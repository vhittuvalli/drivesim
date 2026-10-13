// Train on a Runpod GPU pod instead of the Mac; collection and benchmarks stay local (they need the
// browser sim). The data lives on a Runpod network volume as tar chunks (1.5M small files would be
// slow to read off network storage); each training run gets a fresh pod that unpacks them to its
// local disk, trains with --device cuda, writes its outputs to the volume, and is deleted. Only runs
// that are new (or have changed) since the last sync are uploaded.
//
//   node scripts/remote-train.mjs sync                      upload new runs from a cheap CPU pod
//   node scripts/remote-train.mjs train --init models/policy.pt --epochs 3 --out /tmp/cand
//   node scripts/remote-train.mjs train ... -- --encoder mobilenet     extra train.py arguments
//   node scripts/remote-train.mjs cleanup                   delete pods this script left behind
//
// Needs RUNPOD_API_KEY, RUNPOD_VOLUME (the network volume's id; pods go to its data center) and an
// SSH key (~/.ssh/id_ed25519, or SSH_KEY=<private key path>). Every pod is deleted when the script
// ends, and also from the pod itself after --max-hours, should the Mac sleep or the script die.
//
// Budget: Runpod spending since RUNPOD_BUDGET_SINCE stays under RUNPOD_BUDGET dollars (default $10
// since 2026-10-13). A pod is created only if the spend so far plus the expected cost of the run fits;
// a running pod is deleted when the budget runs out (by this script, and by the pod's own timer).
// The script then exits with code 3 (improve.mjs ends the loop on it).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { parseArgs } from 'node:util';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const API = 'https://api.runpod.io/v2';
// RT_FAKE_POD=<dir> runs the "pod" commands in a local shell with the volume and the pod's disk
// under <dir>, to test the pipeline without renting anything.
const FAKE = process.env.RT_FAKE_POD;
const VOL = FAKE ? `${FAKE}/workspace/drivesim` : '/workspace/drivesim'; // network volume: chunks/, out/, pip cache
const WORK = FAKE ? `${FAKE}/root` : '/root'; // the pod's own disk
const NAME = 'drivesim-train'; // every pod this script creates is named this (cleanup relies on it)
const BUDGET = Number(process.env.RUNPOD_BUDGET ?? 10); // dollars
const SINCE = process.env.RUNPOD_BUDGET_SINCE ?? '2026-10-13T00:00:00Z';
const LEDGER = path.join(ROOT, 'models', 'loop', 'runpod-spend.jsonl'); // one line per pod: what it cost
const OVER_BUDGET = 3; // exit code

const argv = process.argv.slice(2), dash = argv.indexOf('--');
const { positionals, values: opt } = parseArgs({
  args: dash < 0 ? argv : argv.slice(0, dash),
  allowPositionals: true,
  options: {
    init: { type: 'string' },
    epochs: { type: 'string', default: '3' },
    out: { type: 'string' },
    resume: { type: 'boolean', default: false },
    gpu: { type: 'string', default: 'NVIDIA GeForce RTX 4090' },
    image: { type: 'string', default: 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404' },
    workers: { type: 'string', default: '8' },
    'max-hours': { type: 'string', default: '4' },
    parallel: { type: 'string', default: '4' }, // upload streams
    chunk: { type: 'string', default: '48' }, // runs per uploaded tar
    data: { type: 'string', default: path.join(ROOT, 'data') },
  },
});
const extra = dash < 0 ? [] : argv.slice(dash + 1);

const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const KEY = process.env.RUNPOD_API_KEY ?? (FAKE && 'fake');
const SSH_KEY = process.env.SSH_KEY ?? path.join(os.homedir(), '.ssh', 'id_ed25519');
if (!KEY) throw new Error('set RUNPOD_API_KEY');

async function api(method, p, body) {
  const r = await fetch(API + p, {
    method, headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
    body: body && JSON.stringify(body),
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`${method} ${p}: ${r.status} ${text.slice(0, 300)}`);
  return text ? JSON.parse(text) : null;
}

// A shell command; resolves with its stdout, rejects on a non-zero exit. `input` goes to stdin.
function sh(cmd, { input, quiet } = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn('bash', ['-o', 'pipefail', '-c', cmd], { stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, COPYFILE_DISABLE: '1' } });
    let out = '', err = '';
    p.stdout.on('data', (b) => (out += b));
    p.stderr.on('data', (b) => (err += b));
    p.on('close', (code) => (code === 0 ? resolve(out) : reject(new Error(`${quiet ? 'command' : cmd.slice(0, 120)} exited ${code}: ${err.trim().slice(-400)}`))));
    p.stdin.end(input ?? '');
  });
}
const q = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;

// ---- budget ----

class BudgetError extends Error {}
const ledger = () => (fs.existsSync(LEDGER) ? fs.readFileSync(LEDGER, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [])
  .filter((e) => e.date >= SINCE);

// Dollars spent since SINCE: what Runpod has billed, or, when its billing lags behind (it comes in
// hourly buckets), the pods in the ledger plus billed storage, whichever is more.
async function spent() {
  if (FAKE) return ledger().reduce((a, e) => a + e.cost, 0);
  const t = (await api('GET', `/billing?startTime=${SINCE}&bucketSize=day`)).metadata.totals;
  const storage = (t.storageStandardAmount ?? 0) + (t.storageHighPerformanceAmount ?? 0);
  return Math.max(t.totalAmount ?? 0, ledger().reduce((a, e) => a + e.cost, 0) + storage);
}

// What a run of this kind is expected to cost: the dearest one so far, or a cautious default.
function expected(kind, price) {
  const past = ledger().filter((e) => e.kind === kind).map((e) => e.cost);
  return past.length ? Math.max(...past) : price * (kind === 'gpu' ? 1.5 : 3);
}

// ---- pods ----

let pod = null; // { id, ssh, kind, price, created, spentBefore } of the pod this run created
const volume = () => api('GET', `/network-volumes/${process.env.RUNPOD_VOLUME ?? die('set RUNPOD_VOLUME')}`);
function die(m) { throw new Error(m); }

async function createPod(kind) {
  const price = kind === 'gpu' ? 0.89 : 0.06; // until the pod reports its own
  const before = await spent(), need = expected(kind, price);
  log(`Runpod budget: $${before.toFixed(2)} of $${BUDGET} spent since ${SINCE.slice(0, 10)}; this ${kind} run ~$${need.toFixed(2)}`);
  if (before + need > BUDGET) throw new BudgetError(`budget: $${before.toFixed(2)} spent + ~$${need.toFixed(2)} for this run would pass $${BUDGET}`);
  if (FAKE) return void (pod = { id: 'fake', ssh: 'bash -c', kind, price, created: Date.now(), spentBefore: before });
  const vol = await volume();
  const pub = fs.readFileSync(SSH_KEY + '.pub', 'utf8').trim();
  const body = {
    name: NAME, image: opt.image, cloud: 'SECURE', dataCenterIds: [vol.dataCenter], ports: ['22/tcp'],
    mounts: { network: [{ volumeId: vol.id, path: '/workspace' }] }, env: { PUBLIC_KEY: pub },
    ...(kind === 'gpu'
      ? { gpu: { id: opt.gpu, count: 1, minCudaVersion: '12.8', minVcpuCountPerGpu: 8 }, disk: 120 }
      : { cpu: { id: 'cpu3c', vcpuCount: 2 }, disk: 20 }),
  };
  // Stock comes and goes: a few tries before giving up.
  for (let attempt = 1; ; attempt++) {
    try {
      const p = await api('POST', '/pods', body);
      pod = { id: p.id, kind, price: p.cost || price, created: Date.now(), spentBefore: before };
      log(`created ${kind} pod ${p.id} in ${vol.dataCenter} ($${pod.price}/h)`);
      break;
    } catch (e) {
      if (attempt >= 5) throw e;
      log(`pod create failed (${e.message.slice(0, 160)}); retrying in 60 s`);
      await sleep(60_000);
    }
  }
  // Wait for SSH: the pod's public port for 22, then a login that works.
  const t0 = Date.now();
  for (;;) {
    if (Date.now() - t0 > 15 * 60_000) throw new Error(`pod ${pod.id} not reachable over SSH after 15 min`);
    await sleep(10_000);
    const p = await api('GET', `/pods/${pod.id}`).catch(() => null);
    const port = p?.runtime?.ports?.find((x) => x.private === 22 && x.public);
    if (!port?.ip) continue;
    pod.ssh = `ssh -i ${q(SSH_KEY)} -p ${port.public} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ConnectTimeout=20 root@${port.ip}`;
    if (await sh(`${pod.ssh} true`).then(() => true, () => false)) break;
  }
  log(`pod ${pod.id} up`);
  // The pod deletes itself after --max-hours or when the budget runs out, whatever happens to this script.
  if (FAKE) return;
  const hours = Math.min(Number(opt['max-hours']), (BUDGET - before) / pod.price) - (Date.now() - pod.created) / 3.6e6;
  const secs = Math.max(60, Math.round(hours * 3600));
  await sh(`${pod.ssh} ${q(`umask 077; cat > ${WORK}/.rpkey; nohup sh -c 'sleep ${secs}; curl -s -X DELETE -H "Authorization: Bearer $(cat ${WORK}/.rpkey)" ${API}/pods/${pod.id}' < /dev/null >/dev/null 2>&1 &`)}`, { input: KEY, quiet: true });
}

// Stops the work once this pod has used up what is left of the budget.
function checkBudget() {
  if (!pod) return;
  const cost = ((Date.now() - pod.created) / 3.6e6) * pod.price;
  if (pod.spentBefore + cost > BUDGET) throw new BudgetError(`budget: $${(pod.spentBefore + cost).toFixed(2)} of $${BUDGET} used; stopping the pod`);
}

async function deletePod() {
  if (!pod) return;
  const { id, kind, price, created } = pod, hours = (Date.now() - created) / 3.6e6;
  pod = null;
  if (created) fs.mkdirSync(path.dirname(LEDGER), { recursive: true });
  if (created) fs.appendFileSync(LEDGER, JSON.stringify({ date: new Date(created).toISOString(), pod: id, kind, price, hours: +hours.toFixed(3), cost: +(hours * price).toFixed(3) }) + '\n');
  if (FAKE) return;
  for (let i = 0; i < 3; i++) {
    await api('DELETE', `/pods/${id}`).catch(() => {});
    if (!(await api('GET', `/pods/${id}`).then((p) => p && p.status !== 'TERMINATED', () => false))) return log(`deleted pod ${id}`);
    await sleep(5000);
  }
  log(`WARNING: pod ${id} may still be running: delete it in the console or with: node scripts/remote-train.mjs cleanup`);
}

// ---- data sync ----

// Local runs with their file counts; runs written to in the last 2 minutes are still being
// collected and wait for the next sync.
function localRuns() {
  const runs = new Map(), now = Date.now();
  for (const run of fs.readdirSync(opt.data)) {
    const dir = path.join(opt.data, run);
    if (!fs.statSync(dir).isDirectory()) continue;
    let n = 0, newest = fs.statSync(dir).mtimeMs;
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      if (e.isDirectory()) {
        n += fs.readdirSync(path.join(dir, e.name)).length;
        newest = Math.max(newest, fs.statSync(path.join(dir, e.name)).mtimeMs);
      } else n++;
    }
    if (now - newest > 120_000) runs.set(run, n);
  }
  return runs;
}

async function sync() {
  const remote = new Map();
  const lists = await sh(`${pod.ssh} ${q(`mkdir -p ${VOL}/chunks ${VOL}/out; cat ${VOL}/chunks/*.list 2>/dev/null || true`)}`);
  for (const line of lists.split('\n')) {
    const [run, n] = line.trim().split(' ');
    if (run) remote.set(run, Number(n));
  }
  const local = localRuns();
  const todo = [...local].filter(([run, n]) => remote.get(run) !== n).map(([run]) => run).sort();
  if (!todo.length) return log(`data in sync: ${local.size} runs`);
  const chunks = [];
  for (let i = 0; i < todo.length; i += Number(opt.chunk)) chunks.push(todo.slice(i, i + Number(opt.chunk)));
  const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
  log(`uploading ${todo.length} runs in ${chunks.length} chunks`);
  let sent = 0, done = 0;
  const t0 = Date.now(), queue = chunks.map((runs, i) => ({ runs, name: `${stamp}-${String(i).padStart(4, '0')}` }));
  await Promise.all(Array.from({ length: Math.min(Number(opt.parallel), queue.length) }, async () => {
    while (queue.length) {
      const { runs, name } = queue.shift();
      const list = runs.map((r) => `${r} ${local.get(r)}`).join('\n');
      const remoteCmd = `cd ${VOL}/chunks && cat > ${name}.part && mv ${name}.part ${name}.tar && printf '%s\\n' ${q(list)} > ${name}.list && wc -c < ${name}.tar`;
      for (let attempt = 1; ; attempt++) {
        try {
          sent += Number(await sh(`tar cf - -C ${q(opt.data)} ${runs.map(q).join(' ')} | ${pod.ssh} ${q(remoteCmd)}`));
          break;
        } catch (e) {
          if (attempt >= 3) throw e;
          log(`chunk ${name} failed (${e.message.slice(-160)}); again`);
        }
      }
      done++;
      checkBudget();
      log(`chunk ${done}/${chunks.length} · ${(sent / 1e9).toFixed(2)} GB · ${(sent / 1e6 / ((Date.now() - t0) / 1000)).toFixed(1)} MB/s`);
    }
  }));
  log(`uploaded ${todo.length} runs (${(sent / 1e9).toFixed(2)} GB)`);
}

// ---- training ----

async function train() {
  if (!opt.out) throw new Error('--out <path without extension> is required');
  const name = `${path.basename(opt.out)}-${Date.now().toString(36)}`, outR = `${VOL}/out/${name}`;
  // Code and the starting checkpoint go to the pod's own disk.
  await sh(`tar cf - -C ${q(ROOT)} train | ${pod.ssh} ${q(`rm -rf ${WORK}/drivesim && mkdir -p ${WORK}/drivesim && tar xf - -C ${WORK}/drivesim`)}`);
  if (opt.init) await sh(`${pod.ssh} ${q(`cat > ${WORK}/init.pt`)} < ${q(opt.init)}`);
  log('unpacking the data on the pod');
  await sh(`${pod.ssh} ${q(`mkdir -p ${WORK}/data && for t in $(ls ${VOL}/chunks/*.tar | sort); do tar xf "$t" -C ${WORK}/data; done${FAKE ? '' : ` && pip install -q --cache-dir ${VOL}/pip onnx onnxruntime 2>&1 | tail -2`}`)}`);
  const args = ['train/train.py', '--device', FAKE ? 'auto' : 'cuda', '--data', `${WORK}/data`, '--epochs', opt.epochs, '--out', outR, '--workers', opt.workers,
    ...(opt.init ? ['--init', `${WORK}/init.pt`] : []), ...extra];
  // Training runs detached on the pod (a dropped SSH connection doesn't stop it); this polls its log.
  // A data-loader failure resumes once from <out>.last.pt, as in improve.mjs.
  const start = (resume) => sh(`${pod.ssh} ${q(`cd ${WORK}/drivesim && rm -f ${outR}.exit && nohup sh -c ${q(`${FAKE ? `${ROOT}/.venv/bin/python` : 'python'} ${args.map(q).join(' ')}${resume ? ' --resume' : ''} >> ${outR}.log 2>&1; echo $? > ${outR}.exit`)} < /dev/null > /dev/null 2>&1 &`)}`);
  let shown = 0, exit = null;
  const localLog = `${opt.out}.remote.log`;
  fs.mkdirSync(path.dirname(path.resolve(opt.out)), { recursive: true });
  for (const resume of opt.resume ? [true] : [false, true]) {
    log(`training on the pod${resume ? ' (resuming)' : ''}: ${outR}`);
    await start(resume);
    for (;;) {
      await sleep(FAKE ? 5000 : 60_000);
      checkBudget();
      const st = await sh(`${pod.ssh} ${q(`cat ${outR}.exit 2>/dev/null; echo ---; tail -n +${shown + 1} ${outR}.log`)}`).catch(() => null);
      if (st == null) continue; // SSH hiccup: the training goes on
      const [code, newLines] = st.split('---\n');
      if (newLines) {
        fs.appendFileSync(localLog, newLines);
        shown += newLines.split('\n').length - 1;
        for (const l of newLines.split('\n')) if (/^(epoch|wrote|training on|Traceback|\w*Error)/.test(l)) log('  ' + l.slice(0, 200));
      }
      if (code.trim()) { exit = Number(code.trim()); break; }
    }
    if (exit === 0) break;
  }
  if (exit !== 0) throw new Error(`training failed on the pod (exit ${exit}); log in ${localLog}`);
  for (const ext of ['onnx', 'json', 'pt']) await sh(`${pod.ssh} ${q(`cat ${outR}.${ext}`)} > ${q(`${opt.out}.${ext}`)}`);
  log(`wrote ${opt.out}.onnx, .json and .pt`);
}

// ---- main ----

async function cleanup() {
  const pods = (await api('GET', '/pods')).pods.filter((p) => p.name === NAME);
  if (!pods.length) return log(`no ${NAME} pods`);
  for (const p of pods) {
    pod = { id: p.id };
    await deletePod();
  }
}

const cmd = positionals[0];
let stopping = false;
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, async () => {
  if (stopping) return;
  stopping = true;
  log(`${sig}: deleting the pod`);
  await deletePod().catch(() => {});
  process.exit(1);
});
try {
  if (cmd === 'sync') {
    await createPod('cpu');
    await sync();
  } else if (cmd === 'train') {
    await createPod('gpu');
    await sync();
    await train();
  } else if (cmd === 'cleanup') await cleanup();
  else throw new Error('usage: node scripts/remote-train.mjs sync|train|cleanup [options]');
} catch (e) {
  log('ERROR', e.message);
  process.exitCode = e instanceof BudgetError ? OVER_BUDGET : 1;
} finally {
  await deletePod().catch((e) => log('ERROR deleting the pod:', e.message));
}
