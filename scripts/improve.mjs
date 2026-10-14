// The improvement loop: rounds of targeted DAgger, fine-tuning and benchmarking that run
// unattended (overnight). Each round:
//   1. picks the scenarios the best model failed most in its benchmark,
//   2. lets the best model drive new cities (never benchmark cities) with the expert labeling,
//      mostly in those scenarios,
//   3. fine-tunes the best model on all of data/,
//   4. benchmarks the candidate on the same cities as the best model,
//   5. keeps the candidate only if it has fewer takeovers, no more contacts and no more failed
//      scenarios, and commits it.
// The collected data stays in data/ either way, so a rejected round still feeds the next one.
//
//   node scripts/improve.mjs --rounds 4
//   node scripts/improve.mjs --rounds 4 --baseline models/bench-fix-49.json   reuse a benchmark of the current model
//   node scripts/improve.mjs --rounds 4 --remote     fine-tune on a Runpod GPU pod (scripts/remote-train.mjs);
//                                                     the loop ends when the Runpod budget is used up
//
// models/loop/ keeps the state between runs: best.json (the best model's benchmark), history.jsonl
// (one line per round) and history.md (the same as a table). Create models/loop/STOP to end the loop
// after the current round. Needs the dev server (npm run dev).
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import { parseArgs } from 'node:util';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const LOOP = path.join(ROOT, 'models', 'loop');
const { values: opt } = parseArgs({
  options: {
    rounds: { type: 'string', default: '4' },
    // Benchmark cities, never collected on. Nine, not six: always picking the best model on the
    // same six cities partly picked models that suited them (a 29% gain there, 7% on cities 1-3).
    'eval-seeds': { type: 'string', default: '1,2,3,4,5,6,7,8,9' },
    cities: { type: 'string', default: '12' }, // new cities per round
    frames: { type: 'string', default: '2000' }, // frames per city
    epochs: { type: 'string', default: '3' },
    targets: { type: 'string', default: '4' }, // scenarios to focus collection on
    scenarios: { type: 'string', default: '0.6' }, // share of collection episodes that play one of them
    fog: { type: 'string', default: '0.2' },
    'first-seed': { type: 'string', default: '3001' }, // collection city seeds count up from here
    baseline: { type: 'string' }, // an existing benchmark of the current model on the eval seeds
    commit: { type: 'boolean', default: true }, // commit an accepted model (never pushes)
    remote: { type: 'boolean', default: false }, // train on a Runpod pod; needs RUNPOD_API_KEY and RUNPOD_VOLUME
  },
});

const log = (...a) => console.log(new Date().toISOString().slice(0, 19), ...a);
const evalSeeds = opt['eval-seeds'].split(',').map(Number);
const MODEL = (ext) => path.join(ROOT, 'models', `policy.${ext}`);

// A child process with its output appended to `logFile`. With `doneWhen`, a process that has
// printed that line is stopped 30 s later if it hasn't exited (the benchmark has hung after
// writing its results).
function run(cmd, args, logFile, doneWhen = null) {
  return new Promise((resolve) => {
    const out = fs.openSync(logFile, 'a');
    const child = spawn(cmd, args, { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
    let timer = null;
    const onData = (buf) => {
      fs.writeSync(out, buf);
      if (doneWhen && !timer && doneWhen.test(buf.toString())) timer = setTimeout(() => child.kill(), 30_000);
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('close', (code) => {
      clearTimeout(timer);
      fs.closeSync(out);
      resolve(code);
    });
  });
}

function summary(bench) {
  const s = bench.summary;
  return {
    takeovers: s.scenarioTakeovers + s.takeovers, contacts: s.contacts, passed: s.passed, scenarios: s.scenarios,
    clean: s.clean, metersPerTakeover: Math.round(s.metersPerTakeover ?? 0), reasons: s.reasons,
  };
}

// Scenario ids with the most takeovers (failed runs count extra), for focusing collection.
function targets(bench, n) {
  const score = {};
  for (const r of bench.results) {
    if (r.kind !== 'scenario') continue;
    score[r.id] = (score[r.id] ?? 0) + r.takeovers + (r.status === 'passed' ? 0 : 5) + 10 * (r.contacts ?? 0);
  }
  return Object.entries(score).sort((a, b) => b[1] - a[1]).slice(0, n).map(([id]) => id);
}

// Share of collection on the highway: the highway's share of the benchmark's free-drive takeovers,
// between 0.15 and 0.5 so neither road is starved.
function highwayShare(bench) {
  const drives = bench.results.filter((r) => r.kind === 'drive');
  const all = drives.reduce((a, r) => a + r.takeovers, 0) || 1;
  const hw = drives.filter((r) => r.road === 'highway').reduce((a, r) => a + r.takeovers, 0);
  return Math.min(0.5, Math.max(0.15, hw / all));
}

async function benchmark(out, logFile) {
  await run('node', ['scripts/headless.mjs', 'bench', '--seeds', evalSeeds.join(','), '--out', out], logFile, /^wrote /m);
  return fs.existsSync(out) ? JSON.parse(fs.readFileSync(out, 'utf8')) : null;
}

function record(entry) {
  fs.appendFileSync(path.join(LOOP, 'history.jsonl'), JSON.stringify(entry) + '\n');
  const md = path.join(LOOP, 'history.md');
  if (!fs.existsSync(md)) {
    fs.writeFileSync(md, `# Improvement loop\n\nBenchmark cities ${evalSeeds.join(', ')}. Kept = fewer takeovers, no more contacts and no more failed scenarios than the best model.\n\n| Round | Date | Cities collected | Targets | Takeovers | Contacts | Scenarios | m/takeover | Kept |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n`);
  }
  const c = entry.candidate;
  const row = c
    ? `| ${entry.round} | ${entry.date.slice(0, 10)} | ${entry.cities} | ${entry.targets.join(', ')} | ${c.takeovers} (best ${entry.best.takeovers}) | ${c.contacts} (best ${entry.best.contacts}) | ${c.passed}/${c.scenarios} | ${c.metersPerTakeover} | ${entry.kept ? 'yes' : 'no'} |`
    : `| ${entry.round} | ${entry.date.slice(0, 10)} | ${entry.cities} | ${entry.targets.join(', ')} | failed: ${entry.error} | | | | no |`;
  fs.appendFileSync(md, row + '\n');
}

async function main() {
  fs.mkdirSync(LOOP, { recursive: true });
  const ok = await fetch('http://localhost:8000/api/datasets').then((r) => r.ok, () => false);
  if (!ok) throw new Error('dev server not reachable at http://localhost:8000: run npm run dev');
  for (const ext of ['onnx', 'json', 'pt']) if (!fs.existsSync(MODEL(ext))) throw new Error(`missing ${MODEL(ext)}`);
  if (opt.remote && !(process.env.RUNPOD_API_KEY && process.env.RUNPOD_VOLUME)) throw new Error('--remote needs RUNPOD_API_KEY and RUNPOD_VOLUME');

  // The best model's benchmark on the eval seeds.
  const bestFile = path.join(LOOP, 'best.json');
  if (!fs.existsSync(bestFile)) {
    if (opt.baseline) fs.copyFileSync(opt.baseline, bestFile);
    else {
      log('benchmarking the current model as the baseline');
      if (!(await benchmark(bestFile, path.join(LOOP, 'baseline-bench.log')))) throw new Error('baseline benchmark failed');
    }
  }
  let best = JSON.parse(fs.readFileSync(bestFile, 'utf8'));
  const seen = best.seeds ?? best.summary.seeds;
  if (String(seen) !== String(evalSeeds)) throw new Error(`best.json is for seeds ${seen}, not ${evalSeeds}`);
  log('best model:', JSON.stringify(summary(best)));

  const history = fs.existsSync(path.join(LOOP, 'history.jsonl'))
    ? fs.readFileSync(path.join(LOOP, 'history.jsonl'), 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [];
  let nextSeed = Math.max(Number(opt['first-seed']), ...history.map((h) => h.lastSeed + 1));
  const firstRound = (history.at(-1)?.round ?? 0) + 1;

  for (let round = firstRound; round < firstRound + Number(opt.rounds); round++) {
    if (fs.existsSync(path.join(LOOP, 'STOP'))) {
      log('STOP file found: ending');
      break;
    }
    const dir = path.join(LOOP, `round-${round}`);
    fs.mkdirSync(dir, { recursive: true });
    const seeds = Array.from({ length: Number(opt.cities) }, (_, i) => nextSeed + i).filter((s) => !evalSeeds.includes(s));
    nextSeed += Number(opt.cities);
    // Highway scenarios (hw-*) only play in highway episodes: give them a real share when targeted.
    const focus = targets(best, Number(opt.targets));
    const hw = focus.some((id) => id.startsWith('hw-')) ? Math.max(0.35, highwayShare(best)) : highwayShare(best);
    const entry = { round, date: new Date().toISOString(), cities: seeds.length, firstSeed: seeds[0], lastSeed: seeds.at(-1), targets: focus, highway: hw, best: summary(best) };
    const cand = path.join(dir, 'cand'), trainLog = path.join(dir, 'train.log');
    // A round whose process was stopped after training (models/loop/STOP only ends the loop between
    // rounds) still has its candidate: it picks up at the benchmark.
    if (fs.existsSync(`${cand}.onnx`)) log(`round ${round}: resuming at the benchmark with the candidate in ${dir}`);
    else {
      log(`round ${round}: collecting ${seeds.length} cities (${seeds[0]}-${seeds.at(-1)}), focus ${focus.join(',')}, highway ${hw.toFixed(2)}`);

      // 1-2. DAgger with the best model (it is what models/ holds).
      await run('node', ['scripts/headless.mjs', 'collect', '--seeds', seeds.join(','), '--frames', opt.frames, '--neural', '--noise', '0',
        '--highway', hw.toFixed(2), '--fog', opt.fog, '--scenarios', opt.scenarios, '--only', focus.join(',')], path.join(dir, 'collect.log'));

      // 3. Fine-tune the best model on everything (a data-loader failure resumes once). With --remote
      // on a Runpod pod, falling back to the Mac if that fails (no GPU in stock, network down).
      log(`round ${round}: fine-tuning ${opt.epochs} epochs${opt.remote ? ' on Runpod' : ''}`);
      const trainArgs = ['train/train.py', '--init', MODEL('pt'), '--epochs', opt.epochs, '--out', cand];
      const remote = opt.remote ? await run('node', ['scripts/remote-train.mjs', 'train', ...trainArgs.slice(1)], trainLog) : null;
      if (remote === 3) { // scripts/remote-train.mjs: the Runpod budget is used up
        record({ ...entry, error: 'Runpod budget used up' });
        log(`round ${round}: Runpod budget used up (see ${trainLog}); ending`);
        break;
      }
      if (remote) {
        log(`round ${round}: remote training failed, training on the Mac`);
        fs.rmSync(`${cand}.onnx`, { force: true });
      }
      if (!fs.existsSync(`${cand}.onnx`) && (await run('.venv/bin/python', trainArgs, trainLog)) !== 0) await run('.venv/bin/python', [...trainArgs, '--resume'], trainLog);
      if (!fs.existsSync(`${cand}.onnx`)) {
        record({ ...entry, error: 'training failed' });
        log(`round ${round}: training failed, see ${trainLog}`);
        continue;
      }
    }

    // 4. Benchmark the candidate in place of the best model, then put the best back.
    log(`round ${round}: benchmarking the candidate`);
    const keep = { onnx: fs.readFileSync(MODEL('onnx')), json: fs.readFileSync(MODEL('json')) };
    let bench;
    try {
      fs.copyFileSync(`${cand}.onnx`, MODEL('onnx'));
      fs.copyFileSync(`${cand}.json`, MODEL('json'));
      bench = await benchmark(path.join(dir, 'bench.json'), path.join(dir, 'bench.log'));
    } finally {
      fs.writeFileSync(MODEL('onnx'), keep.onnx);
      fs.writeFileSync(MODEL('json'), keep.json);
    }
    if (!bench || String(bench.seeds) !== String(evalSeeds)) {
      record({ ...entry, error: bench ? `benchmark covered seeds ${bench.seeds} only` : 'benchmark failed' });
      log(`round ${round}: benchmark incomplete, candidate not considered`);
      continue;
    }

    // 5. Keep it only if it is better without more contacts or failed scenarios (a failure is a
    // timeout, the car stuck: fewer takeovers alone let round 10 trade 3 of them for 11 takeovers).
    const c = summary(bench), b = summary(best);
    const kept = c.takeovers < b.takeovers && c.contacts <= b.contacts && c.passed >= b.passed;
    record({ ...entry, candidate: c, kept });
    log(`round ${round}: candidate ${c.takeovers} takeovers, ${c.contacts} contacts, ${c.passed}/${c.scenarios} passed vs best ${b.takeovers}, ${b.contacts}, ${b.passed}/${b.scenarios}: ${kept ? 'KEPT' : 'rejected'}`);
    if (!kept) continue;
    fs.copyFileSync(`${cand}.onnx`, MODEL('onnx'));
    fs.copyFileSync(`${cand}.json`, MODEL('json'));
    fs.copyFileSync(`${cand}.pt`, MODEL('pt'));
    fs.copyFileSync(path.join(dir, 'bench.json'), bestFile);
    best = bench;
    if (opt.commit) {
      execFileSync('git', ['add', 'models/policy.onnx', 'models/policy.json'], { cwd: ROOT });
      execFileSync('git', ['commit', '-q', '-m', `chore(model): improvement loop round ${round}\n\nBenchmark cities ${evalSeeds.join(', ')}:\n  takeovers  ${b.takeovers} -> ${c.takeovers}\n  contacts   ${b.contacts} -> ${c.contacts}\n  scenarios  ${b.passed}/${b.scenarios} -> ${c.passed}/${c.scenarios}\n  free drive ${b.metersPerTakeover} -> ${c.metersPerTakeover} m per takeover\nCollected ${seeds.length} new cities, focus: ${focus.join(', ')}.`], { cwd: ROOT });
      log(`round ${round}: committed`);
    }
  }
  log('done');
}

main().catch((e) => {
  log('ERROR', e.stack ?? e);
  process.exit(1);
});
