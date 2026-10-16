// Run the app in headless Chrome for data collection and benchmarks. Needs the dev server
// (npm run dev), puppeteer-core (npm install) and a local Chrome (CHROME=<path> to override).
//
//   node scripts/headless.mjs collect --seeds 11,12,13 --frames 5000                 expert data
//   node scripts/headless.mjs collect --seeds 21,22 --frames 4000 --neural --noise 0  DAgger
//   node scripts/headless.mjs collect --seeds 31,32 --frames 4000 --highway 0.8       mostly highway
//   node scripts/headless.mjs collect ... --neural --scenarios 0.7 --only lead-brake,hw-jam   targeted DAgger
//   node scripts/headless.mjs bench --seed 1 --out models/bench.json
//   node scripts/headless.mjs bench --seeds 1,2,3 --out models/bench.json            three cities, combined
//   node scripts/headless.mjs bench --seeds 1,2,3 --lenient ...    safety driver steps in only for danger
//
// Each seed (city) gets its own browser; --parallel of them run at once.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import puppeteer from 'puppeteer-core';

const CHROMES = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
];
const STALL_S = 120; // give up on a browser whose simulation clock hasn't moved for this long
// ... or that hasn't started it this long after loading. Building a city without a GPU (SwiftShader,
// several browsers at once) takes minutes: LOAD_S=<seconds> raises it, STAGGER_S=<seconds> spaces
// the browsers' starts so they don't all build at once.
const LOAD_S = Number(process.env.LOAD_S ?? 300), STAGGER_S = Number(process.env.STAGGER_S ?? 0);

const { positionals, values: opt } = parseArgs({
  allowPositionals: true,
  options: {
    base: { type: 'string', default: 'http://localhost:8000' },
    seeds: { type: 'string', default: '1,2,3' },
    frames: { type: 'string', default: '5000' },
    noise: { type: 'string', default: '0.5' },
    highway: { type: 'string', default: '0.3' }, // share of collection episodes on the highway
    dense: { type: 'boolean', default: false }, // heavy traffic in every episode
    fog: { type: 'string' }, // share of episodes in fog
    scenarios: { type: 'string', default: '0.35' }, // share of episodes that play a scripted scenario
    only: { type: 'string' }, // scenario ids to pick from, comma-separated (default: all)
    neural: { type: 'boolean', default: false },
    parallel: { type: 'string', default: '3' },
    seed: { type: 'string', default: '1' },
    trials: { type: 'string', default: '2' },
    out: { type: 'string' },
    lenient: { type: 'boolean', default: false }, // bench: the lenient safety driver (src/safety.js)
  },
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const chrome = process.env.CHROME ?? CHROMES.find((c) => fs.existsSync(c));
if (!chrome) throw new Error('Chrome not found: set CHROME=/path/to/chrome');

// Metal on the Mac; on Linux (a Runpod CPU pod, as root) WebGL runs on SwiftShader, the CPU renderer.
// CHROME_ARGS (space-separated) replaces either. The cameras the driver sees render to fixed-size
// targets, so VIEWPORT=<w>x<h> (a smaller main view, cheaper without a GPU) doesn't change them.
const CHROME_ARGS = process.env.CHROME_ARGS?.split(' ') ?? (process.platform === 'linux'
  ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox', '--mute-audio']
  : ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--mute-audio']);
const [VIEW_W, VIEW_H] = (process.env.VIEWPORT ?? '1280x720').split('x').map(Number);

// The app loads three.js and onnxruntime-web from jsdelivr (pinned versions). The browsers here get
// them from a local copy in .cdn-cache/ (filled on first use), so a Wi-Fi drop or network change
// mid-run doesn't break page loads, which cost several improvement-loop benchmarks.
const CDN = 'https://cdn.jsdelivr.net/', CDN_CACHE = path.join(path.dirname(new URL(import.meta.url).pathname), '..', '.cdn-cache');
async function fromCache(req) {
  const url = req.url();
  if (!url.startsWith(CDN)) return req.continue();
  const file = path.join(CDN_CACHE, createHash('sha1').update(url).digest('hex'));
  try {
    if (!fs.existsSync(file)) {
      const r = await fetch(url);
      if (!r.ok) return req.continue();
      fs.mkdirSync(CDN_CACHE, { recursive: true });
      fs.writeFileSync(`${file}.type`, r.headers.get('content-type') ?? 'application/octet-stream');
      fs.writeFileSync(`${file}.part`, Buffer.from(await r.arrayBuffer()));
      fs.renameSync(`${file}.part`, file);
    }
    await req.respond({ status: 200, contentType: fs.readFileSync(`${file}.type`, 'utf8'), body: fs.readFileSync(file), headers: { 'Access-Control-Allow-Origin': '*' } });
  } catch {
    await req.continue().catch(() => {});
  }
}

// A fresh browser on the app. A page that times out loading (dev server or GPU busy) gets a new
// browser and another try, up to LOAD_TRIES, instead of failing the whole collection or benchmark.
const LOAD_TRIES = 4;
async function open(query) {
  const tag = `[${query.match(/seed=(\d+)/)?.[1] ?? '?'}]`;
  for (let attempt = 1; ; attempt++) {
    const browser = await puppeteer.launch({
      executablePath: chrome,
      headless: true,
      args: CHROME_ARGS,
      defaultViewport: { width: VIEW_W, height: VIEW_H },
    });
    try {
      const page = await browser.newPage();
      await page.setRequestInterception(true);
      page.on('request', fromCache);
      page.on('pageerror', (e) => console.log(tag, 'page error:', e.message));
      page.on('console', (m) => m.type() === 'error' && !m.text().includes('404') && console.log(tag, m.text()));
      await page.goto(`${opt.base}/?${query}`, { timeout: 60_000 });
      return { browser, page, tag };
    } catch (e) {
      await browser.close().catch(() => {});
      if (attempt >= LOAD_TRIES) throw e;
      console.log(tag, `page did not load (${e.message}); new browser in 10 s (try ${attempt + 1}/${LOAD_TRIES})`);
      await sleep(10_000);
    }
  }
}

// Poll window.__status() until done(status), the GPU context is lost, the clock stalls, or the run
// hasn't started(status) LOAD_S after opening (a page half-loaded when the network dropped drives
// on without ever starting it); returns the last status (with .lost or .stalled set in those cases).
async function watch({ page, tag }, done, describe, started = () => true) {
  let lastT = null, since = Date.now(), lastLog = 0, st = null, isRunning = false;
  const opened = Date.now();
  for (;;) {
    await sleep(3000);
    st = await page.evaluate(() => window.__status?.() ?? null).catch(() => null);
    const t = st?.t ?? null;
    if (t !== lastT) (lastT = t), (since = Date.now());
    else if (Date.now() - since > (t == null ? LOAD_S : STALL_S) * 1000) {
      console.log(tag, `stalled at sim t=${t}; giving up`);
      return { ...(st ?? {}), stalled: true };
    }
    isRunning ||= Boolean(st && started(st));
    if (!isRunning && Date.now() - opened > LOAD_S * 1000) {
      console.log(tag, `run never started at sim t=${t}; giving up`);
      return { ...(st ?? {}), stalled: true };
    }
    if (st?.contextLost) {
      console.log(tag, `lost the GPU context at sim t=${t?.toFixed?.(0)}`);
      return { ...st, lost: true };
    }
    if (st && Date.now() - lastLog > 20000) {
      console.log(tag, describe(st));
      lastLog = Date.now();
    }
    if (st && done(st)) return st;
  }
}

async function collect() {
  const seeds = opt.seeds.split(',').map(Number), parallel = Number(opt.parallel);
  const describe = (st) => {
    const c = st.collect ?? {};
    return `${c.total ?? 0}/${c.frames} frames · episode ${c.episode} · ${st.fps}${st.error ? ` · ERROR ${st.error}` : ''}`;
  };
  // One city: if the browser loses its GPU context or stalls, reopen it and collect what's left
  // (new runs in data/, same seed), up to 3 times.
  const one = async (seed) => {
    let left = Number(opt.frames), restarts = 0;
    while (left > 0) {
      const q = `seed=${seed}&collect=${left}&noise=${opt.noise}&hwshare=${opt.highway}&speed=8&fx=0${opt.neural ? '&neural=1' : ''}${opt.dense ? '&dense=1' : ''}${opt.fog ? `&fog=${opt.fog}` : ''}&scen=${opt.scenarios}${opt.only ? `&only=${opt.only}` : ''}`;
      const b = await open(q);
      let st;
      try {
        st = await watch(b, (s) => s.collect?.done, describe);
        await sleep(3000); // let the last uploads land
      } finally {
        await b.browser.close();
      }
      left -= st?.collect?.total ?? 0;
      if (!(st?.lost || st?.stalled) || left <= 0) break;
      if (++restarts > 3) {
        console.log(b.tag, `giving up after ${restarts - 1} restarts with ${left} frames left`);
        break;
      }
      console.log(b.tag, `restarting the browser for the remaining ${left} frames (restart ${restarts})`);
    }
    console.log(`[${seed}] finished: ${Number(opt.frames) - Math.max(0, left)}/${opt.frames} frames${restarts ? ` after ${restarts} restart(s)` : ''}`);
  };
  const queue = [...seeds];
  await Promise.all(Array.from({ length: Math.min(parallel, seeds.length) }, async () => {
    while (queue.length) await one(queue.shift());
  }));
  const ds = await (await fetch(`${opt.base}/api/datasets`)).json();
  console.log(`data/: ${ds.runs.length} runs, ${ds.frames} frames`);
}

// One benchmark per seed (city); with several seeds they run in parallel and the summary adds
// them up, since single-seed results swing a lot between runs.
async function bench() {
  // --seeds runs several cities; otherwise the single --seed (default 1).
  const seeds = (process.argv.some((x) => x.startsWith('--seeds')) ? opt.seeds : opt.seed).split(',').map(Number);
  const runOne = async (seed) => {
    for (let attempt = 0; attempt < 2; attempt++) {
      let b;
      try {
        b = await open(`seed=${seed}&bench=1&trials=${opt.trials}&speed=8&fx=0${opt.lenient ? '&safety=lenient' : ''}`);
      } catch (e) {
        console.log(`[${seed}] benchmark page never loaded: ${e.message}`);
        continue;
      }
      let st;
      try {
        st = await watch(b, (s) => s.bench?.done, (s) => `benchmark ${(s.bench?.progress.index ?? 0) + 1}/${s.bench?.progress.total ?? '?'} · ${s.fps}${s.error ? ` · ERROR ${s.error}` : ''}`,
          (s) => s.bench?.progress?.total != null);
      } finally {
        await b.browser.close();
      }
      if (st?.bench?.summary) return { seed, model: st.model, summary: st.bench.summary, results: st.bench.results };
      console.log(`[${seed}] benchmark did not finish${st?.lost ? ' (lost the GPU context)' : ''}${attempt === 0 ? '; starting it again' : ''}`);
    }
    return null;
  };
  // A city that fails twice (e.g. the network dropped while loading) is reported and left out;
  // the cities that finished are still written.
  const runs = [], failed = [], queue = [...seeds];
  await Promise.all(Array.from({ length: Math.min(Number(opt.parallel), seeds.length) }, async (_, i) => {
    await sleep(i * STAGGER_S * 1000);
    while (queue.length) {
      const seed = queue.shift(), r = await runOne(seed);
      if (r) runs.push(r);
      else failed.push(seed);
    }
  }));
  if (!runs.length) throw new Error(`no benchmark finished (seeds ${failed.join(',')})`);
  if (failed.length) console.log(`left out seed(s) ${failed.join(',')}: did not finish twice`);
  runs.sort((a, b) => a.seed - b.seed);
  seeds.splice(0, seeds.length, ...runs.map((r) => r.seed));
  const results = runs.flatMap((r) => r.results.map((x) => ({ seed: r.seed, ...x })));
  const total = (f) => runs.reduce((a, r) => a + f(r.summary), 0);
  const drives = results.filter((r) => r.kind === 'drive');
  const auto = drives.reduce((a, r) => a + r.autoDist, 0), dist = drives.reduce((a, r) => a + r.dist, 0), driveTO = drives.reduce((a, r) => a + r.takeovers, 0);
  const reasons = {};
  for (const r of results) for (const x of r.reasons) reasons[x] = (reasons[x] ?? 0) + 1;
  const summary = {
    seeds, scenarios: total((s) => s.scenarios), passed: total((s) => s.passed), clean: total((s) => s.clean),
    scenarioTakeovers: total((s) => s.scenarioTakeovers), driveKm: dist / 1000, autonomy: dist ? auto / dist : 0,
    takeovers: driveTO, metersPerTakeover: driveTO ? auto / driveTO : null, contacts: total((s) => s.contacts), reasons,
  };
  const out = { seeds, trials: Number(opt.trials), safety: opt.lenient ? 'lenient' : 'strict', date: new Date().toISOString(), model: runs[0]?.model, summary, perSeed: runs.map((r) => ({ seed: r.seed, summary: r.summary })), results };
  for (const r of runs) {
    const s = r.summary;
    console.log(`seed ${r.seed}: ${s.passed}/${s.scenarios} scenarios (${s.clean} clean), ${s.scenarioTakeovers} scenario takeovers, ${s.driveKm.toFixed(2)} km driven, ${s.takeovers} drive takeovers, ${s.contacts} contacts`);
  }
  console.log(JSON.stringify(summary, null, 2));
  if (opt.out) {
    fs.mkdirSync(path.dirname(opt.out), { recursive: true });
    fs.writeFileSync(opt.out, JSON.stringify(out, null, 2));
    console.log(`wrote ${opt.out}`);
  }
}

const ok = await fetch(`${opt.base}/api/datasets`).then((r) => r.ok, () => false);
if (!ok) throw new Error(`dev server not reachable at ${opt.base}: run npm run dev`);
const cmd = { collect, bench }[positionals[0]];
if (!cmd) throw new Error('usage: node scripts/headless.mjs collect|bench [options]');
await cmd();
