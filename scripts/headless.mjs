// Run the app in headless Chrome for data collection and benchmarks. Needs the dev server
// (npm run dev), puppeteer-core (npm install) and a local Chrome (CHROME=<path> to override).
//
//   node scripts/headless.mjs collect --seeds 11,12,13 --frames 5000                 expert data
//   node scripts/headless.mjs collect --seeds 21,22 --frames 4000 --neural --noise 0  DAgger
//   node scripts/headless.mjs collect --seeds 31,32 --frames 4000 --highway 0.8       mostly highway
//   node scripts/headless.mjs bench --seed 1 --out models/bench.json
//   node scripts/headless.mjs bench --seeds 1,2,3 --out models/bench.json            three cities, combined
//
// Each seed (city) gets its own browser; --parallel of them run at once.
import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import puppeteer from 'puppeteer-core';

const CHROMES = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
];
const STALL_S = 120; // give up on a browser whose simulation clock hasn't moved for this long

const { positionals, values: opt } = parseArgs({
  allowPositionals: true,
  options: {
    base: { type: 'string', default: 'http://localhost:8000' },
    seeds: { type: 'string', default: '1,2,3' },
    frames: { type: 'string', default: '5000' },
    noise: { type: 'string', default: '0.5' },
    highway: { type: 'string', default: '0.3' }, // share of collection episodes on the highway
    dense: { type: 'boolean', default: false }, // heavy traffic in every episode
    neural: { type: 'boolean', default: false },
    parallel: { type: 'string', default: '3' },
    seed: { type: 'string', default: '1' },
    trials: { type: 'string', default: '2' },
    out: { type: 'string' },
  },
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const chrome = process.env.CHROME ?? CHROMES.find((c) => fs.existsSync(c));
if (!chrome) throw new Error('Chrome not found: set CHROME=/path/to/chrome');

async function open(query) {
  const browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--mute-audio'],
    defaultViewport: { width: 1280, height: 720 },
  });
  const page = await browser.newPage();
  const tag = `[${query.match(/seed=(\d+)/)?.[1] ?? '?'}]`;
  page.on('pageerror', (e) => console.log(tag, 'page error:', e.message));
  page.on('console', (m) => m.type() === 'error' && !m.text().includes('404') && console.log(tag, m.text()));
  await page.goto(`${opt.base}/?${query}`);
  return { browser, page, tag };
}

// Poll window.__status() until done(status), the GPU context is lost, or the clock stalls; returns
// the last status (with .lost or .stalled set in those cases).
async function watch({ page, tag }, done, describe) {
  let lastT = null, since = Date.now(), lastLog = 0, st = null;
  for (;;) {
    await sleep(3000);
    st = await page.evaluate(() => window.__status?.() ?? null).catch(() => null);
    const t = st?.t ?? null;
    if (t !== lastT) (lastT = t), (since = Date.now());
    else if (Date.now() - since > STALL_S * 1000) {
      console.log(tag, `stalled at sim t=${t}; giving up`);
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
      const q = `seed=${seed}&collect=${left}&noise=${opt.noise}&hwshare=${opt.highway}&speed=8&fx=0${opt.neural ? '&neural=1' : ''}${opt.dense ? '&dense=1' : ''}`;
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
      const b = await open(`seed=${seed}&bench=1&trials=${opt.trials}&speed=8&fx=0`);
      let st;
      try {
        st = await watch(b, (s) => s.bench?.done, (s) => `benchmark ${(s.bench?.progress.index ?? 0) + 1}/${s.bench?.progress.total ?? '?'} · ${s.fps}${s.error ? ` · ERROR ${s.error}` : ''}`);
      } finally {
        await b.browser.close();
      }
      if (st?.bench?.summary) return { seed, model: st.model, summary: st.bench.summary, results: st.bench.results };
      console.log(`[${seed}] benchmark did not finish${st?.lost ? ' (lost the GPU context)' : ''}${attempt === 0 ? '; starting it again' : ''}`);
    }
    throw new Error(`benchmark for seed ${seed} did not finish`);
  };
  const runs = [], queue = [...seeds];
  await Promise.all(Array.from({ length: Math.min(Number(opt.parallel), seeds.length) }, async () => {
    while (queue.length) runs.push(await runOne(queue.shift()));
  }));
  runs.sort((a, b) => a.seed - b.seed);
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
  const out = { seeds, trials: Number(opt.trials), date: new Date().toISOString(), model: runs[0]?.model, summary, perSeed: runs.map((r) => ({ seed: r.seed, summary: r.summary })), results };
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
