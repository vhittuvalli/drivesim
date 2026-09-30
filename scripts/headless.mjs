// Run the app in headless Chrome for data collection and benchmarks. Needs the dev server
// (npm run dev), puppeteer-core (npm install) and a local Chrome (CHROME=<path> to override).
//
//   node scripts/headless.mjs collect --seeds 11,12,13 --frames 5000                 expert data
//   node scripts/headless.mjs collect --seeds 21,22 --frames 4000 --neural --noise 0  DAgger
//   node scripts/headless.mjs bench --seed 1 --out models/bench.json
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

// Poll window.__status() until done(status); returns the last status.
async function watch({ page, tag }, done, describe) {
  let lastT = null, since = Date.now(), lastLog = 0, st = null;
  for (;;) {
    await sleep(3000);
    st = await page.evaluate(() => window.__status?.() ?? null).catch(() => null);
    const t = st?.t ?? null;
    if (t !== lastT) (lastT = t), (since = Date.now());
    else if (Date.now() - since > STALL_S * 1000) {
      console.log(tag, `stalled at sim t=${t}; giving up`);
      return st;
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
  const one = async (seed) => {
    const q = `seed=${seed}&collect=${opt.frames}&noise=${opt.noise}&speed=8&fx=0${opt.neural ? '&neural=1' : ''}`;
    const b = await open(q);
    try {
      const st = await watch(b, (s) => s.collect?.done, describe);
      console.log(b.tag, 'finished:', describe(st ?? {}));
      await sleep(3000); // let the last uploads land
    } finally {
      await b.browser.close();
    }
  };
  const queue = [...seeds];
  await Promise.all(Array.from({ length: Math.min(parallel, seeds.length) }, async () => {
    while (queue.length) await one(queue.shift());
  }));
  const ds = await (await fetch(`${opt.base}/api/datasets`)).json();
  console.log(`data/: ${ds.runs.length} runs, ${ds.frames} frames`);
}

async function bench() {
  const b = await open(`seed=${opt.seed}&bench=1&trials=${opt.trials}&speed=8&fx=0`);
  let st;
  try {
    st = await watch(b, (s) => s.bench?.done, (s) => `benchmark ${(s.bench?.progress.index ?? 0) + 1}/${s.bench?.progress.total ?? '?'} · ${s.fps}${s.error ? ` · ERROR ${s.error}` : ''}`);
  } finally {
    await b.browser.close();
  }
  if (!st?.bench?.summary) throw new Error('benchmark did not finish');
  const out = { seed: Number(opt.seed), trials: Number(opt.trials), date: new Date().toISOString(), model: st.model, summary: st.bench.summary, results: st.bench.results };
  console.log(JSON.stringify(out.summary, null, 2));
  for (const r of out.results) {
    const res = r.kind === 'scenario' ? `${r.status.padEnd(6)} ${r.time.toFixed(1).padStart(5)} s` : `${(r.dist / 1000).toFixed(2)} km`;
    console.log(`  ${(r.name + (r.trial ? ` #${r.trial}` : '')).padEnd(38)} ${res}  takeovers ${r.takeovers}  contacts ${r.contacts}${r.reasons.length ? `  (${[...new Set(r.reasons)].join(', ')})` : ''}`);
  }
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
