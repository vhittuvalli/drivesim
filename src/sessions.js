// Automated sessions driven from the app's frame loop.
//
// CollectSession: drives on its own and streams training data through a Collector, one run
// (data/<run>/) per episode. Each episode randomizes the road (city streets or highway), weather,
// time of day, traffic, pedestrians and double-parked vans, sometimes plays a scripted scenario, and sometimes perturbs the
// steering so the data shows recoveries. If the neural driver is on, the expert only labels:
// that's a DAgger round.
//
// Benchmark: runs the neural driver (with the safety driver) through every scenario and a
// set of free drives in different conditions, and scores it.
//
// Both talk to the app through `app`: {world, setConditions({road, weather, hour, cars, peds, vans, hwCars}),
// startScenario(id | '')}.
import { SCENARIOS } from './scenarios.js';
import { makeSteerNoise } from './collect.js';

const pick = (rand, xs) => xs[Math.floor(rand() * xs.length)];
const weighted = (rand, table) => {
  let r = rand() * Object.values(table).reduce((a, b) => a + b, 0);
  return Object.keys(table).find((k) => (r -= table[k]) < 0);
};
const stamp = () => new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-');

// highway: share of episodes on the highway. dense: heavy traffic (more cars close ahead to learn from).
// fog: share of episodes in fog (null: the usual weather mix).
export function randomConditions(rand, highway = 0.3, dense = false, fog = null) {
  const night = rand() < 0.22;
  return {
    road: rand() < highway ? 'highway' : 'city',
    weather: fog !== null && rand() < fog ? 'fog' : weighted(rand, { clear: 0.4, rain: 0.25, fog: fog !== null ? 0 : 0.15, snow: 0.2 }),
    hour: night ? pick(rand, [4.5 + rand() * 1.5, 19 + rand() * 3]) : 7 + rand() * 11,
    cars: Math.round(dense ? 80 + rand() * 40 : 10 + rand() * 80),
    peds: Math.round(20 + rand() * 130),
    vans: Math.floor(rand() * 8),
    hwCars: Math.round(dense ? 90 + rand() * 60 : 20 + rand() * 80),
  };
}

export class CollectSession {
  // frames: stop after this many (Infinity: until stopped). noise: share of episodes with
  // steering noise. scenarios: share of episodes that play a scripted scenario. highway: share of
  // episodes on the highway. only: scenario ids to pick from (null: all of them), e.g. just the
  // braking scenarios for a targeted DAgger round.
  constructor(app, collector, { seed, frames = Infinity, hz = 10, episodeSeconds = 75, noise = 0.5, scenarios = 0.35, only = null, highway = 0.3, dense = false, fog = null, rand = Math.random, driver = 'expert' } = {}) {
    Object.assign(this, { app, collector, seed, frames, hz, episodeSeconds, noise, scenarios, only, highway, dense, fog, rand, driver });
    this.total = 0;
    this.episode = 0;
    this.done = false;
    this.stamp = stamp();
  }

  get world() {
    return this.app.world;
  }

  start() {
    this.nextEpisode();
  }

  nextEpisode() {
    const { world, rand } = this;
    this.collector.stop();
    this.cond = randomConditions(rand, this.highway, this.dense, this.fog);
    this.nextLane = 0; // when to pick a new preferred highway lane
    this.app.setConditions(this.cond);
    const onRoad = Object.keys(SCENARIOS).filter((id) => (SCENARIOS[id].road ?? 'city') === this.cond.road && (!this.only || this.only.includes(id)));
    this.scenario = onRoad.length && rand() < this.scenarios ? pick(rand, onRoad) : '';
    this.app.startScenario(this.scenario);
    const noisy = rand() < this.noise;
    world.steerNoise = noisy ? makeSteerNoise(rand) : null;
    this.episode++;
    this.run = `${this.stamp}-s${this.seed}-${this.driver === 'neural' ? 'dagger' : 'expert'}-e${String(this.episode).padStart(3, '0')}`;
    this.collector.start(this.run, { seed: this.seed, episode: this.episode, ...this.cond, scenario: this.scenario || null, noise: noisy, driver: this.driver, hz: this.hz });
    this.t0 = world.t;
    this.ended = null;
    this.nextCapture = world.t;
  }

  // A capture is due before stepping past time t.
  due(t) {
    return !this.done && t >= this.nextCapture;
  }

  // After a capture attempt at the due time (ok: a sample was actually taken).
  captured(ok) {
    if (ok) this.total++;
    this.nextCapture = Math.max(this.nextCapture + 1 / this.hz, this.world.t);
  }

  // Once per frame, after stepping.
  update() {
    if (this.done) return;
    const w = this.world;
    if (this.total >= this.frames) return this.stop();
    const run = w.scenario;
    // On the highway every 15-40 s the expert prefers a new lane (or keeping right), so the data
    // covers every lane and plenty of lane changes, not just cruising in the right lane.
    if (!this.scenario && w.road === 'highway' && w.expert.lc && w.t >= this.nextLane) {
      w.expert.lc.preferLane = this.rand() < 0.25 ? null : Math.floor(this.rand() * 3);
      this.nextLane = w.t + 15 + this.rand() * 25;
    }
    if (this.scenario) {
      // Keep recording a couple of seconds past the outcome, then move on.
      if (!run || run.status !== 'running') this.ended ??= w.t;
      if (this.ended !== null && w.t - this.ended > 2) this.nextEpisode();
    } else if (w.t - this.t0 > this.episodeSeconds) this.nextEpisode();
  }

  stop() {
    this.collector.stop();
    this.world.steerNoise = null;
    this.done = true;
  }

  get status() {
    const c = this.cond;
    return { total: this.total, frames: this.frames, episode: this.episode, run: this.run, weather: c?.weather, hour: c?.hour, scenario: this.scenario, noise: !!this.world.steerNoise, done: this.done, error: this.collector.error };
  }
}

// Free-drive conditions for the benchmark: a spread of weather and light.
export const BENCH_DRIVES = [
  { label: 'Clear · day', weather: 'clear', hour: 14 },
  { label: 'Clear · night', weather: 'clear', hour: 21.5 },
  { label: 'Rain · dusk', weather: 'rain', hour: 18 },
  { label: 'Fog · morning', weather: 'fog', hour: 9 },
  { label: 'Snow · day', weather: 'snow', hour: 12 },
  { label: 'Highway · day', road: 'highway', weather: 'clear', hour: 13 },
  { label: 'Highway · rain, night', road: 'highway', weather: 'rain', hour: 21 },
];

export class Benchmark {
  constructor(app, neural, { trials = 2, driveSeconds = 90, cars = 40, peds = 70, vans = 4, hwCars = 60 } = {}) {
    Object.assign(this, { app, neural, driveSeconds });
    const traffic = { cars, peds, vans, hwCars };
    this.items = [];
    for (const id of Object.keys(SCENARIOS)) {
      for (let k = 0; k < trials; k++) this.items.push({ kind: 'scenario', id, name: SCENARIOS[id].name, trial: k + 1, cond: { weather: 'clear', hour: 14, ...traffic } });
    }
    for (const d of BENCH_DRIVES) this.items.push({ kind: 'drive', id: d.label, name: d.label, cond: { road: d.road ?? 'city', weather: d.weather, hour: d.hour, ...traffic } });
    this.results = [];
    this.index = -1;
    this.done = false;
    this.started = new Date().toISOString();
  }

  get world() {
    return this.app.world;
  }

  get current() {
    return this.items[this.index] ?? null;
  }

  start() {
    this.next();
  }

  next() {
    this.index++;
    const item = this.current;
    if (!item) {
      this.done = true;
      this.summary = summarize(this.results);
      return;
    }
    const w = this.world;
    this.app.setConditions(item.cond);
    this.app.startScenario(item.kind === 'scenario' ? item.id : '');
    this.neural.reset();
    w.safety.takeover = 0;
    this.mark = { t: w.t, events: w.safety.events.length, auto: w.safety.autoDist, total: w.safety.totalDist, contacts: w.contacts };
  }

  update() {
    if (this.done) return;
    const w = this.world, item = this.current, m = this.mark;
    let finished = false, extra = {};
    if (item.kind === 'scenario') {
      const run = w.scenario;
      if (run && run.status !== 'running') {
        finished = true;
        extra = { status: run.status, message: run.message, time: run.t, closest: Number.isFinite(run.closest) ? run.closest : null };
      }
    } else if (w.t - m.t >= this.driveSeconds) {
      finished = true;
      extra = { time: w.t - m.t };
    }
    if (!finished) return;
    const events = w.safety.events.slice(m.events);
    this.results.push({
      kind: item.kind, id: item.id, name: item.name, trial: item.trial ?? null, road: item.kind === 'scenario' ? (SCENARIOS[item.id].road ?? 'city') : item.cond.road, weather: item.cond.weather, hour: item.cond.hour, ...extra,
      takeovers: events.length, reasons: events.map((e) => e.reason),
      autoDist: w.safety.autoDist - m.auto, dist: w.safety.totalDist - m.total,
      contacts: item.kind === 'scenario' ? w.contacts : w.contacts - m.contacts,
    });
    this.next();
  }

  get progress() {
    return { index: Math.min(this.index, this.items.length), total: this.items.length, item: this.current };
  }
}

export function summarize(results) {
  const sc = results.filter((r) => r.kind === 'scenario'), dr = results.filter((r) => r.kind === 'drive');
  const sum = (xs, f) => xs.reduce((a, r) => a + f(r), 0);
  const takeovers = sum(dr, (r) => r.takeovers), auto = sum(dr, (r) => r.autoDist), dist = sum(dr, (r) => r.dist);
  return {
    scenarios: sc.length, passed: sc.filter((r) => r.status === 'passed').length,
    clean: sc.filter((r) => r.status === 'passed' && r.takeovers === 0).length,
    scenarioTakeovers: sum(sc, (r) => r.takeovers),
    driveKm: dist / 1000, autonomy: dist ? auto / dist : 0, takeovers,
    metersPerTakeover: takeovers ? auto / takeovers : null,
    contacts: sum(results, (r) => r.contacts),
  };
}
