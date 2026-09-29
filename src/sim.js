// The simulation without rendering: signals, ego (expert or manual driver), NPC traffic,
// pedestrians, contact detection and scenarios. Shared by the browser app, the headless tests
// and the fuzzer; the caller supplies the ego car (with or without a mesh) and the fleet.
import { conditions, LANE_W, ROAD_W, PITCH, STOP_LINE, GRID, nodePos, inGrid } from './config.js';
import { Route, Expert, HALF_LEN } from './planner.js';
import { Signals } from './signals.js';
import { Traffic } from './traffic.js';
import { Crowd } from './peds.js';
import { SCENARIOS } from './scenarios.js';

export const WEATHER = {
  clear: { label: 'Clear', grip: 1, visibility: Infinity },
  rain: { label: 'Rain', grip: 0.7, visibility: 90 },
  fog: { label: 'Fog', grip: 0.95, visibility: 32 },
  snow: { label: 'Snow', grip: 0.5, visibility: 60 },
};

export function setWeather(name) {
  const w = WEATHER[name] ?? WEATHER.clear;
  conditions.grip = w.grip;
  conditions.visibility = w.visibility;
}

// Clearance between the ego footprint and another footprint (half-length hl, half-width hw,
// heading h) at (x, z): separating-axis distance in the ego frame, 0 when they overlap.
function clearance(ego, x, z, h, hl, hw) {
  const dx = x - ego.x, dz = z - ego.z, c = Math.cos(ego.h), s = Math.sin(ego.h);
  const lx = Math.abs(dx * c + dz * s), ly = Math.abs(-dx * s + dz * c);
  const rel = h - ego.h, rc = Math.abs(Math.cos(rel)), rs = Math.abs(Math.sin(rel));
  const gx = lx - HALF_LEN - hl * rc - hw * rs, gy = ly - 0.92 - hl * rs - hw * rc;
  return gx > 0 && gy > 0 ? Math.hypot(gx, gy) : Math.max(gx, gy, 0);
}

// Is point (px, pz) inside a box (half-length hl, half-width hw) centered on pose a?
function inBox(a, px, pz, hl, hw) {
  const dx = px - a.x, dz = pz - a.z;
  const lx = dx * Math.cos(a.h) + dz * Math.sin(a.h), ly = -dx * Math.sin(a.h) + dz * Math.cos(a.h);
  return Math.abs(lx) < hl && Math.abs(ly) < hw;
}

export class World {
  // car: object with x, z, h, v, step(dt, steer, throttle), reset(x, z, h, v).
  constructor({ rand, car, fleet, parked = [], cars = 40, peds = 70, doubleParked = 0 }) {
    this.rand = rand;
    this.car = car;
    this.fleet = fleet;
    this.parked = parked;
    this.signals = new Signals(rand);
    const route = new Route(rand);
    const p0 = route.pts[0], p1 = route.pts[1];
    car.reset(p0.x, p0.z, Math.atan2(p1.z - p0.z, p1.x - p0.x));
    this.expert = new Expert(route, this.signals);
    this.expert.track(car);
    this.traffic = new Traffic(fleet, this.signals, rand);
    this.traffic.setCount(cars, this.expert.agent);
    this.crowd = new Crowd(this.signals, rand);
    this.crowd.setCount(peds);
    this.doubleParked = doubleParked;
    this.traffic.addDoubleParked(doubleParked, this.expert.agent);
    this.t = 0;
    this.manual = null; // input source {read(dt, v) -> {steer, throttle}} while a human drives
    this.scenario = null;
    this.contacts = 0;
    this.lastContact = null;
    this.touching = new Set();
    this.ctrl = null;
    this.hiddenParked = [];
  }

  get ego() {
    return this.expert.agent;
  }

  step(dt) {
    this.t += dt;
    this.signals.update(dt);
    const vehicles = [this.expert.agent, ...this.traffic.agents];
    const agents = [...vehicles, ...this.crowd.agents];
    let c;
    if (this.manual) {
      this.expert.observe(this.car);
      const cmd = this.manual.read(dt, this.car.v);
      c = { steer: cmd.steer, throttle: cmd.throttle, acc: 0, reason: null, manual: true };
    } else {
      c = this.expert.control(this.car, agents, dt);
    }
    this.car.step(dt, c.steer, c.throttle);
    this.traffic.step(dt, agents, this.expert.agent);
    this.crowd.step(dt, vehicles);
    this.detectContacts();
    if (this.scenario) this.tickScenario(dt);
    this.ctrl = c;
    return c;
  }

  // Count new contacts between the ego footprint and any vehicle or pedestrian.
  detectContacts() {
    const ego = this.car;
    const mark = (key, hit, what) => {
      if (hit && !this.touching.has(key)) {
        this.contacts++;
        this.lastContact = { what, t: this.t };
      }
      if (hit) this.touching.add(key);
      else this.touching.delete(key);
    };
    const cars = [...this.traffic.agents, ...this.parked.filter((p) => !p.hidden)];
    for (const a of cars) {
      if (Math.abs(a.x - ego.x) > 8 || Math.abs(a.z - ego.z) > 8) continue;
      const e = a.halfLen - 0.75;
      const hit = [-e, 0, e].some((k) => inBox(ego, a.x + Math.cos(a.h) * k, a.z + Math.sin(a.h) * k, HALF_LEN + 0.6, 0.92 + 0.85));
      mark(`c${a.id ?? `${a.x},${a.z}`}`, hit, a.parked ? 'a parked car' : 'a vehicle');
    }
    for (const p of this.crowd.peds) {
      if (Math.abs(p.x - ego.x) > 5 || Math.abs(p.z - ego.z) > 5) continue;
      mark(`p${p.id}`, inBox(ego, p.x, p.z, HALF_LEN + 0.25, 0.92 + 0.25), 'a pedestrian');
    }
  }

  // ---------- manual driving ----------

  takeOver(input) {
    this.manual = input;
  }

  // Give control back to the expert; it needs a lane to start from.
  handBack() {
    const { route, reason } = Route.fromPose(this.rand, this.car.x, this.car.z, this.car.h);
    if (!route) return { ok: false, reason };
    this.expert.setRoute(route);
    this.expert.track(this.car);
    this.manual = null;
    return { ok: true };
  }

  // Put the car (stopped) on the closest lane that roughly matches its heading, then hand back.
  snapToLane() {
    const { x, z, h } = this.car;
    let best = null;
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        for (const d of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          if (!inGrid(i + d[0], j + d[1])) continue;
          const A = nodePos(i, j);
          const along = Math.min(PITCH - STOP_LINE - 3, Math.max(ROAD_W / 2 + 3, (x - A.x) * d[0] + (z - A.z) * d[1]));
          const p = this.lanePose({ i, j, d }, along);
          const turn = Math.abs(Math.atan2(Math.sin(p.h - h), Math.cos(p.h - h)));
          const cost = Math.hypot(p.x - x, p.z - z) + turn * 8;
          if (!best || cost < best.cost) best = { cost, i, j, d, along };
        }
      }
    }
    this.placeEgo(best);
    this.manual = null;
  }

  // ---------- scenario helpers ----------

  // Pose on the right-hand lane leaving node (i, j) in direction d, `lateral` meters right of it.
  lanePose({ i, j, d }, along, lateral = 0) {
    const A = nodePos(i, j), [dx, dz] = d, rx = -dz, rz = dx, o = LANE_W / 2 + lateral;
    return { x: A.x + dx * along + rx * o, z: A.z + dz * along + rz * o, h: Math.atan2(dz, dx) };
  }

  placeEgo({ i, j, d, along, v = 0, choose = null }) {
    const route = new Route(this.rand, { start: { i, j, d, along }, choose });
    const p0 = route.pts[0], p1 = route.pts[1];
    this.car.reset(p0.x, p0.z, Math.atan2(p1.z - p0.z, p1.x - p0.x), v);
    this.expert.setRoute(route);
    this.expert.track(this.car);
    this.touching.clear();
  }

  // start: {i, j, d, along, lateral?, merge?}; opts as Traffic.spawnScripted plus choose, v.
  spawnCar(start, { choose = null, ...opts } = {}) {
    return this.traffic.spawnScripted(new Route(this.rand, { start, choose }), opts);
  }

  spawnPed(x, z, path, opts) {
    return this.crowd.spawnScripted(x, z, path, opts);
  }

  // Keep background traffic and pedestrians out of areas [{x, z, r}] for this scenario.
  clearArea(areas) {
    this.traffic.avoid = areas;
    this.crowd.avoid = areas;
    this.traffic.clearNear(areas, this.expert.agent);
    this.crowd.clearNear(areas);
  }

  // Temporarily remove parked cars (e.g. where a scenario puts its own).
  hideParked(areas) {
    for (const p of this.parked) {
      if (p.hidden || !areas.some((a) => Math.hypot(p.x - a.x, p.z - a.z) < a.r)) continue;
      p.hidden = true;
      this.fleet.set(p.handle, 0, -1000, 0, false);
      this.hiddenParked.push(p);
    }
  }

  setDoubleParked(n) {
    for (const c of this.traffic.cars.filter((c) => c.scripted && c.agent.hazard && !c.scenario)) this.traffic.remove(c);
    this.doubleParked = n;
    this.traffic.addDoubleParked(n, this.expert.agent);
  }

  // ---------- scenarios ----------

  startScenario(id) {
    this.endScenario();
    const def = SCENARIOS[id];
    if (!def) return null;
    // Scenarios bring their own obstacles.
    for (const c of this.traffic.cars.filter((c) => c.scripted)) this.traffic.remove(c);
    const run = { id, def, t: 0, status: 'running', message: def.goal, flags: {} };
    this.scenario = run;
    const before = new Set(this.traffic.cars);
    def.setup(this, run);
    for (const c of this.traffic.cars) if (!before.has(c)) c.scenario = true;
    this.contacts = 0;
    this.touching.clear();
    return run;
  }

  endScenario() {
    if (!this.scenario) return;
    this.scenario = null;
    for (const c of this.traffic.cars.filter((c) => c.scenario)) this.traffic.remove(c);
    this.crowd.clearScripted();
    this.traffic.avoid = [];
    this.crowd.avoid = [];
    for (const p of this.hiddenParked) {
      p.hidden = false;
      this.fleet.set(p.handle, p.x, p.z, p.h, false);
    }
    this.hiddenParked = [];
    this.setDoubleParked(this.doubleParked);
  }

  tickScenario(dt) {
    const run = this.scenario;
    run.t += dt;
    run.def.update?.(this, run, dt);
    // Closest approach to any scenario actor, body to body.
    const car = this.car;
    for (const c of this.traffic.cars) {
      if (!c.scenario || c.agent.parked) continue;
      const a = c.agent;
      run.closest = Math.min(run.closest ?? Infinity, clearance(car, a.x, a.z, a.h, a.halfLen, c.handle.spec.W / 2));
    }
    for (const p of this.crowd.peds) {
      if (p.scripted) run.closest = Math.min(run.closest ?? Infinity, clearance(car, p.x, p.z, p.h, 0.25, 0.25));
    }
    if (run.status !== 'running') return;
    if (this.contacts > 0) {
      run.status = 'failed';
      run.message = `Contact with ${this.lastContact.what}`;
      return;
    }
    const ok = run.def.check(this, run);
    if (ok) {
      run.status = 'passed';
      run.message = ok;
    } else if (run.t > run.def.timeout) {
      run.status = 'failed';
      run.message = 'Timed out';
    }
  }
}
