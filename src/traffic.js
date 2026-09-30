// Road users other than the ego vehicle.
import { LANE_W, ROAD_W, PITCH, STOP_LINE, GRID, nodePos, inGrid } from './config.js';
import { randomBodyType, randomPaint } from './bodytypes.js';
import { Route, IDM, Overtaker, curveSpeed, signalObstacle, pathAhead, pathObstacle, longitudinal, makeLeftTurnYield } from './planner.js';
import { HW, HW_LENGTH, HighwayRoute, LaneChanger, HAZARD_HANG, loopDist, laneLeader, aheadOnCarriageway } from './highway.js';

// Parked cars fill a fraction of the marked parking stalls (see City.buildMarkings).
export function placeParkedCars(city, fleet, rand, density = 0.3) {
  const parked = [];
  const lateral = LANE_W + (ROAD_W / 2 - LANE_W) / 2; // parking lane center
  const s0 = STOP_LINE + 0.5, s1 = PITCH - STOP_LINE - 0.5;
  for (const seg of city.segments()) {
    const A = nodePos(...seg.a);
    for (const side of [1, -1]) {
      // Stalls are 6.5 m long between the ticks.
      for (let a = s0 + 6; a + 6.5 < s1 - 3; a += 6.5) {
        if (rand() > density) continue;
        const type = randomBodyType(rand);
        if (type === 'van' && rand() < 0.5) continue;
        const h = fleet.acquire(type, randomPaint(rand));
        if (!h) continue;
        const along = a + 3.25 + (rand() - 0.5) * 0.8;
        const off = side * (lateral + (rand() - 0.5) * 0.25);
        // Cars park facing the direction of travel on their side of the street.
        const [x, z, heading] = seg.axis === 'ew'
          ? [A.x + along, A.z + off, side > 0 ? 0 : Math.PI]
          : [A.x - off, A.z + along, side > 0 ? Math.PI / 2 : -Math.PI / 2];
        const jitter = (rand() - 0.5) * 0.04;
        fleet.set(h, x, z, heading + jitter, false);
        parked.push({ kind: 'car', parked: true, x, z, h: heading, v: 0, halfLen: h.spec.L / 2, handle: h });
      }
    }
  }
  return parked;
}

// ---------- moving traffic ----------

const MIN_SPAWN_GAP = 16; // meters to any other agent
const EGO_SPAWN_CLEARANCE = 45;
const HW_SPAWN_GAP = 40; // same carriageway and lane, bumper to bumper
const HW_EGO_CLEARANCE = 90;
const NO_SIGNAL = { gap: Infinity, stop: null };

let nextId = 1;

class NpcCar {
  // opts (scripted vehicles): v, vmax, hold, hazard, slow, parked, ignoreSignals, speedFactor
  constructor(route, handle, rand, opts = {}) {
    this.route = route;
    this.handle = handle;
    this.s = 0;
    this.k = 0;
    this.v = opts.v ?? 0;
    this.stuck = 0;
    this.braking = false;
    this.scripted = !!opts.scripted;
    this.vmax = opts.vmax ?? Infinity;
    this.hold = !!opts.hold; // stationary until released
    this.forceAcc = null; // scripted acceleration override (emergency braking)
    this.ignoreSignals = !!opts.ignoreSignals;
    // Driver personality: some drive a little faster or keep longer gaps.
    this.speedFactor = opts.speedFactor ?? 0.85 + rand() * 0.25;
    this.idm = { ...IDM, T: 1.0 + rand() * 0.8, a: 1.2 + rand() * 0.8 };
    this.ot = new Overtaker(handle.spec.L / 2);
    this.hw = !!route.highway;
    // Highway drivers: their own cruising speed, and lane changes unless scripted.
    this.desired = opts.desired ?? (this.hw ? HW.speed * (0.85 + rand() * 0.25) : Infinity);
    this.lc = this.hw && opts.laneChanges !== false ? new LaneChanger({ politeness: 0.1 + rand() * 0.4, keepRight: handle.type === 'van' ? 0.5 : 0.25 + rand() * 0.1, rand }) : null;
    const p = route.at(0);
    this.agent = {
      id: nextId++, kind: 'car', x: p.x, z: p.z, h: p.h, v: this.v, halfLen: handle.spec.L / 2, s: 0, leg: route.legAt(0),
      hazard: !!opts.hazard, slow: !!opts.slow, parked: !!opts.parked,
    };
  }

  // Place the agent on the route, including any overtaking offset.
  pose() {
    const r = this.route, off = this.ot.offsetAt;
    const p = r.at(this.s, this.k), o = off(this.s);
    const slope = this.ot.active ? (off(this.s + 1) - off(this.s - 1)) / 2 : 0;
    Object.assign(this.agent, {
      x: p.x + Math.sin(p.h) * o, z: p.z - Math.cos(p.h) * o, h: p.h - Math.atan(slope),
      v: this.v, s: this.s, leg: r.legAt(this.s),
    });
    if (this.hw) this.agent.hw = r.stateAt(this.s, this.k);
  }
}

export class Traffic {
  constructor(fleet, signals, rand) {
    this.fleet = fleet;
    this.signals = signals;
    this.rand = rand;
    this.cars = [];
    this.respawns = 0;
    this.avoid = []; // {x, z, r}: keep random spawns out of these areas (scenarios)
    this.time = 0;
  }

  get agents() {
    return this.cars.map((c) => c.agent);
  }

  // Background cars in the city streets.
  get background() {
    return this.cars.filter((c) => !c.scripted && !c.hw);
  }

  // Background cars on the highway.
  get highway() {
    return this.cars.filter((c) => !c.scripted && c.hw);
  }

  // A car on a random highway lane, clear of other traffic and of the ego.
  spawnHighway(ego) {
    const r = this.rand;
    for (let attempt = 0; attempt < 30; attempt++) {
      const dir = r() < 0.5 ? 1 : -1, lane = Math.floor(r() * HW.lanes), q = r() * HW_LENGTH;
      const route = new HighwayRoute({ dir, lane, q });
      const p = route.pts[0], hw = route.stateAt(0);
      if (ego?.hw && ego.hw.dir === dir && Math.abs(loopDist(ego.hw.q, q)) < HW_EGO_CLEARANCE) continue;
      if (ego && Math.hypot(p.x - ego.x, p.z - ego.z) < EGO_SPAWN_CLEARANCE) continue;
      if (this.avoid.some((a) => Math.hypot(p.x - a.x, p.z - a.z) < a.r)) continue;
      const near = this.cars.some((c) => c.hw && c.agent.hw?.dir === dir && Math.abs(c.agent.hw.lat - hw.lat) < HW.laneW && Math.abs(loopDist(c.agent.hw.q, q)) < HW_SPAWN_GAP);
      if (near) continue;
      const type = randomBodyType(r);
      const handle = this.fleet.acquire(type, randomPaint(r));
      if (!handle) return null;
      // Vans stand in for trucks: slower, so there's always someone to pass.
      const car = new NpcCar(route, handle, r, type === 'van' ? { desired: HW.speed * (0.7 + r() * 0.1) } : {});
      car.v = car.desired * 0.9;
      car.pose();
      this.cars.push(car);
      return car;
    }
    return null;
  }

  setHighwayCount(n, ego) {
    let hw = this.highway;
    while (hw.length > n) {
      this.remove(hw[hw.length - 1]);
      hw = this.highway;
    }
    let guard = 0;
    while (this.highway.length < n && guard++ < n * 3) this.spawnHighway(ego);
  }

  spawn(others, ego) {
    for (let attempt = 0; attempt < 30; attempt++) {
      const route = new Route(this.rand);
      const p = route.pts[0];
      if (ego && Math.hypot(p.x - ego.x, p.z - ego.z) < EGO_SPAWN_CLEARANCE) continue;
      if (others.some((a) => a && Math.hypot(p.x - a.x, p.z - a.z) < MIN_SPAWN_GAP)) continue;
      if (this.avoid.some((a) => Math.hypot(p.x - a.x, p.z - a.z) < a.r)) continue;
      const handle = this.fleet.acquire(randomBodyType(this.rand), randomPaint(this.rand));
      if (!handle) return null;
      const car = new NpcCar(route, handle, this.rand);
      car.v = 4 + this.rand() * 4;
      this.cars.push(car);
      return car;
    }
    return null;
  }

  // A vehicle with a given route and behavior (scenarios, double-parked vans).
  spawnScripted(route, { type = 'sedan', paint = 0x8f9499, ...opts } = {}) {
    const handle = this.fleet.acquire(type, paint);
    if (!handle) return null;
    const car = new NpcCar(route, handle, this.rand, { ...opts, scripted: true });
    car.pose();
    this.cars.push(car);
    return car;
  }

  remove(car) {
    const i = this.cars.indexOf(car);
    if (i < 0) return;
    this.cars.splice(i, 1);
    this.fleet.release(car.handle);
  }

  clearScripted() {
    for (const c of this.cars.filter((c) => c.scripted)) this.remove(c);
  }

  // Background cars count (scripted vehicles don't count).
  setCount(n, ego) {
    let bg = this.background;
    while (bg.length > n) {
      this.remove(bg[bg.length - 1]);
      bg = this.background;
    }
    let guard = 0;
    while (this.background.length < n && guard++ < n * 3) this.spawn([...this.agents, ego], ego);
  }

  // Move background cars out of the given areas (they respawn elsewhere).
  clearNear(areas, ego) {
    for (const c of this.cars.filter((c) => !c.scripted)) {
      if (areas.some((a) => Math.hypot(c.agent.x - a.x, c.agent.z - a.z) < a.r)) this.respawn(c, ego);
    }
  }

  respawn(car, ego) {
    this.remove(car);
    this.respawns++;
    if (car.hw) this.spawnHighway(ego);
    else this.spawn([...this.agents, ego], ego);
  }

  // Double-parked delivery vans with hazard lights, mid-block where they can be passed.
  addDoubleParked(n, ego = null) {
    const r = this.rand;
    for (let k = 0; k < n; k++) {
      for (let attempt = 0; attempt < 20; attempt++) {
        const i = Math.floor(r() * GRID), j = Math.floor(r() * GRID);
        const d = [[1, 0], [-1, 0], [0, 1], [0, -1]][Math.floor(r() * 4)];
        if (!inGrid(i + d[0], j + d[1])) continue;
        // Half in the parking lane, far enough from the next stop line to pass it.
        const route = new Route(r, { start: { i, j, d, along: 32 + r() * 14, lateral: 0.5 } });
        const p = route.pts[0];
        if (this.agents.some((a) => Math.hypot(p.x - a.x, p.z - a.z) < 30)) continue;
        if (ego && Math.hypot(p.x - ego.x, p.z - ego.z) < EGO_SPAWN_CLEARANCE) continue;
        if (this.avoid.some((a) => Math.hypot(p.x - a.x, p.z - a.z) < a.r)) continue;
        this.spawnScripted(route, { type: 'van', paint: 0xf2f2f0, hold: true, hazard: true });
        break;
      }
    }
  }

  step(dt, agents, ego) {
    this.time += dt;
    for (const c of [...this.cars]) {
      if (c.hw) {
        this.stepHighway(c, dt, agents, ego);
        continue;
      }
      const r = c.route;
      r.ensure(c.s + 150);
      if (c.k > 300) c.k -= r.trim(c.s);
      while (c.k < r.pts.length - 2 && r.pts[c.k + 1].s <= c.s) c.k++;
      if (c.hold) {
        c.v = 0;
        c.braking = false;
        c.pose();
        continue;
      }

      const v0 = Math.min(curveSpeed(r, c.s, c.k) * c.speedFactor, c.vmax);
      const sig = c.ignoreSignals ? NO_SIGNAL : signalObstacle(r, c.s, c.v, this.signals, makeLeftTurnYield(agents, c.agent));
      const hl = c.agent.halfLen;
      let obs = pathObstacle(pathAhead(r, c.s, c.k, 40, 1.5, c.ot.active ? c.ot.offsetAt : null, hl), agents, c.agent);
      const hang = c.ot.plan({ route: r, s: c.s, k: c.k, v: c.v, v0, obs, agents, self: c.agent, dt });
      if (hang && Number.isFinite(obs.gap)) obs = { ...obs, gap: Math.max(0, obs.gap - hang) };
      let { acc, reason } = longitudinal(c.v, v0, sig, obs, c.idm);
      if (c.forceAcc !== null) acc = c.forceAcc;
      c.v = Math.max(0, c.v + acc * dt);
      c.s += c.v * dt;
      c.braking = acc < -0.8 || c.v < 0.1;
      c.pose();

      // Deadlock breaker: stopped for a long time without a red light ahead.
      const atSignal = reason === 'signal' && sig.stop && sig.stop.state !== 'green';
      c.stuck = c.v < 0.1 && !atSignal ? c.stuck + dt : 0;
      if (c.stuck > 40 && !c.scripted) this.respawn(c, ego);
    }
  }

  // IDM along the lane (looking further ahead at speed) and MOBIL lane changes.
  stepHighway(c, dt, agents, ego) {
    const r = c.route;
    r.ensure(c.s + 200);
    if (c.k > 300) c.k -= r.trim(c.s);
    while (c.k < r.pts.length - 2 && r.pts[c.k + 1].s <= c.s) c.k++;
    if (c.hold) {
      c.v = 0;
      c.braking = false;
      c.pose();
      return;
    }
    // The road ahead changes slowly: refresh the curve speed every 0.1 s.
    c.curveT = (c.curveT ?? 0) - dt;
    if (c.curveT <= 0 || c.curveV === undefined) (c.curveV = curveSpeed(r, c.s, c.k, c.desired)), (c.curveT = 0.1);
    const v0 = Math.min(c.curveV, c.vmax);
    const hl = c.agent.halfLen;
    const horizon = Math.min(140, Math.max(45, c.v * 4.5));
    let obs = pathObstacle(pathAhead(r, c.s, c.k, horizon, 2, null, hl), aheadOnCarriageway(c.agent, agents, horizon), c.agent);
    c.lc?.update({ route: r, s: c.s, k: c.k, v: c.v, v0, agents, self: c.agent, p: c.idm, dt });
    obs = laneLeader(obs, c.agent, r.lane, agents);
    if (obs.agent?.hazard && Number.isFinite(obs.gap)) obs = { ...obs, gap: Math.max(0, obs.gap - HAZARD_HANG) };
    let { acc } = longitudinal(c.v, v0, NO_SIGNAL, obs, c.idm);
    if (c.forceAcc !== null) acc = c.forceAcc;
    c.v = Math.max(0, c.v + acc * dt);
    c.s += c.v * dt;
    c.braking = acc < -0.8 || c.v < 0.1;
    c.pose();
    c.stuck = c.v < 0.1 ? c.stuck + dt : 0;
    if (c.stuck > 40 && !c.scripted) this.respawn(c, ego);
  }

  sync() {
    const blink = Math.floor(this.time / 0.4) % 2 === 0;
    for (const c of this.cars) {
      const lights = c.agent.hazard ? (blink ? 'hazard' : false) : c.braking && !c.agent.parked;
      this.fleet.set(c.handle, c.agent.x, c.agent.z, c.agent.h, lights);
    }
  }
}
