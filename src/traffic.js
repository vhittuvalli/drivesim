// Road users other than the ego vehicle.
import { LANE_W, ROAD_W, PITCH, STOP_LINE, nodePos } from './config.js';
import { randomBodyType, randomPaint } from './bodytypes.js';
import { Route, IDM, curveSpeed, signalObstacle, pathAhead, pathObstacle, longitudinal, makeLeftTurnYield } from './planner.js';

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

let nextId = 1;

class NpcCar {
  constructor(route, handle, rand) {
    this.route = route;
    this.handle = handle;
    this.s = 0;
    this.k = 0;
    this.v = 0;
    this.stuck = 0;
    this.braking = false;
    // Driver personality: some drive a little faster or keep longer gaps.
    this.speedFactor = 0.85 + rand() * 0.25;
    this.idm = { ...IDM, T: 1.0 + rand() * 0.8, a: 1.2 + rand() * 0.8 };
    const p = route.at(0);
    this.agent = { id: nextId++, kind: 'car', x: p.x, z: p.z, h: p.h, v: 0, halfLen: handle.spec.L / 2, s: 0, leg: route.legAt(0) };
  }
}

export class Traffic {
  constructor(fleet, signals, rand) {
    this.fleet = fleet;
    this.signals = signals;
    this.rand = rand;
    this.cars = [];
    this.respawns = 0;
  }

  get agents() {
    return this.cars.map((c) => c.agent);
  }

  spawn(others, ego) {
    for (let attempt = 0; attempt < 30; attempt++) {
      const route = new Route(this.rand);
      const p = route.pts[0];
      if (ego && Math.hypot(p.x - ego.x, p.z - ego.z) < EGO_SPAWN_CLEARANCE) continue;
      if (others.some((a) => a && Math.hypot(p.x - a.x, p.z - a.z) < MIN_SPAWN_GAP)) continue;
      const handle = this.fleet.acquire(randomBodyType(this.rand), randomPaint(this.rand));
      if (!handle) return null;
      const car = new NpcCar(route, handle, this.rand);
      car.v = 4 + this.rand() * 4;
      this.cars.push(car);
      return car;
    }
    return null;
  }

  setCount(n, ego) {
    while (this.cars.length > n) this.fleet.release(this.cars.pop().handle);
    let guard = 0;
    while (this.cars.length < n && guard++ < n * 3) this.spawn([...this.agents, ego], ego);
  }

  respawn(car, ego) {
    const i = this.cars.indexOf(car);
    this.cars.splice(i, 1);
    this.fleet.release(car.handle);
    this.respawns++;
    this.spawn([...this.agents, ego], ego);
  }

  step(dt, agents, ego) {
    for (const c of [...this.cars]) {
      const r = c.route;
      r.ensure(c.s + 150);
      if (c.k > 300) c.k -= r.trim(c.s);
      while (c.k < r.pts.length - 2 && r.pts[c.k + 1].s <= c.s) c.k++;

      const v0 = curveSpeed(r, c.s, c.k) * c.speedFactor;
      const sig = signalObstacle(r, c.s, c.v, this.signals, makeLeftTurnYield(agents, c.agent));
      const obs = pathObstacle(pathAhead(r, c.s, c.k, 40), agents, c.agent);
      const { acc, reason } = longitudinal(c.v, v0, sig, obs, c.idm);
      c.v = Math.max(0, c.v + acc * dt);
      c.s += c.v * dt;
      c.braking = acc < -0.8 || c.v < 0.1;

      const p = r.at(c.s, c.k);
      Object.assign(c.agent, { x: p.x, z: p.z, h: p.h, v: c.v, s: c.s, leg: r.legAt(c.s) });

      // Deadlock breaker: stopped for a long time without a red light ahead.
      const atSignal = reason === 'signal' && sig.stop && sig.stop.state !== 'green';
      c.stuck = c.v < 0.1 && !atSignal ? c.stuck + dt : 0;
      if (c.stuck > 40) this.respawn(c, ego);
    }
  }

  sync() {
    for (const c of this.cars) this.fleet.set(c.handle, c.agent.x, c.agent.z, c.agent.h, c.braking);
  }
}
