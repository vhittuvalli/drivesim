// Pedestrian simulation (no rendering). Pedestrians walk the sidewalks between intersection
// corners, cross at crosswalks during the parallel green phase, and occasionally jaywalk.
import { ROAD_W, SIDEWALK_W, GRID, PITCH, nodePos, inGrid } from './config.js';

const WALK_LINE = ROAD_W / 2 + SIDEWALK_W / 2; // sidewalk centerline, from road center
const P_JAYWALK = 0.06; // chance per sidewalk edge
const REACHED = 0.25;

let nextId = 100000;

class Pedestrian {
  constructor(rand) {
    this.id = nextId++;
    this.lat = -1.1 + rand() * 2.4; // position across the sidewalk
    this.speed = 1.1 + rand() * 0.5;
    this.careless = rand() < 0.3;
    this.x = 0;
    this.z = 0;
    this.h = 0;
    this.v = 0;
    this.corner = null; // {i, j, sx, sz}
    this.prev = null;
    this.queue = []; // waypoints {x, z, mode, ...}
    this.wait = 0;
    this.agent = { id: this.id, kind: 'ped', x: 0, z: 0, h: 0, v: 0 };
  }

  pos(c) {
    const N = nodePos(c.i, c.j);
    const o = WALK_LINE + this.lat;
    return { x: N.x + c.sx * o, z: N.z + c.sz * o };
  }
}

const key = (c) => `${c.i},${c.j},${c.sx},${c.sz}`;

export class Crowd {
  constructor(signals, rand) {
    this.signals = signals;
    this.rand = rand;
    this.peds = [];
    this.jaywalks = 0;
    this.avoid = []; // {x, z, r}: keep random spawns out of these areas (scenarios)
  }

  get agents() {
    return this.peds.map((p) => p.agent);
  }

  // Background pedestrians count (scripted ones don't count).
  setCount(n) {
    const bg = () => this.peds.filter((p) => !p.scripted);
    while (bg().length > n) this.remove(bg().pop());
    while (bg().length < n) this.spawn();
  }

  remove(p) {
    const i = this.peds.indexOf(p);
    if (i >= 0) this.peds.splice(i, 1);
  }

  clearScripted() {
    this.peds = this.peds.filter((p) => !p.scripted);
  }

  // Move background pedestrians out of the given areas (they respawn elsewhere).
  clearNear(areas) {
    for (const p of this.peds.filter((p) => !p.scripted && areas.some((a) => Math.hypot(p.x - a.x, p.z - a.z) < a.r))) {
      this.remove(p);
      this.spawn();
    }
  }

  // A pedestrian at (x, z) that waits (mode 'hold') until the scenario sets queue[0].go, then
  // follows `path` (waypoints {x, z, mode}); afterwards walks to the nearest corner and
  // continues as a normal pedestrian.
  spawnScripted(x, z, path, { speed = 1.4 } = {}) {
    const p = new Pedestrian(this.rand);
    Object.assign(p, { x, z, speed, scripted: true });
    const end = path[path.length - 1];
    const i = Math.round(end.x / PITCH), j = Math.round(end.z / PITCH), N = nodePos(i, j);
    // The nearest corner of the block we end up on (never the far side of a road).
    const corner = { i, j, sx: Math.sign(end.x - N.x) || 1, sz: Math.sign(end.z - N.z) || 1 };
    p.corner = corner;
    p.queue = [{ x, z, mode: 'hold', go: false }, ...path, { ...p.pos(corner), mode: 'walk' }];
    const first = path[0];
    p.h = Math.atan2(first.z - z, first.x - x);
    this.sync(p);
    this.peds.push(p);
    return p;
  }

  spawn() {
    const r = this.rand;
    const p = new Pedestrian(r);
    let c, tries = 0;
    do {
      c = { i: Math.floor(r() * GRID), j: Math.floor(r() * GRID), sx: r() < 0.5 ? -1 : 1, sz: r() < 0.5 ? -1 : 1 };
      // Outward-facing edge corners are dead ends.
      if (!inGrid(c.i + c.sx, c.j) && !inGrid(c.i, c.j + c.sz)) continue;
      const q = p.pos(c);
      if (tries++ < 20 && this.avoid.some((a) => Math.hypot(q.x - a.x, q.z - a.z) < a.r)) continue;
      break;
    } while (true);
    p.corner = c;
    this.plan(p, { sidewalkOnly: true });
    // Start part-way along the first edge.
    const start = p.pos(c), goal = p.queue[0] ?? start;
    const u = r();
    p.x = start.x + (goal.x - start.x) * u;
    p.z = start.z + (goal.z - start.z) * u;
    this.peds.push(p);
    return p;
  }

  // Choose the next move from the current corner and queue its waypoints.
  plan(p, { sidewalkOnly = false } = {}) {
    const { i, j, sx, sz } = p.corner;
    const r = this.rand;
    const moves = [];
    // Along the block edge next to the east-west road, or the north-south road.
    if (inGrid(i + sx, j)) moves.push({ w: 0.35, to: { i: i + sx, j, sx: -sx, sz }, kind: 'walk', road: 'ew' });
    if (inGrid(i, j + sz)) moves.push({ w: 0.35, to: { i, j: j + sz, sx, sz: -sz }, kind: 'walk', road: 'ns' });
    if (!sidewalkOnly) {
      // Crossing the north-south arm means walking east-west: allowed on the EW green.
      if (inGrid(i, j + sz)) moves.push({ w: 0.15, to: { i, j, sx: -sx, sz }, kind: 'cross', axis: 'ew' });
      if (inGrid(i + sx, j)) moves.push({ w: 0.15, to: { i, j, sx, sz: -sz }, kind: 'cross', axis: 'ns' });
    }
    const forward = moves.filter((m) => !p.prev || key(m.to) !== key(p.prev)); // no U-turns
    const pool = forward.length ? forward : moves;
    let x = r() * pool.reduce((a, m) => a + m.w, 0);
    const m = pool.find((o) => (x -= o.w) < 0) ?? pool[0];

    const from = p.pos(p.corner), to = p.pos(m.to);
    if (m.kind === 'cross') {
      p.queue.push({ ...from, mode: 'wait', node: [i, j], axis: m.axis });
      p.queue.push({ ...to, mode: 'cross' });
    } else if (!sidewalkOnly && r() < P_JAYWALK) {
      // Jaywalk mid-block: step across to the opposite sidewalk of the adjacent road.
      const u = 0.3 + r() * 0.4;
      const mx = from.x + (to.x - from.x) * u, mz = from.z + (to.z - from.z) * u;
      const N = nodePos(i, j), o = WALK_LINE + p.lat;
      const far = m.road === 'ew' ? { x: mx, z: N.z - sz * o } : { x: N.x - sx * o, z: mz };
      const flipped = m.road === 'ew' ? { ...m.to, sz: -m.to.sz } : { ...m.to, sx: -m.to.sx };
      p.queue.push({ x: mx, z: mz, mode: 'walk' });
      // If traffic never clears, fall back to finishing the walk on this side.
      p.queue.push({ x: mx, z: mz, mode: 'jaywait', road: m.road, fallback: { corner: m.to, ...to } });
      p.queue.push({ ...far, mode: 'jaywalk' });
      p.queue.push({ ...p.pos(flipped), mode: 'walk' });
      m.to = flipped;
    } else {
      p.queue.push({ ...to, mode: 'walk' });
    }
    p.prev = p.corner;
    p.corner = m.to;
  }

  // Is any vehicle close enough to make a mid-block crossing unsafe?
  trafficNear(p, cars, road) {
    const radius = p.careless ? 12 : 30;
    for (const c of cars) {
      if (c.parked) continue;
      const along = road === 'ew' ? Math.abs(c.x - p.x) : Math.abs(c.z - p.z);
      const across = road === 'ew' ? Math.abs(c.z - p.z) : Math.abs(c.x - p.x);
      if (across < ROAD_W && along < radius) return true;
    }
    return false;
  }

  // A vehicle footprint (plus margin) right in front of us along the walking direction.
  blockedByVehicle(p, ux, uz, cars) {
    const px = p.x + ux * 1.0, pz = p.z + uz * 1.0;
    for (const c of cars) {
      const dx = px - c.x, dz = pz - c.z;
      if (Math.abs(dx) > 5 || Math.abs(dz) > 5) continue;
      const lx = dx * Math.cos(c.h) + dz * Math.sin(c.h), ly = -dx * Math.sin(c.h) + dz * Math.cos(c.h);
      if (Math.abs(lx) < c.halfLen + 0.5 && Math.abs(ly) < 1.4) return true;
    }
    return false;
  }

  step(dt, cars = []) {
    for (const p of this.peds) {
      const wp = p.queue[0];
      if (!wp) {
        this.plan(p);
        continue;
      }
      if (wp.mode === 'hold') {
        p.v = 0;
        if (wp.go) p.queue.shift();
        this.sync(p);
        continue;
      }
      if (wp.mode === 'wait') {
        // Start crossing only at the beginning of the parallel green.
        const ok = this.signals.state(wp.node, wp.axis) === 'green';
        p.v = 0;
        if (ok) p.queue.shift();
        this.sync(p);
        continue;
      }
      if (wp.mode === 'jaywait') {
        p.v = 0;
        p.wait += dt;
        if (!this.trafficNear(p, cars, wp.road)) {
          p.queue.shift();
          p.wait = 0;
          this.jaywalks++;
        } else if (p.wait > 6) {
          const f = wp.fallback;
          p.queue = [{ x: f.x, z: f.z, mode: 'walk' }];
          p.corner = f.corner;
          p.wait = 0;
        }
        this.sync(p);
        continue;
      }
      const dx = wp.x - p.x, dz = wp.z - p.z, d = Math.hypot(dx, dz);
      const onRoad = wp.mode === 'cross' || wp.mode === 'jaywalk';
      if (onRoad && d > REACHED && this.blockedByVehicle(p, dx / d, dz / d, cars)) {
        p.v = 0;
        this.sync(p);
        continue;
      }
      const speed = p.speed * (onRoad ? 1.15 : 1);
      if (d < REACHED) {
        p.queue.shift();
      } else {
        const stepLen = Math.min(d, speed * dt);
        p.x += (dx / d) * stepLen;
        p.z += (dz / d) * stepLen;
        const target = Math.atan2(dz, dx);
        const dh = Math.atan2(Math.sin(target - p.h), Math.cos(target - p.h));
        p.h += dh * Math.min(1, dt * 8);
      }
      p.v = d < REACHED ? 0 : speed;
      this.sync(p);
    }
  }

  sync(p) {
    Object.assign(p.agent, { x: p.x, z: p.z, h: p.h, v: p.v, crossing: p.queue[0]?.mode === 'cross' || p.queue[0]?.mode === 'jaywalk' });
  }
}
