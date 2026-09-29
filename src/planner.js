// Routes on the street grid plus the driving logic shared by every vehicle:
// curvature-aware desired speed, signal compliance with left-turn yielding,
// obstacle detection along the planned path, and IDM car-following.
import { LANE_W, ROAD_W, PITCH, STOP_LINE, GRID, WHEELBASE, MAX_STEER, nodePos, inGrid, clamp } from './config.js';

export const CRUISE = 11; // m/s (~40 km/h, city speed)
export const HALF_LEN = 2.35; // center to front bumper
const A_LAT = 2.2; // comfortable lateral accel
const A_DEC = 2.5; // comfortable decel used for curve speed planning
const LANE = LANE_W / 2;
export const angleWrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// Intelligent Driver Model parameters.
export const IDM = { a: 1.6, b: 2.5, T: 1.3, s0: 2.0, delta: 4 };

export class Route {
  // opts.along: distance from the start node to the spawn point (random if omitted).
  constructor(rand, opts = {}) {
    this.rand = rand;
    this.pts = []; // {x, z, s}
    this.stops = []; // {s, node, axis, d, turn, decision}
    this.turns = []; // {s, kind}
    this.legs = []; // straight approaches: {s0, s1, node, d, stop}
    let i, j, dirs;
    do {
      i = Math.floor(rand() * GRID);
      j = Math.floor(rand() * GRID);
      dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]].filter(([dx, dz]) => inGrid(i + dx, j + dz));
    } while (!dirs.length);
    const [dx, dz] = dirs[Math.floor(rand() * dirs.length)];
    this.node = [i + dx, j + dz];
    this.d = [dx, dz];
    const along = opts.along ?? ROAD_W / 2 + 12 + rand() * (PITCH - ROAD_W - 30);
    const A = nodePos(i, j), rx = -dz, rz = dx;
    this.push(A.x + dx * along + rx * LANE, A.z + dz * along + rz * LANE);
    this.extendStraight();
  }

  get length() {
    return this.pts[this.pts.length - 1].s;
  }

  push(x, z) {
    const last = this.pts[this.pts.length - 1];
    const s = last ? last.s + Math.hypot(x - last.x, z - last.z) : 0;
    if (last && s - last.s < 1e-3) return;
    this.pts.push({ x, z, s });
  }

  line(x1, z1, step = 1) {
    const last = this.pts[this.pts.length - 1];
    const n = Math.max(1, Math.ceil(Math.hypot(x1 - last.x, z1 - last.z) / step));
    for (let k = 1; k <= n; k++) this.push(last.x + ((x1 - last.x) * k) / n, last.z + ((z1 - last.z) * k) / n);
  }

  // Drive up to the entry of this.node, registering the stop line and the approach leg.
  extendStraight() {
    const s0 = this.length;
    const [dx, dz] = this.d, rx = -dz, rz = dx;
    const B = nodePos(...this.node);
    this.line(B.x - dx * (ROAD_W / 2) + rx * LANE, B.z - dz * (ROAD_W / 2) + rz * LANE);
    const stop = { s: this.length - (STOP_LINE - ROAD_W / 2), node: this.node, axis: dx ? 'ew' : 'ns', d: this.d, turn: null, decision: null };
    this.stops.push(stop);
    this.legs.push({ s0, s1: this.length, node: this.node, d: this.d, stop });
  }

  // Choose a maneuver at this.node, drive through the intersection, then to the next node.
  extend() {
    const [dx, dz] = this.d;
    const [i, j] = this.node;
    const options = [
      { d: [dx, dz], kind: 'straight', w: 0.45 },
      { d: [-dz, dx], kind: 'right', w: 0.3 },
      { d: [dz, -dx], kind: 'left', w: 0.3 },
    ].filter((o) => inGrid(i + o.d[0], j + o.d[1]));
    let r = this.rand() * options.reduce((a, o) => a + o.w, 0);
    const choice = options.find((o) => (r -= o.w) < 0) ?? options[0];

    const B = nodePos(i, j);
    const [ex, ez] = [this.pts[this.pts.length - 1].x, this.pts[this.pts.length - 1].z];
    const [d2x, d2z] = choice.d, r2x = -d2z, r2z = d2x;
    const X = { x: B.x + d2x * (ROAD_W / 2) + r2x * LANE, z: B.z + d2z * (ROAD_W / 2) + r2z * LANE };
    this.turns.push({ s: this.length, kind: choice.kind });
    this.stops[this.stops.length - 1].turn = choice.kind;
    if (choice.kind === 'straight') {
      this.line(X.x, X.z);
    } else {
      // Quadratic Bezier through the corner where the two lane lines meet.
      const t = (X.x - ex) * dx + (X.z - ez) * dz;
      const C = { x: ex + dx * t, z: ez + dz * t };
      for (let k = 1; k <= 16; k++) {
        const u = k / 16, a = (1 - u) ** 2, b = 2 * u * (1 - u), c = u * u;
        this.push(a * ex + b * C.x + c * X.x, a * ez + b * C.z + c * X.z);
      }
    }
    this.d = choice.d;
    this.node = [i + d2x, j + d2z];
    this.extendStraight();
  }

  ensure(s) {
    while (this.length < s) this.extend();
  }

  // Drop geometry more than `keep` meters behind s. Returns how many points were removed
  // so callers can shift their index hints.
  trim(s, keep = 60) {
    let n = 0;
    while (n < this.pts.length - 2 && this.pts[n + 1].s < s - keep) n++;
    if (n > 0) this.pts.splice(0, n);
    const cut = s - keep;
    this.stops = this.stops.filter((st) => st.s > cut);
    this.turns = this.turns.filter((t) => t.s > cut);
    this.legs = this.legs.filter((l) => l.s1 + ROAD_W > cut);
    return n;
  }

  // The approach leg the vehicle is on (including the intersection box after it).
  legAt(s) {
    for (const l of this.legs) if (s < l.s1 + ROAD_W) return l;
    return this.legs[this.legs.length - 1];
  }

  // Point at arc length s (linear interpolation), searching from a hint index.
  at(s, hint = 0) {
    let k = Math.max(0, Math.min(hint, this.pts.length - 2));
    while (k > 0 && this.pts[k].s > s) k--;
    while (k < this.pts.length - 2 && this.pts[k + 1].s < s) k++;
    const a = this.pts[k], b = this.pts[k + 1];
    const u = clamp((s - a.s) / (b.s - a.s || 1), 0, 1);
    return { x: a.x + (b.x - a.x) * u, z: a.z + (b.z - a.z) * u, k, h: Math.atan2(b.z - a.z, b.x - a.x) };
  }
}

// ---------- shared driving logic ----------

// Desired speed: the lowest speed that still lets us slow down comfortably for upcoming curvature.
export function curveSpeed(route, s, k, vmax = CRUISE) {
  let vt = vmax;
  for (let d = 0; d < 60; d += 2) {
    const a = route.at(s + d, k), b = route.at(s + d + 4, k);
    const kappa = Math.abs(angleWrap(b.h - a.h)) / 4;
    const vCurve = Math.sqrt(A_LAT / Math.max(kappa, 1e-4));
    vt = Math.min(vt, Math.sqrt(vCurve * vCurve + 2 * A_DEC * Math.max(0, d - 2)));
  }
  return vt;
}

// Next stop line on the route and whether we must stop there. `mustYield(stop)` lets the caller
// hold a permissive left turn for oncoming traffic. gap is front bumper -> stop line.
export function signalObstacle(route, s, v, signals, mustYield = null) {
  for (const st of route.stops) {
    const dist = st.s - s - HALF_LEN;
    if (dist < -1.5) continue;
    if (dist > 90) break;
    const state = signals.state(st.node, st.axis);
    st.state = state;
    st.dist = dist;
    let stop = false;
    if (state === 'green') {
      st.decision = null;
    } else {
      // Latch the stop/go decision on yellow so we don't dither.
      if (!st.decision) st.decision = state === 'red' || (v * v) / (2 * 4.5) < dist + 0.5 ? 'stop' : 'go';
      stop = st.decision === 'stop';
    }
    st.yielding = !stop && dist > -1 && !!mustYield && mustYield(st);
    return { gap: stop || st.yielding ? Math.max(dist, 0) : Infinity, stop: st };
  }
  return { gap: Infinity, stop: null };
}

// Samples of the planned path ahead of the front bumper, reused by obstacle checks.
export function pathAhead(route, s, k, horizon = 45, step = 1.5) {
  const out = [];
  for (let d = 0; d <= horizon; d += step) {
    const p = route.at(s + HALF_LEN + d, k);
    out.push({ x: p.x, z: p.z, h: p.h, d });
  }
  return out;
}

// Closest agent (vehicle or pedestrian) that lies on our path ahead.
// Vehicles are tested at front, center and rear so oblique conflicts inside intersections
// are caught, not just agents whose center sits on our path.
// Returns gap (front bumper to the agent's near edge) and its speed along our path.
export function pathObstacle(samples, agents, self) {
  let best = { gap: Infinity, v: 0, agent: null };
  if (!samples.length) return best;
  const o = samples[0], reach = samples[samples.length - 1].d + 8;
  for (const a of agents) {
    if (a === self || a.parked) continue;
    if (Math.abs(a.x - o.x) > reach || Math.abs(a.z - o.z) > reach) continue;
    let pts, r;
    if (a.kind === 'car') {
      const e = a.halfLen - 0.9, cx = Math.cos(a.h) * e, cz = Math.sin(a.h) * e;
      pts = [[a.x, a.z], [a.x + cx, a.z + cz], [a.x - cx, a.z - cz]];
      r = 1.85;
    } else {
      pts = [[a.x, a.z]];
      r = 1.9;
    }
    const r2 = r * r;
    for (const q of samples) {
      if (q.d >= best.gap) break;
      if (pts.some(([x, z]) => (q.x - x) ** 2 + (q.z - z) ** 2 < r2)) {
        const gap = Math.max(0, q.d - (a.kind === 'car' ? 0.9 : 0.3));
        if (gap < best.gap) {
          const vAlong = a.kind === 'car' ? a.v * Math.max(0, Math.cos(a.h - q.h)) : 0;
          best = { gap, v: vAlong, agent: a };
        }
        break;
      }
    }
  }
  return best;
}

// IDM acceleration toward desired speed v0 with a leader `gap` meters ahead moving at vLead.
export function idm(v, v0, gap, vLead = 0, p = IDM) {
  const free = 1 - Math.pow(v / Math.max(v0, 0.1), p.delta);
  if (!Number.isFinite(gap)) return p.a * free;
  const sStar = p.s0 + Math.max(0, v * p.T + (v * (v - vLead)) / (2 * Math.sqrt(p.a * p.b)));
  return p.a * (free - (sStar / Math.max(gap, 0.05)) ** 2);
}

// Combined longitudinal decision: returns acceleration plus the reason for the binding constraint.
export function longitudinal(v, v0, sig, obs, p = IDM) {
  // Don't block the box: hold at the line if the vehicle ahead is crawling inside or just past
  // the intersection, so we can't get stranded in it when the cross street turns green.
  let sigGap = sig.gap;
  const st = sig.stop;
  if (st && !Number.isFinite(sigGap) && st.dist > -0.5 && st.dist < 20 && obs.agent?.kind === 'car' && obs.v < 2 && obs.gap < st.dist + ROAD_W + 7) {
    sigGap = Math.max(st.dist, 0);
    st.boxBlocked = true;
  } else if (st) {
    st.boxBlocked = false;
  }
  // Stop ~1 m before the line: IDM settles at s0, so shift the line by (s0 - 1).
  const aSig = idm(v, v0, sigGap + (p.s0 - 1), 0, p);
  const aObs = idm(v, v0, obs.gap, obs.v, p);
  const aFree = idm(v, v0, Infinity, 0, p);
  const acc = clamp(Math.min(aSig, aObs, aFree), -8, p.a);
  let reason = null;
  if (acc < aFree - 0.05) {
    if (aObs <= aSig) reason = obs.agent?.kind === 'ped' ? 'pedestrian' : 'vehicle';
    else reason = st?.yielding ? 'yield' : st?.boxBlocked ? 'box' : 'signal';
  }
  return { acc, reason };
}

// Should a vehicle about to turn left at `stop` wait for oncoming traffic?
export function makeLeftTurnYield(agents, self) {
  return (st) => {
    if (st.turn !== 'left' || st.dist > 30) return false;
    const [dx, dz] = st.d;
    for (const a of agents) {
      if (a === self || a.kind !== 'car' || a.parked || !a.leg) continue;
      const l = a.leg;
      if (l.node[0] !== st.node[0] || l.node[1] !== st.node[1]) continue;
      if (l.d[0] !== -dx || l.d[1] !== -dz) continue; // not oncoming
      const toCenter = l.s1 + ROAD_W / 2 - a.s;
      if (l.stop.turn === 'left') {
        // Opposing left turns cross near the center: whoever is already in the box goes first;
        // if both are at the line, the lower id goes first.
        const inBox = toCenter < ROAD_W / 2 + 1 && toCenter > -ROAD_W / 2;
        const atLine = toCenter < ROAD_W / 2 + 12 && toCenter >= ROAD_W / 2 + 1;
        if (inBox || (atLine && st.dist < 8 && a.id < self.id)) return true;
        continue;
      }
      // Gap acceptance: any oncoming car close to the box, or one arriving within ~6 s.
      const toBox = toCenter - ROAD_W / 2;
      if (toCenter > -ROAD_W / 2 && (toBox < 25 || toBox / Math.max(a.v, 0.1) < 6)) return true;
    }
    return false;
  };
}

// ---------- ego expert ----------
// Privileged driver for the ego vehicle: shares the logic above, adds pure-pursuit steering.
export class Expert {
  constructor(route, signals) {
    this.route = route;
    this.signals = signals;
    this.k = 0;
    this.s = 0;
    this.agent = { id: 0, kind: 'car', x: 0, z: 0, h: 0, v: 0, halfLen: HALF_LEN, s: 0, leg: null, ego: true };
  }

  // Project the car onto the route (monotonic forward search).
  track(car) {
    const pts = this.route.pts;
    const d2 = (k) => (pts[k].x - car.x) ** 2 + (pts[k].z - car.z) ** 2;
    let best = this.k, bd = d2(best);
    for (let k = this.k + 1; k < Math.min(pts.length, this.k + 60); k++) {
      const d = d2(k);
      if (d < bd) { bd = d; best = k; }
    }
    this.k = best;
    // Continuous arc length: project onto the segment after (or before) the nearest point.
    const a = pts[Math.max(0, best - 1)], b = pts[Math.min(pts.length - 1, best + 1)];
    const tx = b.x - a.x, tz = b.z - a.z, tl = Math.hypot(tx, tz) || 1;
    this.s = pts[best].s + ((car.x - pts[best].x) * tx + (car.z - pts[best].z) * tz) / tl;
    const p = this.route.at(this.s, best);
    this.lateral = -(car.x - p.x) * Math.sin(p.h) + (car.z - p.z) * Math.cos(p.h);
    Object.assign(this.agent, { x: car.x, z: car.z, h: car.h, v: car.v, s: this.s, leg: this.route.legAt(this.s) });
  }

  control(car, agents = []) {
    const r = this.route;
    r.ensure(this.s + 250);
    if (this.k > 400) this.k -= r.trim(this.s);
    this.track(car);

    // Pure pursuit.
    const Ld = 4 + 0.45 * car.v;
    const tgt = r.at(this.s + Ld, this.k);
    const alpha = angleWrap(Math.atan2(tgt.z - car.z, tgt.x - car.x) - car.h);
    const ld = Math.hypot(tgt.x - car.x, tgt.z - car.z);
    const steer = clamp(Math.atan((2 * WHEELBASE * Math.sin(alpha)) / ld) / MAX_STEER, -1, 1);

    const v0 = curveSpeed(r, this.s, this.k);
    const sig = signalObstacle(r, this.s, car.v, this.signals, makeLeftTurnYield(agents, this.agent));
    const obs = pathObstacle(pathAhead(r, this.s, this.k), agents, this.agent);
    const { acc, reason } = longitudinal(car.v, v0, sig, obs);
    // Hold the brake when stopped so the car doesn't creep.
    let throttle = acc >= 0 ? acc / 3.2 : acc / 7.5;
    if (car.v < 0.3 && acc < 0.2) throttle = -0.5;

    const nextTurn = r.turns.find((t) => t.s > this.s - 5);
    return {
      steer, throttle, acc, reason, target: tgt, signal: sig.stop, lead: Number.isFinite(obs.gap) ? obs : null,
      nextTurn: nextTurn && { kind: nextTurn.kind, dist: nextTurn.s - this.s },
    };
  }
}
