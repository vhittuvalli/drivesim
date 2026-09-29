// Route generation on the street grid and a privileged expert driver that follows it:
// pure-pursuit steering, curvature-aware speed profile, and stopping for signals.
import { LANE_W, ROAD_W, PITCH, STOP_LINE, GRID, WHEELBASE, MAX_STEER, nodePos, inGrid, clamp } from './config.js';

const CRUISE = 11; // m/s (~40 km/h, city speed)
const A_LAT = 2.2; // comfortable lateral accel
const A_DEC = 2.5; // comfortable decel for speed planning
const LANE = LANE_W / 2;
const angleWrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export class Route {
  constructor(rand) {
    this.rand = rand;
    this.pts = []; // {x, z, s}
    this.stops = []; // {s, node, axis, decision}
    this.turns = []; // {s, kind}
    // Start mid-block on an interior street.
    const i = 1 + Math.floor(rand() * (GRID - 2)), j = 1 + Math.floor(rand() * (GRID - 2));
    const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]].filter(([dx, dz]) => inGrid(i + dx, j + dz));
    const [dx, dz] = dirs[Math.floor(rand() * dirs.length)];
    this.node = [i + dx, j + dz];
    this.d = [dx, dz];
    const A = nodePos(i, j), rx = -dz, rz = dx;
    this.push(A.x + dx * (PITCH / 2) + rx * LANE, A.z + dz * (PITCH / 2) + rz * LANE);
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

  // Drive up to the entry of this.node, registering the stop line.
  extendStraight() {
    const [dx, dz] = this.d, rx = -dz, rz = dx;
    const B = nodePos(...this.node);
    this.line(B.x - dx * (ROAD_W / 2) + rx * LANE, B.z - dz * (ROAD_W / 2) + rz * LANE);
    this.stops.push({ s: this.length - (STOP_LINE - ROAD_W / 2), node: this.node, axis: dx ? 'ew' : 'ns', decision: null });
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

export class Expert {
  constructor(route, signals) {
    this.route = route;
    this.signals = signals;
    this.k = 0;
    this.s = 0;
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
  }

  signalLimit(car) {
    for (const st of this.route.stops) {
      const dist = st.s - this.s - 2.6; // front bumper to stop line
      if (dist < -2) continue;
      if (dist > 90) break;
      const state = this.signals.state(st.node, st.axis);
      st.state = state;
      st.dist = dist;
      if (state === 'green') { st.decision = null; return { limit: Infinity, stop: st }; }
      if (!st.decision) {
        const canStop = (car.v * car.v) / (2 * 4.5) < dist + 0.5;
        st.decision = state === 'red' || canStop ? 'stop' : 'go';
      }
      if (st.decision === 'go') return { limit: Infinity, stop: st };
      const room = Math.max(0, dist - 1.5); // aim to stop 1.5 m before the line
      return { limit: Math.sqrt(2 * A_DEC * room), stop: st, room };
    }
    return { limit: Infinity, stop: null };
  }

  control(car) {
    this.route.ensure(this.s + 250);
    this.track(car);
    const r = this.route;

    // Pure pursuit.
    const Ld = 4 + 0.45 * car.v;
    const tgt = r.at(this.s + Ld, this.k);
    const alpha = angleWrap(Math.atan2(tgt.z - car.z, tgt.x - car.x) - car.h);
    const ld = Math.hypot(tgt.x - car.x, tgt.z - car.z);
    const steer = clamp(Math.atan((2 * WHEELBASE * Math.sin(alpha)) / ld) / MAX_STEER, -1, 1);

    // Speed profile: the lowest speed that still lets us slow down for upcoming curvature.
    let vt = CRUISE;
    for (let d = 0; d < 60; d += 2) {
      const a = r.at(this.s + d, this.k), b = r.at(this.s + d + 4, this.k);
      const kappa = Math.abs(angleWrap(b.h - a.h)) / 4;
      const vCurve = Math.sqrt(A_LAT / Math.max(kappa, 1e-4));
      vt = Math.min(vt, Math.sqrt(vCurve * vCurve + 2 * A_DEC * Math.max(0, d - 2)));
    }
    const sig = this.signalLimit(car);
    vt = Math.min(vt, sig.limit);
    let throttle = clamp(0.7 * (vt - car.v), -1, 1);
    if (sig.room !== undefined && sig.limit <= vt + 1e-6) {
      // Kinematic braking to the stop point: a = v^2 / 2d.
      const need = (car.v * car.v) / (2 * Math.max(sig.room, 0.2));
      if (need > 0.6) throttle = Math.min(throttle, -need / 7.5);
      if (sig.room < 0.3 || (vt < 0.3 && car.v < 0.8)) throttle = -0.6;
    }

    const nextTurn = r.turns.find((t) => t.s > this.s - 5);
    return { steer, throttle, vt, target: tgt, signal: sig.stop, nextTurn: nextTurn && { kind: nextTurn.kind, dist: nextTurn.s - this.s } };
  }
}
