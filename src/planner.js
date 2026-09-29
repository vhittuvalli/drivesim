// Routes on the street grid plus the driving logic shared by every vehicle:
// curvature-aware desired speed, signal compliance with left-turn yielding,
// obstacle detection along the planned path, and IDM car-following.
import { LANE_W, ROAD_W, PITCH, STOP_LINE, GRID, WHEELBASE, MAX_STEER, nodePos, inGrid, clamp, conditions } from './config.js';

export const CRUISE = 11; // m/s (~40 km/h, city speed)
export const HALF_LEN = 2.35; // center to front bumper
const A_LAT = 2.2; // comfortable lateral accel
const A_DEC = 2.5; // comfortable decel used for curve speed planning
const LANE = LANE_W / 2;
export const angleWrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// Intelligent Driver Model parameters.
export const IDM = { a: 1.6, b: 2.5, T: 1.3, s0: 2.0, delta: 4 };

// Less grip: longer headways, gentler braking targets.
function forConditions(p) {
  const g = conditions.grip;
  return g === 1 ? p : { ...p, T: p.T / g, b: p.b * g, a: p.a * Math.min(1, g * 1.3) };
}
const smooth = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));

export class Route {
  // opts.start: {i, j, d, along, lateral?, merge?} places the route on the lane leaving node (i, j)
  //   in direction d, `along` meters from the node center (random placement if omitted).
  //   `lateral` starts that many meters right of the lane center (e.g. in the parking lane) and
  //   blends into the lane over `merge` meters.
  // opts.along: distance from the start node to the spawn point for random placement.
  // opts.choose(route) may return 'straight' | 'left' | 'right' to force the next turn.
  constructor(rand, opts = {}) {
    this.rand = rand;
    this.choose = opts.choose ?? null;
    this.pts = []; // {x, z, s}
    this.stops = []; // {s, node, axis, d, turn, decision}
    this.turns = []; // {s, kind}
    this.legs = []; // straight approaches: {s0, s1, node, d, stop}
    let i, j, dx, dz, along;
    if (opts.start) {
      ({ i, j, along } = opts.start);
      [dx, dz] = opts.start.d;
    } else {
      let dirs;
      do {
        i = Math.floor(rand() * GRID);
        j = Math.floor(rand() * GRID);
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]].filter(([ddx, ddz]) => inGrid(i + ddx, j + ddz));
      } while (!dirs.length);
      [dx, dz] = dirs[Math.floor(rand() * dirs.length)];
      along = opts.along ?? ROAD_W / 2 + 12 + rand() * (PITCH - ROAD_W - 30);
    }
    this.node = [i + dx, j + dz];
    this.d = [dx, dz];
    const A = nodePos(i, j), rx = -dz, rz = dx;
    const lat = opts.start?.lateral ?? 0, merge = opts.start?.merge ?? 0;
    this.push(A.x + dx * along + rx * (LANE + lat), A.z + dz * along + rz * (LANE + lat));
    if (lat && merge) {
      // Pull out of the parking lane: quadratic Bezier that ends tangent to the lane.
      const P = (a, l) => ({ x: A.x + dx * a + rx * (LANE + l), z: A.z + dz * a + rz * (LANE + l) });
      const s0 = P(along, lat), C = P(along + merge * 0.5, 0), E = P(along + merge, 0);
      for (let k = 1; k <= 16; k++) {
        const u = k / 16, a = (1 - u) ** 2, b = 2 * u * (1 - u), c = u * u;
        this.push(a * s0.x + b * C.x + c * E.x, a * s0.z + b * C.z + c * E.z);
      }
    }
    this.extendStraight();
  }

  // Route along the lane the pose is on, or null (with a reason) if the pose isn't on a lane.
  static fromPose(rand, x, z, h, opts = {}) {
    const c = Math.cos(h), sn = Math.sin(h);
    const ew = Math.abs(c) >= Math.abs(sn);
    const [dx, dz] = ew ? [Math.sign(c), 0] : [0, Math.sign(sn)];
    const headingErr = Math.abs(angleWrap(h - Math.atan2(dz, dx)));
    if (headingErr > 0.6) return { route: null, reason: 'Line up with the lane first' };
    const t = ew ? x : z, cross = ew ? z : x, sg = ew ? dx : dz;
    const prev = sg > 0 ? Math.floor(t / PITCH) : Math.ceil(t / PITCH);
    const road = Math.round(cross / PITCH);
    const [i, j] = ew ? [prev, road] : [road, prev];
    if (!inGrid(i, j) || !inGrid(i + dx, j + dz)) return { route: null, reason: 'Drive back into the city grid' };
    const along = (t - prev * PITCH) * sg;
    const laneCenter = road * PITCH + (ew ? dx : -dz) * LANE; // right-hand lane
    if (Math.abs(cross - laneCenter) > 2.5) return { route: null, reason: 'Move into the right-hand lane' };
    if (along < ROAD_W / 2 || along > PITCH - STOP_LINE - 2) return { route: null, reason: 'Clear the intersection first' };
    return { route: new Route(rand, { ...opts, start: { i, j, d: [dx, dz], along } }), reason: null };
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
    let choice = options.find((o) => (r -= o.w) < 0) ?? options[0];
    const forced = this.choose?.(this);
    if (forced) choice = options.find((o) => o.kind === forced) ?? choice;

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
// Also capped by visibility: we must be able to stop within the distance we can see.
export function curveSpeed(route, s, k, vmax = CRUISE) {
  const g = conditions.grip;
  let vt = Math.min(vmax, Math.sqrt(2 * A_DEC * g * Math.max(conditions.visibility - 12, 4)));
  for (let d = 0; d < 60; d += 2) {
    const a = route.at(s + d, k), b = route.at(s + d + 4, k);
    const kappa = Math.abs(angleWrap(b.h - a.h)) / 4;
    const vCurve = Math.sqrt((A_LAT * g) / Math.max(kappa, 1e-4));
    vt = Math.min(vt, Math.sqrt(vCurve * vCurve + 2 * A_DEC * g * Math.max(0, d - 2)));
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
      if (!st.decision) st.decision = state === 'red' || (v * v) / (2 * 4.5 * conditions.grip) < dist + 0.5 ? 'stop' : 'go';
      stop = st.decision === 'stop';
    }
    st.yielding = !stop && dist > -1 && !!mustYield && mustYield(st);
    return { gap: stop || st.yielding ? Math.max(dist, 0) : Infinity, stop: st };
  }
  return { gap: Infinity, stop: null };
}

// Samples of the planned path ahead of the front bumper, reused by obstacle checks.
// offsetAt(s) shifts the path left of the lane (overtaking); halfLen is the vehicle's.
export function pathAhead(route, s, k, horizon = 45, step = 1.5, offsetAt = null, halfLen = HALF_LEN) {
  const out = [];
  for (let d = 0; d <= horizon; d += step) {
    const p = route.at(s + halfLen + d, k);
    const o = offsetAt ? offsetAt(s + halfLen + d) : 0;
    out.push({ x: p.x + Math.sin(p.h) * o, z: p.z - Math.cos(p.h) * o, h: p.h, d });
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
      // Anticipate pedestrians who are on the road and moving: they may step into our lane.
      r = a.crossing ? 3.5 : 1.9;
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

// Vehicles that are about to cut across our path (red-light runners, cars pulling out, turning
// traffic), found by constant-velocity prediction. Cars already on our path are left to
// pathObstacle, and so are cars that are going to stop at their own stop line.
export function crossingObstacle(samples, agents, self, v) {
  let best = { gap: Infinity, v: 0, agent: null, predicted: true, at: null };
  if (samples.length < 2) return best;
  const o = samples[0], fx = Math.cos(o.h), fz = Math.sin(o.h);
  for (const a of agents) {
    if (a === self || a.kind !== 'car' || a.parked || a.v < 1) continue;
    const dx = a.x - o.x, dz = a.z - o.z;
    if (Math.abs(dx) > 60 || Math.abs(dz) > 60 || dx * fx + dz * fz < -3) continue;
    const st = a.leg?.stop;
    if (st && st.s - (a.s ?? 0) > 0 && (st.decision === 'stop' || st.yielding || st.boxBlocked)) continue;
    if (samples.some((q) => (q.x - a.x) ** 2 + (q.z - a.z) ** 2 < 4)) continue;
    const cx = Math.cos(a.h) * a.v, cz = Math.sin(a.h) * a.v;
    for (const t of [0.5, 1, 1.5, 2, 2.5]) {
      const px = a.x + cx * t, pz = a.z + cz * t;
      for (const q of samples) {
        if (q.d >= best.gap) break;
        if (Math.cos(a.h - q.h) > 0.97) continue; // same direction: plain car following
        if (Math.abs(q.d / Math.max(v, 1.5) - t) > 1.5) continue; // we won't be there then
        if ((q.x - px) ** 2 + (q.z - pz) ** 2 < 2.4 * 2.4) {
          best = { gap: Math.max(0, q.d - 2), v: 0, agent: a, predicted: true, at: { x: px, z: pz } };
          break;
        }
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
export function longitudinal(v, v0, sig, obs, p0 = IDM) {
  const p = forConditions(p0);
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
  const acc = clamp(Math.min(aSig, aObs, aFree), -8 * conditions.grip, p.a);
  let reason = null;
  if (acc < aFree - 0.05) {
    if (aObs <= aSig) reason = obs.agent?.kind === 'ped' ? 'pedestrian' : obs.predicted ? 'crossing' : 'vehicle';
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

// ---------- overtaking ----------
// Passing a stopped (hazard lights) or slow vehicle on a two-lane street by borrowing the
// oncoming lane. Shared by the ego expert and NPC drivers. The maneuver is a lateral offset of
// the route as a function of arc length: ramp out, hold alongside, ramp back once our rear is
// clear of the passed vehicle. Before committing we check that the oncoming lane stays clear
// for the whole pass and that it finishes before the next stop line; until we're alongside,
// a newly appearing oncoming vehicle makes us abort back into our lane.

// Time and distance to gain `rel` meters on a leader at vl, accelerating from v toward vp.
function passPlan(v, vp, vl, rel) {
  let t = 0, x = 0, vv = v;
  while (x - vl * t < rel && t < 25) {
    vv = Math.min(vp, vv + 1.3 * 0.1);
    x += vv * 0.1;
    t += 0.1;
  }
  return { T: t, D: x };
}

// Is the oncoming lane clear for D meters of passing that takes T seconds?
// (p, fx, fz): our lane-center point and forward direction.
export function oncomingClear(p, fx, fz, agents, self, lead, D, T) {
  const lx = fz, lz = -fx, h = Math.atan2(fz, fx);
  for (const o of agents) {
    if (o === self || o === lead || o.parked) continue;
    const dx = o.x - p.x, dz = o.z - p.z;
    const ahead = dx * fx + dz * fz, lat = dx * lx + dz * lz;
    if (ahead < -20 || ahead > D + 160) continue;
    if (o.kind === 'ped') {
      if (o.crossing && ahead > -2 && ahead < D + 5 && lat > -3 && lat < 7) return { ok: false, by: o };
      continue;
    }
    if (lat < 1.3 || lat > 6) continue; // not in the oncoming lane
    const toward = Math.cos(o.h - h) < -0.3;
    if (toward ? ahead > -4 && ahead < D + o.v * T + 15 : ahead > -10 && ahead < D + 10) return { ok: false, by: o };
  }
  return { ok: true, by: null };
}

export class Overtaker {
  constructor(halfLen = HALF_LEN) {
    this.halfLen = halfLen;
    this.active = null; // {lead, s0, s1, off, ramp, endS, aborted}
    this.wait = 0;
    this.cooldown = 0;
    this.info = null; // for HUD / debug overlay
    this.offsetAt = (s) => {
      const a = this.active;
      if (!a) return 0;
      let f = smooth((s - a.s0) / a.ramp);
      if (a.s1 !== null) f = Math.min(f, 1 - smooth((s - a.s1) / a.ramp));
      return a.off * f;
    };
  }

  // Updates the maneuver; returns extra distance to hold back behind a stopped vehicle so
  // there's room to swing out around it.
  plan({ route, s, k, v, v0, obs, agents, self, dt }) {
    const hl = this.halfLen;
    this.cooldown = Math.max(0, this.cooldown - dt);
    const p = route.at(s, k), fx = Math.cos(p.h), fz = Math.sin(p.h);
    const along = (o) => s + (o.x - p.x) * fx + (o.z - p.z) * fz;

    const a = this.active;
    if (a) {
      const lead = a.lead, leadS = along(lead);
      if (a.s1 === null) {
        if (!agents.includes(lead) || s - hl > leadS + lead.halfLen + 4) {
          a.s1 = s; // our rear is clear: merge back
        } else if (s + hl < leadS - lead.halfLen + 1) {
          const D = Math.max(a.endS - s, 10);
          if (!oncomingClear(p, fx, fz, agents, self, lead, D, D / Math.max(v, 3) + 2).ok) {
            a.s1 = s;
            a.aborted = true;
            this.cooldown = 4;
          }
        }
      } else if (s > a.s1 + a.ramp) {
        this.active = null;
        this.wait = 0;
      }
      this.info = this.active
        ? { state: a.s1 === null ? 'passing' : a.aborted ? 'aborting' : 'returning', why: null, lead, zone: [s, Math.max(s, a.endS)], clear: true }
        : null;
      return 0;
    }

    const lead = obs.agent;
    if (!lead || lead.kind !== 'car' || !(lead.hazard || lead.slow) || obs.gap > 30) {
      this.wait = 0;
      this.info = null;
      return 0;
    }
    this.wait += dt;
    const hang = lead.hazard ? 4 : 0;
    const ramp = clamp(8 + 0.6 * v, 9, 14);
    const vp = Math.min(v0, CRUISE);
    const rel = along(lead) + lead.halfLen + 4 + hl - s;
    const { T, D: Dp } = passPlan(v, vp, lead.v, rel);
    const D = Dp + ramp + 2, endS = s + D, Tall = T + ramp / Math.max(vp, 3);
    const leg = route.legs.find((l) => s - hl >= l.s0 && s < l.s1);
    let why = null;
    if (vp - lead.v < 2.5) why = 'Too slow to pass';
    else if (!leg || endS + hl + 6 > leg.stop.s) why = 'Intersection ahead';
    else {
      const chk = oncomingClear(p, fx, fz, agents, self, lead, D, Tall);
      if (!chk.ok) why = chk.by.kind === 'ped' ? 'Pedestrian crossing' : 'Oncoming traffic';
      else if (this.cooldown > 0 || this.wait < 1.2) why = 'Checking';
    }
    if (!why) this.active = { lead, s0: s, s1: null, off: LANE_W, ramp, endS, aborted: false };
    this.info = { state: why ? 'waiting' : 'passing', why, lead, zone: [s, endS], clear: !why || why === 'Checking' };
    return why ? hang : 0;
  }
}

// ---------- ego expert ----------
// Privileged driver for the ego vehicle: shares the logic above, adds pure-pursuit steering.
export class Expert {
  constructor(route, signals) {
    this.signals = signals;
    this.agent = { id: 0, kind: 'car', x: 0, z: 0, h: 0, v: 0, halfLen: HALF_LEN, s: 0, leg: null, ego: true };
    this.setRoute(route);
  }

  setRoute(route) {
    this.route = route;
    this.k = 0;
    this.s = 0;
    this.lateral = 0;
    this.ot = new Overtaker(HALF_LEN);
  }

  // Keep the agent in sync while someone else is driving.
  observe(car) {
    Object.assign(this.agent, { x: car.x, z: car.z, h: car.h, v: car.v, leg: null });
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
    // Cross-track error from the intended path (lane center, or the passing offset).
    this.lateral = -(car.x - p.x) * Math.sin(p.h) + (car.z - p.z) * Math.cos(p.h) + this.ot.offsetAt(this.s);
    Object.assign(this.agent, { x: car.x, z: car.z, h: car.h, v: car.v, s: this.s, leg: this.route.legAt(this.s) });
  }

  control(car, agents = [], dt = 1 / 60) {
    const r = this.route;
    r.ensure(this.s + 250);
    if (this.k > 400) this.k -= r.trim(this.s);
    this.track(car);
    const off = this.ot.offsetAt;

    // Pure pursuit on the (possibly offset) path.
    const Ld = 4 + 0.45 * car.v;
    const base = r.at(this.s + Ld, this.k), o = off(this.s + Ld);
    const tgt = { x: base.x + Math.sin(base.h) * o, z: base.z - Math.cos(base.h) * o };
    const alpha = angleWrap(Math.atan2(tgt.z - car.z, tgt.x - car.x) - car.h);
    const ld = Math.hypot(tgt.x - car.x, tgt.z - car.z);
    const steer = clamp(Math.atan((2 * WHEELBASE * Math.sin(alpha)) / ld) / MAX_STEER, -1, 1);

    const v0 = curveSpeed(r, this.s, this.k);
    const sig = signalObstacle(r, this.s, car.v, this.signals, makeLeftTurnYield(agents, this.agent));
    const samples = pathAhead(r, this.s, this.k, 45, 1.5, this.ot.active ? off : null);
    let obs = pathObstacle(samples, agents, this.agent);
    const hang = this.ot.plan({ route: r, s: this.s, k: this.k, v: car.v, v0, obs, agents, self: this.agent, dt });
    if (hang && Number.isFinite(obs.gap)) obs = { ...obs, gap: Math.max(0, obs.gap - hang) };
    const cross = crossingObstacle(samples, agents, this.agent, car.v);
    if (cross.gap < obs.gap) obs = cross;
    let { acc, reason } = longitudinal(car.v, v0, sig, obs);
    if (reason === 'vehicle' && this.ot.info?.state === 'waiting' && obs.agent === this.ot.info.lead) reason = 'overtake-wait';
    // Hold the brake when stopped so the car doesn't creep.
    let throttle = acc >= 0 ? acc / 3.2 : acc / 7.5;
    if (car.v < 0.3 && acc < 0.2) throttle = -0.5;

    const nextTurn = r.turns.find((t) => t.s > this.s - 5);
    return {
      steer, throttle, acc, reason, target: tgt, signal: sig.stop, lead: Number.isFinite(obs.gap) ? obs : null,
      nextTurn: nextTurn && { kind: nextTurn.kind, dist: nextTurn.s - this.s },
      samples, overtake: this.ot.info, route: r, s: this.s, k: this.k,
    };
  }
}
