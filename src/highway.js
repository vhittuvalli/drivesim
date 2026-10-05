// Highway: a divided motorway loop around the city, three lanes each way, at highway speed.
// Headless like planner.js: geometry, routes along the lanes, and lane changing (MOBIL).
//
// The centerline (the median) is a closed chain of straights and circular arcs, so position,
// heading and curvature are exact at any arc length u. Traffic on the "forward" carriageway
// (dir = 1) drives with increasing u; dir = -1 drives the other way. A vehicle's progress q
// increases in its direction of travel; lateral offsets are meters right of the median.
// Lanes are numbered from the median: 0 is the left (passing) lane, 2 the right lane.
import { GRID, PITCH, clamp } from './config.js';
import { Path } from './path.js';
import { idm, IDM, angleWrap } from './planner.js';

export const HW = {
  lanes: 3,
  laneW: 3.7,
  median: 1.1, // centerline to the barrier's face
  innerShoulder: 1.2,
  outerShoulder: 3,
  speed: 30, // m/s speed limit (108 km/h)
  cornerR: 300,
};
HW.inner = HW.median + HW.innerShoulder; // centerline to the left edge of lane 0
HW.laneEdge = HW.inner + HW.lanes * HW.laneW; // centerline to the right edge of the right lane
HW.width = HW.laneEdge + HW.outerShoulder; // centerline to the guardrail
export const HW_CRUISE = HW.speed - 1; // the ego expert's desired speed

export const laneOffset = (lane) => HW.inner + HW.laneW * (lane + 0.5);
export const laneOf = (lat) => clamp(Math.round((lat - HW.inner) / HW.laneW - 0.5), 0, HW.lanes - 1);
const mod = (a, n) => ((a % n) + n) % n;

// ---------- centerline ----------
// A rounded square around the city with a gentle S-bend on the east and west sides.
// turn > 0 turns right (heading increases: east -> south).
function buildSegments() {
  const c = ((GRID - 1) * PITCH) / 2, half = 710, R = HW.cornerR;
  const side = 2 * half - 2 * R;
  // S-bend: four arcs of radius sR through angle sA (left, right, right, left): net zero heading
  // and zero lateral shift, bulging outward by 2 sR (1 - cos sA).
  const sR = 450, sA = 0.28, sLen = 4 * sR * Math.sin(sA), gap = (side - sLen) / 2;
  const sBend = [[-1, sR, sA], [1, sR, sA], [1, sR, sA], [-1, sR, sA]].map(([sg, r, a]) => ({ turn: sg * a, R: r }));
  const plan = [
    { L: side }, { turn: Math.PI / 2, R },
    { L: gap }, ...sBend, { L: gap }, { turn: Math.PI / 2, R },
    { L: side }, { turn: Math.PI / 2, R },
    { L: gap }, ...sBend, { L: gap }, { turn: Math.PI / 2, R },
  ];
  let x = c - half + R, z = c - half, h = 0, u = 0;
  const segs = [];
  for (const p of plan) {
    const seg = { u0: u, x0: x, z0: z, h0: h };
    if (p.L) {
      Object.assign(seg, { len: p.L, k: 0 });
      x += Math.cos(h) * p.L;
      z += Math.sin(h) * p.L;
    } else {
      const sg = Math.sign(p.turn), len = Math.abs(p.turn) * p.R;
      // Center of the arc is R to the right (sg > 0) or left of the start.
      Object.assign(seg, { len, k: sg / p.R, cx: x - Math.sin(h) * sg * p.R, cz: z + Math.cos(h) * sg * p.R, R: p.R, sg });
      h += p.turn;
      x = seg.cx + Math.sin(h) * sg * p.R;
      z = seg.cz - Math.cos(h) * sg * p.R;
    }
    u += seg.len;
    segs.push(seg);
  }
  return { segs, length: u };
}

const { segs: SEGS, length: PERIMETER } = buildSegments();
export const HW_LENGTH = PERIMETER;
export const HW_SEGMENTS = SEGS;

// Centerline pose at arc length u: {x, z, h, k (curvature, > 0 turning right)}.
export function centerAt(u) {
  u = mod(u, PERIMETER);
  let seg = SEGS[SEGS.length - 1];
  for (const s of SEGS) {
    if (u < s.u0 + s.len) {
      seg = s;
      break;
    }
  }
  const t = u - seg.u0;
  if (!seg.k) return { x: seg.x0 + Math.cos(seg.h0) * t, z: seg.z0 + Math.sin(seg.h0) * t, h: seg.h0, k: 0 };
  const h = seg.h0 + seg.sg * (t / seg.R);
  return { x: seg.cx + Math.sin(h) * seg.sg * seg.R, z: seg.cz - Math.cos(h) * seg.sg * seg.R, h, k: seg.k };
}

// Point `lat` meters right of the median for a vehicle at progress q in direction dir.
export function lanePoint(dir, q, lat) {
  const c = centerAt(dir > 0 ? q : -q), nx = -Math.sin(c.h), nz = Math.cos(c.h);
  const o = dir > 0 ? lat : -lat;
  return { x: c.x + nx * o, z: c.z + nz * o, h: dir > 0 ? c.h : angleWrap(c.h + Math.PI) };
}

// Coarse lookup table for projecting world points onto the centerline.
const TABLE_STEP = 4;
const TABLE = Array.from({ length: Math.ceil(PERIMETER / TABLE_STEP) }, (_, i) => {
  const c = centerAt(i * TABLE_STEP);
  return [c.x, c.z];
});

// Where a pose is relative to the highway: direction it's heading, progress, lateral offset
// (right of the median in its direction of travel), nearest lane. `dist` is the distance from
// the median; anything over HW.width is off the road.
export function highwayPose(x, z, h) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < TABLE.length; i++) {
    const d = (TABLE[i][0] - x) ** 2 + (TABLE[i][1] - z) ** 2;
    if (d < bd) (bd = d), (best = i);
  }
  let u = best * TABLE_STEP;
  for (let it = 0; it < 3; it++) {
    const c = centerAt(u);
    u += (x - c.x) * Math.cos(c.h) + (z - c.z) * Math.sin(c.h);
  }
  const c = centerAt(u), right = -(x - c.x) * Math.sin(c.h) + (z - c.z) * Math.cos(c.h);
  const dir = Math.cos(h - c.h) >= 0 ? 1 : -1;
  const lat = dir > 0 ? right : -right;
  const pathH = dir > 0 ? c.h : c.h + Math.PI;
  return { dir, q: mod(dir > 0 ? u : -u, PERIMETER), lat, lane: laneOf(lat), dist: Math.abs(right), headingErr: Math.abs(angleWrap(h - pathH)) };
}

export const onHighway = (p) => !!p && p.dist < HW.width + 2;

// Signed distance from progress qa to qb along the loop, in (-L/2, L/2].
export const loopDist = (qa, qb) => {
  const d = mod(qb - qa, PERIMETER);
  return d > PERIMETER / 2 ? d - PERIMETER : d;
};

// ---------- routes ----------

// Minimum-jerk blend 0 -> 1.
const minJerk = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * u * (10 + u * (-15 + 6 * u)));
const STEP = 2;

// Lane changing distance: about 3.5 s, never shorter than 22 m.
export const changeDistance = (v) => clamp(v * 3.5, 22, 110);

// A route along the highway lanes, with the same interface as the grid Route (no stop lines,
// turns or legs). Lateral position is a plan of blends between lane offsets as a function of
// progress q; a lane change replaces the geometry ahead of the car, so the path the vehicle is
// on (and every index into it) stays valid.
export class HighwayRoute extends Path {
  // lat/merge: start `lat` meters right of the median and blend into `lane` over `merge` meters.
  constructor({ dir = 1, lane = 1, q = 0, lat = null, merge = 0 } = {}) {
    super();
    this.highway = true;
    this.dir = dir;
    this.lane = lane;
    const to = laneOffset(lane);
    this.plan = [lat === null || !merge ? { q0: -Infinity, q1: -Infinity, from: to, to } : { q0: q, q1: q + merge, from: lat, to }];
    this.qEnd = q;
    this.addPoint(q);
    this.ensure(200);
  }

  static fromPose(x, z, h, v = 0) {
    const p = highwayPose(x, z, h);
    if (!onHighway(p)) return { route: null, reason: 'Drive back onto the highway' };
    if (p.headingErr > 0.6) return { route: null, reason: 'Line up with the lane first' };
    return { route: new HighwayRoute({ dir: p.dir, lane: p.lane, q: p.q, lat: p.lat, merge: changeDistance(v) }), reason: null };
  }

  latAt(q) {
    let p = this.plan[0];
    for (const x of this.plan) {
      if (x.q0 > q) break;
      p = x;
    }
    if (q >= p.q1) return p.to;
    return p.from + (p.to - p.from) * minJerk((q - p.q0) / (p.q1 - p.q0));
  }

  addPoint(q) {
    const p = lanePoint(this.dir, q, this.latAt(q));
    if (this.push(p.x, p.z)) this.pts[this.pts.length - 1].q = q;
  }

  extend() {
    for (let i = 0; i < 25; i++) this.addPoint((this.qEnd += STEP));
  }

  // Progress at arc length s.
  qAt(s, hint = 0) {
    const p = this.at(s, hint), a = this.pts[p.k], b = this.pts[p.k + 1];
    return a.q + (b.q - a.q) * clamp((s - a.s) / (b.s - a.s || 1), 0, 1);
  }

  // Signed curvature (> 0 turning right) at arc length s, exact rather than from the polyline:
  // the median's curvature seen from this lane, plus the lane-change blend.
  curvatureAt(s, hint = 0) {
    const q = this.qAt(s, hint), c = centerAt(this.dir > 0 ? q : -q), lat = this.latAt(q);
    const k = this.dir > 0 ? c.k : -c.k;
    return k / (1 - k * lat) + (this.latAt(q + 1) - 2 * lat + this.latAt(q - 1));
  }

  // The lane-change blend covering arc length s, or null.
  blendAt(s, hint = 0) {
    const q = this.qAt(s, hint);
    const p = this.plan.findLast((x) => x.q0 <= q);
    return p && q < p.q1 && p.from !== p.to ? p : null;
  }

  // Navigation command: the direction of a lane change in progress.
  commandAt(s, hint = 0) {
    const p = this.blendAt(s, hint);
    return p ? { kind: p.to > p.from ? 'right' : 'left', dist: 0 } : { kind: 'straight', dist: Infinity };
  }

  // Start changing into `lane` at arc length s, completing `dist` meters later.
  changeLane(s, lane, dist, hint = 0) {
    const q0 = this.qAt(s, hint), from = this.latAt(q0);
    let n = this.pts.length;
    while (n > 2 && this.pts[n - 1].q > q0) n--;
    this.pts.length = n;
    this.plan = this.plan.filter((p) => p.q0 < q0);
    this.plan.push({ q0, q1: q0 + dist, from, to: laneOffset(lane) });
    this.lane = lane;
    this.qEnd = this.pts[n - 1].q;
    this.ensure(s + 250);
  }

  trim(s, keep = 60) {
    const n = super.trim(s, keep);
    const q = this.pts[0].q;
    while (this.plan.length > 1 && this.plan[1].q0 <= q) this.plan.shift();
    return n;
  }

  // What other drivers see: {dir, q, lat, target lane}. extraLat: the vehicle's offset from
  // the path (tracking error).
  stateAt(s, hint = 0, extraLat = 0) {
    const q = this.qAt(s, hint);
    return { dir: this.dir, q: mod(q, PERIMETER), lat: this.latAt(q) + extraLat, target: this.lane };
  }
}

// ---------- lane changing (MOBIL) ----------
// "Minimizing Overall Braking Induced by Lane changes" (Kesting, Treiber, Helbing 2007):
// change lanes if our IDM acceleration improves by more than a threshold plus a politeness-
// weighted share of what the change costs the followers, and the new follower wouldn't have to
// brake harder than bSafe. A keep-right bias above the threshold moves drivers back out of the
// passing lanes when the right lane is as good ("keep right except to pass"). preferLane moves
// that bias to another lane (data collection uses it so the expert spends time in every lane).

// Is the vehicle in (or moving into) `lane`?
function occupies(hw, lane) {
  return hw.target === lane || Math.abs((hw.lat - HW.inner) / HW.laneW - 0.5 - lane) < 0.8;
}

// Nearest leader and follower in `lane` (bumper-to-bumper gaps; negative when alongside).
export function neighbors(self, lane, agents, range = 220) {
  const me = self.hw;
  let lead = null, fol = null;
  for (const a of agents) {
    if (a === self || !a.hw || a.hw.dir !== me.dir || !occupies(a.hw, lane)) continue;
    const d = loopDist(me.q, a.hw.q);
    if (Math.abs(d) > range) continue;
    const gap = Math.abs(d) - a.halfLen - self.halfLen;
    if (d >= 0 ? !lead || gap < lead.gap : !fol || gap < fol.gap) {
      const n = { agent: a, gap, v: a.v };
      if (d >= 0) lead = n;
      else fol = n;
    }
  }
  return { lead, fol };
}

// Agents that could be on a highway vehicle's path in the next `horizon` meters: on its
// carriageway and in range along it (anything off the highway can't be).
export function aheadOnCarriageway(self, agents, horizon) {
  const me = self.hw;
  if (!me) return agents;
  return agents.filter((a) => {
    if (a === self || !a.hw || a.hw.dir !== me.dir) return false;
    const d = loopDist(me.q, a.hw.q);
    return d > -12 && d < horizon + 12;
  });
}

// The closer of a path obstacle (planner.pathObstacle) and the leader in our lane, which includes
// a car that is merging into it ahead of us: let it in, even before it's on our path.
export function laneLeader(obs, self, lane, agents) {
  const lead = neighbors(self, lane, agents, 160).lead;
  return lead && Math.max(lead.gap, 0) < obs.gap ? { gap: Math.max(lead.gap, 0), v: lead.v, agent: lead.agent } : obs;
}

const accel = (v, v0, n, p) => (n ? idm(v, v0, Math.max(n.gap, 0.1), n.v, p) : idm(v, v0, Infinity, 0, p));

export class LaneChanger {
  constructor({ politeness = 0.3, threshold = 0.2, keepRight = 0.3, bSafe = 3.5, cooldown = 4, rand = Math.random } = {}) {
    Object.assign(this, { politeness, threshold, keepRight, bSafe, cooldown });
    this.wait = rand() * 0.5; // staggers decisions across drivers
    this.cool = 1;
    this.preferLane = null; // lane the bias pulls toward (null: the right lane)
    this.info = null; // {lane, changing: 'left' | 'right' | null, why}
  }

  // Decide and start a lane change. Returns the new lane or null.
  update({ route, s, k, v, v0, agents, self, p = IDM, dt }) {
    this.cool -= dt;
    this.wait -= dt;
    const blend = route.blendAt(s, k);
    if (blend) {
      this.info = { lane: route.lane, changing: blend.to > blend.from ? 'right' : 'left', why: this.info?.why ?? null };
      return null;
    }
    this.info = { lane: route.lane, changing: null, why: null };
    if (this.cool > 0 || this.wait > 0 || !self.hw) return null;
    this.wait = 0.5;
    const lane = route.lane, cur = neighbors(self, lane, agents);
    // Room to swerve: a lane change bends out slowly at first (half its length to get ~2 m across),
    // so starting one right behind a stopped car clips its corner. Wait for the gap instead.
    if (cur.lead && cur.lead.gap < 0.5 * changeDistance(v) && cur.lead.v < v + 1) return null;
    // A stalled car ahead makes leaving the lane mandatory: at a standstill behind it IDM alone
    // would see nothing to gain.
    const blocked = !!cur.lead?.agent.hazard && cur.lead.gap < 150;
    const aCur = blocked ? Math.min(accel(v, v0, cur.lead, p), -this.bSafe) : accel(v, v0, cur.lead, p);
    // What our follower gains if we leave.
    const oldFol = cur.fol ? accel(cur.fol.v, HW.speed, cur.lead && { gap: cur.fol.gap + 2 * self.halfLen + cur.lead.gap, v: cur.lead.v }, IDM) - accel(cur.fol.v, HW.speed, { gap: cur.fol.gap, v }, IDM) : 0;
    let best = null;
    for (const to of [lane - 1, lane + 1]) {
      if (to < 0 || to >= HW.lanes) continue;
      const n = neighbors(self, to, agents);
      if ((n.lead && n.lead.gap < 3 + v * 0.3) || (n.fol && n.fol.gap < 3)) continue;
      const aNew = accel(v, v0, n.lead, p);
      if (aNew < -this.bSafe) continue;
      let newFol = 0;
      if (n.fol) {
        const after = accel(n.fol.v, HW.speed, { gap: n.fol.gap, v }, IDM);
        if (after < -this.bSafe) continue;
        const before = accel(n.fol.v, HW.speed, n.lead && { gap: n.fol.gap + 2 * self.halfLen + n.lead.gap, v: n.lead.v }, IDM);
        newFol = after - before;
      }
      const target = this.preferLane ?? HW.lanes - 1, toward = Math.abs(to - target) < Math.abs(lane - target);
      const gain = aNew - aCur + this.politeness * (newFol + oldFol) + (toward ? this.keepRight : -this.keepRight);
      if (gain > this.threshold && (!best || gain > best.gain)) {
        const why = blocked ? 'blocked lane' : !toward ? 'passing' : this.preferLane === null ? 'keeping right' : 'lane choice';
        best = { to, gain, why };
      }
    }
    if (!best) return null;
    route.changeLane(s, best.to, changeDistance(v), k);
    this.cool = this.cooldown + changeDistance(v) / Math.max(v, 5);
    this.info = { lane: best.to, changing: best.to > lane ? 'right' : 'left', why: best.why };
    return best.to;
  }
}

// Extra distance to hold back behind a stalled vehicle so there's room to steer around it.
export const HAZARD_HANG = 12;
