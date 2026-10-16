// Safety driver: supervises a learned driver with the expert running in shadow mode. When the
// learned driver leaves its lane, points the wrong way, doesn't brake when the expert would
// be braking hard (for longer than brakeGrace: a driver sampled at 10 Hz can't react on the very
// step the expert's demand jumps), or sits still when the expert would pull away (for longer than
// stallGrace), or is closing on an obstacle with under ttcMin seconds to impact while braking less
// than the expert, the expert takes over for a few seconds (a disengagement). The headline
// metric is autonomous distance per disengagement, as reported by real AV programs.
//
// Lenient mode (`lenient`, ?safety=lenient) takes over only where the learned driver is in danger,
// not where it merely differs from the expert: collision risk, an imminent scrape, running a red
// light, and its center actually outside the lane (LANE_W / 2, not maxLateral). It ignores heading,
// braking later than the expert and slowness to pull away, to tell style differences from errors.
import { angleWrap, HALF_LEN } from './planner.js';
import { WHEELBASE, MAX_STEER, LANE_W } from './config.js';

export const TAKEOVER_SECONDS = 3;

const r2 = (x) => (typeof x === 'number' ? Math.round(x * 100) / 100 : x);

// Where the expert's overtake is: 'out' (swinging into the oncoming lane), 'passing', 'back'
// (merging back), 'ended' (within 2 s after), or null when there is none.
function overtakePhase(expert, sinceEnd) {
  const a = expert?.ot?.active;
  if (!a) return sinceEnd < 2 ? { phase: 'ended', since: r2(sinceEnd) } : null;
  const s = expert.s;
  const phase = s < a.s0 + a.ramp ? 'out' : a.s1 === null || s < a.s1 ? 'passing' : 'back';
  return { phase, offset: r2(expert.ot.offsetAt(s)), full: r2(a.off), aborted: !!a.aborted };
}

// What the learned driver and the expert saw and wanted at a takeover, to tell why it happened
// (e.g. a stall at a green light held by the network's own light brake).
function takeoverContext(world, exp, nn, overtakeEnd = -Infinity) {
  const p = world.policy?.pred, { expert } = world;
  return {
    v: r2(world.car.v),
    // Sideways from the expert's path (overtake swerve included): > 0 right of it, < 0 left.
    lateral: r2(expert?.lateral ?? null),
    overtake: overtakePhase(expert, world.t - overtakeEnd),
    nn: { throttle: r2(nn.throttle), vTarget: r2(nn.vTarget), lightStop: !!nn.lightStop, leadBrake: !!nn.leadBrake },
    expert: { acc: r2(exp.acc), throttle: r2(exp.throttle), reason: exp.reason ?? null },
    signal: exp.signal ? { state: exp.signal.state ?? null, dist: r2(exp.signal.dist) } : null,
    lead: exp.lead && exp.lead.gap < 80 ? { gap: r2(exp.lead.gap), v: r2(exp.lead.v) } : null,
    seen: p ? {
      light: p.light ? Object.fromEntries(Object.entries(p.light).map(([k, v]) => [k, r2(v)])) : null,
      stopDist: r2(p.stopDist ?? null), lead: p.lead ? { gap: r2(p.lead.gap), v: r2(p.lead.v) } : null,
    } : null,
  };
}

// Whether a car at `pose` (x, z, h; the ego's size) touches vehicle `a`: points along a's axis
// inside the ego's footprint grown by a's half width. The benchmark counts contacts with this.
export function touchesPedestrian(pose, p) {
  const dx = p.x - pose.x, dz = p.z - pose.z, c = Math.cos(pose.h), s = Math.sin(pose.h);
  return Math.abs(dx * c + dz * s) < HALF_LEN + 0.25 && Math.abs(-dx * s + dz * c) < 0.92 + 0.25;
}

export function touchesVehicle(pose, a) {
  const e = a.halfLen - 0.75, c = Math.cos(pose.h), s = Math.sin(pose.h);
  return [-e, 0, e].some((k) => {
    const dx = a.x + Math.cos(a.h) * k - pose.x, dz = a.z + Math.sin(a.h) * k - pose.z;
    return Math.abs(dx * c + dz * s) < HALF_LEN + 0.6 && Math.abs(-dx * s + dz * c) < 0.92 + 0.85;
  });
}

export class SafetyDriver {
  constructor({ maxLateral = 1.1, maxHeading = 0.45, brakeDemand = -3, brakeGrace = 0.3, stallGrace = 2.5, ttcMin = 2, ttcGrace = 0.2, maxLateralOvertake = 1.8, scrapeHorizon = 0.8 } = {}) {
    Object.assign(this, { maxLateral, maxHeading, brakeDemand, brakeGrace, stallGrace, ttcMin, ttcGrace, maxLateralOvertake, scrapeHorizon });
    this.enabled = true;
    this.lenient = false; // see the top of this file
    this.reset();
  }

  reset() {
    this.takeover = 0; // seconds of expert control left
    this.underBraking = 0; // seconds the learned driver has been braking less than the expert wants
    this.stalled = 0; // seconds stopped while the expert wants to drive off
    this.closing = 0; // seconds under ttcMin while braking less than the expert
    this.overtakeEnd = -Infinity; // when the expert last had an overtake in progress
    this.events = []; // {t, reason, x, z}
    this.autoDist = 0; // meters driven by the learned driver
    this.totalDist = 0;
  }

  get disengagements() {
    return this.events.length;
  }

  // Meters of autonomous driving per disengagement (Infinity if none yet).
  get distPerDisengagement() {
    return this.events.length ? this.autoDist / this.events.length : Infinity;
  }

  // Why the expert should take over right now, or null.
  check(world, exp, nn, dt = 0) {
    const { expert, car } = world;
    // During an overtake the path itself swings across, and a driver that can't know the expert's
    // exact timing trails it: still in its own lane as the path swings out (right of it), still in
    // the oncoming lane as the path merges back (left of it). Neither is the danger: the vehicle
    // being passed is covered by the collision-risk, late-braking and scrape checks, and the
    // oncoming lane was checked clear by the expert. So allow maxLateralOvertake to the left
    // throughout and for a second after, and to the right while the path is still swinging out
    // (not beside the passed vehicle or after, where parked cars are and 1.5 m is contact).
    const ot = expert.ot?.active;
    if (ot) this.overtakeEnd = world.t;
    const left = expert.lateral < 0 && world.t - this.overtakeEnd < 1;
    const right = expert.lateral > 0 && ot && expert.s < ot.s0 + ot.ramp;
    const maxLateral = this.lenient ? Math.max(this.maxLateral, LANE_W / 2) : this.maxLateral;
    if (Math.abs(expert.lateral) > (left || right ? Math.max(this.maxLateralOvertake, maxLateral) : maxLateral)) return 'left the lane';
    // Heading of the intended path here, including the swerve of an overtake.
    const p = expert.route.at(expert.s, expert.k), off = expert.ot.offsetAt;
    const pathH = p.h - Math.atan((off(expert.s + 1) - off(expert.s - 1)) / 2);
    if (!this.lenient && Math.abs(angleWrap(car.h - pathH)) > this.maxHeading) return 'wrong heading';
    // Stopped at a green light or behind nothing: a learned driver can latch onto its own zero
    // speed and never pull away, which none of the other checks would notice.
    // Not while the expert is itself held by a light, a yield or a blocked box: then it is only
    // creeping up to the line, and staying put is fine.
    const held = exp.reason === 'signal' || exp.reason === 'yield' || exp.reason === 'box';
    const stall = !held && exp.acc > 0.5 && car.v < 0.5 && nn.throttle < 0.05;
    this.stalled = stall ? this.stalled + dt : 0;
    if (!this.lenient && this.stalled > this.stallGrace) return 'did not pull away';
    // Time to collision with the obstacle the expert sees: waiting for the expert's braking demand
    // to outlast brakeGrace was too late to avoid contact (the network braking late in overtakes).
    // A short grace (two 10 Hz observations): a learned driver can't react on the very step the
    // expert's demand jumps.
    const L = exp.lead, rate = L ? car.v - L.v : 0;
    // The threshold grows with speed (2 s up to 10 m/s, 3.5 s at 25 m/s): closing on a stopped jam at
    // highway speed, 2 s left too little room and the takeover came too late to avoid contact.
    const ttcMin = Math.max(this.ttcMin, 1 + car.v / 10);
    const risk = !!L && L.gap < 80 && rate > 0.5 && L.gap / rate < ttcMin && nn.throttle > exp.throttle + 0.3;
    this.closing = risk ? this.closing + dt : 0;
    if (risk && this.closing > this.ttcGrace) return 'collision risk';
    const scrape = this.willScrape(world, nn);
    if (scrape) return scrape === 'pedestrian' ? 'about to hit a pedestrian' : 'about to scrape a vehicle';
    const under = exp.acc < this.brakeDemand && car.v > 1 && nn.throttle > exp.throttle + 0.35;
    this.underBraking = under ? this.underBraking + dt : 0;
    if (under && this.underBraking > this.brakeGrace && (!this.lenient || exp.reason === 'signal')) {
      return { pedestrian: 'late braking for a pedestrian', signal: 'about to run a red light', crossing: 'late braking for a crossing vehicle' }[exp.reason] ?? 'late braking';
    }
    return null;
  }

  // Rolling the learned driver's steering forward scrapeHorizon seconds at the current speed (and
  // other road users straight on at theirs), the car would touch a vehicle or pedestrian it isn't
  // touching now. The lane, heading and lead checks missed the network clipping a parked car beside
  // its path at 4 m/s, and creeping at 0.6 m/s into a pedestrian standing 2.6 m ahead in fog.
  willScrape(world, nn) {
    const { car } = world;
    if (car.v < 0.5 || !world.traffic) return null;
    const close = (a) => Math.abs(a.x - car.x) < 12 && Math.abs(a.z - car.z) < 12;
    const near = [...world.traffic.agents, ...(world.parked ?? []).filter((p) => !p.hidden)].filter((a) => close(a) && !touchesVehicle(car, a));
    const peds = (world.crowd?.peds ?? []).filter((p) => close(p) && !touchesPedestrian(car, p));
    if (!near.length && !peds.length) return null;
    const p = { x: car.x, z: car.z, h: car.h }, target = Math.max(-1, Math.min(1, nn.steer)) * MAX_STEER, dt = 0.1;
    let steer = car.steer ?? target;
    for (let t = dt; t < this.scrapeHorizon + 1e-9; t += dt) {
      steer += Math.max(-1.6 * dt, Math.min(1.6 * dt, target - steer));
      p.h += (car.v / WHEELBASE) * Math.tan(steer) * dt;
      p.x += car.v * Math.cos(p.h) * dt;
      p.z += car.v * Math.sin(p.h) * dt;
      const moved = (a) => ({ ...a, x: a.x + Math.cos(a.h) * a.v * t, z: a.z + Math.sin(a.h) * a.v * t });
      if (near.some((a) => touchesVehicle(p, a.v ? moved(a) : a))) return 'vehicle';
      if (peds.some((q) => touchesPedestrian(p, q.v ? moved(q) : q))) return 'pedestrian';
    }
    return null;
  }

  // Returns the control to apply: the learned driver's, or the expert's during a takeover.
  step(world, exp, nn, dt) {
    const d = world.car.v * dt;
    this.totalDist += d;
    if (this.takeover > 0) {
      this.takeover -= dt;
      this.underBraking = 0;
      this.stalled = 0;
      this.closing = 0;
      return { ...exp, driver: 'safety', nn, expert: exp };
    }
    const reason = this.enabled ? this.check(world, exp, nn, dt) : null;
    if (reason) {
      this.takeover = TAKEOVER_SECONDS;
      this.events.push({ t: world.t, reason, x: world.car.x, z: world.car.z, ctx: takeoverContext(world, exp, nn, this.overtakeEnd) });
      return { ...exp, driver: 'safety', nn, expert: exp, disengaged: reason };
    }
    this.autoDist += d;
    return { ...nn, driver: 'neural', nn, expert: exp };
  }
}
