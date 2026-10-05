// Safety driver: supervises a learned driver with the expert running in shadow mode. When the
// learned driver leaves its lane, points the wrong way, doesn't brake when the expert would
// be braking hard (for longer than brakeGrace: a driver sampled at 10 Hz can't react on the very
// step the expert's demand jumps), or sits still when the expert would pull away (for longer than
// stallGrace), or is closing on an obstacle with under ttcMin seconds to impact while braking less
// than the expert, the expert takes over for a few seconds (a disengagement). The headline
// metric is autonomous distance per disengagement, as reported by real AV programs.
import { angleWrap } from './planner.js';

export const TAKEOVER_SECONDS = 3;

export class SafetyDriver {
  constructor({ maxLateral = 1.1, maxHeading = 0.45, brakeDemand = -3, brakeGrace = 0.3, stallGrace = 2.5, ttcMin = 2, ttcGrace = 0.2, maxLateralOvertake = 1.5 } = {}) {
    Object.assign(this, { maxLateral, maxHeading, brakeDemand, brakeGrace, stallGrace, ttcMin, ttcGrace, maxLateralOvertake });
    this.enabled = true;
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
    // During an overtake (the expert checked the oncoming lane is clear) the path itself swings
    // across; a 10 Hz driver following it trails the merge back by ~1.1 m to the left even with
    // perfect predictions, so allow a little more there, and for a second after while it settles.
    // Only to the left (lateral < 0): to the right are parked cars, and 1.5 m that way is contact.
    if (expert.ot?.active) this.overtakeEnd = world.t;
    const lenient = world.t - this.overtakeEnd < 1 && expert.lateral < 0;
    if (Math.abs(expert.lateral) > (lenient ? this.maxLateralOvertake : this.maxLateral)) return 'left the lane';
    // Heading of the intended path here, including the swerve of an overtake.
    const p = expert.route.at(expert.s, expert.k), off = expert.ot.offsetAt;
    const pathH = p.h - Math.atan((off(expert.s + 1) - off(expert.s - 1)) / 2);
    if (Math.abs(angleWrap(car.h - pathH)) > this.maxHeading) return 'wrong heading';
    // Stopped at a green light or behind nothing: a learned driver can latch onto its own zero
    // speed and never pull away, which none of the other checks would notice.
    // Not while the expert is itself held by a light, a yield or a blocked box: then it is only
    // creeping up to the line, and staying put is fine.
    const held = exp.reason === 'signal' || exp.reason === 'yield' || exp.reason === 'box';
    const stall = !held && exp.acc > 0.5 && car.v < 0.5 && nn.throttle < 0.05;
    this.stalled = stall ? this.stalled + dt : 0;
    if (this.stalled > this.stallGrace) return 'did not pull away';
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
    const under = exp.acc < this.brakeDemand && car.v > 1 && nn.throttle > exp.throttle + 0.35;
    this.underBraking = under ? this.underBraking + dt : 0;
    if (under && this.underBraking > this.brakeGrace) {
      return { pedestrian: 'late braking for a pedestrian', signal: 'about to run a red light', crossing: 'late braking for a crossing vehicle' }[exp.reason] ?? 'late braking';
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
      this.events.push({ t: world.t, reason, x: world.car.x, z: world.car.z });
      return { ...exp, driver: 'safety', nn, expert: exp, disengaged: reason };
    }
    this.autoDist += d;
    return { ...nn, driver: 'neural', nn, expert: exp };
  }
}
