// Safety driver: supervises a learned driver with the expert running in shadow mode. When the
// learned driver leaves its lane, points the wrong way, or doesn't brake when the expert would
// be braking hard (for longer than brakeGrace: a driver sampled at 10 Hz can't react on the very
// step the expert's demand jumps), the expert takes over for a few seconds (a disengagement). The headline
// metric is autonomous distance per disengagement, as reported by real AV programs.
import { angleWrap } from './planner.js';

export const TAKEOVER_SECONDS = 3;

export class SafetyDriver {
  constructor({ maxLateral = 1.1, maxHeading = 0.45, brakeDemand = -3, brakeGrace = 0.3 } = {}) {
    Object.assign(this, { maxLateral, maxHeading, brakeDemand, brakeGrace });
    this.enabled = true;
    this.reset();
  }

  reset() {
    this.takeover = 0; // seconds of expert control left
    this.underBraking = 0; // seconds the learned driver has been braking less than the expert wants
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
    if (Math.abs(expert.lateral) > this.maxLateral) return 'left the lane';
    // Heading of the intended path here, including the swerve of an overtake.
    const p = expert.route.at(expert.s, expert.k), off = expert.ot.offsetAt;
    const pathH = p.h - Math.atan((off(expert.s + 1) - off(expert.s - 1)) / 2);
    if (Math.abs(angleWrap(car.h - pathH)) > this.maxHeading) return 'wrong heading';
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
