// Kinematic bicycle model. Commands are normalized to [-1, 1].

export const WHEELBASE = 2.7;
export const MAX_STEER = 0.55; // rad
export const MAX_SPEED = 16; // m/s
export const CAR_LENGTH = 4.5;
export const CAR_WIDTH = 1.9;

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class Car {
  constructor(x, y, h) {
    this.x = x;
    this.y = y;
    this.h = h;
    this.v = 0;
    this.steer = 0; // actual wheel angle, rad
  }

  step(dt, steerCmd, throttleCmd) {
    const target = clamp(steerCmd, -1, 1) * MAX_STEER;
    const maxRate = 2.5 * dt; // steering actuator rate limit
    this.steer += clamp(target - this.steer, -maxRate, maxRate);

    const t = clamp(throttleCmd, -1, 1);
    const accel = t >= 0 ? 4 * t : 8 * t;
    const drag = 0.002 * this.v * this.v;
    this.v = clamp(this.v + (accel - drag) * dt, 0, MAX_SPEED);

    this.h += (this.v / WHEELBASE) * Math.tan(this.steer) * dt;
    this.x += this.v * Math.cos(this.h) * dt;
    this.y += this.v * Math.sin(this.h) * dt;
  }
}
