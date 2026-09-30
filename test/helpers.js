// Shared test utilities.
import { clamp, conditions, WHEELBASE, MAX_STEER, MAX_SPEED, drag } from '../src/config.js';
import { BODY_TYPES } from '../src/bodytypes.js';

// Same dynamics as src/vehicle.js without the three.js mesh.
export class Car {
  constructor(x, z, h) {
    Object.assign(this, { x, z, h, v: 0, steer: 0 });
  }

  step(dt, s, t) {
    const tg = clamp(s, -1, 1) * MAX_STEER;
    this.steer += clamp(tg - this.steer, -1.6 * dt, 1.6 * dt);
    const a = t >= 0 ? 3.2 * t * Math.min(1, conditions.grip * 1.3) : 7.5 * t * conditions.grip;
    this.v = clamp(this.v + (a - drag(this.v)) * dt, 0, MAX_SPEED);
    this.h += (this.v / WHEELBASE) * Math.tan(this.steer) * dt;
    this.x += this.v * Math.cos(this.h) * dt;
    this.z += this.v * Math.sin(this.h) * dt;
  }

  reset(x, z, h, v = 0) {
    Object.assign(this, { x, z, h, v, steer: 0 });
  }
}

// Minimal stand-in for the instanced renderer.
export class FakeFleet {
  acquire(type) {
    return { type, idx: 0, spec: BODY_TYPES[type] };
  }
  release() {}
  set() {}
}

// Is point (px, pz) inside a vehicle footprint (half-length hl, half-width hw) centered at a?
export function inFootprint(a, px, pz, hl, hw) {
  const dx = px - a.x, dz = pz - a.z;
  const lx = dx * Math.cos(a.h) + dz * Math.sin(a.h), ly = -dx * Math.sin(a.h) + dz * Math.cos(a.h);
  return Math.abs(lx) < hl && Math.abs(ly) < hw;
}
