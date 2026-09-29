// Privileged "expert" driver and safety supervisor.
// Both read ground-truth world state (like HD maps + V2X). The expert is the teacher the
// neural net imitates; the supervisor is a rule-based safety layer that can only brake.

import { LANE_OFFSET } from './world.js';
import { WHEELBASE, MAX_STEER, clamp } from './car.js';

const CRUISE = 12; // m/s
const A_LAT = 3.5; // comfortable lateral accel
const A_BRAKE = 3; // comfortable braking profile

const angleWrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export function laneBlocked(world, s, lane, from = -6, to = 35) {
  return world.cones.some((c) => {
    const g = world.signedGap(s, c.s);
    return g > from && g < to && Math.abs(c.off - lane) < 1.8;
  });
}

export function nextLight(world, s) {
  let light = null, dist = Infinity;
  for (const l of world.lights) {
    const d = world.ahead(s, l.s);
    if (d < dist) { dist = d; light = l; }
  }
  return { light, dist };
}

// Max speed allowed right now so the car can stop before the next non-green light.
export function lightSpeedLimit(world, s, v) {
  const { light, dist } = nextLight(world, s);
  const stopDist = dist - 3.5;
  if (!light || light.state === 'green' || dist > 70 || stopDist < -1) return Infinity;
  const canStop = (v * v) / (2 * 5) < stopDist + 1.5;
  if (light.state === 'red' || canStop) return Math.sqrt(2 * A_BRAKE * Math.max(0, stopDist));
  return Infinity; // yellow and too close to stop: proceed
}

export function expertControl(world, car, nav) {
  const lane = laneBlocked(world, nav.s, LANE_OFFSET) ? -LANE_OFFSET : LANE_OFFSET;

  // Pure pursuit toward a lookahead point in the chosen lane.
  const Ld = 5 + 0.55 * car.v;
  const p = world.pts[world.indexAt(nav.s + Ld)];
  const gx = p.x + p.nx * lane, gy = p.y + p.ny * lane;
  const alpha = angleWrap(Math.atan2(gy - car.y, gx - car.x) - car.h);
  const ld = Math.hypot(gx - car.x, gy - car.y);
  const steer = clamp(Math.atan((2 * WHEELBASE * Math.sin(alpha)) / ld) / MAX_STEER, -1, 1);

  // Speed: cruise, slow for curvature, stop for lights.
  const h0 = world.pts[nav.idx].h;
  const h1 = world.pts[world.indexAt(nav.s + 30)].h;
  const kappa = Math.abs(angleWrap(h1 - h0)) / 30;
  let vt = Math.min(CRUISE, Math.sqrt(A_LAT / Math.max(kappa, 1e-4)));
  vt = Math.min(vt, lightSpeedLimit(world, nav.s, car.v));
  const throttle = vt < 0.1 && car.v < 0.5 ? -1 : clamp(0.9 * (vt - car.v), -1, 1);

  return { steer, throttle, lane, target: { x: gx, y: gy }, vt };
}

// Rule-based safety layer on top of any driver. Only ever reduces throttle.
export function supervise(world, car, nav, cmd) {
  let throttle = cmd.throttle, reason = null;

  const vLim = lightSpeedLimit(world, nav.s, car.v);
  if (car.v > vLim + 0.5 || (vLim < 0.1 && car.v < 1)) {
    throttle = Math.min(throttle, vLim < 0.1 ? -1 : -0.7);
    reason = 'light';
  }

  const brakeDist = 4 + (car.v * car.v) / (2 * 6);
  for (const c of world.cones) {
    const g = world.signedGap(nav.s, c.s);
    if (g > 0 && g < brakeDist && Math.abs(c.off - nav.off) < 1.7) {
      throttle = -1;
      reason = 'obstacle';
      break;
    }
  }
  return { steer: cmd.steer, throttle, reason };
}
