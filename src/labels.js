// Training labels for the driving network, computed from the privileged expert.
//
// The network sees the roof camera, its speed and a high-level command (the turn at the next
// intersection) and predicts, for every command branch, a path of waypoints in the car frame
// plus a target speed. Supervising all branches at once (not just the command that was taken)
// is the dense-supervision idea behind "Learning by Cheating": the privileged planner knows
// where each turn would have gone even when the car didn't take it.
import { ROAD_W, STOP_LINE, mulberry32 } from './config.js';
import { Route } from './planner.js';

export const WP_DIST = [2, 4, 6, 8, 11, 14, 18, 23]; // meters along the path from the car
export const COMMANDS = ['left', 'straight', 'right'];
export const MAX_TARGET_SPEED = 14; // m/s
export const TARGET_HORIZON = 1; // s: target speed = speed the expert will have in this long

// Semantic classes in the label images (R channel = class id, G = depth / 100 m).
export const CLASSES = ['sky', 'road', 'marking', 'sidewalk', 'building', 'vegetation', 'pole', 'traffic light', 'vehicle', 'pedestrian', 'terrain'];
export const DEPTH_RANGE = 100;

// The command for the intersection we're approaching, or still crossing.
export function commandOf(route, s) {
  for (const st of route.stops) {
    // The box ends ROAD_W past the intersection entry, which is (STOP_LINE - ROAD_W/2) past the line.
    if (st.s + STOP_LINE - ROAD_W / 2 + ROAD_W > s) return { kind: st.turn ?? 'straight', dist: st.s - s };
  }
  return { kind: 'straight', dist: Infinity };
}

// (x forward, y left) of a world point in the frame of pose.
export function toEgo(pose, x, z) {
  const dx = x - pose.x, dz = z - pose.z, c = Math.cos(pose.h), s = Math.sin(pose.h);
  return [dx * c + dz * s, dx * s - dz * c];
}

// Waypoints along `route` from arc length s0, shifted left by offsetAt(s) when given.
export function pathWaypoints(route, s0, pose, offsetAt = null, k = 0) {
  return WP_DIST.map((d) => {
    const p = route.at(s0 + d, k), o = offsetAt ? offsetAt(s0 + d) : 0;
    return toEgo(pose, p.x + Math.sin(p.h) * o, p.z - Math.cos(p.h) * o);
  });
}

// Private RNG: building hypothetical routes must not disturb the simulation's random stream.
const labelRand = mulberry32(0x5eed);

// Waypoints for turning `kind` at the next intersection, from the lane the car is on.
function branchWaypoints(pose, kind) {
  let first = true;
  const { route } = Route.fromPose(labelRand, pose.x, pose.z, pose.h, { choose: () => (first ? ((first = false), kind) : null) });
  if (!route) return null;
  route.ensure(WP_DIST[WP_DIST.length - 1] + 30);
  if (route.turns[0]?.kind !== kind) return null; // that turn doesn't exist here (edge of the grid)
  return pathWaypoints(route, 0, pose);
}

// Labels for the current state. `exp` is the expert's control output for this step (it may be
// shadowing another driver, which is what makes DAgger work). Needs the expert to be tracking
// its route, i.e. not while a human drives.
export function makeLabels(world, exp) {
  const { expert, car } = world;
  const route = expert.route, s = expert.s;
  const cmd = commandOf(route, s);
  const overtaking = !!expert.ot.active;
  const taken = pathWaypoints(route, s, car, overtaking ? expert.ot.offsetAt : null, expert.k);
  const wp = { left: null, straight: null, right: null };
  wp[cmd.kind] = taken;
  const horizon = WP_DIST[WP_DIST.length - 1] + 6;
  for (const kind of COMMANDS) {
    if (kind === cmd.kind || overtaking) continue;
    // Far from the intersection every branch just follows the lane.
    wp[kind] = cmd.dist > horizon ? taken : branchWaypoints(car, kind);
  }
  const vTarget = Math.min(MAX_TARGET_SPEED, Math.max(0, car.v + exp.acc * TARGET_HORIZON));
  return { command: cmd.kind, cmdDist: Number.isFinite(cmd.dist) ? cmd.dist : null, wp, vTarget, overtaking };
}
