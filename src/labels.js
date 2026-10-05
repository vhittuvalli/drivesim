// Training labels for the driving network, computed from the privileged expert.
//
// The network sees the roof camera, its speed and a high-level command (the turn at the next
// intersection) and predicts, for every command branch, a path of waypoints in the car frame
// plus a target speed. Supervising all branches at once (not just the command that was taken)
// is the dense-supervision idea behind "Learning by Cheating": the privileged planner knows
// where each turn would have gone even when the car didn't take it.
//
// On the highway the same three commands mean: change to the left lane, keep the lane, change
// to the right lane. The command is the lane change the expert is making; while it keeps its
// lane, the left and right branches are labeled with the lane changes it could start now.
import { ROAD_W, STOP_LINE, mulberry32 } from './config.js';
import { Route } from './planner.js';
import { HW, HighwayRoute, changeDistance, laneOf } from './highway.js';

export const WP_DIST = [2, 4, 6, 8, 11, 14, 18, 23]; // meters along the path from the car
// Above WP_SCALE_SPEED the waypoints spread out in proportion to speed, so they keep covering about
// two seconds of driving: at 30 m/s the last one is 57 m ahead and a lane change is visible.
// Every city speed is below it, so city labels are the same as with fixed spacing.
export const WP_SCALE_SPEED = 12; // m/s
export const wpScale = (v) => Math.max(1, v / WP_SCALE_SPEED);
// Samples carry this; training drops version-1 rows above WP_SCALE_SPEED (fixed spacing).
export const LABEL_VERSION = 4; // 3: light label and the traffic-light camera frame; 4: lead obstacle
export const COMMANDS = ['left', 'straight', 'right'];
// The signal for our approach, as an auxiliary target that teaches the network to look at it:
// 'none' when there is no stop line within LIGHT_RANGE (or we're past it).
export const LIGHTS = ['none', 'red', 'yellow', 'green'];
export const LIGHT_RANGE = 90; // m, the expert's signal horizon (planner.signalObstacle)

// The obstacle the expert is following or stopping for (a vehicle or pedestrian on its path, or one
// about to cross it): front-bumper gap, including the room it leaves behind a stalled vehicle, and
// its speed along the path. With nothing within LEAD_FAR the road counts as clear: the gap is
// LEAD_FAR and the "lead" moves at our own speed.
export const LEAD_FAR = 80; // m

export function leadOf(exp, v) {
  const L = exp.lead;
  if (!L || !(L.gap < LEAD_FAR)) return { gap: LEAD_FAR, v };
  return { gap: Math.max(0, L.gap), v: L.v };
}

export function lightOf(exp) {
  const st = exp.signal;
  return st && st.dist > -1 && st.dist < LIGHT_RANGE && st.state ? st.state : 'none';
}
export const MAX_TARGET_SPEED = 34; // m/s (highway speeds; city driving stays under 14)
export const TARGET_HORIZON = 1; // s: target speed = speed the expert will have in this long

// Semantic classes in the label images (R channel = class id, G = depth / 100 m).
export const CLASSES = ['sky', 'road', 'marking', 'sidewalk', 'building', 'vegetation', 'pole', 'traffic light', 'vehicle', 'pedestrian', 'terrain'];
export const DEPTH_RANGE = 100;

// The command for the intersection we're approaching, or still crossing (on the highway: the
// lane change in progress).
export function commandOf(route, s) {
  if (route.highway) return route.commandAt(s);
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

// Waypoints along `route` from arc length s0, shifted left by offsetAt(s) when given, at
// WP_DIST times `scale`.
export function pathWaypoints(route, s0, pose, offsetAt = null, k = 0, scale = 1) {
  return WP_DIST.map((d0) => {
    const d = d0 * scale;
    const p = route.at(s0 + d, k), o = offsetAt ? offsetAt(s0 + d) : 0;
    return toEgo(pose, p.x + Math.sin(p.h) * o, p.z - Math.cos(p.h) * o);
  });
}

// Private RNG: building hypothetical routes must not disturb the simulation's random stream.
const labelRand = mulberry32(0x5eed);

// Waypoints for turning `kind` at the next intersection, from the lane the car is on.
function branchWaypoints(pose, kind, scale) {
  let first = true;
  const { route } = Route.fromPose(labelRand, pose.x, pose.z, pose.h, { choose: () => (first ? ((first = false), kind) : null) });
  if (!route) return null;
  route.ensure(WP_DIST[WP_DIST.length - 1] * scale + 30);
  if (route.turns[0]?.kind !== kind) return null; // that turn doesn't exist here (edge of the grid)
  return pathWaypoints(route, 0, pose, null, 0, scale);
}

// Waypoints for changing from highway lane `from` to `lane`, starting at progress q.
function laneChangeWaypoints(route, q, pose, from, lane, v) {
  if (lane < 0 || lane >= HW.lanes) return null;
  const hyp = new HighwayRoute({ dir: route.dir, lane: from, q });
  hyp.changeLane(0, lane, changeDistance(v));
  return pathWaypoints(hyp, 0, pose, null, 0, wpScale(v));
}

// Labels for the current state. `exp` is the expert's control output for this step (it may be
// shadowing another driver, which is what makes DAgger work). Needs the expert to be tracking
// its route, i.e. not while a human drives.
export function makeLabels(world, exp) {
  const { expert, car } = world;
  const route = expert.route, s = expert.s;
  const cmd = commandOf(route, s);
  const vTarget = Math.min(MAX_TARGET_SPEED, Math.max(0, car.v + exp.acc * TARGET_HORIZON));
  const scale = wpScale(car.v);
  const lead = leadOf(exp, car.v);
  if (route.highway) {
    const taken = pathWaypoints(route, s, car, null, expert.k, scale);
    const wp = { left: null, straight: null, right: null };
    wp[cmd.kind] = taken;
    // The lane the path is in here (route.lane is already the target once a change is planned).
    const q = route.qAt(s, expert.k), lane = laneOf(route.latAt(q));
    if (cmd.kind === 'straight') {
      wp.left = laneChangeWaypoints(route, q, car, lane, lane - 1, car.v);
      wp.right = laneChangeWaypoints(route, q, car, lane, lane + 1, car.v);
    }
    return { command: cmd.kind, cmdDist: null, wp, vTarget, overtaking: false, road: 'highway', lane, wpScale: scale, light: 'none', lead };
  }
  const overtaking = !!expert.ot.active;
  const taken = pathWaypoints(route, s, car, overtaking ? expert.ot.offsetAt : null, expert.k, scale);
  const wp = { left: null, straight: null, right: null };
  wp[cmd.kind] = taken;
  const horizon = WP_DIST[WP_DIST.length - 1] * scale + 6;
  for (const kind of COMMANDS) {
    if (kind === cmd.kind || overtaking) continue;
    // Far from the intersection every branch just follows the lane.
    wp[kind] = cmd.dist > horizon ? taken : branchWaypoints(car, kind, scale);
  }
  return { command: cmd.kind, cmdDist: Number.isFinite(cmd.dist) ? cmd.dist : null, wp, vTarget, overtaking, road: 'city', lane: null, wpScale: scale, light: lightOf(exp), lead };
}
