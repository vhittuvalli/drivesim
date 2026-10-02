// Neural driver: runs the trained policy (models/policy.onnx, see train/) in the browser with
// ONNX Runtime Web and plugs into World.setPolicy(). The network sees the roof camera (and, if it
// was trained with one, the traffic-light camera) and the car's speed; the route's next turn picks
// the command branch (conditional imitation learning).
//
// Observations are taken at a fixed rate of simulation time and the simulation waits for each
// answer (see due / waiting), so the driver behaves the same however long inference takes.
// Between observations the predicted waypoints are held in world coordinates and followed with
// pure pursuit; the predicted target speed sets the throttle.
//
// Light-aware speed: networks with light and stop-line outputs read the signal ahead, but their
// speed head didn't reliably act on it. When the network is confident the light is red (or amber
// with room to stop), acceleration is capped so the car stops just short of the stop line the
// network predicts, and the brake is held there. Everything the cap uses comes from the network.
import { clamp, WHEELBASE, MAX_STEER } from './config.js';
import { throttleFor } from './planner.js';
import { commandOf, toEgo, TARGET_HORIZON, COMMANDS } from './labels.js';

const ORT_VERSION = '1.30.0';
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

// Catmull-Rom through the car position and the waypoints, a few points per segment: pure pursuit
// on the bare waypoints (up to 5 m apart) would cut tight corners.
function smoothPath(pts, per = 4) {
  const P = [[0, 0], ...pts], out = [];
  for (let i = 0; i < P.length - 1; i++) {
    const p0 = P[Math.max(0, i - 1)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(P.length - 1, i + 2)];
    for (let k = 1; k <= per; k++) {
      const t = k / per, t2 = t * t, t3 = t2 * t;
      const f = (a, b, c, d) => 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - a - 3 * c + d) * t3);
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  return out;
}

// Pure pursuit along a path (car frame: x forward, y left) plus a speed target -> controls,
// with the same geometry and actuator limits as the expert (planner.js). accCap: an upper bound on
// the acceleration (light-aware speed), or null.
export function followPath(pts, v, vTarget, accCap = null) {
  const Ld = 4 + 0.45 * v;
  let prev = [0, 0], tgt = null;
  for (const p of smoothPath(pts)) {
    if (p[0] < 0.3) continue; // passed already (the car moved since the prediction)
    const d = Math.hypot(p[0], p[1]);
    if (d >= Ld) {
      // Interpolate to the point at distance Ld on this segment.
      const d0 = Math.hypot(prev[0], prev[1]), u = clamp((Ld - d0) / (d - d0 || 1), 0, 1);
      tgt = [prev[0] + (p[0] - prev[0]) * u, prev[1] + (p[1] - prev[1]) * u];
      break;
    }
    prev = p;
  }
  tgt ??= prev[0] > 0.3 ? prev : [Ld, 0];
  const alpha = Math.atan2(-tgt[1], tgt[0]); // positive: target is to the right
  const ld = Math.max(1, Math.hypot(tgt[0], tgt[1]));
  const steer = clamp(Math.atan((2 * WHEELBASE * Math.sin(alpha)) / ld) / MAX_STEER, -1, 1);
  let acc = (vTarget - v) / TARGET_HORIZON;
  // The label is clipped at 0 (labels.js), so a zero target understates hard braking at low
  // speed (a stop in well under the horizon): brake at least this firmly.
  if (vTarget < 0.3 && v > 0.3) acc = Math.min(acc, -Math.max(3, 2 * v));
  if (accCap !== null) acc = Math.min(acc, accCap);
  acc = clamp(acc, -7.5, 3.2);
  let throttle = throttleFor(acc, v);
  if (v < 0.5 && (vTarget < 0.25 || (accCap !== null && accCap <= 0))) throttle = -0.5; // hold the brake instead of creeping
  return { steer, throttle, acc, target: tgt };
}

export const LIGHT_STOP = { belief: 0.6, comfortDecel: 0.8, yellowDecel: 4.5, margin: 1, range: 50 };

// Acceleration cap for a signal ahead believed red (or amber that we can stop for) whose stop line
// is `d` meters past the front bumper, at speed v; null if there's nothing to stop for.
export function lightCap(v, d, pRed, pYellow, belief = pRed + pYellow) {
  if (!(d > -1 && d < LIGHT_STOP.range)) return null;
  // Amber: stop only if a firm stop fits before the line (like the expert, planner.signalObstacle).
  const canStop = (v * v) / (2 * LIGHT_STOP.yellowDecel) < d + 0.5;
  const pStop = pRed + (canStop ? pYellow : 0);
  if (belief < LIGHT_STOP.belief || pStop < LIGHT_STOP.belief) return null;
  const room = Math.max(d - LIGHT_STOP.margin, 0.3);
  const need = -(v * v) / (2 * room); // constant deceleration that stops at the margin
  // Far enough away: just don't speed up toward the light; brake once it takes real deceleration.
  return need > -LIGHT_STOP.comfortDecel ? Math.min(0, need) : need;
}

// ImageData (RGBA, rows top to bottom) -> float32 CHW in [0, 1].
function toTensorData(img) {
  const n = img.width * img.height, src = img.data, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = src[i * 4] / 255;
    out[n + i] = src[i * 4 + 1] / 255;
    out[2 * n + i] = src[i * 4 + 2] / 255;
  }
  return out;
}

export class NeuralDriver {
  constructor({ hz = 10 } = {}) {
    this.period = 1 / hz;
    this.session = null;
    this.meta = null;
    this.commands = COMMANDS; // order of the network's command branches
    this.inferMs = 0;
    this.usesTele = false;
    this.lightAware = true; // apply the light-aware speed cap when the network has light outputs
    this.reset();
  }

  get ready() {
    return !!this.session;
  }

  // Loads models/<base>.onnx and its metadata. Throws with a readable message if missing.
  async load(base = 'models/policy') {
    const res = await fetch(`${base}.json`);
    if (!res.ok) throw new Error(`no trained model at ${base}.onnx (collect data, then npm run train)`);
    this.meta = await res.json();
    const ort = await import('onnxruntime-web');
    ort.env.wasm.wasmPaths = ORT_DIST;
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    this.ort = ort;
    this.session = await ort.InferenceSession.create(`${base}.onnx`, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    this.commands = this.meta.commands;
    [, , this.inH, this.inW] = this.meta.inputs.image;
    this.usesTele = !!this.meta.inputs.tele;
    return this.meta;
  }

  // Forget the current prediction (after the car is teleported).
  reset() {
    this.pred = null; // {pose, t, cmd, wp: world-frame paths per command, vTarget[], seg, attention, image}
    this.pending = false;
    this.nextObs = -Infinity;
    this.error = null;
    this.stopBelief = 0; // smoothed P(red or amber) over observations
  }

  // The simulation must not step past this time without a fresh observation.
  waiting(t) {
    return this.pending || t >= this.nextObs;
  }

  // An observation should be rendered and sent now.
  due(t) {
    return this.ready && !this.pending && t >= this.nextObs;
  }

  // Run the network on `rgb` (the roof camera at the car's current pose) and `tele` (the
  // traffic-light camera, for networks that use it). Resolves when the prediction is in place;
  // the simulation stays paused until then.
  async observe(world, rgb, tele = null) {
    const { car, expert } = world;
    const pose = { x: car.x, z: car.z, h: car.h }, t = world.t, v = car.v;
    const cmd = commandOf(expert.route, expert.s).kind;
    this.pending = true;
    this.nextObs = t + this.period;
    const t0 = performance.now();
    try {
      const { Tensor } = this.ort;
      const feeds = {
        image: new Tensor('float32', toTensorData(rgb), [1, 3, this.inH, this.inW]),
        speed: new Tensor('float32', Float32Array.of(v), [1, 1]),
      };
      if (this.usesTele) feeds.tele = new Tensor('float32', toTensorData(tele), [1, 3, ...this.meta.inputs.tele.slice(2)]);
      const out = await this.session.run(feeds);
      // Probabilities for the light ahead (networks trained with the traffic-light camera).
      let light = null;
      if (out.light) {
        const z = Array.from(out.light.data), m = Math.max(...z), e = z.map((x) => Math.exp(x - m)), sum = e.reduce((a, b) => a + b, 0);
        light = Object.fromEntries(this.meta.lights.map((k, i) => [k, e[i] / sum]));
      }
      const wp = out.waypoints.data, nWp = this.meta.n_waypoints;
      const branches = this.commands.map((_, k) => Array.from({ length: nWp }, (_, i) => [wp[(k * nWp + i) * 2], wp[(k * nWp + i) * 2 + 1]]));
      this.setPrediction(pose, cmd, branches, Array.from(out.v_target.data), {
        t, v, seg: out.seg.data, segDims: out.seg.dims, attention: out.attention.data, attnDims: out.attention.dims, image: rgb, tele, light,
        stopDist: out.stop_dist ? out.stop_dist.data[0] : null,
      });
      for (const k of Object.keys(out)) out[k].dispose?.();
      const ms = performance.now() - t0;
      this.inferMs = this.inferMs ? this.inferMs * 0.9 + ms * 0.1 : ms;
    } catch (e) {
      this.error = e.message;
      console.error('Neural driver inference failed', e);
    } finally {
      this.pending = false;
    }
  }

  // branches: per command, waypoints in the frame of `pose` (x forward, y left). They're stored
  // in world coordinates so the path stays put as the car moves until the next observation.
  setPrediction(pose, cmd, branches, vTarget, extra = {}) {
    const c = Math.cos(pose.h), s = Math.sin(pose.h);
    const paths = branches.map((pts) => pts.map(([x, y]) => ({ x: pose.x + x * c + y * s, z: pose.z + x * s - y * c })));
    // One noisy frame shouldn't stop the car (or release it): smooth the light over observations.
    if (extra.light) this.stopBelief = 0.5 * this.stopBelief + 0.5 * (extra.light.red + extra.light.yellow);
    this.pred = { pose, cmd, cmdIndex: this.commands.indexOf(cmd), paths, vTarget, ...extra };
  }

  // The path the network wants to drive for the current command, in world coordinates.
  get path() {
    return this.pred ? this.pred.paths[this.pred.cmdIndex] : null;
  }

  // World.setPolicy() interface. Returns null (the expert drives) until there's a prediction.
  control(world) {
    const p = this.pred;
    if (!p) return null;
    const car = world.car;
    const pts = this.path.map((q) => toEgo(car, q.x, q.z));
    let cap = null;
    if (this.lightAware && p.light && p.stopDist !== null && p.stopDist !== undefined) {
      const d = p.stopDist - toEgo(p.pose, car.x, car.z)[0]; // the line is that much closer now
      cap = lightCap(car.v, d, p.light.red, p.light.yellow, this.stopBelief);
    }
    const out = followPath(pts, car.v, p.vTarget[p.cmdIndex], cap);
    return { steer: out.steer, throttle: out.throttle, acc: out.acc, cmd: p.cmd, vTarget: p.vTarget[p.cmdIndex], lightStop: cap !== null };
  }
}
