// Training-data collection, streamed to the dev server (scripts/serve.py writes data/<run>/):
//   frames/NNNNNN.jpg   roof camera (what the network sees)
//   labels/NNNNNN.png   semantic class (R) and depth (G) of the same view
//   samples.jsonl       one line per frame: command, waypoints for every command branch, target
//                       speed, and what was actually applied (see labels.js)
// Labels always come from the expert, also while the neural driver is in control; that's
// DAgger: the network's own mistakes get labeled with the expert's correction.
import { makeLabels } from './labels.js';

const FLUSH_EVERY = 50;
const MAX_IN_FLIGHT = 64;

async function post(run, name, body, append = false) {
  const r = await fetch(`/api/collect?run=${encodeURIComponent(run)}&name=${encodeURIComponent(name)}${append ? '&append=1' : ''}`, { method: 'POST', body });
  if (!r.ok) throw new Error(`upload ${name}: ${r.status}`);
}

export class Collector {
  constructor(rig) {
    this.rig = rig;
    this.active = false;
    this.inFlight = 0;
    this.count = 0;
    this.error = null;
  }

  // Uploads can't keep up: the caller should stop stepping the sim for a moment.
  get backlogged() {
    return this.inFlight > MAX_IN_FLIGHT;
  }

  start(run, meta) {
    Object.assign(this, { run, active: true, count: 0, rows: [], error: null });
    this.send('meta.json', JSON.stringify({ ...meta, started: new Date().toISOString(), sensor: { width: this.rig.width, height: this.rig.height, fov: this.rig.cam.fov } }, null, 2));
  }

  stop() {
    if (!this.active) return;
    this.flush();
    this.active = false;
  }

  send(name, body, append) {
    this.inFlight++;
    post(this.run, name, body, append)
      .catch((e) => (this.error = e.message))
      .finally(() => this.inFlight--);
  }

  async sendBlob(name, blobPromise) {
    this.inFlight++;
    try {
      await post(this.run, name, await blobPromise);
    } catch (e) {
      this.error = e.message;
    } finally {
      this.inFlight--;
    }
  }

  flush() {
    if (!this.rows.length) return;
    this.send('samples.jsonl', this.rows.map((r) => JSON.stringify(r)).join('\n') + '\n', true);
    this.rows = [];
  }

  // rgb: the ImageData already rendered for this instant. info: {weather, hour, scenario}.
  capture(world, rgb, info) {
    if (!this.active || world.manual || !world.expertCtrl) return;
    const n = ++this.count, id = String(n).padStart(6, '0');
    const lab = makeLabels(world, world.expertCtrl);
    const c = world.ctrl ?? world.expertCtrl;
    this.sendBlob(`frames/${id}.jpg`, this.rig.encode(rgb, 'image/jpeg', 0.92));
    this.sendBlob(`labels/${id}.png`, this.rig.encode(this.rig.renderLabels(world.car), 'image/png'));
    const r2 = (v) => Math.round(v * 100) / 100;
    this.rows.push({
      frame: n, t: r2(world.t), v: r2(world.car.v), command: lab.command, cmd_dist: lab.cmdDist === null ? null : r2(lab.cmdDist),
      wp: Object.fromEntries(Object.entries(lab.wp).map(([k, pts]) => [k, pts && pts.map(([x, y]) => [r2(x), r2(y)])])),
      v_target: r2(lab.vTarget), acc: r2(world.expertCtrl.acc), reason: world.expertCtrl.reason ?? null, overtaking: lab.overtaking,
      steer: r2(c.steer), throttle: r2(c.throttle), driver: c.driver ?? 'expert', noise: r2(c.noise ?? 0),
      weather: info.weather, hour: r2(info.hour), scenario: info.scenario || null,
    });
    if (this.rows.length >= FLUSH_EVERY) this.flush();
  }
}

// Correlated steering noise with occasional larger swerves, so the data shows the car drifting
// off the path and the expert's labels pulling it back (recovery data).
export function makeSteerNoise(rand = Math.random) {
  let n = 0, pulse = 0, pulseV = 0, next = 3 + rand() * 5;
  return (dt) => {
    const g = Math.sqrt(-2 * Math.log(rand() + 1e-9)) * Math.cos(2 * Math.PI * rand());
    n += -n * 1.5 * dt + Math.sqrt(dt) * 0.08 * g;
    next -= dt;
    if (next <= 0) {
      pulse = 0.5 + rand() * 0.8;
      pulseV = (rand() < 0.5 ? -1 : 1) * (0.12 + rand() * 0.22);
      next = 5 + rand() * 8;
    }
    if (pulse > 0) {
      pulse -= dt;
      return n + pulseV;
    }
    return n;
  };
}
