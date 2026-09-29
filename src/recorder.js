// Dataset recorder: renders the roof camera at a fixed rate into a small offscreen target and
// saves JPEG frames plus a CSV of the driving commands and state (for imitation learning).
// The download is a .zip: frames/NNNNNN.jpg, labels.csv, meta.json.
import { SensorRig } from './sensor.js';
import { zipStore } from './zip.js';

const COLUMNS = ['frame', 't', 'steer', 'throttle', 'speed', 'accel', 'x', 'z', 'heading', 'source', 'reason', 'next_turn', 'overtake', 'weather', 'hour'];
const MAX_FRAMES = 6000; // 10 minutes at 10 Hz

export class Recorder {
  constructor(renderer, scene, { width = 320, height = 160, hz = 10 } = {}) {
    Object.assign(this, { width, height, hz });
    this.rig = new SensorRig(renderer, scene, { width, height });
    this.active = false;
    this.frames = [];
    this.rows = [];
  }

  get count() {
    return this.rows.length;
  }

  start(meta) {
    this.active = true;
    this.meta = { ...meta, width: this.width, height: this.height, hz: this.hz, fov: this.rig.cam.fov, started: new Date().toISOString() };
    this.frames = [];
    this.rows = [];
    this.nextT = -Infinity;
  }

  // Call once per rendered frame with the simulation time; captures at `hz` of sim time.
  capture(simT, car, ctrl, info) {
    if (!this.active || simT < this.nextT) return;
    this.nextT = Math.max(this.nextT + 1 / this.hz, simT);
    if (this.rows.length >= MAX_FRAMES) return this.stop();

    const n = this.rows.length + 1;
    this.frames.push(this.rig.encode(this.rig.renderRGB(car), 'image/jpeg', 0.9));
    this.rows.push([
      n, simT.toFixed(3), ctrl.steer.toFixed(4), ctrl.throttle.toFixed(4), car.v.toFixed(3), (car.accel ?? 0).toFixed(3),
      car.x.toFixed(2), car.z.toFixed(2), car.h.toFixed(4), ctrl.manual ? 'manual' : 'expert', ctrl.reason ?? '',
      ctrl.nextTurn?.kind ?? '', ctrl.overtake?.state ?? '', info.weather, info.hour.toFixed(2),
    ]);
  }

  // Stop and download everything recorded so far.
  async stop() {
    if (!this.active) return;
    this.active = false;
    if (!this.rows.length) return;
    const blobs = await Promise.all(this.frames);
    const files = await Promise.all(blobs.map(async (b, i) => ({ name: `frames/${String(i + 1).padStart(6, '0')}.jpg`, data: new Uint8Array(await b.arrayBuffer()) })));
    files.push({ name: 'labels.csv', data: [COLUMNS, ...this.rows].map((r) => r.join(',')).join('\n') + '\n' });
    files.push({ name: 'meta.json', data: JSON.stringify({ ...this.meta, frames: this.rows.length, columns: COLUMNS }, null, 2) });
    const url = URL.createObjectURL(new Blob([zipStore(files)], { type: 'application/zip' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: `drivesim-seed${this.meta.seed}-${Date.now()}.zip` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    this.frames = [];
  }
}
