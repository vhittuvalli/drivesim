// Dataset recorder: renders the roof camera at a fixed rate into a small offscreen target and
// saves JPEG frames plus a CSV of the driving commands and state (for imitation learning).
// The download is a .zip: frames/NNNNNN.jpg, labels.csv, meta.json.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { zipStore } from './zip.js';

const COLUMNS = ['frame', 't', 'steer', 'throttle', 'speed', 'accel', 'x', 'z', 'heading', 'source', 'reason', 'next_turn', 'overtake', 'weather', 'hour'];
const MAX_FRAMES = 6000; // 10 minutes at 10 Hz

export class Recorder {
  constructor(renderer, scene, { width = 320, height = 160, hz = 10 } = {}) {
    Object.assign(this, { renderer, scene, width, height, hz });
    this.cam = new THREE.PerspectiveCamera(70, width / height, 0.1, 2000);
    // Tone mapping and sRGB conversion only happen in OutputPass when rendering offscreen.
    this.composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(width, height, { type: THREE.UnsignedByteType }));
    this.composer.renderToScreen = false;
    this.composer.setPixelRatio(1);
    this.composer.setSize(width, height);
    this.composer.addPass(new RenderPass(scene, this.cam));
    this.composer.addPass(new OutputPass());
    this.pixels = new Uint8Array(width * height * 4);
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d');
    this.active = false;
    this.frames = [];
    this.rows = [];
  }

  get count() {
    return this.rows.length;
  }

  start(meta) {
    this.active = true;
    this.meta = { ...meta, width: this.width, height: this.height, hz: this.hz, fov: this.cam.fov, started: new Date().toISOString() };
    this.frames = [];
    this.rows = [];
    this.nextT = -Infinity;
  }

  // Call once per rendered frame with the simulation time; captures at `hz` of sim time.
  capture(simT, car, ctrl, info) {
    if (!this.active || simT < this.nextT) return;
    this.nextT = Math.max(this.nextT + 1 / this.hz, simT);
    if (this.rows.length >= MAX_FRAMES) return this.stop();

    const fx = Math.cos(car.h), fz = Math.sin(car.h);
    this.cam.position.set(car.x + fx * 0.05, 1.62, car.z + fz * 0.05);
    this.cam.lookAt(car.x + fx * 20, 1.2, car.z + fz * 20);
    this.composer.render();
    const r = this.renderer;
    r.readRenderTargetPixels(this.composer.readBuffer, 0, 0, this.width, this.height, this.pixels);
    r.setRenderTarget(null);

    // GL rows are bottom-up.
    const img = this.ctx.createImageData(this.width, this.height), row = this.width * 4;
    for (let y = 0; y < this.height; y++) img.data.set(this.pixels.subarray((this.height - 1 - y) * row, (this.height - y) * row), y * row);
    this.ctx.putImageData(img, 0, 0);
    const n = this.rows.length + 1;
    this.frames.push(new Promise((res) => this.canvas.toBlob(res, 'image/jpeg', 0.9)));
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
