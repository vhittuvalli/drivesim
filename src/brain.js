// End-to-end driving policy: camera pixels + speed -> [steer, throttle].
// Trained by behavioral cloning on (image, expert action) pairs, all in the browser via TF.js.
/* global tf */

import { SW, SH } from './sensor.js';
import { MAX_SPEED } from './car.js';

const PIX = SW * SH * 3;
const STORAGE_KEY = 'indexeddb://drivesim-pilot';

// Ring buffer of training samples; oldest are overwritten first (keeps DAgger data fresh).
export class Dataset {
  constructor(capacity = 8000) {
    this.capacity = capacity;
    this.images = new Uint8Array(capacity * PIX);
    this.speeds = new Float32Array(capacity);
    this.labels = new Float32Array(capacity * 2);
    this.count = 0;
    this.head = 0;
  }

  add(rgb, v, steer, throttle) {
    const i = this.head;
    this.images.set(rgb, i * PIX);
    this.speeds[i] = v / MAX_SPEED;
    this.labels[i * 2] = steer;
    this.labels[i * 2 + 1] = throttle;
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
  }

  clear() {
    this.count = 0;
    this.head = 0;
  }

  batch(indices) {
    const n = indices.length;
    const img = new Float32Array(n * PIX), spd = new Float32Array(n), y = new Float32Array(n * 2);
    indices.forEach((k, j) => {
      const base = k * PIX, out = j * PIX;
      for (let p = 0; p < PIX; p++) img[out + p] = this.images[base + p] / 255;
      spd[j] = this.speeds[k];
      y[j * 2] = this.labels[k * 2];
      y[j * 2 + 1] = this.labels[k * 2 + 1];
    });
    return [tf.tensor4d(img, [n, SH, SW, 3]), tf.tensor2d(spd, [n, 1]), tf.tensor2d(y, [n, 2])];
  }
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function compile(model) {
  model.compile({ optimizer: tf.train.adam(1e-3), loss: 'meanSquaredError' });
  return model;
}

function buildModel() {
  const img = tf.input({ shape: [SH, SW, 3], name: 'camera' });
  const spd = tf.input({ shape: [1], name: 'speed' });
  let x = img;
  for (const [filters, k] of [[16, 5], [32, 3], [48, 3], [64, 3]]) {
    x = tf.layers.conv2d({ filters, kernelSize: k, strides: 2, padding: 'same', activation: 'relu' }).apply(x);
  }
  x = tf.layers.flatten().apply(x);
  x = tf.layers.concatenate().apply([x, spd]);
  x = tf.layers.dense({ units: 100, activation: 'relu' }).apply(x);
  x = tf.layers.dropout({ rate: 0.2 }).apply(x);
  x = tf.layers.dense({ units: 50, activation: 'relu' }).apply(x);
  const out = tf.layers.dense({ units: 2, activation: 'tanh', name: 'controls' }).apply(x);
  return compile(tf.model({ inputs: [img, spd], outputs: out }));
}

export class Brain {
  constructor() {
    this.model = buildModel();
    this.trained = false;
    this.training = false;
    this.buf = new Float32Array(PIX);
  }

  paramCount() {
    return this.model.countParams();
  }

  predict(rgb, v) {
    for (let i = 0; i < PIX; i++) this.buf[i] = rgb[i] / 255;
    return tf.tidy(() => {
      const out = this.model.predict([tf.tensor4d(this.buf, [1, SH, SW, 3]), tf.tensor2d([v / MAX_SPEED], [1, 1])]);
      return Array.from(out.dataSync());
    });
  }

  async train(ds, { epochs = 8, batchSize = 64, onProgress = () => {}, onEpoch = () => {} } = {}) {
    this.training = true;
    try {
      const idx = shuffle([...Array(ds.count).keys()]);
      const nVal = Math.max(1, Math.floor(idx.length * 0.1));
      const val = idx.slice(0, nVal);
      const train = idx.slice(nVal);
      const batchesPerEpoch = Math.ceil(train.length / batchSize);

      for (let e = 0; e < epochs; e++) {
        shuffle(train);
        let sum = 0, nb = 0;
        for (let i = 0; i < train.length; i += batchSize) {
          const [xi, xs, y] = ds.batch(train.slice(i, i + batchSize));
          const loss = await this.model.trainOnBatch([xi, xs], y);
          tf.dispose([xi, xs, y]);
          sum += Array.isArray(loss) ? loss[0] : loss;
          nb++;
          onProgress((e + nb / batchesPerEpoch) / epochs);
          if (nb % 4 === 0) await tf.nextFrame();
        }

        let vsum = 0, vn = 0;
        for (let i = 0; i < val.length; i += 256) {
          const chunk = val.slice(i, i + 256);
          const [xi, xs, y] = ds.batch(chunk);
          const l = this.model.evaluate([xi, xs], y, { batchSize: 256 });
          vsum += (await l.data())[0] * chunk.length;
          vn += chunk.length;
          tf.dispose([xi, xs, y, l]);
        }
        this.trained = true;
        onEpoch(e + 1, sum / nb, vsum / vn);
      }
    } finally {
      this.training = false;
    }
  }

  reset() {
    this.model.dispose();
    this.model = buildModel();
    this.trained = false;
  }

  async save() {
    await this.model.save(STORAGE_KEY);
  }

  async load() {
    const m = compile(await tf.loadLayersModel(STORAGE_KEY));
    this.model.dispose();
    this.model = m;
    this.trained = true;
  }
}
