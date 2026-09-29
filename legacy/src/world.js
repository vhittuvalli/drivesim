// Procedural world: a closed-loop two-lane road with traffic lights and cone obstacles.
// Units are meters. Coordinates are canvas-style (y points down).
// Lateral offset convention: positive = right lane (driver's right), negative = left lane.

export const ROAD_WIDTH = 8;
export const LANE_OFFSET = 2;
export const GRASS = '#2f4a2a';
export const LIGHT_PHASES = [['green', 10], ['yellow', 2.5], ['red', 7]];
export const LIGHT_COLORS = { red: '#ff2a2a', yellow: '#ffc400', green: '#22dd55' };
const CYCLE = LIGHT_PHASES.reduce((a, [, d]) => a + d, 0);

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function lightState(t) {
  let tt = t % CYCLE;
  for (const [state, dur] of LIGHT_PHASES) {
    if (tt < dur) return state;
    tt -= dur;
  }
  return 'green';
}

export class World {
  constructor(seed = (Math.random() * 2 ** 31) | 0) {
    this.seed = seed;
    const rand = mulberry32(seed);
    this.buildTrack(rand);
    this.placeLights(rand);
    this.placeCones(rand);
  }

  buildTrack(rand) {
    const N = 720, R0 = 170;
    const ph = [rand(), rand(), rand()].map((v) => v * Math.PI * 2);
    const amp = [0.12 + rand() * 0.08, 0.05 + rand() * 0.05, 0.02 + rand() * 0.02];
    const pts = [];
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2;
      const r = R0 * (1 + amp[0] * Math.sin(2 * a + ph[0]) + amp[1] * Math.sin(3 * a + ph[1]) + amp[2] * Math.sin(5 * a + ph[2]));
      pts.push({ x: r * Math.cos(a), y: r * Math.sin(a) });
    }
    const cum = new Float64Array(N);
    let s = 0;
    for (let i = 0; i < N; i++) {
      const p = pts[i], prev = pts[(i - 1 + N) % N], next = pts[(i + 1) % N];
      let tx = next.x - prev.x, ty = next.y - prev.y;
      const m = Math.hypot(tx, ty);
      tx /= m; ty /= m;
      Object.assign(p, { tx, ty, nx: -ty, ny: tx, h: Math.atan2(ty, tx) });
      cum[i] = s;
      s += Math.hypot(next.x - p.x, next.y - p.y);
    }
    this.pts = pts;
    this.cum = cum;
    this.length = s;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    this.bounds = { minX, minY, maxX, maxY };
  }

  placeLights(rand) {
    this.lights = [];
    const n = 4;
    for (let k = 0; k < n; k++) {
      const idx = this.indexAt(((k + 0.5) / n) * this.length + (rand() - 0.5) * 60);
      this.lights.push({ idx, s: this.cum[idx], t: rand() * CYCLE, state: 'green' });
    }
    this.update(0);
  }

  placeCones(rand) {
    this.cones = [];
    let s = 70;
    while (s < this.length - 50) {
      const nearLight = this.lights.some((l) => Math.abs(this.signedGap(s, l.s)) < 45);
      if (!nearLight) {
        const lane = rand() < 0.5 ? -1 : 1;
        const count = 1 + Math.floor(rand() * 3);
        for (let j = 0; j < count; j++) {
          const idx = this.indexAt(s + j * 5);
          const p = this.pts[idx];
          const off = lane * LANE_OFFSET + (rand() - 0.5) * 0.8;
          this.cones.push({ idx, s: this.cum[idx], off, x: p.x + p.nx * off, y: p.y + p.ny * off });
        }
      }
      s += 80 + rand() * 90;
    }
  }

  update(dt) {
    for (const l of this.lights) {
      l.t += dt;
      l.state = lightState(l.t);
    }
  }

  // Index of the track point at arc length s (wraps around the loop).
  indexAt(s) {
    s = ((s % this.length) + this.length) % this.length;
    let lo = 0, hi = this.cum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.cum[mid] <= s) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  // Forward distance along the loop from a to b, in [0, length).
  ahead(a, b) {
    return (((b - a) % this.length) + this.length) % this.length;
  }

  // Signed distance along the loop from a to b, in [-length/2, length/2).
  signedGap(a, b) {
    const d = this.ahead(a, b);
    return d >= this.length / 2 ? d - this.length : d;
  }

  // Project a point onto the track. Pass the previous idx as a hint for a cheap local search.
  nearest(x, y, hint = null) {
    const N = this.pts.length;
    let best = 0, bd = Infinity;
    const scan = (i) => {
      const p = this.pts[i];
      const d = (p.x - x) ** 2 + (p.y - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    };
    if (hint == null) for (let i = 0; i < N; i++) scan(i);
    else for (let k = -40; k <= 40; k++) scan((hint + k + N) % N);
    const p = this.pts[best];
    const off = (x - p.x) * p.nx + (y - p.y) * p.ny;
    const along = (x - p.x) * p.tx + (y - p.y) * p.ty;
    return { idx: best, off, s: (((this.cum[best] + along) % this.length) + this.length) % this.length };
  }

  draw(ctx, { sensor = false } = {}) {
    const pts = this.pts;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'butt';
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.setLineDash([]);
    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = ROAD_WIDTH + 0.8;
    ctx.stroke();
    ctx.strokeStyle = '#4a4d52';
    ctx.lineWidth = ROAD_WIDTH;
    ctx.stroke();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = '#f2f2f2';
    ctx.lineWidth = 0.35;
    ctx.stroke();
    ctx.setLineDash([]);

    for (const l of this.lights) {
      const p = pts[l.idx];
      const half = ROAD_WIDTH / 2;
      ctx.strokeStyle = LIGHT_COLORS[l.state];
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(p.x - p.nx * half, p.y - p.ny * half);
      ctx.lineTo(p.x + p.nx * half, p.y + p.ny * half);
      ctx.stroke();
      if (!sensor) {
        const px = p.x + p.nx * (half + 1.8), py = p.y + p.ny * (half + 1.8);
        ctx.fillStyle = '#111';
        ctx.beginPath(); ctx.arc(px, py, 1.1, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = LIGHT_COLORS[l.state];
        ctx.beginPath(); ctx.arc(px, py, 0.75, 0, Math.PI * 2); ctx.fill();
      }
    }

    for (const c of this.cones) {
      ctx.fillStyle = '#ff7a00';
      ctx.beginPath(); ctx.arc(c.x, c.y, 0.75, 0, Math.PI * 2); ctx.fill();
      if (!sensor) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 0.18;
        ctx.beginPath(); ctx.arc(c.x, c.y, 0.45, 0, Math.PI * 2); ctx.stroke();
      }
    }
  }
}
