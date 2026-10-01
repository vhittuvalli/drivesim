// A polyline with arc length that vehicles follow: the part of a route shared by the street-grid
// Route (planner.js) and the HighwayRoute (highway.js). Subclasses generate points lazily with
// ensure(s) and may register stop lines, turns and approach legs (a highway has none).
import { ROAD_W, clamp } from './config.js';

export class Path {
  constructor() {
    this.pts = []; // {x, z, s}
    this.stops = []; // {s, node, axis, d, turn, decision}
    this.turns = []; // {s, kind}
    this.legs = []; // straight approaches: {s0, s1, node, d, stop}
  }

  get length() {
    return this.pts[this.pts.length - 1].s;
  }

  // Appends a point; returns false if it was too close to the last one to add.
  push(x, z) {
    const last = this.pts[this.pts.length - 1];
    const s = last ? last.s + Math.hypot(x - last.x, z - last.z) : 0;
    if (last && s - last.s < 1e-3) return false;
    this.pts.push({ x, z, s });
    return true;
  }

  line(x1, z1, step = 1) {
    const last = this.pts[this.pts.length - 1];
    const n = Math.max(1, Math.ceil(Math.hypot(x1 - last.x, z1 - last.z) / step));
    for (let k = 1; k <= n; k++) this.push(last.x + ((x1 - last.x) * k) / n, last.z + ((z1 - last.z) * k) / n);
  }

  ensure(s) {
    while (this.length < s) this.extend();
  }

  // Drop geometry more than `keep` meters behind s. Returns how many points were removed
  // so callers can shift their index hints.
  trim(s, keep = 60) {
    let n = 0;
    while (n < this.pts.length - 2 && this.pts[n + 1].s < s - keep) n++;
    if (n > 0) this.pts.splice(0, n);
    const cut = s - keep;
    this.stops = this.stops.filter((st) => st.s > cut);
    this.turns = this.turns.filter((t) => t.s > cut);
    this.legs = this.legs.filter((l) => l.s1 + ROAD_W > cut);
    return n;
  }

  // The approach leg the vehicle is on (including the intersection box after it).
  legAt(s) {
    for (const l of this.legs) if (s < l.s1 + ROAD_W) return l;
    return this.legs[this.legs.length - 1];
  }

  // Point at arc length s (linear interpolation), searching from a hint index.
  at(s, hint = 0) {
    let k = Math.max(0, Math.min(hint, this.pts.length - 2));
    while (k > 0 && this.pts[k].s > s) k--;
    while (k < this.pts.length - 2 && this.pts[k + 1].s < s) k++;
    const a = this.pts[k], b = this.pts[k + 1];
    const u = clamp((s - a.s) / (b.s - a.s || 1), 0, 1);
    return { x: a.x + (b.x - a.x) * u, z: a.z + (b.z - a.z) * u, k, h: Math.atan2(b.z - a.z, b.x - a.x) };
  }
}
