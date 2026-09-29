// Fixed-time signal controller. Each intersection alternates NS and EW phases with yellow and
// an all-red clearance interval, offset randomly so the city isn't perfectly synchronized.
import { GRID } from './config.js';

const PHASES = [
  { ns: 'green', ew: 'red', dur: 14 },
  { ns: 'yellow', ew: 'red', dur: 3.5 },
  { ns: 'red', ew: 'red', dur: 1.5 },
  { ns: 'red', ew: 'green', dur: 14 },
  { ns: 'red', ew: 'yellow', dur: 3.5 },
  { ns: 'red', ew: 'red', dur: 1.5 },
];
const CYCLE = PHASES.reduce((a, p) => a + p.dur, 0);

export class Signals {
  constructor(rand) {
    this.t = 0;
    this.offsets = Array.from({ length: GRID * GRID }, () => rand() * CYCLE);
  }

  // Shift an intersection's cycle so `axis` turned green `into` seconds ago (scenarios).
  force([i, j], axis, into = 0) {
    let start = 0;
    for (const p of PHASES) {
      if (p[axis] === 'green') break;
      start += p.dur;
    }
    this.offsets[i * GRID + j] = (((start + into - this.t) % CYCLE) + CYCLE) % CYCLE;
  }

  update(dt) {
    this.t += dt;
  }

  state([i, j], axis) {
    let tt = (this.t + this.offsets[i * GRID + j]) % CYCLE;
    for (const p of PHASES) {
      if (tt < p.dur) return p[axis];
      tt -= p.dur;
    }
    return 'red';
  }
}
