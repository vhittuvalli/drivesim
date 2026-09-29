// Manual driving input: keyboard (WASD / arrows) or the first connected gamepad
// (left stick steers, right trigger throttle, left trigger brake).
import { clamp } from './config.js';

const LEFT = ['a', 'arrowleft'], RIGHT = ['d', 'arrowright'], UP = ['w', 'arrowup'], DOWN = ['s', 'arrowdown'];
export const DRIVE_KEYS = new Set([...LEFT, ...RIGHT, ...UP, ...DOWN]);

export class DriverInput {
  constructor(target = window) {
    this.keys = new Set();
    this.steer = 0;
    this.source = 'keyboard';
    target.addEventListener('keydown', (e) => {
      if (e.target.closest?.('input, select, textarea')) return;
      this.keys.add(e.key.toLowerCase());
    });
    target.addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    target.addEventListener('blur', () => this.keys.clear());
  }

  read(dt, v) {
    const any = (ks) => ks.some((k) => this.keys.has(k));
    let steer = (any(RIGHT) ? 1 : 0) - (any(LEFT) ? 1 : 0);
    let throttle = (any(UP) ? 1 : 0) - (any(DOWN) ? 1 : 0);
    let analog = false;
    const pad = [...(navigator.getGamepads?.() ?? [])].find(Boolean);
    if (pad) {
      const ax = pad.axes[0] ?? 0, rt = pad.buttons[7]?.value ?? 0, lt = pad.buttons[6]?.value ?? 0;
      if (Math.abs(ax) > 0.08) { steer = ax; analog = true; }
      if (rt > 0.02 || lt > 0.02) throttle = rt - lt;
      if (analog || rt > 0.02 || lt > 0.02) this.source = 'gamepad';
    }
    if (steer || throttle) this.source = analog ? 'gamepad' : this.source;
    // Keys are digital: ramp the wheel, and self-center faster than we turn in.
    if (analog) this.steer = steer;
    else {
      const rate = (steer === 0 ? 3.5 : 2.2) * dt;
      this.steer += clamp(steer - this.steer, -rate, rate);
    }
    // Less steering authority at speed, like a real rack ratio feels.
    return { steer: this.steer / (1 + v / 14), throttle };
  }
}
