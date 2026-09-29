// Headless multi-agent traffic tests (no rendering). Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/config.js';
import { BODY_TYPES } from '../src/bodytypes.js';
import { HALF_LEN } from '../src/planner.js';
import { Signals } from '../src/signals.js';
import { Traffic } from '../src/traffic.js';

// Minimal stand-in for the instanced renderer.
class FakeFleet {
  acquire(type) { return { type, idx: 0, spec: BODY_TYPES[type] }; }
  release() {}
  set() {}
}

function simulate(seed, cars, seconds) {
  const rand = mulberry32(seed);
  const signals = new Signals(rand);
  const traffic = new Traffic(new FakeFleet(), signals, rand);
  traffic.setCount(cars, null);
  const dt = 1 / 30;
  const stats = { collisions: 0, redRuns: 0, meanSpeed: 0, samples: 0 };
  const prevDist = new Map();
  const touching = new Set();
  for (let t = 0; t < seconds / dt; t++) {
    signals.update(dt);
    const agents = traffic.agents;
    traffic.step(dt, agents, null);
    for (const c of traffic.cars) {
      stats.meanSpeed += c.v;
      stats.samples++;
      for (const st of c.route.stops) {
        const d = st.s - c.s - HALF_LEN;
        const before = prevDist.get(st);
        if (before !== undefined && before > 0 && d <= 0 && signals.state(st.node, st.axis) === 'red') stats.redRuns++;
        prevDist.set(st, d);
      }
    }
    // Count new contacts between vehicle footprints (oriented boxes approximated by 2 circles each).
    const A = traffic.agents;
    for (let i = 0; i < A.length; i++) {
      for (let j = i + 1; j < A.length; j++) {
        const a = A[i], b = A[j];
        if (Math.abs(a.x - b.x) > 6 || Math.abs(a.z - b.z) > 6) continue;
        let hit = false;
        for (const sa of [-1.2, 1.2]) for (const sb of [-1.2, 1.2]) {
          const ax = a.x + Math.cos(a.h) * sa, az = a.z + Math.sin(a.h) * sa;
          const bx = b.x + Math.cos(b.h) * sb, bz = b.z + Math.sin(b.h) * sb;
          if ((ax - bx) ** 2 + (az - bz) ** 2 < 1.7 ** 2) hit = true;
        }
        const key = `${i}:${j}`;
        if (hit && !touching.has(key)) { stats.collisions++; touching.add(key); }
        if (!hit) touching.delete(key);
      }
    }
  }
  stats.meanSpeed /= stats.samples;
  stats.respawns = traffic.respawns;
  return stats;
}

for (const seed of [2, 11]) {
  test(`40 NPC cars for 5 minutes: no collisions, no red-light runs, traffic flows (seed ${seed})`, () => {
    const s = simulate(seed, 40, 300);
    assert.equal(s.collisions, 0, `collisions: ${s.collisions}`);
    assert.equal(s.redRuns, 0, `red runs: ${s.redRuns}`);
    assert.ok(s.meanSpeed > 3, `mean speed ${s.meanSpeed.toFixed(2)} m/s`);
    assert.ok(s.respawns <= 2, `deadlock respawns: ${s.respawns}`);
  });
}
