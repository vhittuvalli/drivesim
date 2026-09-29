// Headless pedestrian + traffic interaction tests. Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, ROAD_W, GRID, PITCH } from '../src/config.js';
import { BODY_TYPES } from '../src/bodytypes.js';
import { Signals } from '../src/signals.js';
import { Traffic } from '../src/traffic.js';
import { Crowd } from '../src/peds.js';

class FakeFleet {
  acquire(type) { return { type, idx: 0, spec: BODY_TYPES[type] }; }
  release() {}
  set() {}
}

// Distance from (x, z) to the nearest road centerline, and whether it's inside an intersection box.
function roadInfo(x, z) {
  const fx = ((x % PITCH) + PITCH) % PITCH, fz = ((z % PITCH) + PITCH) % PITCH;
  const dx = Math.min(fx, PITCH - fx), dz = Math.min(fz, PITCH - fz);
  return { onRoad: dx < ROAD_W / 2 || dz < ROAD_W / 2, inBox: dx < ROAD_W / 2 && dz < ROAD_W / 2 };
}

test('pedestrians stay on sidewalks except when crossing', () => {
  const rand = mulberry32(4);
  const signals = new Signals(rand);
  const crowd = new Crowd(signals, rand);
  crowd.setCount(60);
  let offRoute = 0, crossings = 0;
  const lo = -ROAD_W, hi = (GRID - 1) * PITCH + ROAD_W;
  for (let t = 0; t < 180 * 30; t++) {
    signals.update(1 / 30);
    crowd.step(1 / 30, []);
    for (const p of crowd.peds) {
      if (p.x < lo - 10 || p.x > hi + 10 || p.z < lo - 10 || p.z > hi + 10) offRoute++;
      if (roadInfo(p.x, p.z).onRoad) {
        if (p.agent.crossing) crossings++;
        else offRoute++;
      }
    }
  }
  assert.equal(offRoute, 0);
  assert.ok(crossings > 0, 'someone should cross a street');
});

test('pedestrians cross at crosswalks only on the parallel green', () => {
  const rand = mulberry32(9);
  const signals = new Signals(rand);
  const crowd = new Crowd(signals, rand);
  crowd.setCount(80);
  let started = 0, startedOnRed = 0;
  const prevMode = new Map();
  for (let t = 0; t < 240 * 30; t++) {
    signals.update(1 / 30);
    const before = crowd.peds.map((p) => [p, p.queue[0]]);
    crowd.step(1 / 30, []);
    for (const [p, wp] of before) {
      if (wp?.mode === 'wait' && p.queue[0] !== wp) {
        started++;
        if (signals.state(wp.node, wp.axis) !== 'green') startedOnRed++;
      }
      prevMode.set(p, p.queue[0]?.mode);
    }
  }
  assert.ok(started > 10, `crossings started: ${started}`);
  assert.equal(startedOnRed, 0);
});

test('vehicles never hit pedestrians', () => {
  const rand = mulberry32(21);
  const signals = new Signals(rand);
  const traffic = new Traffic(new FakeFleet(), signals, rand);
  const crowd = new Crowd(signals, rand);
  traffic.setCount(40, null);
  crowd.setCount(80);
  const dt = 1 / 30;
  const touching = new Set();
  let hits = 0;
  for (let t = 0; t < 300 / dt; t++) {
    signals.update(dt);
    const cars = traffic.agents;
    const agents = [...cars, ...crowd.agents];
    traffic.step(dt, agents, null);
    crowd.step(dt, cars);
    for (const c of traffic.cars) {
      const a = c.agent;
      for (const p of crowd.peds) {
        const dx = p.x - a.x, dz = p.z - a.z;
        if (Math.abs(dx) > 4 || Math.abs(dz) > 4) continue;
        // Pedestrian inside the car footprint (plus 0.25 m) in the car's frame.
        const lx = dx * Math.cos(a.h) + dz * Math.sin(a.h), ly = -dx * Math.sin(a.h) + dz * Math.cos(a.h);
        const hit = Math.abs(lx) < a.halfLen + 0.25 && Math.abs(ly) < 1.15;
        const k = `${a.id}:${p.id}`;
        if (hit && !touching.has(k)) { hits++; touching.add(k); }
        if (!hit) touching.delete(k);
      }
    }
  }
  assert.equal(hits, 0, `vehicle-pedestrian contacts: ${hits}`);
});
