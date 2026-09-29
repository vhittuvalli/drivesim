// Headless tests for routing and driving logic. Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, clamp, WHEELBASE, MAX_STEER, MAX_SPEED } from '../src/config.js';
import { Route, Expert, idm, IDM, HALF_LEN } from '../src/planner.js';
import { Signals } from '../src/signals.js';

// Same dynamics as src/vehicle.js without the three.js mesh.
class Car {
  constructor(x, z, h) { Object.assign(this, { x, z, h, v: 0, steer: 0 }); }
  step(dt, s, t) {
    const tg = clamp(s, -1, 1) * MAX_STEER;
    this.steer += clamp(tg - this.steer, -1.6 * dt, 1.6 * dt);
    const a = t >= 0 ? 3.2 * t : 7.5 * t;
    this.v = clamp(this.v + (a - 0.0025 * this.v * this.v - (this.v > 0 ? 0.08 : 0)) * dt, 0, MAX_SPEED);
    this.h += (this.v / WHEELBASE) * Math.tan(this.steer) * dt;
    this.x += this.v * Math.cos(this.h) * dt;
    this.z += this.v * Math.sin(this.h) * dt;
  }
}

function drive(seed, seconds) {
  const rand = mulberry32(seed);
  const route = new Route(rand);
  const signals = new Signals(rand);
  const expert = new Expert(route, signals);
  let turns = 0;
  const extend = route.extend.bind(route);
  route.extend = () => { turns++; extend(); };
  const p0 = route.pts[0], p1 = route.pts[1];
  const car = new Car(p0.x, p0.z, Math.atan2(p1.z - p0.z, p1.x - p0.x));
  const dt = 1 / 60;
  const stats = { dist: 0, maxLat: 0, redRuns: 0, turns: 0, stops: 0 };
  const prev = new Map();
  let stopped = false;
  for (let t = 0; t < seconds * 60; t++) {
    signals.update(dt);
    const c = expert.control(car);
    car.step(dt, c.steer, c.throttle);
    stats.dist += car.v * dt;
    stats.maxLat = Math.max(stats.maxLat, Math.abs(expert.lateral));
    if (car.v < 0.1 && !stopped) { stats.stops++; stopped = true; }
    if (car.v > 2) stopped = false;
    for (const st of route.stops) {
      const d = st.s - expert.s - HALF_LEN;
      const before = prev.get(st);
      if (before !== undefined && before > 0 && d <= 0 && signals.state(st.node, st.axis) === 'red') stats.redRuns++;
      prev.set(st, d);
    }
  }
  stats.turns = turns;
  return stats;
}

test('IDM: free road accelerates, standstill gap brakes, equilibrium near s0', () => {
  assert.ok(idm(0, 11, Infinity) > 1);
  assert.ok(idm(10, 11, 5, 0) < -2);
  assert.ok(Math.abs(idm(0, 11, IDM.s0, 0)) < 1e-9 + IDM.a);
  assert.ok(idm(0, 11, IDM.s0 * 0.5, 0) < 0);
});

test('route extends through turns and trims without losing position', () => {
  const route = new Route(mulberry32(3));
  route.ensure(2000);
  assert.ok(route.turns.length > 10);
  assert.ok(route.stops.every((s) => s.turn !== undefined));
  const before = route.at(1500);
  const removed = route.trim(1500);
  assert.ok(removed > 0);
  const after = route.at(1500);
  assert.ok(Math.hypot(before.x - after.x, before.z - after.z) < 1e-6);
});

for (const seed of [1, 7, 99]) {
  test(`expert drives 10 min without running reds or leaving the lane (seed ${seed})`, () => {
    const s = drive(seed, 600);
    assert.ok(s.dist > 2000, `drove ${s.dist.toFixed(0)} m`);
    assert.ok(s.turns > 15);
    assert.ok(s.stops > 5, 'should stop at some lights');
    assert.equal(s.redRuns, 0);
    assert.ok(s.maxLat < 1.0, `max lateral error ${s.maxLat.toFixed(2)} m`);
  });
}
