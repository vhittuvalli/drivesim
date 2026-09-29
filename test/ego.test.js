// End-to-end: the ego expert drives through NPC traffic and pedestrians. Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/config.js';
import { Route, Expert } from '../src/planner.js';
import { Signals } from '../src/signals.js';
import { Traffic } from '../src/traffic.js';
import { Crowd } from '../src/peds.js';
import { Car, FakeFleet, inFootprint } from './helpers.js';

for (const seed of [5, 17]) {
  test(`ego drives 5 min in traffic without contact (seed ${seed})`, () => {
    const rand = mulberry32(seed);
    const signals = new Signals(rand);
    const route = new Route(rand);
    const p0 = route.pts[0], p1 = route.pts[1];
    const car = new Car(p0.x, p0.z, Math.atan2(p1.z - p0.z, p1.x - p0.x));
    const expert = new Expert(route, signals);
    expert.track(car);
    const traffic = new Traffic(new FakeFleet(), signals, rand);
    traffic.setCount(40, expert.agent);
    const crowd = new Crowd(signals, rand);
    crowd.setCount(80);

    const dt = 1 / 60;
    let dist = 0, carHits = 0, pedHits = 0, yields = 0;
    const touching = new Set();
    for (let t = 0; t < 300 / dt; t++) {
      signals.update(dt);
      const vehicles = [expert.agent, ...traffic.agents];
      const agents = [...vehicles, ...crowd.agents];
      const c = expert.control(car, agents);
      if (c.reason === 'pedestrian' || c.reason === 'yield') yields++;
      car.step(dt, c.steer, c.throttle);
      traffic.step(dt, agents, expert.agent);
      crowd.step(dt, vehicles);
      dist += car.v * dt;

      const ego = { x: car.x, z: car.z, h: car.h };
      for (const a of traffic.agents) {
        if (Math.abs(a.x - ego.x) > 7 || Math.abs(a.z - ego.z) > 7) continue;
        // Sample the NPC footprint at front/center/rear against the ego box.
        const hit = [-1.6, 0, 1.6].some((k) => inFootprint(ego, a.x + Math.cos(a.h) * k, a.z + Math.sin(a.h) * k, 2.35 + 0.5, 0.92 + 0.9));
        const key = `c${a.id}`;
        if (hit && !touching.has(key)) { carHits++; touching.add(key); }
        if (!hit) touching.delete(key);
      }
      for (const p of crowd.peds) {
        const hit = inFootprint(ego, p.x, p.z, 2.35 + 0.25, 0.92 + 0.25);
        const key = `p${p.id}`;
        if (hit && !touching.has(key)) { pedHits++; touching.add(key); }
        if (!hit) touching.delete(key);
      }
    }
    assert.ok(dist > 800, `ego made progress: ${dist.toFixed(0)} m`);
    assert.equal(carHits, 0, `ego-vehicle contacts: ${carHits}`);
    assert.equal(pedHits, 0, `ego-pedestrian contacts: ${pedHits}`);
  });
}
