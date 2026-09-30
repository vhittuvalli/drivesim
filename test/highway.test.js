// The highway loop: geometry, the expert and NPC traffic at highway speed with lane changes,
// highway training labels, and handing back from manual driving. Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, clamp } from '../src/config.js';
import { World } from '../src/sim.js';
import { HW, HW_LENGTH, centerAt, lanePoint, laneOffset, highwayPose } from '../src/highway.js';
import { makeLabels, commandOf } from '../src/labels.js';
import { Car, FakeFleet, inFootprint } from './helpers.js';

const world = (seed, highway = 60) => {
  const w = new World({ rand: mulberry32(seed), car: new Car(0, 0, 0), fleet: new FakeFleet(), cars: 10, peds: 10, highway });
  w.setRoad('highway');
  return w;
};

// A slow truck 60 m ahead in the ego's lane, so there's someone to pass.
const slowTruck = (w) => {
  const { dir, q } = w.ego.hw;
  w.traffic.clearNear(w.highwayAreas(-30, 150), w.ego);
  return w.spawnHighwayCar({ dir, lane: w.expert.route.lane, q: q + 60 }, { type: 'van', desired: 20, v: 20, laneChanges: true });
};

test('highway geometry: the loop closes and poses project back onto their lane', () => {
  const a = centerAt(0), b = centerAt(HW_LENGTH - 1e-6);
  assert.ok(Math.hypot(a.x - b.x, a.z - b.z) < 0.01, 'closed loop');
  for (let u = 0; u < HW_LENGTH; u += 97) {
    for (const dir of [1, -1]) {
      const lane = u % 3, q = dir > 0 ? u : HW_LENGTH - u, p = lanePoint(dir, q, laneOffset(lane));
      const r = highwayPose(p.x, p.z, p.h);
      assert.equal(r.dir, dir);
      assert.equal(r.lane, lane);
      assert.ok(Math.abs(r.lat - laneOffset(lane)) < 0.01 && r.headingErr < 1e-6, JSON.stringify(r));
    }
  }
});

test('highway: the expert cruises near the limit, changes lanes, and nobody collides (3 seeds x 90 s)', () => {
  let changes = 0;
  for (const seed of [1, 2, 3]) {
    const w = world(seed, 80);
    slowTruck(w);
    let lane = w.expert.route.lane, vSum = 0, maxLat = 0, overlaps = 0;
    const T = 90 * 60;
    for (let t = 0; t < T; t++) {
      w.step(1 / 60);
      vSum += w.car.v;
      maxLat = Math.max(maxLat, Math.abs(w.expert.lateral));
      if (w.expert.route.lane !== lane) (changes++), (lane = w.expert.route.lane);
      if (t % 20) continue;
      const cars = w.traffic.highway.map((c) => c.agent);
      for (let i = 0; i < cars.length; i++) {
        for (let j = i + 1; j < cars.length; j++) {
          const p = cars[i], q = cars[j];
          if (Math.abs(p.x - q.x) > 8 || Math.abs(p.z - q.z) > 8) continue;
          const e = q.halfLen - 0.5;
          if ([-e, 0, e].some((k) => inFootprint(p, q.x + Math.cos(q.h) * k, q.z + Math.sin(q.h) * k, p.halfLen, 0.9))) overlaps++;
        }
      }
    }
    assert.equal(w.contacts, 0, `seed ${seed}: contact with ${w.lastContact?.what}`);
    assert.equal(overlaps, 0, `seed ${seed}: NPC cars overlapped`);
    assert.equal(w.road, 'highway');
    assert.ok(vSum / T > 24, `seed ${seed}: mean speed ${(vSum / T).toFixed(1)} m/s`);
    assert.ok(maxLat < 0.6, `seed ${seed}: lateral error ${maxLat.toFixed(2)} m`);
  }
  assert.ok(changes >= 3, `lane changes: ${changes}`);
});

test('highway labels: commands are lane changes; left/right branches move over one lane', () => {
  const w = world(4, 80);
  slowTruck(w);
  let straight = 0, changing = 0, spread = 0;
  for (let t = 0; t < 90 * 60; t++) {
    w.step(1 / 60);
    if (t % 30) continue;
    const L = makeLabels(w, w.expertCtrl), r = w.expert.route;
    assert.equal(L.road, 'highway');
    assert.equal(L.command, commandOf(r, w.expert.s).kind);
    assert.ok(L.vTarget <= 34);
    if (L.command === 'straight') {
      straight++;
      assert.equal(L.wp.left === null, L.lane === 0, 'no left branch from the left lane');
      assert.equal(L.wp.right === null, L.lane === HW.lanes - 1, 'no right branch from the right lane');
      const other = L.wp.left ?? L.wp.right;
      // Lane-change branches bend away from the lane, left positive.
      if (L.wp.left) assert.ok(L.wp.left[7][1] > L.wp.straight[7][1]);
      if (L.wp.right) assert.ok(L.wp.right[7][1] < L.wp.straight[7][1]);
      if (other) spread = Math.max(spread, Math.abs(other[7][1] - L.wp.straight[7][1]));
    } else {
      changing++;
      assert.ok(L.wp[L.command] && !L.wp.straight);
    }
  }
  assert.ok(straight > 60 && changing > 0, `straight ${straight}, changing ${changing}`);
  assert.ok(spread > 0.2, `lane-change branches differ from the lane by ${spread.toFixed(2)} m`);
});

test('highway: hand back from manual driving, merging from the shoulder', () => {
  const w = world(5, 0);
  const lane = w.expert.route.lane, steer = { v: 0.07 };
  w.takeOver({ read: () => ({ steer: steer.v, throttle: 0.2 }) });
  // An S-swerve that leaves the car parallel to the lane but about half a lane to the right.
  for (let t = 0; t < 36; t++) w.step(1 / 60);
  steer.v = -0.07;
  for (let t = 0; t < 36; t++) w.step(1 / 60);
  steer.v = 0;
  for (let t = 0; t < 30; t++) w.step(1 / 60);
  const p = highwayPose(w.car.x, w.car.z, w.car.h);
  assert.ok(p.lat - laneOffset(lane) > 1, `moved right: ${(p.lat - laneOffset(lane)).toFixed(2)} m`);
  assert.ok(p.dist < HW.width, 'still on the road');
  const r = w.handBack();
  assert.ok(r.ok, r.reason);
  for (let t = 0; t < 20 * 60; t++) w.step(1 / 60);
  assert.ok(Math.abs(w.expert.lateral) < 0.3, `back in lane: ${w.expert.lateral.toFixed(2)} m`);
  assert.equal(w.contacts, 0);
});

test('highway safety driver: a perfect policy is never disengaged, a drifting one is', () => {
  for (const [name, policy, expectTakeovers] of [
    ['perfect', (_, exp) => ({ steer: exp.steer, throttle: exp.throttle }), false],
    ['drift', (_, exp) => ({ steer: clamp(exp.steer + 0.06, -1, 1), throttle: exp.throttle }), true],
  ]) {
    const w = world(6, 60);
    w.setPolicy({ control: policy });
    for (let t = 0; t < 60 * 60; t++) w.step(1 / 60);
    if (expectTakeovers) assert.ok(w.safety.disengagements > 2, `${name}: ${w.safety.disengagements}`);
    else assert.equal(w.safety.disengagements, 0, `${name}: ${JSON.stringify(w.safety.events[0])}`);
    assert.equal(w.contacts, 0, name);
  }
});
