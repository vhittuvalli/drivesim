// Training labels and the safety driver (the parts of the learned-driver stack that don't need
// a browser). Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, clamp } from '../src/config.js';
import { World } from '../src/sim.js';
import { makeLabels, commandOf, WP_DIST } from '../src/labels.js';
import { Car, FakeFleet } from './helpers.js';

const world = (seed, opts = {}) => new World({ rand: mulberry32(seed), car: new Car(0, 0, 0), fleet: new FakeFleet(), cars: 20, peds: 30, ...opts });

test('labels: waypoints ahead of the car, all branches supervised near intersections', () => {
  const w = world(3);
  let near = 0, far = 0, turnsDiffer = 0;
  for (let t = 0; t < 120 * 60; t++) {
    w.step(1 / 60);
    if (t % 30) continue;
    const L = makeLabels(w, w.expertCtrl);
    const taken = L.wp[L.command];
    assert.ok(taken, 'the taken branch always has a label');
    assert.equal(taken.length, WP_DIST.length);
    // Close to straight ahead at the first waypoint; lateral error is the expert's tracking error.
    assert.ok(taken[0][0] > 0.5 && taken[0][0] < 3 && Math.abs(taken[0][1]) < 1.5, JSON.stringify(taken[0]));
    assert.ok(L.vTarget >= 0 && L.vTarget <= 14);
    if (L.cmdDist !== null && L.cmdDist > 30) {
      far++;
      assert.deepEqual(L.wp.left ?? taken, taken);
    } else if (L.cmdDist !== null && L.cmdDist > 2 && L.cmdDist < 25 && L.wp.left && L.wp.right) {
      near++;
      // Left and right turn paths end on opposite sides.
      if (L.wp.left[7][1] > 3 && L.wp.right[7][1] < -3) turnsDiffer++;
    }
  }
  assert.ok(far > 20 && near > 3, `far ${far}, near ${near}`);
  assert.ok(turnsDiffer >= near * 0.8, `turn branches diverge: ${turnsDiffer}/${near}`);
});

test('labels do not disturb the simulation', () => {
  const a = world(9), b = world(9);
  for (let t = 0; t < 30 * 60; t++) {
    a.step(1 / 60);
    b.step(1 / 60);
    if (t % 6 === 0) makeLabels(a, a.expertCtrl);
  }
  assert.equal(a.car.x, b.car.x);
  assert.equal(a.car.z, b.car.z);
});

test('command is the next turn, held until the car leaves the intersection', () => {
  const w = world(5, { cars: 0, peds: 0 });
  const seen = new Set();
  for (let t = 0; t < 90 * 60; t++) {
    w.step(1 / 60);
    const c = commandOf(w.expert.route, w.expert.s);
    seen.add(c.kind);
    assert.ok(c.dist > -25, `command dist ${c.dist}`);
  }
  assert.ok(seen.size >= 2);
});

test('safety driver: a perfect policy is never disengaged', () => {
  const w = world(2);
  w.setPolicy({ control: (_, exp) => ({ steer: exp.steer, throttle: exp.throttle }) });
  for (let t = 0; t < 120 * 60; t++) w.step(1 / 60);
  assert.equal(w.safety.disengagements, 0);
  assert.ok(w.safety.autoDist > 300);
});

test('safety driver: takes over from a drifting policy and a policy that never brakes', () => {
  for (const [name, policy] of [
    ['drift', (_, exp) => ({ steer: clamp(exp.steer + 0.25, -1, 1), throttle: exp.throttle })],
    ['no brakes', (_, exp) => ({ steer: exp.steer, throttle: Math.max(exp.throttle, 0.3) })],
  ]) {
    const w = world(4);
    w.setPolicy({ control: policy });
    for (let t = 0; t < 120 * 60; t++) w.step(1 / 60);
    assert.ok(w.safety.disengagements > 2, `${name}: ${w.safety.disengagements} takeovers`);
    assert.equal(w.contacts, 0, `${name}: the safety driver prevents contact (${JSON.stringify(w.lastContact)})`);
  }
});
