// Training labels and the safety driver (the parts of the learned-driver stack that don't need
// a browser). Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, clamp } from '../src/config.js';
import { World } from '../src/sim.js';
import { makeLabels, commandOf, WP_DIST, COMMANDS, MAX_TARGET_SPEED } from '../src/labels.js';
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
    assert.ok(L.vTarget >= 0 && L.vTarget <= MAX_TARGET_SPEED);
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

test('neural controller: pure pursuit on a predicted path', async () => {
  const { followPath } = await import('../src/neural.js');
  const straight = WP_DIST.map((d) => [d, 0]);
  const left = WP_DIST.map((d) => [d, 0.02 * d * d]);
  assert.ok(Math.abs(followPath(straight, 8, 8).steer) < 1e-9);
  assert.ok(followPath(left, 8, 8).steer < -0.05, 'a path bending left steers left (negative)');
  assert.ok(followPath(straight, 5, 9).throttle > 0.5);
  assert.ok(followPath(straight, 10, 4).throttle < -0.3);
  assert.equal(followPath(straight, 0.2, 0.1).throttle, -0.5, 'holds the brake when stopped');
});

// An oracle "network" that predicts exactly the expert's labels, observed at 10 Hz and held in
// world coordinates in between: the waypoint representation plus the controller must drive
// (almost) as well as the expert, or no trained network could.
test('neural driver: perfect waypoint predictions drive with (almost) no takeovers', async () => {
  const { NeuralDriver } = await import('../src/neural.js');
  const takeovers = [];
  for (const seed of [2, 6, 22]) {
    const w = world(seed);
    const nn = new NeuralDriver({ hz: 10 });
    w.setPolicy(nn);
    for (let t = 0; t < 150 * 60; t++) {
      // Observe on the neural driver's schedule (its `due` needs a loaded network).
      if (w.t >= nn.nextObs && w.expertCtrl) {
        const L = makeLabels(w, w.expertCtrl);
        nn.setPrediction(w.car, L.command, COMMANDS.map((k) => L.wp[k] ?? L.wp[L.command]), COMMANDS.map(() => L.vTarget));
        nn.nextObs = w.t + nn.period;
      }
      w.step(1 / 60);
    }
    assert.equal(w.contacts, 0, `seed ${seed}: contact with ${w.lastContact?.what}`);
    assert.ok(w.safety.autoDist > 500, `seed ${seed}: drove ${w.safety.autoDist.toFixed(0)} m`);
    takeovers.push(...w.safety.events.map((e) => `seed ${seed}: ${e.reason}`));
  }
  assert.ok(takeovers.length <= 1, `takeovers over ~1.9 km: ${JSON.stringify(takeovers)}`);
});

test('safety driver: takes over from a policy that will not pull away, and the car then moves', () => {
  const w = world(6);
  w.setPolicy({ control: (_, exp) => ({ steer: exp.steer, throttle: -0.5 }) });
  for (let t = 0; t < 60 * 60; t++) w.step(1 / 60);
  const stalls = w.safety.events.filter((e) => e.reason === 'did not pull away').length;
  assert.ok(stalls >= 3, `stall takeovers: ${stalls}`);
  assert.ok(w.safety.totalDist > 40, `the expert drove during takeovers: ${w.safety.totalDist.toFixed(0)} m`);
  assert.equal(w.contacts, 0);
});

test('labels: the light ahead is labeled red, yellow or green near a stop line, none elsewhere', async () => {
  const { lightOf } = await import('../src/labels.js');
  const w = world(3);
  const seen = new Set();
  for (let t = 0; t < 120 * 60; t++) {
    w.step(1 / 60);
    if (t % 15) continue;
    const L = makeLabels(w, w.expertCtrl), st = w.expertCtrl.signal;
    assert.equal(L.light, lightOf(w.expertCtrl));
    if (L.light !== 'none') assert.ok(st && st.dist < 90 && L.light === st.state);
    if (w.expertCtrl.reason === 'signal') assert.ok(['red', 'yellow'].includes(L.light), `stopping for ${L.light}`);
    seen.add(L.light);
  }
  assert.ok(seen.has('red') && seen.has('green') && seen.has('none'), [...seen].join());
});

test('light-aware speed: brakes to stop at the predicted line for red, not for green or far away', async () => {
  const { lightCap, NeuralDriver } = await import('../src/neural.js');
  // 10 m/s, red light, line 20 m ahead: brake at about v^2 / 2(d - 1).
  const a = lightCap(10, 20, 0.95, 0.02);
  assert.ok(a < -2 && a > -3, `decel ${a}`);
  assert.equal(lightCap(10, 20, 0.05, 0.02), null, 'green: no cap');
  assert.equal(lightCap(10, 80, 0.95, 0.02), null, 'beyond range: no cap');
  assert.ok(lightCap(10, 45, 0.95, 0.02) === 0 || lightCap(10, 45, 0.95, 0.02) < 0, 'far: at least no speeding up');
  assert.equal(lightCap(15, 8, 0.02, 0.95), null, 'amber too close to stop for: go through');
  assert.ok(lightCap(8, 30, 0.02, 0.95) <= 0, 'amber with room: stop');
  // Through the driver: a confident red with the line 15 m ahead brakes even though the
  // network's own target speed says keep going; a stopped car holds the brake at the line.
  const nn = new NeuralDriver();
  const pose = { x: 0, z: 0, h: 0 }, straight = [2, 4, 6, 8, 11, 14, 18, 23].map((d) => [d, 0]);
  const red = { none: 0.02, red: 0.95, yellow: 0.01, green: 0.02 };
  for (let i = 0; i < 3; i++) nn.setPrediction(pose, 'straight', [straight, straight, straight], [10, 10, 10], { light: red, stopDist: 15 });
  const moving = nn.control({ car: { x: 0, z: 0, h: 0, v: 10 } });
  assert.ok(moving.throttle < -0.2 && moving.lightStop, JSON.stringify(moving));
  const stopped = nn.control({ car: { x: 13.5, z: 0, h: 0, v: 0.2 } });
  assert.equal(stopped.throttle, -0.5, 'holds the brake at the line');
  nn.lightAware = false;
  assert.ok(nn.control({ car: { x: 0, z: 0, h: 0, v: 10 } }).throttle > 0, 'without the cap it would drive on');
});
