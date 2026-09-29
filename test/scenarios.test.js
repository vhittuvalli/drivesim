// Every scripted scenario, driven by the expert in background traffic. Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/config.js';
import { World } from '../src/sim.js';
import { SCENARIOS } from '../src/scenarios.js';
import { Car, FakeFleet } from './helpers.js';

function play(id, seed) {
  const w = new World({ rand: mulberry32(seed), car: new Car(0, 0, 0), fleet: new FakeFleet(), cars: 30, peds: 50 });
  for (let k = 0; k < 120; k++) w.step(1 / 60);
  const run = w.startScenario(id);
  while (run.status === 'running') w.step(1 / 60);
  return run;
}

for (const id of Object.keys(SCENARIOS)) {
  test(`scenario ${id} passes (seeds 1-3)`, () => {
    for (const seed of [1, 2, 3]) {
      const run = play(id, seed);
      assert.equal(run.status, 'passed', `seed ${seed}: ${run.message} after ${run.t.toFixed(1)} s`);
    }
  });
}

test('the expert brakes for the red-light runner instead of getting lucky', () => {
  const w = new World({ rand: mulberry32(4), car: new Car(0, 0, 0), fleet: new FakeFleet(), cars: 0, peds: 0 });
  const run = w.startScenario('red-runner');
  let sawCrossing = false;
  while (run.status === 'running') {
    w.step(1 / 60);
    if (w.ctrl.reason === 'crossing') sawCrossing = true;
  }
  assert.equal(run.status, 'passed', run.message);
  assert.ok(sawCrossing, 'expected a crossing-vehicle yield');
});

test('overtaking waits for oncoming traffic', () => {
  const run = play('overtake-oncoming', 5);
  assert.equal(run.status, 'passed', run.message);
  assert.ok(run.flags.waited, 'should have waited for the oncoming cars');
});
