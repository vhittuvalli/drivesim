// Route re-rooting, ZIP writer, and a soak with double-parked vans. Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, PITCH, LANE_W } from '../src/config.js';
import { Route } from '../src/planner.js';
import { crc32, zipStore } from '../src/zip.js';
import { soak } from './fuzz.mjs';

test('Route.fromPose starts on the lane the car is in', () => {
  const rand = mulberry32(1);
  // Eastbound between nodes (2,3) and (3,3): lane center is LANE_W/2 south of the road.
  const x = 2 * PITCH + 40, z = 3 * PITCH + LANE_W / 2 + 0.4;
  const { route } = Route.fromPose(rand, x, z, 0.1);
  assert.ok(route);
  assert.deepEqual(route.node, [3, 3]);
  assert.ok(Math.abs(route.pts[0].x - x) < 1e-6 && Math.abs(route.pts[0].z - (z - 0.4)) < 1e-6);
  assert.equal(Route.fromPose(rand, x, z - LANE_W, 0).route, null, 'oncoming lane is refused');
  assert.equal(Route.fromPose(rand, x, z, Math.PI / 2).route, null, 'crosswise heading is refused');
});

test('ZIP writer produces a valid stored archive', () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
  const zip = zipStore([{ name: 'a.txt', data: 'hello' }, { name: 'dir/b.bin', data: new Uint8Array([1, 2, 3]) }]);
  const dv = new DataView(zip.buffer);
  const end = zip.length - 22;
  assert.equal(dv.getUint32(end, true), 0x06054b50);
  assert.equal(dv.getUint16(end + 10, true), 2);
  let p = dv.getUint32(end + 16, true);
  const names = [];
  for (let k = 0; k < 2; k++) {
    assert.equal(dv.getUint32(p, true), 0x02014b50);
    const n = dv.getUint16(p + 28, true), off = dv.getUint32(p + 42, true);
    names.push(new TextDecoder().decode(zip.subarray(p + 46, p + 46 + n)));
    assert.equal(dv.getUint32(off, true), 0x04034b50);
    p += 46 + n;
  }
  assert.deepEqual(names, ['a.txt', 'dir/b.bin']);
});

for (const [seed, weather] of [[3, 'clear'], [8, 'rain']]) {
  test(`3 min with double-parked vans in ${weather}: no contacts, no collisions, cars overtake (seed ${seed})`, () => {
    const s = soak(seed, { minutes: 3, weather });
    assert.equal(s.contacts, 0, `ego contacts: ${JSON.stringify(s.lastContact)}`);
    assert.equal(s.npcCollisions, 0);
    assert.equal(s.redRuns, 0);
    assert.ok(s.npcOvertakes > 0, 'NPCs should pass the vans');
    assert.ok(s.dist > 300, `ego progress ${s.dist.toFixed(0)} m`);
  });
}
