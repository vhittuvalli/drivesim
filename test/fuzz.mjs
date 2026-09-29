// Seed sweep: drive the ego through traffic, pedestrians and double-parked vans for each seed
// and report any contact, NPC-NPC collision or red-light run, with the seed to reproduce it.
//   npm run fuzz -- [seeds=20] [minutes=3] [weather=clear]
import { mulberry32 } from '../src/config.js';
import { HALF_LEN } from '../src/planner.js';
import { World, WEATHER, setWeather } from '../src/sim.js';
import { Car, FakeFleet } from './helpers.js';

export function soak(seed, { minutes = 3, cars = 40, peds = 70, doubleParked = 6, weather = 'clear' } = {}) {
  setWeather(weather);
  const w = new World({ rand: mulberry32(seed), car: new Car(0, 0, 0), fleet: new FakeFleet(), cars, peds, doubleParked });
  const dt = 1 / 60;
  const stats = { seed, dist: 0, contacts: 0, npcCollisions: 0, redRuns: 0, egoOvertakes: 0, npcOvertakes: 0 };
  const touching = new Set(), prevStop = new Map(), overtaking = new Set();
  let egoWas = false;
  for (let t = 0; t < (minutes * 60) / dt; t++) {
    w.step(dt);
    stats.dist += w.car.v * dt;
    const egoNow = !!w.expert.ot.active;
    if (egoNow && !egoWas) stats.egoOvertakes++;
    egoWas = egoNow;
    for (const c of w.traffic.cars) {
      if (c.ot.active && !overtaking.has(c)) stats.npcOvertakes++;
      if (c.ot.active) overtaking.add(c);
      else overtaking.delete(c);
    }
    // Ego red-light runs.
    for (const st of w.expert.route.stops) {
      const d = st.s - w.expert.s - HALF_LEN, before = prevStop.get(st);
      if (before !== undefined && before > 0 && d <= 0 && w.signals.state(st.node, st.axis) === 'red') stats.redRuns++;
      prevStop.set(st, d);
    }
    // NPC-NPC contacts (two circles per car).
    const A = w.traffic.agents;
    for (let i = 0; i < A.length; i++) {
      for (let j = i + 1; j < A.length; j++) {
        const a = A[i], b = A[j];
        if (Math.abs(a.x - b.x) > 7 || Math.abs(a.z - b.z) > 7) continue;
        let hit = false;
        for (const sa of [-1.2, 1.2]) for (const sb of [-1.2, 1.2]) {
          const ax = a.x + Math.cos(a.h) * sa, az = a.z + Math.sin(a.h) * sa;
          const bx = b.x + Math.cos(b.h) * sb, bz = b.z + Math.sin(b.h) * sb;
          if ((ax - bx) ** 2 + (az - bz) ** 2 < 1.7 ** 2) hit = true;
        }
        const key = `${a.id}:${b.id}`;
        if (hit && !touching.has(key)) { stats.npcCollisions++; touching.add(key); }
        if (!hit) touching.delete(key);
      }
    }
  }
  stats.contacts = w.contacts;
  stats.lastContact = w.lastContact;
  setWeather('clear');
  return stats;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [n = 20, minutes = 3, weather = 'clear'] = process.argv.slice(2);
  let bad = 0;
  for (let seed = 1; seed <= +n; seed++) {
    const s = soak(seed, { minutes: +minutes, weather });
    // Slippery roads mean longer headways and slower corners: expect less progress.
    const fail = s.contacts || s.npcCollisions || s.redRuns || s.dist < 150 * +minutes * WEATHER[weather].grip;
    if (fail) bad++;
    console.log(`${fail ? 'FAIL' : 'ok  '} seed ${String(seed).padStart(3)}  ${s.dist.toFixed(0).padStart(5)} m  contacts ${s.contacts}${s.lastContact ? ` (${s.lastContact.what} at ${s.lastContact.t.toFixed(1)} s)` : ''}  npc collisions ${s.npcCollisions}  red runs ${s.redRuns}  overtakes ego ${s.egoOvertakes} / npc ${s.npcOvertakes}`);
  }
  console.log(`${bad} of ${n} seeds failed`);
  process.exit(bad ? 1 : 0);
}
