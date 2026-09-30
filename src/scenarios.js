// Scripted driving scenarios. Each one places the ego and a few scripted actors on a straight
// stretch of the grid (or on the highway: `road: 'highway'`), triggers events as the ego
// approaches, and judges the outcome: any contact fails, `check` returning a message passes, and
// running past `timeout` fails.
//
// A site is a start node A = (i, j) and direction d. N1 = A + d is the next intersection
// (always interior, so cross streets exist on both sides) and N2 = A + 2d the one after.
import { GRID, PITCH, LANE_W, nodePos } from './config.js';
import { HALF_LEN } from './planner.js';
import { HW, HW_LENGTH, loopDist } from './highway.js';

const axisOf = (d) => (d[0] ? 'ew' : 'ns');
const turns = (...kinds) => () => kinds.shift() ?? null;
const straight = (n = 3) => turns(...Array(n).fill('straight'));

function pickSite(rand) {
  for (;;) {
    const d = [[1, 0], [-1, 0], [0, 1], [0, -1]][Math.floor(rand() * 4)];
    const i1 = 1 + Math.floor(rand() * (GRID - 2)), j1 = 1 + Math.floor(rand() * (GRID - 2));
    const i = i1 - d[0], j = j1 - d[1], i2 = i1 + d[0], j2 = j1 + d[1];
    if ([i, j, i2, j2].every((v) => v >= 0 && v < GRID)) {
      return { i, j, d, n1: [i1, j1], n2: [i2, j2], perp: [-d[1], d[0]] };
    }
  }
}

// Distance of (x, z) along the site direction from A, and right of the ego lane center.
function alongOf(site, x, z) {
  const A = nodePos(site.i, site.j);
  return (x - A.x) * site.d[0] + (z - A.z) * site.d[1];
}
function lateralOf(site, x, z) {
  const A = nodePos(site.i, site.j);
  return (x - A.x) * -site.d[1] + (z - A.z) * site.d[0] - LANE_W / 2;
}

// Clearing zones along the site: every 30 m from A to N2.
function siteAreas(site, r = 45) {
  const A = nodePos(site.i, site.j), out = [];
  for (let a = 0; a <= 2 * PITCH; a += 30) out.push({ x: A.x + site.d[0] * a, z: A.z + site.d[1] * a, r });
  return out;
}

function setupSite(w, run, { egoAlong, egoV = 0, choose = straight(), green = true } = {}) {
  const site = pickSite(w.rand);
  run.site = site;
  w.clearArea(siteAreas(site));
  if (green) {
    w.signals.force(site.n1, axisOf(site.d), 0);
    w.signals.force(site.n2, axisOf(site.d), 0);
  }
  w.placeEgo({ i: site.i, j: site.j, d: site.d, along: egoAlong, v: egoV, choose });
  return site;
}

const egoAlong = (w, run) => alongOf(run.site, w.car.x, w.car.z);
const egoInLane = (w, run) => Math.abs(lateralOf(run.site, w.car.x, w.car.z)) < 0.9;
const VAN = { type: 'van', paint: 0xf2f2f0 };

// Ego is ahead of `car` by `margin` meters (bumper to bumper) and back in its lane.
function passed(w, run, car, margin = 6) {
  const a = car.agent, gap = egoAlong(w, run) - alongOf(run.site, a.x, a.z) - HALF_LEN - a.halfLen;
  return gap > margin && egoInLane(w, run);
}

// ---------- highway ----------

// Put the ego on a random stretch of the highway and clear background traffic around it.
function setupHighway(w, run, { lane, v }) {
  const dir = w.rand() < 0.5 ? 1 : -1, q = w.rand() * HW_LENGTH;
  run.hw = { dir, q };
  w.placeEgoHighway({ dir, lane, q, v });
  w.clearArea(w.highwayAreas(-150, 600));
}

// A scripted car `dq` meters ahead of the ego's start (center to center), same direction.
const hwCar = (w, run, lane, dq, opts) => w.spawnHighwayCar({ dir: run.hw.dir, lane, q: run.hw.q + dq }, { v: opts.desired, ...opts });
// Bumper-to-bumper distance from the ego to `car` ahead (negative once the ego is past it).
const hwGap = (w, car) => loopDist(w.ego.hw.q, car.agent.hw.q) - HALF_LEN - car.agent.halfLen;
const egoLane = (w) => Math.round((w.ego.hw.lat - HW.inner) / HW.laneW - 0.5);
const egoInHwLane = (w) => Math.abs((w.ego.hw.lat - HW.inner) / HW.laneW - 0.5 - egoLane(w)) < 0.25;

export const SCENARIOS = {
  'overtake-parked': {
    name: 'Overtake a double-parked van',
    goal: 'Pass the delivery van blocking the lane, using the oncoming lane when it is clear',
    timeout: 60,
    setup(w, run) {
      const site = setupSite(w, run, { egoAlong: 14, egoV: 6 });
      run.van = w.spawnCar({ i: site.i, j: site.j, d: site.d, along: 52, lateral: 0.5 }, { ...VAN, hold: true, hazard: true });
    },
    check: (w, run) => (passed(w, run, run.van) ? 'Passed the van and returned to the lane' : null),
  },

  'overtake-oncoming': {
    name: 'Overtake with oncoming traffic',
    goal: 'A van blocks the lane and cars are coming the other way: wait for a gap, then pass',
    timeout: 70,
    setup(w, run) {
      const site = setupSite(w, run, { egoAlong: 14, egoV: 6 });
      run.van = w.spawnCar({ i: site.i, j: site.j, d: site.d, along: 52, lateral: 0.5 }, { ...VAN, hold: true, hazard: true });
      const back = [-site.d[0], -site.d[1]];
      for (const along of [50, 72]) {
        w.spawnCar({ i: site.n2[0], j: site.n2[1], d: back, along }, { choose: straight(), v: 9, speedFactor: 1 });
      }
    },
    update(w, run) {
      if (w.expert.ot.info?.why === 'Oncoming traffic') run.flags.waited = true;
    },
    check: (w, run) => (passed(w, run, run.van) ? (run.flags.waited ? 'Waited for oncoming traffic, then passed' : 'Passed the van') : null),
  },

  'overtake-slow': {
    name: 'Overtake a slow vehicle',
    goal: 'Pass the slow utility van once the oncoming lane is clear and there is room before the intersection',
    timeout: 60,
    setup(w, run) {
      const site = setupSite(w, run, { egoAlong: 8, egoV: 4 });
      run.slow = w.spawnCar({ i: site.i, j: site.j, d: site.d, along: 22 }, {
        type: 'van', paint: 0xe07b12, choose: straight(), v: 3.5, vmax: 3.5, speedFactor: 1, slow: true,
      });
    },
    check: (w, run) => (passed(w, run, run.slow, 8) ? 'Overtook the slow vehicle' : null),
  },

  'lead-brake': {
    name: 'Lead vehicle emergency stop',
    goal: 'The car ahead slams on its brakes: stop without contact',
    timeout: 30,
    setup(w, run) {
      const site = setupSite(w, run, { egoAlong: 12, egoV: 10 });
      run.lead = w.spawnCar({ i: site.i, j: site.j, d: site.d, along: 34 }, { choose: straight(), v: 10, speedFactor: 1, paint: 0x8a1414 });
    },
    update(w, run, dt) {
      const lead = run.lead;
      if (!run.flags.braked && run.t > 2.5) {
        run.flags.braked = true;
        lead.forceAcc = -8;
      }
      if (run.flags.braked && lead.forceAcc !== null && lead.v === 0) {
        // Stalled: hazards on (the ego may then pass it).
        lead.forceAcc = null;
        lead.hold = true;
        lead.agent.hazard = true;
      }
      run.flags.minGap = Math.min(run.flags.minGap ?? Infinity, alongOf(run.site, lead.agent.x, lead.agent.z) - egoAlong(w, run) - HALF_LEN - lead.agent.halfLen);
    },
    check(w, run) {
      if (!run.lead.hold || w.car.v > 0.1) return null;
      return `Stopped ${run.flags.minGap.toFixed(1)} m behind the car`;
    },
  },

  jaywalker: {
    name: 'Jaywalker between parked cars',
    goal: 'A pedestrian runs out from behind a parked van: give way',
    timeout: 40,
    setup(w, run) {
      const site = setupSite(w, run, { egoAlong: 10, egoV: 10 });
      const van = w.lanePose(site, 50, 3.25);
      w.hideParked([{ x: van.x, z: van.z, r: 12 }]);
      w.spawnCar({ i: site.i, j: site.j, d: site.d, along: 50, lateral: 3.25 }, { ...VAN, paint: 0x3d6ea8, hold: true, parked: true });
      const from = w.lanePose(site, 56.5, 3.9), to = w.lanePose(site, 58, -(LANE_W + 5.1));
      run.ped = w.spawnPed(from.x, from.z, [{ x: to.x, z: to.z, mode: 'jaywalk' }], { speed: 2.3 });
    },
    update(w, run) {
      const ped = run.ped;
      if (!run.flags.go && egoAlong(w, run) + HALF_LEN > 56.5 - 24) {
        run.flags.go = true;
        ped.queue[0].go = true;
      }
      if (run.flags.go) run.flags.minV = Math.min(run.flags.minV ?? Infinity, w.car.v);
    },
    check(w, run) {
      if (!run.flags.go || lateralOf(run.site, run.ped.x, run.ped.z) > -(LANE_W + 4.6)) return null;
      return run.flags.minV < 0.5 ? 'Stopped for the pedestrian' : 'Slowed and let the pedestrian cross';
    },
  },

  'pull-out': {
    name: 'Car pulls out of a parking spot',
    goal: 'A parked car pulls into the lane just ahead: let it merge',
    timeout: 40,
    setup(w, run) {
      const site = setupSite(w, run, { egoAlong: 10, egoV: 10 });
      const spot = w.lanePose(site, 48, 3.25);
      w.hideParked([{ x: spot.x, z: spot.z, r: 9 }]);
      run.car = w.spawnCar({ i: site.i, j: site.j, d: site.d, along: 48, lateral: 3.25, merge: 14 }, {
        choose: straight(), hold: true, parked: true, speedFactor: 0.9, paint: 0x2e4d3a,
      });
    },
    update(w, run) {
      if (!run.flags.go && egoAlong(w, run) + HALF_LEN > 48 - 27) {
        run.flags.go = true;
        Object.assign(run.car, { hold: false, v: 2 });
        run.car.agent.parked = false;
      }
    },
    check: (w, run) => (run.flags.go && egoAlong(w, run) > 80 ? 'Let the car merge and followed it' : null),
  },

  'red-runner': {
    name: 'Red-light runner',
    goal: 'You have a green light, but a car on the cross street is not going to stop',
    timeout: 40,
    setup(w, run) {
      const site = setupSite(w, run, { egoAlong: 12, egoV: 10 });
      run.e = w.rand() < 0.5 ? site.perp : [-site.perp[0], -site.perp[1]];
    },
    update(w, run) {
      const site = run.site, e = run.e;
      if (!run.runner && egoAlong(w, run) > PITCH - 58) {
        const [i, j] = [site.n1[0] - e[0], site.n1[1] - e[1]];
        run.runner = w.spawnCar({ i, j, d: e, along: PITCH - 60 }, {
          choose: straight(), v: 12, speedFactor: 1.15, ignoreSignals: true, paint: 0xb8860b,
        });
        run.runner.scenario = true;
      }
    },
    check(w, run) {
      const r = run.runner;
      if (!r) return null;
      const N = nodePos(...run.site.n1);
      const runnerPast = (r.agent.x - N.x) * run.e[0] + (r.agent.z - N.z) * run.e[1] > 15;
      return runnerPast && egoAlong(w, run) > PITCH + 12 ? 'Yielded to the red-light runner' : null;
    },
  },

  'unprotected-left': {
    name: 'Unprotected left turn',
    goal: 'Turn left on a green light, yielding to oncoming traffic',
    timeout: 45,
    setup(w, run) {
      const site = setupSite(w, run, { egoAlong: 20, egoV: 8, choose: turns('left') });
      const back = [-site.d[0], -site.d[1]];
      for (const along of [10, 30, 50]) {
        w.spawnCar({ i: site.n2[0], j: site.n2[1], d: back, along }, { choose: straight(), v: 10, speedFactor: 1 });
      }
    },
    update(w, run) {
      if (w.ctrl?.reason === 'yield') run.flags.yielded = true;
    },
    check(w, run) {
      const N = nodePos(...run.site.n1), [dx, dz] = run.site.d;
      const left = (w.car.x - N.x) * dz + (w.car.z - N.z) * -dx;
      if (left < 15) return null;
      return run.flags.yielded ? 'Yielded to oncoming traffic, then turned' : 'Completed the left turn';
    },
  },
  'hw-cut-in': {
    road: 'highway',
    name: 'Highway: aggressive cut-in',
    goal: 'A slower car swerves into your lane just ahead and brakes: keep a safe distance',
    timeout: 40,
    setup(w, run) {
      setupHighway(w, run, { lane: 1, v: 27 });
      // Someone alongside on the left, so swerving away isn't an option.
      run.left = hwCar(w, run, 0, -4, { desired: 27, paint: 0x1f3a66 });
      run.cutter = hwCar(w, run, 2, 40, { desired: 21, paint: 0x8a1414 });
    },
    update(w, run) {
      const c = run.cutter, f = run.flags;
      if (!f.cut && hwGap(w, c) < 14) {
        // A 40 m swerve at 21 m/s: ~6 m/s^2 of lateral acceleration, holding speed through it.
        f.cut = true;
        f.cutT = run.t;
        c.route.changeLane(c.s, 1, 40, c.k);
        c.forceAcc = 0;
      }
      if (f.cut && !f.braked && run.t - f.cutT > 2) {
        f.braked = true;
        c.forceAcc = -4;
      }
      if (f.braked && c.forceAcc !== null && c.v < 13) {
        c.forceAcc = null;
        c.desired = 20;
      }
      // Closest approach while it's ahead of us in our lane.
      if (f.cut && Math.abs(w.ego.hw.lat - c.agent.hw.lat) < 2 && hwGap(w, c) > -1) f.minGap = Math.min(f.minGap ?? Infinity, hwGap(w, c));
    },
    check(w, run) {
      const f = run.flags;
      if (!f.cut || run.t - f.cutT < 10) return null;
      return Number.isFinite(f.minGap) ? `Kept ${f.minGap.toFixed(1)} m from the car that cut in` : 'Avoided the car that cut in';
    },
  },

  'hw-stalled': {
    road: 'highway',
    name: 'Highway: stalled car in the lane',
    goal: 'A broken-down car blocks the right lane: find a gap in the traffic beside you and move over',
    timeout: 60,
    setup(w, run) {
      setupHighway(w, run, { lane: 2, v: 27 });
      run.stalled = hwCar(w, run, 2, 280, { desired: 0, hold: true, hazard: true, paint: 0x4a4f55 });
      // Traffic in the middle lane the ego has to merge into.
      for (const dq of [-55, 25, 75]) hwCar(w, run, 1, dq, { desired: 24 });
    },
    check: (w, run) => (hwGap(w, run.stalled) < -2 * HALF_LEN - 8 && egoInHwLane(w) ? 'Changed lanes and passed the stalled car' : null),
  },

  'hw-jam': {
    road: 'highway',
    name: 'Highway: sudden traffic jam',
    goal: 'Traffic ahead brakes hard to a standstill in every lane: stop in time, then move off with it',
    timeout: 50,
    setup(w, run) {
      setupHighway(w, run, { lane: 1, v: 29 });
      run.jam = [];
      for (const [lane, dq] of [[0, 62], [1, 56], [2, 66], [0, 100], [1, 94], [2, 104]]) {
        run.jam.push(hwCar(w, run, lane, dq, { desired: 26, paint: 0x8f9499 }));
      }
    },
    update(w, run) {
      const f = run.flags;
      if (!f.brake && run.t > 4) {
        f.brake = run.t;
        for (const c of run.jam) c.forceAcc = -6.5;
      }
      for (const c of run.jam) if (c.forceAcc !== null && c.v === 0) c.forceAcc = 0;
      if (f.brake && !f.stopped && run.jam.every((c) => c.v === 0)) f.stopped = run.t;
      if (f.stopped && !f.release && run.t - f.stopped > 5) {
        f.release = run.t;
        for (const c of run.jam) c.forceAcc = null;
      }
      if (f.brake) f.minV = Math.min(f.minV ?? Infinity, w.car.v);
    },
    check: (w, run) => (run.flags.release && run.flags.minV < 3 && w.car.v > 15 ? 'Stopped for the jam and moved off with the traffic' : null),
  },
};
