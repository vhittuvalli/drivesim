// Road users other than the ego vehicle.
import { LANE_W, ROAD_W, PITCH, STOP_LINE, nodePos } from './config.js';
import { randomBodyType, randomPaint } from './fleet.js';

// Parked cars fill a fraction of the marked parking stalls (see City.buildMarkings).
export function placeParkedCars(city, fleet, rand, density = 0.3) {
  const parked = [];
  const lateral = LANE_W + (ROAD_W / 2 - LANE_W) / 2; // parking lane center
  const s0 = STOP_LINE + 0.5, s1 = PITCH - STOP_LINE - 0.5;
  for (const seg of city.segments()) {
    const A = nodePos(...seg.a);
    for (const side of [1, -1]) {
      // Stalls are 6.5 m long between the ticks.
      for (let a = s0 + 6; a + 6.5 < s1 - 3; a += 6.5) {
        if (rand() > density) continue;
        const type = randomBodyType(rand);
        if (type === 'van' && rand() < 0.5) continue;
        const h = fleet.acquire(type, randomPaint(rand));
        if (!h) continue;
        const along = a + 3.25 + (rand() - 0.5) * 0.8;
        const off = side * (lateral + (rand() - 0.5) * 0.25);
        // Cars park facing the direction of travel on their side of the street.
        const [x, z, heading] = seg.axis === 'ew'
          ? [A.x + along, A.z + off, side > 0 ? 0 : Math.PI]
          : [A.x - off, A.z + along, side > 0 ? Math.PI / 2 : -Math.PI / 2];
        const jitter = (rand() - 0.5) * 0.04;
        fleet.set(h, x, z, heading + jitter, false);
        parked.push({ kind: 'car', parked: true, x, z, h: heading, v: 0, halfLen: h.spec.L / 2, handle: h });
      }
    }
  }
  return parked;
}
