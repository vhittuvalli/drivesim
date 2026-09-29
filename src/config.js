// City dimensions in meters. Right-hand traffic, one lane each way plus a parking/shoulder lane.
// World axes: x = east, z = south, y = up.

export const LANE_W = 3.5;
export const ROAD_W = 13; // curb to curb
export const SIDEWALK_W = 5;
export const BLOCK = 84; // curb to curb across a block
export const PITCH = ROAD_W + BLOCK; // distance between intersection centers
export const GRID = 7; // intersections per side
export const CURB_H = 0.15;
export const CORNER_R = 5;

export const CROSSWALK_NEAR = ROAD_W / 2 + 1; // crosswalk band, distance from intersection center
export const CROSSWALK_FAR = ROAD_W / 2 + 4.5;
export const STOP_LINE = ROAD_W / 2 + 5; // near edge of stop line (0.5 m wide)

// Ego vehicle
export const WHEELBASE = 2.8;
export const MAX_STEER = 0.6;
export const MAX_SPEED = 20;

export const BAY = 3; // facade window bay width
export const FLOOR = 3.5; // floor height
export const STOREFRONT_H = 4.5;

export const CITY_MIN = -PITCH * 1.5;
export const CITY_SIZE = PITCH * (GRID + 2);

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const nodePos = (i, j) => ({ x: i * PITCH, z: j * PITCH });
export const inGrid = (i, j) => i >= 0 && j >= 0 && i < GRID && j < GRID;
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
