// Vehicle body types and paint distribution (no rendering dependencies, usable in tests).

// Dimensions in meters; profile x runs rear (-) to front (+).
export const BODY_TYPES = {
  sedan: { L: 4.7, W: 1.84, clr: 0.36, wb: 2.8, wr: 0.33, hood: 0.8, belt: 0.98, deck: 1.0, roof: 1.44, ws: 0.95, rf: 0.15, rr: -0.95, rw: -1.75, weight: 0.38 },
  hatch: { L: 4.2, W: 1.78, clr: 0.36, wb: 2.6, wr: 0.32, hood: 0.8, belt: 0.98, deck: 1.02, roof: 1.48, ws: 0.85, rf: 0.05, rr: -1.55, rw: -2.02, weight: 0.2 },
  suv: { L: 4.8, W: 1.94, clr: 0.45, wb: 2.85, wr: 0.38, hood: 1.02, belt: 1.18, deck: 1.2, roof: 1.78, ws: 1.05, rf: 0.25, rr: -1.95, rw: -2.3, weight: 0.25 },
  van: { L: 5.2, W: 1.98, clr: 0.4, wb: 3.3, wr: 0.36, hood: 1.0, belt: 1.2, deck: 1.25, roof: 2.1, ws: 1.75, rf: 1.15, rr: -2.5, rw: -2.58, weight: 0.1 },
  taxi: { L: 4.7, W: 1.84, clr: 0.36, wb: 2.8, wr: 0.33, hood: 0.8, belt: 0.98, deck: 1.0, roof: 1.44, ws: 0.95, rf: 0.15, rr: -0.95, rw: -1.75, weight: 0.07, taxi: true },
};

// Real-world color distribution: white, black, grays/silver dominate.
const PAINTS = [
  [0.22, 0xf2f2f0], [0.18, 0x111214], [0.14, 0x8f9499], [0.12, 0xc3c7cc], [0.08, 0x4a4f55],
  [0.08, 0x1f3a66], [0.07, 0x8a1414], [0.04, 0x2e4d3a], [0.03, 0x6b5a45], [0.03, 0x3d6ea8], [0.02, 0xb8860b],
];

export function randomPaint(rand) {
  let r = rand();
  for (const [w, c] of PAINTS) if ((r -= w) < 0) return c;
  return PAINTS[0][1];
}

export function randomBodyType(rand) {
  let r = rand() * Object.values(BODY_TYPES).reduce((a, t) => a + t.weight, 0);
  for (const [name, t] of Object.entries(BODY_TYPES)) if ((r -= t.weight) < 0) return name;
  return 'sedan';
}
