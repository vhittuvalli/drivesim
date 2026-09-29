// Parametric vehicle bodies rendered with instancing. Every vehicle of a body type shares
// geometry; per-car paint and brake-light state live in per-instance colors, so hundreds of
// cars cost a few dozen draw calls.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

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

const flat = (g) => (g.index ? g.toNonIndexed() : g);

function extrudeProfile(shape, width, bevel) {
  const g = new THREE.ExtrudeGeometry(shape, { depth: width, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel * 0.9, bevelSegments: 2, curveSegments: 8 });
  g.translate(0, 0, -width / 2);
  g.deleteAttribute('uv');
  return flat(g);
}

function box(w, h, d, x, y, z) {
  const g = new THREE.BoxGeometry(w, h, d);
  g.translate(x, y, z);
  g.deleteAttribute('uv');
  return flat(g);
}

function cyl(r, h, x, y, z, seg = 16) {
  const g = new THREE.CylinderGeometry(r, r, h, seg);
  g.rotateX(Math.PI / 2);
  g.translate(x, y, z);
  g.deleteAttribute('uv');
  return flat(g);
}

// Build merged geometry for each material slot of one body type.
function bodyParts(t) {
  const hl = t.L / 2, ax = t.wb / 2, ar = t.wr + 0.08;
  const lower = new THREE.Shape();
  lower.moveTo(-hl + 0.12, t.clr);
  lower.lineTo(-ax - ar, t.clr);
  lower.absarc(-ax, t.wr, ar, Math.PI, 0, true);
  lower.lineTo(ax - ar, t.clr);
  lower.absarc(ax, t.wr, ar, Math.PI, 0, true);
  lower.lineTo(hl - 0.18, t.clr);
  lower.quadraticCurveTo(hl + 0.02, t.clr + 0.04, hl, t.clr + 0.28);
  lower.quadraticCurveTo(hl - 0.04, t.hood - 0.02, hl - 0.3, t.hood);
  lower.lineTo(t.ws, t.belt);
  lower.lineTo(t.rw + 0.05, t.belt + 0.02);
  lower.quadraticCurveTo(-hl + 0.1, t.deck, -hl + 0.02, t.deck - 0.12);
  lower.quadraticCurveTo(-hl - 0.04, t.clr + 0.25, -hl + 0.12, t.clr);

  const glassShape = new THREE.Shape([
    new THREE.Vector2(t.ws, t.belt - 0.03),
    new THREE.Vector2(t.rf, t.roof),
    new THREE.Vector2(t.rr, t.roof),
    new THREE.Vector2(t.rw, t.belt - 0.03),
  ]);
  const roofShape = new THREE.Shape([
    new THREE.Vector2(t.rf - 0.04, t.roof - 0.04),
    new THREE.Vector2(t.rr + 0.04, t.roof - 0.04),
    new THREE.Vector2(t.rr + 0.06, t.roof + 0.025),
    new THREE.Vector2(t.rf - 0.06, t.roof + 0.025),
  ]);

  const paint = [extrudeProfile(lower, t.W - 0.16, 0.08), extrudeProfile(roofShape, t.W - 0.4, 0.03)];
  const pillarX = (t.rf + t.rr) / 2;
  paint.push(box(0.1, t.roof - t.belt, t.W - 0.26, pillarX, (t.roof + t.belt) / 2, 0));
  for (const z of [-1, 1]) paint.push(box(0.14, 0.11, 0.18, t.ws - 0.15, t.belt + 0.08, z * (t.W / 2 + 0.05)));
  const glass = [extrudeProfile(glassShape, t.W - 0.34, 0.04)];

  const trim = [
    box(t.wb + 0.2, 0.09, t.W + 0.02, 0, t.clr + 0.05, 0), // sills
    box(0.05, 0.16, t.W * 0.5, hl + 0.01, t.clr + 0.2, 0), // grille
    box(0.1, 0.12, t.W * 0.8, -hl + 0.04, t.clr + 0.1, 0), // rear bumper insert
  ];
  const chrome = [];
  for (const x of [-ax, ax]) {
    for (const z of [-1, 1]) {
      trim.push(cyl(t.wr, 0.23, x, t.wr, z * (t.W / 2 - 0.13), 18));
      chrome.push(cyl(t.wr * 0.62, 0.235, x, t.wr, z * (t.W / 2 - 0.13), 12));
    }
  }
  const head = [], tail = [];
  for (const z of [-1, 1]) {
    head.push(box(0.1, 0.09, 0.38, hl - 0.06, t.hood - 0.1, z * (t.W / 2 - 0.3)));
    tail.push(box(0.06, 0.11, 0.42, -hl + 0.02, t.deck - 0.16, z * (t.W / 2 - 0.3)));
  }
  if (t.taxi) {
    paint.push(box(0.35, 0.22, 0.9, (t.rf + t.rr) / 2, t.roof + 0.14, 0));
  }
  return {
    paint: mergeGeometries(paint),
    glass: mergeGeometries(glass),
    trim: mergeGeometries(trim),
    chrome: mergeGeometries(chrome),
    head: mergeGeometries(head),
    tail: mergeGeometries(tail),
  };
}

const TAIL_ON = new THREE.Color(5, 0.15, 0.08);
const TAIL_OFF = new THREE.Color(0.55, 0.03, 0.02);
const TAIL_NIGHT = new THREE.Color(1.6, 0.06, 0.03);

export class Fleet {
  constructor(capacityPerType) {
    this.group = new THREE.Group();
    this.types = {};
    const mats = {
      paint: new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0.5, roughness: 0.35, clearcoat: 1, clearcoatRoughness: 0.05 }),
      glass: new THREE.MeshPhysicalMaterial({ color: 0x0b0f13, metalness: 0.3, roughness: 0.05, clearcoat: 1 }),
      trim: new THREE.MeshStandardMaterial({ color: 0x141517, roughness: 0.75, metalness: 0.2 }),
      chrome: new THREE.MeshStandardMaterial({ color: 0xb8bcc0, roughness: 0.25, metalness: 1 }),
      head: new THREE.MeshBasicMaterial({ color: 0xdddddd, toneMapped: false }),
      tail: new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }),
    };
    this.mats = mats;
    for (const [name, spec] of Object.entries(BODY_TYPES)) {
      const parts = bodyParts(spec);
      const meshes = {};
      for (const [slot, geo] of Object.entries(parts)) {
        const m = new THREE.InstancedMesh(geo, mats[slot], capacityPerType);
        m.count = 0;
        m.frustumCulled = false;
        m.castShadow = slot !== 'head' && slot !== 'tail';
        m.receiveShadow = slot === 'paint';
        meshes[slot] = m;
        this.group.add(m);
      }
      this.types[name] = { spec, meshes, free: [], used: 0, capacity: capacityPerType };
    }
    this._m = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._p = new THREE.Vector3();
    this._s = new THREE.Vector3(1, 1, 1);
    this._up = new THREE.Vector3(0, 1, 0);
    this._c = new THREE.Color();
    this.night = 0;
  }

  // Reserve an instance slot; returns a handle or null when the type is full.
  acquire(typeName, paint) {
    const t = this.types[typeName];
    let idx = t.free.pop();
    if (idx === undefined) {
      if (t.used >= t.capacity) return null;
      idx = t.used++;
      for (const m of Object.values(t.meshes)) m.count = t.used;
    }
    const color = typeName === 'taxi' ? 0xf2b705 : paint;
    t.meshes.paint.setColorAt(idx, this._c.set(color));
    t.meshes.paint.instanceColor.needsUpdate = true;
    t.meshes.tail.setColorAt(idx, TAIL_OFF);
    t.meshes.tail.instanceColor.needsUpdate = true;
    return { type: typeName, idx, spec: t.spec };
  }

  release(h) {
    this.set(h, 0, -1000, 0, false);
    this.types[h.type].free.push(h.idx);
  }

  set(h, x, z, heading, braking) {
    const t = this.types[h.type];
    this._q.setFromAxisAngle(this._up, -heading);
    this._p.set(x, 0, z);
    this._m.compose(this._p, this._q, this._s);
    for (const m of Object.values(t.meshes)) {
      m.setMatrixAt(h.idx, this._m);
      m.instanceMatrix.needsUpdate = true;
    }
    t.meshes.tail.setColorAt(h.idx, braking ? TAIL_ON : this.night > 0.3 ? TAIL_NIGHT : TAIL_OFF);
    t.meshes.tail.instanceColor.needsUpdate = true;
  }

  setNight(night) {
    this.night = night;
    this.mats.head.color.setScalar(0.6 + night * 5);
  }
}
