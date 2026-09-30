// Rendering of the highway loop (geometry and driving logic live in highway.js): road surface,
// lane markings, a concrete median barrier, guardrails, lamp posts, overhead sign gantries and
// trees along the verge. Everything is built along the same analytic centerline the vehicles use.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { HW, HW_LENGTH, HW_SEGMENTS, centerAt } from './highway.js';
import { patchGround } from './materials.js';

const STEP = 4; // meters between cross sections

// A strip along the centerline between lateral offsets o0 and o1 (right of the median) at
// height y, or a vertical strip between heights when `wall` is set. UVs in meters.
function strip(u0, u1, o0, o1, y, { wall = null, step = STEP } = {}) {
  if (o0 > o1) [o0, o1] = [o1, o0]; // keep the winding facing up on either side of the median
  const pos = [], uv = [], idx = [];
  const n = Math.max(1, Math.ceil((u1 - u0) / step));
  for (let i = 0; i <= n; i++) {
    const u = u0 + ((u1 - u0) * i) / n, c = centerAt(u), nx = -Math.sin(c.h), nz = Math.cos(c.h);
    if (wall) {
      pos.push(c.x + nx * o0, wall[0], c.z + nz * o0, c.x + nx * o0, wall[1], c.z + nz * o0);
      uv.push(u, wall[0], u, wall[1]);
    } else {
      pos.push(c.x + nx * o0, y, c.z + nz * o0, c.x + nx * o1, y, c.z + nz * o1);
      uv.push(u, o0, u, o1);
    }
    if (i < n) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2); // faces up
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// Extrude a cross-section profile [[o, y], ...] (left to right) along the whole loop.
function extrudeProfile(profile) {
  const n = Math.ceil(HW_LENGTH / STEP), m = profile.length, pos = [], idx = [];
  for (let i = 0; i <= n; i++) {
    const c = centerAt((HW_LENGTH * i) / n), nx = -Math.sin(c.h), nz = Math.cos(c.h);
    for (const [o, y] of profile) pos.push(c.x + nx * o, y, c.z + nz * o);
    if (i === n) break;
    for (let k = 0; k < m - 1; k++) {
      const a = i * m + k, b = a + m;
      idx.push(a, a + 1, b, a + 1, b + 1, b); // faces outward
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// Matrix placing an object at centerline u, offset o, with local x along the road.
function placeAt(u, o = 0, y = 0, turn = 0) {
  const c = centerAt(u);
  const m = new THREE.Matrix4().makeRotationY(-c.h + turn);
  m.setPosition(c.x - Math.sin(c.h) * o, y, c.z + Math.cos(c.h) * o);
  return m;
}

function instanced(geo, mat, matrices, { cast = true } = {}) {
  const mesh = new THREE.InstancedMesh(geo, mat, matrices.length);
  matrices.forEach((m, i) => mesh.setMatrixAt(i, m));
  mesh.castShadow = cast;
  mesh.receiveShadow = true;
  mesh.computeBoundingSphere();
  return mesh;
}

function signTexture(lines) {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 160;
  const g = c.getContext('2d');
  g.fillStyle = '#0f6b3a';
  g.fillRect(0, 0, 512, 160);
  g.strokeStyle = '#f2f2f2';
  g.lineWidth = 6;
  g.strokeRect(8, 8, 496, 144);
  g.fillStyle = '#f2f2f2';
  g.font = 'bold 46px system-ui, sans-serif';
  g.textBaseline = 'middle';
  g.fillText(lines[0], 30, 56);
  g.font = '36px system-ui, sans-serif';
  g.fillText(lines[1], 30, 112);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

// mats: from createMaterials (asphalt textures, lamp, metal). Returns {group, classes}, where
// classes maps the highway's own materials to label classes for the sensor rig.
export function buildHighway(mats, rand) {
  const group = new THREE.Group();
  group.name = 'highway';
  const a = mats.asphalt;
  // Highway asphalt: the city's textures without the city-sized wear and light-pool maps.
  const asphalt = patchGround(new THREE.MeshStandardMaterial({
    map: a.map, normalMap: a.normalMap, roughnessMap: a.roughnessMap, aoMap: a.aoMap,
    color: 0xc4c4c4, roughness: 1, normalScale: new THREE.Vector2(0.8, 0.8),
  }), { antiTile: true, pools: false });
  const marking = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
  const concrete = new THREE.MeshStandardMaterial({ color: 0xb4b0a7, roughness: 0.9 });
  const steel = new THREE.MeshStandardMaterial({ color: 0xa2a8ae, roughness: 0.4, metalness: 0.8, side: THREE.DoubleSide });
  const foliage = new THREE.MeshStandardMaterial({ color: 0x2f4a2a, roughness: 0.95 });
  const add = (mesh, { cast = false } = {}) => {
    mesh.castShadow = cast;
    mesh.receiveShadow = true;
    group.add(mesh);
    return mesh;
  };

  // Road surface, shoulders to guardrails, across the median.
  const W = HW.width + 0.6;
  const road = strip(0, HW_LENGTH, -W, W, 0.02);
  road.setAttribute('uv1', road.attributes.uv);
  add(new THREE.Mesh(road, asphalt));

  // Markings: yellow edge line along the median, white edge line on the shoulder, dashed lane
  // lines (3 m dash, 9 m gap). Vertex colors so it's one draw call.
  const lines = [];
  const colored = (g, rgb) => {
    const n = g.attributes.position.count, col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) col.set(rgb, i * 3);
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.deleteAttribute('uv');
    return g;
  };
  const WHITE = [0.92, 0.92, 0.9], YELLOW = [0.95, 0.72, 0.12], lw = 0.15, y = 0.035;
  for (const sg of [1, -1]) {
    const o = (x) => sg * x;
    lines.push(colored(strip(0, HW_LENGTH, o(HW.inner - lw / 2), o(HW.inner + lw / 2), y), YELLOW));
    lines.push(colored(strip(0, HW_LENGTH, o(HW.laneEdge - lw / 2), o(HW.laneEdge + lw / 2), y), WHITE));
    for (let l = 1; l < HW.lanes; l++) {
      const off = HW.inner + HW.laneW * l;
      for (let u = 0; u + 3 < HW_LENGTH; u += 12) lines.push(colored(strip(u, u + 3, o(off - 0.06), o(off + 0.06), y, { step: 3 }), WHITE));
    }
  }
  add(new THREE.Mesh(mergeGeometries(lines.map((g) => (g.index ? g.toNonIndexed() : g))), marking));

  // Median: concrete jersey barrier.
  add(new THREE.Mesh(extrudeProfile([[-0.34, 0], [-0.24, 0.26], [-0.1, 0.84], [0.1, 0.84], [0.24, 0.26], [0.34, 0]]), concrete), { cast: true });

  // Guardrails: a steel beam on posts every 4 m.
  const posts = [];
  for (const sg of [1, -1]) {
    const o = sg * (HW.width - 0.2);
    add(new THREE.Mesh(strip(0, HW_LENGTH, o, o, 0, { wall: [0.45, 0.78] }), steel), { cast: true });
    for (let u = 0; u < HW_LENGTH; u += 4) posts.push(placeAt(u, o + sg * 0.12, 0.4));
  }
  group.add(instanced(new THREE.BoxGeometry(0.12, 0.8, 0.12), steel, posts));

  // Lamp posts on the median with an arm over each carriageway.
  const lampU = [];
  for (let u = 10; u < HW_LENGTH; u += 70) lampU.push(u);
  const pole = new THREE.CylinderGeometry(0.1, 0.16, 11, 8).translate(0, 5.5, 0);
  const arms = [1, -1].map((sg) => new THREE.BoxGeometry(0.1, 0.1, 3.2).translate(0, 10.9, sg * 1.6));
  const poleGeo = mergeGeometries([pole, ...arms].map((g) => g.toNonIndexed()));
  const head = mergeGeometries([1, -1].map((sg) => new THREE.BoxGeometry(0.35, 0.14, 0.8).translate(0, 10.8, sg * 3.1).toNonIndexed()));
  group.add(instanced(poleGeo, mats.metal, lampU.map((u) => placeAt(u))));
  group.add(instanced(head, mats.lamp, lampU.map((u) => placeAt(u)), { cast: false }));

  // Sign gantries halfway along each straight side, one sign per carriageway facing its traffic.
  const straights = HW_SEGMENTS.filter((s) => !s.k && s.len > 400);
  const names = [['Downtown', 'Exit 3 · 1 km'], ['Airport', 'Next right · 2 km'], ['Harbor', 'Exit 7 · 800 m'], ['University', 'Exit 5 · 1.5 km']];
  const frame = [], boards = [];
  straights.forEach((s, i) => {
    const u = s.u0 + s.len / 2, span = HW.width + 0.8;
    frame.push(new THREE.BoxGeometry(0.4, 7.6, 0.4).translate(0, 3.8, span).applyMatrix4(placeAt(u)));
    frame.push(new THREE.BoxGeometry(0.4, 7.6, 0.4).translate(0, 3.8, -span).applyMatrix4(placeAt(u)));
    frame.push(new THREE.BoxGeometry(0.5, 0.5, 2 * span).translate(0, 7.3, 0).applyMatrix4(placeAt(u)));
    for (const sg of [1, -1]) {
      const [a1, a2] = names[(i + (sg > 0 ? 0 : 2)) % names.length];
      const mat = new THREE.MeshStandardMaterial({ map: signTexture([a1, a2]), roughness: 0.5, emissive: 0xffffff, emissiveIntensity: 0 });
      mat.userData.sign = true;
      // Face the traffic on that side: forward traffic (right of the median) comes from -x.
      const board = new THREE.Mesh(new THREE.PlaneGeometry(8, 2.5), mat);
      board.applyMatrix4(new THREE.Matrix4().makeRotationY(sg > 0 ? -Math.PI / 2 : Math.PI / 2));
      board.applyMatrix4(new THREE.Matrix4().makeTranslation(sg > 0 ? -0.3 : 0.3, 8.8, sg * (HW.inner + 1.5 * HW.laneW)));
      board.applyMatrix4(placeAt(u));
      board.castShadow = true;
      boards.push(board);
      group.add(board);
    }
  });
  add(new THREE.Mesh(mergeGeometries(frame.map((g) => g.toNonIndexed())), mats.metal), { cast: true });

  // Trees scattered along the verges.
  const trees = [], trunks = [];
  for (let u = 0; u < HW_LENGTH; u += 9) {
    for (const sg of [1, -1]) {
      if (rand() < 0.45) continue;
      const o = sg * (HW.width + 6 + rand() * 30), s = 0.7 + rand() * 0.8;
      const m = placeAt(u + rand() * 6, o, 0, rand() * 6).multiply(new THREE.Matrix4().makeScale(s, s * (0.8 + rand() * 0.5), s));
      trees.push(m);
      trunks.push(m);
    }
  }
  const crown = mergeGeometries([new THREE.ConeGeometry(2.2, 5, 7).translate(0, 4.5, 0), new THREE.ConeGeometry(1.6, 3.5, 7).translate(0, 7, 0)].map((g) => g.toNonIndexed()));
  group.add(instanced(crown, foliage, trees));
  group.add(instanced(new THREE.CylinderGeometry(0.18, 0.25, 2.2, 6).translate(0, 1.1, 0), mats.bark, trunks));

  const classes = new Map([
    [asphalt, 'road'], [marking, 'marking'], [concrete, 'building'], [steel, 'building'], [foliage, 'vegetation'],
    ...boards.map((b) => [b.material, 'pole']),
  ]);
  return { group, classes, signs: boards.map((b) => b.material) };
}
