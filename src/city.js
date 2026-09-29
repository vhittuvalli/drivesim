// Procedural city: street grid, sidewalks with rounded corners, lane markings, crosswalks,
// buildings on subdivided lots, street lights, trees and signalized intersections.
import * as THREE from 'three';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import {
  LANE_W, ROAD_W, SIDEWALK_W, BLOCK, PITCH, GRID, CURB_H, CORNER_R, CROSSWALK_NEAR, CROSSWALK_FAR,
  STOP_LINE, BAY, FLOOR, STOREFRONT_H, CITY_MIN, CITY_SIZE, nodePos, inGrid,
} from './config.js';
import { FACADE_W, FACADE_H, STOREFRONT_W, makeDetailMap, makePoolsMap, cityUniforms, worldToMap } from './materials.js';

export const DIRS = [
  { dx: 1, dz: 0, axis: 'ew' },
  { dx: -1, dz: 0, axis: 'ew' },
  { dx: 0, dz: 1, axis: 'ns' },
  { dx: 0, dz: -1, axis: 'ns' },
];

// ---------- geometry builder for merged meshes ----------
class Geo {
  constructor() {
    this.pos = [];
    this.nrm = [];
    this.uv = [];
    this.col = [];
    this.idx = [];
  }
  quad(p, n, uv, color = [1, 1, 1]) {
    const base = this.pos.length / 3;
    for (let k = 0; k < 4; k++) {
      this.pos.push(...p[k]);
      this.nrm.push(...n);
      this.uv.push(...uv[k]);
      this.col.push(...color);
    }
    this.idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  build() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nrm, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    g.setIndex(this.idx);
    g.computeBoundingSphere();
    return g;
  }
}

// Walls of an axis-aligned box footprint, outward-facing, UVs in facade-tile units.
function walls(geo, x0, z0, x1, z1, y0, y1, tileW, tileH, vBase, uOff, vOff, color) {
  const edges = [
    [x0, z1, x1, z1], // south
    [x1, z1, x1, z0], // east
    [x1, z0, x0, z0], // north
    [x0, z0, x0, z1], // west
  ];
  let u = uOff;
  for (const [ax, az, bx, bz] of edges) {
    const len = Math.hypot(bx - ax, bz - az);
    const n = [-(bz - az) / len, 0, (bx - ax) / len];
    const u1 = u + len / tileW;
    const v0 = (y0 - vBase) / tileH + vOff, v1 = (y1 - vBase) / tileH + vOff;
    geo.quad(
      [[ax, y0, az], [bx, y0, bz], [bx, y1, bz], [ax, y1, az]],
      n,
      [[u, v0], [u1, v0], [u1, v1], [u, v1]],
      color,
    );
    u = u1;
  }
}

function roofQuad(geo, x0, z0, x1, z1, y) {
  geo.quad([[x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0]], [0, 1, 0], [[x0, z0], [x0, z1], [x1, z1], [x1, z0]]);
}

function roundedRectShape(x0, z0, x1, z1, r) {
  // Shape (x, y) with y = -z so that rotateX(-PI/2) maps it back onto the ground.
  const s = new THREE.Shape();
  const X0 = x0, X1 = x1, Y0 = -z1, Y1 = -z0;
  s.moveTo(X0 + r, Y0);
  s.lineTo(X1 - r, Y0);
  s.absarc(X1 - r, Y0 + r, r, -Math.PI / 2, 0, false);
  s.lineTo(X1, Y1 - r);
  s.absarc(X1 - r, Y1 - r, r, 0, Math.PI / 2, false);
  s.lineTo(X0 + r, Y1);
  s.absarc(X0 + r, Y1 - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(X0, Y0 + r);
  s.absarc(X0 + r, Y0 + r, r, Math.PI, Math.PI * 1.5, false);
  return s;
}

// ---------- city ----------
export class City {
  constructor(mats, rand) {
    this.mats = mats;
    this.rand = rand;
    this.group = new THREE.Group();
    this.signalHeads = []; // { node:[i,j], axis, index }
    this.lightPositions = [];
    this.center = { x: ((GRID - 1) * PITCH) / 2, z: ((GRID - 1) * PITCH) / 2 };

    this.buildGround();
    this.buildBlocks();
    this.buildMarkings();
    this.buildStreetFurniture();
    this.buildSignals();
    this.buildMaps();
  }

  // Road segments: pairs of adjacent intersections, plus stubs leading out of the grid.
  *segments() {
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        if (i + 1 < GRID) yield { a: [i, j], b: [i + 1, j], axis: 'ew' };
        if (j + 1 < GRID) yield { a: [i, j], b: [i, j + 1], axis: 'ns' };
      }
    }
  }

  buildGround() {
    const size = CITY_SIZE;
    const g = new THREE.PlaneGeometry(size, size, 1, 1);
    g.rotateX(-Math.PI / 2);
    g.translate(CITY_MIN + size / 2, 0, CITY_MIN + size / 2);
    // UVs in meters so textures tile at their real-world size.
    const uv = g.attributes.uv;
    const p = g.attributes.position;
    for (let k = 0; k < uv.count; k++) uv.setXY(k, p.getX(k), -p.getZ(k));
    const m = new THREE.Mesh(g, this.mats.asphalt);
    m.receiveShadow = true;
    this.group.add(m);

    // Far terrain beyond the city, under the fog.
    const far = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000), new THREE.MeshStandardMaterial({ color: 0x3d4034, roughness: 1 }));
    far.rotation.x = -Math.PI / 2;
    far.position.set(this.center.x, -0.05, this.center.z);
    this.group.add(far);
  }

  buildBlocks() {
    const rand = this.rand;
    const geos = { brick: new Geo(), plaster: new Geo(), concrete: new Geo(), glass: new Geo() };
    const shop = new Geo();
    const roof = new Geo();
    const blockGeos = [];
    const maxD = Math.hypot(this.center.x, this.center.z) + PITCH;

    for (let bi = -1; bi < GRID; bi++) {
      for (let bj = -1; bj < GRID; bj++) {
        const x0 = bi * PITCH + ROAD_W / 2, x1 = (bi + 1) * PITCH - ROAD_W / 2;
        const z0 = bj * PITCH + ROAD_W / 2, z1 = (bj + 1) * PITCH - ROAD_W / 2;
        const shape = roundedRectShape(x0, z0, x1, z1, CORNER_R);
        const eg = new THREE.ExtrudeGeometry(shape, { depth: CURB_H, bevelEnabled: false, curveSegments: 8 });
        eg.rotateX(-Math.PI / 2);
        blockGeos.push(eg);

        // Downtown gets taller.
        const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
        const d = Math.hypot(cx - this.center.x, cz - this.center.z) / maxD;
        const downtown = Math.max(0, 1 - d * 1.4);
        this.buildLots(x0 + SIDEWALK_W + 0.5, z0 + SIDEWALK_W + 0.5, x1 - SIDEWALK_W - 0.5, z1 - SIDEWALK_W - 0.5, downtown, geos, shop, roof);
      }
    }

    const blocks = new THREE.Mesh(mergeGeometries(blockGeos, false), [this.mats.sidewalk, this.mats.curb]);
    // mergeGeometries drops groups; rebuild caps/sides using normals: top faces -> sidewalk, sides -> curb.
    blocks.geometry = this.splitCapsAndSides(blocks.geometry);
    blocks.receiveShadow = true;
    this.group.add(blocks);

    for (const [style, geo] of Object.entries(geos)) {
      if (!geo.pos.length) continue;
      const m = new THREE.Mesh(geo.build(), this.mats.facades[style]);
      m.castShadow = m.receiveShadow = true;
      this.group.add(m);
    }
    const s = new THREE.Mesh(shop.build(), this.mats.shop);
    s.castShadow = s.receiveShadow = true;
    const r = new THREE.Mesh(roof.build(), this.mats.roof);
    r.castShadow = r.receiveShadow = true;
    this.group.add(s, r);
  }

  splitCapsAndSides(g) {
    g = g.index ? g.toNonIndexed() : g;
    const n = g.attributes.normal;
    const tris = n.count / 3;
    const caps = [], sides = [];
    for (let t = 0; t < tris; t++) (Math.abs(n.getY(t * 3)) > 0.5 ? caps : sides).push(t);
    const order = [...caps, ...sides];
    const out = new THREE.BufferGeometry();
    for (const name of Object.keys(g.attributes)) {
      const src = g.attributes[name];
      const arr = new Float32Array(src.array.length);
      order.forEach((t, k) => arr.set(src.array.subarray(t * 3 * src.itemSize, (t + 1) * 3 * src.itemSize), k * 3 * src.itemSize));
      out.setAttribute(name, new THREE.BufferAttribute(arr, src.itemSize));
    }
    // Extrude UVs are in meters; scale by the texture's repeat is handled by the texture itself.
    out.addGroup(0, caps.length * 3, 0);
    out.addGroup(caps.length * 3, sides.length * 3, 1);
    out.computeBoundingSphere();
    return out;
  }

  buildLots(x0, z0, x1, z1, downtown, geos, shop, roof) {
    const rand = this.rand;
    // Split into a grid of 2-3 columns and rows, with occasional merged lots.
    const cuts = (a, b) => {
      const n = 2 + Math.floor(rand() * 2);
      const out = [a];
      for (let k = 1; k < n; k++) out.push(a + ((b - a) * k) / n + (rand() - 0.5) * 6);
      out.push(b);
      return out;
    };
    const xs = cuts(x0, x1), zs = cuts(z0, z1);
    for (let i = 0; i < xs.length - 1; i++) {
      for (let j = 0; j < zs.length - 1; j++) {
        // Interior lots of a 3x3 split are courtyards / skipped sometimes.
        const interior = i > 0 && i < xs.length - 2 && j > 0 && j < zs.length - 2;
        if (interior && rand() < 0.6) continue;
        const gap = 0.5 + rand() * 1.5;
        let lx0 = xs[i] + gap, lx1 = xs[i + 1] - gap, lz0 = zs[j] + gap, lz1 = zs[j + 1] - gap;
        // Snap to whole bays, keeping the street-facing edge fixed.
        const w = Math.max(BAY * 3, Math.floor((lx1 - lx0) / BAY) * BAY);
        const dpt = Math.max(BAY * 3, Math.floor((lz1 - lz0) / BAY) * BAY);
        if (i === 0) lx1 = lx0 + w; else lx0 = lx1 - w;
        if (j === 0) lz1 = lz0 + dpt; else lz0 = lz1 - dpt;
        this.building(lx0, lz0, lx1, lz1, downtown, geos, shop, roof);
      }
    }
  }

  building(x0, z0, x1, z1, downtown, geos, shop, roof) {
    const rand = this.rand;
    const tall = downtown * downtown * (40 + rand() * 110) + rand() * 12;
    const floors = Math.max(2, Math.round(tall / FLOOR) + 1 + Math.floor(rand() * 3));
    const h = STOREFRONT_H + floors * FLOOR;
    let style;
    if (h > 45) style = rand() < 0.65 ? 'glass' : 'concrete';
    else if (h > 22) style = ['concrete', 'brick', 'glass', 'plaster'][Math.floor(rand() * 4)];
    else style = rand() < 0.55 ? 'brick' : 'plaster';

    const tint = {
      brick: () => { const k = 0.8 + rand() * 0.3; return [k, k * (0.9 + rand() * 0.1), k * (0.85 + rand() * 0.1)]; },
      plaster: () => {
        const palettes = [[1, 0.93, 0.82], [0.98, 0.88, 0.78], [0.9, 0.9, 0.88], [1, 0.86, 0.7], [0.85, 0.9, 0.95], [0.95, 0.82, 0.78]];
        return palettes[Math.floor(rand() * palettes.length)];
      },
      concrete: () => { const k = 0.85 + rand() * 0.2; return [k, k, k * 0.98]; },
      glass: () => { const k = 0.9 + rand() * 0.15; return [k, k, k]; },
    }[style]();

    const uOff = Math.floor(rand() * 8) * (BAY / FACADE_W);
    const vOff = Math.floor(rand() * 8) * (FLOOR / FACADE_H);
    // Ground-floor retail band.
    walls(shop, x0, z0, x1, z1, 0, STOREFRONT_H, STOREFRONT_W, STOREFRONT_H, 0, rand(), 0, [1, 1, 1]);

    // Tall buildings get a podium and a set-back tower.
    const setback = h > 40 && rand() < 0.6 && x1 - x0 > 18 && z1 - z0 > 18;
    if (setback) {
      const podium = STOREFRONT_H + (2 + Math.floor(rand() * 3)) * FLOOR;
      walls(geos[style], x0, z0, x1, z1, STOREFRONT_H, podium, FACADE_W, FACADE_H, STOREFRONT_H, uOff, vOff, tint);
      roofQuad(roof, x0, z0, x1, z1, podium);
      const inset = BAY * (1 + Math.floor(rand() * 2));
      walls(geos[style], x0 + inset, z0 + inset, x1 - inset, z1 - inset, podium, h, FACADE_W, FACADE_H, STOREFRONT_H, uOff, vOff, tint);
      roofQuad(roof, x0 + inset, z0 + inset, x1 - inset, z1 - inset, h);
      this.rooftop(roof, x0 + inset, z0 + inset, x1 - inset, z1 - inset, h);
    } else {
      walls(geos[style], x0, z0, x1, z1, STOREFRONT_H, h, FACADE_W, FACADE_H, STOREFRONT_H, uOff, vOff, tint);
      roofQuad(roof, x0, z0, x1, z1, h);
      this.rooftop(roof, x0, z0, x1, z1, h);
    }
    // Thin cornice/parapet cap.
    walls(roof, x0 - 0.15, z0 - 0.15, x1 + 0.15, z1 + 0.15, STOREFRONT_H - 0.35, STOREFRONT_H, 2, 2, 0, 0, 0, [0.9, 0.88, 0.85]);
  }

  rooftop(roof, x0, z0, x1, z1, h) {
    const rand = this.rand;
    // Parapet.
    walls(roof, x0, z0, x1, z1, h, h + 0.9, 2, 2, h, 0, 0, [0.85, 0.85, 0.85]);
    const n = Math.floor(rand() * 4);
    for (let k = 0; k < n; k++) {
      const w = 2 + rand() * 4, d = 2 + rand() * 4, hh = 1.5 + rand() * 3;
      const cx = x0 + 2 + rand() * Math.max(0, x1 - x0 - 4 - w), cz = z0 + 2 + rand() * Math.max(0, z1 - z0 - 4 - d);
      walls(roof, cx, cz, cx + w, cz + d, h, h + hh, 2, 2, h, 0, 0, [0.7, 0.72, 0.75]);
      roofQuad(roof, cx, cz, cx + w, cz + d, h + hh);
    }
  }

  // ---------- road markings ----------
  buildMarkings() {
    const geo = new Geo();
    const Y = 0.012;
    const WHITE = [0.92, 0.92, 0.9], YELLOW = [0.95, 0.72, 0.12];
    const rect = (x0, z0, x1, z1, c) => {
      geo.quad([[x0, Y, z0], [x0, Y, z1], [x1, Y, z1], [x1, Y, z0]], [0, 1, 0], [[0, 0], [0, 1], [1, 1], [1, 0]], c);
    };
    // Rect in segment-local coords: along (from node a) and across (+ = right of a->b direction).
    const local = (seg, a0, a1, c0, c1, color) => {
      const A = nodePos(...seg.a);
      if (seg.axis === 'ew') rect(A.x + a0, A.z + c0, A.x + a1, A.z + c1, color);
      else rect(A.x - c1, A.z + a0, A.x - c0, A.z + a1, color);
    };
    const edge = LANE_W + 0.1;
    for (const seg of this.segments()) {
      const s0 = STOP_LINE + 0.5, s1 = PITCH - STOP_LINE - 0.5;
      // Double yellow center line.
      local(seg, s0, s1, 0.1, 0.22, YELLOW);
      local(seg, s0, s1, -0.22, -0.1, YELLOW);
      // Edge lines between travel lane and parking lane.
      local(seg, s0, s1, edge, edge + 0.15, WHITE);
      local(seg, s0, s1, -edge - 0.15, -edge, WHITE);
      // Parking stall ticks.
      for (let a = s0 + 6; a < s1 - 3; a += 6.5) {
        local(seg, a, a + 0.12, edge + 0.15, ROAD_W / 2 - 0.3, WHITE);
        local(seg, a, a + 0.12, -ROAD_W / 2 + 0.3, -edge - 0.15, WHITE);
      }
      // Stop lines: a->b traffic uses the right side (+) and stops before b; b->a uses (-) and stops before a.
      local(seg, PITCH - STOP_LINE - 0.5, PITCH - STOP_LINE, 0.25, edge, WHITE);
      local(seg, STOP_LINE, STOP_LINE + 0.5, -edge, -0.25, WHITE);
    }
    // Continental crosswalks on every arm of every intersection.
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const N = nodePos(i, j);
        for (const d of DIRS) {
          if (!inGrid(i + d.dx, j + d.dz)) continue;
          for (let c = -ROAD_W / 2 + 0.6; c < ROAD_W / 2 - 0.5; c += 1.2) {
            const a0 = CROSSWALK_NEAR, a1 = CROSSWALK_FAR;
            if (d.axis === 'ew') {
              const xa = N.x + d.dx * a0, xb = N.x + d.dx * a1;
              rect(Math.min(xa, xb), N.z + c, Math.max(xa, xb), N.z + c + 0.6, WHITE);
            } else {
              const za = N.z + d.dz * a0, zb = N.z + d.dz * a1;
              rect(N.x + c, Math.min(za, zb), N.x + c + 0.6, Math.max(za, zb), WHITE);
            }
          }
        }
      }
    }
    const m = new THREE.Mesh(geo.build(), this.mats.marking);
    m.receiveShadow = true;
    this.group.add(m);
  }

  // ---------- street lights and trees ----------
  buildStreetFurniture() {
    const rand = this.rand;
    const lights = [];
    const trees = [];
    const place = (x, z, dirX, dirZ, list) => list.push({ x, z, dirX, dirZ });
    const curbOff = ROAD_W / 2 + 0.7;
    // Every road edge in the grid, including the outer ring roads' far sides.
    for (const seg of this.segments()) {
      const A = nodePos(...seg.a);
      for (const side of [-1, 1]) {
        for (let a = 20; a < PITCH - 15; a += 28.5) {
          const [x, z, dx, dz] = seg.axis === 'ew'
            ? [A.x + a, A.z + side * curbOff, 0, -side]
            : [A.x - side * curbOff, A.z + a, side, 0];
          place(x, z, dx, dz, lights);
        }
        for (let a = 34; a < PITCH - 20; a += 28.5) {
          if (rand() < 0.2) continue;
          const off = ROAD_W / 2 + 1.6;
          const a2 = a + (rand() - 0.5) * 3;
          if (seg.axis === 'ew') place(A.x + a2, A.z + side * off, 0, 0, trees);
          else place(A.x + side * off, A.z + a2, 0, 0, trees);
        }
      }
    }
    this.lightPositions = lights;

    // Street lights: pole + arm + luminaire.
    const n = lights.length;
    const poleGeo = new THREE.CylinderGeometry(0.09, 0.14, 8.5, 10);
    poleGeo.translate(0, 4.25, 0);
    const armGeo = new THREE.BoxGeometry(0.08, 0.08, 2.4);
    armGeo.translate(0, 8.3, 1.2);
    const headGeo = new THREE.BoxGeometry(0.45, 0.14, 0.8);
    headGeo.translate(0, 8.25, 2.35);
    const lensGeo = new THREE.PlaneGeometry(0.35, 0.6);
    lensGeo.rotateX(Math.PI / 2);
    lensGeo.translate(0, 8.17, 2.35);
    const poles = new THREE.InstancedMesh(mergeGeometries([poleGeo, armGeo, headGeo]), this.mats.darkMetal, n);
    const lenses = new THREE.InstancedMesh(lensGeo, this.mats.lamp, n);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    lights.forEach((l, k) => {
      // Arm extends along local +z; rotate it to point toward the road.
      q.setFromAxisAngle(up, Math.atan2(l.dirX, l.dirZ));
      m.compose(new THREE.Vector3(l.x, CURB_H, l.z), q, new THREE.Vector3(1, 1, 1));
      poles.setMatrixAt(k, m);
      lenses.setMatrixAt(k, m);
    });
    poles.castShadow = true;
    this.group.add(poles, lenses);

    // Trees: tapered trunk with a few limbs, and a canopy of alpha-tested leaf-cluster cards.
    // Card normals point away from the canopy center so it shades like a soft volume.
    const trunkParts = [new THREE.CylinderGeometry(0.1, 0.2, 3.6, 8).translate(0, 1.8, 0)];
    for (let k = 0; k < 4; k++) {
      const limb = new THREE.CylinderGeometry(0.04, 0.08, 1.8, 6).translate(0, 0.9, 0);
      limb.rotateZ(0.7 + rand() * 0.3);
      limb.rotateY((k / 4) * Math.PI * 2 + rand());
      trunkParts.push(limb.translate(0, 3.1 + rand() * 0.5, 0));
    }
    const trunk = mergeGeometries(trunkParts);
    const cards = [];
    const center = new THREE.Vector3(0, 5.0, 0);
    for (let k = 0; k < 70; k++) {
      const size = 1.3 + rand() * 0.9;
      const card = new THREE.PlaneGeometry(size, size);
      const dir = new THREE.Vector3(rand() - 0.5, (rand() - 0.5) * 0.8, rand() - 0.5).normalize();
      const r = Math.cbrt(rand());
      const pos = dir.clone().multiply(new THREE.Vector3(2.4, 1.8, 2.4)).multiplyScalar(0.35 + 0.65 * r).add(center);
      card.lookAt(new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5));
      card.translate(pos.x, pos.y, pos.z);
      const nrm = card.attributes.normal;
      const p = card.attributes.position;
      for (let v = 0; v < nrm.count; v++) {
        const n = new THREE.Vector3(p.getX(v), p.getY(v), p.getZ(v)).sub(center).normalize();
        nrm.setXYZ(v, n.x, n.y, n.z);
      }
      cards.push(card);
      const back = card.clone();
      const idx = back.index.array;
      for (let t = 0; t < idx.length; t += 3) [idx[t + 1], idx[t + 2]] = [idx[t + 2], idx[t + 1]];
      cards.push(back);
    }
    const canopyGeo = mergeGeometries(cards);
    const trunks = new THREE.InstancedMesh(trunk, this.mats.bark, trees.length);
    const canopies = new THREE.InstancedMesh(canopyGeo, this.mats.foliage, trees.length);
    const color = new THREE.Color();
    trees.forEach((t, k) => {
      const s = 0.75 + rand() * 0.5;
      q.setFromAxisAngle(up, rand() * Math.PI * 2);
      m.compose(new THREE.Vector3(t.x, CURB_H, t.z), q, new THREE.Vector3(s, s * (0.9 + rand() * 0.25), s));
      trunks.setMatrixAt(k, m);
      canopies.setMatrixAt(k, m);
      const b = 0.8 + rand() * 0.35;
      color.setRGB(b * (0.9 + rand() * 0.2), b, b * (0.8 + rand() * 0.2));
      canopies.setColorAt(k, color);
    });
    trunks.castShadow = canopies.castShadow = true;
    canopies.receiveShadow = true;
    this.group.add(trunks, canopies);
  }

  // ---------- traffic signals ----------
  buildSignals() {
    // For each approach into each intersection: a mast-arm pole on the far-right corner with a
    // signal head over the lane, plus a second head on the pole. US-style far-side signals.
    const approaches = [];
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        for (const d of DIRS) {
          // Traffic arriving at (i,j) traveling in direction d comes from (i-dx, j-dz).
          if (!inGrid(i - d.dx, j - d.dz)) continue;
          approaches.push({ node: [i, j], d });
        }
      }
    }
    this.approaches = approaches;
    const n = approaches.length;

    const poleGeo = new THREE.CylinderGeometry(0.13, 0.17, 6.2, 10);
    poleGeo.translate(0, 3.1, 0);
    const armGeo = new THREE.CylinderGeometry(0.07, 0.09, 6.5, 8);
    armGeo.rotateZ(Math.PI / 2);
    armGeo.translate(-3.25, 5.9, 0);
    const poles = new THREE.InstancedMesh(mergeGeometries([poleGeo, armGeo]), this.mats.metal, n);

    // Two heads per approach: overhead (on arm) and pole-mounted.
    const headGeo = new THREE.BoxGeometry(0.34, 1.05, 0.28);
    const backGeo = new THREE.BoxGeometry(0.6, 1.3, 0.03);
    backGeo.translate(0, 0, 0.16);
    const visorGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.22, 12, 1, true, 0, Math.PI);
    visorGeo.rotateX(Math.PI / 2);
    visorGeo.rotateZ(Math.PI / 2);
    const visors = [];
    for (const y of [0.33, 0, -0.33]) visors.push(visorGeo.clone().translate(0, y + 0.02, -0.24));
    const housing = new THREE.InstancedMesh(mergeGeometries([headGeo, backGeo, ...visors]), this.mats.signalHousing, n * 2);
    const lensGeo = new THREE.CircleGeometry(0.11, 16);
    lensGeo.rotateY(Math.PI);
    this.lamps = {};
    const lampMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
    for (const [k, y] of [['red', 0.33], ['yellow', 0], ['green', -0.33]]) {
      const g = lensGeo.clone().translate(0, y, -0.145);
      this.lamps[k] = new THREE.InstancedMesh(g, lampMat, n * 2);
    }

    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), one = new THREE.Vector3(1, 1, 1), up = new THREE.Vector3(0, 1, 0);
    const heads = [];
    approaches.forEach((ap, k) => {
      const N = nodePos(...ap.node);
      const { dx, dz } = ap.d;
      const rx = -dz, rz = dx; // right of travel direction
      const c = ROAD_W / 2 + 1.0;
      const px = N.x + dx * c + rx * c, pz = N.z + dz * c + rz * c;
      // Pole local -x is the arm direction; point it at the road (-right).
      q.setFromAxisAngle(up, Math.atan2(-rz, rx) + 0);
      m.compose(new THREE.Vector3(px, CURB_H, pz), q, one);
      poles.setMatrixAt(k, m);
      // Heads face oncoming traffic (-d). Local -z is the lens side.
      const yaw = Math.atan2(dx, dz);
      q.setFromAxisAngle(up, yaw);
      const overX = N.x + dx * c + rx * (LANE_W / 2), overZ = N.z + dz * c + rz * (LANE_W / 2);
      heads.push({ k, pos: new THREE.Vector3(overX, 5.25, overZ), q: q.clone() });
      heads.push({ k, pos: new THREE.Vector3(px - rx * 0.35, 3.3, pz - rz * 0.35), q: q.clone() });
    });
    heads.forEach((h, idx) => {
      m.compose(h.pos, h.q, one);
      housing.setMatrixAt(idx, m);
      for (const l of Object.values(this.lamps)) l.setMatrixAt(idx, m);
    });
    this.headApproach = heads.map((h) => h.k);
    poles.castShadow = housing.castShadow = true;
    this.group.add(poles, housing, ...Object.values(this.lamps));
  }

  // Called by the signal controller each frame.
  setSignalColors(stateOf) {
    const on = { red: new THREE.Color(6, 0.35, 0.2), yellow: new THREE.Color(6, 3.2, 0.2), green: new THREE.Color(0.2, 5, 2.2) };
    const off = { red: new THREE.Color(0.08, 0.015, 0.01), yellow: new THREE.Color(0.08, 0.05, 0.01), green: new THREE.Color(0.01, 0.06, 0.03) };
    this.headApproach.forEach((k, idx) => {
      const ap = this.approaches[k];
      const state = stateOf(ap.node, ap.d.axis);
      for (const c of ['red', 'yellow', 'green']) this.lamps[c].setColorAt(idx, c === state ? on[c] : off[c]);
    });
    for (const l of Object.values(this.lamps)) l.instanceColor.needsUpdate = true;
  }

  // ---------- city-wide wear map and night light pools ----------
  buildMaps() {
    const rand = this.rand;
    const detail = makeDetailMap((g, s) => {
      const X = worldToMap, W = (m) => m * s;
      // No ctx.filter here: repeated blur filters on a large canvas can wipe it in Chrome.
      // Soft edges come from gradients instead.
      for (const seg of this.segments()) {
        const A = nodePos(...seg.a);
        // Tire tracks: soft, slightly darker bands in each wheel path.
        for (const lane of [-LANE_W / 2, LANE_W / 2]) {
          for (const w of [-0.85, 0.85]) {
            const c = lane + w;
            const [x0, z0] = seg.axis === 'ew' ? [X(A.x), X(A.z + c - 0.6)] : [X(A.x - c - 0.6), X(A.z)];
            const across = W(1.2);
            const grad = seg.axis === 'ew' ? g.createLinearGradient(0, z0, 0, z0 + across) : g.createLinearGradient(x0, 0, x0 + across, 0);
            grad.addColorStop(0, 'rgba(40,40,40,0)');
            grad.addColorStop(0.5, 'rgba(40,40,40,0.07)');
            grad.addColorStop(1, 'rgba(40,40,40,0)');
            g.fillStyle = grad;
            if (seg.axis === 'ew') g.fillRect(x0, z0, W(PITCH), across);
            else g.fillRect(x0, z0, across, W(PITCH));
          }
        }
        // Oil drips at stop lines.
        for (const [a, c] of [[PITCH - STOP_LINE - 3, LANE_W / 2], [STOP_LINE + 3, -LANE_W / 2]]) {
          const [x, z] = seg.axis === 'ew' ? [A.x + a, A.z + c] : [A.x - c, A.z + a];
          const grad = g.createRadialGradient(X(x), X(z), 0, X(x), X(z), W(2.2));
          grad.addColorStop(0, 'rgba(20,20,22,0.35)');
          grad.addColorStop(1, 'rgba(20,20,22,0)');
          g.fillStyle = grad;
          g.fillRect(X(x) - W(3), X(z) - W(3), W(6), W(6));
        }
        // Repair patches (hard-edged, like real cut-and-patch repairs).
        for (let k = 0; k < 3; k++) {
          if (rand() < 0.4) continue;
          const a = 10 + rand() * (PITCH - 20), c = (rand() - 0.5) * (ROAD_W - 3);
          const len = 1.5 + rand() * 6, wid = 1 + rand() * 2.5;
          const tone = rand() < 0.6 ? `rgba(30,30,32,${0.12 + rand() * 0.12})` : `rgba(255,255,255,0.25)`;
          g.fillStyle = tone;
          if (seg.axis === 'ew') g.fillRect(X(A.x + a), X(A.z + c), W(len), W(wid));
          else g.fillRect(X(A.x + c), X(A.z + a), W(wid), W(len));
        }
      }
      // Intersections: heavier wear.
      for (let i = 0; i < GRID; i++) {
        for (let j = 0; j < GRID; j++) {
          const N = nodePos(i, j);
          const grad = g.createRadialGradient(X(N.x), X(N.z), 0, X(N.x), X(N.z), W(ROAD_W * 0.7));
          grad.addColorStop(0, 'rgba(35,35,35,0.12)');
          grad.addColorStop(1, 'rgba(35,35,35,0)');
          g.fillStyle = grad;
          g.fillRect(X(N.x) - W(ROAD_W), X(N.z) - W(ROAD_W), W(ROAD_W * 2), W(ROAD_W * 2));
          // Manhole cover.
          g.fillStyle = 'rgba(25,25,25,0.4)';
          g.beginPath();
          g.arc(X(N.x + (rand() - 0.5) * 4), X(N.z + (rand() - 0.5) * 4), W(0.35), 0, Math.PI * 2);
          g.fill();
        }
      }
    });
    cityUniforms.uDetail.value = detail;

    const pools = makePoolsMap((g, s) => {
      const X = (v) => ((v - CITY_MIN) / CITY_SIZE) * 1024;
      for (const l of this.lightPositions) {
        const x = l.x + l.dirX * 2.35, z = l.z + l.dirZ * 2.35;
        const grad = g.createRadialGradient(X(x), X(z), 0, X(x), X(z), 13 * s);
        grad.addColorStop(0, 'rgba(255,190,120,0.9)');
        grad.addColorStop(0.5, 'rgba(255,170,100,0.35)');
        grad.addColorStop(1, 'rgba(255,160,90,0)');
        g.fillStyle = grad;
        g.fillRect(X(x) - 13 * s, X(z) - 13 * s, 26 * s, 26 * s);
      }
    });
    cityUniforms.uPools.value = pools;
  }
}
