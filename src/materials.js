// Materials: photo-scanned PBR textures (Poly Haven, CC0) plus procedurally painted facades.
// Ground materials get a shader patch that adds large-scale color variation (hides tiling),
// a city-wide wear map (tire tracks, oil, repair patches) and night-time street-light pools.
import * as THREE from 'three';
import { CITY_MIN, CITY_SIZE, BAY, FLOOR, STOREFRONT_H } from './config.js';

const TEX = 'assets/tex/';
const loader = new THREE.TextureLoader();

export const cityUniforms = {
  uMacro: { value: null },
  uDetail: { value: null },
  uPools: { value: null },
  uCityMin: { value: new THREE.Vector2(CITY_MIN, CITY_MIN) },
  uCitySize: { value: new THREE.Vector2(CITY_SIZE, CITY_SIZE) },
  uNight: { value: 0 },
  uWet: { value: 0 }, // rain: darker, glossier ground with puddles
  uSnow: { value: 0 }, // patchy snow cover
};

function loadImage(name) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${name}`));
    img.src = TEX + name;
  });
}

async function loadTex(name, { srgb = false, meters = 1, aniso = 1 } = {}) {
  const t = await loader.loadAsync(TEX + name);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(1 / meters, 1 / meters);
  t.anisotropy = aniso;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ---------- procedural helper textures ----------

// Tileable multi-octave value noise; R and G channels are independent.
function macroNoiseTexture(size = 256) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  const lattice = (n, seed) => {
    let s = seed;
    const r = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    return Array.from({ length: n * n }, r);
  };
  const sample = (grid, n, x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y);
    const fx = x - xi, fy = y - yi;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const at = (i, j) => grid[((j % n + n) % n) * n + ((i % n + n) % n)];
    const a = at(xi, yi), b = at(xi + 1, yi), c2 = at(xi, yi + 1), d = at(xi + 1, yi + 1);
    return a + (b - a) * sx + (c2 - a) * sy + (a - b - c2 + d) * sx * sy;
  };
  const octaves = [[4, 0.5], [8, 0.25], [16, 0.15], [32, 0.1]];
  const grids = [0, 1].map((ch) => octaves.map(([n], k) => lattice(n, 1234 + ch * 97 + k * 13)));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      for (let ch = 0; ch < 2; ch++) {
        let v = 0;
        octaves.forEach(([n, w], k) => (v += w * sample(grids[ch][k], n, (x / size) * n, (y / size) * n)));
        img.data[i + ch] = Math.round(v * 255);
      }
      img.data[i + 2] = 128;
      img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

// Patches a MeshStandardMaterial with world-space variation, wear and light pools.
export function patchGround(mat, { detail = false, pools = true, antiTile = false } = {}) {
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, cityUniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vCityXZ;')
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        vec4 cityWorld = vec4(transformed, 1.0);
        #ifdef USE_INSTANCING
          cityWorld = instanceMatrix * cityWorld;
        #endif
        vCityXZ = (modelMatrix * cityWorld).xz;`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying vec2 vCityXZ;
        uniform sampler2D uMacro, uDetail, uPools;
        uniform vec2 uCityMin, uCitySize;
        uniform float uNight, uWet, uSnow;
        float puddleMask() { return smoothstep(0.5, 0.62, texture2D(uMacro, vCityXZ / 23.0).g); }`,
      )
      .replace(
        '#include <map_fragment>',
        `${antiTile ? `
        #ifdef USE_MAP
          // Anti-tiling: blend with a rotated, rescaled second lookup, switched by low-frequency noise.
          float tileMix = smoothstep(0.35, 0.65, texture2D(uMacro, vCityXZ / 29.0).g);
          vec2 uv2 = mat2(0.8, -0.6, 0.6, 0.8) * vMapUv * 0.63 + vec2(0.31, 0.77);
          diffuseColor *= mix(texture2D(map, vMapUv), texture2D(map, uv2), tileMix);
        #endif` : '#include <map_fragment>'}
        {
          float m = texture2D(uMacro, vCityXZ / 53.0).r * 0.65 + texture2D(uMacro, vCityXZ / 11.3).g * 0.35;
          diffuseColor.rgb *= mix(0.8, 1.15, m);
          ${detail ? 'diffuseColor.rgb *= texture2D(uDetail, (vCityXZ - uCityMin) / uCitySize).rgb;' : ''}
          diffuseColor.rgb *= 1.0 - uWet * (0.3 + 0.2 * puddleMask());
          float snowCover = uSnow * smoothstep(0.32, 0.55, texture2D(uMacro, vCityXZ / 7.0).r * 0.6 + m * 0.4);
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.9, 0.92, 0.95), snowCover);
        }`,
      )
      .replace(
        '#include <roughnessmap_fragment>',
        `#include <roughnessmap_fragment>
        roughnessFactor = mix(roughnessFactor, mix(0.32, 0.04, puddleMask()), uWet);
        roughnessFactor = mix(roughnessFactor, 0.85, uSnow * 0.6);`,
      )
      .replace(
        '#include <emissivemap_fragment>',
        `#include <emissivemap_fragment>
        ${pools ? 'totalEmissiveRadiance += diffuseColor.rgb * texture2D(uPools, (vCityXZ - uCityMin) / uCitySize).rgb * uNight * 2.5;' : ''}`,
      );
  };
  mat.customProgramCacheKey = () => `ground-${detail}-${pools}-${antiTile}`;
  return mat;
}

// ---------- facades ----------
// Each facade texture covers 8 bays x 8 floors (24 m x 28 m). Buildings are snapped to whole
// bays/floors so windows line up with corners. Three canvases per style:
//   color (sRGB), rm (G = roughness, B = metalness), emissive (lit windows at night).

const FACADE_BAYS = 8, FACADE_FLOORS = 8;
export const FACADE_W = FACADE_BAYS * BAY;
export const FACADE_H = FACADE_FLOORS * FLOOR;
export const STOREFRONT_W = FACADE_W;

function canvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return [c, c.getContext('2d')];
}

function tile(g, img, w, h, tw, th) {
  for (let y = 0; y < h; y += th) for (let x = 0; x < w; x += tw) g.drawImage(img, x, y, tw, th);
}

function grime(g, w, h, rand, amount = 40) {
  for (let k = 0; k < amount; k++) {
    const x = rand() * w, y = rand() * h, r = 30 + rand() * 160;
    const grad = g.createRadialGradient(x, y, 0, x, y, r);
    const a = 0.04 + rand() * 0.08;
    grad.addColorStop(0, rand() < 0.7 ? `rgba(30,25,20,${a})` : `rgba(255,250,240,${a * 0.6})`);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grad;
    g.fillRect(x - r, y - r, r * 2, r * 2);
  }
}

function streak(g, x, y, w, len, a = 0.18) {
  const grad = g.createLinearGradient(0, y, 0, y + len);
  grad.addColorStop(0, `rgba(20,18,15,${a})`);
  grad.addColorStop(1, 'rgba(20,18,15,0)');
  g.fillStyle = grad;
  g.fillRect(x, y, w, len);
}

function glassFill(g, x, y, w, h, rand, tone = [70, 85, 100]) {
  const j = (rand() - 0.5) * 30;
  const grad = g.createLinearGradient(x, y, x + w * 0.3, y + h);
  grad.addColorStop(0, `rgb(${tone[0] + 60 + j},${tone[1] + 60 + j},${tone[2] + 60 + j})`);
  grad.addColorStop(0.45, `rgb(${tone[0] + j},${tone[1] + j},${tone[2] + j})`);
  grad.addColorStop(1, `rgb(${tone[0] * 0.4 + j},${tone[1] * 0.4 + j},${tone[2] * 0.45 + j})`);
  g.fillStyle = grad;
  g.fillRect(x, y, w, h);
}

const LIT = ['255,196,130', '255,214,160', '255,232,200', '220,232,255', '255,180,110'];

function litWindow(e, x, y, w, h, rand, strength = 1) {
  const c = LIT[Math.floor(rand() * LIT.length)];
  const a = (0.55 + rand() * 0.45) * strength;
  const grad = e.createLinearGradient(0, y, 0, y + h);
  grad.addColorStop(0, `rgba(${c},${a * 0.7})`);
  grad.addColorStop(1, `rgba(${c},${a})`);
  e.fillStyle = grad;
  e.fillRect(x, y, w, h);
}

const PX = 2048; // color canvas size
const EPX = 512; // rm / emissive canvas size
const K = EPX / PX;
const bayPx = PX / FACADE_BAYS, floorPx = PX / FACADE_FLOORS;
const m2x = bayPx / BAY, m2y = floorPx / FLOOR; // meters -> px

function facade(style, imgs, rand) {
  const [c, g] = canvas(PX, PX);
  const [rmC, rm] = canvas(EPX, EPX);
  const [eC, e] = canvas(EPX, EPX);
  e.fillStyle = '#000';
  e.fillRect(0, 0, EPX, EPX);

  const rmFill = (x, y, w, h, rough, metal) => {
    rm.fillStyle = `rgb(0,${Math.round(rough * 255)},${Math.round(metal * 255)})`;
    rm.fillRect(x * K, y * K, w * K, h * K);
  };

  if (style === 'brick' || style === 'plaster' || style === 'concrete') {
    const base = { brick: imgs.brick, plaster: imgs.plaster, concrete: imgs.concrete }[style];
    const tileM = { brick: 2, plaster: 2, concrete: 8 }[style];
    tile(g, base, PX, PX, tileM * m2x, tileM * m2y);
    if (style === 'concrete') {
      g.fillStyle = 'rgba(235,232,225,0.25)';
      g.fillRect(0, 0, PX, PX);
    }
    rmFill(0, 0, PX, PX, style === 'concrete' ? 0.8 : 0.92, 0);
    grime(g, PX, PX, rand, 60);

    for (let f = 0; f < FACADE_FLOORS; f++) {
      const fy = PX - (f + 1) * floorPx; // canvas top of this floor
      if (style === 'concrete') {
        // Ribbon windows with mullions.
        const wy = fy + 0.55 * m2y, wh = 1.75 * m2y;
        const lit = rand() < 0.45;
        g.fillStyle = '#3a3f44';
        g.fillRect(0, wy - 6, PX, wh + 12);
        for (let b = 0; b < FACADE_BAYS * 2; b++) {
          const x = b * (bayPx / 2);
          glassFill(g, x + 5, wy, bayPx / 2 - 10, wh, rand, [60, 72, 82]);
          if (lit && rand() < 0.8) litWindow(e, (x + 5) * K, wy * K, (bayPx / 2 - 10) * K, wh * K, rand, 0.8);
        }
        rmFill(0, wy, PX, wh, 0.06, 0.55);
        g.fillStyle = 'rgba(210,205,195,0.9)';
        g.fillRect(0, fy + floorPx - 0.12 * m2y, PX, 0.12 * m2y);
        streak(g, 0, wy + wh + 6, PX, 0.4 * m2y, 0.12);
        continue;
      }
      for (let b = 0; b < FACADE_BAYS; b++) {
        const bx = b * bayPx;
        const ww = (style === 'brick' ? 1.3 : 1.15) * m2x, wh = (style === 'brick' ? 1.8 : 2.0) * m2y;
        const wx = bx + (bayPx - ww) / 2, wy = fy + floorPx - 0.9 * m2y - wh;
        // Surround: lintel and sill.
        g.fillStyle = style === 'brick' ? '#b9b2a6' : 'rgba(250,246,238,0.85)';
        g.fillRect(wx - 10, wy - 16, ww + 20, 16);
        g.fillRect(wx - 12, wy + wh, ww + 24, 12);
        // Frame and glass.
        g.fillStyle = style === 'brick' ? '#e9e6df' : '#5b4a3a';
        g.fillRect(wx, wy, ww, wh);
        const inset = 7;
        glassFill(g, wx + inset, wy + inset, ww - inset * 2, wh - inset * 2, rand);
        rmFill(wx + inset, wy + inset, ww - inset * 2, wh - inset * 2, 0.05, 0.35);
        // Interior variety: blinds / curtains.
        const r = rand();
        if (r < 0.25) {
          g.fillStyle = 'rgba(225,220,205,0.75)';
          const bh = (wh - inset * 2) * (0.2 + rand() * 0.6);
          for (let s = 0; s < bh; s += 5) g.fillRect(wx + inset, wy + inset + s, ww - inset * 2, 3);
        } else if (r < 0.45) {
          g.fillStyle = `rgba(${170 + rand() * 60},${140 + rand() * 50},${110 + rand() * 40},0.8)`;
          g.fillRect(wx + inset, wy + inset, (ww - inset * 2) * 0.3, wh - inset * 2);
          g.fillRect(wx + ww - inset - (ww - inset * 2) * 0.3, wy + inset, (ww - inset * 2) * 0.3, wh - inset * 2);
        }
        // Muntins.
        g.fillStyle = style === 'brick' ? '#e9e6df' : '#5b4a3a';
        g.fillRect(wx + ww / 2 - 3, wy, 6, wh);
        g.fillRect(wx, wy + wh * 0.38, ww, 5);
        // Shutters on plaster buildings.
        if (style === 'plaster' && rand() < 0.6) {
          const shutter = ['#3e5a3c', '#5a3a2c', '#39485c', '#6a6a5e'][Math.floor(rand() * 4)];
          g.fillStyle = shutter;
          g.fillRect(wx - ww * 0.48, wy, ww * 0.45, wh);
          g.fillRect(wx + ww * 1.03, wy, ww * 0.45, wh);
          g.fillStyle = 'rgba(0,0,0,0.25)';
          for (let s = 0; s < wh; s += 9) {
            g.fillRect(wx - ww * 0.48, wy + s, ww * 0.45, 2);
            g.fillRect(wx + ww * 1.03, wy + s, ww * 0.45, 2);
          }
        }
        streak(g, wx - 6, wy + wh + 12, ww + 12, 0.8 * m2y, 0.14);
        if (rand() < 0.32) litWindow(e, (wx + inset) * K, (wy + inset) * K, (ww - inset * 2) * K, (wh - inset * 2) * K, rand);
      }
      if (style === 'plaster') {
        g.fillStyle = 'rgba(255,252,245,0.5)';
        g.fillRect(0, fy + floorPx - 0.18 * m2y, PX, 0.18 * m2y);
      }
    }
  } else if (style === 'glass') {
    // Curtain wall: glass panels, aluminum mullions, spandrel band at each slab.
    const tone = [[38, 62, 78], [44, 70, 66], [58, 64, 74], [30, 44, 60]][Math.floor(rand() * 4)];
    for (let f = 0; f < FACADE_FLOORS; f++) {
      const fy = PX - (f + 1) * floorPx;
      const floorLit = rand() < 0.5;
      for (let b = 0; b < FACADE_BAYS * 2; b++) {
        const x = b * (bayPx / 2);
        glassFill(g, x, fy, bayPx / 2, floorPx, rand, tone);
        if (floorLit ? rand() < 0.85 : rand() < 0.1) {
          litWindow(e, x * K, (fy + 0.9 * m2y) * K, (bayPx / 2) * K, (floorPx - 0.9 * m2y) * K, rand, 0.7);
        }
      }
      rmFill(0, fy, PX, floorPx, 0.04, 0.85);
      // Spandrel.
      g.fillStyle = `rgba(${tone[0] * 0.5},${tone[1] * 0.5},${tone[2] * 0.5},0.92)`;
      g.fillRect(0, fy + floorPx - 0.9 * m2y, PX, 0.9 * m2y);
      rmFill(0, fy + floorPx - 0.9 * m2y, PX, 0.9 * m2y, 0.15, 0.7);
      g.fillStyle = '#9ea4a8';
      g.fillRect(0, fy + floorPx - 5, PX, 8);
      rmFill(0, fy + floorPx - 5, PX, 8, 0.35, 0.9);
    }
    g.fillStyle = '#9ea4a8';
    for (let b = 0; b <= FACADE_BAYS * 2; b++) g.fillRect(b * (bayPx / 2) - 4, 0, 8, PX);
    for (let b = 0; b <= FACADE_BAYS * 2; b++) rmFill(b * (bayPx / 2) - 4, 0, 8, PX, 0.35, 0.9);
  }

  return finish(c, rmC, eC);
}

function finish(c, rmC, eC) {
  const color = new THREE.CanvasTexture(c);
  color.colorSpace = THREE.SRGBColorSpace;
  const rmT = new THREE.CanvasTexture(rmC);
  const em = new THREE.CanvasTexture(eC);
  em.colorSpace = THREE.SRGBColorSpace;
  for (const t of [color, rmT, em]) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
  }
  return { color, rm: rmT, emissive: em };
}

// Ground-floor retail: shop windows, doors, signage bands. Covers 24 m x 4.5 m.
function storefront(imgs, rand) {
  const W = PX, H = Math.round((STOREFRONT_H / STOREFRONT_W) * PX);
  const s = PX / STOREFRONT_W; // px per meter
  const [c, g] = canvas(W, H);
  const [rmC, rm] = canvas(EPX, Math.round(H * K));
  const [eC, e] = canvas(EPX, Math.round(H * K));
  e.fillStyle = '#000';
  e.fillRect(0, 0, eC.width, eC.height);
  tile(g, imgs.concrete, W, H, 8 * s, 8 * s);
  g.fillStyle = 'rgba(60,58,55,0.55)';
  g.fillRect(0, 0, W, H);
  rm.fillStyle = 'rgb(0,200,0)';
  rm.fillRect(0, 0, rmC.width, rmC.height);

  const signColors = ['#1f3b57', '#7a1f1f', '#1f5a3a', '#2b2b2b', '#6b4b1f', '#3d2a5a', '#b8b0a0'];
  let x = 0;
  while (x < W - 1) {
    const unitBays = 2 + Math.floor(rand() * 3);
    const uw = Math.min(unitBays * BAY * s, W - x);
    const pier = 0.35 * s;
    const signH = 0.9 * s, plinth = 0.3 * s;
    const top = H - STOREFRONT_H * s + 0.4 * s; // leave a cornice above the sign
    // Sign band.
    g.fillStyle = signColors[Math.floor(rand() * signColors.length)];
    g.fillRect(x + pier, top, uw - pier * 2, signH);
    g.fillStyle = 'rgba(255,255,255,0.85)';
    const letters = 4 + Math.floor(rand() * 7);
    const lw = Math.min(0.35 * s, (uw - pier * 4) / (letters * 1.4));
    const lx0 = x + uw / 2 - (letters * lw * 1.4) / 2;
    for (let k = 0; k < letters; k++) g.fillRect(lx0 + k * lw * 1.4, top + signH * 0.3, lw, signH * 0.4);
    if (rand() < 0.6) {
      e.fillStyle = 'rgba(255,240,220,0.9)';
      for (let k = 0; k < letters; k++) e.fillRect((lx0 + k * lw * 1.4) * K, (top + signH * 0.3) * K, lw * K, signH * 0.4 * K);
    }
    // Shop window.
    const gy = top + signH + 0.1 * s, gh = H - plinth - gy;
    g.fillStyle = '#1d1f22';
    g.fillRect(x + pier, gy, uw - pier * 2, gh);
    glassFill(g, x + pier + 6, gy + 6, uw - pier * 2 - 12, gh - 12, rand, [55, 60, 62]);
    rm.fillStyle = 'rgb(0,15,110)';
    rm.fillRect((x + pier + 6) * K, (gy + 6) * K, (uw - pier * 2 - 12) * K, (gh - 12) * K);
    // Merchandise silhouettes / interior.
    for (let k = 0; k < 6; k++) {
      g.fillStyle = `rgba(${rand() * 120},${rand() * 120},${rand() * 120},0.25)`;
      const w = (0.3 + rand() * 0.8) * s, h = (0.4 + rand() * 1.4) * s;
      g.fillRect(x + pier + rand() * (uw - pier * 2 - w), H - plinth - h - 6, w, h);
    }
    // Door.
    const dw = 1.1 * s, dx = x + pier + (rand() < 0.5 ? 0.3 * s : uw - pier * 2 - dw - 0.3 * s);
    g.fillStyle = '#2a2c2f';
    g.fillRect(dx, gy + 0.2 * s, dw, H - plinth - gy - 0.2 * s);
    glassFill(g, dx + 8, gy + 0.2 * s + 8, dw - 16, H - plinth - gy - 0.2 * s - 16, rand, [40, 44, 46]);
    // Mullions.
    g.fillStyle = '#1d1f22';
    for (let mx = x + pier + 1.5 * s; mx < x + uw - pier - 0.5 * s; mx += 1.5 * s) g.fillRect(mx, gy, 5, gh);
    // Night glow.
    const grad = e.createLinearGradient(0, gy * K, 0, (H - plinth) * K);
    const lc = LIT[Math.floor(rand() * LIT.length)];
    grad.addColorStop(0, `rgba(${lc},0.9)`);
    grad.addColorStop(1, `rgba(${lc},0.6)`);
    e.fillStyle = grad;
    if (rand() < 0.8) e.fillRect((x + pier + 6) * K, (gy + 6) * K, (uw - pier * 2 - 12) * K, (gh - 12) * K);
    // Plinth.
    g.fillStyle = '#3b3936';
    g.fillRect(x, H - plinth, uw, plinth);
    x += uw;
  }
  grime(g, W, H, rand, 20);
  return finish(c, rmC, eC);
}

// ---------- ground-level city maps ----------

const MAP_PX = 2048;
export const worldToMap = (v) => ((v - CITY_MIN) / CITY_SIZE) * MAP_PX;

function mapTexture(c) {
  const t = new THREE.CanvasTexture(c);
  t.flipY = false; // row index == world z
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

export function makeDetailMap(draw) {
  const [c, g] = canvas(MAP_PX, MAP_PX);
  g.fillStyle = '#fff';
  g.fillRect(0, 0, MAP_PX, MAP_PX);
  draw(g, MAP_PX / CITY_SIZE);
  return mapTexture(c);
}

export function makePoolsMap(draw) {
  const [c, g] = canvas(1024, 1024);
  g.fillStyle = '#000';
  g.fillRect(0, 0, 1024, 1024);
  g.globalCompositeOperation = 'lighter';
  draw(g, 1024 / CITY_SIZE);
  return mapTexture(c);
}

// Leaf cluster: many small leaves with individual tone, on transparent background.
function leafTexture(rand) {
  const [c, g] = canvas(512, 512);
  for (let k = 0; k < 900; k++) {
    const r = Math.sqrt(rand()) * 230, a = rand() * Math.PI * 2;
    const x = 256 + Math.cos(a) * r, y = 256 + Math.sin(a) * r;
    const l = 10 + rand() * 12;
    const shade = 0.55 + rand() * 0.6 - (r / 230) * 0.15;
    g.save();
    g.translate(x, y);
    g.rotate(rand() * Math.PI * 2);
    g.fillStyle = `rgb(${Math.round(70 * shade)},${Math.round(118 * shade)},${Math.round(42 * shade)})`;
    g.beginPath();
    g.ellipse(0, 0, l, l * 0.45, 0, 0, Math.PI * 2);
    g.fill();
    g.strokeStyle = `rgba(20,40,10,0.35)`;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(-l, 0);
    g.lineTo(l, 0);
    g.stroke();
    g.restore();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

// ---------- the material library ----------

export async function createMaterials(renderer, rand) {
  const aniso = renderer.capabilities.getMaxAnisotropy();
  const [
    asphaltD, asphaltN, asphaltA,
    paveD, paveN, paveA,
    roofD, roofN,
    brickN, plasterN, concreteN,
    barkD,
    brickImg, plasterImg, concreteImg,
  ] = await Promise.all([
    loadTex('asphalt_02_diff.jpg', { srgb: true, meters: 3, aniso }),
    loadTex('asphalt_02_nor_gl.jpg', { meters: 3, aniso }),
    loadTex('asphalt_02_arm.jpg', { meters: 3, aniso }),
    loadTex('concrete_pavement_diff.jpg', { srgb: true, meters: 1.8, aniso }),
    loadTex('concrete_pavement_nor_gl.jpg', { meters: 1.8, aniso }),
    loadTex('concrete_pavement_arm.jpg', { meters: 1.8, aniso }),
    loadTex('concrete_floor_worn_02_diff.jpg', { srgb: true, meters: 2, aniso }),
    loadTex('concrete_floor_worn_02_nor_gl.jpg', { meters: 2, aniso }),
    loadTex('brick_wall_02_nor_gl.jpg', { meters: 2, aniso }),
    loadTex('clay_plaster_nor_gl.jpg', { meters: 2, aniso }),
    loadTex('concrete_panels_nor_gl.jpg', { meters: 8, aniso }),
    loadTex('bark_brown_02_diff.jpg', { srgb: true, meters: 1, aniso }),
    loadImage('brick_wall_02_diff.jpg'),
    loadImage('clay_plaster_diff.jpg'),
    loadImage('concrete_panels_diff.jpg'),
  ]);
  cityUniforms.uMacro.value = macroNoiseTexture();

  // Asphalt: ARM packs AO (R), roughness (G), metalness (B).
  const asphalt = patchGround(
    new THREE.MeshStandardMaterial({
      map: asphaltD, normalMap: asphaltN, roughnessMap: asphaltA, aoMap: asphaltA,
      color: 0xe6e6e6, roughness: 1, metalness: 0, normalScale: new THREE.Vector2(0.8, 0.8),
    }),
    { detail: true, antiTile: true },
  );
  const sidewalk = patchGround(
    new THREE.MeshStandardMaterial({ map: paveD, normalMap: paveN, roughnessMap: paveA, color: 0xd8d4cc, roughness: 1 }),
  );
  const curb = patchGround(
    new THREE.MeshStandardMaterial({ map: roofD, normalMap: roofN, color: 0xbdb9b0, roughness: 0.9 }),
  );
  const roof = new THREE.MeshStandardMaterial({ map: roofD, normalMap: roofN, color: 0x8a8782, roughness: 0.95 });

  const imgs = { brick: brickImg, plaster: plasterImg, concrete: concreteImg };
  const normals = { brick: brickN, plaster: plasterN, concrete: concreteN };
  const facades = {};
  for (const style of ['brick', 'plaster', 'concrete', 'glass']) {
    const f = facade(style, imgs, rand);
    const n = normals[style];
    facades[style] = new THREE.MeshStandardMaterial({
      map: f.color,
      roughnessMap: f.rm,
      metalnessMap: f.rm,
      emissiveMap: f.emissive,
      emissive: 0xffffff,
      emissiveIntensity: 0,
      normalMap: n ? n.clone() : null,
      normalScale: new THREE.Vector2(0.35, 0.35),
      roughness: 1,
      metalness: 1,
      vertexColors: true,
      envMapIntensity: style === 'glass' ? 1.3 : 1,
    });
    if (n) {
      // Normal map tiles at its real-world size; facade UVs are in 24 m x 28 m units.
      const nm = facades[style].normalMap;
      nm.repeat.set(FACADE_W / (style === 'concrete' ? 8 : 2), FACADE_H / (style === 'concrete' ? 8 : 2));
    }
  }
  const sf = storefront(imgs, rand);
  const shop = new THREE.MeshStandardMaterial({
    map: sf.color, roughnessMap: sf.rm, metalnessMap: sf.rm, emissiveMap: sf.emissive,
    emissive: 0xffffff, emissiveIntensity: 0, roughness: 1, metalness: 1,
  });

  const metal = new THREE.MeshStandardMaterial({ color: 0x5d6166, roughness: 0.55, metalness: 0.8 });
  const darkMetal = new THREE.MeshStandardMaterial({ color: 0x1c1e20, roughness: 0.6, metalness: 0.6 });
  const signalHousing = new THREE.MeshStandardMaterial({ color: 0x1a1a14, roughness: 0.7, metalness: 0.2 });
  const lamp = new THREE.MeshStandardMaterial({ color: 0x333333, emissive: 0xffd9a8, emissiveIntensity: 0, roughness: 0.3 });
  const bark = new THREE.MeshStandardMaterial({ map: barkD, color: 0x8a7a6a, roughness: 0.95 });
  const foliage = new THREE.MeshStandardMaterial({
    map: leafTexture(rand), alphaTest: 0.5, roughness: 0.8, color: 0xffffff,
  });
  const marking = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.65, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  });
  patchGround(marking, { detail: true });

  return { asphalt, sidewalk, curb, roof, facades, shop, metal, darkMetal, signalHousing, lamp, bark, foliage, marking };
}
