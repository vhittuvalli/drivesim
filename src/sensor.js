// The roof camera as a sensor: renders what the driving network sees (tone-mapped sRGB, no
// bloom/AO) into a small offscreen target, and optionally a label image of the same view with
// the semantic class id in R and depth in G (depth / DEPTH_RANGE), used as auxiliary targets.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CLASSES, DEPTH_RANGE } from './labels.js';

export const SENSOR = { width: 256, height: 128, fov: 70, height_m: 1.62 };

// Flat class color with view depth written to G. Works for instanced and skinned meshes.
function labelMaterial(id) {
  const m = new THREE.MeshBasicMaterial({ color: new THREE.Color(id / 255, 0, 0), fog: false, toneMapped: false });
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying float vViewDepth;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nvViewDepth = -mvPosition.z;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vViewDepth;')
      .replace('#include <dithering_fragment>', `#include <dithering_fragment>
      gl_FragColor = vec4(diffuseColor.r, clamp(vViewDepth / ${DEPTH_RANGE.toFixed(1)}, 0.0, 1.0), 0.0, 1.0);`);
  };
  m.customProgramCacheKey = () => 'drivesim-label';
  return m;
}

export class SensorRig {
  // classify(object) -> class index into CLASSES, or -1 to leave it out of the label image.
  constructor(renderer, scene, { width = SENSOR.width, height = SENSOR.height, fov = SENSOR.fov, classify = null } = {}) {
    Object.assign(this, { renderer, scene, width, height, classify });
    this.cam = new THREE.PerspectiveCamera(fov, width / height, 0.1, 2000);
    // Offscreen, tone mapping and sRGB conversion only happen in OutputPass.
    this.composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(width, height, { type: THREE.UnsignedByteType }));
    this.composer.renderToScreen = false;
    this.composer.setPixelRatio(1);
    this.composer.setSize(width, height);
    this.composer.addPass(new RenderPass(scene, this.cam));
    this.composer.addPass(new OutputPass());
    this.labelRT = new THREE.WebGLRenderTarget(width, height, { type: THREE.UnsignedByteType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.labelMats = CLASSES.map((_, i) => labelMaterial(i));
    this.pixels = new Uint8Array(width * height * 4);
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this._c = new THREE.Color();
  }

  // Mount the camera on the car roof, looking down the road.
  mount(pose) {
    const fx = Math.cos(pose.h), fz = Math.sin(pose.h);
    this.cam.position.set(pose.x + fx * 0.05, SENSOR.height_m, pose.z + fz * 0.05);
    this.cam.lookAt(pose.x + fx * 20, 1.2, pose.z + fz * 20);
  }

  // GL rows are bottom-up: copy into an ImageData top-down.
  read(rt) {
    const { width: w, height: h } = this;
    this.renderer.readRenderTargetPixels(rt, 0, 0, w, h, this.pixels);
    this.renderer.setRenderTarget(null);
    const img = new ImageData(w, h), row = w * 4;
    for (let y = 0; y < h; y++) img.data.set(this.pixels.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    return img;
  }

  renderRGB(pose) {
    this.mount(pose);
    this.composer.render();
    return this.read(this.composer.readBuffer);
  }

  renderLabels(pose) {
    this.mount(pose);
    const { scene, renderer } = this;
    const swapped = [], hidden = [];
    scene.traverse((o) => {
      if (!(o.isMesh || o.isLine || o.isPoints || o.isSprite) || !o.visible) return;
      const cls = o.isMesh ? this.classify(o) : -1;
      if (cls < 0) {
        hidden.push(o);
        o.visible = false;
        return;
      }
      swapped.push([o, o.material, o.instanceColor]);
      const m = this.labelMats[cls];
      o.material = Array.isArray(o.material) ? o.material.map(() => m) : m;
      if (o.isInstancedMesh) o.instanceColor = null; // would tint the class color
    });
    const { background, fog } = scene;
    const clear = renderer.getClearColor(this._c).clone(), alpha = renderer.getClearAlpha(), shadows = renderer.shadowMap.autoUpdate;
    scene.background = null;
    scene.fog = null;
    renderer.shadowMap.autoUpdate = false;
    renderer.setRenderTarget(this.labelRT);
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    renderer.render(scene, this.cam);
    const img = this.read(this.labelRT);
    renderer.setClearColor(clear, alpha);
    renderer.shadowMap.autoUpdate = shadows;
    scene.background = background;
    scene.fog = fog;
    for (const [o, m, ic] of swapped) {
      o.material = m;
      if (o.isInstancedMesh) o.instanceColor = ic;
    }
    for (const o of hidden) o.visible = true;
    return img;
  }

  encode(img, type = 'image/jpeg', quality = 0.92) {
    this.ctx.putImageData(img, 0, 0);
    return new Promise((res) => this.canvas.toBlob(res, type, quality));
  }
}

// Build the classifier for the label pass from the scene's known materials and groups.
export function makeClassifier({ mats, city, fleet, crowdGroup, egoMesh, sky }) {
  const id = (name) => CLASSES.indexOf(name);
  const byMat = new Map([
    [mats.asphalt, id('road')], [mats.marking, id('marking')], [mats.sidewalk, id('sidewalk')], [mats.curb, id('sidewalk')],
    [mats.roof, id('building')], [mats.shop, id('building')], ...Object.values(mats.facades).map((m) => [m, id('building')]),
    [mats.bark, id('vegetation')], [mats.foliage, id('vegetation')],
    [mats.metal, id('pole')], [mats.darkMetal, id('pole')], [mats.lamp, id('pole')], [mats.signalHousing, id('traffic light')],
  ]);
  const signalLamps = new Set(Object.values(city.lamps ?? {}));
  const under = (o, root) => {
    for (let p = o; p; p = p.parent) if (p === root) return true;
    return false;
  };
  return (o) => {
    if (o === sky) return id('sky');
    if (signalLamps.has(o)) return id('traffic light');
    const m = Array.isArray(o.material) ? o.material[0] : o.material;
    if (byMat.has(m)) return byMat.get(m);
    if (under(o, fleet.group)) return id('vehicle');
    if (under(o, crowdGroup)) return id('pedestrian');
    if (under(o, egoMesh)) return -1;
    if (under(o, city.group)) return id('terrain');
    return -1;
  };
}
