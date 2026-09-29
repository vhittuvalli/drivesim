// Weather visuals: sky/fog/lighting modifiers per preset and falling rain or snow around the
// camera. The driving effects (grip, visibility) live in sim.js (WEATHER / setWeather).
import * as THREE from 'three';

// sun: direct light scale; fog: fog density multiplier; haze: fog color blend toward gray;
// wet/snow: ground shader; lights: how much vehicles' lights come on in daytime.
export const LOOK = {
  clear: { sun: 1, fog: 1, haze: 0, turbidity: 4, env: 1, wet: 0, snow: 0, lights: 0 },
  rain: { sun: 0.22, fog: 3.2, haze: 0.8, turbidity: 14, env: 0.55, wet: 1, snow: 0, lights: 0.55 },
  fog: { sun: 0.35, fog: 11, haze: 1, turbidity: 16, env: 0.7, wet: 0.3, snow: 0, lights: 0.5 },
  snow: { sun: 0.4, fog: 4, haze: 0.9, turbidity: 12, env: 0.85, wet: 0, snow: 1, lights: 0.35 },
};
export const HAZE = new THREE.Color(0x9aa3ab);

const N = 9000, BOX = 70, TOP = 32;

export class Precipitation {
  constructor(scene) {
    this.kind = 'clear';
    this.drops = new Float32Array(N * 3); // world-space positions (wrapped around the camera)
    for (let i = 0; i < N; i++) this.drops.set([Math.random() * BOX, Math.random() * TOP, Math.random() * BOX], i * 3);
    this.phase = Float32Array.from({ length: N }, () => Math.random() * 6.28);

    this.rainPos = new Float32Array(N * 6);
    const rainGeo = new THREE.BufferGeometry();
    rainGeo.setAttribute('position', new THREE.BufferAttribute(this.rainPos, 3));
    this.rain = new THREE.LineSegments(rainGeo, new THREE.LineBasicMaterial({ color: 0xb8c4d0, transparent: true, opacity: 0.32, depthWrite: false }));

    this.snowPos = new Float32Array(N * 3);
    const snowGeo = new THREE.BufferGeometry();
    snowGeo.setAttribute('position', new THREE.BufferAttribute(this.snowPos, 3));
    this.snow = new THREE.Points(snowGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.09, transparent: true, opacity: 0.85, depthWrite: false }));

    for (const o of [this.rain, this.snow]) {
      o.frustumCulled = false;
      o.visible = false;
      scene.add(o);
    }
    this.t = 0;
  }

  set(kind) {
    this.kind = kind;
    this.rain.visible = kind === 'rain';
    this.snow.visible = kind === 'snow';
  }

  update(dt, camera) {
    if (this.kind !== 'rain' && this.kind !== 'snow') return;
    this.t += dt;
    const rain = this.kind === 'rain';
    const fall = rain ? 11 : 1.3, wind = rain ? 1.2 : 0.6;
    const cx = camera.position.x - BOX / 2, cz = camera.position.z - BOX / 2, cy = Math.max(0, camera.position.y - 12);
    const D = this.drops, out = rain ? this.rainPos : this.snowPos;
    for (let i = 0; i < N; i++) {
      const j = i * 3;
      let y = D[j + 1] - fall * dt;
      if (y < 0) y += TOP;
      D[j + 1] = y;
      D[j] += wind * dt + (rain ? 0 : Math.sin(this.t * 1.3 + this.phase[i]) * 0.4 * dt);
      // Wrap into the box centered on the camera.
      const x = cx + ((((D[j] - cx) % BOX) + BOX) % BOX);
      const z = cz + ((((D[j + 2] - cz) % BOX) + BOX) % BOX);
      if (rain) {
        out[i * 6] = x;
        out[i * 6 + 1] = cy + y;
        out[i * 6 + 2] = z;
        out[i * 6 + 3] = x - wind * 0.05;
        out[i * 6 + 4] = cy + y + 0.55;
        out[i * 6 + 5] = z;
      } else {
        out[j] = x;
        out[j + 1] = cy + y;
        out[j + 2] = z;
      }
    }
    (rain ? this.rain : this.snow).geometry.attributes.position.needsUpdate = true;
  }
}
