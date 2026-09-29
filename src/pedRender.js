// Renders the crowd with an animated, skinned human model (Xbot from the three.js examples,
// loaded from the jsDelivr CDN). Each pedestrian gets its own clothing tint and walk cycle.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

const MODEL_URL = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r170/examples/models/gltf/Xbot.glb';
const WALK_CLIP_SPEED = 1.3; // m/s at timeScale 1
const ANIMATE_RADIUS = 140; // skip animation updates for far-away pedestrians

const CLOTHES = [0x2b2f36, 0x3b4a63, 0x5a3b35, 0x6c6f73, 0x1d2430, 0x7a5c3e, 0x8a8f96, 0x44523a, 0x9b2f2f, 0x2f5f7a, 0xc9c2b4, 0x151515];

export class CrowdRenderer {
  constructor(scene, rand) {
    this.scene = scene;
    this.rand = rand;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.views = new Map(); // ped id -> view
    this.ready = new GLTFLoader().loadAsync(MODEL_URL).then((gltf) => {
      this.gltf = gltf;
      gltf.scene.traverse((o) => {
        if (o.isMesh) {
          o.castShadow = true;
          o.frustumCulled = false;
        }
      });
    });
  }

  makeView(ped) {
    const root = SkeletonUtils.clone(this.gltf.scene);
    const clothes = new THREE.Color(CLOTHES[Math.floor(this.rand() * CLOTHES.length)]);
    root.traverse((o) => {
      if (!o.isMesh) return;
      // Xbot has a 'Surface' mesh (body) and a 'Joints' mesh; recolor both per pedestrian.
      o.material = new THREE.MeshStandardMaterial({
        color: /Joints/i.test(o.name) ? 0x1c1c1e : clothes,
        roughness: 0.8,
        metalness: 0,
      });
    });
    const s = 0.93 + this.rand() * 0.14;
    root.scale.setScalar(s);
    const mixer = new THREE.AnimationMixer(root);
    const clip = (name) => this.gltf.animations.find((a) => a.name === name);
    const walk = mixer.clipAction(clip('walk'));
    const idle = mixer.clipAction(clip('idle'));
    walk.play();
    idle.play();
    walk.time = this.rand() * walk.getClip().duration;
    this.group.add(root);
    return { root, mixer, walk, idle, blend: 1 };
  }

  sync(peds, dt, camera) {
    if (!this.gltf) return;
    const alive = new Set();
    for (const p of peds) {
      alive.add(p.id);
      let v = this.views.get(p.id);
      if (!v) {
        v = this.makeView(p);
        this.views.set(p.id, v);
      }
      v.root.position.set(p.x, 0.15, p.z);
      // Model faces +z; heading h points along (cos h, sin h).
      v.root.rotation.y = Math.PI / 2 - p.h;
      const moving = p.v > 0.05 ? 1 : 0;
      v.blend += (moving - v.blend) * Math.min(1, dt * 6);
      v.walk.setEffectiveWeight(v.blend);
      v.idle.setEffectiveWeight(1 - v.blend);
      v.walk.timeScale = Math.max(0.6, p.v / WALK_CLIP_SPEED);
      const near = camera.position.distanceToSquared(v.root.position) < ANIMATE_RADIUS * ANIMATE_RADIUS;
      v.root.visible = camera.position.distanceToSquared(v.root.position) < 400 * 400;
      if (near) v.mixer.update(dt);
    }
    for (const [id, v] of this.views) {
      if (alive.has(id)) continue;
      this.group.remove(v.root);
      v.root.traverse((o) => o.isMesh && o.material.dispose());
      this.views.delete(id);
    }
  }
}
