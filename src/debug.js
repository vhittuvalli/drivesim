// Planner debug overlay: what the ego expert is thinking, drawn into the scene.
//   cyan ribbon   planned path (bends out while overtaking), dot = pure-pursuit target
//   red box       the vehicle we are following / yielding to (orange: predicted crossing,
//                 with its predicted conflict point)
//   yellow rings  pedestrians on the road nearby (red: the one we're stopping for)
//   stop bar      next stop line in the signal's color
//   lane patch    oncoming-lane check for an overtake (green clear, red blocked, blue passing)
import * as THREE from 'three';
import { LANE_W } from './config.js';

const overlay = (m) => {
  m.depthTest = false;
  m.depthWrite = false;
  m.transparent = true;
  m.toneMapped = false;
  return m;
};
const flat = (mesh) => {
  mesh.renderOrder = 10;
  mesh.frustumCulled = false;
  return mesh;
};

const SIGNAL = { red: 0xff3b30, yellow: 0xffcc00, green: 0x34c759 };

// Lives in its own scene, drawn on top after post-processing (see render()).
export class DebugOverlay {
  constructor() {
    this.scene = new THREE.Scene();
    this.group = new THREE.Group();
    this.group.visible = false;
    this.scene.add(this.group);

    this.ribbon = flat(new THREE.Mesh(new THREE.BufferGeometry(), overlay(new THREE.MeshBasicMaterial({ color: 0x4db8ff, opacity: 0.35, side: THREE.DoubleSide }))));
    this.ribbonPos = new Float32Array(64 * 2 * 3);
    this.ribbon.geometry.setAttribute('position', new THREE.BufferAttribute(this.ribbonPos, 3));
    const idx = [];
    for (let k = 0; k < 63; k++) idx.push(2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2);
    this.ribbon.geometry.setIndex(idx);

    this.target = flat(new THREE.Mesh(new THREE.SphereGeometry(0.25, 12, 8), overlay(new THREE.MeshBasicMaterial({ color: 0x4db8ff }))));
    this.lead = flat(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), overlay(new THREE.LineBasicMaterial({ color: 0xff3b30 }))));
    this.predicted = flat(new THREE.Mesh(new THREE.SphereGeometry(0.45, 12, 8), overlay(new THREE.MeshBasicMaterial({ color: 0xff9500, opacity: 0.8 }))));
    this.stopBar = flat(new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), overlay(new THREE.MeshBasicMaterial({ opacity: 0.7, side: THREE.DoubleSide }))));
    this.zone = flat(new THREE.Mesh(new THREE.BufferGeometry(), overlay(new THREE.MeshBasicMaterial({ opacity: 0.28, side: THREE.DoubleSide }))));
    this.zonePos = new Float32Array(4 * 3);
    this.zone.geometry.setAttribute('position', new THREE.BufferAttribute(this.zonePos, 3));
    this.zone.geometry.setIndex([0, 1, 2, 1, 3, 2]);
    this.rings = [];
    const ringGeo = new THREE.RingGeometry(0.55, 0.75, 24).rotateX(-Math.PI / 2);
    for (let k = 0; k < 16; k++) {
      const r = flat(new THREE.Mesh(ringGeo, overlay(new THREE.MeshBasicMaterial({ color: 0xffcc00 }))));
      this.rings.push(r);
    }
    this.group.add(this.ribbon, this.target, this.lead, this.predicted, this.stopBar, this.zone, ...this.rings);
  }

  set visible(v) {
    this.group.visible = v;
  }

  get visible() {
    return this.group.visible;
  }

  render(renderer, camera) {
    if (!this.group.visible) return;
    const auto = renderer.autoClear;
    renderer.autoClear = false;
    renderer.render(this.scene, camera);
    renderer.autoClear = auto;
  }

  update(world) {
    if (!this.group.visible) return;
    const c = world.ctrl;
    for (const o of this.group.children) o.visible = false;
    if (!c || c.manual || !c.samples) return;
    const y = 0.12;

    // Path ribbon, as wide as the car.
    const S = c.samples;
    const n = Math.min(S.length, 64);
    for (let k = 0; k < 64; k++) {
      const q = S[Math.min(k, n - 1)], lx = Math.sin(q.h) * 0.9, lz = -Math.cos(q.h) * 0.9;
      this.ribbonPos.set([q.x + lx, y, q.z + lz, q.x - lx, y, q.z - lz], k * 6);
    }
    this.ribbon.geometry.attributes.position.needsUpdate = true;
    this.ribbon.geometry.computeBoundingSphere();
    this.ribbon.visible = true;
    this.target.position.set(c.target.x, 0.3, c.target.z);
    this.target.visible = true;

    // Lead / yield target.
    const a = c.lead?.agent;
    if (a && a.kind === 'car') {
      this.lead.position.set(a.x, 0.8, a.z);
      this.lead.rotation.y = -a.h;
      this.lead.scale.set(a.halfLen * 2 + 0.2, 1.7, 2.1);
      this.lead.material.color.set(c.lead.predicted ? 0xff9500 : 0xff3b30);
      this.lead.visible = true;
      if (c.lead.at) {
        this.predicted.position.set(c.lead.at.x, 0.5, c.lead.at.z);
        this.predicted.visible = true;
      }
    }

    // Pedestrians on the road near us.
    let r = 0;
    for (const p of world.crowd.agents) {
      if (r >= this.rings.length) break;
      if (!p.crossing || Math.hypot(p.x - world.car.x, p.z - world.car.z) > 45) continue;
      const ring = this.rings[r++];
      ring.position.set(p.x, 0.1, p.z);
      ring.material.color.set(p === a ? 0xff3b30 : 0xffcc00);
      ring.visible = true;
    }

    // Next stop line.
    const st = c.signal;
    if (st && st.dist < 90) {
      const p = c.route.at(st.s);
      this.stopBar.position.set(p.x, 0.1, p.z);
      this.stopBar.rotation.y = -p.h; // local x runs along the lane
      this.stopBar.scale.set(0.6, 1, LANE_W);
      this.stopBar.material.color.set(SIGNAL[st.state] ?? 0xffffff);
      this.stopBar.visible = true;
    }

    // Overtake check zone on the oncoming lane.
    const ot = c.overtake;
    if (ot) {
      const [s0, s1] = ot.zone, k = c.k;
      const pts = [c.route.at(s0, k), c.route.at(Math.max(s1, s0 + 1), k)];
      pts.forEach((p, i) => {
        for (const [j, off] of [[0, LANE_W * 0.5], [1, LANE_W * 1.5]]) {
          const lx = Math.sin(p.h) * off, lz = -Math.cos(p.h) * off;
          this.zonePos.set([p.x + lx, 0.1, p.z + lz], (i * 2 + j) * 3);
        }
      });
      this.zone.geometry.attributes.position.needsUpdate = true;
      this.zone.geometry.computeBoundingSphere();
      this.zone.material.color.set(ot.state !== 'waiting' ? 0x4db8ff : ot.clear ? 0x34c759 : 0xff3b30);
      this.zone.visible = true;
    }
  }
}
