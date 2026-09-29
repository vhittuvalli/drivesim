// Ego vehicle: kinematic bicycle model + a procedurally modeled sedan with an AV sensor rig.
// Physics heading h: forward = (cos h, sin h) in (x, z). Positive steer turns right.
import * as THREE from 'three';
import { clamp, conditions, WHEELBASE, MAX_STEER, MAX_SPEED } from './config.js';
const WHEEL_R = 0.34;

export class Vehicle {
  constructor(x, z, h) {
    this.x = x;
    this.z = z;
    this.h = h;
    this.v = 0;
    this.steer = 0;
    this.accel = 0;
    this.wheelSpin = 0;
    this.mesh = buildSedan();
  }

  step(dt, steerCmd, throttleCmd) {
    const target = clamp(steerCmd, -1, 1) * MAX_STEER;
    const maxRate = 1.6 * dt;
    this.steer += clamp(target - this.steer, -maxRate, maxRate);
    const t = clamp(throttleCmd, -1, 1);
    this.accel = t >= 0 ? 3.2 * t * Math.min(1, conditions.grip * 1.3) : 7.5 * t * conditions.grip;
    const drag = 0.0025 * this.v * this.v + (this.v > 0 ? 0.08 : 0);
    this.v = clamp(this.v + (this.accel - drag) * dt, 0, MAX_SPEED);
    this.h += (this.v / WHEELBASE) * Math.tan(this.steer) * dt;
    this.x += this.v * Math.cos(this.h) * dt;
    this.z += this.v * Math.sin(this.h) * dt;
    this.wheelSpin -= (this.v * dt) / WHEEL_R;
  }

  // Teleport (scenario setup, respawn); keeps the mesh.
  reset(x, z, h, v = 0) {
    Object.assign(this, { x, z, h, v, steer: 0, accel: 0 });
  }

  // `lights` is how dark it is for headlight purposes (night, rain, fog).
  syncMesh(night, lights = night) {
    const m = this.mesh;
    m.position.set(this.x, 0, this.z);
    m.rotation.y = -this.h;
    // Body pitch/roll from acceleration and cornering.
    const latAcc = (this.v * this.v / WHEELBASE) * Math.tan(this.steer);
    m.userData.body.rotation.z = THREE.MathUtils.lerp(m.userData.body.rotation.z, clamp(-this.accel * 0.004, -0.03, 0.03), 0.1);
    m.userData.body.rotation.x = THREE.MathUtils.lerp(m.userData.body.rotation.x, clamp(latAcc * 0.004, -0.04, 0.04), 0.1);
    for (const w of m.userData.wheels) {
      w.spin.rotation.z = this.wheelSpin;
      if (w.front) w.pivot.rotation.y = -this.steer;
    }
    const braking = this.accel < -0.5;
    m.userData.tail.emissiveIntensity = braking ? 6 : 0.6 + lights * 1.2;
    m.userData.head.emissiveIntensity = 0.3 + lights * 8;
    m.userData.headLight.intensity = lights * 40;
  }
}

function buildSedan() {
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);

  const paint = new THREE.MeshPhysicalMaterial({ color: 0x2e3a48, metalness: 0.6, roughness: 0.38, clearcoat: 1, clearcoatRoughness: 0.03 });
  const glass = new THREE.MeshPhysicalMaterial({ color: 0x0a0e12, metalness: 0.2, roughness: 0.04, clearcoat: 1, envMapIntensity: 1.5 });
  const trim = new THREE.MeshStandardMaterial({ color: 0x111214, roughness: 0.5, metalness: 0.3 });
  const chrome = new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.15, metalness: 1 });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.9 });
  const head = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff4e0, emissiveIntensity: 0.3, roughness: 0.1 });
  const tail = new THREE.MeshStandardMaterial({ color: 0x440000, emissive: 0xff1a0a, emissiveIntensity: 0.6, roughness: 0.2 });
  const sensor = new THREE.MeshStandardMaterial({ color: 0x1b1d20, roughness: 0.3, metalness: 0.5 });

  const shadowed = (mesh) => { mesh.castShadow = true; mesh.receiveShadow = true; return mesh; };
  const extrude = (pts, width, mat, bevel = 0.06) => {
    const s = new THREE.Shape(pts.map(([x, y]) => new THREE.Vector2(x, y)));
    const g = new THREE.ExtrudeGeometry(s, { depth: width, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 4, curveSegments: 16 });
    g.translate(0, 0, -width / 2);
    return shadowed(new THREE.Mesh(g, mat));
  };

  // Lower body profile (x forward, y up) with wheel arches.
  const lower = new THREE.Shape();
  lower.moveTo(-2.3, 0.36);
  lower.lineTo(-1.85, 0.36);
  lower.absarc(-1.42, 0.36, 0.43, Math.PI, 0, true);
  lower.lineTo(0.97, 0.36);
  lower.absarc(1.4, 0.36, 0.43, Math.PI, 0, true);
  lower.lineTo(2.2, 0.36);
  lower.quadraticCurveTo(2.42, 0.4, 2.4, 0.62);
  lower.quadraticCurveTo(2.36, 0.78, 2.1, 0.82);
  lower.lineTo(0.95, 0.98);
  lower.lineTo(-1.7, 1.02);
  lower.quadraticCurveTo(-2.3, 1.0, -2.36, 0.9);
  lower.quadraticCurveTo(-2.42, 0.6, -2.3, 0.36);
  const lowerGeo = new THREE.ExtrudeGeometry(lower, { depth: 1.7, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.07, bevelSegments: 5, curveSegments: 20 });
  lowerGeo.translate(0, 0, -0.85);
  body.add(shadowed(new THREE.Mesh(lowerGeo, paint)));

  // Greenhouse (all glass) and roof skin.
  body.add(extrude([[0.98, 0.96], [0.15, 1.43], [-0.95, 1.45], [-1.78, 1.0]], 1.5, glass, 0.05));
  body.add(extrude([[0.12, 1.43], [-0.92, 1.45], [-0.95, 1.475], [0.1, 1.46]], 1.45, paint, 0.04));
  for (const x of [-0.38]) {
    const pillar = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.44, 1.61), paint));
    pillar.position.set(x, 1.22, 0);
    body.add(pillar);
  }

  // Lights, grille, mirrors, plate.
  for (const z of [-0.62, 0.62]) {
    const hl = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.09, 0.42), head);
    hl.position.set(2.3, 0.72, z);
    hl.rotation.y = z > 0 ? -0.25 : 0.25;
    const tl = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.1, 0.5), tail);
    tl.position.set(-2.46, 0.86, z * 1.05);
    const mirror = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.12, 0.2), paint));
    mirror.position.set(0.8, 1.05, z * 1.55);
    body.add(hl, tl, mirror);
  }
  const lightBar = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.035, 1.3), tail);
  lightBar.position.set(-2.475, 0.88, 0);
  const diffuser = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.14, 1.5), trim);
  diffuser.position.set(-2.42, 0.45, 0);
  body.add(lightBar, diffuser);
  const grille = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.16, 0.9), trim);
  grille.position.set(2.39, 0.55, 0);
  const plate = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.12, 0.5), new THREE.MeshStandardMaterial({ color: 0xf2f2ea, roughness: 0.5 }));
  plate.position.set(-2.5, 0.64, 0);
  const sill = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.08, 1.9), trim);
  sill.position.set(0, 0.4, 0);
  body.add(grille, plate, sill);

  // AV sensor rig: roof rack, spinning lidar, camera pods.
  const rack = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.06, 1.2), sensor));
  rack.position.set(-0.4, 1.53, 0);
  const lidarBase = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.15, 0.1, 24), sensor));
  lidarBase.position.set(-0.4, 1.61, 0);
  const lidar = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.13, 24), new THREE.MeshPhysicalMaterial({ color: 0x0c0c0e, roughness: 0.05, metalness: 0.3, clearcoat: 1 })));
  lidar.position.set(-0.4, 1.72, 0);
  body.add(rack, lidarBase, lidar);
  for (const z of [-0.55, 0, 0.55]) {
    const cam = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.08, 0.1), sensor));
    cam.position.set(0.02, 1.6, z);
    body.add(cam);
  }

  // Wheels: tire + multi-spoke rim, front wheels on steering pivots.
  const wheels = [];
  const tireGeo = new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, 0.24, 32);
  tireGeo.rotateX(Math.PI / 2);
  const rimGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.245, 32);
  rimGeo.rotateX(Math.PI / 2);
  const spokeGeo = new THREE.BoxGeometry(0.05, 0.4, 0.03);
  for (const [x, z, front] of [[1.4, 0.83, true], [1.4, -0.83, true], [-1.42, 0.83, false], [-1.42, -0.83, false]]) {
    const pivot = new THREE.Group();
    pivot.position.set(x, WHEEL_R, z);
    const spin = new THREE.Group();
    spin.add(shadowed(new THREE.Mesh(tireGeo, rubber)));
    spin.add(new THREE.Mesh(rimGeo, trim));
    for (let k = 0; k < 5; k++) {
      const sp = new THREE.Mesh(spokeGeo, chrome);
      sp.rotation.z = (k / 5) * Math.PI;
      sp.position.z = z > 0 ? 0.125 : -0.125;
      spin.add(sp);
    }
    pivot.add(spin);
    root.add(pivot);
    wheels.push({ pivot, spin, front });
  }

  // Soft contact shadow (ambient occlusion under the car).
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 10, 64, 64, 64);
  grad.addColorStop(0, 'rgba(0,0,0,0.75)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const ao = new THREE.Mesh(new THREE.PlaneGeometry(5.6, 2.8), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  ao.rotation.x = -Math.PI / 2;
  ao.position.y = 0.02;
  root.add(ao);

  // Headlight beam for night driving.
  const headLight = new THREE.SpotLight(0xfff1dd, 0, 60, 0.5, 0.45, 1.6);
  headLight.position.set(2.2, 0.75, 0);
  headLight.target.position.set(14, 0, 0);
  root.add(headLight, headLight.target);

  root.userData = { body, wheels, tail, head, headLight };
  return root;
}
