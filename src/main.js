import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { mulberry32, clamp } from './config.js';
import { createMaterials, cityUniforms } from './materials.js';
import { City } from './city.js';
import { Signals } from './signals.js';
import { Vehicle } from './vehicle.js';
import { Route, Expert } from './planner.js';
import { Fleet } from './fleet.js';
import { placeParkedCars, Traffic } from './traffic.js';
import { Crowd } from './peds.js';
import { CrowdRenderer } from './pedRender.js';

const $ = (id) => document.getElementById(id);
const DT = 1 / 60;
const params = new URLSearchParams(location.search);
const seed = Number(params.get('seed')) || ((Math.random() * 1e9) | 0);
const rand = mulberry32(seed);

const settings = {
  hour: Number(params.get('hour') ?? 14.5),
  cam: params.get('cam') ?? 'chase',
  speed: 1,
  cars: Number(params.get('cars') ?? 40),
  peds: Number(params.get('peds') ?? 70),
  paused: false,
  shadows: true,
  bloom: true,
};

// ---------- renderer ----------
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 4000);

const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
composer.addPass(new RenderPass(scene, camera));
const gtao = new GTAOPass(scene, camera, 1, 1);
gtao.updateGtaoMaterial({ radius: 1.2, distanceExponent: 1.5, thickness: 1, scale: 1.1, samples: 12 });
composer.addPass(gtao);
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.25, 0.5, 0.92);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------- sky, sun, environment ----------
const sky = new Sky();
sky.scale.setScalar(3000);
scene.add(sky);
const skyU = sky.material.uniforms;
skyU.turbidity.value = 4;
skyU.rayleigh.value = 1.2;
skyU.mieCoefficient.value = 0.004;
skyU.mieDirectionalG.value = 0.85;

const envScene = new THREE.Scene();
envScene.add(new THREE.Mesh(sky.geometry, sky.material));
const pmrem = new THREE.PMREMGenerator(renderer);
let envRT = null;

const sun = new THREE.DirectionalLight(0xffffff, 3);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
const SH = 110;
Object.assign(sun.shadow.camera, { left: -SH, right: SH, top: SH, bottom: -SH, near: 1, far: 900 });
sun.shadow.camera.updateProjectionMatrix();
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.04;
scene.add(sun, sun.target);
const moon = new THREE.HemisphereLight(0x5a6d99, 0x1a1a22, 0);
scene.add(moon);
scene.fog = new THREE.FogExp2(0xc8d2dc, 0.0022);

const sunDir = new THREE.Vector3();
let night = 0;

function applyTimeOfDay(hour) {
  // Sun path: rises ~6:00 in the east (+x), sets ~18:00 in the west, peaks at 62 degrees.
  const dayT = (hour - 6) / 12;
  const elev = Math.sin(dayT * Math.PI) * 62;
  const azim = 100 + dayT * 160;
  const phi = THREE.MathUtils.degToRad(90 - elev), theta = THREE.MathUtils.degToRad(azim);
  sunDir.setFromSphericalCoords(1, phi, theta);
  skyU.sunPosition.value.copy(sunDir);

  night = clamp((2 - elev) / 8, 0, 1);
  const golden = clamp(1 - Math.abs(elev - 6) / 14, 0, 1);
  sun.intensity = clamp(elev / 10, 0, 1) * 7.5;
  sun.color.setHSL(0.09, 0.4 + golden * 0.5, 0.85 - golden * 0.12);
  moon.intensity = night * 0.9;
  scene.environmentIntensity = THREE.MathUtils.lerp(0.5, 0.06, night);
  renderer.toneMappingExposure = THREE.MathUtils.lerp(1.0, 1.5, night);

  const fogDay = new THREE.Color(0xbfcad6), fogGold = new THREE.Color(0xd8b99a), fogNight = new THREE.Color(0x0b0e15);
  scene.fog.color.copy(fogDay).lerp(fogGold, golden * 0.6).lerp(fogNight, night);
  scene.fog.density = 0.0022 + night * 0.001;

  cityUniforms.uNight.value = night;
  if (mats) {
    for (const m of Object.values(mats.facades)) m.emissiveIntensity = 0.02 + night * 0.7;
    mats.shop.emissiveIntensity = 0.05 + night * 0.8;
    mats.lamp.emissiveIntensity = night * 6;
  }
  fleet?.setNight(night);
  bloom.strength = 0.15 + night * 0.3;
  bloom.threshold = night > 0.3 ? 0.85 : 0.95;

  // Night sky: the Sky shader goes black below the horizon; tint it deep blue.
  scene.background = night > 0.95 ? new THREE.Color(0x05070d) : null;
  sky.visible = night <= 0.95;

  if (envRT) envRT.dispose();
  envRT = pmrem.fromScene(envScene, 0.02);
  scene.environment = envRT.texture;
}

// ---------- world ----------
let mats, city, signals, car, route, expert, fleet, parked, traffic, crowd, crowdView;

async function init() {
  setLoading('Loading photo-scanned textures…');
  mats = await createMaterials(renderer, rand);
  setLoading('Generating city…');
  await new Promise((r) => setTimeout(r, 0));
  city = new City(mats, rand);
  scene.add(city.group);
  signals = new Signals(rand);
  fleet = new Fleet(260);
  scene.add(fleet.group);
  parked = placeParkedCars(city, fleet, rand);
  city.setSignalColors((n, a) => signals.state(n, a));

  route = new Route(rand);
  const p0 = route.pts[0], p1 = route.pts[1];
  car = new Vehicle(p0.x, p0.z, Math.atan2(p1.z - p0.z, p1.x - p0.x));
  scene.add(car.mesh);
  expert = new Expert(route, signals);
  expert.track(car);
  traffic = new Traffic(fleet, signals, rand);
  traffic.setCount(settings.cars, expert.agent);
  crowd = new Crowd(signals, rand);
  crowd.setCount(settings.peds);
  crowdView = new CrowdRenderer(scene, rand);
  setLoading('Loading pedestrians…');
  await crowdView.ready.catch((e) => console.warn('Pedestrian model failed to load', e));

  applyTimeOfDay(settings.hour);
  $('seed').textContent = seed;
  $('hour').value = settings.hour;
  setCam(settings.cam);
  resize();
  setLoading(null);
  requestAnimationFrame(frame);
}

// ---------- cameras ----------
const orbit = new OrbitControls(camera, canvas);
orbit.enabled = false;
orbit.enableDamping = true;
orbit.maxPolarAngle = Math.PI / 2 - 0.02;
const camPos = new THREE.Vector3(), camLook = new THREE.Vector3();
let lastCar = new THREE.Vector3();

function setCam(mode) {
  settings.cam = mode;
  orbit.enabled = mode === 'orbit';
  camera.fov = mode === 'hood' ? 70 : 55;
  camera.updateProjectionMatrix();
  document.querySelectorAll('[data-cam]').forEach((b) => b.classList.toggle('active', b.dataset.cam === mode));
  if (mode === 'orbit') {
    orbit.target.set(car.x, 1, car.z);
    camera.position.set(car.x - 25, 18, car.z + 25);
  }
  lastCar.set(car.x, 0, car.z);
  camPos.set(NaN, 0, 0);
}

function updateCamera(dtReal) {
  const fx = Math.cos(car.h), fz = Math.sin(car.h);
  if (settings.cam === 'chase') {
    const want = new THREE.Vector3(car.x - fx * 8.5, 3.2, car.z - fz * 8.5);
    if (Number.isNaN(camPos.x)) camPos.copy(want);
    camPos.lerp(want, 1 - Math.exp(-dtReal * 4));
    camera.position.copy(camPos);
    camLook.set(car.x + fx * 6, 1.2, car.z + fz * 6);
    camera.lookAt(camLook);
  } else if (settings.cam === 'hood') {
    // Roof-mounted forward camera: this is the viewpoint the driving network will use.
    camera.position.set(car.x + fx * 0.05, 1.62, car.z + fz * 0.05);
    camera.lookAt(car.x + fx * 20, 1.2, car.z + fz * 20);
  } else if (settings.cam === 'top') {
    camera.position.set(car.x - fx * 0.01, 70, car.z - fz * 0.01);
    camera.lookAt(car.x, 0, car.z);
  } else if (settings.cam === 'orbit') {
    const moved = new THREE.Vector3(car.x, 0, car.z).sub(lastCar);
    camera.position.add(moved);
    orbit.target.add(moved);
    orbit.update();
  }
  lastCar.set(car.x, 0, car.z);

  // Shadow frustum follows the camera focus, snapped to texels to avoid shimmering.
  const texel = (2 * SH) / sun.shadow.mapSize.x;
  const fxs = Math.round((car.x + fx * 30) / texel) * texel, fzs = Math.round((car.z + fz * 30) / texel) * texel;
  sun.target.position.set(fxs, 0, fzs);
  sun.position.set(fxs + sunDir.x * 400, sunDir.y * 400, fzs + sunDir.z * 400);
}

// ---------- loop ----------
let last = performance.now(), acc = 0, fpsAcc = 0, fpsN = 0, lastCtrl = null, lastSigT = 0;

function step() {
  signals.update(DT);
  const vehicles = [expert.agent, ...traffic.agents];
  const agents = [...vehicles, ...crowd.agents];
  const c = expert.control(car, agents);
  car.step(DT, c.steer, c.throttle);
  traffic.step(DT, agents, expert.agent);
  crowd.step(DT, vehicles);
  lastCtrl = c;
}

function frame(now) {
  const dtReal = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (!settings.paused) {
    acc += dtReal * settings.speed;
    let n = 0;
    while (acc >= DT && n++ < 40) {
      step();
      acc -= DT;
    }
  }
  if (now - lastSigT > 100) {
    city.setSignalColors((n, a) => signals.state(n, a));
    lastSigT = now;
  }
  car.syncMesh(night);
  traffic.sync();
  crowdView.sync(crowd.peds, settings.paused ? 0 : dtReal * settings.speed, camera);
  updateCamera(dtReal);
  if (settings.bloom) composer.render();
  else renderer.render(scene, camera);

  fpsAcc += dtReal;
  fpsN++;
  if (fpsAcc > 0.5) {
    $('fps').textContent = `${Math.round(fpsN / fpsAcc)} fps`;
    fpsAcc = fpsN = 0;
    updateHud();
  }
  requestAnimationFrame(frame);
}

function updateHud() {
  $('speed').textContent = `${Math.round(car.v * 3.6)}`;
  const c = lastCtrl;
  if (!c) return;
  const turn = c.nextTurn;
  $('maneuver').textContent = turn ? `${{ left: '↰ Left', right: '↱ Right', straight: '↑ Straight' }[turn.kind]} in ${Math.max(0, Math.round(turn.dist))} m` : '';
  const sig = c.signal;
  const el = $('signal');
  if (sig && sig.dist < 90) {
    el.hidden = false;
    el.dataset.state = sig.state;
    $('signal-text').textContent = `${sig.state} · ${Math.max(0, Math.round(sig.dist))} m`;
  } else el.hidden = true;
  const why = { vehicle: 'Following vehicle', pedestrian: 'Yielding to pedestrian', yield: 'Yielding to oncoming traffic', box: 'Waiting for space past the intersection', signal: null }[c.reason];
  $('reason').hidden = !why;
  if (why) $('reason').textContent = c.lead && c.reason !== 'yield' ? `${why} · ${Math.round(c.lead.gap)} m` : why;
  $('traffic-count').textContent = traffic.cars.length;
  $('ped-count').textContent = crowd.peds.length;
  const h = settings.hour;
  $('clock').textContent = `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`;
}

// ---------- UI ----------
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

function setLoading(msg) {
  $('loading').hidden = !msg;
  if (msg) $('loading-text').textContent = msg;
}

document.querySelectorAll('[data-cam]').forEach((b) => b.addEventListener('click', () => setCam(b.dataset.cam)));
let todTimer = null;
$('hour').addEventListener('input', (e) => {
  settings.hour = +e.target.value;
  clearTimeout(todTimer);
  todTimer = setTimeout(() => applyTimeOfDay(settings.hour), 30);
  updateHud();
});
$('cars').value = settings.cars;
$('cars').addEventListener('input', (e) => {
  settings.cars = +e.target.value;
  traffic.setCount(settings.cars, expert.agent);
});
$('peds').value = settings.peds;
$('peds').addEventListener('input', (e) => {
  settings.peds = +e.target.value;
  crowd.setCount(settings.peds);
});
$('sim-speed').addEventListener('change', (e) => (settings.speed = +e.target.value));
$('btn-pause').addEventListener('click', () => {
  settings.paused = !settings.paused;
  $('btn-pause').textContent = settings.paused ? 'Resume' : 'Pause';
});
$('btn-city').addEventListener('click', () => {
  params.set('seed', (Math.random() * 1e9) | 0);
  location.search = params.toString();
});
$('chk-shadows').addEventListener('change', (e) => {
  renderer.shadowMap.enabled = e.target.checked;
  scene.traverse((o) => o.material && (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => (m.needsUpdate = true)));
});
$('chk-bloom').addEventListener('change', (e) => (settings.bloom = e.target.checked));
window.addEventListener('keydown', (e) => {
  const map = { 1: 'chase', 2: 'hood', 3: 'orbit', 4: 'top' };
  if (map[e.key]) setCam(map[e.key]);
  if (e.key === ' ') $('btn-pause').click();
});

window.__dbg = { scene, sun, renderer, camera, cityUniforms, get orbit() { return orbit; }, get crowd() { return crowd; } };
init().catch((err) => {
  console.error(err);
  setLoading(`Failed to start: ${err.message}`);
});
