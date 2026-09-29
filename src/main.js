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
import { Vehicle } from './vehicle.js';
import { Fleet } from './fleet.js';
import { placeParkedCars } from './traffic.js';
import { CrowdRenderer } from './pedRender.js';
import { World, WEATHER, setWeather } from './sim.js';
import { SCENARIOS } from './scenarios.js';
import { DriverInput, DRIVE_KEYS } from './input.js';
import { DebugOverlay } from './debug.js';
import { LOOK, HAZE, Precipitation } from './weather.js';
import { Recorder } from './recorder.js';

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
  vans: Number(params.get('vans') ?? 4), // double-parked delivery vans
  weather: WEATHER[params.get('weather')] ? params.get('weather') : 'clear',
  scenario: SCENARIOS[params.get('scenario')] ? params.get('scenario') : '',
  debug: params.get('debug') === '1',
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
let night = 0, lightsOn = 0;

function applyTimeOfDay(hour) {
  // Sun path: rises ~6:00 in the east (+x), sets ~18:00 in the west, peaks at 62 degrees.
  const dayT = (hour - 6) / 12;
  const elev = Math.sin(dayT * Math.PI) * 62;
  const azim = 100 + dayT * 160;
  const phi = THREE.MathUtils.degToRad(90 - elev), theta = THREE.MathUtils.degToRad(azim);
  sunDir.setFromSphericalCoords(1, phi, theta);
  skyU.sunPosition.value.copy(sunDir);

  const look = LOOK[settings.weather];
  night = clamp((2 - elev) / 8, 0, 1);
  lightsOn = Math.max(night, look.lights);
  const golden = clamp(1 - Math.abs(elev - 6) / 14, 0, 1) * look.sun;
  sun.intensity = clamp(elev / 10, 0, 1) * 7.5 * look.sun;
  sun.color.setHSL(0.09, 0.4 + golden * 0.5, 0.85 - golden * 0.12);
  moon.intensity = night * 0.9;
  skyU.turbidity.value = look.turbidity;
  scene.environmentIntensity = THREE.MathUtils.lerp(0.5, 0.06, night) * look.env;
  renderer.toneMappingExposure = THREE.MathUtils.lerp(1.0, 1.5, night);

  const fogDay = new THREE.Color(0xbfcad6), fogGold = new THREE.Color(0xd8b99a), fogNight = new THREE.Color(0x0b0e15);
  scene.fog.color.copy(fogDay).lerp(fogGold, golden * 0.6).lerp(fogNight, night);
  scene.fog.color.lerp(HAZE.clone().multiplyScalar(THREE.MathUtils.lerp(1, 0.1, night)), look.haze);
  scene.fog.density = (0.0022 + night * 0.001) * look.fog;

  cityUniforms.uNight.value = night;
  cityUniforms.uWet.value = look.wet;
  cityUniforms.uSnow.value = look.snow;
  if (mats) {
    for (const m of Object.values(mats.facades)) m.emissiveIntensity = 0.02 + night * 0.7;
    mats.shop.emissiveIntensity = 0.05 + night * 0.8;
    mats.lamp.emissiveIntensity = night * 6;
  }
  fleet?.setNight(night);
  bloom.strength = 0.15 + night * 0.3;
  bloom.threshold = night > 0.3 ? 0.85 : 0.95;

  // Night sky: the Sky shader goes black below the horizon; tint it deep blue.
  // Overcast weather: a flat sky the color of the fog.
  const overcast = look.haze > 0.5;
  scene.background = overcast ? scene.fog.color.clone() : night > 0.95 ? new THREE.Color(0x05070d) : null;
  sky.visible = !overcast && night <= 0.95;

  if (envRT) envRT.dispose();
  envRT = pmrem.fromScene(envScene, 0.02);
  scene.environment = envRT.texture;
}

// ---------- world ----------
let mats, city, car, fleet, world, crowdView, debugView, precip, recorder;
const driver = new DriverInput();

async function init() {
  setLoading('Loading photo-scanned textures…');
  mats = await createMaterials(renderer, rand);
  setLoading('Generating city…');
  await new Promise((r) => setTimeout(r, 0));
  city = new City(mats, rand);
  scene.add(city.group);
  fleet = new Fleet(260);
  scene.add(fleet.group);
  const parked = placeParkedCars(city, fleet, rand);

  car = new Vehicle(0, 0, 0);
  scene.add(car.mesh);
  setWeather(settings.weather);
  world = new World({ rand, car, fleet, parked, cars: settings.cars, peds: settings.peds, doubleParked: settings.vans });
  city.setSignalColors((n, a) => world.signals.state(n, a));
  crowdView = new CrowdRenderer(scene, rand);
  debugView = new DebugOverlay();
  precip = new Precipitation(scene);
  recorder = new Recorder(renderer, scene);
  setLoading('Loading pedestrians…');
  await crowdView.ready.catch((e) => console.warn('Pedestrian model failed to load', e));

  setWeatherUI(settings.weather);
  setDebug(settings.debug);
  if (settings.scenario) startScenario(settings.scenario);
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
let last = performance.now(), acc = 0, fpsAcc = 0, fpsN = 0, lastSigT = 0, seenContacts = 0;

function frame(now) {
  const dtReal = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (!settings.paused) {
    acc += dtReal * settings.speed;
    let n = 0;
    while (acc >= DT && n++ < 40) {
      world.step(DT);
      acc -= DT;
    }
  }
  if (now - lastSigT > 100) {
    city.setSignalColors((n, a) => world.signals.state(n, a));
    lastSigT = now;
  }
  const simDt = settings.paused ? 0 : dtReal * settings.speed;
  car.syncMesh(night, lightsOn);
  world.traffic.sync();
  crowdView.sync(world.crowd.peds, simDt, camera);
  updateCamera(dtReal);
  debugView.update(world);
  precip.update(simDt, camera);
  if (settings.bloom) composer.render();
  else renderer.render(scene, camera);
  debugView.render(renderer, camera);
  if (recorder.active && world.ctrl) {
    recorder.capture(world.t, car, world.ctrl, { weather: settings.weather, hour: settings.hour });
    $('rec-count').textContent = recorder.count;
  }
  if (world.contacts > seenContacts && !world.scenario) toast(`Contact with ${world.lastContact.what}!`);
  seenContacts = world.contacts;

  fpsAcc += dtReal;
  fpsN++;
  if (fpsAcc > 0.5) {
    $('fps').textContent = `${Math.round(fpsN / fpsAcc)} fps`;
    fpsAcc = fpsN = 0;
    updateHud();
  }
  requestAnimationFrame(frame);
}

const OT_TEXT = { passing: '⇠ Overtaking', returning: '⇢ Merging back', aborting: '↩ Aborting pass' };

function updateHud() {
  $('speed').textContent = `${Math.round(car.v * 3.6)}`;
  updateScenarioCard();
  const c = world.ctrl;
  if (!c) return;
  const turn = c.nextTurn, ot = c.overtake;
  $('maneuver').textContent = c.manual
    ? `Manual · ${driver.source === 'gamepad' ? 'gamepad' : 'WASD / arrows'}`
    : OT_TEXT[ot?.state] ?? (turn ? `${{ left: '↰ Left', right: '↱ Right', straight: '↑ Straight' }[turn.kind]} in ${Math.max(0, Math.round(turn.dist))} m` : '');
  const sig = c.signal;
  const el = $('signal');
  if (sig && sig.dist < 90) {
    el.hidden = false;
    el.dataset.state = sig.state;
    $('signal-text').textContent = `${sig.state} · ${Math.max(0, Math.round(sig.dist))} m`;
  } else el.hidden = true;
  const why = {
    vehicle: 'Following vehicle', pedestrian: 'Yielding to pedestrian', yield: 'Yielding to oncoming traffic',
    box: 'Waiting for space past the intersection', crossing: 'Yielding to crossing vehicle',
    'overtake-wait': `Waiting to pass · ${ot?.why ?? ''}`, signal: null,
  }[c.reason];
  $('reason').hidden = !why;
  if (why) $('reason').textContent = c.lead && !['yield', 'overtake-wait'].includes(c.reason) ? `${why} · ${Math.round(c.lead.gap)} m` : why;
  $('traffic-count').textContent = world.traffic.background.length;
  $('ped-count').textContent = world.crowd.peds.length;
  const h = settings.hour;
  $('clock').textContent = `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`;
}

function updateScenarioCard() {
  const run = world.scenario, card = $('scenario-card');
  card.hidden = !run;
  if (!run) return;
  $('sc-name').textContent = run.def.name;
  $('sc-status').textContent = run.status;
  $('sc-status').dataset.status = run.status;
  $('sc-msg').textContent = run.message;
  $('sc-time').textContent = `${run.t.toFixed(1)} s`;
  $('sc-closest').textContent = Number.isFinite(run.closest) ? `closest ${run.closest.toFixed(1)} m` : '';
}

let toastTimer = null;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 2600);
}

// ---------- scenarios, weather, tools ----------

function startScenario(id) {
  settings.scenario = id;
  $('scenario').value = id;
  if (id) world.startScenario(id);
  else world.endScenario();
  seenContacts = world.contacts;
  setCam(settings.cam);
  updateHud();
}

function setWeatherUI(name) {
  settings.weather = name;
  $('weather').value = name;
  setWeather(name);
  precip.set(name);
  applyTimeOfDay(settings.hour);
}

function setDebug(on) {
  settings.debug = on;
  debugView.visible = on;
  $('btn-debug').classList.toggle('active', on);
}

let handBackFailedAt = -Infinity;
function toggleDrive() {
  if (world.manual) {
    const r = world.handBack();
    if (!r.ok && performance.now() - handBackFailedAt < 4000) {
      world.snapToLane();
      toast('Moved to the nearest lane · expert driving');
    } else if (!r.ok) {
      handBackFailedAt = performance.now();
      toast(`Can't hand back: ${r.reason} · press M again to reset onto the road`);
      return;
    } else toast('Expert driving');
  } else {
    document.activeElement?.blur();
    world.takeOver(driver);
    toast('You have the wheel · WASD / arrows or a gamepad · M to hand back');
  }
  $('btn-drive').classList.toggle('active', !!world.manual);
  updateHud();
}

function toggleRecord() {
  if (recorder.active) {
    recorder.stop();
    toast(`Saved ${recorder.count} frames`);
  } else {
    recorder.start({ seed, weather: settings.weather, hour: settings.hour, scenario: settings.scenario || null });
    toast('Recording the roof camera at 10 Hz · R to stop and download');
  }
  $('btn-rec').classList.toggle('active', recorder.active);
  $('rec').hidden = !recorder.active;
}

function shareLink() {
  const q = new URLSearchParams({ seed, hour: settings.hour.toFixed(2), cam: settings.cam, cars: settings.cars, peds: settings.peds, vans: settings.vans, weather: settings.weather });
  if (settings.scenario) q.set('scenario', settings.scenario);
  if (settings.debug) q.set('debug', '1');
  const url = `${location.origin}${location.pathname}?${q}`;
  history.replaceState(null, '', `?${q}`);
  navigator.clipboard?.writeText(url).then(() => toast('Link copied'), () => prompt('Copy this link', url));
}

// ---------- UI ----------

// Settings panel can be hidden so it doesn't cover the car; remembered per browser.
function setControlsVisible(on) {
  $('controls').hidden = !on;
  $('controls-show').hidden = on;
  try {
    localStorage.setItem('drivesim.controls', on ? 'shown' : 'hidden');
  } catch {}
}
let controlsShown = true;
try {
  controlsShown = localStorage.getItem('drivesim.controls') !== 'hidden';
} catch {}
setControlsVisible(controlsShown);
$('controls-hide').addEventListener('click', () => setControlsVisible(false));
$('controls-show').addEventListener('click', () => setControlsVisible(true));
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
  world.traffic.setCount(settings.cars, world.ego);
});
$('peds').value = settings.peds;
$('peds').addEventListener('input', (e) => {
  settings.peds = +e.target.value;
  world.crowd.setCount(settings.peds);
});
for (const [id, def] of Object.entries(SCENARIOS)) $('scenario').append(new Option(def.name, id));
$('scenario').addEventListener('change', (e) => startScenario(e.target.value));
$('btn-restart').addEventListener('click', () => startScenario(settings.scenario));
for (const [id, w] of Object.entries(WEATHER)) $('weather').append(new Option(w.label, id));
$('weather').addEventListener('change', (e) => setWeatherUI(e.target.value));
$('btn-drive').addEventListener('click', toggleDrive);
$('btn-debug').addEventListener('click', () => setDebug(!settings.debug));
$('btn-rec').addEventListener('click', toggleRecord);
$('btn-link').addEventListener('click', shareLink);
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
  if (e.target.closest?.('input, select, textarea') || !world) return;
  const key = e.key.toLowerCase();
  if (world.manual && DRIVE_KEYS.has(key)) return e.preventDefault();
  const map = { 1: 'chase', 2: 'hood', 3: 'orbit', 4: 'top' };
  if (map[key]) setCam(map[key]);
  else if (key === ' ') {
    e.preventDefault();
    $('btn-pause').click();
  } else if (key === 'm') toggleDrive();
  else if (key === 'o') setDebug(!settings.debug);
  else if (key === 'r') toggleRecord();
  else if (key === 'h') setControlsVisible($('controls').hidden);
});

window.__dbg = { scene, sun, renderer, camera, cityUniforms, get orbit() { return orbit; }, get world() { return world; } };
init().catch((err) => {
  console.error(err);
  setLoading(`Failed to start: ${err.message}`);
});
