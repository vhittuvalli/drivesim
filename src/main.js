import { World, ROAD_WIDTH, LANE_OFFSET, GRASS, LIGHT_COLORS } from './world.js';
import { Car, CAR_LENGTH, CAR_WIDTH, clamp } from './car.js';
import { Camera, FOOTPRINT } from './sensor.js';
import { expertControl, supervise, laneBlocked, nextLight, lightSpeedLimit } from './expert.js';
import { Brain, Dataset } from './brain.js';

const DT = 1 / 60;
const $ = (id) => document.getElementById(id);

// ---------- state ----------
let world, car, nav;
const camera = new Camera();
const brain = new Brain();
const ds = new Dataset(8000);

const settings = { driver: 'expert', supervisor: true, recording: false, noise: 0.15, simSpeed: 1, paused: false, chase: true, zoom: 5 };
const keys = {};
let noise = 0, manualSteer = 0, tick = 0, stall = 0, prevLight = null;
let last = { expert: null, nn: null, cmd: { steer: 0, throttle: 0 }, override: null };

const newStats = () => ({ dist: 0, crashes: 0, violations: 0, interventions: 0, sinceIncident: 0 });
const stats = { expert: newStats(), neural: newStats(), manual: newStats() };

// ---------- world / car lifecycle ----------
function respawn(idx) {
  const N = world.pts.length;
  let guard = 0;
  while (laneBlocked(world, world.cum[idx], LANE_OFFSET, -12, 12) && guard++ < N) idx = (idx + 5) % N;
  const p = world.pts[idx];
  car = new Car(p.x + p.nx * LANE_OFFSET, p.y + p.ny * LANE_OFFSET, p.h);
  nav = world.nearest(car.x, car.y);
  noise = 0;
  stall = 0;
  prevLight = null;
}

function newWorld(seed) {
  world = new World(seed);
  $('seed').textContent = world.seed;
  respawn(0);
  drawMinimapBase();
}

function incident(kind, message) {
  const st = stats[settings.driver];
  st[kind]++;
  st.sinceIncident = 0;
  toast(message);
}

// ---------- simulation step ----------
function gauss() {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

function manualCmd(dt) {
  const target = (keys.ArrowRight ? 1 : 0) - (keys.ArrowLeft ? 1 : 0);
  manualSteer += clamp(target - manualSteer, -3 * dt, 3 * dt);
  return { steer: manualSteer, throttle: (keys.ArrowUp ? 1 : 0) - (keys.ArrowDown ? 1 : 0) };
}

function step(dt) {
  tick++;
  world.update(dt);
  nav = world.nearest(car.x, car.y, nav.idx);

  const exp = expertControl(world, car, nav);
  last.expert = exp;
  let pixels = null;
  const grab = () => (pixels ??= camera.capture(world, car));

  let cmd;
  if (settings.driver === 'expert') {
    // Ornstein-Uhlenbeck steering noise: pushes the car off-center so the dataset contains recoveries.
    noise += -1.2 * noise * dt + settings.noise * 1.55 * Math.sqrt(dt) * gauss();
    cmd = { steer: clamp(exp.steer + noise, -1, 1), throttle: exp.throttle };
  } else if (settings.driver === 'neural') {
    const [steer, throttle] = brain.predict(grab(), car.v);
    last.nn = { steer, throttle };
    cmd = { steer, throttle };
  } else {
    cmd = manualCmd(dt);
  }

  last.override = null;
  if (settings.supervisor && settings.driver !== 'expert') {
    const r = supervise(world, car, nav, cmd);
    if (r.reason && r.throttle < cmd.throttle) last.override = r.reason;
    cmd = { steer: r.steer, throttle: r.throttle };
  }

  // Labels always come from the expert, whoever is driving (DAgger when the NN drives).
  if (settings.recording && tick % 2 === 0) ds.add(grab(), car.v, exp.steer, exp.throttle);

  car.step(dt, cmd.steer, cmd.throttle);
  last.cmd = cmd;

  const st = stats[settings.driver];
  st.dist += car.v * dt;
  st.sinceIncident += car.v * dt;

  const n2 = world.nearest(car.x, car.y, nav.idx);
  nav = n2;

  // Red light violations: detect the moment the "next light" changes after being close.
  const nl = nextLight(world, n2.s);
  if (prevLight && nl.light !== prevLight.light && prevLight.dist < 10 && prevLight.light.state === 'red') {
    incident('violations', 'Ran a red light');
  }
  prevLight = nl;

  // Collisions.
  if (Math.abs(n2.off) > ROAD_WIDTH / 2 + 0.2) {
    incident('crashes', 'Left the road');
    return respawn(n2.idx);
  }
  const ch = Math.cos(car.h), sh = Math.sin(car.h);
  for (const c of world.cones) {
    const dx = c.x - car.x, dy = c.y - car.y;
    if (dx * dx + dy * dy > 36) continue;
    const lx = dx * ch + dy * sh, ly = -dx * sh + dy * ch;
    if (Math.abs(lx) < CAR_LENGTH / 2 + 0.5 && Math.abs(ly) < CAR_WIDTH / 2 + 0.5) {
      incident('crashes', 'Hit a cone');
      return respawn(n2.idx);
    }
  }

  // Stalled with no reason to be stopped -> count as an intervention and tow forward.
  if (car.v < 0.2 && lightSpeedLimit(world, n2.s, car.v) > 0.5) stall += dt;
  else stall = 0;
  if (stall > 5) {
    incident('interventions', 'Stalled — towed forward');
    respawn((n2.idx + 10) % world.pts.length);
  }
}

// ---------- rendering ----------
const view = $('view');
const vctx = view.getContext('2d');
const sensorCtx = $('sensor').getContext('2d');
const mini = $('minimap');
const mctx = mini.getContext('2d');
let miniBase = null, miniT = null;

function drawMinimapBase() {
  const { minX, minY, maxX, maxY } = world.bounds;
  const pad = 12;
  const s = Math.min((mini.width - pad * 2) / (maxX - minX), (mini.height - pad * 2) / (maxY - minY));
  miniT = (x, y) => [pad + (x - minX) * s, pad + (y - minY) * s];
  const c = document.createElement('canvas');
  c.width = mini.width;
  c.height = mini.height;
  const g = c.getContext('2d');
  g.strokeStyle = '#8a8f98';
  g.lineWidth = 3;
  g.beginPath();
  world.pts.forEach((p, i) => g[i ? 'lineTo' : 'moveTo'](...miniT(p.x, p.y)));
  g.closePath();
  g.stroke();
  g.fillStyle = '#ff7a00';
  for (const cone of world.cones) {
    const [x, y] = miniT(cone.x, cone.y);
    g.fillRect(x - 1, y - 1, 2, 2);
  }
  miniBase = c;
}

function drawMinimap() {
  mctx.clearRect(0, 0, mini.width, mini.height);
  mctx.drawImage(miniBase, 0, 0);
  for (const l of world.lights) {
    const [x, y] = miniT(world.pts[l.idx].x, world.pts[l.idx].y);
    mctx.fillStyle = LIGHT_COLORS[l.state];
    mctx.beginPath(); mctx.arc(x, y, 3, 0, Math.PI * 2); mctx.fill();
  }
  const [x, y] = miniT(car.x, car.y);
  mctx.fillStyle = '#4db8ff';
  mctx.strokeStyle = '#fff';
  mctx.lineWidth = 1.5;
  mctx.beginPath(); mctx.arc(x, y, 4, 0, Math.PI * 2); mctx.fill(); mctx.stroke();
}

function drawCar(ctx) {
  ctx.save();
  ctx.translate(car.x, car.y);
  ctx.rotate(car.h);
  const L = CAR_LENGTH, W = CAR_WIDTH;
  ctx.fillStyle = '#111';
  for (const [wx, wy, a] of [[L * 0.3, -W / 2, car.steer], [L * 0.3, W / 2, car.steer], [-L * 0.3, -W / 2, 0], [-L * 0.3, W / 2, 0]]) {
    ctx.save(); ctx.translate(wx, wy); ctx.rotate(a); ctx.fillRect(-0.4, -0.15, 0.8, 0.3); ctx.restore();
  }
  ctx.fillStyle = '#2f8cff';
  ctx.beginPath();
  ctx.roundRect(-L / 2, -W / 2, L, W, 0.5);
  ctx.fill();
  ctx.fillStyle = '#bfe3ff';
  ctx.fillRect(L * 0.08, -W / 2 + 0.2, L * 0.18, W - 0.4);
  ctx.restore();
}

function render() {
  const dpr = window.devicePixelRatio || 1;
  const w = view.clientWidth, h = view.clientHeight;
  if (view.width !== Math.round(w * dpr) || view.height !== Math.round(h * dpr)) {
    view.width = Math.round(w * dpr);
    view.height = Math.round(h * dpr);
  }
  const ctx = vctx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = GRASS;
  ctx.fillRect(0, 0, w, h);
  ctx.translate(w / 2, settings.chase ? h * 0.68 : h / 2);
  if (settings.chase) ctx.rotate(-Math.PI / 2 - car.h);
  ctx.scale(settings.zoom, settings.zoom);
  ctx.translate(-car.x, -car.y);
  world.draw(ctx);

  // Camera footprint.
  ctx.save();
  ctx.translate(car.x, car.y);
  ctx.rotate(car.h + Math.PI / 2);
  ctx.strokeStyle = 'rgba(77,184,255,0.8)';
  ctx.lineWidth = 0.25;
  ctx.setLineDash([1, 1]);
  ctx.strokeRect(FOOTPRINT.x, FOOTPRINT.y, FOOTPRINT.w, FOOTPRINT.h);
  ctx.setLineDash([]);
  ctx.restore();

  // Expert's pure-pursuit target.
  if (last.expert) {
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.beginPath(); ctx.arc(last.expert.target.x, last.expert.target.y, 0.4, 0, Math.PI * 2); ctx.fill();
  }
  drawCar(ctx);

  // Sensor preview (and a display-only NN prediction when the NN isn't driving).
  const px = camera.capture(world, car);
  sensorCtx.drawImage(camera.canvas, 0, 0);
  if (settings.driver !== 'neural' && brain.trained && !brain.training && tick % 3 === 0) {
    const [steer, throttle] = brain.predict(px, car.v);
    last.nn = { steer, throttle };
  }

  drawMinimap();
  updatePanel();
}

// ---------- panel ----------
function setBar(el, v) {
  if (v == null) { el.style.width = '0'; return; }
  v = clamp(v, -1, 1);
  el.style.left = `${v >= 0 ? 50 : 50 + v * 50}%`;
  el.style.width = `${Math.abs(v) * 50}%`;
}

let panelTick = 0;
function updatePanel() {
  setBar($('bar-steer-expert'), last.expert?.steer);
  setBar($('bar-steer-nn'), brain.trained ? last.nn?.steer : null);
  setBar($('bar-steer-cmd'), last.cmd.steer);
  setBar($('bar-thr-expert'), last.expert?.throttle);
  setBar($('bar-thr-nn'), brain.trained ? last.nn?.throttle : null);
  setBar($('bar-thr-cmd'), last.cmd.throttle);

  if (panelTick++ % 6) return;
  const st = stats[settings.driver];
  $('hud-driver').textContent = { expert: 'EXPERT', neural: 'NEURAL NET', manual: 'MANUAL' }[settings.driver];
  $('hud-driver').dataset.driver = settings.driver;
  $('hud-speed').textContent = `${(car.v * 3.6).toFixed(0)} km/h`;
  $('hud-rec').hidden = !settings.recording;
  $('hud-override').hidden = !last.override;
  $('hud-override').textContent = `SUPERVISOR BRAKE · ${last.override ?? ''}`;
  $('hud-paused').hidden = !settings.paused;

  $('m-dist').textContent = `${(st.dist / 1000).toFixed(2)} km`;
  $('m-crashes').textContent = st.crashes;
  $('m-violations').textContent = st.violations;
  $('m-interventions').textContent = st.interventions;
  const incidents = st.crashes + st.violations + st.interventions;
  $('m-mpi').textContent = incidents ? `${(st.dist / 1000 / incidents).toFixed(2)} km` : '—';
  $('m-since').textContent = `${(st.sinceIncident / 1000).toFixed(2)} km`;
  $('metrics-driver').textContent = settings.driver;

  $('ds-count').textContent = `${ds.count.toLocaleString()} / ${ds.capacity.toLocaleString()}`;
  $('ds-fill').style.width = `${(ds.count / ds.capacity) * 100}%`;
  $('btn-record').textContent = settings.recording ? 'Stop recording' : 'Start recording';
  $('btn-record').classList.toggle('active', settings.recording);
  $('btn-train').disabled = brain.training || ds.count < 500;
  $('btn-pause').textContent = settings.paused ? 'Resume' : 'Pause';
  $('model-status').textContent = brain.training ? 'training…' : brain.trained ? 'trained' : 'untrained';
}

let toastTimer = null;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 1600);
}

function log(line) {
  const el = $('train-log');
  el.textContent = `${line}\n${el.textContent}`.slice(0, 4000);
}

function setDriver(d) {
  if (d === 'neural' && !brain.trained) {
    toast('Train (or load) a model first');
    return;
  }
  settings.driver = d;
  noise = 0;
  document.querySelectorAll('[data-driver-btn]').forEach((b) => b.classList.toggle('active', b.dataset.driverBtn === d));
}

// ---------- wiring ----------
document.querySelectorAll('[data-driver-btn]').forEach((b) => b.addEventListener('click', () => setDriver(b.dataset.driverBtn)));
$('chk-supervisor').addEventListener('change', (e) => (settings.supervisor = e.target.checked));
$('chk-chase').addEventListener('change', (e) => (settings.chase = e.target.checked));
$('rng-noise').addEventListener('input', (e) => {
  settings.noise = +e.target.value;
  $('noise-val').textContent = settings.noise.toFixed(2);
});
$('sel-speed').addEventListener('change', (e) => (settings.simSpeed = +e.target.value));
$('btn-record').addEventListener('click', () => (settings.recording = !settings.recording));
$('btn-clear').addEventListener('click', () => { ds.clear(); toast('Dataset cleared'); });
$('btn-pause').addEventListener('click', () => (settings.paused = !settings.paused));
$('btn-world').addEventListener('click', () => newWorld());
$('btn-reset-stats').addEventListener('click', () => (stats[settings.driver] = newStats()));

$('btn-train').addEventListener('click', async () => {
  const epochs = clamp(parseInt($('inp-epochs').value, 10) || 8, 1, 100);
  log(`Training on ${ds.count} samples for ${epochs} epochs…`);
  const t0 = performance.now();
  await brain.train(ds, {
    epochs,
    onProgress: (f) => ($('train-fill').style.width = `${f * 100}%`),
    onEpoch: (e, loss, val) => log(`epoch ${e}/${epochs}  loss ${loss.toFixed(4)}  val ${val.toFixed(4)}`),
  });
  log(`Done in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  toast('Model trained — try the Neural driver');
});
$('btn-save').addEventListener('click', async () => {
  try { await brain.save(); toast('Model saved in this browser'); } catch (e) { toast(`Save failed: ${e.message}`); }
});
$('btn-load').addEventListener('click', async () => {
  try { await brain.load(); toast('Model loaded'); } catch { toast('No saved model found'); }
});
$('btn-reset-model').addEventListener('click', () => {
  if (settings.driver === 'neural') setDriver('expert');
  brain.reset();
  last.nn = null;
  toast('Model reset');
});

view.addEventListener('wheel', (e) => {
  e.preventDefault();
  settings.zoom = clamp(settings.zoom * Math.exp(-e.deltaY * 0.001), 1.5, 20);
}, { passive: false });

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key.startsWith('Arrow')) e.preventDefault();
  keys[e.key] = true;
  if (e.key === '1') setDriver('expert');
  if (e.key === '2') setDriver('neural');
  if (e.key === '3') setDriver('manual');
  if (e.key === 'r') settings.recording = !settings.recording;
  if (e.key === ' ') { e.preventDefault(); settings.paused = !settings.paused; }
});
window.addEventListener('keyup', (e) => (keys[e.key] = false));

// ---------- boot ----------
newWorld();
$('param-count').textContent = brain.paramCount().toLocaleString();
$('tf-backend').textContent = tf.getBackend();
brain.load().then(() => log('Loaded saved model from this browser'), () => {});

function frame() {
  if (!settings.paused) for (let i = 0; i < settings.simSpeed; i++) step(DT);
  render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
