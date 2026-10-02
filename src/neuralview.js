// HUD for the neural driver: what the network sees (camera with its attention map), what it
// segments, a live chart of its steering against the expert's, two steering wheels (the
// network's and the expert's) turning side by side, and the safety driver's tally.
// Also the benchmark scorecard.
import { CLASSES } from './labels.js';
import { MAX_STEER } from './config.js';

// Class colors for the segmentation view (Cityscapes-like), indexed like CLASSES.
const SEG_COLORS = {
  sky: [70, 130, 180], road: [128, 64, 128], marking: [255, 255, 255], sidewalk: [244, 35, 232], building: [70, 70, 70],
  vegetation: [107, 142, 35], pole: [153, 153, 153], 'traffic light': [250, 170, 30], vehicle: [0, 0, 142], pedestrian: [220, 20, 60], terrain: [152, 251, 152],
};
const PALETTE = CLASSES.map((c) => SEG_COLORS[c]);
// Validated categorical pair on the dark HUD surface (dataviz palette, dark steps).
const NN_COLOR = '#3987e5', EXPERT_COLOR = '#d95926';
const CHART_SECONDS = 12;

const $ = (id) => document.getElementById(id);
const fmtDist = (m) => (m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`);

// Steering wheel: a sedan's ~15:1 steering ratio turns the road-wheel angle into the angle a
// driver would turn the wheel (full lock = ±516°). Each wheel follows its driver's command at
// the car's steering actuator rate (vehicle.js: 1.6 rad/s at the road wheels), so it moves the
// way a real wheel would.
const STEER_RATIO = 15;
const WHEEL_DEG = (MAX_STEER * STEER_RATIO * 180) / Math.PI;
const WHEEL_RATE = 1.6 / MAX_STEER; // normalized steer units per second

class Wheel {
  constructor(el) {
    this.el = el;
    this.turn = el.querySelector('.turn');
    this.deg = el.querySelector('.deg');
    this.pedal = el.querySelector('.pedal i');
    this.steer = 0;
  }

  // target: normalized steer command (null: this driver has nothing to say), throttle in [-1, 1].
  update(target, throttle, dt, driving) {
    this.el.classList.toggle('idle', target === null);
    this.el.classList.toggle('driving', driving);
    if (target !== null) this.steer += Math.max(-WHEEL_RATE * dt, Math.min(WHEEL_RATE * dt, target - this.steer));
    const deg = this.steer * WHEEL_DEG;
    this.turn.setAttribute('transform', `rotate(${deg.toFixed(1)})`);
    this.deg.textContent = target === null ? '–' : `${deg >= 0 ? '+' : '−'}${Math.abs(deg).toFixed(0)}°`;
    const t = Math.max(-1, Math.min(1, throttle ?? 0));
    Object.assign(this.pedal.style, t >= 0 ? { left: '50%', width: `${t * 50}%` } : { left: `${50 + t * 50}%`, width: `${-t * 50}%` });
    return deg;
  }
}

export class NeuralView {
  constructor() {
    this.cam = $('nn-cam').getContext('2d');
    this.seg = $('nn-seg').getContext('2d');
    this.chart = $('nn-chart');
    this.cctx = this.chart.getContext('2d');
    this.samples = []; // {t, nn, ex, safety}
    this.shownPred = null;
    this.camImg = null;
    this.segImg = null;
    this.wheelNN = new Wheel($('wheel-nn'));
    this.wheelEx = new Wheel($('wheel-ex'));
  }

  set visible(on) {
    $('neural-panel').hidden = !on;
    $('wheels').hidden = !on;
  }

  // Every rendered frame (dt: simulated seconds since the last frame, 0 while paused).
  animateWheels(world, dt) {
    const c = world.ctrl, ex = world.expertCtrl;
    if (!c || !ex) return;
    const nn = c.nn ?? null;
    const a = this.wheelNN.update(nn ? nn.steer : null, nn?.throttle, dt, c.driver === 'neural');
    const b = this.wheelEx.update(ex.steer, ex.throttle, dt, c.driver !== 'neural');
    $('wheel-diff').textContent = nn ? `${Math.abs(a - b).toFixed(0)}° apart` : 'network starting';
  }

  get visible() {
    return !$('neural-panel').hidden;
  }

  clear() {
    this.samples = [];
  }

  // Once per rendered frame.
  record(world) {
    const c = world.ctrl;
    if (!c?.nn || !world.expertCtrl) return;
    const last = this.samples[this.samples.length - 1];
    if (last && world.t - last.t < 1 / 30) return;
    if (last && world.t < last.t) this.samples = []; // time went backwards (new world)
    this.samples.push({ t: world.t, nn: c.nn.steer, ex: world.expertCtrl.steer, safety: c.driver === 'safety' });
    while (this.samples.length && this.samples[0].t < world.t - CHART_SECONDS) this.samples.shift();
  }

  update(world, neural) {
    if (!this.visible) return;
    const s = world.safety, c = world.ctrl;
    const driver = c?.driver === 'neural' ? 'neural' : c?.driver === 'safety' ? 'safety' : 'waiting';
    $('nn-driver').textContent = { neural: 'Network', safety: 'Safety driver', waiting: 'Starting' }[driver];
    $('nn-driver').dataset.driver = driver;
    $('nn-stats').textContent = `${fmtDist(s.autoDist)} autonomous · ${s.disengagements} takeover${s.disengagements === 1 ? '' : 's'}` +
      (s.disengagements ? ` · ${fmtDist(s.distPerDisengagement)} each` : '');
    const p = neural.pred;
    const last = s.events[s.events.length - 1];
    const light = p?.light && Object.entries(p.light).sort((a, b) => b[1] - a[1])[0];
    $('nn-info').textContent = [
      p ? `cmd ${p.cmd}` : null,
      light ? `light ${light[0]} ${(light[1] * 100).toFixed(0)}%` : null,
      c?.nn?.lightStop ? 'stopping for the light' : null,
      p ? `target ${(p.vTarget[p.cmdIndex] * 3.6).toFixed(0)} km/h` : null,
      neural.inferMs ? `${neural.inferMs.toFixed(0)} ms` : null,
      last && world.t - last.t < 6 ? `took over: ${last.reason}` : null,
    ].filter(Boolean).join(' · ');
    if (p && p !== this.shownPred) {
      this.shownPred = p;
      this.drawCamera(p);
      this.drawSeg(p);
      $('nn-tele-fig').hidden = !p.tele;
      if (p.tele) $('nn-tele').getContext('2d').putImageData(p.tele, 0, 0);
    }
    this.drawChart(world.t);
  }

  // Camera frame the network saw, with its attention map as a heat overlay.
  drawCamera(p) {
    const img = p.image, [, , ah, aw] = p.attnDims, W = img.width, H = img.height;
    this.camImg ??= new ImageData(W, H);
    const out = this.camImg.data, src = img.data, a = p.attention;
    for (let y = 0; y < H; y++) {
      const ay = Math.min(ah - 1, Math.floor((y * ah) / H));
      for (let x = 0; x < W; x++) {
        const k = (y * W + x) * 4, v = a[ay * aw + Math.min(aw - 1, Math.floor((x * aw) / W))];
        // Dim the frame and paint attention in warm colors on top.
        const g = 0.45, heat = Math.min(1, v * 1.4);
        out[k] = src[k] * g * (1 - heat) + 255 * heat;
        out[k + 1] = src[k + 1] * g * (1 - heat) + 200 * heat * heat;
        out[k + 2] = src[k + 2] * g * (1 - heat) + 40 * heat;
        out[k + 3] = 255;
      }
    }
    this.cam.putImageData(this.camImg, 0, 0);
  }

  // Arg-max of the segmentation logits, one pixel per output cell (scaled up by CSS).
  drawSeg(p) {
    const [, nc, h, w] = p.segDims, L = p.seg, n = h * w;
    this.segImg ??= new ImageData(w, h);
    const out = this.segImg.data;
    for (let i = 0; i < n; i++) {
      let best = 0, bv = -Infinity;
      for (let c = 0; c < nc; c++) {
        const v = L[c * n + i];
        if (v > bv) (bv = v), (best = c);
      }
      const col = PALETTE[best];
      out.set([col[0], col[1], col[2], 255], i * 4);
    }
    this.seg.putImageData(this.segImg, 0, 0);
  }

  drawChart(now) {
    const cv = this.chart, ctx = this.cctx, dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = cv.clientWidth, ch = cv.clientHeight;
    if (cv.width !== Math.round(cw * dpr)) (cv.width = Math.round(cw * dpr)), (cv.height = Math.round(ch * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    const pad = 4, X = (t) => cw - ((now - t) / CHART_SECONDS) * cw, Y = (v) => ch / 2 - v * (ch / 2 - pad);
    // Safety-driver spans.
    ctx.fillStyle = 'rgba(255, 255, 255, 0.09)';
    const S = this.samples;
    for (let i = 0; i < S.length; i++) {
      if (!S[i].safety) continue;
      let j = i;
      while (j + 1 < S.length && S[j + 1].safety) j++;
      ctx.fillRect(X(S[i].t), 0, Math.max(1, X(S[j].t) - X(S[i].t)), ch);
      i = j;
    }
    // Zero line and ±0.5 guides.
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, Y(0) + 0.5);
    ctx.lineTo(cw, Y(0) + 0.5);
    ctx.stroke();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.beginPath();
    for (const g of [-0.5, 0.5]) ctx.moveTo(0, Y(g) + 0.5), ctx.lineTo(cw, Y(g) + 0.5);
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    for (const [key, color] of [['ex', EXPERT_COLOR], ['nn', NN_COLOR]]) {
      ctx.strokeStyle = color;
      ctx.beginPath();
      S.forEach((s, i) => (i ? ctx.lineTo(X(s.t), Y(s[key])) : ctx.moveTo(X(s.t), Y(s[key]))));
      ctx.stroke();
    }
    const last = S[S.length - 1];
    $('nn-steer-nn').textContent = last ? last.nn.toFixed(2) : '–';
    $('nn-steer-ex').textContent = last ? last.ex.toFixed(2) : '–';
  }
}

// ---------- benchmark scorecard ----------

export function renderScorecard(bench, { onClose } = {}) {
  const card = $('scorecard');
  card.hidden = false;
  $('score-close').onclick = () => {
    card.hidden = true;
    onClose?.();
  };
  const { index, total, item } = bench.progress;
  $('score-progress').textContent = bench.done ? `Done · ${new Date(bench.started).toLocaleTimeString()}` : `${index + 1}/${total} · ${item?.name ?? ''}${item?.trial ? ` #${item.trial}` : ''}`;
  const rows = [];
  // Scenarios: one row each, one pill per trial.
  const byId = new Map();
  for (const it of bench.items.filter((i) => i.kind === 'scenario')) byId.set(it.id, { name: it.name, runs: [] });
  for (const r of bench.results.filter((r) => r.kind === 'scenario')) byId.get(r.id).runs.push(r);
  for (const [id, { name, runs }] of byId) {
    const pills = runs.map((r) => {
      const cls = r.status === 'passed' ? (r.takeovers ? 'assist' : 'pass') : 'fail';
      const label = r.status === 'passed' ? (r.takeovers ? '✓*' : '✓') : '✗';
      const tip = `${r.message} · ${r.time.toFixed(1)} s · ${r.takeovers} takeover(s)${r.reasons.length ? `: ${r.reasons.join(', ')}` : ''}`;
      return `<span class="res ${cls}" title="${tip.replace(/"/g, '&quot;')}">${label}</span>`;
    });
    const running = bench.current?.id === id;
    rows.push(`<tr><td>${name}</td><td class="pills">${pills.join('')}${running ? '<span class="res run">…</span>' : ''}</td></tr>`);
  }
  $('score-scenarios').innerHTML = rows.join('');
  const drives = bench.results.filter((r) => r.kind === 'drive');
  $('score-drives').innerHTML = drives
    .map((r) => `<tr><td>${r.name}</td><td class="num">${fmtDist(r.dist)}</td><td class="num">${r.takeovers}</td><td class="num">${r.contacts}</td></tr>`)
    .join('') + (bench.current?.kind === 'drive' ? `<tr class="muted"><td>${bench.current.name}</td><td colspan="3">driving…</td></tr>` : '');
  const s = bench.summary;
  $('score-summary').hidden = !s;
  if (s) {
    $('score-summary').innerHTML =
      `<div><b>${s.passed}/${s.scenarios}</b> scenarios passed · <b>${s.clean}</b> without takeovers</div>` +
      `<div><b>${s.metersPerTakeover === null ? '∞' : fmtDist(s.metersPerTakeover)}</b> per takeover · ${(s.autonomy * 100).toFixed(0)}% autonomous over ${s.driveKm.toFixed(2)} km</div>` +
      `<div>${s.contacts} contact${s.contacts === 1 ? '' : 's'}</div>`;
  }
}
