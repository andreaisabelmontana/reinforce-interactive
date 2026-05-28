// Shared utilities for all RL demos
'use strict';

// ---------- Math / RNG ----------
const rand = () => Math.random();
const randn = () => { // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const argmax = (arr) => {
  let best = 0, v = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > v) { v = arr[i]; best = i; }
  return best;
};
const argmaxTies = (arr) => {
  const max = Math.max(...arr);
  const ties = [];
  for (let i = 0; i < arr.length; i++) if (arr[i] === max) ties.push(i);
  return ties[Math.floor(Math.random() * ties.length)];
};
const sumArr = (arr) => arr.reduce((a, b) => a + b, 0);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const lerp = (a, b, t) => a + (b - a) * t;
const fmt = (n, d = 2) => (typeof n === 'number' && isFinite(n)) ? n.toFixed(d) : '—';

// Softmax over array
function softmax(xs, temp = 1) {
  const m = Math.max(...xs);
  const exps = xs.map(x => Math.exp((x - m) / temp));
  const s = sumArr(exps);
  return exps.map(e => e / s);
}
function sampleCategorical(probs) {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

// ---------- Drawing helpers ----------
function clearCanvas(ctx, c = '#050811') {
  ctx.fillStyle = c;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

function drawAxes(ctx, x0, y0, w, h, opts = {}) {
  const { xLabel, yLabel, color = '#2a3454', labelColor = '#8a93b3' } = opts;
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0); ctx.lineTo(x0, y0 - h);
  ctx.moveTo(x0, y0); ctx.lineTo(x0 + w, y0);
  ctx.stroke();
  ctx.fillStyle = labelColor; ctx.font = '11px JetBrains Mono, monospace';
  if (xLabel) ctx.fillText(xLabel, x0 + w - 60, y0 + 14);
  if (yLabel) ctx.fillText(yLabel, x0 - 26, y0 - h + 4);
}

function drawSeries(ctx, data, x0, y0, w, h, opts = {}) {
  const { color = '#7c5cff', lineWidth = 2, yMin, yMax } = opts;
  if (data.length < 2) return;
  const lo = yMin !== undefined ? yMin : Math.min(...data);
  const hi = yMax !== undefined ? yMax : Math.max(...data);
  const range = hi - lo || 1;
  ctx.strokeStyle = color; ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = x0 + (i / Math.max(1, data.length - 1)) * w;
    const y = y0 - ((data[i] - lo) / range) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// HSL-based interpolation for heatmaps
function valueColor(v, vMin = -1, vMax = 1) {
  const t = clamp((v - vMin) / (vMax - vMin || 1), 0, 1);
  // Blue (cold) -> teal -> green -> yellow -> red (hot)
  const hue = (1 - t) * 230;
  return `hsl(${hue}, 75%, ${20 + t * 35}%)`;
}

// ---------- DOM helpers ----------
function $(id) { return document.getElementById(id); }
function on(el, ev, fn) { if (typeof el === 'string') el = $(el); el.addEventListener(ev, fn); }

function makeSlider(id, label, min, max, step, value, onInput) {
  const wrap = document.createElement('label');
  wrap.innerHTML = `${label} <span class="val" id="${id}-v">${value}</span>`;
  const inp = document.createElement('input');
  inp.type = 'range'; inp.id = id; inp.min = min; inp.max = max; inp.step = step; inp.value = value;
  inp.addEventListener('input', () => {
    document.getElementById(`${id}-v`).textContent = inp.value;
    onInput && onInput(parseFloat(inp.value));
  });
  wrap.appendChild(inp);
  return wrap;
}

// ---------- HiDPI canvas ----------
function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// ---------- Animation loop helper ----------
class Loop {
  constructor(stepFn, opts = {}) {
    this.stepFn = stepFn;
    this.running = false;
    this.stepsPerFrame = opts.stepsPerFrame || 1;
    this._raf = null;
  }
  start() {
    if (this.running) return;
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      for (let i = 0; i < this.stepsPerFrame; i++) this.stepFn();
      // Prefer rAF; fall back to setTimeout if backgrounded or unsupported
      if (typeof requestAnimationFrame !== 'undefined' && !document.hidden) {
        this._raf = requestAnimationFrame(tick);
      } else {
        this._raf = setTimeout(tick, 33);
      }
    };
    tick();
  }
  stop() {
    this.running = false;
    if (this._raf) { try { cancelAnimationFrame(this._raf); } catch(_){} try { clearTimeout(this._raf); } catch(_){} }
  }
  toggle() { this.running ? this.stop() : this.start(); }
}

// Mark current page nav link
document.addEventListener('DOMContentLoaded', () => {
  const path = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav a.navlink').forEach(a => {
    const href = a.getAttribute('href');
    if (href === path || (path === '' && href === 'index.html')) a.classList.add('active');
  });
});
