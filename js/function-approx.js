// Function Approximation — 1D regression with different bases
'use strict';

(function(){

const X_MIN = -1, X_MAX = 1;
function trueV(s) { return Math.sin(3 * s) + 0.3 * s; }
function sampleX() { return X_MIN + Math.random() * (X_MAX - X_MIN); }

// Features
function polyFeat(s, n) { const f = new Array(n); let p = 1; for (let i = 0; i < n; i++) { f[i] = p; p *= s; } return f; }
function rbfFeat(s, n) {
  const f = new Array(n);
  const sigma = (X_MAX - X_MIN) / Math.max(2, n - 1);
  for (let i = 0; i < n; i++) {
    const c = X_MIN + i * (X_MAX - X_MIN) / Math.max(1, n - 1);
    f[i] = Math.exp(-Math.pow(s - c, 2) / (2 * sigma * sigma));
  }
  return f;
}
function tileFeat(s, nTilings) {
  // nTilings overlapping, each with 8 tiles. Returns size nTilings * 8
  const tilesPerT = 8;
  const range = X_MAX - X_MIN;
  const tileW = range / tilesPerT;
  const f = new Array(nTilings * tilesPerT).fill(0);
  for (let t = 0; t < nTilings; t++) {
    const offset = (t / nTilings) * tileW;
    const idx = Math.floor((s - X_MIN + offset) / tileW) % tilesPerT;
    f[t * tilesPerT + ((idx + tilesPerT) % tilesPerT)] = 1;
  }
  return f;
}
function fourierFeat(s, n) {
  const sn = (s - X_MIN) / (X_MAX - X_MIN);
  const f = new Array(n);
  for (let i = 0; i < n; i++) f[i] = Math.cos(Math.PI * i * sn);
  return f;
}

// Tiny neural net: 1 -> H -> 1 (tanh)
class TinyNN {
  constructor(h) {
    this.h = h;
    this.W1 = Array.from({length: h}, () => randn() * 0.5);
    this.b1 = new Array(h).fill(0);
    this.W2 = Array.from({length: h}, () => randn() * 0.5);
    this.b2 = 0;
  }
  forward(x) {
    const z = new Array(this.h);
    let y = this.b2;
    for (let i = 0; i < this.h; i++) {
      z[i] = Math.tanh(this.W1[i] * x + this.b1[i]);
      y += this.W2[i] * z[i];
    }
    return { y, z };
  }
  step(x, target, alpha) {
    const { y, z } = this.forward(x);
    const err = y - target;
    // grads
    const dW2 = z.map(zi => err * zi);
    const db2 = err;
    const dW1 = new Array(this.h), db1 = new Array(this.h);
    for (let i = 0; i < this.h; i++) {
      const dz = err * this.W2[i] * (1 - z[i]*z[i]);
      dW1[i] = dz * x;
      db1[i] = dz;
    }
    // update
    for (let i = 0; i < this.h; i++) {
      this.W2[i] -= alpha * dW2[i];
      this.W1[i] -= alpha * dW1[i];
      this.b1[i] -= alpha * db1[i];
    }
    this.b2 -= alpha * db2;
    return err * err;
  }
}

let basis = 'rbf';
let nF = 20;
let w = [];
let nn = null;
let nSeen = 0;
let mseHistory = [];
let samples = []; // visualised noisy samples
const MAX_SAMPLES = 200;

function getFeat(s) {
  if (basis === 'poly') return polyFeat(s, nF);
  if (basis === 'rbf') return rbfFeat(s, nF);
  if (basis === 'tile') return tileFeat(s, Math.max(1, Math.floor(nF / 8)));
  if (basis === 'fourier') return fourierFeat(s, nF);
  return null;
}

function predict(s) {
  if (basis === 'nn') return nn.forward(s).y;
  const f = getFeat(s);
  let y = 0; for (let i = 0; i < f.length; i++) y += w[i] * f[i];
  return y;
}

function reset() {
  basis = document.getElementById('basis').value;
  nF = parseInt(document.getElementById('nf').value);
  if (basis === 'nn') {
    nn = new TinyNN(Math.max(2, Math.min(64, nF)));
  } else {
    // determine dimension via dummy feature call
    const dim = getFeat(0).length;
    w = new Array(dim).fill(0);
  }
  nSeen = 0; mseHistory = []; samples = [];
  redraw(); updateStats();
}

function trainBatch(n) {
  const alpha = parseFloat(document.getElementById('alpha').value);
  const noise = parseFloat(document.getElementById('noise').value);
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const s = sampleX();
    const target = trueV(s) + noise * randn();
    if (basis === 'nn') {
      sse += nn.step(s, target, alpha);
    } else {
      const f = getFeat(s);
      let y = 0; for (let j = 0; j < f.length; j++) y += w[j] * f[j];
      const err = y - target;
      // weight clamp for poly stability
      for (let j = 0; j < f.length; j++) {
        w[j] -= alpha * err * f[j];
        if (Math.abs(w[j]) > 50) w[j] = Math.sign(w[j]) * 50;
      }
      sse += err * err;
    }
    samples.push({ x: s, y: target });
    if (samples.length > MAX_SAMPLES) samples.shift();
    nSeen++;
  }
  // mse on a held-out grid
  let mse = 0; const M = 100;
  for (let i = 0; i < M; i++) {
    const x = X_MIN + i * (X_MAX - X_MIN) / (M - 1);
    const d = predict(x) - trueV(x);
    mse += d * d;
  }
  mse /= M;
  mseHistory.push(mse);
}

const cv = document.getElementById('fa');
const ctx = setupCanvas(cv, 1100, 380);
const mctx = setupCanvas(document.getElementById('mse-chart'), 1100, 180);

function redraw() {
  clearCanvas(ctx);
  // axes
  const pad = 40, top = 20, bot = 360;
  const W = 1100 - pad * 2, H = bot - top;
  const xFor = x => pad + ((x - X_MIN) / (X_MAX - X_MIN)) * W;
  const yMin = -2.5, yMax = 2.5;
  const yFor = y => bot - ((y - yMin) / (yMax - yMin)) * H;
  // grid
  ctx.strokeStyle = '#1a2238'; ctx.lineWidth = 1;
  for (let g = -2; g <= 2; g++) {
    ctx.beginPath(); ctx.moveTo(pad, yFor(g)); ctx.lineTo(pad+W, yFor(g)); ctx.stroke();
  }
  ctx.strokeStyle = '#3a4570';
  ctx.beginPath(); ctx.moveTo(pad, yFor(0)); ctx.lineTo(pad+W, yFor(0)); ctx.stroke();
  ctx.fillStyle = '#8a93b3'; ctx.font = '10px JetBrains Mono';
  ctx.fillText(X_MIN.toFixed(1), pad-10, bot+12);
  ctx.fillText(X_MAX.toFixed(1), pad+W-10, bot+12);
  ctx.fillText('s', pad+W-4, bot+24);
  ctx.fillText('v(s)', pad-30, top+8);

  // Active features (faint) - only for RBF
  if (basis === 'rbf') {
    for (let i = 0; i < nF; i++) {
      const c = X_MIN + i * (X_MAX - X_MIN) / Math.max(1, nF - 1);
      const sigma = (X_MAX - X_MIN) / Math.max(2, nF - 1);
      ctx.strokeStyle = '#7c5cff22'; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = X_MIN; x <= X_MAX; x += (X_MAX - X_MIN) / 200) {
        const y = w[i] * Math.exp(-Math.pow(x - c, 2) / (2 * sigma * sigma));
        if (x === X_MIN) ctx.moveTo(xFor(x), yFor(y)); else ctx.lineTo(xFor(x), yFor(y));
      }
      ctx.stroke();
    }
  }

  // Noisy samples
  ctx.fillStyle = '#ff5c8a';
  for (const s of samples) { ctx.globalAlpha = 0.4; ctx.beginPath(); ctx.arc(xFor(s.x), yFor(s.y), 2, 0, Math.PI*2); ctx.fill(); }
  ctx.globalAlpha = 1;

  // True function
  ctx.strokeStyle = '#2dd4bf'; ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let x = X_MIN; x <= X_MAX; x += (X_MAX - X_MIN) / 400) {
    const y = trueV(x);
    if (x === X_MIN) ctx.moveTo(xFor(x), yFor(y)); else ctx.lineTo(xFor(x), yFor(y));
  }
  ctx.stroke();

  // Learned function
  ctx.strokeStyle = '#00d4ff'; ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let x = X_MIN; x <= X_MAX; x += (X_MAX - X_MIN) / 400) {
    const y = predict(x);
    if (x === X_MIN) ctx.moveTo(xFor(x), yFor(y)); else ctx.lineTo(xFor(x), yFor(y));
  }
  ctx.stroke();

  // MSE chart
  clearCanvas(mctx);
  drawAxes(mctx, 35, 160, 1050, 130, { xLabel: 'training steps', yLabel: 'log MSE' });
  if (mseHistory.length > 1) {
    const logs = mseHistory.map(m => Math.log10(Math.max(m, 1e-6)));
    drawSeries(mctx, logs, 35, 160, 1050, 130, { color: '#7c5cff' });
  }
}

function updateStats() {
  document.getElementById('s-n').textContent = nSeen;
  const mse = mseHistory[mseHistory.length-1];
  document.getElementById('s-mse').textContent = mse !== undefined ? mse.toFixed(4) : '—';
  let wn = 0;
  if (basis === 'nn') {
    for (const x of nn.W1) wn += x*x; for (const x of nn.W2) wn += x*x;
  } else {
    for (const x of w) wn += x*x;
  }
  document.getElementById('s-w').textContent = Math.sqrt(wn).toFixed(2);
  const d = basis === 'nn' ? (nn.h * 3 + 1) : w.length;
  document.getElementById('s-d').textContent = d;
  document.getElementById('nf-v').textContent = nF;
  document.getElementById('alpha-v').textContent = parseFloat(document.getElementById('alpha').value).toFixed(3);
  document.getElementById('noise-v').textContent = parseFloat(document.getElementById('noise').value).toFixed(2);
  document.getElementById('sps-v').textContent = parseInt(document.getElementById('sps').value);
}

const loop = new Loop(() => {
  trainBatch(parseInt(document.getElementById('sps').value));
  redraw(); updateStats();
}, { stepsPerFrame: 1 });

document.getElementById('run').addEventListener('click', () => {
  if (loop.running) { loop.stop(); document.getElementById('run').textContent = '▶ Train'; }
  else { loop.start(); document.getElementById('run').textContent = '⏸ Pause'; }
});
document.getElementById('step').addEventListener('click', () => { trainBatch(100); redraw(); updateStats(); });
document.getElementById('reset').addEventListener('click', () => { loop.stop(); document.getElementById('run').textContent = '▶ Train'; reset(); });
document.getElementById('basis').addEventListener('change', () => { loop.stop(); document.getElementById('run').textContent = '▶ Train'; reset(); });
document.getElementById('nf').addEventListener('input', () => { reset(); });
document.getElementById('alpha').addEventListener('input', updateStats);
document.getElementById('noise').addEventListener('input', updateStats);
document.getElementById('sps').addEventListener('input', updateStats);

reset();
})();
