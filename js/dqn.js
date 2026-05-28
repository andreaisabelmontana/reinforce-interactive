// DQN on Mountain Car — tiny neural net Q(s, a)
'use strict';

(function(){

// ---------- Mountain Car env ----------
const MC = {
  posMin: -1.2, posMax: 0.6,
  velMin: -0.07, velMax: 0.07,
  goalPos: 0.5,
  forceMag: 0.001,
  gravity: 0.0025,
  reset() {
    return { p: -0.5 + (Math.random() - 0.5) * 0.2, v: 0 };
  },
  step(s, a) {
    // a in {0,1,2} -> force in {-1, 0, +1}
    const force = (a - 1);
    let v = s.v + force * this.forceMag + Math.cos(3 * s.p) * (-this.gravity);
    v = clamp(v, this.velMin, this.velMax);
    let p = s.p + v;
    if (p < this.posMin) { p = this.posMin; v = 0; }
    p = clamp(p, this.posMin, this.posMax);
    const done = p >= this.goalPos;
    const r = done ? 0 : -1;
    return { ns: { p, v }, r, done };
  },
  norm(s) {
    // normalise to [-1, 1] for the net
    const np = 2 * (s.p - this.posMin) / (this.posMax - this.posMin) - 1;
    const nv = 2 * (s.v - this.velMin) / (this.velMax - this.velMin) - 1;
    return [np, nv];
  }
};

// ---------- Tiny net Q(s, a): 2 -> H -> 3, ReLU ----------
class QNet {
  constructor(H) {
    this.H = H;
    // He init
    this.W1 = []; this.b1 = new Array(H).fill(0);
    for (let i = 0; i < H; i++) this.W1.push([randn()*Math.sqrt(2/2), randn()*Math.sqrt(2/2)]);
    this.W2 = []; this.b2 = new Array(3).fill(0);
    for (let j = 0; j < 3; j++) {
      const row = new Array(H);
      for (let i = 0; i < H; i++) row[i] = randn() * Math.sqrt(2/H);
      this.W2.push(row);
    }
  }
  forward(x) {
    const h = new Array(this.H), hAct = new Array(this.H);
    for (let i = 0; i < this.H; i++) {
      const z = this.W1[i][0]*x[0] + this.W1[i][1]*x[1] + this.b1[i];
      h[i] = z;
      hAct[i] = z > 0 ? z : 0;
    }
    const q = new Array(3);
    for (let j = 0; j < 3; j++) {
      let s = this.b2[j];
      for (let i = 0; i < this.H; i++) s += this.W2[j][i] * hAct[i];
      q[j] = s;
    }
    return { q, hAct, h };
  }
  copyFrom(other) {
    for (let i = 0; i < this.H; i++) {
      this.W1[i][0] = other.W1[i][0]; this.W1[i][1] = other.W1[i][1];
      this.b1[i] = other.b1[i];
    }
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < this.H; i++) this.W2[j][i] = other.W2[j][i];
      this.b2[j] = other.b2[j];
    }
  }
}

// SGD update for a single sample on action a: ∇ ½(target - Q(s,a))²
function trainOne(net, x, a, target, alpha) {
  const { q, hAct, h } = net.forward(x);
  const err = q[a] - target;
  // dL/dW2[a][i] = err * hAct[i]; dL/db2[a] = err
  for (let i = 0; i < net.H; i++) net.W2[a][i] -= alpha * err * hAct[i];
  net.b2[a] -= alpha * err;
  // dL/dh[i] = err * W2[a][i] * (z>0)
  for (let i = 0; i < net.H; i++) {
    if (h[i] <= 0) continue;
    const dh = err * net.W2[a][i];
    net.W1[i][0] -= alpha * dh * x[0];
    net.W1[i][1] -= alpha * dh * x[1];
    net.b1[i]    -= alpha * dh;
  }
  return err * err;
}

// ---------- Agent state ----------
let net, tgt;
let buffer = []; const MAX_BUF = 5000;
let state = MC.reset();
let totalSteps = 0;
let episodes = 0; let epReturn = 0; let epLen = 0;
let returns = []; let losses = [];
let lastLoss = 0;

function reset() {
  const H = parseInt(document.getElementById('h').value);
  net = new QNet(H); tgt = new QNet(H); tgt.copyFrom(net);
  buffer = []; state = MC.reset();
  totalSteps = 0; episodes = 0; epReturn = 0; epLen = 0;
  returns = []; losses = [];
  redraw(); updateStats();
}

function epsilonNow() {
  const decaySteps = 8000;
  const t = clamp(totalSteps / decaySteps, 0, 1);
  return 1.0 * (1 - t) + 0.05 * t;
}

function stepEnv() {
  const eps = epsilonNow();
  const x = MC.norm(state);
  const { q } = net.forward(x);
  const a = Math.random() < eps ? Math.floor(Math.random() * 3) : argmaxTies(q);
  const { ns, r, done } = MC.step(state, a);
  buffer.push({ s: state, a, r, ns, done });
  if (buffer.length > MAX_BUF) buffer.shift();
  epReturn += r; epLen++;
  totalSteps++;
  state = ns;
  if (done || epLen >= 500) {
    returns.push(epReturn);
    episodes++;
    state = MC.reset(); epReturn = 0; epLen = 0;
  }
  trainStep();
  const ts = parseInt(document.getElementById('ts').value);
  if (totalSteps % ts === 0) tgt.copyFrom(net);
}

function trainStep() {
  const bs = parseInt(document.getElementById('bs').value);
  if (buffer.length < bs) return;
  const alpha = parseFloat(document.getElementById('alpha').value);
  const gamma = parseFloat(document.getElementById('gamma').value);
  let sumLoss = 0;
  for (let i = 0; i < bs; i++) {
    const { s, a, r, ns, done } = buffer[Math.floor(Math.random() * buffer.length)];
    const xs = MC.norm(s), xns = MC.norm(ns);
    const tq = tgt.forward(xns).q;
    const target = done ? r : r + gamma * Math.max(...tq);
    sumLoss += trainOne(net, xs, a, target, alpha);
  }
  lastLoss = sumLoss / bs;
  losses.push(lastLoss);
  if (losses.length > 2000) losses.shift();
}

// ---------- Rendering ----------
const envCtx = setupCanvas(document.getElementById('env'), 540, 260);
const qmapCtx = setupCanvas(document.getElementById('qmap'), 540, 260);
const chCtx = setupCanvas(document.getElementById('charts'), 1100, 220);

function drawEnv() {
  clearCanvas(envCtx);
  const W = 540, H = 260, pad = 20;
  envCtx.strokeStyle = '#1a2238'; envCtx.lineWidth = 1;
  // landscape
  const path = [];
  for (let i = 0; i <= 200; i++) {
    const p = MC.posMin + (i/200) * (MC.posMax - MC.posMin);
    const y = Math.sin(3 * p) * 50 + 150;
    const x = pad + (i/200) * (W - 2*pad);
    path.push([x, y]);
  }
  envCtx.fillStyle = '#0d1424';
  envCtx.beginPath();
  envCtx.moveTo(path[0][0], path[0][1]);
  for (const [x,y] of path) envCtx.lineTo(x, y);
  envCtx.lineTo(W - pad, H);
  envCtx.lineTo(pad, H);
  envCtx.fill();
  envCtx.strokeStyle = '#7c5cff';
  envCtx.beginPath();
  envCtx.moveTo(path[0][0], path[0][1]);
  for (const [x,y] of path) envCtx.lineTo(x, y);
  envCtx.stroke();

  // Goal flag
  const gx = pad + ((MC.goalPos - MC.posMin) / (MC.posMax - MC.posMin)) * (W - 2*pad);
  const gy = Math.sin(3 * MC.goalPos) * 50 + 150;
  envCtx.strokeStyle = '#2dd4bf'; envCtx.lineWidth = 2;
  envCtx.beginPath(); envCtx.moveTo(gx, gy); envCtx.lineTo(gx, gy - 30); envCtx.stroke();
  envCtx.fillStyle = '#2dd4bf';
  envCtx.beginPath(); envCtx.moveTo(gx, gy - 30); envCtx.lineTo(gx + 10, gy - 25); envCtx.lineTo(gx, gy - 20); envCtx.fill();

  // Car
  const cx = pad + ((state.p - MC.posMin) / (MC.posMax - MC.posMin)) * (W - 2*pad);
  const cy = Math.sin(3 * state.p) * 50 + 150 - 8;
  envCtx.fillStyle = '#ff5c8a';
  envCtx.beginPath(); envCtx.arc(cx, cy, 7, 0, Math.PI*2); envCtx.fill();

  // info
  envCtx.fillStyle = '#8a93b3'; envCtx.font = '11px JetBrains Mono';
  envCtx.fillText(`p=${state.p.toFixed(2)}  v=${state.v.toFixed(3)}  step=${epLen}`, 10, 20);
  envCtx.fillText(`ep return so far: ${epReturn}`, 10, 36);
}

function drawQMap() {
  clearCanvas(qmapCtx);
  if (!net) return;
  const W = 540, H = 260, R = 30, C = 60;
  const cellW = W / C, cellH = H / R;
  // compute Q values across grid
  const qmax = new Array(R);
  const qa = new Array(R);
  for (let r = 0; r < R; r++) {
    qmax[r] = new Array(C); qa[r] = new Array(C);
    const v = -1 + 2 * (r / (R-1));
    for (let c = 0; c < C; c++) {
      const p = -1 + 2 * (c / (C-1));
      const q = net.forward([p, v]).q;
      qmax[r][c] = Math.max(...q);
      qa[r][c] = argmax(q);
    }
  }
  // normalise color
  let lo = Infinity, hi = -Infinity;
  for (let r=0; r<R; r++) for (let c=0; c<C; c++) {
    if (qmax[r][c] < lo) lo = qmax[r][c]; if (qmax[r][c] > hi) hi = qmax[r][c];
  }
  if (lo === hi) { lo -= 1; hi += 1; }
  for (let r=0; r<R; r++) for (let c=0; c<C; c++) {
    qmapCtx.fillStyle = valueColor(qmax[r][c], lo, hi);
    qmapCtx.fillRect(c * cellW, H - (r+1) * cellH, cellW + 1, cellH + 1);
  }
  // arrows
  qmapCtx.fillStyle = '#e6ecff';
  qmapCtx.font = '8px system-ui';
  qmapCtx.textAlign = 'center';
  for (let r=0; r<R; r+=3) for (let c=0; c<C; c+=4) {
    const ch = qa[r][c] === 0 ? '←' : qa[r][c] === 2 ? '→' : '·';
    qmapCtx.fillText(ch, c * cellW + cellW/2, H - r * cellH - cellH/2);
  }
  qmapCtx.textAlign = 'start';
  // labels
  qmapCtx.fillStyle = '#8a93b3'; qmapCtx.font = '11px JetBrains Mono';
  qmapCtx.fillText('vel →', W - 60, 16);
  qmapCtx.fillText('← pos', 10, H - 8);
}

function drawCharts() {
  clearCanvas(chCtx);
  drawAxes(chCtx, 35, 200, 1050, 170, { xLabel: 'episode', yLabel: 'G / log L' });
  if (returns.length > 1) {
    const lo = Math.min(...returns), hi = Math.max(...returns);
    drawSeries(chCtx, returns, 35, 200, 1050, 170, { color: '#00d4ff', yMin: lo, yMax: hi });
  }
  if (losses.length > 10) {
    const logs = losses.map(l => Math.log10(Math.max(l, 1e-6)));
    // place on right side scale (just normalised)
    drawSeries(chCtx, logs, 35, 200, 1050, 170, { color: '#ff5c8a44', lineWidth: 1 });
  }
}

function redraw() { drawEnv(); drawQMap(); drawCharts(); }

function updateStats() {
  document.getElementById('s-step').textContent = totalSteps;
  document.getElementById('s-ep').textContent = episodes;
  document.getElementById('s-rb').textContent = buffer.length;
  document.getElementById('s-eps').textContent = epsilonNow().toFixed(2);
  document.getElementById('s-loss').textContent = lastLoss ? lastLoss.toFixed(4) : '—';
  if (returns.length > 0) {
    const tail = returns.slice(-20);
    const avg = tail.reduce((a,b)=>a+b,0) / tail.length;
    document.getElementById('s-avg').textContent = avg.toFixed(1);
    document.getElementById('s-best').textContent = Math.max(...returns).toFixed(0);
  }
  // controls
  document.getElementById('h-v').textContent = document.getElementById('h').value;
  document.getElementById('alpha-v').textContent = parseFloat(document.getElementById('alpha').value).toFixed(4);
  document.getElementById('gamma-v').textContent = parseFloat(document.getElementById('gamma').value).toFixed(2);
  document.getElementById('bs-v').textContent = document.getElementById('bs').value;
  document.getElementById('ts-v').textContent = document.getElementById('ts').value;
  document.getElementById('spf-v').textContent = document.getElementById('spf').value;
}

const loop = new Loop(() => {
  const spf = parseInt(document.getElementById('spf').value);
  for (let i = 0; i < spf; i++) stepEnv();
  // throttle redraw a touch — every frame we still redraw, just don't redraw qmap every frame
  drawEnv(); drawCharts();
  if (totalSteps % 50 === 0) drawQMap();
  updateStats();
}, { stepsPerFrame: 1 });

document.getElementById('run').addEventListener('click', () => {
  if (loop.running) { loop.stop(); document.getElementById('run').textContent = '▶ Train'; }
  else { loop.start(); document.getElementById('run').textContent = '⏸ Pause'; }
});
document.getElementById('step').addEventListener('click', () => { for (let i=0;i<200;i++) stepEnv(); redraw(); updateStats(); });
document.getElementById('reset').addEventListener('click', () => { loop.stop(); document.getElementById('run').textContent = '▶ Train'; reset(); });
document.getElementById('rollout').addEventListener('click', async () => {
  let s = MC.reset(); let G = 0;
  for (let i = 0; i < 500; i++) {
    state = s;
    const q = net.forward(MC.norm(s)).q;
    const a = argmaxTies(q);
    const { ns, r, done } = MC.step(s, a);
    G += r;
    drawEnv();
    s = ns;
    if (done) break;
    await new Promise(r => setTimeout(r, 20));
  }
  document.getElementById('s-best').textContent = Math.max(parseFloat(document.getElementById('s-best').textContent) || -Infinity, G).toFixed(0);
});
['h','alpha','gamma','bs','ts','spf'].forEach(id => document.getElementById(id).addEventListener('input', updateStats));
document.getElementById('h').addEventListener('change', () => { reset(); });

reset();
})();
