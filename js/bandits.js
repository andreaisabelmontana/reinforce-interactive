// Multi-Armed Bandits — interactive demo.
//
// The core estimators and the ε-greedy / UCB / greedy / optimistic selection
// rules mirror the unit-tested module src/bandit.js. This file adds the
// gradient-bandit and Thompson-sampling variants plus all the canvas rendering.
'use strict';

import { RNG, argmax, argmaxTies, softmax } from '../src/rng.js';
import { Loop } from './ui.js';

const rng = new RNG((Date.now() & 0x7fffffff) || 1);
const randn = () => rng.normal();
const sampleCategorical = (probs) => rng.categorical(probs);
const fmt = (n, d = 2) => (typeof n === 'number' && isFinite(n)) ? n.toFixed(d) : '—';

(function () {

let K = 10;
let trueMu = [];       // true means
let Q = [];            // estimated Q values
let N = [];            // pull counts
let H = [];            // gradient bandit preferences
let avgRew = 0;        // baseline for gradient
let lastArm = -1;
let optArm = 0;
let t = 0;
let totalReward = 0;
let optCount = 0;
let rewardHistory = [];
let regretHistory = [];
let cumReward = 0;

function init() {
  K = parseInt(document.getElementById('k').value);
  trueMu = Array.from({length: K}, () => randn());
  optArm = argmax(trueMu);
  Q = new Array(K).fill(0);
  N = new Array(K).fill(0);
  H = new Array(K).fill(0);
  avgRew = 0;
  lastArm = -1; t = 0; totalReward = 0; optCount = 0; cumReward = 0;
  rewardHistory = []; regretHistory = [];
  draw(); updateStats();
}

function selectArm() {
  const strat = document.getElementById('strategy').value;
  const eps = parseFloat(document.getElementById('eps').value);
  const c = parseFloat(document.getElementById('c').value);

  if (strat === 'greedy') return argmaxTies(Q);
  if (strat === 'egreedy') return Math.random() < eps ? Math.floor(Math.random()*K) : argmaxTies(Q);
  if (strat === 'optimistic') return argmaxTies(Q); // Q init handled in init
  if (strat === 'ucb') {
    const ucb = Q.map((q, i) => N[i] === 0 ? Infinity : q + c * Math.sqrt(Math.log(t+1) / N[i]));
    return argmaxTies(ucb);
  }
  if (strat === 'thompson') {
    // Gaussian posterior: mean=Q, var=1/(N+1)
    const samples = Q.map((q, i) => q + randn() / Math.sqrt(N[i] + 1));
    return argmaxTies(samples);
  }
  if (strat === 'grad') {
    const probs = softmax(H);
    return sampleCategorical(probs);
  }
  return Math.floor(Math.random()*K);
}

function pull(a) {
  const r = trueMu[a] + randn(); // reward = N(μ, 1)
  N[a]++;
  const alpha = parseFloat(document.getElementById('alpha').value);
  const useAlpha = alpha > 0;
  if (document.getElementById('strategy').value === 'grad') {
    // Gradient bandit update
    const probs = softmax(H);
    avgRew = avgRew + (r - avgRew) / (t + 1);
    for (let i = 0; i < K; i++) {
      const isA = (i === a) ? 1 : 0;
      H[i] = H[i] + (alpha || 0.1) * (r - avgRew) * (isA - probs[i]);
    }
    Q[a] = Q[a] + (1 / N[a]) * (r - Q[a]); // still track Q for display
  } else {
    if (useAlpha) Q[a] = Q[a] + alpha * (r - Q[a]);
    else Q[a] = Q[a] + (1 / N[a]) * (r - Q[a]);
  }
  lastArm = a;
  t++;
  totalReward += r;
  cumReward += r;
  if (a === optArm) optCount++;
  rewardHistory.push(totalReward / t);
  regretHistory.push(t * trueMu[optArm] - cumReward);
}

function step() {
  // Handle optimistic init lazily
  if (t === 0 && document.getElementById('strategy').value === 'optimistic') {
    Q = new Array(K).fill(5);
  }
  const a = selectArm();
  pull(a);
}

function updateStats() {
  document.getElementById('s-step').textContent = t;
  document.getElementById('s-total').textContent = fmt(totalReward, 2);
  document.getElementById('s-avg').textContent = fmt(t ? totalReward / t : 0, 3);
  document.getElementById('s-opt').textContent = fmt(t ? 100 * optCount / t : 0, 1) + '%';
  document.getElementById('s-reg').textContent = fmt(regretHistory.length ? regretHistory[regretHistory.length-1] : 0, 2);
}

const armCanvas = document.getElementById('armcanvas');
const armCtx = setupCanvas(armCanvas, 1100, 240);
const rewCanvas = document.getElementById('rewchart');
const rewCtx = setupCanvas(rewCanvas, 540, 220);
const regCanvas = document.getElementById('regchart');
const regCtx = setupCanvas(regCanvas, 540, 220);

function draw() {
  // Arms display
  clearCanvas(armCtx);
  const pad = 40, top = 30, bot = 200;
  const W = 1100 - pad * 2;
  const armW = W / K;

  // Y scale: combine true & estimated
  const allVals = trueMu.concat(Q);
  const lo = Math.min(-3, Math.min(...allVals)) - 0.3;
  const hi = Math.max(3, Math.max(...allVals)) + 0.3;
  const range = hi - lo;
  const yFor = v => bot - ((v - lo) / range) * (bot - top);

  // grid
  armCtx.strokeStyle = '#1a2238'; armCtx.lineWidth = 1;
  for (let g = Math.ceil(lo); g <= Math.floor(hi); g++) {
    if (g === 0) continue;
    armCtx.beginPath();
    armCtx.moveTo(pad, yFor(g)); armCtx.lineTo(pad + W, yFor(g));
    armCtx.stroke();
  }
  // zero line
  armCtx.strokeStyle = '#3a4570'; armCtx.beginPath();
  armCtx.moveTo(pad, yFor(0)); armCtx.lineTo(pad+W, yFor(0)); armCtx.stroke();

  // Bars for each arm
  for (let i = 0; i < K; i++) {
    const cx = pad + i * armW + armW / 2;
    // True mu marker (purple horizontal bar)
    const yTrue = yFor(trueMu[i]);
    armCtx.fillStyle = i === optArm ? '#2dd4bf' : '#7c5cff';
    armCtx.fillRect(cx - armW * 0.35, yTrue - 2, armW * 0.7, 4);

    // Q bar (cyan)
    const yQ = yFor(Q[i]);
    armCtx.fillStyle = '#00d4ff';
    const qHeight = Math.abs(yFor(0) - yQ);
    armCtx.globalAlpha = 0.6;
    armCtx.fillRect(cx - armW * 0.25, Math.min(yQ, yFor(0)), armW * 0.5, qHeight);
    armCtx.globalAlpha = 1;

    // Last pull highlight
    if (i === lastArm) {
      armCtx.strokeStyle = '#ff5c8a';
      armCtx.lineWidth = 2;
      armCtx.strokeRect(cx - armW * 0.4, top - 5, armW * 0.8, bot - top + 10);
    }

    // Label
    armCtx.fillStyle = '#8a93b3';
    armCtx.font = '11px JetBrains Mono, monospace';
    armCtx.textAlign = 'center';
    armCtx.fillText(`a${i}`, cx, bot + 14);
    armCtx.fillStyle = '#e6ecff';
    armCtx.font = '10px JetBrains Mono, monospace';
    armCtx.fillText(`N=${N[i]}`, cx, bot + 28);
  }
  armCtx.textAlign = 'start';

  // Reward chart
  clearCanvas(rewCtx);
  drawAxes(rewCtx, 35, 200, 480, 170, { xLabel: 't', yLabel: 'r̄' });
  rewCtx.fillStyle = '#8a93b3'; rewCtx.font = '10px JetBrains Mono';
  rewCtx.fillText(`max=${fmt(trueMu[optArm], 2)}`, 460, 30);
  if (rewardHistory.length > 1) {
    drawSeries(rewCtx, rewardHistory, 35, 200, 480, 170, { color: '#00d4ff', yMin: lo, yMax: hi });
    // optimal line
    rewCtx.strokeStyle = '#2dd4bf66'; rewCtx.setLineDash([4,4]); rewCtx.beginPath();
    const yOpt = 200 - ((trueMu[optArm] - lo) / (hi - lo)) * 170;
    rewCtx.moveTo(35, yOpt); rewCtx.lineTo(35 + 480, yOpt); rewCtx.stroke();
    rewCtx.setLineDash([]);
  }

  // Regret chart
  clearCanvas(regCtx);
  drawAxes(regCtx, 35, 200, 480, 170, { xLabel: 't', yLabel: 'regret' });
  if (regretHistory.length > 1) {
    const rmax = Math.max(1, regretHistory[regretHistory.length-1]);
    drawSeries(regCtx, regretHistory, 35, 200, 480, 170, { color: '#ff5c8a', yMin: 0, yMax: rmax });
  }
}

const loop = new Loop(() => {
  step();
  // Throttle redraw a bit for perf, every step is fine here
  draw(); updateStats();
}, { stepsPerFrame: 1 });

document.getElementById('speed').addEventListener('input', () => {
  loop.stepsPerFrame = parseInt(document.getElementById('speed').value);
  document.getElementById('speed-v').textContent = loop.stepsPerFrame;
});

document.getElementById('run').addEventListener('click', () => {
  if (loop.running) { loop.stop(); document.getElementById('run').textContent = '▶ Run'; }
  else { loop.start(); document.getElementById('run').textContent = '⏸ Pause'; }
});
document.getElementById('step').addEventListener('click', () => { step(); draw(); updateStats(); });
document.getElementById('reset').addEventListener('click', () => { loop.stop(); document.getElementById('run').textContent = '▶ Run'; init(); });
document.getElementById('k').addEventListener('input', () => { loop.stop(); document.getElementById('run').textContent = '▶ Run'; init(); });
document.getElementById('strategy').addEventListener('change', () => { loop.stop(); document.getElementById('run').textContent = '▶ Run'; init(); });

init();
loop.stepsPerFrame = 20;
})();
