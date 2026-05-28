// Genetic Algorithm evolving a linear CartPole policy
'use strict';

(function(){

// ---------- CartPole physics (classic) ----------
const CP = {
  gravity: 9.8,
  massCart: 1.0,
  massPole: 0.1,
  length: 0.5,         // half pole length
  forceMag: 10.0,
  tau: 0.02,
  thresAng: 12 * Math.PI / 180,
  thresPos: 2.4,
  maxSteps: 500,
  reset() { return [randn()*0.05, randn()*0.05, randn()*0.05, randn()*0.05]; },
  step(s, a) {
    const [x, xd, t, td] = s;
    const force = a === 1 ? this.forceMag : -this.forceMag;
    const totalMass = this.massCart + this.massPole;
    const cost = Math.cos(t), sint = Math.sin(t);
    const temp = (force + this.massPole * this.length * td * td * sint) / totalMass;
    const tAcc = (this.gravity * sint - cost * temp) /
      (this.length * (4/3 - this.massPole * cost * cost / totalMass));
    const xAcc = temp - this.massPole * this.length * tAcc * cost / totalMass;
    const nx = x + this.tau * xd;
    const nxd = xd + this.tau * xAcc;
    const nt = t + this.tau * td;
    const ntd = td + this.tau * tAcc;
    const done = Math.abs(nx) > this.thresPos || Math.abs(nt) > this.thresAng;
    return { ns: [nx, nxd, nt, ntd], r: 1, done };
  }
};

function evalGenome(w, reps) {
  let total = 0;
  for (let r = 0; r < reps; r++) {
    let s = CP.reset();
    let steps = 0;
    while (steps < CP.maxSteps) {
      const z = w[0]*s[0] + w[1]*s[1] + w[2]*s[2] + w[3]*s[3];
      const a = z > 0 ? 1 : 0;
      const { ns, done } = CP.step(s, a);
      s = ns; steps++;
      if (done) break;
    }
    total += steps;
  }
  return total / reps;
}

function randGenome() { return [randn(), randn(), randn(), randn()]; }
function mutate(w, sigma) { return w.map(v => v + randn() * sigma); }
function crossover(a, b) {
  // uniform crossover
  return a.map((v, i) => Math.random() < 0.5 ? v : b[i]);
}
function tournament(pop, fit, k = 3) {
  let best = -1, bf = -Infinity;
  for (let i = 0; i < k; i++) {
    const idx = Math.floor(Math.random() * pop.length);
    if (fit[idx] > bf) { bf = fit[idx]; best = idx; }
  }
  return pop[best];
}

let pop = [];
let fit = [];
let gen = 0;
let history = []; // {best, mean, worst}
let lastBest = null;

function reset() {
  const P = parseInt(document.getElementById('pop').value);
  pop = Array.from({length: P}, randGenome);
  fit = [];
  gen = 0; history = [];
  evaluate();
  redraw(); updateStats();
}

function evaluate() {
  const reps = parseInt(document.getElementById('reps').value);
  fit = pop.map(w => evalGenome(w, reps));
}

function runGeneration() {
  const eliteFrac = parseFloat(document.getElementById('elite').value);
  const sigma = parseFloat(document.getElementById('sigma').value);
  const cxp = parseFloat(document.getElementById('cx').value);
  const P = pop.length;
  // sort by fitness
  const idx = fit.map((f, i) => i).sort((a, b) => fit[b] - fit[a]);
  const nElite = Math.max(1, Math.floor(eliteFrac * P));
  const newPop = [];
  for (let i = 0; i < nElite; i++) newPop.push(pop[idx[i]].slice());
  while (newPop.length < P) {
    let child;
    const a = tournament(pop, fit);
    if (Math.random() < cxp) {
      const b = tournament(pop, fit);
      child = crossover(a, b);
    } else child = a.slice();
    child = mutate(child, sigma);
    newPop.push(child);
  }
  pop = newPop;
  evaluate();
  gen++;
  const best = Math.max(...fit), worst = Math.min(...fit);
  const mean = fit.reduce((s,v)=>s+v,0) / fit.length;
  history.push({ best, mean, worst });
  lastBest = pop[idx[0]];
}

// Rendering
const cartCtx = setupCanvas(document.getElementById('cart'), 540, 240);
const popCtx = setupCanvas(document.getElementById('popcv'), 540, 240);
const histCtx = setupCanvas(document.getElementById('histcv'), 1100, 180);

function drawCart(s) {
  clearCanvas(cartCtx);
  const W = 540, H = 240;
  // track
  cartCtx.strokeStyle = '#3a4570'; cartCtx.lineWidth = 1;
  cartCtx.beginPath(); cartCtx.moveTo(20, H - 60); cartCtx.lineTo(W - 20, H - 60); cartCtx.stroke();
  if (!s) return;
  // cart pos
  const cx = W/2 + s[0] * 80;
  const cy = H - 60;
  cartCtx.fillStyle = '#1a2238'; cartCtx.strokeStyle = '#7c5cff';
  cartCtx.lineWidth = 2;
  cartCtx.fillRect(cx - 30, cy - 12, 60, 24);
  cartCtx.strokeRect(cx - 30, cy - 12, 60, 24);
  // pole
  const len = 100;
  const px = cx + Math.sin(s[2]) * len;
  const py = cy - Math.cos(s[2]) * len;
  cartCtx.strokeStyle = '#00d4ff'; cartCtx.lineWidth = 5;
  cartCtx.beginPath(); cartCtx.moveTo(cx, cy); cartCtx.lineTo(px, py); cartCtx.stroke();
  cartCtx.fillStyle = '#00d4ff';
  cartCtx.beginPath(); cartCtx.arc(px, py, 6, 0, Math.PI*2); cartCtx.fill();
}

let visState = null;
let visBest = null;
let visLoop = null;

function startVis() {
  if (visLoop) clearInterval(visLoop);
  visBest = lastBest || pop[0];
  visState = CP.reset();
  let t = 0;
  visLoop = setInterval(() => {
    const w = visBest;
    const z = w[0]*visState[0] + w[1]*visState[1] + w[2]*visState[2] + w[3]*visState[3];
    const a = z > 0 ? 1 : 0;
    const { ns, done } = CP.step(visState, a);
    visState = ns; t++;
    drawCart(visState);
    cartCtx.fillStyle = '#8a93b3'; cartCtx.font = '11px JetBrains Mono';
    cartCtx.fillText(`t=${t}`, 12, 18);
    if (done || t >= 500) { clearInterval(visLoop); visLoop = null; }
  }, 20);
}

function drawPop() {
  clearCanvas(popCtx);
  if (!pop.length) return;
  // project genomes onto first 2 dims for vis
  const xs = pop.map(w => w[0]);
  const ys = pop.map(w => w[2]);
  const fLo = Math.min(...fit), fHi = Math.max(...fit, 1);
  const xLo = Math.min(...xs), xHi = Math.max(...xs);
  for (let i = 0; i < pop.length; i++) {
    const x = 30 + ((xs[i] - xLo) / (xHi - xLo + 1e-6)) * (540 - 60);
    const y = 220 - ((fit[i] - fLo) / (fHi - fLo + 1e-6)) * 180;
    const col = valueColor(fit[i], fLo, fHi);
    popCtx.fillStyle = col;
    popCtx.beginPath(); popCtx.arc(x, y, 5, 0, Math.PI*2); popCtx.fill();
  }
  popCtx.fillStyle = '#8a93b3'; popCtx.font = '11px JetBrains Mono';
  popCtx.fillText('weight[0]', 470, 235);
  popCtx.fillText('fitness', 10, 18);
}

function drawHist() {
  clearCanvas(histCtx);
  drawAxes(histCtx, 35, 160, 1050, 130, { xLabel: 'generation', yLabel: 'fitness' });
  if (history.length > 0) {
    const ymax = Math.max(...history.map(h => h.best), 10);
    drawSeries(histCtx, history.map(h=>h.worst), 35, 160, 1050, 130, { color: '#ff5c8a', yMin: 0, yMax: ymax });
    drawSeries(histCtx, history.map(h=>h.mean), 35, 160, 1050, 130, { color: '#00d4ff', yMin: 0, yMax: ymax });
    drawSeries(histCtx, history.map(h=>h.best), 35, 160, 1050, 130, { color: '#2dd4bf', yMin: 0, yMax: ymax });
  }
}

function redraw() { drawCart(visState); drawPop(); drawHist(); }

function updateStats() {
  document.getElementById('s-g').textContent = gen;
  if (fit.length) {
    document.getElementById('s-best').textContent = Math.max(...fit).toFixed(1);
    document.getElementById('s-avg').textContent = (fit.reduce((a,b)=>a+b,0)/fit.length).toFixed(1);
    const meanW = pop.map(w => Math.sqrt(w.reduce((s,v)=>s+v*v,0))).reduce((a,b)=>a+b,0)/pop.length;
    document.getElementById('s-w').textContent = meanW.toFixed(2);
  }
  ['pop','elite','sigma','cx','reps'].forEach(id => {
    const el = document.getElementById(id);
    document.getElementById(id+'-v').textContent = el.step.includes('.') ? parseFloat(el.value).toFixed(2) : el.value;
  });
}

const loop = new Loop(() => { runGeneration(); redraw(); updateStats(); }, { stepsPerFrame: 1 });

document.getElementById('step').addEventListener('click', () => { runGeneration(); redraw(); updateStats(); });
document.getElementById('run').addEventListener('click', () => {
  if (loop.running) { loop.stop(); document.getElementById('run').textContent = '▶ Auto-run'; }
  else { loop.start(); document.getElementById('run').textContent = '⏸ Pause'; }
});
document.getElementById('reset').addEventListener('click', () => {
  loop.stop(); document.getElementById('run').textContent = '▶ Auto-run';
  if (visLoop) { clearInterval(visLoop); visLoop = null; }
  reset();
});
document.getElementById('vis').addEventListener('click', startVis);
document.getElementById('pop').addEventListener('change', () => reset());
['elite','sigma','cx','reps'].forEach(id => document.getElementById(id).addEventListener('input', updateStats));

reset();
})();
