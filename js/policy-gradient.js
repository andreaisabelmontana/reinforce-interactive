// REINFORCE on a small discrete environment
'use strict';

(function(){

// Environments
function makeCorridor(stochastic = false) {
  const N = 5;
  return {
    nStates: N,
    nActions: 2, // 0 = left, 1 = right
    start: 0,
    isTerminal: (s) => s === N - 1,
    step(s, a) {
      let ns;
      if (stochastic) {
        const wind = Math.random() < 0.2;
        const eff = wind ? 1 - a : a;
        ns = clamp(s + (eff === 1 ? 1 : -1), 0, N - 1);
      } else {
        ns = clamp(s + (a === 1 ? 1 : -1), 0, N - 1);
      }
      const r = ns === N - 1 ? 1 : -0.05;
      return { ns, r, done: ns === N - 1 };
    },
    nodeXs() { return [0.1, 0.3, 0.5, 0.7, 0.9]; }
  };
}

function makeContextualBandit() {
  // 3 states, each one is its own bandit
  const N = 3;
  // optimal action per state: [right, left, right]
  return {
    nStates: N,
    nActions: 2,
    start: () => Math.floor(Math.random() * N),
    isTerminal: () => true,
    step(s, a) {
      const opt = [1, 0, 1][s];
      const r = a === opt ? 1 : -1;
      return { ns: s, r, done: true };
    }
  };
}

let env;
let theta; // [state][action]
let baseline = 0;
let returns = [];
let episodeCount = 0;
let lastTraj = [];
let currentEnvName = 'corridor';

function rebuildEnv() {
  currentEnvName = document.getElementById('env').value;
  if (currentEnvName === 'windy') env = makeCorridor(true);
  else if (currentEnvName === 'bandit2') env = makeContextualBandit();
  else env = makeCorridor(false);
  theta = Array.from({length: env.nStates}, () => new Array(env.nActions).fill(0));
  baseline = 0; returns = []; episodeCount = 0; lastTraj = [];
}

function policyProbs(s) { return softmax(theta[s]); }

function runEpisode() {
  const gamma = parseFloat(document.getElementById('gamma').value);
  let s = typeof env.start === 'function' ? env.start() : env.start;
  const traj = [];
  let t = 0;
  while (t < 50) {
    const probs = policyProbs(s);
    const a = sampleCategorical(probs);
    const { ns, r, done } = env.step(s, a);
    traj.push({ s, a, r, probs });
    s = ns;
    t++;
    if (done) break;
  }
  return traj;
}

function trainEpisode() {
  const gamma = parseFloat(document.getElementById('gamma').value);
  const alpha = parseFloat(document.getElementById('alpha').value);
  const useBaseline = document.getElementById('baseline').checked;
  const traj = runEpisode();
  // compute G_t for each step
  const G = new Array(traj.length).fill(0);
  let acc = 0;
  for (let i = traj.length - 1; i >= 0; i--) { acc = traj[i].r + gamma * acc; G[i] = acc; }
  const G0 = G[0]; // total return

  // Update baseline (running mean)
  baseline = baseline + 0.05 * (G0 - baseline);
  const b = useBaseline ? baseline : 0;

  // Update theta: θ ← θ + α γᵗ (Gₜ - b) ∇log π
  for (let i = 0; i < traj.length; i++) {
    const { s, a, probs } = traj[i];
    const adv = G[i] - b;
    for (let aa = 0; aa < env.nActions; aa++) {
      const grad = (aa === a ? 1 : 0) - probs[aa];
      theta[s][aa] += alpha * Math.pow(gamma, i) * adv * grad;
      // clip for stability
      if (theta[s][aa] > 20) theta[s][aa] = 20;
      if (theta[s][aa] < -20) theta[s][aa] = -20;
    }
  }
  episodeCount++;
  returns.push(G0);
  lastTraj = traj;
  return G0;
}

// Rendering
const pctx = setupCanvas(document.getElementById('pcv'), 540, 240);
const tctx = setupCanvas(document.getElementById('tcv'), 540, 240);
const rctx = setupCanvas(document.getElementById('rch'), 1100, 180);

function redraw() {
  // Policy bars
  clearCanvas(pctx);
  const N = env.nStates;
  const barW = 480 / N;
  const top = 30, bot = 200, H = bot - top;
  pctx.fillStyle = '#8a93b3'; pctx.font = '11px JetBrains Mono';
  pctx.fillText('π(a|s)', 10, 20);
  for (let s = 0; s < N; s++) {
    const probs = policyProbs(s);
    const x = 40 + s * barW + barW * 0.1;
    const w = barW * 0.8;
    pctx.fillStyle = '#1a2238'; pctx.fillRect(x, top, w, H);
    // right (action 1) = cyan
    if (env.nActions === 2) {
      const rH = H * probs[1];
      pctx.fillStyle = '#00d4ff';
      pctx.fillRect(x, bot - rH, w, rH);
      pctx.fillStyle = '#7c5cff';
      pctx.fillRect(x, top, w, H * probs[0]);
      pctx.fillStyle = '#e6ecff';
      pctx.font = '10px JetBrains Mono';
      pctx.textAlign = 'center';
      pctx.fillText(`s${s}`, x + w/2, bot + 15);
      pctx.fillText(`R=${probs[1].toFixed(2)}`, x + w/2, bot + 28);
    }
  }
  pctx.textAlign = 'start';

  // Trajectory
  clearCanvas(tctx);
  if (lastTraj.length && env.nodeXs) {
    const xs = env.nodeXs();
    const cy = 120;
    // nodes
    for (let i = 0; i < xs.length; i++) {
      tctx.fillStyle = i === xs.length - 1 ? '#2dd4bf' : '#1a2238';
      tctx.beginPath(); tctx.arc(xs[i] * 540, cy, 22, 0, Math.PI*2); tctx.fill();
      tctx.strokeStyle = '#3a4570'; tctx.stroke();
      tctx.fillStyle = '#e6ecff';
      tctx.font = '12px JetBrains Mono'; tctx.textAlign = 'center';
      tctx.fillText('s'+i, xs[i] * 540, cy + 4);
    }
    // Path
    tctx.strokeStyle = '#ff5c8a'; tctx.lineWidth = 2;
    tctx.beginPath();
    for (let i = 0; i < lastTraj.length; i++) {
      const x = xs[lastTraj[i].s] * 540;
      const y = cy + (i % 2 === 0 ? -40 : 40);
      if (i === 0) tctx.moveTo(x, y); else tctx.lineTo(x, y);
    }
    tctx.stroke();
    tctx.fillStyle = '#ff5c8a';
    for (let i = 0; i < lastTraj.length; i++) {
      const x = xs[lastTraj[i].s] * 540;
      const y = cy + (i % 2 === 0 ? -40 : 40);
      tctx.beginPath(); tctx.arc(x, y, 4, 0, Math.PI*2); tctx.fill();
    }
    tctx.fillStyle = '#8a93b3'; tctx.font = '11px JetBrains Mono';
    tctx.textAlign = 'start';
    tctx.fillText(`len=${lastTraj.length}  G=${returns[returns.length-1]?.toFixed(2)}`, 10, 230);
  } else {
    tctx.fillStyle = '#8a93b3'; tctx.font = '11px JetBrains Mono';
    tctx.fillText('Contextual bandit — single step per episode', 10, 30);
    // simple bars
    for (let s = 0; s < env.nStates; s++) {
      const probs = policyProbs(s);
      tctx.fillStyle = '#e6ecff';
      tctx.fillText(`s${s}: π(0)=${probs[0].toFixed(2)} π(1)=${probs[1].toFixed(2)}`, 30, 70 + s * 30);
    }
  }
  tctx.textAlign = 'start';

  // Returns chart
  clearCanvas(rctx);
  drawAxes(rctx, 35, 160, 1050, 130, { xLabel: 'episode', yLabel: 'G' });
  if (returns.length > 1) {
    const ma = [];
    let sum = 0; const win = Math.min(50, Math.max(2, Math.floor(returns.length/10)));
    for (let i = 0; i < returns.length; i++) {
      sum += returns[i];
      if (i >= win) sum -= returns[i - win];
      ma.push(sum / Math.min(i+1, win));
    }
    const lo = Math.min(...returns), hi = Math.max(...returns);
    drawSeries(rctx, returns, 35, 160, 1050, 130, { color: '#7c5cff44', yMin: lo, yMax: hi, lineWidth: 1 });
    drawSeries(rctx, ma, 35, 160, 1050, 130, { color: '#00d4ff', yMin: lo, yMax: hi });
  }
}

function updateStats(last) {
  document.getElementById('s-ep').textContent = episodeCount;
  document.getElementById('s-g').textContent = last !== undefined ? last.toFixed(2) : '—';
  if (returns.length > 0) {
    const tail = returns.slice(-50);
    const avg = tail.reduce((a,b)=>a+b,0) / tail.length;
    const v = tail.reduce((a,b)=>a + (b - avg) * (b - avg), 0) / tail.length;
    document.getElementById('s-avg').textContent = avg.toFixed(2);
    document.getElementById('s-var').textContent = v.toFixed(3);
  }
  document.getElementById('s-b').textContent = baseline.toFixed(2);
  document.getElementById('alpha-v').textContent = parseFloat(document.getElementById('alpha').value).toFixed(3);
  document.getElementById('gamma-v').textContent = parseFloat(document.getElementById('gamma').value).toFixed(2);
  document.getElementById('epf-v').textContent = parseInt(document.getElementById('epf').value);
}

const loop = new Loop(() => {
  const epf = parseInt(document.getElementById('epf').value);
  let last;
  for (let i = 0; i < epf; i++) last = trainEpisode();
  redraw(); updateStats(last);
}, { stepsPerFrame: 1 });

document.getElementById('run').addEventListener('click', () => {
  if (loop.running) { loop.stop(); document.getElementById('run').textContent = '▶ Train'; }
  else { loop.start(); document.getElementById('run').textContent = '⏸ Pause'; }
});
document.getElementById('step').addEventListener('click', () => { const g = trainEpisode(); redraw(); updateStats(g); });
document.getElementById('reset').addEventListener('click', () => { loop.stop(); document.getElementById('run').textContent = '▶ Train'; rebuildEnv(); redraw(); updateStats(); });
document.getElementById('env').addEventListener('change', () => { loop.stop(); document.getElementById('run').textContent = '▶ Train'; rebuildEnv(); redraw(); updateStats(); });
['alpha','gamma','epf'].forEach(id => document.getElementById(id).addEventListener('input', updateStats));

rebuildEnv(); redraw(); updateStats();
})();
