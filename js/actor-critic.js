// Compare REINFORCE vs A2C on same env
'use strict';

(function(){

let envCfg;
function makeEnv(name) {
  if (name === 'long') return { N: 10, slip: 0 };
  if (name === 'windy') return { N: 5, slip: 0.2 };
  return { N: 5, slip: 0 };
}

function step(s, a, cfg) {
  let eff = a;
  if (Math.random() < cfg.slip) eff = 1 - a;
  let ns = clamp(s + (eff === 1 ? 1 : -1), 0, cfg.N - 1);
  const r = ns === cfg.N - 1 ? 1 : -0.05;
  return { ns, r, done: ns === cfg.N - 1 };
}

// REINFORCE agent
function makeReinforce(N) {
  return { theta: Array.from({length: N}, () => [0, 0]), baseline: 0, returns: [], nEp: 0 };
}
function trainReinforce(agent, cfg, alpha, gamma) {
  let s = 0;
  const traj = [];
  while (traj.length < 60) {
    const probs = softmax(agent.theta[s]);
    const a = sampleCategorical(probs);
    const { ns, r, done } = step(s, a, cfg);
    traj.push({ s, a, r, probs });
    s = ns;
    if (done) break;
  }
  const G = new Array(traj.length).fill(0);
  let acc = 0;
  for (let i = traj.length - 1; i >= 0; i--) { acc = traj[i].r + gamma * acc; G[i] = acc; }
  agent.baseline += 0.05 * (G[0] - agent.baseline);
  for (let i = 0; i < traj.length; i++) {
    const { s, a, probs } = traj[i];
    const adv = G[i] - agent.baseline;
    for (let aa = 0; aa < 2; aa++) {
      const grad = (aa === a ? 1 : 0) - probs[aa];
      agent.theta[s][aa] += alpha * Math.pow(gamma, i) * adv * grad;
      agent.theta[s][aa] = clamp(agent.theta[s][aa], -20, 20);
    }
  }
  agent.returns.push(G[0]);
  agent.nEp++;
  return G[0];
}

// A2C agent
function makeA2C(N) {
  return { theta: Array.from({length: N}, () => [0, 0]), V: new Array(N).fill(0), returns: [], nEp: 0 };
}
function trainA2C(agent, cfg, alphaA, alphaC, gamma) {
  let s = 0;
  let G = 0, t = 0;
  while (t < 60) {
    const probs = softmax(agent.theta[s]);
    const a = sampleCategorical(probs);
    const { ns, r, done } = step(s, a, cfg);
    const target = done ? r : r + gamma * agent.V[ns];
    const adv = target - agent.V[s];
    // Critic update
    agent.V[s] += alphaC * adv;
    // Actor update
    for (let aa = 0; aa < 2; aa++) {
      const grad = (aa === a ? 1 : 0) - probs[aa];
      agent.theta[s][aa] += alphaA * adv * grad;
      agent.theta[s][aa] = clamp(agent.theta[s][aa], -20, 20);
    }
    G += Math.pow(gamma, t) * r;
    s = ns;
    t++;
    if (done) break;
  }
  agent.returns.push(G);
  agent.nEp++;
  return G;
}

let cfg, rAgent, aAgent;
function reset() {
  cfg = makeEnv(document.getElementById('env').value);
  rAgent = makeReinforce(cfg.N);
  aAgent = makeA2C(cfg.N);
  redraw();
  updateStats();
}

const rpolCtx = setupCanvas(document.getElementById('r-pol'), 540, 160);
const apolCtx = setupCanvas(document.getElementById('a-pol'), 540, 160);
const rchCtx  = setupCanvas(document.getElementById('rchart'), 1100, 220);

function drawPolicyBars(ctx, theta, V, color) {
  clearCanvas(ctx);
  const N = theta.length;
  const barW = 480 / N;
  const top = 20, bot = 130, H = bot - top;
  for (let s = 0; s < N; s++) {
    const probs = softmax(theta[s]);
    const x = 40 + s * barW + barW * 0.1;
    const w = barW * 0.8;
    ctx.fillStyle = '#1a2238'; ctx.fillRect(x, top, w, H);
    ctx.fillStyle = color;
    ctx.fillRect(x, bot - H * probs[1], w, H * probs[1]);
    ctx.fillStyle = '#e6ecff';
    ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'center';
    ctx.fillText(`s${s}`, x + w/2, bot + 14);
    ctx.fillText(probs[1].toFixed(2), x + w/2, top - 4);
    if (V) {
      ctx.fillStyle = '#2dd4bf';
      ctx.fillText(`v=${V[s].toFixed(2)}`, x + w/2, bot + 26);
    }
  }
  ctx.textAlign = 'start';
}

function redraw() {
  drawPolicyBars(rpolCtx, rAgent.theta, null, '#7c5cff');
  drawPolicyBars(apolCtx, aAgent.theta, aAgent.V, '#00d4ff');

  // returns chart
  clearCanvas(rchCtx);
  drawAxes(rchCtx, 35, 200, 1050, 170, { xLabel: 'episode', yLabel: 'G' });
  const all = rAgent.returns.concat(aAgent.returns);
  if (all.length > 1) {
    const lo = Math.min(...all), hi = Math.max(...all);
    // moving averages
    function ma(arr) {
      const out = []; let sum = 0;
      const win = Math.min(50, Math.max(2, Math.floor(arr.length/10)));
      for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
        if (i >= win) sum -= arr[i - win];
        out.push(sum / Math.min(i+1, win));
      }
      return out;
    }
    drawSeries(rchCtx, rAgent.returns, 35, 200, 1050, 170, { color: '#7c5cff33', yMin: lo, yMax: hi, lineWidth: 1 });
    drawSeries(rchCtx, aAgent.returns, 35, 200, 1050, 170, { color: '#00d4ff33', yMin: lo, yMax: hi, lineWidth: 1 });
    drawSeries(rchCtx, ma(rAgent.returns), 35, 200, 1050, 170, { color: '#7c5cff', yMin: lo, yMax: hi, lineWidth: 2 });
    drawSeries(rchCtx, ma(aAgent.returns), 35, 200, 1050, 170, { color: '#00d4ff', yMin: lo, yMax: hi, lineWidth: 2 });
  }
}

function statBlock(prefix, arr, nEp) {
  document.getElementById(`${prefix}-ep`).textContent = nEp;
  if (arr.length > 0) {
    const tail = arr.slice(-50);
    const avg = tail.reduce((a,b)=>a+b,0) / tail.length;
    const v = tail.reduce((a,b)=>a + (b - avg)*(b - avg), 0) / tail.length;
    document.getElementById(`${prefix}-avg`).textContent = avg.toFixed(2);
    document.getElementById(`${prefix}-var`).textContent = v.toFixed(3);
  }
}
function updateStats() {
  statBlock('r', rAgent.returns, rAgent.nEp);
  statBlock('a', aAgent.returns, aAgent.nEp);
  document.getElementById('alphaa-v').textContent = parseFloat(document.getElementById('alphaa').value).toFixed(3);
  document.getElementById('alphac-v').textContent = parseFloat(document.getElementById('alphac').value).toFixed(3);
  document.getElementById('gamma-v').textContent = parseFloat(document.getElementById('gamma').value).toFixed(2);
  document.getElementById('epf-v').textContent = parseInt(document.getElementById('epf').value);
}

const loop = new Loop(() => {
  const alphaA = parseFloat(document.getElementById('alphaa').value);
  const alphaC = parseFloat(document.getElementById('alphac').value);
  const gamma = parseFloat(document.getElementById('gamma').value);
  const epf = parseInt(document.getElementById('epf').value);
  for (let i = 0; i < epf; i++) {
    trainReinforce(rAgent, cfg, alphaA, gamma);
    trainA2C(aAgent, cfg, alphaA, alphaC, gamma);
  }
  redraw(); updateStats();
}, { stepsPerFrame: 1 });

document.getElementById('run').addEventListener('click', () => {
  if (loop.running) { loop.stop(); document.getElementById('run').textContent = '▶ Train Both'; }
  else { loop.start(); document.getElementById('run').textContent = '⏸ Pause'; }
});
document.getElementById('step').addEventListener('click', () => {
  const alphaA = parseFloat(document.getElementById('alphaa').value);
  const alphaC = parseFloat(document.getElementById('alphac').value);
  const gamma = parseFloat(document.getElementById('gamma').value);
  trainReinforce(rAgent, cfg, alphaA, gamma);
  trainA2C(aAgent, cfg, alphaA, alphaC, gamma);
  redraw(); updateStats();
});
document.getElementById('reset').addEventListener('click', () => { loop.stop(); document.getElementById('run').textContent = '▶ Train Both'; reset(); });
document.getElementById('env').addEventListener('change', () => { loop.stop(); document.getElementById('run').textContent = '▶ Train Both'; reset(); });
['alphaa','alphac','gamma','epf'].forEach(id => document.getElementById(id).addEventListener('input', updateStats));

reset();
})();
