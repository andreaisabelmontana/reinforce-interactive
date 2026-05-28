// MC + TD learning on gridworld (cliff walk by default)
'use strict';

(function(){

let gw;
let Q;     // [r][c][a]
let cellSize = 56;
let qctx, tctx, rctx;
let lastTrajectory = [];
let returns = [];
let episodeCount = 0;

function makeFrozen() {
  const rows = 4, cols = 4;
  const gw = new Gridworld(rows, cols, {
    start:{r:0,c:0}, gamma: 0.99, stepReward: -0.01,
    goalReward: 1, pitReward: -1, slip: 0.2
  });
  gw.cells[3][3] = CELL.GOAL;
  gw.cells[1][1] = CELL.PIT;
  gw.cells[1][3] = CELL.PIT;
  gw.cells[2][3] = CELL.PIT;
  gw.cells[3][0] = CELL.PIT;
  return gw;
}

function buildEnv(name) {
  if (name === 'cliff') gw = makeCliffGrid();
  else if (name === 'frozen') gw = makeFrozen();
  else gw = makeDefaultGrid(5, 7);

  // set common controls
  syncParams();

  Q = Array.from({length: gw.rows}, () =>
    Array.from({length: gw.cols}, () => [0,0,0,0]));
  episodeCount = 0; returns = []; lastTrajectory = [];

  qctx = setupCanvas(document.getElementById('qgrid'), gw.cols*cellSize, gw.rows*cellSize);
  tctx = setupCanvas(document.getElementById('traj'),  gw.cols*cellSize, gw.rows*cellSize);
  rctx = setupCanvas(document.getElementById('rchart'), 1000, 180);
  redraw();
  updateStats({});
}

function syncParams() {
  gw.gamma = parseFloat(document.getElementById('gamma').value);
  document.getElementById('alpha-v').textContent = parseFloat(document.getElementById('alpha').value).toFixed(2);
  document.getElementById('eps-v').textContent = parseFloat(document.getElementById('eps').value).toFixed(2);
  document.getElementById('gamma-v').textContent = gw.gamma.toFixed(2);
}

function pickEpsGreedy(r, c, eps) {
  if (Math.random() < eps) return Math.floor(Math.random()*4);
  return argmaxTies(Q[r][c]);
}

function trainEpisode() {
  const alg = document.getElementById('alg').value;
  const alpha = parseFloat(document.getElementById('alpha').value);
  const eps   = parseFloat(document.getElementById('eps').value);
  const gamma = gw.gamma;

  let s = {...gw.start};
  let a = pickEpsGreedy(s.r, s.c, eps);
  let G = 0, steps = 0;
  const traj = [{...s}];
  const episode = []; // for MC

  while (steps < 500) {
    const t = gw.sample(s.r, s.c, a);
    let aNext = -1;
    const sNext = {r: t.r1, c: t.c1};
    if (!t.done) aNext = pickEpsGreedy(sNext.r, sNext.c, eps);

    if (alg === 'qlearn') {
      const target = t.done ? t.reward : t.reward + gamma * Math.max(...Q[sNext.r][sNext.c]);
      Q[s.r][s.c][a] += alpha * (target - Q[s.r][s.c][a]);
    } else if (alg === 'sarsa') {
      const target = t.done ? t.reward : t.reward + gamma * Q[sNext.r][sNext.c][aNext];
      Q[s.r][s.c][a] += alpha * (target - Q[s.r][s.c][a]);
    } else if (alg === 'esarsa') {
      // Expected SARSA with ε-greedy policy
      let expected = 0;
      if (!t.done) {
        const qs = Q[sNext.r][sNext.c];
        const greedy = argmaxTies(qs);
        for (let i = 0; i < 4; i++) {
          const p = (i === greedy ? 1 - eps : 0) + eps / 4;
          expected += p * qs[i];
        }
      }
      const target = t.done ? t.reward : t.reward + gamma * expected;
      Q[s.r][s.c][a] += alpha * (target - Q[s.r][s.c][a]);
    } else if (alg === 'mc') {
      episode.push({s, a, r: t.reward});
    }

    G += Math.pow(gamma, steps) * t.reward;
    traj.push({...sNext});
    steps++;
    s = sNext; a = aNext;
    if (t.done) break;
    if (a < 0) a = pickEpsGreedy(s.r, s.c, eps);
  }

  // MC update at episode end (every-visit)
  if (alg === 'mc') {
    let Gt = 0;
    for (let i = episode.length - 1; i >= 0; i--) {
      const { s: si, a: ai, r: ri } = episode[i];
      Gt = ri + gamma * Gt;
      Q[si.r][si.c][ai] += alpha * (Gt - Q[si.r][si.c][ai]);
    }
  }

  episodeCount++;
  returns.push(G);
  lastTrajectory = traj;
  return { G, steps };
}

function policyFromQ() {
  const p = Array.from({length: gw.rows}, () => new Array(gw.cols).fill(-1));
  for (let r=0; r<gw.rows; r++) for (let c=0; c<gw.cols; c++) {
    if (gw.isWall(r,c) || gw.isTerminal(r,c) || gw.isCliff(r,c)) continue;
    p[r][c] = argmax(Q[r][c]);
  }
  return p;
}

function redraw() {
  const policy = policyFromQ();
  drawGrid(qctx, gw, { Q, policy, showPolicy: true, cellSize });
  // traj canvas
  drawGrid(tctx, gw, { Q, cellSize });
  // overlay trajectory
  if (lastTrajectory.length > 1) {
    tctx.strokeStyle = '#ff5c8a';
    tctx.lineWidth = 3;
    tctx.beginPath();
    for (let i = 0; i < lastTrajectory.length; i++) {
      const x = lastTrajectory[i].c * cellSize + cellSize/2;
      const y = lastTrajectory[i].r * cellSize + cellSize/2;
      if (i === 0) tctx.moveTo(x, y); else tctx.lineTo(x, y);
    }
    tctx.stroke();
    // dot at end
    const last = lastTrajectory[lastTrajectory.length-1];
    tctx.fillStyle = '#ff5c8a';
    tctx.beginPath();
    tctx.arc(last.c*cellSize+cellSize/2, last.r*cellSize+cellSize/2, 6, 0, Math.PI*2);
    tctx.fill();
  }
  // returns chart
  clearCanvas(rctx);
  drawAxes(rctx, 35, 160, 950, 130, { xLabel: 'episode', yLabel: 'G' });
  if (returns.length > 1) {
    // also draw a moving average
    const ma = [];
    let sum = 0;
    const win = Math.max(1, Math.min(50, Math.floor(returns.length/4)));
    for (let i = 0; i < returns.length; i++) {
      sum += returns[i];
      if (i >= win) sum -= returns[i - win];
      ma.push(sum / Math.min(i+1, win));
    }
    const lo = Math.min(...returns), hi = Math.max(...returns);
    drawSeries(rctx, returns, 35, 160, 950, 130, { color: '#7c5cff66', yMin: lo, yMax: hi, lineWidth: 1 });
    drawSeries(rctx, ma, 35, 160, 950, 130, { color: '#00d4ff', yMin: lo, yMax: hi });
  }
}

function updateStats(last) {
  document.getElementById('s-ep').textContent = episodeCount;
  document.getElementById('s-g').textContent = last.G !== undefined ? last.G.toFixed(2) : '—';
  document.getElementById('s-steps').textContent = last.steps ?? '—';
  if (returns.length > 0) {
    const tail = returns.slice(-100);
    const avg = tail.reduce((a,b)=>a+b,0) / tail.length;
    document.getElementById('s-avg').textContent = avg.toFixed(2);
  }
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
document.getElementById('step').addEventListener('click', () => {
  const last = trainEpisode(); redraw(); updateStats(last);
});
document.getElementById('reset').addEventListener('click', () => {
  buildEnv(document.getElementById('env').value);
});
['alpha','eps','gamma','epf'].forEach(id => {
  document.getElementById(id).addEventListener('input', syncParams);
});
document.getElementById('alg').addEventListener('change', () => {
  buildEnv(document.getElementById('env').value);
});
document.getElementById('env').addEventListener('change', () => {
  buildEnv(document.getElementById('env').value);
});

// kickoff
buildEnv('cliff');
})();
