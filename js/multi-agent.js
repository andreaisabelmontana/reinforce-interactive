// Axelrod tournament: iterated prisoner's dilemma
'use strict';

(function(){

// Actions: 0 = C (cooperate), 1 = D (defect)
// Payoff: PAY[a1][a2] = score for player 1
const PAY = [
  [[3,3], [0,5]],
  [[5,0], [1,1]]
];

// Each strategy: function(myHistory, oppHistory) -> action
// Each strategy keeps no internal state besides histories
const STRATS = [
  { name: 'Always C', fn: () => 0, color: '#2dd4bf' },
  { name: 'Always D', fn: () => 1, color: '#ff5c8a' },
  { name: 'Tit-for-Tat', fn: (mine, opp) => opp.length === 0 ? 0 : opp[opp.length-1], color: '#00d4ff' },
  { name: 'TFT-2', fn: (mine, opp) => {
      if (opp.length < 2) return 0;
      if (opp[opp.length-1] === 1 && opp[opp.length-2] === 1) return 1;
      return 0;
    }, color: '#7c5cff' },
  { name: 'Grim', fn: (mine, opp) => opp.includes(1) ? 1 : 0, color: '#f59e0b' },
  { name: 'Pavlov', fn: (mine, opp) => {
      if (mine.length === 0) return 0;
      const last = mine[mine.length-1];
      const opLast = opp[opp.length-1];
      const r = PAY[last][opLast][0];
      return r >= 3 ? last : 1 - last;
    }, color: '#a78bfa' },
  { name: 'Random', fn: () => Math.random() < 0.5 ? 0 : 1, color: '#9ca3af' },
  { name: 'Joss', fn: (mine, opp) => {
      if (opp.length === 0) return 0;
      const tft = opp[opp.length-1];
      if (tft === 0 && Math.random() < 0.1) return 1;
      return tft;
    }, color: '#fb7185' },
  { name: 'Suspicious TFT', fn: (mine, opp) => opp.length === 0 ? 1 : opp[opp.length-1], color: '#fbbf24' },
];

function noisify(a, p) { return Math.random() < p ? 1 - a : a; }

function playMatch(stratA, stratB, rounds, noise = 0) {
  let hA = [], hB = [];
  let sA = 0, sB = 0;
  for (let i = 0; i < rounds; i++) {
    let a = stratA.fn(hA, hB);
    let b = stratB.fn(hB, hA);
    a = noisify(a, noise); b = noisify(b, noise);
    const [pa, pb] = PAY[a][b];
    sA += pa; sB += pb;
    hA.push(a); hB.push(b);
  }
  return { sA, sB, hA, hB };
}

let board = []; // total scores per strategy
let pair = []; // pair[i][j] = avg score for i vs j
let population = STRATS.map(() => 1); // for evolutionary mode
let popHist = []; // [[fractions...]]
let lastDuel = null;

function runTournament() {
  const rounds = parseInt(document.getElementById('rounds').value);
  const noise = parseFloat(document.getElementById('noise').value);
  const N = STRATS.length;
  board = new Array(N).fill(0);
  pair = Array.from({length: N}, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const { sA, sB } = playMatch(STRATS[i], STRATS[j], rounds, noise);
      board[i] += sA;
      pair[i][j] = sA / rounds;
    }
  }
  drawBoard(); drawHeat();
}

function runEvolution() {
  const G = parseInt(document.getElementById('evo').value);
  const rounds = parseInt(document.getElementById('rounds').value);
  const noise = parseFloat(document.getElementById('noise').value);
  const N = STRATS.length;
  population = new Array(N).fill(10);
  popHist = [population.slice()];
  for (let g = 0; g < G; g++) {
    // weighted score where each strat is weighted by pop fractions
    const totalPop = population.reduce((a,b)=>a+b,0);
    const frac = population.map(p => p / totalPop);
    const score = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const { sA } = playMatch(STRATS[i], STRATS[j], rounds, noise);
        score[i] += (sA / rounds) * frac[j];
      }
    }
    // Replicator: pop_i ← pop_i * (score_i / mean_score)
    const meanScore = score.reduce((s,v,i) => s + v * frac[i], 0);
    const next = new Array(N);
    let totalNext = 0;
    for (let i = 0; i < N; i++) {
      next[i] = population[i] * (score[i] / (meanScore + 1e-6));
      if (next[i] < 0.01) next[i] = 0.01;
      totalNext += next[i];
    }
    // renormalise to keep total constant
    for (let i = 0; i < N; i++) next[i] = next[i] * 100 / totalNext;
    population = next;
    popHist.push(population.slice());
  }
  drawPop();
}

// ---------- Rendering ----------
function drawBoard() {
  const N = STRATS.length;
  const ranks = board.map((s, i) => ({ i, s })).sort((a, b) => b.s - a.s);
  const t = document.getElementById('board');
  // clear all but header
  while (t.rows.length > 1) t.deleteRow(1);
  for (let r = 0; r < N; r++) {
    const row = t.insertRow();
    row.insertCell().textContent = (r+1);
    const c2 = row.insertCell();
    c2.innerHTML = `<span style="display:inline-block;width:10px;height:10px;background:${STRATS[ranks[r].i].color};border-radius:2px;margin-right:6px"></span>${STRATS[ranks[r].i].name}`;
    const c3 = row.insertCell(); c3.textContent = ranks[r].s.toFixed(0); c3.style.textAlign = 'right';
    const c4 = row.insertCell(); c4.textContent = (ranks[r].s / (N * parseFloat(document.getElementById('rounds').value))).toFixed(2); c4.style.textAlign = 'right';
  }
}

const heatCtx = setupCanvas(document.getElementById('heat'), 540, 400);
function drawHeat() {
  clearCanvas(heatCtx);
  const N = STRATS.length;
  const lblW = 90;
  const cell = (540 - lblW - 20) / N;
  let lo = Infinity, hi = -Infinity;
  for (let i=0;i<N;i++) for (let j=0;j<N;j++) {
    if (pair[i][j] < lo) lo = pair[i][j]; if (pair[i][j] > hi) hi = pair[i][j];
  }
  if (lo === hi) { lo -= 1; hi += 1; }
  for (let i = 0; i < N; i++) {
    heatCtx.fillStyle = STRATS[i].color;
    heatCtx.font = '10px JetBrains Mono';
    heatCtx.textAlign = 'right';
    heatCtx.fillText(STRATS[i].name, lblW - 5, 30 + i * cell + cell/2 + 3);
    heatCtx.save();
    heatCtx.translate(lblW + i * cell + cell/2, 22);
    heatCtx.rotate(-Math.PI / 4);
    heatCtx.textAlign = 'left';
    heatCtx.fillText(STRATS[i].name, 0, 0);
    heatCtx.restore();
    for (let j = 0; j < N; j++) {
      heatCtx.fillStyle = valueColor(pair[i][j], lo, hi);
      heatCtx.fillRect(lblW + j * cell, 30 + i * cell, cell - 1, cell - 1);
      heatCtx.fillStyle = '#fff';
      heatCtx.font = `${Math.max(8, cell*0.25)}px JetBrains Mono`;
      heatCtx.textAlign = 'center';
      heatCtx.fillText(pair[i][j].toFixed(2), lblW + j * cell + cell/2, 30 + i * cell + cell/2 + 3);
    }
  }
  heatCtx.textAlign = 'start';
}

function fillStratSelects() {
  for (const id of ['p1','p2']) {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    STRATS.forEach((s, i) => {
      const opt = document.createElement('option');
      opt.value = i; opt.textContent = s.name;
      sel.appendChild(opt);
    });
  }
  document.getElementById('p1').value = 2; // TfT
  document.getElementById('p2').value = 1; // Always D
}

const duelCtx = setupCanvas(document.getElementById('duel-canvas'), 1100, 120);
function drawDuel(match) {
  clearCanvas(duelCtx);
  if (!match) return;
  const W = 1100, H = 120;
  const cellW = Math.max(3, (W - 40) / match.hA.length);
  duelCtx.fillStyle = '#8a93b3'; duelCtx.font = '11px JetBrains Mono';
  duelCtx.fillText('P1:', 5, 30);
  duelCtx.fillText('P2:', 5, 80);
  for (let i = 0; i < match.hA.length; i++) {
    duelCtx.fillStyle = match.hA[i] === 0 ? '#2dd4bf' : '#ff5c8a';
    duelCtx.fillRect(35 + i * cellW, 18, cellW - 0.5, 28);
    duelCtx.fillStyle = match.hB[i] === 0 ? '#2dd4bf' : '#ff5c8a';
    duelCtx.fillRect(35 + i * cellW, 66, cellW - 0.5, 28);
  }
  duelCtx.fillStyle = '#8a93b3';
  duelCtx.fillText(`${STRATS[match.i].name} (${match.sA}) vs ${STRATS[match.j].name} (${match.sB})`, 35, 110);
}

const popCtx = setupCanvas(document.getElementById('pop'), 1100, 220);
function drawPop() {
  clearCanvas(popCtx);
  if (popHist.length < 2) {
    popCtx.fillStyle = '#8a93b3'; popCtx.font = '11px JetBrains Mono';
    popCtx.fillText('Set "Evolutionary rounds" > 0 and click Run Tournament.', 35, 110);
    return;
  }
  const N = STRATS.length, T = popHist.length;
  drawAxes(popCtx, 35, 200, 1050, 170, { xLabel: 'generation', yLabel: '% pop' });
  const W = 1050, H = 170;
  // stacked-line: convert to fractions
  const fracs = popHist.map(p => {
    const s = p.reduce((a,b)=>a+b,0);
    return p.map(v => v / s);
  });
  // cumulative bottom
  for (let i = 0; i < N; i++) {
    popCtx.strokeStyle = STRATS[i].color; popCtx.lineWidth = 2;
    popCtx.beginPath();
    for (let t = 0; t < T; t++) {
      const x = 35 + (t / (T-1)) * W;
      const y = 200 - fracs[t][i] * H;
      if (t === 0) popCtx.moveTo(x, y); else popCtx.lineTo(x, y);
    }
    popCtx.stroke();
  }
  // legend
  let lx = 35; popCtx.font = '10px JetBrains Mono'; popCtx.textAlign = 'start';
  for (let i = 0; i < N; i++) {
    popCtx.fillStyle = STRATS[i].color;
    popCtx.fillRect(lx, 6, 9, 9);
    popCtx.fillStyle = '#e6ecff';
    popCtx.fillText(STRATS[i].name, lx + 13, 14);
    lx += STRATS[i].name.length * 7 + 35;
  }
}

function updateStats() {
  document.getElementById('rounds-v').textContent = document.getElementById('rounds').value;
  document.getElementById('noise-v').textContent = parseFloat(document.getElementById('noise').value).toFixed(2);
  document.getElementById('evo-v').textContent = document.getElementById('evo').value;
}

document.getElementById('run').addEventListener('click', () => {
  runTournament();
  const evo = parseInt(document.getElementById('evo').value);
  if (evo > 0) runEvolution();
});
document.getElementById('watch-duel').addEventListener('click', () => {
  const i = parseInt(document.getElementById('p1').value);
  const j = parseInt(document.getElementById('p2').value);
  const rounds = parseInt(document.getElementById('rounds').value);
  const noise = parseFloat(document.getElementById('noise').value);
  const m = playMatch(STRATS[i], STRATS[j], rounds, noise);
  lastDuel = { ...m, i, j };
  drawDuel(lastDuel);
  document.getElementById('duel-info').textContent =
    `${STRATS[i].name} scored ${m.sA} (avg ${(m.sA/rounds).toFixed(2)}) ・ ${STRATS[j].name} scored ${m.sB} (avg ${(m.sB/rounds).toFixed(2)})`;
});
document.getElementById('reset').addEventListener('click', () => {
  board = []; pair = []; popHist = []; lastDuel = null;
  document.getElementById('board').innerHTML = `<tr><th style="text-align:left;color:#8a93b3;padding:0.4rem 0">#</th><th style="text-align:left;color:#8a93b3">Strategy</th><th style="text-align:right;color:#8a93b3">Total</th><th style="text-align:right;color:#8a93b3">Avg/match</th></tr>`;
  clearCanvas(heatCtx); clearCanvas(duelCtx); clearCanvas(popCtx);
});
['rounds','noise','evo'].forEach(id => document.getElementById(id).addEventListener('input', updateStats));

fillStratSelects();
updateStats();
runTournament();
})();
