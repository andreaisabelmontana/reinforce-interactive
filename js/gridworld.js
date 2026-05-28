// Shared Gridworld engine used by MDP, DP, and TD pages
'use strict';

// Cell types
const CELL = { EMPTY: 0, WALL: 1, GOAL: 2, START: 3, PIT: 4, CLIFF: 5 };

class Gridworld {
  constructor(rows, cols, opts = {}) {
    this.rows = rows; this.cols = cols;
    this.cells = Array.from({length: rows}, () => new Array(cols).fill(CELL.EMPTY));
    this.start = opts.start || {r: 0, c: 0};
    this.gamma = opts.gamma ?? 0.95;
    this.stepReward = opts.stepReward ?? -0.04;
    this.goalReward = opts.goalReward ?? 1;
    this.pitReward = opts.pitReward ?? -1;
    this.cliffReward = opts.cliffReward ?? -100;
    this.slip = opts.slip ?? 0.0; // probability of slipping perpendicular
    this.cells[this.start.r][this.start.c] = CELL.START;
  }

  inBounds(r, c) { return r >= 0 && r < this.rows && c >= 0 && c < this.cols; }
  isTerminal(r, c) { const t = this.cells[r][c]; return t === CELL.GOAL || t === CELL.PIT; }
  isWall(r, c) { return this.cells[r][c] === CELL.WALL; }
  isCliff(r, c) { return this.cells[r][c] === CELL.CLIFF; }

  // 4 actions: 0=up, 1=right, 2=down, 3=left
  static ACTIONS = [[-1,0],[0,1],[1,0],[0,-1]];
  static ARROW = ['↑','→','↓','←'];

  // For DP: full transition distribution
  transitions(r, c, a) {
    // returns array of {r1, c1, p, reward, done}
    if (this.isTerminal(r, c)) return [{r1:r, c1:c, p:1, reward:0, done:true}];
    const acts = [a, (a+1)%4, (a+3)%4]; // intended + two perpendicular slips
    const probs = [1 - this.slip, this.slip/2, this.slip/2];
    const out = [];
    for (let i = 0; i < acts.length; i++) {
      if (probs[i] === 0) continue;
      const [dr, dc] = Gridworld.ACTIONS[acts[i]];
      let nr = r + dr, nc = c + dc;
      if (!this.inBounds(nr, nc) || this.isWall(nr, nc)) { nr = r; nc = c; }
      let reward = this.stepReward;
      let done = false;
      const ct = this.cells[nr][nc];
      if (ct === CELL.GOAL) { reward = this.goalReward; done = true; }
      else if (ct === CELL.PIT) { reward = this.pitReward; done = true; }
      else if (ct === CELL.CLIFF) {
        // step into cliff: big neg reward, teleport back to start
        reward = this.cliffReward;
        nr = this.start.r; nc = this.start.c;
        done = false;
      }
      out.push({r1: nr, c1: nc, p: probs[i], reward, done});
    }
    return out;
  }

  // Sample one transition (for model-free methods)
  sample(r, c, a) {
    const ts = this.transitions(r, c, a);
    let pick = Math.random();
    for (const t of ts) { pick -= t.p; if (pick <= 0) return t; }
    return ts[ts.length-1];
  }
}

// Draws a gridworld on a canvas
function drawGrid(ctx, gw, opts = {}) {
  const { V, Q, policy, agent, highlight, showValues = false, showPolicy = false, cellSize = 56 } = opts;
  const W = gw.cols * cellSize, H = gw.rows * cellSize;
  ctx.canvas.width !== W && (ctx.canvas.width = W);
  ctx.canvas.height !== H && (ctx.canvas.height = H);
  clearCanvas(ctx, '#050811');

  // Determine value range for coloring
  let vMin = Infinity, vMax = -Infinity;
  if (V) for (let r=0; r<gw.rows; r++) for (let c=0; c<gw.cols; c++) {
    if (V[r][c] < vMin) vMin = V[r][c];
    if (V[r][c] > vMax) vMax = V[r][c];
  }
  if (Q) for (let r=0; r<gw.rows; r++) for (let c=0; c<gw.cols; c++) {
    const m = Math.max(...Q[r][c]);
    if (m < vMin) vMin = m;
    if (m > vMax) vMax = m;
  }
  if (vMin === vMax) { vMin -= 1; vMax += 1; }

  for (let r = 0; r < gw.rows; r++) {
    for (let c = 0; c < gw.cols; c++) {
      const x = c * cellSize, y = r * cellSize;
      const ct = gw.cells[r][c];
      let bg = '#0d1424';
      if (ct === CELL.WALL) bg = '#3a4570';
      else if (ct === CELL.GOAL) bg = '#1f5d4e';
      else if (ct === CELL.PIT) bg = '#5d1f1f';
      else if (ct === CELL.CLIFF) bg = '#3a0a0a';
      else if (ct === CELL.START) bg = '#1a2238';
      else if (V) {
        bg = valueColor(V[r][c], vMin, vMax);
      } else if (Q) {
        bg = valueColor(Math.max(...Q[r][c]), vMin, vMax);
      }
      ctx.fillStyle = bg;
      ctx.fillRect(x+1, y+1, cellSize-2, cellSize-2);

      // Border
      ctx.strokeStyle = '#0a0e1a'; ctx.lineWidth = 1;
      ctx.strokeRect(x+0.5, y+0.5, cellSize, cellSize);

      // Show value text
      if (showValues && ct !== CELL.WALL) {
        const v = V ? V[r][c] : (Q ? Math.max(...Q[r][c]) : 0);
        ctx.fillStyle = '#e6ecff';
        ctx.font = `${Math.max(9, cellSize*0.18)}px JetBrains Mono, monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(v.toFixed(2), x + cellSize/2, y + cellSize/2 + 3);
      }

      // Show policy arrow
      if (showPolicy && policy && ct === CELL.EMPTY || (showPolicy && policy && ct === CELL.START)) {
        const a = policy[r][c];
        if (a !== undefined && a !== -1) {
          ctx.fillStyle = '#00d4ff';
          ctx.font = `bold ${Math.max(14, cellSize*0.35)}px system-ui`;
          ctx.textAlign = 'center';
          const ar = Gridworld.ARROW[a];
          ctx.fillText(ar, x + cellSize/2, y + cellSize*0.85);
        }
      }

      // Cell type icons
      if (ct === CELL.GOAL) {
        ctx.fillStyle = '#2dd4bf';
        ctx.font = `bold ${cellSize*0.45}px system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText('★', x + cellSize/2, y + cellSize*0.6);
      } else if (ct === CELL.PIT) {
        ctx.fillStyle = '#ff5c8a';
        ctx.font = `bold ${cellSize*0.45}px system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText('✖', x + cellSize/2, y + cellSize*0.6);
      } else if (ct === CELL.CLIFF) {
        ctx.fillStyle = '#ff5c8a';
        ctx.font = `${cellSize*0.4}px system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText('↯', x + cellSize/2, y + cellSize*0.6);
      } else if (ct === CELL.START) {
        ctx.strokeStyle = '#7c5cff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x+5, y+5, cellSize-10, cellSize-10);
        ctx.fillStyle = '#7c5cff';
        ctx.font = `bold 10px JetBrains Mono`;
        ctx.textAlign = 'center';
        ctx.fillText('S', x + cellSize/2, y + cellSize*0.95);
      }

      if (highlight && highlight.r === r && highlight.c === c) {
        ctx.strokeStyle = '#ff5c8a';
        ctx.lineWidth = 3;
        ctx.strokeRect(x+2, y+2, cellSize-4, cellSize-4);
      }
    }
  }

  if (agent) {
    const ax = agent.c * cellSize + cellSize/2;
    const ay = agent.r * cellSize + cellSize/2;
    ctx.fillStyle = '#ff5c8a';
    ctx.beginPath();
    ctx.arc(ax, ay, cellSize*0.22, 0, Math.PI*2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// Click-to-edit utility
function bindGridEditor(canvas, gw, getMode, onChange, cellSize = 56) {
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const c = Math.floor(x / cellSize), r = Math.floor(y / cellSize);
    if (!gw.inBounds(r, c)) return;
    const mode = getMode();
    if (mode === 'start') {
      // clear old start
      for (let i=0;i<gw.rows;i++) for (let j=0;j<gw.cols;j++)
        if (gw.cells[i][j] === CELL.START) gw.cells[i][j] = CELL.EMPTY;
      gw.cells[r][c] = CELL.START;
      gw.start = {r, c};
    } else if (mode === 'wall') gw.cells[r][c] = gw.cells[r][c] === CELL.WALL ? CELL.EMPTY : CELL.WALL;
    else if (mode === 'goal') gw.cells[r][c] = gw.cells[r][c] === CELL.GOAL ? CELL.EMPTY : CELL.GOAL;
    else if (mode === 'pit') gw.cells[r][c] = gw.cells[r][c] === CELL.PIT ? CELL.EMPTY : CELL.PIT;
    else if (mode === 'cliff') gw.cells[r][c] = gw.cells[r][c] === CELL.CLIFF ? CELL.EMPTY : CELL.CLIFF;
    else if (mode === 'erase') gw.cells[r][c] = CELL.EMPTY;
    onChange && onChange();
  });
}

// Initialize a default gridworld
function makeDefaultGrid(rows = 5, cols = 7) {
  const gw = new Gridworld(rows, cols, { start: {r: rows-1, c: 0}, slip: 0.1 });
  gw.cells[0][cols-1] = CELL.GOAL;
  if (rows >= 4) {
    // Add a few walls + a pit
    for (let c = 1; c < cols-2; c++) gw.cells[Math.floor(rows/2)][c] = CELL.WALL;
    gw.cells[Math.floor(rows/2)][Math.floor(cols/2)] = CELL.EMPTY; // gap
    gw.cells[1][cols-1] = CELL.PIT;
  }
  return gw;
}

// Make a cliff-walking grid
function makeCliffGrid() {
  const rows = 4, cols = 10;
  const gw = new Gridworld(rows, cols, {
    start: {r: rows-1, c: 0}, gamma: 1.0, stepReward: -1,
    goalReward: 0, cliffReward: -100, slip: 0.0
  });
  // bottom row middle = cliff, last col bottom = goal
  for (let c = 1; c < cols-1; c++) gw.cells[rows-1][c] = CELL.CLIFF;
  gw.cells[rows-1][cols-1] = CELL.GOAL;
  return gw;
}
