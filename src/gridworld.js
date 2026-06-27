// Gridworld MDP — the shared environment behind the MDP, DP and TD demos.
// Pure model: states (r,c), 4 actions, full transition distribution + sampling.
// No canvas / DOM here; the browser drawing helpers live in js/gridworld.js.

export const CELL = { EMPTY: 0, WALL: 1, GOAL: 2, START: 3, PIT: 4, CLIFF: 5 };

// 4 actions: 0=up, 1=right, 2=down, 3=left
export const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];
export const ARROW = ['↑', '→', '↓', '←'];

export class Gridworld {
  constructor(rows, cols, opts = {}) {
    this.rows = rows;
    this.cols = cols;
    this.cells = Array.from({ length: rows }, () => new Array(cols).fill(CELL.EMPTY));
    this.start = opts.start || { r: 0, c: 0 };
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

  // Full transition distribution for DP.
  // Returns array of {r1, c1, p, reward, done}.
  transitions(r, c, a) {
    if (this.isTerminal(r, c)) return [{ r1: r, c1: c, p: 1, reward: 0, done: true }];
    const acts = [a, (a + 1) % 4, (a + 3) % 4]; // intended + two perpendicular slips
    const probs = [1 - this.slip, this.slip / 2, this.slip / 2];
    const out = [];
    for (let i = 0; i < acts.length; i++) {
      if (probs[i] === 0) continue;
      const [dr, dc] = ACTIONS[acts[i]];
      let nr = r + dr, nc = c + dc;
      if (!this.inBounds(nr, nc) || this.isWall(nr, nc)) { nr = r; nc = c; }
      let reward = this.stepReward;
      let done = false;
      const ct = this.cells[nr][nc];
      if (ct === CELL.GOAL) { reward = this.goalReward; done = true; }
      else if (ct === CELL.PIT) { reward = this.pitReward; done = true; }
      else if (ct === CELL.CLIFF) {
        // step into cliff: big negative reward, teleport back to start, not terminal
        reward = this.cliffReward;
        nr = this.start.r; nc = this.start.c;
        done = false;
      }
      out.push({ r1: nr, c1: nc, p: probs[i], reward, done });
    }
    return out;
  }

  // Sample one transition (for model-free TD/MC). rng is an RNG instance.
  sample(r, c, a, rng) {
    const ts = this.transitions(r, c, a);
    let pick = rng ? rng.random() : Math.random();
    for (const t of ts) { pick -= t.p; if (pick <= 0) return t; }
    return ts[ts.length - 1];
  }
}

// Static class fields kept for parity with the browser script's Gridworld.ACTIONS.
Gridworld.ACTIONS = ACTIONS;
Gridworld.ARROW = ARROW;

// ---- factory grids (faithful to the demos) ----

export function makeDefaultGrid(rows = 5, cols = 7) {
  const gw = new Gridworld(rows, cols, { start: { r: rows - 1, c: 0 }, slip: 0.1 });
  gw.cells[0][cols - 1] = CELL.GOAL;
  if (rows >= 4) {
    for (let c = 1; c < cols - 2; c++) gw.cells[Math.floor(rows / 2)][c] = CELL.WALL;
    gw.cells[Math.floor(rows / 2)][Math.floor(cols / 2)] = CELL.EMPTY; // gap
    gw.cells[1][cols - 1] = CELL.PIT;
  }
  return gw;
}

export function makeCliffGrid() {
  const rows = 4, cols = 10;
  const gw = new Gridworld(rows, cols, {
    start: { r: rows - 1, c: 0 }, gamma: 1.0, stepReward: -1,
    goalReward: 0, cliffReward: -100, slip: 0.0,
  });
  for (let c = 1; c < cols - 1; c++) gw.cells[rows - 1][c] = CELL.CLIFF;
  gw.cells[rows - 1][cols - 1] = CELL.GOAL;
  return gw;
}

// A tiny deterministic grid handy for tests: a 3x3 with the goal in a corner,
// a small step penalty, no slip, so the optimal value/policy are easy to verify.
export function makeTinyGrid() {
  const gw = new Gridworld(3, 3, {
    start: { r: 2, c: 0 }, gamma: 0.9, stepReward: -0.04,
    goalReward: 1, slip: 0.0,
  });
  gw.cells[0][2] = CELL.GOAL;
  return gw;
}
