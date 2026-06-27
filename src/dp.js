// Dynamic programming on a known Gridworld MDP.
// Value iteration and policy iteration, extracted from the DP demo (dp.html).
// Operates on any Gridworld via its transitions() model.

function mkGrid(gw, fill) {
  return Array.from({ length: gw.rows }, () => new Array(gw.cols).fill(fill));
}

// Expected one-step value of taking action a in (r,c) under value function V.
export function actionValue(gw, V, r, c, a) {
  const ts = gw.transitions(r, c, a);
  let q = 0;
  for (const t of ts) q += t.p * (t.reward + gw.gamma * V[t.r1][t.c1]);
  return q;
}

// Greedy policy w.r.t. V. Terminal/wall cells get action -1.
export function greedyPolicy(gw, V) {
  const p = mkGrid(gw, 0);
  for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
    if (gw.isWall(r, c) || gw.isTerminal(r, c)) { p[r][c] = -1; continue; }
    let best = -Infinity, bestA = 0;
    for (let a = 0; a < 4; a++) {
      const q = actionValue(gw, V, r, c, a);
      if (q > best) { best = q; bestA = a; }
    }
    p[r][c] = bestA;
  }
  return p;
}

// Value iteration: sweep the Bellman optimality backup until ||ΔV||∞ < tol.
// Returns { V, policy, iters, delta }.
export function valueIteration(gw, { tol = 1e-6, maxIters = 1000 } = {}) {
  let V = mkGrid(gw, 0);
  let iters = 0, delta = Infinity;
  while (iters < maxIters) {
    let maxD = 0;
    const Vn = mkGrid(gw, 0);
    for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
      if (gw.isWall(r, c) || gw.isTerminal(r, c)) { Vn[r][c] = 0; continue; }
      let best = -Infinity;
      for (let a = 0; a < 4; a++) {
        const q = actionValue(gw, V, r, c, a);
        if (q > best) best = q;
      }
      Vn[r][c] = best;
      maxD = Math.max(maxD, Math.abs(best - V[r][c]));
    }
    V = Vn;
    iters++;
    delta = maxD;
    if (maxD < tol) break;
  }
  return { V, policy: greedyPolicy(gw, V), iters, delta };
}

// One in-place-style policy-evaluation sweep for a fixed policy.
function policyEvalSweep(gw, V, policy) {
  let maxD = 0;
  const Vn = mkGrid(gw, 0);
  for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
    if (gw.isWall(r, c) || gw.isTerminal(r, c)) { Vn[r][c] = 0; continue; }
    const a = policy[r][c];
    Vn[r][c] = actionValue(gw, V, r, c, a);
    maxD = Math.max(maxD, Math.abs(Vn[r][c] - V[r][c]));
  }
  for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) V[r][c] = Vn[r][c];
  return maxD;
}

// Policy iteration: alternate full policy evaluation and greedy improvement
// until the policy is stable. Returns { V, policy, iters }.
export function policyIteration(gw, { tol = 1e-6, evalSweeps = 200, maxIters = 200 } = {}) {
  let V = mkGrid(gw, 0);
  let policy = greedyPolicy(gw, V);
  let iters = 0;
  while (iters < maxIters) {
    // policy evaluation to (near) convergence
    for (let i = 0; i < evalSweeps; i++) {
      if (policyEvalSweep(gw, V, policy) < tol) break;
    }
    // policy improvement
    const newPolicy = greedyPolicy(gw, V);
    let stable = true;
    for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
      if (policy[r][c] !== newPolicy[r][c]) stable = false;
    }
    policy = newPolicy;
    iters++;
    if (stable) break;
  }
  return { V, policy, iters };
}
