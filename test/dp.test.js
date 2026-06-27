import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Gridworld, CELL, makeTinyGrid } from '../src/gridworld.js';
import { valueIteration, policyIteration, actionValue } from '../src/dp.js';

// Bellman optimality residual: for every non-terminal state,
// V(s) should equal max_a Q(s,a).
function bellmanResidual(gw, V) {
  let maxRes = 0;
  for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
    if (gw.isWall(r, c) || gw.isTerminal(r, c)) continue;
    let best = -Infinity;
    for (let a = 0; a < 4; a++) best = Math.max(best, actionValue(gw, V, r, c, a));
    maxRes = Math.max(maxRes, Math.abs(V[r][c] - best));
  }
  return maxRes;
}

test('value iteration converges and satisfies the Bellman optimality equation', () => {
  const gw = makeTinyGrid();
  const { V, policy, iters } = valueIteration(gw, { tol: 1e-9 });
  assert.ok(iters > 1 && iters < 1000);
  assert.ok(bellmanResidual(gw, V) < 1e-6);
  // start value is positive (goal is reachable, reward 1 dominates step penalty)
  assert.ok(V[gw.start.r][gw.start.c] > 0);
  // optimal policy from the start heads toward the goal at (0,2):
  // from (2,0) the best first move is up or right, both strictly improving.
  const a0 = policy[gw.start.r][gw.start.c];
  assert.ok(a0 === 0 || a0 === 1, `start action ${a0}`);
});

test('value iteration recovers the analytic optimum on a deterministic corridor', () => {
  // 1x4 corridor, goal at the right end, no slip, gamma 0.9, step reward -0.04.
  const gw = new Gridworld(1, 4, { start: { r: 0, c: 0 }, gamma: 0.9, stepReward: -0.04, slip: 0, goalReward: 1 });
  gw.cells[0][3] = CELL.GOAL;
  const { V, policy } = valueIteration(gw, { tol: 1e-10 });
  // Optimal: always go right. Value at cell c (distance d=3-c from goal):
  // V = sum_{k=0}^{d-1} gamma^k*(-0.04) + gamma^d * 1  ... last step pays goalReward.
  const g = 0.9, sr = -0.04;
  for (let c = 0; c <= 2; c++) {
    const d = 3 - c;
    let expected = 0;
    for (let k = 0; k < d - 1; k++) expected += Math.pow(g, k) * sr;
    expected += Math.pow(g, d - 1) * 1; // final move into goal pays +1
    assert.ok(Math.abs(V[0][c] - expected) < 1e-6, `c=${c} V=${V[0][c]} expected=${expected}`);
    assert.equal(policy[0][c], 1); // action 1 = right
  }
});

test('policy iteration agrees with value iteration', () => {
  const gw = makeTinyGrid();
  const vi = valueIteration(gw, { tol: 1e-10 });
  const pi = policyIteration(gw, { tol: 1e-10 });
  // values match
  for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
    assert.ok(Math.abs(vi.V[r][c] - pi.V[r][c]) < 1e-4, `(${r},${c}) ${vi.V[r][c]} vs ${pi.V[r][c]}`);
  }
  // greedy policies match on non-terminal cells
  for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
    if (gw.isWall(r, c) || gw.isTerminal(r, c)) continue;
    assert.equal(vi.policy[r][c], pi.policy[r][c], `policy mismatch at (${r},${c})`);
  }
  assert.ok(pi.iters <= vi.iters); // PI converges in fewer outer iterations
});
