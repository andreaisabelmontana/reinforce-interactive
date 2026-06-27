// Temporal-difference control on a Gridworld: Q-learning, SARSA, Expected SARSA.
// Model-free — learns from sampled transitions only. Extracted from the TD demo.

import { RNG, argmaxTies, argmax } from './rng.js';

function mkQ(gw) {
  return Array.from({ length: gw.rows }, () =>
    Array.from({ length: gw.cols }, () => [0, 0, 0, 0]));
}

function epsGreedy(Q, r, c, eps, rng) {
  if (rng.random() < eps) return rng.int(4);
  return argmaxTies(Q[r][c], rng);
}

// Run one episode of TD control, mutating Q in place. Returns { G, steps }.
// alg ∈ {'qlearn','sarsa','esarsa'}.
export function trainEpisode(gw, Q, rng, { alg = 'qlearn', alpha = 0.5, eps = 0.1, maxSteps = 500 } = {}) {
  const gamma = gw.gamma;
  let s = { ...gw.start };
  let a = epsGreedy(Q, s.r, s.c, eps, rng);
  let G = 0, steps = 0;

  while (steps < maxSteps) {
    const t = gw.sample(s.r, s.c, a, rng);
    const sNext = { r: t.r1, c: t.c1 };
    let aNext = -1;
    if (!t.done) aNext = epsGreedy(Q, sNext.r, sNext.c, eps, rng);

    if (alg === 'qlearn') {
      const target = t.done ? t.reward : t.reward + gamma * Math.max(...Q[sNext.r][sNext.c]);
      Q[s.r][s.c][a] += alpha * (target - Q[s.r][s.c][a]);
    } else if (alg === 'sarsa') {
      const target = t.done ? t.reward : t.reward + gamma * Q[sNext.r][sNext.c][aNext];
      Q[s.r][s.c][a] += alpha * (target - Q[s.r][s.c][a]);
    } else if (alg === 'esarsa') {
      let expected = 0;
      if (!t.done) {
        const qs = Q[sNext.r][sNext.c];
        const greedy = argmaxTies(qs, rng);
        for (let i = 0; i < 4; i++) {
          const p = (i === greedy ? 1 - eps : 0) + eps / 4;
          expected += p * qs[i];
        }
      }
      const target = t.done ? t.reward : t.reward + gamma * expected;
      Q[s.r][s.c][a] += alpha * (target - Q[s.r][s.c][a]);
    }

    G += Math.pow(gamma, steps) * t.reward;
    steps++;
    s = sNext;
    a = aNext;
    if (t.done) break;
    if (a < 0) a = epsGreedy(Q, s.r, s.c, eps, rng);
  }
  return { G, steps };
}

// Train for nEpisodes, returning the final Q and the per-episode discounted returns.
export function train(gw, { alg = 'qlearn', alpha = 0.5, eps = 0.1, episodes = 500, seed = 1, maxSteps = 500 } = {}) {
  const rng = new RNG(seed);
  const Q = mkQ(gw);
  const returns = [];
  for (let e = 0; e < episodes; e++) {
    returns.push(trainEpisode(gw, Q, rng, { alg, alpha, eps, maxSteps }).G);
  }
  return { Q, returns };
}

// Greedy policy derived from Q (action -1 on wall/terminal/cliff cells).
export function policyFromQ(gw, Q) {
  const p = Array.from({ length: gw.rows }, () => new Array(gw.cols).fill(-1));
  for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
    if (gw.isWall(r, c) || gw.isTerminal(r, c) || gw.isCliff(r, c)) continue;
    p[r][c] = argmax(Q[r][c]);
  }
  return p;
}

// Roll out a (deterministic, greedy) policy from start and return the
// undiscounted return — used to compare a learned policy against random.
export function rolloutReturn(gw, policy, rng, { maxSteps = 500 } = {}) {
  let s = { ...gw.start };
  let total = 0, steps = 0;
  while (steps < maxSteps) {
    if (gw.isTerminal(s.r, s.c)) break;
    let a = policy[s.r][s.c];
    if (a === undefined || a < 0) a = rng.int(4);
    const t = gw.sample(s.r, s.c, a, rng);
    total += t.reward;
    s = { r: t.r1, c: t.c1 };
    steps++;
    if (t.done) break;
  }
  return total;
}

// Average return of the uniform-random policy from start (baseline for tests).
export function randomReturn(gw, rng, { episodes = 200, maxSteps = 500 } = {}) {
  let sum = 0;
  for (let e = 0; e < episodes; e++) {
    let s = { ...gw.start };
    let steps = 0;
    while (steps < maxSteps) {
      if (gw.isTerminal(s.r, s.c)) break;
      const a = rng.int(4);
      const t = gw.sample(s.r, s.c, a, rng);
      sum += t.reward;
      s = { r: t.r1, c: t.c1 };
      steps++;
      if (t.done) break;
    }
  }
  return sum / episodes;
}
