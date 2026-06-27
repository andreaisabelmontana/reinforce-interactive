import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeTinyGrid } from '../src/gridworld.js';
import { valueIteration } from '../src/dp.js';
import { train, policyFromQ, rolloutReturn, randomReturn } from '../src/td.js';
import { RNG } from '../src/rng.js';

test('Q-learning beats the random policy on the tiny gridworld', () => {
  const gw = makeTinyGrid();
  const { Q } = train(gw, { alg: 'qlearn', alpha: 0.5, eps: 0.1, episodes: 1500, seed: 3 });
  const policy = policyFromQ(gw, Q);

  const evalRng = new RNG(777);
  let learned = 0;
  const E = 300;
  for (let i = 0; i < E; i++) learned += rolloutReturn(gw, policy, evalRng);
  learned /= E;

  const baseline = randomReturn(gw, new RNG(777), { episodes: E });
  assert.ok(learned > baseline, `learned ${learned} should beat random ${baseline}`);
});

test('Q-learning approaches the DP-optimal policy at the start state', () => {
  const gw = makeTinyGrid();
  const { policy: optPolicy } = valueIteration(gw, { tol: 1e-9 });
  const { Q } = train(gw, { alg: 'qlearn', alpha: 0.5, eps: 0.1, episodes: 3000, seed: 11 });
  const learned = policyFromQ(gw, Q);
  // the learned action at the start matches one of the optimal first moves
  const a = learned[gw.start.r][gw.start.c];
  assert.equal(a, optPolicy[gw.start.r][gw.start.c]);
});

test('SARSA also improves over random', () => {
  const gw = makeTinyGrid();
  const { Q } = train(gw, { alg: 'sarsa', alpha: 0.5, eps: 0.1, episodes: 1500, seed: 5 });
  const policy = policyFromQ(gw, Q);
  const evalRng = new RNG(31);
  let learned = 0;
  const E = 300;
  for (let i = 0; i < E; i++) learned += rolloutReturn(gw, policy, evalRng);
  learned /= E;
  const baseline = randomReturn(gw, new RNG(31), { episodes: E });
  assert.ok(learned > baseline, `SARSA ${learned} should beat random ${baseline}`);
});

test('learning curve trends upward (later returns beat early returns)', () => {
  const gw = makeTinyGrid();
  const { returns } = train(gw, { alg: 'qlearn', alpha: 0.5, eps: 0.1, episodes: 1200, seed: 9 });
  const early = returns.slice(0, 200).reduce((a, b) => a + b, 0) / 200;
  const late = returns.slice(-200).reduce((a, b) => a + b, 0) / 200;
  assert.ok(late > early, `late ${late} should exceed early ${early}`);
});
