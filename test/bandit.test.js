import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Bandit } from '../src/bandit.js';

test('epsilon-greedy pulls the best arm a majority of the time', () => {
  const b = new Bandit({ k: 10, seed: 4, strategy: 'egreedy', eps: 0.1 });
  const stats = b.run(5000);
  assert.ok(stats.optFraction > 0.5, `optFraction ${stats.optFraction}`);
});

test('UCB pulls the best arm a majority of the time', () => {
  const b = new Bandit({ k: 10, seed: 4, strategy: 'ucb', c: 2 });
  const stats = b.run(5000);
  assert.ok(stats.optFraction > 0.5, `optFraction ${stats.optFraction}`);
});

test('epsilon-greedy beats greedy on average across many testbeds', () => {
  // Pure greedy can lock onto whichever arm it sampled high first, so on any
  // single seed it may get lucky. Averaged over many random testbeds it is
  // worse than ε-greedy — the classic Sutton & Barto 10-armed-bandit result.
  let egSum = 0, greedySum = 0;
  const M = 60;
  for (let seed = 0; seed < M; seed++) {
    egSum += new Bandit({ k: 10, seed, strategy: 'egreedy', eps: 0.1 }).run(2000).optFraction;
    greedySum += new Bandit({ k: 10, seed, strategy: 'greedy' }).run(2000).optFraction;
  }
  const eg = egSum / M, greedy = greedySum / M;
  assert.ok(eg > greedy, `mean optFraction: ε-greedy ${eg} vs greedy ${greedy}`);
});

test('average regret shrinks over time (sublinear cumulative regret)', () => {
  const b = new Bandit({ k: 10, seed: 8, strategy: 'egreedy', eps: 0.1 });
  b.run(500);
  const early = b.regret / b.t; // avg regret after 500
  b.run(9500); // now 10000 total
  const late = b.regret / b.t; // avg regret after 10000
  assert.ok(late < early, `avg regret should fall: early ${early}, late ${late}`);
  assert.ok(late >= 0, 'cumulative regret is non-negative');
});

test('UCB achieves lower average regret than greedy', () => {
  const ucb = new Bandit({ k: 10, seed: 14, strategy: 'ucb', c: 2 }).run(5000);
  const greedy = new Bandit({ k: 10, seed: 14, strategy: 'greedy' }).run(5000);
  assert.ok(ucb.avgRegret < greedy.avgRegret, `ucb ${ucb.avgRegret} vs greedy ${greedy.avgRegret}`);
});
