// Multi-armed bandit — k Gaussian arms, action-value estimates, and the
// action-selection strategies from the Bandits demo (ε-greedy, UCB, greedy,
// optimistic init). Seeded for reproducible regret curves.

import { RNG, argmaxTies, argmax } from './rng.js';

// A k-armed testbed: arm i pays reward ~ N(mu_i, 1).
export class Bandit {
  // strategy ∈ {'greedy','egreedy','ucb','optimistic'}
  constructor({ k = 10, seed = 1, strategy = 'egreedy', eps = 0.1, c = 2, optimistic = 5 } = {}) {
    this.k = k;
    this.rng = new RNG(seed);
    this.strategy = strategy;
    this.eps = eps;
    this.c = c;
    this.trueMu = Array.from({ length: k }, () => this.rng.normal());
    this.optArm = argmax(this.trueMu);
    this.Q = new Array(k).fill(strategy === 'optimistic' ? optimistic : 0);
    this.N = new Array(k).fill(0);
    this.t = 0;
    this.cumReward = 0;
    this.optCount = 0;
    this.regret = 0; // cumulative regret = t*mu* - sum rewards
  }

  selectArm() {
    if (this.strategy === 'egreedy') {
      return this.rng.random() < this.eps ? this.rng.int(this.k) : argmaxTies(this.Q, this.rng);
    }
    if (this.strategy === 'ucb') {
      const ucb = this.Q.map((q, i) =>
        this.N[i] === 0 ? Infinity : q + this.c * Math.sqrt(Math.log(this.t + 1) / this.N[i]));
      return argmaxTies(ucb, this.rng);
    }
    // 'greedy' and 'optimistic' both act greedily; optimistic differs only in Q init.
    return argmaxTies(this.Q, this.rng);
  }

  // Pull arm a, observe reward ~ N(mu_a, 1), and update the sample-average estimate.
  pull(a) {
    const r = this.trueMu[a] + this.rng.normal();
    this.N[a]++;
    this.Q[a] += (1 / this.N[a]) * (r - this.Q[a]);
    this.t++;
    this.cumReward += r;
    if (a === this.optArm) this.optCount++;
    this.regret = this.t * this.trueMu[this.optArm] - this.cumReward;
    return r;
  }

  step() {
    const a = this.selectArm();
    this.pull(a);
    return a;
  }

  // Run n steps. Returns summary stats including the fraction of optimal pulls
  // and average regret per step.
  run(n) {
    for (let i = 0; i < n; i++) this.step();
    return {
      steps: this.t,
      optFraction: this.optCount / this.t,
      avgReward: this.cumReward / this.t,
      regret: this.regret,
      avgRegret: this.regret / this.t,
    };
  }
}
