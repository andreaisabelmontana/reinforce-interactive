// Seedable PRNG + math helpers shared by the RL modules.
// Framework-free, no DOM. Used by both the browser demos and the tests.

// mulberry32: tiny, fast, well-distributed 32-bit PRNG. Deterministic given a seed.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A small RNG object bundling uniform / gaussian / integer / categorical draws.
export class RNG {
  constructor(seed = 12345) {
    this.next = mulberry32(seed);
  }
  // uniform in [0, 1)
  random() {
    return this.next();
  }
  // integer in [0, n)
  int(n) {
    return Math.floor(this.next() * n);
  }
  // standard normal via Box-Muller
  normal() {
    let u = 0, v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  // sample an index from a probability vector (must sum to 1)
  categorical(probs) {
    const r = this.next();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return i;
    }
    return probs.length - 1;
  }
}

// argmax with no tie-breaking (first max wins) — matches js/common.js argmax.
export function argmax(arr) {
  let best = 0, v = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > v) { v = arr[i]; best = i; }
  return best;
}

// argmax with random tie-breaking among equal maxima. Needs an rng for determinism.
export function argmaxTies(arr, rng) {
  const max = Math.max(...arr);
  const ties = [];
  for (let i = 0; i < arr.length; i++) if (arr[i] === max) ties.push(i);
  const pick = rng ? rng.int(ties.length) : Math.floor(Math.random() * ties.length);
  return ties[ties.length === 1 ? 0 : pick];
}

export function softmax(xs, temp = 1) {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp((x - m) / temp));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}
