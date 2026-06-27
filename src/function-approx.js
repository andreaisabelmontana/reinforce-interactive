// Linear function approximation: feature bases + the semi-gradient SGD update
// from the Function Approximation demo. Approximates v(s) = wᵀφ(s).

import { RNG } from './rng.js';

export const X_MIN = -1, X_MAX = 1;

// The target value function the demo regresses against.
export function trueV(s) { return Math.sin(3 * s) + 0.3 * s; }

// ---- feature bases (identical to js/function-approx.js) ----

export function polyFeat(s, n) {
  const f = new Array(n);
  let p = 1;
  for (let i = 0; i < n; i++) { f[i] = p; p *= s; }
  return f;
}

export function rbfFeat(s, n) {
  const f = new Array(n);
  const sigma = (X_MAX - X_MIN) / Math.max(2, n - 1);
  for (let i = 0; i < n; i++) {
    const c = X_MIN + (i * (X_MAX - X_MIN)) / Math.max(1, n - 1);
    f[i] = Math.exp(-Math.pow(s - c, 2) / (2 * sigma * sigma));
  }
  return f;
}

export function tileFeat(s, nTilings) {
  const tilesPerT = 8;
  const range = X_MAX - X_MIN;
  const tileW = range / tilesPerT;
  const f = new Array(nTilings * tilesPerT).fill(0);
  for (let t = 0; t < nTilings; t++) {
    const offset = (t / nTilings) * tileW;
    const idx = Math.floor((s - X_MIN + offset) / tileW) % tilesPerT;
    f[t * tilesPerT + ((idx + tilesPerT) % tilesPerT)] = 1;
  }
  return f;
}

export function fourierFeat(s, n) {
  const sn = (s - X_MIN) / (X_MAX - X_MIN);
  const f = new Array(n);
  for (let i = 0; i < n; i++) f[i] = Math.cos(Math.PI * i * sn);
  return f;
}

export const BASES = { poly: polyFeat, rbf: rbfFeat, tile: tileFeat, fourier: fourierFeat };

// A linear value-function approximator: prediction wᵀφ(s) and the
// semi-gradient SGD weight update w ← w − α(ŷ − target)φ(s).
export class LinearVFA {
  constructor(featureFn, dim, { clamp = 50 } = {}) {
    this.featureFn = featureFn;
    this.w = new Array(dim).fill(0);
    this.clamp = clamp;
  }

  predict(s) {
    const f = this.featureFn(s);
    let y = 0;
    for (let i = 0; i < f.length; i++) y += this.w[i] * f[i];
    return y;
  }

  // One SGD step toward target. Returns the squared error before the update.
  update(s, target, alpha) {
    const f = this.featureFn(s);
    let y = 0;
    for (let i = 0; i < f.length; i++) y += this.w[i] * f[i];
    const err = y - target;
    for (let i = 0; i < f.length; i++) {
      this.w[i] -= alpha * err * f[i];
      if (Math.abs(this.w[i]) > this.clamp) this.w[i] = Math.sign(this.w[i]) * this.clamp;
    }
    return err * err;
  }
}

// Train a LinearVFA on noisy samples of trueV and return the model plus
// the mean squared error over an evenly spaced grid.
export function fit({ basis = 'rbf', dim = 20, alpha = 0.05, noise = 0, steps = 5000, seed = 1 } = {}) {
  const featureFn = (s) => BASES[basis](s, basis === 'tile' ? Math.max(1, Math.floor(dim / 8)) : dim);
  const probeLen = featureFn(0).length;
  const model = new LinearVFA(featureFn, probeLen);
  const rng = new RNG(seed);
  for (let i = 0; i < steps; i++) {
    const s = X_MIN + rng.random() * (X_MAX - X_MIN);
    const target = trueV(s) + noise * rng.normal();
    model.update(s, target, alpha);
  }
  return { model, mse: gridMSE(model) };
}

export function gridMSE(model, M = 100) {
  let mse = 0;
  for (let i = 0; i < M; i++) {
    const x = X_MIN + (i * (X_MAX - X_MIN)) / (M - 1);
    const d = model.predict(x) - trueV(x);
    mse += d * d;
  }
  return mse / M;
}
