import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LinearVFA, polyFeat, rbfFeat, fourierFeat, tileFeat, fit, trueV, gridMSE } from '../src/function-approx.js';

test('feature bases return the expected dimensions', () => {
  assert.equal(polyFeat(0.3, 5).length, 5);
  assert.equal(rbfFeat(0.3, 12).length, 12);
  assert.equal(fourierFeat(0.3, 8).length, 8);
  assert.equal(tileFeat(0.3, 4).length, 4 * 8);
});

test('a single tile is active per tiling', () => {
  const f = tileFeat(0.0, 3);
  const active = f.reduce((a, b) => a + b, 0);
  assert.equal(active, 3); // one active tile per tiling
});

test('linear update reduces error toward a constant target', () => {
  const model = new LinearVFA((s) => [1, s], 2); // bias + slope
  let last = Infinity;
  for (let i = 0; i < 200; i++) last = model.update(0.5, 1.0, 0.1);
  assert.ok(model.predict(0.5) > 0.9 && model.predict(0.5) <= 1.0 + 1e-6);
  assert.ok(last < 1e-3);
});

test('RBF regression fits the target function well', () => {
  const { mse } = fit({ basis: 'rbf', dim: 24, alpha: 0.1, noise: 0, steps: 20000, seed: 2 });
  assert.ok(mse < 0.05, `rbf mse ${mse}`);
});

test('fourier regression also drives MSE down', () => {
  const { mse } = fit({ basis: 'fourier', dim: 16, alpha: 0.05, noise: 0, steps: 20000, seed: 2 });
  assert.ok(mse < 0.05, `fourier mse ${mse}`);
});

test('an untrained model has higher error than a trained one', () => {
  const fresh = new LinearVFA((s) => rbfFeat(s, 24), 24);
  const before = gridMSE(fresh);
  const { mse } = fit({ basis: 'rbf', dim: 24, alpha: 0.1, noise: 0, steps: 20000, seed: 2 });
  assert.ok(mse < before, `trained ${mse} should beat untrained ${before}`);
  assert.equal(typeof trueV(0.5), 'number');
});
