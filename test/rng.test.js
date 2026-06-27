import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RNG, mulberry32, argmax, argmaxTies, softmax } from '../src/rng.js';

test('mulberry32 is deterministic for a fixed seed', () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  for (let i = 0; i < 100; i++) assert.equal(a(), b());
});

test('different seeds produce different streams', () => {
  const a = mulberry32(1)();
  const b = mulberry32(2)();
  assert.notEqual(a, b);
});

test('RNG.random stays in [0,1) and RNG.int in [0,n)', () => {
  const rng = new RNG(7);
  for (let i = 0; i < 1000; i++) {
    const u = rng.random();
    assert.ok(u >= 0 && u < 1);
    const k = rng.int(5);
    assert.ok(Number.isInteger(k) && k >= 0 && k < 5);
  }
});

test('RNG.normal has roughly zero mean and unit variance', () => {
  const rng = new RNG(123);
  const N = 20000;
  let sum = 0, sumSq = 0;
  for (let i = 0; i < N; i++) {
    const x = rng.normal();
    sum += x;
    sumSq += x * x;
  }
  const mean = sum / N;
  const variance = sumSq / N - mean * mean;
  assert.ok(Math.abs(mean) < 0.05, `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.1, `variance ${variance}`);
});

test('argmax and softmax behave', () => {
  assert.equal(argmax([1, 5, 3]), 1);
  assert.equal(argmaxTies([2, 2, 2], new RNG(1)) < 3, true);
  const p = softmax([0, 0, 0]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  for (const x of p) assert.ok(Math.abs(x - 1 / 3) < 1e-12);
});
