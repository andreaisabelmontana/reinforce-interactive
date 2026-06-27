import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Gridworld, CELL, makeCliffGrid, makeTinyGrid } from '../src/gridworld.js';
import { RNG } from '../src/rng.js';

test('a clear move yields the step reward and the intended neighbour', () => {
  const gw = new Gridworld(3, 3, { start: { r: 2, c: 0 }, slip: 0, stepReward: -0.04 });
  // action 0 = up from (2,0) -> (1,0)
  const ts = gw.transitions(2, 0, 0);
  assert.equal(ts.length, 1); // no slip => single outcome
  assert.deepEqual({ r: ts[0].r1, c: ts[0].c1 }, { r: 1, c: 0 });
  assert.equal(ts[0].reward, -0.04);
  assert.equal(ts[0].done, false);
});

test('moving off the grid keeps the agent in place', () => {
  const gw = new Gridworld(3, 3, { start: { r: 0, c: 0 }, slip: 0 });
  // action 0 = up from top row stays put
  const ts = gw.transitions(0, 0, 0);
  assert.deepEqual({ r: ts[0].r1, c: ts[0].c1 }, { r: 0, c: 0 });
});

test('walls block movement', () => {
  const gw = new Gridworld(3, 3, { start: { r: 2, c: 0 }, slip: 0 });
  gw.cells[1][0] = CELL.WALL;
  const ts = gw.transitions(2, 0, 0); // up into the wall
  assert.deepEqual({ r: ts[0].r1, c: ts[0].c1 }, { r: 2, c: 0 });
});

test('the goal is terminal and pays the goal reward', () => {
  const gw = makeTinyGrid(); // goal at (0,2)
  // step into goal from (1,2) by going up
  const ts = gw.transitions(1, 2, 0);
  const goalOutcome = ts.find((t) => t.r1 === 0 && t.c1 === 2);
  assert.ok(goalOutcome);
  assert.equal(goalOutcome.reward, gw.goalReward);
  assert.equal(goalOutcome.done, true);
  // once in the goal, it is absorbing
  assert.equal(gw.isTerminal(0, 2), true);
  const abs = gw.transitions(0, 2, 1);
  assert.equal(abs.length, 1);
  assert.equal(abs[0].done, true);
  assert.equal(abs[0].reward, 0);
});

test('slip spreads probability over intended + perpendicular moves and sums to 1', () => {
  const gw = new Gridworld(3, 3, { start: { r: 2, c: 1 }, slip: 0.2 });
  const ts = gw.transitions(1, 1, 0); // up, with slip
  const total = ts.reduce((a, t) => a + t.p, 0);
  assert.ok(Math.abs(total - 1) < 1e-12, `probs sum to ${total}`);
  const intended = ts.reduce((m, t) => Math.max(m, t.p), 0);
  assert.ok(Math.abs(intended - 0.8) < 1e-12);
});

test('cliff: large penalty, teleport to start, not terminal', () => {
  const gw = makeCliffGrid(); // bottom row middle cells are cliffs
  // from start (3,0) moving right (action 1) steps onto a cliff cell
  const ts = gw.transitions(3, 0, 1);
  assert.equal(ts[0].reward, -100);
  assert.deepEqual({ r: ts[0].r1, c: ts[0].c1 }, { r: gw.start.r, c: gw.start.c });
  assert.equal(ts[0].done, false);
});

test('sampling respects the transition distribution', () => {
  const gw = new Gridworld(3, 3, { start: { r: 2, c: 1 }, slip: 0.2 });
  const rng = new RNG(99);
  const counts = {};
  const N = 40000;
  for (let i = 0; i < N; i++) {
    const t = gw.sample(1, 1, 0, rng);
    const key = `${t.r1},${t.c1}`;
    counts[key] = (counts[key] || 0) + 1;
  }
  // intended cell (0,1) should get ~80% of samples
  assert.ok(Math.abs(counts['0,1'] / N - 0.8) < 0.02, JSON.stringify(counts));
});
