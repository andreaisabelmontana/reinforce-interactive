// Browser bridge for the Gridworld engine.
//
// The MDP model (states, actions, transitions, rewards, sampling) lives in the
// framework-free, unit-tested module src/gridworld.js — this file re-exports it
// so the pages run exactly the code the test suite verifies, and adds the
// canvas drawing + click-to-edit helpers that only make sense in the browser.
//
// Loaded as <script type="module">. The drawing helpers use the globals defined
// by js/common.js (clearCanvas, valueColor), which is loaded first as a classic
// script.
'use strict';

import {
  Gridworld, CELL, ACTIONS, ARROW,
  makeDefaultGrid, makeCliffGrid, makeTinyGrid,
} from '../src/gridworld.js';

export { Gridworld, CELL, ACTIONS, ARROW, makeDefaultGrid, makeCliffGrid, makeTinyGrid };

// Draws a gridworld on a canvas (browser-only).
export function drawGrid(ctx, gw, opts = {}) {
  const { V, Q, policy, agent, highlight, showValues = false, showPolicy = false, cellSize = 56 } = opts;
  const W = gw.cols * cellSize, H = gw.rows * cellSize;
  ctx.canvas.width !== W && (ctx.canvas.width = W);
  ctx.canvas.height !== H && (ctx.canvas.height = H);
  clearCanvas(ctx, '#050811');

  let vMin = Infinity, vMax = -Infinity;
  if (V) for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
    if (V[r][c] < vMin) vMin = V[r][c];
    if (V[r][c] > vMax) vMax = V[r][c];
  }
  if (Q) for (let r = 0; r < gw.rows; r++) for (let c = 0; c < gw.cols; c++) {
    const m = Math.max(...Q[r][c]);
    if (m < vMin) vMin = m;
    if (m > vMax) vMax = m;
  }
  if (vMin === vMax) { vMin -= 1; vMax += 1; }

  for (let r = 0; r < gw.rows; r++) {
    for (let c = 0; c < gw.cols; c++) {
      const x = c * cellSize, y = r * cellSize;
      const ct = gw.cells[r][c];
      let bg = '#0d1424';
      if (ct === CELL.WALL) bg = '#3a4570';
      else if (ct === CELL.GOAL) bg = '#1f5d4e';
      else if (ct === CELL.PIT) bg = '#5d1f1f';
      else if (ct === CELL.CLIFF) bg = '#3a0a0a';
      else if (ct === CELL.START) bg = '#1a2238';
      else if (V) {
        bg = valueColor(V[r][c], vMin, vMax);
      } else if (Q) {
        bg = valueColor(Math.max(...Q[r][c]), vMin, vMax);
      }
      ctx.fillStyle = bg;
      ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

      ctx.strokeStyle = '#0a0e1a'; ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, cellSize, cellSize);

      if (showValues && ct !== CELL.WALL) {
        const v = V ? V[r][c] : (Q ? Math.max(...Q[r][c]) : 0);
        ctx.fillStyle = '#e6ecff';
        ctx.font = `${Math.max(9, cellSize * 0.18)}px JetBrains Mono, monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(v.toFixed(2), x + cellSize / 2, y + cellSize / 2 + 3);
      }

      if (showPolicy && policy && ct === CELL.EMPTY || (showPolicy && policy && ct === CELL.START)) {
        const a = policy[r][c];
        if (a !== undefined && a !== -1) {
          ctx.fillStyle = '#00d4ff';
          ctx.font = `bold ${Math.max(14, cellSize * 0.35)}px system-ui`;
          ctx.textAlign = 'center';
          const ar = ARROW[a];
          ctx.fillText(ar, x + cellSize / 2, y + cellSize * 0.85);
        }
      }

      if (ct === CELL.GOAL) {
        ctx.fillStyle = '#2dd4bf';
        ctx.font = `bold ${cellSize * 0.45}px system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText('★', x + cellSize / 2, y + cellSize * 0.6);
      } else if (ct === CELL.PIT) {
        ctx.fillStyle = '#ff5c8a';
        ctx.font = `bold ${cellSize * 0.45}px system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText('✖', x + cellSize / 2, y + cellSize * 0.6);
      } else if (ct === CELL.CLIFF) {
        ctx.fillStyle = '#ff5c8a';
        ctx.font = `${cellSize * 0.4}px system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText('↯', x + cellSize / 2, y + cellSize * 0.6);
      } else if (ct === CELL.START) {
        ctx.strokeStyle = '#7c5cff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 5, y + 5, cellSize - 10, cellSize - 10);
        ctx.fillStyle = '#7c5cff';
        ctx.font = `bold 10px JetBrains Mono`;
        ctx.textAlign = 'center';
        ctx.fillText('S', x + cellSize / 2, y + cellSize * 0.95);
      }

      if (highlight && highlight.r === r && highlight.c === c) {
        ctx.strokeStyle = '#ff5c8a';
        ctx.lineWidth = 3;
        ctx.strokeRect(x + 2, y + 2, cellSize - 4, cellSize - 4);
      }
    }
  }

  if (agent) {
    const ax = agent.c * cellSize + cellSize / 2;
    const ay = agent.r * cellSize + cellSize / 2;
    ctx.fillStyle = '#ff5c8a';
    ctx.beginPath();
    ctx.arc(ax, ay, cellSize * 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// Click-to-edit utility (browser-only).
export function bindGridEditor(canvas, gw, getMode, onChange, cellSize = 56) {
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const c = Math.floor(x / cellSize), r = Math.floor(y / cellSize);
    if (!gw.inBounds(r, c)) return;
    const mode = getMode();
    if (mode === 'start') {
      for (let i = 0; i < gw.rows; i++) for (let j = 0; j < gw.cols; j++)
        if (gw.cells[i][j] === CELL.START) gw.cells[i][j] = CELL.EMPTY;
      gw.cells[r][c] = CELL.START;
      gw.start = { r, c };
    } else if (mode === 'wall') gw.cells[r][c] = gw.cells[r][c] === CELL.WALL ? CELL.EMPTY : CELL.WALL;
    else if (mode === 'goal') gw.cells[r][c] = gw.cells[r][c] === CELL.GOAL ? CELL.EMPTY : CELL.GOAL;
    else if (mode === 'pit') gw.cells[r][c] = gw.cells[r][c] === CELL.PIT ? CELL.EMPTY : CELL.PIT;
    else if (mode === 'cliff') gw.cells[r][c] = gw.cells[r][c] === CELL.CLIFF ? CELL.EMPTY : CELL.CLIFF;
    else if (mode === 'erase') gw.cells[r][c] = CELL.EMPTY;
    onChange && onChange();
  });
}
