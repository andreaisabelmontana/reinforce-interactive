// Browser-only animation loop helper, importable by the ES-module demo scripts.
// Mirrors the Loop class in js/common.js (which classic-script pages still use).
'use strict';

export class Loop {
  constructor(stepFn, opts = {}) {
    this.stepFn = stepFn;
    this.running = false;
    this.stepsPerFrame = opts.stepsPerFrame || 1;
    this._raf = null;
  }
  start() {
    if (this.running) return;
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      for (let i = 0; i < this.stepsPerFrame; i++) this.stepFn();
      if (typeof requestAnimationFrame !== 'undefined' && !document.hidden) {
        this._raf = requestAnimationFrame(tick);
      } else {
        this._raf = setTimeout(tick, 33);
      }
    };
    tick();
  }
  stop() {
    this.running = false;
    if (this._raf) {
      try { cancelAnimationFrame(this._raf); } catch (_) {}
      try { clearTimeout(this._raf); } catch (_) {}
    }
  }
  toggle() { this.running ? this.stop() : this.start(); }
}
