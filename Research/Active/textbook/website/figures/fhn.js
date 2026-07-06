/* FitzHugh–Nagumo phase-plane widget (Fig 2.1).
   Drag to place an initial condition; sliders vary I_ext / a / b / ε; watch the
   RK4-integrated trajectory. Ported from the durable prototype
   (plans/2026-07-03_refresh_prototype.html). Themeable via --fig-* tokens.

   Contract: mount(canvas, params) -> { redraw, destroy }. */
import { fit, cvar, reducedMotion, slider, button } from './_canvas.js';

const VMIN = -2.6, VMAX = 2.6, WMIN = -1.3, WMAX = 2.2, M = 34;

export function mount(canvas, params) {
  const p = Object.assign({ I: 0.5, a: 0.7, b: 0.8, eps: 0.08 }, params || {});
  let ic = [-1.0, -0.6], v = ic[0], w = ic[1], traj = [[v, w]];
  let playing = !reducedMotion(), raf = null, destroyed = false;

  const deriv = (v, w) => [v - v * v * v / 3 - w + p.I, p.eps * (v + p.a - p.b * w)];
  function rk4(v, w, dt) {
    const a = deriv(v, w),
      b = deriv(v + 0.5 * dt * a[0], w + 0.5 * dt * a[1]),
      c = deriv(v + 0.5 * dt * b[0], w + 0.5 * dt * b[1]),
      d = deriv(v + dt * c[0], w + dt * c[1]);
    return [v + dt / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]),
            w + dt / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1])];
  }
  function restart() { v = ic[0]; w = ic[1]; traj = [[v, w]]; }

  function classify() {
    if (traj.length < 200) return 'transient';
    const seg = traj.slice(-200);
    let mn = 9, mx = -9;
    for (const s of seg) { if (s[0] < mn) mn = s[0]; if (s[0] > mx) mx = s[0]; }
    return (mx - mn > 1.5) ? 'limit cycle' : 'rest';
  }

  let geo = null;
  function draw() {
    const s = fit(canvas), ctx = s.ctx, W = s.w, H = s.h;
    const cr = cvar('--fig-crimson'), te = cvar('--fig-teal'), am = cvar('--fig-amber'),
      ax = cvar('--fig-axis'), gr = cvar('--fig-grid'), st = cvar('--fig-stage');
    const X = vv => M + (vv - VMIN) / (VMAX - VMIN) * (W - M * 1.5);
    const Y = ww => (H - M) - (ww - WMIN) / (WMAX - WMIN) * (H - M * 1.6);
    geo = { X, Y, W, H };
    ctx.fillStyle = st; ctx.fillRect(0, 0, W, H);
    // grid
    ctx.strokeStyle = gr; ctx.lineWidth = 1;
    for (let gv = -2; gv <= 2; gv++) { ctx.beginPath(); ctx.moveTo(X(gv), Y(WMAX)); ctx.lineTo(X(gv), Y(WMIN)); ctx.stroke(); }
    for (let gw = -1; gw <= 2; gw++) { ctx.beginPath(); ctx.moveTo(X(VMIN), Y(gw)); ctx.lineTo(X(VMAX), Y(gw)); ctx.stroke(); }
    // axes
    ctx.strokeStyle = ax; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(X(VMIN), Y(0)); ctx.lineTo(X(VMAX), Y(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(0), Y(WMIN)); ctx.lineTo(X(0), Y(WMAX)); ctx.stroke();
    ctx.globalAlpha = 1;
    // v-nullcline  w = v - v^3/3 + I
    ctx.strokeStyle = cr; ctx.lineWidth = 2.6; ctx.beginPath();
    for (let i = 0; i <= 120; i++) { const vv = VMIN + (VMAX - VMIN) * i / 120; const ww = vv - vv * vv * vv / 3 + p.I; i ? ctx.lineTo(X(vv), Y(ww)) : ctx.moveTo(X(vv), Y(ww)); }
    ctx.stroke();
    // w-nullcline  w = (v + a) / b
    ctx.strokeStyle = te; ctx.lineWidth = 2.4; ctx.beginPath();
    ctx.moveTo(X(VMIN), Y((VMIN + p.a) / p.b)); ctx.lineTo(X(VMAX), Y((VMAX + p.a) / p.b)); ctx.stroke();
    // trajectory
    ctx.strokeStyle = am; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let k = 0; k < traj.length; k++) { const t = traj[k]; k ? ctx.lineTo(X(t[0]), Y(t[1])) : ctx.moveTo(X(t[0]), Y(t[1])); }
    ctx.stroke();
    const head = traj[traj.length - 1];
    ctx.fillStyle = am; ctx.beginPath(); ctx.arc(X(head[0]), Y(head[1]), 4.5, 0, 7); ctx.fill();
    // initial-condition marker
    ctx.strokeStyle = cr; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(ic[0]), Y(ic[1]), 6, 0, 7); ctx.stroke();
    // legend (mono)
    ctx.font = '11px ' + (cvar('--font-mono') || 'monospace'); ctx.textBaseline = 'middle';
    const leg = [['v-nullcline', cr], ['w-nullcline', te], ['trajectory', am]];
    leg.forEach((l, i) => {
      const y = 16 + i * 16;
      ctx.strokeStyle = l[1]; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(12, y); ctx.lineTo(28, y); ctx.stroke();
      ctx.fillStyle = ax; ctx.fillText(l[0], 34, y);
    });
  }

  function step() {
    if (destroyed) return;
    if (playing) {
      for (let n = 0; n < 3; n++) { const r = rk4(v, w, 0.05); v = r[0]; w = r[1]; traj.push([v, w]); }
      if (traj.length > 1400) traj.splice(0, traj.length - 1400);
    }
    draw();
    if (regimeEl) { const st = classify(); regimeEl.textContent = st; regimeEl.style.color = st === 'limit cycle' ? 'var(--highlight)' : 'var(--text)'; }
    raf = requestAnimationFrame(step);
  }

  // ── pointer → set initial condition ──
  let dragging = false;
  function place(e) {
    if (!geo) draw();
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const vv = VMIN + (px - M) / (geo.W - M * 1.5) * (VMAX - VMIN);
    const ww = WMIN + ((geo.H - M) - py) / (geo.H - M * 1.6) * (WMAX - WMIN);
    ic = [Math.max(VMIN, Math.min(VMAX, vv)), Math.max(WMIN, Math.min(WMAX, ww))];
    restart();
  }
  const onDown = e => { dragging = true; canvas.setPointerCapture(e.pointerId); place(e); };
  const onMove = e => { if (dragging) place(e); };
  const onUp = () => { dragging = false; };
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);

  // ── control rail ──
  const fig = canvas.closest('.fig');
  const controls = fig && fig.querySelector('.fig-controls');
  let regimeEl = null, playBtn = null;
  if (controls) {
    controls.innerHTML = '';
    const grid = document.createElement('div'); grid.className = 'fw-grid'; controls.appendChild(grid);
    slider(grid, { label: 'I_ext — stimulus', min: 0, max: 1.2, step: 0.01, value: p.I, fmt: v => v.toFixed(2) }, v => { p.I = v; restart(); if (!playing) draw(); });
    slider(grid, { label: 'a — recovery offset', min: 0, max: 1.2, step: 0.01, value: p.a, fmt: v => v.toFixed(2) }, v => { p.a = v; restart(); if (!playing) draw(); });
    slider(grid, { label: 'b — recovery slope', min: 0.2, max: 1.4, step: 0.01, value: p.b, fmt: v => v.toFixed(2) }, v => { p.b = v; restart(); if (!playing) draw(); });
    slider(grid, { label: 'ε — time-scale ratio', min: 0.02, max: 0.3, step: 0.005, value: p.eps, fmt: v => v.toFixed(3) }, v => { p.eps = v; restart(); if (!playing) draw(); });
    const row = document.createElement('div'); row.className = 'fw-btnrow'; controls.appendChild(row);
    playBtn = button(row, playing ? 'Pause' : 'Play', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; }, true);
    button(row, 'Reset', () => { ic = [-1.0, -0.6]; restart(); if (!playing) draw(); });
    const rd = document.createElement('div'); rd.className = 'fw-readout';
    rd.innerHTML = '<span class="fw-name">regime</span> <b></b>'; regimeEl = rd.querySelector('b');
    controls.appendChild(rd);
  }

  // reduced motion: integrate a static frame, no animation loop
  if (reducedMotion()) {
    for (let i = 0; i < 900; i++) { const r = rk4(v, w, 0.05); v = r[0]; w = r[1]; traj.push([v, w]); }
    draw();
    if (regimeEl) regimeEl.textContent = classify();
  } else {
    step();
  }

  return {
    redraw() { if (!destroyed) draw(); },
    destroy() {
      destroyed = true;
      if (raf) cancelAnimationFrame(raf);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
    }
  };
}
