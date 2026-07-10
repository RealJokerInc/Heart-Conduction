/* Ion-Current Playground — pick a cardiac_core ionic engine, tune conductances,
   watch the action-potential morphology respond. Every trace is EXACT real-engine
   output precomputed by website/build/gen_ap_traces.py (sliders snap to grid levels).
   Contract: mount(canvas, params) -> { redraw, destroy }. */
import { fit, cvar, reducedMotion } from './_canvas.js';

const ENGINES = [
  { id: 'ttp06',  label: 'TTP06',        sub: 'human ventricle', cellTypes: ['ENDO', 'EPI', 'M_CELL'] },
  { id: 'ord',    label: "O'Hara–Rudy",  sub: 'human ventricle', cellTypes: ['ENDO', 'EPI', 'M_CELL'] },
  { id: 'phas13', label: 'PHAS13',       sub: 'hiPSC — spontaneous', cellTypes: null, spontaneous: true },
  { id: 'mhas13', label: 'MHAS13',       sub: 'matured hiPSC', cellTypes: null },
];
const CT_LABEL = { ENDO: 'Endo', EPI: 'Epi', M_CELL: 'M-cell' };

export function mount(canvas, params) {
  const fig = canvas.closest('.ap-explorer') || canvas.parentElement;
  const stage = canvas.parentElement;
  let destroyed = false, data = null;
  let engine = 'ttp06', cellType = 'EPI';
  let gridIdx = [2, 2, 2];          // grid slider positions (index into gridLevels; 2 == 1.0×)
  let isolate = null;               // {knob, level} when isolating a single current, else null
  let gridMap = null;               // "i,j,k" -> trace

  // ── DOM scaffold ─────────────────────────────────────────────
  const bar = el('div', 'apx-enginebar');
  const cttoggle = el('div', 'apx-ct');
  stage.parentElement.insertBefore(bar, stage);
  const metrics = el('div', 'apx-metrics');
  stage.appendChild(metrics);
  const controls = fig.querySelector('.fig-controls');
  controls.innerHTML = '';
  const combineWrap = el('div', 'apx-panel'); controls.appendChild(combineWrap);
  const isolateWrap = el('div', 'apx-panel'); controls.appendChild(isolateWrap);
  const glossary = el('div', 'apx-glossary'); controls.appendChild(glossary);

  // engine buttons
  ENGINES.forEach(e => {
    const b = el('button', 'apx-eng' + (e.id === engine ? ' active' : ''));
    b.type = 'button';
    b.innerHTML = `<b>${e.label}</b><span>${e.sub}</span>`;
    b.addEventListener('click', () => selectEngine(e.id));
    b._eid = e.id; bar.appendChild(b);
  });
  bar.appendChild(cttoggle);

  function el(tag, cls) { const n = document.createElement(tag); if (cls) n.className = cls; return n; }
  function b64toVm(s) {
    const bin = atob(s), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    const out = new Float32Array(u.length), span = data.vmax - data.vmin;
    for (let i = 0; i < u.length; i++) out[i] = data.vmin + (u[i] / 255) * span;
    return out;
  }

  // ── data selection ───────────────────────────────────────────
  async function selectEngine(id) {
    engine = id;
    const e = ENGINES.find(x => x.id === id);
    if (e.cellTypes && !e.cellTypes.includes(cellType)) cellType = 'EPI';
    bar.querySelectorAll('.apx-eng').forEach(x => x.classList.toggle('active', x._eid === id));
    await load();
  }
  async function load() {
    const e = ENGINES.find(x => x.id === engine);
    const fname = engine + (e.cellTypes ? '_' + cellType.toLowerCase() : '') + '.json';
    try {
      const r = await fetch('data/ap_explorer/' + fname);
      data = await r.json();
    } catch (err) { metrics.textContent = 'data not available: ' + fname; return; }
    gridMap = {};
    data.grid.forEach(g => { gridMap[g.idx.join(',')] = g; });
    gridIdx = [2, 2, 2]; isolate = null;
    renderCellTypes(e); renderCombine(); renderIsolate(); renderGlossary();
    draw();
  }

  function currentTrace() {
    if (isolate) {
      const arr = data.sweeps[isolate.knob];
      return arr[isolate.level];
    }
    return gridMap[gridIdx.join(',')] || data.baseline;
  }

  // ── control panels ───────────────────────────────────────────
  function renderCellTypes(e) {
    cttoggle.innerHTML = '';
    if (!e.cellTypes) return;
    e.cellTypes.forEach(ct => {
      const b = el('button', 'apx-ctbtn' + (ct === cellType ? ' active' : ''));
      b.type = 'button'; b.textContent = CT_LABEL[ct];
      b.addEventListener('click', () => { cellType = ct; load(); });
      cttoggle.appendChild(b);
    });
  }
  function renderCombine() {
    combineWrap.innerHTML = '<div class="apx-ptitle">Combine — tune together</div>';
    data.gridAxes.forEach((knobId, ax) => {
      const meta = data.knobs.find(k => k.id === knobId);
      const row = sliderRow(meta.label, data.gridLevels, gridIdx[ax], (li) => {
        gridIdx[ax] = li; isolate = null; syncIsolateReset(); draw();
      });
      combineWrap.appendChild(row.node); row.node._ax = ax; combineWrap['_r' + ax] = row;
    });
  }
  function renderIsolate() {
    isolateWrap.innerHTML = '<div class="apx-ptitle">Isolate one current — others held at baseline</div>';
    const sel = el('select', 'apx-select');
    data.knobs.filter(k => k.tier === 'sweep').forEach(k => {
      const o = document.createElement('option'); o.value = k.id; o.textContent = k.label; sel.appendChild(o);
    });
    const holder = el('div');
    const buildSlider = () => {
      holder.innerHTML = '';
      const k = sel.value;
      const li0 = data.sweepLevels.indexOf(1.0);
      const row = sliderRow('scale ×', data.sweepLevels, li0, (li) => {
        isolate = (data.sweepLevels[li] === 1.0) ? null : { knob: k, level: li };
        combineWrap.querySelectorAll('input').forEach((inp, ax) => { inp.value = 2; combineWrap['_r' + ax].setLabel(2); });
        gridIdx = [2, 2, 2];
        draw();
      });
      holder.appendChild(row.node);
    };
    sel.addEventListener('change', buildSlider);
    isolateWrap.appendChild(sel); isolateWrap.appendChild(holder); buildSlider();
  }
  function syncIsolateReset() { /* moving a combine slider clears isolation (handled in draw) */ }
  function renderGlossary() {
    glossary.innerHTML = '<div class="apx-ptitle">Currents in this engine</div>';
    data.knobs.forEach(k => {
      const row = el('div', 'apx-gitem' + (k.unfamiliar ? ' unfamiliar' : ''));
      row.innerHTML = `<span>${k.label}</span>`;
      glossary.appendChild(row);
    });
  }

  function sliderRow(label, levels, idx, onInput) {
    const node = el('div', 'apx-ctrl');
    const row = el('div', 'fw-row');
    const nm = el('span', 'fw-name'); nm.textContent = label;
    const vl = el('span', 'fw-val'); vl.textContent = fmt(levels[idx]);
    const inp = document.createElement('input');
    inp.type = 'range'; inp.min = 0; inp.max = levels.length - 1; inp.step = 1; inp.value = idx;
    inp.setAttribute('aria-label', label);
    inp.addEventListener('input', () => { vl.textContent = fmt(levels[+inp.value]); onInput(+inp.value); });
    row.append(nm, vl); node.append(row, inp);
    return { node, setLabel: (i) => { vl.textContent = fmt(levels[i]); } };
  }
  function fmt(x) { return (x === 1.0 ? '1.00' : x.toFixed(2)) + '×'; }

  // ── plot ─────────────────────────────────────────────────────
  function draw() {
    if (destroyed || !data) return;
    const s = fit(canvas), ctx = s.ctx, W = s.w, H = s.h;
    const st = cvar('--fig-stage'), gr = cvar('--fig-grid'), ax = cvar('--fig-axis'),
      mut = cvar('--fig-muted'), cr = cvar('--fig-crimson');
    const padL = 42, padB = 30, padT = 14, padR = 14, pw = W - padL - padR, ph = H - padT - padB;
    const tmax = data.t_ms[data.t_ms.length - 1], vlo = data.vmin, vhi = data.vmax;
    const X = t => padL + t / tmax * pw, Y = v => padT + ph - (v - vlo) / (vhi - vlo) * ph;
    ctx.fillStyle = st; ctx.fillRect(0, 0, W, H);
    // grid + y ticks (mV)
    ctx.strokeStyle = gr; ctx.fillStyle = ax; ctx.lineWidth = 1;
    ctx.font = '10px ' + (cvar('--font-mono') || 'monospace'); ctx.textBaseline = 'middle';
    for (let mv = -80; mv <= 40; mv += 40) {
      ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(padL, Y(mv)); ctx.lineTo(W - padR, Y(mv)); ctx.stroke();
      ctx.globalAlpha = 1; ctx.textAlign = 'right'; ctx.fillText(mv, padL - 6, Y(mv));
    }
    ctx.textAlign = 'center';
    for (let t = 0; t <= tmax; t += tmax / 4) ctx.fillText(Math.round(t), X(t), H - padB + 12);
    ctx.fillText('mV', 12, padT + 4); ctx.fillText('t (ms)', W - padR - 20, H - 6);
    // baseline (grey) + current (crimson)
    const drawTrace = (vm, color, w) => {
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineJoin = 'round'; ctx.beginPath();
      for (let i = 0; i < vm.length; i++) { const x = X(data.t_ms[i]), y = Y(vm[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    };
    drawTrace(b64toVm(data.baseline.vm), mut, 1.5);
    const cur = currentTrace();
    drawTrace(b64toVm(cur.vm), cr, 2.4);
    renderMetrics(cur);
  }
  function renderMetrics(tr) {
    const cells = [['APD₉₀', tr.apd90, 'ms'], ['APD₅₀', tr.apd50, 'ms'], ['dV/dt', tr.dvdt, 'V/s'],
      ['V_rest', tr.vrest, 'mV'], ['V_peak', tr.vpeak, 'mV']];
    if (data.beating === 'spontaneous') cells.push(['cycle', tr.cl, 'ms']);
    metrics.innerHTML = cells.map(([k, v, u]) =>
      `<div class="apx-m"><b>${v == null ? '—' : v}</b><span>${k}${u ? ' · ' + u : ''}</span></div>`).join('');
  }

  load();
  return { redraw: draw, destroy() { destroyed = true; if (bar.parentElement) bar.remove(); } };
}
