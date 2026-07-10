#!/usr/bin/env python3
"""Generate exact real-engine AP traces for the Ion-Current Playground.

For each production cardiac_core ionic engine (TTP06, ORd, PHAS13, MHAS13), pace a
single cell over a grid of conductance-knob scalings and record the resulting
action potential. Every trace is REAL engine output (no interpolation): the page
snaps sliders to these precomputed levels.

Batching: all cells of a config (baseline + primary-knob grid + per-knob 1-D sweeps)
run in ONE batched pacing loop — conductance fields are set to per-cell tensors that
broadcast inside `model.step` (verified: currents multiply `p.G* * (V-E)` elementwise).

Output: website/data/ap_explorer/<engine>[_<celltype>].json  (Vm quantized to uint8+base64).

Usage:
    python website/build/gen_ap_traces.py --engines ttp06:EPI            # one config (probe)
    python website/build/gen_ap_traces.py                                # full bank
Tunables: --beats, --dt, --bcl, --compile. Run on CPU (default) or --device cuda.
"""
import argparse, base64, itertools, json, pathlib, time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]          # website/
OUT = ROOT / "data" / "ap_explorer"

# ── knob metadata: field name on model.params → (label, is-unfamiliar) ──
KNOB_LABELS = {
    'GNa': 'I_Na — fast sodium (upstroke)', 'g_Na': 'I_Na — fast sodium (upstroke)',
    'PCa': 'I_CaL — L-type calcium (plateau)', 'g_CaL': 'I_CaL — L-type calcium (plateau)',
    'GKr': 'I_Kr — rapid delayed-rectifier K⁺', 'g_Kr': 'I_Kr — rapid delayed-rectifier K⁺',
    'GKs': 'I_Ks — slow delayed-rectifier K⁺', 'g_Ks': 'I_Ks — slow delayed-rectifier K⁺',
    'GK1': 'I_K1 — inward-rectifier K⁺ (resting)', 'g_K1': 'I_K1 — inward-rectifier K⁺ (resting)',
    'Gto': 'I_to — transient-outward K⁺ (notch)', 'g_to': 'I_to — transient-outward K⁺ (notch)',
    'GNaL': 'I_NaL — late sodium', 'g_f': 'I_f — funny current (HCN pacemaker)★',
    'kNaCa': 'I_NaCa — Na/Ca exchanger★', 'Gncx': 'I_NaCa — Na/Ca exchanger★',
    'PNaK': 'I_NaK — Na/K ATPase pump★', 'Pnak': 'I_NaK — Na/K ATPase pump★',
    'GpCa': 'I_pCa — sarcolemmal Ca pump★', 'g_pCa': 'I_pCa — sarcolemmal Ca pump★',
    'GpK': 'I_pK — plateau K⁺', 'GbNa': 'I_bNa — background Na leak★',
    'GbCa': 'I_bCa — background Ca leak★', 'g_bNa': 'I_bNa — background Na leak★',
    'g_bCa': 'I_bCa — background Ca leak★',
}

# ── per-engine configuration ──
ENGINES = {
    'ttp06': dict(cell_types=['ENDO', 'EPI', 'M_CELL'], beating='paced',
                  primary=['GNa', 'PCa', 'GKr'],
                  secondary=['GKs', 'GK1', 'Gto', 'GpCa', 'GbNa', 'GbCa']),
    'ord':   dict(cell_types=['ENDO', 'EPI', 'M_CELL'], beating='paced',
                  primary=['GNa', 'PCa', 'GKr'],
                  secondary=['GNaL', 'GKs', 'GK1', 'Gto', 'GpCa']),
    'phas13': dict(cell_types=[None], beating='spontaneous',
                   primary=['g_f', 'g_CaL', 'g_Kr'],
                   secondary=['g_Na', 'g_Ks', 'g_K1', 'g_to', 'kNaCa', 'PNaK']),
    'mhas13': dict(cell_types=[None], beating='paced',
                   primary=['g_Na', 'g_CaL', 'g_Kr'],
                   secondary=['g_Ks', 'g_to', 'kNaCa', 'PNaK']),
}
GRID_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]                 # combinable primary knobs
SWEEP_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # isolated 1-D sweeps
VMIN, VMAX = -95.0, 55.0                                # int8 quantization window (mV)


def build_specs(cfg):
    """Return (specs, grid_index, sweep_index). Each spec = {knob: scale} for all knobs."""
    primary, secondary = cfg['primary'], cfg['secondary']
    specs, grid_index, sweep_index = [], [], {}
    base = {k: 1.0 for k in primary + secondary}
    specs.append(dict(base)); baseline_i = 0                       # index 0 = baseline
    # primary grid (cross-product)
    for combo in itertools.product(range(len(GRID_LEVELS)), repeat=len(primary)):
        s = dict(base)
        for ki, li in zip(primary, combo):
            s[ki] = GRID_LEVELS[li]
        grid_index.append({'idx': list(combo), 'spec': len(specs)})
        specs.append(s)
    # 1-D isolated sweeps
    for kn in secondary:
        sweep_index[kn] = []
        for lv in SWEEP_LEVELS:
            s = dict(base); s[kn] = lv
            sweep_index[kn].append({'level': lv, 'spec': len(specs)})
            specs.append(s)
    return specs, grid_index, sweep_index, baseline_i


def apply_knobs(model, specs):
    """Set each knob field on model.params to an (N,) tensor of scale×default.
    Returns (N, defaults) where defaults[kn] is the original scalar (scale=1.0) value."""
    N = len(specs)
    dev, dt_ = model.device, model.dtype
    defaults = {}
    for kn in specs[0].keys():
        defaults[kn] = float(getattr(model.params, kn))
        scales = torch.tensor([s[kn] for s in specs], dtype=dt_, device=dev)
        setattr(model.params, kn, scales * defaults[kn])
    return N, defaults


def pace(model, N, dt, bcl, beats, spontaneous, stepfn, record_ms=None):
    """Batched pacing/free-run. Records the tail window at FULL dt resolution so
    metrics (Vpeak, dV/dt_max — sub-ms upstroke) are exact. Returns (t_ms, Vm[T,N]).
    Paced: records the last beat. Spontaneous: records the last `record_ms` window."""
    V = torch.full((N,), model.V_rest, dtype=model.dtype, device=model.device)
    states = model.get_initial_state(n_cells=N)
    steps_per_beat = int(round(bcl / dt))
    n_steps = beats * steps_per_beat
    rec_len = int(round((record_ms if record_ms else bcl) / dt))
    rec_from = n_steps - rec_len
    Vfull = torch.empty((rec_len, N), dtype=model.dtype)
    zeros = torch.zeros(N, dtype=model.dtype, device=model.device)
    stimvec = torch.full((N,), -80.0, dtype=model.dtype, device=model.device)
    for i in range(n_steps):
        if spontaneous:
            Istim = None
        else:
            phase = (i % steps_per_beat) * dt
            Istim = stimvec if phase < 1.0 else zeros
        V, states = stepfn(V, states, dt, Istim)
        if i >= rec_from:
            Vfull[i - rec_from] = V.detach().to('cpu')
    tfull = np.arange(rec_len) * dt
    return tfull, Vfull.numpy()                               # (T,), (T, N) at full dt


def _apd_single(t, v, pk, lo, hi):
    """APD90/APD50 for the single beat peaking at index pk, searched within [lo, hi)."""
    seg_v, seg_t = v[lo:hi], t[lo:hi]
    pkl = pk - lo
    vmx, vmn = float(v[pk]), float(seg_v.min())
    if vmx - vmn < 20:
        return {'apd90': None, 'apd50': None}
    res = {}
    for pct, key in ((0.9, 'apd90'), (0.5, 'apd50')):
        thr = vmx - pct * (vmx - vmn)
        up = next((k for k in range(pkl) if seg_v[k] >= thr), None)
        dn = next((k for k in range(pkl, len(seg_v)) if seg_v[k] <= thr), None)
        res[key] = round(float(seg_t[dn] - seg_t[up]), 1) if up is not None and dn is not None else None
    return res


def metrics(t, v, spontaneous):
    """Per-cell AP metrics. Paced: whole trace is one beat. Spontaneous: CL from peak
    spacing and APD measured on a single (complete) beat — NOT across the multi-beat window."""
    v = np.asarray(v); t = np.asarray(t)
    vmin, vmax = float(v.min()), float(v.max())
    out = dict(vrest=round(vmin, 1), vpeak=round(vmax, 1), apd90=None, apd50=None, dvdt=None)
    if vmax - vmin < 20:                                       # no real AP (e.g. quiescent)
        return out
    out['dvdt'] = round(float((np.diff(v) / np.diff(t)).max()), 1)
    if spontaneous:
        raw = [k for k in range(1, len(v) - 1) if v[k] > 0 and v[k] >= v[k-1] and v[k] > v[k+1]]
        peaks = []                                            # dedupe bumps within one AP (<200 ms)
        for k in raw:
            if not peaks or t[k] - t[peaks[-1]] > 200:
                peaks.append(k)
        if len(peaks) >= 2:
            out['cl'] = round(float(t[peaks[-1]] - t[peaks[-2]]), 1)
            lo = peaks[-3] if len(peaks) >= 3 else 0           # bound the 2nd-to-last (complete) beat
            out.update(_apd_single(t, v, peaks[-2], lo, peaks[-1]))
        return out
    out.update(_apd_single(t, v, int(v.argmax()), 0, len(v)))  # paced
    return out


def quantize(v):
    q = np.clip((v - VMIN) / (VMAX - VMIN) * 255.0, 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode('ascii')


def gen_config(engine, celltype, args, stepfn_wrap):
    from cardiac_core.ionic.registry import build_ionic_model
    cfg = ENGINES[engine]
    spontaneous = cfg['beating'] == 'spontaneous'
    model = build_ionic_model(engine, cell_type=celltype or 'ENDO', device=args.device)
    specs, grid_index, sweep_index, base_i = build_specs(cfg)
    N, defaults = apply_knobs(model, specs)
    stepfn = stepfn_wrap(model)
    t0 = time.time()
    if spontaneous:
        # free-run (no stimulus) long enough to settle, record a window with ≥2 cycles
        t, Vm = pace(model, N, args.dt, args.spont_ms, 1, True, stepfn, record_ms=args.spont_record_ms)
    else:
        t, Vm = pace(model, N, args.dt, args.bcl, args.beats, False, stepfn)
    dur = time.time() - t0
    # metrics on FULL-res trace; store a ~400-sample downsample for plotting
    T = Vm.shape[0]
    stride = max(1, T // 400)
    t_ds = t[::stride]
    knobs_meta = [dict(id=k, label=KNOB_LABELS.get(k, k), default=defaults[k],
                       tier='grid' if k in cfg['primary'] else 'sweep',
                       unfamiliar=KNOB_LABELS.get(k, '').endswith('★')) for k in cfg['primary'] + cfg['secondary']]
    def trace(i):
        m = metrics(t, Vm[:, i], spontaneous)           # exact metrics (full dt)
        return {'vm': quantize(Vm[::stride, i]), **m}   # downsampled for the plot
    data = dict(
        engine=engine, cellType=celltype, beating=cfg['beating'], bcl=args.bcl,
        t_ms=[round(float(x), 2) for x in t_ds], vmin=VMIN, vmax=VMAX,
        gridLevels=GRID_LEVELS, sweepLevels=SWEEP_LEVELS,
        knobs=knobs_meta, gridAxes=cfg['primary'],
        baseline=trace(base_i),
        grid=[{'idx': g['idx'], **trace(g['spec'])} for g in grid_index],
        sweeps={kn: [{'level': e['level'], **trace(e['spec'])} for e in sweep_index[kn]]
                for kn in sweep_index},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fname = f"{engine}" + (f"_{celltype.lower()}" if celltype else "") + ".json"
    (OUT / fname).write_text(json.dumps(data))
    b = data['baseline']
    print(f"  {engine:7s} {str(celltype):7s}: {N} cells, {args.beats} beats in {dur:.0f}s "
          f"| baseline APD90={b['apd90']} dVdt={b['dvdt']} Vpeak={b['vpeak']}"
          + (f" CL={b.get('cl')}" if spontaneous else "") + f" → {fname} ({(OUT/fname).stat().st_size//1024} KB)",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engines', default='', help='comma list e.g. ttp06:EPI,ord:ENDO (default: all)')
    ap.add_argument('--beats', type=int, default=12)
    ap.add_argument('--dt', type=float, default=0.02)
    ap.add_argument('--bcl', type=float, default=1000.0)
    ap.add_argument('--spont-ms', type=float, default=15000.0, help='PHAS13 free-run duration')
    ap.add_argument('--spont-record-ms', type=float, default=4000.0, help='PHAS13 recorded tail window')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()

    def stepfn_wrap(model):
        return torch.compile(model.step) if args.compile else model.step

    if args.engines:
        jobs = []
        for tok in args.engines.split(','):
            e, _, ct = tok.partition(':')
            jobs.append((e, ct or None))
    else:
        jobs = [(e, ct) for e, cfg in ENGINES.items() for ct in cfg['cell_types']]
    print(f"generating {len(jobs)} configs on {args.device}, {args.beats} beats, dt={args.dt}", flush=True)
    for e, ct in jobs:
        gen_config(e, ct, args, stepfn_wrap)


if __name__ == '__main__':
    main()
