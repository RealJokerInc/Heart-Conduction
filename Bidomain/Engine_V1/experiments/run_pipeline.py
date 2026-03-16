#!/usr/bin/env python
"""
Triangle Merger Pipeline — Automated experiment + visualization + evaluation + report.

Usage:
    python run_pipeline.py                    # Full run (~45 min)
    python run_pipeline.py --quick            # Quick validation (~1 min)
    python run_pipeline.py --skip-experiment  # Use existing saved data
    python run_pipeline.py --skip-viz         # Skip visualization
    python run_pipeline.py --only-report      # Only evaluation + report
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.join(_HERE, '..')
_TESTS = os.path.join(_ENGINE, 'tests')
sys.path.insert(0, _ENGINE)
sys.path.insert(0, _TESTS)
sys.path.insert(0, _HERE)

torch.set_default_dtype(torch.float64)

# ============================================================
# Physical constants (independent of grid)
# ============================================================
SIGMA_I = 1.74
SIGMA_E = 6.25
CHI = 1400.0
CM = 1.0
D_I = SIGMA_I / (CHI * CM)
D_E = SIGMA_E / (CHI * CM)
D_EFF = D_I * D_E / (D_I + D_E)
KLEBER_RATIO = math.sqrt((D_I + D_E) / D_E)

# Grid parameter sets
FULL_PARAMS = dict(NX=1001, NY=161, DX=0.05, DT=0.01,
                   T_END=800.0, SAVE_EVERY=25.0, LX=50.0, LY=8.0)
QUICK_PARAMS = dict(NX=401, NY=41, DX=0.05, DT=0.01,
                    T_END=200.0, SAVE_EVERY=25.0, LX=20.0, LY=2.0)


def get_output_dir(quick=False):
    subdir = 'triangle_merger_quick' if quick else 'triangle_merger'
    return os.path.join(
        _HERE, '..', '..', '..', 'Research',
        'Q5_boundary_conduction_speedup', subdir)


def patch_module(mod, params, output_dir):
    """Patch module-level constants (NX, NY, LX, LY, T_END, OUTPUT_DIR, etc.)."""
    for key, val in params.items():
        if hasattr(mod, key):
            setattr(mod, key, val)
    mod.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Phase 1 — Run Experiment
# ============================================================
def phase1_experiment(quick=False):
    """Run 3 simulation configs, verify, and save data."""
    import triangle_merger as tm

    params = QUICK_PARAMS if quick else FULL_PARAMS
    output_dir = get_output_dir(quick)
    patch_module(tm, params, output_dir)

    configs = [
        ('monodomain_mehrstellen', tm.run_monodomain_mehrstellen),
        ('bidomain_5pt', lambda: tm.run_bidomain(stencil='5pt')),
        ('bidomain_mehrstellen', lambda: tm.run_bidomain(stencil='mehrstellen')),
    ]

    results = {}
    timings = {}
    for name, fn in configs:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")
        t0 = time.time()
        try:
            results[name] = fn()
            timings[name] = time.time() - t0
            print(f"  Completed in {timings[name]:.0f}s")
        except Exception as e:
            timings[name] = -(time.time() - t0)
            print(f"  FAILED after {abs(timings[name]):.0f}s: {e}")
            import traceback
            traceback.print_exc()

    if len(results) == 3:
        tm.verify_results(results['monodomain_mehrstellen'],
                          results['bidomain_5pt'],
                          results['bidomain_mehrstellen'])

    if results:
        tm.save_data(list(results.values()))

    with open(os.path.join(output_dir, 'timings.json'), 'w') as f:
        json.dump(timings, f, indent=2)

    return timings


# ============================================================
# Phase 2 — Run Visualization
# ============================================================
def phase2_viz(quick=False):
    """Generate all 7 plots."""
    import triangle_merger_viz as viz

    params = QUICK_PARAMS if quick else FULL_PARAMS
    output_dir = get_output_dir(quick)
    patch_module(viz, params, output_dir)

    viz.main()

    expected = [
        'wavefront_evolution.png', 'Vm_heatmaps.png', 'lead_vs_time.png',
        'front_range_vs_time.png', 'cv_profile_steady.png',
        'stencil_comparison.png', 'isochrone_map.png',
    ]
    all_ok = True
    for fname in expected:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {fname}")
            all_ok = False
        elif os.path.getsize(path) < 10240:
            print(f"  TOO SMALL: {fname} ({os.path.getsize(path)} bytes)")
            all_ok = False
    return all_ok


# ============================================================
# Phase 3 — Evaluate
# ============================================================
def _load_config(output_dir, name):
    """Load saved fronts, activation times, and times for one config."""
    fronts = torch.load(os.path.join(output_dir, f'{name}_fronts.pt'),
                        weights_only=True)
    act_time = torch.load(os.path.join(output_dir, f'{name}_act_time.pt'),
                          weights_only=True)
    with open(os.path.join(output_dir, f'{name}_times.json')) as f:
        times = json.load(f)
    return dict(fronts=fronts, act_time=act_time, times=times)


def _find_snap(times, target, tol=13.0):
    """Return index of snapshot closest to target within tolerance, or None."""
    for i, t in enumerate(times):
        if abs(t - target) < tol:
            return i
    return None


def phase3_evaluate(quick=False):
    """Compute quantitative metrics from saved data."""
    output_dir = get_output_dir(quick)
    params = QUICK_PARAMS if quick else FULL_PARAMS
    DX = params['DX']
    NY = params['NY']
    T_END = params['T_END']

    # Load data
    configs = {}
    for name in ['monodomain_mehrstellen', 'bidomain_5pt', 'bidomain_mehrstellen']:
        try:
            configs[name] = _load_config(output_dir, name)
        except FileNotFoundError as e:
            print(f"  WARNING: Missing data for {name}: {e}")

    metrics = dict(mode='quick' if quick else 'full', params=params)

    # --- 1. Assertion checks ---
    assertions = {}

    # 1a. Monodomain flat wavefront
    if 'monodomain_mehrstellen' in configs:
        mono = configs['monodomain_mehrstellen']
        max_dev = 0.0
        for i in range(len(mono['times'])):
            front = mono['fronts'][i].float()
            active = front > 0
            if active.sum() > 2:
                dev = (front[active].max() - front[active].min()).item() * DX
                max_dev = max(max_dev, dev)
        assertions['mono_flat'] = dict(
            passed=max_dev < 0.1, max_deviation_cm=max_dev, criterion='< 0.1 cm')

    # 1b. Edge leads center (bidomain mehrstellen, t > 50ms)
    if 'bidomain_mehrstellen' in configs:
        bi9 = configs['bidomain_mehrstellen']
        fail_times = []
        for i, t in enumerate(bi9['times']):
            if t < 50:
                continue
            front = bi9['fronts'][i]
            center = front[NY // 2].item()
            edge = max(front[1].item(), front[-2].item())
            if edge <= center and center > 0:
                fail_times.append(t)
        assertions['edge_leads'] = dict(passed=len(fail_times) == 0,
                                        fail_times=fail_times)

    # 1c. No NaN/Inf
    nan_ok = True
    for cfg in configs.values():
        if not torch.isfinite(cfg['fronts']).all():
            nan_ok = False
        finite_act = cfg['act_time'][cfg['act_time'] != float('inf')]
        if finite_act.numel() > 0 and not torch.isfinite(finite_act).all():
            nan_ok = False
    assertions['no_nan'] = dict(passed=nan_ok)

    # 1d. Wave stays in domain
    in_domain = True
    for cfg in configs.values():
        max_cm = cfg['fronts'].float().max().item() * DX
        if max_cm > params['LX'] - 2.0:
            in_domain = False
    assertions['in_domain'] = dict(passed=in_domain)

    metrics['assertions'] = assertions

    # --- 2. Triangle merger detection ---
    merger = {}
    for bi_name in ['bidomain_5pt', 'bidomain_mehrstellen']:
        if bi_name not in configs:
            continue
        bi = configs[bi_name]
        times_arr = np.array(bi['times'])
        fronts = bi['fronts'].float().numpy()

        ranges = []
        for i in range(len(times_arr)):
            f = fronts[i]
            active = f > 0
            if active.sum() > 2:
                ranges.append((f[active].max() - f[active].min()) * DX)
            else:
                ranges.append(0.0)
        ranges = np.array(ranges)

        peak_idx = int(np.argmax(ranges))
        peak_range = float(ranges[peak_idx])
        peak_time = float(times_arr[peak_idx])

        # Merger time: when range drops to 50% of peak after peak
        merger_time = None
        half_peak = peak_range * 0.5
        for i in range(peak_idx + 1, len(ranges)):
            if ranges[i] < half_peak:
                merger_time = float(times_arr[i])
                break

        ss_range = float(np.mean(ranges[-3:])) if len(ranges) >= 3 else float(ranges[-1])

        merger[bi_name] = dict(
            peak_range_cm=peak_range, peak_time_ms=peak_time,
            merger_time_ms=merger_time, steady_state_range_cm=ss_range)

    metrics['merger'] = merger

    # --- 3. Steady-state shape ---
    shape = {}
    for bi_name in ['bidomain_5pt', 'bidomain_mehrstellen']:
        if bi_name not in configs:
            continue
        bi = configs[bi_name]
        front = bi['fronts'][-1].float()
        center = front[NY // 2].item()
        edge = max(front[1].item(), front[-2].item())
        lead_cm = (edge - center) * DX

        if abs(lead_cm) < 0.05:
            shape_type = 'flat'
        elif lead_cm > 0:
            shape_type = 'triangle (edge leads)'
        else:
            shape_type = 'inverted (center leads)'
        shape[bi_name] = dict(edge_center_lead_cm=float(lead_cm), shape=shape_type)

    metrics['shape'] = shape

    # --- 4. Kleber CV ratio ---
    cv = {}
    for bi_name in ['bidomain_5pt', 'bidomain_mehrstellen']:
        if bi_name not in configs:
            continue
        bi = configs[bi_name]
        times_arr = np.array(bi['times'])
        fronts = bi['fronts'].float().numpy()

        late_start = 600.0 if not quick else T_END * 0.6
        late_end = T_END
        late_mask = (times_arr >= late_start) & (times_arr <= late_end)
        if late_mask.sum() < 2:
            cv[bi_name] = dict(cv_center_cm_s=None, cv_edge_cm_s=None,
                               ratio=None, theory_ratio=KLEBER_RATIO,
                               late_window=f'{late_start:.0f}-{late_end:.0f}ms')
            continue

        late_idx = np.where(late_mask)[0]
        cv_profiles = []
        for k in range(len(late_idx) - 1):
            i1, i2 = late_idx[k], late_idx[k + 1]
            dx_front = (fronts[i2] - fronts[i1]) * DX
            dt_front = times_arr[i2] - times_arr[i1]
            cv_profiles.append(dx_front / dt_front * 1000)  # cm/s

        if cv_profiles:
            cv_mean = np.mean(cv_profiles, axis=0)
            cv_center = float(cv_mean[NY // 2])
            cv_edge = float(max(cv_mean[1], cv_mean[-2]))
            ratio = cv_edge / cv_center if cv_center > 0 else None
        else:
            cv_center = cv_edge = ratio = None

        cv[bi_name] = dict(
            cv_center_cm_s=cv_center, cv_edge_cm_s=cv_edge,
            ratio=ratio, theory_ratio=KLEBER_RATIO,
            late_window=f'{late_start:.0f}-{late_end:.0f}ms')

    metrics['cv'] = cv

    # --- 5. Stencil comparison ---
    stencil_diff = {}
    if 'bidomain_5pt' in configs and 'bidomain_mehrstellen' in configs:
        bi5, bi9 = configs['bidomain_5pt'], configs['bidomain_mehrstellen']
        key_times = [200, 400, 600, 800] if not quick else [50, 100, 150, 200]
        key_times = [t for t in key_times if t <= T_END]

        for kt in key_times:
            idx5 = _find_snap(bi5['times'], kt)
            idx9 = _find_snap(bi9['times'], kt)
            if idx5 is not None and idx9 is not None:
                diff = (bi5['fronts'][idx5].float() -
                        bi9['fronts'][idx9].float()).abs() * DX
                stencil_diff[f't={kt:.0f}ms'] = dict(
                    max_cm=float(diff.max()), mean_cm=float(diff.mean()))

    metrics['stencil_diff'] = stencil_diff

    # Save metrics
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=_convert)

    return metrics


# ============================================================
# Phase 4 — Write Report
# ============================================================
def phase4_report(metrics, quick=False):
    """Generate REPORT.md from computed metrics."""
    output_dir = get_output_dir(quick)
    params = metrics['params']
    mode = 'Quick' if quick else 'Full'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    L = []  # output lines
    L.append(f'# Triangle Merger Experiment Report')
    L.append(f'')
    L.append(f'Generated: {timestamp} | Mode: **{mode}**')
    L.append(f'')

    # -- Executive Summary --
    L.append('## Executive Summary')
    L.append('')
    merger = metrics.get('merger', {})
    bi9_m = merger.get('bidomain_mehrstellen', {})
    cv_data = metrics.get('cv', {})
    bi9_cv = cv_data.get('bidomain_mehrstellen', {})

    if bi9_m.get('merger_time_ms'):
        L.append(f"Triangle merger observed: peak wavefront range of "
                 f"{bi9_m['peak_range_cm']:.2f} cm at t={bi9_m['peak_time_ms']:.0f}ms, "
                 f"merging to 50% by t={bi9_m['merger_time_ms']:.0f}ms.")
    elif bi9_m:
        L.append(f"Triangle formation observed: peak wavefront range of "
                 f"{bi9_m['peak_range_cm']:.2f} cm at t={bi9_m['peak_time_ms']:.0f}ms. "
                 f"Merger not completed within simulation time.")

    if bi9_cv and bi9_cv.get('ratio'):
        L.append(f"Kleber CV ratio: {bi9_cv['ratio']:.4f} "
                 f"(theory: {KLEBER_RATIO:.4f}).")

    stencil = metrics.get('stencil_diff', {})
    if stencil:
        max_d = max(v['max_cm'] for v in stencil.values())
        L.append(f"Stencil difference (5pt vs Mehrstellen): max {max_d:.3f} cm.")
    L.append('')

    # -- Setup --
    L.append('## Setup')
    L.append('')
    L.append('| Parameter | Value |')
    L.append('|-----------|-------|')
    L.append(f"| Domain | {params['LX']:.0f} x {params['LY']:.0f} cm |")
    L.append(f"| Grid | {params['NX']} x {params['NY']} |")
    L.append(f"| dx | {params['DX']} cm |")
    L.append(f"| dt | {params['DT']} ms |")
    L.append(f"| T_end | {params['T_END']:.0f} ms |")
    L.append(f"| sigma_i | {SIGMA_I} mS/cm |")
    L.append(f"| sigma_e | {SIGMA_E} mS/cm |")
    L.append(f"| D_i | {D_I:.6f} cm\u00b2/ms |")
    L.append(f"| D_e | {D_E:.6f} cm\u00b2/ms |")
    L.append(f"| D_eff | {D_EFF:.6f} cm\u00b2/ms |")
    L.append(f"| Kleber theory ratio | {KLEBER_RATIO:.4f} |")
    L.append('')

    # -- Configurations --
    L.append('## Configurations')
    L.append('')
    timings_path = os.path.join(output_dir, 'timings.json')
    timings = {}
    if os.path.exists(timings_path):
        with open(timings_path) as f:
            timings = json.load(f)

    L.append('| Config | BCs | Stencil | Wall-clock |')
    L.append('|--------|-----|---------|------------|')
    for name, bc, st in [
        ('monodomain_mehrstellen', 'Neumann', 'Mehrstellen 9pt'),
        ('bidomain_5pt', 'bath_tb', '5-point'),
        ('bidomain_mehrstellen', 'bath_tb', 'Mehrstellen 9pt'),
    ]:
        t = timings.get(name)
        t_str = f"{abs(t):.0f}s{' (FAILED)' if t < 0 else ''}" if t is not None else 'N/A'
        L.append(f'| {name} | {bc} | {st} | {t_str} |')
    L.append('')

    # -- Results --
    L.append('## Results')
    L.append('')

    # 1. Assertions
    L.append('### 1. Embedded Assertions')
    L.append('')
    L.append('| Check | Result | Detail |')
    L.append('|-------|--------|--------|')
    asserts = metrics.get('assertions', {})

    a = asserts.get('mono_flat', {})
    L.append(f"| Monodomain flat | {'PASS' if a.get('passed') else 'FAIL'} | "
             f"max deviation = {a.get('max_deviation_cm', 0):.3f} cm "
             f"({a.get('criterion', '')}) |")

    a = asserts.get('edge_leads', {})
    detail = 'all times' if a.get('passed') else f"failed at t={a.get('fail_times', [])}"
    L.append(f"| Edge leads center | {'PASS' if a.get('passed') else 'FAIL'} | {detail} |")

    a = asserts.get('no_nan', {})
    L.append(f"| No NaN/Inf | {'PASS' if a.get('passed') else 'FAIL'} | \u2014 |")

    a = asserts.get('in_domain', {})
    L.append(f"| Wave in domain | {'PASS' if a.get('passed') else 'FAIL'} | \u2014 |")
    L.append('')

    # 2. Triangle Merger
    L.append('### 2. Triangle Merger')
    L.append('')
    if merger:
        L.append('| Config | Peak range (cm) | Peak time (ms) | Merger time (ms) | SS range (cm) |')
        L.append('|--------|-----------------|----------------|------------------|---------------|')
        for name, m in merger.items():
            mt = f"{m['merger_time_ms']:.0f}" if m.get('merger_time_ms') else 'N/A'
            L.append(f"| {name} | {m['peak_range_cm']:.3f} | "
                     f"{m['peak_time_ms']:.0f} | {mt} | "
                     f"{m['steady_state_range_cm']:.3f} |")
        L.append('')
        L.append('![Front range vs time](front_range_vs_time.png)')
    else:
        L.append('No merger data available.')
    L.append('')

    # 3. Steady-State Shape
    L.append('### 3. Steady-State Shape')
    L.append('')
    shape = metrics.get('shape', {})
    if shape:
        L.append('| Config | Edge-center lead (cm) | Shape |')
        L.append('|--------|-----------------------|-------|')
        for name, s in shape.items():
            L.append(f"| {name} | {s['edge_center_lead_cm']:.3f} | {s['shape']} |")
        L.append('')
        L.append('![Wavefront evolution](wavefront_evolution.png)')
    L.append('')

    # 4. Kleber Effect
    L.append('### 4. Kleber Effect')
    L.append('')
    if cv_data:
        L.append('| Config | CV center (cm/s) | CV edge (cm/s) | Ratio | Theory |')
        L.append('|--------|------------------|----------------|-------|--------|')
        for name, c in cv_data.items():
            def _f(v, fmt='.1f'):
                return f'{v:{fmt}}' if v is not None else 'N/A'
            L.append(f"| {name} | {_f(c.get('cv_center_cm_s'))} | "
                     f"{_f(c.get('cv_edge_cm_s'))} | "
                     f"{_f(c.get('ratio'), '.4f')} | "
                     f"{c.get('theory_ratio', KLEBER_RATIO):.4f} |")
        L.append('')
        L.append('![CV profile](cv_profile_steady.png)')
    L.append('')

    # 5. Stencil Comparison
    L.append('### 5. Stencil Comparison (5pt vs Mehrstellen)')
    L.append('')
    if stencil:
        L.append('| Time | Max |front_5pt - front_9pt| (cm) | Mean (cm) |')
        L.append('|------|-------------------------------|-----------|')
        for tl, d in stencil.items():
            L.append(f"| {tl} | {d['max_cm']:.4f} | {d['mean_cm']:.4f} |")
        L.append('')
        L.append('![Stencil comparison](stencil_comparison.png)')
    L.append('')

    # -- Visualizations --
    L.append('## Visualizations')
    L.append('')
    viz_files = [
        ('wavefront_evolution.png', 'Wavefront deviation vs monodomain flat reference (2x4 grid)'),
        ('Vm_heatmaps.png', 'Vm heatmaps zoomed to wavefront (2x4 grid)'),
        ('lead_vs_time.png', 'Edge/quarter lead distance vs time'),
        ('front_range_vs_time.png', 'Total wavefront range (max-min) vs time'),
        ('cv_profile_steady.png', 'Local CV as function of y at late time'),
        ('stencil_comparison.png', 'Absolute front difference between stencils'),
        ('isochrone_map.png', 'Activation time isochrone contours'),
    ]
    for fname, desc in viz_files:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            size = os.path.getsize(path) // 1024
            L.append(f'- `{fname}` ({size} KB) \u2014 {desc}')
        else:
            L.append(f'- `{fname}` (MISSING) \u2014 {desc}')
    L.append('')

    # -- Conclusions --
    L.append('## Conclusions')
    L.append('')
    all_pass = all(a.get('passed', False) for a in asserts.values())
    integrity_msg = ('All 4 assertions passed.' if all_pass
                     else 'Some assertions failed \u2014 see table above.')
    L.append(f"1. **Simulation integrity**: {integrity_msg}")

    if bi9_m:
        if bi9_m.get('merger_time_ms'):
            L.append(f"2. **Triangle merger**: Confirmed. Triangular wavefront develops due to "
                     f"bath-coupled boundary speedup, peaks at t={bi9_m['peak_time_ms']:.0f}ms, "
                     f"then merges as the faster edge waves coalesce.")
        else:
            L.append(f"2. **Triangle merger**: Triangle formation observed (peak range "
                     f"{bi9_m['peak_range_cm']:.2f} cm) but full merger not reached in "
                     f"{params['T_END']:.0f}ms simulation.")

    if bi9_cv and bi9_cv.get('ratio'):
        pct = abs(bi9_cv['ratio'] - KLEBER_RATIO) / KLEBER_RATIO * 100
        kleber_msg = ('Good agreement.' if pct < 10
                     else 'Significant deviation \u2014 see discussion.')
        L.append(f"3. **Kleber effect**: Measured CV ratio = {bi9_cv['ratio']:.4f}, "
                 f"theory = {KLEBER_RATIO:.4f} ({pct:.1f}% difference). {kleber_msg}")

    stencil_vals = list(stencil.values())
    if stencil_vals:
        max_diff = max(v['max_cm'] for v in stencil_vals)
        L.append(f"4. **Stencil effect**: Maximum front difference = {max_diff:.3f} cm. "
                 f"{'Negligible impact on wavefront position.' if max_diff < 0.5 else 'Measurable stencil isotropy effect.'}")

    L.append('')

    report_path = os.path.join(output_dir, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(L))

    print(f"\nReport written to: {report_path}")
    return report_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Triangle Merger Pipeline')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 401x41 grid, 200ms (~1 min)')
    parser.add_argument('--skip-experiment', action='store_true',
                        help='Skip Phase 1, use existing data')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip Phase 2')
    parser.add_argument('--only-report', action='store_true',
                        help='Only Phase 3+4 (assumes data + plots exist)')
    args = parser.parse_args()

    quick = args.quick
    mode = 'QUICK' if quick else 'FULL'
    output_dir = get_output_dir(quick)

    print(f"Triangle Merger Pipeline \u2014 {mode} mode")
    print(f"Output: {output_dir}")
    print('=' * 60)

    t_total = time.time()

    # Phase 1
    if not args.skip_experiment and not args.only_report:
        print(f"\n{'=' * 60}")
        print('PHASE 1: Run Experiment')
        print('=' * 60)
        timings = phase1_experiment(quick=quick)
        for name, t in timings.items():
            status = f"{t:.0f}s" if t > 0 else f"FAILED ({abs(t):.0f}s)"
            print(f"  {name}: {status}")

    # Phase 2
    if not args.skip_viz and not args.only_report:
        print(f"\n{'=' * 60}")
        print('PHASE 2: Run Visualization')
        print('=' * 60)
        viz_ok = phase2_viz(quick=quick)
        print(f"  Visualization: {'OK' if viz_ok else 'ISSUES'}")

    # Phase 3
    print(f"\n{'=' * 60}")
    print('PHASE 3: Evaluate')
    print('=' * 60)
    metrics = phase3_evaluate(quick=quick)
    for name, a in metrics.get('assertions', {}).items():
        print(f"  {name}: {'PASS' if a.get('passed') else 'FAIL'}")

    # Phase 4
    print(f"\n{'=' * 60}")
    print('PHASE 4: Write Report')
    print('=' * 60)
    report_path = phase4_report(metrics, quick=quick)

    total = time.time() - t_total
    print(f"\nPipeline complete in {total:.0f}s")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()
