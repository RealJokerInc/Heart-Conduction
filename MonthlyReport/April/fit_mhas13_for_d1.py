#!/usr/bin/env python3
"""
Fit MHAS13 (cell only, tier 2) and persist best theta as JSON for D1 plotting.
Mirrors run_mhas13.py Steps 1-2; skips tissue fit for speed.

Output: MonthlyReport/April/figures/D1_mhas13_theta.json
"""
import sys
import os
import json
import torch
from time import perf_counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Optimizer', 'V1'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Monodomain', 'Engine_V5.4'))

from tuner.config import TuningConfig, TuningTargets, get_param_names, theta_to_dict
from tuner.cell_runner import run_single_cell, run_single_cell_batch
from tuner.cell_fitter import fit_cell
from tuner.metrics import measure_apd, measure_dvdt_max, measure_v_rest


def main():
    config = TuningConfig(
        ionic_model='mhas13',
        tier=2,
        device='cpu',
        dt=0.02,
        dt_cell=0.2,
        dx_cm=0.04,
        cable_length_cm=1.5,
        n_beats=5,
        pacing_cl=1000.0,
        stim_amplitude=-40.0,
        stim_duration=2.0,
        n_iterations=10,
        seed=42,
    )
    targets = TuningTargets(
        apd_90=350.0,
        cv_longitudinal=15.0,
        cv_transverse=7.5,
        dvdt_max=25.0,
        spontaneous_cl=None,
        dvdt_max_upper=120.0,
        v_peak_max=60.0,
        v_rest_range=(-92.0, -70.0),
    )

    n_params = len(get_param_names(config.tier))
    print(f"MHAS13 cell fit — tier {config.tier}, {n_params} params, "
          f"n_initial={2*n_params}, n_iter={config.n_iterations}")
    print("=" * 70)

    # Baseline reference
    theta_base = torch.ones(1, n_params, dtype=torch.float64)
    t_arr_b, V_all_b = run_single_cell_batch(theta_base, config)
    V_b = V_all_b[0]
    apd_b = measure_apd(V_b, t_arr_b)
    dvdt_b = measure_dvdt_max(V_b, t_arr_b)
    vr_b = measure_v_rest(V_b, t_arr_b)
    print(f"Baseline: APD90={apd_b}  dVdt={dvdt_b:.1f}  V_rest={vr_b:.1f}")

    # Fit
    print("\nFitting...")
    t0 = perf_counter()
    result = fit_cell(config, targets, n_initial=2*n_params,
                      n_iterations=config.n_iterations, verbose=False)
    t = perf_counter() - t0
    print(f"Fit done in {t:.0f}s ({result.n_feasible}/{result.n_total} feasible)")

    # Best by APD-error objective
    apd_errors = -result.pareto_Y[:, 0]
    best_idx = int(apd_errors.argmin().item())
    best_theta = result.pareto_X[best_idx]
    best_dict = theta_to_dict(best_theta, config.tier)

    cell_best = run_single_cell(best_theta, config)
    print(f"\nBest result: APD90={cell_best.apd90:.1f}  dVdt={cell_best.dvdt_max:.1f}  "
          f"Vpeak={cell_best.v_peak:.1f}")
    for k, v in best_dict.items():
        print(f"   {k}: {v:.4f}")

    out = {
        'tier': int(config.tier),
        'param_names': get_param_names(config.tier),
        'theta_array': [float(x) for x in best_theta.tolist()],
        'theta_dict': {k: float(v) for k, v in best_dict.items()},
        'baseline_apd_ms': float(apd_b) if apd_b else None,
        'baseline_dvdt_Vps': float(dvdt_b) if dvdt_b else None,
        'fitted_apd_ms': float(cell_best.apd90) if cell_best.apd90 else None,
        'fitted_dvdt_Vps': float(cell_best.dvdt_max) if cell_best.dvdt_max else None,
        'fitted_vrest_mV': float(cell_best.v_rest),
        'config': {
            'dt_cell': config.dt_cell,
            'pacing_cl': config.pacing_cl,
            'stim_amplitude': config.stim_amplitude,
            'stim_duration': config.stim_duration,
            'n_beats': config.n_beats,
        },
    }
    out_dir = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'D1_mhas13_theta.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
