#!/usr/bin/env python3
"""
Optimizer V1 — First Test Run (Optimized)

Speed improvements applied:
  S1: Batched cell evaluation (M cells in one sim)
  S2: Ionic subcycling in tissue runner (dt_cell/dt fewer ionic evals)
  S4: dt_cell=0.2ms (validated stable)
  S5: Analytical CV∝√D tissue fitting (1-3 sims instead of 12+)
  Cable shortened to 1.5cm
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'Monodomain', 'Engine_V5.4'))

import torch
from time import perf_counter

from tuner.config import TuningConfig, TuningTargets, get_param_names, theta_to_dict
from tuner.cell_runner import run_single_cell, run_single_cell_batch, extract_biomarkers_batch
from tuner.tissue_runner import run_cv_measurement
from tuner.cell_fitter import fit_cell
from tuner.tissue_fitter import fit_tissue


def main():
    device = 'cpu'  # Batched CPU faster than GPU for single-cell ODE

    config = TuningConfig(
        ionic_model='phas13',
        tier=1,
        device=device,
        dt=0.02,                    # tissue dt (CFL)
        dt_cell=0.2,               # cell dt (validated stable)
        dx_cm=0.04,                 # from spiral_wave_s1s2
        cable_length_cm=1.5,        # shorter cable (faster)
        n_beats=5,
        pacing_cl=1000.0,
        stim_amplitude=-5.0,
        stim_duration=2.0,
        n_iterations=10,
    )

    targets = TuningTargets(
        apd_90=350.0,
        cv_longitudinal=15.0,
        cv_transverse=7.5,
        dvdt_max=25.0,
        spontaneous_cl=None,
    )

    print("=" * 70)
    print("Optimizer V1 — First Test (Optimized)")
    print(f"  Device:  {device}")
    print(f"  Cell dt: {config.dt_cell}ms  |  Tissue dt: {config.dt}ms")
    print(f"  dx: {config.dx_cm}cm  |  Cable: {config.cable_length_cm}cm")
    print(f"  Tier {config.tier}: {get_param_names(config.tier)}")
    print("=" * 70)
    t_start = perf_counter()

    # Step 1: Baseline
    print("\n--- Step 1: Baseline ---")
    theta_base = torch.ones(1, 6, dtype=torch.float64)
    t0 = perf_counter()
    t_arr, V_all = run_single_cell_batch(theta_base, config)
    t_cell = perf_counter() - t0

    from tuner.metrics import measure_apd, measure_dvdt_max, measure_v_rest, measure_peak
    V = V_all[0]
    apd = measure_apd(V, t_arr)
    dvdt = measure_dvdt_max(V, t_arr)
    print(f"  Cell: {t_cell:.1f}s  APD={'N/A' if apd is None else f'{apd:.0f}'}ms  "
          f"dVdt={dvdt:.1f}V/s  Vpeak={measure_peak(V):.1f}mV")

    D_ref = 0.0001
    t0 = perf_counter()
    cv_base = run_cv_measurement(theta_base[0], D_ref, config, n_beats=3)
    t_tissue = perf_counter() - t0
    cv_str = f"{cv_base.cv:.1f}" if cv_base.cv else "N/A"
    print(f"  Tissue: {t_tissue:.1f}s  CV={cv_str}cm/s at D={D_ref}")

    # Step 2: Cell fitter
    print("\n--- Step 2: Cell Fitter (8 initial + 10 BO, batched) ---")
    t0 = perf_counter()
    cell_fit = fit_cell(config, targets, n_initial=8, n_iterations=10, verbose=True)
    t_fit = perf_counter() - t0
    print(f"\n  Cell fitter: {t_fit:.1f}s total")

    apd_errors = -cell_fit.pareto_Y[:, 0]
    best_idx = apd_errors.argmin().item()
    best_theta = cell_fit.pareto_X[best_idx]
    best_dict = theta_to_dict(best_theta, config.tier)
    print(f"  Best: { {k: f'{v:.3f}' for k,v in best_dict.items()} }")

    cell_best = run_single_cell(best_theta, config)
    if cell_best.apd90:
        print(f"  APD90={cell_best.apd90:.1f}ms (Δ={cell_best.apd90-targets.apd_90:+.1f})  "
              f"dVdt={cell_best.dvdt_max:.1f}V/s")

    # Step 3: Tissue fitter (analytical)
    print("\n--- Step 3: Tissue Fitter (analytical CV∝√D) ---")
    t0 = perf_counter()
    tissue_fit = fit_tissue(best_theta, config, targets, verbose=True)
    t_tissue_fit = perf_counter() - t0
    print(f"\n  Tissue fitter: {t_tissue_fit:.1f}s ({tissue_fit.n_sims} sims)")

    # Summary
    t_total = perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"COMPLETE — {t_total:.0f}s total")
    if cell_best.apd90:
        print(f"  APD: {cell_best.apd90:.0f}ms (target {targets.apd_90:.0f})")
    if cell_best.dvdt_max:
        print(f"  dVdt: {cell_best.dvdt_max:.0f}V/s (target {targets.dvdt_max:.0f})")
    print(f"  D_long={tissue_fit.D_long:.6f} -> CV={tissue_fit.cv_long_achieved:.1f}cm/s "
          f"(target {targets.cv_longitudinal:.1f})")
    print(f"  D_trans={tissue_fit.D_trans:.6f} -> CV={tissue_fit.cv_trans_achieved:.1f}cm/s "
          f"(target {targets.cv_transverse:.1f})")
    print(f"  Params: { {k: f'{v:.3f}' for k,v in best_dict.items()} }")
    print("=" * 70)


if __name__ == '__main__':
    main()
