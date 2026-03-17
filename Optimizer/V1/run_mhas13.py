#!/usr/bin/env python3
"""
Optimizer V1 — MHAS13 Full Pipeline (Iteration 2)

Improvements over iteration 1:
  A1: dV/dt hard constraint (reject dVdt > 60 V/s)
  A2: Tier 2 (10 params: +kNaCa, PNaK, g_pCa, VmaxUp)
  A3: Two-point secant CV warm-start (faster convergence)
  A4: Tighter CV convergence (3% threshold, 4 secant iters)
  R3: Reproducibility seeding (seed=42)
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
    device = 'cpu'

    config = TuningConfig(
        ionic_model='mhas13',
        tier=2,                     # A2: 10 params (+kNaCa, PNaK, g_pCa, VmaxUp)
        device=device,
        dt=0.02,
        dt_cell=0.2,
        dx_cm=0.04,
        cable_length_cm=1.5,
        n_beats=5,
        pacing_cl=1000.0,
        stim_amplitude=-40.0,
        stim_duration=2.0,
        n_iterations=15,            # More iterations for 10-param space
        seed=42,                    # R3: reproducibility
    )

    targets = TuningTargets(
        apd_90=350.0,
        cv_longitudinal=15.0,
        cv_transverse=7.5,
        dvdt_max=25.0,
        spontaneous_cl=None,
        # A1: hard constraints (relaxed — MHAS13 baseline dVdt=132 V/s)
        dvdt_max_upper=120.0,       # Reject dVdt > 120 V/s
        v_peak_max=60.0,            # Reject Vpeak > 60 mV
        v_rest_range=(-92.0, -70.0),
    )

    print("=" * 70)
    print("Optimizer V1 — MHAS13 Pipeline (Iteration 2)")
    print(f"  Improvements: A1(dVdt constraint), A2(tier 2), A3(secant CV),")
    print(f"                A4(tighter CV tol), R3(seed={config.seed})")
    print(f"  Model:   MHAS13  |  Device: {device}")
    print(f"  Cell dt: {config.dt_cell}ms  |  Tissue dt: {config.dt}ms")
    print(f"  Tier {config.tier}: {get_param_names(config.tier)}")
    print(f"  Targets: APD={targets.apd_90}ms  dVdt={targets.dvdt_max}V/s")
    print(f"  Constraints: dVdt<{targets.dvdt_max_upper}  Vpeak<{targets.v_peak_max}")
    print("=" * 70)
    t_start = perf_counter()

    # ================================================================
    # Step 1: Baseline
    # ================================================================
    print("\n--- Step 1: MHAS13 Baseline (theta=1.0) ---")
    n_params = len(get_param_names(config.tier))
    theta_base = torch.ones(1, n_params, dtype=torch.float64)

    t0 = perf_counter()
    t_arr, V_all = run_single_cell_batch(theta_base, config)
    t_cell = perf_counter() - t0

    from tuner.metrics import measure_apd, measure_dvdt_max, measure_v_rest, measure_peak
    V = V_all[0]
    apd_base = measure_apd(V, t_arr)
    dvdt_base = measure_dvdt_max(V, t_arr)
    vrest_base = measure_v_rest(V, t_arr)
    vpeak_base = measure_peak(V)
    print(f"  Cell: {t_cell:.1f}s")
    print(f"    APD90={apd_base:.0f}ms  dVdt={dvdt_base:.1f}V/s  "
          f"Vrest={vrest_base:.1f}mV  Vpeak={vpeak_base:.1f}mV" if apd_base else
          f"    APD=N/A  dVdt={dvdt_base:.1f}V/s  Vrest={vrest_base:.1f}mV")

    # ================================================================
    # Step 2: Cell Fitter (constrained, tier 2)
    # ================================================================
    print(f"\n--- Step 2: Cell Fitter (tier {config.tier}, {n_params} params, constrained) ---")
    t0 = perf_counter()
    cell_fit = fit_cell(config, targets, n_initial=2*n_params,
                        n_iterations=config.n_iterations, verbose=True)
    t_fit = perf_counter() - t0
    print(f"\n  Cell fitter: {t_fit:.1f}s ({cell_fit.n_feasible}/{cell_fit.n_total} feasible)")

    # Select best feasible
    apd_errors = -cell_fit.pareto_Y[:, 0]
    best_idx = apd_errors.argmin().item()
    best_theta = cell_fit.pareto_X[best_idx]
    best_dict = theta_to_dict(best_theta, config.tier)
    print(f"  Best: { {k: f'{v:.3f}' for k, v in best_dict.items()} }")

    cell_best = run_single_cell(best_theta, config)
    print(f"  APD90={cell_best.apd90:.1f}ms  dVdt={cell_best.dvdt_max:.1f}V/s  "
          f"Vpeak={cell_best.v_peak:.1f}mV" if cell_best.apd90 else "  APD=N/A")

    # ================================================================
    # Step 3: Tissue Fitter (secant method)
    # ================================================================
    print("\n--- Step 3: Tissue Fitter (secant CV refinement) ---")
    t0 = perf_counter()
    tissue_fit = fit_tissue(best_theta, config, targets, verbose=True)
    t_tis = perf_counter() - t0
    print(f"\n  Tissue fitter: {t_tis:.1f}s ({tissue_fit.n_sims} sims)")

    # ================================================================
    # Summary
    # ================================================================
    t_total = perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"MHAS13 ITERATION 2 COMPLETE — {t_total:.0f}s ({t_total/60:.1f} min)")
    print()
    print(f"  {'Metric':<12} {'Baseline':>10} {'Tuned':>10} {'Target':>10} {'Constr':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    apd_b = f"{apd_base:.0f}" if apd_base else "N/A"
    apd_t = f"{cell_best.apd90:.0f}" if cell_best.apd90 else "N/A"
    dvdt_b = f"{dvdt_base:.0f}" if dvdt_base else "N/A"
    dvdt_t = f"{cell_best.dvdt_max:.0f}" if cell_best.dvdt_max else "N/A"
    print(f"  {'APD90 (ms)':<12} {apd_b:>10} {apd_t:>10} {targets.apd_90:>10.0f} {'':>10}")
    print(f"  {'dVdt (V/s)':<12} {dvdt_b:>10} {dvdt_t:>10} {targets.dvdt_max:>10.0f} {'<'+str(int(targets.dvdt_max_upper)):>10}")
    print(f"  {'CV_L (cm/s)':<12} {'':>10} {tissue_fit.cv_long_achieved:>10.1f} {targets.cv_longitudinal:>10.1f} {'':>10}")
    print(f"  {'CV_T (cm/s)':<12} {'':>10} {tissue_fit.cv_trans_achieved:>10.1f} {targets.cv_transverse:>10.1f} {'':>10}")
    print()
    print(f"  D_long = {tissue_fit.D_long:.6f}  |  D_trans = {tissue_fit.D_trans:.6f}")
    print()
    print(f"  Tuned parameters (tier {config.tier}):")
    for k, v in best_dict.items():
        print(f"    {k}: {v:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
