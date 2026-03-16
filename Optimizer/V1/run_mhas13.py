#!/usr/bin/env python3
"""
Optimizer V1 — MHAS13 Full Pipeline Run

Tunes MHAS13 (matured hiPSC-CM) ionic parameters to target:
  APD90 = 350 ms, dV/dt = 25 V/s, CV_long = 15 cm/s, CV_trans = 7.5 cm/s

Uses spiral_wave_s1s2 tissue parameters (dt=0.02ms, dx=0.04cm).
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
        ionic_model='mhas13',       # Matured hiPSC-CM
        tier=1,                     # 6 ionic params
        device=device,
        dt=0.02,                    # tissue dt (spiral_wave_s1s2)
        dt_cell=0.2,               # cell dt
        dx_cm=0.04,                 # spatial (spiral_wave_s1s2)
        cable_length_cm=1.5,
        n_beats=5,
        pacing_cl=1000.0,           # 1 Hz pacing
        stim_amplitude=-40.0,       # MHAS13 threshold ~-15 A/F
        stim_duration=2.0,
        n_iterations=10,            # BO iterations
    )

    targets = TuningTargets(
        apd_90=350.0,              # ms
        cv_longitudinal=15.0,      # cm/s
        cv_transverse=7.5,         # cm/s
        dvdt_max=25.0,             # V/s
        spontaneous_cl=None,       # Not applicable (quiescent model)
    )

    print("=" * 70)
    print("Optimizer V1 — MHAS13 Full Pipeline")
    print(f"  Model:   MHAS13 (matured hiPSC-CM, TTP06 IK1, no If)")
    print(f"  Device:  {device}")
    print(f"  Cell dt: {config.dt_cell}ms  |  Tissue dt: {config.dt}ms")
    print(f"  dx: {config.dx_cm}cm  |  Cable: {config.cable_length_cm}cm")
    print(f"  Stim: {config.stim_amplitude} A/F, {config.stim_duration}ms")
    print(f"  Tier {config.tier}: {get_param_names(config.tier)}")
    print(f"  Targets: APD={targets.apd_90}ms  dVdt={targets.dvdt_max}V/s")
    print(f"           CV_L={targets.cv_longitudinal}  CV_T={targets.cv_transverse} cm/s")
    print("=" * 70)
    t_start = perf_counter()

    # ================================================================
    # Step 1: Baseline characterization
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

    apd_s = f"{apd_base:.0f}" if apd_base else "N/A"
    dvdt_s = f"{dvdt_base:.1f}" if dvdt_base else "N/A"
    print(f"  Cell: {t_cell:.1f}s")
    print(f"    APD90  = {apd_s} ms (target {targets.apd_90})")
    print(f"    dVdt   = {dvdt_s} V/s (target {targets.dvdt_max})")
    print(f"    V_rest = {vrest_base:.1f} mV")
    print(f"    V_peak = {vpeak_base:.1f} mV")

    # Tissue baseline
    D_ref = 0.0001
    t0 = perf_counter()
    cv_base = run_cv_measurement(theta_base[0], D_ref, config, n_beats=3)
    t_tis = perf_counter() - t0
    cv_s = f"{cv_base.cv:.1f}" if cv_base.cv else "N/A"
    print(f"  Tissue: {t_tis:.1f}s")
    print(f"    CV = {cv_s} cm/s at D={D_ref}")

    # ================================================================
    # Step 2: Cell Fitter
    # ================================================================
    print("\n--- Step 2: Cell Fitter (8 initial + 10 BO, batched) ---")
    t0 = perf_counter()
    cell_fit = fit_cell(config, targets, n_initial=8, n_iterations=10,
                        verbose=True)
    t_fit = perf_counter() - t0
    print(f"\n  Cell fitter: {t_fit:.1f}s total")
    print(f"  Pareto front: {cell_fit.pareto_X.shape[0]} points")

    # Select best from Pareto
    apd_errors = -cell_fit.pareto_Y[:, 0]
    best_idx = apd_errors.argmin().item()
    best_theta = cell_fit.pareto_X[best_idx]
    best_dict = theta_to_dict(best_theta, config.tier)
    print(f"  Best: { {k: f'{v:.3f}' for k, v in best_dict.items()} }")

    # Evaluate best
    cell_best = run_single_cell(best_theta, config)
    apd_best = f"{cell_best.apd90:.1f}" if cell_best.apd90 else "N/A"
    dvdt_best = f"{cell_best.dvdt_max:.1f}" if cell_best.dvdt_max else "N/A"
    print(f"  Best APD90 = {apd_best} ms")
    print(f"  Best dVdt  = {dvdt_best} V/s")

    # ================================================================
    # Step 3: Tissue Fitter
    # ================================================================
    print("\n--- Step 3: Tissue Fitter (analytical CV~sqrt(D)) ---")
    t0 = perf_counter()
    tissue_fit = fit_tissue(best_theta, config, targets, verbose=True)
    t_tis_fit = perf_counter() - t0
    print(f"\n  Tissue fitter: {t_tis_fit:.1f}s ({tissue_fit.n_sims} sims)")

    # ================================================================
    # Summary
    # ================================================================
    t_total = perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"MHAS13 PIPELINE COMPLETE — {t_total:.0f}s total ({t_total/60:.1f} min)")
    print()
    print(f"  {'Metric':<12} {'Baseline':>10} {'Tuned':>10} {'Target':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    apd_b = f"{apd_base:.0f}" if apd_base else "N/A"
    apd_t = f"{cell_best.apd90:.0f}" if cell_best.apd90 else "N/A"
    dvdt_b = f"{dvdt_base:.0f}" if dvdt_base else "N/A"
    dvdt_t = f"{cell_best.dvdt_max:.0f}" if cell_best.dvdt_max else "N/A"
    print(f"  {'APD90 (ms)':<12} {apd_b:>10} {apd_t:>10} {targets.apd_90:>10.0f}")
    print(f"  {'dVdt (V/s)':<12} {dvdt_b:>10} {dvdt_t:>10} {targets.dvdt_max:>10.0f}")
    print(f"  {'CV_L (cm/s)':<12} {cv_s:>10} {tissue_fit.cv_long_achieved:>10.1f} {targets.cv_longitudinal:>10.1f}")
    print(f"  {'CV_T (cm/s)':<12} {'—':>10} {tissue_fit.cv_trans_achieved:>10.1f} {targets.cv_transverse:>10.1f}")
    print()
    print(f"  Tissue: D_long = {tissue_fit.D_long:.6f} cm²/ms")
    print(f"          D_trans = {tissue_fit.D_trans:.6f} cm²/ms")
    print()
    print(f"  Tuned ionic scaling factors:")
    for k, v in best_dict.items():
        print(f"    {k}: {v:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
