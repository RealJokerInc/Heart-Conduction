#!/usr/bin/env python3
"""
Optimizer V1 — MHAS13 Bidomain Pipeline

Same targets as monodomain run:
  APD90 = 350 ms, dV/dt = 25 V/s, CV_long = 15 cm/s, CV_trans = 7.5 cm/s
  dt = 0.02 ms, dx = 0.04 cm (from spiral_wave_s1s2)

Uses the Bidomain V1 engine for tissue CV measurement with the
decoupled Gauss-Seidel solver (parabolic Vm → elliptic phi_e).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'Monodomain', 'Engine_V5.4'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'Bidomain', 'Engine_V1'))

import torch
from time import perf_counter

from tuner.config import TuningConfig, TuningTargets, get_param_names, theta_to_dict
from tuner.cell_runner import run_single_cell, run_single_cell_batch
from tuner.cell_fitter import fit_cell
from tuner.tissue_fitter import fit_tissue


def main():
    device = 'cpu'

    config = TuningConfig(
        ionic_model='mhas13',
        tier=1,
        device=device,
        dt=0.02,                    # tissue dt (spiral_wave_s1s2)
        dt_cell=0.2,               # cell dt
        dx_cm=0.04,                 # spatial (spiral_wave_s1s2)
        cable_length_cm=1.5,
        n_beats=5,
        pacing_cl=1000.0,
        stim_amplitude=-40.0,
        stim_duration=2.0,
        n_iterations=10,
        # Bidomain-specific
        engine='bidomain',
        De_Di_ratio=3.597,          # physiological D_e/D_i
        bc_type='insulated',
        bidomain_splitting='strang',
        elliptic_solver='auto',
    )

    targets = TuningTargets(
        apd_90=350.0,
        cv_longitudinal=15.0,
        cv_transverse=7.5,
        dvdt_max=25.0,
        spontaneous_cl=None,
    )

    print("=" * 70)
    print("Optimizer V1 — MHAS13 BIDOMAIN Pipeline")
    print(f"  Model:   MHAS13 (matured hiPSC-CM)")
    print(f"  Engine:  BIDOMAIN (decoupled GS, {config.elliptic_solver} elliptic)")
    print(f"  Device:  {device}")
    print(f"  Cell dt: {config.dt_cell}ms  |  Tissue dt: {config.dt}ms")
    print(f"  dx: {config.dx_cm}cm  |  Cable: {config.cable_length_cm}cm")
    print(f"  D_e/D_i ratio: {config.De_Di_ratio:.3f}")
    print(f"  BC: {config.bc_type}")
    print(f"  Stim: {config.stim_amplitude} A/F, {config.stim_duration}ms")
    print(f"  Tier {config.tier}: {get_param_names(config.tier)}")
    print(f"  Targets: APD={targets.apd_90}ms  dVdt={targets.dvdt_max}V/s")
    print(f"           CV_L={targets.cv_longitudinal}  CV_T={targets.cv_transverse} cm/s")
    print("=" * 70)
    t_start = perf_counter()

    # ================================================================
    # Step 1: Baseline (cell — same as monodomain, engine-agnostic)
    # ================================================================
    print("\n--- Step 1: MHAS13 Baseline ---")
    n_params = len(get_param_names(config.tier))
    theta_base = torch.ones(1, n_params, dtype=torch.float64)

    t0 = perf_counter()
    t_arr, V_all = run_single_cell_batch(theta_base, config)
    t_cell = perf_counter() - t0

    from tuner.metrics import measure_apd, measure_dvdt_max, measure_v_rest, measure_peak
    V = V_all[0]
    apd_base = measure_apd(V, t_arr)
    dvdt_base = measure_dvdt_max(V, t_arr)

    apd_s = f"{apd_base:.0f}" if apd_base else "N/A"
    dvdt_s = f"{dvdt_base:.1f}" if dvdt_base else "N/A"
    print(f"  Cell: {t_cell:.1f}s")
    print(f"    APD90  = {apd_s} ms  |  dVdt = {dvdt_s} V/s")
    print(f"    V_rest = {measure_v_rest(V, t_arr):.1f} mV  |  V_peak = {measure_peak(V):.1f} mV")

    # Bidomain tissue baseline
    from tuner.tissue_runner_bidomain import run_cv_measurement_bidomain
    r = config.De_Di_ratio
    D_eff_ref = 0.0001
    D_i_ref = D_eff_ref * (1.0 + r) / r
    D_e_ref = D_i_ref * r
    print(f"\n  Bidomain tissue baseline:")
    print(f"    D_eff={D_eff_ref}, D_i={D_i_ref:.6f}, D_e={D_e_ref:.6f}")

    t0 = perf_counter()
    cv_base = run_cv_measurement_bidomain(
        theta_base[0], D_i_ref, D_e_ref, config, n_beats=3)
    t_tis = perf_counter() - t0
    cv_s = f"{cv_base.cv:.1f}" if cv_base.cv else "N/A"
    print(f"    Tissue: {t_tis:.1f}s  |  CV = {cv_s} cm/s")

    # ================================================================
    # Step 2: Cell Fitter (engine-agnostic, same as monodomain)
    # ================================================================
    print("\n--- Step 2: Cell Fitter (8 initial + 10 BO, batched) ---")
    t0 = perf_counter()
    cell_fit = fit_cell(config, targets, n_initial=8, n_iterations=10,
                        verbose=True)
    t_fit = perf_counter() - t0
    print(f"\n  Cell fitter: {t_fit:.1f}s total")

    apd_errors = -cell_fit.pareto_Y[:, 0]
    best_idx = apd_errors.argmin().item()
    best_theta = cell_fit.pareto_X[best_idx]
    best_dict = theta_to_dict(best_theta, config.tier)
    print(f"  Best: { {k: f'{v:.3f}' for k, v in best_dict.items()} }")

    cell_best = run_single_cell(best_theta, config)
    apd_best = f"{cell_best.apd90:.1f}" if cell_best.apd90 else "N/A"
    dvdt_best = f"{cell_best.dvdt_max:.1f}" if cell_best.dvdt_max else "N/A"
    print(f"  APD90 = {apd_best} ms  |  dVdt = {dvdt_best} V/s")

    # ================================================================
    # Step 3: Tissue Fitter (BIDOMAIN engine)
    # ================================================================
    print("\n--- Step 3: Tissue Fitter (BIDOMAIN, analytical CV~sqrt(D_eff)) ---")
    t0 = perf_counter()
    tissue_fit = fit_tissue(best_theta, config, targets, verbose=True)
    t_tis_fit = perf_counter() - t0
    print(f"\n  Tissue fitter: {t_tis_fit:.1f}s ({tissue_fit.n_sims} sims)")

    # ================================================================
    # Summary
    # ================================================================
    t_total = perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"MHAS13 BIDOMAIN PIPELINE COMPLETE — {t_total:.0f}s ({t_total/60:.1f} min)")
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
    print(f"  Tissue (D_eff):")
    print(f"    D_eff_long  = {tissue_fit.D_long:.6f} cm²/ms")
    print(f"    D_eff_trans = {tissue_fit.D_trans:.6f} cm²/ms")
    if tissue_fit.D_i_long is not None:
        print(f"  Bidomain decomposition (D_e/D_i = {config.De_Di_ratio:.3f}):")
        print(f"    Long:  D_i = {tissue_fit.D_i_long:.6f}, D_e = {tissue_fit.D_e_long:.6f}")
        print(f"    Trans: D_i = {tissue_fit.D_i_trans:.6f}, D_e = {tissue_fit.D_e_trans:.6f}")
    print()
    print(f"  Tuned ionic scaling factors:")
    for k, v in best_dict.items():
        print(f"    {k}: {v:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
