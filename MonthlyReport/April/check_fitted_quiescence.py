#!/usr/bin/env python3
"""
Test whether fitted MHAS13 (from D1_mhas13_theta.json) remains quiescent
without external stimulus. The optimizer's fit objective and constraints
did not include quiescence — this verifies that the maturation property
survived the parameter sweep.

Quiescence criterion: in 10s of free-running simulation, V never rises
above V_threshold = -40 mV (arbitrary AP-detection floor).
"""
import sys
import os
import json
import torch
import numpy as np
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Optimizer', 'V1'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Monodomain', 'Engine_V5.4'))

from tuner.config import TuningConfig
from tuner.cell_runner import run_single_cell_batch


def main():
    theta_path = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures',
                              'D1_mhas13_theta.json')
    with open(theta_path) as f:
        meta = json.load(f)

    print("Fitted MHAS13 theta:")
    for k, v in meta['theta_dict'].items():
        delta = (v - 1.0) * 100  # baseline is 1.0 = 100%
        flag = ""
        if k == 'g_K1' and v < 0.5:
            flag = "  ⚠ REDUCED — could compromise quiescence"
        print(f"   {k:<10} = {v:.4f}  ({delta:+.0f}%)  {flag}")

    config = TuningConfig(
        ionic_model='mhas13', tier=2, device='cpu',
        dt=0.02, dt_cell=0.05,
        dx_cm=0.04, cable_length_cm=1.5,
        n_beats=10, pacing_cl=1000.0,
        stim_amplitude=-40.0, stim_duration=2.0,
    )
    theta = torch.tensor([meta['theta_array']], dtype=torch.float64)

    print("\nRunning free-run (no stimulus) for 10 s...")
    t0 = time.perf_counter()
    t_arr, V_all = run_single_cell_batch(
        theta, config, n_beats=10, save_last_n_beats=10, spontaneous=True,
    )
    print(f"   done ({time.perf_counter() - t0:.1f}s)")

    V = V_all[0]
    V_max = float(np.max(V))
    V_min = float(np.min(V))
    V_mean_late = float(np.mean(V[len(V)//2:]))

    print(f"\nResults over t=[{t_arr[0]:.0f}, {t_arr[-1]:.0f}] ms:")
    print(f"   V_max  = {V_max:+.1f} mV")
    print(f"   V_min  = {V_min:+.1f} mV")
    print(f"   V_mean (last 5s) = {V_mean_late:+.1f} mV")

    # Quiescence check
    threshold = -40.0
    fired = V_max > threshold
    print()
    if fired:
        # Count APs (peaks crossing threshold)
        above = V > threshold
        edges = np.diff(above.astype(int))
        n_aps = int(np.sum(edges == 1))
        print(f"   ❌ NOT QUIESCENT — V rose above {threshold} mV {n_aps} time(s)")
        print(f"   The fit broke MHAS13's maturation property.")
    else:
        print(f"   ✓ QUIESCENT — V stayed below {threshold} mV (max {V_max:.1f} mV)")
        print(f"   Maturation property survived the fit.")


if __name__ == '__main__':
    main()
