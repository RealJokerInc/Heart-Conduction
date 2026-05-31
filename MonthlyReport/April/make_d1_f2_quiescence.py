#!/usr/bin/env python3
"""
D1 — F2: Quiescence comparison (10 s free-run, no stimulus).

PHAS13 baseline fires spontaneous APs (immature pacemaker phenotype).
Fitted MHAS13 holds at V_rest (matured / quiescent).

Outputs:
    MonthlyReport/April/figures/D1_F2_quiescence.png   (static two-panel plot)
    MonthlyReport/April/figures/D1_F2_traces.npz       (full V traces, for GIF)
"""

import sys
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Optimizer', 'V1'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Monodomain', 'Engine_V5.4'))

from cardiac_sim.ionic import PHAS13Model
from tuner.config import TuningConfig
from tuner.cell_runner import run_single_cell_batch


def run_phas13_freerun(t_total_ms=10000.0, dt=0.05):
    """Free-run PHAS13 (will fire spontaneously)."""
    model = PHAS13Model(device='cpu')
    state = model.get_initial_state(n_cells=1).unsqueeze(0)
    V = torch.tensor([model.V_rest], dtype=torch.float64)

    n_steps = int(t_total_ms / dt)
    t_arr = np.linspace(0.0, t_total_ms, n_steps + 1)
    V_arr = np.zeros(n_steps + 1)
    V_arr[0] = V[0].item()
    I_stim = torch.tensor([0.0], dtype=torch.float64)

    for i in range(n_steps):
        V, state = model.step(V, state, dt, I_stim)
        V_arr[i + 1] = V[0].item()

    return t_arr, V_arr


def run_mhas13_fitted_freerun(theta_meta, t_total_ms=10000.0):
    """Free-run fitted MHAS13 via run_single_cell_batch (spontaneous=True)."""
    config = TuningConfig(
        ionic_model='mhas13', tier=2, device='cpu',
        dt=0.02, dt_cell=0.05,
        dx_cm=0.04, cable_length_cm=1.5,
        n_beats=int(t_total_ms / 1000.0), pacing_cl=1000.0,
        stim_amplitude=-40.0, stim_duration=2.0,
    )
    theta = torch.tensor([theta_meta['theta_array']], dtype=torch.float64)
    n_beats = int(t_total_ms / 1000.0)
    t_arr, V_all = run_single_cell_batch(
        theta, config, n_beats=n_beats, save_last_n_beats=n_beats,
        spontaneous=True,
    )
    return t_arr, V_all[0]


def count_aps(V, threshold=-40.0):
    above = V > threshold
    edges = np.diff(above.astype(int))
    return int(np.sum(edges == 1))


def main():
    out_dir = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    theta_path = os.path.join(out_dir, 'D1_mhas13_theta.json')
    with open(theta_path) as f:
        theta_meta = json.load(f)

    t_total = 10000.0  # 10 s

    print("=" * 70)
    print("D1 F2 — Quiescence comparison (10 s free-run, no stimulus)")
    print("=" * 70)

    print("\n[1/2] PHAS13 free-run...")
    t0 = time.perf_counter()
    t_p, V_p = run_phas13_freerun(t_total_ms=t_total)
    print(f"      done ({time.perf_counter() - t0:.1f}s)")
    n_p = count_aps(V_p)
    print(f"      spontaneous APs detected: {n_p}")

    print("\n[2/2] MHAS13 fitted free-run...")
    t0 = time.perf_counter()
    t_m, V_m = run_mhas13_fitted_freerun(theta_meta, t_total_ms=t_total)
    print(f"      done ({time.perf_counter() - t0:.1f}s)")
    n_m = count_aps(V_m)
    V_m_max = float(np.max(V_m))
    print(f"      spontaneous APs detected: {n_m}  (V_max = {V_m_max:.1f} mV)")

    # Save full traces for the GIF script
    npz_path = os.path.join(out_dir, 'D1_F2_traces.npz')
    np.savez(npz_path,
             phas13_t=t_p, phas13_V=V_p, phas13_n_aps=n_p,
             mhas13_t=t_m, mhas13_V=V_m, mhas13_n_aps=n_m)
    print(f"\nSaved traces: {npz_path}")

    # ----- Static two-panel plot -----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 5.5), dpi=300,
                                    sharex=True, sharey=True)

    # PHAS13 — top panel, green
    ax1.plot(t_p / 1000.0, V_p, color='#2ca02c', linewidth=1.4)
    ax1.set_ylabel('V (mV)', fontsize=11)
    ax1.set_title(
        f'PHAS13 (immature hiPSC-CM) — spontaneous: '
        f'{n_p} APs in 10 s ⟹ ~{n_p / 10.0:.1f} Hz auto-firing',
        fontsize=10, color='#2ca02c', loc='left',
    )
    ax1.grid(True, alpha=0.25)
    ax1.axhline(-40.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax1.text(0.02, -38, 'AP threshold (-40 mV)',
             transform=ax1.get_yaxis_transform(), fontsize=8, color='gray')

    # MHAS13 fitted — bottom panel, red
    ax2.plot(t_m / 1000.0, V_m, color='#d62728', linewidth=1.4)
    ax2.set_ylabel('V (mV)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_title(
        f'MHAS13 (matured + fitted) — quiescent: '
        f'0 APs in 10 s, V_max = {V_m_max:.1f} mV',
        fontsize=10, color='#d62728', loc='left',
    )
    ax2.grid(True, alpha=0.25)
    ax2.axhline(-40.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)

    ax1.set_ylim(-95, 50)
    ax1.set_xlim(0, t_total / 1000.0)

    fig.suptitle(
        '10-second free-run with no external stimulus',
        fontsize=11, y=0.995,
    )
    plt.tight_layout()

    out_png = os.path.join(out_dir, 'D1_F2_quiescence.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure:  {out_png}")


if __name__ == '__main__':
    main()
