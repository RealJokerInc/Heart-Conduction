#!/usr/bin/env python3
"""
D1 — F1: Single-AP comparison of PHAS13 vs FITTED MHAS13 vs TTP06 vs ORd.

MHAS13 uses tier-2 fitted theta from D1_mhas13_theta.json (produced by
fit_mhas13_for_d1.py). PHAS13 / TTP06 / ORd use baseline parameters.

Output: MonthlyReport/April/figures/D1_F1_ap_comparison_4models.png

Protocol:
  - PHAS13: free-run (no stimulus); capture one spontaneous AP after 5s warmup.
  - MHAS13 (fitted): paced via run_single_cell_batch with optimizer theta.
  - TTP06 EPI / ORd EPI: paced via baseline model.step(), -52 µA/µF, 2 ms.
  - All traces aligned at upstroke (t = 0 = max dV/dt).
  - Window: (-100, 1000) ms.
"""

import sys
import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Optimizer', 'V1'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Monodomain', 'Engine_V5.4'))

from cardiac_sim.ionic import (
    PHAS13Model, TTP06Model, ORdModel, CellType,
)
from tuner.config import TuningConfig
from tuner.cell_runner import run_single_cell_batch


def _init_2d(model):
    state_1d = model.get_initial_state(n_cells=1)
    state = state_1d.unsqueeze(0)  # (1, n_states)
    V = torch.tensor([model.V_rest], dtype=torch.float64)  # (1,)
    return V, state


def run_paced(model, dt=0.05, bcl=1000.0, n_beats=8,
              stim_amp=-52.0, stim_dur=2.0):
    """Paced via standalone model.step() loop."""
    V, state = _init_2d(model)
    t_total = n_beats * bcl
    n_steps = int(t_total / dt)
    t_arr = np.linspace(0.0, t_total, n_steps + 1)
    V_arr = np.zeros(n_steps + 1)
    V_arr[0] = V[0].item()

    for i in range(n_steps):
        t = i * dt
        beat_phase = t % bcl
        stim_val = stim_amp if (0.0 <= beat_phase < stim_dur) else 0.0
        I_stim = torch.tensor([stim_val], dtype=torch.float64)
        V, state = model.step(V, state, dt, I_stim)
        V_arr[i + 1] = V[0].item()

    return t_arr, V_arr


def run_freerun(model, dt=0.05, t_total=12000.0):
    """Free-run (PHAS13 spontaneous)."""
    V, state = _init_2d(model)
    n_steps = int(t_total / dt)
    t_arr = np.linspace(0.0, t_total, n_steps + 1)
    V_arr = np.zeros(n_steps + 1)
    V_arr[0] = V[0].item()
    I_stim = torch.tensor([0.0], dtype=torch.float64)

    for i in range(n_steps):
        V, state = model.step(V, state, dt, I_stim)
        V_arr[i + 1] = V[0].item()

    return t_arr, V_arr


def run_mhas13_fitted(theta_dict_or_array, n_beats=8, save_last=3):
    """Run MHAS13 with optimizer-fitted theta via run_single_cell_batch."""
    config = TuningConfig(
        ionic_model='mhas13',
        tier=2,
        device='cpu',
        dt=0.02,
        dt_cell=0.05,        # match other models for visual alignment
        dx_cm=0.04,
        cable_length_cm=1.5,
        n_beats=n_beats,
        pacing_cl=1000.0,
        stim_amplitude=-40.0,
        stim_duration=2.0,
    )
    if isinstance(theta_dict_or_array, dict):
        theta = torch.tensor(
            [theta_dict_or_array['theta_array']], dtype=torch.float64,
        )
    else:
        theta = torch.tensor([list(theta_dict_or_array)], dtype=torch.float64)

    t_arr, V_all = run_single_cell_batch(
        theta, config, n_beats=n_beats, save_last_n_beats=save_last,
    )
    return t_arr, V_all[0]


def find_upstroke_time(t, V, t_min=None):
    dVdt = np.gradient(V, t)
    if t_min is not None:
        mask = t >= t_min
        if not mask.any():
            raise RuntimeError(f"No samples with t >= {t_min}")
        dVdt_masked = dVdt.copy()
        dVdt_masked[~mask] = -np.inf
        idx = int(np.argmax(dVdt_masked))
    else:
        idx = int(np.argmax(dVdt))
    return float(t[idx])


def extract_window(t, V, t_center, window=(-100.0, 1000.0)):
    t_rel = t - t_center
    mask = (t_rel >= window[0]) & (t_rel <= window[1])
    return t_rel[mask], V[mask]


def main():
    out_dir = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'D1_F1_ap_comparison_4models.png')
    theta_path = os.path.join(out_dir, 'D1_mhas13_theta.json')

    if not os.path.exists(theta_path):
        raise FileNotFoundError(
            f"Fitted theta not found: {theta_path}. "
            f"Run MonthlyReport/April/fit_mhas13_for_d1.py first."
        )
    with open(theta_path) as f:
        theta_meta = json.load(f)
    print(f"Loaded fitted MHAS13 theta: APD={theta_meta['fitted_apd_ms']:.1f} ms, "
          f"dVdt={theta_meta['fitted_dvdt_Vps']:.1f} V/s")

    device = 'cpu'
    dt = 0.05
    window = (-100.0, 1000.0)

    print("=" * 70)
    print("D1 F1 — 4-model AP comparison (FITTED MHAS13)")
    print("=" * 70)

    # PHAS13 — spontaneous
    print("\n[1/4] PHAS13 (free-run, 12 s)...")
    t0 = time.perf_counter()
    phas13 = PHAS13Model(device=device)
    t1, V1 = run_freerun(phas13, dt=dt, t_total=12000.0)
    up1 = find_upstroke_time(t1, V1, t_min=5000.0)
    print(f"      done ({time.perf_counter() - t0:.1f}s); upstroke t={up1:.1f} ms")

    # MHAS13 — fitted, paced (uses dt_cell=0.05 internally)
    print("\n[2/4] MHAS13 fitted (8 beats @ BCL=1000, theta from optimizer)...")
    t0 = time.perf_counter()
    t2, V2 = run_mhas13_fitted(theta_meta, n_beats=8, save_last=3)
    up2 = find_upstroke_time(t2, V2, t_min=t2[0] + 1500.0)  # last-saved-beat upstroke
    print(f"      done ({time.perf_counter() - t0:.1f}s); upstroke t={up2:.1f} ms")

    # TTP06 — paced, EPI
    print("\n[3/4] TTP06 EPI (8 beats @ BCL=1000)...")
    t0 = time.perf_counter()
    ttp06 = TTP06Model(cell_type=CellType.EPI, device=device)
    t3, V3 = run_paced(ttp06, dt=dt, n_beats=8)
    up3 = find_upstroke_time(t3, V3, t_min=6500.0)
    print(f"      done ({time.perf_counter() - t0:.1f}s); upstroke t={up3:.1f} ms")

    # ORd — paced, EPI
    print("\n[4/4] ORd EPI (8 beats @ BCL=1000)...")
    t0 = time.perf_counter()
    ord_m = ORdModel(cell_type=CellType.EPI, device=device)
    t4, V4 = run_paced(ord_m, dt=dt, n_beats=8)
    up4 = find_upstroke_time(t4, V4, t_min=6500.0)
    print(f"      done ({time.perf_counter() - t0:.1f}s); upstroke t={up4:.1f} ms")

    # Align around upstroke
    t1a, V1a = extract_window(t1, V1, up1, window=window)
    t2a, V2a = extract_window(t2, V2, up2, window=window)
    t3a, V3a = extract_window(t3, V3, up3, window=window)
    t4a, V4a = extract_window(t4, V4, up4, window=window)

    # Save raw aligned traces for reproducibility
    np.savez(
        os.path.join(out_dir, 'D1_F1_traces.npz'),
        phas13_t=t1a, phas13_V=V1a,
        mhas13_t=t2a, mhas13_V=V2a,
        ttp06_t=t3a, ttp06_V=V3a,
        ord_t=t4a, ord_V=V4a,
    )

    # Compute V_rest for legend (from beat-start window before upstroke)
    vr_phas13 = float(np.median(V1a[(t1a > -90) & (t1a < -10)]))
    vr_mhas13 = float(np.median(V2a[(t2a > -90) & (t2a < -10)]))
    vr_ttp06 = float(np.median(V3a[(t3a > -90) & (t3a < -10)]))
    vr_ord = float(np.median(V4a[(t4a > -90) & (t4a < -10)]))

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(10.0, 5.0), dpi=300)

    # PHAS13 — green solid
    ax.plot(t1a, V1a, '-', color='#2ca02c', linewidth=1.8,
            label=f'PHAS13 (spontaneous, V_rest = {vr_phas13:.1f} mV)')
    # MHAS13 fitted — red solid bold (the headline)
    ax.plot(t2a, V2a, '-', color='#d62728', linewidth=2.6,
            label=f'MHAS13 fitted (V_rest = {vr_mhas13:.1f} mV, '
                  f'APD₉₀ = {theta_meta["fitted_apd_ms"]:.0f} ms)')
    # TTP06 — black dotted (standard)
    ax.plot(t3a, V3a, ':', color='black', linewidth=1.5,
            label=f'TTP06 EPI (adult, V_rest = {vr_ttp06:.1f} mV)')
    # ORd — black finer-graded dotted (denser dot pattern)
    ax.plot(t4a, V4a, color='black', linewidth=1.5,
            linestyle=(0, (1, 1)),
            label=f'ORd EPI (adult, V_rest = {vr_ord:.1f} mV)')

    ax.axhline(0.0, color='k', linewidth=0.4, alpha=0.3)
    ax.axvline(0.0, color='k', linewidth=0.4, alpha=0.3, linestyle=':')

    ax.set_xlabel('Time relative to upstroke (ms)', fontsize=11)
    ax.set_ylabel('Membrane potential V (mV)', fontsize=11)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(-95, 50)
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        'Single-AP comparison: hiPSC-CM (PHAS13) vs matured-fitted (MHAS13) '
        'vs adult ventricular (TTP06, ORd)',
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved figure: {out_png}")
    print(f"Saved traces: {os.path.join(out_dir, 'D1_F1_traces.npz')}")


if __name__ == '__main__':
    main()
