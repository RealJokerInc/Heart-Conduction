#!/usr/bin/env python
"""
Conductivity Sweep — Edge lead scaling with D_eff.

Hypothesis: steady-state edge lead ~ 1/sqrt(D_eff) for isotropic scaling.

5 configs, all bidomain Mehrstellen + bath_tb BCs:
  0.5x iso:  sigma scaled 0.5x → slower CV, same Kleber ratio
  1x iso:    baseline
  2x iso:    sigma scaled 2x → faster CV, same Kleber ratio
  4x iso:    sigma scaled 4x → fastest CV, same Kleber ratio
  4x σ_i:    sigma_i only 4x → faster CV, HIGHER Kleber ratio

Output: Research/Q5_boundary_conduction_speedup/conductivity_sweep/
"""

import sys
import os
import time
import json
import math

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.join(_HERE, '..')
_TESTS = os.path.join(_ENGINE, 'tests')
sys.path.insert(0, _ENGINE)
sys.path.insert(0, _TESTS)

torch.set_default_dtype(torch.float64)

from cv_shared import build_bidomain_sim

# ============================================================
# Constants
# ============================================================
CHI = 1400.0
CM = 1.0
SIGMA_I_BASE = 1.74
SIGMA_E_BASE = 6.25
THRESHOLD = -30.0

NX, NY = 1001, 161
DX = 0.05
DT = 0.01
T_END = 400.0
SAVE_EVERY = 25.0

STIM_WIDTH = 5 * DX
STIM_START = 1.0
STIM_DUR = 2.0
STIM_AMP = -80.0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = os.path.join(
    _HERE, '..', '..', '..', 'Research',
    'Q5_boundary_conduction_speedup', 'conductivity_sweep')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIGS = [
    {'label': '0.5x iso',           'sigma_i': SIGMA_I_BASE * 0.5, 'sigma_e': SIGMA_E_BASE * 0.5},
    {'label': '1x iso (baseline)',   'sigma_i': SIGMA_I_BASE,       'sigma_e': SIGMA_E_BASE},
    {'label': '2x iso',             'sigma_i': SIGMA_I_BASE * 2,   'sigma_e': SIGMA_E_BASE * 2},
    {'label': '4x iso',             'sigma_i': SIGMA_I_BASE * 4,   'sigma_e': SIGMA_E_BASE * 4},
    {'label': '4x sigma_i only',    'sigma_i': SIGMA_I_BASE * 4,   'sigma_e': SIGMA_E_BASE},
]


# ============================================================
# Wavefront extraction
# ============================================================
def extract_wavefront(V, threshold=THRESHOLD):
    front = torch.zeros(V.shape[1], dtype=torch.long, device=V.device)
    for j in range(V.shape[1]):
        activated = (V[:, j] > threshold).nonzero(as_tuple=False)
        if activated.numel() > 0:
            front[j] = activated[-1, 0]
    return front


# ============================================================
# Run one config
# ============================================================
def run_config(cfg):
    D_i = cfg['sigma_i'] / (CHI * CM)
    D_e = cfg['sigma_e'] / (CHI * CM)
    D_eff = D_i * D_e / (D_i + D_e)
    kleber = math.sqrt((D_i + D_e) / D_e)

    print(f"\n{'=' * 60}")
    print(f"  {cfg['label']}")
    print(f"  D_i={D_i:.6f}  D_e={D_e:.6f}  D_eff={D_eff:.6f}  Kleber={kleber:.4f}")
    print(f"{'=' * 60}")

    sim, grid = build_bidomain_sim(
        nx=NX, ny=NY, dx=DX, dt=DT, D_i=D_i, D_e=D_e,
        bc_type='bath_tb', stencil='mehrstellen',
        stim_width=STIM_WIDTH, stim_start=STIM_START,
        stim_dur=STIM_DUR, stim_amp=STIM_AMP,
        device=DEVICE,
    )

    fronts = []
    times = []
    t0 = time.time()
    for state in sim.run(t_end=T_END, save_every=SAVE_EVERY):
        V = grid.flat_to_grid(state.Vm.cpu())
        front = extract_wavefront(V)
        fronts.append(front)
        times.append(state.t)

        center = front[NY // 2].item()
        edge = max(front[1].item(), front[-2].item())
        lead = (edge - center) * DX
        elapsed = time.time() - t0
        print(f"    t={state.t:6.0f}ms  lead={lead:.3f}cm  elapsed={elapsed:.0f}s")

    total = time.time() - t0

    # Steady-state edge lead (last 3 snapshots)
    leads = []
    for front in fronts[-3:]:
        center = front[NY // 2].item()
        edge = max(front[1].item(), front[-2].item())
        leads.append((edge - center) * DX)
    ss_lead = float(np.mean(leads))

    print(f"  DONE in {total:.0f}s — SS edge lead = {ss_lead:.3f} cm")

    return dict(
        label=cfg['label'],
        sigma_i=cfg['sigma_i'], sigma_e=cfg['sigma_e'],
        D_i=D_i, D_e=D_e, D_eff=D_eff,
        kleber_ratio=kleber,
        ss_edge_lead_cm=ss_lead,
        wall_clock_s=total,
        times=times,
        leads_vs_time=[
            float((max(f[1].item(), f[-2].item()) - f[NY // 2].item()) * DX)
            for f in fronts
        ],
    )


# ============================================================
# Plot
# ============================================================
def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    iso = [r for r in results if 'iso' in r['label']]
    non_iso = [r for r in results if 'iso' not in r['label']]

    # --- Panel 1: Edge lead vs D_eff ---
    ax = axes[0]
    for r in results:
        c = 'blue' if 'iso' in r['label'] else 'red'
        m = 'o' if 'iso' in r['label'] else 's'
        ax.scatter(r['D_eff'] * 1000, r['ss_edge_lead_cm'], c=c, marker=m,
                   s=100, zorder=5, label=r['label'])

    # 1/sqrt(D) prediction (fitted to baseline)
    if len(iso) >= 2:
        base = iso[1]  # 1x baseline
        D_range = np.linspace(
            min(r['D_eff'] for r in results) * 0.7,
            max(r['D_eff'] for r in results) * 1.3, 100)
        pred = base['ss_edge_lead_cm'] * np.sqrt(base['D_eff'] / D_range)
        ax.plot(D_range * 1000, pred, 'g--', linewidth=2,
                label='Prediction: lead ~ 1/\u221aD')

    ax.set_xlabel('D_eff (\u00d710\u207b\u00b3 cm\u00b2/ms)')
    ax.set_ylabel('Steady-state edge lead (cm)')
    ax.set_title('Edge Lead vs D_eff')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Edge lead vs Kleber ratio ---
    ax = axes[1]
    for r in results:
        c = 'blue' if 'iso' in r['label'] else 'red'
        m = 'o' if 'iso' in r['label'] else 's'
        ax.scatter(r['kleber_ratio'], r['ss_edge_lead_cm'], c=c, marker=m,
                   s=100, zorder=5, label=r['label'])

    ax.set_xlabel('Kleber ratio \u221a((\u03c3\u1d62+\u03c3\u2091)/\u03c3\u2091)')
    ax.set_ylabel('Steady-state edge lead (cm)')
    ax.set_title('Edge Lead vs Kleber Ratio')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Lead vs time for all configs ---
    ax = axes[2]
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        color = cmap(i / max(len(results) - 1, 1))
        ls = '-' if 'iso' in r['label'] else '--'
        ax.plot(r['times'], r['leads_vs_time'], color=color, linestyle=ls,
                linewidth=2, label=r['label'])

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Edge lead (cm)')
    ax.set_title('Edge Lead Evolution')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'conductivity_sweep.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    print("Conductivity Sweep — Edge Lead vs D_eff")
    print(f"Grid: {NX}x{NY}, dx={DX}, T_END={T_END}ms, Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print('=' * 60)

    results = []
    for cfg in CONFIGS:
        try:
            results.append(run_config(cfg))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Save results (without large arrays)
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Label':<25} {'D_eff':>10} {'Kleber':>8} {'SS Lead':>10} {'Time':>8}")
    print('-' * 65)
    for r in results:
        print(f"{r['label']:<25} {r['D_eff']:>10.6f} {r['kleber_ratio']:>8.4f} "
              f"{r['ss_edge_lead_cm']:>9.3f}cm {r['wall_clock_s']:>7.0f}s")

    # Theory check
    iso = [r for r in results if 'iso' in r['label']]
    if len(iso) >= 2:
        print(f"\nTheory: lead * sqrt(D_eff) should be constant for iso configs")
        for r in iso:
            product = r['ss_edge_lead_cm'] * math.sqrt(r['D_eff'])
            print(f"  {r['label']:<25} {r['ss_edge_lead_cm']:.3f} * "
                  f"{math.sqrt(r['D_eff']):.6f} = {product:.6f}")

    plot_results(results)


if __name__ == '__main__':
    main()
