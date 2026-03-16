#!/usr/bin/env python
"""
Anisotropic Conductivity Test — 2:1 ratio with lower longitudinal CV.

Compare against isotropic baseline to test whether anisotropy + lower speed
produces a bowl or sharper triangle.

Eikonal prediction: lead ~ sqrt(D_long) / D_trans. With 2:1 anisotropy and
lower D_long, the halved D_trans dominates → LARGER lead (sharper triangle).
The user's hypothesis is that slower propagation → bowl shape.

Configs:
  1. Isotropic baseline: D_i=0.001243, D_e=0.004464 (CV ~47 cm/s)
  2. Anisotropic 2:1, lower speed:
     D_i_fiber=0.0008, D_i_cross=0.0004
     D_e_fiber=0.003,  D_e_cross=0.0015
     (CV_long ~38 cm/s, Kleber ~1.126)
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

# ============================================================
# Parameters
# ============================================================
CHI = 1400.0
CM = 1.0
THRESHOLD = -30.0

NX, NY = 1001, 161
DX = 0.05
DT = 0.01
T_END = 500.0
SAVE_EVERY = 25.0

STIM_WIDTH = 5 * DX
STIM_START = 1.0
STIM_DUR = 2.0
STIM_AMP = -80.0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = os.path.join(
    _HERE, '..', '..', '..', 'Research',
    'Q5_boundary_conduction_speedup', 'anisotropic_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Baseline isotropic
SIGMA_I = 1.74
SIGMA_E = 6.25
D_I = SIGMA_I / (CHI * CM)
D_E = SIGMA_E / (CHI * CM)

# Anisotropic 2:1, lower longitudinal speed
# Fiber (longitudinal, x): lower than baseline
# Cross (transverse, y): half of fiber
D_I_FIBER = 0.0008     # < baseline 0.001243
D_I_CROSS = 0.0004     # 2:1 ratio
D_E_FIBER = 0.003      # < baseline 0.004464
D_E_CROSS = 0.0015     # 2:1 ratio


def extract_wavefront(V, threshold=THRESHOLD):
    front = torch.zeros(V.shape[1], dtype=torch.long, device=V.device)
    for j in range(V.shape[1]):
        activated = (V[:, j] > threshold).nonzero(as_tuple=False)
        if activated.numel() > 0:
            front[j] = activated[-1, 0]
    return front


# ============================================================
# Build isotropic simulation (reuses build_bidomain_sim)
# ============================================================
def build_isotropic():
    from cv_shared import build_bidomain_sim
    return build_bidomain_sim(
        nx=NX, ny=NY, dx=DX, dt=DT, D_i=D_I, D_e=D_E,
        bc_type='bath_tb', stencil='5pt',
        stim_width=STIM_WIDTH, stim_start=STIM_START,
        stim_dur=STIM_DUR, stim_amp=STIM_AMP,
        device=DEVICE,
    )


# ============================================================
# Build anisotropic simulation (manual — fiber conductivity)
# ============================================================
def build_anisotropic():
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, Edge
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import (
        BidomainFDMDiscretization)
    from cardiac_sim.tissue_builder.stimulus import (
        StimulusProtocol, left_edge_region)
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    Lx = DX * (NX - 1)
    Ly = DX * (NY - 1)

    boundary_spec = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
    grid = StructuredGrid(Nx=NX, Ny=NY, Lx=Lx, Ly=Ly,
                          boundary_spec=boundary_spec,
                          _device=torch.device(DEVICE))

    # Fiber-based anisotropic conductivity: theta=0 → fibers along x
    cond = BidomainConductivity(
        D_i_fiber=D_I_FIBER, D_i_cross=D_I_CROSS,
        D_e_fiber=D_E_FIBER, D_e_cross=D_E_CROSS,
        theta=torch.zeros(NX, NY, dtype=torch.float64),
    )

    # Anisotropic uses standard 9pt stencil, not Mehrstellen
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0, stencil='5pt')

    stimulus = StimulusProtocol()
    stimulus.add_stimulus(
        region=left_edge_region(width=STIM_WIDTH),
        start_time=STIM_START, duration=STIM_DUR, amplitude=STIM_AMP,
    )

    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06', stimulus=stimulus,
        dt=DT, splitting='strang', parabolic_solver='pcg',
        elliptic_solver='auto', theta=0.5,
    )
    return sim, grid


# ============================================================
# Run one config
# ============================================================
def run_sim(label, sim, grid):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

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
        print(f"    t={state.t:6.0f}ms  lead={lead:.3f}cm  "
              f"front=[{front.min().item()}-{front.max().item()}]  "
              f"elapsed={elapsed:.0f}s")

    total = time.time() - t0

    # Steady-state lead (last 3 snapshots)
    leads = []
    for front in fronts[-3:]:
        center = front[NY // 2].item()
        edge = max(front[1].item(), front[-2].item())
        leads.append((edge - center) * DX)
    ss_lead = float(np.mean(leads))

    # Wavefront profile at last snapshot
    front_final = fronts[-1].float().numpy() * DX

    print(f"  DONE in {total:.0f}s — SS edge lead = {ss_lead:.3f} cm")

    return dict(
        label=label, ss_edge_lead_cm=ss_lead, wall_clock_s=total,
        times=times, front_final=front_final,
        leads_vs_time=[
            float((max(f[1].item(), f[-2].item()) - f[NY // 2].item()) * DX)
            for f in fronts
        ],
    )


# ============================================================
# Main
# ============================================================
def main():
    D_eff_iso = D_I * D_E / (D_I + D_E)
    D_eff_long = D_I_FIBER * D_E_FIBER / (D_I_FIBER + D_E_FIBER)
    D_eff_trans = D_I_CROSS * D_E_CROSS / (D_I_CROSS + D_E_CROSS)
    kleber_iso = math.sqrt((D_I + D_E) / D_E)
    kleber_aniso = math.sqrt((D_I_CROSS + D_E_CROSS) / D_E_CROSS)

    print("Anisotropic Conductivity Test")
    print(f"Grid: {NX}x{NY}, dx={DX}, T_END={T_END}ms, Device: {DEVICE}")
    print(f"\nIsotropic baseline:")
    print(f"  D_i={D_I:.6f}  D_e={D_E:.6f}  D_eff={D_eff_iso:.6f}  Kleber={kleber_iso:.4f}")
    print(f"\nAnisotropic 2:1:")
    print(f"  D_i_fiber={D_I_FIBER:.6f}  D_i_cross={D_I_CROSS:.6f}")
    print(f"  D_e_fiber={D_E_FIBER:.6f}  D_e_cross={D_E_CROSS:.6f}")
    print(f"  D_eff_long={D_eff_long:.6f}  D_eff_trans={D_eff_trans:.6f}")
    print(f"  Kleber (transverse, at y-boundary)={kleber_aniso:.4f}")
    print(f"\nEikonal prediction:")
    print(f"  lead_iso ~ 1/sqrt(D_eff) ∝ {1/math.sqrt(D_eff_iso):.1f}")
    print(f"  lead_aniso ~ sqrt(D_long)/D_trans ∝ {math.sqrt(D_eff_long)/D_eff_trans:.1f}")
    pred_ratio = (math.sqrt(D_eff_long) / D_eff_trans) / (1 / math.sqrt(D_eff_iso))
    print(f"  Predicted ratio (aniso/iso): {pred_ratio:.2f}x")
    print(f"Output: {OUTPUT_DIR}")
    print('=' * 60)

    # Run isotropic baseline
    sim_iso, grid_iso = build_isotropic()
    r_iso = run_sim('Isotropic baseline (5pt)', sim_iso, grid_iso)

    # Run anisotropic
    sim_aniso, grid_aniso = build_anisotropic()
    r_aniso = run_sim('Anisotropic 2:1 lower speed (5pt)', sim_aniso, grid_aniso)

    results = [r_iso, r_aniso]

    # Save
    results_save = []
    for r in results:
        rs = {k: v for k, v in r.items() if k != 'front_final'}
        rs['front_final'] = r['front_final'].tolist()
        results_save.append(rs)

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Isotropic:   SS lead = {r_iso['ss_edge_lead_cm']:.3f} cm")
    print(f"  Anisotropic: SS lead = {r_aniso['ss_edge_lead_cm']:.3f} cm")
    ratio = r_aniso['ss_edge_lead_cm'] / r_iso['ss_edge_lead_cm'] \
        if r_iso['ss_edge_lead_cm'] > 0 else float('inf')
    print(f"  Ratio: {ratio:.2f}x (predicted: {pred_ratio:.2f}x)")
    if ratio > 1.2:
        print(f"  → Anisotropic has LARGER lead (sharper triangle)")
    elif ratio < 0.8:
        print(f"  → Anisotropic has SMALLER lead (more bowl-like)")
    else:
        print(f"  → Similar lead distances")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    y_cm = np.arange(NY) * DX

    # Panel 1: Lead vs time
    ax = axes[0]
    for r, c in zip(results, ['blue', 'red']):
        ax.plot(r['times'], r['leads_vs_time'], color=c, linewidth=2,
                label=r['label'])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Edge lead (cm)')
    ax.set_title('Edge Lead Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Final wavefront profile
    ax = axes[1]
    for r, c in zip(results, ['blue', 'red']):
        # Normalize: subtract center position
        front = np.array(r['front_final'])
        center = front[NY // 2]
        ax.plot(y_cm, front - center, color=c, linewidth=2, label=r['label'])
    ax.set_xlabel('y (cm)')
    ax.set_ylabel('Front position relative to center (cm)')
    ax.set_title('Steady-State Wavefront Shape')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Normalized shape comparison
    ax = axes[2]
    for r, c in zip(results, ['blue', 'red']):
        front = np.array(r['front_final'])
        center = front[NY // 2]
        dev = front - center
        if dev.max() > 0:
            dev_norm = dev / dev.max()
        else:
            dev_norm = dev
        ax.plot(y_cm / y_cm[-1], dev_norm, color=c, linewidth=2,
                label=r['label'])
    ax.set_xlabel('y / Ly (normalized)')
    ax.set_ylabel('Normalized deviation')
    ax.set_title('Normalized Shape (bowl vs triangle?)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'anisotropic_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {path}")


if __name__ == '__main__':
    main()
