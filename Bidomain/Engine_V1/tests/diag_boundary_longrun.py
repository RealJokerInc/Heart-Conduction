#!/usr/bin/env python3
"""
Diagnostic: Long-run boundary artifact on 8x8cm domain.

Matches the interactive spiral wave parameters:
  - 8x8 cm domain, dx=0.025 (321x321)
  - dt=0.01, Strang splitting
  - S1 voltage clamp left edge
  - Run for 100ms

Compares: insulated vs bath_tb vs monodomain control.
Saves wavefront shape at 20, 40, 60, 80, 100 ms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, Edge
from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
from cardiac_sim.tissue_builder.stimulus import StimulusProtocol

# === Match interactive sim parameters ===
# Use coarser grid for tractable CPU runtime (~5 min total)
DOMAIN = 8.0
DX = 0.05           # 0.05 cm = 500 um (4x fewer nodes than interactive)
DT = 0.01
NX = int(DOMAIN / DX) + 1  # 161
NY = NX                      # 161

SIGMA_I, SIGMA_E = 1.74, 6.25
CHI, CM = 1400.0, 1.0
D_I = SIGMA_I / (CHI * CM)
D_E = SIGMA_E / (CHI * CM)
D_EFF = D_I * D_E / (D_I + D_E)

S1_CELLS = max(3, int(0.2 / DX))  # 8 cells = 0.2 cm
T_END = 100.0
SNAP_TIMES = [20.0, 40.0, 60.0, 80.0, 100.0]
THRESHOLD = -30.0

OUT_DIR = os.path.join(os.path.dirname(__file__), 'diag_boundary')


def run_bidomain(bc_name, bc_spec):
    """Run bidomain and collect wavefront snapshots."""
    print(f"\n  [{bc_name}] Building {NX}x{NY} grid...")
    grid = StructuredGrid(Nx=NX, Ny=NY, Lx=DOMAIN, Ly=DOMAIN,
                          boundary_spec=bc_spec)
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    spatial = BidomainFDMDiscretization(grid, cond, chi=1.0, Cm=1.0)

    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06',
        stimulus=StimulusProtocol(), dt=DT,
        splitting='strang', elliptic_solver='auto', theta=0.5)
    print(f"  [{bc_name}] Solver: {sim._elliptic_solver_name}")

    # S1 voltage clamp
    V_grid = grid.flat_to_grid(sim.state.Vm)
    V_grid[:S1_CELLS, :] = 20.0
    sim.state.Vm = grid.grid_to_flat(V_grid)

    state = sim.state
    splitting = sim.splitting
    snapshots = {}
    snap_idx = 0
    n_steps = int(T_END / DT + 0.5)

    t0 = time.time()
    for step_i in range(n_steps):
        splitting.step(state, DT)
        state.t += DT

        if snap_idx < len(SNAP_TIMES) and state.t >= SNAP_TIMES[snap_idx] - 1e-12:
            Vm = grid.flat_to_grid(state.Vm).clone().numpy()
            snapshots[SNAP_TIMES[snap_idx]] = Vm
            snap_idx += 1
            print(f"  [{bc_name}] t={SNAP_TIMES[snap_idx-1]:.0f}ms "
                  f"({time.time()-t0:.0f}s elapsed)")

    print(f"  [{bc_name}] Total: {time.time()-t0:.0f}s")
    return snapshots


def run_monodomain():
    """Monodomain FDM control."""
    from cardiac_sim.ionic.ttp06.model import TTP06Model

    print(f"\n  [monodomain] Building {NX}x{NY}...")
    model = TTP06Model(device='cpu')
    V = torch.full((NX, NY), model.V_rest, dtype=torch.float64)
    S = model.get_initial_state(NX * NY).reshape(NX, NY, -1)

    D = D_EFF
    alpha = D / (DX * DX)

    # S1 voltage clamp
    V[:S1_CELLS, :] = 20.0

    snapshots = {}
    snap_idx = 0
    t = 0.0
    n_steps = int(T_END / DT + 0.5)

    t0 = time.time()
    for step_i in range(n_steps):
        V_flat = V.reshape(-1)
        S_flat = S.reshape(-1, S.shape[-1])
        Iion = model.compute_Iion(V_flat, S_flat)
        V_flat = V_flat + DT * (-Iion)

        gate_inf = model.compute_gate_steady_states(V_flat, S_flat)
        gate_tau = model.compute_gate_time_constants(V_flat, S_flat)
        for k, gi in enumerate(model.gate_indices):
            tau_k = gate_tau[:, k].clamp(min=1e-6)
            inf_k = gate_inf[:, k]
            S_flat[:, gi] = inf_k + (S_flat[:, gi] - inf_k) * torch.exp(-DT / tau_k)
        conc_rates = model.compute_concentration_rates(V_flat, S_flat)
        for k, ci in enumerate(model.concentration_indices):
            S_flat[:, ci] = S_flat[:, ci] + DT * conc_rates[:, k]

        V = V_flat.reshape(NX, NY)
        S = S_flat.reshape(NX, NY, -1)

        V_pad = torch.nn.functional.pad(
            V.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate'
        ).squeeze()
        lap = (V_pad[1:-1, 2:] + V_pad[1:-1, :-2] +
               V_pad[2:, 1:-1] + V_pad[:-2, 1:-1] -
               4 * V_pad[1:-1, 1:-1])
        V = V + DT * alpha * lap

        t += DT
        if snap_idx < len(SNAP_TIMES) and t >= SNAP_TIMES[snap_idx] - 1e-12:
            snapshots[SNAP_TIMES[snap_idx]] = V.clone().numpy()
            snap_idx += 1
            print(f"  [monodomain] t={SNAP_TIMES[snap_idx-1]:.0f}ms "
                  f"({time.time()-t0:.0f}s elapsed)")

    print(f"  [monodomain] Total: {time.time()-t0:.0f}s")
    return snapshots


def find_wavefront(V_grid, threshold=THRESHOLD):
    nx, ny = V_grid.shape
    front = np.full(ny, np.nan)
    for j in range(ny):
        above = np.where(V_grid[:, j] > threshold)[0]
        if len(above) > 0:
            front[j] = above[-1]
    return front


def main():
    print("=" * 70)
    print("LONG-RUN BOUNDARY DIAGNOSTIC (8x8cm, 100ms)")
    print("=" * 70)
    print(f"  Grid: {NX}x{NY}, dx={DX}, dt={DT}")
    print(f"  D_i={D_I:.6f}, D_e={D_E:.6f}, D_eff={D_EFF:.6f}")
    print(f"  S1: {S1_CELLS} cols voltage clamp")

    all_results = {}
    all_results['monodomain'] = run_monodomain()
    all_results['insulated'] = run_bidomain('insulated', BoundarySpec.insulated())
    all_results['bath_tb'] = run_bidomain('bath_tb',
        BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM]))

    os.makedirs(OUT_DIR, exist_ok=True)
    y_cm = np.arange(NY) * DX

    # --- Wavefront comparison at each timepoint ---
    fig, axes = plt.subplots(1, len(SNAP_TIMES), figsize=(5 * len(SNAP_TIMES), 5),
                             squeeze=False)
    for ti, t_snap in enumerate(SNAP_TIMES):
        ax = axes[0, ti]
        for name in ['monodomain', 'insulated', 'bath_tb']:
            Vm = all_results[name].get(t_snap)
            if Vm is None:
                continue
            front = find_wavefront(Vm) * DX
            ax.plot(y_cm, front, '-', linewidth=1.5, label=name)
        ax.set_title(f't={t_snap:.0f}ms')
        ax.set_xlabel('y (cm)')
        if ti == 0:
            ax.set_ylabel('Wavefront x (cm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('8x8cm Domain — Wavefront Shape Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'longrun_wavefront.png'), dpi=150)
    plt.close()

    # --- Vm heatmaps at t=60, 100 ---
    for t_snap in [60.0, 100.0]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
        for i, name in enumerate(['monodomain', 'insulated', 'bath_tb']):
            Vm = all_results[name].get(t_snap)
            if Vm is None:
                continue
            ax = axes[0, i]
            im = ax.imshow(Vm.T, origin='lower', aspect='equal',
                           extent=[0, DOMAIN, 0, DOMAIN],
                           vmin=-90, vmax=40, cmap='jet')
            front = find_wavefront(Vm)
            valid = ~np.isnan(front)
            if valid.any():
                ax.plot(front[valid] * DX, y_cm[valid], 'w-', linewidth=1.5)
            ax.set_title(f'{name}\nt={t_snap:.0f}ms')
            ax.set_xlabel('x (cm)')
            if i == 0:
                ax.set_ylabel('y (cm)')
        plt.colorbar(im, ax=axes[0, -1], label='Vm (mV)', shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'longrun_Vm_t{t_snap:.0f}.png'), dpi=150)
        plt.close()

    # --- Print wavefront delta at edges ---
    print("\n" + "=" * 70)
    print("WAVEFRONT EDGE SPEEDUP (x_edge - x_center, in cm)")
    print("=" * 70)
    y_c = NY // 2
    y_bot, y_top = 1, NY - 2
    for t_snap in SNAP_TIMES:
        print(f"\n  t={t_snap:.0f}ms:")
        for name in ['monodomain', 'insulated', 'bath_tb']:
            Vm = all_results[name].get(t_snap)
            if Vm is None:
                continue
            front = find_wavefront(Vm) * DX
            xc = front[y_c]
            db = front[y_bot] - xc if not np.isnan(front[y_bot]) else float('nan')
            dt_val = front[y_top] - xc if not np.isnan(front[y_top]) else float('nan')
            xc_s = f"{xc:.3f}" if not np.isnan(xc) else "N/A"
            db_s = f"{db:+.3f}" if not np.isnan(db) else "N/A"
            dt_s = f"{dt_val:+.3f}" if not np.isnan(dt_val) else "N/A"
            print(f"    {name:<15} center={xc_s}  bot={db_s}  top={dt_s}")

    print(f"\n  Plots saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
