#!/usr/bin/env python3
"""
Diagnostic: Boundary Artifact Investigation

Runs plane wave propagation with all 4 BC modes + monodomain control.
Saves voltage and phi_e snapshots as PNG heatmaps.
Measures wavefront position at center vs boundary rows.

Goal: determine if boundary speedup artifact is:
  (a) A bug (same result regardless of BC mode)
  (b) A bidomain physics effect (differs from monodomain even with Neumann)
  (c) A display/measurement issue

Output: diag_boundary/ directory with PNG snapshots and CSV data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import time

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, Edge
from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
from cardiac_sim.tissue_builder.stimulus import StimulusProtocol

# Try matplotlib; fall back to raw data if unavailable
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, saving raw .npy files only")


# === Parameters (same for all runs) ===
NX, NY = 121, 61       # 3.0 x 1.5 cm at dx=0.025
DX = 0.025
DT = 0.01
LX = DX * (NX - 1)     # 3.0 cm
LY = DX * (NY - 1)     # 1.5 cm

# Conductivities
SIGMA_I, SIGMA_E = 1.74, 6.25
CHI, CM = 1400.0, 1.0
D_I = SIGMA_I / (CHI * CM)
D_E = SIGMA_E / (CHI * CM)
D_EFF = D_I * D_E / (D_I + D_E)

# Stimulus: voltage clamp left edge (matching interactive script)
S1_CELLS = 5

# Timing
T_END = 25.0            # ms (wave travels ~1.3 cm)
SNAP_TIMES = [5.0, 10.0, 15.0, 20.0, 25.0]

# Wavefront detection
THRESHOLD = -30.0       # mV

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), 'diag_boundary')


# === BC modes ===
BC_CONFIGS = {
    'insulated':  BoundarySpec.insulated(),
    'bath_all':   BoundarySpec.bath_coupled(),
    'bath_tb':    BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM]),
    'bath_lr':    BoundarySpec.bath_coupled_edges([Edge.LEFT, Edge.RIGHT]),
}


def run_bidomain(bc_name, bc_spec):
    """Run bidomain simulation with given BCs, return snapshots."""
    print(f"\n  [{bc_name}] Building...")
    grid = StructuredGrid(Nx=NX, Ny=NY, Lx=LX, Ly=LY,
                          boundary_spec=bc_spec)
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    spatial = BidomainFDMDiscretization(grid, cond, chi=1.0, Cm=1.0)

    sim = BidomainSimulation(
        spatial=spatial,
        ionic_model='ttp06',
        stimulus=StimulusProtocol(),  # empty — using voltage clamp
        dt=DT,
        splitting='strang',
        elliptic_solver='auto',
        theta=0.5,
    )
    print(f"  [{bc_name}] Elliptic solver: {sim._elliptic_solver_name}")

    # Apply S1: voltage clamp left edge
    V_grid = grid.flat_to_grid(sim.state.Vm)
    V_grid[:S1_CELLS, :] = 20.0
    sim.state.Vm = grid.grid_to_flat(V_grid)

    # Run and collect snapshots
    state = sim.state
    splitting = sim.splitting
    snapshots = {}
    snap_idx = 0

    t0 = time.time()
    n_steps = int(T_END / DT + 0.5)
    next_snap = SNAP_TIMES[0] if SNAP_TIMES else T_END + 1

    for step_i in range(n_steps):
        splitting.step(state, DT)
        state.t += DT

        if snap_idx < len(SNAP_TIMES) and state.t >= SNAP_TIMES[snap_idx] - 1e-12:
            t_snap = SNAP_TIMES[snap_idx]
            V = grid.flat_to_grid(state.Vm).clone().numpy()
            phi_e = grid.flat_to_grid(state.phi_e).clone().numpy()
            snapshots[t_snap] = {'Vm': V, 'phi_e': phi_e}
            snap_idx += 1

    elapsed = time.time() - t0
    print(f"  [{bc_name}] Done in {elapsed:.1f}s ({n_steps} steps)")
    return snapshots


def run_monodomain():
    """Run monodomain FDM control (explicit Euler + Rush-Larsen)."""
    from cardiac_sim.ionic.ttp06.model import TTP06Model

    print(f"\n  [monodomain] Building...")
    model = TTP06Model(device='cpu')
    V = torch.full((NX, NY), model.V_rest, dtype=torch.float64)
    S = model.get_initial_state(NX * NY).reshape(NX, NY, -1)

    # Stability check
    D = D_EFF
    alpha = D / (DX * DX)
    dt_max = DX * DX / (4 * D)
    dt = min(DT, dt_max * 0.8)
    if dt < DT:
        print(f"  [monodomain] CFL limit: using dt={dt:.4f} instead of {DT}")

    # Apply S1: voltage clamp
    V[:S1_CELLS, :] = 20.0

    snapshots = {}
    snap_idx = 0
    t = 0.0
    n_steps = int(T_END / dt + 0.5)

    t0 = time.time()
    for step_i in range(n_steps):
        # Ionic step
        V_flat = V.reshape(-1)
        S_flat = S.reshape(-1, S.shape[-1])

        Iion = model.compute_Iion(V_flat, S_flat)
        V_flat = V_flat + dt * (-Iion)

        gate_inf = model.compute_gate_steady_states(V_flat, S_flat)
        gate_tau = model.compute_gate_time_constants(V_flat, S_flat)
        for k, gi in enumerate(model.gate_indices):
            tau_k = gate_tau[:, k].clamp(min=1e-6)
            inf_k = gate_inf[:, k]
            S_flat[:, gi] = inf_k + (S_flat[:, gi] - inf_k) * torch.exp(-dt / tau_k)
        conc_rates = model.compute_concentration_rates(V_flat, S_flat)
        for k, ci in enumerate(model.concentration_indices):
            S_flat[:, ci] = S_flat[:, ci] + dt * conc_rates[:, k]

        V = V_flat.reshape(NX, NY)
        S = S_flat.reshape(NX, NY, -1)

        # Diffusion step (explicit, Neumann)
        V_pad = torch.nn.functional.pad(
            V.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate'
        ).squeeze()
        lap = (V_pad[1:-1, 2:] + V_pad[1:-1, :-2] +
               V_pad[2:, 1:-1] + V_pad[:-2, 1:-1] -
               4 * V_pad[1:-1, 1:-1])
        V = V + dt * alpha * lap

        t += dt

        if snap_idx < len(SNAP_TIMES) and t >= SNAP_TIMES[snap_idx] - 1e-12:
            t_snap = SNAP_TIMES[snap_idx]
            snapshots[t_snap] = {'Vm': V.clone().numpy(), 'phi_e': None}
            snap_idx += 1

    elapsed = time.time() - t0
    print(f"  [monodomain] Done in {elapsed:.1f}s ({n_steps} steps, dt={dt:.4f})")
    return snapshots


def find_wavefront(V_grid, threshold=THRESHOLD):
    """Find wavefront x-position at each y-row.

    Returns array of shape (NY,) with x-position (in grid indices) of the
    rightmost cell above threshold. NaN if wave hasn't reached that row.
    """
    nx, ny = V_grid.shape
    front = np.full(ny, np.nan)
    for j in range(ny):
        above = np.where(V_grid[:, j] > threshold)[0]
        if len(above) > 0:
            front[j] = above[-1]  # rightmost activated cell
    return front


def save_plots(all_results, out_dir):
    """Save diagnostic PNG plots."""
    os.makedirs(out_dir, exist_ok=True)

    names = list(all_results.keys())
    n_modes = len(names)
    y_cm = np.arange(NY) * DX
    y_center = NY // 2

    for t_snap in SNAP_TIMES:
        # --- Voltage heatmaps ---
        fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4),
                                 squeeze=False)
        for i, name in enumerate(names):
            snap = all_results[name].get(t_snap)
            if snap is None:
                continue
            Vm = snap['Vm']
            ax = axes[0, i]
            # Display: x horizontal, y vertical, y increasing upward
            im = ax.imshow(Vm.T, origin='lower', aspect='auto',
                           extent=[0, LX, 0, LY],
                           vmin=-90, vmax=40, cmap='jet')
            ax.set_title(f'{name}\nt={t_snap:.0f}ms', fontsize=10)
            ax.set_xlabel('x (cm)')
            if i == 0:
                ax.set_ylabel('y (cm)')
            # Mark wavefront contour
            front = find_wavefront(Vm)
            valid = ~np.isnan(front)
            if valid.any():
                ax.plot(front[valid] * DX, y_cm[valid], 'w-', linewidth=1.5)

        plt.colorbar(im, ax=axes[0, -1], label='Vm (mV)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'Vm_t{t_snap:.0f}.png'), dpi=150)
        plt.close()

        # --- phi_e heatmaps (bidomain only) ---
        bidomain_names = [n for n in names if n != 'monodomain']
        if bidomain_names:
            fig, axes = plt.subplots(1, len(bidomain_names),
                                     figsize=(4 * len(bidomain_names), 4),
                                     squeeze=False)
            for i, name in enumerate(bidomain_names):
                snap = all_results[name].get(t_snap)
                if snap is None or snap['phi_e'] is None:
                    continue
                phi_e = snap['phi_e']
                ax = axes[0, i]
                vmax = max(abs(phi_e.min()), abs(phi_e.max()), 1e-6)
                ax.imshow(phi_e.T, origin='lower', aspect='auto',
                          extent=[0, LX, 0, LY],
                          vmin=-vmax, vmax=vmax, cmap='RdBu_r')
                ax.set_title(f'{name} phi_e\nt={t_snap:.0f}ms', fontsize=10)
                ax.set_xlabel('x (cm)')
                if i == 0:
                    ax.set_ylabel('y (cm)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'phi_e_t{t_snap:.0f}.png'), dpi=150)
            plt.close()

    # --- Wavefront position vs y at final time ---
    t_final = SNAP_TIMES[-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        snap = all_results[name].get(t_final)
        if snap is None:
            continue
        front = find_wavefront(snap['Vm'])
        front_cm = front * DX
        ax.plot(y_cm, front_cm, '-o', markersize=2, label=name)
    ax.set_xlabel('y position (cm)')
    ax.set_ylabel('Wavefront x position (cm)')
    ax.set_title(f'Wavefront shape at t={t_final:.0f}ms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wavefront_comparison.png'), dpi=150)
    plt.close()

    # --- Wavefront position vs y at ALL times ---
    fig, axes = plt.subplots(1, len(SNAP_TIMES), figsize=(4 * len(SNAP_TIMES), 4),
                             squeeze=False)
    for ti, t_snap in enumerate(SNAP_TIMES):
        ax = axes[0, ti]
        for name in names:
            snap = all_results[name].get(t_snap)
            if snap is None:
                continue
            front = find_wavefront(snap['Vm'])
            front_cm = front * DX
            ax.plot(y_cm, front_cm, '-', markersize=1, label=name)
        ax.set_title(f't={t_snap:.0f}ms', fontsize=10)
        ax.set_xlabel('y (cm)')
        if ti == 0:
            ax.set_ylabel('Wavefront x (cm)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wavefront_all_times.png'), dpi=150)
    plt.close()

    print(f"\n  Plots saved to {out_dir}/")


def save_raw(all_results, out_dir):
    """Save raw numpy arrays for inspection."""
    os.makedirs(out_dir, exist_ok=True)
    for name, snaps in all_results.items():
        for t_snap, data in snaps.items():
            prefix = f"{name}_t{t_snap:.0f}"
            np.save(os.path.join(out_dir, f'{prefix}_Vm.npy'), data['Vm'])
            if data['phi_e'] is not None:
                np.save(os.path.join(out_dir, f'{prefix}_phi_e.npy'), data['phi_e'])


def print_wavefront_table(all_results):
    """Print wavefront positions at center and edges."""
    print("\n" + "=" * 80)
    print("WAVEFRONT POSITION (x in cm) at each timepoint")
    print("=" * 80)
    names = list(all_results.keys())

    for t_snap in SNAP_TIMES:
        print(f"\n  t = {t_snap:.0f} ms:")
        print(f"  {'Mode':<15} {'x_center':>10} {'x_edge(y=1)':>12} "
              f"{'x_edge(y=N-2)':>14} {'delta_bot':>10} {'delta_top':>10}")
        print(f"  {'-'*13:<15} {'-'*10:>10} {'-'*12:>12} {'-'*14:>14} "
              f"{'-'*10:>10} {'-'*10:>10}")

        for name in names:
            snap = all_results[name].get(t_snap)
            if snap is None:
                continue
            front = find_wavefront(snap['Vm'])
            y_c = NY // 2
            y_bot = 1
            y_top = NY - 2

            xc = front[y_c] * DX if not np.isnan(front[y_c]) else float('nan')
            xb = front[y_bot] * DX if not np.isnan(front[y_bot]) else float('nan')
            xt = front[y_top] * DX if not np.isnan(front[y_top]) else float('nan')

            db = (xb - xc) if not (np.isnan(xb) or np.isnan(xc)) else float('nan')
            dt_val = (xt - xc) if not (np.isnan(xt) or np.isnan(xc)) else float('nan')

            def fmt(v):
                return f"{v:.4f}" if not np.isnan(v) else "N/A"

            print(f"  {name:<15} {fmt(xc):>10} {fmt(xb):>12} {fmt(xt):>14} "
                  f"{fmt(db):>10} {fmt(dt_val):>10}")


def main():
    print("=" * 70)
    print("BOUNDARY ARTIFACT DIAGNOSTIC")
    print("=" * 70)
    print(f"  Grid: {NX} x {NY} ({NX*NY:,} nodes)")
    print(f"  Domain: {LX:.2f} x {LY:.2f} cm")
    print(f"  dx = {DX}, dt = {DT}")
    print(f"  D_i = {D_I:.6f}, D_e = {D_E:.6f}, D_eff = {D_EFF:.6f}")
    print(f"  S1: voltage clamp, left {S1_CELLS} columns")
    print(f"  Snapshots at: {SNAP_TIMES} ms")

    all_results = {}

    # Run monodomain control first
    all_results['monodomain'] = run_monodomain()

    # Run all bidomain BC modes
    for bc_name, bc_spec in BC_CONFIGS.items():
        all_results[bc_name] = run_bidomain(bc_name, bc_spec)

    # Print numerical wavefront data
    print_wavefront_table(all_results)

    # Save outputs
    if HAS_MPL:
        save_plots(all_results, OUT_DIR)
    save_raw(all_results, OUT_DIR)

    print("\nDone. Check diag_boundary/ for output files.")


if __name__ == '__main__':
    main()
