#!/usr/bin/env python
"""
Triangle Merger Experiment — Kleber Boundary Speedup Wavefront Characterization

Domain: 50x8 cm, dx=dy=0.05, NX=1001, NY=161, dt=0.01, T_END=800ms, GPU

Three configurations:
1. monodomain_mehrstellen — explicit FDM + Rush-Larsen, Mehrstellen 9pt,
   D_eff, Neumann BCs. Flat-wavefront reference.
2. bidomain_5pt — bath_tb BCs, 5-point stencil. Baseline for comparison.
3. bidomain_mehrstellen — bath_tb BCs, Mehrstellen 9-point. Primary result.

Snapshots every 25ms (32 total). Full Vm grid + wavefront profile.
Activation time tracking for isochrone map.

Output: Research/Q5_boundary_conduction_speedup/triangle_merger/
"""

import sys
import os
import time
import json
import torch

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.join(_HERE, '..')
_TESTS = os.path.join(_ENGINE, 'tests')
sys.path.insert(0, _ENGINE)
sys.path.insert(0, _TESTS)

torch.set_default_dtype(torch.float64)

# ============================================================
# Parameters
# ============================================================
NX, NY = 1001, 161
DX = 0.05       # cm
DT = 0.01       # ms
T_END = 800.0   # ms
SAVE_EVERY = 25.0  # ms
THRESHOLD = -30.0  # mV activation threshold

# Domain
LX = DX * (NX - 1)  # 50 cm
LY = DX * (NY - 1)  # 8 cm

# Conductivity (same as cv_shared)
SIGMA_I = 1.74
SIGMA_E = 6.25
CHI = 1400.0
CM = 1.0
D_I = SIGMA_I / (CHI * CM)
D_E = SIGMA_E / (CHI * CM)
D_EFF = D_I * D_E / (D_I + D_E)

# Stimulus
STIM_COLS = 5
STIM_WIDTH = STIM_COLS * DX
STIM_START = 1.0
STIM_DUR = 2.0
STIM_AMP = -80.0

# Output directory
OUTPUT_DIR = os.path.join(
    _HERE, '..', '..', '..', 'Research',
    'Q5_boundary_conduction_speedup', 'triangle_merger')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Domain: {NX}x{NY} = {LX:.0f}x{LY:.0f} cm, dx={DX}, dt={DT}")
print(f"D_i={D_I:.6f}, D_e={D_E:.6f}, D_eff={D_EFF:.6f} cm²/ms")
print(f"Output: {OUTPUT_DIR}")


# ============================================================
# Wavefront extraction
# ============================================================
def extract_wavefront(V, threshold=THRESHOLD):
    """Find the leading wavefront x-position for each y-row.

    Returns array of shape (NY,) with the x-index of the leading
    wavefront at each row. Returns 0 where the wave hasn't arrived.
    """
    front = torch.zeros(V.shape[1], dtype=torch.long, device=V.device)
    for j in range(V.shape[1]):
        col = V[:, j]
        activated = (col > threshold).nonzero(as_tuple=False)
        if activated.numel() > 0:
            front[j] = activated[-1, 0]  # rightmost activated node
    return front


# ============================================================
# Run 1: Monodomain Mehrstellen (flat reference)
# ============================================================
def run_monodomain_mehrstellen():
    """Explicit Euler + Rush-Larsen + Mehrstellen Laplacian, Neumann BCs."""
    from cardiac_sim.ionic.ttp06.model import TTP06Model
    from cardiac_sim.simulation.classical.discretization.fdm import (
        _neumann_tridiag, _sparse_kron, _speye)

    print("\n" + "=" * 60)
    print("Config 1: Monodomain Mehrstellen (flat reference)")
    print("=" * 60)

    device = torch.device(DEVICE)
    model = TTP06Model(device=DEVICE)

    V = torch.full((NX, NY), model.V_rest, device=device)
    S = model.get_initial_state(NX * NY).reshape(NX, NY, -1)

    # Build Mehrstellen Laplacian on GPU
    h = DX
    T_x = _neumann_tridiag(NX, device, torch.float64)
    T_y = _neumann_tridiag(NY, device, torch.float64)
    I_x = _speye(NX, device, torch.float64)
    I_y = _speye(NY, device, torch.float64)
    L = (D_EFF / (6.0 * h * h)) * (
        6.0 * (_sparse_kron(I_x, T_y) + _sparse_kron(T_x, I_y))
        + _sparse_kron(T_x, T_y)
    ).coalesce()

    # CFL check
    lam_max = D_EFF * 32.0 / (6.0 * h * h)
    dt_max = 2.0 / lam_max
    dt = DT
    if dt > dt_max:
        print(f"  WARNING: dt={dt} > CFL={dt_max:.4f}, using {dt_max*0.8:.4f}")
        dt = dt_max * 0.8

    stim_mask = torch.zeros(NX, NY, dtype=torch.bool, device=device)
    stim_mask[:STIM_COLS, :] = True

    # Activation time tracking
    act_time = torch.full((NX, NY), float('inf'), device=device)
    activated = torch.zeros(NX, NY, dtype=torch.bool, device=device)

    snapshots = []
    fronts = []
    times = []
    t = 0.0
    next_save = SAVE_EVERY
    n_steps = int(T_END / dt + 0.5)
    t0 = time.time()

    for step_i in range(n_steps):
        V_flat = V.reshape(-1)
        S_flat = S.reshape(-1, S.shape[-1])

        Iion = model.compute_Iion(V_flat, S_flat)
        Istim = torch.zeros_like(V_flat)
        if STIM_START <= t < STIM_START + STIM_DUR:
            Istim[stim_mask.reshape(-1)] = STIM_AMP

        V_flat = V_flat + dt * (-(Iion + Istim))

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

        # Diffusion
        lap = torch.sparse.mm(L, V.reshape(-1, 1)).squeeze(1).reshape(NX, NY)
        V = V + dt * lap

        t += dt

        # Track activation times
        newly_act = (V > THRESHOLD) & ~activated
        if newly_act.any():
            act_time[newly_act] = t
            activated |= newly_act

        if t >= next_save - 1e-12:
            next_save += SAVE_EVERY
            front = extract_wavefront(V)
            snapshots.append(V.cpu().clone())
            fronts.append(front.cpu().clone())
            times.append(t)
            elapsed = time.time() - t0
            print(f"  t={t:.0f}ms  front_range=[{front.min().item()}-{front.max().item()}]  "
                  f"elapsed={elapsed:.0f}s")

    total = time.time() - t0
    print(f"  DONE in {total:.0f}s ({n_steps} steps)")

    return {
        'name': 'monodomain_mehrstellen',
        'snapshots': snapshots,
        'fronts': fronts,
        'times': times,
        'act_time': act_time.cpu(),
    }


# ============================================================
# Run 2/3: Bidomain (5pt or Mehrstellen)
# ============================================================
def run_bidomain(stencil='5pt'):
    """Bidomain with bath_tb BCs."""
    from cv_shared import build_bidomain_sim

    label = f"bidomain_{stencil.replace('-', '')}"
    print("\n" + "=" * 60)
    print(f"Config: {label} (bath_tb BCs)")
    print("=" * 60)

    sim, grid = build_bidomain_sim(
        nx=NX, ny=NY, dx=DX, dt=DT, D_i=D_I, D_e=D_E,
        bc_type='bath_tb', stencil=stencil,
        stim_width=STIM_WIDTH, stim_start=STIM_START,
        stim_dur=STIM_DUR, stim_amp=STIM_AMP,
        device=DEVICE,
    )

    # Activation time tracking
    act_time = torch.full((NX, NY), float('inf'))
    activated = torch.zeros(NX, NY, dtype=torch.bool)

    snapshots = []
    fronts = []
    times = []
    t0 = time.time()

    for state in sim.run(t_end=T_END, save_every=SAVE_EVERY):
        V = grid.flat_to_grid(state.Vm.cpu())
        front = extract_wavefront(V)
        snapshots.append(V.clone())
        fronts.append(front.clone())
        times.append(state.t)

        # Track activation
        newly_act = (V > THRESHOLD) & ~activated
        if newly_act.any():
            act_time[newly_act] = state.t
            activated |= newly_act

        elapsed = time.time() - t0
        print(f"  t={state.t:.0f}ms  front_range=[{front.min().item()}-{front.max().item()}]  "
              f"elapsed={elapsed:.0f}s")

    total = time.time() - t0
    print(f"  DONE in {total:.0f}s")

    return {
        'name': label,
        'snapshots': snapshots,
        'fronts': fronts,
        'times': times,
        'act_time': act_time,
    }


# ============================================================
# Assertions
# ============================================================
def verify_results(mono, bi5, bi9):
    """Embedded assertions from the plan."""
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    all_ok = True

    # 1. Monodomain wavefront flat
    for i, (front, t) in enumerate(zip(mono['fronts'], mono['times'])):
        if front.max().item() == 0:
            continue  # wave hasn't started
        front_cm = front.float() * DX
        deviation = (front_cm.max() - front_cm.min()).item()
        if deviation > 0.1:
            print(f"  FAIL: Mono wavefront not flat at t={t:.0f}ms: deviation={deviation:.3f}cm")
            all_ok = False

    if all_ok:
        print("  PASS: Monodomain wavefront flat (deviation < 0.1cm)")

    # 2. Bidomain edge leads center for t > 50ms
    edge_leads = True
    for front, t in zip(bi9['fronts'], bi9['times']):
        if t < 50:
            continue
        if front.max().item() == 0:
            continue
        center = front[NY // 2].item()
        edge_top = front[-2].item()   # second from top (first interior)
        edge_bot = front[1].item()     # second from bottom
        edge = max(edge_top, edge_bot)
        if edge <= center and center > 0:
            print(f"  WARN: Edge not leading at t={t:.0f}ms: edge={edge}, center={center}")
            edge_leads = False

    if edge_leads:
        print("  PASS: Bidomain edge leads center for t > 50ms")

    # 3. No NaN/Inf
    nan_ok = True
    for result in [mono, bi5, bi9]:
        for V in result['snapshots']:
            if not torch.isfinite(V).all():
                print(f"  FAIL: NaN/Inf in {result['name']}")
                nan_ok = False
                break
    if nan_ok:
        print("  PASS: No NaN/Inf in any snapshot")

    # 4. Wave stays within domain
    in_domain = True
    for result in [mono, bi5, bi9]:
        for front in result['fronts']:
            max_front = front.max().item() * DX
            if max_front > 48:
                print(f"  FAIL: Wave exceeded 48cm in {result['name']}: {max_front:.1f}cm")
                in_domain = False
    if in_domain:
        print("  PASS: Wave stays within domain (< 48cm)")


# ============================================================
# Save data
# ============================================================
def save_data(results):
    """Save fronts and activation times for visualization."""
    for r in results:
        name = r['name']

        # Save fronts as tensor
        fronts_tensor = torch.stack(r['fronts'])  # (n_snaps, NY)
        torch.save(fronts_tensor, os.path.join(OUTPUT_DIR, f'{name}_fronts.pt'))

        # Save activation time map
        torch.save(r['act_time'], os.path.join(OUTPUT_DIR, f'{name}_act_time.pt'))

        # Save times
        with open(os.path.join(OUTPUT_DIR, f'{name}_times.json'), 'w') as f:
            json.dump(r['times'], f)

        # Save last snapshot Vm
        if r['snapshots']:
            torch.save(r['snapshots'][-1], os.path.join(OUTPUT_DIR, f'{name}_Vm_final.pt'))

        # Save snapshots at key times (200, 400, 600, 800ms)
        key_times = [200, 400, 600, 800]
        for kt in key_times:
            idx = None
            for i, t in enumerate(r['times']):
                if abs(t - kt) < SAVE_EVERY / 2:
                    idx = i
                    break
            if idx is not None:
                torch.save(r['snapshots'][idx],
                           os.path.join(OUTPUT_DIR, f'{name}_Vm_{kt}ms.pt'))

    print(f"\nData saved to {OUTPUT_DIR}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Triangle Merger Experiment")
    print("=" * 60)

    # Run all three configurations
    mono = run_monodomain_mehrstellen()
    bi5 = run_bidomain(stencil='5pt')
    bi9 = run_bidomain(stencil='mehrstellen')

    # Verify
    verify_results(mono, bi5, bi9)

    # Save
    save_data([mono, bi5, bi9])

    print("\nExperiment complete. Run visualizations next.")
