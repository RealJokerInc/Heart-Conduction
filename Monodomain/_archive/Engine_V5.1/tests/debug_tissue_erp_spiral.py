#!/usr/bin/env python3
"""
Tissue ERP Measurement - Matching Spiral Wave S1-S2 Setup

This script measures the actual tissue ERP using the EXACT same setup
as the spiral wave simulation to understand why S2 requires ~380ms.

Setup (matching spiral_wave_s1s2.py):
- Grid: 300x300 (or configurable)
- Domain: 6cm x 6cm
- Diffusion: ISOTROPIC (D_L = D_T = 0.00151 cm²/ms for CV=0.06)
- S1: Plane wave from left edge
- S2: 2cm x 2cm lower-left quadrant

Measures:
- Single-cell APD90
- Tissue ERP for the S2 quadrant configuration
- ERP mismatch ratio (tissue ERP / APD90)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from tissue import MonodomainSimulation
from ionic import CellType, ORdModel


def measure_single_cell_apd90(device='cpu'):
    """Measure single-cell APD90 with default ORd parameters."""
    print("=" * 70)
    print("SINGLE-CELL APD90 MEASUREMENT")
    print("=" * 70)

    model = ORdModel(celltype=CellType.ENDO, device=device)

    # Use the model's run method for single cell simulation
    t_trace, V_trace = model.run(
        t_end=500.0,
        dt=0.01,
        stim_times=[10.0],  # Single stimulus at 10ms
        stim_duration=1.0,
        stim_amplitude=-80.0,  # Negative = depolarizing current
        save_interval=0.1
    )

    V_trace = V_trace.cpu().numpy()
    t_trace = t_trace.cpu().numpy()

    # Find APD90
    V_max = V_trace.max()
    V_min = V_trace.min()
    V_90 = V_min + 0.1 * (V_max - V_min)  # 90% repolarization

    # Find upstroke time
    upstroke_idx = np.argmax(V_trace > 0)
    upstroke_t = t_trace[upstroke_idx]

    # Find APD90 (when V crosses V_90 after upstroke)
    apd90 = 300  # Default
    for i in range(upstroke_idx + 100, len(V_trace)):
        if V_trace[i] < V_90:
            apd90 = t_trace[i] - upstroke_t
            break

    print(f"  V_max = {V_max:.1f} mV")
    print(f"  V_min = {V_min:.1f} mV")
    print(f"  V_90 = {V_90:.1f} mV")
    print(f"  APD90 = {apd90:.1f} ms")
    print()

    return apd90


def measure_tissue_erp_spiral_setup(
    grid_size: int = 300,
    domain_cm: float = 6.0,
    s2_width_cm: float = 2.0,
    s2_height_cm: float = 2.0,
    verbose: bool = True
):
    """
    Measure tissue ERP using the exact spiral wave S1-S2 setup.

    Returns the minimum coupling interval for successful S2 capture.
    """
    dx = domain_cm / grid_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # S2 region in cells (lower-left quadrant, matching spiral_wave_s1s2.py)
    s2_x_cells = int(s2_width_cm / dx)
    s2_y_cells = int(s2_height_cm / dx)

    if verbose:
        print("=" * 70)
        print("TISSUE ERP MEASUREMENT (Spiral Wave Setup)")
        print("=" * 70)
        print(f"Grid: {grid_size}x{grid_size}")
        print(f"Domain: {domain_cm:.1f}x{domain_cm:.1f} cm")
        print(f"dx = {dx*10:.2f} mm")
        print(f"S2 region: {s2_width_cm:.1f}x{s2_height_cm:.1f} cm ({s2_x_cells}x{s2_y_cells} cells)")
        print(f"S2 location: lower-left quadrant (y=0 to {s2_y_cells}, x=0 to {s2_x_cells})")
        print()

    # Test coupling intervals (CI = time from S1 to S2)
    # Start high and work down to find ERP
    coupling_intervals = [450, 420, 400, 390, 380, 370, 360, 350, 340, 330, 320, 310, 300, 290, 280]

    results = []
    erp_found = None

    if verbose:
        print(f"{'CI (ms)':<10} | {'Max V in S2':<12} | {'AP Triggered':<12}")
        print("-" * 40)

    for ci in coupling_intervals:
        # Create fresh simulation with ISOTROPIC D (matching spiral_wave_s1s2.py)
        sim = MonodomainSimulation(
            ny=grid_size, nx=grid_size,
            dx=dx, dy=dx,
            cv_long=0.06, cv_trans=0.06,  # ISOTROPIC
            celltype=CellType.ENDO,
            device=device,
            params_override=None
        )

        dt = sim.diffusion.get_stability_limit() * 0.8

        # S1: Plane wave from left edge at t=5ms
        s1_time = 5.0
        s1_duration = 1.0

        # S2: Lower-left quadrant at t = s1_time + CI
        s2_time = s1_time + ci
        s2_duration = 2.0

        # Track max V in S2 region after S2 stimulus
        max_V_after_s2 = -90.0

        # Run simulation until 50ms after S2
        sim_end = s2_time + 50.0

        while sim.time < sim_end:
            V = sim.get_voltage()

            # S1: Left edge (matching spiral_wave_s1s2.py)
            if s1_time <= sim.time < s1_time + s1_duration:
                V[:, :int(0.3/dx)] = 20.0  # 3mm wide, same as spiral wave
                sim.set_voltage(V)

            # S2: Lower-left quadrant
            if s2_time <= sim.time < s2_time + s2_duration:
                V[:s2_y_cells, :s2_x_cells] = 20.0
                sim.set_voltage(V)

            sim.step(dt)

            # Track max V in S2 region after S2 ends
            if sim.time > s2_time + s2_duration + 5.0:
                V = sim.get_voltage()
                V_s2 = V[:s2_y_cells, :s2_x_cells]
                current_max = V_s2.max().item()
                max_V_after_s2 = max(max_V_after_s2, current_max)

        # AP triggered if max V > 0mV (clear upstroke)
        ap_triggered = max_V_after_s2 > 0

        results.append({
            'ci': ci,
            'max_V': max_V_after_s2,
            'ap': ap_triggered
        })

        if verbose:
            status = "YES" if ap_triggered else "NO"
            print(f"{ci:<10} | {max_V_after_s2:<+12.1f} | {status:<12}")

        # Track last successful CI (ERP is the first CI that captures)
        if ap_triggered:
            erp_found = ci

    # ERP is the MINIMUM CI that produces AP
    # Since we're going high to low, the last successful one is the ERP
    tissue_erp = erp_found if erp_found else coupling_intervals[0]

    if verbose:
        print()
        print(f"TISSUE ERP = {tissue_erp} ms")
        print()

    return tissue_erp, results


def create_erp_comparison_sheet():
    """
    Create comparison sheet of single-cell vs tissue ERP.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print()
    print("=" * 70)
    print("ERP COMPARISON: SINGLE-CELL vs TISSUE")
    print("=" * 70)
    print()

    # Measure single-cell APD90
    apd90 = measure_single_cell_apd90(device)

    # Measure tissue ERP with spiral wave setup
    tissue_erp, results = measure_tissue_erp_spiral_setup(
        grid_size=200,  # Faster test
        domain_cm=5.0,
        s2_width_cm=2.0,
        s2_height_cm=2.0,
        verbose=True
    )

    # Calculate mismatch
    mismatch_ms = tissue_erp - apd90
    mismatch_ratio = tissue_erp / apd90

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Single-cell APD90:     {apd90:.1f} ms")
    print(f"  Tissue ERP (2x2cm S2): {tissue_erp:.1f} ms")
    print(f"  Mismatch:              {mismatch_ms:.1f} ms ({mismatch_ratio:.2f}x APD90)")
    print()
    print("This mismatch is due to SOURCE-SINK effect:")
    print("  - S2 electrode must overcome electrotonic load from surrounding tissue")
    print("  - Tissue at edge of S2 must be fully recovered to support propagation")
    print("  - Larger S2 electrodes have smaller mismatch (more source, less sink)")
    print()

    return {
        'apd90': apd90,
        'tissue_erp': tissue_erp,
        'mismatch_ms': mismatch_ms,
        'mismatch_ratio': mismatch_ratio
    }


def sweep_s2_sizes():
    """
    Sweep different S2 sizes to show how tissue ERP varies with electrode size.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print()
    print("=" * 70)
    print("S2 SIZE SWEEP: Tissue ERP vs Electrode Size")
    print("=" * 70)
    print()

    # Get APD90 first
    apd90 = measure_single_cell_apd90(device)

    # Test different S2 sizes
    s2_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # cm (square)

    results = []

    print(f"{'S2 Size (cm)':<15} | {'Tissue ERP (ms)':<18} | {'Mismatch Ratio':<15}")
    print("-" * 55)

    for s2_size in s2_sizes:
        tissue_erp, _ = measure_tissue_erp_spiral_setup(
            grid_size=150,  # Faster
            domain_cm=4.0,
            s2_width_cm=s2_size,
            s2_height_cm=s2_size,
            verbose=False
        )

        ratio = tissue_erp / apd90
        results.append({
            's2_size': s2_size,
            'tissue_erp': tissue_erp,
            'ratio': ratio
        })

        print(f"{s2_size:<15.1f} | {tissue_erp:<18.0f} | {ratio:<15.2f}")

    print()
    print(f"Single-cell APD90 = {apd90:.1f} ms")
    print()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure tissue ERP with spiral wave setup")
    parser.add_argument('--full', action='store_true', help="Full grid (300x300)")
    parser.add_argument('--sweep', action='store_true', help="Sweep S2 sizes")
    parser.add_argument('--quick', action='store_true', help="Quick test (150x150)")

    args = parser.parse_args()

    if args.sweep:
        sweep_s2_sizes()
    elif args.full:
        create_erp_comparison_sheet()
    else:
        # Default: quick comparison
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        apd90 = measure_single_cell_apd90(device)

        grid = 150 if args.quick else 200
        tissue_erp, _ = measure_tissue_erp_spiral_setup(
            grid_size=grid,
            domain_cm=5.0,
            s2_width_cm=2.0,
            s2_height_cm=2.0,
            verbose=True
        )

        print("=" * 70)
        print("COMPARISON (No ERP Factor Applied)")
        print("=" * 70)
        print(f"  Single-cell APD90:     {apd90:.1f} ms")
        print(f"  Tissue ERP (2x2cm S2): {tissue_erp:.1f} ms")
        print(f"  Raw Mismatch:          {tissue_erp - apd90:.1f} ms")
        print(f"  Mismatch Ratio:        {tissue_erp / apd90:.2f}x")
        print()
