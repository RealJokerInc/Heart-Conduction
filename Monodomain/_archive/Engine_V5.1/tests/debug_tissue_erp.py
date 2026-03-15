#!/usr/bin/env python3
"""
Phase 0.D: 2D Tissue ERP Measurement

This script measures the Effective Refractory Period (ERP) in 2D tissue
with different S2 electrode sizes to quantify the source-sink effect.

Key hypothesis: Larger S2 electrodes should have lower tissue ERP because
they reduce source-sink mismatch (more "source" current relative to "sink").

Expected results:
- Small S2 (point): Tissue ERP >> APD90 (high source-sink mismatch)
- Large S2 (plane wave): Tissue ERP ≈ APD90 (minimal source-sink mismatch)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from tissue import MonodomainSimulation
from ionic import CellType

def measure_tissue_erp(
    s2_size_cm: float,
    grid_size: int = 200,
    domain_cm: float = 5.0,
    verbose: bool = True
) -> dict:
    """
    Measure tissue ERP for a given S2 electrode size.

    Args:
        s2_size_cm: Size of S2 electrode (square, in cm)
        grid_size: Number of grid cells
        domain_cm: Physical domain size in cm
        verbose: Print detailed output

    Returns:
        Dict with ERP, APD90, and other measurements
    """
    dx = domain_cm / grid_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert S2 size to grid cells
    s2_cells = max(2, int(s2_size_cm / dx))
    s2_actual_cm = s2_cells * dx

    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing S2 size: {s2_size_cm:.2f} cm ({s2_cells} cells)")
        print(f"{'='*60}")

    # Test coupling intervals
    coupling_intervals = [200, 250, 280, 300, 320, 340, 360, 380, 400, 450, 500]

    results = []
    erp_found = None

    for ci in coupling_intervals:
        # Create fresh simulation for each test
        sim = MonodomainSimulation(
            ny=grid_size, nx=grid_size,
            dx=dx, dy=dx,
            cv_long=0.06, cv_trans=0.02,
            celltype=CellType.ENDO,
            device=device,
            params_override=None  # Normal APD
        )

        dt = sim.diffusion.get_stability_limit() * 0.8

        # S1: Full left edge plane wave at t=5ms
        s1_time = 5.0
        s1_duration = 1.0

        # S2: Square electrode at center at t = s1_time + CI
        s2_time = s1_time + ci
        s2_duration = 2.0  # Longer stimulus for S2

        cy, cx = grid_size // 2, grid_size // 2
        s2_y_start = cy - s2_cells // 2
        s2_y_end = cy + s2_cells // 2
        s2_x_start = cx - s2_cells // 2
        s2_x_end = cx + s2_cells // 2

        # Track max V after S2 in the S2 region
        max_V_after_s2 = -90.0
        s2_applied = False

        # Run simulation
        sim_end = s2_time + 50.0  # Run 50ms after S2

        while sim.time < sim_end:
            # Apply stimuli
            V = sim.get_voltage()

            # S1: Left edge plane wave
            if s1_time <= sim.time < s1_time + s1_duration:
                V[:, :5] = 20.0
                sim.set_voltage(V)

            # S2: Central electrode
            if s2_time <= sim.time < s2_time + s2_duration:
                V[s2_y_start:s2_y_end, s2_x_start:s2_x_end] = 20.0
                sim.set_voltage(V)
                s2_applied = True

            # Step simulation
            sim.step(dt)

            # Track max V in S2 region after S2
            if sim.time > s2_time + s2_duration:
                V = sim.get_voltage()
                V_s2_region = V[s2_y_start:s2_y_end, s2_x_start:s2_x_end]
                current_max = V_s2_region.max().item()
                max_V_after_s2 = max(max_V_after_s2, current_max)

        # Determine if AP was triggered (max V > 0mV indicates upstroke)
        ap_triggered = max_V_after_s2 > 0

        results.append({
            'ci': ci,
            'max_V': max_V_after_s2,
            'ap': ap_triggered
        })

        if verbose:
            status = "AP" if ap_triggered else "No AP"
            print(f"  CI={ci:3d}ms: max_V={max_V_after_s2:+6.1f}mV -> {status}")

        # Find ERP (first CI that triggers AP)
        if ap_triggered and erp_found is None:
            erp_found = ci

    # If no ERP found within range, it's > max tested
    if erp_found is None:
        erp_found = coupling_intervals[-1] + 50  # Mark as > max

    return {
        's2_size_cm': s2_actual_cm,
        's2_cells': s2_cells,
        'erp': erp_found,
        'results': results
    }


def run_s2_size_sweep():
    """Test tissue ERP with different S2 electrode sizes."""

    print("="*70)
    print("PHASE 0.D: 2D TISSUE ERP vs S2 ELECTRODE SIZE")
    print("="*70)
    print()
    print("Testing hypothesis: Larger S2 electrodes reduce source-sink mismatch")
    print("                    and should show lower tissue ERP")
    print()

    # Test different S2 sizes
    s2_sizes = [0.1, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0]  # cm

    results = []

    for s2_size in s2_sizes:
        result = measure_tissue_erp(s2_size, verbose=True)
        results.append(result)

    # Summary
    print()
    print("="*70)
    print("SUMMARY: TISSUE ERP vs S2 SIZE")
    print("="*70)
    print()
    print(f"{'S2 Size (cm)':>12} | {'S2 Cells':>10} | {'Tissue ERP (ms)':>15}")
    print("-"*45)

    for r in results:
        erp_str = f"{r['erp']}" if r['erp'] <= 500 else ">500"
        print(f"{r['s2_size_cm']:>12.2f} | {r['s2_cells']:>10} | {erp_str:>15}")

    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()

    # Check if larger S2 gives lower ERP
    erps = [r['erp'] for r in results]
    sizes = [r['s2_size_cm'] for r in results]

    if erps[0] > erps[-1]:
        print("CONFIRMED: Larger S2 electrodes have LOWER tissue ERP")
        print("           This validates the source-sink mismatch hypothesis")
        print()
        print(f"  Small S2 ({sizes[0]:.1f}cm): Tissue ERP = {erps[0]}ms")
        print(f"  Large S2 ({sizes[-1]:.1f}cm): Tissue ERP = {erps[-1]}ms")
        print(f"  Difference: {erps[0] - erps[-1]}ms")
    else:
        print("UNEXPECTED: S2 size does not significantly affect tissue ERP")
        print("            Need to investigate other factors")

    print()
    print("RECOMMENDATIONS FOR SPIRAL WAVE S1-S2 PROTOCOL:")
    print("-"*50)

    # Find minimum S2 size for reasonable ERP
    for i, r in enumerate(results):
        if r['erp'] <= 320:  # Within ~20ms of APD90
            print(f"  Minimum S2 size for ERP ≤ 320ms: {r['s2_size_cm']:.1f}cm")
            break
    else:
        print("  Even largest S2 tested has high ERP!")
        print("  Consider using plane wave S2 stimulus")

    return results


def test_plane_wave_s2():
    """Test tissue ERP with full plane wave S2 (no source-sink mismatch)."""

    print()
    print("="*70)
    print("CONTROL TEST: PLANE WAVE S2 (No source-sink mismatch)")
    print("="*70)

    grid_size = 200
    domain_cm = 5.0
    dx = domain_cm / grid_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    coupling_intervals = [200, 220, 240, 260, 280, 300, 320, 340]

    print("\nTesting S2 as FULL plane wave (left edge):")
    print("-"*50)

    erp_found = None

    for ci in coupling_intervals:
        sim = MonodomainSimulation(
            ny=grid_size, nx=grid_size,
            dx=dx, dy=dx,
            cv_long=0.06, cv_trans=0.02,
            celltype=CellType.ENDO,
            device=device,
            params_override=None
        )

        dt = sim.diffusion.get_stability_limit() * 0.8

        s1_time = 5.0
        s1_duration = 1.0
        s2_time = s1_time + ci
        s2_duration = 2.0

        # Track max V after S2 on left edge
        max_V_after_s2 = -90.0

        sim_end = s2_time + 50.0

        while sim.time < sim_end:
            V = sim.get_voltage()

            # S1: Left edge
            if s1_time <= sim.time < s1_time + s1_duration:
                V[:, :5] = 20.0
                sim.set_voltage(V)

            # S2: Also left edge (plane wave)
            if s2_time <= sim.time < s2_time + s2_duration:
                V[:, :5] = 20.0
                sim.set_voltage(V)

            sim.step(dt)

            if sim.time > s2_time + s2_duration:
                V = sim.get_voltage()
                current_max = V[:, 5:20].max().item()  # Check just beyond stimulus region
                max_V_after_s2 = max(max_V_after_s2, current_max)

        ap_triggered = max_V_after_s2 > 0
        status = "AP" if ap_triggered else "No AP"
        print(f"  CI={ci:3d}ms: max_V={max_V_after_s2:+6.1f}mV -> {status}")

        if ap_triggered and erp_found is None:
            erp_found = ci

    if erp_found:
        print(f"\nPLANE WAVE ERP = {erp_found}ms")
        print("This should be close to APD90 (~297ms) since there's no source-sink mismatch")
    else:
        print("\nNo ERP found in tested range!")

    return erp_found


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure 2D tissue ERP vs S2 size")
    parser.add_argument('--sweep', action='store_true', help="Run full S2 size sweep")
    parser.add_argument('--plane', action='store_true', help="Test plane wave S2")
    parser.add_argument('--quick', action='store_true', help="Quick test with small grid")

    args = parser.parse_args()

    if args.plane:
        test_plane_wave_s2()
    elif args.sweep or not any([args.plane, args.quick]):
        run_s2_size_sweep()

    if args.quick:
        # Quick test with smaller grid
        print("\nQuick test mode (reduced grid size)...")
        result = measure_tissue_erp(s2_size_cm=1.0, grid_size=100, domain_cm=3.0)
        print(f"\nQuick test result: ERP = {result['erp']}ms for S2 size = {result['s2_size_cm']:.2f}cm")
