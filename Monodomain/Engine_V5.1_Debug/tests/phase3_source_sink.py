"""
Phase 3: Source-Sink Effect Quantification

Tests the effect of S2 electrode size on tissue ERP to quantify source-sink mismatch.

Expected Results:
- 1x1 cell electrode: ERP = APD + 80-100ms (maximum mismatch)
- 5x5 cell electrode: ERP = APD + 40-60ms (moderate mismatch)
- 10x10 cell electrode: ERP = APD + 20-40ms (reduced mismatch)
- Full edge (plane wave): ERP ~ APD (no mismatch)

References:
- PMC6301915: Tissue geometry affects ERP by up to 38ms
- PMC5874259: Source-sink mismatch causes functional conduction block
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tissue.simulation import MonodomainSimulation
from ionic import StateIndex, CellType

def print_status(msg):
    print(f"[STATUS] {msg}")

def check_propagation(V_history, threshold=-40.0):
    """
    Check if S2 propagated across the tissue.

    Returns True if activation spreads to >50% of tissue.
    """
    n_times, ny, nx = V_history.shape

    # Check final portion of simulation
    V_end = V_history[-10:, :, :]  # Last 10 frames

    # Count cells that activated (went above threshold)
    activated = np.any(V_end > threshold, axis=0)
    activation_ratio = np.sum(activated) / (ny * nx)

    return activation_ratio > 0.5, activation_ratio

def measure_tissue_erp(sim_class, electrode_size, nx=100, ny=100, dx=0.01):
    """
    Measure tissue ERP for a given S2 electrode size using binary search.

    Parameters
    ----------
    sim_class : type
        MonodomainSimulation class
    electrode_size : int or str
        Size of S2 electrode (e.g., 1, 5, 10) or 'edge' for full edge
    nx, ny : int
        Grid dimensions
    dx : float
        Grid spacing (cm)

    Returns
    -------
    erp : float
        Effective refractory period (ms)
    details : dict
        Additional information
    """
    # Binary search bounds
    ci_low = 240   # Below this, always fails
    ci_high = 400  # Above this, always succeeds
    tolerance = 5  # ms

    print_status(f"Testing electrode size: {electrode_size}")

    results = []

    while ci_high - ci_low > tolerance:
        ci_test = (ci_low + ci_high) / 2

        # Create fresh simulation
        sim = sim_class(
            ny=ny, nx=nx, dx=dx, dy=dx,
            cv_long=0.06, cv_trans=0.02,
            celltype=CellType.ENDO,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # S1: Left edge stimulus
        sim.add_stimulus(
            region=(slice(None), slice(0, 5)),
            start_time=10.0,
            duration=1.0
        )

        # S2: Variable size electrode at center
        if electrode_size == 'edge':
            # Full bottom edge - plane wave
            s2_region = (slice(None), slice(0, 5))
        else:
            # Square electrode at center
            size = electrode_size
            center_y, center_x = ny // 2, nx // 2
            half = size // 2
            s2_region = (
                slice(max(0, center_y - half), min(ny, center_y + half + 1)),
                slice(max(0, center_x - half), min(nx, center_x + half + 1))
            )

        s2_time = 10.0 + ci_test  # CI measured from S1
        sim.add_stimulus(
            region=s2_region,
            start_time=s2_time,
            duration=1.0
        )

        # Run simulation
        t_end = s2_time + 150  # Allow time for propagation
        try:
            t, V = sim.run(t_end, dt=0.02, save_interval=5.0)

            # Check if S2 propagated
            propagated, ratio = check_propagation(V, threshold=-40.0)

            results.append((ci_test, propagated, ratio))

            if propagated:
                ci_high = ci_test
                print(f"  CI={ci_test:.0f}ms: PROPAGATED ({ratio*100:.0f}% activated)")
            else:
                ci_low = ci_test
                print(f"  CI={ci_test:.0f}ms: BLOCKED ({ratio*100:.0f}% activated)")

        except Exception as e:
            print(f"  CI={ci_test:.0f}ms: ERROR - {e}")
            ci_low = ci_test

    erp = ci_high

    return erp, {'results': results, 'electrode_size': electrode_size}

def run_phase3_quick():
    """
    Quick Phase 3 test with smaller grid and fewer electrode sizes.
    """
    print("=" * 70)
    print("PHASE 3: SOURCE-SINK EFFECT QUANTIFICATION (Quick Test)")
    print("=" * 70)
    print()

    # Smaller grid for faster testing
    nx, ny = 50, 50
    dx = 0.02  # 0.02 cm = 200 um spacing (coarser for speed)

    print(f"Grid: {nx}x{ny} cells")
    print(f"Spacing: {dx*10:.0f} mm")
    print(f"Domain: {nx*dx:.1f} x {ny*dx:.1f} cm")
    print()

    # Test different electrode sizes
    electrode_configs = [
        (1, "1x1 (point)"),
        (5, "5x5 (small patch)"),
        ('edge', "Full edge (plane wave)")
    ]

    results = []

    for size, desc in electrode_configs:
        print("-" * 50)
        erp, details = measure_tissue_erp(
            MonodomainSimulation,
            electrode_size=size,
            nx=nx, ny=ny, dx=dx
        )
        results.append({
            'size': size,
            'desc': desc,
            'erp': erp,
            'details': details
        })
        print(f"Result: ERP = {erp:.0f} ms")
        print()

    # Summary
    print("=" * 70)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 70)
    print()

    apd90 = 272  # From validation

    print(f"{'Electrode':<25} {'ERP (ms)':<12} {'PRR (ms)':<12} {'Expected PRR'}")
    print("-" * 65)

    for r in results:
        prr = r['erp'] - apd90
        if r['size'] == 1:
            expected = "80-100 ms"
        elif r['size'] == 5:
            expected = "40-60 ms"
        elif r['size'] == 'edge':
            expected = "~0 ms"
        else:
            expected = "20-40 ms"

        print(f"{r['desc']:<25} {r['erp']:<12.0f} {prr:<12.0f} {expected}")

    print()
    print("=" * 70)

    return results

def run_phase3_full():
    """
    Full Phase 3 test with standard grid and all electrode sizes.
    """
    print("=" * 70)
    print("PHASE 3: SOURCE-SINK EFFECT QUANTIFICATION")
    print("=" * 70)
    print()

    # Standard grid
    nx, ny = 100, 100
    dx = 0.01  # 0.01 cm = 100 um spacing

    print(f"Grid: {nx}x{ny} cells")
    print(f"Spacing: {dx*10:.1f} mm")
    print(f"Domain: {nx*dx:.1f} x {ny*dx:.1f} cm")
    print()

    # Test different electrode sizes
    electrode_configs = [
        (1, "1x1 (single cell)"),
        (3, "3x3 (tiny patch)"),
        (5, "5x5 (small patch)"),
        (10, "10x10 (medium patch)"),
        (20, "20x20 (large patch)"),
        ('edge', "Full edge (plane wave)")
    ]

    results = []

    for size, desc in electrode_configs:
        print("-" * 50)
        erp, details = measure_tissue_erp(
            MonodomainSimulation,
            electrode_size=size,
            nx=nx, ny=ny, dx=dx
        )
        results.append({
            'size': size,
            'desc': desc,
            'erp': erp,
            'details': details
        })
        print(f"Result: ERP = {erp:.0f} ms")
        print()

    # Summary
    print("=" * 70)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 70)
    print()

    apd90 = 272  # From validation

    print(f"{'Electrode':<25} {'ERP (ms)':<12} {'PRR (ms)':<12}")
    print("-" * 50)

    for r in results:
        prr = r['erp'] - apd90
        print(f"{r['desc']:<25} {r['erp']:<12.0f} {prr:<12.0f}")

    print()

    # Analysis
    print("ANALYSIS:")
    print()

    if len(results) >= 2:
        erp_point = results[0]['erp']
        erp_edge = results[-1]['erp']

        print(f"Source-sink effect magnitude: {erp_point - erp_edge:.0f} ms")
        print(f"  (difference between point and plane wave stimulation)")
        print()

        # Check if results match expected pattern
        if erp_point > erp_edge + 50:
            print("CONCLUSION: Source-sink effect CONFIRMED")
            print("  Point stimulation requires longer recovery than plane wave.")
        else:
            print("CONCLUSION: Source-sink effect SMALLER than expected")
            print("  May indicate model or test configuration issues.")

    print()
    print("=" * 70)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "phase3_data")
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "source_sink_results.csv")
    with open(results_file, 'w') as f:
        f.write("electrode_size,description,erp_ms,prr_ms\n")
        for r in results:
            size_str = str(r['size']) if r['size'] != 'edge' else 'edge'
            prr = r['erp'] - apd90
            f.write(f"{size_str},{r['desc']},{r['erp']:.0f},{prr:.0f}\n")

    print(f"Results saved to: {results_file}")

    return results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3: Source-Sink Quantification')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with smaller grid')
    args = parser.parse_args()

    if args.quick:
        run_phase3_quick()
    else:
        run_phase3_full()
