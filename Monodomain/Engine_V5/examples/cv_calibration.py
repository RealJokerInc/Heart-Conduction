"""
CV Calibration Script

Tests conduction velocity at multiple mesh sizes to verify
mesh-independent CV with the new CV-based parameter scaling.

Target human ventricular CV:
- Longitudinal: 0.6 m/s (0.06 cm/ms)
- Transverse: 0.2 m/s (0.02 cm/ms)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from time import perf_counter

from ionic import CellType
from tissue import MonodomainSimulation, get_diffusion_params


def measure_cv(dx: float, cv_target_long: float = 0.06, cv_target_trans: float = 0.02):
    """
    Run simulation and measure CV at given mesh size.

    Returns measured CV and comparison to target.
    """
    # Grid size to cover same physical domain (~1 cm x 0.5 cm)
    domain_x = 1.0  # cm
    domain_y = 0.5  # cm

    nx = int(domain_x / dx)
    ny = int(domain_y / dx)

    # Get D values for this mesh size
    D_L, D_T = get_diffusion_params(dx, cv_target_long, cv_target_trans)

    print(f"\n{'='*60}")
    print(f"Mesh: dx = {dx*1000:.0f} um, grid = {ny}x{nx}")
    print(f"Target CV: long={cv_target_long*10:.1f} m/s, trans={cv_target_trans*10:.1f} m/s")
    print(f"Computed D: D_L={D_L:.6f}, D_T={D_T:.6f} cm^2/ms")

    sim = MonodomainSimulation(
        ny=ny, nx=nx,
        dx=dx, dy=dx,
        cv_long=cv_target_long,
        cv_trans=cv_target_trans,
        celltype=CellType.ENDO,
        splitting='godunov'
    )

    # Check that D values match
    print(f"Simulation D: D_L={sim.D_L:.6f}, D_T={sim.D_T:.6f} cm^2/ms")

    # Stability limit
    dt_max = sim.diffusion.get_stability_limit()
    dt = min(0.02, dt_max * 0.9)
    print(f"Stability limit: {dt_max:.4f} ms, using dt={dt:.4f} ms")

    # Stimulus at left edge
    sim.add_stimulus(
        region=(slice(None), slice(0, max(3, int(0.03/dx)))),
        start_time=1.0,
        duration=1.0
    )

    # Run for enough time to cross domain
    t_end = 80.0  # ms

    print(f"Running simulation for {t_end} ms...")
    start = perf_counter()
    t, V = sim.run(t_end, dt=dt, save_interval=1.0)
    elapsed = perf_counter() - start
    print(f"Completed in {elapsed:.1f}s")

    # Compute activation time and CV
    act_time = sim.compute_activation_time(V, threshold=-40.0)
    cv_measured = sim.compute_conduction_velocity(act_time, direction='x')

    # Compute error
    error_pct = 100 * (cv_measured - cv_target_long) / cv_target_long

    print(f"\nResults:")
    print(f"  Measured CV: {cv_measured*10:.2f} m/s ({cv_measured:.4f} cm/ms)")
    print(f"  Target CV:   {cv_target_long*10:.2f} m/s")
    print(f"  Error:       {error_pct:+.1f}%")

    return {
        'dx': dx,
        'nx': nx, 'ny': ny,
        'D_L': D_L, 'D_T': D_T,
        'cv_target': cv_target_long,
        'cv_measured': cv_measured,
        'error_pct': error_pct,
        'time': elapsed
    }


def run_calibration():
    """Run CV calibration at multiple mesh sizes."""

    print("="*60)
    print("CONDUCTION VELOCITY CALIBRATION")
    print("="*60)
    print("\nTarget: Human ventricular tissue")
    print("  CV_longitudinal = 0.6 m/s (0.06 cm/ms)")
    print("  CV_transverse = 0.2 m/s (0.02 cm/ms)")

    # Test mesh sizes from fine to coarse
    mesh_sizes = [0.005, 0.0075, 0.01, 0.015, 0.02]  # cm

    results = []
    for dx in mesh_sizes:
        try:
            result = measure_cv(dx)
            results.append(result)
        except Exception as e:
            print(f"ERROR at dx={dx}: {e}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'dx (um)':<10} {'Grid':<12} {'D_L':<12} {'CV (m/s)':<12} {'Error':<10}")
    print("-"*60)

    for r in results:
        print(f"{r['dx']*1000:<10.0f} {r['ny']}x{r['nx']:<8} {r['D_L']:<12.6f} "
              f"{r['cv_measured']*10:<12.2f} {r['error_pct']:+.1f}%")

    # Check if CV is consistent
    cv_values = [r['cv_measured'] for r in results]
    cv_mean = np.mean(cv_values)
    cv_std = np.std(cv_values)
    cv_range = max(cv_values) - min(cv_values)

    print("-"*60)
    print(f"CV range: {cv_range*10:.3f} m/s (std={cv_std*10:.3f} m/s)")

    if cv_range * 10 < 0.05:  # Within 5 cm/s
        print("PASS: CV is mesh-independent (within 5%)")
    else:
        print("NEEDS TUNING: CV varies significantly with mesh size")
        print("Adjust TAU_FOOT or correction factor in diffusion.py")

    return results


if __name__ == '__main__':
    results = run_calibration()
