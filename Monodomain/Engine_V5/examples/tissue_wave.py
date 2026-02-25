"""
Tissue Wave Propagation Example

Tests monodomain solver with planar wave propagation.
Validates conduction velocity against expected values.

Expected CV for human ventricular tissue:
- Longitudinal: ~0.5-0.7 m/s (0.05-0.07 cm/ms)
- Transverse: ~0.2-0.3 m/s (0.02-0.03 cm/ms)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from ionic import CellType
from tissue import MonodomainSimulation, estimate_cv_from_params


def run_planar_wave_test(
    nx: int = 100,
    ny: int = 50,
    dx: float = 0.01,  # 100 um
    cv_long: float = 0.06,  # Target CV: 0.6 m/s
    cv_trans: float = 0.02,  # Target CV: 0.2 m/s
    fiber_angle: float = 0.0,
    t_end: float = 100.0,
    dt: float = 0.02
):
    """
    Run planar wave propagation test.

    Stimulates left edge and measures conduction velocity.
    Uses CV-based parameter selection for mesh-independent results.
    """
    print("=" * 60)
    print("Planar Wave Propagation Test")
    print("=" * 60)

    # Target CV
    print(f"\nTarget CV:")
    print(f"  Longitudinal: {cv_long*10:.2f} m/s ({cv_long:.4f} cm/ms)")
    print(f"  Transverse: {cv_trans*10:.2f} m/s ({cv_trans:.4f} cm/ms)")

    # Create simulation with CV-based parameters
    print(f"\nGrid: {ny} x {nx} cells, dx = {dx*1000:.0f} um")
    print(f"Domain: {ny*dx:.2f} x {nx*dx:.2f} cm")

    sim = MonodomainSimulation(
        ny=ny, nx=nx,
        dx=dx, dy=dx,
        cv_long=cv_long,
        cv_trans=cv_trans,
        fiber_angle=fiber_angle,
        celltype=CellType.ENDO,
        splitting='godunov'
    )

    print(f"Computed D_L = {sim.D_L:.6f}, D_T = {sim.D_T:.6f} cm^2/ms")

    # Check stability
    dt_max = sim.diffusion.get_stability_limit()
    print(f"Stability limit: dt_max = {dt_max:.4f} ms")
    print(f"Using dt = {dt} ms")

    if dt > dt_max:
        print("WARNING: Time step exceeds stability limit!")
        dt = dt_max * 0.9
        print(f"Reducing to dt = {dt:.4f} ms")

    # Add stimulus at left edge (wider region, longer duration for reliable activation)
    stim_width = max(3, int(0.03 / dx))  # At least 3 cells or 0.3mm
    sim.add_stimulus(
        region=(slice(None), slice(0, stim_width)),
        start_time=1.0,
        duration=2.0
    )

    # Run simulation
    print(f"\nRunning simulation for {t_end} ms...")
    start_time = perf_counter()

    def progress(t, t_total):
        pct = 100 * t / t_total
        elapsed = perf_counter() - start_time
        eta = elapsed / max(t, 0.1) * (t_total - t)
        print(f"\r  t = {t:.1f} / {t_total:.0f} ms ({pct:.0f}%) ETA: {eta:.0f}s", end="")

    t, V = sim.run(t_end, dt=dt, save_interval=1.0, progress_callback=progress)
    print(f"\n  Completed in {perf_counter() - start_time:.1f} s")

    # Compute activation time
    print("\nComputing activation times...")
    act_time = sim.compute_activation_time(V, threshold=-40.0)

    # Compute CV
    cv_x = sim.compute_conduction_velocity(act_time, direction='x')
    cv_y = sim.compute_conduction_velocity(act_time, direction='y')

    print(f"\nMeasured conduction velocity:")
    print(f"  CV_x (along fibers): {cv_x*10:.2f} m/s ({cv_x:.4f} cm/ms)")
    if not np.isnan(cv_y):
        print(f"  CV_y (across fibers): {cv_y*10:.2f} m/s ({cv_y:.4f} cm/ms)")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Voltage snapshots
    times_to_show = [10, 30, 50]
    for idx, t_show in enumerate(times_to_show):
        t_idx = np.argmin(np.abs(t - t_show))
        ax = axes[0, idx]
        im = ax.imshow(V[t_idx], cmap='RdBu_r', vmin=-90, vmax=40,
                      origin='lower', aspect='equal')
        ax.set_title(f't = {t[t_idx]:.0f} ms')
        ax.set_xlabel('x (cells)')
        ax.set_ylabel('y (cells)')
        plt.colorbar(im, ax=ax, label='V (mV)')

    # Activation time map
    ax = axes[1, 0]
    im = ax.imshow(act_time, cmap='viridis', origin='lower', aspect='equal')
    ax.set_title('Activation Time')
    ax.set_xlabel('x (cells)')
    ax.set_ylabel('y (cells)')
    plt.colorbar(im, ax=ax, label='Time (ms)')

    # Voltage trace at center
    ax = axes[1, 1]
    mid_i, mid_j = ny // 2, nx // 2
    ax.plot(t, V[:, mid_i, mid_j], 'b-', lw=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title(f'Voltage at ({mid_i}, {mid_j})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_end])

    # CV profile
    ax = axes[1, 2]
    mid_row_act = act_time[ny // 2, :]
    x_coords = np.arange(nx) * dx
    valid = ~np.isnan(mid_row_act)
    ax.plot(mid_row_act[valid], x_coords[valid], 'b.-')
    ax.set_xlabel('Activation Time (ms)')
    ax.set_ylabel('Distance (cm)')
    ax.set_title(f'CV = {cv_x*10:.2f} m/s')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'planar_wave_test.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.show()

    return sim, t, V, act_time


def run_anisotropy_test():
    """
    Test anisotropic propagation with fibers at 45 degrees.
    """
    print("\n" + "=" * 60)
    print("Anisotropic Propagation Test (45 degree fibers)")
    print("=" * 60)

    nx, ny = 80, 80
    dx = 0.01
    cv_long = 0.06  # 0.6 m/s
    cv_trans = 0.02  # 0.2 m/s
    fiber_angle = np.pi / 4  # 45 degrees

    sim = MonodomainSimulation(
        ny=ny, nx=nx,
        dx=dx, dy=dx,
        cv_long=cv_long,
        cv_trans=cv_trans,
        fiber_angle=fiber_angle,
        celltype=CellType.ENDO
    )

    print(f"Target CV: long={cv_long*10:.1f} m/s, trans={cv_trans*10:.1f} m/s")
    print(f"Computed D_L = {sim.D_L:.6f}, D_T = {sim.D_T:.6f} cm^2/ms")

    # Point stimulus at center
    center = ny // 2
    sim.add_stimulus(
        region=(slice(center-2, center+2), slice(center-2, center+2)),
        start_time=1.0,
        duration=1.0
    )

    dt = sim.diffusion.get_stability_limit() * 0.8
    print(f"Using dt = {dt:.4f} ms")

    print("Running simulation...")
    start = perf_counter()
    t, V = sim.run(60.0, dt=dt, save_interval=2.0)
    print(f"Completed in {perf_counter() - start:.1f} s")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    times = [10, 20, 30, 40]
    for idx, t_show in enumerate(times):
        t_idx = np.argmin(np.abs(t - t_show))
        ax = axes[idx]
        im = ax.imshow(V[t_idx], cmap='RdBu_r', vmin=-90, vmax=40,
                      origin='lower', aspect='equal')
        ax.set_title(f't = {t[t_idx]:.0f} ms')

        # Draw fiber direction
        cx, cy = nx // 2, ny // 2
        length = 15
        ax.arrow(cx, cy,
                length * np.cos(fiber_angle),
                length * np.sin(fiber_angle),
                head_width=3, head_length=2, fc='green', ec='green')
        ax.text(cx + 5, cy - 5, 'fiber', color='green', fontsize=8)

        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'anisotropic_wave_test.png')
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {os.path.abspath(out_path)}")
    plt.show()


if __name__ == '__main__':
    # Run planar wave test with CV-based parameters
    sim, t, V, act_time = run_planar_wave_test(
        nx=100, ny=50,
        cv_long=0.06,  # Target 0.6 m/s
        cv_trans=0.02,  # Target 0.2 m/s
        t_end=80.0
    )

    # Optionally run anisotropy test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--aniso', action='store_true', help='Run anisotropy test')
    args, _ = parser.parse_known_args()

    if args.aniso:
        run_anisotropy_test()
