"""
Debug Border Speedup Effect
============================

This script investigates why there's no border speedup observed near infarct boundaries.

The expected physics:
- At a no-flux boundary, current cannot escape
- This effectively "reflects" the current back into the tissue
- The wavefront should speed up because it has less "load" to charge

The issue: Current implementation vs correct implementation of no-flux BC at internal interfaces.

Author: Generated with Claude Code
Date: 2025-12-10
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution

import numpy as np
import numba
from numba import jit, prange
import matplotlib.pyplot as plt
from parameters import default_fk_params, SpatialParams, PhysicalConstants
from fenton_karma import FentonKarmaModel


# =============================================================================
# Two Different Implementations of No-Flux at Internal Boundaries
# =============================================================================

@jit(nopython=True)
def diffusion_WRONG_noflux(u, mask, Dxx_dt_dx2, Dyy_dt_dy2):
    """
    CURRENT IMPLEMENTATION (no speedup).

    At infarct boundaries, replaces neighbor value with center value.
    This gives: lap = neighbor - center (one-sided difference)
    But no doubling factor!
    """
    ny, nx = u.shape
    u_new = np.copy(u)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if mask[i, j] < 0.5:
                continue

            m_left = mask[i, j-1]
            m_right = mask[i, j+1]
            m_down = mask[i-1, j]
            m_up = mask[i+1, j]

            # WRONG: Just mirror the value
            u_left = u[i, j-1] if m_left > 0.5 else u[i, j]
            u_right = u[i, j+1] if m_right > 0.5 else u[i, j]
            u_down = u[i-1, j] if m_down > 0.5 else u[i, j]
            u_up = u[i+1, j] if m_up > 0.5 else u[i, j]

            lap_x = u_right - 2.0 * u[i, j] + u_left
            lap_y = u_up - 2.0 * u[i, j] + u_down

            u_new[i, j] = u[i, j] + Dxx_dt_dx2 * lap_x + Dyy_dt_dy2 * lap_y

    return u_new


@jit(nopython=True)
def diffusion_CORRECT_noflux(u, mask, Dxx_dt_dx2, Dyy_dt_dy2):
    """
    CORRECT IMPLEMENTATION (with speedup).

    At infarct boundaries, use doubled gradient from interior side.
    This properly implements ∂u/∂n = 0 at the interface.

    Key insight: For Neumann BC, the ghost point mirrors the interior,
    so the second derivative becomes 2*(u_interior - u_boundary).
    """
    ny, nx = u.shape
    u_new = np.copy(u)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if mask[i, j] < 0.5:
                continue

            m_left = mask[i, j-1]
            m_right = mask[i, j+1]
            m_down = mask[i-1, j]
            m_up = mask[i+1, j]

            # Count valid neighbors
            n_valid_x = (1 if m_left > 0.5 else 0) + (1 if m_right > 0.5 else 0)
            n_valid_y = (1 if m_down > 0.5 else 0) + (1 if m_up > 0.5 else 0)

            # Compute laplacian with proper Neumann BC
            # Standard interior
            if n_valid_x == 2:
                lap_x = u[i, j+1] - 2.0 * u[i, j] + u[i, j-1]
            elif n_valid_x == 1:
                # One-sided Neumann: double the interior gradient
                if m_left > 0.5:
                    lap_x = 2.0 * (u[i, j-1] - u[i, j])  # Only left valid
                else:
                    lap_x = 2.0 * (u[i, j+1] - u[i, j])  # Only right valid
            else:
                lap_x = 0.0  # Both neighbors are infarct

            if n_valid_y == 2:
                lap_y = u[i+1, j] - 2.0 * u[i, j] + u[i-1, j]
            elif n_valid_y == 1:
                if m_down > 0.5:
                    lap_y = 2.0 * (u[i-1, j] - u[i, j])  # Only down valid
                else:
                    lap_y = 2.0 * (u[i+1, j] - u[i, j])  # Only up valid
            else:
                lap_y = 0.0  # Both neighbors are infarct

            u_new[i, j] = u[i, j] + Dxx_dt_dx2 * lap_x + Dyy_dt_dy2 * lap_y

    return u_new


# =============================================================================
# Simple 1D Test
# =============================================================================

def test_1d_speedup():
    """
    Simple 1D test: wave propagating along a channel with one boundary.

    Compare:
    - Interior propagation (symmetric diffusion)
    - Boundary propagation (should be faster with correct BC)
    """
    print("=" * 60)
    print("1D BOUNDARY SPEEDUP TEST")
    print("=" * 60)

    # Setup
    nx = 200
    dx = 0.5  # mm
    dt = 0.02  # ms
    D = 0.1  # mm²/ms
    D_dt_dx2 = D * dt / (dx * dx)

    print(f"Grid: {nx} points, dx={dx} mm")
    print(f"Diffusion: D={D} mm²/ms")
    print(f"CFL number: {D_dt_dx2:.4f}")

    # Two test cases:
    # 1. Interior: point at center, symmetric diffusion
    # 2. Boundary: point at left edge with no-flux BC

    u_interior = np.zeros(nx)
    u_boundary_wrong = np.zeros(nx)
    u_boundary_correct = np.zeros(nx)

    # Initial conditions: Gaussian at x=50
    x0_interior = 50
    x0_boundary = 10  # Near left edge
    sigma = 2.0

    x = np.arange(nx) * dx
    u_interior = np.exp(-((x - x0_interior*dx)**2) / (2*sigma**2))
    u_boundary_wrong = np.exp(-((x - x0_boundary*dx)**2) / (2*sigma**2))
    u_boundary_correct = np.exp(-((x - x0_boundary*dx)**2) / (2*sigma**2))

    # Simulate
    n_steps = 500

    for step in range(n_steps):
        # Interior: standard second derivative
        u_new = np.zeros_like(u_interior)
        for j in range(1, nx-1):
            lap = u_interior[j+1] - 2*u_interior[j] + u_interior[j-1]
            u_new[j] = u_interior[j] + D_dt_dx2 * lap
        u_interior = u_new

        # Boundary WRONG: left edge with simple mirroring
        u_new = np.zeros_like(u_boundary_wrong)
        for j in range(1, nx-1):
            lap = u_boundary_wrong[j+1] - 2*u_boundary_wrong[j] + u_boundary_wrong[j-1]
            u_new[j] = u_boundary_wrong[j] + D_dt_dx2 * lap
        # Left edge: wrong treatment (just set equal to neighbor)
        u_new[0] = u_boundary_wrong[0] + D_dt_dx2 * (u_boundary_wrong[1] - u_boundary_wrong[0])
        u_boundary_wrong = u_new

        # Boundary CORRECT: left edge with doubled gradient
        u_new = np.zeros_like(u_boundary_correct)
        for j in range(1, nx-1):
            lap = u_boundary_correct[j+1] - 2*u_boundary_correct[j] + u_boundary_correct[j-1]
            u_new[j] = u_boundary_correct[j] + D_dt_dx2 * lap
        # Left edge: correct Neumann BC (doubled gradient)
        u_new[0] = u_boundary_correct[0] + D_dt_dx2 * 2.0 * (u_boundary_correct[1] - u_boundary_correct[0])
        u_boundary_correct = u_new

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, u_interior, 'b-', linewidth=2, label='Interior (symmetric)')
    ax.plot(x, u_boundary_wrong, 'r--', linewidth=2, label='Boundary WRONG (no speedup)')
    ax.plot(x, u_boundary_correct, 'g-', linewidth=2, label='Boundary CORRECT (speedup)')
    ax.set_xlabel('x [mm]', fontsize=12)
    ax.set_ylabel('u', fontsize=12)
    ax.set_title(f'1D Diffusion after {n_steps*dt:.1f} ms', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/debug_1d_speedup.png', dpi=150)
    print(f"\nPlot saved: images/debug_1d_speedup.png")

    # Measure center of mass (proxy for wavefront position)
    x_com_interior = np.sum(x * u_interior) / np.sum(u_interior)
    x_com_wrong = np.sum(x * u_boundary_wrong) / np.sum(u_boundary_wrong)
    x_com_correct = np.sum(x * u_boundary_correct) / np.sum(u_boundary_correct)

    print(f"\nCenter of mass positions:")
    print(f"  Interior:        {x_com_interior:.2f} mm")
    print(f"  Boundary WRONG:  {x_com_wrong:.2f} mm")
    print(f"  Boundary CORRECT:{x_com_correct:.2f} mm")

    return fig


# =============================================================================
# 2D Comparison Test
# =============================================================================

def test_2d_speedup():
    """
    2D test: Compare wave propagation with different boundary treatments.
    """
    print("\n" + "=" * 60)
    print("2D BOUNDARY SPEEDUP TEST")
    print("=" * 60)

    # Setup
    nx, ny = 121, 121
    dx, dy = 0.5, 0.5
    dt = 0.02
    Dxx = 0.1  # mm²/ms
    Dyy = 0.05  # mm²/ms (anisotropic)

    Dxx_dt_dx2 = Dxx * dt / (dx * dx)
    Dyy_dt_dy2 = Dyy * dt / (dy * dy)

    print(f"Grid: {nx}×{ny}, dx={dx} mm")
    print(f"Diffusion: Dxx={Dxx}, Dyy={Dyy} mm²/ms")

    # Create domain with horizontal rectangle infarct in center
    mask = np.ones((ny, nx))
    infarct_y_start = ny//2 - 8  # 8 cells = 4mm half-height
    infarct_y_end = ny//2 + 8
    infarct_x_start = nx//2 - 40  # 40 cells = 20mm half-width
    infarct_x_end = nx//2 + 40
    mask[infarct_y_start:infarct_y_end, infarct_x_start:infarct_x_end] = 0

    print(f"Infarct: rows {infarct_y_start}-{infarct_y_end}, cols {infarct_x_start}-{infarct_x_end}")

    # Initialize with stimulus on left edge
    u_wrong = np.zeros((ny, nx))
    u_correct = np.zeros((ny, nx))

    # Planar wave stimulus on left
    u_wrong[:, :5] = 1.0
    u_correct[:, :5] = 1.0

    # Apply mask
    u_wrong *= mask
    u_correct *= mask

    # Simulate and record snapshots
    snapshots = []
    times = [0, 10, 20, 30, 40]  # ms
    step_times = [int(t/dt) for t in times]

    n_steps = int(50 / dt)  # 50 ms total

    for step in range(n_steps):
        u_wrong = diffusion_WRONG_noflux(u_wrong, mask, Dxx_dt_dx2, Dyy_dt_dy2)
        u_correct = diffusion_CORRECT_noflux(u_correct, mask, Dxx_dt_dx2, Dyy_dt_dy2)

        # Simple ionic model: threshold + decay
        # u_wrong = np.where(u_wrong > 0.13, np.minimum(u_wrong + 0.02, 1.0), u_wrong * 0.999)
        # u_correct = np.where(u_correct > 0.13, np.minimum(u_correct + 0.02, 1.0), u_correct * 0.999)

        u_wrong *= mask
        u_correct *= mask

        if step in step_times:
            snapshots.append((step * dt, u_wrong.copy(), u_correct.copy()))
            print(f"  t = {step*dt:.1f} ms")

    # Plot comparison
    n_snaps = len(snapshots)
    fig, axes = plt.subplots(2, n_snaps, figsize=(4*n_snaps, 8))

    extent = [0, nx*dx, 0, ny*dy]

    for i, (t, uw, uc) in enumerate(snapshots):
        # Mask out infarct for visualization
        uw_vis = np.ma.masked_where(mask < 0.5, uw)
        uc_vis = np.ma.masked_where(mask < 0.5, uc)

        axes[0, i].imshow(uw_vis, origin='lower', extent=extent, cmap='turbo', vmin=0, vmax=1)
        axes[0, i].set_title(f'WRONG BC\nt = {t:.0f} ms', fontsize=10)
        axes[0, i].set_xlabel('x [mm]')
        if i == 0:
            axes[0, i].set_ylabel('y [mm]')

        axes[1, i].imshow(uc_vis, origin='lower', extent=extent, cmap='turbo', vmin=0, vmax=1)
        axes[1, i].set_title(f'CORRECT BC\nt = {t:.0f} ms', fontsize=10)
        axes[1, i].set_xlabel('x [mm]')
        if i == 0:
            axes[1, i].set_ylabel('y [mm]')

        # Draw infarct boundary
        for ax in [axes[0, i], axes[1, i]]:
            rect = plt.Rectangle((infarct_x_start*dx, infarct_y_start*dy),
                                  (infarct_x_end-infarct_x_start)*dx,
                                  (infarct_y_end-infarct_y_start)*dy,
                                  fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

    plt.suptitle('2D Diffusion: Wrong vs Correct No-Flux BC at Infarct Border', fontsize=14)
    plt.tight_layout()
    plt.savefig('images/debug_2d_speedup.png', dpi=150)
    print(f"\nPlot saved: images/debug_2d_speedup.png")

    # Measure wavefront position along a line just above the infarct
    measure_y = infarct_y_end + 2  # 1mm above infarct
    print(f"\nMeasuring wavefront at y = {measure_y} (row {measure_y})")

    # Find 50% crossing point
    def find_wavefront(u_line, threshold=0.5):
        above = u_line > threshold
        if not np.any(above):
            return 0
        return np.argmax(above)

    wf_wrong = find_wavefront(snapshots[-1][1][measure_y, :])
    wf_correct = find_wavefront(snapshots[-1][2][measure_y, :])

    print(f"Wavefront position at t = {snapshots[-1][0]:.0f} ms:")
    print(f"  WRONG BC:   x = {wf_wrong * dx:.1f} mm (grid point {wf_wrong})")
    print(f"  CORRECT BC: x = {wf_correct * dx:.1f} mm (grid point {wf_correct})")
    print(f"  Difference: {(wf_correct - wf_wrong) * dx:.1f} mm")

    return fig


# =============================================================================
# Test with Full FK Model
# =============================================================================

def test_full_fk_speedup():
    """
    Test with full Fenton-Karma ionic model.
    """
    print("\n" + "=" * 60)
    print("FULL FK MODEL BORDER SPEEDUP TEST")
    print("=" * 60)

    # Setup
    nx, ny = 121, 121
    dx, dy = 0.5, 0.5
    dt = 0.02
    Dxx = 0.1
    Dyy = 0.05

    Dxx_dt_dx2 = Dxx * dt / (dx * dx)
    Dyy_dt_dy2 = Dyy * dt / (dy * dy)

    # Create FK model
    params = default_fk_params()
    model = FentonKarmaModel(params=params, dt=dt)

    # Create mask with horizontal rectangle infarct
    mask = np.ones((ny, nx))
    infarct_y_start = ny//2 - 8
    infarct_y_end = ny//2 + 8
    infarct_x_start = 20
    infarct_x_end = 100
    mask[infarct_y_start:infarct_y_end, infarct_x_start:infarct_x_end] = 0

    # Initialize two states
    state_wrong = model.initialize_state((ny, nx))
    state_correct = model.initialize_state((ny, nx))

    # Apply planar stimulus
    state_wrong['u'][:, :5] = 1.0
    state_correct['u'][:, :5] = 1.0

    # Apply mask
    for key in ['u', 'v', 'w']:
        state_wrong[key] *= mask
        state_correct[key] *= mask

    # Measure points
    measure_y_border = infarct_y_end + 2  # 1mm above infarct
    measure_y_free = infarct_y_end + 20   # 10mm above infarct
    measure_x = [30, 50, 70, 90]  # x positions to measure

    activation_times_wrong = {(x, measure_y_border): None for x in measure_x}
    activation_times_wrong.update({(x, measure_y_free): None for x in measure_x})
    activation_times_correct = {(x, measure_y_border): None for x in measure_x}
    activation_times_correct.update({(x, measure_y_free): None for x in measure_x})

    # Simulate
    n_steps = int(80 / dt)  # 80 ms
    threshold = 0.5

    print("Running simulation...")
    for step in range(n_steps):
        t = step * dt

        # Diffusion
        state_wrong['u'] = diffusion_WRONG_noflux(state_wrong['u'], mask, Dxx_dt_dx2, Dyy_dt_dy2)
        state_correct['u'] = diffusion_CORRECT_noflux(state_correct['u'], mask, Dxx_dt_dx2, Dyy_dt_dy2)

        # Ionic
        I_stim = np.zeros((ny, nx))
        model.ionic_step(state_wrong, I_stim)
        model.ionic_step(state_correct, I_stim)

        # Apply mask
        for key in ['u', 'v', 'w']:
            state_wrong[key] *= mask
            state_correct[key] *= mask

        # Record activation times
        for (x, y), t_act in activation_times_wrong.items():
            if t_act is None and state_wrong['u'][y, x] > threshold:
                activation_times_wrong[(x, y)] = t

        for (x, y), t_act in activation_times_correct.items():
            if t_act is None and state_correct['u'][y, x] > threshold:
                activation_times_correct[(x, y)] = t

    # Print results
    print("\nActivation Times:")
    print(f"{'Position':<15} {'WRONG BC':<12} {'CORRECT BC':<12} {'Difference':<10}")
    print("-" * 50)

    for x in measure_x:
        key_border = (x, measure_y_border)
        key_free = (x, measure_y_free)

        t_border_wrong = activation_times_wrong[key_border]
        t_border_correct = activation_times_correct[key_border]
        t_free_wrong = activation_times_wrong[key_free]
        t_free_correct = activation_times_correct[key_free]

        if t_border_wrong and t_border_correct:
            diff_border = t_border_correct - t_border_wrong
            print(f"Border x={x*dx:.0f}mm:  {t_border_wrong:.1f} ms    {t_border_correct:.1f} ms    {diff_border:+.1f} ms")

        if t_free_wrong and t_free_correct:
            diff_free = t_free_correct - t_free_wrong
            print(f"Free x={x*dx:.0f}mm:    {t_free_wrong:.1f} ms    {t_free_correct:.1f} ms    {diff_free:+.1f} ms")

    # Compute CV
    print("\nConduction Velocities:")

    for label, times_dict in [("WRONG BC", activation_times_wrong), ("CORRECT BC", activation_times_correct)]:
        t1_border = times_dict[(measure_x[0], measure_y_border)]
        t2_border = times_dict[(measure_x[-1], measure_y_border)]
        t1_free = times_dict[(measure_x[0], measure_y_free)]
        t2_free = times_dict[(measure_x[-1], measure_y_free)]

        if t1_border and t2_border:
            dist = (measure_x[-1] - measure_x[0]) * dx
            cv_border = dist / (t2_border - t1_border)
            print(f"  {label} - Near border: CV = {cv_border:.2f} mm/ms = {cv_border*1000:.0f} mm/s")

        if t1_free and t2_free:
            dist = (measure_x[-1] - measure_x[0]) * dx
            cv_free = dist / (t2_free - t1_free)
            print(f"  {label} - Free tissue: CV = {cv_free:.2f} mm/ms = {cv_free*1000:.0f} mm/s")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING BORDER SPEEDUP EFFECT")
    print("=" * 60)
    print("""
The issue: No-flux boundary conditions at internal interfaces (infarct borders)
are not being applied correctly.

WRONG implementation:
    u_neighbor = u_center if neighbor is infarct
    lap = u_right - 2*u_center + u_left
    -> At boundary: lap = u_interior - u_center (single-sided)

CORRECT implementation:
    At boundary: lap = 2.0 * (u_interior - u_center)
    -> Doubled gradient to properly reflect the BC

This is what's done for domain boundaries but NOT for infarct boundaries!
""")

    # Run tests
    test_1d_speedup()
    test_2d_speedup()
    test_full_fk_speedup()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The lack of border speedup is due to incorrect implementation of Neumann BC
at internal interfaces. The fix is to use doubled gradients when a neighbor
is an infarct cell, matching the treatment used for domain boundaries.

To fix infarct_border_speedup.py and spiral_wave_sim.py:
Change from:
    u_right = u[i, j+1] if m_right > 0.5 else u[i, j]
    lap_x = u_right - 2.0 * u[i, j] + u_left

To:
    if n_valid_x == 2:
        lap_x = u[i,j+1] - 2*u[i,j] + u[i,j-1]
    elif n_valid_x == 1:
        if m_left > 0.5:
            lap_x = 2.0 * (u[i,j-1] - u[i,j])
        else:
            lap_x = 2.0 * (u[i,j+1] - u[i,j])
    else:
        lap_x = 0.0
""")

    plt.show()
