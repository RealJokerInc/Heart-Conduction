"""
Cardiac Wave Propagation - Flat Domain with Pulse Trains
=========================================================

Uniform right-aligning fibers with pulse train stimulation capability.

Features:
- No infarct (flat domain)
- Uniform fiber orientation (rightward)
- Pulse train stimulation (multiple pulses)
- Numba-accelerated
- Fixed diffusion computation (like V2)

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
from parameters import create_default_parameters
from aliev_panfilov_fixed import AlievPanfilovModel


# =============================================================================
# Numba-Accelerated Kernels (from V2)
# =============================================================================

@numba.jit(nopython=True, parallel=False, cache=True)
def ionic_step_numba(V, w, dt, I_stim, k, a, epsilon0, mu1, mu2,
                     epsilon_rest, V_threshold, k_sigmoid):
    """
    Numba-accelerated ionic model step.

    No tissue mask needed - entire domain is healthy tissue.
    """
    ny, nx = V.shape

    for i in range(ny):
        for j in range(nx):
            v = V[i, j]
            w_val = w[i, j]
            stim = I_stim[i, j]

            # Ionic current
            I_ion = k * v * (v - a) * (v - 1.0) + v * w_val

            # Voltage-dependent epsilon (with safeguard for division)
            # Ensure v + mu2 > 0 to avoid division by zero
            v_safe = max(v, -mu2 + 0.01)
            sigmoid_rest = 1.0 / (1.0 + np.exp(k_sigmoid * (v_safe - V_threshold)))
            eps = epsilon0 + epsilon_rest * sigmoid_rest + (mu1 * w_val) / (v_safe + mu2)

            # Update equations
            dVdt = -I_ion + stim
            dwdt = eps * (-k * v * (v - a - 1.0) - w_val)

            # Apply updates with explicit Euler
            v_new = v + dt * dVdt
            w_new = w_val + dt * dwdt

            # FIX #2 (NEW): Clamp V to [0, 1] range (Aliev-Panfilov model bounds)
            # This prevents both overshoot (V > 1) and undershoot (V < 0)
            V[i, j] = max(0.0, min(v_new, 1.0))
            w[i, j] = max(w_new, 0.0)  # Recovery variable stays non-negative


def compute_diffusion_flux_based(V, Dxx, Dyy, Dxy, dx, dy):
    """
    Compute diffusion term using flux-based approach.

    Fixed version: Uses one-sided differences at boundaries.
    """
    ny, nx = V.shape

    # Initialize gradients
    dVdx = np.zeros_like(V)
    dVdy = np.zeros_like(V)

    # Gradient in x direction (along columns, axis=1)
    # Interior points (centered difference)
    dVdx[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2.0 * dx)

    # Boundaries (one-sided for Neumann BC)
    dVdx[:, 0] = (V[:, 1] - V[:, 0]) / dx  # Left edge
    dVdx[:, -1] = (V[:, -1] - V[:, -2]) / dx  # Right edge

    # Gradient in y direction (along rows, axis=0)
    # Interior points (centered difference)
    dVdy[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2.0 * dy)

    # Boundaries (one-sided for Neumann BC)
    dVdy[0, :] = (V[1, :] - V[0, :]) / dy  # Bottom edge
    dVdy[-1, :] = (V[-1, :] - V[-2, :]) / dy  # Top edge

    # Compute flux components: J = D * grad(V)
    Jx = Dxx * dVdx + Dxy * dVdy
    Jy = Dxy * dVdx + Dyy * dVdy

    # Divergence of flux
    div_J = np.zeros_like(V)

    # ∂Jx/∂x (along columns, axis=1)
    div_J[:, 1:-1] += (Jx[:, 2:] - Jx[:, :-2]) / (2.0 * dx)
    div_J[:, 0] += (Jx[:, 1] - Jx[:, 0]) / dx
    div_J[:, -1] += (Jx[:, -1] - Jx[:, -2]) / dx

    # ∂Jy/∂y (along rows, axis=0)
    div_J[1:-1, :] += (Jy[2:, :] - Jy[:-2, :]) / (2.0 * dy)
    div_J[0, :] += (Jy[1, :] - Jy[0, :]) / dy
    div_J[-1, :] += (Jy[-1, :] - Jy[-2, :]) / dy

    return div_J


# =============================================================================
# Main Simulation Class
# =============================================================================

class FlatSimulation:
    """
    Flat domain cardiac simulation with uniform fibers and pulse trains.
    """

    def __init__(
        self,
        domain_size=80.0,
        resolution=0.5,
        D_parallel=1.0,
        D_perp=0.5,
        T_scale=10.0,
        fiber_angle=0.0  # degrees, 0 = rightward
    ):
        """Initialize flat simulation."""

        print("=" * 70)
        print("FLAT DOMAIN SIMULATION - Uniform Fibers + Pulse Trains")
        print("=" * 70)

        # Create mesh
        self.Lx = domain_size
        self.Ly = domain_size
        self.dx = resolution
        self.dy = resolution
        self.nx = int(np.round(self.Lx / self.dx)) + 1
        self.ny = int(np.round(self.Ly / self.dy)) + 1

        print(f"\nDomain: {self.Lx} × {self.Ly} mm")
        print(f"Grid: {self.nx} × {self.ny} points")
        print(f"Resolution: {self.dx} mm")

        # Uniform fiber field (all pointing in same direction)
        self.fiber_angle = fiber_angle
        theta = np.radians(fiber_angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Diffusion tensor components (constant everywhere)
        # D = R @ diag(D_parallel, D_perp) @ R^T
        self.Dxx = D_parallel * cos_t**2 + D_perp * sin_t**2
        self.Dyy = D_parallel * sin_t**2 + D_perp * cos_t**2
        self.Dxy = (D_parallel - D_perp) * cos_t * sin_t

        # Broadcast to full arrays
        self.Dxx = np.full((self.ny, self.nx), self.Dxx)
        self.Dyy = np.full((self.ny, self.nx), self.Dyy)
        self.Dxy = np.full((self.ny, self.nx), self.Dxy)

        print(f"\nFiber orientation: {fiber_angle}° (rightward)")
        print(f"Diffusion: D_parallel={D_parallel}, D_perp={D_perp}")
        print(f"  Dxx={self.Dxx[0,0]:.3f}, Dyy={self.Dyy[0,0]:.3f}, Dxy={self.Dxy[0,0]:.3f}")

        # Ionic model
        params = create_default_parameters()
        params.ionic.epsilon_rest = 0.05
        self.ionic_model = AlievPanfilovModel(params.ionic, params.physical, T_scale=T_scale)

        # Store ionic parameters for Numba
        self.k = params.ionic.k
        self.a = params.ionic.a
        self.epsilon0 = params.ionic.epsilon0
        self.mu1 = params.ionic.mu1
        self.mu2 = params.ionic.mu2
        self.epsilon_rest = params.ionic.epsilon_rest
        self.V_threshold = params.ionic.V_threshold
        self.k_sigmoid = params.ionic.k_sigmoid

        # State variables
        self.V = np.zeros((self.ny, self.nx))
        self.w = np.zeros((self.ny, self.nx))

        # Stimulus mask (left edge by default)
        self.stim_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.stim_mask[:, 0] = True

        print(f"Stimulus points: {np.sum(self.stim_mask)} (left edge)")
        print(f"\n✓ Flat simulation ready!")
        print(f"  • Uniform {fiber_angle}° fibers")
        print(f"  • Pulse train capability")
        print(f"  • Numba acceleration")

    def step(self, dt, I_stim):
        """
        Single time step with operator splitting.

        1. Diffusion
        2. Reaction (ionic model)
        """
        # Step 1: Diffusion
        div_J = compute_diffusion_flux_based(self.V, self.Dxx, self.Dyy, self.Dxy, self.dx, self.dy)
        self.V += dt * div_J

        # Step 2: Reaction (Numba-accelerated)
        dtau = self.ionic_model.time_to_tau(dt)

        ionic_step_numba(
            self.V, self.w, dtau, I_stim,
            self.k, self.a, self.epsilon0, self.mu1, self.mu2,
            self.epsilon_rest, self.V_threshold, self.k_sigmoid
        )

    def create_pulse_train(
        self,
        amplitude=30.0,
        pulse_duration=2.0,
        start_times=[5.0, 200.0, 400.0],
        location='left'  # 'left', 'center', or custom mask
    ):
        """
        Create pulse train stimulus function.

        Parameters
        ----------
        amplitude : float
            Stimulus amplitude in mV
        pulse_duration : float
            Duration of each pulse in ms
        start_times : list of float
            Start time of each pulse in ms
        location : str or ndarray
            'left' = left edge (default)
            'center' = 10mm radius circle at center
            ndarray = custom boolean mask

        Returns
        -------
        stim_func : callable
            Stimulus function: stim_func(t) -> I_stim array
        """
        V_range = self.ionic_model.V_range
        amplitude_norm = amplitude / V_range

        # Determine stimulus location
        if isinstance(location, np.ndarray):
            stim_mask = location
        elif location == 'left':
            stim_mask = np.zeros((self.ny, self.nx), dtype=bool)
            stim_mask[:, 0] = True
        elif location == 'center':
            # 10mm radius circle at center
            center_x, center_y = self.Lx / 2, self.Ly / 2
            x = np.linspace(0, self.Lx, self.nx)
            y = np.linspace(0, self.Ly, self.ny)
            X, Y = np.meshgrid(x, y)
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            stim_mask = dist <= 10.0
        else:
            raise ValueError(f"Unknown location: {location}")

        def stim_func(t):
            I_stim = np.zeros((self.ny, self.nx))
            # Check if t is within any pulse window
            for t_start in start_times:
                if t_start <= t < t_start + pulse_duration:
                    I_stim[stim_mask] = amplitude_norm
                    break
            return I_stim

        print(f"\nPulse train created:")
        print(f"  Amplitude: {amplitude} mV (normalized: {amplitude_norm:.4f})")
        print(f"  Duration: {pulse_duration} ms")
        print(f"  Pulses: {len(start_times)}")
        print(f"  Times: {start_times}")
        print(f"  Location: {location} ({np.sum(stim_mask)} points)")

        return stim_func

    def simulate(
        self,
        t_end=400.0,
        dt=0.005,  # FIX: reduced from 0.01 for stability
        stim_func=None,
        save_every_ms=2.0,
        verbose=True
    ):
        """Run simulation."""
        n_steps = int(np.ceil(t_end / dt))
        save_every = max(1, int(np.round(save_every_ms / dt)))
        n_saves = (n_steps // save_every) + 1

        times = np.zeros(n_saves)
        V_history = np.zeros((n_saves, self.ny, self.nx))

        times[0] = 0.0
        V_history[0] = self.V.copy()
        save_idx = 1

        if verbose:
            print(f"\nRunning simulation:")
            print(f"  Duration: {t_end} ms")
            print(f"  Time step: {dt} ms")
            print(f"  Steps: {n_steps:,}")
            print(f"  Saves: {n_saves}")

        for step in range(n_steps):
            t = step * dt

            # Get stimulus
            if stim_func is not None:
                I_stim = stim_func(t)
            else:
                I_stim = np.zeros((self.ny, self.nx))

            # Step forward
            self.step(dt, I_stim)

            # Save
            if (step + 1) % save_every == 0 and save_idx < n_saves:
                times[save_idx] = t + dt
                V_history[save_idx] = self.V.copy()
                save_idx += 1

                if verbose and save_idx % 20 == 0:
                    V_max = np.max(self.V)
                    V_right = np.max(self.V[:, -1])  # Right edge
                    print(f"    t = {t+dt:.1f} ms | V_max = {V_max:.4f} | V_right_edge = {V_right:.4f}")

        if verbose:
            print(f"  ✓ Complete! Saved {save_idx} snapshots")

        return times[:save_idx], V_history[:save_idx]

    def animate(
        self,
        times,
        V_history,
        skip_frames=5,
        interval=30,
        save_path=None
    ):
        """Create animation."""
        V_hist_mV = self.ionic_model.voltage_to_physical(V_history)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            V_hist_mV[0],
            origin='lower',
            extent=[0, self.Lx, 0, self.Ly],
            cmap='turbo',
            vmin=self.ionic_model.V_rest,
            vmax=self.ionic_model.V_peak,
            aspect='equal'
        )

        plt.colorbar(im, ax=ax, label='Voltage (mV)', fraction=0.046)

        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        title = ax.set_title(f't = {times[0]:.1f} ms', fontsize=14, fontweight='bold')

        def update(frame_idx):
            frame_idx = frame_idx * skip_frames
            if frame_idx >= len(times):
                frame_idx = len(times) - 1

            im.set_data(V_hist_mV[frame_idx])
            title.set_text(f't = {times[frame_idx]:.1f} ms')
            return [im, title]

        n_frames = len(times) // skip_frames
        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=True,
            repeat=True
        )

        if save_path:
            print(f"\nSaving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=30, dpi=100)
            print("✓ Animation saved!")

        return fig, anim


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    import time

    # Create simulation
    sim = FlatSimulation(
        domain_size=80.0,
        resolution=0.5,
        D_parallel=1.0,
        D_perp=0.5,
        T_scale=10.0,
        fiber_angle=0.0  # Rightward
    )

    # Create pulse train (3 pulses)
    stim_func = sim.create_pulse_train(
        amplitude=30.0,
        pulse_duration=2.0,
        start_times=[5.0, 300.0, 600.0],  # S1, S2, S3
        location='left'
    )

    # Run simulation
    print("\n" + "=" * 70)
    print("RUNNING PULSE TRAIN SIMULATION")
    print("=" * 70)

    start_time = time.time()

    times, V_hist = sim.simulate(
        t_end=800.0,
        dt=0.005,  # FIX: reduced for stability
        stim_func=stim_func,
        save_every_ms=2.0,
        verbose=True
    )

    elapsed = time.time() - start_time
    print(f"\n✓ Simulation complete in {elapsed:.2f} seconds")
    print(f"  Performance: {800.0/elapsed:.2f} ms/sec simulated")

    # Animate
    print("\n" + "=" * 70)
    print("CREATING ANIMATION")
    print("=" * 70)

    fig, anim = sim.animate(times, V_hist, skip_frames=5, save_path='flat_pulse_train.gif')

    plt.show()
