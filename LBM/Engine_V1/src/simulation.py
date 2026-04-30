"""LBM Simulation orchestrator.

Coordinates the full simulation loop:
1. Ionic source computation
2. LBM step (collision + streaming + BC)
3. Ionic state update
4. Stimulus delivery
5. Output recording

Layer 1: stateful class wrapping Layer 2 pure functions.
"""

import torch
from torch import Tensor
from typing import Optional, List

from .lattice import D2Q5, D2Q9
from .diffusion import tau_from_D
from .state import create_lbm_state, recover_voltage
from .step import lbm_step_d2q5_bgk, lbm_step_d2q9_bgk
from .solver.rush_larsen import ionic_step, compute_source_term


class Stimulus:
    """Stimulus protocol specification."""
    def __init__(self, mask: Tensor, start: float, duration: float,
                 amplitude: float):
        self.mask = mask          # (Nx, Ny) bool
        self.start = start        # ms
        self.duration = duration  # ms
        self.amplitude = amplitude  # pA/pF


class LBMSimulation:
    """Complete LBM cardiac simulation.

    Args:
        Nx, Ny: grid dimensions
        dx: spatial resolution (cm)
        dt: time step (ms)
        D: diffusion coefficient (cm^2/ms)
        ionic_model: IonicModel instance (e.g., TTP06Model)
        Cm: membrane capacitance (uF/cm^2), default 1.0. Used for source
            term scaling: R = -(I_ion + I_stim) / Cm
        lattice: 'd2q5' or 'd2q9' (default 'd2q5')
        bounce_masks: dict of boundary masks (if None, uses full-grid rectangular)
    """

    def __init__(self, Nx: int, Ny: int, dx: float, dt: float,
                 D: float, ionic_model, Cm: float = 1.0,
                 lattice: str = 'd2q5', bounce_masks: dict = None):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dt = dt
        self.D = D
        self.ionic_model = ionic_model
        self.Cm = Cm
        self.device = ionic_model.device
        self.dtype = ionic_model.dtype

        # Lattice setup
        if lattice == 'd2q5':
            self.lattice = D2Q5()
            self._step_fn = lbm_step_d2q5_bgk
        elif lattice == 'd2q9':
            self.lattice = D2Q9()
            self._step_fn = lbm_step_d2q9_bgk
        else:
            raise ValueError(f"Unknown lattice: {lattice}")

        self.w = torch.tensor(self.lattice.w, device=self.device, dtype=self.dtype)
        tau = tau_from_D(D, dx, dt)
        self.omega = 1.0 / tau

        # Boundary masks
        if bounce_masks is not None:
            self.bounce_masks = bounce_masks
        else:
            self.bounce_masks = self._make_rect_masks()

        # State
        n_cells = Nx * Ny
        V_init = torch.full((Nx, Ny), ionic_model.V_rest,
                            device=self.device, dtype=self.dtype)
        self.f = self.w[:, None, None] * V_init[None, :, :]
        self.V = V_init
        self.ionic_states = ionic_model.get_initial_state(n_cells=n_cells)
        self.t = 0.0

        # Stimuli
        self.stimuli: List[Stimulus] = []

    def _make_rect_masks(self) -> dict:
        """Create bounce masks for rectangular full-grid domain."""
        masks = {}
        for a in range(1, self.lattice.Q):
            m = torch.zeros(self.Nx, self.Ny, dtype=torch.bool,
                           device=self.device)
            ex, ey = self.lattice.e[a]
            if ex == 1:   m[-1, :] = True
            if ex == -1:  m[0, :] = True
            if ey == 1:   m[:, -1] = True
            if ey == -1:  m[:, 0] = True
            masks[a] = m
        return masks

    def add_stimulus(self, mask: Tensor, start: float, duration: float,
                     amplitude: float = -80.0):
        """Add a stimulus protocol."""
        self.stimuli.append(Stimulus(mask, start, duration, amplitude))

    def _get_I_stim(self) -> Tensor:
        """Compute current stimulus at time self.t."""
        I_stim = torch.zeros(self.Nx, self.Ny, device=self.device,
                             dtype=self.dtype)
        for s in self.stimuli:
            if s.start <= self.t < s.start + s.duration:
                I_stim[s.mask] = s.amplitude
        return I_stim

    def step(self):
        """Advance simulation by one time step."""
        I_stim = self._get_I_stim()
        V_flat = self.V.reshape(-1)
        I_stim_flat = I_stim.reshape(-1)

        # 1. Compute ionic source (I_ion computed once, reused below)
        I_ion = self.ionic_model.compute_Iion(V_flat, self.ionic_states)
        R_flat = compute_source_term(I_ion, I_stim_flat, self.Cm)
        R = R_flat.reshape(self.Nx, self.Ny)

        # 2-6. LBM step (collide -> stream -> BC -> recover V)
        self.f, self.V = self._step_fn(
            self.f, self.V, R, self.dt, self.omega, self.w,
            self.bounce_masks
        )

        # 7. Update ionic states only (V comes from distributions, not ionic ODE)
        V_flat = self.V.reshape(-1)
        self.ionic_states = ionic_step(
            self.ionic_model, V_flat, self.ionic_states, self.dt
        )

        # 8. Advance time
        self.t += self.dt

    def run(self, t_end: float, save_every: float = 1.0):
        """Run simulation to t_end, saving V snapshots.

        Returns:
            times: list of saved time points
            V_history: list of V snapshots (each Nx x Ny)
        """
        save_interval = max(1, int(save_every / self.dt))
        times = []
        V_history = []
        step_count = 0

        while self.t < t_end - self.dt / 2:
            self.step()
            step_count += 1
            if step_count % save_interval == 0:
                times.append(self.t)
                V_history.append(self.V.clone())

        return times, V_history

    def get_activation_times(self, threshold: float = -30.0) -> Tensor:
        """Compute activation time at each node (first time V > threshold).

        Must be called during run via callback or by post-processing V_history.
        This is a post-processing utility, not a live tracker.
        """
        raise NotImplementedError("Use run() output to compute activation times")


def measure_cv(V_history: list, times: list, x1: int, x2: int,
               y: int, dx: float, threshold: float = -30.0) -> float:
    """Measure conduction velocity between two x-positions at row y.

    Args:
        V_history: list of (Nx, Ny) voltage snapshots
        times: list of corresponding time points
        x1, x2: x-positions to measure between (x2 > x1)
        y: y-position (row)
        dx: spatial resolution (cm)
        threshold: activation threshold (mV)

    Returns:
        cv: conduction velocity (cm/ms)
    """
    t1 = None
    t2 = None
    for V, t in zip(V_history, times):
        if t1 is None and V[x1, y].item() > threshold:
            t1 = t
        if t2 is None and V[x2, y].item() > threshold:
            t2 = t
        if t1 is not None and t2 is not None:
            break

    if t1 is None or t2 is None or t2 <= t1:
        return float('nan')

    distance = abs(x2 - x1) * dx
    return distance / (t2 - t1)
