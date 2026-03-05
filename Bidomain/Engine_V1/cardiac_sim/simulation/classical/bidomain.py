"""
BidomainSimulation — Top-Level Orchestrator

Builds state, solvers, and splitting strategy from config strings.
Follows V5.4's MonodomainSimulation pattern.

Ref: improvement.md L1686-1766 (runtime step spec)
Ref: improvement.md L1943-2018 (user API)
"""

import torch
from typing import Optional, Union

from .state import BidomainState
from .discretization.base import BidomainSpatialDiscretization


class BidomainSimulation:
    """
    Top-level orchestrator for bidomain cardiac simulations.

    Parameters
    ----------
    spatial : BidomainSpatialDiscretization
        FDM discretization (provides stencils, grid, conductivity)
    ionic_model : str or IonicModel
        Ionic model name ('ttp06', 'ord') or instance
    stimulus : StimulusProtocol or None
        Stimulus events
    dt : float
        Time step (ms)
    splitting : str
        'strang' or 'godunov'
    ionic_solver : str
        'rush_larsen' or 'forward_euler'
    diffusion_solver : str
        'decoupled' (only option currently)
    parabolic_solver : str
        'pcg', 'chebyshev', or 'spectral'
    elliptic_solver : str
        'auto', 'spectral', 'pcg_spectral', 'pcg', or 'pcg_gmg'
    theta : float
        Implicitness (0.5 = CN, 1.0 = BDF1)
    """

    def __init__(self, spatial, ionic_model, stimulus=None, dt=0.02,
                 splitting='strang', ionic_solver='rush_larsen',
                 diffusion_solver='decoupled', parabolic_solver='pcg',
                 elliptic_solver='auto', theta=0.5, device=None):

        self.dt = dt
        self._spatial = spatial

        # Resolve device from spatial discretization if not specified
        if device is None:
            device = spatial.grid.device
        self.device = device

        # 1. Resolve ionic model
        model = _resolve_ionic_model(ionic_model, device=device)

        # 2. Auto-select elliptic solver
        if elliptic_solver == 'auto':
            elliptic_solver = self._auto_select_elliptic_solver(spatial)
        self._elliptic_solver_name = elliptic_solver

        # 3. Build solver chain
        para_ls = _build_linear_solver(parabolic_solver, spatial)
        ellip_ls = _build_linear_solver(elliptic_solver, spatial)
        diffusion = _build_diffusion_solver(
            diffusion_solver, spatial, dt, para_ls, ellip_ls, theta)
        ionic = _build_ionic_solver(ionic_solver, model)
        self.splitting = _build_splitting(splitting, ionic, diffusion)

        # 4. Build state
        self.state = _build_state(spatial, model, stimulus, device=device)

    @staticmethod
    def _auto_select_elliptic_solver(spatial):
        """Read BoundarySpec from mesh and pick best solver tier."""
        bc = spatial.grid.boundary_spec
        cond = spatial._conductivity

        if bc.phi_e_spectral_eligible and cond.is_isotropic:
            return 'spectral'
        elif bc.phi_e_spectral_eligible:
            return 'pcg_spectral'
        else:
            return 'pcg_gmg'

    def run(self, t_end, save_every=1.0, callback=None):
        """
        Run simulation, yielding state at save points.

        Parameters
        ----------
        t_end : float
            End time (ms)
        save_every : float
            Save interval (ms)
        callback : callable, optional
            Called with state at each save point. Return True to stop.

        Yields
        ------
        state : BidomainState
        """
        state = self.state
        dt = self.dt
        next_save = save_every

        while state.t < t_end - 1e-12:
            self.splitting.step(state, dt)
            state.t += dt

            if state.t >= next_save - 1e-12:
                next_save += save_every
                if callback and callback(state):
                    break
                yield state


# === Factory Functions ===

def _resolve_ionic_model(ionic_model, device=None):
    """Resolve ionic model from string or instance."""
    if device is None:
        device = torch.device('cpu')
    dev_str = str(device) if not isinstance(device, str) else device
    if isinstance(ionic_model, str):
        name = ionic_model.lower()
        if name == 'ttp06':
            from ...ionic import TTP06Model
            return TTP06Model(device=dev_str)
        elif name == 'ord':
            from ...ionic import ORdModel
            return ORdModel(device=dev_str)
        else:
            raise ValueError(f"Unknown ionic model: {ionic_model}")
    return ionic_model


def _build_ionic_solver(name, ionic_model):
    """Build IonicSolver from string."""
    if name == 'rush_larsen':
        from .solver.ionic_stepping.rush_larsen import RushLarsenSolver
        return RushLarsenSolver(ionic_model)
    elif name == 'forward_euler':
        from .solver.ionic_stepping.base import IonicSolver
        # ForwardEuler is the base class default behavior
        # Create a concrete subclass
        class ForwardEulerSolver(IonicSolver):
            def step(self, state, dt):
                model = self.ionic_model
                V = state.V
                S = state.ionic_states
                Iion = model.compute_Iion(V, S)
                Istim = self._evaluate_Istim(state)
                state.V = V + dt * (-(Iion + Istim))
                gate_inf = model.compute_gate_steady_states(V, S)
                gate_tau = model.compute_gate_time_constants(V, S)
                self._update_gates(S, gate_inf, gate_tau, dt)
                conc_rates = model.compute_concentration_rates(V, S)
                for i, idx in enumerate(model.concentration_indices):
                    S[:, idx] = S[:, idx] + dt * conc_rates[:, i]
        return ForwardEulerSolver(ionic_model)
    else:
        raise ValueError(f"Unknown ionic solver: {name}")


def _build_linear_solver(name, spatial):
    """Build LinearSolver from string."""
    if name == 'pcg':
        from .solver.linear_solver.pcg import PCGSolver
        return PCGSolver(max_iters=500, tol=1e-8)
    elif name == 'chebyshev':
        from .solver.linear_solver.chebyshev import ChebyshevSolver
        return ChebyshevSolver()
    elif name == 'spectral':
        from .solver.linear_solver.spectral import SpectralSolver
        grid = spatial.grid
        nx, ny = grid.Nx, grid.Ny
        dx = grid.Lx / (nx - 1)
        dy = grid.Ly / (ny - 1)
        D = spatial._conductivity.D_i + spatial._conductivity.D_e
        bc = grid.boundary_spec
        # spectral_transform returns 'dct'/'dst', but SpectralSolver expects
        # physics names 'neumann'/'dirichlet'
        transform_to_bc = {'dct': 'neumann', 'dst': 'dirichlet', 'fft': 'periodic'}
        bc_type = transform_to_bc[bc.spectral_transform]
        return SpectralSolver(nx, ny, dx, dy, D, bc_type=bc_type)
    elif name == 'pcg_spectral':
        from .solver.linear_solver.pcg_spectral import PCGSpectralSolver
        grid = spatial.grid
        nx, ny = grid.Nx, grid.Ny
        dx = grid.Lx / (nx - 1)
        dy = grid.Ly / (ny - 1)
        D = spatial._conductivity.D_i + spatial._conductivity.D_e
        bc = grid.boundary_spec
        transform_to_bc = {'dct': 'neumann', 'dst': 'dirichlet', 'fft': 'periodic'}
        bc_type = transform_to_bc[bc.spectral_transform]
        return PCGSpectralSolver(nx, ny, dx, dy, D, bc_type=bc_type)
    elif name == 'pcg_gmg':
        # Stub — falls back to PCG
        from .solver.linear_solver.pcg import PCGSolver
        return PCGSolver(max_iters=500, tol=1e-8)
    else:
        raise ValueError(f"Unknown linear solver: {name}")


def _build_diffusion_solver(name, spatial, dt, para_ls, ellip_ls, theta):
    """Build BidomainDiffusionSolver from string."""
    if name == 'decoupled':
        from .solver.diffusion_stepping.decoupled import DecoupledBidomainDiffusionSolver
        return DecoupledBidomainDiffusionSolver(
            spatial, dt, para_ls, ellip_ls, theta=theta)
    else:
        raise ValueError(f"Unknown diffusion solver: {name}")


def _build_splitting(name, ionic_solver, diffusion_solver):
    """Build SplittingStrategy from string."""
    if name == 'strang':
        from .solver.splitting.strang import StrangSplitting
        return StrangSplitting(ionic_solver, diffusion_solver)
    elif name == 'godunov':
        from .solver.splitting.godunov import GodunovSplitting
        return GodunovSplitting(ionic_solver, diffusion_solver)
    else:
        raise ValueError(f"Unknown splitting strategy: {name}")


def _build_state(spatial, ionic_model, stimulus, device=None):
    """Build BidomainState from spatial discretization and ionic model."""
    x, y = spatial.coordinates
    n_dof = spatial.n_dof
    if device is None:
        device = spatial.grid.device
    dtype = torch.float64

    Vm = torch.full((n_dof,), ionic_model.V_rest, device=device, dtype=dtype)
    phi_e = torch.zeros(n_dof, device=device, dtype=dtype)
    ionic_states = ionic_model.get_initial_state(n_dof)

    # Process stimulus
    stim_masks = torch.zeros(0, n_dof, device=device, dtype=dtype)
    stim_starts = []
    stim_durations = []
    stim_amplitudes = []
    stim_amplitudes_e = []

    if stimulus is not None:
        masks = []
        for s in stimulus.stimuli:
            mask = s.get_mask(x, y).to(device=device, dtype=dtype)
            masks.append(mask)
            stim_starts.append(s.start_time)
            stim_durations.append(s.duration)
            stim_amplitudes.append(s.amplitude)
            stim_amplitudes_e.append(0.0)
        if masks:
            stim_masks = torch.stack(masks)

    return BidomainState(
        spatial=spatial,
        n_dof=n_dof,
        x=x, y=y,
        Vm=Vm,
        phi_e=phi_e,
        ionic_states=ionic_states,
        gate_indices=ionic_model.gate_indices,
        concentration_indices=ionic_model.concentration_indices,
        stim_masks=stim_masks,
        stim_starts=stim_starts,
        stim_durations=stim_durations,
        stim_amplitudes=stim_amplitudes,
        stim_amplitudes_e=stim_amplitudes_e,
    )
