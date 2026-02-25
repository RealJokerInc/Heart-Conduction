"""
Rush-Larsen Ionic Solver

Exponential integrator for gating variables, Forward Euler for concentrations.
Primary ionic solver for cardiac simulations.

Ref: improvement.md:L1214-1248
Ref: V5.3/IMPLEMENTATION.md:L620-638
"""

from typing import TYPE_CHECKING
import torch

from .base import IonicSolver

if TYPE_CHECKING:
    from .....ionic.base import IonicModel
    from ....state import SimulationState


class RushLarsenSolver(IonicSolver):
    """
    Rush-Larsen exponential integrator for ionic models.

    Uses:
    - Exponential integration for gating variables (Rush-Larsen formula)
    - Forward Euler for concentrations
    - Forward Euler for voltage

    The Rush-Larsen formula for a gating variable x with dx/dt = (x_inf - x) / tau:
        x_new = x_inf - (x_inf - x_old) * exp(-dt/tau)

    This is unconditionally stable for any dt.

    Parameters
    ----------
    ionic_model : IonicModel
        The ionic model providing compute_Iion, compute_gate_steady_states,
        compute_gate_time_constants, compute_concentration_rates
    """

    def __init__(self, ionic_model: 'IonicModel'):
        super().__init__(ionic_model)

    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance ionic variables by dt using Rush-Larsen integration.

        Modifies state.V and state.ionic_states in-place.

        Order of operations (matches V5.3):
        1. Compute Iion from current state
        2. Compute gate_inf, gate_tau from current state (OLD voltage)
        3. Update voltage (Forward Euler)
        4. Update gates (Rush-Larsen using values from step 2)
        5. Update concentrations (Forward Euler)

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        model = self.ionic_model
        V = state.V          # (n_dof,)
        S = state.ionic_states  # (n_dof, n_states)

        # 1. Compute ionic current from current state (BEFORE any updates)
        Iion = model.compute_Iion(V, S)  # (n_dof,)

        # 2. Compute gate steady-states and time constants BEFORE voltage update
        # This matches V5.3 which uses OLD voltage for gate updates
        gate_inf = model.compute_gate_steady_states(V, S)  # (n_dof, n_gates)
        gate_tau = model.compute_gate_time_constants(V, S)  # (n_dof, n_gates)

        # 3. Evaluate stimulus at current time
        Istim = self._evaluate_Istim(state)  # (n_dof,)

        # 4. Forward Euler on voltage
        # V5.3 convention: dV = -(Iion + Istim)
        # A negative Istim (e.g., -80 uA/uF) reduces total current, depolarizing V
        state.V = V + dt * (-(Iion + Istim))

        # 5. Rush-Larsen exponential integration on gates
        # Uses gate_inf/tau computed from OLD state (step 2)
        gate_indices = model.gate_indices
        for i, idx in enumerate(gate_indices):
            x = S[:, idx]
            x_inf = gate_inf[:, i]
            tau = gate_tau[:, i]
            S[:, idx] = x_inf - (x_inf - x) * torch.exp(-dt / tau)

        # 6. Forward Euler on concentrations
        conc_rates = model.compute_concentration_rates(V, S)  # (n_dof, n_conc)

        conc_indices = model.concentration_indices
        for i, idx in enumerate(conc_indices):
            S[:, idx] = S[:, idx] + dt * conc_rates[:, i]

    def _update_gates(
        self,
        ionic_states: torch.Tensor,
        gate_inf: torch.Tensor,
        gate_tau: torch.Tensor,
        dt: float
    ) -> None:
        """
        Update gating variables using Rush-Larsen exponential integration.

        x_new = x_inf - (x_inf - x_old) * exp(-dt/tau)
        """
        model = self.ionic_model
        gate_indices = model.gate_indices
        for i, idx in enumerate(gate_indices):
            x = ionic_states[:, idx]
            x_inf = gate_inf[:, i]
            tau = gate_tau[:, i]
            ionic_states[:, idx] = x_inf - (x_inf - x) * torch.exp(-dt / tau)
