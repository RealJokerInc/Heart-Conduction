"""
Forward Euler Ionic Solver

Simple explicit Forward Euler for all ionic ODEs.
Requires smaller time steps than Rush-Larsen for stability.

Ref: improvement.md:L1006-1031
"""

from typing import TYPE_CHECKING
import torch

from .base import IonicSolver

if TYPE_CHECKING:
    from .....ionic.base import IonicModel
    from ....state import SimulationState


class ForwardEulerIonicSolver(IonicSolver):
    """
    Simple Forward Euler integrator for ionic models.

    Uses Forward Euler for all variables (voltage, gates, concentrations).
    Less stable than Rush-Larsen, requires smaller time steps.

    For gating variables, dx/dt = (x_inf - x) / tau, so:
        x_new = x_old + dt * (x_inf - x_old) / tau

    Parameters
    ----------
    ionic_model : IonicModel
        The ionic model providing compute functions
    """

    def __init__(self, ionic_model: 'IonicModel'):
        super().__init__(ionic_model)

    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance ionic variables by dt using Forward Euler.

        Modifies state.V and state.ionic_states in-place.

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        model = self.ionic_model
        V = state.V
        S = state.ionic_states

        # 1. Compute ionic current
        Iion = model.compute_Iion(V, S)

        # 2. Evaluate stimulus
        Istim = self._evaluate_Istim(state)

        # 3. Forward Euler on voltage
        # V5.3 convention: dV = -(Iion + Istim)
        state.V = V + dt * (-(Iion + Istim))

        # 4. Forward Euler on gates
        # dx/dt = (x_inf - x) / tau
        gate_inf = model.compute_gate_steady_states(V, S)
        gate_tau = model.compute_gate_time_constants(V, S)

        gate_indices = model.gate_indices
        for i, idx in enumerate(gate_indices):
            x = S[:, idx]
            x_inf = gate_inf[:, i]
            tau = gate_tau[:, i]
            # Forward Euler: x_new = x + dt * dx/dt
            S[:, idx] = x + dt * (x_inf - x) / tau

        # 5. Forward Euler on concentrations
        conc_rates = model.compute_concentration_rates(V, S)

        conc_indices = model.concentration_indices
        for i, idx in enumerate(conc_indices):
            S[:, idx] = S[:, idx] + dt * conc_rates[:, i]
