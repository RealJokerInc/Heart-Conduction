"""IonicNODE: wraps IonicStage1.dzdt as a torchdiffeq ODE function.

Training:  odeint_adjoint(node, z0, t_eval)  — adjoint shapes vector field geometry
Inference: node.euler_step(z, V, dt)         — no solver, works for any dt

Zero new learned parameters. All dynamics in stage1.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from torchdiffeq import odeint, odeint_adjoint

from .stage1 import IonicStage1


class IonicNODE(nn.Module):

    def __init__(self, stage1: IonicStage1):
        super().__init__()
        self.stage1 = stage1
        self._V_traj: Optional[Tensor] = None   # (B, T) or (T,)
        self._t_grid: Optional[Tensor] = None   # (T,) cumulative times
        self.nfe: int = 0  # number of function evaluations (reset per integrate())

    def set_v_trajectory(self, V_traj: Tensor, t_grid: Tensor) -> None:
        """Store V(t) for interpolation during integrate(). Call before integrate()."""
        self._V_traj = V_traj
        self._t_grid = t_grid

    def clear_v_trajectory(self) -> None:
        self._V_traj = None
        self._t_grid = None

    def _interpolate_V(self, t: Tensor) -> Tensor:
        """Linear interpolation of V at continuous scalar time t.

        V_traj has T points (one per segment step). t_grid has T+1 points
        (cumulative dt, including t=0). Interpolation maps t to V values using
        t_grid[:-1] as the knot times (each V[i] corresponds to t_grid[i]).
        At t >= t_grid[T-1], clamp to V_traj[T-1] (last value).
        """
        assert self._V_traj is not None, "Call set_v_trajectory() before integrate()"
        t_grid = self._t_grid
        T = self._V_traj.shape[-1]  # number of V samples
        t_c = t.clamp(t_grid[0], t_grid[T - 1])  # clamp to V_traj range
        idx = (torch.searchsorted(t_grid[:T].contiguous(), t_c) - 1).clamp(0, T - 2)
        t0, t1 = t_grid[idx], t_grid[idx + 1]
        frac = ((t_c - t0) / (t1 - t0 + 1e-12)).clamp(0.0, 1.0)
        if self._V_traj.dim() == 1:
            return self._V_traj[idx] + frac * (self._V_traj[idx + 1] - self._V_traj[idx])
        else:
            return self._V_traj[:, idx] + frac * (self._V_traj[:, idx + 1] - self._V_traj[:, idx])

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """torchdiffeq interface: (scalar t, state z) -> dz/dt."""
        self.nfe += 1
        V = self._interpolate_V(t)
        return self.stage1.dzdt(z, V)

    def integrate(
        self,
        z0: Tensor,
        t_eval: Tensor,
        method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-3,
        adjoint: bool = False,
    ) -> Tensor:
        """Integrate from t_eval[0] to t_eval[-1], return z at each t_eval point.

        Args:
            adjoint: If True, use odeint_adjoint (O(1) memory). If False, use
                odeint (backprop through solver — more memory, stable early training).
        Returns: (N, B, carried_dim) or (N, carried_dim) if unbatched.
        """
        self.nfe = 0  # reset for this integration
        if adjoint:
            return odeint_adjoint(
                self, z0, t_eval,
                method=method, rtol=rtol, atol=atol,
                adjoint_params=tuple(self.stage1.parameters()),
            )
        else:
            return odeint(
                self, z0, t_eval,
                method=method, rtol=rtol, atol=atol,
            )

    def euler_step(self, z: Tensor, V: Tensor, dt: float) -> Tensor:
        """Euler inference step. No solver. Works for any dt value.

        Args:
            z: (B, carried_dim) or (carried_dim,)
            V: (B,) or scalar
            dt: timestep in ms (float — dt-independence guaranteed by training)
        Returns:
            z_next: same shape as z
        """
        return z + dt * self.stage1.dzdt(z, V)
