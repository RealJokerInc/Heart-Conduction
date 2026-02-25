"""
Time Stepping Schemes for Monodomain Equation

Provides implicit time integrators:
- Crank-Nicolson (θ=0.5) - default, 2nd order
- BDF1 (Backward Euler) - 1st order, L-stable
- BDF2 - 2nd order, A-stable

The monodomain equation (semi-discrete):
    M·dV/dt = -K·V - M·Iion + M·Istim

Each scheme reformulates this as a linear system to solve at each time step.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import torch

from .linear import pcg_solve, sparse_mv


class TimeScheme(Enum):
    """Available time integration schemes."""
    CN = 'CN'        # Crank-Nicolson (θ=0.5)
    BDF1 = 'BDF1'    # Backward Euler
    BDF2 = 'BDF2'    # 2nd order BDF


@dataclass
class TimeStepperConfig:
    """Configuration for time stepping."""
    scheme: TimeScheme = TimeScheme.CN
    theta: float = 0.5           # For generalized θ-method (CN uses 0.5)
    pcg_tol: float = 1e-8        # PCG tolerance
    pcg_max_iter: int = 500      # PCG max iterations
    use_warm_start: bool = True  # Use previous solution as initial guess


class CrankNicolsonStepper:
    """
    Crank-Nicolson time integrator.

    Discretizes: M·dV/dt = -K·V + M·f

    As: (M + θ·dt·K)·V^{n+1} = (M - (1-θ)·dt·K)·V^n + dt·M·f^{n+θ}

    With θ=0.5 (centered):
    (M + 0.5·dt·K)·V^{n+1} = (M - 0.5·dt·K)·V^n + dt·M·f

    where f = Istim - Iion (evaluated at time n or n+0.5)
    """

    def __init__(
        self,
        M: torch.Tensor,
        K: torch.Tensor,
        theta: float = 0.5,
        config: Optional[TimeStepperConfig] = None
    ):
        """
        Initialize Crank-Nicolson stepper.

        Parameters
        ----------
        M : torch.Tensor
            Mass matrix (sparse COO)
        K : torch.Tensor
            Stiffness matrix (sparse COO)
        theta : float
            Implicitness parameter (0.5 = CN, 1.0 = Backward Euler)
        config : TimeStepperConfig, optional
            Solver configuration
        """
        self.M = M
        self.K = K
        self.theta = theta
        self.config = config or TimeStepperConfig()

        self.device = M.device
        self.dtype = M.dtype
        self.n = M.shape[0]

        # Cached matrices for fixed dt
        self._cached_dt = None
        self._A_lhs = None
        self._B_rhs = None

        # Warm start
        self._last_solution = None

    def _build_matrices(self, dt: float):
        """Build LHS and RHS matrices for given dt."""
        if self._cached_dt != dt:
            # LHS: A = M + θ·dt·K
            self._A_lhs = (self.M + self.theta * dt * self.K).coalesce()

            # RHS coefficient: B = M - (1-θ)·dt·K
            self._B_rhs = (self.M - (1 - self.theta) * dt * self.K).coalesce()

            self._cached_dt = dt

    def step(
        self,
        V: torch.Tensor,
        f: torch.Tensor,
        dt: float,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Advance voltage by one time step.

        Parameters
        ----------
        V : torch.Tensor
            Current voltage (n_nodes,)
        f : torch.Tensor
            Source term f = Istim - Iion (n_nodes,)
        dt : float
            Time step size (ms)
        return_stats : bool
            If True, return solver stats

        Returns
        -------
        V_new : torch.Tensor
            Voltage at next time step
        stats : SolverStats, optional
            Solver statistics
        """
        # Build matrices if needed
        self._build_matrices(dt)

        # RHS: b = B·V + dt·M·f
        b = sparse_mv(self._B_rhs, V) + dt * sparse_mv(self.M, f)

        # Initial guess for PCG
        x0 = self._last_solution if self.config.use_warm_start else None

        # Solve: A·V^{n+1} = b
        result = pcg_solve(
            self._A_lhs, b,
            x0=x0,
            tol=self.config.pcg_tol,
            max_iter=self.config.pcg_max_iter,
            return_stats=return_stats
        )

        if return_stats:
            V_new, stats = result
            self._last_solution = V_new.clone()
            return V_new, stats
        else:
            V_new = result
            self._last_solution = V_new.clone()
            return V_new


class BDFStepper:
    """
    BDF (Backward Differentiation Formula) time integrator.

    BDF1 (Backward Euler):
        (M/dt + K)·V^{n+1} = M·V^n/dt + M·f

    BDF2 (2nd order):
        (3M/(2dt) + K)·V^{n+1} = 2M·V^n/dt - M·V^{n-1}/(2dt) + M·f

    BDF2 automatically falls back to BDF1 for the first step when history
    is not available.
    """

    def __init__(
        self,
        M: torch.Tensor,
        K: torch.Tensor,
        order: int = 2,
        config: Optional[TimeStepperConfig] = None
    ):
        """
        Initialize BDF stepper.

        Parameters
        ----------
        M : torch.Tensor
            Mass matrix (sparse COO)
        K : torch.Tensor
            Stiffness matrix (sparse COO)
        order : int
            BDF order (1 or 2)
        config : TimeStepperConfig, optional
            Solver configuration
        """
        self.M = M
        self.K = K
        self.order = order
        self.config = config or TimeStepperConfig()

        self.device = M.device
        self.dtype = M.dtype
        self.n = M.shape[0]

        # History for BDF2
        self.history: List[torch.Tensor] = []

        # Cached matrices
        self._cached_dt = None
        self._A_bdf1 = None
        self._A_bdf2 = None

        # Warm start
        self._last_solution = None

    def _build_matrices(self, dt: float):
        """Build LHS matrices for given dt."""
        if self._cached_dt != dt:
            # BDF1: A = M/dt + K
            self._A_bdf1 = (self.M * (1.0 / dt) + self.K).coalesce()

            # BDF2: A = 3M/(2dt) + K
            self._A_bdf2 = (self.M * (1.5 / dt) + self.K).coalesce()

            self._cached_dt = dt

    def push_history(self, V: torch.Tensor):
        """Add state to history."""
        self.history.append(V.clone())
        if len(self.history) > self.order:
            self.history.pop(0)

    @property
    def initialized(self) -> bool:
        """Check if BDF2 has enough history."""
        return len(self.history) >= self.order

    def step(
        self,
        V: torch.Tensor,
        f: torch.Tensor,
        dt: float,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Advance voltage by one time step.

        Parameters
        ----------
        V : torch.Tensor
            Current voltage (n_nodes,)
        f : torch.Tensor
            Source term f = Istim - Iion (n_nodes,)
        dt : float
            Time step size (ms)
        return_stats : bool
            If True, return solver stats

        Returns
        -------
        V_new : torch.Tensor
            Voltage at next time step
        """
        # Build matrices if needed
        self._build_matrices(dt)

        # Store current state in history
        self.push_history(V)

        # Choose BDF1 or BDF2
        use_bdf2 = (self.order == 2 and len(self.history) >= 2)

        if use_bdf2:
            # BDF2: b = 2M·V^n/dt - M·V^{n-1}/(2dt) + M·f
            A = self._A_bdf2
            V_n = self.history[-1]    # Current (just pushed)
            V_nm1 = self.history[-2]  # Previous
            b = (sparse_mv(self.M, V_n) * (2.0 / dt) -
                 sparse_mv(self.M, V_nm1) * (0.5 / dt) +
                 sparse_mv(self.M, f))
        else:
            # BDF1: b = M·V^n/dt + M·f
            A = self._A_bdf1
            V_n = self.history[-1]
            b = sparse_mv(self.M, V_n) * (1.0 / dt) + sparse_mv(self.M, f)

        # Initial guess for PCG
        x0 = self._last_solution if self.config.use_warm_start else None

        # Solve: A·V^{n+1} = b
        result = pcg_solve(
            A, b,
            x0=x0,
            tol=self.config.pcg_tol,
            max_iter=self.config.pcg_max_iter,
            return_stats=return_stats
        )

        if return_stats:
            V_new, stats = result
            self._last_solution = V_new.clone()
            return V_new, stats
        else:
            V_new = result
            self._last_solution = V_new.clone()
            return V_new

    def reset_history(self):
        """Clear history (for restarting simulation)."""
        self.history.clear()


def create_time_stepper(
    M: torch.Tensor,
    K: torch.Tensor,
    scheme: str = 'CN',
    config: Optional[TimeStepperConfig] = None
):
    """
    Factory function to create time stepper.

    Parameters
    ----------
    M : torch.Tensor
        Mass matrix
    K : torch.Tensor
        Stiffness matrix
    scheme : str
        Time scheme: 'CN', 'BDF1', or 'BDF2'
    config : TimeStepperConfig, optional
        Configuration

    Returns
    -------
    stepper : CrankNicolsonStepper or BDFStepper
    """
    scheme = scheme.upper()

    if scheme == 'CN':
        return CrankNicolsonStepper(M, K, theta=0.5, config=config)
    elif scheme == 'BDF1':
        return BDFStepper(M, K, order=1, config=config)
    elif scheme == 'BDF2':
        return BDFStepper(M, K, order=2, config=config)
    else:
        raise ValueError(f"Unknown time scheme: {scheme}")


def heat_equation_step_analytic(
    u0: torch.Tensor,
    sigma2: float,
    D: float,
    t: float,
    x: torch.Tensor,
    y: torch.Tensor,
    x0: float = 0.5,
    y0: float = 0.5
) -> torch.Tensor:
    """
    Analytic solution for 2D heat equation with Gaussian initial condition.

    Initial: u(x,y,0) = exp(-((x-x0)² + (y-y0)²) / (4σ²))
    Solution: u(x,y,t) = (σ²/(σ²+Dt)) * exp(-((x-x0)² + (y-y0)²) / (4(σ²+Dt)))

    Parameters
    ----------
    u0 : torch.Tensor
        Initial condition (not used, for interface compatibility)
    sigma2 : float
        Initial Gaussian variance
    D : float
        Diffusion coefficient
    t : float
        Time
    x, y : torch.Tensor
        Coordinates
    x0, y0 : float
        Center of Gaussian

    Returns
    -------
    u : torch.Tensor
        Analytic solution at time t
    """
    denom = sigma2 + D * t
    return (sigma2 / denom) * torch.exp(-((x - x0)**2 + (y - y0)**2) / (4 * denom))
