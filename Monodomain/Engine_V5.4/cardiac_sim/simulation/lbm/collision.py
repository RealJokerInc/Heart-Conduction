"""
LBM Collision Operators

BGK (Single Relaxation Time) and MRT (Multiple Relaxation Time) collision
operators for the monodomain equation.

BGK:
    f*_i = f_i - (1/τ)(f_i - f_eq_i) + dt·w_i·S

MRT:
    f* = f - M^{-1}·Λ·(Mf - Mf_eq) + dt·w·S

where S is the source term from ionic currents.

Ref: Research/04_LBM_EP:L129-186
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from .d2q5 import D2Q5, d2q5
from .d3q7 import D3Q7, d3q7


class CollisionOperator(ABC):
    """Abstract base class for LBM collision operators."""

    @abstractmethod
    def collide(
        self,
        f: torch.Tensor,
        V: torch.Tensor,
        source: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Apply collision operator.

        Parameters
        ----------
        f : torch.Tensor
            Distribution functions, shape (Q, Nx, Ny) or (Q, Nz, Ny, Nx)
        V : torch.Tensor
            Macroscopic voltage, shape (Nx, Ny) or (Nz, Ny, Nx)
        source : torch.Tensor
            Source term (ionic current / (χ·Cm)), same shape as V
        dt : float
            Time step

        Returns
        -------
        f_star : torch.Tensor
            Post-collision distribution
        """
        pass


class BGKCollision(CollisionOperator):
    """
    BGK (Bhatnagar-Gross-Krook) single relaxation time collision.

    f*_i = f_i - (1/τ)(f_i - f_eq_i) + dt·w_i·S

    where f_eq_i = w_i·V (equilibrium for pure diffusion).

    Parameters
    ----------
    tau : float
        Relaxation time. Must be > 0.5 for stability.
    lattice : D2Q5 or D3Q7
        Lattice definition
    device : torch.device
        Computation device
    dtype : torch.dtype
        Data type
    """

    def __init__(
        self,
        tau: float,
        lattice: D2Q5 = d2q5,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float64
    ):
        if tau <= 0.5:
            raise ValueError(f"τ must be > 0.5 for stability, got {tau}")

        self.tau = tau
        self.omega = 1.0 / tau  # Collision frequency
        self.lattice = lattice
        self.device = device
        self.dtype = dtype

        # Pre-compute weights tensor
        self.w = lattice.get_w_tensor(device, dtype)

    def collide(
        self,
        f: torch.Tensor,
        V: torch.Tensor,
        source: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Apply BGK collision.

        f*_i = f_i - ω·(f_i - f_eq_i) + dt·w_i·S

        where ω = 1/τ and f_eq_i = w_i·V.
        """
        Q = self.lattice.Q

        # Compute equilibrium: f_eq_i = w_i * V
        # w has shape (Q,), V has shape (Nx, Ny) or (Nz, Ny, Nx)
        # Need to broadcast correctly
        if V.dim() == 2:
            # 2D: (Nx, Ny) -> (Q, Nx, Ny)
            f_eq = self.w.view(Q, 1, 1) * V.unsqueeze(0)
        else:
            # 3D: (Nz, Ny, Nx) -> (Q, Nz, Ny, Nx)
            f_eq = self.w.view(Q, 1, 1, 1) * V.unsqueeze(0)

        # Collision: f* = f - ω·(f - f_eq)
        f_star = f - self.omega * (f - f_eq)

        # Add source term: + dt·w·S
        if V.dim() == 2:
            source_term = dt * self.w.view(Q, 1, 1) * source.unsqueeze(0)
        else:
            source_term = dt * self.w.view(Q, 1, 1, 1) * source.unsqueeze(0)

        f_star = f_star + source_term

        return f_star


class MRTCollision(CollisionOperator):
    """
    MRT (Multiple Relaxation Time) collision operator.

    Allows anisotropic diffusion by using different relaxation rates
    for different moments. Essential for cardiac tissue with fiber anisotropy.

    For D2Q5:
        M = [[1,  1,  1,  1,  1],    # conserved (V)
             [0,  1, -1,  0,  0],    # j_x (x-flux)
             [0,  0,  0,  1, -1],    # j_y (y-flux)
             [0,  1,  1, -1, -1],    # e (energy-like)
             [0,  1,  1,  1,  1]]    # ε (higher moment)

    Relaxation rates:
        - Conserved moment (row 0): s_0 = 0 (no relaxation, mass conservation)
        - Flux moments (rows 1-2): s_x, s_y derived from D_xx, D_yy
        - Higher moments: s_e, s_ε tunable for stability

    Parameters
    ----------
    D_xx : float
        Diffusion coefficient in x-direction
    D_yy : float
        Diffusion coefficient in y-direction
    dx : float
        Grid spacing
    dt : float
        Time step
    lattice : D2Q5
        Lattice definition (currently only D2Q5 supported)
    device : torch.device
        Computation device
    dtype : torch.dtype
        Data type
    s_higher : float
        Relaxation rate for higher moments (default 1.0 for stability)
    """

    def __init__(
        self,
        D_xx: float,
        D_yy: float,
        dx: float,
        dt: float,
        lattice: D2Q5 = d2q5,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float64,
        s_higher: float = 1.0
    ):
        if not isinstance(lattice, D2Q5):
            raise NotImplementedError("MRT currently only supports D2Q5")

        self.lattice = lattice
        self.device = device
        self.dtype = dtype
        self.dx = dx
        self.dt = dt

        # Compute relaxation rates from diffusion coefficients
        # τ_x = 0.5 + 3·D_xx·dt/dx²
        # s_x = 1/τ_x
        tau_x = 0.5 + 3.0 * D_xx * dt / (dx * dx)
        tau_y = 0.5 + 3.0 * D_yy * dt / (dx * dx)

        if tau_x <= 0.5 or tau_y <= 0.5:
            raise ValueError(
                f"Unstable: τ_x={tau_x:.4f}, τ_y={tau_y:.4f}. "
                f"Need τ > 0.5. Reduce D or dt."
            )

        s_x = 1.0 / tau_x
        s_y = 1.0 / tau_y

        # Build transformation matrix M (5x5 for D2Q5)
        # Columns correspond to: rest, +x, -x, +y, -y
        self.M = torch.tensor([
            [1,  1,  1,  1,  1],    # 0: conserved (V = Σf_i)
            [0,  1, -1,  0,  0],    # 1: x-flux
            [0,  0,  0,  1, -1],    # 2: y-flux
            [0,  1,  1, -1, -1],    # 3: energy-like
            [0,  1,  1,  1,  1],    # 4: higher moment (ε)
        ], device=device, dtype=dtype)

        # Build inverse transformation matrix
        self.M_inv = torch.linalg.inv(self.M)

        # Diagonal relaxation matrix S
        # s_0 = 0 (conserved), s_1 = s_x, s_2 = s_y, s_3 = s_4 = s_higher
        self.S_diag = torch.tensor(
            [0.0, s_x, s_y, s_higher, s_higher],
            device=device, dtype=dtype
        )

        # Pre-compute M_inv @ S for efficiency
        # We need M_inv @ diag(S) which is M_inv with columns scaled by S
        self.M_inv_S = self.M_inv * self.S_diag.unsqueeze(0)

        # Weights tensor
        self.w = lattice.get_w_tensor(device, dtype)

    def collide(
        self,
        f: torch.Tensor,
        V: torch.Tensor,
        source: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Apply MRT collision.

        f* = f - M^{-1}·S·(Mf - Mf_eq) + dt·w·S_source

        In moment space:
            m = M·f
            m_eq = M·f_eq  (equilibrium moments)
            Δm = S·(m - m_eq)
            f* = f - M^{-1}·Δm
        """
        Q = self.lattice.Q
        Nx, Ny = V.shape

        # Reshape f from (Q, Nx, Ny) to (Q, Nx*Ny) for matrix operations
        f_flat = f.reshape(Q, -1)

        # Transform to moment space: m = M @ f
        m = self.M @ f_flat  # (Q, Ny*Nx)

        # Compute equilibrium distribution: f_eq = w * V
        V_flat = V.flatten()
        f_eq = self.w.unsqueeze(1) * V_flat.unsqueeze(0)  # (Q, Ny*Nx)

        # Equilibrium in moment space: m_eq = M @ f_eq
        m_eq = self.M @ f_eq  # (Q, Ny*Nx)

        # Relaxation: Δm = S @ (m - m_eq)
        # Since S is diagonal, this is element-wise
        delta_m = self.S_diag.unsqueeze(1) * (m - m_eq)

        # Transform back: f* = f - M_inv @ Δm
        f_star_flat = f_flat - self.M_inv @ delta_m

        # Add source term
        source_flat = source.flatten()
        source_contrib = dt * self.w.unsqueeze(1) * source_flat.unsqueeze(0)
        f_star_flat = f_star_flat + source_contrib

        # Reshape back to (Q, Nx, Ny)
        f_star = f_star_flat.reshape(Q, Nx, Ny)

        return f_star


def create_isotropic_bgk(
    D: float,
    dx: float,
    dt: float,
    lattice: D2Q5 = d2q5,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float64
) -> BGKCollision:
    """
    Create BGK collision operator for isotropic diffusion.

    Convenience function that computes τ from D, dx, dt.

    Parameters
    ----------
    D : float
        Isotropic diffusion coefficient
    dx : float
        Grid spacing
    dt : float
        Time step
    lattice : D2Q5
        Lattice definition
    device : torch.device
        Computation device
    dtype : torch.dtype
        Data type

    Returns
    -------
    collision : BGKCollision
    """
    tau = lattice.tau_from_D(D, dx, dt)
    return BGKCollision(tau, lattice, device, dtype)


def create_anisotropic_mrt(
    D_fiber: float,
    D_cross: float,
    fiber_angle: float,
    dx: float,
    dt: float,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float64
) -> MRTCollision:
    """
    Create MRT collision operator for anisotropic diffusion.

    Computes D_xx, D_yy from fiber/cross diffusion and fiber angle.

    Parameters
    ----------
    D_fiber : float
        Diffusion along fiber direction
    D_cross : float
        Diffusion across fiber direction
    fiber_angle : float
        Fiber angle in radians (0 = aligned with x-axis)
    dx : float
        Grid spacing
    dt : float
        Time step
    device : torch.device
        Computation device
    dtype : torch.dtype
        Data type

    Returns
    -------
    collision : MRTCollision
    """
    import math

    cos_a = math.cos(fiber_angle)
    sin_a = math.sin(fiber_angle)

    # Rotate diffusion tensor
    # D = R @ diag(D_fiber, D_cross) @ R^T
    D_xx = D_fiber * cos_a**2 + D_cross * sin_a**2
    D_yy = D_fiber * sin_a**2 + D_cross * cos_a**2
    # D_xy = (D_fiber - D_cross) * cos_a * sin_a  # Not used in MRT diagonal

    return MRTCollision(D_xx, D_yy, dx, dt, d2q5, device, dtype)
