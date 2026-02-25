"""
D2Q5 Lattice Definition

2D lattice with 5 velocities: 1 rest + 4 cardinal directions.
Optimized for diffusion problems on Cartesian grids.

Velocity vectors:
    e_0 = (0, 0)   # rest
    e_1 = (1, 0)   # +x (East)
    e_2 = (-1, 0)  # -x (West)
    e_3 = (0, 1)   # +y (North)
    e_4 = (0, -1)  # -y (South)

Weights: w_0 = 1/3, w_1..4 = 1/6

Speed of sound squared: c_s² = 1/3

Diffusion-relaxation relation (BGK):
    D = (1/3) * (τ - 0.5) * dx² / dt
    τ = 0.5 + 3 * D * dt / dx²

Ref: Research/04_LBM_EP:L105-125
"""

import torch
from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class D2Q5:
    """
    D2Q5 lattice constants and utilities.

    This is a frozen dataclass containing all lattice-specific constants
    needed for LBM simulation. All values are computed once at import time.
    """

    # Number of velocities
    Q: int = 5

    # Number of spatial dimensions
    D: int = 2

    # Speed of sound squared
    cs2: float = 1.0 / 3.0

    # Velocity vectors as tuples: (dx, dy) for each direction
    # Index 0 = rest, 1 = +x, 2 = -x, 3 = +y, 4 = -y
    e: Tuple[Tuple[int, int], ...] = (
        (0, 0),   # 0: rest
        (1, 0),   # 1: +x (East)
        (-1, 0),  # 2: -x (West)
        (0, 1),   # 3: +y (North)
        (0, -1),  # 4: -y (South)
    )

    # Weights
    w: Tuple[float, ...] = (
        1.0 / 3.0,  # rest
        1.0 / 6.0,  # +x
        1.0 / 6.0,  # -x
        1.0 / 6.0,  # +y
        1.0 / 6.0,  # -y
    )

    # Opposite direction mapping: opposite[i] gives the index of -e_i
    opposite: Tuple[int, ...] = (
        0,  # rest -> rest
        2,  # +x -> -x
        1,  # -x -> +x
        4,  # +y -> -y
        3,  # -y -> +y
    )

    def get_e_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get velocity vectors as a tensor.

        Returns
        -------
        e : torch.Tensor
            Shape (Q, 2) = (5, 2), velocity vectors
        """
        return torch.tensor(self.e, device=device, dtype=dtype)

    def get_w_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get weights as a tensor.

        Returns
        -------
        w : torch.Tensor
            Shape (Q,) = (5,), lattice weights
        """
        return torch.tensor(self.w, device=device, dtype=dtype)

    def tau_from_D(self, D: float, dx: float, dt: float) -> float:
        """
        Compute relaxation time τ from diffusion coefficient.

        τ = 0.5 + 3 * D * dt / dx²

        Parameters
        ----------
        D : float
            Diffusion coefficient (cm²/ms)
        dx : float
            Grid spacing (cm)
        dt : float
            Time step (ms)

        Returns
        -------
        tau : float
            Relaxation time (must be > 0.5 for stability)
        """
        return 0.5 + 3.0 * D * dt / (dx * dx)

    def D_from_tau(self, tau: float, dx: float, dt: float) -> float:
        """
        Compute diffusion coefficient from relaxation time.

        D = (τ - 0.5) * dx² / (3 * dt)

        Parameters
        ----------
        tau : float
            Relaxation time
        dx : float
            Grid spacing (cm)
        dt : float
            Time step (ms)

        Returns
        -------
        D : float
            Diffusion coefficient (cm²/ms)
        """
        return (tau - 0.5) * dx * dx / (3.0 * dt)

    def check_stability(self, D: float, dx: float, dt: float) -> Tuple[bool, float]:
        """
        Check if parameters satisfy stability condition τ > 0.5.

        Parameters
        ----------
        D : float
            Diffusion coefficient
        dx : float
            Grid spacing
        dt : float
            Time step

        Returns
        -------
        is_stable : bool
            True if τ > 0.5
        tau : float
            The computed τ value
        """
        tau = self.tau_from_D(D, dx, dt)
        return tau > 0.5, tau


# Singleton instance for convenience
d2q5 = D2Q5()
