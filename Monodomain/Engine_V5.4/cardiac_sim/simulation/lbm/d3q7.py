"""
D3Q7 Lattice Definition

3D lattice with 7 velocities: 1 rest + 6 face-centered directions.
Direct 3D extension of D2Q5 for volumetric cardiac simulations.

Velocity vectors:
    e_0 = (0, 0, 0)   # rest
    e_1 = (1, 0, 0)   # +x
    e_2 = (-1, 0, 0)  # -x
    e_3 = (0, 1, 0)   # +y
    e_4 = (0, -1, 0)  # -y
    e_5 = (0, 0, 1)   # +z
    e_6 = (0, 0, -1)  # -z

Weights: w_0 = 1/4, w_1..6 = 1/8 (from Rapaka et al.)

Note: Some references use w_0 = 1/2, w_1..6 = 1/12.
We use 1/4, 1/8 to match the original LBM-EP paper.

Speed of sound squared: c_s² = 1/3

Ref: Research/04_LBM_EP:L95-103
"""

import torch
from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class D3Q7:
    """
    D3Q7 lattice constants and utilities.

    This is a frozen dataclass containing all lattice-specific constants
    needed for 3D LBM simulation.
    """

    # Number of velocities
    Q: int = 7

    # Number of spatial dimensions
    D: int = 3

    # Speed of sound squared
    cs2: float = 1.0 / 3.0

    # Velocity vectors as tuples: (dx, dy, dz) for each direction
    e: Tuple[Tuple[int, int, int], ...] = (
        (0, 0, 0),    # 0: rest
        (1, 0, 0),    # 1: +x
        (-1, 0, 0),   # 2: -x
        (0, 1, 0),    # 3: +y
        (0, -1, 0),   # 4: -y
        (0, 0, 1),    # 5: +z
        (0, 0, -1),   # 6: -z
    )

    # Weights (Rapaka et al. convention)
    w: Tuple[float, ...] = (
        1.0 / 4.0,  # rest
        1.0 / 8.0,  # +x
        1.0 / 8.0,  # -x
        1.0 / 8.0,  # +y
        1.0 / 8.0,  # -y
        1.0 / 8.0,  # +z
        1.0 / 8.0,  # -z
    )

    # Opposite direction mapping
    opposite: Tuple[int, ...] = (
        0,  # rest -> rest
        2,  # +x -> -x
        1,  # -x -> +x
        4,  # +y -> -y
        3,  # -y -> +y
        6,  # +z -> -z
        5,  # -z -> +z
    )

    def get_e_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get velocity vectors as a tensor.

        Returns
        -------
        e : torch.Tensor
            Shape (Q, 3) = (7, 3), velocity vectors
        """
        return torch.tensor(self.e, device=device, dtype=dtype)

    def get_w_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get weights as a tensor.

        Returns
        -------
        w : torch.Tensor
            Shape (Q,) = (7,), lattice weights
        """
        return torch.tensor(self.w, device=device, dtype=dtype)

    def tau_from_D(self, D: float, dx: float, dt: float) -> float:
        """
        Compute relaxation time τ from diffusion coefficient.

        For D3Q7 with w_0=1/4, w_i=1/8:
        τ = 0.5 + 4 * D * dt / dx²

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
        # For D3Q7 with these weights: D = cs² * (τ - 0.5) * dx² / dt
        # cs² = 1/3, so τ = 0.5 + 3 * D * dt / dx² (same as D2Q5)
        return 0.5 + 3.0 * D * dt / (dx * dx)

    def D_from_tau(self, tau: float, dx: float, dt: float) -> float:
        """
        Compute diffusion coefficient from relaxation time.

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
d3q7 = D3Q7()
