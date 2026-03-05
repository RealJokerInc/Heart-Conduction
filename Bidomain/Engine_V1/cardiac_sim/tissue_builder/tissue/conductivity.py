"""
Bidomain Conductivity — Paired intracellular and extracellular conductivity.

Ref: improvement.md L234-270
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class BidomainConductivity:
    """
    Paired intracellular and extracellular conductivity for bidomain.

    Supports three modes:
    1. Scalar isotropic: D_i, D_e (simplest)
    2. Per-node fields: D_i_field, D_e_field (heterogeneous tissue)
    3. Fiber-based: D_i_fiber/D_i_cross + theta field (anisotropic)

    Default values derived from human ventricular tissue:
        sigma_i = 1.74 mS/cm, sigma_e = 6.25 mS/cm
        chi = 1400 cm^-1, Cm = 1.0 uF/cm^2
        D = sigma / (chi * Cm)
    """

    # Scalar isotropic (simplest)
    D_i: float = 0.00124       # cm^2/ms (sigma_i / chi*Cm)
    D_e: float = 0.00446       # cm^2/ms (sigma_e / chi*Cm)

    # Per-node fields (for heterogeneous tissue)
    D_i_field: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None  # (Dxx_i, Dyy_i, Dxy_i)
    D_e_field: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    # Fiber-based (for anisotropic tissue)
    D_i_fiber: Optional[float] = None   # Along fibers
    D_i_cross: Optional[float] = None   # Across fibers
    D_e_fiber: Optional[float] = None
    D_e_cross: Optional[float] = None
    theta: Optional[torch.Tensor] = None  # Fiber angle field (radians)

    def get_effective_monodomain_D(self) -> float:
        """D_eff = D_i * D_e / (D_i + D_e) for validation against monodomain."""
        return self.D_i * self.D_e / (self.D_i + self.D_e)

    def get_boundary_enhanced_D(self) -> float:
        """D at tissue-bath boundary = D_i (Kleber effect)."""
        return self.D_i

    @property
    def is_isotropic(self) -> bool:
        """True if conductivity is scalar isotropic (no field, no fibers)."""
        return (self.D_i_field is None and
                self.D_e_field is None and
                self.theta is None)

    @property
    def D_sum(self) -> float:
        """D_i + D_e (for isotropic elliptic operator)."""
        return self.D_i + self.D_e
