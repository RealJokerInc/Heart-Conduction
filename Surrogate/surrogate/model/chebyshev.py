"""Per-dimension Chebyshev polynomial readout layer.

Maps latent state z ∈ R^n_dims to scalar I_ion via degree-K Chebyshev
polynomials per dimension, plus a direct Vm pathway.

Parameters (default n_dims=16, degree=3):
    C       (16, 4)  Chebyshev coefficients per dim    64
    b_vm    (1,)     Direct Vm bias                     1
    b       (1,)     Scalar bias                        1
    Total                                              66
"""

import torch
import torch.nn as nn
from torch import Tensor


class ChebyshevReadout(nn.Module):
    """KAN-style Chebyshev readout: per-dim polynomial → scalar current.

    Each latent dimension is independently mapped through a Chebyshev
    polynomial of degree K, then summed with a direct Vm term:

        I_ion = Σ_k φ_k(z_k) + b_vm · Vm + b

    where φ_k(z_k) = C[k,:] · [T₀(z̃_k), T₁(z̃_k), ..., T_K(z̃_k)]
    and z̃_k is z_k normalized to [-1, 1] using registered bounds.

    Args:
        n_dims: Number of latent dimensions (default 16).
        degree: Maximum Chebyshev degree K (default 3).
    """

    def __init__(self, n_dims: int = 16, degree: int = 3):
        super().__init__()
        self.n_dims = n_dims
        self.degree = degree

        # Chebyshev coefficients — zero init so model starts as I_ion = b_vm*Vm + b
        self.C = nn.Parameter(torch.zeros(n_dims, degree + 1))
        self.b_vm = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

        # Normalization bounds — default [-1, 1] (identity normalization)
        self.register_buffer("z_min", -torch.ones(n_dims))
        self.register_buffer("z_max", torch.ones(n_dims))

    def set_bounds(self, z_min: Tensor, z_max: Tensor) -> None:
        """Update normalization buffers from training data statistics.

        Args:
            z_min: Per-dim minimum (n_dims,).
            z_max: Per-dim maximum (n_dims,).
        """
        self.z_min.copy_(z_min.detach())
        self.z_max.copy_(z_max.detach())

    def forward(self, z: Tensor, Vm: Tensor) -> Tensor:
        """Compute I_ion from latent state and membrane voltage.

        Args:
            z: Latent state (batch, n_dims) or (n_dims,).
            Vm: Membrane voltage (batch,) or scalar.

        Returns:
            I_ion: Ionic current (batch,) or scalar.
        """
        eps = 1e-8
        # Normalize to [-1, 1]
        z_norm = 2.0 * (z - self.z_min) / (self.z_max - self.z_min + eps) - 1.0
        z_norm = z_norm.clamp(-1.0, 1.0)

        # Chebyshev basis via recurrence: T₀=1, T₁=z̃, Tₙ=2z̃Tₙ₋₁-Tₙ₋₂
        T = [torch.ones_like(z_norm), z_norm]
        for j in range(2, self.degree + 1):
            T.append(2.0 * z_norm * T[-1] - T[-2])
        T = torch.stack(T, dim=-1)  # (..., n_dims, degree+1)

        # Per-dim polynomial evaluation and sum
        phi = (self.C * T).sum(dim=-1)  # (..., n_dims)
        I_ion = phi.sum(dim=-1) + self.b_vm * Vm + self.b  # (...,)
        return I_ion
