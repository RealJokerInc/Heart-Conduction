"""Lattice base class for LBM velocity sets."""

from abc import ABC, abstractmethod
import torch
from torch import Tensor


class Lattice(ABC):
    """Abstract base for D2Qn lattice definitions.

    Subclasses define Q, e, w, opposite, cs2 as class-level constants.
    """

    Q: int                          # Number of discrete velocities
    D: int = 2                      # Spatial dimensions
    cs2: float = 1.0 / 3.0         # Speed of sound squared
    e: tuple                        # Velocity vectors: ((ex, ey), ...)
    w: tuple                        # Weights: (w_0, w_1, ...)
    opposite: tuple                 # Opposite direction indices

    def get_e_tensor(self, device: torch.device = None, dtype: torch.dtype = torch.float64) -> Tensor:
        """Return velocity vectors as (Q, 2) tensor."""
        return torch.tensor(self.e, device=device, dtype=dtype)

    def get_w_tensor(self, device: torch.device = None, dtype: torch.dtype = torch.float64) -> Tensor:
        """Return weights as (Q,) tensor."""
        return torch.tensor(self.w, device=device, dtype=dtype)
