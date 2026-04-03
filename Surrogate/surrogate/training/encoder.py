"""Temporary encoder: true ionic states (14) -> ionic latent (16).

Training scaffold only. Used in Phase A1 (autoencoder) and Phase B (teacher forcing).
Discarded after Phase B — never part of inference.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_


class TemporaryEncoder(nn.Module):
    """Maps true ionic states to ionic latent. Training scaffold only.

    Simple 2-layer MLP with GELU. Intentionally simple so the latent space
    it creates is reproducible by the attention mechanism during Phase B.
    """

    def __init__(self, n_ionic_targets: int = 14, ionic_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_ionic_targets, ionic_dim),
            nn.GELU(),
            nn.Linear(ionic_dim, ionic_dim),
        )
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.net[0].weight)
        xavier_uniform_(self.net[2].weight)

    def forward(self, ionic_states: Tensor) -> Tensor:
        """(B, 14) -> (B, 16) or (14,) -> (16,)."""
        return self.net(ionic_states)


def make_carried_state(
    encoder: TemporaryEncoder,
    ionic_states: Tensor,
    concentrations: Tensor,
) -> Tensor:
    """Build carried_state from encoder output + concentrations.

    Args:
        encoder: Temporary encoder mapping (B, 14) -> (B, 16).
        ionic_states: True ionic states (B, 14) or (14,).
        concentrations: True concentrations (B, 4) or (4,).

    Returns:
        carried_state: (B, 20) or (20,) = cat([encoder(ionic_states), concentrations]).
    """
    latent = encoder(ionic_states)
    return torch.cat([latent, concentrations], dim=-1)
