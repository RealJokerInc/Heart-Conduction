"""Ionic surrogate model — 3-stage autoregressive latent predictor.

Architecture (v2, 886 FLOPs, 3.7× Rush-Larsen):
    Stage 1: n×1 cross-attention — per-dim gated update toward voltage target
    Stage 2: 2× split GELU cross-channel — spectrally-normed coupling
    Stage 3: KAN Chebyshev K=3 readout — per-dim polynomial → I_ion

Parameters (default latent_dim=16, attn_dim=8, degree=3):
    ┌─────────────────────────┬──────────────┬───────────┐
    │ Component               │ Inference    │ Training  │
    ├─────────────────────────┼──────────────┼───────────┤
    │ Stage 1: W_q (16,8)    │ 128          │ 128       │
    │ Stage 1: W_k (2→8)     │ 16           │ 16        │
    │ Stage 1: W_v (2→8)     │ 16           │ 16        │
    │ Stage 1: W_out (8,16)  │ 128          │ 128       │
    │ Stage 2a: W_cc1+b      │ 144          │ 144       │
    │ Stage 2b: W_cc2+b      │ 144          │ 144       │
    │ Stage 3: C+b_vm+b      │ 66           │ 66        │
    │ Scaffold: W_dec+b_dec  │ —            │ 306       │
    ├─────────────────────────┼──────────────┼───────────┤
    │ Total                   │ 642          │ 948       │
    └─────────────────────────┴──────────────┴───────────┘
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm

from .chebyshev import ChebyshevReadout


class IonicSurrogate(nn.Module):
    """3-stage ionic surrogate: cross-attention → split GELU → Chebyshev readout.

    Autoregressive: each call takes (latent_prev, Vm, dt) and returns
    (latent_new, I_ion, gates_pred). The latent state is carried forward
    across time steps.

    Args:
        latent_dim: Latent state dimension (default 16).
        attn_dim: Attention/projection dimension (default 8).
        cheby_degree: Chebyshev polynomial degree (default 3).
        split: Split point for GELU gating (default 8).
        n_gates: Number of ionic gates for scaffold decoder (default 18).
        scaffold: Whether to include the training scaffold decoder.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        attn_dim: int = 8,
        cheby_degree: int = 3,
        split: int = 8,
        n_gates: int = 18,
        scaffold: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.attn_dim = attn_dim
        self.split = split

        # --- Stage 1: n×1 cross-attention ---
        # W_q is a raw Parameter (NOT nn.Linear) — per-dim query
        self.W_q = nn.Parameter(torch.empty(latent_dim, attn_dim))
        self.W_k = nn.Linear(2, attn_dim, bias=False)
        self.W_v = nn.Linear(2, attn_dim, bias=False)
        self.W_out = nn.Parameter(torch.empty(attn_dim, latent_dim))
        self.scale = 1.0 / math.sqrt(attn_dim)

        # --- Stage 2: two-round split GELU ---
        # CRITICAL: init weights BEFORE spectral_norm wrapping
        _cc1 = nn.Linear(split, latent_dim)
        _cc2 = nn.Linear(split, latent_dim)
        xavier_uniform_(_cc1.weight)
        xavier_uniform_(_cc2.weight)
        self.cc1 = spectral_norm(_cc1)
        self.cc2 = spectral_norm(_cc2)

        # --- Stage 3: Chebyshev readout ---
        self.readout = ChebyshevReadout(latent_dim, cheby_degree)

        # --- Scaffold decoder (training only) ---
        if scaffold:
            self.decoder = nn.Linear(latent_dim, n_gates)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for attention weights. Readout already zero-init."""
        xavier_uniform_(self.W_q)
        xavier_uniform_(self.W_k.weight)
        xavier_uniform_(self.W_v.weight)
        xavier_uniform_(self.W_out)
        # cc1, cc2: already initialized before spectral_norm wrapping
        # readout: zero-init in ChebyshevReadout.__init__
        # decoder: default init (overwritten by Phase A training)

    def forward(
        self, latent_prev: Tensor, Vm: Tensor, dt: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Forward pass: one autoregressive step.

        Args:
            latent_prev: Previous latent state (batch, latent_dim) or (latent_dim,).
            Vm: Membrane voltage (batch,) or scalar.
            dt: Time step (batch,) or scalar.

        Returns:
            latent_new: Updated latent state, same shape as latent_prev.
            I_ion: Predicted ionic current, same shape as Vm.
            gates_pred: Predicted gate values (batch, n_gates) or None if no scaffold.
        """
        # Handle unbatched input
        squeezed = latent_prev.dim() == 1
        if squeezed:
            latent_prev = latent_prev.unsqueeze(0)
            Vm = Vm.view(1)
            dt = dt.view(1)

        # --- Stage 1: n×1 cross-attention ---
        x = torch.stack([Vm, dt], dim=-1)           # (B, 2)
        k = self.W_k(x)                              # (B, 8)
        v = self.W_v(x)                              # (B, 8)

        # Per-dim query: latent[b,d] * W_q[d,:] → (B, 16, 8)
        q = latent_prev.unsqueeze(-1) * self.W_q     # (B,16,1) * (16,8) → (B,16,8)

        # Attention score: dot product of q and k per dim
        score = (q * k.unsqueeze(1)).sum(-1) * self.scale  # (B, 16)
        gate = torch.sigmoid(score)                   # (B, 16)

        # Per-dim target from voltage value
        target = v @ self.W_out                       # (B, 16)

        # Contractive update: interpolate toward target
        latent_mid = latent_prev + gate * (target - latent_prev)

        # --- Stage 2: two-round split GELU cross-channel ---
        s = self.split

        # Round 1
        g1 = F.gelu(latent_mid[:, :s]) * latent_mid[:, s:]
        corr1 = self.cc1(g1)
        corr1 = corr1 / (corr1.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)  # RMSNorm
        latent_a = latent_mid + corr1

        # Round 2
        g2 = F.gelu(latent_a[:, :s]) * latent_a[:, s:]
        corr2 = self.cc2(g2)
        corr2 = corr2 / (corr2.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)  # RMSNorm
        latent_new = latent_a + corr2

        # --- Stage 3: Chebyshev readout ---
        I_ion = self.readout(latent_new, Vm)

        # --- Scaffold decoder ---
        gates_pred = None
        if hasattr(self, "decoder"):
            gates_pred = torch.sigmoid(self.decoder(latent_new))

        # Restore unbatched shape
        if squeezed:
            latent_new = latent_new.squeeze(0)
            I_ion = I_ion.squeeze(0)
            if gates_pred is not None:
                gates_pred = gates_pred.squeeze(0)

        return latent_new, I_ion, gates_pred

    def inference_param_count(self) -> int:
        """Count parameters excluding scaffold decoder."""
        total = sum(p.numel() for p in self.parameters())
        if hasattr(self, "decoder"):
            total -= sum(p.numel() for p in self.decoder.parameters())
        return total

    def remove_scaffold(self) -> None:
        """Remove scaffold decoder for production inference. Idempotent."""
        if hasattr(self, "decoder"):
            del self.decoder
