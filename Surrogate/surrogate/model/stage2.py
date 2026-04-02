"""Stage 2: Cross-attention current readout.

Cross-attention over conductance latent (queries) and normalized environment
tokens (keys/values), followed by an output MLP.  Produces scalar I_ion.

No softmax -- attention scores can be negative, which is physically meaningful
(the driving-force term Vm - E can be negative).

Architecture (small TTP06 config, cond_dim=8):
    Q = cond_lat * e_q          (B, C, d)        C=8, d=4
    K = env_norm * e_k          (B, 9, d)
    V = env_norm * e_v          (B, 9, d_v)       d_v=1
    scores = Q @ K^T / sqrt(d)  (B, C, 9)        no softmax
    attended = scores @ V        (B, C, d_v) -> squeeze -> (B, C)
    I_ion = MLP(attended)        (B, 1) -> squeeze -> (B,)

Parameters (small config):
    e_q  (8, 4)   = 32
    e_k  (9, 4)   = 36
    e_v  (9, 1)   = 9
    W1   (8, 4)+b = 36
    W2   (4, 1)+b = 5
    Total          = 118
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class ConductanceAttention(nn.Module):
    """Cross-attention: conductance latent (queries) x environment tokens (keys/values).

    No softmax — scores can be negative (physically meaningful: driving force Vm - E
    can be negative). Each conductance dim queries the environment to determine its
    current contribution.
    """

    def __init__(self, cond_dim: int, n_env: int, attn_dim: int, d_v: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(attn_dim)
        self.e_q = nn.Parameter(torch.randn(cond_dim, attn_dim) * 0.1)
        self.e_k = nn.Parameter(torch.randn(n_env, attn_dim) * 0.1)
        self.e_v = nn.Parameter(torch.randn(n_env, d_v) * 0.1)

    def forward(self, conductance_latent: Tensor, env_normalized: Tensor) -> Tensor:
        # i=batch, j=cond_dim, k=attn_dim, l=n_env, m=d_v
        Q = torch.einsum('ij,jk->ijk', conductance_latent, self.e_q)   # (B, C, d)
        K = torch.einsum('il,lk->ilk', env_normalized, self.e_k)        # (B, 9, d)
        V = torch.einsum('il,lm->ilm', env_normalized, self.e_v)        # (B, 9, d_v)
        scores = torch.einsum('ijk,ilk->ijl', Q, K) * self.scale        # (B, C, 9)
        attended = torch.einsum('ijl,ilm->ijm', scores, V).squeeze(-1)  # (B, C)
        return attended


class IonicStage2(nn.Module):
    """Cross-attention readout: conductance latent + environment -> I_ion.

    Args:
        cond_dim:   Conductance latent dimension (default 8).
        n_env:      Number of environment tokens (default 9).
        attn_dim:   Attention projection dimension for Q/K (default 4).
        d_v:        Value dimension (default 1).
        mlp_hidden: Output MLP hidden dimension (default 4).
    """

    def __init__(
        self,
        cond_dim: int = 8,
        n_env: int = 9,
        attn_dim: int = 4,
        d_v: int = 1,
        mlp_hidden: int = 4,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_env = n_env
        self.scale = 1.0 / math.sqrt(attn_dim)

        # --- Attention ---
        self.conductance_attention = ConductanceAttention(cond_dim, n_env, attn_dim, d_v)

        # --- Output MLP ---
        self.output_mlp = nn.Sequential(
            nn.Linear(cond_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

        # Zero-init biases so zero conductance -> zero output
        nn.init.zeros_(self.output_mlp[0].bias)
        nn.init.zeros_(self.output_mlp[2].bias)

    def forward(self, conductance_latent: Tensor, env_normalized: Tensor) -> Tensor:
        """Compute I_ion from conductance latent and normalized environment.

        Args:
            conductance_latent: (B, cond_dim) or (cond_dim,) conductance state.
            env_normalized:     (B, n_env) or (n_env,) normalized environment tokens.

        Returns:
            I_ion: (B,) or scalar.
        """
        unbatched = conductance_latent.dim() == 1
        if unbatched:
            conductance_latent = conductance_latent.unsqueeze(0)
            env_normalized = env_normalized.unsqueeze(0)

        B = conductance_latent.shape[0]
        assert conductance_latent.shape == (B, self.cond_dim), (
            f"Expected conductance_latent shape (B, {self.cond_dim}), "
            f"got {conductance_latent.shape}"
        )
        assert env_normalized.shape == (B, self.n_env), (
            f"Expected env_normalized shape (B, {self.n_env}), "
            f"got {env_normalized.shape}"
        )

        attended = self.conductance_attention(conductance_latent, env_normalized)  # (B, C)
        I_ion = self.output_mlp(attended).squeeze(-1)                              # (B,)

        if unbatched:
            I_ion = I_ion.squeeze(0)

        return I_ion
