"""Stage 1: Attention + MLP + Compression for ionic surrogate v3.

State evolution engine (off critical path). Neural ODE dynamics function.

dzdt(z, Vm) -> dz/dt:
    voltage_attention(CARRIED_DIM, d=ATTN_DIM) to [Vm]
    -> delta = z_mid - z (attention rate)
    -> split: ionic_delta (IONIC_DIM) + conc_delta (CONC_DIM)
    -> ionic: Pre-RMSNorm -> ionic_mixing_mlp -> residual_bypass(ionic_delta, correction)
    -> cat([ionic_rate, conc_delta]) = dz/dt

forward(z, Vm) -> compression + scaffold (no dynamics):
    -> _compress(z) -> conductance_latent
    -> scaffold decoders (training only):
        ionic_state_decoder(IONIC_DIM -> N_IONIC_TARGETS): all recoverable ionic states
        gate_conductance_decoder(COND_DIM -> N_CONDUCTANCE_TARGETS): effective gate products
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_


# === Hyperparameter defaults (small TTP06 config) ===
IONIC_DIM = 16
CONC_DIM = 4
CARRIED_DIM = IONIC_DIM + CONC_DIM  # 20
ATTN_DIM = 4
COND_DIM = 8
MLP_HIDDEN = 16
COMP_H1 = 12
COMP_H2 = 12
N_IONIC_TARGETS = 14     # 12 HH gates (m-Xs) + RR + Ca_SR = 14 ionic state targets
N_CONDUCTANCE_TARGETS = 5 # G_Na(m³hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1Xr2), G_Ks(Xs²)
ALPHA_INIT = -5.0  # sigmoid(-5) ~ 0.007, near-pure residual at init
BETA_INIT = -5.0


def rms_norm(x: Tensor) -> Tensor:
    """Zero-parameter RMSNorm. Normalizes by root-mean-square, no learned scale."""
    return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)


def residual_bypass(base: Tensor, correction: Tensor, logit: Tensor) -> Tensor:
    """Additive residual bypass. Identity path unconditional; correction adds on top.

    alpha -> 0: pass base unchanged. alpha -> 1: add full correction to base.
    For NODE: derivative = attention_rate + alpha * mlp_correction (clean additive sum).
    """
    alpha = torch.sigmoid(logit)
    return base + alpha * correction


class VoltageAttention(nn.Module):
    """Per-dim voltage-gated attention over carried state.

    Each carried state dim independently queries [Vm] to produce a gate
    and target. Contractive update: z = old + gate * (target - old).
    dt removed for NODE pivot — vector field f(z,V) must be dt-independent.
    """

    def __init__(self, carried_dim: int, attn_dim: int):
        super().__init__()
        self.W_q = nn.Parameter(torch.empty(carried_dim, attn_dim))
        self.W_k = nn.Linear(1, attn_dim, bias=False)
        self.W_v = nn.Linear(1, attn_dim, bias=False)
        self.W_out = nn.Parameter(torch.empty(attn_dim, carried_dim))
        self.scale = 1.0 / math.sqrt(attn_dim)
        xavier_uniform_(self.W_q)
        xavier_uniform_(self.W_k.weight)
        xavier_uniform_(self.W_v.weight)
        xavier_uniform_(self.W_out)

    def forward(self, carried_state: Tensor, Vm: Tensor) -> Tensor:
        x = Vm.unsqueeze(-1)                                           # (B, 1)
        k = self.W_k(x)                                                # (B, d)
        v = self.W_v(x)                                                # (B, d)
        q = torch.einsum('ij,jk->ijk', carried_state, self.W_q)       # (B, D, d)
        score = torch.einsum('ijk,ik->ij', q, k) * self.scale         # (B, D)
        gate = torch.sigmoid(score)                                    # (B, D)
        target = v @ self.W_out                                        # (B, D)
        return carried_state + gate * (target - carried_state)         # (B, D)


class IonicStage1(nn.Module):
    """Stage 1: attention + ionic mixing + gate conductance compression.

    Takes carried_state (ionic + concentrations), produces updated state,
    conductance latent, and scaffold predictions.

    Concentrations split off AFTER attention, BEFORE MLP -- they do NOT
    go through the ionic mixing MLP. Only ionic dims get the MLP correction.

    Args:
        ionic_dim: Latent ionic state dims (default 16).
        conc_dim: Explicit concentration dims (default 4).
        attn_dim: Attention projection dimension (default 4).
        cond_dim: Conductance latent after compression (default 8).
        mlp_hidden: Ionic mixing MLP hidden dim (default 16).
        comp_h1: Compression first hidden layer (default 12).
        comp_h2: Compression second hidden layer (default 12).
        n_ionic_targets: State decoder targets (default 14: 12 HH gates + RR + Ca_SR).
        n_conductance_targets: Conductance decoder targets (default 5: effective gate products).
        scaffold: Whether to include training scaffold decoders.
    """

    def __init__(
        self,
        ionic_dim: int = IONIC_DIM,
        conc_dim: int = CONC_DIM,
        attn_dim: int = ATTN_DIM,
        cond_dim: int = COND_DIM,
        mlp_hidden: int = MLP_HIDDEN,
        comp_h1: int = COMP_H1,
        comp_h2: int = COMP_H2,
        n_ionic_targets: int = N_IONIC_TARGETS,
        n_conductance_targets: int = N_CONDUCTANCE_TARGETS,
        scaffold: bool = True,
    ):
        super().__init__()
        self.ionic_dim = ionic_dim
        self.conc_dim = conc_dim
        self.carried_dim = ionic_dim + conc_dim
        self.cond_dim = cond_dim

        # --- Attention ---
        self.voltage_attention = VoltageAttention(self.carried_dim, attn_dim)

        # --- Ionic mixing MLP (cross-dim communication on ionic dims) ---
        self.ionic_mixing_mlp = nn.Sequential(
            nn.Linear(ionic_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, ionic_dim),
        )
        self.ionic_mixing_logit = nn.Parameter(torch.full((ionic_dim,), ALPHA_INIT))

        # --- Gate conductance compression (full carried_state → effective conductances) ---
        self.gate_conductance_linear = nn.Linear(self.carried_dim, cond_dim, bias=False)
        self.gate_conductance_mlp = nn.Sequential(
            nn.Linear(self.carried_dim, comp_h1),
            nn.GELU(),
            nn.Linear(comp_h1, comp_h2),
            nn.GELU(),
            nn.Linear(comp_h2, cond_dim),
        )
        self.gate_conductance_logit = nn.Parameter(torch.full((cond_dim,), BETA_INIT))

        # --- Scaffold decoders (training only) ---
        if scaffold:
            # Decoder 1: ionic_state → full ionic set (gates + Ca_SR + RR)
            self.ionic_state_decoder = nn.Linear(ionic_dim, n_ionic_targets)
            # Decoder 2: conductance_latent → effective gate products (m³hj, dff2fCass, etc.)
            self.gate_conductance_decoder = nn.Linear(cond_dim, n_conductance_targets)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for MLP and compression weights."""
        xavier_uniform_(self.ionic_mixing_mlp[0].weight)
        xavier_uniform_(self.ionic_mixing_mlp[2].weight)
        xavier_uniform_(self.gate_conductance_linear.weight)
        xavier_uniform_(self.gate_conductance_mlp[0].weight)
        xavier_uniform_(self.gate_conductance_mlp[2].weight)
        xavier_uniform_(self.gate_conductance_mlp[4].weight)

    def _compress(self, carried_state: Tensor) -> Tensor:
        """Run gate conductance compression on carried_state. No dynamics."""
        linear_path = self.gate_conductance_linear(carried_state)
        nonlinear_path = self.gate_conductance_mlp(carried_state)
        return residual_bypass(linear_path, nonlinear_path, self.gate_conductance_logit)

    def dzdt(self, z: Tensor, Vm: Tensor) -> Tensor:
        """Compute dz/dt for ODE integration. No compression, no scaffold decoders.

        Returns a RATE (not a displacement). ODE solver integrates this;
        Euler inference does z + dt * dzdt(z, V).

        Args:
            z: carried_state (B, carried_dim) or (carried_dim,)
            Vm: membrane voltage (B,) or scalar
        Returns:
            dz_dt: rate of change, same shape as z
        """
        squeezed = z.dim() == 1
        if squeezed:
            z = z.unsqueeze(0)
            Vm = Vm.view(1)

        z_mid = self.voltage_attention(z, Vm)           # z + gate*(target-z)
        delta = z_mid - z                               # attention rate for all dims

        ionic_delta = delta[:, :self.ionic_dim]
        conc_delta = delta[:, self.ionic_dim:]

        correction = self.ionic_mixing_mlp(rms_norm(z_mid[:, :self.ionic_dim]))
        ionic_rate = residual_bypass(ionic_delta, correction, self.ionic_mixing_logit)

        dz_dt = torch.cat([ionic_rate, conc_delta], dim=-1)

        if squeezed:
            dz_dt = dz_dt.squeeze(0)
        return dz_dt

    def forward(
        self, carried_state: Tensor, Vm: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """Run compression + scaffold on carried_state. Does NOT advance state.

        State advancement is done by IonicNODE.euler_step() or odeint_adjoint.
        This method produces conductance_latent and scaffold predictions from
        an already-advanced state.

        Args:
            carried_state: Carried state (B, carried_dim) or (carried_dim,).
            Vm: Membrane voltage (B,) or scalar.

        Returns:
            carried_state: Same as input (no dynamics).
            conductance_latent: Compressed gate conductance (B, cond_dim) or (cond_dim,).
            concentrations: Concentration slice (B, conc_dim) or (conc_dim,).
            ionic_state_pred: Ionic state decoder predictions, or None if no scaffold.
            conductance_pred: Conductance decoder predictions, or None if no scaffold.
        """
        assert carried_state.shape[-1] == self.carried_dim, (
            f"Expected carried_state dim {self.carried_dim}, got {carried_state.shape[-1]}"
        )

        # Handle unbatched input
        squeezed = carried_state.dim() == 1
        if squeezed:
            carried_state = carried_state.unsqueeze(0)
            Vm = Vm.view(1)

        ionic_new = carried_state[:, :self.ionic_dim]
        conc_new = carried_state[:, self.ionic_dim:]

        # === Gate conductance compression (full carried_state → cond_dim) ===
        conductance_latent = self._compress(carried_state)

        # === Scaffold decoders (training only) ===
        ionic_state_pred = None
        conductance_pred = None
        if hasattr(self, "ionic_state_decoder"):
            ionic_state_pred = self.ionic_state_decoder(ionic_new)
            conductance_pred = self.gate_conductance_decoder(conductance_latent)

        # Restore unbatched shape
        if squeezed:
            carried_state = carried_state.squeeze(0)
            conductance_latent = conductance_latent.squeeze(0)
            conc_new = conc_new.squeeze(0)
            if ionic_state_pred is not None:
                ionic_state_pred = ionic_state_pred.squeeze(0)
                conductance_pred = conductance_pred.squeeze(0)

        return (
            carried_state,
            conductance_latent,
            conc_new,
            ionic_state_pred,
            conductance_pred,
        )

    def remove_scaffold(self) -> None:
        """Remove scaffold decoders for production inference. Idempotent."""
        if hasattr(self, "ionic_state_decoder"):
            del self.ionic_state_decoder
        if hasattr(self, "gate_conductance_decoder"):
            del self.gate_conductance_decoder

    def inference_param_count(self) -> int:
        """Count parameters excluding scaffold decoders."""
        total = sum(p.numel() for p in self.parameters())
        if hasattr(self, "ionic_state_decoder"):
            total -= sum(p.numel() for p in self.ionic_state_decoder.parameters())
        if hasattr(self, "gate_conductance_decoder"):
            total -= sum(p.numel() for p in self.gate_conductance_decoder.parameters())
        return total
