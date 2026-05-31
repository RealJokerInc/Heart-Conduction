"""Stage 1: Unified StateRateMLP + Compression for ionic surrogate v4 (Session 27).

State evolution engine (off critical path). Neural ODE dynamics function.

v4 changes (2026-04-19):
    - IONIC_DIM 16 -> 20 (added slack for slow-variable tracking)
    - Unified rate path: StateRateMLP replaces IonicRateMLP + conc_kan
      (5-layer MLP with pre-LayerNorm + gated full-path linear skip)
    - ionic_state_decoder.bias frozen at TTP06 physiological rest
      so decoder(z=0) = rest by construction

dzdt(z, Vm) -> dz/dt:
    state_rate_mlp(z, Vm) -> rate   (single call, full-state in, full-rate out)

forward(z, Vm) -> compression + scaffold (no dynamics):
    -> _compress(z) -> conductance_latent
    -> scaffold decoders (training only):
        ionic_state_decoder(IONIC_DIM -> N_IONIC_TARGETS)  bias frozen at rest
        gate_conductance_decoder(COND_DIM -> N_CONDUCTANCE_TARGETS)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_


# === Hyperparameter defaults (small TTP06 config, v4) ===
IONIC_DIM = 20
CONC_DIM = 4
CARRIED_DIM = IONIC_DIM + CONC_DIM  # 24
COND_DIM = 8
H_STATE_MLP = 32
COMP_H1 = 12
COMP_H2 = 12
N_IONIC_TARGETS = 14     # 12 HH gates (m-Xs) + RR + Ca_SR
N_CONDUCTANCE_TARGETS = 5 # G_Na(m³hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1Xr2), G_Ks(Xs²)
BETA_INIT = -5.0


# === Physics-informed rest state (Session 27, 2026-04-19) ===
# TTP06 physiological rest ionic_state (14 dims, from BCL=2000 t=0 sample).
# Frozen as ionic_state_decoder.bias so decoder(z=0) = rest by construction.
# Latent semantics: z represents deviation from rest.
# Order matches V3Preprocessor scaffold-decoder targets:
#   [m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs, RR, CaSR].
TTP06_REST_IONIC_STATE = torch.tensor([
    0.001720,  # m
    0.744400,  # h
    0.704500,  # j
    0.000000,  # r
    0.999998,  # s
    0.000034,  # d
    0.788800,  # f
    0.975500,  # f2
    0.995300,  # fCass
    0.006210,  # Xr1
    0.471200,  # Xr2
    0.001720,  # Xs
    0.907300,  # RR
    3.640000,  # CaSR
], dtype=torch.float64)


def residual_bypass(base: Tensor, correction: Tensor, logit: Tensor) -> Tensor:
    """Additive residual bypass. Used by gate conductance compression."""
    alpha = torch.sigmoid(logit)
    return base + alpha * correction


class StateRateMLP(nn.Module):
    """Unified rate predictor: (z_full, Vm) -> dz/dt. Replaces IonicRateMLP + conc_kan.

    Architecture (Session 27):
      - 5-layer MLP (stem + 4 hidden), width H_STATE_MLP=32, GELU
      - Pre-LayerNorm on each hidden Linear's input (LLaMA style)
      - Linear readout with zero-init weight and bias (rate=0 at init -> ODE stable)
      - Gated full-path linear skip (input -> rate), alpha=sigmoid(logit), logit init BETA_INIT=-5 -> alpha~0.007 (near-dormant; logit=0 / alpha=0.5 caused 1e90 divergence on 2026-04-19 flash training)

    Input dim: carried_dim + 1 (for Vm).  Output dim: carried_dim.
    Default shape: 25 -> 32 -> 32 -> 32 -> 32 -> 32 -> 24.
    """

    def __init__(self, carried_dim: int = CARRIED_DIM, hidden: int = H_STATE_MLP):
        super().__init__()
        in_dim = carried_dim + 1
        self.fc1 = nn.Linear(in_dim, hidden)    # stem
        self.fc2 = nn.Linear(hidden, hidden)    # hidden 1
        self.fc3 = nn.Linear(hidden, hidden)    # hidden 2
        self.fc4 = nn.Linear(hidden, hidden)    # hidden 3
        self.fc5 = nn.Linear(hidden, hidden)    # hidden 4
        # Pre-norm on each hidden Linear's input; five norms (one before readout)
        self.ln1 = nn.LayerNorm(hidden)         # before fc2
        self.ln2 = nn.LayerNorm(hidden)         # before fc3
        self.ln3 = nn.LayerNorm(hidden)         # before fc4
        self.ln4 = nn.LayerNorm(hidden)         # before fc5
        self.ln5 = nn.LayerNorm(hidden)         # before readout
        # Readout: zero-init for ODE stability
        self.readout = nn.Linear(hidden, carried_dim)
        # Gated full-path skip. Init logit = BETA_INIT (-5) -> alpha ~= 0.007,
        # skip is near-dormant at init. Same pattern as gate_conductance_logit:
        # gives the model a learnable linear-vs-deep knob but keeps the ODE stable
        # at init (rate ~= 0 from both deep-zero-init and skip-near-zero-gate).
        self.skip = nn.Linear(in_dim, carried_dim, bias=False)
        self.skip_logit = nn.Parameter(torch.full((carried_dim,), BETA_INIT))

    def forward(self, z: Tensor, Vm: Tensor) -> Tensor:
        x = torch.cat([z, Vm.unsqueeze(-1)], dim=-1)
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(self.ln1(h)))
        h = F.gelu(self.fc3(self.ln2(h)))
        h = F.gelu(self.fc4(self.ln3(h)))
        h = F.gelu(self.fc5(self.ln4(h)))
        rate_deep = self.readout(self.ln5(h))
        rate_skip = self.skip(x)
        alpha = torch.sigmoid(self.skip_logit)
        return rate_deep + alpha * rate_skip


class IonicStage1(nn.Module):
    """Stage 1 (v4): unified rate predictor + gate conductance compression.

    Args:
        ionic_dim: Latent ionic state dims (default 20).
        conc_dim: Explicit concentration dims (default 4).
        cond_dim: Conductance latent after compression (default 8).
        hidden: StateRateMLP hidden width (default 32).
        comp_h1: Compression first hidden layer (default 12).
        comp_h2: Compression second hidden layer (default 12).
        n_ionic_targets: State decoder targets (default 14).
        n_conductance_targets: Conductance decoder targets (default 5).
        scaffold: Whether to include training scaffold decoders.
    """

    def __init__(
        self,
        ionic_dim: int = IONIC_DIM,
        conc_dim: int = CONC_DIM,
        cond_dim: int = COND_DIM,
        hidden: int = H_STATE_MLP,
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

        # --- Unified state rate predictor: (z_full, Vm) -> dz/dt ---
        self.state_rate_mlp = StateRateMLP(self.carried_dim, hidden)

        # --- Input centering buffer (Session 28, v4 hotfix 2026-04-20) ---
        # The skip path's raw K_i=138 / V=-85 inputs produced ~150mV * 0.42 Xavier
        # weight = 63-magnitude rate contributions during dopri5 integration on
        # the t1 training launch, driving val_loss to 1.88e9 / epoch 0 with only
        # ~24% reduction per epoch. Subtracting a fixed reference "rest input"
        # gives StateRateMLP a centered input (all slots near 0 at rest) without
        # adding any learnable parameters. Non-trainable buffer -> doesn't affect
        # gradient flow and doesn't break checkpoint compatibility (except that
        # v3 checkpoints already can't load due to dim changes).
        # Use the default dtype so ``.double()`` (called by Stage1 constructors in
        # most call sites) cascades; tests that instantiate without ``.double()``
        # still work because the buffer matches the model's float32 parameters.
        _ref_dtype = torch.get_default_dtype()
        INIT_CONC_ORDER = torch.tensor(
            [10.0, 138.0, 1e-4, 2e-4], dtype=_ref_dtype
        )  # [Na_i, K_i, Ca_i, Ca_ss] — matches node_rollout.INIT_CONC
        V_REST_MV = -85.23
        input_ref = torch.zeros(self.carried_dim + 1, dtype=_ref_dtype)
        input_ref[self.ionic_dim:self.carried_dim] = INIT_CONC_ORDER[:conc_dim]
        input_ref[self.carried_dim] = V_REST_MV
        self.register_buffer("input_ref", input_ref)

        # --- Gate conductance compression (full carried_state -> effective conductances) ---
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
            self.ionic_state_decoder = nn.Linear(ionic_dim, n_ionic_targets)
            self.gate_conductance_decoder = nn.Linear(cond_dim, n_conductance_targets)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights.

        StateRateMLP: Xavier on hidden Linears and skip; zero-init on readout
        (rate ~= 0 at init -> ODE stable). skip_logit is constructor-initialized
        to BETA_INIT (-5), giving alpha=sigmoid(-5)~=0.007; NOT zero. Zero-init
        (alpha=0.5) with Xavier skip + V=-85 mV produced rate ~20/dim and an
        ODE blowup to 1e90 on the 2026-04-19 flash training. Do not re-init
        skip_logit here.

        Compression: Xavier on both paths; logit init BETA_INIT (~-5 -> alpha~0.007).

        Scaffold (Session 27): ionic_state_decoder.bias frozen at TTP06 rest
        so decoder(z=0) = rest by construction; latent = deviation from rest.
        """
        # StateRateMLP hidden Linears + skip
        for fc in (self.state_rate_mlp.fc1, self.state_rate_mlp.fc2,
                   self.state_rate_mlp.fc3, self.state_rate_mlp.fc4,
                   self.state_rate_mlp.fc5):
            xavier_uniform_(fc.weight)
        xavier_uniform_(self.state_rate_mlp.skip.weight)
        # Zero-init readout (both weight and bias) -> deep contribution is 0 at init
        nn.init.zeros_(self.state_rate_mlp.readout.weight)
        nn.init.zeros_(self.state_rate_mlp.readout.bias)

        # Compression
        xavier_uniform_(self.gate_conductance_linear.weight)
        xavier_uniform_(self.gate_conductance_mlp[0].weight)
        xavier_uniform_(self.gate_conductance_mlp[2].weight)
        xavier_uniform_(self.gate_conductance_mlp[4].weight)

        # Physics-informed rest attractor: pin decoder bias to TTP06 rest
        self.pin_rest_bias()

    def pin_rest_bias(self) -> None:
        """Pin ionic_state_decoder.bias to TTP06 physiological rest and freeze.

        Makes decoder(z=0) = TTP06_REST_IONIC_STATE by construction. Latent
        semantics become "deviation from rest".

        CALL AFTER load_state_dict() -- torch.nn.Module.load_state_dict overwrites
        the bias with the checkpoint value, silently breaking the rest guarantee.
        Idempotent; no-op if scaffold is absent.
        """
        if not hasattr(self, "ionic_state_decoder"):
            return
        assert self.ionic_state_decoder.out_features == TTP06_REST_IONIC_STATE.numel(), (
            f"Decoder out_features ({self.ionic_state_decoder.out_features}) "
            f"does not match TTP06_REST_IONIC_STATE length "
            f"({TTP06_REST_IONIC_STATE.numel()})"
        )
        with torch.no_grad():
            self.ionic_state_decoder.bias.copy_(
                TTP06_REST_IONIC_STATE.to(
                    dtype=self.ionic_state_decoder.bias.dtype,
                    device=self.ionic_state_decoder.bias.device,
                )
            )
        self.ionic_state_decoder.bias.requires_grad_(False)

    def _compress(self, carried_state: Tensor) -> Tensor:
        """Run gate conductance compression on carried_state. No dynamics."""
        linear_path = self.gate_conductance_linear(carried_state)
        nonlinear_path = self.gate_conductance_mlp(carried_state)
        return residual_bypass(linear_path, nonlinear_path, self.gate_conductance_logit)

    def dzdt(self, z: Tensor, Vm: Tensor) -> Tensor:
        """Compute dz/dt for ODE integration. Unified: one call to StateRateMLP.

        Returns a RATE. ODE solver integrates this; Euler inference does
        z + dt * dzdt(z, V).

        Input centering (Session 28): both the deep LayerNorm and the linear
        skip see a centered input, so the K_i=138 / V=-85 spikes don't drive
        integrator blow-up at init. The buffer is non-trainable so gradients
        and checkpoint shape are unchanged.

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

        z_centered = z - self.input_ref[: self.carried_dim]
        Vm_centered = Vm - self.input_ref[self.carried_dim]
        dz_dt = self.state_rate_mlp(z_centered, Vm_centered)

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
        """
        assert carried_state.shape[-1] == self.carried_dim, (
            f"Expected carried_state dim {self.carried_dim}, got {carried_state.shape[-1]}"
        )

        squeezed = carried_state.dim() == 1
        if squeezed:
            carried_state = carried_state.unsqueeze(0)
            Vm = Vm.view(1)

        ionic_new = carried_state[:, :self.ionic_dim]
        conc_new = carried_state[:, self.ionic_dim:]

        conductance_latent = self._compress(carried_state)

        ionic_state_pred = None
        conductance_pred = None
        if hasattr(self, "ionic_state_decoder"):
            ionic_state_pred = self.ionic_state_decoder(ionic_new)
            conductance_pred = self.gate_conductance_decoder(conductance_latent)

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
