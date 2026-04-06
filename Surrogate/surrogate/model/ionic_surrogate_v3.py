"""IonicSurrogateV3 orchestrator: Stage 1 + Nernst + Stage 2.

Combines the three v3 components into the full autoregressive ionic surrogate.
Key design constraint: Stage 2 reads PREVIOUS conductance latent and PREVIOUS
concentrations, NOT Stage 1's current-step output. This mirrors the operator
splitting in the bidomain simulator where I_ion at time t depends on the state
at time t, while Stage 1 advances the state to t+1.

Forward signature:
    (carried_state, Vm, dt, cond_lat_prev, conc_prev) -> dict

Autoregressive loop pattern:
    for each timestep:
        out = model(carried, Vm, dt, cond_lat_prev, conc_prev)
        carried     = out['carried_state']
        cond_lat_prev = out['conductance_latent']
        conc_prev   = out['concentrations']
        I_ion       = out['I_ion']
        # ... use I_ion for diffusion step ...
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .nernst import NernstComputer
from .stage1 import IonicStage1
from .stage2 import IonicStage2

# Re-export default hyperparameters from stage1 for convenience
from .stage1 import (
    IONIC_DIM,
    CONC_DIM,
    ATTN_DIM,
    COND_DIM,
    MLP_HIDDEN,
    COMP_H1,
    COMP_H2,
    N_IONIC_TARGETS,
    N_CONDUCTANCE_TARGETS,
)
from .stage2 import IonicStage2


class IonicSurrogateV3(nn.Module):
    """Full v3 ionic surrogate: Stage 1 (state evolution) + Nernst + Stage 2 (current readout).

    Stage 1 advances the carried state to t+1 and produces new conductance
    latent and concentrations. Stage 2 computes I_ion from the PREVIOUS step's
    conductance latent and concentrations (not Stage 1's output). This matches
    the simulator's operator splitting: I_ion(t) depends on state(t), while
    state(t+1) is computed in parallel.

    Args:
        ionic_dim: Latent ionic state dims (default 16).
        conc_dim: Explicit concentration dims (default 4).
        attn_dim: Stage 1 attention projection dimension (default 4).
        cond_dim: Conductance latent after compression (default 8).
        mlp_hidden: Ionic mixing MLP hidden dim (default 16).
        comp_h1: Compression first hidden layer (default 12).
        comp_h2: Compression second hidden layer (default 12).
        n_ionic_targets: Scaffold target: ionic states (default 14).
        n_conductance_targets: Scaffold target: conductance products (default 5).
        scaffold: Whether to include training scaffold decoders in Stage 1.
        n_env: Number of environment tokens (default 9).
        stage2_attn: Stage 2 attention dim for Q/K (default 4).
        stage2_dv: Stage 2 value dim (default 1).
        stage2_mlp_h: Stage 2 output MLP hidden dim (default 4).
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
        n_env: int = 9,
        stage2_attn: int = 4,
        stage2_dv: int = 1,
        stage2_mlp_h: int = 4,
    ):
        super().__init__()
        self.ionic_dim = ionic_dim
        self.conc_dim = conc_dim
        self.carried_dim = ionic_dim + conc_dim
        self.cond_dim = cond_dim

        # --- Sub-modules ---
        self.stage1 = IonicStage1(
            ionic_dim=ionic_dim,
            conc_dim=conc_dim,
            attn_dim=attn_dim,
            cond_dim=cond_dim,
            mlp_hidden=mlp_hidden,
            comp_h1=comp_h1,
            comp_h2=comp_h2,
            n_ionic_targets=n_ionic_targets,
            n_conductance_targets=n_conductance_targets,
            scaffold=scaffold,
        )
        self.nernst = NernstComputer()
        self.stage2 = IonicStage2(
            cond_dim=cond_dim,
            n_env=n_env,
            attn_dim=stage2_attn,
            d_v=stage2_dv,
            mlp_hidden=stage2_mlp_h,
        )

    def forward(
        self,
        carried_state: Tensor,
        Vm: Tensor,
        dt: Tensor,
        cond_lat_prev: Tensor,
        conc_prev: Tensor,
    ) -> Dict[str, Any]:
        """One autoregressive step.

        Args:
            carried_state: Previous carried state (B, carried_dim) or (carried_dim,).
            Vm: Membrane voltage (B,) or scalar.
            dt: Time step (B,) or scalar.
            cond_lat_prev: PREVIOUS step's conductance latent (B, cond_dim) or (cond_dim,).
            conc_prev: PREVIOUS step's concentrations (B, conc_dim) or (conc_dim,).
                       Order: [Na_i, K_i, Ca_i, Ca_ss].

        Returns:
            dict with keys:
                carried_state: Updated carried state, same shape as input.
                conductance_latent: New conductance latent (for next step's Stage 2).
                concentrations: New concentrations (for next step's Stage 2).
                I_ion: Ionic current scalar (B,) or scalar.
                ionic_state_pred: Ionic state decoder predictions (or None).
                conductance_pred: Conductance decoder predictions (or None).
        """
        # --- Shape assertions ---
        assert carried_state.shape[-1] == self.carried_dim, (
            f"Expected carried_state dim {self.carried_dim}, got {carried_state.shape[-1]}"
        )
        assert cond_lat_prev.shape[-1] == self.cond_dim, (
            f"Expected cond_lat_prev dim {self.cond_dim}, got {cond_lat_prev.shape[-1]}"
        )
        assert conc_prev.shape[-1] == self.conc_dim, (
            f"Expected conc_prev dim {self.conc_dim}, got {conc_prev.shape[-1]}"
        )

        # --- Stage 1: compression + scaffold (no dynamics — state advanced by IonicNODE) ---
        # dt kept in V3 signature for backward compat but not passed to Stage1
        cs_new, cond_new, conc_new, ionic_state_pred, conductance_pred = self.stage1(
            carried_state, Vm
        )

        # --- Nernst on PREV concentrations ---
        # conc_prev order: [Na_i, K_i, Ca_i, Ca_ss]
        # Handle both batched (B, 4) and unbatched (4,)
        if conc_prev.dim() == 1:
            Na_i_prev = conc_prev[0]
            K_i_prev = conc_prev[1]
            Ca_i_prev = conc_prev[2]
            Ca_ss_prev = conc_prev[3]
        else:
            Na_i_prev = conc_prev[:, 0]
            K_i_prev = conc_prev[:, 1]
            Ca_i_prev = conc_prev[:, 2]
            Ca_ss_prev = conc_prev[:, 3]

        E_Na, E_K, E_Ca, E_Ks = self.nernst(Na_i_prev, K_i_prev, Ca_i_prev)

        # --- Environment normalization ---
        env_norm = self.nernst.normalize_environment(
            Vm, E_Na, E_K, E_Ca, E_Ks,
            Na_i_prev, K_i_prev, Ca_i_prev, Ca_ss_prev,
        )

        # --- Stage 2: current readout from PREV state ---
        I_ion = self.stage2(cond_lat_prev, env_norm)

        return {
            "carried_state": cs_new,
            "conductance_latent": cond_new,
            "concentrations": conc_new,
            "I_ion": I_ion,
            "ionic_state_pred": ionic_state_pred,
            "conductance_pred": conductance_pred,
        }

    def remove_scaffold(self) -> None:
        """Remove training scaffold decoders. Delegates to Stage 1. Idempotent."""
        self.stage1.remove_scaffold()

    def inference_param_count(self) -> int:
        """Count inference parameters (Stage 1 + Stage 2, Nernst has 0 learned params)."""
        return self.stage1.inference_param_count() + sum(
            p.numel() for p in self.stage2.parameters()
        )
