"""
Optimizer V1 — Batched Ionic Step (PHAS13 / MHAS13)

Simulates M parameter sets simultaneously. Each cell has its own
conductance scaling but shares the same gating kinetics (which depend
only on V, not conductances).

Supports two IK1 modes:
- 'phas13': native Paci IK1 formulation (spontaneous model)
- 'mhas13': TTP06 IK1 formulation + g_f=0 (matured, quiescent)

The conductance tensor has shape (M, 14) with columns ordered as in
PHAS13_REGISTRY (tier 1-3). Non-tuned parameters use published defaults.
"""

import sys
import os
import torch
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'Monodomain', 'Engine_V5.4'))

from cardiac_sim.ionic.phas13.parameters import (
    StateIndex, PHAS13Parameters, V_REST,
    get_initial_state as _get_initial_state,
)
from cardiac_sim.ionic.phas13.gating import (
    rush_larsen,
    INa_m_inf, INa_m_tau, INa_h_inf, INa_h_tau, INa_j_inf, INa_j_tau,
    ICaL_d_inf, ICaL_d_tau, ICaL_f1_inf, ICaL_f1_tau,
    ICaL_f2_inf, ICaL_f2_tau, ICaL_fCa_inf, FCAL_TAU,
    IKr_Xr1_inf, IKr_Xr1_tau, IKr_Xr2_inf, IKr_Xr2_tau,
    IKs_Xs_inf, IKs_Xs_tau,
    Ito_q_inf, Ito_q_tau, Ito_r_inf, Ito_r_tau,
    If_Xf_inf, If_Xf_tau,
)
from cardiac_sim.ionic.phas13.currents import (
    E_Na, E_K, E_Ca, E_Ks, RTONF, R, T, F,
)
from cardiac_sim.ionic.phas13.calcium import (
    i_up, i_rel, i_leak, buffering_factor_cyt, buffering_factor_sr,
    update_g_rel,
)
from cardiac_sim.ionic.phas13.gating import safe_exp

from .config import PHAS13_REGISTRY, get_param_names


# ============================================================================
# Conductance indices — column positions in the (M, 14) conductance tensor
# ============================================================================

# All 14 tunable params in registry order
_ALL_PARAM_NAMES = list(PHAS13_REGISTRY.keys())
C_g_Na   = _ALL_PARAM_NAMES.index('g_Na')
C_g_CaL  = _ALL_PARAM_NAMES.index('g_CaL')
C_g_Kr   = _ALL_PARAM_NAMES.index('g_Kr')
C_g_Ks   = _ALL_PARAM_NAMES.index('g_Ks')
C_g_K1   = _ALL_PARAM_NAMES.index('g_K1')
C_g_to   = _ALL_PARAM_NAMES.index('g_to')
C_kNaCa  = _ALL_PARAM_NAMES.index('kNaCa')
C_PNaK   = _ALL_PARAM_NAMES.index('PNaK')
C_g_pCa  = _ALL_PARAM_NAMES.index('g_pCa')
C_VmaxUp = _ALL_PARAM_NAMES.index('VmaxUp')
C_g_f    = _ALL_PARAM_NAMES.index('g_f')
C_g_bNa  = _ALL_PARAM_NAMES.index('g_bNa')
C_g_bCa  = _ALL_PARAM_NAMES.index('g_bCa')
C_V_leak = _ALL_PARAM_NAMES.index('V_leak')

N_ALL_PARAMS = len(_ALL_PARAM_NAMES)


def build_conductance_tensor(theta_batch: torch.Tensor, tier: int,
                             dtype=torch.float64,
                             device='cpu',
                             ionic_model: str = 'mhas13') -> torch.Tensor:
    """
    Convert (M, n_tier_params) scaling factors to (M, 14) actual conductance values.

    Non-tuned parameters (tiers above `tier`) use published defaults (scale=1.0).
    For MHAS13: g_f is forced to 0 (If suppressed).
    """
    M = theta_batch.shape[0]
    # Start with all-ones (published defaults)
    cond = torch.ones(M, N_ALL_PARAMS, dtype=dtype, device=device)

    # Fill in the tuned parameters
    tier_names = get_param_names(tier)
    for col_in_tier, name in enumerate(tier_names):
        col_in_all = _ALL_PARAM_NAMES.index(name)
        cond[:, col_in_all] = theta_batch[:, col_in_tier]

    # Convert scaling factors to actual values
    for i, name in enumerate(_ALL_PARAM_NAMES):
        cond[:, i] *= PHAS13_REGISTRY[name].published

    # MHAS13 maturation: suppress If
    if ionic_model == 'mhas13':
        cond[:, C_g_f] = 0.0

    return cond


def batch_step(
    V: torch.Tensor,               # (M,)
    states: torch.Tensor,           # (M, 17)
    dt: float,
    cond: torch.Tensor,             # (M, 14) actual conductance values
    I_stim: Optional[torch.Tensor] = None,  # (M,) or None
    ionic_model: str = 'mhas13',    # 'mhas13' uses TTP06 IK1; 'phas13' uses Paci IK1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Advance M cells by one timestep with per-cell conductances.

    Supports PHAS13 (Paci IK1) and MHAS13 (TTP06 IK1 + g_f=0).
    """
    p = PHAS13Parameters()  # For non-tunable constants (Nao, Ki, Cao, etc.)

    # Extract states: all (M,)
    Nai = states[:, StateIndex.Nai]
    Cai = states[:, StateIndex.Cai]
    CaSR = states[:, StateIndex.CaSR]
    m = states[:, StateIndex.m]
    h = states[:, StateIndex.h]
    j = states[:, StateIndex.j]
    d = states[:, StateIndex.d]
    f1 = states[:, StateIndex.f1]
    f2 = states[:, StateIndex.f2]
    fCa = states[:, StateIndex.fCa]
    Xr1 = states[:, StateIndex.Xr1]
    Xr2 = states[:, StateIndex.Xr2]
    Xs = states[:, StateIndex.Xs]
    q = states[:, StateIndex.q]
    r_gate = states[:, StateIndex.r_gate]
    Xf = states[:, StateIndex.Xf]
    g_rel = states[:, StateIndex.g_rel]

    # Extract per-cell conductances: all (M,)
    g_Na  = cond[:, C_g_Na]
    g_CaL = cond[:, C_g_CaL]
    g_Kr  = cond[:, C_g_Kr]
    g_Ks  = cond[:, C_g_Ks]
    g_K1  = cond[:, C_g_K1]
    g_to  = cond[:, C_g_to]
    kNaCa = cond[:, C_kNaCa]
    PNaK  = cond[:, C_PNaK]
    g_pCa = cond[:, C_g_pCa]
    VmaxUp_val = cond[:, C_VmaxUp]
    g_f   = cond[:, C_g_f]
    g_bNa = cond[:, C_g_bNa]
    g_bCa = cond[:, C_g_bCa]
    V_leak_val = cond[:, C_V_leak]

    # ---- Compute ionic currents (all (M,)) ----
    # Reversal potentials
    ENa = RTONF * torch.log(p.Nao / Nai)
    EK = E_K(p.Ki, p.Ko)  # scalar
    ECa = 0.5 * RTONF * torch.log(p.Cao / Cai)
    EKs = RTONF * torch.log((p.Ko + p.PkNa * p.Nao) / (p.Ki + p.PkNa * Nai))

    # INa
    iNa = g_Na * (m ** 3) * h * j * (V - ENa)

    # ICaL (GHK)
    zfrt = 2.0 * V * F / (R * T)
    ghk_normal = (4.0 * V * F * F / (R * T) *
                  (Cai * safe_exp(zfrt) - 0.341 * p.Cao) /
                  (safe_exp(zfrt) - 1.0))
    ghk_limit = 2.0 * F * (Cai - 0.341 * p.Cao)
    ghk = torch.where(torch.abs(V) > 0.01, ghk_normal, ghk_limit)
    iCaL = g_CaL * d * f1 * f2 * fCa * ghk

    # IKr
    iKr = g_Kr * Xr1 * Xr2 * torch.sqrt(torch.tensor(p.Ko / 5.4)) * (V - EK)

    # IKs (Ca-dependent)
    k_ks = 1.0 + 0.6 / (1.0 + (3.8e-5 / Cai) ** 1.4)
    iKs = g_Ks * (Xs ** 2) * (V - EKs) * k_ks

    # IK1 — model-dependent formulation
    VmEK = V - EK
    if ionic_model == 'mhas13':
        # TTP06 IK1 formulation (Verkerk 2019) with fixed GK1_ttp06
        from cardiac_sim.ionic.mhas13.parameters import MHAS13Parameters
        GK1_ttp06 = MHAS13Parameters().GK1_ttp06
        alpha_k1 = 0.1 / (1.0 + safe_exp(0.06 * (VmEK - 200.0)))
        beta_k1 = (3.0 * safe_exp(0.0002 * (VmEK + 100.0)) +
                    safe_exp(0.1 * (VmEK - 10.0))) / \
                   (1.0 + safe_exp(-0.5 * VmEK))
        xK1 = alpha_k1 / (alpha_k1 + beta_k1)
        iK1 = GK1_ttp06 * (p.Ko / 5.4) ** 0.5 * xK1 * (V - EK)
    else:
        # Native Paci IK1 formulation
        alpha_k1 = 3.91 / (1.0 + safe_exp(0.5942 * (VmEK - 200.0)))
        beta_k1 = ((-1.509 * safe_exp(0.0002 * (VmEK + 100.0)) +
                    safe_exp(0.5886 * (VmEK - 10.0))) /
                   (1.0 + safe_exp(0.4547 * VmEK)))
        inf_k1 = alpha_k1 / (alpha_k1 + beta_k1)
        iK1 = g_K1 * inf_k1 * torch.sqrt(torch.tensor(p.Ko / 5.4)) * (V - EK)

    # Ito
    ito = g_to * q * r_gate * (V - EK)

    # If
    i_f = g_f * Xf * (V - p.E_f)

    # INaCa
    vfrt = V * F / (R * T)
    iNaCa = kNaCa * (
        safe_exp(p.gamma_ncx * vfrt) * (Nai ** 3) * p.Cao -
        safe_exp((p.gamma_ncx - 1.0) * vfrt) * (p.Nao ** 3) * Cai * p.alpha_ncx
    ) / (
        ((p.KmNai ** 3) + (p.Nao ** 3)) *
        (p.KmCa + p.Cao) *
        (1.0 + p.Ksat * safe_exp((p.gamma_ncx - 1.0) * vfrt))
    )

    # INaK
    iNaK = (PNaK * p.Ko / (p.Ko + p.Km_K) * Nai / (Nai + p.Km_Na) /
            (1.0 + 0.1245 * safe_exp(-0.1 * vfrt) +
             0.0353 * safe_exp(-vfrt)))

    # IpCa
    ipCa = g_pCa * Cai / (p.KpCa + Cai)

    # IbNa
    ibNa = g_bNa * (V - ENa)

    # IbCa
    ibCa = g_bCa * (V - ECa)

    # Total ionic current
    I_ion = (iNa + iCaL + iKr + iKs + iK1 + ito + i_f +
             iNaCa + iNaK + ipCa + ibNa + ibCa)

    if I_stim is not None:
        I_ion = I_ion + I_stim

    # Update membrane potential
    V_new = V + (-I_ion) * dt

    # ---- Update gating variables (depend on V only, not conductances) ----
    m_new = rush_larsen(m, INa_m_inf(V), INa_m_tau(V), dt)
    h_new = rush_larsen(h, INa_h_inf(V), INa_h_tau(V), dt)
    j_new = rush_larsen(j, INa_j_inf(V), INa_j_tau(V), dt)

    d_new = rush_larsen(d, ICaL_d_inf(V), ICaL_d_tau(V), dt)

    # f1: Ca-dependent tau scaling
    f1_inf = ICaL_f1_inf(V)
    f1_tau_base = ICaL_f1_tau(V)
    constf1 = torch.where(f1_inf > f1,
                           1.0 + 1433.0 * (Cai - 50.0e-6),
                           torch.ones_like(f1))
    f1_new = rush_larsen(f1, f1_inf, f1_tau_base * constf1, dt)

    f2_new = rush_larsen(f2, ICaL_f2_inf(V), ICaL_f2_tau(V), dt)

    # fCa: conditional Forward Euler
    fCa_inf = ICaL_fCa_inf(Cai)
    constfCa = torch.where((V > -60.0) & (fCa_inf > fCa),
                            torch.zeros_like(fCa), torch.ones_like(fCa))
    fCa_new = torch.clamp(fCa + constfCa * (fCa_inf - fCa) / FCAL_TAU * dt,
                           0.0, 1.0)

    Xr1_new = rush_larsen(Xr1, IKr_Xr1_inf(V, p.Cao), IKr_Xr1_tau(V), dt)
    Xr2_new = rush_larsen(Xr2, IKr_Xr2_inf(V), IKr_Xr2_tau(V), dt)
    Xs_new = rush_larsen(Xs, IKs_Xs_inf(V), IKs_Xs_tau(V), dt)

    q_new = rush_larsen(q, Ito_q_inf(V), Ito_q_tau(V), dt)
    r_gate_new = rush_larsen(r_gate, Ito_r_inf(V), Ito_r_tau(V), dt)

    Xf_new = rush_larsen(Xf, If_Xf_inf(V), If_Xf_tau(V), dt)

    # ---- Update concentrations (per-cell VmaxUp, V_leak) ----
    inv_VcF = p.Cm / (p.F * p.Vc) / 1000.0
    Vc_Vsr = p.Vc / p.V_SR

    Iup = i_up(Cai, VmaxUp=VmaxUp_val, Kup=p.Kup)
    Irel = i_rel(CaSR, d, g_rel)
    Ileak = i_leak(CaSR, Cai, V_leak=V_leak_val)

    INa_total = iNa + ibNa + 3.0 * iNaK + 3.0 * iNaCa
    dNai = -INa_total * inv_VcF

    ICa_total = iCaL + ibCa + ipCa - 2.0 * iNaCa
    dCai_unbuffered = (Ileak - Iup + Irel -
                       ICa_total * p.Cm / (2.0 * p.Vc * p.F) / 1000.0)
    dCai = dCai_unbuffered * buffering_factor_cyt(Cai)

    dCaSR_unbuffered = Vc_Vsr * (Iup - Irel - Ileak)
    dCaSR = dCaSR_unbuffered * buffering_factor_sr(CaSR)

    g_rel_new = update_g_rel(V, g_rel, Cai, dt)

    Nai_new = torch.clamp(Nai + dNai * dt, min=1.0)
    Cai_new = torch.clamp(Cai + dCai * dt, min=1e-8)
    CaSR_new = torch.clamp(CaSR + dCaSR * dt, min=1e-4)

    # Assemble new state
    new_states = torch.stack([
        Nai_new, Cai_new, CaSR_new,
        m_new, h_new, j_new,
        d_new, f1_new, f2_new, fCa_new,
        Xr1_new, Xr2_new, Xs_new,
        q_new, r_gate_new, Xf_new,
        g_rel_new,
    ], dim=-1)

    return V_new, new_states
