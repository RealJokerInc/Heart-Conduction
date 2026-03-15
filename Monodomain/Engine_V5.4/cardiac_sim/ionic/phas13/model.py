"""
PHAS13 hiPSC-CM Model Implementation

Paci-Hyttinen-Aalto-Setala-Severi 2013 human induced pluripotent stem
cell-derived cardiomyocyte model. Implements the IonicModel interface
for use with the simulation engine.

17 ionic state variables (gates + concentrations), V stored separately.
12 ionic currents. Spontaneous beating via If (funny current).

Reference:
Paci M, Hyttinen J, Aalto-Setala K, Severi S (2013).
"Computational Models of Ventricular- and Atrial-Like Human Induced
Pluripotent Stem Cell Derived Cardiomyocytes."
Ann Biomed Eng 41(11):2334-2348.
"""

import torch
from typing import Optional, Tuple

from ..base import IonicModel
from .parameters import (
    StateIndex, PHAS13Parameters,
    get_initial_state as _get_initial_state,
    V_REST, STATE_NAMES,
)
from .gating import (
    rush_larsen,
    # INa gates
    INa_m_inf, INa_m_tau, INa_h_inf, INa_h_tau, INa_j_inf, INa_j_tau,
    # ICaL gates
    ICaL_d_inf, ICaL_d_tau, ICaL_f1_inf, ICaL_f1_tau,
    ICaL_f2_inf, ICaL_f2_tau, ICaL_fCa_inf, FCAL_TAU,
    # IKr gates
    IKr_Xr1_inf, IKr_Xr1_tau, IKr_Xr2_inf, IKr_Xr2_tau,
    # IKs gate
    IKs_Xs_inf, IKs_Xs_tau,
    # Ito gates
    Ito_q_inf, Ito_q_tau, Ito_r_inf, Ito_r_tau,
    # If gate
    If_Xf_inf, If_Xf_tau,
)
from .currents import (
    I_Na, I_CaL, I_Kr, I_Ks, I_K1, I_to, I_f,
    I_NaCa, I_NaK, I_pCa, I_bNa, I_bCa,
)
from .calcium import update_concentrations


class PHAS13Model(IonicModel):
    """
    PHAS13 hiPSC-CM ionic model.

    Parameters
    ----------
    device : str or torch.device
        Computation device (default: 'cuda' if available)
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(device)
        self.params = PHAS13Parameters()

    @property
    def name(self) -> str:
        return "PHAS13"

    @property
    def n_states(self) -> int:
        return StateIndex.N_STATES

    @property
    def V_rest(self) -> float:
        return V_REST

    @property
    def state_names(self) -> tuple:
        return STATE_NAMES

    @property
    def gate_indices(self):
        return [
            StateIndex.m, StateIndex.h, StateIndex.j,           # INa
            StateIndex.d, StateIndex.f1, StateIndex.f2,          # ICaL (voltage)
            StateIndex.fCa,                                      # ICaL (Ca-dependent)
            StateIndex.Xr1, StateIndex.Xr2,                     # IKr
            StateIndex.Xs,                                        # IKs
            StateIndex.q, StateIndex.r_gate,                     # Ito
            StateIndex.Xf,                                        # If
            StateIndex.g_rel,                                     # RyR
        ]

    @property
    def concentration_indices(self):
        return [
            StateIndex.Nai, StateIndex.Cai, StateIndex.CaSR,
        ]

    def get_initial_state(self, n_cells: int = 1) -> torch.Tensor:
        initial = _get_initial_state(self.device, self.dtype)
        if n_cells == 1:
            return initial
        return initial.unsqueeze(0).expand(n_cells, -1).clone()

    def step(self, V: torch.Tensor, ionic_states: torch.Tensor, dt: float,
             I_stim: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance state by one time step.

        Uses Rush-Larsen for standard gates, conditional Forward Euler for
        fCa and g_rel, Forward Euler for concentrations.
        """
        single_cell = ionic_states.dim() == 1
        if single_cell:
            ionic_states = ionic_states.unsqueeze(0)
            V = V.unsqueeze(0)

        # Extract states
        Nai = ionic_states[:, StateIndex.Nai]
        Cai = ionic_states[:, StateIndex.Cai]
        CaSR = ionic_states[:, StateIndex.CaSR]

        m = ionic_states[:, StateIndex.m]
        h = ionic_states[:, StateIndex.h]
        j = ionic_states[:, StateIndex.j]

        d = ionic_states[:, StateIndex.d]
        f1 = ionic_states[:, StateIndex.f1]
        f2 = ionic_states[:, StateIndex.f2]
        fCa = ionic_states[:, StateIndex.fCa]

        Xr1 = ionic_states[:, StateIndex.Xr1]
        Xr2 = ionic_states[:, StateIndex.Xr2]
        Xs = ionic_states[:, StateIndex.Xs]

        q = ionic_states[:, StateIndex.q]
        r_gate = ionic_states[:, StateIndex.r_gate]
        Xf = ionic_states[:, StateIndex.Xf]
        g_rel = ionic_states[:, StateIndex.g_rel]

        p = self.params

        # Compute ionic currents
        iNa = I_Na(V, m, h, j, Nai, p.g_Na, p.Nao)
        iCaL = I_CaL(V, d, f1, f2, fCa, Cai, p.g_CaL, p.Cao)
        iKr = I_Kr(V, Xr1, Xr2, p.g_Kr, p.Ki, p.Ko)
        iKs = I_Ks(V, Xs, Nai, Cai, p.g_Ks, p.Ki, p.Ko, p.Nao, p.PkNa)
        iK1 = I_K1(V, p.g_K1, p.Ki, p.Ko)
        ito = I_to(V, q, r_gate, p.g_to, p.Ki, p.Ko)
        i_f = I_f(V, Xf, p.g_f, p.E_f)
        iNaCa = I_NaCa(V, Nai, Cai, p.kNaCa, p.Cao, p.Nao,
                        p.KmNai, p.KmCa, p.Ksat, p.alpha_ncx, p.gamma_ncx)
        iNaK = I_NaK(V, Nai, p.PNaK, p.Ki, p.Ko, p.Km_K, p.Km_Na)
        ipCa = I_pCa(Cai, p.g_pCa, p.KpCa)
        ibNa = I_bNa(V, Nai, p.g_bNa, p.Nao)
        ibCa = I_bCa(V, Cai, p.g_bCa, p.Cao)

        # Total ionic current
        I_ion = (iNa + iCaL + iKr + iKs + iK1 + ito + i_f +
                 iNaCa + iNaK + ipCa + ibNa + ibCa)

        if I_stim is not None:
            I_ion = I_ion + I_stim

        # Update membrane potential
        V_new = V + (-I_ion) * dt

        # ---- Update gating variables ----

        # Standard Rush-Larsen gates (11 gates)
        m_new = rush_larsen(m, INa_m_inf(V), INa_m_tau(V), dt)
        h_new = rush_larsen(h, INa_h_inf(V), INa_h_tau(V), dt)
        j_new = rush_larsen(j, INa_j_inf(V), INa_j_tau(V), dt)

        d_new = rush_larsen(d, ICaL_d_inf(V), ICaL_d_tau(V), dt)

        # f1: Ca-dependent tau scaling (constf1)
        f1_inf = ICaL_f1_inf(V)
        f1_tau_base = ICaL_f1_tau(V)
        constf1 = torch.where(
            f1_inf > f1,
            1.0 + 1433.0 * (Cai - 50.0e-6),
            torch.ones_like(f1)
        )
        f1_tau = f1_tau_base * constf1
        f1_new = rush_larsen(f1, f1_inf, f1_tau, dt)

        f2_new = rush_larsen(f2, ICaL_f2_inf(V), ICaL_f2_tau(V), dt)

        # fCa: conditional Forward Euler (Ca-dependent, freezes at V > -60)
        fCa_inf = ICaL_fCa_inf(Cai)
        constfCa = torch.where(
            (V > -60.0) & (fCa_inf > fCa),
            torch.zeros_like(fCa),
            torch.ones_like(fCa)
        )
        fCa_new = fCa + constfCa * (fCa_inf - fCa) / FCAL_TAU * dt
        fCa_new = torch.clamp(fCa_new, 0.0, 1.0)

        Xr1_new = rush_larsen(Xr1, IKr_Xr1_inf(V, p.Cao), IKr_Xr1_tau(V), dt)
        Xr2_new = rush_larsen(Xr2, IKr_Xr2_inf(V), IKr_Xr2_tau(V), dt)
        Xs_new = rush_larsen(Xs, IKs_Xs_inf(V), IKs_Xs_tau(V), dt)

        q_new = rush_larsen(q, Ito_q_inf(V), Ito_q_tau(V), dt)
        r_gate_new = rush_larsen(r_gate, Ito_r_inf(V), Ito_r_tau(V), dt)

        Xf_new = rush_larsen(Xf, If_Xf_inf(V), If_Xf_tau(V), dt)

        # ---- Update concentrations ----
        Nai_new, Cai_new, CaSR_new, g_rel_new = update_concentrations(
            V, Nai, Cai, CaSR, g_rel, d,
            iNa, iCaL, iNaCa, iNaK, ipCa, ibNa, ibCa,
            dt, p.Cm, p.Vc, p.V_SR, p.F,
        )

        # Assemble new state
        new_ionic_states = torch.stack([
            Nai_new, Cai_new, CaSR_new,
            m_new, h_new, j_new,
            d_new, f1_new, f2_new, fCa_new,
            Xr1_new, Xr2_new, Xs_new,
            q_new, r_gate_new, Xf_new,
            g_rel_new,
        ], dim=-1)

        if single_cell:
            new_ionic_states = new_ionic_states.squeeze(0)
            V_new = V_new.squeeze(0)

        return V_new, new_ionic_states

    def compute_Iion(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        single_cell = ionic_states.dim() == 1
        if single_cell:
            ionic_states = ionic_states.unsqueeze(0)
            V = V.unsqueeze(0)

        Nai = ionic_states[:, StateIndex.Nai]
        Cai = ionic_states[:, StateIndex.Cai]
        m = ionic_states[:, StateIndex.m]
        h = ionic_states[:, StateIndex.h]
        j = ionic_states[:, StateIndex.j]
        d = ionic_states[:, StateIndex.d]
        f1 = ionic_states[:, StateIndex.f1]
        f2 = ionic_states[:, StateIndex.f2]
        fCa = ionic_states[:, StateIndex.fCa]
        Xr1 = ionic_states[:, StateIndex.Xr1]
        Xr2 = ionic_states[:, StateIndex.Xr2]
        Xs = ionic_states[:, StateIndex.Xs]
        q = ionic_states[:, StateIndex.q]
        r_gate = ionic_states[:, StateIndex.r_gate]
        Xf = ionic_states[:, StateIndex.Xf]

        p = self.params

        I_ion = (
            I_Na(V, m, h, j, Nai, p.g_Na, p.Nao) +
            I_CaL(V, d, f1, f2, fCa, Cai, p.g_CaL, p.Cao) +
            I_Kr(V, Xr1, Xr2, p.g_Kr, p.Ki, p.Ko) +
            I_Ks(V, Xs, Nai, Cai, p.g_Ks, p.Ki, p.Ko, p.Nao, p.PkNa) +
            I_K1(V, p.g_K1, p.Ki, p.Ko) +
            I_to(V, q, r_gate, p.g_to, p.Ki, p.Ko) +
            I_f(V, Xf, p.g_f, p.E_f) +
            I_NaCa(V, Nai, Cai, p.kNaCa, p.Cao, p.Nao,
                    p.KmNai, p.KmCa, p.Ksat, p.alpha_ncx, p.gamma_ncx) +
            I_NaK(V, Nai, p.PNaK, p.Ki, p.Ko, p.Km_K, p.Km_Na) +
            I_pCa(Cai, p.g_pCa, p.KpCa) +
            I_bNa(V, Nai, p.g_bNa, p.Nao) +
            I_bCa(V, Cai, p.g_bCa, p.Cao)
        )

        if single_cell:
            I_ion = I_ion.squeeze(0)
        return I_ion

    def compute_gate_steady_states(self, V: torch.Tensor,
                                   ionic_states: torch.Tensor) -> torch.Tensor:
        single_cell = V.dim() == 0
        if single_cell:
            V = V.unsqueeze(0)
            ionic_states = ionic_states.unsqueeze(0)

        Cai = ionic_states[:, StateIndex.Cai]

        result = torch.stack([
            INa_m_inf(V), INa_h_inf(V), INa_j_inf(V),
            ICaL_d_inf(V), ICaL_f1_inf(V), ICaL_f2_inf(V),
            ICaL_fCa_inf(Cai),
            IKr_Xr1_inf(V, self.params.Cao), IKr_Xr2_inf(V),
            IKs_Xs_inf(V),
            Ito_q_inf(V), Ito_r_inf(V),
            If_Xf_inf(V),
            torch.ones_like(V),  # g_rel_inf placeholder (Ca-dependent)
        ], dim=-1)

        return result

    def compute_gate_time_constants(self, V: torch.Tensor,
                                    ionic_states: torch.Tensor) -> torch.Tensor:
        single_cell = V.dim() == 0
        if single_cell:
            V = V.unsqueeze(0)
            ionic_states = ionic_states.unsqueeze(0)

        result = torch.stack([
            INa_m_tau(V), INa_h_tau(V), INa_j_tau(V),
            ICaL_d_tau(V), ICaL_f1_tau(V), ICaL_f2_tau(V),
            torch.full_like(V, FCAL_TAU),   # fCa tau (constant 2 ms)
            IKr_Xr1_tau(V), IKr_Xr2_tau(V),
            IKs_Xs_tau(V),
            Ito_q_tau(V), Ito_r_tau(V),
            If_Xf_tau(V),
            torch.full_like(V, self.params.tau_g),  # g_rel tau
        ], dim=-1)

        return result

    def compute_concentration_rates(self, V: torch.Tensor,
                                    ionic_states: torch.Tensor) -> torch.Tensor:
        single_cell = V.dim() == 0
        if single_cell:
            V = V.unsqueeze(0)
            ionic_states = ionic_states.unsqueeze(0)

        Nai = ionic_states[:, StateIndex.Nai]
        Cai = ionic_states[:, StateIndex.Cai]
        CaSR = ionic_states[:, StateIndex.CaSR]
        m = ionic_states[:, StateIndex.m]
        h = ionic_states[:, StateIndex.h]
        j = ionic_states[:, StateIndex.j]
        d = ionic_states[:, StateIndex.d]
        f1 = ionic_states[:, StateIndex.f1]
        f2 = ionic_states[:, StateIndex.f2]
        fCa = ionic_states[:, StateIndex.fCa]
        Xr1 = ionic_states[:, StateIndex.Xr1]
        Xr2 = ionic_states[:, StateIndex.Xr2]
        Xs = ionic_states[:, StateIndex.Xs]
        q = ionic_states[:, StateIndex.q]
        r_gate = ionic_states[:, StateIndex.r_gate]
        Xf = ionic_states[:, StateIndex.Xf]
        g_rel = ionic_states[:, StateIndex.g_rel]

        p = self.params

        # Compute currents
        iNa = I_Na(V, m, h, j, Nai, p.g_Na, p.Nao)
        iCaL = I_CaL(V, d, f1, f2, fCa, Cai, p.g_CaL, p.Cao)
        iNaCa = I_NaCa(V, Nai, Cai, p.kNaCa, p.Cao, p.Nao,
                        p.KmNai, p.KmCa, p.Ksat, p.alpha_ncx, p.gamma_ncx)
        iNaK = I_NaK(V, Nai, p.PNaK, p.Ki, p.Ko, p.Km_K, p.Km_Na)
        ipCa = I_pCa(Cai, p.g_pCa, p.KpCa)
        ibNa = I_bNa(V, Nai, p.g_bNa, p.Nao)
        ibCa = I_bCa(V, Cai, p.g_bCa, p.Cao)

        # Concentration conversion factor
        inv_VcF = p.Cm / (p.F * p.Vc) / 1000.0
        Vc_Vsr = p.Vc / p.V_SR

        from .calcium import i_up, i_rel, i_leak
        from .calcium import buffering_factor_cyt, buffering_factor_sr

        Iup = i_up(Cai)
        Irel = i_rel(CaSR, d, g_rel)
        Ileak = i_leak(CaSR, Cai)

        # dNai/dt
        INa_total = iNa + ibNa + 3.0 * iNaK + 3.0 * iNaCa
        dNai_dt = -INa_total * inv_VcF

        # dCai/dt
        ICa_total = iCaL + ibCa + ipCa - 2.0 * iNaCa
        dCai_unbuffered = (Ileak - Iup + Irel -
                           ICa_total * p.Cm / (2.0 * p.Vc * p.F) / 1000.0)
        dCai_dt = dCai_unbuffered * buffering_factor_cyt(Cai)

        # dCaSR/dt
        dCaSR_unbuffered = Vc_Vsr * (Iup - Irel - Ileak)
        dCaSR_dt = dCaSR_unbuffered * buffering_factor_sr(CaSR)

        result = torch.stack([dNai_dt, dCai_dt, dCaSR_dt], dim=-1)
        return result
