"""
O'Hara-Rudy 2011 (ORd) Ventricular Myocyte Model - PyTorch GPU Implementation

Complete implementation supporting:
- Single cell simulation
- Batch tissue simulation (vectorized over grid)
- Rush-Larsen exponential integration for gating variables
- Forward Euler for concentrations
- CaMKII phosphorylation effects

Reference:
O'Hara T, et al. (2011). "Simulation of the Undiseased Human Cardiac
Ventricular Action Potential: Model Formulation and Experimental Validation."
PLoS Comput Biol 7(5): e1002061.
"""

import torch
from typing import Optional, Dict, Tuple

from ..base import IonicModel, CellType
from .parameters import (
    StateIndex, ORdParameters, STATE_NAMES, V_REST,
    get_celltype_parameters, get_initial_state as _get_initial_state
)
from .gating import (
    safe_exp, rush_larsen,
    # INa
    INa_m_inf, INa_m_tau, INa_h_inf, INa_hf_tau, INa_hs_tau,
    INa_j_inf, INa_j_tau, INa_hsp_inf, INa_hsp_tau, INa_jp_tau,
    # INaL
    INaL_mL_inf, INaL_mL_tau, INaL_hL_inf, INaL_hL_tau,
    INaL_hLp_inf, INaL_hLp_tau,
    # Ito
    Ito_a_inf, Ito_a_tau, Ito_i_inf, Ito_iF_tau, Ito_iS_tau,
    Ito_ap_inf, Ito_delta_epi,
    # ICaL
    ICaL_d_inf, ICaL_d_tau, ICaL_f_inf, ICaL_ff_tau, ICaL_fs_tau,
    ICaL_fcaf_tau, ICaL_fcas_tau,
    ICaL_jca_tau, ICaL_ffp_tau, ICaL_fcafp_tau,
    # IKr
    IKr_xr_inf, IKr_xrf_tau, IKr_xrs_tau,
    # IKs
    IKs_xs1_inf, IKs_xs1_tau, IKs_xs2_tau,
    # IK1
    IK1_xk1_inf, IK1_xk1_tau,
)
from .currents import (
    E_K, E_Na,
    I_Na, I_NaL, I_to, I_CaL, I_Kr, I_Ks, I_K1,
    I_NaCa, I_NaK, I_Nab, I_Cab, I_Kb, I_pCa
)
from .calcium import (
    J_rel, J_up, update_concentrations
)
from .camkii import (
    compute_CaMKa, fCaMKp, update_CaMKt
)


class ORdModel(IonicModel):
    """
    O'Hara-Rudy 2011 ventricular myocyte model.

    GPU-accelerated PyTorch implementation supporting both single-cell
    and tissue-level (batch) simulations.

    V (membrane potential) is stored and passed separately from ionic_states.

    Parameters
    ----------
    cell_type : CellType
        Cell type variant (ENDO, EPI, or M_CELL)
    device : str
        PyTorch device ('cuda' or 'cpu')
    params_override : dict, optional
        Dictionary of parameter names and values to override defaults.

    Attributes
    ----------
    params : ORdParameters
        Model parameters for the cell type
    device : torch.device
        Computation device
    dtype : torch.dtype
        Data type (float64)
    """

    def __init__(self, cell_type: CellType = CellType.ENDO, device: str = 'cuda',
                 params_override: Optional[Dict[str, float]] = None):
        super().__init__(device)
        self.cell_type = cell_type

        # Get cell-type specific parameters
        self.params = get_celltype_parameters(cell_type)

        # Apply optional parameter overrides
        if params_override is not None:
            for param_name, param_value in params_override.items():
                if hasattr(self.params, param_name):
                    setattr(self.params, param_name, param_value)
                else:
                    raise ValueError(f"Unknown parameter: {param_name}")

        # Precompute geometry factors
        self._setup_geometry()

        # For epicardial Ito delta_epi (only EPI, NOT M_CELL per ORd C++ code)
        self._is_epi = cell_type == CellType.EPI

    # =========================================================================
    # IonicModel ABC Properties
    # =========================================================================

    @property
    def name(self) -> str:
        """Model name."""
        return "ORd"

    @property
    def n_states(self) -> int:
        """Number of ionic state variables (excludes V)."""
        return StateIndex.N_STATES

    @property
    def state_names(self) -> Tuple[str, ...]:
        """Names of all ionic state variables in order (excludes V)."""
        return STATE_NAMES

    @property
    def V_rest(self) -> float:
        """Resting membrane potential (mV)."""
        return V_REST

    @property
    def gate_indices(self):
        """Indices of gating variables for Rush-Larsen integration."""
        return [
            # INa
            StateIndex.m, StateIndex.hf, StateIndex.hs, StateIndex.j,
            StateIndex.hsp, StateIndex.jp,
            # INaL
            StateIndex.mL, StateIndex.hL, StateIndex.hLp,
            # Ito
            StateIndex.a, StateIndex.iF, StateIndex.iS,
            StateIndex.ap, StateIndex.iFp, StateIndex.iSp,
            # ICaL (nca is Forward Euler, handled separately)
            StateIndex.d, StateIndex.ff, StateIndex.fs,
            StateIndex.fcaf, StateIndex.fcas, StateIndex.jca,
            StateIndex.ffp, StateIndex.fcafp,
            # IKr
            StateIndex.xrf, StateIndex.xrs,
            # IKs
            StateIndex.xs1, StateIndex.xs2,
            # IK1
            StateIndex.xk1,
        ]

    @property
    def concentration_indices(self):
        """Indices of concentration variables for Forward Euler."""
        return [
            StateIndex.nai, StateIndex.nass,
            StateIndex.ki, StateIndex.kss,
            StateIndex.cai, StateIndex.cass,
            StateIndex.cansr, StateIndex.cajsr,
            StateIndex.Jrelnp, StateIndex.Jrelp,
            StateIndex.CaMKt,
            StateIndex.nca,  # nca is a Forward Euler gate (not Rush-Larsen)
        ]

    # =========================================================================
    # Initialization
    # =========================================================================

    def _setup_geometry(self):
        """Precompute geometry-related constants."""
        p = self.params

        # Cell geometry
        import math
        vcell = 1000 * math.pi * p.rad**2 * p.L
        Ageo = 2 * math.pi * p.rad**2 + 2 * math.pi * p.rad * p.L

        self.Acap = 2 * Ageo
        self.vmyo = 0.68 * vcell
        self.vnsr = 0.0552 * vcell
        self.vjsr = 0.0048 * vcell
        self.vss = 0.02 * vcell

    def get_initial_state(self, n_cells: int = 1) -> torch.Tensor:
        """
        Get initial ionic state tensor (excludes V).

        Parameters
        ----------
        n_cells : int
            Number of cells (1 for single cell, >1 for tissue)

        Returns
        -------
        ionic_states : torch.Tensor
            Initial ionic state tensor of shape (n_cells, n_states) or (n_states,) if n_cells=1
        """
        initial = _get_initial_state(self.device, self.dtype)

        if n_cells == 1:
            return initial
        else:
            # Expand to (n_cells, n_states)
            return initial.unsqueeze(0).expand(n_cells, -1).clone()

    def get_initial_state_tissue(self, ny: int, nx: int) -> torch.Tensor:
        """
        Get initial ionic state tensor for 2D tissue simulation (excludes V).

        Parameters
        ----------
        ny, nx : int
            Grid dimensions

        Returns
        -------
        ionic_states : torch.Tensor
            Initial ionic states (ny, nx, 40) on GPU
        """
        initial = _get_initial_state(self.device, self.dtype)
        ionic_states = initial.unsqueeze(0).unsqueeze(0).expand(ny, nx, -1).clone()
        return ionic_states

    # =========================================================================
    # Gate Updates
    # =========================================================================

    def _update_gates(self, V: torch.Tensor, ionic_states: torch.Tensor,
                      dt: float) -> torch.Tensor:
        """
        Update all gating variables using Rush-Larsen integration.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (...,)
        ionic_states : torch.Tensor
            Ionic state tensor (..., 40)
        dt : float
            Time step (ms)

        Returns
        -------
        ionic_states : torch.Tensor
            Updated ionic state tensor
        """
        cass = ionic_states[..., StateIndex.cass]

        # =================================================================
        # INa gates
        # =================================================================
        # m gate
        m_inf = INa_m_inf(V)
        m_tau = INa_m_tau(V)
        ionic_states[..., StateIndex.m] = rush_larsen(
            ionic_states[..., StateIndex.m], m_inf, m_tau, dt
        )

        # hf gate
        h_inf = INa_h_inf(V)
        hf_tau = INa_hf_tau(V)
        ionic_states[..., StateIndex.hf] = rush_larsen(
            ionic_states[..., StateIndex.hf], h_inf, hf_tau, dt
        )

        # hs gate
        hs_tau = INa_hs_tau(V, self.params.tau_hs_scale)
        ionic_states[..., StateIndex.hs] = rush_larsen(
            ionic_states[..., StateIndex.hs], h_inf, hs_tau, dt
        )

        # j gate
        j_inf = INa_j_inf(V)
        j_tau = INa_j_tau(V)
        ionic_states[..., StateIndex.j] = rush_larsen(
            ionic_states[..., StateIndex.j], j_inf, j_tau, dt
        )

        # hsp gate (phosphorylated)
        hsp_inf = INa_hsp_inf(V)
        hsp_tau = INa_hsp_tau(V, self.params.tau_hsp_scale)
        ionic_states[..., StateIndex.hsp] = rush_larsen(
            ionic_states[..., StateIndex.hsp], hsp_inf, hsp_tau, dt
        )

        # jp gate (phosphorylated) - uses h_inf (not hsp_inf)
        jp_inf = h_inf
        jp_tau = INa_jp_tau(V)
        ionic_states[..., StateIndex.jp] = rush_larsen(
            ionic_states[..., StateIndex.jp], jp_inf, jp_tau, dt
        )

        # =================================================================
        # INaL gates
        # =================================================================
        mL_inf = INaL_mL_inf(V)
        mL_tau = INaL_mL_tau(V)
        ionic_states[..., StateIndex.mL] = rush_larsen(
            ionic_states[..., StateIndex.mL], mL_inf, mL_tau, dt
        )

        hL_inf = INaL_hL_inf(V)
        hL_tau = INaL_hL_tau(V)
        ionic_states[..., StateIndex.hL] = rush_larsen(
            ionic_states[..., StateIndex.hL], hL_inf, hL_tau, dt
        )

        hLp_inf = INaL_hLp_inf(V)
        hLp_tau = INaL_hLp_tau(V)
        ionic_states[..., StateIndex.hLp] = rush_larsen(
            ionic_states[..., StateIndex.hLp], hLp_inf, hLp_tau, dt
        )

        # =================================================================
        # Ito gates
        # =================================================================
        a_inf = Ito_a_inf(V)
        a_tau = Ito_a_tau(V)
        ionic_states[..., StateIndex.a] = rush_larsen(
            ionic_states[..., StateIndex.a], a_inf, a_tau, dt
        )

        i_inf = Ito_i_inf(V)
        iF_tau = Ito_iF_tau(V)
        iS_tau = Ito_iS_tau(V)

        # Apply delta_epi scaling for EPI cells only
        if self._is_epi:
            delta_epi = Ito_delta_epi(V)
            iF_tau = iF_tau * delta_epi
            iS_tau = iS_tau * delta_epi

        ionic_states[..., StateIndex.iF] = rush_larsen(
            ionic_states[..., StateIndex.iF], i_inf, iF_tau, dt
        )
        ionic_states[..., StateIndex.iS] = rush_larsen(
            ionic_states[..., StateIndex.iS], i_inf, iS_tau, dt
        )

        # Phosphorylated variants
        ap_inf = Ito_ap_inf(V)
        ionic_states[..., StateIndex.ap] = rush_larsen(
            ionic_states[..., StateIndex.ap], ap_inf, a_tau, dt
        )

        # Phosphorylated Ito inactivation has CaMKII-dependent time constants
        dti_develop = 1.354 + 1.0e-4 / (safe_exp((V - 167.4) / 15.89) +
                                         safe_exp(-(V - 12.23) / 0.2154))
        dti_recover = 1.0 - 0.5 / (1.0 + safe_exp((V + 70.0) / 20.0))
        iFp_tau = dti_develop * dti_recover * iF_tau
        iSp_tau = dti_develop * dti_recover * iS_tau

        ionic_states[..., StateIndex.iFp] = rush_larsen(
            ionic_states[..., StateIndex.iFp], i_inf, iFp_tau, dt
        )
        ionic_states[..., StateIndex.iSp] = rush_larsen(
            ionic_states[..., StateIndex.iSp], i_inf, iSp_tau, dt
        )

        # =================================================================
        # ICaL gates
        # =================================================================
        d_inf = ICaL_d_inf(V)
        d_tau = ICaL_d_tau(V)
        ionic_states[..., StateIndex.d] = rush_larsen(
            ionic_states[..., StateIndex.d], d_inf, d_tau, dt
        )

        f_inf = ICaL_f_inf(V)
        ff_tau = ICaL_ff_tau(V)
        fs_tau = ICaL_fs_tau(V)
        ionic_states[..., StateIndex.ff] = rush_larsen(
            ionic_states[..., StateIndex.ff], f_inf, ff_tau, dt
        )
        ionic_states[..., StateIndex.fs] = rush_larsen(
            ionic_states[..., StateIndex.fs], f_inf, fs_tau, dt
        )

        # Ca-dependent inactivation gates use f_inf (voltage)
        fcaf_tau = ICaL_fcaf_tau(V)
        fcas_tau = ICaL_fcas_tau(V)
        ionic_states[..., StateIndex.fcaf] = rush_larsen(
            ionic_states[..., StateIndex.fcaf], f_inf, fcaf_tau, dt
        )
        ionic_states[..., StateIndex.fcas] = rush_larsen(
            ionic_states[..., StateIndex.fcas], f_inf, fcas_tau, dt
        )

        # jca uses f_inf and fixed tau=75ms
        jca_tau = 75.0 * torch.ones_like(V)
        ionic_states[..., StateIndex.jca] = rush_larsen(
            ionic_states[..., StateIndex.jca], f_inf, jca_tau, dt
        )

        # Phosphorylated variants
        ffp_tau = ICaL_ffp_tau(V)
        ionic_states[..., StateIndex.ffp] = rush_larsen(
            ionic_states[..., StateIndex.ffp], f_inf, ffp_tau, dt
        )

        fcafp_tau = ICaL_fcafp_tau(V)
        ionic_states[..., StateIndex.fcafp] = rush_larsen(
            ionic_states[..., StateIndex.fcafp], f_inf, fcafp_tau, dt
        )

        # nca gate (special handling - forward Euler with jca-dependent kinetics)
        Kmn = 0.002  # mM
        k2n = 1000.0  # /ms
        jca = ionic_states[..., StateIndex.jca]
        nca = ionic_states[..., StateIndex.nca]

        km2n = jca * 1.0
        anca = 1.0 / (k2n / (km2n + 1e-10) + (1.0 + Kmn / (cass + 1e-10)) ** 4.0)

        dnca = anca * k2n - nca * km2n
        ionic_states[..., StateIndex.nca] = nca + dt * dnca

        # =================================================================
        # IKr gates
        # =================================================================
        xr_inf = IKr_xr_inf(V)
        xrf_tau = IKr_xrf_tau(V)
        xrs_tau = IKr_xrs_tau(V)
        ionic_states[..., StateIndex.xrf] = rush_larsen(
            ionic_states[..., StateIndex.xrf], xr_inf, xrf_tau, dt
        )
        ionic_states[..., StateIndex.xrs] = rush_larsen(
            ionic_states[..., StateIndex.xrs], xr_inf, xrs_tau, dt
        )

        # =================================================================
        # IKs gates
        # =================================================================
        xs1_inf = IKs_xs1_inf(V)
        xs1_tau = IKs_xs1_tau(V)
        xs2_tau = IKs_xs2_tau(V)
        ionic_states[..., StateIndex.xs1] = rush_larsen(
            ionic_states[..., StateIndex.xs1], xs1_inf, xs1_tau, dt
        )
        ionic_states[..., StateIndex.xs2] = rush_larsen(
            ionic_states[..., StateIndex.xs2], xs1_inf, xs2_tau, dt
        )

        # =================================================================
        # IK1 gate
        # =================================================================
        xk1_inf = IK1_xk1_inf(V, self.params.ko)
        xk1_tau = IK1_xk1_tau(V)
        ionic_states[..., StateIndex.xk1] = rush_larsen(
            ionic_states[..., StateIndex.xk1], xk1_inf, xk1_tau, dt
        )

        return ionic_states

    # =========================================================================
    # Current Computation
    # =========================================================================

    def compute_currents(self, V: torch.Tensor, ionic_states: torch.Tensor,
                         fCaMKp_val: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all ionic currents.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (...,)
        ionic_states : torch.Tensor
            Ionic state tensor (..., 40)
        fCaMKp_val : torch.Tensor
            CaMKII phosphorylation factor

        Returns
        -------
        currents : dict
            Dictionary of all currents
        """
        p = self.params

        # Extract ionic state variables
        nai = ionic_states[..., StateIndex.nai]
        nass = ionic_states[..., StateIndex.nass]
        ki = ionic_states[..., StateIndex.ki]
        kss = ionic_states[..., StateIndex.kss]
        cai = ionic_states[..., StateIndex.cai]
        cass = ionic_states[..., StateIndex.cass]

        # Gates
        m = ionic_states[..., StateIndex.m]
        hf = ionic_states[..., StateIndex.hf]
        hs = ionic_states[..., StateIndex.hs]
        j = ionic_states[..., StateIndex.j]
        hsp = ionic_states[..., StateIndex.hsp]
        jp = ionic_states[..., StateIndex.jp]
        mL = ionic_states[..., StateIndex.mL]
        hL = ionic_states[..., StateIndex.hL]
        hLp = ionic_states[..., StateIndex.hLp]
        a = ionic_states[..., StateIndex.a]
        iF = ionic_states[..., StateIndex.iF]
        iS = ionic_states[..., StateIndex.iS]
        ap = ionic_states[..., StateIndex.ap]
        iFp = ionic_states[..., StateIndex.iFp]
        iSp = ionic_states[..., StateIndex.iSp]
        d = ionic_states[..., StateIndex.d]
        ff = ionic_states[..., StateIndex.ff]
        fs = ionic_states[..., StateIndex.fs]
        fcaf = ionic_states[..., StateIndex.fcaf]
        fcas = ionic_states[..., StateIndex.fcas]
        jca = ionic_states[..., StateIndex.jca]
        nca = ionic_states[..., StateIndex.nca]
        ffp = ionic_states[..., StateIndex.ffp]
        fcafp = ionic_states[..., StateIndex.fcafp]
        xrf = ionic_states[..., StateIndex.xrf]
        xrs = ionic_states[..., StateIndex.xrs]
        xs1 = ionic_states[..., StateIndex.xs1]
        xs2 = ionic_states[..., StateIndex.xs2]
        xk1 = ionic_states[..., StateIndex.xk1]

        # Compute currents
        INa = I_Na(V, m, hf, hs, j, hsp, jp, nai, fCaMKp_val,
                   p.GNa, p.nao)

        INaL = I_NaL(V, mL, hL, hLp, nai, fCaMKp_val,
                     p.GNaL, p.nao, p.GNaL_scale)

        delta_epi = Ito_delta_epi(V) if self._is_epi else None
        Ito = I_to(V, a, iF, iS, ap, iFp, iSp, ki, fCaMKp_val,
                   p.Gto, p.ko, p.Gto_scale, delta_epi)

        ICaL, ICaNa, ICaK = I_CaL(V, d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp,
                                   cass, nass, kss, fCaMKp_val,
                                   p.PCa, p.cao, p.nao, p.ko, p.PCa_scale)

        IKr = I_Kr(V, xrf, xrs, ki, p.GKr, p.ko, p.GKr_scale)

        IKs = I_Ks(V, xs1, xs2, ki, nai, cai, p.GKs, p.ko, p.nao, p.GKs_scale)

        IK1 = I_K1(V, xk1, ki, p.GK1, p.ko, p.GK1_scale)

        INaCa_i, INaCa_ss = I_NaCa(V, nai, nass, cai, cass,
                                    p.Gncx, p.cao, p.nao, p.Gncx_scale)

        INaK = I_NaK(V, nai, nass, ki, p.Pnak, p.ko, p.nao, p.Pnak_scale)

        INab = I_Nab(V, nai, p.PNab, p.nao)
        ICab = I_Cab(V, cai, p.PCab, p.cao)
        IKb = I_Kb(V, ki, p.GKb, p.ko, p.GKb_scale)
        IpCa = I_pCa(cai, p.GpCa)

        return {
            'INa': INa, 'INaL': INaL, 'Ito': Ito,
            'ICaL': ICaL, 'ICaNa': ICaNa, 'ICaK': ICaK,
            'IKr': IKr, 'IKs': IKs, 'IK1': IK1,
            'INaCa_i': INaCa_i, 'INaCa_ss': INaCa_ss,
            'INaK': INaK, 'INab': INab, 'ICab': ICab,
            'IKb': IKb, 'IpCa': IpCa
        }

    def compute_Iion(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute total ionic current.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (...,)
        ionic_states : torch.Tensor
            Ionic state tensor (..., n_states)

        Returns
        -------
        Iion : torch.Tensor
            Total ionic current (uA/uF)
        """
        p = self.params

        # Compute CaMKII activation
        CaMKt = ionic_states[..., StateIndex.CaMKt]
        cass = ionic_states[..., StateIndex.cass]
        CaMKb, CaMKa = compute_CaMKa(CaMKt, cass, p.CaMKo, p.KmCaM)
        fCaMKp_val = fCaMKp(CaMKa, p.KmCaMK)

        # Compute all currents
        currents = self.compute_currents(V, ionic_states, fCaMKp_val)

        # Sum all currents
        Iion = (currents['INa'] + currents['INaL'] + currents['Ito'] +
                currents['ICaL'] + currents['ICaNa'] + currents['ICaK'] +
                currents['IKr'] + currents['IKs'] + currents['IK1'] +
                currents['INaCa_i'] + currents['INaCa_ss'] +
                currents['INaK'] + currents['INab'] + currents['ICab'] +
                currents['IKb'] + currents['IpCa'])

        return Iion

    def compute_gate_steady_states(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute steady-state values for all gating variables.

        Returns (n_cells, 28) in gate_indices order.
        Note: ORd has 28 Rush-Larsen gates (nca is Forward Euler).
        """

        # INa
        m_inf = INa_m_inf(V)
        h_inf = INa_h_inf(V)  # shared by hf, hs, jp
        j_inf = INa_j_inf(V)
        hsp_inf = INa_hsp_inf(V)

        # INaL
        mL_inf = INaL_mL_inf(V)
        hL_inf = INaL_hL_inf(V)
        hLp_inf = INaL_hLp_inf(V)

        # Ito
        a_inf = Ito_a_inf(V)
        i_inf = Ito_i_inf(V)  # shared by iF, iS, iFp, iSp
        ap_inf = Ito_ap_inf(V)

        # ICaL
        d_inf = ICaL_d_inf(V)
        f_inf = ICaL_f_inf(V)  # shared by ff, fs, fcaf, fcas, jca, ffp, fcafp

        # IKr
        xr_inf = IKr_xr_inf(V)  # shared by xrf, xrs

        # IKs
        xs1_inf = IKs_xs1_inf(V)  # shared by xs1, xs2

        # IK1
        xk1_inf = IK1_xk1_inf(V, self.params.ko)

        return torch.stack([
            # INa: m, hf, hs, j, hsp, jp
            m_inf, h_inf, h_inf, j_inf, hsp_inf, h_inf,
            # INaL: mL, hL, hLp
            mL_inf, hL_inf, hLp_inf,
            # Ito: a, iF, iS, ap, iFp, iSp
            a_inf, i_inf, i_inf, ap_inf, i_inf, i_inf,
            # ICaL: d, ff, fs, fcaf, fcas, jca, ffp, fcafp
            d_inf, f_inf, f_inf, f_inf, f_inf, f_inf, f_inf, f_inf,
            # IKr: xrf, xrs
            xr_inf, xr_inf,
            # IKs: xs1, xs2
            xs1_inf, xs1_inf,
            # IK1: xk1
            xk1_inf,
        ], dim=-1)

    def compute_gate_time_constants(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute time constants for all gating variables.

        Returns (n_cells, 28) in gate_indices order.
        Note: Some time constants depend on CaMKII phosphorylation state.
        """
        p = self.params

        # INa
        m_tau = INa_m_tau(V)
        hf_tau = INa_hf_tau(V)
        hs_tau = INa_hs_tau(V, p.tau_hs_scale)
        j_tau = INa_j_tau(V)
        hsp_tau = INa_hsp_tau(V, p.tau_hsp_scale)
        jp_tau = INa_jp_tau(V)

        # INaL
        mL_tau = INaL_mL_tau(V)
        hL_tau = INaL_hL_tau(V)
        hLp_tau = INaL_hLp_tau(V)

        # Ito
        a_tau = Ito_a_tau(V)
        iF_tau = Ito_iF_tau(V)
        iS_tau = Ito_iS_tau(V)

        if self._is_epi:
            delta_epi = Ito_delta_epi(V)
            iF_tau = iF_tau * delta_epi
            iS_tau = iS_tau * delta_epi

        # Phosphorylated Ito inactivation
        dti_develop = 1.354 + 1.0e-4 / (safe_exp((V - 167.4) / 15.89) +
                                         safe_exp(-(V - 12.23) / 0.2154))
        dti_recover = 1.0 - 0.5 / (1.0 + safe_exp((V + 70.0) / 20.0))
        iFp_tau = dti_develop * dti_recover * iF_tau
        iSp_tau = dti_develop * dti_recover * iS_tau

        # ICaL
        d_tau = ICaL_d_tau(V)
        ff_tau = ICaL_ff_tau(V)
        fs_tau = ICaL_fs_tau(V)
        fcaf_tau = ICaL_fcaf_tau(V)
        fcas_tau = ICaL_fcas_tau(V)
        jca_tau = 75.0 * torch.ones_like(V)
        ffp_tau = ICaL_ffp_tau(V)
        fcafp_tau = ICaL_fcafp_tau(V)

        # IKr
        xrf_tau = IKr_xrf_tau(V)
        xrs_tau = IKr_xrs_tau(V)

        # IKs
        xs1_tau = IKs_xs1_tau(V)
        xs2_tau = IKs_xs2_tau(V)

        # IK1
        xk1_tau = IK1_xk1_tau(V)

        return torch.stack([
            # INa
            m_tau, hf_tau, hs_tau, j_tau, hsp_tau, jp_tau,
            # INaL
            mL_tau, hL_tau, hLp_tau,
            # Ito
            a_tau, iF_tau, iS_tau, a_tau, iFp_tau, iSp_tau,
            # ICaL
            d_tau, ff_tau, fs_tau, fcaf_tau, fcas_tau, jca_tau, ffp_tau, fcafp_tau,
            # IKr
            xrf_tau, xrs_tau,
            # IKs
            xs1_tau, xs2_tau,
            # IK1
            xk1_tau,
        ], dim=-1)

    def compute_concentration_rates(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute time derivatives for all concentration variables.

        Returns (n_cells, 12) in concentration_indices order:
        nai, nass, ki, kss, cai, cass, cansr, cajsr, Jrelnp, Jrelp, CaMKt, nca

        Note: This is complex because ORd has SR release dynamics, CaMKII,
        and nca gate that all use Forward Euler.
        """
        raise NotImplementedError(
            "ORd compute_concentration_rates is complex due to SR release "
            "and CaMKII dynamics. Use model.step() for full ORd time stepping. "
            "For external IonicSolver support, TTP06 is the primary target."
        )

    # =========================================================================
    # Time Stepping
    # =========================================================================

    def step(self, V: torch.Tensor, ionic_states: torch.Tensor, dt: float,
             Istim: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance model by one time step.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (...,)
        ionic_states : torch.Tensor
            Ionic state tensor (..., 40)
        dt : float
            Time step (ms)
        Istim : torch.Tensor, optional
            Stimulus current (uA/uF), same shape as V or scalar

        Returns
        -------
        V_new : torch.Tensor
            Updated membrane potential
        ionic_states : torch.Tensor
            Updated ionic state tensor
        """
        p = self.params

        if Istim is None:
            Istim = torch.zeros_like(V)

        # 1. Compute CaMKII activation (using current state)
        CaMKt = ionic_states[..., StateIndex.CaMKt]
        cass = ionic_states[..., StateIndex.cass]
        CaMKb, CaMKa = compute_CaMKa(CaMKt, cass, p.CaMKo, p.KmCaM)
        fCaMKp_val = fCaMKp(CaMKa, p.KmCaMK)

        # 2. Compute all currents BEFORE updating gates
        currents = self.compute_currents(V, ionic_states, fCaMKp_val)

        # 3. Update gating variables (Rush-Larsen) - AFTER computing currents
        ionic_states = self._update_gates(V, ionic_states, dt)

        # 4. Compute total ionic current
        Iion = (currents['INa'] + currents['INaL'] + currents['Ito'] +
                currents['ICaL'] + currents['ICaNa'] + currents['ICaK'] +
                currents['IKr'] + currents['IKs'] + currents['IK1'] +
                currents['INaCa_i'] + currents['INaCa_ss'] +
                currents['INaK'] + currents['INab'] + currents['ICab'] +
                currents['IKb'] + currents['IpCa'])

        # 5. Update voltage
        dV = -dt * (Iion + Istim) / p.Cm
        V_new = V + dV

        # 6. Compute SR fluxes
        Jrel, Jrelnp_new, Jrelp_new = J_rel(
            ionic_states[..., StateIndex.cajsr], currents['ICaL'],
            ionic_states[..., StateIndex.Jrelnp], ionic_states[..., StateIndex.Jrelp],
            fCaMKp_val, dt, p.a_rel, p.bt, p.cajsr_half, p.Jrel_scale
        )
        ionic_states[..., StateIndex.Jrelnp] = Jrelnp_new
        ionic_states[..., StateIndex.Jrelp] = Jrelp_new

        Jup = J_up(
            ionic_states[..., StateIndex.cai], ionic_states[..., StateIndex.cansr],
            fCaMKp_val, p.Jup_max, p.Kmup, p.nsrbar, p.Jup_scale
        )

        # 7. Update concentrations
        (nai_new, nass_new, ki_new, kss_new,
         cai_new, cass_new, cansr_new, cajsr_new) = update_concentrations(
            ionic_states[..., StateIndex.nai], ionic_states[..., StateIndex.nass],
            ionic_states[..., StateIndex.ki], ionic_states[..., StateIndex.kss],
            ionic_states[..., StateIndex.cai], ionic_states[..., StateIndex.cass],
            ionic_states[..., StateIndex.cansr], ionic_states[..., StateIndex.cajsr],
            currents['INa'], currents['INaL'],
            currents['ICaL'], currents['ICaNa'], currents['ICaK'],
            currents['ICab'], currents['INab'], currents['IpCa'],
            currents['INaCa_i'], currents['INaCa_ss'],
            currents['INaK'],
            currents['IKr'], currents['IKs'], currents['IK1'],
            currents['Ito'], currents['IKb'],
            Istim,
            Jrel, Jup,
            dt,
            self.Acap, self.vmyo, self.vnsr, self.vjsr, self.vss,
            p.tau_diff_Na, p.tau_diff_K, p.tau_diff_Ca, p.tau_tr,
            p.cmdnmax, p.kmcmdn, p.trpnmax, p.kmtrpn,
            p.BSRmax, p.KmBSR, p.BSLmax, p.KmBSL,
            p.csqnmax, p.kmcsqn,
            p.cmdnmax_scale
        )

        ionic_states[..., StateIndex.nai] = nai_new
        ionic_states[..., StateIndex.nass] = nass_new
        ionic_states[..., StateIndex.ki] = ki_new
        ionic_states[..., StateIndex.kss] = kss_new
        ionic_states[..., StateIndex.cai] = cai_new
        ionic_states[..., StateIndex.cass] = cass_new
        ionic_states[..., StateIndex.cansr] = cansr_new
        ionic_states[..., StateIndex.cajsr] = cajsr_new

        # 8. Update CaMKII
        ionic_states[..., StateIndex.CaMKt] = update_CaMKt(
            CaMKt, CaMKb, CaMKa, dt, p.aCaMK, p.bCaMK
        )

        return V_new, ionic_states
