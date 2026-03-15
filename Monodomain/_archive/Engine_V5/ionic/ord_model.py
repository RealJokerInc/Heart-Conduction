"""
O'Hara-Rudy (ORd 2011) Model Class

Main model class integrating all components:
- Gating kinetics
- Ionic currents
- Calcium handling
- CaMKII signaling

Reference: O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.
"""

import numpy as np
from typing import Optional, Tuple

from .parameters import ORdParameters, StateIndex, CellType, DEFAULT_PARAMS
from . import gating
from . import currents
from . import calcium
from . import camkii


class ORdModel:
    """
    O'Hara-Rudy 2011 human ventricular myocyte model.

    Features:
    - 41 state variables
    - 15 ionic currents
    - CaMKII signaling
    - Subspace calcium handling
    - Cell-type specific parameters (endo/epi/M-cell)
    """

    def __init__(self, params: Optional[ORdParameters] = None,
                 celltype: CellType = CellType.ENDO):
        """
        Initialize ORd model.

        Args:
            params: Model parameters (uses defaults if None)
            celltype: Cell type (ENDO, EPI, or M_CELL)
        """
        self.params = params if params is not None else DEFAULT_PARAMS
        self.celltype = celltype
        self.scales = self.params.get_celltype_scales(celltype)

        # Cache scaled parameters
        self._setup_scaled_params()

    def _setup_scaled_params(self):
        """Apply cell-type scaling to parameters."""
        p = self.params
        s = self.scales

        # Scaled conductances
        self.GNaL = p.GNaL * s['GNaL']
        self.Gto = p.Gto * s['Gto']
        self.PCa = p.PCa * s['PCa']
        self.GKr = p.GKr * s['GKr']
        self.GKs = p.GKs * s['GKs']
        self.GK1 = p.GK1 * s['GK1']
        self.Gncx = p.Gncx * s['Gncx']
        self.Pnak = p.Pnak * s['Pnak']
        self.GKb = p.GKb * s['GKb']
        self.cmdnmax = p.cmdnmax * s['cmdnmax']

    def get_initial_state(self) -> np.ndarray:
        """Get initial state vector."""
        return self.params.get_initial_state(self.celltype)

    def compute_currents(self, y: np.ndarray, CaMKa: float) -> dict:
        """
        Compute all ionic currents.

        Args:
            y: State vector (41 variables)
            CaMKa: Active CaMKII

        Returns:
            Dictionary of all currents
        """
        p = self.params
        V = y[StateIndex.V]

        # Concentrations
        nai = y[StateIndex.nai]
        nass = y[StateIndex.nass]
        ki = y[StateIndex.ki]
        kss = y[StateIndex.kss]
        cai = y[StateIndex.cai]
        cass = y[StateIndex.cass]

        # CaMKII phosphorylation fraction
        fCaMKp = camkii.fCaMKp(CaMKa, p.KmCaMK)

        # INa
        INa = currents.I_Na(
            V, y[StateIndex.m], y[StateIndex.hf], y[StateIndex.hs],
            y[StateIndex.j], y[StateIndex.hsp], y[StateIndex.jp],
            nai, p.nao, fCaMKp, p.GNa
        )

        # INaL
        INaL = currents.I_NaL(
            V, y[StateIndex.mL], y[StateIndex.hL], y[StateIndex.hLp],
            nai, p.nao, fCaMKp, self.GNaL
        )

        # Ito
        Ito = currents.I_to(
            V, y[StateIndex.a], y[StateIndex.iF], y[StateIndex.iS],
            y[StateIndex.ap], y[StateIndex.iFp], y[StateIndex.iSp],
            ki, p.ko, fCaMKp, self.Gto
        )

        # ICaL (returns tuple)
        ICaL, ICaNa, ICaK = currents.I_CaL(
            V, y[StateIndex.d], y[StateIndex.ff], y[StateIndex.fs],
            y[StateIndex.fcaf], y[StateIndex.fcas], y[StateIndex.jca],
            y[StateIndex.nca], y[StateIndex.ffp], y[StateIndex.fcafp],
            cass, nass, kss, p.cao, p.nao, p.ko, fCaMKp, self.PCa
        )

        # IKr
        IKr = currents.I_Kr(
            V, y[StateIndex.xrf], y[StateIndex.xrs],
            ki, p.ko, self.GKr
        )

        # IKs
        IKs = currents.I_Ks(
            V, y[StateIndex.xs1], y[StateIndex.xs2],
            ki, nai, p.ko, p.nao, cai, self.GKs
        )

        # IK1
        IK1 = currents.I_K1(
            V, y[StateIndex.xk1], ki, p.ko, self.GK1
        )

        # INaCa
        INaCa_i = currents.I_NaCa_i(V, nai, cai, p.nao, p.cao, self.Gncx)
        INaCa_ss = currents.I_NaCa_ss(V, nass, cass, p.nao, p.cao, self.Gncx)
        INaCa = INaCa_i + INaCa_ss

        # INaK
        INaK = currents.I_NaK(V, nai, ki, p.nao, p.ko, self.Pnak)

        # Background currents
        IKb = currents.I_Kb(V, ki, p.ko, self.GKb)
        INab = currents.I_Nab(V, nai, p.nao, p.PNab)
        ICab = currents.I_Cab(V, cai, p.cao, p.PCab)
        IpCa = currents.I_pCa(cai, p.GpCa)

        return {
            'INa': INa, 'INaL': INaL, 'Ito': Ito,
            'ICaL': ICaL, 'ICaNa': ICaNa, 'ICaK': ICaK,
            'IKr': IKr, 'IKs': IKs, 'IK1': IK1,
            'INaCa_i': INaCa_i, 'INaCa_ss': INaCa_ss, 'INaCa': INaCa,
            'INaK': INaK,
            'IKb': IKb, 'INab': INab, 'ICab': ICab, 'IpCa': IpCa
        }

    def step(self, y: np.ndarray, dt: float, Istim: float = 0.0) -> np.ndarray:
        """
        Advance model by one time step.

        Args:
            y: Current state vector
            dt: Time step (ms)
            Istim: Stimulus current (uA/uF)

        Returns:
            Updated state vector
        """
        p = self.params
        y_new = y.copy()

        V = y[StateIndex.V]

        # Get concentrations
        nai = y[StateIndex.nai]
        nass = y[StateIndex.nass]
        ki = y[StateIndex.ki]
        kss = y[StateIndex.kss]
        cai = y[StateIndex.cai]
        cass = y[StateIndex.cass]
        cansr = y[StateIndex.cansr]
        cajsr = y[StateIndex.cajsr]

        # CaMKII
        CaMKt = y[StateIndex.CaMKt]
        CaMKb, CaMKa = camkii.compute_CaMKa(CaMKt, cass, p.CaMKo, p.KmCaM)
        fCaMKp = camkii.fCaMKp(CaMKa, p.KmCaMK)

        # Update CaMKII
        dCaMKt = camkii.dCaMKt_dt(CaMKb, CaMKt, p.aCaMK, p.bCaMK)
        y_new[StateIndex.CaMKt] = CaMKt + dt * dCaMKt

        # =================================================================
        # Update gating variables (Rush-Larsen)
        # =================================================================

        # INa gates
        y_new[StateIndex.m] = gating.rush_larsen(
            y[StateIndex.m], gating.INa_m_inf(V), gating.INa_m_tau(V), dt)
        y_new[StateIndex.hf] = gating.rush_larsen(
            y[StateIndex.hf], gating.INa_h_inf(V), gating.INa_hf_tau(V), dt)
        y_new[StateIndex.hs] = gating.rush_larsen(
            y[StateIndex.hs], gating.INa_h_inf(V), gating.INa_hs_tau(V), dt)
        y_new[StateIndex.j] = gating.rush_larsen(
            y[StateIndex.j], gating.INa_h_inf(V), gating.INa_j_tau(V), dt)
        y_new[StateIndex.hsp] = gating.rush_larsen(
            y[StateIndex.hsp], gating.INa_hsp_inf(V), 3.0 * gating.INa_hs_tau(V), dt)
        y_new[StateIndex.jp] = gating.rush_larsen(
            y[StateIndex.jp], gating.INa_h_inf(V), 1.46 * gating.INa_j_tau(V), dt)

        # INaL gates
        y_new[StateIndex.mL] = gating.rush_larsen(
            y[StateIndex.mL], gating.INaL_mL_inf(V), gating.INa_m_tau(V), dt)
        y_new[StateIndex.hL] = gating.rush_larsen(
            y[StateIndex.hL], gating.INaL_hL_inf(V), p.thL, dt)
        y_new[StateIndex.hLp] = gating.rush_larsen(
            y[StateIndex.hLp], gating.INaL_hLp_inf(V), 3.0 * p.thL, dt)

        # Ito gates
        celltype_int = int(self.celltype)
        y_new[StateIndex.a] = gating.rush_larsen(
            y[StateIndex.a], gating.Ito_a_inf(V), gating.Ito_a_tau(V), dt)
        y_new[StateIndex.iF] = gating.rush_larsen(
            y[StateIndex.iF], gating.Ito_i_inf(V), gating.Ito_iF_tau(V, celltype_int), dt)
        y_new[StateIndex.iS] = gating.rush_larsen(
            y[StateIndex.iS], gating.Ito_i_inf(V), gating.Ito_iS_tau(V, celltype_int), dt)
        y_new[StateIndex.ap] = gating.rush_larsen(
            y[StateIndex.ap], gating.Ito_ap_inf(V), gating.Ito_a_tau(V), dt)

        # Phosphorylated Ito inactivation
        tiF = gating.Ito_iF_tau(V, celltype_int)
        tiS = gating.Ito_iS_tau(V, celltype_int)
        dti_develop = 1.354 + 1.0e-4 / (np.exp((V - 167.4) / 15.89) + np.exp(-(V - 12.23) / 0.2154))
        dti_recover = 1.0 - 0.5 / (1.0 + np.exp((V + 70.0) / 20.0))
        tiFp = dti_develop * dti_recover * tiF
        tiSp = dti_develop * dti_recover * tiS
        y_new[StateIndex.iFp] = gating.rush_larsen(
            y[StateIndex.iFp], gating.Ito_i_inf(V), tiFp, dt)
        y_new[StateIndex.iSp] = gating.rush_larsen(
            y[StateIndex.iSp], gating.Ito_i_inf(V), tiSp, dt)

        # ICaL gates
        y_new[StateIndex.d] = gating.rush_larsen(
            y[StateIndex.d], gating.ICaL_d_inf(V), gating.ICaL_d_tau(V), dt)
        y_new[StateIndex.ff] = gating.rush_larsen(
            y[StateIndex.ff], gating.ICaL_f_inf(V), gating.ICaL_ff_tau(V), dt)
        y_new[StateIndex.fs] = gating.rush_larsen(
            y[StateIndex.fs], gating.ICaL_f_inf(V), gating.ICaL_fs_tau(V), dt)
        y_new[StateIndex.fcaf] = gating.rush_larsen(
            y[StateIndex.fcaf], gating.ICaL_f_inf(V), gating.ICaL_fcaf_tau(V), dt)
        y_new[StateIndex.fcas] = gating.rush_larsen(
            y[StateIndex.fcas], gating.ICaL_f_inf(V), gating.ICaL_fcas_tau(V), dt)
        y_new[StateIndex.jca] = gating.rush_larsen(
            y[StateIndex.jca], gating.ICaL_f_inf(V), 75.0, dt)
        y_new[StateIndex.ffp] = gating.rush_larsen(
            y[StateIndex.ffp], gating.ICaL_f_inf(V), 2.5 * gating.ICaL_ff_tau(V), dt)
        y_new[StateIndex.fcafp] = gating.rush_larsen(
            y[StateIndex.fcafp], gating.ICaL_f_inf(V), 2.5 * gating.ICaL_fcaf_tau(V), dt)

        # nca (Ca/calmodulin binding) - special handling
        Kmn = 0.002
        k2n = 1000.0
        km2n = y[StateIndex.jca] * 1.0
        anca = 1.0 / (k2n / km2n + (1.0 + Kmn / cass) ** 4.0)
        dnca = anca * k2n - y[StateIndex.nca] * km2n
        y_new[StateIndex.nca] = y[StateIndex.nca] + dt * dnca

        # IKr gates
        y_new[StateIndex.xrf] = gating.rush_larsen(
            y[StateIndex.xrf], gating.IKr_xr_inf(V), gating.IKr_xrf_tau(V), dt)
        y_new[StateIndex.xrs] = gating.rush_larsen(
            y[StateIndex.xrs], gating.IKr_xr_inf(V), gating.IKr_xrs_tau(V), dt)

        # IKs gates
        y_new[StateIndex.xs1] = gating.rush_larsen(
            y[StateIndex.xs1], gating.IKs_xs1_inf(V), gating.IKs_xs1_tau(V), dt)
        y_new[StateIndex.xs2] = gating.rush_larsen(
            y[StateIndex.xs2], gating.IKs_xs1_inf(V), gating.IKs_xs2_tau(V), dt)

        # IK1 gate
        y_new[StateIndex.xk1] = gating.rush_larsen(
            y[StateIndex.xk1], gating.IK1_xk1_inf(V, p.ko), gating.IK1_xk1_tau(V), dt)

        # =================================================================
        # Compute currents
        # =================================================================
        I = self.compute_currents(y, CaMKa)

        # =================================================================
        # Calcium handling
        # =================================================================

        # Diffusion fluxes
        JdiffNa = calcium.J_diffNa(nass, nai)
        JdiffK = calcium.J_diffK(kss, ki)
        Jdiff = calcium.J_diffCa(cass, cai)

        # SR release
        Jrel, Jrelnp_new, Jrelp_new = calcium.compute_Jrel(
            y[StateIndex.Jrelnp], y[StateIndex.Jrelp], fCaMKp,
            I['ICaL'], cajsr, dt, p.bt, celltype_int
        )
        y_new[StateIndex.Jrelnp] = Jrelnp_new
        y_new[StateIndex.Jrelp] = Jrelp_new

        # SERCA uptake
        Jup, Jleak = calcium.compute_Jup(cai, cansr, fCaMKp, celltype_int)

        # SR transfer
        Jtr = calcium.J_tr(cansr, cajsr)

        # Buffering factors
        Bcai = calcium.beta_cai(cai, self.cmdnmax, p.kmcmdn, p.trpnmax, p.kmtrpn)
        Bcass = calcium.beta_cass(cass, p.BSRmax, p.KmBSR, p.BSLmax, p.KmBSL)
        Bcajsr = calcium.beta_cajsr(cajsr, p.csqnmax, p.kmcsqn)

        # =================================================================
        # Update concentrations
        # =================================================================

        # Na+ concentrations
        dnai = -(I['INa'] + I['INaL'] + 3.0 * I['INaCa_i'] + 3.0 * I['INaK'] + I['INab']) * \
               p.Acap / (p.F * p.vmyo) + JdiffNa * p.vss / p.vmyo
        dnass = -(I['ICaNa'] + 3.0 * I['INaCa_ss']) * p.Acap / (p.F * p.vss) - JdiffNa
        y_new[StateIndex.nai] = nai + dt * dnai
        y_new[StateIndex.nass] = nass + dt * dnass

        # K+ concentrations
        dki = -(I['Ito'] + I['IKr'] + I['IKs'] + I['IK1'] + I['IKb'] + Istim - 2.0 * I['INaK']) * \
              p.Acap / (p.F * p.vmyo) + JdiffK * p.vss / p.vmyo
        dkss = -I['ICaK'] * p.Acap / (p.F * p.vss) - JdiffK
        y_new[StateIndex.ki] = ki + dt * dki
        y_new[StateIndex.kss] = kss + dt * dkss

        # Ca2+ concentrations
        dcai = calcium.dcai_dt(I['ICaL'], I['ICab'], I['IpCa'], I['INaCa_i'],
                               Jdiff, Jup, Bcai, p.vmyo, p.vnsr, p.vss, p.Acap, p.F)
        dcass = calcium.dcass_dt(I['ICaL'], I['INaCa_ss'], Jrel, Jdiff,
                                 Bcass, p.vss, p.vjsr, p.Acap, p.F)
        dcansr = calcium.dcansr_dt(Jup, Jtr, p.vjsr, p.vnsr)
        dcajsr = calcium.dcajsr_dt(Jtr, Jrel, Bcajsr)

        y_new[StateIndex.cai] = max(1e-8, cai + dt * dcai)
        y_new[StateIndex.cass] = max(1e-8, cass + dt * dcass)
        y_new[StateIndex.cansr] = max(1e-8, cansr + dt * dcansr)
        y_new[StateIndex.cajsr] = max(1e-8, cajsr + dt * dcajsr)

        # =================================================================
        # Update membrane potential
        # =================================================================
        Iion = (I['INa'] + I['INaL'] + I['Ito'] + I['ICaL'] + I['ICaNa'] + I['ICaK'] +
                I['IKr'] + I['IKs'] + I['IK1'] + I['INaCa'] + I['INaK'] +
                I['INab'] + I['IKb'] + I['IpCa'] + I['ICab'] + Istim)

        y_new[StateIndex.V] = V - dt * Iion

        return y_new

    def simulate(self, t_span: Tuple[float, float], dt: float = 0.01,
                 bcl: float = 1000.0, stim_duration: float = 0.5,
                 stim_amplitude: float = 80.0, stim_start: float = 0.0,
                 y0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation over time span.

        Args:
            t_span: (t_start, t_end) in ms
            dt: Time step (ms)
            bcl: Basic cycle length (ms)
            stim_duration: Stimulus duration (ms)
            stim_amplitude: Stimulus amplitude (uA/uF, positive value)
            stim_start: Stimulus start offset (ms)
            y0: Initial state (uses defaults if None)

        Returns:
            Tuple of (t, y) where t is time array and y is state array
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt) + 1

        t = np.linspace(t_start, t_end, n_steps)
        y = np.zeros((n_steps, StateIndex.N_STATES))

        # Initial conditions
        y[0] = y0 if y0 is not None else self.get_initial_state()

        # Time stepping
        for i in range(1, n_steps):
            # Determine stimulus
            t_in_cycle = (t[i - 1] - stim_start) % bcl
            if t_in_cycle < stim_duration:
                Istim = -stim_amplitude  # Negative for depolarizing
            else:
                Istim = 0.0

            y[i] = self.step(y[i - 1], dt, Istim)

        return t, y
