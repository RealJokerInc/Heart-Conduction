"""
TTP06 Model Implementation

ten Tusscher-Panfilov 2006 human ventricular myocyte model.
Implements the IonicModel interface for use with the simulation engine.

18 ionic state variables (gates + concentrations), V stored separately.
12 ionic currents.

Supports optional LUT (Lookup Table) acceleration for voltage-dependent
gating functions, providing 2-5x speedup for large tissue simulations.

Supports custom cell type configurations via CellTypeConfig for
study-specific parameter variations.

Reference:
ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a
human ventricular tissue model." Am J Physiol Heart Circ Physiol.
"""

import torch
from typing import Optional, Union, Tuple

from ..base import IonicModel, CellType
from .parameters import (
    StateIndex, TTP06Parameters, get_celltype_parameters,
    get_initial_state as _get_initial_state, V_REST
)
from .celltypes.standard import CellTypeConfig
from .gating import (
    rush_larsen,
    # INa gates
    INa_m_inf, INa_m_tau, INa_h_inf, INa_h_tau, INa_j_inf, INa_j_tau,
    # Ito gates
    Ito_r_inf, Ito_r_tau,
    Ito_s_inf_endo, Ito_s_inf_epi, Ito_s_tau_endo, Ito_s_tau_epi,
    # ICaL gates
    ICaL_d_inf, ICaL_d_tau, ICaL_f_inf, ICaL_f_tau,
    ICaL_f2_inf, ICaL_f2_tau, ICaL_fCass_inf, ICaL_fCass_tau,
    # IKr gates
    IKr_Xr1_inf, IKr_Xr1_tau, IKr_Xr2_inf, IKr_Xr2_tau,
    # IKs gate
    IKs_Xs_inf, IKs_Xs_tau,
)
from .currents import (
    I_Na, I_to, I_CaL, I_Kr, I_Ks, I_K1,
    I_NaCa, I_NaK, I_pCa, I_pK, I_bNa, I_bCa
)
from .calcium import update_concentrations


class TTP06Model(IonicModel):
    """
    ten Tusscher-Panfilov 2006 human ventricular myocyte model.

    Parameters
    ----------
    cell_type : CellType
        Cell type: ENDO, EPI, or M_CELL (default: ENDO)
    device : torch.device
        Device for tensors (default: cuda if available)
    dtype : torch.dtype
        Data type (default: float64)
    use_lut : bool
        Use lookup tables for voltage-dependent gating (default: False)
        Provides 2-5x speedup for large tissue simulations
    config : CellTypeConfig, optional
        Custom cell type configuration with parameter overrides.
        If provided, overrides the cell_type parameter settings.
    """

    def __init__(
        self,
        cell_type: CellType = CellType.ENDO,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        use_lut: bool = False,
        config: Optional[CellTypeConfig] = None
    ):
        self.cell_type = cell_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.use_lut = use_lut
        self._lut = None
        self._config = config

        # Get base parameters from cell_type
        self.params = get_celltype_parameters(cell_type)

        # Apply custom config overrides if provided
        if config is not None:
            self._apply_config(config)

        if use_lut:
            from ..lut import get_ttp06_lut
            self._lut = get_ttp06_lut(self.device)

    def _apply_config(self, config: CellTypeConfig):
        """Apply parameter overrides from a CellTypeConfig."""
        overrides = config.get_overrides()
        for param, value in overrides.items():
            if param == 'use_epi_ito_kinetics':
                # Handle kinetics flag separately
                continue
            if hasattr(self.params, param):
                setattr(self.params, param, value)

        # Store kinetics flag for use in step()
        self._use_epi_ito_kinetics = config.use_epi_ito_kinetics

    @property
    def use_epi_ito_kinetics(self) -> bool:
        """Whether to use epicardial Ito inactivation kinetics."""
        if hasattr(self, '_use_epi_ito_kinetics'):
            return self._use_epi_ito_kinetics
        # Default: use EPI kinetics for EPI and M_CELL
        return self.cell_type != CellType.ENDO

    @classmethod
    def from_config(
        cls,
        config: CellTypeConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        use_lut: bool = False,
        base_cell_type: CellType = CellType.EPI
    ) -> 'TTP06Model':
        """
        Create a TTP06Model from a custom CellTypeConfig.

        Parameters
        ----------
        config : CellTypeConfig
            Custom cell type configuration
        device : torch.device, optional
            Device for tensors
        dtype : torch.dtype
            Data type (default: float64)
        use_lut : bool
            Use lookup tables (default: False)
        base_cell_type : CellType
            Base cell type for unspecified parameters (default: EPI)

        Returns
        -------
        TTP06Model
            Model instance with custom parameters
        """
        return cls(
            cell_type=base_cell_type,
            device=device,
            dtype=dtype,
            use_lut=use_lut,
            config=config
        )

    @property
    def name(self) -> str:
        return "TTP06"

    @property
    def n_states(self) -> int:
        return StateIndex.N_STATES

    @property
    def V_rest(self) -> float:
        return V_REST

    @property
    def state_names(self) -> tuple:
        from .parameters import STATE_NAMES
        return STATE_NAMES

    @property
    def gate_indices(self):
        return [
            StateIndex.m, StateIndex.h, StateIndex.j,         # INa
            StateIndex.r, StateIndex.s,                        # Ito
            StateIndex.d, StateIndex.f, StateIndex.f2,         # ICaL (voltage)
            StateIndex.fCass,                                  # ICaL (Ca-dependent)
            StateIndex.Xr1, StateIndex.Xr2,                   # IKr
            StateIndex.Xs,                                     # IKs
        ]

    @property
    def concentration_indices(self):
        return [
            StateIndex.Ki, StateIndex.Nai,
            StateIndex.Cai, StateIndex.CaSR, StateIndex.CaSS,
            StateIndex.RR,
        ]

    def get_initial_state(self, n_cells: int = 1) -> torch.Tensor:
        """
        Get initial ionic state tensor (excludes V).

        Parameters
        ----------
        n_cells : int
            Number of cells (default: 1)

        Returns
        -------
        torch.Tensor
            Shape (18,) if n_cells=1, else (n_cells, 18)
        """
        initial = _get_initial_state(self.device, self.dtype)

        if n_cells == 1:
            return initial
        else:
            return initial.unsqueeze(0).expand(n_cells, -1).clone()

    def step(self, V: torch.Tensor, ionic_states: torch.Tensor, dt: float,
             I_stim: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance state by one time step.

        Uses Rush-Larsen for gating variables, Forward Euler for concentrations.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (mV), shape () or (n_cells,)
        ionic_states : torch.Tensor
            Ionic state tensor, shape (18,) or (n_cells, 18)
        dt : float
            Time step (ms)
        I_stim : torch.Tensor, optional
            Stimulus current (pA/pF), same shape as V

        Returns
        -------
        V_new : torch.Tensor
            Updated membrane potential
        new_ionic_states : torch.Tensor
            Updated ionic state tensor
        """
        # Handle both single-cell and multi-cell cases
        single_cell = ionic_states.dim() == 1
        if single_cell:
            ionic_states = ionic_states.unsqueeze(0)
            V = V.unsqueeze(0)

        # Extract ionic states
        Ki = ionic_states[:, StateIndex.Ki]
        Nai = ionic_states[:, StateIndex.Nai]
        Cai = ionic_states[:, StateIndex.Cai]
        CaSR = ionic_states[:, StateIndex.CaSR]
        CaSS = ionic_states[:, StateIndex.CaSS]

        m = ionic_states[:, StateIndex.m]
        h = ionic_states[:, StateIndex.h]
        j = ionic_states[:, StateIndex.j]

        r = ionic_states[:, StateIndex.r]
        s = ionic_states[:, StateIndex.s]

        d = ionic_states[:, StateIndex.d]
        f = ionic_states[:, StateIndex.f]
        f2 = ionic_states[:, StateIndex.f2]
        fCass = ionic_states[:, StateIndex.fCass]

        Xr1 = ionic_states[:, StateIndex.Xr1]
        Xr2 = ionic_states[:, StateIndex.Xr2]
        Xs = ionic_states[:, StateIndex.Xs]

        RR = ionic_states[:, StateIndex.RR]

        # Get parameters
        p = self.params

        # Compute ionic currents
        INa = I_Na(V, m, h, j, Nai, p.GNa, p.Nao)
        Ito = I_to(V, r, s, Ki, p.Gto, p.Ko)
        ICaL = I_CaL(V, d, f, f2, fCass, CaSS, p.PCa, p.Cao)
        IKr = I_Kr(V, Xr1, Xr2, Ki, p.GKr, p.Ko)
        IKs = I_Ks(V, Xs, Ki, Nai, p.GKs, p.Ko, p.Nao)
        IK1 = I_K1(V, Ki, p.GK1, p.Ko)
        INaCa = I_NaCa(V, Nai, Cai, p.KNaCa, p.Cao, p.Nao,
                       p.KmNai, p.KmCa, p.ksat, p.alpha_ncx, p.gamma_ncx)
        INaK = I_NaK(V, Nai, Ki, p.PNaK, p.Ko, p.Nao, p.KmK, p.KmNa)
        IpCa = I_pCa(Cai, p.GpCa, p.KpCa)
        IpK = I_pK(V, Ki, p.GpK, p.Ko)
        IbNa = I_bNa(V, Nai, p.GbNa, p.Nao)
        IbCa = I_bCa(V, Cai, p.GbCa, p.Cao)

        # Total ionic current
        I_ion = (INa + Ito + ICaL + IKr + IKs + IK1 +
                 INaCa + INaK + IpCa + IpK + IbNa + IbCa)

        # Add stimulus if provided
        if I_stim is not None:
            I_ion = I_ion + I_stim

        # Update membrane potential (Forward Euler)
        dV = -I_ion
        V_new = V + dV * dt

        # Update gating variables (Rush-Larsen)
        if self.use_lut and self._lut is not None:
            # Use lookup tables for voltage-dependent gating
            g = self._lut.get_all_gating(V, celltype_is_endo=(self.cell_type == CellType.ENDO))

            m_new = rush_larsen(m, g['m_inf'], g['m_tau'], dt)
            h_new = rush_larsen(h, g['h_inf'], g['h_tau'], dt)
            j_new = rush_larsen(j, g['j_inf'], g['j_tau'], dt)
            r_new = rush_larsen(r, g['r_inf'], g['r_tau'], dt)
            s_new = rush_larsen(s, g['s_inf'], g['s_tau'], dt)
            d_new = rush_larsen(d, g['d_inf'], g['d_tau'], dt)
            f_new = rush_larsen(f, g['f_inf'], g['f_tau'], dt)
            f2_new = rush_larsen(f2, g['f2_inf'], g['f2_tau'], dt)
            Xr1_new = rush_larsen(Xr1, g['Xr1_inf'], g['Xr1_tau'], dt)
            Xr2_new = rush_larsen(Xr2, g['Xr2_inf'], g['Xr2_tau'], dt)
            Xs_new = rush_larsen(Xs, g['Xs_inf'], g['Xs_tau'], dt)
        else:
            # Direct function evaluation
            # INa gates
            m_new = rush_larsen(m, INa_m_inf(V), INa_m_tau(V), dt)
            h_new = rush_larsen(h, INa_h_inf(V), INa_h_tau(V), dt)
            j_new = rush_larsen(j, INa_j_inf(V), INa_j_tau(V), dt)

            # Ito gates (cell-type dependent kinetics)
            r_new = rush_larsen(r, Ito_r_inf(V), Ito_r_tau(V), dt)

            if self.use_epi_ito_kinetics:
                s_new = rush_larsen(s, Ito_s_inf_epi(V), Ito_s_tau_epi(V), dt)
            else:
                s_new = rush_larsen(s, Ito_s_inf_endo(V), Ito_s_tau_endo(V), dt)

            # ICaL gates
            d_new = rush_larsen(d, ICaL_d_inf(V), ICaL_d_tau(V), dt)
            f_new = rush_larsen(f, ICaL_f_inf(V), ICaL_f_tau(V), dt)
            f2_new = rush_larsen(f2, ICaL_f2_inf(V), ICaL_f2_tau(V), dt)

            # IKr gates
            Xr1_new = rush_larsen(Xr1, IKr_Xr1_inf(V), IKr_Xr1_tau(V), dt)
            Xr2_new = rush_larsen(Xr2, IKr_Xr2_inf(V), IKr_Xr2_tau(V), dt)

            # IKs gate
            Xs_new = rush_larsen(Xs, IKs_Xs_inf(V), IKs_Xs_tau(V), dt)

        # fCass is Ca-dependent, not voltage-dependent - always compute directly
        fCass_new = rush_larsen(fCass, ICaL_fCass_inf(CaSS), ICaL_fCass_tau(CaSS), dt)

        # Update concentrations
        Ki_new, Nai_new, Cai_new, CaSR_new, CaSS_new, RR_new = update_concentrations(
            V, Ki, Nai, Cai, CaSR, CaSS, RR,
            INa, ICaL, Ito, IKr, IKs, IK1,
            INaCa, INaK, IpCa, IpK, IbNa, IbCa,
            dt, p.Cm, p.Vc, p.Vsr, p.Vss
        )

        # Assemble new ionic state
        new_ionic_states = torch.stack([
            Ki_new, Nai_new, Cai_new, CaSR_new, CaSS_new,
            m_new, h_new, j_new,
            r_new, s_new,
            d_new, f_new, f2_new, fCass_new,
            Xr1_new, Xr2_new, Xs_new,
            RR_new
        ], dim=-1)

        if single_cell:
            new_ionic_states = new_ionic_states.squeeze(0)
            V_new = V_new.squeeze(0)

        return V_new, new_ionic_states

    def compute_Iion(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute total ionic current.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential
        ionic_states : torch.Tensor
            Ionic state tensor

        Returns
        -------
        torch.Tensor
            Total ionic current (pA/pF)
        """
        # Handle both single-cell and multi-cell cases
        single_cell = ionic_states.dim() == 1
        if single_cell:
            ionic_states = ionic_states.unsqueeze(0)
            V = V.unsqueeze(0)

        # Extract states
        Ki = ionic_states[:, StateIndex.Ki]
        Nai = ionic_states[:, StateIndex.Nai]
        Cai = ionic_states[:, StateIndex.Cai]
        CaSS = ionic_states[:, StateIndex.CaSS]

        m = ionic_states[:, StateIndex.m]
        h = ionic_states[:, StateIndex.h]
        j = ionic_states[:, StateIndex.j]
        r = ionic_states[:, StateIndex.r]
        s = ionic_states[:, StateIndex.s]
        d = ionic_states[:, StateIndex.d]
        f = ionic_states[:, StateIndex.f]
        f2 = ionic_states[:, StateIndex.f2]
        fCass = ionic_states[:, StateIndex.fCass]
        Xr1 = ionic_states[:, StateIndex.Xr1]
        Xr2 = ionic_states[:, StateIndex.Xr2]
        Xs = ionic_states[:, StateIndex.Xs]

        p = self.params

        # Compute currents
        INa = I_Na(V, m, h, j, Nai, p.GNa, p.Nao)
        Ito = I_to(V, r, s, Ki, p.Gto, p.Ko)
        ICaL = I_CaL(V, d, f, f2, fCass, CaSS, p.PCa, p.Cao)
        IKr = I_Kr(V, Xr1, Xr2, Ki, p.GKr, p.Ko)
        IKs = I_Ks(V, Xs, Ki, Nai, p.GKs, p.Ko, p.Nao)
        IK1 = I_K1(V, Ki, p.GK1, p.Ko)
        INaCa = I_NaCa(V, Nai, Cai, p.KNaCa, p.Cao, p.Nao,
                       p.KmNai, p.KmCa, p.ksat, p.alpha_ncx, p.gamma_ncx)
        INaK = I_NaK(V, Nai, Ki, p.PNaK, p.Ko, p.Nao, p.KmK, p.KmNa)
        IpCa = I_pCa(Cai, p.GpCa, p.KpCa)
        IpK = I_pK(V, Ki, p.GpK, p.Ko)
        IbNa = I_bNa(V, Nai, p.GbNa, p.Nao)
        IbCa = I_bCa(V, Cai, p.GbCa, p.Cao)

        I_ion = (INa + Ito + ICaL + IKr + IKs + IK1 +
                 INaCa + INaK + IpCa + IpK + IbNa + IbCa)

        if single_cell:
            I_ion = I_ion.squeeze(0)

        return I_ion

    def compute_gate_steady_states(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute steady-state values for all 12 gating variables.

        Returns (n_cells, 12) in gate_indices order:
        m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs
        """
        single_cell = V.dim() == 0
        if single_cell:
            V = V.unsqueeze(0)
            ionic_states = ionic_states.unsqueeze(0)

        CaSS = ionic_states[:, StateIndex.CaSS]

        # Ito s_inf depends on cell type
        if self.use_epi_ito_kinetics:
            s_inf = Ito_s_inf_epi(V)
        else:
            s_inf = Ito_s_inf_endo(V)

        result = torch.stack([
            INa_m_inf(V), INa_h_inf(V), INa_j_inf(V),
            Ito_r_inf(V), s_inf,
            ICaL_d_inf(V), ICaL_f_inf(V), ICaL_f2_inf(V),
            ICaL_fCass_inf(CaSS),
            IKr_Xr1_inf(V), IKr_Xr2_inf(V),
            IKs_Xs_inf(V),
        ], dim=-1)

        return result

    def compute_gate_time_constants(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute time constants for all 12 gating variables.

        Returns (n_cells, 12) in gate_indices order.
        """
        single_cell = V.dim() == 0
        if single_cell:
            V = V.unsqueeze(0)
            ionic_states = ionic_states.unsqueeze(0)

        CaSS = ionic_states[:, StateIndex.CaSS]

        if self.use_epi_ito_kinetics:
            s_tau = Ito_s_tau_epi(V)
        else:
            s_tau = Ito_s_tau_endo(V)

        result = torch.stack([
            INa_m_tau(V), INa_h_tau(V), INa_j_tau(V),
            Ito_r_tau(V), s_tau,
            ICaL_d_tau(V), ICaL_f_tau(V), ICaL_f2_tau(V),
            ICaL_fCass_tau(CaSS),
            IKr_Xr1_tau(V), IKr_Xr2_tau(V),
            IKs_Xs_tau(V),
        ], dim=-1)

        return result

    def compute_concentration_rates(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute time derivatives for all 6 concentration variables.

        Returns (n_cells, 6) in concentration_indices order:
        dKi/dt, dNai/dt, dCai/dt, dCaSR/dt, dCaSS/dt, dRR/dt
        """
        single_cell = V.dim() == 0
        if single_cell:
            V = V.unsqueeze(0)
            ionic_states = ionic_states.unsqueeze(0)

        Ki = ionic_states[:, StateIndex.Ki]
        Nai = ionic_states[:, StateIndex.Nai]
        Cai = ionic_states[:, StateIndex.Cai]
        CaSR = ionic_states[:, StateIndex.CaSR]
        CaSS = ionic_states[:, StateIndex.CaSS]
        RR = ionic_states[:, StateIndex.RR]
        m = ionic_states[:, StateIndex.m]
        h = ionic_states[:, StateIndex.h]
        j = ionic_states[:, StateIndex.j]
        r = ionic_states[:, StateIndex.r]
        s = ionic_states[:, StateIndex.s]
        d = ionic_states[:, StateIndex.d]
        f = ionic_states[:, StateIndex.f]
        f2 = ionic_states[:, StateIndex.f2]
        fCass = ionic_states[:, StateIndex.fCass]
        Xr1 = ionic_states[:, StateIndex.Xr1]
        Xr2 = ionic_states[:, StateIndex.Xr2]
        Xs = ionic_states[:, StateIndex.Xs]

        p = self.params

        # Compute all ionic currents
        i_Na = I_Na(V, m, h, j, Nai, p.GNa, p.Nao)
        i_to = I_to(V, r, s, Ki, p.Gto, p.Ko)
        i_CaL = I_CaL(V, d, f, f2, fCass, CaSS, p.PCa, p.Cao)
        i_Kr = I_Kr(V, Xr1, Xr2, Ki, p.GKr, p.Ko)
        i_Ks = I_Ks(V, Xs, Ki, Nai, p.GKs, p.Ko, p.Nao)
        i_K1 = I_K1(V, Ki, p.GK1, p.Ko)
        i_NaCa = I_NaCa(V, Nai, Cai, p.KNaCa, p.Cao, p.Nao,
                         p.KmNai, p.KmCa, p.ksat, p.alpha_ncx, p.gamma_ncx)
        i_NaK = I_NaK(V, Nai, Ki, p.PNaK, p.Ko, p.Nao, p.KmK, p.KmNa)
        i_pCa = I_pCa(Cai, p.GpCa, p.KpCa)
        i_pK = I_pK(V, Ki, p.GpK, p.Ko)
        i_bNa = I_bNa(V, Nai, p.GbNa, p.Nao)
        i_bCa = I_bCa(V, Cai, p.GbCa, p.Cao)

        # Physical constants
        F_const = 96485.3415

        # Volume conversion factors
        inv_VcF = p.Cm / (p.Vc * F_const) * 1000.0
        inv_VssF = p.Cm / (p.Vss * F_const) * 1000.0
        Vsr_Vc = p.Vsr / p.Vc

        # Ca handling fluxes
        from .calcium import I_up, I_rel, I_leak, I_xfer
        from .calcium import buffering_factor_cyt, buffering_factor_sr, buffering_factor_ss

        Iup = I_up(Cai)
        Irel, dRR = I_rel(CaSR, CaSS, RR)
        Ileak = I_leak(CaSR, Cai)
        Ixfer = I_xfer(CaSS, Cai)

        # dKi/dt
        IK_total = i_K1 + i_to + i_Kr + i_Ks + i_pK - 2.0 * i_NaK
        dKi_dt = -IK_total * inv_VcF

        # dNai/dt
        INa_total = i_Na + i_bNa + 3.0 * i_NaK + 3.0 * i_NaCa
        dNai_dt = -INa_total * inv_VcF

        # dCai/dt (with buffering)
        ICa_sarcolemma = i_bCa + i_pCa - 2.0 * i_NaCa
        dCai_unbuffered = (Ileak - Iup) * Vsr_Vc + Ixfer - ICa_sarcolemma * inv_VcF / 2.0
        dCai_dt = dCai_unbuffered * buffering_factor_cyt(Cai)

        # dCaSR/dt (with buffering)
        dCaSR_unbuffered = Iup - Irel * p.Vss / p.Vsr - Ileak
        dCaSR_dt = dCaSR_unbuffered * buffering_factor_sr(CaSR)

        # dCaSS/dt (with buffering)
        dCaSS_unbuffered = Irel - Ixfer * p.Vc / p.Vss - i_CaL * inv_VssF / 2.0
        dCaSS_dt = dCaSS_unbuffered * buffering_factor_ss(CaSS)

        result = torch.stack([dKi_dt, dNai_dt, dCai_dt, dCaSR_dt, dCaSS_dt, dRR], dim=-1)
        return result
