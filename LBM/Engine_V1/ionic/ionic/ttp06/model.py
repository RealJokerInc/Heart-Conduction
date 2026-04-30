"""
TTP06 Model Implementation

ten Tusscher-Panfilov 2006 human ventricular myocyte model.
Implements the IonicModel interface for use with the simulation engine.

19 state variables, 12 ionic currents.

Supports optional LUT (Lookup Table) acceleration for voltage-dependent
gating functions, providing 2-5x speedup for large tissue simulations.

Supports custom cell type configurations via CellTypeConfig for
study-specific parameter variations.

Reference:
ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a
human ventricular tissue model." Am J Physiol Heart Circ Physiol.
"""

import torch
from typing import Optional, Union

from ionic.base import IonicModel, CellType
from ionic.ttp06.parameters import (
    StateIndex, TTP06Parameters, get_celltype_parameters,
    get_initial_state as _get_initial_state
)
from ionic.ttp06.celltypes.standard import CellTypeConfig
from ionic.ttp06.gating import (
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
from ionic.ttp06.currents import (
    I_Na, I_to, I_CaL, I_Kr, I_Ks, I_K1,
    I_NaCa, I_NaK, I_pCa, I_pK, I_bNa, I_bCa
)
from ionic.ttp06.calcium import update_concentrations


class TTP06Model(IonicModel):
    """
    ten Tusscher-Panfilov 2006 human ventricular myocyte model.

    Parameters
    ----------
    celltype : CellType
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
        If provided, overrides the celltype parameter settings.
    """

    def __init__(
        self,
        celltype: CellType = CellType.ENDO,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        use_lut: bool = False,
        config: Optional[CellTypeConfig] = None
    ):
        self.celltype = celltype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.use_lut = use_lut
        self._lut = None
        self._config = config

        # Get base parameters from celltype
        self.params = get_celltype_parameters(celltype)

        # Apply custom config overrides if provided
        if config is not None:
            self._apply_config(config)

        if use_lut:
            from ionic.lut import get_ttp06_lut
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
        return self.celltype != CellType.ENDO

    @classmethod
    def from_config(
        cls,
        config: CellTypeConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        use_lut: bool = False,
        base_celltype: CellType = CellType.EPI
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
        base_celltype : CellType
            Base cell type for unspecified parameters (default: EPI)

        Returns
        -------
        TTP06Model
            Model instance with custom parameters
        """
        return cls(
            celltype=base_celltype,
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
    def state_names(self) -> tuple:
        from ionic.ttp06.parameters import STATE_NAMES
        return STATE_NAMES

    @property
    def V_index(self) -> int:
        return StateIndex.V

    def get_initial_state(self, n_cells: int = 1) -> torch.Tensor:
        """
        Get initial state tensor.

        Parameters
        ----------
        n_cells : int
            Number of cells (default: 1)

        Returns
        -------
        torch.Tensor
            Shape (19,) if n_cells=1, else (n_cells, 19)
        """
        initial = _get_initial_state(self.device, self.dtype)

        if n_cells == 1:
            return initial
        else:
            return initial.unsqueeze(0).expand(n_cells, -1).clone()

    def step(self, states: torch.Tensor, dt: float,
             I_stim: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Advance state by one time step.

        Uses Rush-Larsen for gating variables, Forward Euler for concentrations.

        Parameters
        ----------
        states : torch.Tensor
            State tensor, shape (19,) or (n_cells, 19)
        dt : float
            Time step (ms)
        I_stim : torch.Tensor, optional
            Stimulus current (pA/pF), same shape as V

        Returns
        -------
        torch.Tensor
            Updated state tensor
        """
        # Handle both single-cell and multi-cell cases
        single_cell = states.dim() == 1
        if single_cell:
            states = states.unsqueeze(0)

        # Extract states
        V = states[:, StateIndex.V]
        Ki = states[:, StateIndex.Ki]
        Nai = states[:, StateIndex.Nai]
        Cai = states[:, StateIndex.Cai]
        CaSR = states[:, StateIndex.CaSR]
        CaSS = states[:, StateIndex.CaSS]

        m = states[:, StateIndex.m]
        h = states[:, StateIndex.h]
        j = states[:, StateIndex.j]

        r = states[:, StateIndex.r]
        s = states[:, StateIndex.s]

        d = states[:, StateIndex.d]
        f = states[:, StateIndex.f]
        f2 = states[:, StateIndex.f2]
        fCass = states[:, StateIndex.fCass]

        Xr1 = states[:, StateIndex.Xr1]
        Xr2 = states[:, StateIndex.Xr2]
        Xs = states[:, StateIndex.Xs]

        RR = states[:, StateIndex.RR]

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
            g = self._lut.get_all_gating(V, celltype_is_endo=(self.celltype == CellType.ENDO))

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

        # Assemble new state
        new_states = torch.stack([
            V_new, Ki_new, Nai_new, Cai_new, CaSR_new, CaSS_new,
            m_new, h_new, j_new,
            r_new, s_new,
            d_new, f_new, f2_new, fCass_new,
            Xr1_new, Xr2_new, Xs_new,
            RR_new
        ], dim=-1)

        if single_cell:
            new_states = new_states.squeeze(0)

        return new_states

    def compute_Iion(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute total ionic current.

        Parameters
        ----------
        states : torch.Tensor
            State tensor

        Returns
        -------
        torch.Tensor
            Total ionic current (pA/pF)
        """
        # Handle both single-cell and multi-cell cases
        single_cell = states.dim() == 1
        if single_cell:
            states = states.unsqueeze(0)

        # Extract states
        V = states[:, StateIndex.V]
        Ki = states[:, StateIndex.Ki]
        Nai = states[:, StateIndex.Nai]
        Cai = states[:, StateIndex.Cai]
        CaSS = states[:, StateIndex.CaSS]

        m = states[:, StateIndex.m]
        h = states[:, StateIndex.h]
        j = states[:, StateIndex.j]
        r = states[:, StateIndex.r]
        s = states[:, StateIndex.s]
        d = states[:, StateIndex.d]
        f = states[:, StateIndex.f]
        f2 = states[:, StateIndex.f2]
        fCass = states[:, StateIndex.fCass]
        Xr1 = states[:, StateIndex.Xr1]
        Xr2 = states[:, StateIndex.Xr2]
        Xs = states[:, StateIndex.Xs]

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
