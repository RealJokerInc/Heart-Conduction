"""
O'Hara-Rudy 2011 Model Parameters

Contains:
- StateIndex: Indices into state tensor
- ORdParameters: All model constants and scaling factors
"""

from enum import IntEnum
from dataclasses import dataclass
import torch

from ..base import CellType


class StateIndex(IntEnum):
    """
    Indices into the ionic state tensor (40 state variables).

    V is stored separately from ionic states.
    The ionic state tensor has shape (..., 40) where the last dimension
    contains these variables in order.
    """
    # Ion concentrations - bulk
    nai = 0         # Intracellular Na+ (mM)
    ki = 1          # Intracellular K+ (mM)
    cai = 2         # Intracellular Ca2+ (mM)
    cansr = 3       # Network SR Ca2+ (mM)

    # Ion concentrations - subspace
    nass = 4        # Subspace Na+ (mM)
    kss = 5         # Subspace K+ (mM)
    cass = 6        # Subspace Ca2+ (mM)
    cajsr = 7       # Junctional SR Ca2+ (mM)

    # INa (fast sodium) gates
    m = 8           # Activation
    hf = 9          # Fast inactivation
    hs = 10         # Slow inactivation
    j = 11          # Recovery from inactivation
    hsp = 12        # Phosphorylated fast inactivation
    jp = 13         # Phosphorylated recovery

    # INaL (late sodium) gates
    mL = 14         # Activation
    hL = 15         # Inactivation
    hLp = 16        # Phosphorylated inactivation

    # Ito (transient outward) gates
    a = 17          # Activation
    iF = 18         # Fast inactivation
    iS = 19         # Slow inactivation
    ap = 20         # Phosphorylated activation
    iFp = 21        # Phosphorylated fast inactivation
    iSp = 22        # Phosphorylated slow inactivation

    # ICaL (L-type calcium) gates
    d = 23          # Activation
    ff = 24         # Fast voltage inactivation
    fs = 25         # Slow voltage inactivation
    fcaf = 26       # Fast Ca-dependent inactivation
    fcas = 27       # Slow Ca-dependent inactivation
    jca = 28        # Ca-dependent recovery
    nca = 29        # Ca-dependent inactivation factor
    ffp = 30        # Phosphorylated fast inactivation
    fcafp = 31      # Phosphorylated Ca-dependent inactivation

    # IKr (rapid delayed rectifier) gates
    xrf = 32        # Fast activation
    xrs = 33        # Slow activation

    # IKs (slow delayed rectifier) gates
    xs1 = 34        # Activation gate 1
    xs2 = 35        # Activation gate 2

    # IK1 (inward rectifier) gate
    xk1 = 36        # Activation

    # SR release
    Jrelnp = 37     # Non-phosphorylated release
    Jrelp = 38      # Phosphorylated release

    # CaMKII
    CaMKt = 39      # Trapped CaMKII fraction

    # Total number of states (gates + concentrations, excludes V)
    N_STATES = 40


# State variable names in order (excludes V)
STATE_NAMES = (
    'nai', 'ki', 'cai', 'cansr', 'nass', 'kss', 'cass', 'cajsr',
    'm', 'hf', 'hs', 'j', 'hsp', 'jp',
    'mL', 'hL', 'hLp',
    'a', 'iF', 'iS', 'ap', 'iFp', 'iSp',
    'd', 'ff', 'fs', 'fcaf', 'fcas', 'jca', 'nca', 'ffp', 'fcafp',
    'xrf', 'xrs', 'xs1', 'xs2', 'xk1',
    'Jrelnp', 'Jrelp', 'CaMKt'
)


@dataclass
class ORdParameters:
    """
    O'Hara-Rudy 2011 model parameters.

    All parameters use consistent units:
    - Voltage: mV
    - Current: uA/uF (normalized to membrane capacitance)
    - Concentration: mM
    - Time: ms
    - Conductance: mS/uF
    """

    # Physical constants
    R: float = 8314.0       # Gas constant (mJ/(mol·K))
    T: float = 310.0        # Temperature (K) - 37°C
    F: float = 96485.0      # Faraday constant (C/mol)

    # Derived constant
    @property
    def RTF(self) -> float:
        """R*T/F in mV (used for Nernst potentials)."""
        return self.R * self.T / self.F  # ~26.71 mV at 37°C

    # Cell geometry
    L: float = 0.01         # Cell length (cm)
    rad: float = 0.0011     # Cell radius (cm)

    @property
    def vcell(self) -> float:
        """Cell volume (uL)."""
        import math
        return 1000 * math.pi * self.rad**2 * self.L

    @property
    def Ageo(self) -> float:
        """Geometric membrane area (cm²)."""
        import math
        return 2 * math.pi * self.rad**2 + 2 * math.pi * self.rad * self.L

    @property
    def Acap(self) -> float:
        """Capacitive membrane area (cm²)."""
        return 2 * self.Ageo

    @property
    def vmyo(self) -> float:
        """Myoplasm volume (uL)."""
        return 0.68 * self.vcell

    @property
    def vnsr(self) -> float:
        """Network SR volume (uL)."""
        return 0.0552 * self.vcell

    @property
    def vjsr(self) -> float:
        """Junctional SR volume (uL)."""
        return 0.0048 * self.vcell

    @property
    def vss(self) -> float:
        """Subspace volume (uL)."""
        return 0.02 * self.vcell

    # Membrane capacitance
    Cm: float = 1.0         # uF/cm²

    # Extracellular concentrations
    nao: float = 140.0      # Na+ (mM)
    cao: float = 1.8        # Ca2+ (mM)
    ko: float = 5.4         # K+ (mM)

    # Maximum conductances (mS/uF) - base values, scaled by cell type
    GNa: float = 75.0       # INa
    GNaL: float = 0.0075    # INaL (will be scaled)
    Gto: float = 0.02       # Ito (will be scaled)
    GKr: float = 0.046      # IKr (will be scaled)
    GKs: float = 0.0034     # IKs (will be scaled)
    GK1: float = 0.1908     # IK1
    GKb: float = 0.003      # IKb
    GpCa: float = 0.0005    # IpCa
    PCa: float = 0.0001     # ICaL permeability (cm/s)
    PNab: float = 3.75e-10  # INab permeability
    PCab: float = 2.5e-8    # ICab permeability

    # INaCa parameters
    Gncx: float = 0.0008    # Scaling factor
    KmCaAct: float = 150e-6 # Ca activation Km (mM)
    kna1: float = 15.0      # Na binding rate 1
    kna2: float = 5.0       # Na binding rate 2
    kna3: float = 88.12     # Na binding rate 3
    kasymm: float = 12.5    # Asymmetry factor
    wna: float = 6e4        # Na occlusion rate
    wca: float = 6e4        # Ca occlusion rate
    wnaca: float = 5e3      # Na-Ca occlusion rate
    kcaon: float = 1.5e6    # Ca on-rate
    kcaoff: float = 5e3     # Ca off-rate
    qna: float = 0.5224     # Na charge movement
    qca: float = 0.167      # Ca charge movement

    # INaK parameters
    Pnak: float = 30.0      # Maximum flux
    k1p: float = 949.5      # Rate constant
    k1m: float = 182.4      # Rate constant
    k2p: float = 687.2      # Rate constant
    k2m: float = 39.4       # Rate constant
    k3p: float = 1899.0     # Rate constant
    k3m: float = 79300.0    # Rate constant
    k4p: float = 639.0      # Rate constant
    k4m: float = 40.0       # Rate constant
    Knai0: float = 9.073    # Na affinity
    Knao0: float = 27.78    # External Na affinity
    delta_eNa: float = -0.1550  # Voltage dependence
    Kki: float = 0.5        # K affinity
    Kko: float = 0.3582     # External K affinity
    MgADP: float = 0.05     # ADP concentration
    MgATP: float = 9.8      # ATP concentration
    Kmgatp: float = 1.698e-7    # ATP affinity
    H: float = 1e-7         # H+ concentration
    eP: float = 4.2         # Phosphorylation energy
    Khp: float = 1.698e-7   # H+ affinity
    Knap: float = 224.0     # Na+ affinity for phosphorylation
    Kxkur: float = 292.0    # K+ affinity

    # Calcium handling parameters
    cmdnmax: float = 0.05   # Calmodulin max (mM)
    kmcmdn: float = 0.00238 # Calmodulin Km (mM)
    trpnmax: float = 0.07   # Troponin max (mM)
    kmtrpn: float = 0.0005  # Troponin Km (mM)
    BSRmax: float = 0.047   # SR binding site max (mM)
    KmBSR: float = 0.00087  # SR binding Km (mM)
    BSLmax: float = 1.124   # SL binding site max (mM)
    KmBSL: float = 0.0087   # SL binding Km (mM)
    csqnmax: float = 10.0   # Calsequestrin max (mM)
    kmcsqn: float = 0.8     # Calsequestrin Km (mM)

    # SERCA parameters
    Jup_max: float = 0.004375  # Max uptake rate (mM/ms)
    Jup_b: float = 1.0      # Baseline scaling
    Kmup: float = 0.00092   # Uptake Km (mM)
    nsrbar: float = 15.0    # NSR max concentration (mM)

    # SR release parameters
    bt: float = 4.75        # Release time constant base (ms)
    a_rel: float = 0.5      # Release amplitude (mM/ms)
    cajsr_half: float = 1.5 # JSR Ca for half-max release (mM)

    # CaMKII parameters
    CaMKo: float = 0.05     # Total CaMKII (fraction)
    KmCaM: float = 0.0015   # CaM affinity (mM)
    KmCaMK: float = 0.15    # CaMKII affinity
    aCaMK: float = 0.05     # Trapping rate (/ms)
    bCaMK: float = 0.00068  # Release rate (/ms)

    # Diffusion time constants (ms)
    tau_diff_Na: float = 2.0    # Na diffusion subspace->cytosol
    tau_diff_K: float = 2.0     # K diffusion subspace->cytosol
    tau_diff_Ca: float = 0.2    # Ca diffusion subspace->cytosol
    tau_tr: float = 100.0       # NSR->JSR transfer

    # Cell-type specific scaling factors (set by get_celltype_parameters)
    GNaL_scale: float = 1.0
    Gto_scale: float = 1.0
    GKr_scale: float = 1.0
    GKs_scale: float = 1.0
    GK1_scale: float = 1.0
    GKb_scale: float = 1.0      # IKb conductance scale
    Gncx_scale: float = 1.0
    Pnak_scale: float = 1.0     # INaK pump scale
    PCa_scale: float = 1.0      # ICaL permeability scale
    Jup_scale: float = 1.0
    Jrel_scale: float = 1.0     # SR release scale
    cmdnmax_scale: float = 1.0  # Calmodulin buffering scale
    tau_hs_scale: float = 1.0
    tau_hsp_scale: float = 1.0


def get_celltype_parameters(celltype: CellType) -> ORdParameters:
    """
    Get parameters scaled for specific cell type.

    Parameters
    ----------
    celltype : CellType
        ENDO, EPI, or M_CELL

    Returns
    -------
    ORdParameters
        Parameters with cell-type specific scaling
    """
    params = ORdParameters()

    if celltype == CellType.ENDO:
        # Endocardial - baseline (all scales = 1.0)
        pass  # All defaults are 1.0

    elif celltype == CellType.EPI:
        # Epicardial - shorter APD
        params.GNaL_scale = 0.6
        params.Gto_scale = 4.0      # Much larger Ito
        params.PCa_scale = 1.2      # Larger ICaL
        params.GKr_scale = 1.3
        params.GKs_scale = 1.4
        params.GK1_scale = 1.2
        params.GKb_scale = 0.6      # Smaller IKb
        params.Gncx_scale = 1.1
        params.Pnak_scale = 0.9     # Smaller INaK
        params.Jup_scale = 1.3
        params.cmdnmax_scale = 1.3  # More calmodulin buffering

    elif celltype == CellType.M_CELL:
        # M-cell - longest APD
        params.Gto_scale = 4.0
        params.PCa_scale = 2.5      # Much larger ICaL
        params.GKr_scale = 0.8      # Smaller IKr
        params.GK1_scale = 1.3
        params.Gncx_scale = 1.4
        params.Pnak_scale = 0.7     # Much smaller INaK
        params.Jrel_scale = 1.7     # Larger SR release

    return params


def get_initial_state(device: torch.device = None,
                      dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Get initial ionic state tensor for ORd model (excludes V).

    These values represent a cell at rest (quasi-steady state).

    Parameters
    ----------
    device : torch.device
        Target device (default: cuda if available)
    dtype : torch.dtype
        Data type (default: float64)

    Returns
    -------
    torch.Tensor
        Initial ionic state tensor of shape (40,)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.zeros(StateIndex.N_STATES, dtype=dtype, device=device)

    # Ion concentrations
    state[StateIndex.nai] = 7.0
    state[StateIndex.nass] = 7.0
    state[StateIndex.ki] = 145.0
    state[StateIndex.kss] = 145.0
    state[StateIndex.cai] = 1.0e-4
    state[StateIndex.cass] = 1.0e-4
    state[StateIndex.cansr] = 1.2
    state[StateIndex.cajsr] = 1.2

    # INa gates - FIXED: Use steady-state values at V_rest = -87.5 mV
    # h_inf(-87.5) = 1/(1+exp((-87.5+82.9)/6.086)) = 0.6804
    # hsp_inf(-87.5) = 1/(1+exp((-87.5+89.1)/6.086)) = 0.4347
    state[StateIndex.m] = 0.0077  # m_inf at -87.5 mV
    state[StateIndex.hf] = 0.6804  # h_inf at -87.5 mV
    state[StateIndex.hs] = 0.6804  # h_inf at -87.5 mV
    state[StateIndex.j] = 0.6804   # j_inf = h_inf at -87.5 mV
    state[StateIndex.hsp] = 0.4347  # hsp_inf at -87.5 mV
    state[StateIndex.jp] = 0.4347   # jp_inf = hsp_inf at -87.5 mV

    # INaL gates - FIXED: Use steady-state values at V_rest = -87.5 mV
    state[StateIndex.mL] = 0.0002  # mL_inf at -87.5 mV
    state[StateIndex.hL] = 0.4963  # hL_inf at -87.5 mV
    state[StateIndex.hLp] = 0.3010  # hLp_inf at -87.5 mV

    # Ito gates
    state[StateIndex.a] = 0.0
    state[StateIndex.iF] = 1.0
    state[StateIndex.iS] = 1.0
    state[StateIndex.ap] = 0.0
    state[StateIndex.iFp] = 1.0
    state[StateIndex.iSp] = 1.0

    # ICaL gates
    state[StateIndex.d] = 0.0
    state[StateIndex.ff] = 1.0
    state[StateIndex.fs] = 1.0
    state[StateIndex.fcaf] = 1.0
    state[StateIndex.fcas] = 1.0
    state[StateIndex.jca] = 1.0
    state[StateIndex.nca] = 0.0
    state[StateIndex.ffp] = 1.0
    state[StateIndex.fcafp] = 1.0

    # IKr gates
    state[StateIndex.xrf] = 0.0
    state[StateIndex.xrs] = 0.0

    # IKs gates
    state[StateIndex.xs1] = 0.0
    state[StateIndex.xs2] = 0.0

    # IK1 gate
    state[StateIndex.xk1] = 1.0

    # SR release
    state[StateIndex.Jrelnp] = 0.0
    state[StateIndex.Jrelp] = 0.0

    # CaMKII
    state[StateIndex.CaMKt] = 0.0

    return state


# Resting membrane potential (mV) — separated from ionic states
V_REST = -87.5
