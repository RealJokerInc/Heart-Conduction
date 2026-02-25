"""
O'Hara-Rudy 2011 Model Parameters

Contains:
- StateIndex: Indices into state tensor
- CellType: Endocardial, Epicardial, M-cell variants
- ORdParameters: All model constants and scaling factors
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict
import torch


class StateIndex(IntEnum):
    """
    Indices into the state tensor (41 state variables).

    The state tensor has shape (..., 41) where the last dimension
    contains these variables in order.

    NOTE: Order must match V5 exactly for compatibility!
    V5 groups: bulk concentrations (nai, ki, cai, cansr) then
               subspace concentrations (nass, kss, cass, cajsr)
    """
    # Membrane potential
    V = 0           # Transmembrane potential (mV)

    # Ion concentrations - bulk (matches V5 ordering)
    nai = 1         # Intracellular Na+ (mM)
    ki = 2          # Intracellular K+ (mM)
    cai = 3         # Intracellular Ca2+ (mM)
    cansr = 4       # Network SR Ca2+ (mM)

    # Ion concentrations - subspace (matches V5 ordering)
    nass = 5        # Subspace Na+ (mM)
    kss = 6         # Subspace K+ (mM)
    cass = 7        # Subspace Ca2+ (mM)
    cajsr = 8       # Junctional SR Ca2+ (mM)

    # INa (fast sodium) gates
    m = 9           # Activation
    hf = 10         # Fast inactivation
    hs = 11         # Slow inactivation
    j = 12          # Recovery from inactivation
    hsp = 13        # Phosphorylated fast inactivation
    jp = 14         # Phosphorylated recovery

    # INaL (late sodium) gates
    mL = 15         # Activation
    hL = 16         # Inactivation
    hLp = 17        # Phosphorylated inactivation

    # Ito (transient outward) gates
    a = 18          # Activation
    iF = 19         # Fast inactivation
    iS = 20         # Slow inactivation
    ap = 21         # Phosphorylated activation
    iFp = 22        # Phosphorylated fast inactivation
    iSp = 23        # Phosphorylated slow inactivation

    # ICaL (L-type calcium) gates
    d = 24          # Activation
    ff = 25         # Fast voltage inactivation
    fs = 26         # Slow voltage inactivation
    fcaf = 27       # Fast Ca-dependent inactivation
    fcas = 28       # Slow Ca-dependent inactivation
    jca = 29        # Ca-dependent recovery
    nca = 30        # Ca-dependent inactivation factor
    ffp = 31        # Phosphorylated fast inactivation
    fcafp = 32      # Phosphorylated Ca-dependent inactivation

    # IKr (rapid delayed rectifier) gates
    xrf = 33        # Fast activation
    xrs = 34        # Slow activation

    # IKs (slow delayed rectifier) gates
    xs1 = 35        # Activation gate 1
    xs2 = 36        # Activation gate 2

    # IK1 (inward rectifier) gate
    xk1 = 37        # Activation

    # SR release
    Jrelnp = 38     # Non-phosphorylated release
    Jrelp = 39      # Phosphorylated release

    # CaMKII
    CaMKt = 40      # Trapped CaMKII fraction

    # Total number of states
    N_STATES = 41


class CellType(IntEnum):
    """Cell type variants with different ion channel expression."""
    ENDO = 0    # Endocardial
    EPI = 1     # Epicardial
    M_CELL = 2  # Mid-myocardial (M-cell)


@dataclass
class ORdParameters:
    """
    O'Hara-Rudy 2011 model parameters.

    All parameters use consistent units:
    - Voltage: mV
    - Current: µA/µF (normalized to membrane capacitance)
    - Concentration: mM
    - Time: ms
    - Conductance: mS/µF
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
        """Cell volume (µL)."""
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
        """Myoplasm volume (µL)."""
        return 0.68 * self.vcell

    @property
    def vnsr(self) -> float:
        """Network SR volume (µL)."""
        return 0.0552 * self.vcell

    @property
    def vjsr(self) -> float:
        """Junctional SR volume (µL)."""
        return 0.0048 * self.vcell

    @property
    def vss(self) -> float:
        """Subspace volume (µL)."""
        return 0.02 * self.vcell

    # Membrane capacitance
    Cm: float = 1.0         # µF/cm²

    # Extracellular concentrations
    nao: float = 140.0      # Na+ (mM)
    cao: float = 1.8        # Ca2+ (mM)
    ko: float = 5.4         # K+ (mM)

    # Maximum conductances (mS/µF) - base values, scaled by cell type
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
    cajsr_half: float = 1.5 # JSR Ca for half-max release (mM) - from ORd C++

    # CaMKII parameters
    CaMKo: float = 0.05     # Total CaMKII (fraction)
    KmCaM: float = 0.0015   # CaM affinity (mM)
    KmCaMK: float = 0.15    # CaMKII affinity
    aCaMK: float = 0.05     # Trapping rate (/ms)
    bCaMK: float = 0.00068  # Release rate (/ms)

    # Diffusion time constants (ms)
    tau_diff_Na: float = 2.0    # Na diffusion subspace→cytosol
    tau_diff_K: float = 2.0     # K diffusion subspace→cytosol
    tau_diff_Ca: float = 0.2    # Ca diffusion subspace→cytosol
    tau_tr: float = 100.0       # NSR→JSR transfer

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
        # Epicardial - shorter APD (from ORd C++ and V5)
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
    Get initial state tensor for ORd model.

    These values represent a cell at rest (quasi-steady state).

    Parameters
    ----------
    device : torch.device
        Target device (default: cuda)
    dtype : torch.dtype
        Data type (default: float64)

    Returns
    -------
    torch.Tensor
        Initial state tensor of shape (41,)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.zeros(StateIndex.N_STATES, dtype=dtype, device=device)

    # Initial values from original ORd C++ code (same as V5)
    # Resting membrane potential
    state[StateIndex.V] = -87.5

    # Ion concentrations
    state[StateIndex.nai] = 7.0
    state[StateIndex.nass] = 7.0
    state[StateIndex.ki] = 145.0
    state[StateIndex.kss] = 145.0
    state[StateIndex.cai] = 1.0e-4
    state[StateIndex.cass] = 1.0e-4
    state[StateIndex.cansr] = 1.2
    state[StateIndex.cajsr] = 1.2

    # INa gates
    state[StateIndex.m] = 0.0
    state[StateIndex.hf] = 1.0
    state[StateIndex.hs] = 1.0
    state[StateIndex.j] = 1.0
    state[StateIndex.hsp] = 1.0
    state[StateIndex.jp] = 1.0

    # INaL gates
    state[StateIndex.mL] = 0.0
    state[StateIndex.hL] = 1.0
    state[StateIndex.hLp] = 1.0

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
