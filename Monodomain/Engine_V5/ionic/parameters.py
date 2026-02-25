"""
O'Hara-Rudy (ORd 2011) Model Parameters

Complete parameter set for the ORd human ventricular action potential model.
All values from the original C++/MATLAB implementation.

Reference: O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class CellType(IntEnum):
    """Ventricular cell type enumeration."""
    ENDO = 0    # Endocardial
    EPI = 1     # Epicardial
    M_CELL = 2  # Mid-myocardial (M-cell)


class StateIndex:
    """
    ORd State Variable Indices (0-based).

    Total: 41 state variables
    """
    # Membrane potential (1)
    V = 0

    # Ion concentrations - bulk (4)
    nai = 1      # Intracellular Na+
    ki = 2       # Intracellular K+
    cai = 3      # Intracellular Ca2+
    cansr = 4    # Network SR Ca2+

    # Ion concentrations - subspace (4)
    nass = 5     # Subspace Na+
    kss = 6      # Subspace K+
    cass = 7     # Subspace Ca2+
    cajsr = 8    # Junctional SR Ca2+

    # INa gates (6)
    m = 9        # Activation
    hf = 10      # Fast inactivation
    hs = 11      # Slow inactivation
    j = 12       # Recovery
    hsp = 13     # Slow inactivation (phosphorylated)
    jp = 14      # Recovery (phosphorylated)

    # INaL gates (3)
    mL = 15      # Activation
    hL = 16      # Inactivation
    hLp = 17     # Inactivation (phosphorylated)

    # Ito gates (6)
    a = 18       # Activation
    iF = 19      # Fast inactivation
    iS = 20      # Slow inactivation
    ap = 21      # Activation (phosphorylated)
    iFp = 22     # Fast inactivation (phosphorylated)
    iSp = 23     # Slow inactivation (phosphorylated)

    # ICaL gates (9)
    d = 24       # Activation
    ff = 25      # Fast voltage inactivation
    fs = 26      # Slow voltage inactivation
    fcaf = 27    # Fast Ca inactivation
    fcas = 28    # Slow Ca inactivation
    jca = 29     # Ca-dependent recovery
    nca = 30     # Ca/calmodulin binding
    ffp = 31     # Fast voltage inactivation (phosphorylated)
    fcafp = 32   # Fast Ca inactivation (phosphorylated)

    # IKr gates (2)
    xrf = 33     # Fast activation
    xrs = 34     # Slow activation

    # IKs gates (2)
    xs1 = 35     # Activation 1
    xs2 = 36     # Activation 2

    # IK1 gate (1)
    xk1 = 37     # Activation

    # SR release (2)
    Jrelnp = 38  # Release (non-phosphorylated)
    Jrelp = 39   # Release (phosphorylated)

    # CaMKII (1)
    CaMKt = 40   # Trapped CaMKII

    # Total count
    N_STATES = 41


# State variable names for debugging/output
STATE_NAMES = [
    'V', 'nai', 'ki', 'cai', 'cansr',
    'nass', 'kss', 'cass', 'cajsr',
    'm', 'hf', 'hs', 'j', 'hsp', 'jp',
    'mL', 'hL', 'hLp',
    'a', 'iF', 'iS', 'ap', 'iFp', 'iSp',
    'd', 'ff', 'fs', 'fcaf', 'fcas', 'jca', 'nca', 'ffp', 'fcafp',
    'xrf', 'xrs',
    'xs1', 'xs2',
    'xk1',
    'Jrelnp', 'Jrelp',
    'CaMKt'
]


@dataclass
class ORdParameters:
    """
    Complete parameter set for ORd 2011 model.

    All values from original C++/MATLAB code.
    """

    # =========================================================================
    # Physical Constants
    # =========================================================================
    F: float = 96485.0          # Faraday constant (C/mol)
    R: float = 8314.0           # Gas constant (mJ/(mol·K))
    T: float = 310.0            # Temperature (K)

    @property
    def RTF(self) -> float:
        """R*T/F for Nernst potential calculations."""
        return self.R * self.T / self.F

    # =========================================================================
    # Cell Geometry
    # =========================================================================
    L: float = 0.01             # Cell length (cm)
    rad: float = 0.0011         # Cell radius (cm)

    @property
    def vcell(self) -> float:
        """Cell volume (uL)."""
        return 1000.0 * 3.14159 * self.rad * self.rad * self.L

    @property
    def Ageo(self) -> float:
        """Geometric membrane area (cm^2)."""
        return 2.0 * 3.14159 * self.rad * self.rad + 2.0 * 3.14159 * self.rad * self.L

    @property
    def Acap(self) -> float:
        """Capacitive membrane area (cm^2)."""
        return 2.0 * self.Ageo

    @property
    def vmyo(self) -> float:
        """Myoplasm volume (uL) - 68% of cell."""
        return 0.68 * self.vcell

    @property
    def vnsr(self) -> float:
        """Network SR volume (uL) - 5.52% of cell."""
        return 0.0552 * self.vcell

    @property
    def vjsr(self) -> float:
        """Junctional SR volume (uL) - 0.48% of cell."""
        return 0.0048 * self.vcell

    @property
    def vss(self) -> float:
        """Subspace volume (uL) - 2% of cell."""
        return 0.02 * self.vcell

    # =========================================================================
    # Extracellular Concentrations (mM)
    # =========================================================================
    nao: float = 140.0          # Extracellular Na+
    ko: float = 5.4             # Extracellular K+
    cao: float = 1.8            # Extracellular Ca2+

    # =========================================================================
    # Stimulus
    # =========================================================================
    stim_amp: float = -80.0     # Stimulus amplitude (uA/uF)
    stim_duration: float = 0.5  # Stimulus duration (ms)

    # =========================================================================
    # CaMKII Parameters
    # =========================================================================
    aCaMK: float = 0.05         # CaMKII activation rate
    bCaMK: float = 0.00068      # CaMKII deactivation rate
    CaMKo: float = 0.05         # Total CaMKII concentration
    KmCaM: float = 0.0015       # CaM binding Kd (mM)
    KmCaMK: float = 0.15        # CaMKII effect Kd

    # =========================================================================
    # INa - Fast Sodium Current
    # =========================================================================
    GNa: float = 75.0           # Max conductance (mS/uF)

    # =========================================================================
    # INaL - Late Sodium Current
    # =========================================================================
    GNaL: float = 0.0075        # Max conductance (mS/uF)
    thL: float = 200.0          # hL time constant (ms)

    # =========================================================================
    # Ito - Transient Outward K+ Current
    # =========================================================================
    Gto: float = 0.02           # Max conductance (mS/uF)

    # =========================================================================
    # ICaL - L-type Calcium Current
    # =========================================================================
    PCa: float = 0.0001         # Ca permeability (cm/s)
    PCaNa: float = 0.00125      # Na permeability ratio (* PCa)
    PCaK: float = 3.574e-4      # K permeability ratio (* PCa)

    # =========================================================================
    # IKr - Rapid Delayed Rectifier K+ Current
    # =========================================================================
    GKr: float = 0.046          # Max conductance (mS/uF)

    # =========================================================================
    # IKs - Slow Delayed Rectifier K+ Current
    # =========================================================================
    GKs: float = 0.0034         # Max conductance (mS/uF)

    # =========================================================================
    # IK1 - Inward Rectifier K+ Current
    # =========================================================================
    GK1: float = 0.1908         # Max conductance (mS/uF)

    # =========================================================================
    # INaCa - Na/Ca Exchanger
    # =========================================================================
    Gncx: float = 0.0008        # Scaling factor

    # =========================================================================
    # INaK - Na/K Pump
    # =========================================================================
    Pnak: float = 30.0          # Max pump rate (uA/uF)

    # =========================================================================
    # Background Currents
    # =========================================================================
    GKb: float = 0.003          # IKb conductance (mS/uF)
    PNab: float = 3.75e-10      # INab permeability (cm/s)
    PCab: float = 2.5e-8        # ICab permeability (cm/s)
    GpCa: float = 0.0005        # IpCa max current (uA/uF)

    # =========================================================================
    # SR Calcium Handling
    # =========================================================================
    bt: float = 4.75            # Release time constant base (ms)

    # =========================================================================
    # Calcium Buffering
    # =========================================================================
    cmdnmax: float = 0.05       # Calmodulin max (mM)
    kmcmdn: float = 0.00238     # Calmodulin Kd (mM)
    trpnmax: float = 0.07       # Troponin max (mM)
    kmtrpn: float = 0.0005      # Troponin Kd (mM)
    BSRmax: float = 0.047       # SR binding sites max (mM)
    KmBSR: float = 0.00087      # SR binding Kd (mM)
    BSLmax: float = 1.124       # Sarcolemmal binding max (mM)
    KmBSL: float = 0.0087       # Sarcolemmal binding Kd (mM)
    csqnmax: float = 10.0       # Calsequestrin max (mM)
    kmcsqn: float = 0.8         # Calsequestrin Kd (mM)

    # =========================================================================
    # Cell Type Scaling
    # =========================================================================
    def get_celltype_scales(self, celltype: CellType) -> dict:
        """
        Get scaling factors for cell-type specific parameters.

        Args:
            celltype: CellType enum (ENDO, EPI, M_CELL)

        Returns:
            Dictionary of parameter scaling factors
        """
        scales = {
            'GNaL': 1.0,
            'Gto': 1.0,
            'PCa': 1.0,
            'GKr': 1.0,
            'GKs': 1.0,
            'GK1': 1.0,
            'Gncx': 1.0,
            'Pnak': 1.0,
            'GKb': 1.0,
            'Jrel_scale': 1.0,
            'Jup_scale': 1.0,
            'cmdnmax': 1.0,
        }

        if celltype == CellType.EPI:
            scales['GNaL'] = 0.6
            scales['Gto'] = 4.0
            scales['PCa'] = 1.2
            scales['GKr'] = 1.3
            scales['GKs'] = 1.4
            scales['GK1'] = 1.2
            scales['Gncx'] = 1.1
            scales['Pnak'] = 0.9
            scales['GKb'] = 0.6
            scales['Jup_scale'] = 1.3
            scales['cmdnmax'] = 1.3

        elif celltype == CellType.M_CELL:
            scales['Gto'] = 4.0
            scales['PCa'] = 2.5
            scales['GKr'] = 0.8
            scales['GK1'] = 1.3
            scales['Gncx'] = 1.4
            scales['Pnak'] = 0.7
            scales['Jrel_scale'] = 1.7

        return scales

    # =========================================================================
    # Initial Conditions
    # =========================================================================
    def get_initial_state(self, celltype: CellType = CellType.ENDO) -> np.ndarray:
        """
        Return initial state vector (41 variables).

        Values from original ORd C++ code.
        """
        y0 = np.zeros(StateIndex.N_STATES, dtype=np.float64)

        # Membrane potential
        y0[StateIndex.V] = -87.5

        # Ion concentrations - bulk
        y0[StateIndex.nai] = 7.0
        y0[StateIndex.ki] = 145.0
        y0[StateIndex.cai] = 1.0e-4
        y0[StateIndex.cansr] = 1.2

        # Ion concentrations - subspace
        y0[StateIndex.nass] = 7.0
        y0[StateIndex.kss] = 145.0
        y0[StateIndex.cass] = 1.0e-4
        y0[StateIndex.cajsr] = 1.2

        # INa gates
        y0[StateIndex.m] = 0.0
        y0[StateIndex.hf] = 1.0
        y0[StateIndex.hs] = 1.0
        y0[StateIndex.j] = 1.0
        y0[StateIndex.hsp] = 1.0
        y0[StateIndex.jp] = 1.0

        # INaL gates
        y0[StateIndex.mL] = 0.0
        y0[StateIndex.hL] = 1.0
        y0[StateIndex.hLp] = 1.0

        # Ito gates
        y0[StateIndex.a] = 0.0
        y0[StateIndex.iF] = 1.0
        y0[StateIndex.iS] = 1.0
        y0[StateIndex.ap] = 0.0
        y0[StateIndex.iFp] = 1.0
        y0[StateIndex.iSp] = 1.0

        # ICaL gates
        y0[StateIndex.d] = 0.0
        y0[StateIndex.ff] = 1.0
        y0[StateIndex.fs] = 1.0
        y0[StateIndex.fcaf] = 1.0
        y0[StateIndex.fcas] = 1.0
        y0[StateIndex.jca] = 1.0
        y0[StateIndex.nca] = 0.0
        y0[StateIndex.ffp] = 1.0
        y0[StateIndex.fcafp] = 1.0

        # IKr gates
        y0[StateIndex.xrf] = 0.0
        y0[StateIndex.xrs] = 0.0

        # IKs gates
        y0[StateIndex.xs1] = 0.0
        y0[StateIndex.xs2] = 0.0

        # IK1 gate
        y0[StateIndex.xk1] = 1.0

        # SR release
        y0[StateIndex.Jrelnp] = 0.0
        y0[StateIndex.Jrelp] = 0.0

        # CaMKII
        y0[StateIndex.CaMKt] = 0.0

        return y0


# Default parameter instance
DEFAULT_PARAMS = ORdParameters()
