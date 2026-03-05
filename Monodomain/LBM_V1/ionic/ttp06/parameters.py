"""
ten Tusscher-Panfilov 2006 Model Parameters

Contains:
- StateIndex: Indices into state tensor (19 states)
- TTP06Parameters: All model constants and scaling factors

Reference:
ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a
human ventricular tissue model." Am J Physiol Heart Circ Physiol.
"""

from enum import IntEnum
from dataclasses import dataclass
import torch

from ..base import CellType


class StateIndex(IntEnum):
    """
    Indices into the ionic state tensor (18 state variables).

    V is stored separately from ionic states.
    Order follows original TTP06 publication (minus V).
    """
    # Ion concentrations
    Ki = 0          # Intracellular K+ (mM)
    Nai = 1         # Intracellular Na+ (mM)
    Cai = 2         # Intracellular Ca2+ cytoplasm (mM)
    CaSR = 3        # SR Ca2+ (mM)
    CaSS = 4        # Subspace Ca2+ (mM)

    # INa gates
    m = 5           # Activation
    h = 6           # Fast inactivation
    j = 7           # Slow inactivation

    # Ito gates
    r = 8           # Activation
    s = 9           # Inactivation

    # ICaL gates
    d = 10          # Activation
    f = 11          # Voltage inactivation
    f2 = 12         # Voltage inactivation 2
    fCass = 13      # Ca-dependent inactivation

    # IKr gates
    Xr1 = 14        # Activation
    Xr2 = 15        # Inactivation (time-independent in tissue)

    # IKs gate
    Xs = 16         # Activation

    # RyR (Ryanodine receptor)
    RR = 17         # Release fraction

    # Total number of states (gates + concentrations, excludes V)
    N_STATES = 18


# State variable names in order (excludes V)
STATE_NAMES = (
    'Ki', 'Nai', 'Cai', 'CaSR', 'CaSS',
    'm', 'h', 'j',
    'r', 's',
    'd', 'f', 'f2', 'fCass',
    'Xr1', 'Xr2', 'Xs',
    'RR'
)


@dataclass
class TTP06Parameters:
    """
    ten Tusscher-Panfilov 2006 model parameters.

    All parameters use consistent units:
    - Voltage: mV
    - Current: pA/pF (normalized to membrane capacitance)
    - Concentration: mM
    - Time: ms
    - Conductance: nS/pF
    """

    # Physical constants
    R: float = 8314.472     # Gas constant (J/(mol·K))
    T: float = 310.0        # Temperature (K) - 37°C
    F: float = 96485.3415   # Faraday constant (C/mol)

    @property
    def RTONF(self) -> float:
        """R*T/F in mV."""
        return self.R * self.T / self.F

    # Cell geometry
    Cm: float = 0.185       # Cell capacitance (uF)
    Vc: float = 16.404      # Cell volume (pL)
    Vsr: float = 1.094      # SR volume (pL)
    Vss: float = 0.05468    # Subspace volume (pL)

    # Extracellular concentrations
    Ko: float = 5.4         # K+ (mM)
    Nao: float = 140.0      # Na+ (mM)
    Cao: float = 2.0        # Ca2+ (mM)

    # Maximum conductances (nS/pF)
    GNa: float = 14.838     # INa
    GK1: float = 5.405      # IK1
    Gto: float = 0.294      # Ito (ENDO default, EPI=0.294, M=0.294)
    GKr: float = 0.153      # IKr
    GKs: float = 0.392      # IKs (ENDO default)
    GpCa: float = 0.1238    # IpCa
    GpK: float = 0.0146     # IpK
    GbNa: float = 0.00029   # INab
    GbCa: float = 0.000592  # ICab

    # ICaL permeability
    PCa: float = 0.0000398  # cm/s (3.98e-5)

    # INaCa parameters
    KNaCa: float = 1000.0   # Maximum exchange rate (pA/pF)
    KmNai: float = 87.5     # Na affinity (mM)
    KmCa: float = 1.38      # Ca affinity (mM)
    ksat: float = 0.1       # Saturation factor
    alpha_ncx: float = 2.5  # Voltage dependence factor
    gamma_ncx: float = 0.35 # Position of energy barrier

    # INaK parameters
    PNaK: float = 2.724     # Maximum pump rate (pA/pF)
    KmK: float = 1.0        # K affinity (mM)
    KmNa: float = 40.0      # Na affinity (mM)

    # IpCa parameters
    KpCa: float = 0.0005    # Ca affinity (mM)

    # Calcium handling
    Kup: float = 0.00025    # SERCA Km (mM)
    Vmax_up: float = 0.006375  # SERCA Vmax (mM/ms)
    Vrel: float = 0.102     # RyR release rate (1/ms)
    Vleak: float = 0.00036  # SR leak rate (1/ms)
    Vxfer: float = 0.0038   # SS-cytosol transfer rate (1/ms)

    # Ca buffering
    Bufc: float = 0.2       # Cytoplasmic buffer (mM)
    Kbufc: float = 0.001    # Buffer affinity (mM)
    Bufsr: float = 10.0     # SR buffer (mM)
    Kbufsr: float = 0.3     # SR buffer affinity (mM)
    Bufss: float = 0.4      # SS buffer (mM)
    Kbufss: float = 0.00025 # SS buffer affinity (mM)

    # SR release parameters
    k1_prime: float = 0.15  # RyR on rate (/mM^2/ms)
    k2_prime: float = 0.045 # RyR off rate (/mM/ms)
    k3: float = 0.060       # RyR inactivation rate (/ms)
    k4: float = 0.005       # RyR recovery rate (/ms)
    EC: float = 1.5         # Ca50 for CICR (mM)
    maxsr: float = 2.5      # Max SR release factor
    minsr: float = 1.0      # Min SR release factor

    # Cell-type specific scaling (set by get_celltype_parameters)
    Gto_scale: float = 1.0
    GKs_scale: float = 1.0
    s_inf_scale: float = 1.0  # For epicardial Ito inactivation


def get_celltype_parameters(celltype: CellType) -> TTP06Parameters:
    """
    Get parameters scaled for specific cell type.

    Parameters
    ----------
    celltype : CellType
        ENDO, EPI, or M_CELL

    Returns
    -------
    TTP06Parameters
        Parameters with cell-type specific scaling
    """
    params = TTP06Parameters()

    if celltype == CellType.ENDO:
        # Endocardial - baseline
        params.Gto = 0.073  # Much smaller Ito
        params.GKs = 0.392

    elif celltype == CellType.EPI:
        # Epicardial - larger Ito, different kinetics
        params.Gto = 0.294  # 4x larger than ENDO
        params.GKs = 0.392
        params.s_inf_scale = 1.0  # EPI uses different s_inf formula

    elif celltype == CellType.M_CELL:
        # M-cell - intermediate Ito, smaller IKs
        params.Gto = 0.294
        params.GKs = 0.098  # 0.25x smaller IKs for longer APD

    return params


def get_initial_state(device: torch.device = None,
                      dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Get initial ionic state tensor for TTP06 model (excludes V).

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
        Initial ionic state tensor of shape (18,)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.zeros(StateIndex.N_STATES, dtype=dtype, device=device)

    # Initial values from CellML reference (epicardial steady state)
    # Reference: models.cellml.org/exposure/de5058f16f829f91a1e4e5990a10ed71
    state[StateIndex.Ki] = 136.89     # mM
    state[StateIndex.Nai] = 8.604     # mM
    state[StateIndex.Cai] = 0.000126  # mM (126 nM)
    state[StateIndex.CaSR] = 3.64     # mM
    state[StateIndex.CaSS] = 0.00036  # mM (360 nM)

    # INa gates (steady state at rest)
    state[StateIndex.m] = 0.00172
    state[StateIndex.h] = 0.7444
    state[StateIndex.j] = 0.7045

    # Ito gates
    state[StateIndex.r] = 2.42e-8
    state[StateIndex.s] = 0.999998

    # ICaL gates
    state[StateIndex.d] = 3.373e-5
    state[StateIndex.f] = 0.7888
    state[StateIndex.f2] = 0.9755
    state[StateIndex.fCass] = 0.9953

    # IKr gates
    state[StateIndex.Xr1] = 0.00621
    state[StateIndex.Xr2] = 0.4712

    # IKs gate
    state[StateIndex.Xs] = 0.00172

    # RyR
    state[StateIndex.RR] = 0.9073

    return state


# Resting membrane potential (mV) — separated from ionic states
V_REST = -85.23
