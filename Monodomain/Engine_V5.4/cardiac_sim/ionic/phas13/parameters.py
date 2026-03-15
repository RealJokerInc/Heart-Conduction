"""
PHAS13 hiPSC-CM Model Parameters

Contains:
- StateIndex: Indices into state tensor (17 states, V separate)
- PHAS13Parameters: All model constants
- get_initial_state(): Published initial conditions

Reference:
Paci M, Hyttinen J, Aalto-Setala K, Severi S (2013).
"Computational Models of Ventricular- and Atrial-Like Human Induced
Pluripotent Stem Cell Derived Cardiomyocytes."
Ann Biomed Eng 41(11):2334-2348.
"""

from enum import IntEnum
from dataclasses import dataclass
import torch


class StateIndex(IntEnum):
    """
    Indices into the ionic state tensor (17 state variables).
    V is stored separately from ionic states.
    """
    # Concentrations (Forward Euler)
    Nai    = 0      # Intracellular Na+ (mM)
    Cai    = 1      # Intracellular Ca2+ cytoplasm (mM)
    CaSR   = 2      # SR Ca2+ (mM)

    # INa gates (Rush-Larsen)
    m      = 3      # Activation
    h      = 4      # Fast inactivation
    j      = 5      # Slow inactivation

    # ICaL gates (d,f1,f2: Rush-Larsen; fCa: conditional FE)
    d      = 6      # Activation
    f1     = 7      # Voltage inactivation 1
    f2     = 8      # Voltage inactivation 2
    fCa    = 9      # Ca-dependent inactivation

    # IKr gates (Rush-Larsen)
    Xr1    = 10     # Activation
    Xr2    = 11     # Inactivation

    # IKs gate (Rush-Larsen)
    Xs     = 12     # Activation

    # Ito gates (Rush-Larsen)
    q      = 13     # Inactivation
    r_gate = 14     # Activation (named r_gate to avoid clash)

    # If gate (Rush-Larsen)
    Xf     = 15     # Activation

    # Ca release (conditional Forward Euler)
    g_rel  = 16     # RyR inactivation gate

    N_STATES = 17


# State variable names in order (excludes V)
STATE_NAMES = (
    'Nai', 'Cai', 'CaSR',
    'm', 'h', 'j',
    'd', 'f1', 'f2', 'fCa',
    'Xr1', 'Xr2', 'Xs',
    'q', 'r_gate', 'Xf',
    'g_rel',
)


@dataclass
class PHAS13Parameters:
    """
    PHAS13 hiPSC-CM model parameters.

    Units convention:
    - Voltage: mV (converted from .mmt Volts)
    - Time: ms (converted from .mmt seconds)
    - Conductance: ohmic conductances divided by 1000 for mV driving forces
    - Current: A/F (= pA/pF = uA/uF)
    - Concentration: mM
    """

    # Physical constants (same as TTP06 convention: R in mJ/(mol*K) for mV arithmetic)
    R: float = 8314.472         # Gas constant (mJ/(mol*K))
    T: float = 310.0            # Temperature (K)
    F: float = 96485.3415       # Faraday constant (C/mol)

    @property
    def RTONF(self) -> float:
        """R*T/F in mV."""
        return self.R * self.T / self.F

    # Cell geometry
    Cm: float = 9.87109e-11     # Cell capacitance (F)
    Vc: float = 8800.0e-18      # Cell volume (m^3, from 8800 um^3)
    V_SR: float = 583.73e-18    # SR volume (m^3, from 583.73 um^3)

    # Fixed intracellular K+
    Ki: float = 150.0           # mM (not a state variable)

    # Extracellular concentrations (mM)
    Ko: float = 5.4
    Nao: float = 151.0
    Cao: float = 1.8

    # --- Ohmic conductances (S/F in .mmt, divided by 1000 for mV convention) ---
    g_Na: float = 3.6712302         # INa (3671.2302 S/F / 1000)
    g_K1: float = 0.0281492         # IK1 (28.1492 / 1000)
    g_Kr: float = 0.0298667         # IKr (29.8667 / 1000)
    g_Ks: float = 0.002041          # IKs (2.041 / 1000)
    g_to: float = 0.0299038         # Ito (29.9038 / 1000)
    g_f: float = 0.03010312         # If  (30.10312 / 1000)
    g_bCa: float = 0.00069264       # IbCa (0.69264 / 1000)
    g_bNa: float = 0.0009           # IbNa (0.9 / 1000)

    # --- Non-ohmic current parameters (keep as-is, A/F) ---
    # ICaL (GHK formulation)
    g_CaL: float = 8.635702e-5     # L/F/ms

    # INaCa
    kNaCa: float = 4900.0          # A/F
    KmCa: float = 1.38             # mM
    KmNai: float = 87.5            # mM
    Ksat: float = 0.1
    alpha_ncx: float = 2.8571432
    gamma_ncx: float = 0.35

    # INaK
    PNaK: float = 1.841424         # A/F
    Km_K: float = 1.0              # mM
    Km_Na: float = 40.0            # mM

    # IpCa
    g_pCa: float = 0.4125          # A/F
    KpCa: float = 0.0005           # mM

    # If reversal
    E_f: float = -17.0             # mV (from -0.017 V)

    # IKs Na permeability
    PkNa: float = 0.03

    # --- Calcium handling (rates converted from /s to /ms) ---
    VmaxUp: float = 5.6064e-4      # SERCA Vmax (mM/ms, from 0.56064 mM/s)
    Kup: float = 0.00025           # SERCA Km (mM)
    a_rel: float = 0.016464        # RyR release (mM/ms, from 16.464 mM/s)
    b_rel: float = 0.25            # RyR half-sat (mM)
    c_rel: float = 0.008232        # RyR release basal (mM/ms, from 8.232 mM/s)
    V_leak: float = 4.4444e-7      # SR leak rate (1/ms, from 4.4444e-4 1/s)
    tau_g: float = 2.0             # g_rel time constant (ms, from 0.002 s)

    # Ca buffering
    Buf_C: float = 0.25            # Cytoplasmic buffer total (mM)
    Kbuf_C: float = 0.001          # Cytoplasmic buffer Kd (mM)
    Buf_SR: float = 10.0           # SR buffer total (mM)
    Kbuf_SR: float = 0.3           # SR buffer Kd (mM)


def get_initial_state(device: torch.device = None,
                      dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Get initial ionic state tensor for PHAS13 model (excludes V).

    Values from published CellML/Myokit model (paci-2013-ventricular.mmt).

    Returns
    -------
    torch.Tensor
        Initial ionic state tensor of shape (17,)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.zeros(StateIndex.N_STATES, dtype=dtype, device=device)

    # Concentrations
    state[StateIndex.Nai]    = 10.9248496211574      # mM
    state[StateIndex.Cai]    = 1.80773974140477e-5   # mM
    state[StateIndex.CaSR]   = 0.273423475193100     # mM

    # INa gates
    state[StateIndex.m]      = 0.102953468725004
    state[StateIndex.h]      = 0.786926637881461
    state[StateIndex.j]      = 0.253943221774722

    # ICaL gates
    state[StateIndex.d]      = 8.96088425225182e-5
    state[StateIndex.f1]     = 0.970411811263976
    state[StateIndex.f2]     = 0.999965815466749
    state[StateIndex.fCa]    = 0.998925296531804

    # IKr gates
    state[StateIndex.Xr1]    = 7.78547011240132e-3
    state[StateIndex.Xr2]    = 0.432162576531617

    # IKs gate
    state[StateIndex.Xs]     = 0.0322944866983666

    # Ito gates
    state[StateIndex.q]      = 0.839295925773219
    state[StateIndex.r_gate] = 5.73289893326379e-3

    # If gate
    state[StateIndex.Xf]     = 0.100615100568753

    # RyR inactivation
    state[StateIndex.g_rel]  = 0.999999981028517

    return state


# Resting membrane potential (mV) — from .mmt initial V = -0.074334 V
V_REST = -74.334005762384
