"""
LRd07 Model Parameters

All parameter values are taken directly from constantsLRd.m
from the Livshitz & Rudy 2007 MATLAB implementation.

Reference: Livshitz LM, Rudy Y. Am J Physiol Heart Circ Physiol. 2007;292(6):H2854-66.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LRd07Parameters:
    """
    Complete parameter set for LRd07 cardiac cell model.

    All values match constantsLRd.m exactly.
    """

    # =========================================================================
    # Physical Constants
    # =========================================================================
    F: float = 96485.0          # Faraday constant (C/mol)
    R: float = 8314.0           # Gas constant (mJ/(mol·K))
    T: float = 310.0            # Temperature (K)

    # Derived constant
    @property
    def frt(self) -> float:
        """F / (R * T) - used in Nernst and GHK equations"""
        return self.F / (self.R * self.T)

    # =========================================================================
    # Cell Geometry
    # =========================================================================
    l: float = 0.01             # Cell length (cm)
    a: float = 0.0011           # Cell radius (cm)

    @property
    def vcell(self) -> float:
        """Cell volume (uL)"""
        return 1000 * np.pi * self.a * self.a * self.l

    @property
    def ageo(self) -> float:
        """Geometric membrane area (cm^2)"""
        return 2 * np.pi * self.a * self.a + 2 * np.pi * self.a * self.l

    @property
    def Acap(self) -> float:
        """Capacitive membrane area (cm^2)"""
        return self.ageo * 2

    @property
    def vmyo(self) -> float:
        """Myoplasm volume (uL) - 68% of cell"""
        return self.vcell * 0.68

    @property
    def vsr(self) -> float:
        """Total SR volume (uL) - 6% of cell"""
        return self.vcell * 0.06

    @property
    def vnsr(self) -> float:
        """Network SR volume (uL) - 5.52% of cell"""
        return self.vcell * 0.0552

    @property
    def vjsr(self) -> float:
        """Junctional SR volume (uL) - 0.48% of cell"""
        return self.vcell * 0.0048

    @property
    def vss(self) -> float:
        """Subspace volume (uL) - 2% of cell"""
        return self.vcell * 0.02

    @property
    def AF(self) -> float:
        """Acap / F - conversion factor for concentration changes"""
        return self.Acap / self.F

    # =========================================================================
    # Extracellular Concentrations (mM)
    # =========================================================================
    Na_o: float = 140.0         # Extracellular Na+ (mM)
    K_o: float = 4.5            # Extracellular K+ (mM)
    Ca_o: float = 1.8           # Extracellular Ca2+ (mM)

    # =========================================================================
    # Stimulus Parameters
    # =========================================================================
    I_stim: float = 80.0        # Stimulus current (uA/uF)
    stim_duration: float = 0.5  # Stimulus duration (ms)
    stim_start: float = 0.0     # Stimulus start time (ms)

    # =========================================================================
    # INa - Fast Sodium Current
    # =========================================================================
    G_Na: float = 16.0          # Max conductance (mS/cm^2)
    G_Nab: float = 0.004        # Background Na conductance (mS/cm^2)

    # =========================================================================
    # ICaL - L-type Calcium Current (GHK formulation)
    # =========================================================================
    P_Ca: float = 5.4e-4        # Ca permeability (cm/s)
    P_Na: float = 6.75e-7       # Na permeability through L-type (cm/s)
    P_K: float = 1.93e-7        # K permeability through L-type (cm/s)

    # Activity coefficients
    gamma_Cai: float = 1.0      # Intracellular Ca activity coefficient
    gamma_Cao: float = 0.341    # Extracellular Ca activity coefficient
    gamma_Nai: float = 0.75     # Intracellular Na activity coefficient
    gamma_Nao: float = 0.75     # Extracellular Na activity coefficient
    gamma_Ki: float = 0.75      # Intracellular K activity coefficient
    gamma_Ko: float = 0.75      # Extracellular K activity coefficient

    # fCa gating
    Km_Ca: float = 6e-4         # Half-saturation for Ca-dependent inactivation (mM)

    # =========================================================================
    # ICaT - T-type Calcium Current
    # =========================================================================
    g_CaT: float = 0.05         # Max conductance (mS/cm^2)

    # =========================================================================
    # ICab - Background Calcium Current
    # =========================================================================
    g_Cab: float = 0.003016     # Conductance (mS/cm^2)

    # =========================================================================
    # IKr - Rapid Delayed Rectifier K+ Current
    # =========================================================================
    g_Kr_max: float = 0.02614   # Max conductance (mS/cm^2)

    # =========================================================================
    # IKs - Slow Delayed Rectifier K+ Current (Ca-dependent)
    # =========================================================================
    G_Ks_max: float = 0.433     # Max conductance (mS/cm^2)
    prnak: float = 0.01833      # Na/K permeability ratio for IKs reversal

    # Ca-dependence parameters
    IKs_Ca_max: float = 0.6     # Max enhancement factor
    IKs_Ca_Kd: float = 38e-6    # Half-saturation for Ca enhancement (mM)

    # =========================================================================
    # IK1 - Inward Rectifier K+ Current
    # =========================================================================
    G_K1_max: float = 0.75      # Max conductance (mS/cm^2)

    # =========================================================================
    # IKp - Plateau K+ Current
    # =========================================================================
    G_Kp_max: float = 0.00552   # Max conductance (mS/cm^2)

    # =========================================================================
    # INaCa - Na/Ca Exchanger
    # =========================================================================
    c1: float = 0.00025         # Scaling factor (uA/uF)
    c2: float = 0.0001          # Saturation factor
    gamma_NCX: float = 0.15     # Voltage dependence position

    # =========================================================================
    # INaK - Na/K Pump
    # =========================================================================
    I_NaK_max: float = 2.25     # Max pump current (uA/uF)
    Km_Nai: float = 10.0        # Half-saturation for Na (mM)
    Km_Ko: float = 1.5          # Half-saturation for K (mM)

    # =========================================================================
    # IpCa - Sarcolemmal Ca Pump
    # =========================================================================
    I_pCa_max: float = 1.15     # Max pump current (uA/uF)
    Km_pCa: float = 0.5e-3      # Half-saturation (mM)

    # =========================================================================
    # SR Calcium Handling
    # =========================================================================
    # SERCA (SR Ca-ATPase) uptake
    I_up_max: float = 0.00875   # Max uptake rate (mM/ms)
    Km_up: float = 0.00092      # Half-saturation for uptake (mM)
    NSR_max: float = 15.0       # Max NSR concentration (mM)

    # NSR to JSR transfer
    tau_tr: float = 120.0       # Transfer time constant (ms)

    # SR Release (Rel) - Key LRd07 feature
    tau_Rel: float = 4.75       # Base release time constant (ms)
    kappa: float = 0.125        # Release scaling factor
    K_Rel_ss: float = 1.0       # Half-saturation for JSR dependence (mM)
    qn: float = 9.0             # Hill coefficient for JSR dependence

    @property
    def alpha_Rel(self) -> float:
        """Release rate constant = tau * kappa"""
        return self.tau_Rel * self.kappa

    # =========================================================================
    # Calcium Buffering
    # =========================================================================
    # Cytosolic buffers
    CMDN_max: float = 0.050     # Total calmodulin concentration (mM)
    TRPN_max: float = 0.070     # Total troponin concentration (mM)
    Km_CMDN: float = 0.00238    # Calmodulin Ca affinity (mM)
    Km_TRPN: float = 0.0005     # Troponin Ca affinity (mM)

    # JSR buffer
    CSQN_max: float = 10.0      # Total calsequestrin concentration (mM)
    Km_CSQN: float = 0.8        # Calsequestrin Ca affinity (mM)

    # =========================================================================
    # Initial Conditions (from MATLAB steady-state at BCL=400ms)
    # =========================================================================
    def get_initial_state(self) -> np.ndarray:
        """
        Return initial state vector y(1:18).

        Values are steady-state from 100-beat simulation at BCL=400ms.

        State variable ordering (MATLAB indices):
        y(1)  = V        Membrane potential (mV)
        y(2)  = H        INa fast inactivation
        y(3)  = m        INa activation
        y(4)  = J        INa slow inactivation
        y(5)  = d        ICaL activation
        y(6)  = f        ICaL voltage inactivation
        y(7)  = xr       IKr activation
        y(8)  = ca_T     Total cytosolic Ca (buffered) (mM)
        y(9)  = Na_i     Intracellular Na (mM)
        y(10) = K_i      Intracellular K (mM)
        y(11) = jsr_T    Total JSR Ca (buffered) (mM)
        y(12) = nsr      NSR Ca (mM)
        y(13) = xs       IKs activation (fast)
        y(14) = B        ICaT activation
        y(15) = G        ICaT inactivation
        y(16) = xs2      IKs activation (slow)
        y(17) = Rel      SR release flux (mM/ms)
        y(18) = Over     Overload variable (unused)
        """
        # Steady-state initial conditions for BCL=400ms
        # From MATLAB Initial_conditions2007.mat
        return np.array([
            -89.450638,     # V (mV)
            0.9944204,      # H
            0.0007330838,   # m
            0.99611274,     # J
            0.0,            # d (essentially 0 at rest)
            0.99753546,     # f
            0.00027845222,  # xr
            0.025696491,    # ca_T (mM) -> ~218 nM free
            16.684466,      # Na_i (mM)
            139.65796,      # K_i (mM)
            7.9266129,      # jsr_T (mM) -> ~1.46 mM free
            2.7182251,      # nsr (mM)
            0.027929689,    # xs
            0.00092611766,  # B
            0.95421487,     # G
            0.075776491,    # xs2
            0.0,            # Rel
            0.0,            # Over
        ], dtype=np.float64)

    def get_state_names(self) -> list:
        """Return list of state variable names in order."""
        return [
            'V', 'H', 'm', 'J', 'd', 'f', 'xr',
            'ca_T', 'Na_i', 'K_i', 'jsr_T', 'nsr',
            'xs', 'B', 'G', 'xs2', 'Rel', 'Over'
        ]

    def get_state_units(self) -> list:
        """Return list of state variable units in order."""
        return [
            'mV', '-', '-', '-', '-', '-', '-',
            'mM', 'mM', 'mM', 'mM', 'mM',
            '-', '-', '-', '-', 'mM/ms', '-'
        ]


# Create a default parameter instance for convenience
DEFAULT_PARAMS = LRd07Parameters()


# State variable indices (0-based Python indexing)
class StateIndex:
    """State variable indices for easy access."""
    V = 0
    H = 1
    m = 2
    J = 3
    d = 4
    f = 5
    xr = 6
    ca_T = 7
    Na_i = 8
    K_i = 9
    jsr_T = 10
    nsr = 11
    xs = 12
    B = 13
    G = 14
    xs2 = 15
    Rel = 16
    Over = 17

    N_STATES = 18
