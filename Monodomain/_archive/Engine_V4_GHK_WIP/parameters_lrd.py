"""
Luo-Rudy 1994 (LRd) Parameter Definitions
==========================================

All parameters for the Luo-Rudy 1994 dynamic cardiac ventricular action potential model.

Units:
- Voltage: mV
- Time: ms
- Concentration: mM
- Conductance: mS/cm²
- Current density: µA/cm²
- Capacitance: µF/cm²
- Permeability: cm/s

Reference:
    Luo CH, Rudy Y. Circ Res. 1994;74(6):1071-1096.
    CellML: models.cellml.org/e/81

Author: Generated with Claude Code
Date: 2024-12
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


# =============================================================================
# Physical Constants
# =============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """Universal physical constants."""
    R: float = 8314.0       # Gas constant [mJ/(mol·K)] = [J/(kmol·K)]
    T: float = 310.0        # Temperature [K] (37°C)
    F: float = 96485.0      # Faraday constant [C/mol]

    @property
    def RTF(self) -> float:
        """R*T/F ~ 26.7 mV at 37°C."""
        return self.R * self.T / self.F

    @property
    def FRT(self) -> float:
        """F/(R*T) ~ 0.0374 1/mV at 37°C."""
        return self.F / (self.R * self.T)


# =============================================================================
# Cell Geometry
# =============================================================================

@dataclass(frozen=True)
class CellGeometry:
    """Cell geometry parameters for concentration calculations."""
    Cm: float = 1.0          # Membrane capacitance [µF/cm²]
    A_cap: float = 1.534e-4  # Capacitive membrane area [cm²]
    V_myo: float = 25.84e-6  # Myoplasm volume [µL] = [cm³ * 1e-3]
    V_jsr: float = 0.16e-6   # Junctional SR volume [µL]
    V_nsr: float = 2.1e-6    # Network SR volume [µL]

    @property
    def V_jsr_V_myo(self) -> float:
        """Volume ratio V_jsr / V_myo."""
        return self.V_jsr / self.V_myo

    @property
    def V_nsr_V_myo(self) -> float:
        """Volume ratio V_nsr / V_myo."""
        return self.V_nsr / self.V_myo


# =============================================================================
# Extracellular Concentrations (Fixed)
# =============================================================================

@dataclass(frozen=True)
class ExtracellularConcentrations:
    """Fixed extracellular ion concentrations [mM]."""
    Na_o: float = 140.0     # Extracellular Na+ [mM]
    K_o: float = 5.4        # Extracellular K+ [mM]
    Ca_o: float = 1.8       # Extracellular Ca2+ [mM]


# =============================================================================
# Fast Sodium Current (I_Na) Parameters
# =============================================================================

@dataclass(frozen=True)
class INaParams:
    """Fast sodium current parameters."""
    g_Na: float = 16.0      # Max conductance [mS/cm²]
    # Note: Original LR91 used 23 mS/cm², reduced in LRd94


# =============================================================================
# L-Type Calcium Current (I_CaL) Parameters - GHK Formulation
# =============================================================================

@dataclass(frozen=True)
class ICaLParams:
    """L-type calcium current parameters with GHK permeabilities."""
    # Permeabilities [cm/s]
    P_Ca: float = 5.4e-4    # Ca2+ permeability
    P_Na: float = 6.75e-7   # Na+ permeability through L-type channel
    P_K: float = 1.93e-7    # K+ permeability through L-type channel

    # Activity coefficients (dimensionless)
    gamma_Cai: float = 1.0      # Intracellular Ca2+ activity coefficient
    gamma_Cao: float = 0.341    # Extracellular Ca2+ activity coefficient
    gamma_Nai: float = 0.75     # Intracellular Na+ activity coefficient
    gamma_Nao: float = 0.75     # Extracellular Na+ activity coefficient
    gamma_Ki: float = 0.75      # Intracellular K+ activity coefficient
    gamma_Ko: float = 0.75      # Extracellular K+ activity coefficient

    # Calcium-dependent inactivation
    Km_fCa: float = 0.6e-3  # Half-saturation for f_Ca gate [mM]


# =============================================================================
# Potassium Current Parameters
# =============================================================================

@dataclass(frozen=True)
class IKParams:
    """Time-dependent potassium current (IK) parameters."""
    g_K_max: float = 0.282  # Max conductance [mS/cm²]
    PR_NaK: float = 0.01833 # Na/K permeability ratio for EK calculation


@dataclass(frozen=True)
class IK1Params:
    """Inward rectifier potassium current (IK1) parameters."""
    g_K1_max: float = 0.6047  # Max conductance [mS/cm²]


@dataclass(frozen=True)
class IKpParams:
    """Plateau potassium current (IKp) parameters."""
    g_Kp: float = 0.0183    # Conductance [mS/cm²]


# =============================================================================
# Na+/Ca2+ Exchanger Parameters
# =============================================================================

@dataclass(frozen=True)
class INaCaParams:
    """Na+/Ca2+ exchanger parameters."""
    k_NaCa: float = 2000.0  # Scaling factor [µA/cm²]
    Km_Na: float = 87.5     # Na+ half-saturation [mM]
    Km_Ca: float = 1.38     # Ca2+ half-saturation [mM]
    k_sat: float = 0.1      # Saturation factor at negative potentials
    eta: float = 0.35       # Position of energy barrier (0-1)


# =============================================================================
# Na+/K+ Pump Parameters
# =============================================================================

@dataclass(frozen=True)
class INaKParams:
    """Na+/K+ ATPase pump parameters."""
    I_NaK_max: float = 1.5  # Max pump current [µA/cm²]
    Km_Nai: float = 10.0    # Intracellular Na+ half-saturation [mM]
    Km_Ko: float = 1.5      # Extracellular K+ half-saturation [mM]


# =============================================================================
# Sarcolemmal Ca2+ Pump Parameters
# =============================================================================

@dataclass(frozen=True)
class IpCaParams:
    """Sarcolemmal Ca2+ pump parameters."""
    I_pCa_max: float = 1.15  # Max pump current [µA/cm²]
    Km_pCa: float = 0.5e-3   # Ca2+ half-saturation [mM]


# =============================================================================
# Background Current Parameters
# =============================================================================

@dataclass(frozen=True)
class BackgroundParams:
    """Background leak current parameters."""
    g_Na_b: float = 0.001   # Na+ background conductance [mS/cm²]
    g_Ca_b: float = 0.003   # Ca2+ background conductance [mS/cm²]


# =============================================================================
# Sarcoplasmic Reticulum Parameters
# =============================================================================

@dataclass(frozen=True)
class SRParams:
    """Sarcoplasmic reticulum Ca2+ handling parameters."""
    # Release from JSR (CICR)
    G_rel_max: float = 60.0     # Max release conductance [1/ms]
    tau_on: float = 2.0         # Activation time constant [ms]
    tau_off: float = 2.0        # Deactivation time constant [ms]
    Km_rel: float = 0.8e-3      # Ca2+ sensitivity for release [mM]
    delta_Ca_th: float = 0.18e-3  # Threshold for CICR [mM]

    # Uptake into NSR (SERCA)
    I_up_max: float = 0.005     # Max uptake rate [mM/ms]
    Km_up: float = 0.92e-3      # Ca2+ half-saturation for uptake [mM]

    # Transfer NSR -> JSR
    tau_tr: float = 180.0       # Transfer time constant [ms]

    # Leak from NSR
    K_leak: float = 0.0         # Leak rate (often set to balance uptake)


# =============================================================================
# Calcium Buffering Parameters
# =============================================================================

@dataclass(frozen=True)
class BufferParams:
    """Calcium buffering parameters."""
    # Cytoplasmic buffers
    TRPN_tot: float = 0.070     # Total troponin [mM]
    Km_TRPN: float = 0.5e-3     # Troponin Ca2+ affinity [mM]
    CMDN_tot: float = 0.050     # Total calmodulin [mM]
    Km_CMDN: float = 2.38e-3    # Calmodulin Ca2+ affinity [mM]

    # JSR buffer
    CSQN_tot: float = 10.0      # Total calsequestrin [mM]
    Km_CSQN: float = 0.8        # Calsequestrin Ca2+ affinity [mM]


# =============================================================================
# Initial Conditions
# =============================================================================

@dataclass
class LRdInitialConditions:
    """
    Initial conditions for all 13 state variables.

    Values from CellML model at steady-state rest.
    """
    # Membrane potential
    V: float = -84.624      # [mV]

    # INa gating variables
    m: float = 0.00136      # Activation gate
    h: float = 0.9814       # Fast inactivation gate
    j: float = 0.9905       # Slow inactivation gate

    # ICaL gating variables
    d: float = 3.0e-6       # Activation gate
    f: float = 1.0          # Voltage inactivation gate
    f_Ca: float = 1.0       # Calcium inactivation gate

    # IK gating variable
    X: float = 0.0057       # Activation gate

    # Intracellular concentrations
    Na_i: float = 10.0      # [mM]
    K_i: float = 145.0      # [mM]
    Ca_i: float = 0.12e-3   # [mM] = 0.12 µM

    # SR calcium
    Ca_jsr: float = 1.8     # JSR Ca2+ [mM]
    Ca_nsr: float = 1.8     # NSR Ca2+ [mM]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'V': self.V, 'm': self.m, 'h': self.h, 'j': self.j,
            'd': self.d, 'f': self.f, 'f_Ca': self.f_Ca, 'X': self.X,
            'Na_i': self.Na_i, 'K_i': self.K_i, 'Ca_i': self.Ca_i,
            'Ca_jsr': self.Ca_jsr, 'Ca_nsr': self.Ca_nsr
        }

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in standard order."""
        return np.array([
            self.V, self.m, self.h, self.j,
            self.d, self.f, self.f_Ca, self.X,
            self.Na_i, self.K_i, self.Ca_i,
            self.Ca_jsr, self.Ca_nsr
        ], dtype=np.float64)


# =============================================================================
# Complete Parameter Set
# =============================================================================

@dataclass
class LRdParams:
    """
    Complete Luo-Rudy 1994 parameter set.

    Aggregates all parameter groups for convenient access.
    """
    physical: PhysicalConstants = field(default_factory=PhysicalConstants)
    geometry: CellGeometry = field(default_factory=CellGeometry)
    ext_conc: ExtracellularConcentrations = field(default_factory=ExtracellularConcentrations)
    i_na: INaParams = field(default_factory=INaParams)
    i_cal: ICaLParams = field(default_factory=ICaLParams)
    i_k: IKParams = field(default_factory=IKParams)
    i_k1: IK1Params = field(default_factory=IK1Params)
    i_kp: IKpParams = field(default_factory=IKpParams)
    i_naca: INaCaParams = field(default_factory=INaCaParams)
    i_nak: INaKParams = field(default_factory=INaKParams)
    i_pca: IpCaParams = field(default_factory=IpCaParams)
    background: BackgroundParams = field(default_factory=BackgroundParams)
    sr: SRParams = field(default_factory=SRParams)
    buffers: BufferParams = field(default_factory=BufferParams)

    def summary(self) -> str:
        """Generate parameter summary."""
        lines = [
            "=" * 60,
            "Luo-Rudy 1994 (LRd) Parameters",
            "=" * 60,
            "",
            "Physical Constants:",
            f"  R = {self.physical.R} mJ/(mol·K)",
            f"  T = {self.physical.T} K ({self.physical.T - 273.15}°C)",
            f"  F = {self.physical.F} C/mol",
            f"  RTF = {self.physical.RTF:.3f} mV",
            "",
            "Cell Geometry:",
            f"  Cm = {self.geometry.Cm} µF/cm²",
            f"  V_myo = {self.geometry.V_myo*1e6:.2f} µL",
            f"  V_jsr = {self.geometry.V_jsr*1e6:.2f} µL",
            f"  V_nsr = {self.geometry.V_nsr*1e6:.2f} µL",
            "",
            "Extracellular Concentrations:",
            f"  Na_o = {self.ext_conc.Na_o} mM",
            f"  K_o = {self.ext_conc.K_o} mM",
            f"  Ca_o = {self.ext_conc.Ca_o} mM",
            "",
            "Conductances & Permeabilities:",
            f"  g_Na = {self.i_na.g_Na} mS/cm²",
            f"  P_Ca = {self.i_cal.P_Ca} cm/s",
            f"  g_K = {self.i_k.g_K_max} mS/cm²",
            f"  g_K1 = {self.i_k1.g_K1_max} mS/cm²",
            f"  g_Kp = {self.i_kp.g_Kp} mS/cm²",
            "",
            "Pumps & Exchangers:",
            f"  I_NaK_max = {self.i_nak.I_NaK_max} µA/cm²",
            f"  k_NaCa = {self.i_naca.k_NaCa} µA/cm²",
            f"  I_pCa_max = {self.i_pca.I_pCa_max} µA/cm²",
            "",
            "SR Parameters:",
            f"  G_rel_max = {self.sr.G_rel_max} 1/ms",
            f"  I_up_max = {self.sr.I_up_max} mM/ms",
            f"  tau_tr = {self.sr.tau_tr} ms",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Default Factory Functions
# =============================================================================

def default_params() -> LRdParams:
    """Get default LRd94 parameters."""
    return LRdParams()


def default_initial_conditions() -> LRdInitialConditions:
    """Get default initial conditions at rest."""
    return LRdInitialConditions()


# =============================================================================
# State Variable Index Mapping
# =============================================================================

# Standard ordering for state variables (used in Numba kernels)
STATE_INDICES = {
    'V': 0,
    'm': 1,
    'h': 2,
    'j': 3,
    'd': 4,
    'f': 5,
    'f_Ca': 6,
    'X': 7,
    'Na_i': 8,
    'K_i': 9,
    'Ca_i': 10,
    'Ca_jsr': 11,
    'Ca_nsr': 12,
}

N_STATES = 13


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing LRd94 Parameters")
    print()

    # Create default parameters
    params = default_params()
    print(params.summary())

    # Create initial conditions
    ic = default_initial_conditions()
    print("\nInitial Conditions:")
    for name, value in ic.to_dict().items():
        print(f"  {name} = {value}")

    # Test array conversion
    state_array = ic.to_array()
    print(f"\nState array shape: {state_array.shape}")
    print(f"State array: {state_array}")

    # Verify RTF
    print(f"\nRTF = {params.physical.RTF:.4f} mV (expected ~26.7 mV)")

    # Verify volume ratios
    print(f"V_jsr/V_myo = {params.geometry.V_jsr_V_myo:.6f}")
    print(f"V_nsr/V_myo = {params.geometry.V_nsr_V_myo:.6f}")
