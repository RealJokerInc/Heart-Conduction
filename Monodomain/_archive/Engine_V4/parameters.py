"""
Luo-Rudy 1994 Parameter Management
==================================

Parameter definitions for the Luo-Rudy 1994 dynamic cardiac ionic model.
Values taken directly from CellML repository implementation.

Reference:
    Luo CH, Rudy Y. "A dynamic model of the cardiac ventricular action potential.
    I. Simulations of ionic currents and concentration changes."
    Circ Res. 1994;74(6):1071-1096.

CellML Source:
    https://models.cellml.org/e/81/luo_rudy_1994.cellml

Unit System:
    CellML uses mm² for area, we convert to cm² (multiply conductances by 100)
    - Time: ms
    - Voltage: mV
    - Current: µA/cm²
    - Conductance: mS/cm²
    - Concentration: mM
    - Permeability: mm/ms (for GHK)

Author: Generated with Claude Code
Date: 2025-12-11
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


# =============================================================================
# Physical Constants (from CellML)
# =============================================================================

@dataclass
class PhysicalConstants:
    """Universal physical constants for electrochemistry."""
    R: float = 8314.5       # Gas constant [mJ/(mol*K)] - CellML value
    T: float = 310.0        # Temperature [K] (37°C)
    F: float = 96845.0      # Faraday constant [C/mol] - CellML value

    @property
    def RTF(self) -> float:
        """R*T/F ~ 26.6 mV at 37°C."""
        return self.R * self.T / self.F

    @property
    def FRT(self) -> float:
        """F/(R*T) ~ 0.0376 /mV for exponential terms."""
        return self.F / (self.R * self.T)


# =============================================================================
# Cell Geometry Parameters (from CellML)
# =============================================================================

@dataclass
class CellGeometry:
    """Cardiac myocyte geometry parameters."""
    C_m: float = 1.0          # Membrane capacitance [µF/cm²] (= 0.01 µF/mm² × 100)
    A_m: float = 200.0        # Surface/volume ratio [1/cm] (Am in CellML)
    V_myo: float = 0.68       # Myoplasm volume fraction
    V_jsr: float = 0.0048     # Junctional SR volume fraction
    V_nsr: float = 0.0552     # Network SR volume fraction


# =============================================================================
# Luo-Rudy 1994 Model Parameters (Original CellML values)
# =============================================================================

@dataclass
class LRd94Params:
    """
    Luo-Rudy 1994 ionic model parameters.

    All values from CellML repository, converted to cm² basis where needed.
    Original CellML conductances (mS/mm²) multiplied by 100 for mS/cm².
    """
    # ===================
    # Extracellular Concentrations [mM]
    # ===================
    Na_o: float = 140.0      # Extracellular Na+
    K_o: float = 5.4         # Extracellular K+
    Ca_o: float = 1.8        # Extracellular Ca2+

    # ===================
    # Maximal Conductances [mS/cm²] (CellML × 100)
    # ===================
    G_Na: float = 16.0       # Fast Na+ (0.16 × 100)
    G_K: float = 0.282       # Time-dependent K+ (2.82e-3 × 100)
    G_K1: float = 0.75       # Inward rectifier K+ (7.5e-3 × 100)
    G_Kp: float = 0.0183     # Plateau K+ (1.83e-4 × 100)
    G_Na_b: float = 0.00141  # Background Na+ (1.41e-5 × 100)
    G_Ca_b: float = 0.003016 # Background Ca2+ (3.016e-5 × 100)

    # ===================
    # L-type Ca2+ Channel - GHK Permeabilities [mm/ms]
    # (NOT conductance - these go directly into GHK equation)
    # ===================
    P_Ca: float = 5.4e-6     # Ca2+ permeability
    P_Na: float = 6.75e-9    # Na+ permeability through L-type
    P_K: float = 1.93e-9     # K+ permeability through L-type

    # ===================
    # Activity Coefficients (dimensionless)
    # ===================
    gamma_Cai: float = 1.0   # Intracellular Ca2+
    gamma_Cao: float = 0.34  # Extracellular Ca2+
    gamma_Nai: float = 0.75  # Intracellular Na+
    gamma_Nao: float = 0.75  # Extracellular Na+
    gamma_Ki: float = 0.75   # Intracellular K+
    gamma_Ko: float = 0.75   # Extracellular K+

    # ===================
    # f_Ca gate parameter
    # ===================
    Km_Ca: float = 0.6e-3    # Ca2+ half-inactivation [mM] (= 0.6 µM)

    # ===================
    # Na+/K+ ATPase (Pump) [µA/cm²]
    # ===================
    I_NaK_bar: float = 1.5   # Maximum pump current (1.5e-2 × 100)
    K_m_Na_i: float = 10.0   # Na+ half-saturation [mM]
    K_m_K_o: float = 1.5     # K+ half-saturation [mM]

    # ===================
    # Sarcolemmal Ca2+ Pump [µA/cm²]
    # ===================
    I_pCa_bar: float = 1.15  # Maximum pump current (1.15e-2 × 100)
    K_m_pCa: float = 0.5e-3  # Ca2+ half-saturation [mM]

    # ===================
    # Na+/Ca2+ Exchanger
    # ===================
    k_NaCa: float = 2000.0   # Exchanger scaling (20.0 × 100 for area)
    K_m_Na: float = 87.5     # Na+ half-saturation [mM]
    K_m_Ca: float = 1.38     # Ca2+ half-saturation [mM]
    k_sat: float = 0.1       # Saturation factor
    eta: float = 0.35        # Voltage dependence position

    # ===================
    # Non-specific Ca-activated Current
    # ===================
    P_ns_Ca: float = 1.75e-9  # Permeability [mm/ms]
    K_m_ns_Ca: float = 1.2e-3 # Ca2+ activation [mM]

    # ===================
    # Sarcoplasmic Reticulum (SR)
    # ===================
    G_rel_max: float = 15.0   # Max release conductance [1/ms]
    tau_on: float = 2.0       # CICR activation time constant [ms]
    tau_off: float = 2.0      # CICR deactivation time constant [ms]
    tau_tr: float = 180.0     # JSR-NSR transfer time constant [ms]
    K_m_rel: float = 0.8e-3   # Release Ca2+ sensitivity [mM]
    K_m_up: float = 0.92e-3   # Uptake Ca2+ sensitivity [mM]
    I_up_bar: float = 0.005   # Maximum SR uptake rate [mM/ms]
    Ca_NSR_max: float = 15.0  # Maximum NSR Ca2+ [mM]

    # ===================
    # Intracellular Buffers
    # ===================
    TRPN_tot: float = 0.070    # Total troponin [mM]
    K_m_TRPN: float = 0.0005   # Troponin Kd [mM]
    CMDN_tot: float = 0.050    # Total calmodulin [mM]
    K_m_CMDN: float = 0.00238  # Calmodulin Kd [mM]
    CSQN_tot: float = 10.0     # Total calsequestrin [mM]
    K_m_CSQN: float = 0.8      # Calsequestrin Kd [mM]

    def validate(self) -> None:
        """Check parameters are in valid ranges."""
        assert self.G_Na > 0, "G_Na must be positive"
        assert self.P_Ca > 0, "P_Ca must be positive"
        assert self.G_K > 0, "G_K must be positive"
        assert self.G_K1 > 0, "G_K1 must be positive"
        assert self.Na_o > 0, "Na_o must be positive"
        assert self.K_o > 0, "K_o must be positive"
        assert self.Ca_o > 0, "Ca_o must be positive"

    def summary(self) -> str:
        """Human-readable parameter summary."""
        lines = [
            "Luo-Rudy 1994 Parameters (CellML Reference)",
            "=" * 50,
            "",
            "Extracellular Concentrations [mM]:",
            f"  Na_o     = {self.Na_o:8.2f}",
            f"  K_o      = {self.K_o:8.2f}",
            f"  Ca_o     = {self.Ca_o:8.2f}",
            "",
            "Conductances [mS/cm²]:",
            f"  G_Na     = {self.G_Na:8.4f}  (Fast Na+)",
            f"  G_K      = {self.G_K:8.4f}  (Time-dep K+)",
            f"  G_K1     = {self.G_K1:8.4f}  (Inward rectifier)",
            f"  G_Kp     = {self.G_Kp:8.4f}  (Plateau K+)",
            "",
            "L-type Ca2+ Permeabilities [mm/ms]:",
            f"  P_Ca     = {self.P_Ca:10.2e}",
            f"  P_Na     = {self.P_Na:10.2e}",
            f"  P_K      = {self.P_K:10.2e}",
            "",
            "Activity Coefficients:",
            f"  γ_Cai/Cao = {self.gamma_Cai}/{self.gamma_Cao}",
            f"  γ_Nai/Nao = {self.gamma_Nai}/{self.gamma_Nao}",
            f"  γ_Ki/Ko   = {self.gamma_Ki}/{self.gamma_Ko}",
            "",
            "SR Parameters:",
            f"  G_rel_max = {self.G_rel_max:8.2f} /ms",
            f"  I_up_bar  = {self.I_up_bar:8.5f} mM/ms",
            f"  tau_tr    = {self.tau_tr:8.2f} ms",
        ]
        return "\n".join(lines)


# =============================================================================
# Initial Conditions (from CellML)
# =============================================================================

@dataclass
class LRd94InitialConditions:
    """Initial state values for LRd94 model at rest (CellML values)."""
    V: float = -84.624       # Membrane potential [mV]
    m: float = 0.0           # Na+ activation
    h: float = 1.0           # Na+ fast inactivation
    j: float = 1.0           # Na+ slow inactivation
    d: float = 0.0           # L-type Ca2+ activation
    f: float = 1.0           # L-type Ca2+ inactivation
    Na_i: float = 10.0       # Intracellular Na+ [mM]
    K_i: float = 145.0       # Intracellular K+ [mM]
    Ca_i: float = 0.12e-3    # Intracellular Ca2+ [mM] (= 120 nM)
    Ca_jsr: float = 1.8      # Junctional SR Ca2+ [mM]
    Ca_nsr: float = 1.8      # Network SR Ca2+ [mM]
    # CICR trigger variables
    APtrack: float = 0.0     # AP tracking for CICR
    APtrack2: float = 0.0    # Secondary AP tracking
    APtrack3: float = 0.0    # Tertiary AP tracking

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'V': self.V,
            'm': self.m, 'h': self.h, 'j': self.j,
            'd': self.d, 'f': self.f,
            'Na_i': self.Na_i, 'K_i': self.K_i, 'Ca_i': self.Ca_i,
            'Ca_jsr': self.Ca_jsr, 'Ca_nsr': self.Ca_nsr,
        }

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.V,
            self.m, self.h, self.j,
            self.d, self.f,
            self.Na_i, self.K_i, self.Ca_i,
            self.Ca_jsr, self.Ca_nsr,
        ], dtype=np.float64)


# State variable indices
STATE_INDICES = {
    'V': 0,
    'm': 1, 'h': 2, 'j': 3,
    'd': 4, 'f': 5,
    'Na_i': 6, 'K_i': 7, 'Ca_i': 8,
    'Ca_jsr': 9, 'Ca_nsr': 10,
}

STATE_NAMES = list(STATE_INDICES.keys())
N_STATES = len(STATE_NAMES)


# =============================================================================
# Spatial Parameters
# =============================================================================

@dataclass
class SpatialParams:
    """Spatial domain and diffusion parameters."""
    Lx: float = 80.0         # Domain size x [mm]
    Ly: float = 80.0         # Domain size y [mm]
    dx: float = 0.5          # Grid spacing [mm]
    dy: float = 0.5          # Grid spacing [mm]
    D_parallel: float = 0.1  # Diffusion along fibers [mm²/ms]
    D_perp: float = 0.05     # Diffusion perpendicular [mm²/ms]
    fiber_angle: float = 0.0 # Fiber angle [degrees]

    @property
    def nx(self) -> int:
        return int(np.round(self.Lx / self.dx)) + 1

    @property
    def ny(self) -> int:
        return int(np.round(self.Ly / self.dy)) + 1

    def check_stability(self, dt: float) -> dict:
        """Check CFL stability condition."""
        D_max = max(self.D_parallel, self.D_perp)
        dx_min = min(self.dx, self.dy)
        r = D_max * dt / (dx_min ** 2)
        return {
            'r': r,
            'stable': r < 0.25,
            'dt_max': 0.25 * dx_min**2 / D_max,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def default_params() -> LRd94Params:
    """Get default LRd94 parameters (original CellML values, APD ~300ms)."""
    return LRd94Params()


def default_initial_conditions() -> LRd94InitialConditions:
    """Get default initial conditions."""
    return LRd94InitialConditions()


def default_spatial_params() -> SpatialParams:
    """Get default spatial parameters."""
    return SpatialParams()


def default_physical_constants() -> PhysicalConstants:
    """Get default physical constants."""
    return PhysicalConstants()


def default_cell_geometry() -> CellGeometry:
    """Get default cell geometry."""
    return CellGeometry()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("LRd94 Parameter System (CellML Reference)")
    print("=" * 60)

    params = default_params()
    params.validate()
    print(params.summary())

    print("\n" + "=" * 60)

    ic = default_initial_conditions()
    print("\nInitial Conditions:")
    for name, value in ic.to_dict().items():
        if name == 'V':
            print(f"  {name:8s} = {value:12.4f} mV")
        elif 'Ca' in name:
            print(f"  {name:8s} = {value*1e6:12.1f} nM" if value < 0.01 else f"  {name:8s} = {value:12.4f} mM")
        else:
            print(f"  {name:8s} = {value:12.4f}")

    phys = default_physical_constants()
    print(f"\nPhysical Constants:")
    print(f"  R = {phys.R} mJ/(mol·K)")
    print(f"  T = {phys.T} K")
    print(f"  F = {phys.F} C/mol")
    print(f"  RT/F = {phys.RTF:.2f} mV")
    print(f"  F/RT = {phys.FRT:.4f} /mV")

    print("\n" + "=" * 60)
    print("Parameter system loaded successfully!")
