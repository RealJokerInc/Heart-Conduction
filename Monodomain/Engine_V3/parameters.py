"""
Fenton-Karma Parameter Management
=================================

Parameter definitions for the Fenton-Karma 3-variable cardiac ionic model.

All parameters are in PHYSICAL UNITS:
- Time: ms (milliseconds)
- Space: mm (millimeters)
- Voltage: dimensionless [0, 1] internally, mV externally
- Diffusion: mm²/ms

Internal scaling by dt is handled in the Simulation class.

Parameter Sets:
- PARAMSET_3: Stable spirals (recommended starting point)
- PARAMSET_6: Similar to MATLAB fk2d.m
- See docs/FK_IMPLEMENTATION_RESEARCH.md for all 10 sets

References:
- Fenton & Karma 1998, Chaos 8:20-47
- Fenton et al. 2002, Chaos 12:852-892
- cardiax (github.com/epignatelli/cardiax)

Author: Generated with Claude Code
Date: 2025-12-10
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


# =============================================================================
# Physical Constants
# =============================================================================

@dataclass
class PhysicalConstants:
    """
    Physical constants for voltage conversion.

    The FK model uses dimensionless voltage u ∈ [0, 1].
    Physical voltage: V_mV = V_rest + u * (V_peak - V_rest)
    """
    V_rest: float = -85.0   # Resting potential [mV]
    V_peak: float = 40.0    # Peak depolarization [mV]

    @property
    def V_range(self) -> float:
        """Voltage range [mV]."""
        return self.V_peak - self.V_rest  # = 125 mV

    def to_physical(self, u: np.ndarray) -> np.ndarray:
        """Convert dimensionless u to physical mV."""
        return self.V_rest + u * self.V_range

    def to_dimensionless(self, V_mV: np.ndarray) -> np.ndarray:
        """Convert physical mV to dimensionless u."""
        return (V_mV - self.V_rest) / self.V_range


# =============================================================================
# Fenton-Karma Parameters
# =============================================================================

@dataclass
class FKParams:
    """
    Fenton-Karma model parameters.

    All time constants in ms. Voltage thresholds are dimensionless.

    The model has 14 parameters controlling:
    - Fast inward current (J_fi): tau_d, tau_v_plus, tau_v1_minus, tau_v2_minus
    - Slow outward current (J_so): tau_0, tau_r
    - Slow inward current (J_si): tau_si, k, u_csi
    - Thresholds: u_c, u_v
    - Membrane: Cm

    Note: tau_v_minus is VOLTAGE-DEPENDENT:
        tau_v_minus = u_v < u ? tau_v2_minus : tau_v1_minus
    """
    # Fast inward current (J_fi) - Depolarization
    tau_d: float = 0.25           # Activation time constant [ms]
    tau_v_plus: float = 3.33      # v-gate closing [ms]
    tau_v1_minus: float = 19.6    # v-gate opening (u > u_v) [ms]
    tau_v2_minus: float = 1250.0  # v-gate opening (u < u_v) [ms]

    # Slow outward current (J_so) - Repolarization
    tau_0: float = 12.5           # Below threshold [ms]
    tau_r: float = 33.33          # Above threshold [ms]

    # Slow inward current (J_si) - Plateau (Ca-like)
    tau_si: float = 29.0          # Time constant [ms]
    tau_w_plus: float = 870.0     # w-gate closing [ms] - MAIN APD CONTROL
    tau_w_minus: float = 41.0     # w-gate opening [ms]
    k: float = 10.0               # Tanh steepness for J_si
    u_csi: float = 0.85           # Threshold for J_si

    # Voltage thresholds (dimensionless)
    u_c: float = 0.13             # Main excitation threshold
    u_v: float = 0.04             # Secondary threshold (tau_v_minus switch)

    # Membrane capacitance (usually 1.0)
    Cm: float = 1.0

    def validate(self) -> None:
        """Check parameters are in valid ranges."""
        assert self.tau_d > 0, "tau_d must be positive"
        assert self.tau_v_plus > 0, "tau_v_plus must be positive"
        assert self.tau_v1_minus > 0, "tau_v1_minus must be positive"
        assert self.tau_v2_minus > 0, "tau_v2_minus must be positive"
        assert self.tau_0 > 0, "tau_0 must be positive"
        assert self.tau_r > 0, "tau_r must be positive"
        assert self.tau_si > 0, "tau_si must be positive"
        assert self.tau_w_plus > 0, "tau_w_plus must be positive"
        assert self.tau_w_minus > 0, "tau_w_minus must be positive"
        assert self.k > 0, "k must be positive"
        assert 0 < self.u_c < 1, "u_c must be in (0, 1)"
        assert 0 < self.u_csi < 1, "u_csi must be in (0, 1)"
        assert 0 <= self.u_v < self.u_c, "u_v must be in [0, u_c)"
        assert self.Cm > 0, "Cm must be positive"

    def to_tuple(self) -> tuple:
        """Convert to tuple for Numba kernel (order matters!)."""
        return (
            self.tau_d, self.tau_v_plus, self.tau_v1_minus, self.tau_v2_minus,
            self.tau_0, self.tau_r,
            self.tau_si, self.tau_w_plus, self.tau_w_minus, self.k, self.u_csi,
            self.u_c, self.u_v, self.Cm
        )

    def summary(self) -> str:
        """Human-readable parameter summary."""
        lines = [
            "Fenton-Karma Parameters",
            "=" * 40,
            "",
            "Fast Inward Current (J_fi):",
            f"  tau_d       = {self.tau_d:8.3f} ms",
            f"  tau_v_plus  = {self.tau_v_plus:8.3f} ms",
            f"  tau_v1_minus= {self.tau_v1_minus:8.3f} ms",
            f"  tau_v2_minus= {self.tau_v2_minus:8.3f} ms",
            "",
            "Slow Outward Current (J_so):",
            f"  tau_0       = {self.tau_0:8.3f} ms",
            f"  tau_r       = {self.tau_r:8.3f} ms",
            "",
            "Slow Inward Current (J_si):",
            f"  tau_si      = {self.tau_si:8.3f} ms",
            f"  tau_w_plus  = {self.tau_w_plus:8.3f} ms (APD control)",
            f"  tau_w_minus = {self.tau_w_minus:8.3f} ms",
            f"  k           = {self.k:8.3f}",
            f"  u_csi       = {self.u_csi:8.3f}",
            "",
            "Thresholds:",
            f"  u_c         = {self.u_c:8.3f}",
            f"  u_v         = {self.u_v:8.3f}",
            f"  Cm          = {self.Cm:8.3f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Spatial Parameters
# =============================================================================

@dataclass
class SpatialParams:
    """
    Spatial domain and diffusion parameters.

    All in physical units (mm, mm²/ms).

    Default diffusion coefficients tuned for CV ≈ 500 mm/s:
    - D_parallel = 0.1 mm²/ms (along fibers)
    - D_perp = 0.05 mm²/ms (2:1 anisotropy)
    """
    # Domain size [mm]
    Lx: float = 80.0
    Ly: float = 80.0

    # Grid spacing [mm]
    dx: float = 0.5
    dy: float = 0.5

    # Diffusion coefficients [mm²/ms]
    # Tuned for CV ≈ 500 mm/s (target from V2)
    D_parallel: float = 0.1    # Along fibers
    D_perp: float = 0.05       # Perpendicular to fibers (2:1 ratio)

    # Fiber angle [degrees] (0 = horizontal/rightward)
    fiber_angle: float = 0.0

    @property
    def nx(self) -> int:
        """Number of grid points in x."""
        return int(np.round(self.Lx / self.dx)) + 1

    @property
    def ny(self) -> int:
        """Number of grid points in y."""
        return int(np.round(self.Ly / self.dy)) + 1

    @property
    def anisotropy_ratio(self) -> float:
        """Ratio D_parallel / D_perp."""
        return self.D_parallel / self.D_perp

    def diffusion_tensor(self) -> tuple:
        """
        Compute diffusion tensor components (Dxx, Dyy, Dxy).

        D = R @ diag(D_parallel, D_perp) @ R^T
        where R is rotation matrix for fiber angle.
        """
        theta = np.radians(self.fiber_angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        Dxx = self.D_parallel * cos_t**2 + self.D_perp * sin_t**2
        Dyy = self.D_parallel * sin_t**2 + self.D_perp * cos_t**2
        Dxy = (self.D_parallel - self.D_perp) * cos_t * sin_t

        return Dxx, Dyy, Dxy

    def check_stability(self, dt: float) -> dict:
        """
        Check CFL stability condition.

        For explicit Euler: r = D * dt / dx² < 0.25
        """
        D_max = max(self.D_parallel, self.D_perp)
        dx_min = min(self.dx, self.dy)

        r = D_max * dt / (dx_min ** 2)
        dt_max = 0.25 * dx_min**2 / D_max

        return {
            'r': r,
            'stable': r < 0.25,
            'dt_max': dt_max,
            'D_max': D_max,
            'dx_min': dx_min,
        }


# =============================================================================
# Pre-defined Parameter Sets (from Fenton et al. 2002)
# =============================================================================

def get_paramset_3() -> FKParams:
    """
    Parameter Set 3: Stable spiral waves.

    Recommended starting point - produces clean spirals without breakup.
    APD approximately 200-300 ms depending on pacing.
    """
    return FKParams(
        tau_d=0.25,
        tau_v_plus=3.33,
        tau_v1_minus=19.6,
        tau_v2_minus=1250.0,
        tau_0=12.5,
        tau_r=33.33,
        tau_si=29.0,
        tau_w_plus=870.0,
        tau_w_minus=41.0,
        k=10.0,
        u_csi=0.85,
        u_c=0.13,
        u_v=0.04,
        Cm=1.0,
    )


def get_paramset_6() -> FKParams:
    """
    Parameter Set 6: Similar to MATLAB fk2d.m.

    Slightly different dynamics, good for comparison.
    """
    return FKParams(
        tau_d=0.395,
        tau_v_plus=3.33,
        tau_v1_minus=9.0,
        tau_v2_minus=8.0,
        tau_0=9.0,
        tau_r=33.33,
        tau_si=29.0,
        tau_w_plus=250.0,
        tau_w_minus=60.0,
        k=15.0,
        u_csi=0.5,
        u_c=0.13,
        u_v=0.04,
        Cm=1.0,
    )


def get_paramset_1a() -> FKParams:
    """Parameter Set 1A: From cardiax."""
    return FKParams(
        tau_d=0.41,
        tau_v_plus=3.33,
        tau_v1_minus=19.6,
        tau_v2_minus=1000.0,
        tau_0=8.3,
        tau_r=50.0,
        tau_si=45.0,
        tau_w_plus=667.0,
        tau_w_minus=11.0,
        k=10.0,
        u_csi=0.85,
        u_c=0.13,
        u_v=0.0055,
        Cm=1.0,
    )


def get_paramset_4a() -> FKParams:
    """Parameter Set 4A: Different threshold behavior."""
    return FKParams(
        tau_d=0.407,
        tau_v_plus=3.33,
        tau_v1_minus=15.6,
        tau_v2_minus=5.0,
        tau_0=9.0,
        tau_r=34.0,
        tau_si=26.5,
        tau_w_plus=350.0,
        tau_w_minus=80.0,
        k=15.0,
        u_csi=0.45,
        u_c=0.15,
        u_v=0.04,
        Cm=1.0,
    )


def get_paramset_apd250() -> FKParams:
    """
    Custom Parameter Set: APD90 = 250 ms target.

    Based on Set 3, tuned for APD90 = 250 ms.
    Use this for matching our V2 target parameters.
    """
    return FKParams(
        tau_d=0.25,
        tau_v_plus=3.33,
        tau_v1_minus=19.6,
        tau_v2_minus=1250.0,
        tau_0=12.5,
        tau_r=33.33,
        tau_si=29.0,
        tau_w_plus=1700.0,    # Tuned for APD90 ≈ 250 ms
        tau_w_minus=41.0,
        k=10.0,
        u_csi=0.85,
        u_c=0.13,
        u_v=0.04,
        Cm=1.0,
    )


# Dictionary of all presets
PARAM_PRESETS = {
    '3': get_paramset_3,
    '6': get_paramset_6,
    '1a': get_paramset_1a,
    '4a': get_paramset_4a,
    'apd250': get_paramset_apd250,  # Our target!
}


def get_preset(name: str) -> FKParams:
    """
    Get a preset parameter set by name.

    Available: '3', '6', '1a', '4a'
    """
    if name not in PARAM_PRESETS:
        available = ', '.join(PARAM_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PARAM_PRESETS[name]()


# =============================================================================
# Default Configuration
# =============================================================================

def default_fk_params() -> FKParams:
    """Get default FK parameters (APD250 - our target)."""
    return get_paramset_apd250()


def default_spatial_params() -> SpatialParams:
    """Get default spatial parameters."""
    return SpatialParams()


def default_physical_constants() -> PhysicalConstants:
    """Get default physical constants."""
    return PhysicalConstants()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Parameter System")
    print("=" * 60)

    # Test FK params
    params = default_fk_params()
    params.validate()
    print(params.summary())

    print("\n" + "=" * 60)

    # Test spatial params
    spatial = default_spatial_params()
    print(f"\nSpatial Parameters:")
    print(f"  Domain: {spatial.Lx} × {spatial.Ly} mm")
    print(f"  Grid: {spatial.nx} × {spatial.ny} points")
    print(f"  dx = {spatial.dx} mm")
    print(f"  D_parallel = {spatial.D_parallel} mm²/ms")
    print(f"  D_perp = {spatial.D_perp} mm²/ms")
    print(f"  Anisotropy = {spatial.anisotropy_ratio}:1")

    # Check stability
    dt = 0.02  # ms
    stability = spatial.check_stability(dt)
    print(f"\nStability check (dt = {dt} ms):")
    print(f"  CFL number r = {stability['r']:.4f}")
    print(f"  dt_max = {stability['dt_max']:.4f} ms")
    print(f"  Status: {'✓ STABLE' if stability['stable'] else '✗ UNSTABLE'}")

    # Test voltage conversion
    phys = default_physical_constants()
    print(f"\nVoltage Conversion:")
    print(f"  u=0.0 → {phys.to_physical(0.0):.1f} mV (rest)")
    print(f"  u=0.5 → {phys.to_physical(0.5):.1f} mV")
    print(f"  u=1.0 → {phys.to_physical(1.0):.1f} mV (peak)")
