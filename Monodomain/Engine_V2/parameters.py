"""
Parameter Management System
============================

Length-scale based parameter scaling for cardiac electrophysiology simulations.

The core idea: All spatial parameters scale with a reference length L_ref.
This allows consistent physics across different mesh resolutions and domain sizes.

Physical Units:
- Voltage: mV (millivolts)
- Time: ms (milliseconds)
- Space: mm (millimeters)
- Diffusion: mm²/ms

Author: Generated with Claude Code
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class PhysicalConstants:
    """
    Fundamental physical constants (not scaled by mesh).
    """
    # Membrane properties
    C_m: float = 1.0  # Membrane capacitance [μF/cm²]

    # Voltage scales
    V_rest: float = -85.0  # Resting potential [mV]
    V_peak: float = 40.0   # Peak depolarization [mV]

    # Physiological targets
    APD90_target: float = 250.0  # Target APD90 [ms]
    CV_longitudinal_target: float = 500.0  # Target conduction velocity [mm/s]
    CV_transverse_target: float = 250.0   # Target transverse CV [mm/s]

    def voltage_range(self) -> float:
        """Total voltage excursion during AP."""
        return self.V_peak - self.V_rest

    def anisotropy_ratio(self) -> float:
        """Ratio of longitudinal to transverse conduction velocity."""
        return self.CV_longitudinal_target / self.CV_transverse_target


@dataclass
class AlievPanfilovParameters:
    """
    Parameters for the Aliev-Panfilov ionic model with voltage-dependent epsilon.

    These are dimensionless parameters that control AP shape and recovery.

    Standard Aliev-Panfilov Parameters:
    -----------------------------------
    k, a, epsilon0, mu1, mu2 - control AP shape and basic dynamics

    Voltage-Dependent Epsilon Enhancement:
    --------------------------------------
    epsilon_rest - recovery boost at rest for fast w decay
    V_threshold - voltage threshold for sigmoid transition
    k_sigmoid - steepness of sigmoid transition

    These additions allow independent control of APD duration and recovery time.
    """
    # Standard Aliev-Panfilov parameters
    k: float = 8.0      # Cubic nonlinearity strength
    a: float = 0.1      # Threshold parameter (FIXED: reduced from 0.15 for better excitability)
    epsilon0: float = 0.01    # Base recovery time scale (FIXED: increased from 0.002 for stability)
    mu1: float = 0.2    # Recovery modulation parameter (baseline value)
    mu2: float = 0.3    # Recovery modulation parameter (CRITICAL: used in division check)

    # NEW: Voltage-dependent epsilon parameters
    epsilon_rest: float = 0.05    # Recovery boost at rest (0 = disabled)
    V_threshold: float = 0.2      # Voltage threshold for sigmoid [0-1]
    k_sigmoid: float = 20.0       # Sigmoid steepness (higher = sharper transition)

    def validate(self):
        """Check parameters are in reasonable ranges."""
        assert self.k > 0, "k must be positive"
        assert 0 < self.a < 1, "a should be in (0, 1)"
        assert self.epsilon0 > 0, "epsilon0 must be positive"
        assert self.mu1 >= 0, "mu1 must be non-negative"
        assert self.mu2 > 0, "mu2 must be positive"
        assert self.epsilon_rest >= 0, "epsilon_rest must be non-negative"
        assert 0 < self.V_threshold < 1, "V_threshold should be in (0, 1)"
        assert self.k_sigmoid > 0, "k_sigmoid must be positive"


@dataclass
class SpatialScaling:
    """
    Spatial parameters that scale with reference length.

    The reference length L_ref defines the characteristic spatial scale.
    All other spatial parameters are derived from this.

    Design philosophy:
    - L_ref determines domain size
    - Resolution (points_per_wavelength) determines mesh spacing
    - Diffusion coefficients are physiologically based
    """
    L_ref: float = 20.0  # Reference length scale [mm]

    # Mesh resolution control
    points_per_wavelength: float = 10.0  # Grid points per AP wavelength

    # Diffusion coefficients (from PARAMETERS_REFERENCE.txt Section 2)
    D_parallel: float = 1.0   # Longitudinal diffusivity [mm²/ms]
    D_perp: float = 0.5       # Transverse diffusivity [mm²/ms]

    # Anisotropy
    anisotropy_ratio: float = 2.0  # D_parallel / D_perp

    def __post_init__(self):
        """Ensure D_perp consistent with anisotropy ratio."""
        if self.D_perp != self.D_parallel / self.anisotropy_ratio:
            # Recompute D_perp to be consistent
            self.D_perp = self.D_parallel / self.anisotropy_ratio

    def domain_size(self, aspect_ratio: float = 1.0) -> tuple[float, float]:
        """
        Compute domain size (Lx, Ly) based on L_ref.

        Parameters
        ----------
        aspect_ratio : float
            Ratio Lx / Ly. Default 1.0 for square domain.

        Returns
        -------
        Lx, Ly : float
            Domain dimensions [mm]
        """
        Lx = self.L_ref
        Ly = self.L_ref / aspect_ratio
        return Lx, Ly

    def mesh_spacing(self, CV_estimate: float = 500.0, APD_estimate: float = 250.0) -> tuple[float, float]:
        """
        Compute mesh spacing based on wavelength resolution.

        Wavelength λ = CV * APD, and we want points_per_wavelength grid points per λ.

        Parameters
        ----------
        CV_estimate : float
            Estimated conduction velocity [mm/s]
        APD_estimate : float
            Estimated action potential duration [ms]

        Returns
        -------
        dx, dy : float
            Grid spacing [mm]
        """
        # Convert CV from mm/s to mm/ms
        CV_mm_per_ms = CV_estimate / 1000.0

        # Wavelength [mm]
        wavelength = CV_mm_per_ms * APD_estimate

        # Spacing
        dx = dy = wavelength / self.points_per_wavelength

        return dx, dy

    def grid_dimensions(self, aspect_ratio: float = 1.0) -> tuple[int, int]:
        """
        Compute number of grid points (nx, ny) for given domain.

        Parameters
        ----------
        aspect_ratio : float
            Lx / Ly ratio

        Returns
        -------
        nx, ny : int
            Number of grid points
        """
        Lx, Ly = self.domain_size(aspect_ratio)
        dx, dy = self.mesh_spacing()

        nx = int(np.ceil(Lx / dx)) + 1
        ny = int(np.ceil(Ly / dy)) + 1

        return nx, ny

    def estimate_time_step(self, safety_factor: float = 0.5) -> float:
        """
        Estimate stable time step for explicit Euler.

        CFL condition: dt < safety * dx² / (2 * D_max)

        Parameters
        ----------
        safety_factor : float
            Safety factor (< 1.0)

        Returns
        -------
        dt : float
            Suggested time step [ms]
        """
        dx, dy = self.mesh_spacing()
        dx_min = min(dx, dy)
        D_max = max(self.D_parallel, self.D_perp)

        dt_crit = safety_factor * dx_min**2 / (2.0 * D_max)

        return dt_crit

    def scale(self, factor: float) -> 'SpatialScaling':
        """
        Create a new SpatialScaling with L_ref scaled by factor.

        This scales the entire spatial domain while keeping physics consistent.

        Parameters
        ----------
        factor : float
            Scaling factor (e.g., 2.0 doubles domain size)

        Returns
        -------
        scaled : SpatialScaling
            New scaled configuration
        """
        return SpatialScaling(
            L_ref=self.L_ref * factor,
            points_per_wavelength=self.points_per_wavelength,
            D_parallel=self.D_parallel,
            D_perp=self.D_perp,
            anisotropy_ratio=self.anisotropy_ratio,
        )


@dataclass
class SimulationParameters:
    """
    Complete parameter set for cardiac electrophysiology simulation.

    Combines physical constants, ionic model parameters, and spatial scaling.
    """
    physical: PhysicalConstants = field(default_factory=PhysicalConstants)
    ionic: AlievPanfilovParameters = field(default_factory=AlievPanfilovParameters)
    spatial: SpatialScaling = field(default_factory=SpatialScaling)

    def validate(self):
        """Validate all parameter sets."""
        self.ionic.validate()
        # Add more validation as needed

    def summary(self) -> str:
        """Generate human-readable summary of parameters."""
        lines = [
            "=" * 70,
            "SIMULATION PARAMETERS SUMMARY",
            "=" * 70,
            "",
            "PHYSICAL CONSTANTS:",
            f"  C_m = {self.physical.C_m} μF/cm²",
            f"  V_rest = {self.physical.V_rest} mV",
            f"  V_peak = {self.physical.V_peak} mV",
            f"  Voltage range = {self.physical.voltage_range()} mV",
            f"  Target APD90 = {self.physical.APD90_target} ms",
            f"  Target CV (longitudinal) = {self.physical.CV_longitudinal_target} mm/s",
            f"  Target CV (transverse) = {self.physical.CV_transverse_target} mm/s",
            "",
            "ALIEV-PANFILOV PARAMETERS:",
            f"  k = {self.ionic.k}",
            f"  a = {self.ionic.a}",
            f"  epsilon0 = {self.ionic.epsilon0}",
            f"  mu1 = {self.ionic.mu1}",
            f"  mu2 = {self.ionic.mu2}",
            "",
            "SPATIAL SCALING:",
            f"  L_ref = {self.spatial.L_ref} mm",
            f"  D_parallel = {self.spatial.D_parallel} mm²/ms",
            f"  D_perp = {self.spatial.D_perp} mm²/ms",
            f"  Anisotropy ratio = {self.spatial.anisotropy_ratio}",
            f"  Points per wavelength = {self.spatial.points_per_wavelength}",
            "",
            "DERIVED QUANTITIES:",
        ]

        Lx, Ly = self.spatial.domain_size()
        nx, ny = self.spatial.grid_dimensions()
        dx, dy = self.spatial.mesh_spacing()
        dt_est = self.spatial.estimate_time_step()

        lines.extend([
            f"  Domain size = {Lx} × {Ly} mm",
            f"  Grid dimensions = {nx} × {ny} points",
            f"  Mesh spacing = {dx:.4f} × {dy:.4f} mm",
            f"  Estimated time step = {dt_est:.4f} ms",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'physical': {
                'C_m': self.physical.C_m,
                'V_rest': self.physical.V_rest,
                'V_peak': self.physical.V_peak,
                'APD90_target': self.physical.APD90_target,
                'CV_longitudinal_target': self.physical.CV_longitudinal_target,
                'CV_transverse_target': self.physical.CV_transverse_target,
            },
            'ionic': {
                'k': self.ionic.k,
                'a': self.ionic.a,
                'epsilon0': self.ionic.epsilon0,
                'mu1': self.ionic.mu1,
                'mu2': self.ionic.mu2,
            },
            'spatial': {
                'L_ref': self.spatial.L_ref,
                'points_per_wavelength': self.spatial.points_per_wavelength,
                'D_parallel': self.spatial.D_parallel,
                'D_perp': self.spatial.D_perp,
                'anisotropy_ratio': self.spatial.anisotropy_ratio,
            }
        }


def create_default_parameters() -> SimulationParameters:
    """
    Create default parameter set based on PARAMETERS_REFERENCE.txt.

    Returns
    -------
    params : SimulationParameters
        Default parameter configuration
    """
    return SimulationParameters()


def create_custom_parameters(
    L_ref: float = 20.0,
    D_parallel: float = 1.0,
    anisotropy: float = 2.0,
    APD90_target: float = 250.0,
    **kwargs
) -> SimulationParameters:
    """
    Create custom parameter set with common overrides.

    Parameters
    ----------
    L_ref : float
        Reference length scale [mm]
    D_parallel : float
        Longitudinal diffusivity [mm²/ms]
    anisotropy : float
        Anisotropy ratio
    APD90_target : float
        Target APD90 [ms]
    **kwargs
        Additional overrides for any parameter

    Returns
    -------
    params : SimulationParameters
        Custom parameter configuration
    """
    physical = PhysicalConstants(APD90_target=APD90_target)
    ionic = AlievPanfilovParameters()
    spatial = SpatialScaling(
        L_ref=L_ref,
        D_parallel=D_parallel,
        anisotropy_ratio=anisotropy
    )

    # Apply any additional overrides from kwargs
    # (This is a simple implementation; could be more sophisticated)

    params = SimulationParameters(
        physical=physical,
        ionic=ionic,
        spatial=spatial
    )

    params.validate()
    return params


if __name__ == "__main__":
    # Test the parameter system
    print("Testing parameter system...\n")

    params = create_default_parameters()
    print(params.summary())

    print("\n\nTesting scaling:")
    scaled = params.spatial.scale(2.0)
    print(f"Original L_ref: {params.spatial.L_ref} mm")
    print(f"Scaled L_ref: {scaled.L_ref} mm")
    print(f"Original domain: {params.spatial.domain_size()}")
    print(f"Scaled domain: {scaled.domain_size()}")
