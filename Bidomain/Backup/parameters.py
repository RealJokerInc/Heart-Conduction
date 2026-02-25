"""
Bidomain Model Parameters
=========================

Physical constants and default parameters for bidomain cardiac simulation.
Based on literature values from:
- Sundnes et al., "Computing the Electrical Activity in the Heart"
- Clerc, 1976 (conductivity measurements)
- Various ionic model papers
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BidomainParameters:
    """Parameters for the bidomain model."""

    # Membrane properties
    C_m: float = 1.0        # Membrane capacitance [μF/cm²]
    chi: float = 1400.0     # Surface-to-volume ratio [1/cm]

    # Intracellular conductivity tensor [mS/cm]
    # (longitudinal, transverse) - along and across fiber direction
    sigma_i_l: float = 3.0       # Longitudinal intracellular
    sigma_i_t: float = 0.31525   # Transverse intracellular

    # Extracellular conductivity tensor [mS/cm]
    sigma_e_l: float = 2.0       # Longitudinal extracellular
    sigma_e_t: float = 1.3514    # Transverse extracellular

    # Derived effective conductivities for monodomain comparison
    @property
    def sigma_i(self) -> np.ndarray:
        """Intracellular conductivity tensor (2x2 diagonal)."""
        return np.diag([self.sigma_i_l, self.sigma_i_t])

    @property
    def sigma_e(self) -> np.ndarray:
        """Extracellular conductivity tensor (2x2 diagonal)."""
        return np.diag([self.sigma_e_l, self.sigma_e_t])

    @property
    def sigma_bulk(self) -> np.ndarray:
        """Bulk conductivity σ_i + σ_e."""
        return self.sigma_i + self.sigma_e

    @property
    def anisotropy_ratio_i(self) -> float:
        """Intracellular anisotropy ratio (longitudinal/transverse)."""
        return self.sigma_i_l / self.sigma_i_t

    @property
    def anisotropy_ratio_e(self) -> float:
        """Extracellular anisotropy ratio (longitudinal/transverse)."""
        return self.sigma_e_l / self.sigma_e_t


@dataclass
class MeshParameters:
    """Spatial discretization parameters."""

    # Domain size [cm]
    Lx: float = 4.0    # Domain length in x
    Ly: float = 4.0    # Domain length in y

    # Grid resolution
    dx: float = 0.01   # Grid spacing [cm] (100 μm)
    dy: float = 0.01   # Grid spacing [cm]

    @property
    def nx(self) -> int:
        """Number of grid points in x."""
        return int(self.Lx / self.dx) + 1

    @property
    def ny(self) -> int:
        """Number of grid points in y."""
        return int(self.Ly / self.dy) + 1

    @property
    def n_nodes(self) -> int:
        """Total number of grid nodes."""
        return self.nx * self.ny


@dataclass
class TimeParameters:
    """Temporal discretization parameters."""

    dt: float = 0.01       # Time step [ms]
    t_end: float = 100.0   # Simulation end time [ms]

    # Stimulus parameters
    stim_start: float = 1.0      # Stimulus start time [ms]
    stim_duration: float = 1.0   # Stimulus duration [ms]
    stim_amplitude: float = 200.0  # Stimulus amplitude [μA/cm³]

    @property
    def n_steps(self) -> int:
        """Total number of time steps."""
        return int(self.t_end / self.dt)


@dataclass
class IonicModelParameters:
    """Parameters for ionic current model."""

    # Resting potential and threshold
    V_rest: float = -85.0     # Resting potential [mV]
    V_threshold: float = -60.0  # Activation threshold [mV]
    V_peak: float = 40.0      # Peak action potential [mV]

    # Ion concentrations (for LRd model) [mM]
    K_o: float = 5.4          # Extracellular potassium (normal)
    K_i: float = 145.0        # Intracellular potassium
    Na_o: float = 140.0       # Extracellular sodium
    Na_i: float = 10.0        # Intracellular sodium
    Ca_o: float = 1.8         # Extracellular calcium

    # Ischemic [K]o range
    K_o_ischemic: float = 10.0  # Elevated [K]o in ischemia


@dataclass
class SolverParameters:
    """Linear solver and numerical scheme parameters."""

    # Operator splitting
    splitting_scheme: str = 'godunov'  # 'godunov' or 'strang'

    # Linear solver for elliptic problem
    elliptic_solver: str = 'direct'  # 'direct', 'cg', 'gmres', 'amg'
    elliptic_tol: float = 1e-8       # Iterative solver tolerance
    elliptic_maxiter: int = 1000     # Max iterations

    # Preconditioner
    preconditioner: str = 'ilu'      # 'ilu', 'jacobi', 'amg'

    # Parabolic solver
    parabolic_scheme: str = 'cn'     # 'cn' (Crank-Nicolson), 'be' (backward Euler)

    # ODE solver for ionic model
    ode_scheme: str = 'fe'           # 'fe' (forward Euler), 'rl' (Rush-Larsen)


def default_bidomain_params() -> BidomainParameters:
    """Return default bidomain parameters."""
    return BidomainParameters()


def default_mesh_params() -> MeshParameters:
    """Return default mesh parameters."""
    return MeshParameters()


def default_time_params() -> TimeParameters:
    """Return default time parameters."""
    return TimeParameters()


def default_ionic_params() -> IonicModelParameters:
    """Return default ionic model parameters."""
    return IonicModelParameters()


def default_solver_params() -> SolverParameters:
    """Return default solver parameters."""
    return SolverParameters()


def create_simulation_config(
    bidomain: BidomainParameters = None,
    mesh: MeshParameters = None,
    time: TimeParameters = None,
    ionic: IonicModelParameters = None,
    solver: SolverParameters = None
) -> Dict[str, Any]:
    """Create a complete simulation configuration."""
    return {
        'bidomain': bidomain or default_bidomain_params(),
        'mesh': mesh or default_mesh_params(),
        'time': time or default_time_params(),
        'ionic': ionic or default_ionic_params(),
        'solver': solver or default_solver_params()
    }


# Quick validation
if __name__ == '__main__':
    params = default_bidomain_params()
    print("Bidomain Parameters:")
    print(f"  C_m = {params.C_m} μF/cm²")
    print(f"  χ = {params.chi} 1/cm")
    print(f"  σ_i = diag({params.sigma_i_l}, {params.sigma_i_t}) mS/cm")
    print(f"  σ_e = diag({params.sigma_e_l}, {params.sigma_e_t}) mS/cm")
    print(f"  Intracellular anisotropy ratio: {params.anisotropy_ratio_i:.1f}:1")
    print(f"  Extracellular anisotropy ratio: {params.anisotropy_ratio_e:.1f}:1")

    mesh = default_mesh_params()
    print(f"\nMesh: {mesh.nx} x {mesh.ny} = {mesh.n_nodes} nodes")
    print(f"  Domain: {mesh.Lx} x {mesh.Ly} cm")
    print(f"  Resolution: {mesh.dx*1e4:.0f} μm")

    time = default_time_params()
    print(f"\nTime: {time.n_steps} steps, dt = {time.dt} ms")
