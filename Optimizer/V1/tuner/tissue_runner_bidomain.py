"""
Optimizer V1 — Bidomain Tissue Runner

Measures conduction velocity using the Bidomain V1 engine's
BidomainSimulation solver (decoupled Gauss-Seidel, spectral/PCG/GMG).

Uses a narrow 2D strip (Nx × Ny_strip) since the bidomain elliptic
solve requires a 2D domain. CV measured from Vm activation times
at probe points along the center row.
"""

import sys
import os
import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass

# Add Bidomain engine path — must be FIRST to shadow monodomain cardiac_sim
_BIDOMAIN_PATH = os.path.join(os.path.dirname(__file__),
                              '..', '..', '..', 'Bidomain', 'Engine_V1')
_BIDOMAIN_PATH = os.path.abspath(_BIDOMAIN_PATH)


def _import_bidomain():
    """Import Bidomain engine modules, temporarily overriding sys.path."""
    import importlib
    # Save current cardiac_sim if loaded
    saved = sys.modules.pop('cardiac_sim', None)
    # Also remove any submodules
    to_remove = [k for k in sys.modules if k.startswith('cardiac_sim.')]
    saved_subs = {k: sys.modules.pop(k) for k in to_remove}

    sys.path.insert(0, _BIDOMAIN_PATH)
    try:
        import cardiac_sim
        importlib.reload(cardiac_sim)
        from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
        from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
        from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
        from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
        from cardiac_sim.tissue_builder.stimulus import StimulusProtocol
        from cardiac_sim.tissue_builder.stimulus.regions import left_edge_region
        from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        return (StructuredGrid, BoundarySpec, BidomainConductivity,
                BidomainFDMDiscretization, StimulusProtocol,
                left_edge_region, BidomainSimulation, MHAS13Model)
    finally:
        # Restore monodomain modules
        sys.path.remove(_BIDOMAIN_PATH)
        # Don't restore — keep bidomain modules cached for this session


# Import once at module load
(StructuredGrid, BoundarySpec, BidomainConductivity,
 BidomainFDMDiscretization, StimulusProtocol,
 left_edge_region, BidomainSimulation, MHAS13Model) = _import_bidomain()

from .config import TuningConfig, apply_scaling, theta_to_dict


@dataclass
class CVResult:
    """Result from a CV measurement."""
    cv: Optional[float] = None
    converged: bool = True
    D_i: float = 0.0
    D_e: float = 0.0
    D_eff: float = 0.0


NY_STRIP = 5  # Narrow strip width for 2D bidomain


def run_cv_measurement_bidomain(
    theta_ionic: torch.Tensor,
    D_i: float,
    D_e: float,
    config: TuningConfig,
    cable_length_cm: float = None,
    dx_cm: float = None,
    n_beats: int = 3,
) -> CVResult:
    """
    Measure CV using the Bidomain V1 engine.

    Builds a narrow 2D strip, paces from the left edge, and measures
    activation time at two probe points along the center row.

    Parameters
    ----------
    theta_ionic : ionic scaling factors
    D_i : intracellular diffusivity (cm²/ms)
    D_e : extracellular diffusivity (cm²/ms)
    config : TuningConfig
    cable_length_cm : strip length (default: config.cable_length_cm)
    dx_cm : spatial resolution (default: config.dx_cm)
    n_beats : pacing beats
    """
    if cable_length_cm is None:
        cable_length_cm = config.cable_length_cm
    if dx_cm is None:
        dx_cm = config.dx_cm

    dt = config.dt
    device = config.device
    dtype = config.dtype

    Nx = int(cable_length_cm / dx_cm) + 1
    Ny = NY_STRIP
    Lx = dx_cm * (Nx - 1)
    Ly = dx_cm * (Ny - 1)

    D_eff = D_i * D_e / (D_i + D_e)

    # 1. Grid with boundary conditions
    if config.bc_type == 'bath':
        boundary_spec = BoundarySpec.bath_coupled()
    else:
        boundary_spec = BoundarySpec.insulated()

    grid = StructuredGrid(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        boundary_spec=boundary_spec,
        _device=torch.device(device),
        _dtype=dtype,
    )

    # 2. Conductivity
    cond = BidomainConductivity(D_i=D_i, D_e=D_e)

    # 3. Spatial discretization (FDM)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    # 4. Stimulus protocol — pacing from left edge
    stimulus = StimulusProtocol()
    stim_width = dx_cm * max(3, Nx // 20)  # Left ~5%

    for beat in range(n_beats):
        stimulus.add_stimulus(
            region=left_edge_region(width=stim_width),
            start_time=1.0 + beat * config.pacing_cl,
            duration=config.stim_duration,
            amplitude=config.stim_amplitude,
        )

    # 5. Create ionic model with scaled parameters
    ionic_model = MHAS13Model(device=device)
    theta_dict = theta_to_dict(theta_ionic, config.tier)
    apply_scaling(ionic_model.params, theta_dict)

    # 6. Build BidomainSimulation
    sim = BidomainSimulation(
        spatial=spatial,
        ionic_model=ionic_model,
        stimulus=stimulus,
        dt=dt,
        splitting=config.bidomain_splitting,
        elliptic_solver=config.elliptic_solver,
        device=device,
    )

    # 7. Run simulation and track activation
    probe1_x = Nx // 4
    probe2_x = 3 * Nx // 4
    probe_y = Ny // 2
    distance_cm = (probe2_x - probe1_x) * dx_cm

    activation_threshold = -30.0  # mV
    t_act1 = None
    t_act2 = None
    V_prev_p1 = ionic_model.V_rest
    V_prev_p2 = ionic_model.V_rest

    # Only track activation during last beat
    last_beat_start = 1.0 + (n_beats - 1) * config.pacing_cl
    total_time = 1.0 + n_beats * config.pacing_cl + 50.0  # Extra 50ms after last stim

    save_every = dt * 50  # Save every 50 steps for activation tracking

    try:
        for state in sim.run(t_end=total_time, save_every=save_every):
            t = state.t

            if t < last_beat_start:
                continue

            # Reset activation tracking at start of last beat
            if t - last_beat_start < save_every * 2:
                t_act1 = None
                t_act2 = None

            # Get Vm at probe points
            V_grid = grid.flat_to_grid(state.Vm)
            v1 = V_grid[probe1_x, probe_y].item()
            v2 = V_grid[probe2_x, probe_y].item()

            # Detect upstroke crossing
            if t_act1 is None and v1 > activation_threshold and V_prev_p1 <= activation_threshold:
                t_act1 = t

            if t_act2 is None and v2 > activation_threshold and V_prev_p2 <= activation_threshold:
                t_act2 = t

            V_prev_p1 = v1
            V_prev_p2 = v2

            # Early exit once both probes activated
            if t_act1 is not None and t_act2 is not None:
                break

    except Exception as e:
        return CVResult(converged=False, D_i=D_i, D_e=D_e, D_eff=D_eff)

    # Calculate CV
    if t_act1 is not None and t_act2 is not None and t_act2 > t_act1:
        cv = distance_cm / (t_act2 - t_act1) * 1000.0  # cm/ms → cm/s
        return CVResult(cv=cv, converged=True, D_i=D_i, D_e=D_e, D_eff=D_eff)

    return CVResult(converged=False, D_i=D_i, D_e=D_e, D_eff=D_eff)
