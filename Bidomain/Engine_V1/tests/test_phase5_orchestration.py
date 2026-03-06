"""
Phase 5 Validation Tests — BidomainSimulation Orchestrator

Tests 5-T1 through 5-T6 from PROGRESS.md.
(5-T7, 5-T8 postponed — require long TTP06 runs.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
import numpy as np


def _make_spatial(nx=10, ny=10, lx=1.0, ly=1.0, D_i=0.00124, D_e=0.00446, bc='insulated'):
    """Helper: create BidomainFDMDiscretization."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

    grid = StructuredGrid.create_rectangle(Lx=lx, Ly=ly, Nx=nx, Ny=ny)
    if bc == 'insulated':
        grid.boundary_spec = BoundarySpec.insulated()
    elif bc == 'bath':
        grid.boundary_spec = BoundarySpec.bath_coupled()
    elif bc == 'mixed':
        from cardiac_sim.tissue_builder.mesh.boundary import Edge
        grid.boundary_spec = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
    cond = BidomainConductivity(D_i=D_i, D_e=D_e)
    return BidomainFDMDiscretization(grid, cond, chi=1400.0, Cm=1.0)


# === 5-T1: Factory string configs ===

def test_factory_construction():
    """5-T1: Construct BidomainSimulation with all default strings."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
    from cardiac_sim.tissue_builder.stimulus import StimulusProtocol, left_edge_region

    spatial = _make_spatial(nx=10, ny=10)

    stimulus = StimulusProtocol()
    stimulus.add_stimulus(
        region=left_edge_region(width=0.1),
        start_time=0.0, duration=1.0, amplitude=-52.0
    )

    sim = BidomainSimulation(
        spatial=spatial,
        ionic_model='ttp06',
        stimulus=stimulus,
        dt=0.02,
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='decoupled',
        parabolic_solver='pcg',
        elliptic_solver='auto',
        theta=0.5
    )

    assert sim.state is not None
    assert sim.state.n_dof == 100  # 10x10
    assert sim.state.Vm.shape == (100,)
    assert sim.state.phi_e.shape == (100,)
    assert sim.state.ionic_states.shape[0] == 100
    assert sim.splitting is not None


# === 5-T2: Auto-solver: Neumann isotropic ===

def test_auto_solver_neumann_iso():
    """5-T2: Insulated + isotropic -> 'spectral'."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    spatial = _make_spatial(nx=10, ny=10, bc='insulated')
    sim = BidomainSimulation(spatial=spatial, ionic_model='ttp06', dt=0.02)
    assert sim._elliptic_solver_name == 'spectral'


# === 5-T3: Auto-solver: Dirichlet isotropic ===

def test_auto_solver_dirichlet_iso():
    """5-T3: Bath-coupled + isotropic -> 'spectral'."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    spatial = _make_spatial(nx=10, ny=10, bc='bath')
    sim = BidomainSimulation(spatial=spatial, ionic_model='ttp06', dt=0.02)
    assert sim._elliptic_solver_name == 'spectral'


# === 5-T4: Auto-solver: anisotropic ===

def test_auto_solver_aniso():
    """5-T4: Anisotropic + uniform BCs -> 'pcg_spectral'."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    grid = StructuredGrid.create_rectangle(Lx=1.0, Ly=1.0, Nx=10, Ny=10)
    grid.boundary_spec = BoundarySpec.insulated()

    # Create anisotropic conductivity (fiber + cross-fiber)
    cond = BidomainConductivity(
        D_i_fiber=0.003, D_i_cross=0.0003,
        D_e_fiber=0.002, D_e_cross=0.001,
        theta=torch.zeros(10, 10, dtype=torch.float64)  # uniform fiber angle
    )
    spatial = BidomainFDMDiscretization(grid, cond, chi=1400.0, Cm=1.0)

    sim = BidomainSimulation(spatial=spatial, ionic_model='ttp06', dt=0.02)
    assert sim._elliptic_solver_name == 'pcg_spectral'


# === 5-T5: Auto-solver: mixed BCs ===

def test_auto_solver_mixed():
    """5-T5: Mixed per-axis BCs (Neumann-x, Dirichlet-y) -> 'spectral'."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    spatial = _make_spatial(nx=10, ny=10, bc='mixed')
    sim = BidomainSimulation(spatial=spatial, ionic_model='ttp06', dt=0.02)
    # Per-axis uniform BCs are spectrally eligible (DCT-x, DST-y)
    assert sim._elliptic_solver_name == 'spectral'


# === 5-T6: Strang splitting call order ===

def test_strang_splitting_order():
    """5-T6: Run 10 steps, verify ionic called twice per step, diffusion once."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    spatial = _make_spatial(nx=10, ny=10, bc='insulated')
    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06', dt=0.02,
        splitting='strang', elliptic_solver='pcg'
    )

    # Instrument the solvers to count calls
    ionic_calls = [0]
    diffusion_calls = [0]

    original_ionic_step = sim.splitting.ionic_solver.step
    original_diffusion_step = sim.splitting.diffusion_solver.step

    def counted_ionic_step(state, dt):
        ionic_calls[0] += 1
        return original_ionic_step(state, dt)

    def counted_diffusion_step(state, dt):
        diffusion_calls[0] += 1
        return original_diffusion_step(state, dt)

    sim.splitting.ionic_solver.step = counted_ionic_step
    sim.splitting.diffusion_solver.step = counted_diffusion_step

    # Run 10 steps manually
    for _ in range(10):
        sim.splitting.step(sim.state, sim.dt)
        sim.state.t += sim.dt

    # Strang: 2 ionic calls + 1 diffusion call per step
    assert ionic_calls[0] == 20, f"Expected 20 ionic calls, got {ionic_calls[0]}"
    assert diffusion_calls[0] == 10, f"Expected 10 diffusion calls, got {diffusion_calls[0]}"


# === 5-T7: run() generator works ===

def test_run_generator():
    """Verify run() yields states at correct times."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    spatial = _make_spatial(nx=10, ny=10, bc='insulated')
    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06', dt=0.02,
        elliptic_solver='pcg'
    )

    save_times = []
    for state in sim.run(t_end=0.1, save_every=0.04):
        save_times.append(state.t)
        if len(save_times) >= 3:
            break

    assert len(save_times) >= 2, f"Expected at least 2 save points, got {len(save_times)}"
    # First save at ~0.04, second at ~0.08
    assert abs(save_times[0] - 0.04) < 0.02 + 1e-10
    assert abs(save_times[1] - 0.08) < 0.02 + 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
