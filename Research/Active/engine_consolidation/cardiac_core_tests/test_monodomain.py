"""Tests for cardiac_core.api.monodomain() — simplified monodomain API."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Engine is now vendored under cardiac_core.monodomain — no sys.path insert needed.

from cardiac_core import monodomain, create_cardiac_mesh, save_cardiac_mesh


class TestMonodomainFromData:
    """Test monodomain() with CardiacMeshData directly (no file)."""

    def test_basic_run(self):
        """Create mesh → monodomain → run 5ms → snapshots have correct shape."""
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05, D=0.001)
        sim = monodomain(mesh, device='cpu')

        snapshots = list(sim.snapshots(5.0, save_every=1.0))
        assert len(snapshots) >= 4

        Nx = round(1.0 / 0.05) + 1  # 21
        Ny = round(0.5 / 0.05) + 1  # 11
        for snap in snapshots:
            assert snap.V.shape == (Nx, Ny)
            assert snap.phi_e is None
            assert snap.Nx == Nx
            assert snap.Ny == Ny
            assert snap.dx == 0.05
            assert snap.dy == 0.05

    def test_stimulus_activates(self):
        """Stimulus should depolarize tissue — V_max should exceed resting."""
        mesh = create_cardiac_mesh(
            Lx=1.0, Ly=0.5, dx=0.05,
            stim_amplitude=-80.0,
            stim_start=0.0,
            stim_duration=2.0,
        )
        sim = monodomain(mesh, device='cpu')

        # Run past stimulus
        for snap in sim.snapshots(5.0, save_every=1.0):
            pass
        V_max = snap.V.max().item()
        # TTP06 resting ~-86 mV, depolarized should be >0 mV
        assert V_max > -50.0, f"V_max={V_max}, stimulus didn't activate"

    def test_properties(self):
        """V and t properties work."""
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        sim = monodomain(mesh, device='cpu')

        assert sim.t == 0.0
        Nx = round(0.5 / 0.05) + 1
        assert sim.V.shape == (Nx, Nx)


class TestMonodomainFromFile:
    """Test monodomain() with .npz file path."""

    def test_from_file(self, tmp_path):
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05)
        path = str(tmp_path / "mesh.npz")
        save_cardiac_mesh(path, mesh)

        sim = monodomain(path, device='cpu')
        snapshots = list(sim.snapshots(3.0, save_every=1.0))
        assert len(snapshots) >= 2

    def test_override_dt(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05, dt=0.02)
        sim = monodomain(mesh, dt=0.01, device='cpu')
        # Should use overridden dt
        assert sim._engine.dt == 0.01


class TestEngineIsV55:
    """The monodomain factory must run the Cm-correct V5.5 engine.

    NOT a path/`state.Cm` substring check (brittle against stale sys.modules,
    and V5.4's SimulationState has no Cm field anyway). Instead a BEHAVIORAL
    Cm-sensitivity check on the reaction step: from an identical mid-AP state,
    one Rush-Larsen reaction step at Cm=2 must give exactly half the voltage
    change of Cm=1 (V5.5 divides the reaction by Cm; V5.4 does not, so V5.4
    would give equal dV). The reaction step touches neither diffusion nor
    `state.spatial`, so it isolates the `/Cm` cleanly.
    """

    def test_reaction_divides_by_cm(self):
        # Construct via the factory first — this loads the V5.5 engine into
        # sys.modules (via _prepare_engine) before we import engine internals.
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        sim = monodomain(mesh, device='cpu')
        model = sim._engine._ionic_model  # the actual TTP06Model instance

        from cardiac_core._monodomain.simulation.classical.solver.ionic_time_stepping.rush_larsen import (
            RushLarsenSolver,
        )
        from cardiac_core._monodomain.simulation.classical.state import SimulationState

        n = 4
        dt = 0.01
        # Mid-AP depolarized voltage → Iion is substantially non-zero, so the
        # /Cm distinction is a real, large effect (not resting-state noise).
        V0 = torch.full((n,), -20.0, dtype=torch.float64)
        ionic0 = model.get_initial_state(n_cells=n)
        if ionic0.dtype != torch.float64:
            ionic0 = ionic0.to(torch.float64)

        def one_reaction_step(Cm):
            state = SimulationState(
                spatial=None,
                n_dof=n,
                x=torch.zeros(n, dtype=torch.float64),
                y=torch.zeros(n, dtype=torch.float64),
                V=V0.clone(),
                ionic_states=ionic0.clone(),
                gate_indices=model.gate_indices,
                concentration_indices=model.concentration_indices,
                Cm=Cm,
            )
            RushLarsenSolver(model).step(state, dt)
            return state.V - V0  # pure reaction dV (no diffusion in the ionic step)

        dV_cm1 = one_reaction_step(1.0)
        dV_cm2 = one_reaction_step(2.0)

        # V5.5 signature: reaction dV scales as 1/Cm.
        torch.testing.assert_close(dV_cm2, dV_cm1 / 2.0, atol=1e-12, rtol=0.0)
        # And Cm must MATTER — V5.4 (no /Cm) would make these equal.
        assert not torch.allclose(dV_cm2, dV_cm1, atol=1e-6), \
            "reaction is Cm-insensitive — factory is running V5.4, not V5.5"
        assert dV_cm1.abs().max() > 1e-3, "mid-AP reaction dV unexpectedly tiny"

    def test_engine_module_under_v55(self):
        """Secondary (soft) check: the live rush_larsen module file is V5.5."""
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        monodomain(mesh, device='cpu')  # construct via the vendored cardiac_core._monodomain solver
        import cardiac_core._monodomain.simulation.classical.solver.ionic_time_stepping.rush_larsen as rl
        assert 'cardiac_core' in rl.__file__ and '_monodomain' in rl.__file__, \
            f"rush_larsen not under cardiac_core/_monodomain: {rl.__file__}"
        assert 'Engine_V5' not in rl.__file__   # the original engine folder is NOT on the import path


class TestMonodomainMatchesDirect:
    """Verify wrapper produces same result as direct construction (V5.5; Cm=1 ≡ V5.4)."""

    def test_matches_direct(self):
        """Same mesh → wrapper vs direct → identical V after 3ms."""
        from cardiac_core.mesh.structured import StructuredGrid
        from cardiac_core._monodomain import FDMDiscretization, MonodomainSimulation
        from cardiac_core.stimulus.protocol import StimulusProtocol

        dx = 0.05
        Lx, Ly = 1.0, 0.5
        D = 0.001
        Nx = round(Lx / dx) + 1
        Ny = round(Ly / dx) + 1

        # --- Direct construction ---
        grid_direct = StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny, device='cpu')
        spatial_direct = FDMDiscretization(grid_direct, D=D, chi=1400.0, Cm=1.0)

        x, y = grid_direct.coordinates
        stim_direct = StimulusProtocol()
        stim_direct.add_stimulus(
            region=(x < 0.1),
            start_time=1.0,
            duration=2.0,
            amplitude=-80.0,
        )

        sim_direct = MonodomainSimulation(
            spatial=spatial_direct,
            ionic_model='ttp06',
            stimulus=stim_direct,
            dt=0.02,
            splitting='strang',
            diffusion_solver='crank_nicolson',
            linear_solver='pcg',
            cell_type='ENDO',  # match wrapper default
        )

        # --- Wrapper construction ---
        mesh = create_cardiac_mesh(Lx=Lx, Ly=Ly, dx=dx, D=D)
        sim_wrapper = monodomain(mesh, device='cpu')

        # Run both for 3ms
        V_direct = None
        for state in sim_direct.run(3.0, save_every=3.0):
            V_direct = state.V.clone()

        V_wrapper = None
        for snap in sim_wrapper.snapshots(3.0, save_every=3.0):
            V_wrapper = snap.V.clone()

        # Compare (wrapper returns grid, direct returns flat)
        V_direct_grid = grid_direct.flat_to_grid(V_direct)
        assert torch.allclose(V_wrapper, V_direct_grid, atol=1e-10), \
            f"Max diff: {(V_wrapper - V_direct_grid).abs().max()}"
