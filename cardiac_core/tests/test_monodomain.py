"""Tests for cardiac_core.api.monodomain() — simplified monodomain API."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure engine is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "Monodomain" / "Engine_V5.4"))

from cardiac_core import monodomain, create_cardiac_mesh, save_cardiac_mesh


class TestMonodomainFromData:
    """Test monodomain() with CardiacMeshData directly (no file)."""

    def test_basic_run(self):
        """Create mesh → monodomain → run 5ms → snapshots have correct shape."""
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05, D=0.001)
        sim = monodomain(mesh, device='cpu')

        snapshots = list(sim.run(5.0, save_every=1.0))
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
        for snap in sim.run(5.0, save_every=1.0):
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
        snapshots = list(sim.run(3.0, save_every=1.0))
        assert len(snapshots) >= 2

    def test_override_dt(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05, dt=0.02)
        sim = monodomain(mesh, dt=0.01, device='cpu')
        # Should use overridden dt
        assert sim._engine.dt == 0.01


class TestMonodomainMatchesDirect:
    """Verify wrapper produces same result as direct V5.4 construction."""

    def test_matches_direct(self):
        """Same mesh → wrapper vs direct → identical V after 3ms."""
        from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
        from cardiac_sim.simulation.classical.discretization_scheme import FDMDiscretization
        from cardiac_sim.simulation.classical import MonodomainSimulation
        from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol

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
        for snap in sim_wrapper.run(3.0, save_every=3.0):
            V_wrapper = snap.V.clone()

        # Compare (wrapper returns grid, direct returns flat)
        V_direct_grid = grid_direct.flat_to_grid(V_direct)
        assert torch.allclose(V_wrapper, V_direct_grid, atol=1e-10), \
            f"Max diff: {(V_wrapper - V_direct_grid).abs().max()}"
