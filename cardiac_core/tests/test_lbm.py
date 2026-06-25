"""Tests for cardiac_core.api.lbm() — simplified LBM API."""

import numpy as np
import pytest
import torch

from cardiac_core import lbm, create_cardiac_mesh, save_cardiac_mesh


class TestLBMFromData:
    """Test lbm() with CardiacMeshData directly."""

    def test_basic_run(self):
        """Create mesh → lbm → run 5ms → snapshots have correct shape."""
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.025, D=0.001, dt=0.005)
        sim = lbm(mesh, device='cpu')

        snapshots = list(sim.snapshots(5.0, save_every=1.0))
        assert len(snapshots) >= 4

        Nx = round(0.5 / 0.025) + 1  # 21
        Ny = Nx
        for snap in snapshots:
            assert snap.V.shape == (Nx, Ny)
            assert snap.phi_e is None
            assert snap.Nx == Nx
            assert snap.Ny == Ny

    def test_generator_yields_snapshots(self):
        """run() yields SimulationSnapshot instances with increasing time."""
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.025, D=0.001, dt=0.005)
        sim = lbm(mesh, device='cpu')

        times = []
        for snap in sim.snapshots(3.0, save_every=1.0):
            times.append(snap.t)

        assert len(times) >= 2
        # Times should be strictly increasing
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_stimulus_activates(self):
        """Stimulus should depolarize tissue."""
        mesh = create_cardiac_mesh(
            Lx=0.5, Ly=0.5, dx=0.025, D=0.001, dt=0.005,
            stim_amplitude=-80.0,
            stim_start=0.0,
            stim_duration=2.0,
        )
        sim = lbm(mesh, device='cpu')

        for snap in sim.snapshots(5.0, save_every=1.0):
            pass
        V_max = snap.V.max().item()
        assert V_max > -50.0, f"V_max={V_max}, stimulus didn't activate"

    def test_properties(self):
        """V and t properties work."""
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.025, D=0.001, dt=0.005)
        sim = lbm(mesh, device='cpu')

        assert sim.t == 0.0
        Nx = round(0.5 / 0.025) + 1
        assert sim.V.shape == (Nx, Nx)


class TestLBMFromFile:
    """Test lbm() with .npz file path."""

    def test_from_file(self, tmp_path):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.025, D=0.001, dt=0.005)
        path = str(tmp_path / "mesh.npz")
        save_cardiac_mesh(path, mesh)

        sim = lbm(path, device='cpu')
        snapshots = list(sim.snapshots(3.0, save_every=1.0))
        assert len(snapshots) >= 2


class TestLBMMatchesDirect:
    """Verify wrapper produces same result as direct LBM V1 construction."""

    def test_matches_direct(self):
        """Same params → wrapper vs direct → identical V after 3ms."""
        from cardiac_core._lbm.simulation import LBMSimulation
        from cardiac_core.ionic import TTP06Model

        dx = 0.025
        Nx = 21
        Ny = 21
        dt = 0.005
        D = 0.001

        # --- Direct construction ---
        ionic = TTP06Model(device='cpu')
        sim_direct = LBMSimulation(
            Nx=Nx, Ny=Ny, dx=dx, dt=dt,
            D=D, ionic_model=ionic, Cm=1.0,
            lattice='d2q5',
        )
        stim_mask = torch.zeros(Nx, Ny, dtype=torch.bool)
        stim_mask[:4, :] = True  # x < 0.1
        sim_direct.add_stimulus(stim_mask, start=1.0, duration=2.0, amplitude=-80.0)

        # --- Wrapper construction ---
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=dx, D=D, dt=dt)
        sim_wrapper = lbm(mesh, device='cpu')

        # Run both for 3ms
        times_d, V_hist_d = sim_direct.run(t_end=3.0, save_every=3.0)

        V_wrapper = None
        for snap in sim_wrapper.snapshots(3.0, save_every=3.0):
            V_wrapper = snap.V.clone()

        V_direct = V_hist_d[-1]
        assert torch.allclose(V_wrapper, V_direct, atol=1e-10), \
            f"Max diff: {(V_wrapper - V_direct).abs().max()}"
