"""Integration tests — same mesh → all 3 engines → correct snapshots."""

import numpy as np
import pytest
import torch

from cardiac_core import (
    monodomain, bidomain, lbm,
    create_cardiac_mesh, save_cardiac_mesh,
    SimulationSnapshot,
)


class TestAllEnginesSameMesh:
    """Same mesh file → all 3 engines run and produce valid output."""

    @pytest.fixture
    def mesh(self):
        return create_cardiac_mesh(
            Lx=0.5, Ly=0.5, dx=0.05, D=0.001,
            stim_amplitude=-80.0,
            stim_start=0.0,
            stim_duration=2.0,
        )

    def test_monodomain(self, mesh):
        sim = monodomain(mesh, device='cpu')
        snaps = list(sim.snapshots(5.0, save_every=1.0))
        assert len(snaps) >= 4
        assert isinstance(snaps[0], SimulationSnapshot)
        assert snaps[-1].V.max() > -50.0

    def test_bidomain(self, mesh):
        sim = bidomain(mesh, device='cpu')
        snaps = list(sim.snapshots(5.0, save_every=1.0))
        assert len(snaps) >= 4
        assert isinstance(snaps[0], SimulationSnapshot)
        assert snaps[0].phi_e is not None
        assert snaps[-1].V.max() > -50.0

    def test_lbm(self, mesh):
        sim = lbm(mesh, dt=0.005, device='cpu')
        snaps = list(sim.snapshots(5.0, save_every=1.0))
        assert len(snaps) >= 4
        assert isinstance(snaps[0], SimulationSnapshot)
        assert snaps[-1].V.max() > -50.0

    def test_all_same_grid_shape(self, mesh):
        """All engines return same (Nx, Ny) grid shape."""
        Nx = round(0.5 / 0.05) + 1  # 11
        Ny = Nx

        sim_m = monodomain(mesh, device='cpu')
        sim_b = bidomain(mesh, device='cpu')
        sim_l = lbm(mesh, dt=0.005, device='cpu')

        for snap in sim_m.snapshots(2.0, save_every=2.0):
            assert snap.V.shape == (Nx, Ny)
        for snap in sim_b.snapshots(2.0, save_every=2.0):
            assert snap.V.shape == (Nx, Ny)
            assert snap.phi_e.shape == (Nx, Ny)
        for snap in sim_l.snapshots(2.0, save_every=2.0):
            assert snap.V.shape == (Nx, Ny)


class TestFromFileSameMesh:
    """Same .npz file → all 3 engines."""

    def test_all_from_file(self, tmp_path):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05, D=0.001)
        path = str(tmp_path / "shared.npz")
        save_cardiac_mesh(path, mesh)

        # All 3 from same file
        sim_m = monodomain(path, device='cpu')
        sim_b = bidomain(path, device='cpu')
        sim_l = lbm(path, dt=0.005, device='cpu')

        assert list(sim_m.snapshots(2.0, save_every=2.0))
        assert list(sim_b.snapshots(2.0, save_every=2.0))
        assert list(sim_l.snapshots(2.0, save_every=2.0))
