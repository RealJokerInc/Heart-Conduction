"""Tests for cardiac_core.run — one-shot simulation functions."""

import sys
from pathlib import Path

import torch
import pytest

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "Monodomain" / "Engine_V5.4"))
sys.path.insert(0, str(_project_root / "Bidomain" / "Engine_V1"))
sys.path.insert(0, str(_project_root / "LBM" / "Engine_V1"))

from cardiac_core import (
    create_cardiac_mesh, run_monodomain, run_bidomain, run_lbm, simulate,
)


class TestRunMonodomain:
    def test_returns_tensors(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        times, V = run_monodomain(mesh, t_end=5.0, save_every=1.0)

        assert isinstance(times, torch.Tensor)
        assert isinstance(V, torch.Tensor)
        assert times.ndim == 1
        assert V.ndim == 3  # (n_saves, Nx, Ny)
        assert V.shape[0] == times.shape[0]

    def test_correct_grid_shape(self):
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05)
        Nx = round(1.0 / 0.05) + 1
        Ny = round(0.5 / 0.05) + 1

        times, V = run_monodomain(mesh, t_end=3.0, save_every=1.0)
        assert V.shape[1] == Nx
        assert V.shape[2] == Ny

    def test_stimulus_activates(self):
        mesh = create_cardiac_mesh(
            Lx=0.5, Ly=0.5, dx=0.05,
            stim_amplitude=-80.0, stim_start=0.0, stim_duration=2.0,
        )
        times, V = run_monodomain(mesh, t_end=5.0, save_every=1.0)
        assert V[-1].max() > -50.0

    def test_output_device_cpu(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        times, V = run_monodomain(mesh, t_end=3.0, output_device='cpu')
        assert V.device.type == 'cpu'
        assert times.device.type == 'cpu'


class TestRunBidomain:
    def test_returns_three_tensors(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        times, V, phi_e = run_bidomain(mesh, t_end=3.0, save_every=1.0)

        assert isinstance(phi_e, torch.Tensor)
        assert phi_e.shape == V.shape
        assert times.shape[0] == V.shape[0]


class TestRunLBM:
    def test_returns_tensors(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.025, dt=0.005)
        times, V = run_lbm(mesh, t_end=3.0, save_every=1.0)

        assert isinstance(V, torch.Tensor)
        assert V.ndim == 3
        assert times.shape[0] == V.shape[0]


class TestSimulate:
    def test_monodomain(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        result = simulate(mesh, t_end=3.0, engine='monodomain')

        assert result.V.ndim == 3
        assert result.phi_e is None

    def test_bidomain(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        result = simulate(mesh, t_end=3.0, engine='bidomain')

        assert result.phi_e is not None
        assert result.phi_e.shape == result.V.shape

    def test_lbm(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.025)
        result = simulate(mesh, t_end=3.0, engine='lbm', dt=0.005)

        assert result.V.ndim == 3
        assert result.phi_e is None

    def test_invalid_engine(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        with pytest.raises(ValueError, match="Unknown engine"):
            simulate(mesh, t_end=1.0, engine='invalid')

    def test_matches_run_function(self):
        """simulate(engine='monodomain') == run_monodomain()."""
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)

        times_r, V_r = run_monodomain(mesh, t_end=3.0, save_every=1.0)
        result = simulate(mesh, t_end=3.0, save_every=1.0, engine='monodomain')

        assert torch.allclose(times_r, result.times)
        assert torch.allclose(V_r, result.V)

    def test_vm_is_canonical(self):
        """Vm is the canonical field; .V is a read-only alias (same tensor); dx/dy populated."""
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        result = simulate(mesh, t_end=3.0, engine='monodomain')

        # Vm is the real field; V aliases it (identity, not a copy).
        assert result.Vm is result.V
        assert result.Vm.ndim == 3
        # Run helpers now populate spacing for the analysis hooks.
        assert result.dx == 0.05 and result.dy == 0.05
        assert result.ionic_states is None

    def test_snapshot_vm_alias(self):
        """SimulationSnapshot.Vm is canonical; .V aliases it."""
        from cardiac_core import monodomain
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        sim = monodomain(mesh, device='cpu')
        snap = next(sim.snapshots(2.0, save_every=1.0))
        assert snap.Vm is snap.V
        assert snap.Vm.shape == (sim.Nx, sim.Ny)
