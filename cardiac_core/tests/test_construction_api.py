"""Tests for the declarative construction API (Phase 4): factories from
(Grid, ionic_model, ConductivityConfig, stimulus) + reset/with_/stimulate + the Simulation Protocol."""

import torch

from cardiac_core import (
    monodomain, bidomain, lbm,
    Grid, ConductivityConfig, Simulation, create_cardiac_mesh, Stim,
)
from cardiac_core import analysis


def _left_edge_stim(g, width=0.05, amplitude=-52.0, start=1.0, duration=2.0):
    return Stim.from_region(g, (lambda x, y: x < width), start_time=start,
                            duration=duration, amplitude=amplitude)


def _collect_vt(sim, t_end, save_every=0.5):
    times, V = [], []
    for snap in sim.snapshots(t_end, save_every=save_every):
        times.append(snap.t)
        V.append(snap.Vm)
    return torch.tensor(times, dtype=torch.float64), torch.stack(V)


class TestDeclarativeConstruction:
    def test_monodomain_from_grid(self):
        g = Grid(40, 20, 0.05)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        sim = monodomain(g, 'ttp06', cond, _left_edge_stim(g))
        assert sim.Nx == 40 and sim.Ny == 20
        assert sim.dx == 0.05
        snap = next(sim.snapshots(2.0, save_every=1.0))
        assert snap.Vm.shape == (40, 20)

    def test_legacy_mesh_path(self):
        # Positional CardiacMeshData auto-detected as mesh=.
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05)
        sim = monodomain(mesh, device='cpu')
        snaps = list(sim.snapshots(3.0, save_every=1.0))
        assert len(snaps) >= 2

    def test_bidomain_from_grid(self):
        g = Grid(30, 30, 0.05)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        sim = bidomain(g, 'ttp06', cond, _left_edge_stim(g))
        assert sim.engine_type == 'bidomain'
        snap = next(sim.snapshots(2.0, save_every=1.0))
        assert snap.Vm.shape == (30, 30)
        assert snap.phi_e is not None

    def test_lbm_from_grid(self):
        g = Grid(40, 20, 0.025)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        sim = lbm(g, 'ttp06', cond, _left_edge_stim(g), dt=0.005)
        assert sim.engine_type == 'lbm'
        snap = next(sim.snapshots(1.0, save_every=0.5))
        assert snap.Vm.shape == (40, 20)


class TestConductivityMapping:
    def test_monodomain_cv_via_config(self):
        """CV through the assembled CardiacMeshData lands in a physiological band.

        Mis-scaling (e.g. an extra /chi) would put CV ~1000x off (~0.05 cm/s).
        """
        g = Grid(200, 50, 0.01)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        sim = monodomain(g, 'ttp06', cond, _left_edge_stim(g, width=0.05))
        # Front ~50 cm/s = 0.05 cm/ms; x2=100 (1.0 cm) activates ~21 ms → t_end=40 ms for margin.
        times, V = _collect_vt(sim, t_end=40.0, save_every=0.5)
        cv = analysis.conduction_velocity(V, times, dx=g.dx, x1=20, x2=100, y=25, threshold=-20.0)
        assert 10.0 < cv < 100.0, f"CV={cv} cm/s out of physiological band (mis-scaling?)"


class TestSimulationSurface:
    def test_introspection(self):
        g = Grid(20, 20, 0.05)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        sim = monodomain(g, 'ttp06', cond, _left_edge_stim(g))
        assert sim.dt == 0.02       # default
        assert sim.Cm == 1.0
        assert sim.ionic_model == 'ttp06'

    def test_with_is_functional(self):
        g = Grid(20, 20, 0.05)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        s = monodomain(g, 'ttp06', cond, _left_edge_stim(g))
        s2 = s.with_(dt=0.01)
        assert s2.dt == 0.01
        assert s.dt == 0.02         # original untouched
        assert s is not s2

    def test_reset(self):
        g = Grid(20, 20, 0.05)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        sim = monodomain(g, 'ttp06', cond, _left_edge_stim(g))
        list(sim.snapshots(3.0, save_every=1.0))
        assert sim.t > 0.0
        sim.reset()
        assert sim.t == 0.0
        assert sim.Vm.max().item() < -50.0   # back to rest (TTP06 ~ -86 mV)

    def test_stimulate_adds(self):
        # Build WITHOUT a stimulus, then add one via stimulate().
        g = Grid(40, 20, 0.05)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        sim = monodomain(g, 'ttp06', cond, stimulus=None)
        sim.stimulate(region=(lambda x, y: x < 0.05), start_time=0.0, duration=2.0, amplitude=-52.0)
        for snap in sim.snapshots(5.0, save_every=1.0):
            pass
        assert snap.Vm.max().item() > -50.0, "stimulate() did not activate tissue"

    def test_stimulate_mesh_path(self):
        # Legacy mesh-built sim: stimulate() appends to data.stimuli and reset/run works.
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05)
        sim = monodomain(mesh, device='cpu')
        n_before = len(sim._data.stimuli)
        sim.stimulate(region=(lambda x, y: x > 0.9), start_time=0.0, duration=2.0, amplitude=-52.0)
        assert len(sim._data.stimuli) == n_before + 1
        snaps = list(sim.snapshots(3.0, save_every=1.0))
        assert len(snaps) >= 2

    def test_protocol_isinstance(self):
        g = Grid(20, 20, 0.05)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        sim = monodomain(g, 'ttp06', cond, _left_edge_stim(g))
        assert isinstance(sim, Simulation)
