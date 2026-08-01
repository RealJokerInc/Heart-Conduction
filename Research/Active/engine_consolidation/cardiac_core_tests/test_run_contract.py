"""The unified run() eager/batch contract + SimulationResult analysis hooks."""

import pytest
import torch

from cardiac_core import monodomain, lbm, Grid, ConductivityConfig, Stim
from cardiac_core import analysis
from cardiac_core.run import SimulationResult

COND = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)


def _stim(g):
    return Stim.from_region(g, (lambda x, y: x < 0.05),
                            start_time=1.0, duration=2.0, amplitude=-52.0)


class TestRunContract:
    def test_eager_returns_result(self):
        g = Grid(20, 20, 0.05)
        sim = monodomain(g, 'ttp06', COND, _stim(g))
        r = sim.run(t_end=10.0, save_every=1.0)
        assert isinstance(r, SimulationResult)
        assert r.Vm.ndim == 3
        assert r.Vm.shape[1:] == (20, 20)
        assert r.Vm.shape[0] == r.times.shape[0]
        assert r.dx == 0.05 and r.dy == 0.05

    def test_batch_streams(self):
        g = Grid(20, 20, 0.05)
        sim = monodomain(g, 'ttp06', COND, _stim(g))
        chunks = list(sim.run(t_end=10.0, save_every=1.0, batch=3))
        assert len(chunks) > 1
        assert all(isinstance(c, SimulationResult) for c in chunks)
        assert all(c.Vm.shape[0] <= 3 for c in chunks)
        # Concatenated chunks equal the eager result.
        eager = monodomain(g, 'ttp06', COND, _stim(g)).run(t_end=10.0, save_every=1.0)
        cat = torch.cat([c.Vm for c in chunks], dim=0)
        assert cat.shape == eager.Vm.shape
        torch.testing.assert_close(cat, eager.Vm)

    def test_default_record_no_ionic(self):
        g = Grid(15, 15, 0.05)
        sim = monodomain(g, 'ttp06', COND, _stim(g))
        r = sim.run(t_end=3.0, save_every=1.0)
        assert r.ionic_states is None

    def test_record_ionic_states_monodomain(self):
        g = Grid(15, 15, 0.05)
        sim = monodomain(g, 'ttp06', COND, _stim(g))
        r = sim.run(t_end=5.0, save_every=1.0, record=("Vm", "ionic_states"))
        assert r.ionic_states is not None
        T = r.times.shape[0]
        # (T, n_states, Nx, Ny) — a REAL tensor, not merely 'is not None'.
        assert r.ionic_states.ndim == 4
        assert r.ionic_states.shape[0] == T
        assert r.ionic_states.shape[2:] == (15, 15)
        assert r.ionic_states.shape[1] >= 14  # TTP06 has ~18 states

    def test_record_ionic_states_lbm_not_implemented(self):
        g = Grid(20, 20, 0.025)
        sim = lbm(g, 'ttp06', COND, _stim(g), dt=0.005)
        with pytest.raises(NotImplementedError):
            sim.run(t_end=1.0, save_every=0.5, record=("Vm", "ionic_states"))

    def test_snapshots_still_generates(self):
        # The back-compat generator alias still yields SimulationSnapshot frames.
        g = Grid(15, 15, 0.05)
        sim = monodomain(g, 'ttp06', COND, _stim(g))
        snaps = list(sim.snapshots(3.0, save_every=1.0))
        assert len(snaps) >= 2
        assert snaps[0].Vm.shape == (15, 15)


@pytest.fixture(scope="module")
def propagating_result():
    """One propagating monodomain run (front ~50 cm/s) reused by the hook tests."""
    g = Grid(150, 20, 0.01)
    sim = monodomain(g, 'ttp06', COND, _stim(g))
    return sim.run(t_end=30.0, save_every=0.5)


class TestResultHooks:
    def test_result_cv_hook(self, propagating_result):
        r = propagating_result
        cv_hook = r.cv(x1=20, x2=80, y=10)
        cv_direct = analysis.conduction_velocity(r.Vm, r.times, r.dx, 20, 80, 10)
        assert cv_hook == cv_direct           # thin delegator — identical
        assert 10.0 < cv_hook < 100.0         # physiological (and not nan)

    def test_result_apd_hook(self, propagating_result):
        apd = propagating_result.apd()
        assert apd.shape == (150, 20)

    def test_result_lat_hook(self, propagating_result):
        lat = propagating_result.lat()
        assert lat.shape == (150, 20)
