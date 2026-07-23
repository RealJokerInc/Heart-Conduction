"""Tests for the mid-run advanced features (solver-hardening Step 2).

Masked voltage clamp (clamp_voltage / add_clamp_protocol / release_clamp) and mid-run state
injection (set_voltage / set_state / get_state / state_names). The unclamped path stays on the
fast engine loop — test_integrity.py is the bit-identical golden guard for that.
"""

import warnings

import numpy as np
import pytest
import torch

from cardiac_core import monodomain, bidomain, lbm, Grid, ConductivityConfig


def _sim(engine=monodomain, nx=60, ny=20, dx=0.02):
    g = Grid(nx, ny, dx)
    cond = ConductivityConfig.isotropic(1.4)
    stim = {"region": lambda x, y: x < 3 * dx, "start_time": 1.0, "duration": 2.0, "amplitude": -52.0}
    return monodomain(g, "ttp06", cond, stim, dt=0.05) if engine is monodomain \
        else engine(g, "ttp06", cond, stim, dt=0.05)


# --- state injection --------------------------------------------------------

def test_state_names_match_columns():
    sim = _sim()
    names = sim.state_names
    assert len(names) == 18                          # TTP06 gates + concentrations, V excluded
    assert sim.get_state(names[0]).shape == (60, 20)
    assert "Cai" in names and "m" in names and "Nai" in names
    assert "V" not in names and "Vm" not in names    # voltage is separate


def test_get_state_resting_physiological():
    sim = _sim()
    cai = sim.get_state("Cai")
    assert cai.shape == (60, 20)
    assert 1e-5 < float(cai.mean()) < 1e-3   # resting Ca_i band


def test_set_voltage_roundtrip():
    sim = _sim()
    field = torch.full((60, 20), -85.0, dtype=torch.float64)
    field[:30, :] = 20.0
    sim.set_voltage(field)
    assert float(sim.Vm[:30].mean()) == pytest.approx(20.0, abs=1e-9)
    assert float(sim.Vm[30:].mean()) == pytest.approx(-85.0, abs=1e-9)


def test_set_state_roundtrip():
    sim = _sim()
    sim.set_state("Cai", torch.full((60, 20), 0.001, dtype=torch.float64))
    assert float(sim.get_state("Cai").mean()) == pytest.approx(0.001, abs=1e-9)


def test_set_state_bad_name_raises():
    sim = _sim()
    with pytest.raises(ValueError, match="unknown ionic state"):
        sim.set_state("not_a_state", torch.zeros((60, 20), dtype=torch.float64))


# --- voltage clamp ----------------------------------------------------------

def test_clamp_holds_region_every_frame():
    sim = _sim()
    mask = np.zeros((60, 20), bool)
    mask[:10, :] = True
    sim.clamp_voltage(mask, 10.0)
    r = sim.run(t_end=15.0, save_every=1.0)
    # clamped strip pinned at 10 mV in every saved frame
    assert torch.allclose(r.Vm[:, :10, :], torch.tensor(10.0, dtype=torch.float64), atol=1e-6)
    # far region is free to evolve (not pinned to 10)
    assert abs(float(r.Vm[-1, 50:, :].mean()) - 10.0) > 5.0


def test_clamp_window_start_end():
    sim = _sim()
    mask = np.zeros((60, 20), bool)
    mask[:10, :] = True
    sim.clamp_voltage(mask, 30.0, start_time=5.0, duration=4.0)
    r = sim.run(t_end=12.0, save_every=1.0)
    t = r.times
    # Check frames clearly inside the [5,9) window and clearly outside it (avoid the
    # exact-boundary frame, where a save-time ULP vs the clamp cutoff is inherently ambiguous).
    inside = (t > 5.5) & (t < 8.5)         # t = 6, 7, 8
    outside = (t < 4.5) | (t > 9.5)        # t <= 4  and  t >= 10
    assert torch.allclose(r.Vm[inside][:, :10, :], torch.tensor(30.0, dtype=torch.float64), atol=1e-6)
    assert not torch.allclose(r.Vm[outside][:, :10, :], torch.tensor(30.0, dtype=torch.float64), atol=1e-6)


def test_clamp_callable_timevarying():
    sim = _sim()
    mask = np.zeros((60, 20), bool)
    mask[:10, :] = True
    sim.clamp_voltage(mask, lambda t: -80.0 if t < 6.0 else 0.0)
    r = sim.run(t_end=10.0, save_every=1.0)
    lo = r.times < 6.0
    hi = r.times >= 6.0
    assert torch.allclose(r.Vm[lo][:, :10, :], torch.tensor(-80.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(r.Vm[hi][:, :10, :], torch.tensor(0.0, dtype=torch.float64), atol=1e-6)


def test_add_clamp_protocol():
    sim = _sim()
    mask = np.zeros((60, 20), bool)
    mask[:10, :] = True
    sim.add_clamp_protocol(mask, [(-80.0, 4.0), (-20.0, 4.0)], start_time=0.0)
    r = sim.run(t_end=8.0, save_every=1.0)
    # clearly inside each step (skip the t=4 boundary frame)
    early = (r.times > 0.5) & (r.times < 3.5)     # t = 1, 2, 3
    late = (r.times > 4.5) & (r.times < 7.5)       # t = 5, 6, 7
    assert torch.allclose(r.Vm[early][:, :10, :], torch.tensor(-80.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(r.Vm[late][:, :10, :], torch.tensor(-20.0, dtype=torch.float64), atol=1e-6)


def test_release_clamp_returns_to_fast_path():
    sim = _sim()
    mask = np.zeros((60, 20), bool)
    mask[:10, :] = True
    sim.clamp_voltage(mask, 10.0)
    assert sim._clamp_mask is not None
    sim.release_clamp()
    assert sim._clamp_mask is None


def test_clamp_result_matches_unclamped_when_released():
    # A clamp then release must reproduce the plain run bit-for-bit (fast path restored).
    a = _sim()
    b = _sim()
    b.clamp_voltage(np.ones((60, 20), bool), 0.0)
    b.release_clamp()
    ra = a.run(t_end=6.0, save_every=1.0)
    rb = b.run(t_end=6.0, save_every=1.0)
    assert torch.equal(ra.Vm, rb.Vm)


# --- LBM guards -------------------------------------------------------------

def test_lbm_clamp_supported_but_injection_raises():
    # Stim-object Phase 1.4: LBM gained a NATIVE additive voltage clamp (Σf→value, flux-preserving),
    # so clamp_voltage NO LONGER raises for LBM (see test_stim.py::TestClampHolds for the held-value
    # checks). Other stateful mid-run ops still raise — V is a lattice-population moment (Σf), not a
    # stored per-node field, so set_voltage/set_state can't write it back.
    sim = _sim(engine=lbm)
    sim.clamp_voltage(np.ones((60, 20), bool), 10.0)      # native LBM clamp — no longer raises
    assert sim._lbm_clamp is not None                      # registered on the wrapper (re-pushed on reset)
    with pytest.raises(NotImplementedError):
        sim.set_voltage(torch.zeros((60, 20), dtype=torch.float64))


# --- bidomain parity --------------------------------------------------------

def test_clamp_frame_cadence_matches_unclamped_bidomain():
    # Regression (audit R1, Lane B): _stepping_run must use the engine's OWN save-cadence
    # tolerances. Bidomain.run uses t_end-1e-12; the trigger is save_every == dt, where a
    # wrong tolerance emits one extra trailing frame past t_end (101 vs 100).
    g = Grid(24, 16, 0.02)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = {"region": lambda x, y: x < 0.06, "start_time": 1.0, "duration": 2.0, "amplitude": -52.0}
    ctrl = bidomain(g, "ttp06", cond, stim, dt=0.05).run(t_end=5.0, save_every=0.05)  # save_every == dt
    cl_sim = bidomain(g, "ttp06", cond, stim, dt=0.05)
    m = np.zeros((24, 16), bool); m[:6, :] = True
    cl_sim.clamp_voltage(m, 10.0)
    cl = cl_sim.run(t_end=5.0, save_every=0.05)
    assert cl.Vm.shape[0] == ctrl.Vm.shape[0]          # same frame count
    assert torch.allclose(cl.times, ctrl.times)         # same frame times
    assert float(cl.times[-1]) <= 5.0 + 1e-9            # no overshoot past t_end


def test_clamp_works_on_bidomain():
    g = Grid(40, 16, 0.02)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = {"region": lambda x, y: x < 0.06, "start_time": 1.0, "duration": 2.0, "amplitude": -52.0}
    sim = bidomain(g, "ttp06", cond, stim, dt=0.05)
    mask = np.zeros((40, 16), bool)
    mask[:8, :] = True
    sim.clamp_voltage(mask, 15.0)
    r = sim.run(t_end=10.0, save_every=1.0)
    assert torch.allclose(r.Vm[:, :8, :], torch.tensor(15.0, dtype=torch.float64), atol=1e-6)
