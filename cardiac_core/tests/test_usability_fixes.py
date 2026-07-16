"""Regression tests for the P0/P1 usability-audit fixes (engine_consolidation PLAN 2026-07-16).

One (or two) focused test(s) per bug: B1 GPU device-mismatch, B3/B4 apd_at peak/notch,
B5 Grid(N,1), B6 forward_euler CFL guard, B7 record= validation, B8 NaN-fill masked nodes.
These lock in the fix; test_integrity.py (full-rectangle) is the bit-identical golden guard.
"""

import warnings

import numpy as np
import pytest
import torch

from cardiac_core import monodomain, bidomain, Grid, ConductivityConfig, create_cardiac_mesh
from cardiac_core.analysis import apd_at
from cardiac_core.mesh.structured import StructuredGrid


def _stim(width=0.05, amplitude=-52.0, start=1.0, duration=2.0):
    return {'region': (lambda x, y: x < width), 'start_time': start,
            'duration': duration, 'amplitude': amplitude}


# ===========================================================================
# B1 — GPU device-mismatch in _result_from (declarative .run() path)
# ===========================================================================
def test_result_times_device_matches_vm():
    """CPU smoke: run() must return times and Vm on the same device."""
    sim = monodomain(create_cardiac_mesh(0.2, 0.1, 0.02, D=1e-3, chi=1.0), stimulus=_stim())
    r = sim.run(3.0, 1.0)
    assert r.times.device == r.Vm.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_result_analysis_runs_on_cuda():
    """On cuda the snapshots carry Vm on GPU; times must too, else lat()/cv() raise."""
    sim = monodomain(create_cardiac_mesh(0.4, 0.1, 0.02, D=1e-3, chi=1.0),
                     stimulus=_stim(), device='cuda')
    r = sim.run(4.0, 1.0)
    assert r.times.device == r.Vm.device
    # These index times by a (GPU) argmax index — the pre-fix CPU times crashed here.
    lat = r.lat()
    _ = r.cv(2, 15, 2)
    assert lat.shape == (r.Vm.shape[1], r.Vm.shape[2])


# ===========================================================================
# B8 — NaN-fill out-of-domain (masked) nodes
# ===========================================================================
def test_flat_to_grid_nan_fills_masked():
    """Direct contract: masked-out nodes come back NaN, active nodes finite."""
    mask = torch.ones(9, 9, dtype=torch.bool)
    mask[3:6, 3:6] = False
    sg = StructuredGrid.from_mask(mask, 0.02, 0.02)
    flat = torch.arange(sg.n_dof, dtype=torch.float64) + 1.0  # all finite, nonzero
    grid = sg.flat_to_grid(flat)
    assert torch.isnan(grid[~mask]).all()
    assert torch.isfinite(grid[mask]).all()


def test_masked_nodes_are_nan_mono():
    mask = np.ones((21, 21), dtype=bool)
    mask[8:13, 8:13] = False   # central hole
    sim = monodomain(create_cardiac_mesh(0.4, 0.4, 0.02, D=1e-3, chi=1.0, mask=mask),
                     stimulus=_stim())
    r = sim.run(4.0, 1.0)
    hole = torch.tensor(~mask)
    assert torch.isnan(r.Vm[:, hole]).all(), "masked nodes must be NaN, not 0.0 mV"
    assert torch.isfinite(r.Vm[:, torch.tensor(mask)]).all(), "active tissue must stay finite"
    # lat/apd auto-correct: masked hole is not counted as activated
    assert torch.isnan(r.lat()[hole]).all()


def test_masked_nodes_are_nan_bidomain():
    mask = np.ones((15, 15), dtype=bool)
    mask[6:9, 6:9] = False
    # The default 'auto' elliptic solver picks the spectral path, which reshapes to the
    # full rectangle and can't take a masked (irregular) domain; PCG handles the hole.
    sim = bidomain(create_cardiac_mesh(0.28, 0.28, 0.02, D=1e-3, chi=1.0, mask=mask),
                   stimulus=_stim(), elliptic_solver='pcg')
    r = sim.run(4.0, 1.0)
    hole = torch.tensor(~mask)
    assert torch.isnan(r.Vm[:, hole]).all(), "masked Vm must be NaN"
    assert r.phi_e is not None
    assert torch.isnan(r.phi_e[:, hole]).all(), "masked phi_e must be NaN"


def test_full_rectangle_has_no_nan():
    """Golden-shape guard: an unmasked run must contain no NaN anywhere."""
    sim = monodomain(create_cardiac_mesh(0.2, 0.2, 0.02, D=1e-3, chi=1.0), stimulus=_stim())
    r = sim.run(4.0, 1.0)
    assert torch.isfinite(r.Vm).all()


# ===========================================================================
# B3/B4 — apd_at peak-over-remaining + spike-and-dome notch
# ===========================================================================
def _const_trace(values):
    """Build a (T,1,1) trace tensor + times from a 1-D value list."""
    v = torch.tensor(values, dtype=torch.float64).reshape(-1, 1, 1)
    t = torch.arange(v.shape[0], dtype=torch.float64)
    return v, t


def test_apd_multibeat_not_corrupted():
    """A later, taller beat must not inflate an earlier beat's APD (B3)."""
    rest = -85.0
    trace = [rest] * 10                       # diastole
    trace += [20.0] * 45                       # beat 1 plateau (peak +20)
    trace += list(np.linspace(20.0, rest, 6))  # beat-1 repolarization
    trace += [rest] * (160 - len(trace))       # long diastole to idx 160
    trace += [40.0] * 45                        # beat 2 plateau — TALLER (+40)
    trace += list(np.linspace(40.0, rest, 6))   # beat-2 repolarization
    V, times = _const_trace(trace)
    apd1 = apd_at(V, times, 0, 0, repol=0.9)
    # Beat-1 APD is ~ (its own repol time − its activation). The old code maxed over the
    # whole tail (beat 2, +40) and searched beat 2's repolarization → ~195 ms (corrupted).
    assert 30.0 < apd1 < 80.0, f"beat-1 APD corrupted by beat 2: {apd1}"


def test_apd_notch_dome_aware():
    """APD30 on a spike-and-dome must land on the final repolarization, not the notch (B4)."""
    rest = -85.0
    trace = [rest] * 10        # diastole
    trace += [40.0]            # idx 10: spike peak +40
    trace += [20.0, 0.0, 0.0]  # idx 11-13: notch dips below V_repol30 (= +2.5)
    trace += [10.0] * 147      # idx 14-160: dome plateau (+10, above +2.5)
    trace += list(np.linspace(10.0, rest, 40))  # final repolarization
    V, times = _const_trace(trace)

    apd30_dome = apd_at(V, times, 0, 0, repol=0.3)             # default dome_aware=True
    apd30_first = apd_at(V, times, 0, 0, repol=0.3, dome_aware=False)
    assert apd30_dome > 100.0, f"dome-aware APD30 should reach the dome repol: {apd30_dome}"
    assert apd30_first < 30.0, f"first-crossing fallback should land on the notch: {apd30_first}"


def test_apd_single_clean_ap_unchanged():
    """A monotonic single AP: dome-aware == first-crossing (no regression)."""
    rest = -85.0
    trace = [rest] * 5 + [20.0] * 30 + list(np.linspace(20.0, rest, 30)) + [rest] * 10
    V, times = _const_trace(trace)
    a_dome = apd_at(V, times, 0, 0, repol=0.9)
    a_first = apd_at(V, times, 0, 0, repol=0.9, dome_aware=False)
    assert a_dome == pytest.approx(a_first)   # dome-aware must not change a monotonic AP
    assert 45.0 < a_dome < 70.0               # 30-pt plateau + ~90% of a 30-pt repol


# ===========================================================================
# B5 — Grid(N, 1) 1-D cable (ZeroDivisionError guard)
# ===========================================================================
def test_grid_1d_cable_constructs():
    g = Grid(101, 1, 0.02)      # would ZeroDivisionError at dy = Ly/(Ny-1)
    assert g.Nx == 101 and g.Ny == 1
    sg = g._structured_grid()   # forces StructuredGrid.__post_init__
    assert sg.dx == pytest.approx(0.02)
    assert sg.dy == pytest.approx(0.02)  # degenerate axis inherits dx


def test_structuredgrid_single_column():
    sg = StructuredGrid.create_rectangle(0.0, 2.0, 1, 101)  # Nx==1 cable along y
    assert sg.dy == pytest.approx(0.02)
    assert sg.dx == pytest.approx(0.02)


def test_monodomain_1d_cable_runs():
    g = Grid(101, 1, 0.02)
    cond = ConductivityConfig.isotropic(1.4, chi=1.0)
    sim = monodomain(g, 'ttp06', cond, _stim(width=0.06))
    r = sim.run(4.0, 1.0)
    assert r.Vm.shape == (r.Vm.shape[0], 101, 1)


# ===========================================================================
# B7 — validate record= keys
# ===========================================================================
def test_record_rejects_unknown():
    sim = monodomain(create_cardiac_mesh(0.2, 0.1, 0.02, D=1e-3, chi=1.0), stimulus=_stim())
    with pytest.raises(ValueError, match="record key"):
        sim.run(2.0, 1.0, record=("Vm", "I_Kr"))


def test_record_known_keys_ok():
    sim = monodomain(create_cardiac_mesh(0.2, 0.1, 0.02, D=1e-3, chi=1.0), stimulus=_stim())
    sim.run(2.0, 1.0, record=("Vm",))
    sim.run(2.0, 1.0, record=("Vm", "ionic_states"))


# ===========================================================================
# B6 — forward_euler CFL stability guard
# ===========================================================================
def test_forward_euler_stability_warns():
    """dt above chi*Cm*min(dx,dy)^2/(4*D_max) must warn; below must not."""
    mesh = create_cardiac_mesh(0.4, 0.4, 0.02, D=1e-3, chi=1.0)
    # dt_max = 1*1*0.02^2/(4*1e-3) = 0.1 ms.
    sim = monodomain(mesh, stimulus=_stim(), diffusion_solver='forward_euler', dt=0.5)
    with pytest.warns(UserWarning, match="stability limit"):
        sim.run(1.0, 1.0)

    # dt below the limit → no CFL warning.
    sim_ok = monodomain(mesh, stimulus=_stim(), diffusion_solver='forward_euler', dt=0.02)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim_ok.run(1.0, 1.0)
    assert not any("stability limit" in str(w.message) for w in caught)
