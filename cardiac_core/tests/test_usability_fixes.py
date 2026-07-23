"""Regression tests for the P0/P1 usability-audit fixes (engine_consolidation PLAN 2026-07-16).

One (or two) focused test(s) per bug: B1 GPU device-mismatch, B3/B4 apd_at peak/notch,
B5 Grid(N,1), B6 forward_euler CFL guard, B7 record= validation, B8 NaN-fill masked nodes.
These lock in the fix; test_integrity.py (full-rectangle) is the bit-identical golden guard.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiac_core import monodomain, bidomain, lbm, Grid, ConductivityConfig, create_cardiac_mesh, Stim
from cardiac_core import analysis
from cardiac_core.analysis import apd_at
from cardiac_core.mesh.structured import StructuredGrid


def _stim(width=0.05, amplitude=-52.0, start=1.0, duration=2.0):
    # Dict form — retained for the legacy mesh= factory path (monodomain(mesh, stimulus=...)),
    # which drops stimulus= and never emits the dict-deprecation warning.
    return {'region': (lambda x, y: x < width), 'start_time': start,
            'duration': duration, 'amplitude': amplitude}


def _gstim(g, width=0.05, amplitude=-52.0, start=1.0, duration=2.0):
    # Stim form for the declarative Grid factory path (monodomain(grid, ...)); resolved eagerly
    # against the grid, byte-identical to the dict once normalized.
    return Stim.from_region(g, (lambda x, y: x < width), start_time=start,
                            duration=duration, amplitude=amplitude)


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
    sim = monodomain(g, 'ttp06', cond, _gstim(g, width=0.06))
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


# ===========================================================================
# B2 — fast spectral solver path (DCT/FFT) works through the factory
# ===========================================================================
def test_dct_solver_runs_and_matches_pcg():
    """linear_solver='dct' is EXACT vs CN on the isotropic-uniform full-rect Neumann
    default — compare the Vm FIELD (CV is frame-quantized and would hide a wrong solve)."""
    g = Grid(60, 6, 0.02)
    cond = ConductivityConfig.isotropic(1.4)  # eff D = 1.4/1400 = 1e-3 cm^2/ms
    stim = _gstim(g, width=0.06)
    r_pcg = monodomain(g, 'ttp06', cond, stim, linear_solver='pcg').run(40.0, 0.5)
    r_dct = monodomain(g, 'ttp06', cond, stim, linear_solver='dct').run(40.0, 0.5)
    assert torch.isfinite(r_dct.Vm).all()
    assert torch.max(torch.abs(r_dct.Vm - r_pcg.Vm)).item() < 0.5   # mV; DCT≈CN to ~1e-3
    assert r_dct.cv(10, 45, 3) == pytest.approx(r_pcg.cv(10, 45, 3), rel=0.05)


def test_spectral_solvers_reject_unsupported_configs():
    """DCT/FFT invert an idealized eigen-operator; the factory must REJECT (not silently
    mis-solve) masked / anisotropic / bdf2 configs, and reject fft (Neumann meshes)."""
    cond = ConductivityConfig.isotropic(1.4)
    # fft is never valid via the mono factory (it builds Neumann meshes)
    g_fft = Grid(32, 8, 0.02)
    with pytest.raises(ValueError, match="fft"):
        monodomain(g_fft, 'ttp06', cond, _gstim(g_fft, width=0.06), linear_solver='fft')
    # dct on a masked domain
    mask = np.ones((20, 12), dtype=bool); mask[8:11, 5:8] = False
    with pytest.raises(ValueError, match="masked|silently wrong"):
        monodomain(create_cardiac_mesh(0.38, 0.22, 0.02, D=1e-3, chi=1.0, mask=mask),
                   linear_solver='dct')
    # dct on anisotropic D
    aniso = ConductivityConfig.anisotropic(2.0, 0.5, 0.0)
    g_aniso = Grid(30, 20, 0.02)
    with pytest.raises(ValueError, match="anisotropic|silently wrong"):
        monodomain(g_aniso, 'ttp06', aniso, _gstim(g_aniso, width=0.06), linear_solver='dct')
    # dct with bdf2 (its bootstrap step switches operators)
    g_bdf2 = Grid(30, 8, 0.02)
    with pytest.raises(ValueError, match="scheme|silently wrong"):
        monodomain(g_bdf2, 'ttp06', cond, _gstim(g_bdf2, width=0.06),
                   linear_solver='dct', diffusion_solver='bdf2')


# ===========================================================================
# Phase 3 — stub de-trap + scale_conductance / set_conductivity implementations
# ===========================================================================
def test_stubs_have_informative_errors():
    """Still-unimplemented stubs raise NotImplementedError with a message naming the route.

    NOTE: get_state / set_state / set_voltage / clamp_voltage / add_clamp_protocol /
    release_clamp were IMPLEMENTED on the solver-hardening branch (mid-run voltage clamp +
    state injection) — they are tested in test_advanced_features.py and no longer stubs.
    The methods below remain unimplemented.
    """
    sim = monodomain(create_cardiac_mesh(0.2, 0.12, 0.02, D=1e-3, chi=1.0), stimulus=_stim())
    for call in (lambda: sim.compute_cv(0.0, 0.1, 0.05),
                 lambda: sim.set_parameter('GNa', 1.0),
                 lambda: sim.add_probe('apex', 0.05, 0.05)):
        with pytest.raises(NotImplementedError) as ei:
            call()
        msg = str(ei.value).lower()
        assert len(msg) > 20, "stub error must be informative, not bare"
        assert ("not implemented" in msg) or ("not a sim method" in msg), msg


def test_scale_conductance_changes_apd():
    """Reducing GKr (IKr block) prolongs APD; an unknown conductance raises."""
    g = Grid(12, 4, 0.02)
    cond = ConductivityConfig.isotropic(1.4)
    stim = _gstim(g, width=0.06)
    r0 = monodomain(g, 'ttp06', cond, stim, dt=0.1).run(600.0, 2.0)
    sim = monodomain(g, 'ttp06', cond, stim, dt=0.1)
    sim.scale_conductance('GKr', 0.4)
    r1 = sim.run(600.0, 2.0)
    apd0, apd1 = float(r0.apd()[2, 2]), float(r1.apd()[2, 2])
    assert apd0 == apd0 and apd1 == apd1, "both APDs must be finite"
    assert apd1 > apd0 + 5.0, f"reduced GKr should prolong APD: {apd0} -> {apd1}"
    with pytest.raises(ValueError, match="not a scalable conductance"):
        sim.scale_conductance('NotAConductance', 0.5)
    with pytest.raises(ValueError, match="not a scalable conductance"):
        sim.scale_conductance('Nao', 0.5)   # a concentration, not a conductance — must reject


def test_set_conductivity_scar_blocks():
    """set_conductivity(mask, D=0) makes an inexcitable scar; the wave routes around."""
    g = Grid(24, 20, 0.02)
    cond = ConductivityConfig.isotropic(1.4)
    sim = monodomain(g, 'ttp06', cond, _gstim(g, width=0.06), dt=0.1)
    scar = np.zeros((24, 20), dtype=bool)
    scar[11:14, 6:16] = True   # vertical barrier (open lanes at rows 0-5 and 16-19)
    sim.set_conductivity(scar, D=0.0)
    r = sim.run(220.0, 4.0)
    scar_t = torch.as_tensor(scar)
    assert not (r.Vm[:, scar_t] >= -20.0).any(), "scar nodes must never activate"
    lat = r.lat()
    # A node in the scar's SHADOW (just past it, mid-band) can only be reached by routing
    # around the barrier — and must therefore activate LATER than the open-lane node at the
    # same x. If set_conductivity no-op'd, the shadow node would activate straight-through.
    lat_shadow = lat[15, 10].item()   # behind the scar (col 15 > scar col 13, row 10 in-band)
    lat_open = lat[15, 2].item()      # same x, open lane (row 2)
    assert lat_shadow == lat_shadow, "shadow node should eventually activate via routing"
    assert lat_open == lat_open and lat_shadow > lat_open + 2.0, \
        "shadow node must activate later than the open lane (detour around the scar)"


# --- audit follow-ups: the conductivity/conductance methods must also work on the
#     bidomain sigma path and must not flip cell type (found by the Phase-3 audit) ---
def test_set_conductivity_scar_blocks_declarative_bidomain():
    """A declarative bidomain stores conductivity as sigma fields (not D_xx); a scar
    must zero those too (else it silently no-ops and the wave passes through)."""
    g = Grid(24, 16, 0.02)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    sim = bidomain(g, 'ttp06', cond, _gstim(g, width=0.06), dt=0.1)
    scar = np.zeros((24, 16), dtype=bool)
    scar[11:14, 5:12] = True
    sim.set_conductivity(scar, D=0.0)
    assert float(np.unique(sim._data.sigma_i[0][scar])[0]) == 0.0, "sigma must be zeroed"
    r = sim.run(140.0, 4.0)
    assert not (r.Vm[:, torch.as_tensor(scar)] >= -20.0).any(), "scar must block on bidomain"


def test_set_conductivity_nonzero_D_on_sigma_bidomain_raises():
    """An absolute nonzero D has no unambiguous sigma meaning — raise, don't silently
    apply it to the ignored D_xx field."""
    g = Grid(16, 12, 0.02)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    sim = bidomain(g, 'ttp06', cond, _gstim(g), dt=0.1)
    scar = np.zeros((16, 12), dtype=bool); scar[7:9, 4:8] = True
    with pytest.raises(NotImplementedError, match="bidomain"):
        sim.set_conductivity(scar, D=5e-4)


def test_scale_conductance_preserves_celltype_on_bidomain():
    """scale_conductance must scale only the target conductance, not flip cell type —
    the bidomain factory builds ENDO, so a name-rebuild would jump Gto 0.073->0.294."""
    g = Grid(16, 12, 0.02)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    sim = bidomain(g, 'ttp06', cond, _gstim(g), dt=0.1)
    gto0 = sim._live_ionic_model().params.Gto
    gkr0 = sim._live_ionic_model().params.GKr
    sim.scale_conductance('GKr', 0.5)
    assert sim._live_ionic_model().params.Gto == gto0, "cell-type conductances must not change"
    assert sim._live_ionic_model().params.GKr == pytest.approx(gkr0 * 0.5)


def test_scale_conductivity_scales_sigma_on_bidomain():
    g = Grid(16, 12, 0.02)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    sim = bidomain(g, 'ttp06', cond, _gstim(g), dt=0.1)
    zone = np.zeros((16, 12), dtype=bool); zone[6:10, 4:8] = True
    si0 = float(sim._data.sigma_i[0][7, 6])
    sim.scale_conductivity(zone, 0.25)
    assert float(sim._data.sigma_i[0][7, 6]) == pytest.approx(si0 * 0.25)


def test_scale_conductance_on_lbm():
    """scale_conductance works on LBM too (deep-copies the live model — ENDO-consistent)."""
    g = Grid(16, 8, 0.02)
    cond = ConductivityConfig.isotropic(1.4)
    sim = lbm(g, 'ttp06', cond, _gstim(g, width=0.06), dt=0.005)
    gkr0 = sim._live_ionic_model().params.GKr
    sim.scale_conductance('GKr', 0.5)
    assert sim._live_ionic_model().params.GKr == pytest.approx(gkr0 * 0.5)


def test_regional_conductivity_on_lbm_raises_clearly():
    """A regional scar makes D non-uniform, which the LBM path can't represent — the
    error must name LBM, not the misleading 'oblique fibers (D_xy != 0)' message."""
    g = Grid(16, 12, 0.02)
    cond = ConductivityConfig.isotropic(1.4)
    sim = lbm(g, 'ttp06', cond, _gstim(g), dt=0.005)
    scar = np.zeros((16, 12), dtype=bool); scar[7:9, 4:8] = True
    with pytest.raises(NotImplementedError, match="LBM"):
        sim.set_conductivity(scar, D=0.0)


# ===========================================================================
# Phase 4 — the cheatsheet's runnable example must actually execute (doc canary)
# ===========================================================================
def test_cheatsheet_examples_execute():
    """Extract the tagged runnable block from API_CHEATSHEET.md and exec it, so the
    documented calls (construct/run/record/dct/scale_conductance/set_conductivity/
    dominant_frequency/save_result/load_result) can't silently drift from the code."""
    cheatsheet = Path(__file__).resolve().parents[1] / "API_CHEATSHEET.md"
    text = cheatsheet.read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    runnable = [b for b in blocks if b.lstrip().startswith("# runnable-canary")]
    assert runnable, "no runnable-canary block found in API_CHEATSHEET.md"
    for block in runnable:
        exec(compile(block, "<cheatsheet>", "exec"), {"__name__": "__cheatsheet__"})


# ===========================================================================
# Phase 5 (P2) — aggregate / per-beat / axis analysis helpers + guards
# ===========================================================================
def _planar_wave_x(Nx=30, Ny=6, n=60, idx_per_save=2):
    times = torch.arange(n, dtype=torch.float64)
    V = torch.full((n, Nx, Ny), -85.0, dtype=torch.float64)
    for t in range(n):
        front = t * idx_per_save
        if front > 0:
            V[t, :min(front, Nx), :] = 20.0
    return times, V


def test_dominant_frequency_map():
    n = 2000
    times = torch.arange(n, dtype=torch.float64)          # 1 ms spacing
    trace = torch.sin(2 * torch.pi * 5.0 * times / 1000.0)  # 5 Hz
    V = trace.reshape(n, 1, 1).expand(n, 4, 3).clone()
    dfm = analysis.dominant_frequency_map(V, times)         # 0.5 Hz resolution → no warn
    assert dfm.shape == (4, 3)
    assert dfm[0, 0].item() == pytest.approx(5.0, abs=0.6)


def test_df_map_resolution_warns():
    n = 100
    times = torch.arange(n, dtype=torch.float64)
    V = torch.sin(2 * torch.pi * 5.0 * times / 1000.0).reshape(n, 1, 1).expand(n, 3, 3).clone()
    with pytest.warns(UserWarning, match="resolution"):
        analysis.dominant_frequency_map(V, times)


def test_cv_between_diagonal_axis():
    times, V = _planar_wave_x(Nx=30, Ny=6, idx_per_save=2)
    cv = analysis.cv_between(V, times, (5, 2), (15, 2), dx=0.02)
    assert cv == pytest.approx(40.0, rel=0.2)   # 2 idx/ms * 0.02 cm = 0.04 cm/ms = 40 cm/s
    # genuinely off-axis: (15,4) activates at the same frame as (15,2) (wave is planar in x),
    # so the extra dy leg lengthens the path -> higher CV. If the dy term were dropped this
    # would equal the on-axis value. dist=hypot(0.2,0.04)=0.2040 cm over the same 5 ms.
    cv_diag = analysis.cv_between(V, times, (5, 2), (15, 4), dx=0.02, dy=0.02)
    assert cv_diag == pytest.approx(40.79, rel=0.03)
    assert cv_diag > cv + 0.3   # the dy component is actually used


def test_radial_cv_point_source():
    Nx, Ny, n = 31, 31, 45
    cx, cy = 15, 15
    times = torch.arange(n, dtype=torch.float64)
    ii, jj = torch.meshgrid(torch.arange(Nx), torch.arange(Ny), indexing='ij')
    frame = torch.round(torch.sqrt(((ii - cx) ** 2 + (jj - cy) ** 2).double())).long()
    V = torch.full((n, Nx, Ny), -85.0, dtype=torch.float64)
    for t in range(n):
        V[t][frame <= t] = 20.0
    rcv = analysis.radial_cv(V, times, (cx, cy), dx=0.02)
    assert torch.isnan(rcv[cx, cy])                          # center is NaN
    assert rcv[25, 15].item() == pytest.approx(20.0, rel=0.15)  # dx/1ms*1000 = 20 cm/s


def test_apd_per_beat_multibeat():
    rest = -85.0
    trace = [rest] * 10 + [20.0] * 45 + list(np.linspace(20.0, rest, 6))
    trace += [rest] * (160 - len(trace))
    trace += [40.0] * 45 + list(np.linspace(40.0, rest, 6))
    V = torch.tensor(trace, dtype=torch.float64).reshape(-1, 1, 1)
    times = torch.arange(V.shape[0], dtype=torch.float64)
    apds = analysis.apd_per_beat(V, times, 0, 0, repol=0.9)
    assert apds.numel() == 2
    assert torch.isfinite(apds).all()
    assert 30.0 < apds[0].item() < 80.0     # beat 1 not corrupted by the taller beat 2


def test_restitution_slope_alternans_threshold():
    DI = torch.tensor([50.0, 100.0, 150.0, 200.0])
    APD = torch.tensor([200.0, 260.0, 290.0, 310.0])  # slopes 1.2, 0.6, 0.4 at mids 75/125/175
    res = analysis.restitution_slope(DI, APD)
    assert res['max_slope'] == pytest.approx(1.2, abs=1e-9)
    # DI* interpolates the descending slope=1 crossing between mids 75 (slope 1.2) and
    # 125 (slope 0.6): 75 + (1.2-1)/(1.2-0.6)*50 = 91.67 — NOT the steep short-DI endpoint.
    assert res['DI_star'] == pytest.approx(91.667, abs=0.1)
    # no crossing (all slopes < 1) -> NaN
    flat = analysis.restitution_slope(torch.tensor([50.0, 100.0]), torch.tensor([200.0, 210.0]))
    assert flat['DI_star'] != flat['DI_star']  # NaN
    assert analysis.restitution_slope(torch.tensor([1.0]), torch.tensor([2.0]))['n'] == 1


def test_zero_node_stimulus_warns_stimulate():
    sim = monodomain(create_cardiac_mesh(0.2, 0.1, 0.02, D=1e-3, chi=1.0))
    with pytest.warns(UserWarning, match="0 tissue nodes"):
        sim.stimulate(lambda x, y: x < -1.0)   # region selects nothing


def test_zero_node_stimulus_warns_declarative():
    g = Grid(10, 6, 0.02)
    cond = ConductivityConfig.isotropic(1.4)
    bad = Stim.from_region(g, (lambda x, y: x < -1.0), start_time=1.0, duration=2.0, amplitude=-52.0)
    with pytest.warns(UserWarning, match="0 tissue nodes"):
        monodomain(g, 'ttp06', cond, bad)


def test_result_p2_hooks_wired():
    """The SimulationResult P2 hooks delegate to analysis without error."""
    times, V = _planar_wave_x(Nx=20, Ny=5, idx_per_save=2)
    from cardiac_core.run import SimulationResult
    r = SimulationResult(times=times, Vm=V, dx=0.02, dy=0.02)
    assert r.cv_between((3, 2), (12, 2)) == pytest.approx(40.0, rel=0.25)
    assert r.radial_cv((0, 2)).shape == (20, 5)


# ===========================================================================
# Round-2 audit remediation
# ===========================================================================
def test_scale_conductance_hipsc_lowercase_names():
    """hiPSC models (paci/phas13/mhas13) name conductances lowercase g_* — scale_conductance
    must accept those (regression: the G*/P* filter rejected them) and reject the
    dimensionless gamma_ncx / PkNa that merely start with g/p."""
    g = Grid(12, 4, 0.02)
    cond = ConductivityConfig.isotropic(1.4)
    sim = monodomain(g, 'paci', cond, _gstim(g, width=0.06), dt=0.02)
    gna0 = sim._live_ionic_model().params.g_Na
    sim.scale_conductance('g_Na', 0.5)
    assert sim._live_ionic_model().params.g_Na == pytest.approx(gna0 * 0.5)
    for bad in ('gamma_ncx', 'PkNa'):
        with pytest.raises(ValueError, match="not a scalable conductance"):
            sim.scale_conductance(bad, 0.5)


def test_dominant_frequency_map_nan_at_masked_node():
    """A masked (NaN) node must be NaN in the DF map, not a phantom low frequency (B8 + P5)."""
    n = 2000
    times = torch.arange(n, dtype=torch.float64)
    V = torch.sin(2 * torch.pi * 5.0 * times / 1000.0).reshape(n, 1, 1).expand(n, 4, 3).clone()
    V[:, 1, 1] = float('nan')
    dfm = analysis.dominant_frequency_map(V, times)
    assert torch.isnan(dfm[1, 1]), "masked node must be NaN"
    assert dfm[0, 0].item() == pytest.approx(5.0, abs=0.6)   # finite nodes unaffected


def test_radial_cv_warns_on_dead_center():
    Nx, Ny, n = 10, 10, 20
    times = torch.arange(n, dtype=torch.float64)
    V = torch.full((n, Nx, Ny), -85.0, dtype=torch.float64)   # nothing activates
    with pytest.warns(UserWarning, match="never activates"):
        rcv = analysis.radial_cv(V, times, (5, 5), dx=0.02)
    assert torch.isnan(rcv).all()


def test_restitution_slope_last_crossing():
    """On a non-monotonic curve with two descending slope=1 crossings, DI* is the LARGER-DI
    (alternans-boundary) one, not the first."""
    DI = torch.tensor([50., 100., 150., 200., 250.])
    APD = torch.tensor([200., 275., 300., 365., 385.])   # slopes 1.5, 0.5, 1.3, 0.4
    res = analysis.restitution_slope(DI, APD)
    assert res['DI_star'] > 150.0, f"should pick the larger-DI crossing: {res['DI_star']}"


def test_bidomain_masked_default_elliptic_solver_works():
    """A masked bidomain on the DEFAULT elliptic_solver='auto' must not crash — auto now
    falls back to pcg (spectral/pcg_spectral reshape phi_e to the full rectangle)."""
    mask = np.ones((15, 15), dtype=bool); mask[6:9, 6:9] = False
    sim = bidomain(create_cardiac_mesh(0.28, 0.28, 0.02, D=1e-3, chi=1.0, mask=mask),
                   stimulus=_stim())   # no explicit elliptic_solver
    assert sim._engine._elliptic_solver_name == 'pcg'
    r = sim.run(4.0, 1.0)
    assert torch.isnan(r.Vm[:, torch.tensor(~mask)]).all()


def test_dct_rejected_on_nondefault_stencil_and_boundary():
    """Cover the stencil/boundary_mode branches of the spectral gate (were untested)."""
    cond = ConductivityConfig.isotropic(1.4)
    g = Grid(30, 8, 0.02)
    stim = _gstim(g, width=0.06)
    with pytest.raises(ValueError, match="stencil|silently wrong"):
        monodomain(g, 'ttp06', cond, stim,
                   linear_solver='dct', stencil='moore8_iso')
    with pytest.raises(ValueError, match="boundary_mode|silently wrong"):
        monodomain(g, 'ttp06', cond, stim,
                   linear_solver='dct', boundary_mode='rest_pad')
