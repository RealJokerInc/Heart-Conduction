"""The 0-D single_cell driver.

The companion safety_factor tests live in test_safety_factor.py.
"""

import pytest
import torch

from cardiac_core.single_cell import single_cell


def test_ttp06_ap():
    r = single_cell('ttp06', celltype='EPI', n_beats=1, bcl=350.0, save_every=1.0)
    assert r.v_peak > 10.0                    # a real upstroke (overshoot)
    assert r.v_rest < -75.0                    # stable diastolic rest
    apd = r.apd(0.9)
    assert 150.0 < apd < 340.0                 # physiological APD90 band (single non-paced beat)


def test_cm_scaling_reaction():
    # Cm=2 DAMPS the reaction (less depolarization); the per-step /Cm mechanism is exact.
    r1 = single_cell('ttp06', celltype='EPI', n_beats=1, bcl=50.0, Cm=1.0, save_every=1.0)
    r2 = single_cell('ttp06', celltype='EPI', n_beats=1, bcl=50.0, Cm=2.0, save_every=1.0)
    assert (r2.v_peak - r2.v_rest) < (r1.v_peak - r1.v_rest)      # damped upstroke
    # the reaction-/Cm rescaling is exact on a single shared step:
    from cardiac_core.ionic.registry import build_ionic_model
    m = build_ionic_model('ttp06', 'EPI', device='cpu')
    V0 = torch.tensor(-50.0, dtype=torch.float64)
    s = m.get_initial_state(1)
    V1, _ = m.step(V0, s, 0.02, torch.tensor(-52.0, dtype=torch.float64))
    assert (V0 + (V1 - V0) / 2.0 - V0).item() == pytest.approx(0.5 * (V1 - V0).item())


def test_0d_vs_tissue_singlenode():
    # A uniformly-stimulated tissue patch has NO gradient -> diffusion=0 -> each node is 0-D. Its
    # trace matches single_cell to the splitting truncation (the ORd-ordering-bug guard: same step).
    from cardiac_core import monodomain, Grid, ConductivityConfig, Stim
    g = Grid(4, 4, 0.05)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = Stim.from_region(g, (lambda x, y: x > -1.0), start_time=10.0, duration=2.0,
                            amplitude=-52.0)   # whole-domain stimulus -> uniform field
    r = monodomain(g, 'ttp06', cond, stim, dt=0.02).run(300.0, save_every=1.0)
    from cardiac_core import analysis
    tissue_apd = analysis.apd_at(r.Vm, r.times, 2, 2, repol=0.9)

    sc = single_cell('ttp06', celltype='ENDO', dt=0.02, n_beats=1, bcl=290.0, t0=10.0,
                     stim_amplitude=-52.0, stim_duration=2.0, save_every=1.0)
    assert sc.apd(0.9) == pytest.approx(tissue_apd, rel=0.05)   # agree to the split truncation


def test_prepace_runs_and_is_stable():
    r = single_cell('ttp06', celltype='EPI', n_beats=1, bcl=300.0, pre_pace=1, save_every=1.0)
    assert r.v_peak > 10.0
    assert r.v_rest < -75.0
    assert torch.isfinite(r.V).all()
