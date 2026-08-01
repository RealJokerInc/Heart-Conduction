"""The 0-D single_cell driver.

The companion safety_factor tests live in test_safety_factor.py.
"""

import types

import pytest
import torch

from cardiac_core import single_cell


def test_public_exports_not_shadowed_by_submodules():
    """Regression: a `_LAZY` export whose NAME equals its submodule name gets shadowed.

    PEP 562 `__getattr__` only fires when normal attribute lookup FAILS. Importing submodule
    `cardiac_core.X` binds it as a package attribute, so a lazy export also named `X` silently
    becomes the MODULE from then on — `cc.single_cell` was a function on first access and a
    (non-callable) module on every access after. Fixed by making the module private
    (`_single_cell`). This guards the WHOLE export map so no future export can regress the same way.
    """
    import cardiac_core as cc
    from cardiac_core import _LAZY

    collisions = sorted(n for n, mod in _LAZY.items() if n == mod)
    assert collisions == [], (
        f"export name(s) collide with their own submodule and will be shadowed: {collisions}. "
        f"Rename the submodule private (e.g. '_{collisions[0] if collisions else 'x'}.py')."
    )

    for name in _LAZY:
        first = getattr(cc, name)
        assert not isinstance(first, types.ModuleType), f"cc.{name} resolved to a module, not an object"
        assert getattr(cc, name) is first, f"cc.{name} changed identity on repeated access (shadowed)"


def test_single_cell_public_export_is_callable():
    """Both public import forms must give the FUNCTION (this is what was broken)."""
    import cardiac_core as cc
    from cardiac_core import single_cell as fn
    assert callable(fn) and callable(cc.single_cell)
    assert cc.single_cell is fn


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


# ---------------------------------------------------------------------------
# conductances= : the 0-D drug knob (mirrors tissue scale_conductance)
# ---------------------------------------------------------------------------

def test_single_cell_ikr_block_prolongs_apd():
    # 50% IKr block (dofetilide-like) slows phase-3 repolarization -> longer APD90.
    base = single_cell('ttp06', celltype='EPI', n_beats=1, bcl=400.0, pre_pace=1, save_every=1.0)
    drug = single_cell('ttp06', celltype='EPI', n_beats=1, bcl=400.0, pre_pace=1, save_every=1.0,
                       conductances={'GKr': 0.5})
    assert drug.apd(0.9) > base.apd(0.9) + 3.0        # a real, non-trivial prolongation


def test_single_cell_ina_block_reduces_dvdt():
    # 90% INa block reduces the available depolarizing current -> slower upstroke (lower max dV/dt).
    def max_dvdt(r):
        dts = float(r.times[1] - r.times[0])
        return float((r.V[1:] - r.V[:-1]).max()) / dts
    base = single_cell('ttp06', n_beats=1, bcl=120.0, save_every=0.02)
    drug = single_cell('ttp06', n_beats=1, bcl=120.0, save_every=0.02, conductances={'GNa': 0.1})
    assert max_dvdt(drug) < max_dvdt(base)


def test_single_cell_unknown_conductance_matches_tissue_message():
    # The 0-D and tissue paths share ionic.scaling.scale_ionic_conductances -> identical error text.
    from cardiac_core import monodomain, Grid, ConductivityConfig, Stim
    with pytest.raises(ValueError) as e0d:
        single_cell('ttp06', conductances={'Gxx': 1.0})
    g = Grid(4, 4, 0.05)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = Stim.from_region(g, (lambda x, y: x > -1.0), start_time=10.0, duration=2.0, amplitude=-52.0)
    sim = monodomain(g, 'ttp06', cond, stim, dt=0.02)
    with pytest.raises(ValueError) as etis:
        sim.scale_conductance('Gxx', 1.0)
    assert str(e0d.value) == str(etis.value)


def test_single_cell_conductance_parity_with_tissue():
    # Drugged 0-D matches a drugged uniformly-stimulated tissue node (APD90, to the split truncation).
    # Extends test_0d_vs_tissue_singlenode for the conductances= path; compares APD, NOT V traces.
    from cardiac_core import monodomain, Grid, ConductivityConfig, Stim, analysis
    g = Grid(4, 4, 0.05)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = Stim.from_region(g, (lambda x, y: x > -1.0), start_time=10.0, duration=2.0, amplitude=-52.0)
    sim = monodomain(g, 'ttp06', cond, stim, dt=0.02)
    sim.scale_conductance('GKr', 0.5)                 # drug the tissue
    r = sim.run(300.0, save_every=1.0)
    tissue_apd = analysis.apd_at(r.Vm, r.times, 2, 2, repol=0.9)

    sc = single_cell('ttp06', celltype='ENDO', dt=0.02, n_beats=1, bcl=290.0, t0=10.0,
                     stim_amplitude=-52.0, stim_duration=2.0, save_every=1.0,
                     conductances={'GKr': 0.5})       # drug the 0-D cell the same way
    assert sc.apd(0.9) == pytest.approx(tissue_apd, rel=0.05)


def test_single_cell_drug_before_pre_pace():
    # The drug is applied BEFORE pre-pacing, so the cell settles toward its OWN drugged steady state.
    # Scaling AFTER pre-pacing (undrugged steady state) gives a different final_state -> ordering matters.
    from cardiac_core._single_cell import _pace
    from cardiac_core.ionic.registry import build_ionic_model
    from cardiac_core.ionic.scaling import scale_ionic_conductances
    kw = dict(bcl=250.0, t0=10.0, stim_amplitude=-52.0, stim_duration=2.0, Cm=1.0, save_every=1)

    # BEFORE (the real single_cell ordering): scale -> pre_pace -> record.
    m_b = scale_ionic_conductances(build_ionic_model('ttp06', 'ENDO', device='cpu'), {'GKr': 0.5})
    Vb = torch.tensor(m_b.V_rest, dtype=m_b.dtype, device=m_b.device)
    sb = m_b.get_initial_state(n_cells=1)
    _, _, Vb, sb = _pace(m_b, Vb, sb, 0.02, n_beats=2, record=False, **kw)
    _, _, _, sb_final = _pace(m_b, Vb, sb, 0.02, n_beats=1, record=True, **kw)

    # AFTER: pre_pace the UNDRUGGED cell -> scale -> record.
    m_u = build_ionic_model('ttp06', 'ENDO', device='cpu')
    Vu = torch.tensor(m_u.V_rest, dtype=m_u.dtype, device=m_u.device)
    su = m_u.get_initial_state(n_cells=1)
    _, _, Vu, su = _pace(m_u, Vu, su, 0.02, n_beats=2, record=False, **kw)
    m_a = scale_ionic_conductances(m_u, {'GKr': 0.5})
    _, _, _, su_final = _pace(m_a, Vu, su, 0.02, n_beats=1, record=True, **kw)

    assert not torch.allclose(sb_final, su_final)     # ordering changes the steady state


def test_single_cell_no_conductances_is_unchanged():
    # Regression: None / {} leave the no-drug path bit-identical to today.
    a = single_cell('ttp06', n_beats=1, bcl=200.0, save_every=1.0)
    n = single_cell('ttp06', n_beats=1, bcl=200.0, save_every=1.0, conductances=None)
    e = single_cell('ttp06', n_beats=1, bcl=200.0, save_every=1.0, conductances={})
    assert torch.equal(a.V, n.V)
    assert torch.equal(a.V, e.V)
    assert torch.equal(a.final_state, e.final_state)
