"""
Tests for decision_space.py — unified decision-space registry + apply() (Step 3.1).

Fast (no simulation): only builds the scaled model + mesh and checks the flat vector
maps to the expected scaled conductances and per-axis diffusion, with D_trans FREE.
"""

import pytest


def _cfg():
    from tuner.config import TuningConfig
    # Tiny mesh (domain 1 mm / dx 0.5 mm → 3×3) — apply() only builds it, no sim.
    return TuningConfig(device='cpu', ionic_model='mhas13', tier=2,
                        domain_mm=1.0, dx_mm=0.5, dt=0.02)


def _vector(axes, overrides):
    """Build a vector aligned to axes: overrides by name, else 1.0."""
    return [overrides.get(a.name, 1.0) for a in axes]


def test_build_axes_structure():
    from tuner.decision_space import build_axes
    axes = build_axes(tier=2, gNa_floor=0.15, include_kinetics=False)
    names = [a.name for a in axes]
    assert 'g_Na' in names and 'D_long' in names and 'D_trans' in names
    gNa = next(a for a in axes if a.name == 'g_Na')
    assert gNa.bounds[0] == 0.15                       # widened floor (lock-2)
    subs = {a.subsystem for a in axes}
    assert 'cond' in subs and 'diffusion' in subs
    # D_long / D_trans are BOTH present as free diffusion axes
    diff = [a.name for a in axes if a.subsystem == 'diffusion']
    assert set(diff) == {'D_long', 'D_trans'}


def test_apply_roundtrip():
    from tuner.decision_space import build_axes, apply
    from tuner.config import PHAS13_REGISTRY
    cfg = _cfg()
    axes = build_axes(tier=2, include_kinetics=False)
    vec = _vector(axes, {'g_Na': 0.5, 'D_long': 1e-4, 'D_trans': 5e-5})

    model, mesh = apply(vec, axes, cfg)

    assert model.params.g_Na == pytest.approx(PHAS13_REGISTRY['g_Na'].published * 0.5)
    assert float(mesh.D_xx.max()) == pytest.approx(1e-4)      # D_long
    assert float(mesh.D_yy.max()) == pytest.approx(5e-5)      # D_trans (free)


def test_dx_axis():
    """include_dx adds a tunable grid axis; apply() uses it for the mesh; dx_of reads it."""
    from tuner.decision_space import build_axes, apply, dx_of
    cfg = _cfg()
    axes = build_axes(tier=2, include_kinetics=False, include_dx=True,
                      dx_bounds_cm=(0.002, 0.05))
    dx_ax = [a for a in axes if a.name == 'dx_cm']
    assert len(dx_ax) == 1 and dx_ax[0].subsystem == 'grid'
    assert dx_ax[0].bounds == (0.002, 0.05)

    vec = _vector(axes, {'dx_cm': 0.02, 'D_long': 1e-4, 'D_trans': 5e-5})
    assert dx_of(vec, axes) == pytest.approx(0.02)
    _model, mesh = apply(vec, axes, cfg)
    assert float(mesh.dx) == pytest.approx(0.02)        # dx_cm=0.02 → 0.2 mm mesh

    # dx-absent build → dx_of returns None (back-compat)
    axes0 = build_axes(tier=2, include_kinetics=False, include_dx=False)
    assert dx_of(_vector(axes0, {}), axes0) is None


def test_dtrans_free():
    from tuner.decision_space import build_axes, apply
    cfg = _cfg()
    axes = build_axes(tier=2, include_kinetics=False)

    _, m1 = apply(_vector(axes, {'D_long': 1e-4, 'D_trans': 5e-5}), axes, cfg)
    _, m2 = apply(_vector(axes, {'D_long': 1e-4, 'D_trans': 2e-5}), axes, cfg)

    assert float(m1.D_xx.max()) == pytest.approx(float(m2.D_xx.max()))   # D_long tied
    assert float(m1.D_yy.max()) != pytest.approx(float(m2.D_yy.max()))   # D_trans free
