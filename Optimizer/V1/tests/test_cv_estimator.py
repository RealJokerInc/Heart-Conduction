"""
Tests for cv_estimator.py — convergence-aware CV (PLAN Step 1.2, architecture §4).

Calibrated (baseline MHAS13, chip cable):
- RESOLVABLE: D=2e-4, ladder (0.004,0.002,0.001) → converged, r*/dx = 4.2/8.3/16.8,
  CV plateaus 12.0→11.9 (finest two agree 0.8%).
- BLOCKED (corrupted band): D=5e-5, ladder (0.02,0.015,0.01) → CV finite (4.7/5.3/5.9)
  but r*/dx = 0.53/0.63/0.85 (all <3) → converged=False (refuses to extrapolate a
  grid-corrupted trend — the whole point of the estimator).
"""

import math

import pytest


def _chip_cfg():
    from tuner.config import TuningConfig
    return TuningConfig(device='cpu', ionic_model='mhas13', tier=1,
                        dx_cm=0.01, cable_length_cm=0.5, dt=0.02,
                        stim_amplitude=-40.0, stim_start=1.0, engine='monodomain')


def test_rstar_cm_units():
    """Pure-function unit check (no sim): r* = D/(CV/1000) in cm; NaN if non-propagating."""
    from tuner.cv_estimator import rstar_cm
    assert rstar_cm(2e-4, 12.0) == pytest.approx(2e-4 / (12.0 / 1000.0))   # 0.01667 cm
    assert math.isnan(rstar_cm(2e-4, float('nan')))
    assert math.isnan(rstar_cm(2e-4, 0.0))


@pytest.mark.slow
def test_reports_blocked():
    """A D whose ladder stays in the corrupted band (r*/dx<3) → converged=False,
    even though the wave propagates (CV finite). The estimator must NOT extrapolate."""
    from tuner.cv_estimator import resolved_cv
    cfg = _chip_cfg()
    res = resolved_cv(None, 5e-5, cfg, dx_ladder=(0.02, 0.015, 0.01))
    assert res['converged'] is False
    assert res['rstar_over_dx'] < 3.0
    assert math.isnan(res['cv_resolved'])
    assert len(res['rungs']) == 3


@pytest.mark.slow
def test_resolved_stable():
    """A well-resolved D → converged=True with r*/dx≥3 at every rung and a CV that
    plateaus across the finest two rungs (≤3%)."""
    from tuner.cv_estimator import resolved_cv
    cfg = _chip_cfg()
    res = resolved_cv(None, 2e-4, cfg, dx_ladder=(0.004, 0.002, 0.001))
    assert res['converged'] is True
    assert res['rstar_over_dx'] >= 3.0
    assert math.isfinite(res['cv_resolved']) and res['cv_resolved'] > 0
    assert all(r['rstar_over_dx'] >= 3.0 for r in res['rungs'])
    cv_coarse, cv_fine = res['rungs'][-2]['cv'], res['rungs'][-1]['cv']
    assert abs(cv_fine - cv_coarse) / cv_coarse <= 0.03      # plateau reached
