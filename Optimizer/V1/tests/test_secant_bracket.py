"""
Tests for the secant D-fit bracket-DOWN fix (PLAN Step 1.1).

The old `_fit_D_for_cv` bumped D UP ×4 on a non-propagating start; for slow chip
targets the propagating window sits BELOW D0=1e-3, so the up-bump walked into the
high-D NaN zone and returned a FAKE D (0.004, NaN) — a Known Failure that produced
the garbage `D=0.004, CV=nan` tissue records. The fix brackets DOWN into the window
and returns (NaN, NaN) — never a fake D — when nothing propagates.

Calibration (baseline MHAS13, dx=0.01 cm): CV=nan @ D≥5e-4 (above window), then
12.4 / 8.4 / 5.9 / 3.9 cm/s @ D = 2e-4 / 1e-4 / 5e-5 / 2.5e-5. So target CV=6 lands
at D≈5e-5 (in the window), reached by bracketing DOWN from D0=1e-3.
"""

import math

import pytest


def _chip_cfg():
    from tuner.config import TuningConfig
    return TuningConfig(device='cpu', ionic_model='mhas13', tier=1,
                        dx_cm=0.01, cable_length_cm=0.5, dt=0.02,
                        stim_amplitude=-40.0, stim_start=1.0, engine='monodomain')


@pytest.mark.slow
def test_brackets_down():
    """target CV=6 → the fit brackets DOWN into the window; finite CV in tol; D in
    the window (well below D0); NEVER the old fake-D fallback (0.004)."""
    from tuner.cc_runner import fit_D_for_cv
    cfg = _chip_cfg()
    D, cv = fit_D_for_cv(None, 6.0, cfg, D0=1e-3, n=8, tol=0.02)

    assert math.isfinite(cv) and cv > 0
    assert abs(cv - 6.0) / 6.0 < 0.10                     # secant converged near target
    assert 3e-5 <= D <= 2e-4                              # in the propagating window
    assert D < 1e-3                                       # bracketed DOWN from D0
    assert not math.isclose(D, 0.004, rel_tol=1e-3)       # never the fake fallback


@pytest.mark.slow
def test_no_fake_fallback_when_blocked():
    """When no propagating D exists down to D_lo, return (NaN, NaN) — not a fake D.

    Force it by putting D_lo ABOVE the propagating window: D0=1e-3 is NaN and the
    bracket-down cannot descend below D_lo=6e-4 (still above the window), so there is
    no propagating point to return."""
    from tuner.cc_runner import fit_D_for_cv
    cfg = _chip_cfg()
    D, cv = fit_D_for_cv(None, 6.0, cfg, D0=1e-3, D_lo=6e-4, n=8)

    assert math.isnan(D) and math.isnan(cv)               # honest infeasibility
    assert not math.isclose(D, 0.004, rel_tol=1e-3) if math.isfinite(D) else True
