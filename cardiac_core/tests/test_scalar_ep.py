"""Phase-6 (analysis.fields): scalar EP metrics — wavelength, di, the multi-beat APD baseline fix,
and the protocol-based ERP.
"""

import warnings

import numpy as np
import pytest
import torch

from cardiac_core import analysis
from cardiac_core.analysis import wavelength, di, apd_at, apd_per_beat, restitution_curve


# ======================================================================
# Step 6.1 — wavelength + di
# ======================================================================

def test_wavelength_units():
    # λ = CV·ERP: 50 cm/s · 200 ms / 1000 = 10 cm.
    assert wavelength(50.0, 200.0, kind="erp") == pytest.approx(10.0)


def test_wavelength_apd_proxy_warns():
    with pytest.warns(UserWarning, match="underestimate"):
        wavelength(50.0, 200.0, kind="apd")


def test_wavelength_bad_kind_raises():
    with pytest.raises(ValueError):
        wavelength(50.0, 200.0, kind="bogus")


def test_di_algebraic():
    assert di(1000.0, 280.0) == pytest.approx(720.0)


# ======================================================================
# Step 6.1 — the per-beat diastolic APD baseline fix (three functions)
# ======================================================================

def _beat(v_rest, v_peak, n_rest=10, n_plateau=40, n_repol=20):
    return ([v_rest] * n_rest + [v_peak] * n_plateau
            + list(np.linspace(v_peak, v_rest, n_repol)))


def _drifting_trace():
    # Three beats with a CONSTANT peak (+20) but a strongly rising diastolic baseline (-85, -60,
    # -35). Each beat repolarizes toward its OWN rest. With the old trace[0]=-85 baseline the later
    # beats' V_repol (-74.5) sits BELOW their actual rest, so they never cross -> NaN; the per-beat
    # foot fix measures each against its own diastole -> all finite and (equal shape) near-equal.
    seq = _beat(-85.0, 20.0) + _beat(-60.0, 20.0) + _beat(-35.0, 20.0) + [-35.0] * 10
    V = torch.tensor(seq, dtype=torch.float64).reshape(-1, 1, 1)
    times = torch.arange(V.shape[0], dtype=torch.float64)
    return V, times


def test_apd_per_beat_drifting_baseline():
    V, times = _drifting_trace()
    apds = apd_per_beat(V, times, 0, 0, repol=0.9)
    assert apds.numel() == 3
    assert torch.isfinite(apds).all()                 # each beat measured against its OWN diastole
    # equal amplitude + shape -> near-equal APDs (the old code returned NaN for the drifted beats)
    assert (apds.max() - apds.min()).item() < 3.0


def test_apd_at_uses_beat_diastole():
    V, times = _drifting_trace()
    # apd_at on the drifting trace: beat-2 upstroke; must measure against ~-70, not -85.
    # (bound to a node; the single-beat regression is covered by test_analysis.py staying green.)
    a = apd_at(V, times, 0, 0, repol=0.9)
    assert np.isfinite(a)


def test_restitution_curve_drifting_baseline():
    V, times = _drifting_trace()
    DI, APD = restitution_curve(V, times, 0, 0, repol=0.9)
    assert torch.isfinite(APD).all()
    assert APD.numel() >= 1


def test_apd_at_single_beat_regression():
    # A clean single beat from rest: the per-beat diastole == trace[0], so the value is unchanged.
    seq = _beat(-85.0, 20.0) + [-85.0] * 10
    V = torch.tensor(seq, dtype=torch.float64).reshape(-1, 1, 1)
    times = torch.arange(V.shape[0], dtype=torch.float64)
    a = apd_at(V, times, 0, 0, repol=0.9)
    # V_repol = 20 - 0.9*(20-(-85)) = -74.5; reached ~90% down the 20-frame linear repol.
    assert a == pytest.approx(40 + 0.9 * 20, abs=1.5)


# ======================================================================
# Step 6.2 — protocol-based ERP (smoke)
# ======================================================================

def test_post_repol_refractoriness_algebra():
    from cardiac_core.protocols import post_repol_refractoriness, erp_proxy
    assert post_repol_refractoriness(300.0, 250.0) == pytest.approx(50.0)
    assert erp_proxy(275.0) == pytest.approx(275.0)


def test_erp_smoke_runs_and_returns_number():
    # A tiny cable + the lightweight phas13 phase model so the S1S2 bisection stays fast; we only
    # assert erp() runs the sims, bisects, and returns a number in range (or nan, warned).
    from cardiac_core.protocols import erp
    from cardiac_core import Grid, ConductivityConfig
    g = Grid(20, 3, 0.03)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = erp(g, 'phas13', cond, bcl=250.0, n_s1=2, ci_min=30.0, ci_max=220.0, tol=60.0,
                  dt=0.05, diffusion_solver='forward_euler', linear_solver='none')
    assert isinstance(val, float)
    assert np.isnan(val) or (30.0 <= val <= 220.0)
