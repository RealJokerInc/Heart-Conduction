"""
Tests for metrics.py — AP biomarker extraction on synthetic traces.
"""

import pytest
import numpy as np


def _make_synthetic_ap(dt=0.1, apd=300.0, v_rest=-80.0, v_peak=30.0,
                       cl=1000.0, n_beats=3):
    """Generate a synthetic AP trace for testing metrics.

    First AP starts at t=100ms so v_rest window (first 50ms) sees rest.
    """
    delay = 100.0  # ms before first AP
    t_total = delay + cl * n_beats
    t = np.arange(0, t_total, dt)
    V = np.full_like(t, v_rest)

    for beat in range(n_beats):
        t_start = delay + beat * cl
        # Simple triangular AP
        upstroke_ms = 2.0
        for i, ti in enumerate(t):
            if t_start <= ti < t_start + upstroke_ms:
                frac = (ti - t_start) / upstroke_ms
                V[i] = v_rest + (v_peak - v_rest) * frac
            elif t_start + upstroke_ms <= ti < t_start + apd:
                frac = (ti - t_start - upstroke_ms) / (apd - upstroke_ms)
                V[i] = v_peak - (v_peak - v_rest) * frac
    return t, V


class TestMetrics:
    """Phase I: Metric extraction tests."""

    def test_detect_aps(self):
        """Detect correct number of APs in synthetic trace."""
        from tuner.metrics import detect_aps
        t, V = _make_synthetic_ap(n_beats=3)
        aps = detect_aps(V, t)
        assert len(aps) == 3

    def test_measure_apd(self):
        """APD90 matches synthetic trace."""
        from tuner.metrics import measure_apd
        t, V = _make_synthetic_ap(apd=300.0, n_beats=3)
        apd = measure_apd(V, t, fraction=0.9)
        # Triangular AP: APD90 measured from peak is ~90% of total repolarization
        # For a linear ramp, 90% repol ≈ 0.9 * (apd - upstroke) ≈ 268 ms
        assert apd is not None
        assert 200 < apd < 350

    def test_measure_dvdt_max(self):
        """dV/dt_max positive for synthetic AP."""
        from tuner.metrics import measure_dvdt_max
        t, V = _make_synthetic_ap()
        dvdt = measure_dvdt_max(V, t)
        assert dvdt is not None
        assert dvdt > 0

    def test_measure_v_rest(self):
        """V_rest returns minimum (diastolic) voltage."""
        from tuner.metrics import measure_v_rest
        t, V = _make_synthetic_ap(v_rest=-80.0)
        vr = measure_v_rest(V, t)
        assert vr == pytest.approx(-80.0, abs=0.5)

    def test_measure_peak(self):
        """Peak voltage matches synthetic trace."""
        from tuner.metrics import measure_peak
        t, V = _make_synthetic_ap(v_peak=30.0)
        vp = measure_peak(V)
        assert vp == pytest.approx(30.0, abs=1.0)

    def test_measure_cl(self):
        """CL matches synthetic trace."""
        from tuner.metrics import measure_cl
        t, V = _make_synthetic_ap(cl=1000.0, n_beats=3)
        cl_val = measure_cl(V, t)
        assert cl_val is not None
        assert cl_val == pytest.approx(1000.0, abs=5.0)

    def test_no_ap_returns_none(self):
        """No AP in flat trace returns None."""
        from tuner.metrics import measure_apd, measure_dvdt_max, measure_cl
        t = np.arange(0, 1000, 0.1)
        V = np.full_like(t, -80.0)
        assert measure_apd(V, t) is None
        assert measure_dvdt_max(V, t) is None
        assert measure_cl(V, t) is None
