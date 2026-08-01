"""Self-test for the eikonal metrics (front normal / CV_n / curvature / fit).

The analysis pipeline must recover known answers on synthetic activation-time
(LAT) fields before it is trusted on a physics run, so that a diverging fan's
dynamic slowing is never confused with purely kinematic spreading.
"""
import numpy as np

from cardiac_core.analysis import front_metrics, fit_eikonal, activation_time_interp

CV0 = 0.06          # cm/ms
DX = 0.005          # cm (50 um)
N = 201


def _grid():
    i = np.arange(N) * DX
    X, Y = np.meshgrid(i, i, indexing="ij")  # (Nx, Ny), axis0 = x
    cx = cy = (N - 1) / 2 * DX
    R = np.hypot(X - cx, Y - cy)
    return X, Y, R


def test_radial_synthetic():
    """lat = r/CV0 -> cv_n flat = CV0, kappa = 1/r, fit gives D_eik ~ 0."""
    _, _, R = _grid()
    lat = R / CV0
    m = front_metrics(lat, DX)
    ann = (R > 0.15) & (R < 0.35)

    cv_med = np.nanmedian(m["cv_n"][ann])
    assert abs(cv_med - CV0) / CV0 < 0.03, f"cv_n {cv_med} != {CV0}"

    rel = np.abs(m["kappa"][ann] - 1.0 / R[ann]) / (1.0 / R[ann])
    assert np.nanmedian(rel) < 0.10, f"kappa rel-err {np.nanmedian(rel)}"

    fit = fit_eikonal(m["cv_n"], m["kappa"], mask=ann)
    assert abs(fit["CV0"] - CV0) / CV0 < 0.03, fit
    assert abs(fit["D_eik"]) < 2e-4, fit  # synthetic has no real curvature-slowing


def test_planar_synthetic():
    """lat = x/CV0 -> cv_n = CV0, kappa = 0 (flat front)."""
    X, _, _ = _grid()
    lat = X / CV0
    m = front_metrics(lat, DX)
    interior = np.zeros((N, N), bool)
    interior[3:-3, 3:-3] = True

    cv_med = np.nanmedian(m["cv_n"][interior])
    assert abs(cv_med - CV0) / CV0 < 0.01, cv_med
    assert abs(np.nanmedian(m["kappa"][interior])) < 0.5, np.nanmedian(m["kappa"][interior])


def test_activation_time_interp_linear():
    """A linear-in-time ramp crossing threshold returns the interpolated time."""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    # one cell: V goes -60 -> -50 -> -30 -> -10; crosses -40 between t=1 and t=2
    V = np.array([-60.0, -50.0, -30.0, -10.0]).reshape(4, 1, 1)
    lat = activation_time_interp(V, times, threshold=-40.0)
    # linear interp between (1,-50) and (2,-30): t = 1 + (-40 - -50)/20 = 1.5
    assert abs(lat[0, 0] - 1.5) < 1e-9, lat[0, 0]
