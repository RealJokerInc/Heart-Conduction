"""
Smoke test for diag_hipsc_window (PLAN Step 1.3).

Runs the hiPSC-window sweep on a tiny grid (no media I/O) and checks it returns the
architecture-relevant keys ('window', 'nan_cause') and a valid per-point cause.
"""

import pytest


@pytest.mark.slow
def test_hipsc_window_runs():
    from tuner.config import TuningConfig
    from diag_hipsc_window import diag_hipsc_window

    cfg = TuningConfig(device='cpu', ionic_model='mhas13', tier=2,
                       dx_cm=0.02, cable_length_cm=0.3, dt=0.02,
                       stim_amplitude=-40.0, stim_start=1.0, engine='monodomain')
    # Partial θ dict (apply_scaling handles a subset); exercises the sweep cheaply.
    theta = {'g_Na': 0.5, 'g_CaL': 1.0}

    res = diag_hipsc_window(theta, cfg, [1e-3, 1e-4], save_media=False)

    assert 'window' in res
    assert 'nan_cause' in res
    assert 'rows' in res and len(res['rows']) == 2
    for r in res['rows']:
        assert r['cause'] in ('propagates', 'over_depolarization',
                              'no_capture', 'source_sink_block')
        assert 'rstar_over_dx' in r and 'vmax' in r
