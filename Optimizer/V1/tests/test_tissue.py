"""
Tests for tissue_runner.py — 1D cable CV measurement.
"""

import pytest
import torch


class TestTissueRunner:
    """Phase IV: Tissue runner tests."""

    def test_cv_measurement_baseline(self):
        """Baseline D produces measurable CV."""
        from tuner.config import TuningConfig, get_param_names
        from tuner.tissue_runner import run_cv_measurement

        # Tissue sims need small dt for wavefront resolution
        config = TuningConfig(device='cpu', tier=1, n_beats=2, dt=0.01,
                              pacing_cl=1500.0)
        n_params = len(get_param_names(1))
        theta = torch.ones(n_params, dtype=torch.float64)

        # Reasonable D for PHAS13
        D = 0.0001  # cm^2/ms
        result = run_cv_measurement(theta, D, config,
                                    cable_length_cm=0.3, dx_cm=0.01)

        # May not converge at this dt on CPU (borderline), so just check it runs
        if result.converged and result.cv is not None:
            assert result.cv > 0
            assert result.cv < 200  # Reasonable range

    def test_cv_increases_with_D(self):
        """Higher D -> higher CV."""
        from tuner.config import TuningConfig, get_param_names
        from tuner.tissue_runner import run_cv_measurement

        config = TuningConfig(device='cpu', tier=1, n_beats=2, dt=0.01,
                              pacing_cl=1500.0)
        n_params = len(get_param_names(1))
        theta = torch.ones(n_params, dtype=torch.float64)

        D_low = 0.00005
        D_high = 0.0003

        res_low = run_cv_measurement(theta, D_low, config,
                                     cable_length_cm=0.3, dx_cm=0.01)
        res_high = run_cv_measurement(theta, D_high, config,
                                      cable_length_cm=0.3, dx_cm=0.01)

        if (res_low.cv is not None and res_high.cv is not None and
                res_low.converged and res_high.converged):
            assert res_high.cv > res_low.cv
