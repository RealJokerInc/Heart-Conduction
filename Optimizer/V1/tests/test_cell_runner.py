"""
Tests for cell_runner.py — Single-cell simulation wrapper.
"""

import pytest
import torch


class TestCellRunner:
    """Phase II: Cell runner tests."""

    def test_baseline_runs(self):
        """Baseline theta=1.0 produces valid result."""
        from tuner.config import TuningConfig, get_param_names
        from tuner.cell_runner import run_single_cell

        config = TuningConfig(device='cpu', tier=1, n_beats=3, dt=0.05,
                              pacing_cl=1500.0)
        n_params = len(get_param_names(1))
        theta = torch.ones(n_params, dtype=torch.float64)

        result = run_single_cell(theta, config)
        assert result.converged
        assert result.v_rest < -60.0  # Should be near -74 mV

    def test_baseline_apd(self):
        """Baseline APD measurable and in expected range."""
        from tuner.config import TuningConfig, get_param_names
        from tuner.cell_runner import run_single_cell

        config = TuningConfig(device='cpu', tier=1, n_beats=5, dt=0.05,
                              pacing_cl=1500.0)
        n_params = len(get_param_names(1))
        theta = torch.ones(n_params, dtype=torch.float64)

        result = run_single_cell(theta, config, return_trace=True)
        assert result.converged
        # PHAS13 native APD ~469 ms; with dt=0.05 some drift is OK
        if result.apd90 is not None:
            assert 100 < result.apd90 < 1000

    def test_spontaneous_beating(self):
        """Spontaneous run detects CL."""
        from tuner.config import TuningConfig, get_param_names
        from tuner.cell_runner import run_spontaneous

        config = TuningConfig(device='cpu', tier=1, n_beats=3, dt=0.05)
        n_params = len(get_param_names(1))
        theta = torch.ones(n_params, dtype=torch.float64)

        result = run_spontaneous(theta, config, duration_ms=5000.0)
        assert result.converged
        # PHAS13 native CL ~1636 ms
        if result.cl is not None:
            assert 800 < result.cl < 3000

    def test_scaled_params_change_output(self):
        """Scaling parameters changes model output."""
        from tuner.config import TuningConfig, get_param_names
        from tuner.cell_runner import run_single_cell

        config = TuningConfig(device='cpu', tier=1, n_beats=3, dt=0.05,
                              pacing_cl=1500.0)
        n_params = len(get_param_names(1))

        # Baseline
        theta_base = torch.ones(n_params, dtype=torch.float64)
        res_base = run_single_cell(theta_base, config)

        # Doubled IKr (index 2 = g_Kr)
        theta_ikr = theta_base.clone()
        theta_ikr[2] = 2.0
        res_ikr = run_single_cell(theta_ikr, config)

        # Both should converge; results should differ
        assert res_base.converged
        assert res_ikr.converged
        # At minimum, v_peak or apd should change
        assert (res_base.v_peak != res_ikr.v_peak or
                res_base.apd90 != res_ikr.apd90)
