"""
Tests for cell_fitter.py — BayesOpt cell fitting (smoke test).
"""

import pytest
import torch


@pytest.mark.slow
class TestCellFitter:
    """Phase III: Cell fitter smoke tests (require BoTorch)."""

    def test_smoke_5_iterations(self):
        """BO loop runs with 5 iterations without error."""
        from tuner.config import TuningConfig, TuningTargets
        from tuner.cell_fitter import fit_cell

        config = TuningConfig(
            device='cpu', tier=1, n_beats=5, dt=0.01,
            pacing_cl=1500.0, n_iterations=5,
        )
        targets = TuningTargets(spontaneous_cl=None)  # Skip CL objective

        result = fit_cell(config, targets, n_initial=4, n_iterations=3,
                          verbose=True)

        assert result.pareto_X.shape[0] >= 1
        assert result.pareto_Y.shape[0] >= 1
        assert result.all_X.shape[0] == 4 + 3  # initial + iterations
        assert len(result.param_names) == 6
        assert len(result.objective_names) == 2
