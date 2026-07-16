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
            # This smoke test exercises the BO loop machinery, not the ionic
            # backend — pin the fast batched V5.4 path (the cardiac_core AP path
            # is covered by test_cell_runner_cc). The 200-iter BO cell fit is
            # retired under the joint architecture regardless.
            ionic_backend='cardiac_sim',
        )
        targets = TuningTargets(spontaneous_cl=None)  # Skip CL objective

        result = fit_cell(config, targets, n_initial=4, n_iterations=3,
                          verbose=True)

        assert result.pareto_X.shape[0] >= 1
        assert result.pareto_Y.shape[0] >= 1
        # fit_cell batches q = min(4, max(1, n_iterations - iteration)) candidates
        # per iteration (descending schedule), so with n_iterations=3 the totals are
        # n_initial + (3 + 2 + 1) = 4 + 6 = 10 — not n_initial + n_iterations.
        expected = 4 + sum(min(4, max(1, 3 - i)) for i in range(3))
        assert result.all_X.shape[0] == expected
        assert len(result.param_names) == 6
        assert len(result.objective_names) == 2
