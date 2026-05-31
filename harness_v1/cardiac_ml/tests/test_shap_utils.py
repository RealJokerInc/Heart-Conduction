"""Step 5.2 tests — SHAP utils smoke.

Uses minimal dimensions (T=5 steps, N=2 samples, M=1 baseline, nsamples=3) to
keep runtime under a few seconds. Full-scale SHAP is O(nsamples * (N+M))
NODE forward passes — see PLAN.md Step 5.2 Risk.
"""
from __future__ import annotations

import pytest
import torch

from cardiac_ml.analysis.shap_utils import (
    _run_node_to_final_z,
    kernel_shap_v_only,
    plot_shap_summary,
)


def _make_tiny_node():
    from surrogate.model.node import IonicNODE
    from surrogate.model.stage1 import IonicStage1
    stage1 = IonicStage1().to(dtype=torch.float64)
    return IonicNODE(stage1).to(dtype=torch.float64)


def test_run_node_to_final_z_shape():
    """Minimal integrate call returns (carried_dim,) final state."""
    node = _make_tiny_node()
    V = torch.linspace(-85.0, 20.0, 5, dtype=torch.float64)
    z_final = _run_node_to_final_z(node, V)
    assert z_final.shape == (node.stage1.carried_dim,)


@pytest.mark.slow
def test_kernel_shap_runs():
    """Sanity: KernelExplainer returns a shap.Explanation with matching data."""
    shap = pytest.importorskip("shap")
    node = _make_tiny_node()
    V_samples = torch.linspace(-85.0, 20.0, 5, dtype=torch.float64).repeat(2, 1)
    baseline = torch.full((1, 5), -85.0, dtype=torch.float64)
    explanation = kernel_shap_v_only(node, V_samples, baseline, nsamples=3)
    assert hasattr(explanation, "values")
    assert hasattr(explanation, "data")
    # data matches input shape
    assert explanation.data.shape == V_samples.shape


def test_no_deep_explainer_usage():
    """Enforce KernelExplainer-only policy (H-4). Docstring mentions OK;
    actual invocation (`shap.DeepExplainer(...)`) is not."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "analysis" / "shap_utils.py").read_text()
    assert "shap.DeepExplainer" not in src, (
        "shap_utils.py must not call shap.DeepExplainer — use KernelExplainer only."
    )


@pytest.mark.slow
def test_plot_summary_writes_png(tmp_path):
    """plot_shap_summary writes a non-empty PNG."""
    shap = pytest.importorskip("shap")
    node = _make_tiny_node()
    V_samples = torch.linspace(-85.0, 20.0, 5, dtype=torch.float64).repeat(2, 1)
    baseline = torch.full((1, 5), -85.0, dtype=torch.float64)
    explanation = kernel_shap_v_only(node, V_samples, baseline, nsamples=3)
    png = tmp_path / "summary.png"
    plot_shap_summary(explanation, str(png))
    assert png.is_file() and png.stat().st_size > 0
