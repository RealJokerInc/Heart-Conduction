"""SHAP utilities for trained NODE checkpoints.

KernelExplainer-only (per OPEN-4 + audit H-4). DeepExplainer is incompatible
with torchdiffeq's gradient routing (adjoint replaces `.backward()`), so we
treat the model as a black-box `V -> z_final` function.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    import shap
except ImportError:  # pragma: no cover — shap pinned in env
    shap = None  # type: ignore[assignment]


def _run_node_to_final_z(model: torch.nn.Module, V: torch.Tensor) -> torch.Tensor:
    """Integrate IonicNODE with a single V trajectory and return final state.

    Uses the NODE's own `integrate` + V-traj lifecycle. V is expected as a
    1D tensor of length T (ms sampling implicit via set_v_trajectory).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    V = V.to(device=device, dtype=dtype)
    T = V.shape[-1]
    # Uniform 0.1-ms sampling — matches run_multi_bcl.py default beat resolution.
    dt = torch.full((T,), 0.1, dtype=dtype, device=device)
    t_grid = torch.cat([torch.zeros(1, dtype=dtype, device=device), dt.cumsum(0)])
    model.set_v_trajectory(V.unsqueeze(0) if V.dim() == 1 else V, t_grid)

    B = 1 if V.dim() == 1 else V.shape[0]
    carried_dim = model.stage1.carried_dim
    z0 = torch.zeros(B, carried_dim, dtype=dtype, device=device)
    # Resting concentrations (Layer 0 physics).
    INIT_CONC = torch.tensor([10.0, 138.0, 0.0001, 0.0002], dtype=dtype, device=device)
    z0[:, model.stage1.ionic_dim:] = INIT_CONC

    t_eval = torch.tensor([0.0, t_grid[-1].item()], dtype=dtype, device=device)
    z_traj = model.integrate(z0, t_eval, method="dopri5", rtol=1e-3, atol=1e-3,
                             adjoint=False)
    model.clear_v_trajectory()
    return z_traj[-1].squeeze(0)  # (carried_dim,)


def kernel_shap_v_only(
    model: torch.nn.Module,
    V_samples: torch.Tensor,
    baseline: torch.Tensor,
    nsamples: int = 50,
) -> Any:
    """V-only KernelExplainer over a trained NODE checkpoint.

    Args:
        model: eval-mode IonicNODE.
        V_samples: (N, T) voltage trajectories to explain.
        baseline: (M, T) reference trajectories (e.g. resting-V).
        nsamples: number of coalition samples per prediction — trades cost
            for SHAP-value fidelity.
    Returns:
        shap.Explanation (values shape matches V_samples, plus output dim).
    """
    if shap is None:  # pragma: no cover
        raise ImportError("shap is not installed; `pip install shap` to use this path.")
    model.eval()

    def predict(V_batch_np: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for V_row in V_batch_np:
                z_final = _run_node_to_final_z(
                    model, torch.from_numpy(V_row).double()
                )
                outputs.append(z_final.detach().cpu().numpy())
        return np.stack(outputs)

    explainer = shap.KernelExplainer(predict, baseline.cpu().numpy())
    shap_values = explainer.shap_values(V_samples.cpu().numpy(), nsamples=nsamples)
    return shap.Explanation(
        values=shap_values,
        data=V_samples.cpu().numpy(),
        feature_names=[f"V[t={t}]" for t in range(V_samples.shape[-1])],
    )


def plot_shap_summary(explanation: Any, output_path: str) -> None:
    """Save a SHAP summary plot as PNG at `output_path`."""
    if shap is None:  # pragma: no cover
        raise ImportError("shap is not installed; `pip install shap` to use this path.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # KernelExplainer returns (N, T) or (N, T, output_dim). Flatten output dim
    # by summing |shap| across outputs for a single summary view.
    vals = explanation.values
    if isinstance(vals, list):
        vals = np.stack(vals, axis=-1)
    if vals.ndim == 3:
        vals = np.abs(vals).sum(axis=-1)
    shap.summary_plot(
        vals, explanation.data, feature_names=explanation.feature_names,
        show=False,
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close("all")
