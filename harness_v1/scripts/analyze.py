"""Post-hoc SHAP analysis of a trained NODE checkpoint.

Usage:
    python scripts/analyze.py experiment=ionic_node_t1 \
        run_id=<mlflow_run_id> \
        output_dir=./shap_out \
        shap_nsamples=20

Loads `best.pt` from the specified MLflow run, rebuilds the model from the
composed config, and saves a V-only SHAP summary PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SURROGATE = _ROOT / "Surrogate"
if _SURROGATE.is_dir() and str(_SURROGATE) not in sys.path:
    sys.path.insert(0, str(_SURROGATE))

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cardiac_ml.analysis.shap_utils import kernel_shap_v_only, plot_shap_summary
from cardiac_ml.conf_schemas import _register

_register()


def _sample_v_trajectories(T: int = 300, N: int = 3, dtype=torch.float64) -> torch.Tensor:
    """Synthetic V trajectories: resting voltage + small Gaussian perturbation.

    Replaced with real beats from the data loader for production analysis.
    """
    rng = torch.Generator().manual_seed(0)
    base = torch.full((T,), -85.0, dtype=dtype)
    return base + 2.0 * torch.randn(N, T, generator=rng, dtype=dtype)


def _resting_v_baseline(T: int = 300, M: int = 2, dtype=torch.float64) -> torch.Tensor:
    return torch.full((M, T), -85.0, dtype=dtype)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_id = cfg.get("run_id")
    assert run_id, "Required: run_id=<mlflow_run_id> (see mlflow ui)"

    output_dir = Path(cfg.get("output_dir", "./shap_out"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rebuild model from the experiment's config.
    device = torch.device(cfg.training.device)
    dtype = getattr(torch, cfg.training.dtype)
    model = instantiate(cfg.model).to(device=device, dtype=dtype)

    # 2. Load best.pt from the MLflow run.
    import mlflow
    tracking_uri = cfg.tracking.get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    ckpt_path = Path(client.download_artifacts(run_id, "best.pt", str(output_dir)))
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # ModelCheckpoint saves a plain state_dict (not a {stage1_state_dict: ...} wrapper).
    if isinstance(state, dict) and "stage1_state_dict" in state:
        model.stage1.load_state_dict(state["stage1_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # 3. Build V samples + baseline.
    T_ms = int(cfg.get("shap_T_ms", 300))
    N = int(cfg.get("shap_N", 3))
    M = int(cfg.get("shap_M", 2))
    V_samples = _sample_v_trajectories(T=T_ms, N=N, dtype=dtype)
    baseline = _resting_v_baseline(T=T_ms, M=M, dtype=dtype)

    # 4. Run SHAP + plot.
    nsamples = int(cfg.get("shap_nsamples", 20))
    print(f"Running KernelSHAP: N={N}, T={T_ms}, baseline M={M}, nsamples={nsamples}")
    explanation = kernel_shap_v_only(model, V_samples, baseline, nsamples=nsamples)
    png_path = output_dir / "shap_v_only.png"
    plot_shap_summary(explanation, str(png_path))
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
