"""Step 4.2 tests — NODE configs + multi_bcl_loader sanity.

Covers:
- conf/experiment/ionic_node_t1 composes; training.phase_name == "A1".
- cardiac_ml.data.multi_bcl_loader imports; make_loader is callable.
- _extract_beats produces correct shape / metadata for a synthetic T1 dict.
- _single_beat_collate unsqueezes tensors, preserves non-tensor metadata.
- min_beat filter matches oracle (`_beat >= 15` → last 5 of 20).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from cardiac_ml.conf_schemas import _register
from cardiac_ml.data.multi_bcl_loader import (
    _extract_beats,
    _single_beat_collate,
    make_loader,
    MultiBCLBeatDataset,
)

CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")


@pytest.fixture(autouse=True)
def _register_and_reset_hydra():
    _register()
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_ionic_node_t1_composes():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=ionic_node_t1"])
    assert cfg.training.phase_name == "A1"
    assert cfg.training.optimizer._target_ == "torch.optim.AdamW"
    assert "node_train_step" in cfg.training.train_step_fn.path
    assert cfg.data.train._target_.endswith("multi_bcl_loader.make_loader")


def test_multi_bcl_loader_importable():
    mod = importlib.import_module("cardiac_ml.data.multi_bcl_loader")
    assert callable(mod.make_loader)
    assert hasattr(mod, "MultiBCLBeatDataset")


def _synthetic_t1(bcls: list[int], n_beats: int = 20) -> dict:
    """Build a T1-like dict sized for `bcls` at 0.01-ms resolution."""
    total_steps = sum(int(b / 0.01) for b in bcls) * n_beats
    return {
        "Vm": torch.linspace(-85.0, 20.0, total_steps, dtype=torch.float64),
        "dt": torch.full((total_steps,), 0.01, dtype=torch.float64),
        "ionic_states": torch.zeros(total_steps, 14, dtype=torch.float64),
        "concentrations": torch.zeros(total_steps, 4, dtype=torch.float64),
        "conductance_products": torch.zeros(total_steps, 5, dtype=torch.float64),
    }


def test_extract_beats_shape():
    """One BCL=500ms → 500ms / 0.1ms = 5000 subsampled points per beat."""
    data = _synthetic_t1(bcls=[500], n_beats=2)
    beats = _extract_beats(data, bcls=[500], n_beats=2)
    assert len(beats) == 2
    for beat in beats:
        for key in ("Vm", "dt", "ionic_states", "concentrations",
                    "conductance_products"):
            assert key in beat
        assert beat["Vm"].shape[0] == 5000
        assert beat["_bcl"] == 500
        assert beat["_tier"] == "T1"
    # dt scaled by SUBSAMPLE=10 → 0.1 ms per step.
    assert torch.allclose(beats[0]["dt"][:5],
                          torch.full((5,), 0.1, dtype=torch.float64))


def test_min_beat_filter_matches_oracle():
    """`min_beat=15` over `n_beats=20` keeps beats 15-19 (5 per BCL)."""
    data = _synthetic_t1(bcls=[500], n_beats=20)
    beats = _extract_beats(data, bcls=[500], n_beats=20, min_beat=15)
    assert len(beats) == 5
    assert [b["_beat"] for b in beats] == [15, 16, 17, 18, 19]


def test_collate_unsqueezes_batch_dim():
    seg = {
        "Vm": torch.zeros(10, dtype=torch.float64),
        "ionic_states": torch.zeros(10, 14, dtype=torch.float64),
        "_bcl": 500,
        "_tier": "T1",
    }
    out = _single_beat_collate([seg])
    assert out["Vm"].shape == (1, 10)
    assert out["ionic_states"].shape == (1, 10, 14)
    assert out["Vm"].dtype == torch.float64
    assert out["_bcl"] == 500
    assert out["_tier"] == "T1"


def test_make_loader_end_to_end(tmp_path):
    """Write a tiny synthetic cache, load via make_loader, iterate once."""
    data = _synthetic_t1(bcls=[500], n_beats=2)
    cache_path = tmp_path / "synthetic_tier01.pt"
    torch.save(data, cache_path)

    loader = make_loader(cache_path, bcls=[500], n_beats=2, min_beat=0,
                         batch_size=1, shuffle=False)
    batch = next(iter(loader))
    assert batch["Vm"].shape == (1, 5000)
    assert batch["_bcl"] == 500


def test_make_loader_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="cache not found"):
        make_loader(tmp_path / "does_not_exist.pt", bcls=[500], n_beats=1)
