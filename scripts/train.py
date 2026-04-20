"""Entry point for cardiac_ml training.

Usage:
    python scripts/train.py experiment=<name>
    python scripts/train.py experiment=ionic_node_t1 training.optimizer.lr=5e-4
    python scripts/train.py --cfg job    # dump composed config

PEP 562 note: `cardiac_ml.__init__.py` uses lazy import; `Trainer` is
resolved only on first access.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path so `cardiac_ml` resolves when invoked as a script
# (`python scripts/train.py`). When invoked via `python -m scripts.train`
# from project root this is redundant but harmless.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# Also Surrogate/ for model `_target_`s.
_SURROGATE = _ROOT / "Surrogate"
if _SURROGATE.is_dir() and str(_SURROGATE) not in sys.path:
    sys.path.insert(0, str(_SURROGATE))

import hydra
from omegaconf import DictConfig

from cardiac_ml import Trainer
from cardiac_ml.conf_schemas import _register

# Round-3 MED-6: register structured configs with Hydra's ConfigStore at
# entry-point import time, BEFORE @hydra.main resolves the defaults list.
# Must not live in cardiac_ml/__init__.py (would break lazy import pattern).
_register()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
