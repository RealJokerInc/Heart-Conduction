"""Factory builders for IonicNODE — supports optional warm-start from a
stage1-only checkpoint.

Mirrors Surrogate/run_multi_bcl.py:88-101: instantiate `IonicStage1(scaffold=True)`,
optionally load `ckpt["stage1_state_dict"]`, wrap with `IonicNODE(stage1)`.

Used by conf/model/ionic_node.yaml via Hydra `_target_`:

    _target_: cardiac_ml.model.ionic_node_factory.make_node
    scaffold: true
    stage1_ckpt: ${oc.env:WARM_START_CKPT,null}
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from surrogate.model.node import IonicNODE
from surrogate.model.stage1 import IonicStage1


def make_node(
    scaffold: bool = True,
    stage1_ckpt: Optional[str] = None,
) -> IonicNODE:
    """Instantiate IonicNODE, optionally warm-starting stage1 from `stage1_ckpt`.

    Args:
        scaffold: pass-through to IonicStage1 constructor.
        stage1_ckpt: path to a `.pt` file containing `"stage1_state_dict"`.
            If None (or empty / "null"), skip the warm start. Missing files
            raise FileNotFoundError to fail loud.
    """
    stage1 = IonicStage1(scaffold=scaffold)
    if stage1_ckpt and str(stage1_ckpt).lower() != "null":
        path = Path(stage1_ckpt)
        if not path.is_file():
            raise FileNotFoundError(
                f"stage1_ckpt not found: {path}. Supply a valid path or set to null."
            )
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        if "stage1_state_dict" not in ckpt:
            raise KeyError(
                f"warm-start ckpt {path} missing 'stage1_state_dict' key "
                f"(found: {list(ckpt.keys())})"
            )
        stage1.load_state_dict(ckpt["stage1_state_dict"])
    return IonicNODE(stage1)
