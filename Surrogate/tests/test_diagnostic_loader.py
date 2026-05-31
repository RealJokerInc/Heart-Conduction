"""Diagnostic-loader dual-format tests (Session 27, Step 3.2).

The ``integrator_error_budget.py`` diagnostic must accept both the v3 wrapper
format (``{"stage1_state_dict": ...}``) and cardiac_ml's flat ``IonicNODE``
``state_dict()``. After loading, it must re-pin the rest-attractor bias.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from surrogate.model.node import IonicNODE
from surrogate.model.stage1 import IonicStage1, TTP06_REST_IONIC_STATE

# The diagnostic lives at `Surrogate/diagnostics/integrator_error_budget.py`
# (top-level, outside the `surrogate` package). Load it by file path so tests
# don't depend on `diagnostics` becoming an importable package.
_DIAG_PATH = Path(__file__).resolve().parents[1] / "diagnostics" / "integrator_error_budget.py"
_spec = importlib.util.spec_from_file_location("_diag_mod", _DIAG_PATH)
_diag = importlib.util.module_from_spec(_spec)
sys.modules["_diag_mod"] = _diag
_spec.loader.exec_module(_diag)
_load_stage1_state = _diag._load_stage1_state


def _expected_rest(bias: torch.Tensor) -> torch.Tensor:
    return TTP06_REST_IONIC_STATE.to(dtype=bias.dtype, device=bias.device)


def test_diagnostic_loads_legacy_ckpt(tmp_path):
    """v3 wrapper format round-trips through the loader."""
    donor = IonicStage1(scaffold=True)
    target = IonicStage1(scaffold=True)
    _load_stage1_state({"stage1_state_dict": donor.state_dict()}, target)
    ref = list(donor.state_dict().items())[0][0]
    assert torch.allclose(
        target.state_dict()[ref], donor.state_dict()[ref]
    )


def test_diagnostic_loads_cardiac_ml_ckpt(tmp_path):
    """Flat ``IonicNODE.state_dict()`` with ``stage1.*`` prefix round-trips."""
    donor = IonicStage1(scaffold=True)
    node = IonicNODE(donor)
    target = IonicStage1(scaffold=True)
    _load_stage1_state(node.state_dict(), target)
    # weight round-tripped (not bias — bias is frozen at rest on both sides)
    assert torch.allclose(
        target.ionic_state_decoder.weight, donor.ionic_state_decoder.weight
    )


def test_diagnostic_pin_rest_bias_after_load():
    """After a bad-bias load + re-pin, bias equals TTP06_REST_IONIC_STATE."""
    donor = IonicStage1(scaffold=True)
    with torch.no_grad():
        donor.ionic_state_decoder.bias.copy_(
            torch.randn(14, dtype=donor.ionic_state_decoder.bias.dtype)
        )
    donor.ionic_state_decoder.bias.requires_grad_(True)
    target = IonicStage1(scaffold=True)
    _load_stage1_state({"stage1_state_dict": donor.state_dict()}, target)
    target.pin_rest_bias()
    bias = target.ionic_state_decoder.bias
    assert bias.requires_grad is False
    assert torch.allclose(bias.detach(), _expected_rest(bias), atol=1e-6)
