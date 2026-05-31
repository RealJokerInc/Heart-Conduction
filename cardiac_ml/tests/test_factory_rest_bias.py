"""Rest-bias pin invariants for the ionic-node factory (Session 27, Step 3.0).

``IonicStage1._init_weights`` freezes ``ionic_state_decoder.bias`` at
``TTP06_REST_IONIC_STATE`` so ``decoder(z=0) = rest`` by construction. The cold
path is safe, but ``load_state_dict`` silently overwrites the bias with the
checkpoint value. The factory must re-pin after any warm start; these tests
lock that contract.
"""
from __future__ import annotations

import torch

from cardiac_ml.model.ionic_node_factory import make_node
from surrogate.model.stage1 import IonicStage1, TTP06_REST_IONIC_STATE


def _expected_rest(bias: torch.Tensor) -> torch.Tensor:
    """Match `pin_rest_bias`'s dtype/device coercion so comparisons don't
    trigger the float32/float64 rest anchor."""
    return TTP06_REST_IONIC_STATE.to(dtype=bias.dtype, device=bias.device)


def test_factory_pins_rest_bias_cold_start():
    node = make_node(scaffold=True)
    bias = node.stage1.ionic_state_decoder.bias
    assert bias.requires_grad is False
    assert torch.allclose(bias.detach(), _expected_rest(bias), atol=1e-6)


def test_factory_pins_rest_bias_after_warm_start(tmp_path):
    donor = IonicStage1(scaffold=True)
    # Corrupt the donor's bias so the warm-load would regress the invariant
    # unless the factory re-pins.
    with torch.no_grad():
        donor.ionic_state_decoder.bias.copy_(
            torch.randn(TTP06_REST_IONIC_STATE.numel(), dtype=donor.ionic_state_decoder.bias.dtype)
        )
    donor.ionic_state_decoder.bias.requires_grad_(True)

    ckpt_path = tmp_path / "warm_start.pt"
    torch.save({"stage1_state_dict": donor.state_dict()}, ckpt_path)

    node = make_node(scaffold=True, stage1_ckpt=str(ckpt_path))
    bias = node.stage1.ionic_state_decoder.bias
    assert bias.requires_grad is False, "Factory must re-freeze bias after load_state_dict"
    assert torch.allclose(bias.detach(), _expected_rest(bias), atol=1e-6), (
        "Factory must overwrite the corrupted bias with TTP06_REST_IONIC_STATE"
    )


def test_v4_stage1_loads_via_factory():
    """Factory wraps a v4 IonicStage1 (ionic_dim=20, state_rate_mlp present)."""
    node = make_node(scaffold=True)
    stage1 = node.stage1
    assert stage1.ionic_dim == 20, (
        f"Expected ionic_dim=20 (v4), got {stage1.ionic_dim} — regression?"
    )
    assert stage1.carried_dim == 24
    assert hasattr(stage1, "state_rate_mlp"), (
        "v4 stage1 must expose state_rate_mlp — factory regressed to v3 arch"
    )
    assert not hasattr(stage1, "ionic_rate_mlp"), (
        "v3 ionic_rate_mlp must not appear on v4 stage1"
    )
    assert stage1.ionic_state_decoder.bias.requires_grad is False
