"""Voltage-clamp steady-state integrator (Session 27, Step 2.1).

These tests validate the V5.4 composition path and lock the clamp-rest
artifact's schema. The held-V fixed point differs from
``TTP06_REST_IONIC_STATE`` by design (paced vs. clamped steady state); tests
compute the clamped-rest reference once per session and use it for identity
and perturbation checks.
"""
from __future__ import annotations

import sys

import pytest
import torch

from surrogate.data import voltage_clamp_ss as vcs
from surrogate.data.voltage_clamp_ss import compute_z_ss_grid
from surrogate.model.stage1 import TTP06_REST_IONIC_STATE


@pytest.fixture(scope="module")
def clamped_rest() -> torch.Tensor:
    """z_ss(-85.23) under V-clamp — computed once per test module (slow)."""
    out = compute_z_ss_grid([-85.23])
    assert out["converged"].all(), "Fixture integration should converge"
    return out["z_ss_grid"][0]


def test_z_ss_dtype_shape():
    out = compute_z_ss_grid([-85.23])
    assert out["V_grid"].shape == (1,)
    assert out["z_ss_grid"].shape == (1, 14)
    assert out["V_grid"].dtype == torch.float64
    assert out["z_ss_grid"].dtype == torch.float64
    assert out["converged"].dtype == torch.bool


def test_z_ss_converged_early():
    out = compute_z_ss_grid([-85.23], max_t_ms=2000.0)
    assert out["converged"].all(), (
        "Clamped integration must converge before max_t_ms; silent timeout "
        "would return a drifting pseudo-state."
    )


def test_z_ss_from_rest_is_stable(clamped_rest: torch.Tensor):
    """Integrator is stable when started at the clamped fixed point."""
    out = compute_z_ss_grid([-85.23], initial_state=clamped_rest.clone())
    assert out["converged"].all()
    diff = (out["z_ss_grid"][0] - clamped_rest).abs().max().item()
    assert diff < 1e-3, f"Clamped rest is not a fixed point — drifted {diff:.3e}"


def test_z_ss_from_perturbed_initial_converges_to_rest(clamped_rest: torch.Tensor):
    """A ~5% perturbation in each dim relaxes back to the clamped fixed point.

    Uses a fixed-seed generator so the test is reproducible regardless of
    which tests ran before this one. Perturbs relative to ``clamped_rest``
    (the real attractor under clamp) rather than ``TTP06_REST_IONIC_STATE``
    (paced-steady-state values — not a fixed point of the clamped system).
    """
    gen = torch.Generator().manual_seed(0xC0FFEE)
    perturbation = 0.05 * torch.randn(14, generator=gen, dtype=torch.float64)
    perturbed = clamped_rest + perturbation
    out = compute_z_ss_grid([-85.23], initial_state=perturbed)
    assert out["converged"].all()
    diff = (out["z_ss_grid"][0] - clamped_rest).abs().max().item()
    assert diff < 1e-1, (
        f"Perturbed start did not relax near clamped rest (diff {diff:.3e}). "
        "Broken RHS or wrong clamp values would fail this."
    )


def test_z_ss_uses_v54_rhs():
    """Clamp module IS linked to V5.4's TTP06 RHS, not a hand-rolled one."""
    assert "cardiac_sim.ionic.ttp06.model" in sys.modules, (
        "voltage_clamp_ss.py must import V5.4's TTP06Model, not reimplement it"
    )
    # The adapter names are the three primitive wrappers plus _step/_rate_14.
    # Any symbol starting with _TTP06_RHS_ or tokens like 'rhs' at module level
    # would indicate an in-file reimplementation — assert the inventory.
    module_symbols = [name for name in dir(vcs) if not name.startswith("__")]
    for name in module_symbols:
        assert not name.startswith("_TTP06_RHS_"), (
            f"Unexpected RHS-like symbol {name!r} suggests reimplementation"
        )


def test_z_ss_cached_artifact(tmp_path, monkeypatch):
    """`python -m surrogate.data.voltage_clamp_ss` writes the expected schema."""
    monkeypatch.setattr(vcs, "_DEFAULT_OUT", tmp_path / "z_ss_grid.pt")
    vcs.main()
    loaded = torch.load(tmp_path / "z_ss_grid.pt", weights_only=False)
    assert set(loaded.keys()) == {"V_grid", "z_ss_grid", "converged"}
    assert loaded["V_grid"].tolist() == [-85.23]
    assert loaded["z_ss_grid"].shape == (1, 14)
    assert loaded["z_ss_grid"].dtype == torch.float64


def test_rest_attractor_anchor_defined():
    """`TTP06_REST_IONIC_STATE` is the 14-dim anchor referenced by the clamp."""
    assert TTP06_REST_IONIC_STATE.shape == (14,)
    assert TTP06_REST_IONIC_STATE.dtype == torch.float64
