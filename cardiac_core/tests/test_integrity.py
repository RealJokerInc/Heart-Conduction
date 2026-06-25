"""Integrity gate (engine_consolidation Phase 0+).

After each vendoring phase, the vendored engine must reproduce its pre-vendor golden output
BIT-IDENTICALLY (atol=0), and the original engine source trees must stay byte-unchanged.

Captured by `_integrity/make_goldens.py`. These tests run every phase and must stay green —
a failure means the code-move changed numerics (behavioral regression) or touched an original.
"""

import os
import json

import torch
import pytest

from cardiac_core.tests._integrity.make_goldens import (
    canonical_sim, tree_hash, ENGINE_SRC, HERE,
)


def _check_golden(engine: str):
    path = os.path.join(HERE, f"golden_{engine}.pt")
    if not os.path.exists(path):
        pytest.skip(f"golden_{engine}.pt not captured — run _integrity/make_goldens.py")
    g = torch.load(path, weights_only=False)
    r = canonical_sim(engine)
    assert torch.equal(r.times.cpu(), g["times"]), f"{engine}: times drifted from golden"
    assert torch.equal(r.Vm.cpu(), g["Vm"]), f"{engine}: Vm NOT bit-identical to pre-vendor golden"
    if "phi_e" in g:
        assert torch.equal(r.phi_e.cpu(), g["phi_e"]), f"{engine}: phi_e NOT bit-identical to golden"


def test_monodomain_matches_golden():
    _check_golden("monodomain")


def test_bidomain_matches_golden():
    _check_golden("bidomain")


def test_lbm_matches_golden():
    _check_golden("lbm")


def test_originals_untouched():
    path = os.path.join(HERE, "engine_src_sha.json")
    if not os.path.exists(path):
        pytest.skip("engine_src_sha.json not captured — run _integrity/make_goldens.py")
    baseline = json.load(open(path))
    for e, src in ENGINE_SRC.items():
        assert tree_hash(src) == baseline[e], (
            f"{e} ORIGINAL source tree changed ({src}) — vendoring must be copy-only"
        )
