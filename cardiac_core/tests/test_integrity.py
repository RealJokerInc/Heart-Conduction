"""Integrity gate (engine_consolidation Phase 0+).

After each vendoring phase, the vendored engine must reproduce its pre-vendor golden output
BIT-IDENTICALLY (atol=0), and the original engine source trees must stay byte-unchanged.

Captured by `_integrity/make_goldens.py`. These tests run every phase and must stay green —
a failure means the code-move changed numerics (behavioral regression) or touched an original.

Vendoring policy (2026-06-30): the vendored `cardiac_core` tree is the LIVING source and an
intentional fix may edit an original in place (kept in sync) — regenerate the goldens + hashes
in the SAME change via `make_goldens.py`. Missing goldens hard-FAIL (not skip) so a deleted
golden cannot silently disable the drift guard (Audit #42).
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
        pytest.fail(f"golden_{engine}.pt missing — regenerate via _integrity/make_goldens.py "
                    f"(a deleted golden silently disables the numerics drift guard, Audit #42)")
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
        pytest.fail("engine_src_sha.json missing — regenerate via _integrity/make_goldens.py (Audit #42)")
    baseline = json.load(open(path))
    for e, src in ENGINE_SRC.items():
        assert tree_hash(src) == baseline[e], (
            f"{e} ORIGINAL source tree changed ({src}) — vendoring must be copy-only"
        )
