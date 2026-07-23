"""Integrity gate for the vendored engines.

Each engine must reproduce its recorded golden output BIT-IDENTICALLY (atol=0), so that a
refactor cannot silently change numerics.

Goldens are captured by `_integrity/make_goldens.py`. These tests must stay green — a failure
means the code move changed numerics (a behavioral regression) or touched an original tree.

An intentional numerics change must regenerate the goldens in the SAME change, via
`make_goldens.py`. Missing goldens hard-FAIL rather than skip, so that deleting a golden
cannot silently disable the drift guard.
"""

import os

import torch
import pytest

from cardiac_core.tests._integrity.make_goldens import canonical_sim, HERE


def _check_golden(engine: str):
    path = os.path.join(HERE, f"golden_{engine}.pt")
    if not os.path.exists(path):
        pytest.fail(f"golden_{engine}.pt missing — regenerate via _integrity/make_goldens.py "
                    f"(a deleted golden silently disables the numerics drift guard)")
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

