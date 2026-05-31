"""Phase-2 import smoke tests.

Covers Step 2.1 checklist: every sub-package importable; top-level
`cardiac_ml.Trainer` access raises a clean ImportError (not AttributeError
or silent None) before Step 3.4 lands the real Trainer.
"""
from __future__ import annotations

import importlib

import pytest


def test_all_init_files_importable():
    """Every sub-package under cardiac_ml/ imports without error."""
    for mod in (
        "cardiac_ml",
        "cardiac_ml.training",
        "cardiac_ml.analysis",
        "cardiac_ml.utils",
        "cardiac_ml.tests",
    ):
        importlib.import_module(mod)


def test_trainer_accessible_via_lazy_getattr():
    """Post Step 3.4: `cardiac_ml.Trainer` resolves cleanly via PEP 562
    `__getattr__`. Was a "raises ImportError" test during Phase 2 when
    trainer.py was still a stub."""
    import cardiac_ml
    from cardiac_ml.training.trainer import Trainer as RealTrainer
    assert cardiac_ml.Trainer is RealTrainer


def test_unknown_attribute_raises_attribute_error():
    """PEP 562 __getattr__ falls through to AttributeError for unknown names."""
    import cardiac_ml
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = cardiac_ml.NonExistent
