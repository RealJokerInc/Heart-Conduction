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


def test_trainer_lazy_raises_clean_import_error():
    """Before Step 3.4, `from cardiac_ml import Trainer` raises ImportError
    with a message pointing at the missing implementation. Must NOT raise
    AttributeError (which would mean PEP 562 __getattr__ isn't wired) and
    must NOT silently bind a placeholder."""
    import cardiac_ml
    with pytest.raises(ImportError, match="not yet implemented"):
        _ = cardiac_ml.Trainer


def test_unknown_attribute_raises_attribute_error():
    """PEP 562 __getattr__ falls through to AttributeError for unknown names."""
    import cardiac_ml
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = cardiac_ml.NonExistent
