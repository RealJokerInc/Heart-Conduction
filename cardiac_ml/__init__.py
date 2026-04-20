"""cardiac_ml: project-wide ML training harness.

Public API surface is deliberately small — only `Trainer` is re-exported.
Access is lazy via PEP 562 `__getattr__` so sibling imports (e.g.
`cardiac_ml.training`, `cardiac_ml.utils`) work before Step 3.4 lands
the Trainer implementation.
"""
from __future__ import annotations

__all__ = ["Trainer"]


def __getattr__(name: str):
    if name == "Trainer":
        try:
            from cardiac_ml.training.trainer import Trainer
        except ImportError as e:
            raise ImportError(
                "cardiac_ml.Trainer is not yet implemented. "
                "See PLAN.md Step 3.4. Underlying error: " + str(e)
            ) from e
        return Trainer
    raise AttributeError(f"module 'cardiac_ml' has no attribute {name!r}")
