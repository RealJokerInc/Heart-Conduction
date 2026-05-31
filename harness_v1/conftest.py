"""Project-root pytest conftest.

Adds Surrogate/ to sys.path so `import surrogate` works from anywhere —
needed by cardiac_ml tests that walk `_target_` strings and try to
import them (Round-3 MED-6 mechanism). The Surrogate tree is not
installed as a package; this mirrors the CWD=Surrogate invocation
pattern documented in CLAUDE.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SURROGATE = _ROOT / "Surrogate"
if _SURROGATE.is_dir() and str(_SURROGATE) not in sys.path:
    sys.path.insert(0, str(_SURROGATE))
