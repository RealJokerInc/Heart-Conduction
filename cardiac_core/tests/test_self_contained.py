"""cardiac_core is self-contained.

Durable guard — fails if any future edit reintroduces a cross-folder import into the original
engine trees or the deleted `_prepare_engine` sys.modules hack. Matches real IMPORT statements and
the hack token, NOT docstring/path-string mentions, so it won't false-positive on prose.
"""

import os
import re
import importlib

import pytest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../cardiac_core

# A real cross-folder import into an original engine package (`cardiac_sim` or LBM's `src`).
_BAD_IMPORT = re.compile(r'^\s*(from|import)\s+(cardiac_sim|src)\b')
# The deleted hack — only a real CALL/def (`_prepare_engine(`) is a regression; prose mentions
# ("no _prepare_engine)", "via _prepare_engine") are fine and are ignored (call form requires `(`).
_HACK = re.compile(r'_prepare_engine\(')

# Files allowed to mention the originals:
#  - this guard file itself (contains the patterns/messages)
#  - the firewall-gate driver, which deliberately subprocess-drives the original V5.5 cable harness
_EXCLUDE = {"test_self_contained.py"}


def _iter_py():
    for dp, dn, fn in os.walk(PKG_ROOT):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py") and f not in _EXCLUDE:
                yield os.path.join(dp, f)


def test_no_cross_folder_imports():
    bad = []
    for path in _iter_py():
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if _BAD_IMPORT.match(line):
                    bad.append(f"{os.path.relpath(path, PKG_ROOT)}:{i}: {line.strip()}")
    assert not bad, "cardiac_core imports the original engine trees:\n" + "\n".join(bad)


def test_no_prepare_engine_hack():
    bad = []
    for path in _iter_py():
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                code = line.split("#", 1)[0]   # ignore inline comments
                if _HACK.search(code):
                    bad.append(f"{os.path.relpath(path, PKG_ROOT)}:{i}: {line.strip()}")
    assert not bad, "_prepare_engine hack reintroduced:\n" + "\n".join(bad)


@pytest.mark.parametrize("sub", ["ionic", "mesh", "stimulus", "fields", "video", "image",
                                 "_monodomain", "_bidomain", "_lbm"])
def test_subpackage_importable(sub):
    importlib.import_module(f"cardiac_core.{sub}")
