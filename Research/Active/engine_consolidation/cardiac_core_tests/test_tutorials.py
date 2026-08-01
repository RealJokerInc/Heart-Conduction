"""Anti-rot gate for the tutorial notebooks, exposed to pytest / ``/verify``.

Executing all 12 tutorial notebooks is a few minutes of real simulation, so this
is **skipped by default** — it would dominate the fast unit-test suite. Opt in to
run it (e.g. before a release, or in CI):

    CARDIAC_RUN_TUTORIAL_GATE=1 pytest cardiac_core/tests/test_tutorials.py

For ad-hoc use, run the standalone runner instead:

    python cardiac_core/tutorials/run_all_tutorials.py

Each notebook is executed on an in-memory copy (its committed empty outputs are
left untouched); a broken cell fails the corresponding parametrized test.
"""
import glob
import os

import pytest

_TUTORIALS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "cardiac_core_tutorials"))
_NOTEBOOKS = sorted(glob.glob(os.path.join(_TUTORIALS, "*.ipynb")))

pytestmark = pytest.mark.skipif(
    os.environ.get("CARDIAC_RUN_TUTORIAL_GATE") != "1",
    reason="tutorial execute-all gate is slow (~9 min); set CARDIAC_RUN_TUTORIAL_GATE=1 to run",
)


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=[os.path.basename(p) for p in _NOTEBOOKS])
def test_tutorial_notebook_executes(nb_path):
    """Each tutorial notebook runs top-to-bottom with no cell errors."""
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbconvert")
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    # Raises CellExecutionError (→ test failure) if any cell errors; in-memory only.
    ep.preprocess(nb, {"metadata": {"path": _TUTORIALS}})
