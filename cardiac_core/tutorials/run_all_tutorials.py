#!/usr/bin/env python
"""Execute-all anti-rot gate for the cardiac_core tutorial notebooks.

Runs every tutorial notebook headless in a fresh kernel via nbconvert, top to
bottom, and FAILS if any cell raises. This is what keeps the notebooks from
silently rotting as cardiac_core changes: a breaking API change surfaces here as
a failed cell instead of as a broken lesson a reader hits.

Each notebook is executed on an IN-MEMORY copy and discarded — the committed
notebooks keep their empty output cells (a reader generates the numbers and
figures themselves by running the notebook).

Usage (from the repo root, in the project environment):
    python cardiac_core/tutorials/run_all_tutorials.py           # all notebooks
    python cardiac_core/tutorials/run_all_tutorials.py 05_       # only names containing "05_"

Exit code 0 = every notebook ran clean; 1 = at least one failed; 2 = setup error.
Requires nbformat + nbconvert (+ the ipykernel-provided "python3" kernel). The
whole series is a few minutes of real simulation, so this is an on-demand / CI
gate, not something to run on every quick edit. The pytest wrapper
(cardiac_core/tests/test_tutorials.py) is skipped unless CARDIAC_RUN_TUTORIAL_GATE=1.
"""
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PER_CELL_TIMEOUT = 600  # seconds per cell; the reentry / dx-sweep cells are the slow ones


def main(argv):
    try:
        import nbformat
        from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
    except ImportError as exc:  # pragma: no cover - environment guard
        print(f"anti-rot gate needs nbformat + nbconvert: {exc}", file=sys.stderr)
        print("install: pip install nbformat nbconvert", file=sys.stderr)
        return 2

    pattern = argv[1] if len(argv) > 1 else ""
    notebooks = sorted(
        p for p in glob.glob(os.path.join(HERE, "*.ipynb")) if pattern in os.path.basename(p)
    )
    if not notebooks:
        print(f"no tutorial notebooks match {pattern!r} in {HERE}", file=sys.stderr)
        return 2

    print(f"anti-rot gate: executing {len(notebooks)} notebook(s) via nbconvert\n")
    failures = []
    for path in notebooks:
        name = os.path.basename(path)
        nb = nbformat.read(path, as_version=4)
        ep = ExecutePreprocessor(timeout=PER_CELL_TIMEOUT, kernel_name="python3")
        t0 = time.time()
        try:
            # In-memory execution: the committed .ipynb is NOT rewritten with outputs.
            ep.preprocess(nb, {"metadata": {"path": HERE}})
            print(f"  PASS   {name:<30} ({time.time() - t0:5.0f}s)")
        except CellExecutionError as exc:
            failures.append(name)
            last = str(exc).strip().splitlines()[-1] if str(exc).strip() else "cell error"
            print(f"  FAIL   {name:<30} ({time.time() - t0:5.0f}s)  {last[:200]}")
        except Exception as exc:  # noqa: BLE001 - report any executor/kernel failure
            failures.append(name)
            print(f"  ERROR  {name:<30} ({time.time() - t0:5.0f}s)  {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"anti-rot gate FAILED: {len(failures)}/{len(notebooks)} broke — {', '.join(failures)}")
        return 1
    print(f"anti-rot gate PASSED: all {len(notebooks)} notebooks executed clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
