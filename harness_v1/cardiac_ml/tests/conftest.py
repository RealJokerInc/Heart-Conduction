"""pytest fixtures for cardiac_ml tests.

Round-2 M-4 + Round-3 MED-10: `mlflow_tmpdir` is `autouse=True, scope="session"`
so ALL tests automatically pick it up without explicit injection. Prevents
tests from polluting the top-level `./mlruns/` directory.

Subprocess-based tests (Step 3.6 end-to-end) must STILL pass
`tracking.tracking_uri=./mlruns_test` as a Hydra CLI override — the
subprocess is a separate Python process and doesn't inherit pytest
fixtures.
"""
from __future__ import annotations

import os

import mlflow
import pytest


@pytest.fixture(autouse=True, scope="session")
def mlflow_tmpdir(tmp_path_factory):
    """Redirect MLflow's tracking URI to a per-session tmpdir for all tests."""
    root = tmp_path_factory.mktemp("mlruns")
    uri = f"file:{root}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    yield root
    # Cleanup happens via tmp_path_factory's own teardown.
