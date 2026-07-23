"""Test-suite fixtures for cardiac_core.

Keeps the suite from writing simulation media into the source tree: several visualization and
video tests render real output through ``media_path``, whose default root is the caller's
project directory. Redirecting it to a temporary directory for the whole session makes the
suite self-contained and leaves no artifacts behind.
"""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _media_root(tmp_path_factory):
    """Send all media written during the test session to a temporary directory."""
    root = tmp_path_factory.mktemp("cardiac_core_media")
    previous = os.environ.get("CARDIAC_MEDIA_ROOT")
    os.environ["CARDIAC_MEDIA_ROOT"] = str(root)
    try:
        yield root
    finally:
        if previous is None:
            os.environ.pop("CARDIAC_MEDIA_ROOT", None)
        else:
            os.environ["CARDIAC_MEDIA_ROOT"] = previous
