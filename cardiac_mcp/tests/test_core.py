"""Tests for cardiac_mcp.core — the gate logic (fast) + one end-to-end simulate (slow).

The gate tests are pure (no cardiac_core), so they run instantly. They redirect core.LAB /
core.NOTEBOOK to a tmp dir so the real Lab/ is never touched.
"""
from __future__ import annotations

import base64
import json

import pytest

from cardiac_mcp import core


@pytest.fixture
def tmp_lab(tmp_path, monkeypatch):
    lab = tmp_path / "Lab"
    lab.mkdir()
    monkeypatch.setattr(core, "LAB", lab)
    monkeypatch.setattr(core, "NOTEBOOK", lab / "NOTEBOOK.md")
    return lab


# ----------------------------------------------------------------------------- helpers
def test_slugify():
    assert core._slugify("Measure CV in a strip!") == "measure-cv-in-a-strip"
    assert core._slugify("") == "experiment"
    assert len(core._slugify("x" * 100)) <= 40


def test_grid_dims_spans_exact_domain():
    Nx, Ny = core._grid_dims(2.0, 0.5, 0.01)
    assert (Nx, Ny) == (201, 51)
    assert round(0.01 * (Nx - 1), 6) == 2.0  # Grid.Lx == length_cm


def test_render_run_script_is_valid_python():
    res = core.build_manifest(goal="strip cv", date="2026-06-26")
    payload = json.loads(base64.urlsafe_b64decode(res["experiment_token"]))
    script = core.render_run_script(payload["params"])
    compile(script, "run.py", "exec")  # raises SyntaxError if malformed
    assert "cc.monodomain(" in script
    assert "cc.ConductivityConfig.bidomain(" in script


# ----------------------------------------------------------------------------- the gate
def test_build_manifest_shape():
    res = core.build_manifest(goal="Measure CV", date="2026-06-26")
    assert "EXPERIMENT MANIFEST" in res["manifest_text"]
    assert "Confirm, or tell me what to change." in res["manifest_text"]
    assert res["slug"] == "measure-cv"
    assert isinstance(res["experiment_token"], str) and len(res["experiment_token"]) > 0


def test_commit_refuses_without_confirmation(tmp_lab):
    res = core.build_manifest(goal="gate test", date="2026-06-26")
    with pytest.raises(ValueError, match="confirmed=False"):
        core.commit_experiment(res["experiment_token"], confirmed=False)
    assert list(tmp_lab.iterdir()) == []  # nothing written


def test_commit_refuses_tampered_token(tmp_lab):
    res = core.build_manifest(goal="tamper test", date="2026-06-26")
    payload = json.loads(base64.urlsafe_b64decode(res["experiment_token"]))
    payload["params"]["sigma_i"] = 99.0  # alter a param, keep the old signature
    bad = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(ValueError, match="integrity check failed"):
        core.commit_experiment(bad, confirmed=True)
    assert list(tmp_lab.iterdir()) == []


def test_commit_writes_folder_and_notebook(tmp_lab):
    res = core.build_manifest(goal="Strip CV control", date="2026-06-26", scientist="LC")
    out = core.commit_experiment(res["experiment_token"], confirmed=True)
    d = tmp_lab / "2026-06-26_strip-cv-control"
    assert d.is_dir()
    assert (d / "MANIFEST.md").read_text().startswith("EXPERIMENT MANIFEST")
    assert "Scientist:    LC" in (d / "MANIFEST.md").read_text()
    compile((d / "run.py").read_text(), "run.py", "exec")
    nb = (tmp_lab / "NOTEBOOK.md").read_text()
    assert "strip-cv-control" in nb and "built" in nb
    assert out["status"] == "built"


def test_commit_never_overwrites(tmp_lab):
    res = core.build_manifest(goal="dup", date="2026-06-26")
    a = core.commit_experiment(res["experiment_token"], confirmed=True)
    b = core.commit_experiment(res["experiment_token"], confirmed=True)
    assert a["experiment_dir"] != b["experiment_dir"]
    assert b["experiment_dir"].endswith("dup-02")


def test_notebook_status_update(tmp_lab):
    core._append_notebook_row("2026-06-26", "myslug", "a goal", "monodomain", "built")
    core._update_notebook_status("2026-06-26_myslug", "done", 57.1)
    nb = (tmp_lab / "NOTEBOOK.md").read_text()
    assert "done" in nb and "CV=57.1 cm/s" in nb
    assert "built" not in nb.split("myslug")[1].split("\n")[0]


# ------------------------------------------------------------------ input validation + server metadata
def test_run_experiment_rejects_traversal():
    # MUST validate inputs: an absolute path or a `..`-escape must NOT execute anything outside Lab/.
    # The guard is lexical after .resolve(), so it raises whether or not Lab/ exists.
    with pytest.raises(ValueError, match="inside Lab/"):
        core.run_experiment("/etc")
    with pytest.raises(ValueError, match="inside Lab/"):
        core.run_experiment("../../etc")


def test_build_manifest_rejects_bad_date():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        core.build_manifest(goal="x", date="../../x")
    res = core.build_manifest(goal="x", date="2026-06-28")  # valid date still works
    assert res["slug"] == "x"


def test_server_metadata():
    # serverInfo.version is our package version (not the SDK's), and annotations are set per-tool.
    import asyncio

    from cardiac_mcp.server import mcp

    assert mcp._mcp_server.version == "0.1.0"
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}  # list_tools() returns a list
    assert tools["simulate"].annotations.readOnlyHint is True
    assert tools["run_experiment"].annotations.destructiveHint is True


# ----------------------------------------------------------------------------- end-to-end (slow)
@pytest.mark.slow
def test_simulate_end_to_end():
    r = core.simulate(length_cm=1.5, width_cm=0.4, dx=0.02, t_end_ms=25.0)
    assert r["activated"] is True
    assert r["cv_cm_per_s"] is not None and 20 < r["cv_cm_per_s"] < 120
    assert r["note"] == "ok"
    assert r["grid"]["Nx"] == 76
