"""cardiac_mcp.core — transport-agnostic logic for the cardiac-core MCP server.

ALL behaviour lives here as plain functions over ``cardiac_core`` + the ``Lab/`` notebook.
``server.py`` only wires these to the MCP transport, so:
  * the same functions back stdio (now) and streamable-HTTP (later) unchanged, and
  * they are unit-testable without an MCP client (see ``tests/test_core.py``).

Two tracks (the user's "both, as separate tools" decision):
  * DIRECT / exploration  -> ``simulate``           (ephemeral; returns numbers; no Lab/ record)
  * GATED  / recorded     -> ``build_manifest`` + ``commit_experiment`` + ``run_experiment``
    The gate is STRUCTURAL: ``build_manifest`` returns a self-signed ``experiment_token`` that
    embeds the exact manifest + params; ``commit_experiment`` refuses unless that token verifies
    AND ``confirmed=True`` — so what gets written is provably what the scientist reviewed.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB = REPO_ROOT / "Lab"
NOTEBOOK = LAB / "NOTEBOOK.md"
CHEATSHEET = REPO_ROOT / "cardiac_core" / "API_CHEATSHEET.md"

_DIVIDER = "─" * 50  # ─ x50, matching the /sim-experiment manifest template

SERVER_INSTRUCTIONS = """\
cardiac-core: run cardiac electrophysiology simulations via the unified cardiac_core API.

Always read the `cardiac://cheatsheet` resource before constructing parameters — it is the ONLY
authoritative API surface (never invent signatures).

Two tracks:
  * Quick look  -> `simulate(...)`  : ephemeral CV measurement, no record. Use for exploration.
  * Recorded experiment (the accountability path):
      1. `build_manifest(...)`  -> returns a plain-text manifest + an `experiment_token`.
      2. SHOW the manifest to the scientist; get their explicit "go".
      3. `commit_experiment(experiment_token, confirmed=True)` -> writes Lab/{date}_{slug}/.
      4. (optional) `run_experiment(experiment_dir)` -> executes the script, records the result.
  NEVER call commit_experiment before the scientist confirms the manifest. That gate is the point.
"""


# ----------------------------------------------------------------------------- tool result models
# Typed returns so FastMCP emits an MCP `outputSchema` and returns `structuredContent` (spec SHOULD).
# Values are unchanged — only the annotations are new (nested variable shapes stay `dict[str, Any]`).
class SimulateResult(TypedDict):
    engine: str
    ionic: str
    grid: dict[str, Any]
    conductivity: dict[str, Any]
    cv_cm_per_s: float | None
    cv_indices: dict[str, int]
    activated: bool
    frames: list[int]
    note: str


class ManifestResult(TypedDict):
    manifest_text: str
    slug: str
    experiment_token: str
    next: str


class CommitResult(TypedDict):
    experiment_dir: str
    files: list[str]
    status: str
    next: str


class RunResult(TypedDict):
    experiment_dir: str
    status: str
    cv_cm_per_s: float | None
    returncode: int
    stdout: str
    stderr: str


class ListResult(TypedDict):
    count: int
    experiments: list[str]


_ENGINE_WHY = {
    "monodomain": "single potential, fast — default",
    "bidomain": "models the surrounding bath / tissue edge / boundary loading",
    "lbm": "lattice-Boltzmann (explicit request)",
}


# ----------------------------------------------------------------------------- helpers
def _slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:maxlen].strip("-")) or "experiment"


def _today() -> str:
    return datetime.date.today().isoformat()


def _grid_dims(length_cm: float, width_cm: float, dx: float) -> tuple[int, int]:
    # Grid.Lx = dx*(Nx-1) -> span exactly length_cm x width_cm
    return round(length_cm / dx) + 1, round(width_cm / dx) + 1


def _sanity_note(cv: float | None, activated: bool) -> str:
    import math

    if not activated:
        return "WARNING: tissue never activated — stimulus too weak or t_end too short."
    if cv is None or math.isnan(cv) or cv <= 0:
        return "WARNING: CV is NaN/0 — front may not have reached x2; increase t_end or check indices."
    if cv > 200:
        return "WARNING: CV unphysiologically high (>200 cm/s) — check dx / conductivity."
    return "ok"


# ----------------------------------------------------------------------------- DIRECT track
def simulate(
    length_cm: float = 2.0,
    width_cm: float = 0.5,
    dx: float = 0.02,
    sigma_i: float = 1.74,
    sigma_e: float = 6.25,
    ionic: str = "ttp06",
    engine: str = "monodomain",
    stim_width_cm: float = 0.05,
    t_end_ms: float = 30.0,
    save_every_ms: float = 0.5,
) -> SimulateResult:
    """Run a quick conduction-velocity simulation on a tissue strip and return the result.

    EXPLORATION tool — ephemeral: returns numbers, writes no Lab/ record. Use it to try parameters
    before committing a recorded experiment. Defaults to a coarse dx=0.02 strip (~8s); pass a finer
    dx (e.g. 0.01) for publication fidelity (slower, ~40s).

    Args:
        length_cm: tissue length (cm).
        width_cm: tissue width (cm).
        dx: grid spacing (cm); smaller = finer + slower.
        sigma_i: intracellular conductivity (mS/cm).
        sigma_e: extracellular conductivity (mS/cm); lower = weaker coupling (fibrosis-like).
        ionic: cell model, "ttp06" or "ord".
        engine: "monodomain" (default) or "bidomain".
        stim_width_cm: stimulate the left edge up to this x (cm).
        t_end_ms: simulated duration (ms); give the front time to cross (front ~50 cm/s).
        save_every_ms: snapshot interval (ms).

    Returns:
        dict with cv_cm_per_s, activated, grid, conductivity, and a sanity 'note'.
    """
    import cardiac_core as cc

    if engine not in ("monodomain", "bidomain"):
        raise ValueError("simulate(): engine must be 'monodomain' or 'bidomain' (lbm via the gated path).")
    Nx, Ny = _grid_dims(length_cm, width_cm, dx)
    g = cc.Grid(Nx, Ny, dx)
    cond = cc.ConductivityConfig.bidomain(sigma_i, sigma_e, chi=1400.0)
    stim = {
        "region": lambda x, y: x < stim_width_cm,
        "start_time": 1.0,
        "duration": 2.0,
        "amplitude": -80.0,
    }
    factory = {"monodomain": cc.monodomain, "bidomain": cc.bidomain}[engine]
    sim = factory(g, ionic, cond, stim)
    r = sim.run(t_end=t_end_ms, save_every=save_every_ms)

    x1 = round(0.2 / dx)
    x2 = round(min(length_cm * 0.5, 1.0) / dx)
    cv_raw = float(r.cv(x1=x1, x2=x2, y=Ny // 2))
    activated = bool((r.Vm[-1] > 0).any())
    import math

    cv = None if math.isnan(cv_raw) else round(cv_raw, 2)
    return {
        "engine": engine,
        "ionic": ionic,
        "grid": {"Nx": Nx, "Ny": Ny, "dx": dx,
                 "Lx_cm": round(dx * (Nx - 1), 4), "Ly_cm": round(dx * (Ny - 1), 4)},
        "conductivity": {"sigma_i": sigma_i, "sigma_e": sigma_e,
                         "sigma_eff": round(float(cond.sigma_eff), 4),
                         "D_eff": float(cond.D_eff)},
        "cv_cm_per_s": cv,
        "cv_indices": {"x1": x1, "x2": x2, "y": Ny // 2},
        "activated": activated,
        "frames": list(r.Vm.shape),
        "note": _sanity_note(cv, activated),
    }


# ----------------------------------------------------------------------------- GATED track
def render_manifest(p: dict) -> str:
    """Render the plain-text experiment manifest (matches the /sim-experiment template)."""
    why = _ENGINE_WHY.get(p["engine"], "")
    lines = [
        "EXPERIMENT MANIFEST — please confirm before I run",
        _DIVIDER,
        f"Goal:         {p['goal']}",
        f"Engine:       {p['engine']}  ({why})",
        f"Ionic model:  {p['ionic']}",
        f"Geometry:     {p['length_cm']} × {p['width_cm']} cm,  dx = {p['dx']} cm   "
        f"({p['Nx']} × {p['Ny']} grid)",
        f"Tissue:       σ_i={p['sigma_i']}, σ_e={p['sigma_e']} mS/cm  "
        f"(bidomain; χ={p['chi']}, Cm={p['Cm']})",
        f"Delivery:     single stimulus, left edge x<{p['stim_width_cm']} cm, "
        f"t={p['stim_start_ms']} ms, {p['stim_amp']} µA/µF",
        f"Sim length:   t_end = {p['t_end_ms']} ms,  dt = {p['dt_ms']} ms,  "
        f"save every {p['save_every_ms']} ms",
        f"Measure:      {p['measure']}",
        f"Outputs:      conduction-velocity printout (+ media via /sim-media)",
        f"Script:       Lab/{p['date']}_{p['slug']}/run.py",
    ]
    if p.get("scientist") or p.get("hypothesis"):
        lines.append("── optional " + "─" * 39)
        if p.get("scientist"):
            lines.append(f"Scientist:    {p['scientist']}")
        if p.get("hypothesis"):
            lines.append(f"Hypothesis:   {p['hypothesis']}")
    if p["measure"] != "cv":
        lines.append(f"NOTE:         v1 generates a CV script; measure='{p['measure']}' not yet wired.")
    lines += [_DIVIDER, "Confirm, or tell me what to change."]
    return "\n".join(lines)


def render_run_script(p: dict) -> str:
    """Render a self-contained run.py from confirmed params (mirrors run-template.py)."""
    return f'''"""
{p["title"]} — generated by the cardiac-core MCP server on {p["date"]}
Goal: {p["goal"]}
Manifest: ./MANIFEST.md   (the confirmed parameters — the accountability record)
Run:  conda run -n heart-conduction python run.py

Generated against cardiac_core/API_CHEATSHEET.md. Edit the PARAMETERS block to iterate;
you do not need to touch the cardiac_core calls below it.
"""
import cardiac_core as cc

# ============================================================
# PARAMETERS  — edit these
# ============================================================
LENGTH_CM     = {p["length_cm"]}
WIDTH_CM      = {p["width_cm"]}
DX            = {p["dx"]}
SIGMA_I       = {p["sigma_i"]}
SIGMA_E       = {p["sigma_e"]}
IONIC         = "{p["ionic"]}"
STIM_WIDTH_CM = {p["stim_width_cm"]}
STIM_START_MS = {p["stim_start_ms"]}
STIM_AMP      = {p["stim_amp"]}
T_END_MS      = {p["t_end_ms"]}
SAVE_EVERY_MS = {p["save_every_ms"]}
SLUG          = "{p["slug"]}"
MAKE_MEDIA    = False
# ============================================================

Nx, Ny = round(LENGTH_CM / DX) + 1, round(WIDTH_CM / DX) + 1

grid = cc.Grid(Nx, Ny, DX)
cond = cc.ConductivityConfig.bidomain(SIGMA_I, SIGMA_E, chi={p["chi"]})
stim = {{"region": lambda x, y: x < STIM_WIDTH_CM, "start_time": STIM_START_MS,
        "duration": 2.0, "amplitude": STIM_AMP}}

sim = cc.{p["engine"]}(grid, IONIC, cond, stim)
result = sim.run(t_end=T_END_MS, save_every=SAVE_EVERY_MS)

cv = result.cv(x1=round(0.2 / DX), x2=round(min(LENGTH_CM * 0.5, 1.0) / DX), y=Ny // 2)
print(f"conduction velocity = {{cv:.1f}} cm/s")

if MAKE_MEDIA:
    from cardiac_core import propagation_video, apd_map_figure
    print("video:", propagation_video(result, SLUG, bulk=True))
    print("apd:  ", apd_map_figure(result, SLUG, bulk=True))
'''


def _sign_payload(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_manifest(
    goal: str,
    engine: str = "monodomain",
    ionic: str = "ttp06",
    length_cm: float = 2.0,
    width_cm: float = 0.5,
    dx: float = 0.01,
    sigma_i: float = 1.74,
    sigma_e: float = 6.25,
    chi: float = 1400.0,
    Cm: float = 1.0,
    stim_width_cm: float = 0.05,
    stim_start_ms: float = 1.0,
    stim_amp: float = -80.0,
    t_end_ms: float = 40.0,
    dt_ms: float = 0.02,
    save_every_ms: float = 0.5,
    measure: str = "cv",
    date: str | None = None,
    scientist: str | None = None,
    hypothesis: str | None = None,
) -> ManifestResult:
    """Build (but do NOT run) a recorded experiment: compute params, render the manifest, return a token.

    This is step 1 of the accountability gate. Show the returned ``manifest_text`` to the scientist;
    only after their explicit confirmation call ``commit_experiment(experiment_token, confirmed=True)``.

    Args mirror the cheatsheet recipe (CV strip). ``date`` should be today's date (YYYY-MM-DD);
    defaults to the server's clock if omitted.

    Returns:
        dict with ``manifest_text`` (show this), ``slug``, and an opaque self-signed
        ``experiment_token`` to pass to ``commit_experiment``.
    """
    if engine not in _ENGINE_WHY:
        raise ValueError(f"build_manifest(): unknown engine '{engine}'.")
    date = date or _today()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"build_manifest(): date must be YYYY-MM-DD (got {date!r}).")
    slug = _slugify(goal)
    Nx, Ny = _grid_dims(length_cm, width_cm, dx)
    params = {
        "goal": goal, "title": goal, "engine": engine, "ionic": ionic,
        "length_cm": length_cm, "width_cm": width_cm, "dx": dx, "Nx": Nx, "Ny": Ny,
        "sigma_i": sigma_i, "sigma_e": sigma_e, "chi": chi, "Cm": Cm,
        "stim_width_cm": stim_width_cm, "stim_start_ms": stim_start_ms, "stim_amp": stim_amp,
        "t_end_ms": t_end_ms, "dt_ms": dt_ms, "save_every_ms": save_every_ms,
        "measure": measure, "date": date, "slug": slug,
        "scientist": scientist, "hypothesis": hypothesis,
    }
    manifest_text = render_manifest(params)
    payload = {"manifest_text": manifest_text, "params": params}
    payload["sig"] = _sign_payload({"manifest_text": manifest_text, "params": params})
    token = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return {
        "manifest_text": manifest_text,
        "slug": slug,
        "experiment_token": token,
        "next": ("Show manifest_text to the scientist verbatim. On their explicit 'go', call "
                 "commit_experiment(experiment_token, confirmed=True)."),
    }


def commit_experiment(experiment_token: str, confirmed: bool = False) -> CommitResult:
    """Write a recorded experiment to Lab/ — the gated step. REQUIRES scientist confirmation.

    Refuses unless ``confirmed=True`` AND the ``experiment_token`` from ``build_manifest`` verifies
    intact (so the committed script is provably the one the scientist reviewed). Writes
    ``Lab/{date}_{slug}/`` containing ``MANIFEST.md`` (verbatim) + ``run.py``, and appends a row to
    ``Lab/NOTEBOOK.md``. Never overwrites an existing experiment folder (suffixes the slug).

    Args:
        experiment_token: the opaque token returned by ``build_manifest``.
        confirmed: must be True — set only after the scientist explicitly approved the manifest.

    Returns:
        dict with ``experiment_dir``, written ``files``, and ``status`` = "built".
    """
    if not confirmed:
        raise ValueError(
            "GATE: confirmed=False. Show the manifest, get the scientist's explicit 'go', "
            "then call commit_experiment(experiment_token, confirmed=True).")
    try:
        payload = json.loads(base64.urlsafe_b64decode(experiment_token.encode()))
        sig = payload.pop("sig")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"GATE: malformed experiment_token ({exc}). Re-run build_manifest.") from exc
    if _sign_payload(payload) != sig:
        raise ValueError(
            "GATE: token integrity check failed — manifest/params were altered after build_manifest. "
            "Re-run build_manifest and re-confirm.")

    manifest_text = payload["manifest_text"]
    params = payload["params"]
    date, slug = params["date"], params["slug"]

    d = LAB / f"{date}_{slug}"
    n = 2
    while d.exists():
        d = LAB / f"{date}_{slug}-{n:02d}"
        n += 1
    d.mkdir(parents=True)
    (d / "MANIFEST.md").write_text(manifest_text + "\n")
    (d / "run.py").write_text(render_run_script(params))
    _append_notebook_row(date, d.name.split("_", 1)[1], params["goal"], params["engine"], "built")

    try:
        experiment_dir = str(d.relative_to(REPO_ROOT))
    except ValueError:  # LAB redirected outside the repo (e.g. under test)
        experiment_dir = str(d)
    return {
        "experiment_dir": experiment_dir,
        "files": ["MANIFEST.md", "run.py"],
        "status": "built",
        "next": "Optionally call run_experiment(experiment_dir) to execute and record the result.",
    }


def run_experiment(experiment_dir: str, timeout_s: int = 900) -> RunResult:
    """Execute a committed experiment's run.py, sanity-check the result, and record it.

    Runs the script with the same Python/env from the repo root, parses the conduction velocity,
    updates the ``Lab/NOTEBOOK.md`` row (built -> done | failed) and appends the result to the
    experiment's ``MANIFEST.md``.

    Args:
        experiment_dir: path like "Lab/2026-06-26_my-experiment" (as returned by commit_experiment).
        timeout_s: hard wall-clock cap on the run.

    Returns:
        dict with ``status`` (done|failed), ``cv_cm_per_s``, and trimmed stdout/stderr.
    """
    d = (REPO_ROOT / experiment_dir).resolve()
    if not d.is_relative_to(LAB.resolve()):
        raise ValueError(
            f"run_experiment(): experiment_dir must be inside Lab/ (got {experiment_dir!r}).")
    run_py = d / "run.py"
    if not run_py.exists():
        raise ValueError(f"run_experiment(): no run.py at {experiment_dir}")
    proc = subprocess.run(
        [sys.executable, str(run_py)], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=timeout_s,
    )
    cv = _parse_cv(proc.stdout)
    ok = proc.returncode == 0 and cv is not None and 0 < cv < 200
    status = "done" if ok else "failed"
    _update_notebook_status(d.name, status, cv)
    _append_manifest_result(d, status, cv, proc.stdout, proc.stderr)
    return {
        "experiment_dir": experiment_dir,
        "status": status,
        "cv_cm_per_s": cv,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": ("" if ok else proc.stderr[-1500:]),
    }


def list_experiments() -> ListResult:
    """List recorded experiments in Lab/ (folders containing a MANIFEST.md)."""
    if not LAB.exists():
        return {"count": 0, "experiments": []}
    names = sorted(p.parent.name for p in LAB.glob("*/MANIFEST.md"))
    return {"count": len(names), "experiments": names}


# ----------------------------------------------------------------------------- notebook I/O
_NOTEBOOK_HEADER = (
    "# Lab notebook\n\n"
    "| date | slug | goal | engine | status | result |\n"
    "|------|------|------|--------|--------|--------|\n"
)


def _append_notebook_row(date: str, slug: str, goal: str, engine: str, status: str) -> None:
    if not NOTEBOOK.exists():
        NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
        NOTEBOOK.write_text(_NOTEBOOK_HEADER)
    row = f"| {date} | {slug} | {goal} | {engine} | {status} | — |\n"
    with NOTEBOOK.open("a") as fh:
        fh.write(row)


def _update_notebook_status(folder_name: str, status: str, cv: float | None) -> None:
    if not NOTEBOOK.exists():
        return
    date, slug = folder_name.split("_", 1)
    result = f"CV={cv:.1f} cm/s" if (status == "done" and cv is not None) else \
        ("CV=NaN/blew up" if status == "failed" else "—")
    out = []
    for line in NOTEBOOK.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        # cells: ['', date, slug, goal, engine, status, result, '']
        if len(cells) >= 7 and cells[1] == date and cells[2] == slug:
            cells[5] = status
            cells[6] = result
            line = "| " + " | ".join(cells[1:7]) + " |"
        out.append(line)
    NOTEBOOK.write_text("\n".join(out) + "\n")


def _append_manifest_result(d: Path, status: str, cv: float | None, stdout: str, stderr: str) -> None:
    man = d / "MANIFEST.md"
    if not man.exists():
        return
    block = [f"\n\n## Result ({status})", ""]
    if cv is not None:
        block.append(f"- conduction velocity = {cv:.1f} cm/s")
    block.append(f"- status: {status}")
    if status == "failed" and stderr.strip():
        block.append(f"- stderr (tail): {stderr.strip()[-400:]}")
    with man.open("a") as fh:
        fh.write("\n".join(block) + "\n")


def _parse_cv(stdout: str) -> float | None:
    m = re.search(r"conduction velocity\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*cm/s", stdout)
    return float(m.group(1)) if m else None


# ----------------------------------------------------------------------------- resources
def read_cheatsheet() -> str:
    """Return the canonical cardiac_core API cheatsheet (the only API source to generate against)."""
    return CHEATSHEET.read_text() if CHEATSHEET.exists() else "API_CHEATSHEET.md not found."


def read_notebook() -> str:
    """Return the Lab notebook index (all recorded experiments)."""
    return NOTEBOOK.read_text() if NOTEBOOK.exists() else "No Lab/NOTEBOOK.md yet."
