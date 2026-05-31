# -*- coding: utf-8 -*-
"""Experiment orchestrator: take a config dict, run the engine, write a self-
contained experiment directory.

Usage:
    from configs import BIDIRECTIONAL, make
    from experiment import run_experiment

    # by named config
    run_experiment(BIDIRECTIONAL)

    # ad-hoc override
    run_experiment(make({"name": "tweak", "rule": {"threshold": 30}}))

Each run produces:
    outputs/experiments/{date}_{name}/
        config.json         exact config dict that produced the run
        iso.npz             isochrone arrays + V_final (+ snaps/activity if enabled)
        isochrone.png       2D heatmap with white contour lines
        per_column_lat.png  the camel-toe diagnostic
        summary.txt         filled / max_step / edge−mid table
        metadata.json       wall time + git hash + tanks_vec dtype + np version

INDEX.md at the experiments root gets one new line per run.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import configs
import tanks_vec

EXPERIMENTS_ROOT = Path("outputs/experiments")


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _make_run_dir(name: str) -> Path:
    today = _dt.date.today().isoformat()
    base = EXPERIMENTS_ROOT / f"{today}_{name}"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    # name collision: append a numeric suffix
    for i in range(1, 100):
        d = EXPERIMENTS_ROOT / f"{today}_{name}_{i:02d}"
        if not d.exists():
            d.mkdir(parents=True)
            return d
    raise RuntimeError(f"too many runs today named {name!r}")


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------

def plot_isochrone(iso: np.ndarray, path: Path, title: str = "") -> None:
    Ny, Nx = iso.shape
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    iso_f = iso.astype(float)
    iso_plot = np.where(iso_f >= 0, iso_f, np.nan)
    iso_max = max(int(np.nanmax(iso_plot)) if np.isfinite(np.nanmax(iso_plot)) else 1, 1)
    im = ax.imshow(iso_plot, origin="upper", cmap="plasma",
                   aspect="equal", vmin=0, vmax=iso_max)
    levels = np.linspace(iso_max * 0.05, iso_max * 0.95, 14)
    ax.contour(iso_plot, levels=levels, colors="white",
               linewidths=0.7, alpha=0.7)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if title:
        ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label="step of first crossing")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_per_column_lat(iso: np.ndarray, path: Path,
                        sample_cols=(3, 8, 18, 30, 45, 60),
                        title: str = "") -> None:
    Ny = iso.shape[0]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sample_cols)))
    for k, c in enumerate(sample_cols):
        col = iso[:, c].astype(float)
        col = np.where(col >= 0, col, np.nan)
        if np.all(np.isnan(col)):
            ax.plot([], [], color=cmap[k], label=f"x={c} (not reached)")
            continue
        ax.plot(np.arange(Ny), col - float(np.nanmean(col)),
                color=cmap[k], lw=1.5, label=f"x={c}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("y (row)")
    ax.set_ylabel("iso[y, x] − col-mean   (negative = fires earlier)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(iso: np.ndarray, path: Path,
                  sample_cols=(3, 8, 18, 30, 45, 60)) -> str:
    Ny, Nx = iso.shape
    filled = int((iso >= 0).sum())
    max_step = int(iso.max())
    lines = [
        f"filled tanks       : {filled}/{Nx*Ny}",
        f"max isochrone step : {max_step}",
        "",
        f"{'col':>4}  {'top':>6} {'mid':>6} {'bot':>6}  {'edge−mid':>9}  shape",
    ]
    for c in sample_cols:
        col = iso[:, c].astype(float)
        col = np.where(col >= 0, col, np.nan)
        if np.all(np.isnan(col)):
            lines.append(f"{c:>4}  not reached")
            continue
        top = col[0]; mid = col[Ny // 2]; bot = col[-1]
        edge = 0.5 * (top + bot)
        delta = edge - mid if not (np.isnan(edge) or np.isnan(mid)) else float("nan")
        if np.isnan(delta):
            shape = "—"
        elif delta < -1:
            shape = "CAMEL (boundary speedup)"
        elif delta > 1:
            shape = "crescent (boundary slowdown)"
        else:
            shape = "flat"
        lines.append(f"{c:>4}  {top:>6.0f} {mid:>6.0f} {bot:>6.0f}  {delta:>+9.1f}  {shape}")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def append_index(experiments_root: Path, name: str, run_dir: Path,
                 description: str, headline: str) -> None:
    index = experiments_root / "INDEX.md"
    if not index.exists():
        index.write_text(
            "# Experiments index\n\n"
            "Auto-appended by `experiment.run_experiment`. One line per run.\n\n"
            "| date | name | description | headline | dir |\n"
            "|------|------|-------------|----------|-----|\n"
        )
    today = _dt.date.today().isoformat()
    rel = run_dir.relative_to(experiments_root)
    desc_short = description.replace("|", "/").strip() or "—"
    head_short = headline.replace("|", "/").strip() or "—"
    with index.open("a") as f:
        f.write(f"| {today} | `{name}` | {desc_short} | {head_short} | `{rel}/` |\n")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_experiment(config: dict[str, Any], output_root: Path | str = EXPERIMENTS_ROOT) -> Path:
    """Run one experiment from a config dict. Returns the run directory."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(config)
    name = cfg["name"]
    run_dir = _make_run_dir(name)
    print(f"[{name}] running -> {run_dir}", flush=True)

    geom = cfg["geometry"]
    inlet_cells, outlet_cells = configs.resolve_geometry(geom)

    rule = cfg["rule"]
    pipes = cfg["pipes"]
    bc = cfg["boundary"]
    sim = cfg["sim"]

    t0 = time.perf_counter()
    out = tanks_vec.run(
        Nx=geom["Nx"], Ny=geom["Ny"],
        mode=rule["type"], steps=sim["steps"],
        inlet_cells=inlet_cells, outlet_cells=outlet_cells,
        threshold=rule["threshold"], max_volume=rule["max_volume"],
        max_pump=rule["max_pump"], gradient_k=rule["gradient_k"],
        directionality=pipes["directionality"], boundary=bc["type"],
        damping_cap=rule["damping_cap"],
        record_history=sim["record_history"],
        snap_every=sim["snap_every"],
    )
    elapsed = time.perf_counter() - t0
    iso = out["iso"]
    print(f"[{name}] sim done in {elapsed:.2f}s, max_step={int(iso.max())}, "
          f"filled={int((iso >= 0).sum())}/{geom['Nx']*geom['Ny']}", flush=True)

    # Write artefacts
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    npz_payload = {"iso": iso, "V_final": out["V"]}
    if sim["record_history"]:
        npz_payload["snaps"] = out["snaps"]
        npz_payload["snap_steps"] = out["snap_steps"]
        npz_payload["activity"] = out["activity"]
    np.savez(run_dir / "iso.npz", **npz_payload)

    title = f"{name} — {rule['type']} / {pipes['directionality']} / BC {bc['type']}"
    plot_isochrone(iso, run_dir / "isochrone.png", title=title)
    plot_per_column_lat(iso, run_dir / "per_column_lat.png",
                        sample_cols=tuple(sim["sample_cols"]),
                        title=title)
    summary_text = write_summary(iso, run_dir / "summary.txt",
                                 sample_cols=tuple(sim["sample_cols"]))

    # Headline for INDEX (edge−mid at col 18)
    Ny = geom["Ny"]
    col18 = iso[:, 18].astype(float)
    col18 = np.where(col18 >= 0, col18, np.nan)
    edge = 0.5 * (col18[0] + col18[-1])
    mid = col18[Ny // 2]
    if not (np.isnan(edge) or np.isnan(mid)):
        d = edge - mid
        if d < -1:
            headline = f"camel (Δ@x=18 = {d:+.1f})"
        elif d > 1:
            headline = f"crescent (Δ@x=18 = {d:+.1f})"
        else:
            headline = f"flat (Δ@x=18 = {d:+.1f})"
    else:
        headline = "x=18 not reached"

    metadata = {
        "name": name,
        "wall_time_s": elapsed,
        "git_hash": _git_hash(),
        "numpy_version": np.__version__,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "headline": headline,
        "filled": int((iso >= 0).sum()),
        "max_step": int(iso.max()),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    append_index(output_root, name, run_dir, cfg.get("description", ""), headline)

    print(f"[{name}] {headline}")
    return run_dir


def run_experiment_by_name(name: str, **overrides) -> Path:
    """Look up a named config in configs.REGISTRY, optionally apply overrides."""
    if name not in configs.REGISTRY:
        raise KeyError(f"no config named {name!r}; known: {sorted(configs.REGISTRY)}")
    cfg = copy.deepcopy(configs.REGISTRY[name])
    if overrides:
        configs._deep_update(cfg, overrides)
    return run_experiment(cfg)


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="config name from configs.REGISTRY (or 'all')")
    args = ap.parse_args()
    if args.name == "all":
        for name in configs.REGISTRY:
            run_experiment_by_name(name)
    else:
        run_experiment_by_name(args.name)


if __name__ == "__main__":
    _cli()
