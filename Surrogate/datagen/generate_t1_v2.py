"""T1 v2 generator — steady-state pacing, 35 BCLs x 50 beats.

Uses ``BatchGenerator`` (parallel GPU, torch.compile) to batch multiple BCLs
per forward pass. BCLs are auto-grouped into chunks so that
``n_cells * max_steps_in_chunk`` stays under ``--cell-step-budget``
(default 40M → ~14 GB GPU for the 47-col float64 recording buffers).

Writes one HDF5 file per (tier, celltype) to
``/media/shared/norepinephrine/surrogate_data_v2/raw/tier01_{celltype}.h5``
with the v2 schema (column_names/units in file attrs, short group names,
gzip-4 compression, per-group stim/beat metadata, quality flags).

Also writes the split sidecar JSON and a provenance log.

Usage:
    python -m datagen.generate_t1_v2 --celltype EPI
    python -m datagen.generate_t1_v2 --celltype EPI --smoke     # 2 BCLs x 5 beats

Run from Surrogate/ as working directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surrogate.data.schema import (  # noqa: E402
    COLUMN_NAMES,
    COLUMN_UNITS,
    COLUMN_BOUNDS,
    DATASET_VERSION,
    N_COLUMNS,
    column_groups_json,
)
from surrogate.data.batch_generator import BatchGenerator  # noqa: E402
from surrogate.data.single_cell_generator import TraceData  # noqa: E402
from surrogate.data.protocols import SteadyStatePacing  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen_t1_v2")

DEFAULT_ROOT = Path("/media/shared/norepinephrine/surrogate_data_v2")

BCLS_LOW = list(range(200, 310, 10))          # 200..300 step 10  (11)
BCLS_MID = list(range(350, 1050, 50))         # 350..1000 step 50 (14)
BCLS_HIGH = list(range(1100, 2100, 100))      # 1100..2000 step 100 (10)
ALL_BCLS: List[int] = BCLS_LOW + BCLS_MID + BCLS_HIGH   # 35

VAL_BCLS = [220, 280, 450, 900, 1200, 1800]
TEST_BCLS = [240, 260, 550, 850, 1500, 1900]
assert set(VAL_BCLS).issubset(ALL_BCLS)
assert set(TEST_BCLS).issubset(ALL_BCLS)
assert not (set(VAL_BCLS) & set(TEST_BCLS))

N_BEATS = 50
WARMUP_BEATS = (0, 14)
TRAIN_BEATS_WITHIN = (15, 39)
VAL_BEATS_WITHIN = (40, 44)
TEST_BEATS_WITHIN = (45, 49)

DT_MS = 0.01

# Default memory budget: 40M cell-steps × 46 doubles × 8 bytes ≈ 14 GB of
# recording tensors per chunk. Fits comfortably in 33 GB Blackwell VRAM
# alongside schedule tensors, model state, and compile-graph overhead.
DEFAULT_CELL_STEP_BUDGET = 40_000_000


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT.parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _set_string_array_attr(obj, name: str, values) -> None:
    arr = np.array(list(values), dtype=h5py.string_dtype(encoding="utf-8"))
    obj.attrs.create(name, arr)


def group_bcls(bcls: List[int], n_beats: int, cell_step_budget: int) -> List[List[int]]:
    """Greedy-group BCLs so that ``len(group) * max(bcl_ms*n_beats/dt)`` <= budget.

    Sorts ascending so each chunk contains BCLs of similar duration — avoids
    wasting the recording buffer on short BCLs padded to a large max.
    """
    bcls_sorted = sorted(bcls)
    groups: List[List[int]] = []
    current: List[int] = []

    def steps(bcl: int) -> int:
        return int(bcl * n_beats / DT_MS)

    for bcl in bcls_sorted:
        if not current:
            current.append(bcl)
            continue
        max_bcl_if_added = max(current[-1], bcl)  # sorted ⇒ = bcl
        cost = (len(current) + 1) * steps(max_bcl_if_added)
        if cost > cell_step_budget:
            groups.append(current)
            current = [bcl]
        else:
            current.append(bcl)
    if current:
        groups.append(current)
    return groups


def validate_trace(data: torch.Tensor, protocol_name: str) -> List[str]:
    warnings: List[str] = []
    arr = data.cpu().numpy() if data.is_cuda else data.numpy()
    if arr.shape[1] != N_COLUMNS:
        raise ValueError(f"{protocol_name}: got {arr.shape[1]} cols, expected {N_COLUMNS}")
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise ValueError(f"{protocol_name}: {n_nan} NaN + {n_inf} inf values")
    for col_name, (lo, hi) in COLUMN_BOUNDS.items():
        idx = COLUMN_NAMES.index(col_name)
        col = arr[:, idx]
        if lo is not None and col.min() < lo:
            warnings.append(f"{col_name} below bound: min={col.min():.6g} < {lo}")
        if hi is not None and col.max() > hi:
            warnings.append(f"{col_name} above bound: max={col.max():.6g} > {hi}")
    return warnings


def compute_capture_flag(data: np.ndarray, beat_boundaries_idx: List[int],
                         capture_window_steps: int = 500) -> bool:
    vm = data[:, 0]
    for start in beat_boundaries_idx[:-1]:
        end = min(start + capture_window_steps, len(vm))
        if end - start < 2:
            return False
        if vm[start:end].max() <= 0.0:
            return False
    return True


def compute_alternans_flag(data: np.ndarray, beat_boundaries_idx: List[int],
                            warmup_beats: int = 15) -> bool:
    vm = data[:, 0]
    if len(beat_boundaries_idx) < warmup_beats + 3:
        return False
    apds: List[float] = []
    for i in range(warmup_beats, len(beat_boundaries_idx) - 1):
        lo, hi = beat_boundaries_idx[i], beat_boundaries_idx[i + 1]
        beat_vm = vm[lo:hi]
        peak = beat_vm.max()
        if peak <= 0.0:
            apds.append(0.0)
            continue
        threshold = peak - 0.9 * (peak - beat_vm[0])
        above = np.where(beat_vm > threshold)[0]
        apds.append(float(above[-1] - above[0]) if above.size else 0.0)
    if len(apds) < 3:
        return False
    apds_arr = np.array(apds)
    diffs = np.abs(np.diff(apds_arr))
    means = 0.5 * (apds_arr[1:] + apds_arr[:-1])
    rel = np.where(means > 0, diffs / np.maximum(means, 1e-9), 0.0)
    return bool((rel > 0.10).mean() > 0.5)


def write_group(f: h5py.File, name: str, trace_data: torch.Tensor,
                bcl_ms: int, n_beats: int) -> Dict:
    warnings = validate_trace(trace_data, name)

    arr = trace_data.cpu().numpy() if trace_data.is_cuda else trace_data.numpy()
    n_steps = arr.shape[0]
    steps_per_beat = int(round(bcl_ms / DT_MS))
    beat_boundaries_idx = [b * steps_per_beat for b in range(n_beats + 1)]
    beat_boundaries_idx[-1] = min(beat_boundaries_idx[-1], n_steps)
    stim_onsets_ms = [b * bcl_ms for b in range(n_beats)]

    capture = compute_capture_flag(arr, beat_boundaries_idx)
    alternans = compute_alternans_flag(arr, beat_boundaries_idx)

    grp = f.create_group(name)
    ds = grp.create_dataset(
        "data",
        data=arr,
        dtype=np.float64,
        chunks=(min(65536, n_steps), N_COLUMNS),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    ds.attrs["shape"] = np.array(arr.shape, dtype=np.int64)

    grp.attrs["protocol_type"] = "steady_pacing"
    grp.attrs["bcl_ms"] = float(bcl_ms)
    grp.attrs["n_beats"] = int(n_beats)
    grp.attrs["duration_ms"] = float(bcl_ms * n_beats)
    grp.attrs["n_timesteps"] = int(n_steps)
    grp.attrs["stim_amplitude_pA_pF"] = -80.0
    grp.attrs["stim_duration_ms"] = 1.0
    grp.attrs["stim_onsets_ms"] = np.array(stim_onsets_ms, dtype=np.float64)
    grp.attrs["beat_boundaries_idx"] = np.array(beat_boundaries_idx, dtype=np.int64)
    grp.attrs["capture_flag"] = bool(capture)
    grp.attrs["alternans_flag"] = bool(alternans)

    return {
        "name": name,
        "bcl_ms": bcl_ms,
        "n_timesteps": n_steps,
        "capture_flag": capture,
        "alternans_flag": alternans,
        "warnings": warnings,
    }


def write_file_attrs(f: h5py.File, celltype: str, git_sha: str) -> None:
    f.attrs["dataset_version"] = DATASET_VERSION
    f.attrs["tier_id"] = 1
    f.attrs["tier_description"] = "Steady-state pacing, 35 BCLs x 50 beats"
    f.attrs["cell_type"] = celltype
    f.attrs["ionic_model"] = "TTP06"
    f.attrs["dt_ms"] = DT_MS
    f.attrs["simulator_engine"] = "Bidomain/Engine_V1 (TTP06Model via BatchGenerator)"
    f.attrs["simulator_commit"] = git_sha
    f.attrs["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _set_string_array_attr(f, "column_names", COLUMN_NAMES)
    _set_string_array_attr(f, "column_units", COLUMN_UNITS)
    f.attrs["column_groups"] = column_groups_json()


def write_split_json(path: Path, celltype: str, bcls: List[int]) -> None:
    val_bcls = [b for b in VAL_BCLS if b in bcls]
    test_bcls = [b for b in TEST_BCLS if b in bcls]
    train_bcls = sorted(b for b in bcls if b not in val_bcls and b not in test_bcls)
    payload = {
        "dataset_version": DATASET_VERSION,
        "tier": 1,
        "cell_type": celltype,
        "strategy": "two_axis",
        "rationale": (
            "Regime-stratified across-BCL holdout (low 200-300 / mid 350-1000 / "
            "high 1100-2000) plus within-BCL beat holdout on train BCLs."
        ),
        "across_bcl": {
            "train": train_bcls,
            "val": val_bcls,
            "test": test_bcls,
        },
        "within_bcl": {
            "applies_to": "across_bcl.train",
            "warmup_beats": list(WARMUP_BEATS),
            "train_beats": list(TRAIN_BEATS_WITHIN),
            "val_beats": list(VAL_BEATS_WITHIN),
            "test_beats": list(TEST_BEATS_WITHIN),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_provenance_json(path: Path, celltype: str, git_sha: str,
                           per_group: List[Dict], total_seconds: float,
                           batch_groups: List[List[int]]) -> None:
    payload = {
        "dataset_version": DATASET_VERSION,
        "tier": 1,
        "cell_type": celltype,
        "git_sha": git_sha,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_seconds": round(total_seconds, 2),
        "batch_groups": batch_groups,
        "n_groups": len(per_group),
        "groups": per_group,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def generate(celltype: str, output_h5: Path, split_json: Path,
              provenance_json: Path, bcls: List[int], n_beats: int,
              device: str, cell_step_budget: int) -> None:
    git_sha = _git_sha()
    log.info(f"celltype={celltype} device={device} git={git_sha[:8]} "
             f"bcls={len(bcls)} beats={n_beats} budget={cell_step_budget:,}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    split_json.parent.mkdir(parents=True, exist_ok=True)
    provenance_json.parent.mkdir(parents=True, exist_ok=True)

    batch_groups = group_bcls(bcls, n_beats, cell_step_budget)
    log.info(f"grouped into {len(batch_groups)} batches: "
             + ", ".join(f"[{g[0]}..{g[-1]}]({len(g)})" for g in batch_groups))

    gen = BatchGenerator(cell_type=celltype, device=device, use_compile=(device == "cuda"))

    t_global = time.time()
    per_group_metrics: List[Dict] = []

    with h5py.File(output_h5, "w") as f:
        write_file_attrs(f, celltype, git_sha)

        for batch_idx, bcl_chunk in enumerate(batch_groups):
            protocols = [
                SteadyStatePacing(bcl=float(b), n_beats=n_beats, dt_default=DT_MS)
                for b in bcl_chunk
            ]
            log.info(f"batch {batch_idx + 1}/{len(batch_groups)}: "
                     f"BCLs {bcl_chunk[0]}..{bcl_chunk[-1]} ({len(bcl_chunk)} cells)")
            t0 = time.time()
            traces = gen.run_batch(protocols, progress_interval=30.0)
            batch_elapsed = time.time() - t0
            log.info(f"  batch done in {batch_elapsed:.1f}s")

            for bcl, trace in zip(bcl_chunk, traces):
                name = f"bcl{bcl}"
                metrics = write_group(f, name, trace.data, bcl_ms=bcl, n_beats=n_beats)
                metrics["batch_idx"] = batch_idx
                per_group_metrics.append(metrics)
                warn_note = (" ! " + "; ".join(metrics["warnings"])) if metrics["warnings"] else ""
                log.info(
                    f"  wrote {name} ({metrics['n_timesteps']:,} steps) "
                    f"capture={metrics['capture_flag']} "
                    f"altern={metrics['alternans_flag']}{warn_note}"
                )

            # Free batch memory before next group
            del traces
            if device == "cuda":
                torch.cuda.empty_cache()

    write_split_json(split_json, celltype, bcls)
    write_provenance_json(provenance_json, celltype, git_sha,
                          per_group_metrics, time.time() - t_global,
                          batch_groups)

    log.info(f"wrote {output_h5}  ({output_h5.stat().st_size / 1e9:.2f} GB)")
    log.info(f"wrote {split_json}")
    log.info(f"wrote {provenance_json}")
    log.info(f"total elapsed: {(time.time() - t_global) / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate T1 v2 steady-state data")
    p.add_argument("--celltype", default="EPI", choices=["EPI", "ENDO", "M_CELL"])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cell-step-budget", type=int, default=DEFAULT_CELL_STEP_BUDGET,
                   help="Max cell-steps per BatchGenerator call (governs chunk size)")
    p.add_argument("--smoke", action="store_true",
                   help="Run 2 BCLs x 5 beats as a sanity check")
    args = p.parse_args()

    bcls = ALL_BCLS
    n_beats = N_BEATS
    output_name = f"tier01_{args.celltype.lower()}.h5"
    split_name = f"tier01_{args.celltype.lower()}_v2.json"
    prov_name = f"tier01_{args.celltype.lower()}_genlog.json"

    if args.smoke:
        bcls = [500, 1000]
        n_beats = 5
        output_name = f"tier01_{args.celltype.lower()}_smoke.h5"
        split_name = f"tier01_{args.celltype.lower()}_smoke_v2.json"
        prov_name = f"tier01_{args.celltype.lower()}_smoke_genlog.json"
        log.info("SMOKE mode — 2 BCLs x 5 beats")

    output_h5 = args.root / "raw" / output_name
    split_json = args.root / "splits" / split_name
    provenance_json = args.root / "provenance" / prov_name

    generate(
        celltype=args.celltype,
        output_h5=output_h5,
        split_json=split_json,
        provenance_json=provenance_json,
        bcls=bcls,
        n_beats=n_beats,
        device=args.device,
        cell_step_budget=args.cell_step_budget,
    )


if __name__ == "__main__":
    main()
