"""Inference speed comparison: TTP06 simulator vs IonicSurrogate Stage 1.

For a grid of (n_cells, device), times N_STEPS_BENCH steps of each forward
path after N_WARMUP warmup steps, then reports cell-steps/sec and the
surrogate/TTP06 speedup ratio.

- TTP06: ``TTP06Model.step`` (the full biophysics step — gates via
  Rush-Larsen, concentrations via Euler, ionic currents, state update).
- Surrogate: ``IonicStage1.dzdt`` + manual Euler update
  ``z <- z + dt * dzdt(z, V)``. Matches what Stage 1 does on the
  off-critical-path lane during tissue simulation.

Both use torch.compile by default. The surrogate has no scaffold decoder
on the inference path (IonicStage1(scaffold=False) strips the decoders).

Usage (CPU, non-interfering with GPU jobs):
    python benchmarks/speed_ttp06_vs_surrogate.py --device cpu

Usage (GPU, when GPU is idle):
    python benchmarks/speed_ttp06_vs_surrogate.py --device cuda

Run from Surrogate/ as working directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BIDOMAIN_ROOT = Path(__file__).resolve().parents[2] / "Bidomain" / "Engine_V1"
if str(_BIDOMAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BIDOMAIN_ROOT))

from cardiac_sim.ionic.ttp06.model import TTP06Model  # noqa: E402
from cardiac_sim.ionic.ttp06.parameters import V_REST  # noqa: E402
from cardiac_sim.ionic.base import CellType  # noqa: E402

from surrogate.model.stage1 import IonicStage1  # noqa: E402
from surrogate.training.node_rollout import INIT_CONC, V_REST_MV  # noqa: E402

DT_MS = 0.01
N_STEPS_BENCH = 30_000        # 300 ms of AP at dt=0.01
N_WARMUP = 200                # enough to trigger compile + caches

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_ttp06(n_cells: int, device: torch.device, use_compile: bool) -> float:
    """cell-steps/sec for bare TTP06.step loop (no recording, no I/O)."""
    model = TTP06Model(cell_type=CellType.EPI, device=device)
    step_fn = torch.compile(model.step) if use_compile else model.step

    V = torch.full((n_cells,), V_REST, dtype=torch.float64, device=device)
    states = model.get_initial_state(n_cells=n_cells).to(device)
    I_stim = torch.zeros(n_cells, dtype=torch.float64, device=device)

    for _ in range(N_WARMUP):
        V, states = step_fn(V, states, DT_MS, I_stim=I_stim)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(N_STEPS_BENCH):
        V, states = step_fn(V, states, DT_MS, I_stim=I_stim)
    _sync(device)
    elapsed = time.perf_counter() - t0
    return n_cells * N_STEPS_BENCH / elapsed


def bench_surrogate(n_cells: int, device: torch.device, use_compile: bool,
                     scaffold: bool = False) -> float:
    """cell-steps/sec for IonicStage1.dzdt + forward Euler update."""
    stage1 = IonicStage1(scaffold=scaffold).to(dtype=torch.float64, device=device)
    stage1.eval()
    fn = torch.compile(stage1.dzdt) if use_compile else stage1.dzdt

    z = torch.zeros(n_cells, stage1.carried_dim, dtype=torch.float64, device=device)
    z[:, stage1.ionic_dim:] = INIT_CONC.to(device)
    V = torch.full((n_cells,), V_REST_MV, dtype=torch.float64, device=device)
    dt = torch.tensor(DT_MS, dtype=torch.float64, device=device)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            z = z + dt * fn(z, V)
        _sync(device)

        t0 = time.perf_counter()
        for _ in range(N_STEPS_BENCH):
            z = z + dt * fn(z, V)
        _sync(device)
    elapsed = time.perf_counter() - t0
    return n_cells * N_STEPS_BENCH / elapsed


def inference_params(scaffold: bool = False) -> int:
    s = IonicStage1(scaffold=scaffold)
    return sum(p.numel() for p in s.parameters())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[10, 100, 1_000, 10_000])
    p.add_argument("--scaffold", action="store_true",
                   help="Include scaffold decoders in surrogate (training-time config)")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON output path")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")
    use_compile = not args.no_compile

    print(f"device={device}  compile={use_compile}  dt={DT_MS}ms  "
          f"steps={N_STEPS_BENCH:,}  warmup={N_WARMUP}")
    print(f"surrogate params (inference{', with scaffold' if args.scaffold else ''}): "
          f"{inference_params(args.scaffold):,}")
    print()
    header = (
        f"{'n_cells':>8}  "
        f"{'TTP06 cs/s':>14}  {'TTP06 μs/step':>14}  "
        f"{'Surr  cs/s':>14}  {'Surr  μs/step':>14}  "
        f"{'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for n in args.sizes:
        ttp06_rate = bench_ttp06(n, device, use_compile)
        surr_rate = bench_surrogate(n, device, use_compile, scaffold=args.scaffold)
        ttp06_us = 1e6 / (ttp06_rate / n)
        surr_us = 1e6 / (surr_rate / n)
        speedup = surr_rate / ttp06_rate
        print(
            f"{n:>8,}  "
            f"{ttp06_rate:>13,.0f}  {ttp06_us:>13.2f}  "
            f"{surr_rate:>13,.0f}  {surr_us:>13.2f}  "
            f"{speedup:>7.2f}x"
        )
        results.append({
            "n_cells": n,
            "ttp06_cell_steps_per_sec": ttp06_rate,
            "ttp06_us_per_step_per_cell": ttp06_us,
            "surrogate_cell_steps_per_sec": surr_rate,
            "surrogate_us_per_step_per_cell": surr_us,
            "speedup_surr_over_ttp06": speedup,
        })

    summary = {
        "device": str(device),
        "compile": use_compile,
        "dt_ms": DT_MS,
        "n_steps": N_STEPS_BENCH,
        "n_warmup": N_WARMUP,
        "scaffold": args.scaffold,
        "surrogate_inference_params": inference_params(args.scaffold),
        "rows": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
