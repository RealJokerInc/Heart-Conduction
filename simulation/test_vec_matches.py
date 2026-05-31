# -*- coding: utf-8 -*-
"""Regression test: vectorised simulator vs OOP simulator.

Runs small grids through both `tanks_channel_states` (OOP) and `tanks_vec`
(numpy) and reports the max tank-wise absolute difference in final V and
the max difference in isochrone step.

Expected differences:
    - max |V_oop - V_vec| <= ~1e-10 (float non-associativity from different
      summation orders in the Jacobi update)
    - max |iso_oop - iso_vec| <= ~1 (if a tank crosses threshold right at
      the noise boundary, it may register one step earlier or later)

Any larger discrepancy indicates a real bug (indexing, flux routing,
boundary handling, or rule mismatch).
"""

from __future__ import annotations

import time

import numpy as np

import tanks_vec
from tanks_channel_states import build_grid, history_to_isochrone, run_sim


def run_oop(Nx, Ny, mode, steps, inlet_cells, outlet_cells,
            threshold=45.0, gradient_k=0.05):
    tanks, channels = build_grid(
        Nx, Ny,
        channel_state=mode,
        threshold=threshold,
        gradient_k=gradient_k,
        inlet_cells=inlet_cells,
        outlet_cells=outlet_cells,
    )
    # record_every=1 so every step is captured -> exact iso comparison
    t0 = time.perf_counter()
    history = run_sim(tanks, channels, Nx, Ny, steps=steps, record_every=1)
    elapsed = time.perf_counter() - t0

    V_final = np.zeros((Ny, Nx), dtype=np.float64)
    for t in tanks:
        x = t.id // Ny
        y = t.id % Ny
        V_final[y, x] = t.current_volume

    iso = history_to_isochrone(history, threshold=threshold, record_every=1)
    return V_final, iso, elapsed


def run_vec(Nx, Ny, mode, steps, inlet_cells, outlet_cells,
            threshold=45.0, gradient_k=0.05):
    t0 = time.perf_counter()
    out = tanks_vec.run(
        Nx, Ny, mode, steps,
        inlet_cells=inlet_cells,
        outlet_cells=outlet_cells,
        threshold=threshold,
        gradient_k=gradient_k,
    )
    elapsed = time.perf_counter() - t0
    return out["V"], out["iso"], elapsed


def compare_case(mode, geometry, Nx=30, Ny=20, steps=100):
    if geometry == "line":
        inlet_cells = [(0, y) for y in range(Ny)]
        outlet_cells = [(Nx - 1, y) for y in range(Ny)]
    elif geometry == "point":
        inlet_cells = [(5, 4), (5, 6), (5, 7)]
        outlet_cells = [(Nx - 1, y) for y in range(Ny)]
    else:
        raise ValueError(geometry)

    V_oop, iso_oop, t_oop = run_oop(Nx, Ny, mode, steps, inlet_cells, outlet_cells)
    V_vec, iso_vec, t_vec = run_vec(Nx, Ny, mode, steps, inlet_cells, outlet_cells)

    # Compare final V
    abs_V = np.abs(V_oop - V_vec)
    max_abs_V = float(abs_V.max())
    mean_V = float(V_oop.mean())

    # Compare iso (both where both reached)
    both_reached = (iso_oop >= 0) & (iso_vec >= 0)
    only_oop = (iso_oop >= 0) & ~(iso_vec >= 0)
    only_vec = ~(iso_oop >= 0) & (iso_vec >= 0)
    if both_reached.any():
        diff_iso = iso_oop[both_reached].astype(int) - iso_vec[both_reached].astype(int)
        max_abs_iso = int(np.abs(diff_iso).max())
    else:
        max_abs_iso = 0

    return {
        "mode": mode,
        "geometry": geometry,
        "Nx": Nx, "Ny": Ny, "steps": steps,
        "max_abs_V": max_abs_V,
        "mean_V": mean_V,
        "max_abs_iso": max_abs_iso,
        "iso_only_oop_cells": int(only_oop.sum()),
        "iso_only_vec_cells": int(only_vec.sum()),
        "t_oop": t_oop,
        "t_vec": t_vec,
        "speedup": t_oop / max(t_vec, 1e-9),
    }


def main():
    print(f"{'mode':9} {'geom':6} {'max|ΔV|':>11} {'max|Δiso|':>9} "
          f"{'only_oop':>9} {'only_vec':>9} {'t_oop(s)':>10} {'t_vec(s)':>10} {'speedup':>8}")
    all_max_V = 0.0
    all_max_iso = 0
    for mode in ("constant", "gradient"):
        for geom in ("line", "point"):
            r = compare_case(mode, geom)
            all_max_V = max(all_max_V, r["max_abs_V"])
            all_max_iso = max(all_max_iso, r["max_abs_iso"])
            print(
                f"{r['mode']:9} {r['geometry']:6} "
                f"{r['max_abs_V']:11.3e} {r['max_abs_iso']:9d} "
                f"{r['iso_only_oop_cells']:9d} {r['iso_only_vec_cells']:9d} "
                f"{r['t_oop']:10.3f} {r['t_vec']:10.3f} {r['speedup']:8.1f}x"
            )

    print()
    if all_max_V > 1e-8 or all_max_iso > 1:
        print(f"FAIL: max|ΔV| = {all_max_V:.3e}, max|Δiso| = {all_max_iso}")
    else:
        print(f"PASS: max|ΔV| = {all_max_V:.3e} (<= 1e-8), max|Δiso| = {all_max_iso} (<= 1)")


if __name__ == "__main__":
    main()
