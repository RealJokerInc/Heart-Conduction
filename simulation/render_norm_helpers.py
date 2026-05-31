"""Shared helpers for normalized isochrone / per-column LAT figures.

Normalization principle: factor out wave-slowing.
  iso[y, x] = step at which cell (y, x) crossed threshold (or -1 if never).
  mean_x[x] = mean over y of iso[y, x]   (column arrival time).
  Wave slows down outward → mean_x grows non-linearly with x.

For isochrones: instead of contour levels at evenly-spaced ABSOLUTE step
numbers (which compress outer columns), pick contour levels at evenly-spaced
x-positions, using the corresponding mean_x values as time levels.

For per-column LAT: instead of plotting (mean_iso − iso) in step units, plot
(mean_iso − iso) / Δmean_x — fractional lag relative to the column's own
traversal time.
"""
from __future__ import annotations

import numpy as np


def column_arrival_times(iso: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_idx, mean_x) for fully-activated columns."""
    invalid = iso < 0
    ok = ~invalid.any(axis=0)
    iso_f = iso.astype(np.float64)
    iso_f[invalid] = np.nan
    x_idx = np.where(ok)[0]
    mean_x = np.nanmean(iso_f[:, x_idx], axis=0)
    return x_idx, mean_x


def x_evenly_spaced_levels(iso: np.ndarray, num_levels: int) -> np.ndarray:
    """Return contour levels (step values) that correspond to evenly-spaced
    x-positions of the wavefront's mean. So `num_levels` contours, each at the
    time when the wave reaches a different x-position (evenly distributed).
    """
    x_idx, mean_x = column_arrival_times(iso)
    if len(x_idx) < 2:
        return np.array([])
    # Pick num_levels x-positions evenly across the reached range
    x_pick = np.linspace(x_idx[0], x_idx[-1], num_levels).astype(int)
    levels = []
    for x in x_pick:
        # find the closest activated column to x
        if x in x_idx:
            i = int(np.where(x_idx == x)[0][0])
            levels.append(mean_x[i])
        else:
            i = int(np.argmin(np.abs(x_idx - x)))
            levels.append(mean_x[i])
    levels = np.array(sorted(set(levels)))
    return levels


def per_column_dev_normalized(iso: np.ndarray, x: int) -> tuple[np.ndarray, float]:
    """Return (dev_y, dt) for column x, where:
      dev_y[y] = mean_x − iso[y, x]    (positive = ahead of column mean, in steps)
      dt       = traversal time (mean_x − mean_{x_prev})   for normalization
    Returns dev / dt (fractional lag).
    """
    col_iso = iso[:, x].astype(np.float64)
    valid = col_iso >= 0
    if not valid.all():
        return None, None  # don't plot partial columns; caller handles it
    mean_iso = float(col_iso.mean())
    dev = mean_iso - col_iso
    # Estimate per-column traversal time
    x_idx, mean_x = column_arrival_times(iso)
    if x not in x_idx:
        return dev, 1.0
    i = int(np.where(x_idx == x)[0][0])
    if i == 0:
        # use forward difference
        dt = mean_x[1] - mean_x[0] if len(mean_x) > 1 else 1.0
    else:
        dt = mean_x[i] - mean_x[i - 1]
    if dt <= 0:
        dt = 1.0
    return dev, dt
