"""
Optimizer V1 — AP Biomarker Extraction

Functions for measuring action potential characteristics from voltage traces.
All functions operate on numpy arrays or torch tensors.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class APInfo:
    """Information about a single detected action potential."""
    start_idx: int      # Index of upstroke start
    peak_idx: int       # Index of peak voltage
    end_idx: int        # Index of repolarization (APD90 crossing)
    start_time: float
    peak_time: float
    end_time: float
    v_peak: float
    v_min: float        # Minimum voltage before this AP


def detect_aps(V: np.ndarray, t: np.ndarray,
               v_threshold: float = -20.0,
               min_peak: float = 0.0,
               min_interval_ms: float = 100.0) -> List[APInfo]:
    """
    Detect action potentials in a voltage trace.

    Parameters
    ----------
    V : (N,) voltage trace in mV
    t : (N,) time trace in ms
    v_threshold : threshold for detecting upstroke crossing
    min_peak : minimum peak voltage to accept
    min_interval_ms : minimum interval between peaks

    Returns
    -------
    List of APInfo for each detected AP.
    """
    aps = []

    # Find peaks: V[i] > V[i-1] and V[i] > V[i+1] and V[i] > min_peak
    peak_indices = []
    for i in range(1, len(V) - 1):
        if V[i] > V[i - 1] and V[i] > V[i + 1] and V[i] > min_peak:
            if not peak_indices or (t[i] - t[peak_indices[-1]]) > min_interval_ms:
                peak_indices.append(i)

    for peak_idx in peak_indices:
        v_peak = V[peak_idx]

        # Find start: last crossing of v_threshold before peak
        start_idx = peak_idx
        for k in range(peak_idx - 1, -1, -1):
            if V[k] < v_threshold:
                start_idx = k
                break

        # v_min: minimum voltage in window before this AP
        search_start = max(0, start_idx - int(200.0 / (t[1] - t[0])))
        v_min = V[search_start:start_idx + 1].min()

        # Find end: APD90 crossing after peak
        v90 = v_peak - 0.9 * (v_peak - v_min)
        end_idx = len(V) - 1
        for k in range(peak_idx + 1, len(V)):
            if V[k] < v90:
                end_idx = k
                break

        aps.append(APInfo(
            start_idx=start_idx,
            peak_idx=peak_idx,
            end_idx=end_idx,
            start_time=t[start_idx],
            peak_time=t[peak_idx],
            end_time=t[end_idx],
            v_peak=v_peak,
            v_min=v_min,
        ))

    return aps


def measure_apd(V: np.ndarray, t: np.ndarray,
                fraction: float = 0.9) -> Optional[float]:
    """
    Measure APD at given repolarization fraction from the last complete AP.

    Returns APD in ms, or None if no AP detected.
    """
    aps = detect_aps(V, t)
    if not aps:
        return None

    # Use last complete AP
    ap = aps[-1]
    v_repol = ap.v_peak - fraction * (ap.v_peak - ap.v_min)

    # Find repolarization crossing after peak
    for k in range(ap.peak_idx + 1, len(V)):
        if V[k] < v_repol:
            # Linear interpolation for precision
            if k > 0 and V[k - 1] >= v_repol:
                frac = (v_repol - V[k - 1]) / (V[k] - V[k - 1])
                t_cross = t[k - 1] + frac * (t[k] - t[k - 1])
            else:
                t_cross = t[k]
            return t_cross - ap.peak_time

    return None


def measure_dvdt_max(V: np.ndarray, t: np.ndarray) -> Optional[float]:
    """
    Measure maximum upstroke velocity (dV/dt_max) in V/s.

    Returns value in V/s (= mV/ms), or None if no AP detected.
    """
    aps = detect_aps(V, t)
    if not aps:
        return None

    ap = aps[-1]
    # Search around upstroke
    search_start = max(0, ap.start_idx - 10)
    search_end = min(len(V) - 1, ap.peak_idx + 10)

    dVdt = np.diff(V[search_start:search_end]) / np.diff(t[search_start:search_end])
    return float(dVdt.max())


def measure_v_rest(V: np.ndarray, t: np.ndarray,
                   window_ms: float = 50.0) -> float:
    """
    Measure resting potential as the minimum voltage in the trace.

    For paced traces this captures the diastolic potential.
    For spontaneous traces this captures the MDP (maximum diastolic potential).
    """
    return float(V.min())


def measure_peak(V: np.ndarray) -> float:
    """Measure peak voltage."""
    return float(V.max())


def measure_cl(V: np.ndarray, t: np.ndarray) -> Optional[float]:
    """
    Measure spontaneous cycle length from consecutive peaks.

    Returns mean CL in ms, or None if fewer than 2 peaks.
    """
    aps = detect_aps(V, t)
    if len(aps) < 2:
        return None

    cls = [aps[i + 1].peak_time - aps[i].peak_time
           for i in range(len(aps) - 1)]
    return float(np.mean(cls))


def measure_restitution(V_traces: List[np.ndarray],
                        t_traces: List[np.ndarray],
                        di_values: List[float]) -> List[Tuple[float, float]]:
    """
    Measure APD restitution curve from S1-S2 protocol results.

    Parameters
    ----------
    V_traces : list of voltage traces, one per DI value
    t_traces : list of time arrays
    di_values : diastolic intervals in ms

    Returns
    -------
    List of (DI, APD90) tuples for successful measurements.
    """
    curve = []
    for V, t, di in zip(V_traces, t_traces, di_values):
        apd = measure_apd(V, t)
        if apd is not None:
            curve.append((di, apd))
    return curve
