"""
Pure analysis functions — tensor in, tensor out.

All functions operate on plain tensors. No simulation objects needed.
Works on whatever device the input tensors are on (CPU or GPU).

    lat = activation_time(V_history, times)
    cv = conduction_velocity(V_history, times, dx, x1, x2, y)
    apd = apd_map(V_history, times)
"""

import math
import numpy as np
import torch


def activation_time(
    V: torch.Tensor,
    times: torch.Tensor,
    threshold: float = -20.0,
) -> torch.Tensor:
    """Compute activation time at each node.

    Finds the first time V crosses threshold (upstroke) at each grid point.

    Parameters
    ----------
    V : torch.Tensor
        (n_saves, Nx, Ny) voltage history.
    times : torch.Tensor
        (n_saves,) time points in ms.
    threshold : float
        Activation threshold (mV).

    Returns
    -------
    torch.Tensor
        (Nx, Ny) activation time in ms. NaN where not activated.
    """
    # (n_saves, Nx, Ny) bool: True where V >= threshold
    above = V >= threshold
    # First time index where above is True along time axis
    # any() along time to find which nodes ever activated
    ever_activated = above.any(dim=0)  # (Nx, Ny)

    # argmax on bool gives first True index
    first_idx = above.to(torch.int8).argmax(dim=0)  # (Nx, Ny)

    # Map index to time
    lat = times[first_idx]  # (Nx, Ny)
    lat[~ever_activated] = float('nan')
    return lat


def conduction_velocity(
    V: torch.Tensor,
    times: torch.Tensor,
    dx: float,
    x1: int,
    x2: int,
    y: int,
    threshold: float = -20.0,
) -> float:
    """Measure conduction velocity between two x-indices at row y.

    Parameters
    ----------
    V : torch.Tensor
        (n_saves, Nx, Ny) voltage history.
    times : torch.Tensor
        (n_saves,) time points in ms.
    dx : float
        Grid spacing (cm).
    x1, x2 : int
        x-indices to measure between. x2 > x1.
    y : int
        y-index (row).
    threshold : float
        Activation threshold (mV).

    Returns
    -------
    float
        Conduction velocity in cm/s. NaN if activation not detected at either point.
    """
    trace1 = V[:, x1, y]  # (n_saves,)
    trace2 = V[:, x2, y]

    above1 = trace1 >= threshold
    above2 = trace2 >= threshold

    if not above1.any() or not above2.any():
        return float('nan')

    t1 = times[above1.to(torch.int8).argmax()].item()
    t2 = times[above2.to(torch.int8).argmax()].item()

    dt = t2 - t1
    if abs(dt) < 1e-12:
        return float('nan')

    distance = abs(x2 - x1) * dx  # cm
    return (distance / dt) * 1000.0  # cm/ms → cm/s


def apd_at(
    V: torch.Tensor,
    times: torch.Tensor,
    ix: int,
    iy: int,
    repol: float = 0.9,
    threshold: float = -20.0,
) -> float:
    """Compute action potential duration at a single grid point.

    APD is measured from activation to repol% repolarization.
    APD90 (repol=0.9) = time from upstroke to 90% return toward resting.

    Parameters
    ----------
    V : torch.Tensor
        (n_saves, Nx, Ny) voltage history.
    times : torch.Tensor
        (n_saves,) time points in ms.
    ix, iy : int
        Grid indices.
    repol : float
        Repolarization fraction (0.9 = APD90, 0.5 = APD50).
    threshold : float
        Activation threshold (mV).

    Returns
    -------
    float
        APD in ms. NaN if no complete AP detected.
    """
    trace = V[:, ix, iy]  # (n_saves,)

    above = trace >= threshold
    if not above.any():
        return float('nan')

    act_idx = above.to(torch.int8).argmax().item()
    V_peak = trace[act_idx:].max().item()
    V_rest = trace[0].item()  # assume resting at t=0

    # Repolarization voltage
    V_repol = V_peak - repol * (V_peak - V_rest)

    # Find first time after peak where V drops below V_repol
    peak_idx = act_idx + trace[act_idx:].argmax().item()
    post_peak = trace[peak_idx:]

    below = post_peak <= V_repol
    if not below.any():
        return float('nan')  # AP didn't complete

    repol_idx = peak_idx + below.to(torch.int8).argmax().item()
    return times[repol_idx].item() - times[act_idx].item()


def apd_map(
    V: torch.Tensor,
    times: torch.Tensor,
    repol: float = 0.9,
    threshold: float = -20.0,
) -> torch.Tensor:
    """Compute APD at every grid point.

    Parameters
    ----------
    V : torch.Tensor
        (n_saves, Nx, Ny) voltage history.
    times : torch.Tensor
        (n_saves,) time points in ms.
    repol : float
        Repolarization fraction (0.9 = APD90).
    threshold : float
        Activation threshold (mV).

    Returns
    -------
    torch.Tensor
        (Nx, Ny) APD in ms. NaN where no complete AP.
    """
    _, Nx, Ny = V.shape
    result = torch.full((Nx, Ny), float('nan'), device=V.device, dtype=V.dtype)
    for ix in range(Nx):
        for iy in range(Ny):
            result[ix, iy] = apd_at(V, times, ix, iy, repol, threshold)
    return result


def dominant_frequency(
    V: torch.Tensor,
    times: torch.Tensor,
    ix: int,
    iy: int,
) -> float:
    """Compute dominant frequency of voltage trace at a point.

    Uses FFT to find peak frequency. Useful for fibrillation analysis.

    Parameters
    ----------
    V : torch.Tensor
        (n_saves, Nx, Ny) voltage history.
    times : torch.Tensor
        (n_saves,) time points in ms.
    ix, iy : int
        Grid indices.

    Returns
    -------
    float
        Dominant frequency in Hz.
    """
    trace = V[:, ix, iy]
    n = trace.shape[0]
    dt_ms = (times[-1] - times[0]).item() / (n - 1)  # ms
    fs = 1000.0 / dt_ms  # Hz

    # FFT, take magnitude, ignore DC
    spectrum = torch.fft.rfft(trace - trace.mean())
    magnitudes = spectrum.abs()
    magnitudes[0] = 0  # ignore DC

    peak_idx = magnitudes.argmax().item()
    freqs = torch.fft.rfftfreq(n, d=dt_ms / 1000.0)  # Hz
    return freqs[peak_idx].item()


def wavefront_mask(
    V_snapshot: torch.Tensor,
    threshold: float = -20.0,
) -> torch.Tensor:
    """Detect wavefront in a single voltage frame.

    Wavefront = nodes above threshold that have at least one neighbor below.

    Parameters
    ----------
    V_snapshot : torch.Tensor
        (Nx, Ny) single voltage frame.
    threshold : float
        Activation threshold (mV).

    Returns
    -------
    torch.Tensor
        (Nx, Ny) bool — True at wavefront nodes.
    """
    above = V_snapshot >= threshold

    # Shift in 4 cardinal directions, check if any neighbor is below
    padded = torch.nn.functional.pad(above.unsqueeze(0).unsqueeze(0).float(),
                                     (1, 1, 1, 1), mode='replicate')
    padded = padded.squeeze()

    Nx, Ny = V_snapshot.shape
    has_below_neighbor = (
        (~padded[0:Nx, 1:Ny+1].bool()) |   # left neighbor
        (~padded[2:Nx+2, 1:Ny+1].bool()) | # right neighbor
        (~padded[1:Nx+1, 0:Ny].bool()) |   # top neighbor
        (~padded[1:Nx+1, 2:Ny+2].bool())   # bottom neighbor
    )

    return above & has_below_neighbor


# ============================================================================
# Phase analysis (spiral wave / reentry detection)
# ============================================================================


def phase_map(
    V: torch.Tensor,
    times: torch.Tensor,
    t_idx: int,
) -> torch.Tensor:
    """Compute instantaneous phase at every node using Hilbert transform.

    Phase is defined on [-pi, pi]. Used for detecting spiral wave tips
    (phase singularities).

    Parameters
    ----------
    V : torch.Tensor
        (n_saves, Nx, Ny) voltage history.
    times : torch.Tensor
        (n_saves,) time points in ms.
    t_idx : int
        Time index at which to compute phase.

    Returns
    -------
    torch.Tensor
        (Nx, Ny) phase in radians [-pi, pi].
    """
    _, Nx, Ny = V.shape
    # Reshape to (Nx*Ny, n_saves) for batch Hilbert
    traces = V.permute(1, 2, 0).reshape(-1, V.shape[0])  # (Nx*Ny, n_saves)

    # Hilbert transform via FFT
    n = traces.shape[1]
    F = torch.fft.fft(traces - traces.mean(dim=1, keepdim=True), dim=1)
    h = torch.zeros(n, device=V.device, dtype=V.dtype)
    h[0] = 1
    h[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    analytic = torch.fft.ifft(F * h.unsqueeze(0), dim=1)

    # Phase at requested time index
    phase = torch.atan2(analytic[:, t_idx].imag, analytic[:, t_idx].real)
    return phase.reshape(Nx, Ny)


def phase_singularities(
    phase: torch.Tensor,
) -> torch.Tensor:
    """Detect phase singularities (spiral wave tips).

    A phase singularity is a point where the phase wraps through a full
    2*pi around a closed loop. Detected via topological charge: sum of
    phase differences around each 2x2 plaquette.

    Parameters
    ----------
    phase : torch.Tensor
        (Nx, Ny) phase map in radians [-pi, pi].

    Returns
    -------
    torch.Tensor
        (Nx-1, Ny-1) float — topological charge at each plaquette center.
        Values near +1 or -1 indicate singularities. 0 = no singularity.
    """
    # Phase differences along edges of each 2x2 plaquette
    def _wrap(d):
        """Wrap phase difference to [-pi, pi]."""
        return (d + torch.pi) % (2 * torch.pi) - torch.pi

    # Four edges of each plaquette (counterclockwise)
    d1 = _wrap(phase[1:, :-1] - phase[:-1, :-1])  # bottom edge
    d2 = _wrap(phase[1:, 1:] - phase[1:, :-1])     # right edge
    d3 = _wrap(phase[:-1, 1:] - phase[1:, 1:])     # top edge
    d4 = _wrap(phase[:-1, :-1] - phase[:-1, 1:])   # left edge

    # Topological charge = sum / (2*pi), should be 0 or ±1
    charge = (d1 + d2 + d3 + d4) / (2 * torch.pi)
    return charge


# ============================================================================
# Restitution
# ============================================================================


def restitution_curve(
    V: torch.Tensor,
    times: torch.Tensor,
    ix: int,
    iy: int,
    repol: float = 0.9,
    threshold: float = -20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract APD restitution curve from a multi-beat recording.

    Measures APD and preceding diastolic interval (DI) for each beat.

    Parameters
    ----------
    V : torch.Tensor
        (n_saves, Nx, Ny) voltage history with multiple APs.
    times : torch.Tensor
        (n_saves,) time points in ms.
    ix, iy : int
        Grid indices for measurement point.
    repol : float
        Repolarization fraction (0.9 = APD90).
    threshold : float
        Activation threshold (mV).

    Returns
    -------
    DI : torch.Tensor
        (n_beats-1,) diastolic intervals in ms.
    APD : torch.Tensor
        (n_beats-1,) action potential durations in ms.
    """
    trace = V[:, ix, iy]
    n = trace.shape[0]

    # Find all upstroke crossings
    above = trace >= threshold
    # Detect rising edges: was below, now above
    rising = (~above[:-1]) & above[1:]
    rising_indices = torch.where(rising)[0] + 1  # (n_crossings,)

    if len(rising_indices) < 2:
        return torch.tensor([]), torch.tensor([])

    # For each beat, compute APD
    apd_list = []
    repol_time_list = []
    act_time_list = []

    for beat_start in rising_indices:
        act_t = times[beat_start].item()
        act_time_list.append(act_t)

        # Find peak after activation
        remaining = trace[beat_start:]
        if len(remaining) < 3:
            apd_list.append(float('nan'))
            repol_time_list.append(float('nan'))
            continue

        V_peak = remaining.max().item()
        V_rest = trace[0].item()
        V_repol = V_peak - repol * (V_peak - V_rest)

        peak_idx = beat_start + remaining.argmax().item()
        post_peak = trace[peak_idx:]
        below = post_peak <= V_repol

        if not below.any():
            apd_list.append(float('nan'))
            repol_time_list.append(float('nan'))
            continue

        repol_idx = peak_idx + below.to(torch.int8).argmax().item()
        apd_val = times[repol_idx].item() - act_t
        apd_list.append(apd_val)
        repol_time_list.append(times[repol_idx].item())

    # DI = time from end of AP_n to start of AP_{n+1}
    DI_list = []
    APD_out = []
    for i in range(len(act_time_list) - 1):
        if np.isnan(repol_time_list[i]) or np.isnan(apd_list[i + 1]):
            continue
        di = act_time_list[i + 1] - repol_time_list[i]
        if di > 0:
            DI_list.append(di)
            APD_out.append(apd_list[i + 1])

    return torch.tensor(DI_list, dtype=torch.float64), torch.tensor(APD_out, dtype=torch.float64)
