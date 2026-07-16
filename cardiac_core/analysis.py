"""
Pure analysis functions — tensor in, tensor out.

All functions operate on plain tensors. No simulation objects needed.
Works on whatever device the input tensors are on (CPU or GPU).

    lat = activation_time(V_history, times)
    cv = conduction_velocity(V_history, times, dx, x1, x2, y)
    apd = apd_map(V_history, times)
"""

import math
import warnings

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
    if V.shape[0] == 0:   # empty run (0 save-points) — return NaN map, don't argmax an empty axis (F1)
        return torch.full(V.shape[1:], float('nan'), dtype=times.dtype, device=V.device)
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
    dome_aware: bool = True,
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
    dome_aware : bool
        If True (default), the repolarization endpoint is the LAST crossing of
        V_repol within the beat, so a spike-and-dome morphology's early notch
        does not truncate low-repol APDs (B4). Set False for a plain
        first-crossing endpoint.

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

    # B3: bound the peak/repolarization search to THIS beat — the window ends at
    # the next upstroke (or the trace end), so a later, taller beat cannot inflate
    # this beat's V_peak (which previously maxed over the entire remaining trace).
    rising = (~above[:-1]) & above[1:]
    rising_idx = torch.where(rising)[0] + 1  # all upstroke indices
    future = rising_idx[rising_idx > act_idx]
    end = int(future[0].item()) if future.numel() > 0 else trace.shape[0]

    beat = trace[act_idx:end]
    V_peak = beat.max().item()
    V_rest = trace[0].item()  # assume resting at t=0

    # Repolarization voltage
    V_repol = V_peak - repol * (V_peak - V_rest)

    peak_idx = act_idx + int(beat.argmax().item())
    post_peak = trace[peak_idx:end]

    below = post_peak <= V_repol
    if not below.any():
        return float('nan')  # AP didn't complete within this beat

    if dome_aware:
        # B4: the LAST above->below crossing of V_repol is the final repolarization;
        # the first crossing can land on an early spike-and-dome notch for low repol%.
        # For a monotonic (dome-free) repolarization there is one crossing, so this
        # is identical to the first-crossing result.
        crossings = (~below[:-1]) & below[1:]
        cross_idx = torch.where(crossings)[0]
        local = (int(cross_idx[-1].item()) + 1 if cross_idx.numel() > 0
                 else int(below.to(torch.int8).argmax().item()))
    else:
        local = int(below.to(torch.int8).argmax().item())

    repol_idx = peak_idx + local
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
    if V.shape[0] == 0:   # empty run — all-NaN map (skip the per-node loop over an empty axis) (F1)
        return result
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
    if V.shape[0] == 0:   # empty run — no spectrum to take (F1 empty-run hardening)
        return float('nan')
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
    if V.shape[0] == 0:   # empty run — no trace to Hilbert-transform (F1 empty-run hardening)
        return torch.full((Nx, Ny), float('nan'), device=V.device, dtype=V.dtype)
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


# ---------------------------------------------------------------------------
# Eikonal / source-sink metrics (source_sink_mismatch_investigation)
#
# Separate the dynamic curvature slowing (-D*kappa) from kinematic fanning, so a
# diverging "fan" can be quantified rather than dismissed. Used by the Fig-4C/D
# campaign (Research/Active/source_sink_mismatch_investigation/FIG4C_BLOCK_TEST_PLAN.md).
#
# Sign convention: n_hat = grad(LAT)/|grad(LAT)| points along propagation (LAT
# increases downstream). kappa = div(n_hat) is +1/r for a convex/expanding front,
# matching the eikonal relation  CV_n = CV0 - D*kappa.
# ---------------------------------------------------------------------------

def activation_time_interp(V, times, threshold: float = -40.0):
    """Interpolated activation-time (LAT) map — numpy, sub-frame accurate.

    Unlike :func:`activation_time` (nearest-frame, torch), this linearly
    interpolates the threshold crossing between saved frames, which the eikonal
    CV = 1/|grad LAT| needs to avoid frame-quantized velocity artifacts.

    Parameters
    ----------
    V : array (n_saves, Nx, Ny) voltage history.
    times : array (n_saves,) ms.
    threshold : crossing level (mV). Default -40 (matches the diag scripts).

    Returns
    -------
    np.ndarray (Nx, Ny) LAT in ms; NaN where never activated.
    """
    V = np.asarray(V)
    times = np.asarray(times, dtype=float)
    above = V >= threshold
    ever = above.any(axis=0)
    idx = np.argmax(above, axis=0)                 # first crossing frame
    idxc = np.clip(idx, 1, len(times) - 1)
    v1 = np.take_along_axis(V, idxc[None], 0)[0]
    v0 = np.take_along_axis(V, (idxc - 1)[None], 0)[0]
    t1 = times[idxc]
    t0 = times[idxc - 1]
    denom = np.where(v1 == v0, 1.0, v1 - v0)
    lat = t0 + (threshold - v0) * (t1 - t0) / denom
    lat[idx == 0] = times[0]
    lat[~ever] = np.nan
    return lat


def _smooth3(a):
    """NaN-aware 3x3 box mean (edge-padded). Reduces div(n_hat) noise."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(a, 1, mode="edge")
    win = sliding_window_view(pad, (3, 3))
    with np.errstate(invalid="ignore"):
        out = np.nanmean(win, axis=(-1, -2))
    return out


def front_metrics(lat, dx: float, smooth: bool = True) -> dict:
    """Front normal, normal conduction velocity, and curvature from a LAT map.

    Parameters
    ----------
    lat : array (Nx, Ny) activation times (ms), NaN off-front.
    dx  : grid spacing (cm).
    smooth : 3x3-smooth the unit-normal field before taking div (curvature).

    Returns
    -------
    dict with numpy (Nx, Ny) arrays:
        cv_n  : 1/|grad LAT|  (cm/ms), front-normal conduction velocity
        kappa : div(n_hat)    (1/cm),  + = convex/expanding
        n_x, n_y : unit propagation-direction components
    """
    lat = np.asarray(lat, dtype=float)
    gx, gy = np.gradient(lat, dx)                   # ms/cm
    mag = np.hypot(gx, gy)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_n = np.where(mag > 1e-9, 1.0 / mag, np.nan)
        nx = np.where(mag > 1e-9, gx / mag, np.nan)
        ny = np.where(mag > 1e-9, gy / mag, np.nan)
    if smooth:
        nx, ny = _smooth3(nx), _smooth3(ny)
    dnx, _ = np.gradient(nx, dx)
    _, dny = np.gradient(ny, dx)
    kappa = dnx + dny                              # 1/cm
    return {"cv_n": cv_n, "kappa": kappa, "n_x": nx, "n_y": ny}


def fit_eikonal(cv_n, kappa, mask=None) -> dict:
    """Linear fit CV_n = CV0 - D*kappa over valid cells.

    Returns dict(CV0 [cm/ms], D_eik [cm^2/ms], r2, r_star=D/CV0 [cm], n).
    """
    cv = np.asarray(cv_n, dtype=float).ravel()
    k = np.asarray(kappa, dtype=float).ravel()
    good = np.isfinite(cv) & np.isfinite(k)
    if mask is not None:
        good &= np.asarray(mask).ravel().astype(bool)
    cv, k = cv[good], k[good]
    if cv.size < 3:
        return {"CV0": np.nan, "D_eik": np.nan, "r2": np.nan,
                "r_star": np.nan, "n": int(cv.size)}
    slope, intercept = np.polyfit(k, cv, 1)        # cv = slope*k + intercept
    D_eik = -slope
    CV0 = intercept
    pred = slope * k + intercept
    ss_res = float(np.sum((cv - pred) ** 2))
    ss_tot = float(np.sum((cv - cv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r_star = D_eik / CV0 if (np.isfinite(CV0) and abs(CV0) > 1e-12) else np.nan
    return {"CV0": float(CV0), "D_eik": float(D_eik), "r2": float(r2),
            "r_star": float(r_star), "n": int(cv.size)}


# ============================================================================
# Aggregate / per-beat / axis analysis (P2 usability helpers)
# ============================================================================


def dominant_frequency_map(
    V: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Dominant frequency at EVERY node (batched rfft), for fibrillation mapping.

    Like :func:`dominant_frequency` but returns an ``(Nx, Ny)`` map in one FFT over the
    flattened field. Warns when the frequency resolution ``1/(n*dt)`` is coarse (record
    longer / smaller ``save_every`` to resolve DF differences — B10).

    Returns ``(Nx, Ny)`` DF in Hz; NaN map for an empty run.
    """
    if V.shape[0] == 0:
        return torch.full(V.shape[1:], float('nan'), device=V.device, dtype=V.dtype)
    n, Nx, Ny = V.shape
    if n < 4:
        return torch.full((Nx, Ny), float('nan'), device=V.device, dtype=V.dtype)
    dt_ms = (times[-1] - times[0]).item() / (n - 1)
    df_res_hz = 1000.0 / (n * dt_ms)
    if df_res_hz > 0.5:
        warnings.warn(
            f"dominant_frequency_map: frequency resolution is {df_res_hz:.2f} Hz "
            f"({n} samples at dt={dt_ms:.3g} ms); DF differences below this are "
            f"unresolved — record longer or use a smaller save_every.",
            stacklevel=2,
        )
    traces = V.reshape(n, -1)                       # (n, Nx*Ny)
    traces = traces - traces.mean(dim=0, keepdim=True)
    spec = torch.fft.rfft(traces, dim=0).abs()      # (nf, Nx*Ny)
    spec[0] = 0.0                                    # ignore DC
    freqs = torch.fft.rfftfreq(n, d=dt_ms / 1000.0).to(V.device)  # Hz
    peak = spec.argmax(dim=0)                        # (Nx*Ny,)
    df = freqs[peak].reshape(Nx, Ny)
    # Masked/out-of-domain nodes are NaN across time (B8); their rfft is NaN and argmax
    # would return a phantom frequency — set them NaN, not a bogus low frequency.
    non_finite = ~torch.isfinite(V).all(dim=0)      # (Nx, Ny)
    if non_finite.any():
        df = df.clone()
        df[non_finite] = float('nan')
    return df


def cv_between(
    V: torch.Tensor,
    times: torch.Tensor,
    p1: tuple,
    p2: tuple,
    dx: float,
    dy: float = None,
    threshold: float = -20.0,
) -> float:
    """Conduction velocity along the line between two nodes ``p1=(ix,iy)`` and ``p2=(ix,iy)``.

    Generalizes :func:`conduction_velocity` (which is x-axis only) to any direction —
    the CV is the Euclidean distance between the points over their activation-time
    difference. NaN if either point never activates or they co-activate.
    """
    dy = dx if dy is None else dy
    (i1, j1), (i2, j2) = p1, p2
    t1_series = V[:, i1, j1] >= threshold
    t2_series = V[:, i2, j2] >= threshold
    if not t1_series.any() or not t2_series.any():
        return float('nan')
    t1 = times[t1_series.to(torch.int8).argmax()].item()
    t2 = times[t2_series.to(torch.int8).argmax()].item()
    dt = t2 - t1
    if abs(dt) < 1e-12:
        return float('nan')
    dist = math.hypot((i2 - i1) * dx, (j2 - j1) * dy)  # cm
    return (dist / abs(dt)) * 1000.0                    # cm/s


def radial_cv(
    V: torch.Tensor,
    times: torch.Tensor,
    center: tuple,
    dx: float,
    dy: float = None,
    threshold: float = -20.0,
) -> torch.Tensor:
    """Outward conduction-velocity map from a point source at ``center=(ix,iy)``.

    For a point-stimulated expanding wave, each activated node's radial CV is its
    distance from ``center`` over ``(LAT[node] - LAT[center])``. Returns an ``(Nx, Ny)``
    map in cm/s; NaN at the center, at nodes that never activate, and where the LAT
    difference is non-positive (upstream of the source).
    """
    dy = dx if dy is None else dy
    ci, cj = center
    lat = activation_time(V, times, threshold)        # (Nx, Ny), NaN where unactivated
    Nx, Ny = lat.shape
    if not bool(torch.isfinite(lat[ci, cj])):
        warnings.warn(
            f"radial_cv: center {center} never activates (LAT is NaN there), so the whole "
            f"map is NaN — pass the point-source node as center.",
            stacklevel=2,
        )
    ii = torch.arange(Nx, device=V.device).reshape(Nx, 1).to(lat.dtype)
    jj = torch.arange(Ny, device=V.device).reshape(1, Ny).to(lat.dtype)
    dist = torch.sqrt(((ii - ci) * dx) ** 2 + ((jj - cj) * dy) ** 2)   # cm
    dlat = lat - lat[ci, cj]                           # ms
    cv = torch.full((Nx, Ny), float('nan'), device=V.device, dtype=lat.dtype)
    ok = torch.isfinite(dlat) & (dlat > 1e-9)
    cv[ok] = (dist[ok] / dlat[ok]) * 1000.0            # cm/s
    return cv


def apd_per_beat(
    V: torch.Tensor,
    times: torch.Tensor,
    ix: int,
    iy: int,
    repol: float = 0.9,
    threshold: float = -20.0,
    dome_aware: bool = True,
) -> torch.Tensor:
    """APD of EACH beat at a node (multi-beat recording).

    Each beat's peak/repolarization is bounded to that beat (so a later beat can't
    corrupt an earlier one — same rule as :func:`apd_at`). Returns a ``(n_beats,)``
    tensor; a beat that doesn't complete within the run is NaN.
    """
    trace = V[:, ix, iy]
    above = trace >= threshold
    # Beats are detected by their UPSTROKE (a below->above crossing). A beat already in
    # progress at t=0 (trace starts depolarized) has no clean upstroke and no valid
    # resting reference, so it is NOT measured (measuring it gave a bogus 0.0 ms).
    rising = (~above[:-1]) & above[1:]
    starts = torch.where(rising)[0] + 1
    if starts.numel() == 0:
        return torch.tensor([], dtype=torch.float64, device=V.device)
    V_rest = trace[0].item()
    out = []
    for k in range(len(starts)):
        s = int(starts[k].item())
        end = int(starts[k + 1].item()) if k + 1 < len(starts) else trace.shape[0]
        beat = trace[s:end]
        V_peak = beat.max().item()
        V_repol = V_peak - repol * (V_peak - V_rest)
        pk = s + int(beat.argmax().item())
        post = trace[pk:end]
        below = post <= V_repol
        if not below.any():
            out.append(float('nan'))
            continue
        if dome_aware:
            crossings = (~below[:-1]) & below[1:]
            ci = torch.where(crossings)[0]
            local = (int(ci[-1].item()) + 1 if ci.numel() > 0
                     else int(below.to(torch.int8).argmax().item()))
        else:
            local = int(below.to(torch.int8).argmax().item())
        out.append(times[pk + local].item() - times[s].item())
    return torch.tensor(out, dtype=torch.float64, device=V.device)


def restitution_slope(DI, APD) -> dict:
    """Max APD-restitution slope and DI* (the DI where slope crosses 1, alternans onset).

    Takes the ``(DI, APD)`` arrays from :func:`restitution_curve`. Guards the divide by
    a zero DI-spacing (B13). Returns dict(max_slope, DI_star [ms], n). ``DI_star`` is NaN
    if the slope never reaches 1 (or < 2 points).
    """
    DI = np.asarray(DI.cpu() if isinstance(DI, torch.Tensor) else DI, dtype=float)
    APD = np.asarray(APD.cpu() if isinstance(APD, torch.Tensor) else APD, dtype=float)
    if DI.size < 2:
        return {"max_slope": float('nan'), "DI_star": float('nan'), "n": int(DI.size)}
    order = np.argsort(DI)
    DI, APD = DI[order], APD[order]
    dDI = np.diff(DI)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(np.abs(dDI) > 1e-9, np.diff(APD) / dDI, np.nan)  # B13 guard
    mid_DI = 0.5 * (DI[1:] + DI[:-1])
    max_slope = float(np.nanmax(slopes)) if np.isfinite(slopes).any() else float('nan')
    # DI* = the DI where the (normally decreasing) slope descends through 1 — the
    # alternans-onset boundary. Interpolate between the two midpoints that bracket
    # slope=1 (slope[k] >= 1 > slope[k+1]) going ascending in DI. NaN if the slope
    # never crosses 1 within the sampled range.
    DI_star = float('nan')
    for k in range(len(slopes) - 1):
        s0, s1 = slopes[k], slopes[k + 1]
        if np.isfinite(s0) and np.isfinite(s1) and s0 >= 1.0 > s1:
            frac = (s0 - 1.0) / (s0 - s1)            # in [0, 1]
            DI_star = float(mid_DI[k] + frac * (mid_DI[k + 1] - mid_DI[k]))
            # Keep scanning: the LAST (largest-DI) descending crossing is the alternans
            # boundary above which the slope stays < 1. On a normal monotone-decreasing
            # restitution curve there is exactly one crossing, so this is unchanged.
    return {"max_slope": max_slope, "DI_star": DI_star, "n": int(DI.size)}
