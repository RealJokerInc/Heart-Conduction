# -*- coding: utf-8 -*-
"""Vectorised storage-tank simulator (numpy).

Mathematically equivalent to tanks_channel_states.py but operates on
(Ny, Nx) arrays directly instead of iterating Channel objects. Speedup
vs. the OOP version is typically 100-500x.

Feature flags (orthogonal axes, all default to John's original behaviour):

  mode             rule for per-channel pump rate
                       'constant' : max_pump · sqrt((V_src - θ)/(V_max - θ)),
                                    capped at max_pump (John's rule).
                       'gradient' : k · (V_src - V_dst)  (Fickian).

  directionality   how the pipe gates on V_src vs V_dst
                       'one_way'        : pipe fires only when V_src > V_dst
                                          (John's original).
                       'bidirectional'  : pipe fires whenever V_src > θ;
                                          both directions of each pipe pair
                                          fire and net flux is f(V_A) - f(V_B).

  boundary         how out-of-bounds neighbours are handled
                       'zero_pad'    : ghost cells valued 0 (no-flux Neumann).
                       'reflect_y'   : mirror padding on y boundaries only.
                       'reflect_all' : mirror padding on all boundaries.
                       For reflection modes, ghost-bound flux is folded back
                       into the mirror real cell (mass-conserving).

  damping_cap      whether to clamp pump amount when it exceeds |V_src-V_dst|
                       True (John's "minimal case" branch) | False.

The Jacobi update is unchanged: per-step, all channels compute amt from the
current V; updates land in flux_in/flux_out buffers; V is updated at end.
Inlets/outlets are forced AFTER the update each step.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


MOORE_8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return b where b[y, x] = a[y - dy, x - dx], zero-padded at edges."""
    out = np.roll(a, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        out[:dy, :] = 0.0
    elif dy < 0:
        out[dy:, :] = 0.0
    if dx > 0:
        out[:, :dx] = 0.0
    elif dx < 0:
        out[:, dx:] = 0.0
    return out


def _valid_mask(Ny: int, Nx: int, dy: int, dx: int) -> np.ndarray:
    """True at (y, x) iff (y+dy, x+dx) lies inside the grid."""
    m = np.ones((Ny, Nx), dtype=bool)
    if dy == 1:
        m[-1, :] = False
    elif dy == -1:
        m[0, :] = False
    if dx == 1:
        m[:, -1] = False
    elif dx == -1:
        m[:, 0] = False
    return m


def _fold_ghosts_to_real(flux_p: np.ndarray, Ny: int, Nx: int,
                         pad_y: int, pad_x: int) -> np.ndarray:
    """Fold ghost-cell flux contributions back into mirror real cells (mass-
    conserving reflection BC). Returns the (Ny, Nx) real-cell flux."""
    real = flux_p[pad_y:pad_y + Ny, pad_x:pad_x + Nx].copy()
    if pad_y > 0:
        real[1, :] += flux_p[0, pad_x:pad_x + Nx]
        real[Ny - 2, :] += flux_p[Ny + pad_y, pad_x:pad_x + Nx]
    if pad_x > 0:
        real[:, 1] += flux_p[pad_y:pad_y + Ny, 0]
        real[:, Nx - 2] += flux_p[pad_y:pad_y + Ny, Nx + pad_x]
    if pad_y > 0 and pad_x > 0:
        real[1, 1] += flux_p[0, 0]
        real[1, Nx - 2] += flux_p[0, Nx + pad_x]
        real[Ny - 2, 1] += flux_p[Ny + pad_y, 0]
        real[Ny - 2, Nx - 2] += flux_p[Ny + pad_y, Nx + pad_x]
    return real


def run(
    Nx: int,
    Ny: int,
    mode: str,
    steps: int,
    inlet_cells: Sequence[tuple[int, int]] | None = None,
    outlet_cells: Sequence[tuple[int, int]] | None = None,
    threshold: float = 45.0,
    max_volume: float = 100.0,
    max_pump: float = 5.0,
    gradient_k: float = 0.05,
    directionality: str = "one_way",
    boundary: str = "zero_pad",
    damping_cap: bool = True,
    record_isochrone: bool = True,
    record_history: bool = False,
    snap_every: int = 100,
    connectivity: str = "moore8",
    threshold_gate: bool = True,
) -> dict:
    """Run the vectorised tank simulation. Returns a dict with at least
    'V' and 'iso'; if record_history, also 'snaps', 'snap_steps', 'activity'.

    Ablation knobs (added 2026-04-29 for boundary-effect attribution):
      connectivity     'moore8'    : 8-neighbour (cardinals + diagonals; John's default)
                       'cardinal4' : 4-neighbour (cardinals only; matches monodomain
                                     5-point Laplacian connectivity).
      threshold_gate   True  : pipes only fire when V_src > threshold (John's default;
                              the "fired_p" gate that creates discrete activation).
                       False : drop the threshold gate; pipes fire as long as
                              gap > 0 (one_way) or always (bidirectional).
                              Reduces system to discrete diffusion.
    """
    if directionality not in ("one_way", "bidirectional"):
        raise ValueError(f"unknown directionality: {directionality!r}")
    if boundary not in ("zero_pad", "reflect_y", "reflect_all"):
        raise ValueError(f"unknown boundary: {boundary!r}")
    if connectivity not in ("moore8", "cardinal4", "moore8_iso"):
        raise ValueError(f"unknown connectivity: {connectivity!r}")

    # direction_weight[(dy, dx)] scales the pump rate per direction.
    # 'moore8'      : uniform weight 1 on all 8 (John's default).
    #                 D_eff = (1 + 2*1) * k = 3k.  CFL borderline at k=0.08.
    # 'cardinal4'   : weight 1 on the 4 cardinals only.
    #                 D_eff = (1 + 0)   * k = 1k.  Stable.
    # 'moore8_iso'  : Patra-Kaluza isotropic 9-pt: cardinals 4/6, diagonals 1/6.
    #                 The 1/6 prefactor is the canonical normalisation that
    #                 makes the stencil reproduce the continuum Laplacian
    #                 magnitude (NOT just the ratio). Without 1/6, the
    #                 effective D_eff = 6k, which violates CFL for k=0.08
    #                 and produces grid-scale mosaic instability.
    #                 With 1/6:  D_eff = (4/6 + 2/6) * k = 1k.  Stable.
    #                 Deficit ratio is unchanged: (4/6 + 1/6) / 1 = 5/6.
    if connectivity == "moore8":
        directions = MOORE_8
        direction_weight = {d: 1.0 for d in MOORE_8}
    elif connectivity == "moore8_iso":
        directions = MOORE_8
        direction_weight = {
            d: (4.0/6.0 if (d[0] == 0) ^ (d[1] == 0) else 1.0/6.0)
            for d in MOORE_8
        }
    else:  # cardinal4
        directions = [(dy, dx) for (dy, dx) in MOORE_8 if (dy == 0) ^ (dx == 0)]
        direction_weight = {d: 1.0 for d in directions}

    pad_y = 1 if boundary in ("reflect_y", "reflect_all") else 0
    pad_x = 1 if boundary == "reflect_all" else 0

    V = np.zeros((Ny, Nx), dtype=np.float64)
    iso = np.full((Ny, Nx), -1, dtype=np.int32)

    inlet_mask = np.zeros((Ny, Nx), dtype=bool)
    outlet_mask = np.zeros((Ny, Nx), dtype=bool)
    if inlet_cells:
        for x, y in inlet_cells:
            inlet_mask[y, x] = True
    if outlet_cells:
        for x, y in outlet_cells:
            outlet_mask[y, x] = True

    pumpfactor = np.sqrt(max_volume - threshold)

    snaps: list[np.ndarray] = []
    snap_steps: list[int] = []
    activity = np.zeros(steps, dtype=np.float64) if record_history else None

    V_prev = V.copy()

    for step in range(steps):
        if pad_y > 0 or pad_x > 0:
            Vp = np.pad(V, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
        else:
            Vp = V

        Ny_p, Nx_p = Vp.shape
        flux_in_p = np.zeros_like(Vp)
        flux_out_p = np.zeros_like(Vp)
        fired_p = Vp > threshold

        for dy, dx in directions:
            valid = _valid_mask(Ny_p, Nx_p, dy, dx)
            V_dst = _shift(Vp, -dy, -dx)
            gap = Vp - V_dst

            # threshold_gate=False removes the fired_p constraint, leaving the
            # gate as just (gap > 0) & valid for one_way (pure Fickian) or
            # just valid for bidirectional.
            if threshold_gate:
                gate = fired_p
            else:
                gate = np.ones_like(fired_p)
            if directionality == "one_way":
                gate = gate & (gap > 0) & valid
            else:  # bidirectional
                gate = gate & valid

            if mode == "constant":
                base = max_pump * np.sqrt(np.clip(Vp - threshold, 0.0, None)) / pumpfactor
                base = np.minimum(base, max_pump)
                if damping_cap:
                    over = base > np.abs(gap)
                    if directionality == "one_way":
                        amt = np.where(over, gap / 4.0, base)
                    else:
                        # In bidirectional mode the cap can fire when gap is
                        # negative; clamp to non-negative to keep this pipe's
                        # contribution physically meaningful (the reverse pipe
                        # handles the V_dst > V_src case in its own iteration).
                        amt = np.where(over, np.maximum(gap / 4.0, 0.0), base)
                else:
                    amt = base
                amt = np.where(gate, amt, 0.0)
                amt = np.maximum(amt, 0.0)
            elif mode == "gradient":
                amt = np.where(gate, gradient_k * gap, 0.0)
            else:
                raise ValueError(f"unknown mode: {mode!r}")

            # Per-direction weight (1 for uniform Moore-8, 4/1 for isotropic 9pt).
            w_dir = direction_weight[(dy, dx)]
            if w_dir != 1.0:
                amt = amt * w_dir

            flux_out_p += amt
            flux_in_p += _shift(amt, dy, dx)

        if pad_y > 0 or pad_x > 0:
            real_in = _fold_ghosts_to_real(flux_in_p, Ny, Nx, pad_y, pad_x)
            real_out = flux_out_p[pad_y:pad_y + Ny, pad_x:pad_x + Nx]
        else:
            real_in = flux_in_p
            real_out = flux_out_p

        V_new = V - real_out + real_in
        np.clip(V_new, 0.0, max_volume, out=V_new)
        V_new[inlet_mask] = max_volume
        V_new[outlet_mask] = 0.0

        if record_history:
            activity[step] = float(np.linalg.norm(V_new - V_prev))
            V_prev = V.copy()
            if step % snap_every == 0 or step == steps - 1:
                snaps.append(V_new.copy())
                snap_steps.append(step)

        V = V_new

        if record_isochrone:
            newly = (V > threshold) & (iso < 0)
            if newly.any():
                iso[newly] = step

    out = {"V": V, "iso": iso}
    if record_history:
        out["snaps"] = np.array(snaps)
        out["snap_steps"] = np.array(snap_steps)
        out["activity"] = activity
    return out
