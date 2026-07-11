"""
Optimizer V1 — unified decision-space registry + apply() (PLAN Step 3.1 / arch §5-6).

The joint fit's decision vector is HETEROGENEOUS — ionic conductances, (optional) Na
kinetics, and tissue diffusion — living in disconnected homes today (PHAS13_REGISTRY;
kinetic instance attrs; TISSUE_PARAMS). This module gives each axis a uniform
`{name, subsystem, bounds, apply}` declaration behind ONE
    apply(vector) -> (scaled_model, mesh)
so the fit is genuinely joint, not three bolted-on stages. Presupposes the P-1 backend
unification (one cardiac_core model for AP + CV).

Constraint graph (arch §6): HARD = definitional only (r*/dx ≥ k); SOFT = leading-order
priors used as WARM-STARTS, never hard ties (CV∝√D, 2:1 anisotropy). In particular
D_trans is a FREE axis (arch §6 correction) — the 2:1 relation is a warm-start, NOT
`D_trans = D_long/ratio²`. Caveat (arch open-Q7): free D_trans only has teeth once
CV_T is an INDEPENDENT measured target; with PARKER's derived CV_T it relands near
D_long/ratio² — recorded, not fenced.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

from .config import get_params_for_tier, TISSUE_PARAMS
from .cc_runner import _build_model
from .chip import chip_mesh, ANISOTROPY_RATIO


@dataclass
class Axis:
    """One decision-vector axis."""
    name: str
    subsystem: str          # 'cond' | 'kinetic' | 'diffusion'
    bounds: tuple           # (lo, hi)


# Constraint graph (arch §6). Hard = definitional; soft = warm-start priors only.
CONSTRAINT_GRAPH = {
    "hard": ["rstar_over_dx >= k"],                         # resolve source-sink (k≈3)
    "soft": ["CV ∝ sqrt(D)", f"anisotropy CV_L/CV_T ≈ {ANISOTROPY_RATIO}:1"],
}


def _kinetic_axes() -> List[Axis]:
    """Na-kinetic axes IF P1.5 registered them in config (else empty).

    Kept a soft/optional import so decision_space works both before (conductance-only)
    and after the gated kinetics model change — decision_space imports FROM config, not
    the reverse (arch §5)."""
    try:
        from .config import KINETIC_REGISTRY
    except ImportError:
        return []
    return [Axis(name, "kinetic", bounds) for name, bounds in KINETIC_REGISTRY.items()]


def build_axes(tier: int = 2, *, gNa_floor: Optional[float] = None,
               include_kinetics: bool = True, include_dx: bool = False,
               dx_bounds_cm=(0.002, 0.05)) -> List[Axis]:
    """Assemble the decision axes: conductances (tier) + D_long/D_trans (FREE) +
    optional kinetics + optional dx. `gNa_floor` widens the g_Na lower bound for the
    sweep (arch lock-2: ≤0.17).

    `include_dx` promotes the grid spacing to a PHYSICAL decision axis (subsystem
    'grid') — the tissue's effective functional-unit/coupling scale — instead of a
    fixed resolved value. This lets the fit reach slow, source-sink-limited CV (which
    lives at coarse dx / r*/dx<3) at a moderate non-blocking D, provided the SAME dx is
    used for the downstream reentry application (fit↔app consistency, so it's a physical
    commitment, not grid error). `dx_bounds_cm` bounds the plausible functional-unit
    scale (default 0.02–0.5 mm)."""
    axes: List[Axis] = []
    for name, spec in get_params_for_tier(tier).items():
        lo, hi = spec.bounds
        if name == "g_Na" and gNa_floor is not None:
            lo = gNa_floor
        axes.append(Axis(name, "cond", (lo, hi)))
    # Diffusion — both FREE (arch §6): D_trans is NOT tied to D_long.
    axes.append(Axis("D_long", "diffusion", TISSUE_PARAMS["D_long"]))
    axes.append(Axis("D_trans", "diffusion", TISSUE_PARAMS["D_trans"]))
    if include_kinetics:
        axes.extend(_kinetic_axes())
    if include_dx:
        axes.append(Axis("dx_cm", "grid", tuple(dx_bounds_cm)))
    return axes


def bounds_arrays(axes: List[Axis]):
    """Return (lower, upper) lists aligned to `axes` (for the optimizer box)."""
    lo = [a.bounds[0] for a in axes]
    hi = [a.bounds[1] for a in axes]
    return lo, hi


def apply(vector, axes: List[Axis], config, *, base_theta: Optional[dict] = None):
    """Materialize a flat decision `vector` (aligned to `axes`) into (model, mesh).

    - 'cond' axes → conductance scaling on the ionic model (over `base_theta`).
    - 'kinetic' axes → per-instance attributes on the model (e.g. tau_m_scale).
    - 'diffusion' axes (D_long, D_trans) → per-axis mesh D (D_trans FREE).
    """
    cond, kinetic = dict(base_theta or {}), {}
    D_long = D_trans = dx_cm = None
    for ax, val in zip(axes, vector):
        v = float(val)
        if ax.subsystem == "cond":
            cond[ax.name] = v
        elif ax.subsystem == "kinetic":
            kinetic[ax.name] = v
        elif ax.name == "D_long":
            D_long = v
        elif ax.name == "D_trans":
            D_trans = v
        elif ax.name == "dx_cm":
            dx_cm = v

    model = _build_model(cond, config)
    for kname, kval in kinetic.items():        # P1.5 kinetic multipliers (no-op if absent)
        setattr(model, kname, kval)

    dx_mm = (dx_cm * 10.0) if dx_cm is not None else getattr(config, "dx_mm", 0.1)
    mesh = chip_mesh(domain_mm=getattr(config, "domain_mm", 16.0), dx_mm=dx_mm,
                     D_long=D_long, D_trans=D_trans,
                     ionic_model=config.ionic_model, dt=config.dt)
    return model, mesh


def dx_of(vector, axes):
    """The candidate's tissue dx (cm) if a 'dx_cm' axis exists, else None."""
    for ax, val in zip(axes, vector):
        if ax.name == "dx_cm":
            return float(val)
    return None
