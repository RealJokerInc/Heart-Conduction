"""
Optimizer V1 — convergence-aware CV estimator (PLAN Step 1.2 / architecture §4).

FIT vs RESOLVE: dx/dt are NUMERICAL and must be RESOLVED, never fit. The discretized
measurement is CV_num = CV_phys + ε(θ,D,dx,dt), ε→0 as dx→0. If the optimizer moved
dx to hit a target it would exploit ε (a grid fudge that evaporates on the real chip).
So CV is measured by a dx-LADDER (fix θ,D,dt; vary dx ONLY) and extrapolated to the
resolved limit; dx moves to CANCEL ε, not to achieve a target.

Two hard limits (architecture §4):
- You cannot extrapolate THROUGH a block: below r*/dx≈1, CV_num→NaN (a bifurcation).
- The usable floor is r*/dx≈3, NOT 1: in 1<r*/dx<3 the wave propagates but CV0 is
  grid-corrupted and even sign-inverts (source_sink S0b) — a WRONG trend. So every
  ladder rung must sit at r*/dx≳k (k=3); if the finest rung is still below k, the
  point is "not resolvable at this ladder" → converged=False (refine dx or report).

r* = D/CV (electrotonic space constant): D [cm²/ms], CV [cm/s] → CV/1000 [cm/ms] →
r* = D/(CV/1000) [cm]. r*/dx uses dx [cm].

Solver is implicit Crank-Nicolson (unconditionally stable) → dt is ACCURACY-bounded,
not CFL-bounded. `check_dt=True` verifies the finest-dx CV is dt-converged (halve dt
once; ΔCV ≤ tol). (config.py:44's "CFL: dx²/4D" comment is stale for the CN solver.)
"""

import math
from dataclasses import replace

from .cc_runner import run_1d_cable

# Default ladder (cm). Fine enough that a chip-window D (r*≈60–160 µm) can reach
# r*/dx≥3 (needs dx≲r*/3≈20–50 µm). Coarse→fine; the finest rung is the reference.
DEFAULT_DX_LADDER = (0.004, 0.002, 0.001)
RSTAR_OVER_DX_FLOOR = 3.0        # k — resolve source-sink, not just marginally propagate


def rstar_cm(D: float, cv_cm_s: float) -> float:
    """Electrotonic space constant r* = D/CV in cm (NaN if non-propagating)."""
    if not (math.isfinite(cv_cm_s) and cv_cm_s > 0):
        return float("nan")
    return D / (cv_cm_s / 1000.0)


def _extrapolate(cvs, dxs):
    """Richardson to dx→0 assuming CV(dx)=CV0 + a·dx (leading order), from the finest
    two rungs. If they already plateau, this returns ≈ the finest CV. cvs/dxs are
    coarse→fine."""
    cv1, cv2 = cvs[-2], cvs[-1]
    dx1, dx2 = dxs[-2], dxs[-1]
    if dx1 == dx2:
        return cv2
    return cv2 + (cv2 - cv1) * dx2 / (dx1 - dx2)


def resolved_cv(theta_ionic, D: float, config, *, dx_ladder=DEFAULT_DX_LADDER,
                k: float = RSTAR_OVER_DX_FLOOR, t_end=None, check_dt: bool = False,
                dt_tol: float = 0.03) -> dict:
    """Resolved CV for (θ, D): dx-ladder → extrapolated CV + achieved r*/dx.

    Returns
    -------
    dict with:
      cv_resolved     : extrapolated CV (cm/s), or NaN if not resolvable
      rstar           : r* at cv_resolved (cm), or NaN
      rstar_over_dx   : r*/dx at the FINEST rung, or NaN
      converged       : True only if every rung propagates AND r*/dx≥k (extrapolable)
      rungs           : per-dx [{dx, cv, rstar, rstar_over_dx}]
      dt_adequate     : (only if check_dt) True if halving dt at the finest dx moves CV ≤ dt_tol
    """
    rungs = []
    for dx in dx_ladder:
        cfg = replace(config, dx_cm=dx)
        cv = run_1d_cable(theta_ionic, D, cfg, t_end=t_end)
        rs = rstar_cm(D, cv)
        rungs.append({"dx": dx, "cv": cv, "rstar": rs,
                      "rstar_over_dx": (rs / dx) if math.isfinite(rs) else float("nan")})

    finest = rungs[-1]
    # Every rung must propagate AND be at r*/dx≥k — never extrapolate through the
    # block (NaN) or the corrupted 1<r*/dx<3 band (wrong trend).
    resolvable = all(
        math.isfinite(r["cv"]) and r["cv"] > 0
        and math.isfinite(r["rstar_over_dx"]) and r["rstar_over_dx"] >= k
        for r in rungs
    )
    if not resolvable:
        return {"cv_resolved": float("nan"), "rstar": finest["rstar"],
                "rstar_over_dx": finest["rstar_over_dx"], "converged": False,
                "rungs": rungs}

    cv_resolved = _extrapolate([r["cv"] for r in rungs], [r["dx"] for r in rungs])
    rs_res = rstar_cm(D, cv_resolved)
    out = {"cv_resolved": cv_resolved, "rstar": rs_res,
           "rstar_over_dx": (rs_res / finest["dx"]) if math.isfinite(rs_res) else float("nan"),
           "converged": True, "rungs": rungs}

    if check_dt:
        dx_fine = finest["dx"]
        cfg_fine = replace(config, dx_cm=dx_fine)
        cv_base = run_1d_cable(theta_ionic, D, cfg_fine, t_end=t_end)
        cfg_halfdt = replace(cfg_fine, dt=config.dt * 0.5)
        cv_half = run_1d_cable(theta_ionic, D, cfg_halfdt, t_end=t_end)
        if math.isfinite(cv_base) and cv_base > 0 and math.isfinite(cv_half):
            out["dt_adequate"] = abs(cv_half - cv_base) / cv_base <= dt_tol
        else:
            out["dt_adequate"] = False
    return out
