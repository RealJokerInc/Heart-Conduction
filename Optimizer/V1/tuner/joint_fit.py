"""
Optimizer V1 — constrained-scalarization joint fit on a GP emulator (Steps 3.2/3.3).

The V2 joint fit that supersedes the sequential cell→tissue pipeline (architecture §7).
It reuses the joint_refiner GP-emulator PATTERN but rebuilds it on the P-1 backend and
the resolution-aware constraint graph:

  - decision vector spans {conductances, (kinetics), D_long, D_trans} via decision_space
    (D_trans FREE); ONE apply(vector) → (kinetic-scaled model, per-axis mesh).
  - each design point is evaluated on cardiac_core (cell AP via cell_runner_cc, tissue
    CV via cc_runner) — ONE model for both observables, so kinetics is identifiable.
  - the block region is a NON-STATIONARY CLIFF: blocked propagation returns NaN, which
    is MASKED (a feasibility classifier), NOT smoothed by a 50.0 penalty the GP would
    interpolate through (the joint_refiner bug, architecture §7). CV GPs train on
    resolved+feasible points only; NaN never enters a GP target (isfinite guard).
  - METHOD is constrained scalarization, NOT 4-obj qNEHVI: minimize AP-morphology error
    s.t. CV_L/CV_T tol + r*/dx≥k + dV/dt band. It SURFACES infeasibility explicitly
    (which lock binds) instead of returning silent dominated compromises.

Cost: training points run sims at a FIXED resolved dx (feasibility-map-chosen) + an
r*/dx≥k filter — equivalent to a per-point ladder where feasible (r*/dx≥3 ⇒ CV
resolved) but ~Nx cheaper; the resolved_cv LADDER is reserved for final top-k
validation. All heavy work is injectable (`evaluate_fn`) so tests use a synthetic
oracle (no sims).
"""
from dataclasses import dataclass, field
from dataclasses import replace as _replace
import math
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from .decision_space import Axis, apply, bounds_arrays
from .cc_runner import run_1d_cable
from .cell_runner_cc import run_single_cell_cc
from .cv_estimator import rstar_cm, resolved_cv

RSTAR_OVER_DX_FLOOR = 3.0


@dataclass
class PointEval:
    """One evaluated design point (all metrics; NaN where blocked)."""
    vector: list
    apd90: float
    dvdt: float
    cv_l: float
    cv_t: float
    rstar_over_dx_l: float
    rstar_over_dx_t: float
    feasible_sim: bool          # propagated AND resolved (r*/dx≥k) on BOTH axes


@dataclass
class JointFitResult:
    theta: Dict[str, float]
    kinetics: Dict[str, float]
    D_long: float
    D_trans: float
    achieved: Dict[str, float]
    infeasible: bool = False
    binding_lock: Optional[str] = None


@dataclass
class InfeasReport:
    """Returned when no feasible (θ, D) exists — names the binding lock instead of a
    fake fit (architecture §7: surface infeasibility explicitly)."""
    binding_lock: str
    detail: str
    infeasible: bool = True


# --------------------------------------------------------------------------- eval
def _extract_D(vector, axes):
    D_long = D_trans = None
    for ax, v in zip(axes, vector):
        if ax.name == "D_long":
            D_long = float(v)
        elif ax.name == "D_trans":
            D_trans = float(v)
    return D_long, D_trans


def make_sim_evaluator(config, axes, base_theta, resolved_dx_cm, *,
                       k=RSTAR_OVER_DX_FLOOR, n_beats_cell=4):
    """The real (cardiac_core) evaluator: apply(vector) → cell AP + tissue CV at the
    resolved dx, on ONE kinetic-scaled model. Blocked axes → NaN CV → feasible_sim=False."""
    cfg_dx = _replace(config, dx_cm=resolved_dx_cm)

    def ev(vector):
        model, _mesh = apply(vector, axes, config, base_theta=base_theta)
        cell = run_single_cell_cc(None, config, model=model, n_beats=n_beats_cell)
        D_long, D_trans = _extract_D(vector, axes)
        cv_l = run_1d_cable(None, D_long, cfg_dx, model=model)
        cv_t = run_1d_cable(None, D_trans, cfg_dx, model=model)
        rox_l = (rstar_cm(D_long, cv_l) / resolved_dx_cm) if math.isfinite(cv_l) else float("nan")
        rox_t = (rstar_cm(D_trans, cv_t) / resolved_dx_cm) if math.isfinite(cv_t) else float("nan")
        feasible = (math.isfinite(cv_l) and math.isfinite(cv_t)
                    and rox_l >= k and rox_t >= k)
        apd = cell.apd90 if cell.apd90 is not None else float("nan")
        dvdt = cell.dvdt_max if cell.dvdt_max is not None else float("nan")
        return PointEval(list(vector), apd, dvdt, cv_l, cv_t, rox_l, rox_t, bool(feasible))

    return ev


def build_training_set(axes, evaluate_fn, *, n=40, seed=42, seed_points=None):
    """Sobol design over the decision box → evaluated PointEvals (X, evals). `seed_points`
    (a list of known-propagating warm-start vectors) are prepended so the CV GP has
    feasible anchors — pure Sobol rarely lands in the thin slow-CV feasible manifold."""
    lo, hi = bounds_arrays(axes)
    bounds = torch.tensor([lo, hi], dtype=torch.float64)
    X = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed).squeeze(1)
    if seed_points:
        S = torch.tensor(seed_points, dtype=torch.float64)
        X = torch.cat([S, X], dim=0)
    evals = [evaluate_fn(x.tolist()) for x in X]
    return X, evals


# ------------------------------------------------------------------------ emulator
def _fit_gp(X, y):
    # Normalize inputs to the unit cube — the D axes (~1e-4) and conductance scales
    # (~1) span 4 orders of magnitude; without this the RBF kernel cannot fit CV and
    # the D-solve fails (GP over-predicts CV at low D → D driven to the floor).
    gp = SingleTaskGP(X, y, outcome_transform=Standardize(m=1),
                      input_transform=Normalize(d=X.shape[-1]))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


def _feature_idx(axes, predicate):
    return [i for i, a in enumerate(axes) if predicate(a)]


def build_emulator(X, evals, axes):
    """GPs trained on REDUCED feature sets so CV-irrelevant axes don't add noise to the
    CV emulator (architecture open-Q8 — the fix that makes a high-dim fit tractable):
      - cv_l ~ f(g_Na, kinetics, D_long); cv_t ~ f(g_Na, kinetics, D_trans)  (the
        repolarizing/Ca conductances do NOT drive the upstroke-limited CV);
      - apd, dvdt ~ f(conductances, kinetics)  (D does not affect the 0-D cell AP);
      - feasibility ~ f(full vector)  (depends on D AND excitability).
    NaN CV never enters a CV GP target (isfinite guard — block masking, NOT a 50.0 penalty).
    Each GP carries its feature indices so prediction/D-solve slice consistently."""
    feas = torch.tensor([e.feasible_sim for e in evals])
    y_feas = torch.tensor([[1.0 if e.feasible_sim else 0.0] for e in evals], dtype=torch.float64)
    gp_feas = _fit_gp(X, y_feas)
    feas_idx = list(range(X.shape[-1]))

    cvl_idx = _feature_idx(axes, lambda a: a.name == "g_Na" or a.subsystem == "kinetic" or a.name == "D_long")
    cvt_idx = _feature_idx(axes, lambda a: a.name == "g_Na" or a.subsystem == "kinetic" or a.name == "D_trans")
    ap_idx = _feature_idx(axes, lambda a: a.subsystem in ("cond", "kinetic"))

    def _gp_on(row_ok, attr, idx):
        rows = [(x, getattr(e, attr)) for x, e in zip(X, evals)
                if row_ok(e) and math.isfinite(getattr(e, attr))]
        if len(rows) < 2:
            return None
        Xg = torch.stack([r[0] for r in rows])[:, idx]
        yg = torch.tensor([[r[1]] for r in rows], dtype=torch.float64)
        assert torch.isfinite(yg).all(), f"NaN leaked into {attr} GP target"
        return _fit_gp(Xg, yg)

    gp_apd = _gp_on(lambda e: True, "apd90", ap_idx)
    gp_dvdt = _gp_on(lambda e: True, "dvdt", ap_idx)
    gp_cvl = _gp_on(lambda e: e.feasible_sim, "cv_l", cvl_idx)
    gp_cvt = _gp_on(lambda e: e.feasible_sim, "cv_t", cvt_idx)
    return {"feas": gp_feas, "feas_idx": feas_idx,
            "apd": gp_apd, "dvdt": gp_dvdt, "ap_idx": ap_idx,
            "cvl": gp_cvl, "cvl_idx": cvl_idx, "cvt": gp_cvt, "cvt_idx": cvt_idx,
            "n_feasible": int(feas.sum())}


def _pred(gp, C, idx):
    with torch.no_grad():
        return gp.posterior(C[:, idx]).mean.squeeze(-1)


def _solve_D_batch(gp_cv, gp_idx, C, idx_D, target_cv, D_lo, D_hi, n_iter=30):
    """Batched bisection: for each candidate, find D (at axis idx_D) whose EMULATED CV
    equals target_cv (CV is monotone-increasing in D). Vectorized over all candidates.
    gp_idx = the CV GP's feature columns. Clamps to [D_lo, D_hi] when the endpoints
    don't bracket the target."""
    n = C.shape[0]
    lo = torch.full((n,), float(D_lo), dtype=C.dtype)
    hi = torch.full((n,), float(D_hi), dtype=C.dtype)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        Cm = C.clone()
        Cm[:, idx_D] = mid
        cv = _pred(gp_cv, Cm, gp_idx)
        below = cv < target_cv          # need MORE D → raise the lower bound
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- constrained scalarize
def refine_joint_cc(axes, evaluate_fn, targets, *, config=None, base_theta=None,
                    n_training=40, n_candidates=4000, n_validate=15, seed=42,
                    cv_tol=0.12, dvdt_band=(20.0, 130.0), k=RSTAR_OVER_DX_FLOOR,
                    emulator_margin=0.6, seed_points=None, verbose=False):
    """Joint fit: train emulator, run constrained scalarization on it, validate top-k.

    Objective: minimize AP-morphology error (|APD−target|) subject to hard constraints
    CV_L/CV_T within tol, r*/dx≥k (from D/CV), and dV/dt in band. Returns a validated
    JointFitResult, or an InfeasReport naming the binding lock when the feasible set is
    empty (architecture §7).
    """
    CV_L, CV_T = targets.cv_longitudinal, targets.cv_transverse
    X, evals = build_training_set(axes, evaluate_fn, n=n_training, seed=seed,
                                  seed_points=seed_points)
    emu = build_emulator(X, evals, axes)

    # If almost nothing propagated+resolved, the binding lock is resolution/excitability.
    if emu["n_feasible"] < 2 or emu["cvt"] is None or emu["cvl"] is None:
        return InfeasReport(
            binding_lock="resolution/excitability",
            detail=(f"only {emu['n_feasible']}/{n_training} training points reached "
                    f"r*/dx≥{k} on both axes — CV_T={CV_T} unresolvable in this box "
                    f"(refine dx / widen g_Na / add kinetics)."))

    lo, hi = bounds_arrays(axes)
    bounds = torch.tensor([lo, hi], dtype=torch.float64)
    C = draw_sobol_samples(bounds=bounds, n=n_candidates, q=1, seed=seed + 1).squeeze(1)

    idx_Dl = [i for i, a in enumerate(axes) if a.name == "D_long"][0]
    idx_Dt = [i for i, a in enumerate(axes) if a.name == "D_trans"][0]
    dx_cm = getattr(config, "dx_mm", 1.0) / 10.0 if config is not None else None

    # SOLVE D on the emulator so every candidate HITS its CV target (D determines CV
    # given θ — the emulator analog of the secant). Random D rarely lands within tol of
    # BOTH CV_L and CV_T at once (a thin manifold); solving makes the CV constraints hit
    # by construction, then dV/dt + r*/dx + feasibility do the real carving.
    Dl_range = next(a.bounds for a in axes if a.name == "D_long")
    Dt_range = next(a.bounds for a in axes if a.name == "D_trans")
    C[:, idx_Dl] = _solve_D_batch(emu["cvl"], emu["cvl_idx"], C, idx_Dl, CV_L, Dl_range[0], Dl_range[1])
    C[:, idx_Dt] = _solve_D_batch(emu["cvt"], emu["cvt_idx"], C, idx_Dt, CV_T, Dt_range[0], Dt_range[1])

    feas_p = _pred(emu["feas"], C, emu["feas_idx"])
    cvl_p = _pred(emu["cvl"], C, emu["cvl_idx"])
    cvt_p = _pred(emu["cvt"], C, emu["cvt_idx"])
    apd_p = _pred(emu["apd"], C, emu["ap_idx"]) if emu["apd"] is not None else torch.full((len(C),), float("nan"))
    dvdt_p = _pred(emu["dvdt"], C, emu["ap_idx"]) if emu["dvdt"] is not None else torch.full((len(C),), float("nan"))

    # constraint masks (track each so we can name the binding lock). Candidate CV must
    # be well INSIDE tol (× emulator_margin) so GP drift doesn't push the validated
    # exact CV outside tol — the fix for the "emulator_drift" failure mode.
    m_feas = feas_p > 0.5
    m_cvl = (cvl_p - CV_L).abs() / CV_L < cv_tol * emulator_margin
    m_cvt = (cvt_p - CV_T).abs() / CV_T < cv_tol * emulator_margin
    m_dvdt = (dvdt_p >= dvdt_band[0]) & (dvdt_p <= dvdt_band[1])
    if dx_cm:
        # r*/dx from the SOLVED D and the TARGET CV (CV≈target by construction).
        rox_l = (C[:, idx_Dl] / (CV_L / 1000.0)) / dx_cm
        rox_t = (C[:, idx_Dt] / (CV_T / 1000.0)) / dx_cm
        m_rox = (rox_l >= k) & (rox_t >= k)
    else:
        m_rox = torch.ones(len(C), dtype=torch.bool)

    feasible = m_feas & m_cvl & m_cvt & m_dvdt & m_rox
    if not feasible.any():
        # name the lock that eliminated the most CV-feasible candidates
        cv_ok = m_feas & m_cvl & m_cvt
        lock = _binding_lock(m_feas, m_cvl, m_cvt, m_dvdt, m_rox, len(C))
        return InfeasReport(binding_lock=lock,
                            detail=f"no candidate satisfied all constraints "
                                   f"(feas={int(m_feas.sum())}, cvL={int(m_cvl.sum())}, "
                                   f"cvT={int(m_cvt.sum())}, dvdt={int(m_dvdt.sum())}, "
                                   f"r*/dx={int(m_rox.sum())} of {len(C)}).")

    # rank feasible by AP-morphology error (|APD − target|); validate top-k on the oracle
    ap_err = (apd_p - targets.apd_90).abs()
    ap_err = torch.where(feasible, ap_err, torch.full_like(ap_err, float("inf")))
    order = torch.argsort(ap_err)[:n_validate]

    best = None
    for i in order.tolist():
        ev = evaluate_fn(C[i].tolist())
        if not ev.feasible_sim:
            continue
        cvl_ok = abs(ev.cv_l - CV_L) / CV_L < cv_tol
        cvt_ok = abs(ev.cv_t - CV_T) / CV_T < cv_tol
        if cvl_ok and cvt_ok and (best is None or
                                  abs(ev.apd90 - targets.apd_90) < abs(best.apd90 - targets.apd_90)):
            best = ev

    if best is None:
        return InfeasReport(binding_lock="emulator_drift",
                            detail="emulator-feasible candidates failed real validation "
                                   "(refit near the validation set).")

    theta, kinetics, D_long, D_trans = _unpack(best.vector, axes, base_theta)
    return JointFitResult(
        theta=theta, kinetics=kinetics, D_long=D_long, D_trans=D_trans,
        achieved={"apd90": best.apd90, "dvdt": best.dvdt, "cv_l": best.cv_l,
                  "cv_t": best.cv_t, "rstar_over_dx_l": best.rstar_over_dx_l,
                  "rstar_over_dx_t": best.rstar_over_dx_t})


def _binding_lock(m_feas, m_cvl, m_cvt, m_dvdt, m_rox, n):
    """Name the constraint that makes the feasible set empty, from the per-constraint
    pass counts (D is already solved to hit each CV target, so m_rox is usually all-pass;
    the binding wall is normally CV_T or the CV_L∩CV_T anisotropy, not resolution)."""
    counts = {
        "feasibility/propagation": int(m_feas.sum()),
        "CV_L match": int(m_cvl.sum()),
        "CV_T match (slow transverse target)": int(m_cvt.sum()),
        "dV/dt band": int(m_dvdt.sum()),
        "r*/dx>=3 resolution": int(m_rox.sum()),
    }
    both_cv = int((m_cvl & m_cvt).sum())
    if both_cv == 0 and counts["CV_L match"] > 0 and counts["CV_T match (slow transverse target)"] > 0:
        return ("anisotropy — CV_L and CV_T not simultaneously reachable "
                f"(CV_L {counts['CV_L match']}/{n}, CV_T {counts['CV_T match (slow transverse target)']}/{n}, both 0)")
    binding = min(counts, key=counts.get)
    return f"{binding} — fewest passing ({counts[binding]}/{n})"


def _unpack(vector, axes, base_theta):
    theta, kinetics = dict(base_theta or {}), {}
    D_long = D_trans = None
    for ax, v in zip(axes, vector):
        v = float(v)
        if ax.subsystem == "cond":
            theta[ax.name] = v
        elif ax.subsystem == "kinetic":
            kinetics[ax.name] = v
        elif ax.name == "D_long":
            D_long = v
        elif ax.name == "D_trans":
            D_trans = v
    return theta, kinetics, D_long, D_trans
