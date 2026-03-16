"""
Optimizer V1 — Tissue Fitter (Analytical CV Scaling)

Uses CV ∝ √D to compute D from a single reference simulation,
then verifies with one confirmation sim. Falls back to BO refinement
only if the analytical estimate is off by > 10%.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .config import TuningConfig, TuningTargets, TISSUE_PARAMS
from .tissue_runner import run_cv_measurement, CVResult


@dataclass
class TissueFitResult:
    """Result from tissue fitting."""
    D_long: float
    D_trans: float
    cv_long_achieved: float
    cv_trans_achieved: float
    all_D: torch.Tensor
    all_cv: torch.Tensor
    n_sims: int = 0             # Total cable sims run


def fit_tissue(theta_ionic: torch.Tensor,
               config: TuningConfig,
               targets: TuningTargets,
               n_iterations: int = 30,
               verbose: bool = True) -> TissueFitResult:
    """
    Fit D_long and D_trans using analytical CV ∝ √D scaling.

    Strategy:
    1. Run ONE reference sim at D_ref to measure CV_ref
    2. Compute D analytically: D = (CV_target / CV_ref)² × D_ref
    3. Verify with ONE confirmation sim
    4. If error > 10%, do Newton refinement (1-2 more sims)
    """
    dtype = config.dtype
    n_sims = 0

    # Step 1: Reference simulation
    D_ref = 0.0001  # cm²/ms
    if verbose:
        print(f"  Reference sim at D = {D_ref}...")
    ref_result = run_cv_measurement(theta_ionic, D_ref, config, n_beats=3)
    n_sims += 1

    if ref_result.cv is None or not ref_result.converged:
        # Try a larger D
        D_ref = 0.0003
        if verbose:
            print(f"  Retry at D = {D_ref}...")
        ref_result = run_cv_measurement(theta_ionic, D_ref, config, n_beats=3)
        n_sims += 1

    if ref_result.cv is None or not ref_result.converged:
        if verbose:
            print("  WARNING: Reference CV failed, using defaults")
        return TissueFitResult(
            D_long=D_ref, D_trans=D_ref * 0.5,
            cv_long_achieved=0.0, cv_trans_achieved=0.0,
            all_D=torch.tensor([[D_ref]], dtype=dtype),
            all_cv=torch.tensor([[0.0]], dtype=dtype),
            n_sims=n_sims,
        )

    cv_ref = ref_result.cv
    if verbose:
        print(f"  Reference: CV = {cv_ref:.1f} cm/s at D = {D_ref}")

    # Step 2: Analytical estimate
    D_long_est = (targets.cv_longitudinal / cv_ref) ** 2 * D_ref
    D_trans_est = (targets.cv_transverse / cv_ref) ** 2 * D_ref

    D_long_est = np.clip(D_long_est, *TISSUE_PARAMS['D_long'])
    D_trans_est = np.clip(D_trans_est, *TISSUE_PARAMS['D_trans'])

    if verbose:
        print(f"  Analytical: D_long = {D_long_est:.6f}, D_trans = {D_trans_est:.6f}")

    # Step 3: Verify D_long
    D_long_opt, cv_long, n = _verify_and_refine(
        theta_ionic, config, targets.cv_longitudinal,
        D_long_est, cv_ref, D_ref, TISSUE_PARAMS['D_long'],
        verbose, "D_long")
    n_sims += n

    # Step 4: Verify D_trans
    D_trans_opt, cv_trans, n = _verify_and_refine(
        theta_ionic, config, targets.cv_transverse,
        D_trans_est, cv_ref, D_ref, TISSUE_PARAMS['D_trans'],
        verbose, "D_trans")
    n_sims += n

    if verbose:
        print(f"  Total tissue sims: {n_sims}")

    return TissueFitResult(
        D_long=D_long_opt,
        D_trans=D_trans_opt,
        cv_long_achieved=cv_long,
        cv_trans_achieved=cv_trans,
        all_D=torch.tensor([[D_long_opt, D_trans_opt]], dtype=dtype),
        all_cv=torch.tensor([[cv_long, cv_trans]], dtype=dtype),
        n_sims=n_sims,
    )


def _verify_and_refine(theta_ionic, config, cv_target, D_est,
                       cv_ref, D_ref, D_bounds,
                       verbose, label) -> Tuple[float, float, int]:
    """
    Verify analytical D estimate, refine with Newton if off by > 10%.

    Returns (D_optimal, cv_achieved, n_sims).
    """
    n_sims = 0

    # Verification sim
    result = run_cv_measurement(theta_ionic, D_est, config, n_beats=3)
    n_sims += 1

    if result.cv is None or not result.converged:
        if verbose:
            print(f"  {label}: verification failed at D = {D_est:.6f}")
        return D_est, 0.0, n_sims

    cv_achieved = result.cv
    rel_error = abs(cv_achieved - cv_target) / cv_target

    if verbose:
        print(f"  {label}: D = {D_est:.6f} -> CV = {cv_achieved:.1f} cm/s "
              f"(target {cv_target:.1f}, err {rel_error*100:.1f}%)")

    if rel_error < 0.10:
        return D_est, cv_achieved, n_sims

    # Newton refinement: use the two data points (D_ref, cv_ref) and (D_est, cv_achieved)
    # CV ∝ √D → dCV/dD = CV / (2D)
    # Newton step: D_new = D_est + (cv_target - cv_achieved) / (dCV/dD)
    for _ in range(2):  # Max 2 Newton steps
        dCV_dD = cv_achieved / (2.0 * D_est)
        if abs(dCV_dD) < 1e-10:
            break
        D_new = D_est + (cv_target - cv_achieved) / dCV_dD
        D_new = np.clip(D_new, *D_bounds)

        result = run_cv_measurement(theta_ionic, D_new, config, n_beats=3)
        n_sims += 1

        if result.cv is None or not result.converged:
            break

        D_est = D_new
        cv_achieved = result.cv
        rel_error = abs(cv_achieved - cv_target) / cv_target

        if verbose:
            print(f"  {label}: Newton -> D = {D_est:.6f} -> CV = {cv_achieved:.1f} "
                  f"(err {rel_error*100:.1f}%)")

        if rel_error < 0.05:
            break

    return D_est, cv_achieved, n_sims
