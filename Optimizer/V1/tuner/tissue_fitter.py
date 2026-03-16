"""
Optimizer V1 — Tissue Fitter (Analytical CV Scaling)

Supports both monodomain (single D) and bidomain (D_i, D_e with ratio
constraint). Uses CV ∝ √D_eff for analytical warm-start, then verifies
with real tissue simulations.

For bidomain: D_eff = D_i·D_e/(D_i+D_e), with D_e/D_i = ratio (fixed).
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
    D_long: float               # Monodomain D (or D_eff for bidomain)
    D_trans: float
    cv_long_achieved: float
    cv_trans_achieved: float
    all_D: torch.Tensor
    all_cv: torch.Tensor
    n_sims: int = 0
    # Bidomain-specific
    D_i_long: Optional[float] = None
    D_e_long: Optional[float] = None
    D_i_trans: Optional[float] = None
    D_e_trans: Optional[float] = None


def _run_cv(theta_ionic, D_or_Deff, config, n_beats=3):
    """
    Run CV measurement on the appropriate engine.

    For monodomain: D_or_Deff is the diffusion coefficient D.
    For bidomain: D_or_Deff is D_eff, decomposed into (D_i, D_e) via ratio.
    """
    if config.engine == 'bidomain':
        from .tissue_runner_bidomain import run_cv_measurement_bidomain
        r = config.De_Di_ratio
        # D_eff = D_i * r / (1 + r) → D_i = D_eff * (1 + r) / r
        D_i = D_or_Deff * (1.0 + r) / r
        D_e = D_i * r
        return run_cv_measurement_bidomain(
            theta_ionic, D_i, D_e, config, n_beats=n_beats)
    else:
        return run_cv_measurement(
            theta_ionic, D_or_Deff, config, n_beats=n_beats)


def fit_tissue(theta_ionic: torch.Tensor,
               config: TuningConfig,
               targets: TuningTargets,
               n_iterations: int = 30,
               verbose: bool = True) -> TissueFitResult:
    """
    Fit tissue diffusion parameters using analytical CV ∝ √D_eff scaling.

    For bidomain, optimizes D_eff then decomposes into (D_i, D_e) using
    the fixed ratio D_e/D_i = config.De_Di_ratio.
    """
    dtype = config.dtype
    n_sims = 0
    is_bidomain = config.engine == 'bidomain'

    if verbose and is_bidomain:
        print(f"  Engine: BIDOMAIN (D_e/D_i ratio = {config.De_Di_ratio:.3f})")

    # Step 1: Reference simulation
    D_ref = 0.0001  # D_eff reference
    if verbose:
        if is_bidomain:
            r = config.De_Di_ratio
            D_i_ref = D_ref * (1.0 + r) / r
            D_e_ref = D_i_ref * r
            print(f"  Reference: D_eff={D_ref}, D_i={D_i_ref:.6f}, D_e={D_e_ref:.6f}")
        else:
            print(f"  Reference sim at D = {D_ref}...")

    ref_result = _run_cv(theta_ionic, D_ref, config, n_beats=3)
    n_sims += 1

    if ref_result.cv is None or not ref_result.converged:
        D_ref = 0.0003
        if verbose:
            print(f"  Retry at D_eff = {D_ref}...")
        ref_result = _run_cv(theta_ionic, D_ref, config, n_beats=3)
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
        print(f"  Reference: CV = {cv_ref:.1f} cm/s at D_eff = {D_ref}")

    # Step 2: Analytical estimate
    Deff_long_est = (targets.cv_longitudinal / cv_ref) ** 2 * D_ref
    Deff_trans_est = (targets.cv_transverse / cv_ref) ** 2 * D_ref

    Deff_long_est = np.clip(Deff_long_est, *TISSUE_PARAMS['D_long'])
    Deff_trans_est = np.clip(Deff_trans_est, *TISSUE_PARAMS['D_trans'])

    if verbose:
        print(f"  Analytical: D_eff_long = {Deff_long_est:.6f}, "
              f"D_eff_trans = {Deff_trans_est:.6f}")

    # Step 3: Verify D_long
    Deff_long_opt, cv_long, n = _verify_and_refine(
        theta_ionic, config, targets.cv_longitudinal,
        Deff_long_est, cv_ref, D_ref, TISSUE_PARAMS['D_long'],
        verbose, "D_long")
    n_sims += n

    # Step 4: Verify D_trans
    Deff_trans_opt, cv_trans, n = _verify_and_refine(
        theta_ionic, config, targets.cv_transverse,
        Deff_trans_est, cv_ref, D_ref, TISSUE_PARAMS['D_trans'],
        verbose, "D_trans")
    n_sims += n

    if verbose:
        print(f"  Total tissue sims: {n_sims}")

    # Build result with bidomain decomposition if applicable
    result = TissueFitResult(
        D_long=Deff_long_opt,
        D_trans=Deff_trans_opt,
        cv_long_achieved=cv_long,
        cv_trans_achieved=cv_trans,
        all_D=torch.tensor([[Deff_long_opt, Deff_trans_opt]], dtype=dtype),
        all_cv=torch.tensor([[cv_long, cv_trans]], dtype=dtype),
        n_sims=n_sims,
    )

    if is_bidomain:
        r = config.De_Di_ratio
        result.D_i_long = Deff_long_opt * (1.0 + r) / r
        result.D_e_long = result.D_i_long * r
        result.D_i_trans = Deff_trans_opt * (1.0 + r) / r
        result.D_e_trans = result.D_i_trans * r

        if verbose:
            print(f"  Bidomain decomposition (ratio={r:.3f}):")
            print(f"    Long:  D_i={result.D_i_long:.6f}, D_e={result.D_e_long:.6f}")
            print(f"    Trans: D_i={result.D_i_trans:.6f}, D_e={result.D_e_trans:.6f}")

    return result


def _verify_and_refine(theta_ionic, config, cv_target, D_est,
                       cv_ref, D_ref, D_bounds,
                       verbose, label) -> Tuple[float, float, int]:
    """Verify analytical D_eff estimate, refine with Newton if off by > 10%."""
    n_sims = 0

    result = _run_cv(theta_ionic, D_est, config, n_beats=3)
    n_sims += 1

    if result.cv is None or not result.converged:
        if verbose:
            print(f"  {label}: verification failed at D_eff = {D_est:.6f}")
        return D_est, 0.0, n_sims

    cv_achieved = result.cv
    rel_error = abs(cv_achieved - cv_target) / cv_target

    if verbose:
        print(f"  {label}: D_eff = {D_est:.6f} -> CV = {cv_achieved:.1f} cm/s "
              f"(target {cv_target:.1f}, err {rel_error*100:.1f}%)")

    if rel_error < 0.10:
        return D_est, cv_achieved, n_sims

    for _ in range(2):
        dCV_dD = cv_achieved / (2.0 * D_est)
        if abs(dCV_dD) < 1e-10:
            break
        D_new = D_est + (cv_target - cv_achieved) / dCV_dD
        D_new = np.clip(D_new, *D_bounds)

        result = _run_cv(theta_ionic, D_new, config, n_beats=3)
        n_sims += 1

        if result.cv is None or not result.converged:
            break

        D_est = D_new
        cv_achieved = result.cv
        rel_error = abs(cv_achieved - cv_target) / cv_target

        if verbose:
            print(f"  {label}: Newton -> D_eff = {D_est:.6f} -> CV = {cv_achieved:.1f} "
                  f"(err {rel_error*100:.1f}%)")

        if rel_error < 0.05:
            break

    return D_est, cv_achieved, n_sims
