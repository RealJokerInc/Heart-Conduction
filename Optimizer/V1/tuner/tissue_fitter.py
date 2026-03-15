"""
Optimizer V1 — Tissue Fitter (D Optimization)

Single-objective BayesOpt to find D_long and D_trans that match
target conduction velocities.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from .config import TuningConfig, TuningTargets, TISSUE_PARAMS
from .tissue_runner import run_cv_measurement, CVResult


@dataclass
class TissueFitResult:
    """Result from tissue fitting."""
    D_long: float               # Optimal longitudinal D (cm^2/ms)
    D_trans: float              # Optimal transverse D (cm^2/ms)
    cv_long_achieved: float     # Achieved longitudinal CV (cm/s)
    cv_trans_achieved: float    # Achieved transverse CV (cm/s)
    all_D: torch.Tensor         # All evaluated D values
    all_cv: torch.Tensor        # All measured CVs


def _analytical_warmstart(cv_target: float, cv_baseline: float,
                          D_baseline: float) -> float:
    """
    Estimate D from target CV using CV ∝ sqrt(D).

    D_init = (cv_target / cv_baseline)^2 * D_baseline
    """
    if cv_baseline <= 0:
        return D_baseline
    return (cv_target / cv_baseline) ** 2 * D_baseline


def fit_tissue(theta_ionic: torch.Tensor,
               config: TuningConfig,
               targets: TuningTargets,
               n_iterations: int = 30,
               verbose: bool = True) -> TissueFitResult:
    """
    Fit D_long and D_trans to match target CVs.

    Parameters
    ----------
    theta_ionic : best ionic parameter scaling from cell fitter
    config : TuningConfig
    targets : TuningTargets
    n_iterations : BO evaluations
    verbose : print progress

    Returns
    -------
    TissueFitResult
    """
    dtype = config.dtype

    # Step 1: Measure baseline CV with default D
    D_default = 0.0001  # cm^2/ms (reasonable starting point)
    baseline = run_cv_measurement(theta_ionic, D_default, config)

    if baseline.cv is None or not baseline.converged:
        if verbose:
            print("  Baseline CV measurement failed, using fallback D_default")
        cv_baseline = 10.0  # Assume something reasonable
    else:
        cv_baseline = baseline.cv
        if verbose:
            print(f"  Baseline CV = {cv_baseline:.1f} cm/s at D = {D_default}")

    # Step 2: Analytical warm-start
    D_long_init = _analytical_warmstart(targets.cv_longitudinal, cv_baseline, D_default)
    D_trans_init = _analytical_warmstart(targets.cv_transverse, cv_baseline, D_default)

    # Clamp to bounds
    D_long_init = np.clip(D_long_init, *TISSUE_PARAMS['D_long'])
    D_trans_init = np.clip(D_trans_init, *TISSUE_PARAMS['D_trans'])

    if verbose:
        print(f"  Warm-start: D_long = {D_long_init:.6f}, D_trans = {D_trans_init:.6f}")

    # Step 3: Optimize D_long
    D_long_opt, cv_long = _optimize_D(
        theta_ionic, config,
        target_cv=targets.cv_longitudinal,
        D_init=D_long_init,
        D_bounds=TISSUE_PARAMS['D_long'],
        n_iterations=n_iterations // 2,
        verbose=verbose,
        label="D_long",
    )

    # Step 4: Optimize D_trans
    D_trans_opt, cv_trans = _optimize_D(
        theta_ionic, config,
        target_cv=targets.cv_transverse,
        D_init=D_trans_init,
        D_bounds=TISSUE_PARAMS['D_trans'],
        n_iterations=n_iterations // 2,
        verbose=verbose,
        label="D_trans",
    )

    return TissueFitResult(
        D_long=D_long_opt,
        D_trans=D_trans_opt,
        cv_long_achieved=cv_long,
        cv_trans_achieved=cv_trans,
        all_D=torch.tensor([[D_long_opt, D_trans_opt]], dtype=dtype),
        all_cv=torch.tensor([[cv_long, cv_trans]], dtype=dtype),
    )


def _optimize_D(theta_ionic: torch.Tensor,
                config: TuningConfig,
                target_cv: float,
                D_init: float,
                D_bounds: Tuple[float, float],
                n_iterations: int = 15,
                verbose: bool = True,
                label: str = "D") -> Tuple[float, float]:
    """
    Single-objective BO for one diffusion coefficient.

    Returns
    -------
    (D_optimal, cv_achieved)
    """
    dtype = config.dtype
    bounds = torch.tensor([[D_bounds[0]], [D_bounds[1]]], dtype=dtype)

    # Initial evaluations: warm-start + 2 boundary probes
    D_values = [D_init,
                max(D_bounds[0], D_init * 0.5),
                min(D_bounds[1], D_init * 2.0)]

    train_X = torch.tensor([[d] for d in D_values], dtype=dtype)
    train_Y = torch.zeros(len(D_values), 1, dtype=dtype)

    for i, D in enumerate(D_values):
        result = run_cv_measurement(theta_ionic, D, config)
        if result.cv is not None and result.converged:
            cv_err = (result.cv - target_cv) ** 2
        else:
            cv_err = 10000.0
        train_Y[i, 0] = -cv_err  # Maximize (minimize error)

    # BO loop
    for iteration in range(n_iterations):
        model = SingleTaskGP(
            train_X, train_Y,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq = qExpectedImprovement(
            model=model,
            best_f=train_Y.max(),
        )

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=64,
        )

        new_D = candidate.squeeze().item()
        result = run_cv_measurement(theta_ionic, new_D, config)
        if result.cv is not None and result.converged:
            cv_err = (result.cv - target_cv) ** 2
        else:
            cv_err = 10000.0

        train_X = torch.cat([train_X, candidate.view(1, 1)])
        train_Y = torch.cat([train_Y, torch.tensor([[-cv_err]], dtype=dtype)])

    # Find best
    best_idx = train_Y.argmax()
    D_opt = train_X[best_idx, 0].item()

    # Evaluate best
    final_result = run_cv_measurement(theta_ionic, D_opt, config)
    cv_achieved = final_result.cv if final_result.cv is not None else 0.0

    if verbose:
        print(f"  {label}: D = {D_opt:.6f} -> CV = {cv_achieved:.1f} cm/s "
              f"(target {target_cv:.1f})")

    return D_opt, cv_achieved
