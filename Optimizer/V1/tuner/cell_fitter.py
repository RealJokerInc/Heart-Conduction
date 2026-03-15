"""
Optimizer V1 — Cell Fitter (Multi-Objective BayesOpt)

Uses BoTorch qNEHVI to find Pareto-optimal ionic parameter scalings
that minimize APD error, dV/dt error, and (optionally) CL error.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from .config import (
    TuningConfig, TuningTargets,
    get_bounds_tensor, get_param_names, get_params_for_tier,
)
from .cell_runner import run_single_cell, run_spontaneous, CellResult


@dataclass
class CellFitResult:
    """Result from cell fitting."""
    pareto_X: torch.Tensor          # (n_pareto, n_params) scaling factors
    pareto_Y: torch.Tensor          # (n_pareto, n_objectives) objective values
    all_X: torch.Tensor             # (n_total, n_params) all evaluated points
    all_Y: torch.Tensor             # (n_total, n_objectives) all evaluations
    param_names: List[str]
    objective_names: List[str]


def _evaluate_single(theta: torch.Tensor, config: TuningConfig,
                     targets: TuningTargets) -> torch.Tensor:
    """
    Evaluate a single theta vector -> objective vector.

    Objectives (all to be MAXIMIZED for BoTorch, so we negate errors):
    - neg_apd_error: -|APD90_measured - APD90_target|
    - neg_dvdt_error: -|dvdt_measured - dvdt_target|
    - neg_cl_error: -|CL_measured - CL_target| (if target exists)

    Returns
    -------
    (n_obj,) tensor of objective values (higher is better).
    """
    result = run_single_cell(theta, config)

    objectives = []

    # APD90 error
    if result.apd90 is not None and result.converged:
        apd_err = abs(result.apd90 - targets.apd_90)
    else:
        apd_err = 1000.0  # Penalty for failed sim
    objectives.append(-apd_err)

    # dV/dt max error
    if result.dvdt_max is not None and result.converged:
        dvdt_err = abs(result.dvdt_max - targets.dvdt_max)
    else:
        dvdt_err = 500.0
    objectives.append(-dvdt_err)

    # CL error (optional)
    if targets.spontaneous_cl is not None:
        spont = run_spontaneous(theta, config, duration_ms=5000.0)
        if spont.cl is not None and spont.converged:
            cl_err = abs(spont.cl - targets.spontaneous_cl)
        else:
            cl_err = 2000.0
        objectives.append(-cl_err)

    return torch.tensor(objectives, dtype=config.dtype)


def _generate_initial_design(bounds: torch.Tensor, n_points: int,
                             dtype=torch.float64) -> torch.Tensor:
    """Generate Latin Hypercube initial design."""
    # Use Sobol quasi-random for better space coverage
    candidates = draw_sobol_samples(bounds=bounds, n=n_points, q=1).squeeze(1)
    return candidates


def fit_cell(config: TuningConfig, targets: TuningTargets,
             n_initial: Optional[int] = None,
             n_iterations: Optional[int] = None,
             verbose: bool = True) -> CellFitResult:
    """
    Run multi-objective BayesOpt for ionic parameter tuning.

    Parameters
    ----------
    config : TuningConfig
    targets : TuningTargets
    n_initial : initial design points (default: 2 * n_params)
    n_iterations : BO iterations (default: config.n_iterations)
    verbose : print progress

    Returns
    -------
    CellFitResult with Pareto front and all evaluations.
    """
    bounds = get_bounds_tensor(config.tier, config.dtype)
    param_names = get_param_names(config.tier)
    n_params = len(param_names)

    if n_initial is None:
        n_initial = config.n_initial if config.n_initial > 0 else 2 * n_params
    if n_iterations is None:
        n_iterations = config.n_iterations

    # Determine number of objectives
    n_obj = 2
    obj_names = ['neg_apd_error', 'neg_dvdt_error']
    if targets.spontaneous_cl is not None:
        n_obj = 3
        obj_names.append('neg_cl_error')

    # Reference point for hypervolume (worst acceptable values)
    ref_point = torch.tensor([-500.0] * n_obj, dtype=config.dtype)

    # Phase 1: Initial design
    if verbose:
        print(f"Cell Fitter: {n_params} params (tier {config.tier}), "
              f"{n_obj} objectives")
        print(f"  Initial design: {n_initial} points")

    X_init = _generate_initial_design(bounds, n_initial, config.dtype)
    Y_init = torch.zeros(n_initial, n_obj, dtype=config.dtype)

    for i in range(n_initial):
        Y_init[i] = _evaluate_single(X_init[i], config, targets)
        if verbose and (i + 1) % 5 == 0:
            print(f"  Initial {i + 1}/{n_initial}")

    train_X = X_init.clone()
    train_Y = Y_init.clone()

    # Phase 2: BayesOpt loop
    if verbose:
        print(f"  BO iterations: {n_iterations}")

    for iteration in range(n_iterations):
        # Fit GP model
        model = SingleTaskGP(
            train_X, train_Y,
            outcome_transform=Standardize(m=n_obj),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Acquisition function
        acq = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            prune_baseline=True,
        )

        # Optimize acquisition
        candidate, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        # Evaluate candidate
        new_X = candidate.squeeze(0)
        new_Y = _evaluate_single(new_X, config, targets)

        # Update training data
        train_X = torch.cat([train_X, new_X.unsqueeze(0)])
        train_Y = torch.cat([train_Y, new_Y.unsqueeze(0)])

        if verbose and (iteration + 1) % 10 == 0:
            # Report best APD error so far
            best_apd = train_Y[:, 0].max().item()
            print(f"  Iter {iteration + 1}/{n_iterations}: "
                  f"best APD error = {-best_apd:.1f} ms")

    # Extract Pareto front
    pareto_mask = is_non_dominated(train_Y)
    pareto_X = train_X[pareto_mask]
    pareto_Y = train_Y[pareto_mask]

    if verbose:
        print(f"  Done. Pareto front: {pareto_X.shape[0]} points")

    return CellFitResult(
        pareto_X=pareto_X,
        pareto_Y=pareto_Y,
        all_X=train_X,
        all_Y=train_Y,
        param_names=param_names,
        objective_names=obj_names,
    )
