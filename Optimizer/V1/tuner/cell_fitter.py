"""
Optimizer V1 — Cell Fitter (Multi-Objective BayesOpt)

Uses BoTorch qLogNEHVI to find Pareto-optimal ionic parameter scalings.
Evaluates candidates in batches for GPU acceleration.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from .config import (
    TuningConfig, TuningTargets,
    get_bounds_tensor, get_param_names,
)
from .cell_runner import (
    run_single_cell_batch, extract_biomarkers_batch,
    run_spontaneous, CellResult,
)


@dataclass
class CellFitResult:
    """Result from cell fitting."""
    pareto_X: torch.Tensor
    pareto_Y: torch.Tensor
    all_X: torch.Tensor
    all_Y: torch.Tensor
    param_names: List[str]
    objective_names: List[str]


def _evaluate_batch(theta_batch: torch.Tensor, config: TuningConfig,
                    targets: TuningTargets) -> torch.Tensor:
    """
    Evaluate M theta vectors -> (M, n_obj) objective matrix.

    All M cells simulated simultaneously in one batched run.
    """
    M = theta_batch.shape[0]
    n_obj = 2
    if targets.spontaneous_cl is not None:
        n_obj = 3

    # Run all M cells in one batch
    t, V_all = run_single_cell_batch(theta_batch, config)
    results = extract_biomarkers_batch(t, V_all, config, targets)

    Y = torch.zeros(M, n_obj, dtype=config.dtype)

    for i, res in enumerate(results):
        # APD90 error
        if res.apd90 is not None and res.converged:
            Y[i, 0] = -abs(res.apd90 - targets.apd_90)
        else:
            Y[i, 0] = -1000.0

        # dV/dt error
        if res.dvdt_max is not None and res.converged:
            Y[i, 1] = -abs(res.dvdt_max - targets.dvdt_max)
        else:
            Y[i, 1] = -500.0

    # CL objective (requires separate spontaneous runs — not batched yet)
    if targets.spontaneous_cl is not None:
        for i in range(M):
            spont = run_spontaneous(theta_batch[i], config, duration_ms=5000.0)
            if spont.cl is not None and spont.converged:
                Y[i, 2] = -abs(spont.cl - targets.spontaneous_cl)
            else:
                Y[i, 2] = -2000.0

    return Y


def fit_cell(config: TuningConfig, targets: TuningTargets,
             n_initial: Optional[int] = None,
             n_iterations: Optional[int] = None,
             verbose: bool = True) -> CellFitResult:
    """
    Run multi-objective BayesOpt with batched evaluation.
    """
    bounds = get_bounds_tensor(config.tier, config.dtype)
    param_names = get_param_names(config.tier)
    n_params = len(param_names)

    if n_initial is None:
        n_initial = config.n_initial if config.n_initial > 0 else 2 * n_params
    if n_iterations is None:
        n_iterations = config.n_iterations

    n_obj = 2
    obj_names = ['neg_apd_error', 'neg_dvdt_error']
    if targets.spontaneous_cl is not None:
        n_obj = 3
        obj_names.append('neg_cl_error')

    ref_point = torch.tensor([-500.0] * n_obj, dtype=config.dtype)

    # Phase 1: Initial design — evaluate ALL initial points in one batch
    if verbose:
        print(f"Cell Fitter: {n_params} params (tier {config.tier}), "
              f"{n_obj} objectives, device={config.device}")
        print(f"  Initial design: {n_initial} points (batched)")

    X_init = draw_sobol_samples(bounds=bounds, n=n_initial, q=1).squeeze(1)

    from time import perf_counter
    t0 = perf_counter()
    Y_init = _evaluate_batch(X_init, config, targets)
    t_batch = perf_counter() - t0

    if verbose:
        print(f"  Initial batch: {t_batch:.1f}s ({t_batch/n_initial:.1f}s/eval amortized)")

    train_X = X_init.clone()
    train_Y = Y_init.clone()

    # Phase 2: BayesOpt loop
    if verbose:
        print(f"  BO iterations: {n_iterations}")

    for iteration in range(n_iterations):
        model = SingleTaskGP(
            train_X, train_Y,
            outcome_transform=Standardize(m=n_obj),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            prune_baseline=True,
        )

        # Request q=4 candidates per iteration for batch evaluation
        q_batch = min(4, max(1, n_iterations - iteration))
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q_batch,
            num_restarts=10,
            raw_samples=256,
        )

        # Evaluate batch
        new_X = candidate.squeeze(0) if q_batch == 1 else candidate
        if new_X.dim() == 1:
            new_X = new_X.unsqueeze(0)

        new_Y = _evaluate_batch(new_X, config, targets)

        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])

        if verbose and (iteration + 1) % max(1, n_iterations // 5) == 0:
            best_apd = train_Y[:, 0].max().item()
            print(f"  Iter {iteration + 1}/{n_iterations}: "
                  f"best APD error = {-best_apd:.1f} ms, "
                  f"{train_X.shape[0]} total evals")

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
