"""
Optimizer V1 — Cell Fitter (Constrained Multi-Objective BayesOpt)

Uses BoTorch qLogNEHVI with hard constraints on dV/dt, V_peak, V_rest.
Candidates violating constraints receive heavy penalties.
Evaluates candidates in batches for performance.
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
from .cell_result import CellResult
# The V5.4 (`cardiac_sim`) AP functions in cell_runner are imported LAZILY inside
# _evaluate_batch so the default cardiac_core AP path never triggers the cardiac_sim
# import at module load (P-1 backend unification).


CONSTRAINT_PENALTY = -2000.0  # Heavy penalty for constraint violations


@dataclass
class CellFitResult:
    """Result from cell fitting."""
    pareto_X: torch.Tensor
    pareto_Y: torch.Tensor
    all_X: torch.Tensor
    all_Y: torch.Tensor
    param_names: List[str]
    objective_names: List[str]
    n_feasible: int = 0         # How many evals passed all constraints
    n_total: int = 0


def _check_constraints(res: CellResult, targets: TuningTargets) -> bool:
    """Return True if result passes all hard constraints."""
    if not res.converged:
        return False
    if res.dvdt_max is not None and res.dvdt_max > targets.dvdt_max_upper:
        return False
    if res.v_peak > targets.v_peak_max:
        return False
    if res.v_rest < targets.v_rest_range[0] or res.v_rest > targets.v_rest_range[1]:
        return False
    return True


def _evaluate_batch(theta_batch: torch.Tensor, config: TuningConfig,
                    targets: TuningTargets) -> torch.Tensor:
    """
    Evaluate M theta vectors -> (M, n_obj) objective matrix.

    Applies hard constraints: infeasible candidates get CONSTRAINT_PENALTY.
    Feasible candidates get negative absolute errors (higher = better).
    """
    M = theta_batch.shape[0]
    n_obj = 2
    if targets.spontaneous_cl is not None:
        n_obj = 3

    if config.ionic_backend == 'cardiac_core':
        from .cell_runner_cc import run_cell_batch_cc
        results = run_cell_batch_cc(theta_batch, config, targets)
    else:
        from .cell_runner import run_single_cell_batch, extract_biomarkers_batch
        t, V_all = run_single_cell_batch(theta_batch, config)
        results = extract_biomarkers_batch(t, V_all, config, targets)

    Y = torch.full((M, n_obj), CONSTRAINT_PENALTY, dtype=config.dtype)

    for i, res in enumerate(results):
        feasible = _check_constraints(res, targets)

        if not feasible:
            # All objectives get penalty
            continue

        # APD90 error (feasible)
        if res.apd90 is not None:
            Y[i, 0] = -abs(res.apd90 - targets.apd_90)
        else:
            Y[i, 0] = -1000.0

        # dV/dt error (feasible — already passed constraint)
        if res.dvdt_max is not None:
            Y[i, 1] = -abs(res.dvdt_max - targets.dvdt_max)
        else:
            Y[i, 1] = -500.0

    # CL objective (optional, sequential)
    if targets.spontaneous_cl is not None:
        for i in range(M):
            if Y[i, 0] <= CONSTRAINT_PENALTY + 1:
                continue  # Already penalized
            # Deprecated spontaneous-CL objective: MHAS13 is quiescent, so this only
            # has meaning on the V5.4 path (lazy import — off the default AP path).
            from .cell_runner import run_spontaneous
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
    Run constrained multi-objective BayesOpt with batched evaluation.
    """
    # Reproducibility (R3)
    torch.manual_seed(config.seed)

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

    if verbose:
        print(f"Cell Fitter: {n_params} params (tier {config.tier}), "
              f"{n_obj} objectives, device={config.device}")
        print(f"  Constraints: dVdt<{targets.dvdt_max_upper}V/s, "
              f"Vpeak<{targets.v_peak_max}mV, "
              f"Vrest∈[{targets.v_rest_range[0]},{targets.v_rest_range[1]}]mV")
        print(f"  Initial design: {n_initial} points (batched)")

    X_init = draw_sobol_samples(bounds=bounds, n=n_initial, q=1,
                                seed=config.seed).squeeze(1)

    from time import perf_counter
    t0 = perf_counter()
    Y_init = _evaluate_batch(X_init, config, targets)
    t_batch = perf_counter() - t0

    n_feasible_init = (Y_init[:, 0] > CONSTRAINT_PENALTY + 1).sum().item()
    if verbose:
        print(f"  Initial batch: {t_batch:.1f}s ({t_batch/n_initial:.1f}s/eval), "
              f"{n_feasible_init}/{n_initial} feasible")

    train_X = X_init.clone()
    train_Y = Y_init.clone()

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

        q_batch = min(4, max(1, n_iterations - iteration))
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q_batch,
            num_restarts=10,
            raw_samples=256,
        )

        new_X = candidate.squeeze(0) if q_batch == 1 else candidate
        if new_X.dim() == 1:
            new_X = new_X.unsqueeze(0)

        new_Y = _evaluate_batch(new_X, config, targets)

        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])

        if verbose and (iteration + 1) % max(1, n_iterations // 5) == 0:
            feasible_mask = train_Y[:, 0] > CONSTRAINT_PENALTY + 1
            n_feas = feasible_mask.sum().item()
            if n_feas > 0:
                best_apd = train_Y[feasible_mask, 0].max().item()
                best_dvdt = train_Y[feasible_mask, 1].max().item()
                print(f"  Iter {iteration + 1}/{n_iterations}: "
                      f"APD err={-best_apd:.1f}ms, dVdt err={-best_dvdt:.1f}V/s, "
                      f"{n_feas}/{train_X.shape[0]} feasible")
            else:
                print(f"  Iter {iteration + 1}/{n_iterations}: "
                      f"0/{train_X.shape[0]} feasible (relaxing search)")

    # Extract Pareto front from FEASIBLE solutions only
    feasible_mask = train_Y[:, 0] > CONSTRAINT_PENALTY + 1
    n_feasible = feasible_mask.sum().item()

    if n_feasible > 0:
        feasible_X = train_X[feasible_mask]
        feasible_Y = train_Y[feasible_mask]
        pareto_mask = is_non_dominated(feasible_Y)
        pareto_X = feasible_X[pareto_mask]
        pareto_Y = feasible_Y[pareto_mask]
    else:
        # Fallback: return best overall (least penalized)
        best_idx = train_Y[:, 0].argmax()
        pareto_X = train_X[best_idx:best_idx+1]
        pareto_Y = train_Y[best_idx:best_idx+1]

    if verbose:
        print(f"  Done. Pareto front: {pareto_X.shape[0]} points "
              f"({n_feasible}/{train_X.shape[0]} feasible)")

    return CellFitResult(
        pareto_X=pareto_X,
        pareto_Y=pareto_Y,
        all_X=train_X,
        all_Y=train_Y,
        param_names=param_names,
        objective_names=obj_names,
        n_feasible=n_feasible,
        n_total=train_X.shape[0],
    )
