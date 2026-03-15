"""
Optimizer V1 — Joint Refiner

Builds a GP emulator from Phase III Pareto front × Phase IV D perturbations,
then runs NSGA-II on the emulator to co-optimize ionic + tissue parameters.
Active learning validates top candidates on real simulator.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from .config import (
    TuningConfig, TuningTargets,
    get_bounds_tensor, get_param_names, TISSUE_PARAMS,
)
from .cell_runner import run_single_cell, CellResult
from .tissue_runner import run_cv_measurement, CVResult


@dataclass
class JointResult:
    """Result from joint refinement."""
    theta_ionic: Dict[str, float]   # Optimal ionic scaling factors
    D_long: float                   # Optimal D_long (cm^2/ms)
    D_trans: float                  # Optimal D_trans (cm^2/ms)
    pareto_X: np.ndarray            # NSGA-II Pareto front inputs
    pareto_F: np.ndarray            # NSGA-II Pareto front objectives
    validation_results: List[Dict]  # Real-sim validation of top candidates


class EmulatorProblem(ElementwiseProblem):
    """Pymoo problem wrapping GP emulator for NSGA-II."""

    def __init__(self, gp_models: List[SingleTaskGP],
                 bounds_lower: np.ndarray,
                 bounds_upper: np.ndarray,
                 targets: TuningTargets,
                 n_ionic: int):
        n_var = len(bounds_lower)
        n_obj = len(gp_models)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=bounds_lower,
            xu=bounds_upper,
        )
        self.gp_models = gp_models
        self.targets = targets
        self.n_ionic = n_ionic

    def _evaluate(self, x, out, *args, **kwargs):
        X_tensor = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
        objectives = []
        for gp in self.gp_models:
            with torch.no_grad():
                posterior = gp.posterior(X_tensor)
                mean = posterior.mean.item()
            objectives.append(mean)
        out["F"] = np.array(objectives)


def _build_training_data(pareto_X: torch.Tensor,
                         D_long: float, D_trans: float,
                         config: TuningConfig,
                         targets: TuningTargets,
                         n_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build training data from Pareto front × D perturbations.

    Returns (X, Y) where X = [ionic_scaling..., D_long, D_trans]
    and Y = [apd_err, dvdt_err, cv_long_err, cv_trans_err].
    """
    n_pareto = pareto_X.shape[0]
    n_ionic = pareto_X.shape[1]
    dtype = config.dtype

    # Sample from Pareto front with D perturbations
    D_long_perturbs = np.linspace(
        max(TISSUE_PARAMS['D_long'][0], D_long * 0.5),
        min(TISSUE_PARAMS['D_long'][1], D_long * 2.0),
        max(3, n_samples // n_pareto)
    )
    D_trans_perturbs = np.linspace(
        max(TISSUE_PARAMS['D_trans'][0], D_trans * 0.5),
        min(TISSUE_PARAMS['D_trans'][1], D_trans * 2.0),
        max(3, n_samples // n_pareto)
    )

    X_list = []
    Y_list = []

    for i in range(min(n_pareto, n_samples)):
        theta_i = pareto_X[i % n_pareto]

        for dl in D_long_perturbs[:3]:  # Limit combinations
            for dt_val in D_trans_perturbs[:3]:
                x = torch.cat([theta_i, torch.tensor([dl, dt_val], dtype=dtype)])
                X_list.append(x)

                # Evaluate: cell metrics
                cell_result = run_single_cell(theta_i, config)
                apd_err = abs(cell_result.apd90 - targets.apd_90) if cell_result.apd90 else 500.0
                dvdt_err = abs(cell_result.dvdt_max - targets.dvdt_max) if cell_result.dvdt_max else 200.0

                # Evaluate: CV
                cv_result = run_cv_measurement(theta_i, dl, config)
                cv_long_err = abs(cv_result.cv - targets.cv_longitudinal) if cv_result.cv else 50.0

                cv_trans_result = run_cv_measurement(theta_i, dt_val, config)
                cv_trans_err = abs(cv_trans_result.cv - targets.cv_transverse) if cv_trans_result.cv else 50.0

                Y_list.append(torch.tensor([apd_err, dvdt_err, cv_long_err, cv_trans_err],
                                           dtype=dtype))

                if len(X_list) >= n_samples:
                    break
            if len(X_list) >= n_samples:
                break
        if len(X_list) >= n_samples:
            break

    return torch.stack(X_list), torch.stack(Y_list)


def _build_emulator(X: torch.Tensor, Y: torch.Tensor) -> List[SingleTaskGP]:
    """Build independent GP for each output dimension."""
    gps = []
    for j in range(Y.shape[1]):
        gp = SingleTaskGP(
            X, Y[:, j:j+1],
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gps.append(gp)
    return gps


def refine_joint(pareto_X: torch.Tensor,
                 D_long: float, D_trans: float,
                 config: TuningConfig,
                 targets: TuningTargets,
                 n_training: int = 50,
                 n_nsga2_gen: int = 100,
                 n_validate: int = 5,
                 verbose: bool = True) -> JointResult:
    """
    Joint refinement of ionic + tissue parameters.

    Parameters
    ----------
    pareto_X : Pareto front from cell fitter
    D_long : D_long from tissue fitter
    D_trans : D_trans from tissue fitter
    config : TuningConfig
    targets : TuningTargets
    n_training : number of training sims for emulator
    n_nsga2_gen : NSGA-II generations
    n_validate : number of top candidates to validate on real sim

    Returns
    -------
    JointResult
    """
    n_ionic = pareto_X.shape[1]
    param_names = get_param_names(config.tier)

    if verbose:
        print(f"Joint Refiner: building emulator from {n_training} training sims")

    # Step 1: Build training data
    X_train, Y_train = _build_training_data(
        pareto_X, D_long, D_trans, config, targets, n_training
    )

    if verbose:
        print(f"  Training data: {X_train.shape[0]} points, "
              f"{Y_train.shape[1]} outputs")

    # Step 2: Build GP emulator
    gps = _build_emulator(X_train, Y_train)

    if verbose:
        print(f"  GP emulator built ({len(gps)} GPs)")

    # Step 3: NSGA-II optimization on emulator
    ionic_bounds = get_bounds_tensor(config.tier)
    bounds_lower = np.concatenate([
        ionic_bounds[0].numpy(),
        [TISSUE_PARAMS['D_long'][0], TISSUE_PARAMS['D_trans'][0]]
    ])
    bounds_upper = np.concatenate([
        ionic_bounds[1].numpy(),
        [TISSUE_PARAMS['D_long'][1], TISSUE_PARAMS['D_trans'][1]]
    ])

    problem = EmulatorProblem(
        gp_models=gps,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        targets=targets,
        n_ionic=n_ionic,
    )

    algorithm = NSGA2(pop_size=100)
    termination = get_termination("n_gen", n_nsga2_gen)

    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=False,
    )

    if verbose:
        print(f"  NSGA-II: {res.F.shape[0]} Pareto points")

    # Step 4: Validate top candidates on real simulator
    if res.F is not None and len(res.F) > 0:
        # Sort by sum of objectives (all-errors)
        total_err = res.F.sum(axis=1)
        top_indices = total_err.argsort()[:n_validate]

        validation_results = []
        for idx in top_indices:
            x = res.X[idx]
            theta = torch.tensor(x[:n_ionic], dtype=config.dtype)
            dl = x[n_ionic]
            dt_val = x[n_ionic + 1]

            cell_res = run_single_cell(theta, config)
            cv_res = run_cv_measurement(theta, dl, config)

            vr = {
                'theta': {name: x[i] for i, name in enumerate(param_names)},
                'D_long': dl,
                'D_trans': dt_val,
                'apd90': cell_res.apd90,
                'dvdt_max': cell_res.dvdt_max,
                'cv': cv_res.cv,
                'converged': cell_res.converged and cv_res.converged,
            }
            validation_results.append(vr)

            if verbose:
                apd_str = f"{vr['apd90']:.0f}" if vr['apd90'] else "N/A"
                cv_str = f"{vr['cv']:.1f}" if vr['cv'] else "N/A"
                print(f"  Candidate {idx}: APD={apd_str} ms, CV={cv_str} cm/s")

        # Pick best validated candidate
        valid = [v for v in validation_results if v['converged']]
        if valid:
            # Prefer closest to APD target
            best = min(valid,
                       key=lambda v: abs((v['apd90'] or 500) - targets.apd_90))
        else:
            best = validation_results[0]

        return JointResult(
            theta_ionic=best['theta'],
            D_long=best['D_long'],
            D_trans=best['D_trans'],
            pareto_X=res.X,
            pareto_F=res.F,
            validation_results=validation_results,
        )

    # Fallback: return Phase III/IV results
    best_pareto_idx = 0
    return JointResult(
        theta_ionic={name: pareto_X[best_pareto_idx, i].item()
                     for i, name in enumerate(param_names)},
        D_long=D_long,
        D_trans=D_trans,
        pareto_X=res.X if res.X is not None else np.array([]),
        pareto_F=res.F if res.F is not None else np.array([]),
        validation_results=[],
    )
