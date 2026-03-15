"""
Optimizer V1 — Pipeline Orchestrator

Runs the full 4-phase optimization pipeline:
  Phase I-II:  Cell fitter (single-cell BayesOpt)
  Phase III:   Tissue fitter (D optimization)
  Phase IV:    Joint refinement (GP emulator + NSGA-II)
  Phase V:     Validation
"""

import torch
import json
import os
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict

from .config import TuningConfig, TuningTargets, get_param_names, theta_to_dict
from .cell_fitter import fit_cell, CellFitResult
from .tissue_fitter import fit_tissue, TissueFitResult
from .joint_refiner import refine_joint, JointResult
from .validator import validate, ValidationResult


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    theta_ionic: Dict[str, float]
    D_long: float
    D_trans: float
    cell_fit: Optional[CellFitResult] = None
    tissue_fit: Optional[TissueFitResult] = None
    joint: Optional[JointResult] = None
    validation: Optional[ValidationResult] = None


def run_pipeline(config: TuningConfig = None,
                 targets: TuningTargets = None,
                 skip_joint: bool = False,
                 skip_validation: bool = False,
                 verbose: bool = True,
                 save_dir: Optional[str] = None) -> PipelineResult:
    """
    Run the full optimization pipeline.

    Parameters
    ----------
    config : TuningConfig (default: tier 1, 200 iterations)
    targets : TuningTargets (default: PHAS13 targets)
    skip_joint : skip Phase IV joint refinement
    skip_validation : skip Phase V validation
    verbose : print progress
    save_dir : directory to save intermediate results

    Returns
    -------
    PipelineResult
    """
    if config is None:
        config = TuningConfig()
    if targets is None:
        targets = TuningTargets()

    if verbose:
        print("=" * 60)
        print("Optimizer V1 — PHAS13 Tuning Pipeline")
        print(f"  Tier: {config.tier} ({len(get_param_names(config.tier))} params)")
        print(f"  Device: {config.device}")
        print(f"  Target APD90: {targets.apd_90} ms")
        print(f"  Target CV: {targets.cv_longitudinal} / {targets.cv_transverse} cm/s")
        print("=" * 60)

    # ======== Phase I-II: Cell Fitter ========
    if verbose:
        print("\n--- Phase I-II: Cell Fitter ---")

    cell_result = fit_cell(config, targets, verbose=verbose)

    # Select best from Pareto front (closest to APD target)
    best_idx = 0
    if cell_result.pareto_Y.shape[0] > 0:
        apd_errors = -cell_result.pareto_Y[:, 0]  # Negated errors
        best_idx = apd_errors.argmin().item()

    best_theta = cell_result.pareto_X[best_idx]
    theta_dict = theta_to_dict(best_theta, config.tier)

    if verbose:
        print(f"\n  Best cell params: {theta_dict}")

    # ======== Phase III: Tissue Fitter ========
    if verbose:
        print("\n--- Phase III: Tissue Fitter ---")

    tissue_result = fit_tissue(best_theta, config, targets, verbose=verbose)

    if verbose:
        print(f"\n  D_long = {tissue_result.D_long:.6f} "
              f"(CV = {tissue_result.cv_long_achieved:.1f} cm/s)")
        print(f"  D_trans = {tissue_result.D_trans:.6f} "
              f"(CV = {tissue_result.cv_trans_achieved:.1f} cm/s)")

    # ======== Phase IV: Joint Refinement ========
    joint_result = None
    final_theta = theta_dict
    final_D_long = tissue_result.D_long
    final_D_trans = tissue_result.D_trans

    if not skip_joint:
        if verbose:
            print("\n--- Phase IV: Joint Refinement ---")

        joint_result = refine_joint(
            cell_result.pareto_X,
            tissue_result.D_long,
            tissue_result.D_trans,
            config, targets,
            verbose=verbose,
        )

        final_theta = joint_result.theta_ionic
        final_D_long = joint_result.D_long
        final_D_trans = joint_result.D_trans

    # ======== Phase V: Validation ========
    validation_result = None
    if not skip_validation:
        if verbose:
            print("\n--- Phase V: Validation ---")

        validation_result = validate(
            final_theta, final_D_long, final_D_trans,
            config, targets, verbose=verbose,
        )

    # ======== Save Results ========
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _save_results(save_dir, final_theta, final_D_long, final_D_trans,
                      config, targets)

    result = PipelineResult(
        theta_ionic=final_theta,
        D_long=final_D_long,
        D_trans=final_D_trans,
        cell_fit=cell_result,
        tissue_fit=tissue_result,
        joint=joint_result,
        validation=validation_result,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline complete.")
        print(f"  Ionic params: {final_theta}")
        print(f"  D_long = {final_D_long:.6f}, D_trans = {final_D_trans:.6f}")
        if validation_result:
            print(f"  Validation: {validation_result.n_passed}/"
                  f"{validation_result.n_total} passed")
        print("=" * 60)

    return result


def _save_results(save_dir: str, theta: Dict, D_long: float, D_trans: float,
                  config: TuningConfig, targets: TuningTargets):
    """Save results to JSON."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'theta_ionic': theta,
        'D_long': D_long,
        'D_trans': D_trans,
        'config': {
            'ionic_model': config.ionic_model,
            'tier': config.tier,
            'n_iterations': config.n_iterations,
            'pacing_cl': config.pacing_cl,
        },
        'targets': {
            'apd_90': targets.apd_90,
            'cv_longitudinal': targets.cv_longitudinal,
            'cv_transverse': targets.cv_transverse,
            'dvdt_max': targets.dvdt_max,
        },
    }

    path = os.path.join(save_dir, 'tuning_result.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
