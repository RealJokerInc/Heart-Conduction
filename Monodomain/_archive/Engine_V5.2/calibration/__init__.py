"""
Calibration Pipeline for Diffusion Parameters.

This module provides optimization-based calibration using 1D cable simulations
and 2D tissue simulations to find diffusion coefficients (D_L, D_T) that match
target electrophysiological parameters.

Main Components:
    - Cable1D: 1D monodomain solver for CV/APD measurement
    - SingleCellERP: Single-cell ERP measurement via S1-S2 protocol
    - Tissue2D: 2D tissue ERP measurement (central stim, edge probes)
    - DiffusionCalibrator: Differential Evolution optimizer

2D Tissue ERP Measurement:
    The optimizer uses 2D tissue simulation for ERP:
    - Square mesh sized proportional to wavelength (CV × ERP) with margin
    - Central stimulus (S1, S2)
    - Probes at x-terminus (right edge) and y-terminus (top edge)
    - Single ERP = minimum S1-S2 interval where BOTH probes activate
    - Both D_x (longitudinal) and D_y (transverse) affect the result

Quick Start:
    >>> from calibration import calibrate_diffusion
    >>> result = calibrate_diffusion(
    ...     cv_longitudinal=0.06,    # 60 cm/s
    ...     anisotropy_ratio=3.0,    # D_L / D_T
    ...     erp_tissue_target=320.0, # Target tissue ERP (ms)
    ...     dx=0.01,                 # Mesh spacing (cm)
    ...     dt=0.02                  # Time step (ms)
    ... )
    >>> print(f"D_L = {result.D_longitudinal}")
    >>> print(f"D_T = {result.D_transverse}")
    >>> print(f"ERP (2D tissue) = {result.erp_tissue}")

The calibration jointly optimizes both D_L and D_T because:
    1. CV scales as √D (CV ∝ √D)
    2. Tissue ERP depends on source-sink coupling in both directions
    3. Simple D_T = D_L/α scaling fails to satisfy ERP targets

Tiered Optimization:
    Tier 1: Fixed dt - try with default time step (faster)
    Tier 2: Variable dt - if Tier 1 fails, allow dt to vary for convergence

    dt is regularized toward default (w_dt_reg = 0.01) so D accuracy is prioritized.

Loss Function:
    L = w_cv_L × (CV_L_meas - CV_L_target)² / CV_L²
      + w_cv_T × (CV_T_meas - CV_T_target)² / CV_T²
      + w_apd × max(0, APD_min - APD)² / APD_min²
      + w_erp_tissue × (ERP_tissue - ERP_target)² / ERP_target²

Default Weights:
    - w_cv_L = 1.0, w_cv_T = 1.0
    - w_apd = 0.5 (penalty only)
    - w_erp_tissue = 5.0 (primary objective)
"""

from .cable_1d import (
    Cable1D,
    CableConfig,
    MeasurementResult,
    measure_cv_apd
)

from .erp_measurement import (
    SingleCellERP,
    TissueERP,
    S1S2Config,
    ERPResult,
    measure_erp_single_cell,
    measure_erp_tissue
)

from .tissue_erp_2d import (
    Tissue2D,
    Tissue2DConfig,
    ERPResult2D,
    measure_tissue_erp_2d
)

from .optimizer import (
    DiffusionCalibrator,
    CalibrationTargets,
    CalibrationWeights,
    CalibrationConfig,
    CalibrationResult,
    calibrate_diffusion
)

__all__ = [
    # Cable solver
    'Cable1D',
    'CableConfig',
    'MeasurementResult',
    'measure_cv_apd',

    # ERP measurement (single-cell & 1D tissue)
    'SingleCellERP',
    'TissueERP',
    'S1S2Config',
    'ERPResult',
    'measure_erp_single_cell',
    'measure_erp_tissue',

    # 2D tissue ERP
    'Tissue2D',
    'Tissue2DConfig',
    'ERPResult2D',
    'measure_tissue_erp_2d',

    # Optimizer
    'DiffusionCalibrator',
    'CalibrationTargets',
    'CalibrationWeights',
    'CalibrationConfig',
    'CalibrationResult',
    'calibrate_diffusion',
]
