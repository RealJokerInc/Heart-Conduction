"""
Tissue-Level Simulation Components

Contains diffusion operators and 2D tissue simulation.
"""

from tissue.diffusion import (
    DiffusionOperator,
    compute_diffusion_tensor,
    compute_diffusion_fvm,
    compute_diffusion_fvm_uniform,
    compute_diffusion_isotropic,
    # CV-based parameter computation
    compute_D_from_cv,
    compute_cv_from_D,
    get_diffusion_params,
    # Physical constants
    CHI,
    CM,
    CV_LONGITUDINAL_DEFAULT,
    CV_TRANSVERSE_DEFAULT,
    CV_EMPIRICAL_CONSTANT,
    D_L_REF,
    D_T_REF,
)

from tissue.simulation import (
    MonodomainSimulation,
    create_spiral_wave_ic,
    estimate_cv_from_params,
)

__all__ = [
    # Diffusion
    'DiffusionOperator',
    'compute_diffusion_tensor',
    'compute_diffusion_fvm',
    'compute_diffusion_fvm_uniform',
    'compute_diffusion_isotropic',
    # CV-based parameters
    'compute_D_from_cv',
    'compute_cv_from_D',
    'get_diffusion_params',
    # Simulation
    'MonodomainSimulation',
    'create_spiral_wave_ic',
    'estimate_cv_from_params',
    # Constants
    'CHI',
    'CM',
    'CV_LONGITUDINAL_DEFAULT',
    'CV_TRANSVERSE_DEFAULT',
    'CV_EMPIRICAL_CONSTANT',
    'D_L_REF',
    'D_T_REF',
]
