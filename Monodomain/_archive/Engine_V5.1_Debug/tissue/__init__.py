"""
Tissue Simulation Module for Engine V5.1

GPU-accelerated monodomain tissue simulation using PyTorch.

Components:
- DiffusionOperator: FVM-based diffusion operator with anisotropy support
- MonodomainSimulation: Full tissue simulation with ionic + diffusion coupling

Physical Model:
    χ·Cm·∂V/∂t = -χ·Iion(V, u) + ∇·(D·∇V) + Istim

Where:
- χ = 1400 cm⁻¹ (surface-to-volume ratio)
- Cm = 1.0 µF/cm² (membrane capacitance)
- D = diffusion tensor (anisotropic, fiber-aligned)
- Iion = ionic current from O'Hara-Rudy model

Example:
    from tissue import MonodomainSimulation
    from ionic import CellType

    # Create 100x100 tissue (1cm x 1cm at 100µm resolution)
    sim = MonodomainSimulation(
        ny=100, nx=100,
        dx=0.01, dy=0.01,
        cv_long=0.06,  # 0.6 m/s
        cv_trans=0.02,  # 0.2 m/s
        celltype=CellType.ENDO,
        device='cuda'
    )

    # Add stimulus
    sim.add_stimulus(
        region=(slice(None), slice(0, 5)),
        start_time=0.0,
        duration=1.0
    )

    # Run simulation
    t, V = sim.run(t_end=100.0, dt=0.02)
"""

from .diffusion import (
    DiffusionOperator,
    compute_D_from_cv,
    compute_cv_from_D,
    get_diffusion_params,
    compute_D_min,
    validate_D_for_mesh,
    CHI,
    CM,
    CV_LONGITUDINAL_DEFAULT,
    CV_TRANSVERSE_DEFAULT,
    K_BASE,
    APD_REF,
    APD_ALPHA,
    SAFETY_MARGIN_DEFAULT,
)

from .simulation import (
    MonodomainSimulation,
    create_spiral_wave_ic,
)

from .mesh import (
    MeshBuilder,
    MeshConfig,
    CableMesh,
    CableConfig,
    create_default_mesh,
    create_cv_tuning_cable,
    # Tuned diffusion coefficients
    D_L_DEFAULT,
    D_T_DEFAULT,
    D_TUNED_CV06,
    D_TUNED_CV03,
)

__all__ = [
    # Diffusion
    'DiffusionOperator',
    'compute_D_from_cv',
    'compute_cv_from_D',
    'get_diffusion_params',
    'compute_D_min',
    'validate_D_for_mesh',
    'CHI',
    'CM',
    'CV_LONGITUDINAL_DEFAULT',
    'CV_TRANSVERSE_DEFAULT',
    'K_BASE',
    'APD_REF',
    'APD_ALPHA',
    'SAFETY_MARGIN_DEFAULT',
    # Simulation
    'MonodomainSimulation',
    'create_spiral_wave_ic',
    # Mesh
    'MeshBuilder',
    'MeshConfig',
    'CableMesh',
    'CableConfig',
    'create_default_mesh',
    'create_cv_tuning_cable',
    # Tuned diffusion coefficients
    'D_L_DEFAULT',
    'D_T_DEFAULT',
    'D_TUNED_CV06',
    'D_TUNED_CV03',
]
