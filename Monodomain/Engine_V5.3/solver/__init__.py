"""
Solvers for Monodomain Simulation

This module provides:
- PCG solver with Jacobi preconditioner
- Time stepping schemes (CN, BDF1, BDF2)
- Sparse matrix-vector operations
"""

from .linear import (
    pcg_solve,
    sparse_mv,
    extract_diagonal,
    SolverStats,
    apply_dirichlet_bc,
)

from .time_stepping import (
    TimeScheme,
    TimeStepperConfig,
    CrankNicolsonStepper,
    BDFStepper,
    create_time_stepper,
)

__all__ = [
    # Linear solver
    'pcg_solve',
    'sparse_mv',
    'extract_diagonal',
    'SolverStats',
    'apply_dirichlet_bc',
    # Time stepping
    'TimeScheme',
    'TimeStepperConfig',
    'CrankNicolsonStepper',
    'BDFStepper',
    'create_time_stepper',
]
