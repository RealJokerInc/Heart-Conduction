"""
Discretization Scheme Package

Spatial discretization methods that provide operators for time stepping:
- SpatialDiscretization: Abstract base class
- FEMDiscretization: Finite Element Method (unstructured triangular mesh)
- FDMDiscretization: Finite Difference Method (structured grid, 9-point stencil)
- FVMDiscretization: Finite Volume Method (structured grid, cell-centered)
"""

from .base import SpatialDiscretization, MassType, DiffusionOperators, sparse_mv
from .fem import FEMDiscretization
from .fdm import FDMDiscretization
from .fvm import FVMDiscretization

__all__ = [
    'SpatialDiscretization',
    'MassType',
    'DiffusionOperators',
    'sparse_mv',
    'FEMDiscretization',
    'FDMDiscretization',
    'FVMDiscretization',
]
