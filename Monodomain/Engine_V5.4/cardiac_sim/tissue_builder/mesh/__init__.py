"""
Mesh Subpackage

Provides computational mesh types:
- Mesh: Abstract base class
- TriangularMesh: Unstructured triangles (FEM)
- StructuredGrid: Cartesian grid (FDM/FVM/LBM) [Phase 2]
"""

from .base import Mesh
from .triangular import TriangularMesh
from .structured import StructuredGrid

__all__ = [
    'Mesh',
    'TriangularMesh',
    'StructuredGrid',
]
