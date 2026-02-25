"""
FEM Infrastructure for Monodomain Simulation

This module provides:
- TriangularMesh: 2D triangular mesh generation and handling
- Matrix assembly: Mass and stiffness matrix construction
- P1 basis functions
"""

from .mesh import TriangularMesh
from .assembly import assemble_matrices, assemble_mass_matrix, assemble_stiffness_matrix

__all__ = [
    'TriangularMesh',
    'assemble_matrices',
    'assemble_mass_matrix',
    'assemble_stiffness_matrix',
]
