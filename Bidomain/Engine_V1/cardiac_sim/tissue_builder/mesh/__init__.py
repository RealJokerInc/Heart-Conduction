"""
Mesh Subpackage

Provides computational mesh types and boundary condition protocol.
"""

from .base import Mesh
from .structured import StructuredGrid
from .boundary import BoundarySpec, BCType, Edge, EdgeBC

__all__ = [
    'Mesh',
    'StructuredGrid',
    'BoundarySpec',
    'BCType',
    'Edge',
    'EdgeBC',
]
