"""cardiac_core.mesh — shared structured-grid geometry.

The single home for `StructuredGrid` and the `boundary` types, shared by every engine. Geometry is
structured-grid only (no unstructured/FEM meshes). `structured.py` carries the full feature set used
by both monodomain and bidomain: the monodomain essentials plus `boundary_spec` and the
`edge_masks`/`dirichlet_mask_phi_e`/`neumann_mask_phi_e` phi_e properties that only bidomain needs.
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
