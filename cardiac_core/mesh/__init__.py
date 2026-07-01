"""cardiac_core.mesh — shared structured-grid geometry (engine_consolidation).

The single home for `StructuredGrid` (+ `boundary` types), vendored
once from the engines. (FEM `TriangularMesh` was removed 2026-06-30 — structured-grid only.) `structured.py` is the bidomain superset (it adds `boundary_spec` + the
`edge_masks`/`dirichlet_mask_phi_e`/`neumann_mask_phi_e` phi_e props to mono's; shared methods are
byte-identical). Union of the mono + bidomain exports — neither engine's `__init__` worked verbatim.
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
