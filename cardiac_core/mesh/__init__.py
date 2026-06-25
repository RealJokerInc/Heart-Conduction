"""cardiac_core.mesh — shared structured-grid geometry (engine_consolidation).

The single home for `StructuredGrid` (+ `boundary` types) and the FEM `TriangularMesh`, vendored
once from the engines. `structured.py` is the bidomain superset (it adds `boundary_spec` + the
`edge_masks`/`dirichlet_mask_phi_e`/`neumann_mask_phi_e` phi_e props to mono's; shared methods are
byte-identical). Union of the mono + bidomain exports — neither engine's `__init__` worked verbatim.
"""

from .base import Mesh
from .structured import StructuredGrid
from .triangular import TriangularMesh
from .boundary import BoundarySpec, BCType, Edge, EdgeBC

__all__ = [
    'Mesh',
    'StructuredGrid',
    'TriangularMesh',
    'BoundarySpec',
    'BCType',
    'Edge',
    'EdgeBC',
]
