"""cardiac_core._lbm — the vendored LBM V1 solver (engine_consolidation).

Copied verbatim from LBM/Engine_V1/src (fully relative-import internally; it receives the ionic
model as an OBJECT, so there were zero cross-imports to rewrite). Private package (underscore) so it
does not shadow the public lbm() factory. The original src/__init__.py was docstring-only, so this
facade adds the LBMSimulation re-export.
"""

from .simulation import LBMSimulation

__all__ = ["LBMSimulation"]
