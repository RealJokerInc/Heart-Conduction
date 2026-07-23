"""cardiac_core._lbm — the lattice-Boltzmann monodomain solver.

Self-contained: fully relative-import internally, and it receives the ionic model as an object
rather than importing one. Private package (leading underscore) so it does not shadow the public
``lbm()`` factory; this facade re-exports :class:`LBMSimulation`.
"""

from .simulation import LBMSimulation

__all__ = ["LBMSimulation"]
