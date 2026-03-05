"""
Boundary Condition Protocol for Bidomain Simulations

Per-edge, per-variable BC specification. Stored in StructuredGrid at init time,
read by discretization and solver layers.

Ref: improvement.md L272-597
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional


class BCType(Enum):
    """Boundary condition types for bidomain variables."""
    NEUMANN = "neumann"          # Zero-flux: n·D·grad(u) = 0
    DIRICHLET = "dirichlet"      # Fixed value: u = value


class Edge(Enum):
    """Named edges of a rectangular domain."""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class EdgeBC:
    """BC specification for one variable on one edge."""
    bc_type: BCType = BCType.NEUMANN
    value: float = 0.0           # Dirichlet value (ignored for Neumann)


@dataclass
class BoundarySpec:
    """
    Complete boundary condition specification for a bidomain simulation.

    Encodes per-edge, per-variable BC types. Stored in the mesh at init time,
    read by discretization and solver layers. No BC logic leaks into strings.

    Attributes
    ----------
    Vm : Dict[Edge, EdgeBC]
        BC for transmembrane potential on each edge.
        Default: Neumann everywhere (membrane is always insulated).
    phi_e : Dict[Edge, EdgeBC]
        BC for extracellular potential on each edge.
        Default: Neumann everywhere (fully insulated tissue).
    """
    Vm: Dict[Edge, EdgeBC] = field(default_factory=lambda: {e: EdgeBC() for e in Edge})
    phi_e: Dict[Edge, EdgeBC] = field(default_factory=lambda: {e: EdgeBC() for e in Edge})

    # --- Derived properties (computed once, queried many times) ---

    @property
    def phi_e_has_null_space(self) -> bool:
        """True if phi_e has a constant null space (all edges Neumann)."""
        return all(bc.bc_type == BCType.NEUMANN for bc in self.phi_e.values())

    @property
    def phi_e_uniform_bc(self) -> Optional[BCType]:
        """If all edges have the same BC type, return it. Else None."""
        types = {bc.bc_type for bc in self.phi_e.values()}
        return types.pop() if len(types) == 1 else None

    @property
    def phi_e_spectral_eligible(self) -> bool:
        """True if spectral solver (Tier 1/2) can be used for phi_e.
        Requires all edges to have the same BC type."""
        return self.phi_e_uniform_bc is not None

    @property
    def spectral_transform(self) -> Optional[str]:
        """Which spectral transform to use, or None if mixed BCs."""
        bc = self.phi_e_uniform_bc
        if bc == BCType.NEUMANN:
            return 'dct'
        elif bc == BCType.DIRICHLET:
            return 'dst'
        return None

    def get_bc(self, variable: str, edge: Edge) -> EdgeBC:
        """Get BC for a specific variable and edge."""
        if variable == 'Vm':
            return self.Vm[edge]
        elif variable == 'phi_e':
            return self.phi_e[edge]
        raise ValueError(f"Unknown variable: {variable}")

    # --- Factory methods for common configurations ---

    @classmethod
    def insulated(cls) -> 'BoundarySpec':
        """Standard insulated tissue — Neumann everywhere on all variables."""
        return cls()

    @classmethod
    def bath_coupled(cls, bath_value: float = 0.0) -> 'BoundarySpec':
        """Bath-coupled on ALL edges — phi_e Dirichlet everywhere."""
        return cls(
            phi_e={edge: EdgeBC(BCType.DIRICHLET, bath_value) for edge in Edge},
        )

    @classmethod
    def bath_coupled_edges(cls, bath_edges: list, bath_value: float = 0.0) -> 'BoundarySpec':
        """Partial bath — specified edges are bath-coupled, rest insulated."""
        phi_e = {}
        for edge in Edge:
            if edge in bath_edges or edge.value in bath_edges:
                phi_e[edge] = EdgeBC(BCType.DIRICHLET, bath_value)
            else:
                phi_e[edge] = EdgeBC(BCType.NEUMANN)
        return cls(phi_e=phi_e)
