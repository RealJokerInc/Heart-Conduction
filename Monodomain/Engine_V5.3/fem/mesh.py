"""
Triangular Mesh for 2D FEM

Provides mesh generation, manipulation, and utility functions for
2D triangular finite element meshes.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import torch


@dataclass
class TriangularMesh:
    """
    2D triangular mesh for FEM simulations.

    Attributes
    ----------
    nodes : torch.Tensor
        Node coordinates, shape (n_nodes, 2)
    elements : torch.Tensor
        Element connectivity (node indices), shape (n_elements, 3)
    boundary_nodes : torch.Tensor
        Indices of nodes on the boundary
    n_nodes : int
        Number of nodes
    n_elements : int
        Number of elements
    device : torch.device
        Device where tensors are stored
    dtype : torch.dtype
        Data type for floating point tensors
    """

    nodes: torch.Tensor
    elements: torch.Tensor
    boundary_nodes: torch.Tensor
    n_nodes: int
    n_elements: int
    device: torch.device = None
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        if self.device is None:
            self.device = self.nodes.device

    @classmethod
    def create_rectangle(
        cls,
        Lx: float,
        Ly: float,
        nx: int,
        ny: int,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64
    ) -> 'TriangularMesh':
        """
        Create a structured triangular mesh on a rectangular domain.

        The mesh is created by dividing the rectangle into (nx-1) x (ny-1) squares,
        then splitting each square into 2 triangles with alternating diagonals
        for better isotropy.

        Parameters
        ----------
        Lx : float
            Domain length in x-direction (cm)
        Ly : float
            Domain length in y-direction (cm)
        nx : int
            Number of nodes in x-direction
        ny : int
            Number of nodes in y-direction
        device : str
            Device ('cuda' or 'cpu')
        dtype : torch.dtype
            Data type for coordinates

        Returns
        -------
        TriangularMesh
            Generated mesh
        """
        device = torch.device(device)

        # Create node grid
        x = torch.linspace(0, Lx, nx, device=device, dtype=dtype)
        y = torch.linspace(0, Ly, ny, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        nodes = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        # Create elements (2 triangles per quad cell)
        # Use alternating diagonal pattern for better isotropy
        elements = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                # Node indices for this quad
                n00 = i * ny + j          # bottom-left
                n10 = (i + 1) * ny + j    # bottom-right
                n01 = i * ny + (j + 1)    # top-left
                n11 = (i + 1) * ny + (j + 1)  # top-right

                # Alternate diagonal direction for isotropy
                if (i + j) % 2 == 0:
                    # Diagonal: n00 -> n11
                    elements.append([n00, n10, n11])
                    elements.append([n00, n11, n01])
                else:
                    # Diagonal: n10 -> n01
                    elements.append([n00, n10, n01])
                    elements.append([n10, n11, n01])

        elements = torch.tensor(elements, dtype=torch.long, device=device)

        # Find boundary nodes
        boundary_nodes = cls._find_boundary_nodes(nodes, Lx, Ly)

        return cls(
            nodes=nodes,
            elements=elements,
            boundary_nodes=boundary_nodes,
            n_nodes=nodes.shape[0],
            n_elements=elements.shape[0],
            device=device,
            dtype=dtype
        )

    @staticmethod
    def _find_boundary_nodes(
        nodes: torch.Tensor,
        Lx: float,
        Ly: float,
        tol: float = 1e-10
    ) -> torch.Tensor:
        """Find indices of nodes on the domain boundary."""
        x, y = nodes[:, 0], nodes[:, 1]
        on_left = x < tol
        on_right = x > Lx - tol
        on_bottom = y < tol
        on_top = y > Ly - tol
        boundary_mask = on_left | on_right | on_bottom | on_top
        return torch.where(boundary_mask)[0]

    def get_element_coordinates(self) -> torch.Tensor:
        """
        Get coordinates of all element vertices.

        Returns
        -------
        torch.Tensor
            Shape (n_elements, 3, 2) - coordinates for each vertex of each element
        """
        return self.nodes[self.elements]

    def compute_element_areas(self) -> torch.Tensor:
        """
        Compute area of each element.

        Returns
        -------
        torch.Tensor
            Shape (n_elements,) - area of each triangle
        """
        coords = self.get_element_coordinates()
        x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
        x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
        x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

        # Area = 0.5 * |det([x2-x1, x3-x1; y2-y1, y3-y1])|
        areas = 0.5 * torch.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        return areas

    def get_node_region(
        self,
        region_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Get indices of nodes within a region defined by a function.

        Parameters
        ----------
        region_func : callable
            Function (x, y) -> bool tensor indicating nodes in region

        Returns
        -------
        torch.Tensor
            Indices of nodes in the region
        """
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        mask = region_func(x, y)
        return torch.where(mask)[0]

    def get_dx(self) -> float:
        """Estimate mesh spacing (average edge length)."""
        coords = self.get_element_coordinates()

        # Compute edge lengths for first element as estimate
        e1 = torch.norm(coords[0, 1] - coords[0, 0])
        e2 = torch.norm(coords[0, 2] - coords[0, 1])
        e3 = torch.norm(coords[0, 0] - coords[0, 2])

        return ((e1 + e2 + e3) / 3).item()

    def to(self, device: torch.device) -> 'TriangularMesh':
        """Move mesh to specified device."""
        return TriangularMesh(
            nodes=self.nodes.to(device),
            elements=self.elements.to(device),
            boundary_nodes=self.boundary_nodes.to(device),
            n_nodes=self.n_nodes,
            n_elements=self.n_elements,
            device=device,
            dtype=self.dtype
        )

    def __repr__(self) -> str:
        return (f"TriangularMesh(n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
                f"device={self.device})")
