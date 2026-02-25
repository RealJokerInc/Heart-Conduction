"""
Mesh loader — reads mesh.npz files exported by mesh_builder.

Converts the .npz contents into a StructuredGrid and per-node
conductivity arrays ready for FDM/FVM discretization.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
import torch

from .structured import StructuredGrid


@dataclass
class MeshData:
    """
    Container for loaded mesh data.

    Attributes
    ----------
    grid : StructuredGrid
        The spatial grid with domain mask.
    D_xx : torch.Tensor
        Per-active-node conductivity D_xx, shape (n_dof,).
    D_yy : torch.Tensor
        Per-active-node conductivity D_yy, shape (n_dof,).
    D_xy : torch.Tensor
        Per-active-node conductivity D_xy, shape (n_dof,).
    D_xx_grid : torch.Tensor
        Per-grid-node conductivity D_xx, shape (Nx, Ny). For FDM/FVM.
    D_yy_grid : torch.Tensor
        Per-grid-node conductivity D_yy, shape (Nx, Ny). For FDM/FVM.
    D_xy_grid : torch.Tensor
        Per-grid-node conductivity D_xy, shape (Nx, Ny). For FDM/FVM.
    group_labels : List[str]
        Label for each tissue group.
    group_cell_types : List[str]
        Cell type for each tissue group.
    """
    grid: StructuredGrid
    D_xx: torch.Tensor
    D_yy: torch.Tensor
    D_xy: torch.Tensor
    D_xx_grid: torch.Tensor
    D_yy_grid: torch.Tensor
    D_xy_grid: torch.Tensor
    group_labels: List[str]
    group_cell_types: List[str]


def load_mesh(
    path: str,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float64,
) -> MeshData:
    """
    Load a mesh.npz file and return a MeshData object.

    The .npz contains arrays in grid convention (Nx, Ny) matching
    StructuredGrid's indexing='ij' layout. No transpose needed.

    Parameters
    ----------
    path : str
        Path to the .npz file.
    device : str
        Torch device ('cpu', 'cuda', 'mps').
    dtype : torch.dtype
        Float precision for tensors.

    Returns
    -------
    MeshData
        Grid + per-node conductivity + metadata.
    """
    data = np.load(path, allow_pickle=True)

    # Scalars
    dx = float(data['dx'])
    dy = float(data['dy'])

    # Arrays already in grid convention (Nx, Ny)
    mask = data['mask'].astype(bool)
    D_xx_np = data['D_xx'].astype(np.float64)
    D_yy_np = data['D_yy'].astype(np.float64)
    D_xy_np = data['D_xy'].astype(np.float64)

    # Create StructuredGrid from mask
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    grid = StructuredGrid.from_mask(mask_tensor, dx, dy, device=device, dtype=dtype)

    # Grid-convention tensors (Nx, Ny) — for FDM/FVM D_field parameter
    D_xx_grid = torch.tensor(D_xx_np, device=device, dtype=dtype)
    D_yy_grid = torch.tensor(D_yy_np, device=device, dtype=dtype)
    D_xy_grid = torch.tensor(D_xy_np, device=device, dtype=dtype)

    # Flatten conductivity to active-node ordering (same as grid.grid_to_flat)
    D_xx_flat = torch.tensor(D_xx_np[mask], device=device, dtype=dtype)
    D_yy_flat = torch.tensor(D_yy_np[mask], device=device, dtype=dtype)
    D_xy_flat = torch.tensor(D_xy_np[mask], device=device, dtype=dtype)

    # Metadata
    group_labels = list(data['group_labels'].astype(str))
    group_cell_types = list(data['group_cell_types'].astype(str))

    return MeshData(
        grid=grid,
        D_xx=D_xx_flat,
        D_yy=D_yy_flat,
        D_xy=D_xy_flat,
        D_xx_grid=D_xx_grid,
        D_yy_grid=D_yy_grid,
        D_xy_grid=D_xy_grid,
        group_labels=group_labels,
        group_cell_types=group_cell_types,
    )
