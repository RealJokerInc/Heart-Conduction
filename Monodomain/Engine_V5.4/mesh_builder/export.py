"""
MeshBuilder export functionality.

Exports a MeshBuilderSession to a .npz file containing all geometry
and material data needed by the simulation engine loader.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from .session import MeshBuilderSession


def export_mesh(session: MeshBuilderSession, output_path: str) -> str:
    """
    Export a configured MeshBuilderSession to a .npz file.

    The .npz contains arrays in grid convention (Nx, Ny) matching
    StructuredGrid's indexing='ij' layout:

        mask        : (Nx, Ny) bool    — True = active tissue
        dx, dy      : float            — grid spacing (cm)
        Lx, Ly      : float            — domain dimensions (cm)
        D_xx, D_yy, D_xy : (Nx, Ny) float64 — conductivity tensor per node
        label_map   : (Nx, Ny) int32   — group index per node (-1 = background)
        group_labels     : str[]        — label per tissue group
        group_cell_types : str[]        — cell type per tissue group

    Parameters
    ----------
    session : MeshBuilderSession
        A fully configured session (all tissue groups configured, dimensions set).
    output_path : str
        Path for the output .npz file.

    Returns
    -------
    str
        The resolved output path.
    """
    if session.image_array is None:
        raise ValueError("No image loaded in session.")
    if not session.all_groups_configured:
        unconfigured = session.unconfigured_groups
        labels = [g.label or f"color={g.color}" for g in unconfigured]
        raise ValueError(f"Unconfigured groups: {labels}")

    img = session.image_array  # (H, W, C) where C = 3 or 4
    h, w = img.shape[:2]
    n_channels = img.shape[2] if img.ndim == 3 else 1

    # Target mesh resolution from session dimensions
    nx, ny = session.get_mesh_resolution()
    tissue_w, tissue_h = session.tissue_dimensions
    dx = session.dx
    dy = dx  # Square pixels

    # Build a color lookup: color_tuple → (group_index, CellGroup)
    # Only tissue groups get indices 0..n-1; background gets -1
    tissue_groups = sorted(
        [g for g in session.color_groups.values() if not g.is_background],
        key=lambda g: g.pixel_count,
        reverse=True,
    )
    background_colors = set()
    for g in session.color_groups.values():
        if g.is_background:
            background_colors.add(g.color)

    color_to_index = {}
    for idx, group in enumerate(tissue_groups):
        color_to_index[group.color] = idx

    group_labels = np.array([g.label or "" for g in tissue_groups])
    group_cell_types = np.array([g.cell_type or "" for g in tissue_groups])

    # Resample image to mesh resolution (nx, ny)
    # image is (H, W, C), mesh is (ny, nx) where ny = rows, nx = cols
    # Use nearest-neighbor to preserve color identity
    if h != ny or w != nx:
        row_indices = np.round(np.linspace(0, h - 1, ny)).astype(int)
        col_indices = np.round(np.linspace(0, w - 1, nx)).astype(int)
        resampled = img[np.ix_(row_indices, col_indices)]
    else:
        resampled = img

    # Build arrays in image convention (ny, nx) = (rows, cols)
    mask_img = np.zeros((ny, nx), dtype=bool)
    label_img = np.full((ny, nx), -1, dtype=np.int32)
    Dxx_img = np.zeros((ny, nx), dtype=np.float64)
    Dyy_img = np.zeros((ny, nx), dtype=np.float64)
    Dxy_img = np.zeros((ny, nx), dtype=np.float64)

    for row in range(ny):
        for col in range(nx):
            pixel = resampled[row, col]
            color = tuple(int(c) for c in pixel[:n_channels])

            if color in background_colors:
                continue

            if color in color_to_index:
                idx = color_to_index[color]
                group = tissue_groups[idx]
                mask_img[row, col] = True
                label_img[row, col] = idx
                if group.conductivity_tensor is not None:
                    Dxx_img[row, col] = group.conductivity_tensor[0, 0]
                    Dyy_img[row, col] = group.conductivity_tensor[1, 1]
                    Dxy_img[row, col] = group.conductivity_tensor[0, 1]

    # Transpose to grid convention (Nx, Ny) for StructuredGrid indexing='ij'
    mask = mask_img.T
    label_map = label_img.T
    D_xx = Dxx_img.T
    D_yy = Dyy_img.T
    D_xy = Dxy_img.T

    # Compute domain dimensions
    Lx = dx * (nx - 1)
    Ly = dy * (ny - 1)

    # Save — all 2D arrays are (Nx, Ny) grid convention
    output_path = str(Path(output_path).resolve())
    np.savez(
        output_path,
        mask=mask,
        dx=np.float64(dx),
        dy=np.float64(dy),
        Lx=np.float64(Lx),
        Ly=np.float64(Ly),
        D_xx=D_xx,
        D_yy=D_yy,
        D_xy=D_xy,
        label_map=label_map,
        group_labels=group_labels,
        group_cell_types=group_cell_types,
    )

    return output_path
