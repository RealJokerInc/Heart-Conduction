"""
Geometry helpers — create masks, distance fields, and regions.

All functions return numpy arrays (Nx, Ny) suitable for use in
CardiacMeshData or as masks to simulation methods.

    scar = circle_mask(Nx, Ny, dx, center=(1.0, 0.5), radius=0.2)
    border = boundary_distance(mask) < 0.1
"""

import numpy as np


def _coordinate_grids(Nx: int, Ny: int, dx: float, dy: float = None):
    """Return (Nx, Ny) coordinate arrays for x and y."""
    if dy is None:
        dy = dx
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    return np.meshgrid(x, y, indexing='ij')


def circle_mask(
    Nx: int, Ny: int, dx: float,
    center: tuple[float, float],
    radius: float,
    dy: float = None,
) -> np.ndarray:
    """Create a circular mask.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    dx : float
        Grid spacing (cm). dy defaults to dx.
    center : (float, float)
        Center coordinates (x_cm, y_cm).
    radius : float
        Radius (cm).

    Returns
    -------
    np.ndarray
        (Nx, Ny) bool.
    """
    X, Y = _coordinate_grids(Nx, Ny, dx, dy)
    return ((X - center[0])**2 + (Y - center[1])**2) <= radius**2


def rectangle_mask(
    Nx: int, Ny: int, dx: float,
    x0: float, y0: float,
    x1: float, y1: float,
    dy: float = None,
) -> np.ndarray:
    """Create a rectangular mask.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    dx : float
        Grid spacing (cm).
    x0, y0 : float
        Lower-left corner (cm).
    x1, y1 : float
        Upper-right corner (cm).

    Returns
    -------
    np.ndarray
        (Nx, Ny) bool.
    """
    X, Y = _coordinate_grids(Nx, Ny, dx, dy)
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def annulus_mask(
    Nx: int, Ny: int, dx: float,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    dy: float = None,
) -> np.ndarray:
    """Create an annular (ring) mask.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    dx : float
        Grid spacing (cm).
    center : (float, float)
        Center coordinates (cm).
    inner_radius, outer_radius : float
        Inner and outer radii (cm).

    Returns
    -------
    np.ndarray
        (Nx, Ny) bool.
    """
    X, Y = _coordinate_grids(Nx, Ny, dx, dy)
    r2 = (X - center[0])**2 + (Y - center[1])**2
    return (r2 >= inner_radius**2) & (r2 <= outer_radius**2)


def left_edge_mask(Nx: int, Ny: int, dx: float, width: float) -> np.ndarray:
    """Mask for left edge of domain (x < width)."""
    x = np.arange(Nx) * dx
    mask = np.zeros((Nx, Ny), dtype=bool)
    mask[x < width, :] = True
    return mask


def right_edge_mask(Nx: int, Ny: int, dx: float, width: float) -> np.ndarray:
    """Mask for right edge of domain (x > Lx - width)."""
    x = np.arange(Nx) * dx
    Lx = x[-1]
    mask = np.zeros((Nx, Ny), dtype=bool)
    mask[x > Lx - width, :] = True
    return mask


def bottom_edge_mask(Nx: int, Ny: int, dx: float, width: float, dy: float = None) -> np.ndarray:
    """Mask for the bottom edge of the domain (low y: ``y < width``). ``dy`` defaults to ``dx``."""
    dy = dx if dy is None else dy
    y = np.arange(Ny) * dy
    mask = np.zeros((Nx, Ny), dtype=bool)
    mask[:, y < width] = True
    return mask


def top_edge_mask(Nx: int, Ny: int, dx: float, width: float, dy: float = None) -> np.ndarray:
    """Mask for the top edge of the domain (high y: ``y > Ly - width``). ``dy`` defaults to ``dx``."""
    dy = dx if dy is None else dy
    y = np.arange(Ny) * dy
    Ly = y[-1]
    mask = np.zeros((Nx, Ny), dtype=bool)
    mask[:, y > Ly - width] = True
    return mask


def point_distance(
    Nx: int, Ny: int, dx: float,
    center: tuple[float, float],
    dy: float = None,
) -> np.ndarray:
    """Euclidean distance from a point at every grid node.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    dx : float
        Grid spacing (cm).
    center : (float, float)
        Point coordinates (x_cm, y_cm). Matches the ``center=`` convention of
        ``circle_mask``/``annulus_mask``.

    Returns
    -------
    np.ndarray
        (Nx, Ny) float64 distance in cm.
    """
    X, Y = _coordinate_grids(Nx, Ny, dx, dy)
    return np.sqrt((X - center[0])**2 + (Y - center[1])**2)


def boundary_distance(mask: np.ndarray, dx: float) -> np.ndarray:
    """Distance from tissue boundary at every active node.

    Uses chamfer (city-block) approximation — fast, O(Nx*Ny).

    Parameters
    ----------
    mask : np.ndarray
        (Nx, Ny) bool tissue mask.
    dx : float
        Grid spacing (cm).

    Returns
    -------
    np.ndarray
        (Nx, Ny) float64 distance in cm. 0 at boundary, positive inside,
        NaN outside tissue.
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask).astype(np.float64) * dx
    dist[~mask] = np.nan
    return dist


def fiber_field_uniform(Nx: int, Ny: int, angle: float = 0.0) -> np.ndarray:
    """Uniform fiber angle field.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    angle : float
        Fiber angle in radians (0 = x-axis).

    Returns
    -------
    np.ndarray
        (Nx, Ny) float64 angle field.
    """
    return np.full((Nx, Ny), angle, dtype=np.float64)


def fiber_field_transmural(
    Nx: int, Ny: int,
    angle_endo: float = -1.047,
    angle_epi: float = 1.047,
) -> np.ndarray:
    """Transmural fiber rotation (linear from endo to epi angle).

    Assumes y=0 is endocardium, y=Ny-1 is epicardium.
    Default: -60 to +60 degrees (-pi/3 to pi/3).

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    angle_endo : float
        Endo fiber angle (radians).
    angle_epi : float
        Epi fiber angle (radians).

    Returns
    -------
    np.ndarray
        (Nx, Ny) float64 angle field.
    """
    t = np.linspace(0, 1, Ny)
    angles_1d = angle_endo + t * (angle_epi - angle_endo)
    return np.broadcast_to(angles_1d[np.newaxis, :], (Nx, Ny)).copy()
