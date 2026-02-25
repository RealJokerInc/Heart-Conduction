"""
Utility helpers to build meshes for the 2-D monodomain solver.

This module focuses on constructing a rectangular tissue mesh whose fiber
direction points purely left-to-right (theta = 0 everywhere) and defining
stimulus helpers along the left-most vertical boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


from version3 import build_D_components
@dataclass
class TissueMesh:
    """
    Container holding the geometric / diffusion information for a rectangular
    sheet of tissue.
    """

    nx: int
    ny: int
    Lx: float
    Ly: float
    dx: float
    dy: float
    Dxx: np.ndarray
    Dxy: np.ndarray
    Dyy: np.ndarray
    epsilon_tissue: np.ndarray
    theta: np.ndarray
    stim_mask: np.ndarray

    def empty_state(self, V_rest: float = 0.0, w_rest: float = 0.0):
        """Allocate voltage / recovery arrays initialized at the resting state."""
        V = np.full((self.ny, self.nx), V_rest, dtype=float)
        w = np.full_like(V, w_rest)
        return V, w

    def extent(self):
        """Return matplotlib-friendly plot extent = (xmin, xmax, ymin, ymax)."""
        return (0.0, self.Lx, 0.0, self.Ly)

    def periodic_left_edge_stimulus(
        self,
        amplitude: float = 20.0,
        period_ms: float = 6.0,
        pulse_ms: float = 1.0,
        start_time: float = 0.0,
    ):
        """
        Build a callable I_stim(t) that applies a periodic pulse on the left edge.
        """
        mask = self.stim_mask.copy()
        ny, nx = mask.shape

        def _stimulus(t: float):
            field = np.zeros((ny, nx), dtype=float)
            if t >= start_time:
                phase = (t - start_time) % period_ms
                if phase < pulse_ms:
                    field[mask] = amplitude
            return field

        return _stimulus

    def pulse_train_left_edge_stimulus(
        self,
        amplitude: float = 20.0,
        pulse_ms: float = 1.0,
        interval_ms: float = 8.0,
        n_pulses: int | None = 5,
        start_time: float = 0.0,
    ):
        """
        Build a finite pulse train stimulus on the left border. If n_pulses is
        None the train becomes periodic for the entire simulation.
        """
        mask = self.stim_mask.copy()
        ny, nx = mask.shape

        def _stimulus(t: float):
            field = np.zeros((ny, nx), dtype=float)
            if t < start_time:
                return field

            elapsed = t - start_time
            pulse_index = int(elapsed // interval_ms)
            if n_pulses is not None and pulse_index >= n_pulses:
                return field

            phase = elapsed - pulse_index * interval_ms
            if phase < pulse_ms:
                field[mask] = amplitude
            return field

        return _stimulus


def create_left_to_right_mesh(
    nx: int = 160,
    ny: int = 120,
    Lx: float = 20.0,
    Ly: float = 15.0,
    D_parallel: float = 1.0,
    D_perp: float = 0.2,
    epsilon_scale: float = 1.0,
) -> TissueMesh:
    """
    Construct a uniform rectangular grid whose fibers point from left to right.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    theta = np.zeros((ny, nx), dtype=float)  # 0 rad => horizontal fibers
    epsilon_tissue = np.full((ny, nx), epsilon_scale, dtype=float)

    Dxx, Dxy, Dyy = build_D_components(theta, D_parallel, D_perp, epsilon_tissue)

    stim_mask = np.zeros((ny, nx), dtype=bool)
    stim_mask[:, 0] = True  # left-most vertical border

    return TissueMesh(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dx=dx,
        dy=dy,
        Dxx=Dxx,
        Dxy=Dxy,
        Dyy=Dyy,
        epsilon_tissue=epsilon_tissue,
        theta=theta,
        stim_mask=stim_mask,
    )


def create_circular_infarct_mesh(
    nx: int = 160,
    ny: int = 160,
    Lx: float = 20.0,
    Ly: float = 20.0,
    infarct_radius: float = 3.0,
    D_parallel: float = 1.0,
    D_perp: float = 0.2,
    eps_healthy: float = 1.0,
    eps_infarct: float = 0.0,
    flow_strength: float = 1.0,
    boundary_boost: float = 3.0,
    boost_width: float | None = None,
) -> TissueMesh:
    """
    Build a mesh containing a circular infarct with zero diffusivity inside.

    Fibers wrap tangentially around the infarct (theta field follows the polar
    angle), so diffusion aligns with concentric circles.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    cx = Lx / 2.0
    cy = Ly / 2.0
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    epsilon_tissue = np.full((ny, nx), eps_healthy, dtype=float)
    epsilon_tissue[r <= infarct_radius] = eps_infarct

    dx_c = X - cx
    dy_c = Y - cy
    r_safe = np.maximum(r, 1e-6)
    cos_phi = dx_c / r_safe
    sin_phi = dy_c / r_safe

    tangential_x = -sin_phi
    tangential_y = cos_phi

    if boost_width is None:
        boost_width = 0.5 * infarct_radius

    distance_from_boundary = np.abs(r - infarct_radius)
    ring_mask = distance_from_boundary <= boost_width

    r_safe = np.maximum(r, infarct_radius + 1e-6)
    ratio = (infarct_radius / r_safe) ** 2
    flow_x = flow_strength * (1.0 - ratio * np.cos(2 * np.arctan2(dy_c, dx_c)))
    flow_y = -flow_strength * ratio * np.sin(2 * np.arctan2(dy_c, dx_c))

    flow_x[ring_mask] = flow_strength * tangential_x[ring_mask]
    flow_y[ring_mask] = flow_strength * tangential_y[ring_mask]

    norm = np.hypot(flow_x, flow_y)
    norm[norm == 0.0] = 1.0
    flow_x /= norm
    flow_y /= norm

    theta = np.arctan2(flow_y, flow_x)

    boost = 1.0 + boundary_boost * np.exp(-(distance_from_boundary / boost_width) ** 2)
    D_parallel_map = D_parallel * boost

    Dxx, Dxy, Dyy = build_D_components(theta, D_parallel_map, D_perp, epsilon_tissue)

    stim_mask = np.zeros((ny, nx), dtype=bool)
    stim_mask[:, 0] = True

    return TissueMesh(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dx=dx,
        dy=dy,
        Dxx=Dxx,
        Dxy=Dxy,
        Dyy=Dyy,
        epsilon_tissue=epsilon_tissue,
        theta=theta,
        stim_mask=stim_mask,
    )


def create_spiral_mesh(
    nx: int = 160,
    ny: int = 160,
    Lx: float = 20.0,
    Ly: float = 20.0,
    spiral_pitch: float = 0.1,
    D_parallel: float = 1.0,
    D_perp: float = 0.2,
    epsilon_scale: float = 1.0,
) -> TissueMesh:
    """
    Build a mesh whose fibers follow a logarithmic spiral emanating from the origin.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    theta = np.arctan2(Y, X) + spiral_pitch * np.sqrt(X**2 + Y**2)

    epsilon_tissue = np.full((ny, nx), epsilon_scale, dtype=float)

    Dxx, Dxy, Dyy = build_D_components(theta, D_parallel, D_perp, epsilon_tissue)

    stim_mask = np.zeros((ny, nx), dtype=bool)
    stim_mask[:, 0] = True  # left boundary for vertical line stimulation

    return TissueMesh(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dx=dx,
        dy=dy,
        Dxx=Dxx,
        Dxy=Dxy,
        Dyy=Dyy,
        epsilon_tissue=epsilon_tissue,
        theta=theta,
        stim_mask=stim_mask,
    )

def create_flow_around_infarct_mesh(
    nx: int = 160,
    ny: int = 160,
    Lx: float = 20.0,
    Ly: float = 20.0,
    infarct_radius: float = 3.0,
    D_parallel: float = 1.0,
    D_perp: float = 0.2,
    eps_healthy: float = 1.0,
    eps_infarct: float = 0.0,
    flow_strength: float = 1.0,
    boundary_boost: float = 3.0,
    boost_width: float | None = None,
) -> TissueMesh:
    """
    Build a mesh with a circular infarct at the centre and a fiber field that
    follows an incompressible potential flow around that infarct.

    Far away: fibers ~ +x direction (bulk flow from the left).
    Near the infarct: fibers become tangential to the circular boundary, creating
    a conduction "highway" around the infarct.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Centre of the infarct (and cylinder in the flow analogy)
    cx = Lx / 2.0
    cy = Ly / 2.0

    dx_c = X - cx
    dy_c = Y - cy
    r = np.sqrt(dx_c * dx_c + dy_c * dy_c)

    # Tissue mask: healthy outside, infarct inside
    epsilon_tissue = np.full((ny, nx), eps_healthy, dtype=float)
    epsilon_tissue[r <= infarct_radius] = eps_infarct

    # ----------------------------------------------------------------------
    # Potential flow around a cylinder with uniform inflow from the left.
    # ----------------------------------------------------------------------
    r_safe = np.maximum(r, infarct_radius * 1.001)
    phi = np.arctan2(dy_c, dx_c)
    U_inf = flow_strength

    # Polar components of velocity
    u_r = U_inf * (1.0 - (infarct_radius ** 2) / (r_safe ** 2)) * np.cos(phi)
    u_phi = -U_inf * (1.0 + (infarct_radius ** 2) / (r_safe ** 2)) * np.sin(phi)

    # Convert to Cartesian components
    u_x = u_r * np.cos(phi) - u_phi * np.sin(phi)
    u_y = u_r * np.sin(phi) + u_phi * np.cos(phi)

    # Inside infarct: arbitrary direction (diffusion will be ~0 anyway)
    inside = r <= infarct_radius
    u_x[inside] = 1.0
    u_y[inside] = 0.0

    # Normalise to get pure directions
    speed = np.hypot(u_x, u_y)
    speed[speed == 0.0] = 1.0
    u_x /= speed
    u_y /= speed

    # Fiber angle field = direction of the "flow"
    theta = np.arctan2(u_y, u_x)

    # ----------------------------------------------------------------------
    # Boost D_parallel in a ring around the infarct boundary.
    # ----------------------------------------------------------------------
    if boost_width is None:
        boost_width = 0.5 * infarct_radius

    distance_from_boundary = np.abs(r - infarct_radius)
    boost = 1.0 + boundary_boost * np.exp(
        - (distance_from_boundary / boost_width) ** 2
    )

    D_parallel_map = D_parallel * boost

    # Build diffusion tensor components aligned with the fiber field
    Dxx, Dxy, Dyy = build_D_components(
        theta,
        D_parallel_map,
        D_perp,
        epsilon_tissue,
    )

    # Stimulus: left boundary, as usual
    stim_mask = np.zeros((ny, nx), dtype=bool)
    stim_mask[:, 0] = True

    return TissueMesh(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dx=dx,
        dy=dy,
        Dxx=Dxx,
        Dxy=Dxy,
        Dyy=Dyy,
        epsilon_tissue=epsilon_tissue,
        theta=theta,
        stim_mask=stim_mask,
    )


__all__ = [
    "TissueMesh",
    "create_left_to_right_mesh",
    "create_circular_infarct_mesh",
    "create_spiral_mesh",
    "create_flow_around_infarct_mesh",
]

