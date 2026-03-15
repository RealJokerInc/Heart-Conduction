"""
Finite Volume Method (FVM) Diffusion Operator for Monodomain Equation

Implements anisotropic diffusion with support for spatially varying fiber orientation.

The monodomain equation:
    chi * (Cm * dV/dt + Iion) = div(D * grad(V))

Where D is the conductivity tensor:
    D = R(theta) * [[D_L, 0], [0, D_T]] * R(theta)^T

FVM computes fluxes through cell faces, ensuring conservation.

CV-based diffusion scaling:
    From cable theory: CV = 2 * sqrt(D / (chi * Cm * tau_foot))
    So: D = (CV/2)^2 * chi * Cm * tau_foot

    Human ventricular CV targets:
    - Longitudinal: 0.5-0.7 m/s (0.05-0.07 cm/ms)
    - Transverse: 0.17-0.25 m/s (0.017-0.025 cm/ms)
    - Anisotropy ratio: ~3:1
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional


# Physical constants
CHI = 1400.0      # Surface-to-volume ratio (cm^-1)
CM = 1.0          # Membrane capacitance (uF/cm^2)

# Empirical calibration from ORd model simulations
# Measured: D_L = 0.001498 cm^2/ms at dx = 0.01 cm gives CV ≈ 0.0586 cm/ms
# This gives: k = CV / sqrt(D) = 0.0586 / sqrt(0.001498) = 1.514
CV_EMPIRICAL_CONSTANT = 1.514  # cm^0.5 / ms^0.5

# Human ventricular CV targets (cm/ms)
CV_LONGITUDINAL_DEFAULT = 0.06   # 0.6 m/s
CV_TRANSVERSE_DEFAULT = 0.02    # 0.2 m/s

# Reference diffusion coefficients at dx = 0.01 cm (100 um)
D_L_REF = 0.00157  # For CV = 0.06 cm/ms (0.6 m/s)
D_T_REF = 0.00017  # For CV = 0.02 cm/ms (0.2 m/s)


def compute_D_from_cv(
    cv: float,
    dx: float,
    dx_ref: float = 0.01
) -> float:
    """
    Compute diffusion coefficient to achieve target CV at given mesh size.

    Uses empirical calibration from ORd model simulations rather than
    pure cable theory, which gives more accurate results.

    The relationship is: CV = k * sqrt(D) where k is empirically determined
    from simulations.

    Solving for D: D = (CV / k)^2

    A mesh-dependent correction is applied to compensate for numerical
    discretization errors at coarser meshes.

    Parameters
    ----------
    cv : float
        Target conduction velocity (cm/ms).
        Note: 1 m/s = 0.1 cm/ms
    dx : float
        Mesh spacing (cm)
    dx_ref : float
        Reference mesh size for correction (cm)

    Returns
    -------
    D : float
        Diffusion coefficient (cm^2/ms)
    """
    # Base D from empirical relationship: CV = k * sqrt(D)
    D_base = (cv / CV_EMPIRICAL_CONSTANT)**2

    # Mesh correction factor (empirically tuned)
    # At coarser meshes, numerical dispersion INCREASES CV, so we REDUCE D
    # At finer meshes, we're closer to the continuous limit
    ratio = dx / dx_ref
    if ratio > 1.0:
        # Coarser mesh: reduce D to compensate for numerical dispersion
        # Calibrated for ORd model with FVM diffusion
        correction = 1.0 / (1.0 + 0.04 * (ratio - 1.0))
    elif ratio < 1.0:
        # Finer mesh: slight increase (less numerical effect)
        correction = 1.0 + 0.02 * (1.0 - ratio)
    else:
        correction = 1.0

    return D_base * correction


def compute_cv_from_D(D: float) -> float:
    """
    Estimate CV from diffusion coefficient using empirical relationship.

    Uses the empirically calibrated formula: CV = k * sqrt(D)

    Parameters
    ----------
    D : float
        Diffusion coefficient (cm^2/ms)

    Returns
    -------
    cv : float
        Estimated conduction velocity (cm/ms)
    """
    return CV_EMPIRICAL_CONSTANT * np.sqrt(D)


def get_diffusion_params(
    dx: float,
    cv_long: float = CV_LONGITUDINAL_DEFAULT,
    cv_trans: float = CV_TRANSVERSE_DEFAULT
) -> Tuple[float, float]:
    """
    Get diffusion coefficients for target CVs at given mesh size.

    This is the main entry point for CV-based parameter selection.

    Parameters
    ----------
    dx : float
        Mesh spacing (cm)
    cv_long : float
        Target longitudinal CV (cm/ms), default 0.06 (0.6 m/s)
    cv_trans : float
        Target transverse CV (cm/ms), default 0.02 (0.2 m/s)

    Returns
    -------
    D_L, D_T : float
        Longitudinal and transverse diffusion coefficients (cm^2/ms)

    Examples
    --------
    >>> D_L, D_T = get_diffusion_params(dx=0.01)  # 100 um mesh
    >>> D_L, D_T = get_diffusion_params(dx=0.02, cv_long=0.05, cv_trans=0.02)
    """
    D_L = compute_D_from_cv(cv_long, dx)
    D_T = compute_D_from_cv(cv_trans, dx)

    return D_L, D_T


@njit(cache=True)
def compute_diffusion_tensor(theta, D_L, D_T):
    """
    Compute full 2x2 diffusion tensor from fiber angle.

    Parameters
    ----------
    theta : float
        Fiber angle in radians (0 = fibers along x-axis)
    D_L : float
        Longitudinal diffusion coefficient (along fiber)
    D_T : float
        Transverse diffusion coefficient (across fiber)

    Returns
    -------
    D_xx, D_yy, D_xy : float
        Components of symmetric diffusion tensor
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos2 = cos_t * cos_t
    sin2 = sin_t * sin_t

    D_xx = D_L * cos2 + D_T * sin2
    D_yy = D_L * sin2 + D_T * cos2
    D_xy = (D_L - D_T) * cos_t * sin_t

    return D_xx, D_yy, D_xy


@njit(cache=True)
def fvm_flux_east(V, D_xx, D_xy, i, j, dx, dy, ny, nx):
    """
    Compute flux through the east face of cell (i, j).

    Flux_east = -(D_xx * dV/dx + D_xy * dV/dy) at east face

    Uses central differences for gradients at face centers.
    """
    # dV/dx at east face (between i,j and i,j+1)
    dVdx = (V[i, j+1] - V[i, j]) / dx

    # dV/dy at east face (average of north-south gradients at adjacent cells)
    # Handle boundaries
    if i == 0:
        # Forward difference at bottom boundary
        dVdy_left = (V[i+1, j] - V[i, j]) / dy
        dVdy_right = (V[i+1, j+1] - V[i, j+1]) / dy
    elif i == ny - 1:
        # Backward difference at top boundary
        dVdy_left = (V[i, j] - V[i-1, j]) / dy
        dVdy_right = (V[i, j+1] - V[i-1, j+1]) / dy
    else:
        # Central difference
        dVdy_left = (V[i+1, j] - V[i-1, j]) / (2 * dy)
        dVdy_right = (V[i+1, j+1] - V[i-1, j+1]) / (2 * dy)

    dVdy = 0.5 * (dVdy_left + dVdy_right)

    # Average tensor components at face
    D_xx_face = 0.5 * (D_xx[i, j] + D_xx[i, j+1])
    D_xy_face = 0.5 * (D_xy[i, j] + D_xy[i, j+1])

    # Flux = -D * grad(V) dot n_east, n_east = [1, 0]
    flux = D_xx_face * dVdx + D_xy_face * dVdy

    return flux


@njit(cache=True)
def fvm_flux_north(V, D_yy, D_xy, i, j, dx, dy, ny, nx):
    """
    Compute flux through the north face of cell (i, j).

    Flux_north = -(D_xy * dV/dx + D_yy * dV/dy) at north face
    """
    # dV/dy at north face (between i,j and i+1,j)
    dVdy = (V[i+1, j] - V[i, j]) / dy

    # dV/dx at north face (average of east-west gradients)
    if j == 0:
        dVdx_bottom = (V[i, j+1] - V[i, j]) / dx
        dVdx_top = (V[i+1, j+1] - V[i+1, j]) / dx
    elif j == nx - 1:
        dVdx_bottom = (V[i, j] - V[i, j-1]) / dx
        dVdx_top = (V[i+1, j] - V[i+1, j-1]) / dx
    else:
        dVdx_bottom = (V[i, j+1] - V[i, j-1]) / (2 * dx)
        dVdx_top = (V[i+1, j+1] - V[i+1, j-1]) / (2 * dx)

    dVdx = 0.5 * (dVdx_bottom + dVdx_top)

    # Average tensor at face
    D_yy_face = 0.5 * (D_yy[i, j] + D_yy[i+1, j])
    D_xy_face = 0.5 * (D_xy[i, j] + D_xy[i+1, j])

    # Flux = -D * grad(V) dot n_north, n_north = [0, 1]
    flux = D_xy_face * dVdx + D_yy_face * dVdy

    return flux


@njit(parallel=True, cache=True)
def compute_diffusion_fvm(V, D_xx, D_yy, D_xy, dx, dy):
    """
    Compute diffusion term using Finite Volume Method.

    div(D * grad(V)) at each cell center using flux balance.

    Parameters
    ----------
    V : ndarray (ny, nx)
        Membrane potential field
    D_xx, D_yy, D_xy : ndarray (ny, nx)
        Diffusion tensor components at each cell
    dx, dy : float
        Grid spacing

    Returns
    -------
    diff : ndarray (ny, nx)
        Diffusion term (div(D*grad(V))) at each cell
    """
    ny, nx = V.shape
    diff = np.zeros_like(V)

    for i in prange(ny):
        for j in range(nx):
            # Flux through east face
            if j < nx - 1:
                flux_e = fvm_flux_east(V, D_xx, D_xy, i, j, dx, dy, ny, nx)
            else:
                flux_e = 0.0  # No-flux BC at east boundary

            # Flux through west face
            if j > 0:
                flux_w = fvm_flux_east(V, D_xx, D_xy, i, j-1, dx, dy, ny, nx)
            else:
                flux_w = 0.0  # No-flux BC at west boundary

            # Flux through north face
            if i < ny - 1:
                flux_n = fvm_flux_north(V, D_yy, D_xy, i, j, dx, dy, ny, nx)
            else:
                flux_n = 0.0  # No-flux BC at north boundary

            # Flux through south face
            if i > 0:
                flux_s = fvm_flux_north(V, D_yy, D_xy, i-1, j, dx, dy, ny, nx)
            else:
                flux_s = 0.0  # No-flux BC at south boundary

            # Net flux into cell (conservation form)
            # div(D*grad(V)) = (flux_e - flux_w)/dx + (flux_n - flux_s)/dy
            diff[i, j] = (flux_e - flux_w) / dx + (flux_n - flux_s) / dy

    return diff


@njit(parallel=True, cache=True)
def compute_diffusion_fvm_uniform(V, D_L, D_T, theta, dx, dy):
    """
    Optimized FVM for uniform (constant) fiber angle.

    When theta is constant everywhere, we can precompute the tensor.

    Parameters
    ----------
    V : ndarray (ny, nx)
        Membrane potential field
    D_L : float
        Longitudinal diffusion coefficient
    D_T : float
        Transverse diffusion coefficient
    theta : float
        Uniform fiber angle in radians
    dx, dy : float
        Grid spacing

    Returns
    -------
    diff : ndarray (ny, nx)
        Diffusion term
    """
    ny, nx = V.shape
    diff = np.zeros_like(V)

    # Precompute tensor (constant everywhere)
    D_xx, D_yy, D_xy = compute_diffusion_tensor(theta, D_L, D_T)

    for i in prange(ny):
        for j in range(nx):
            # East flux
            if j < nx - 1:
                dVdx_e = (V[i, j+1] - V[i, j]) / dx

                if i == 0:
                    dVdy_e = 0.5 * ((V[i+1, j] - V[i, j]) / dy +
                                   (V[i+1, j+1] - V[i, j+1]) / dy)
                elif i == ny - 1:
                    dVdy_e = 0.5 * ((V[i, j] - V[i-1, j]) / dy +
                                   (V[i, j+1] - V[i-1, j+1]) / dy)
                else:
                    dVdy_e = 0.25 * ((V[i+1, j] - V[i-1, j]) / dy +
                                    (V[i+1, j+1] - V[i-1, j+1]) / dy)

                flux_e = D_xx * dVdx_e + D_xy * dVdy_e
            else:
                flux_e = 0.0

            # West flux
            if j > 0:
                dVdx_w = (V[i, j] - V[i, j-1]) / dx

                if i == 0:
                    dVdy_w = 0.5 * ((V[i+1, j-1] - V[i, j-1]) / dy +
                                   (V[i+1, j] - V[i, j]) / dy)
                elif i == ny - 1:
                    dVdy_w = 0.5 * ((V[i, j-1] - V[i-1, j-1]) / dy +
                                   (V[i, j] - V[i-1, j]) / dy)
                else:
                    dVdy_w = 0.25 * ((V[i+1, j-1] - V[i-1, j-1]) / dy +
                                    (V[i+1, j] - V[i-1, j]) / dy)

                flux_w = D_xx * dVdx_w + D_xy * dVdy_w
            else:
                flux_w = 0.0

            # North flux
            if i < ny - 1:
                dVdy_n = (V[i+1, j] - V[i, j]) / dy

                if j == 0:
                    dVdx_n = 0.5 * ((V[i, j+1] - V[i, j]) / dx +
                                   (V[i+1, j+1] - V[i+1, j]) / dx)
                elif j == nx - 1:
                    dVdx_n = 0.5 * ((V[i, j] - V[i, j-1]) / dx +
                                   (V[i+1, j] - V[i+1, j-1]) / dx)
                else:
                    dVdx_n = 0.25 * ((V[i, j+1] - V[i, j-1]) / dx +
                                    (V[i+1, j+1] - V[i+1, j-1]) / dx)

                flux_n = D_xy * dVdx_n + D_yy * dVdy_n
            else:
                flux_n = 0.0

            # South flux
            if i > 0:
                dVdy_s = (V[i, j] - V[i-1, j]) / dy

                if j == 0:
                    dVdx_s = 0.5 * ((V[i-1, j+1] - V[i-1, j]) / dx +
                                   (V[i, j+1] - V[i, j]) / dx)
                elif j == nx - 1:
                    dVdx_s = 0.5 * ((V[i-1, j] - V[i-1, j-1]) / dx +
                                   (V[i, j] - V[i, j-1]) / dx)
                else:
                    dVdx_s = 0.25 * ((V[i-1, j+1] - V[i-1, j-1]) / dx +
                                    (V[i, j+1] - V[i, j-1]) / dx)

                flux_s = D_xy * dVdx_s + D_yy * dVdy_s
            else:
                flux_s = 0.0

            diff[i, j] = (flux_e - flux_w) / dx + (flux_n - flux_s) / dy

    return diff


@njit(parallel=True, cache=True)
def compute_diffusion_isotropic(V, D, dx, dy):
    """
    Simplified FVM for isotropic diffusion (no fiber orientation).

    Reduces to standard 5-point Laplacian with no-flux BC.

    Parameters
    ----------
    V : ndarray (ny, nx)
        Membrane potential field
    D : float
        Isotropic diffusion coefficient
    dx, dy : float
        Grid spacing

    Returns
    -------
    diff : ndarray (ny, nx)
        D * Laplacian(V)
    """
    ny, nx = V.shape
    diff = np.zeros_like(V)

    for i in prange(ny):
        for j in range(nx):
            # Get neighbors with no-flux BC (Neumann)
            V_c = V[i, j]

            V_e = V[i, j+1] if j < nx - 1 else V_c
            V_w = V[i, j-1] if j > 0 else V_c
            V_n = V[i+1, j] if i < ny - 1 else V_c
            V_s = V[i-1, j] if i > 0 else V_c

            # 5-point Laplacian
            laplacian = ((V_e - 2*V_c + V_w) / (dx * dx) +
                        (V_n - 2*V_c + V_s) / (dy * dy))

            diff[i, j] = D * laplacian

    return diff


class DiffusionOperator:
    """
    FVM-based diffusion operator for monodomain equation.

    Supports:
    - Isotropic diffusion
    - Anisotropic with uniform fiber angle
    - Anisotropic with spatially varying fiber field

    Parameters
    ----------
    ny, nx : int
        Grid dimensions
    dx, dy : float
        Grid spacing (cm)
    D_L : float
        Longitudinal diffusion coefficient (cm^2/ms)
    D_T : float
        Transverse diffusion coefficient (cm^2/ms)
    fiber_angle : float or ndarray
        Fiber angle in radians. If scalar, uniform orientation.
        If ndarray (ny, nx), spatially varying.
    """

    def __init__(self, ny, nx, dx, dy, D_L=0.001, D_T=0.00025, fiber_angle=0.0):
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.dy = dy
        self.D_L = D_L
        self.D_T = D_T

        # Determine mode based on fiber angle
        if np.isscalar(fiber_angle):
            self.uniform_fiber = True
            self.fiber_angle = float(fiber_angle)
            self.D_xx = None
            self.D_yy = None
            self.D_xy = None
        else:
            self.uniform_fiber = False
            self.fiber_angle = np.asarray(fiber_angle)
            # Precompute tensor field
            self._precompute_tensor_field()

        # Check for isotropic case
        self.isotropic = (abs(D_L - D_T) < 1e-12)

    def _precompute_tensor_field(self):
        """Precompute diffusion tensor at each cell for varying fiber field."""
        self.D_xx = np.zeros((self.ny, self.nx))
        self.D_yy = np.zeros((self.ny, self.nx))
        self.D_xy = np.zeros((self.ny, self.nx))

        for i in range(self.ny):
            for j in range(self.nx):
                theta = self.fiber_angle[i, j]
                self.D_xx[i, j], self.D_yy[i, j], self.D_xy[i, j] = \
                    compute_diffusion_tensor(theta, self.D_L, self.D_T)

    def apply(self, V):
        """
        Apply diffusion operator to voltage field.

        Parameters
        ----------
        V : ndarray (ny, nx)
            Membrane potential field

        Returns
        -------
        diff : ndarray (ny, nx)
            div(D * grad(V))
        """
        if self.isotropic:
            return compute_diffusion_isotropic(V, self.D_L, self.dx, self.dy)
        elif self.uniform_fiber:
            return compute_diffusion_fvm_uniform(
                V, self.D_L, self.D_T, self.fiber_angle, self.dx, self.dy
            )
        else:
            return compute_diffusion_fvm(
                V, self.D_xx, self.D_yy, self.D_xy, self.dx, self.dy
            )

    def get_stability_limit(self, safety=0.9):
        """
        Compute maximum stable time step for explicit integration.

        Uses von Neumann stability analysis for diffusion equation.

        Parameters
        ----------
        safety : float
            Safety factor (0 < safety <= 1)

        Returns
        -------
        dt_max : float
            Maximum stable time step (ms)
        """
        # For anisotropic diffusion, use maximum eigenvalue
        D_max = max(self.D_L, self.D_T)

        # Stability criterion: dt <= dx^2 / (4 * D_max) for 2D
        dt_max = 0.25 * min(self.dx**2, self.dy**2) / D_max

        return safety * dt_max
