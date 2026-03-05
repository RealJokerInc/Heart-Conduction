"""Diffusion coefficient <-> LBM relaxation parameter conversions.

Pipeline: sigma -> D -> tau (CONVENTIONS.md)
"""

import math


def sigma_to_D(sigma_l: float, sigma_t: float, fiber_angle: float,
               chi: float, Cm: float) -> tuple[float, float, float]:
    """Convert conductivity tensor to diffusion tensor.

    D = sigma / (chi * Cm)
    sigma = sigma_t * I + (sigma_l - sigma_t) * a * a^T

    Returns (D_xx, D_yy, D_xy).
    """
    cos_a = math.cos(fiber_angle)
    sin_a = math.sin(fiber_angle)

    sigma_xx = sigma_t + (sigma_l - sigma_t) * cos_a ** 2
    sigma_yy = sigma_t + (sigma_l - sigma_t) * sin_a ** 2
    sigma_xy = (sigma_l - sigma_t) * cos_a * sin_a

    scale = 1.0 / (chi * Cm)
    return sigma_xx * scale, sigma_yy * scale, sigma_xy * scale


def tau_from_D(D: float, dx: float, dt: float, cs2: float = 1.0 / 3.0) -> float:
    """Scalar BGK relaxation time from isotropic diffusion coefficient.

    tau = 0.5 + D * dt / (cs2 * dx^2)
    """
    return 0.5 + D * dt / (cs2 * dx * dx)


def tau_tensor_from_D(D_xx: float, D_yy: float, D_xy: float,
                      dx: float, dt: float,
                      cs2: float = 1.0 / 3.0) -> tuple[float, float, float]:
    """MRT relaxation times from anisotropic diffusion tensor.

    tau_ij = delta_ij/2 + D_ij * dt / (cs2 * dx^2)

    Returns (tau_xx, tau_yy, tau_xy).
    """
    scale = dt / (cs2 * dx * dx)
    tau_xx = 0.5 + D_xx * scale
    tau_yy = 0.5 + D_yy * scale
    tau_xy = D_xy * scale  # No delta_ij term for off-diagonal
    return tau_xx, tau_yy, tau_xy


def check_stability(D: float, dx: float, dt: float,
                    cs2: float = 1.0 / 3.0) -> tuple[bool, float]:
    """Check BGK stability: tau > 0.5.

    Returns (is_stable, tau_value).
    """
    tau = tau_from_D(D, dx, dt, cs2)
    return tau > 0.5, tau


def check_stability_tensor(D_xx: float, D_yy: float, D_xy: float,
                           dx: float, dt: float,
                           cs2: float = 1.0 / 3.0) -> tuple[bool, float]:
    """Check MRT stability: all eigenvalues of tau tensor > 0.5.

    Returns (is_stable, tau_min).
    """
    tau_xx, tau_yy, tau_xy = tau_tensor_from_D(D_xx, D_yy, D_xy, dx, dt, cs2)

    # Eigenvalues of 2x2 symmetric [[tau_xx, tau_xy], [tau_xy, tau_yy]]
    trace = tau_xx + tau_yy
    det = tau_xx * tau_yy - tau_xy * tau_xy
    disc = math.sqrt(max(0.0, (trace / 2) ** 2 - det))
    tau_min = trace / 2 - disc
    return tau_min > 0.5, tau_min
