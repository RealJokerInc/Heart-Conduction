"""MRT collision operator for D2Q9 lattice.

Moment space (9 moments, Lallemand & Luo 2000):
  Row 0: rho  (conserved, s_0 = 0)
  Row 1: e    (energy, s_e)
  Row 2: eps  (energy-square, s_eps)
  Row 3: j_x  (x-flux, s_j)
  Row 4: q_x  (energy-flux x, s_q)
  Row 5: j_y  (y-flux, s_j)
  Row 6: q_y  (energy-flux y, s_q)
  Row 7: p_xx (stress diff, s_pxx) -- encodes D_xx - D_yy
  Row 8: p_xy (shear stress, s_pxy) -- encodes D_xy

Layer 2: pure function, torch.compile compatible.
"""

import torch
from torch import Tensor

# Lallemand & Luo (2000) M matrix for D2Q9
# Column order matches CONVENTIONS.md: rest, E, W, N, S, NE, NW, SW, SE
_M_D2Q9 = torch.tensor([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1],   # rho
    [-4, -1, -1, -1, -1,  2,  2,  2,  2],   # e
    [ 4, -2, -2, -2, -2,  1,  1,  1,  1],   # eps
    [ 0,  1, -1,  0,  0,  1, -1, -1,  1],   # j_x
    [ 0, -2,  2,  0,  0,  1, -1, -1,  1],   # q_x
    [ 0,  0,  0,  1, -1,  1,  1, -1, -1],   # j_y
    [ 0,  0,  0, -2,  2,  1,  1, -1, -1],   # q_y
    [ 0,  1,  1, -1, -1,  0,  0,  0,  0],   # p_xx
    [ 0,  0,  0,  0,  0,  1, -1,  1, -1],   # p_xy
], dtype=torch.float64)

_M_inv_D2Q9 = torch.linalg.inv(_M_D2Q9)

# Equilibrium moment coefficients: m_eq = coeff * V
# For diffusion (no advection): j_x = j_y = 0, higher moments from e_eq, eps_eq
# e_eq = -4*V + 3*(jx^2+jy^2)/rho -> for diffusion (j=0): e_eq = -4*V... but
# with the normalization in Lallemand & Luo, e_eq depends on the conserved density.
# For reaction-diffusion LBM: rho = V, u = 0
# e_eq = -2*V (from the standard D2Q9 equilibrium)
# eps_eq = V
# All flux/stress moments: 0
_meq_coeff_D2Q9 = torch.tensor([
    1.0,    # rho_eq = V
    -2.0,   # e_eq = -2*V
    1.0,    # eps_eq = V
    0.0,    # j_x_eq = 0
    0.0,    # q_x_eq = 0
    0.0,    # j_y_eq = 0
    0.0,    # q_y_eq = 0
    0.0,    # p_xx_eq = 0
    0.0,    # p_xy_eq = 0
], dtype=torch.float64)


def mrt_collide_d2q9(f: Tensor, V: Tensor, R: Tensor, dt: float,
                     s_e: float, s_eps: float,
                     s_jx: float, s_q: float,
                     s_pxx: float, s_pxy: float,
                     w: Tensor,
                     s_jy: float = None) -> Tensor:
    """MRT collision for D2Q9 with full anisotropic diffusion tensor.

    Diffusion tensor components (Chapman-Enskog):
        D_xx = cs2 * (1/s_jx - 0.5) * dt
        D_yy = cs2 * (1/s_jy - 0.5) * dt
    For D_xy != 0, rotate s_jx/s_jy via moment-space transformation (Phase 8).

    Args:
        f: (9, Nx, Ny) distributions
        V: (Nx, Ny) voltage
        R: (Nx, Ny) source
        dt: time step
        s_e: relaxation rate for energy (free parameter)
        s_eps: relaxation rate for energy-square (free parameter)
        s_jx: relaxation rate for j_x moment → controls D_xx
        s_q: relaxation rate for energy-flux moments (free parameter)
        s_pxx: relaxation rate for p_xx (free parameter for stability)
        s_pxy: relaxation rate for p_xy (free parameter for stability)
        w: (9,) weights
        s_jy: relaxation rate for j_y moment → controls D_yy.
              If None, defaults to s_jx (isotropic).

    Returns:
        f_star: (9, Nx, Ny)
    """
    if s_jy is None:
        s_jy = s_jx

    dev, dtp = f.device, f.dtype
    M = _M_D2Q9.to(device=dev, dtype=dtp)
    M_inv = _M_inv_D2Q9.to(device=dev, dtype=dtp)
    meq_c = _meq_coeff_D2Q9.to(device=dev, dtype=dtp)

    Q, Nx, Ny = f.shape
    f_flat = f.reshape(Q, -1)  # (9, N)

    # Transform to moment space
    m = M @ f_flat  # (9, N)

    # Equilibrium moments
    V_flat = V.reshape(-1)
    m_eq = meq_c[:, None] * V_flat[None, :]

    # Relaxation rates: S = diag(0, s_e, s_eps, s_jx, s_q, s_jy, s_q, s_pxx, s_pxy)
    S = torch.tensor([0.0, s_e, s_eps, s_jx, s_q, s_jy, s_q, s_pxx, s_pxy],
                     device=dev, dtype=dtp)

    # Relax
    m_star = m - S[:, None] * (m - m_eq)

    # Transform back
    f_star = (M_inv @ m_star).reshape(Q, Nx, Ny)

    # Add source in distribution space: dt * w_i * R (Campos Eq. 11)
    f_star = f_star + dt * w[:, None, None] * R[None, :, :]

    return f_star
