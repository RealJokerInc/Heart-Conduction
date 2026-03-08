"""MRT collision operator for D2Q5 lattice.

Moment space (5 moments):
  Row 0: rho  (conserved, s_0 = 0)
  Row 1: j_x  (x-flux, s_x)
  Row 2: j_y  (y-flux, s_y)
  Row 3: e    (energy = -4*f0 + f1+f2+f3+f4, s_e)
  Row 4: p_xx (stress diff = f1+f2-f3-f4, s_pxx) -- encodes D_xx - D_yy

D2Q5 can encode D_xx != D_yy via s_x, s_y and s_pxx, but NOT D_xy.

Layer 2: pure function, torch.compile compatible.
"""

import torch
from torch import Tensor

# D2Q5 transformation matrix
# Orthogonal rows, full rank 5
_M_D2Q5 = torch.tensor([
    [ 1,  1,  1,  1,  1],   # rho = sum(f)
    [ 0,  1, -1,  0,  0],   # j_x = f1 - f2
    [ 0,  0,  0,  1, -1],   # j_y = f3 - f4
    [-4,  1,  1,  1,  1],   # e = -4*f0 + f1+f2+f3+f4
    [ 0,  1,  1, -1, -1],   # p_xx = f1+f2 - f3-f4 (stress diff)
], dtype=torch.float64)

_M_inv_D2Q5 = torch.linalg.inv(_M_D2Q5)

# Equilibrium moment coefficients: m_eq = coeff * V
# rho_eq = V, j_x_eq = 0, j_y_eq = 0, e_eq = -2/3*V, p_xx_eq = 0
# Derivation: e = [-4,1,1,1,1] . [V/3, V/6, V/6, V/6, V/6]
#            = -4V/3 + 4(V/6) = -4V/3 + 2V/3 = -2V/3
_meq_coeff_D2Q5 = torch.tensor([1.0, 0.0, 0.0, -2.0/3.0, 0.0], dtype=torch.float64)


def mrt_collide_d2q5(f: Tensor, V: Tensor, R: Tensor, dt: float,
                     s_x: float, s_y: float,
                     s_e: float, s_pxx: float,
                     w: Tensor) -> Tensor:
    """MRT collision for D2Q5 with axis-aligned anisotropy.

    Args:
        f: (5, Nx, Ny) distributions
        V: (Nx, Ny) voltage
        R: (Nx, Ny) source
        dt: time step
        s_x, s_y: relaxation rates for flux moments (from D_xx, D_yy)
        s_e: relaxation rate for energy moment (free, stability)
        s_pxx: relaxation rate for stress difference (from D_xx - D_yy)
        w: (5,) weights

    Returns:
        f_star: (5, Nx, Ny)
    """
    dev, dtp = f.device, f.dtype
    M = _M_D2Q5.to(device=dev, dtype=dtp)
    M_inv = _M_inv_D2Q5.to(device=dev, dtype=dtp)
    meq_c = _meq_coeff_D2Q5.to(device=dev, dtype=dtp)

    Q, Nx, Ny = f.shape
    f_flat = f.reshape(Q, -1)  # (5, N)

    # Transform to moment space
    m = M @ f_flat  # (5, N)

    # Equilibrium moments
    V_flat = V.reshape(-1)
    m_eq = meq_c[:, None] * V_flat[None, :]

    # Relaxation rates diagonal: [s_0=0, s_x, s_y, s_e, s_pxx]
    S = torch.tensor([0.0, s_x, s_y, s_e, s_pxx], device=dev, dtype=dtp)

    # Relax
    m_star = m - S[:, None] * (m - m_eq)

    # Transform back
    f_star = (M_inv @ m_star).reshape(Q, Nx, Ny)

    # Add source in distribution space: dt * w_i * R (Campos Eq. 11)
    f_star = f_star + dt * w[:, None, None] * R[None, :, :]

    return f_star
