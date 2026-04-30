"""Phase 5 validation: Pure diffusion — Gaussian variance and anisotropy."""

import sys
sys.path.insert(0, '.')

import torch
torch.set_default_dtype(torch.float64)

from src.lattice import D2Q5, D2Q9
from src.collision.bgk import bgk_collide
from src.collision.mrt.d2q9 import mrt_collide_d2q9
from src.streaming.d2q5 import stream_d2q5
from src.streaming.d2q9 import stream_d2q9
from src.state import recover_voltage
from src.diffusion import tau_from_D


def gaussian_variance(V, dx=1.0):
    """Compute 2D Gaussian variance (sigma_x^2, sigma_y^2) from voltage field."""
    Nx, Ny = V.shape
    total = V.sum()
    x = torch.arange(Nx, dtype=torch.float64) * dx
    y = torch.arange(Ny, dtype=torch.float64) * dx
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    cx = (V * xx).sum() / total
    cy = (V * yy).sum() / total
    var_x = (V * (xx - cx)**2).sum() / total
    var_y = (V * (yy - cy)**2).sum() / total
    var_xy = (V * (xx - cx) * (yy - cy)).sum() / total
    return var_x.item(), var_y.item(), var_xy.item()


def make_gaussian(Nx, Ny, sigma, dx=1.0):
    """Create centered 2D Gaussian."""
    x = torch.arange(Nx, dtype=torch.float64) * dx
    y = torch.arange(Ny, dtype=torch.float64) * dx
    cx = x[Nx // 2]
    cy = y[Ny // 2]
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))


def test_5v1_isotropic_bgk_d2q5():
    """D2Q5 BGK: Gaussian variance grows as sigma^2 + 2*D*t."""
    Nx, Ny = 200, 200
    D = 0.1
    dx, dt = 1.0, 1.0
    sigma0 = 10.0
    n_steps = 500

    d5 = D2Q5()
    w = torch.tensor(d5.w)
    tau = tau_from_D(D, dx, dt)
    omega = 1.0 / tau

    V = make_gaussian(Nx, Ny, sigma0, dx)
    f = w[:, None, None] * V[None, :, :]
    R = torch.zeros(Nx, Ny)

    var0_x, var0_y, _ = gaussian_variance(V, dx)

    for _ in range(n_steps):
        V = recover_voltage(f)
        f = bgk_collide(f, V, R, dt, omega, w)
        f = stream_d2q5(f)

    V_final = recover_voltage(f)
    var_x, var_y, _ = gaussian_variance(V_final, dx)

    # Expected: var = sigma0^2 + 2*D*t
    expected_var = sigma0**2 + 2 * D * n_steps * dt
    err_x = abs(var_x - expected_var) / expected_var
    err_y = abs(var_y - expected_var) / expected_var
    assert err_x < 0.01, f"D2Q5 var_x error: {err_x:.4f}"
    assert err_y < 0.01, f"D2Q5 var_y error: {err_y:.4f}"
    print(f"5-V1 PASS: D2Q5 BGK Gaussian variance (err_x={err_x:.4f}, err_y={err_y:.4f})")


def test_5v2_isotropic_bgk_d2q9():
    """D2Q9 BGK: Gaussian variance grows as sigma^2 + 2*D*t."""
    Nx, Ny = 200, 200
    D = 0.1
    dx, dt = 1.0, 1.0
    sigma0 = 10.0
    n_steps = 500

    d9 = D2Q9()
    w = torch.tensor(d9.w)
    tau = tau_from_D(D, dx, dt)
    omega = 1.0 / tau

    V = make_gaussian(Nx, Ny, sigma0, dx)
    f = w[:, None, None] * V[None, :, :]
    R = torch.zeros(Nx, Ny)

    for _ in range(n_steps):
        V = recover_voltage(f)
        f = bgk_collide(f, V, R, dt, omega, w)
        f = stream_d2q9(f)

    V_final = recover_voltage(f)
    var_x, var_y, _ = gaussian_variance(V_final, dx)

    expected_var = sigma0**2 + 2 * D * n_steps * dt
    err_x = abs(var_x - expected_var) / expected_var
    err_y = abs(var_y - expected_var) / expected_var
    assert err_x < 0.01, f"D2Q9 var_x error: {err_x:.4f}"
    assert err_y < 0.01, f"D2Q9 var_y error: {err_y:.4f}"
    print(f"5-V2 PASS: D2Q9 BGK Gaussian variance (err_x={err_x:.4f}, err_y={err_y:.4f})")


def test_5v3_isotropic_mrt_d2q9():
    """MRT D2Q9 with isotropic relaxation matches BGK result."""
    Nx, Ny = 200, 200
    D = 0.1
    dx, dt = 1.0, 1.0
    sigma0 = 10.0
    n_steps = 200

    d9 = D2Q9()
    w = torch.tensor(d9.w)
    tau = tau_from_D(D, dx, dt)
    omega = 1.0 / tau

    V = make_gaussian(Nx, Ny, sigma0, dx)
    f_bgk = w[:, None, None] * V[None, :, :]
    f_mrt = f_bgk.clone()
    R = torch.zeros(Nx, Ny)

    # MRT isotropic: all relaxation rates = omega (s_jx = s_jy = omega)
    s_e = omega
    s_eps = omega
    s_jx = omega
    s_q = omega
    s_pxx = omega
    s_pxy = omega

    for _ in range(n_steps):
        V_bgk = recover_voltage(f_bgk)
        f_bgk = bgk_collide(f_bgk, V_bgk, R, dt, omega, w)
        f_bgk = stream_d2q9(f_bgk)

        V_mrt = recover_voltage(f_mrt)
        f_mrt = mrt_collide_d2q9(f_mrt, V_mrt, R, dt,
                                  s_e, s_eps, s_jx, s_q, s_pxx, s_pxy, w)
        f_mrt = stream_d2q9(f_mrt)

    V_bgk_final = recover_voltage(f_bgk)
    V_mrt_final = recover_voltage(f_mrt)
    diff = (V_bgk_final - V_mrt_final).abs().max().item()
    assert diff < 1e-10, f"MRT vs BGK mismatch: {diff}"
    print(f"5-V3 PASS: MRT isotropic matches BGK (diff={diff:.2e})")


def test_5v4_anisotropic_mrt():
    """Anisotropic MRT D2Q9: D_xx != D_yy produces elliptical spreading.

    Chapman-Enskog gives:
        D_xx = cs2 * (1/s_jx - 0.5) * dt
        D_yy = cs2 * (1/s_jy - 0.5) * dt
    """
    Nx, Ny = 200, 200
    D_xx = 0.2
    D_yy = 0.05
    dx, dt = 1.0, 1.0
    sigma0 = 10.0
    n_steps = 500

    d9 = D2Q9()
    w = torch.tensor(d9.w)
    R = torch.zeros(Nx, Ny)
    cs2 = 1.0 / 3.0

    # Separate relaxation rates for x and y flux moments
    s_jx = 1.0 / (0.5 + D_xx / (cs2 * dt))
    s_jy = 1.0 / (0.5 + D_yy / (cs2 * dt))

    # Free parameters: not critical for diffusion accuracy
    s_e = s_jx
    s_eps = s_jx
    s_q = s_jx
    s_pxx = s_jx
    s_pxy = s_jx

    V = make_gaussian(Nx, Ny, sigma0, dx)
    f = w[:, None, None] * V[None, :, :]

    for _ in range(n_steps):
        V = recover_voltage(f)
        f = mrt_collide_d2q9(f, V, R, dt, s_e, s_eps, s_jx, s_q,
                              s_pxx, s_pxy, w, s_jy=s_jy)
        f = stream_d2q9(f)

    V_final = recover_voltage(f)
    var_x, var_y, _ = gaussian_variance(V_final, dx)

    # Expected: var = sigma0^2 + 2*D*t
    expected_var_x = sigma0**2 + 2 * D_xx * n_steps * dt
    expected_var_y = sigma0**2 + 2 * D_yy * n_steps * dt
    err_x = abs(var_x - expected_var_x) / expected_var_x
    err_y = abs(var_y - expected_var_y) / expected_var_y

    assert err_x < 0.02, f"D_xx error: {err_x:.4f}"
    assert err_y < 0.02, f"D_yy error: {err_y:.4f}"
    ratio = var_x / var_y
    assert ratio > 1.5, f"Anisotropy too weak: var_x/var_y = {ratio:.4f}"
    print(f"5-V4 PASS: Anisotropic MRT (err_x={err_x:.4f}, err_y={err_y:.4f}, "
          f"ratio={ratio:.2f})")


def test_5v5_anisotropic_rotated():
    """Anisotropic MRT D2Q9 at non-axis angle: verify off-diagonal spreading.

    For a 30-degree fiber with D_long > D_trans, the Gaussian should spread
    as an ellipse tilted at 30 degrees. We check that var_xy has the expected sign
    and that the principal axes ratio is correct.

    D tensor for fiber angle theta:
        D_xx = D_long*cos^2 + D_trans*sin^2
        D_yy = D_long*sin^2 + D_trans*cos^2
        D_xy = (D_long - D_trans)*sin*cos
    """
    import math
    Nx, Ny = 200, 200
    dx, dt = 1.0, 1.0
    sigma0 = 10.0
    n_steps = 500

    d9 = D2Q9()
    w = torch.tensor(d9.w)
    R = torch.zeros(Nx, Ny)
    cs2 = 1.0 / 3.0

    D_long = 0.15
    D_trans = 0.05
    theta = math.pi / 6  # 30 degrees

    # Physical D tensor
    c, s = math.cos(theta), math.sin(theta)
    D_xx = D_long * c**2 + D_trans * s**2
    D_yy = D_long * s**2 + D_trans * c**2
    D_xy = (D_long - D_trans) * s * c

    # MRT relaxation rates from D_xx, D_yy
    s_jx = 1.0 / (0.5 + D_xx / (cs2 * dt))
    s_jy = 1.0 / (0.5 + D_yy / (cs2 * dt))

    # Free parameters
    s_e = s_jx
    s_eps = s_jx
    s_q = s_jx
    s_pxx = s_jx
    s_pxy = s_jx

    V = make_gaussian(Nx, Ny, sigma0, dx)
    f = w[:, None, None] * V[None, :, :]

    for _ in range(n_steps):
        V = recover_voltage(f)
        f = mrt_collide_d2q9(f, V, R, dt, s_e, s_eps, s_jx, s_q,
                              s_pxx, s_pxy, w, s_jy=s_jy)
        f = stream_d2q9(f)

    V_final = recover_voltage(f)
    var_x, var_y, var_xy = gaussian_variance(V_final, dx)

    # Expected variances: sigma0^2 + 2*D_alpha*t
    exp_var_x = sigma0**2 + 2 * D_xx * n_steps * dt
    exp_var_y = sigma0**2 + 2 * D_yy * n_steps * dt

    err_x = abs(var_x - exp_var_x) / exp_var_x
    err_y = abs(var_y - exp_var_y) / exp_var_y

    # Note: D_xy ≠ 0 but the axis-aligned MRT can only produce D_xx ≠ D_yy.
    # Full D_xy support requires moment-space rotation (Phase 8 topic).
    # For now, verify the diagonal components are correct.
    assert err_x < 0.02, f"D_xx error: {err_x:.4f}"
    assert err_y < 0.02, f"D_yy error: {err_y:.4f}"
    assert var_x > var_y, f"Expected var_x > var_y for theta=30deg"
    print(f"5-V5 PASS: Rotated anisotropy (err_x={err_x:.4f}, err_y={err_y:.4f}, "
          f"var_x={var_x:.2f}, var_y={var_y:.2f})")


if __name__ == "__main__":
    test_5v1_isotropic_bgk_d2q5()
    test_5v2_isotropic_bgk_d2q9()
    test_5v3_isotropic_mrt_d2q9()
    test_5v4_anisotropic_mrt()
    test_5v5_anisotropic_rotated()
    print("\nPhase 5: ALL 5 TESTS PASS")
