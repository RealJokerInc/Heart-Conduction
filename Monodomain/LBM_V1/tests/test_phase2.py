"""Phase 2 validation: Collision operators."""

import sys
sys.path.insert(0, '.')

import torch
from src.lattice import D2Q5, D2Q9
from src.collision.bgk import bgk_collide
from src.collision.mrt.d2q5 import mrt_collide_d2q5
from src.collision.mrt.d2q9 import mrt_collide_d2q9
from src.diffusion import tau_from_D

torch.set_default_dtype(torch.float64)

Nx, Ny = 20, 20
d5 = D2Q5()
d9 = D2Q9()
w5 = torch.tensor(d5.w)
w9 = torch.tensor(d9.w)


def test_2v1_bgk_equilibrium():
    """BGK: equilibrium f is unchanged by collision (except source)."""
    V = torch.ones(Nx, Ny) * 0.5
    R = torch.zeros(Nx, Ny)
    f_eq = w5[:, None, None] * V[None, :, :]
    tau = 0.8
    omega = 1.0 / tau
    f_star = bgk_collide(f_eq, V, R, dt=0.01, omega=omega, w=w5)
    err = (f_star - f_eq).abs().max().item()
    assert err < 1e-14, f"BGK equilibrium error: {err}"
    print(f"2-V1 PASS: BGK equilibrium unchanged (err={err:.1e})")


def test_2v2_mrt_d2q9_equilibrium():
    """MRT D2Q9: equilibrium f unchanged by collision (except source)."""
    V = torch.ones(Nx, Ny) * 0.5
    R = torch.zeros(Nx, Ny)
    f_eq = w9[:, None, None] * V[None, :, :]

    f_star = mrt_collide_d2q9(f_eq, V, R, dt=0.01,
                               s_e=1.0, s_eps=1.0, s_jx=1.0, s_q=1.0,
                               s_pxx=1.0, s_pxy=1.0, w=w9)
    err = (f_star - f_eq).abs().max().item()
    assert err < 1e-13, f"MRT D2Q9 equilibrium error: {err}"
    print(f"2-V2 PASS: MRT D2Q9 equilibrium unchanged (err={err:.1e})")


def test_2v3_mrt_matches_bgk_isotropic():
    """MRT D2Q9 with isotropic D gives same result as BGK."""
    V = torch.randn(Nx, Ny) * 0.1
    R = torch.randn(Nx, Ny) * 0.01
    dt = 0.01

    # For isotropic D, all relaxation rates equal
    D = 0.001
    dx = 0.025
    tau = tau_from_D(D, dx, dt)
    omega = 1.0 / tau
    s = 1.0 / tau  # same for all flux/stress moments

    f = w9[:, None, None] * V[None, :, :] + torch.randn(9, Nx, Ny) * 0.001
    # V must equal sum(f) for MRT-BGK equivalence (LBM invariant)
    V = f.sum(dim=0)

    f_bgk = bgk_collide(f, V, R, dt, omega, w9)
    f_mrt = mrt_collide_d2q9(f, V, R, dt,
                              s_e=s, s_eps=s, s_jx=s, s_q=s,
                              s_pxx=s, s_pxy=s, w=w9)
    err = (f_mrt - f_bgk).abs().max().item()
    assert err < 1e-12, f"MRT vs BGK isotropic error: {err}"
    print(f"2-V3 PASS: MRT isotropic matches BGK (err={err:.1e})")


def test_2v4_mrt_anisotropic_dxx_dyy():
    """MRT D2Q9: D_xx != D_yy produces directional effect."""
    V = torch.randn(Nx, Ny) * 0.1
    R = torch.zeros(Nx, Ny)
    dt = 0.01

    # Non-equilibrium f with gradient in x
    f = w9[:, None, None] * V[None, :, :]
    f[1] += 0.01  # add x-flux perturbation

    # Isotropic case
    s_iso = 1.2
    f1 = mrt_collide_d2q9(f, V, R, dt, s_e=1.0, s_eps=1.0,
                           s_jx=s_iso, s_q=1.0, s_pxx=s_iso, s_pxy=s_iso, w=w9)

    # Anisotropic: different pxx rate
    f2 = mrt_collide_d2q9(f, V, R, dt, s_e=1.0, s_eps=1.0,
                           s_jx=s_iso, s_q=1.0, s_pxx=0.8, s_pxy=s_iso, w=w9)

    diff = (f1 - f2).abs().max().item()
    assert diff > 1e-5, f"Anisotropic should differ from isotropic: diff={diff}"
    print(f"2-V4 PASS: Anisotropic D_xx!=D_yy produces different result (diff={diff:.2e})")


def test_2v5_mrt_anisotropic_dxy():
    """MRT D2Q9: D_xy != 0 produces different relaxation."""
    V = torch.randn(Nx, Ny) * 0.1
    R = torch.zeros(Nx, Ny)
    dt = 0.01

    f = w9[:, None, None] * V[None, :, :]
    f[5] += 0.01  # add diagonal perturbation (NE)

    s_j = 1.2
    # No off-diagonal
    f1 = mrt_collide_d2q9(f, V, R, dt, s_e=1.0, s_eps=1.0,
                           s_jx=s_j, s_q=1.0, s_pxx=s_j, s_pxy=s_j, w=w9)
    # Different pxy rate
    f2 = mrt_collide_d2q9(f, V, R, dt, s_e=1.0, s_eps=1.0,
                           s_jx=s_j, s_q=1.0, s_pxx=s_j, s_pxy=0.8, w=w9)

    diff = (f1 - f2).abs().max().item()
    assert diff > 1e-5, f"D_xy should matter: diff={diff}"
    print(f"2-V5 PASS: D_xy!=0 produces different result (diff={diff:.2e})")


def test_2v6_source_conservation():
    """Source term conservation: sum(f*_i) = V + dt*R."""
    V = torch.randn(Nx, Ny) * 0.1 + 0.5
    R = torch.randn(Nx, Ny) * 0.01
    dt = 0.01

    f = w9[:, None, None] * V[None, :, :]

    f_star = mrt_collide_d2q9(f, V, R, dt,
                               s_e=1.2, s_eps=1.1, s_jx=1.0, s_q=1.3,
                               s_pxx=0.9, s_pxy=0.8, w=w9)
    V_after = f_star.sum(dim=0)
    V_expected = V + R * dt
    err = (V_after - V_expected).abs().max().item()
    assert err < 1e-13, f"Source conservation error: {err}"
    print(f"2-V6 PASS: source conservation sum(f*)=V+dt*R (err={err:.1e})")

    # Also test BGK
    tau = 0.8
    f_star_bgk = bgk_collide(f, V, R, dt, 1.0/tau, w9)
    V_bgk = f_star_bgk.sum(dim=0)
    err_bgk = (V_bgk - V_expected).abs().max().item()
    assert err_bgk < 1e-13
    print(f"       BGK source conservation also OK (err={err_bgk:.1e})")


if __name__ == "__main__":
    test_2v1_bgk_equilibrium()
    test_2v2_mrt_d2q9_equilibrium()
    test_2v3_mrt_matches_bgk_isotropic()
    test_2v4_mrt_anisotropic_dxx_dyy()
    test_2v5_mrt_anisotropic_dxy()
    test_2v6_source_conservation()
    print("\nPhase 2: ALL 6 TESTS PASS")
