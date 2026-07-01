"""Phase 0 Step 0.2 — LBM per-axis anisotropy via D2Q9-MRT.

0.2c (the correctness gate): a pure-diffusion benchmark with **dx != dt** that
recovers the per-axis diffusion tensor from second-moment growth. dx != dt is
essential — the correct mapping tau = 0.5 + D*dt/(cs2*dx^2) and the wrong
lattice-unit form tau = 0.5 + D/(cs2*dt) coincide when dx == dt, so a dx==dt
benchmark would pass for both and prove nothing.

Plus engine guards (0.2a) and the cardiac_core lbm() wrapper routing (0.2b).
Covers PLAN Phase 0 Step 0.2.
"""
import numpy as np
import pytest
import torch

from cardiac_core import create_cardiac_mesh, run_lbm
from cardiac_core.ionic import MHAS13Model
from cardiac_core._lbm.simulation import LBMSimulation
from cardiac_core._lbm.step import lbm_step_d2q9_mrt
from cardiac_core._lbm.diffusion import tau_tensor_from_D
from cardiac_core._lbm.lattice import D2Q9


def _diffuse_recover_D(D_xx, D_yy, dx, dt, nsteps=200, Nx=81, Ny=81, sigma0=0.05):
    """Evolve a Gaussian blob under pure D2Q9-MRT diffusion (R=0); recover D
    from the second-moment growth: sigma^2(t) = sigma^2(0) + 2 D t (per axis)."""
    cs2 = 1.0 / 3.0
    tau_xx, tau_yy, _ = tau_tensor_from_D(D_xx, D_yy, 0.0, dx, dt, cs2)
    s_jx, s_jy = 1.0 / tau_xx, 1.0 / tau_yy
    w = torch.tensor(D2Q9().w, dtype=torch.float64)

    xs = (torch.arange(Nx, dtype=torch.float64) - Nx // 2) * dx
    ys = (torch.arange(Ny, dtype=torch.float64) - Ny // 2) * dx
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    V = torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma0 ** 2))
    f = w[:, None, None] * V[None, :, :]
    R = torch.zeros(Nx, Ny, dtype=torch.float64)
    bounce = {a: torch.zeros(Nx, Ny, dtype=torch.bool) for a in range(1, 9)}  # no reflection

    def moments(Vf):
        m0 = Vf.sum()
        mx, my = (X * Vf).sum() / m0, (Y * Vf).sum() / m0
        sxx = ((X - mx) ** 2 * Vf).sum() / m0
        syy = ((Y - my) ** 2 * Vf).sum() / m0
        return float(sxx), float(syy)

    sxx0, syy0 = moments(V)
    for _ in range(nsteps):
        f, V = lbm_step_d2q9_mrt(f, V, R, dt, w, 1.0, 1.0, s_jx, 1.0, 1.0, 1.0,
                                 bounce, s_jy=s_jy)
    sxx1, syy1 = moments(V)
    T = nsteps * dt
    return (sxx1 - sxx0) / (2 * T), (syy1 - syy0) / (2 * T)


def test_mrt_recovers_D_tensor():
    """0.2c gate: recover anisotropic D_xx, D_yy from moment growth (dx != dt)."""
    dx, dt = 0.025, 0.01          # dx != dt — discriminates the correct mapping
    D_xx, D_yy = 0.002, 0.0005    # 4:1
    rec_xx, rec_yy = _diffuse_recover_D(D_xx, D_yy, dx, dt)
    # Tolerance 8%: the slow (transverse) axis spreads little -> larger relative
    # moment-measurement error. The WRONG lattice-unit mapping would be ~6x off
    # at this dx!=dt, so 8% still decisively confirms tau = 0.5 + D*dt/(cs2*dx^2).
    assert abs(rec_xx - D_xx) / D_xx < 0.08, (rec_xx, D_xx)
    assert abs(rec_yy - D_yy) / D_yy < 0.08, (rec_yy, D_yy)
    assert rec_xx > rec_yy


def test_mrt_isotropic_recovers_D():
    """Isotropic MRT (D_xx == D_yy) recovers the scalar D on both axes."""
    dx, dt = 0.025, 0.01
    D = 0.001
    rec_xx, rec_yy = _diffuse_recover_D(D, D, dx, dt)
    assert abs(rec_xx - D) / D < 0.05
    assert abs(rec_yy - D) / D < 0.05
    assert abs(rec_xx - rec_yy) / D < 0.05      # symmetric


def test_mrt_guards():
    """collision='mrt' rejects d2q5 and non-canonical weights."""
    model = MHAS13Model(device="cpu")
    kw = dict(Nx=8, Ny=8, dx=0.025, dt=0.01, D=0.001, ionic_model=model)
    with pytest.raises(ValueError):
        LBMSimulation(collision="mrt", lattice="d2q5", **kw)
    with pytest.raises(ValueError):
        LBMSimulation(collision="mrt", lattice="d2q9",
                      weights_mode="uniform_8", **kw)
    # sane MRT construction succeeds
    sim = LBMSimulation(collision="mrt", lattice="d2q9",
                        D=0.001, D_yy=0.00025,
                        Nx=8, Ny=8, dx=0.025, dt=0.01, ionic_model=model)
    assert sim.collision == "mrt"
    assert sim.s_jx < sim.s_jy        # s = 1/tau; larger D_xx -> larger tau -> SMALLER s_jx


def test_lbm_wrapper_routes_anisotropic_to_mrt():
    """0.2b: an anisotropic CardiacMeshData runs via run_lbm (no ValueError)."""
    mesh = create_cardiac_mesh(Lx=0.4, Ly=0.4, dx=0.02, D=0.002, D_yy=0.0005,
                               ionic_model="mhas13", dt=0.01, chi=1.0)
    times, V = run_lbm(mesh, t_end=2.0, save_every=1.0, dt=0.01, device="cpu")
    assert V.shape[1:] == mesh.mask.shape
    assert torch.isfinite(V).all()


def test_lbm_wrapper_isotropic_still_bgk():
    """Isotropic mesh still runs (BGK path unchanged)."""
    mesh = create_cardiac_mesh(Lx=0.4, Ly=0.4, dx=0.02, D=0.001,
                               ionic_model="mhas13", dt=0.01, chi=1.0)
    times, V = run_lbm(mesh, t_end=2.0, save_every=1.0, dt=0.01, device="cpu")
    assert torch.isfinite(V).all()
