"""Phase 0 Step 0.1 — cardiac_core tuning seam.

Per-axis mesh diffusion (`create_cardiac_mesh(D_yy=...)`) + IonicModel-instance
pass-through (the LBM factory no longer `.lower()`-crashes on an instance, and
mono/bidomain already accept instances + per-axis D).

Covers PLAN Phase 0 Step 0.1 (Research/Active/ionic_model_optimization/PLAN.md).

NOTE (convention): `create_cardiac_mesh(D=...)` treats `D` as RAW; the
membrane-effective diffusivity is `D/(χ·Cm)` in every engine. To pass an
already-effective `D` (cm²/ms), set **chi=1.0** so `D/(χ·Cm)=D` (as the chip
meshes do). See Research/Active/engine_consolidation (Audit #2/#8/#21).
"""
import numpy as np
import torch

from cardiac_core import create_cardiac_mesh, run_monodomain, run_lbm
from cardiac_core.ionic import MHAS13Model

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _cv(times, V, p1, p2, dx, threshold=-20.0):
    """CV (cm/ms) between two grid points from a (n, Nx, Ny) V history."""
    (i1, j1), (i2, j2) = p1, p2
    t1 = t2 = None
    for k in range(V.shape[0]):
        tk = times[k].item() if torch.is_tensor(times) else times[k]
        if t1 is None and V[k, i1, j1].item() > threshold:
            t1 = tk
        if t2 is None and V[k, i2, j2].item() > threshold:
            t2 = tk
    if t1 is None or t2 is None or t2 <= t1:
        return float("nan")
    dist = (((i2 - i1) ** 2 + (j2 - j1) ** 2) ** 0.5) * dx
    return dist / (t2 - t1)


def test_create_mesh_anisotropic():
    """create_cardiac_mesh(D_yy=...) yields D_xx != D_yy; default stays isotropic."""
    iso = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
    assert np.allclose(iso.D_xx, iso.D_yy)

    aniso = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05, D=0.001, D_yy=0.00025, chi=1.0)
    assert np.allclose(aniso.D_xx, 0.001)
    assert np.allclose(aniso.D_yy, 0.00025)
    assert np.allclose(aniso.D_xy, 0.0)
    assert not np.allclose(aniso.D_xx, aniso.D_yy)


def test_lbm_accepts_instance():
    """lbm()/run_lbm accept a pre-built IonicModel instance (no .lower() crash)."""
    model = MHAS13Model(device="cpu")
    mesh = create_cardiac_mesh(Lx=0.4, Ly=0.1, dx=0.02, D=0.001,
                               ionic_model="mhas13", dt=0.02, chi=1.0)
    times, V = run_lbm(mesh, t_end=2.0, save_every=1.0,
                       ionic_model=model, dt=0.02, device="cpu")
    assert V.shape[1:] == mesh.mask.shape
    assert torch.isfinite(V).all()


def test_scaled_instance_changes_output():
    """A g_Kr-scaled instance changes the mono V trajectory vs baseline."""
    mesh = create_cardiac_mesh(Lx=0.4, Ly=0.1, dx=0.02, D=0.001, chi=1.0,
                               ionic_model="mhas13", dt=0.02, stim_amplitude=-52.0)
    base = MHAS13Model(device=DEV)
    scaled = MHAS13Model(device=DEV)
    scaled.params.g_Kr = scaled.params.g_Kr * 0.5   # apply_scaling-style mutation

    _, V0 = run_monodomain(mesh, t_end=30.0, save_every=2.0,
                           ionic_model=base, dt=0.02, device=DEV)
    _, V1 = run_monodomain(mesh, t_end=30.0, save_every=2.0,
                           ionic_model=scaled, dt=0.02, device=DEV)
    assert not torch.allclose(V0, V1)


def test_mono_mesh_anisotropy_changes_cv():
    """Mono honors per-axis mesh D: CV along x (D_xx) > CV along y (D_yy).

    Effective-D mesh -> chi=1.0; fine dx + narrow strong stim for a clean front.
    """
    dx = 0.01
    Dxx, Dyy = 0.001, 0.00025          # 4:1 in D -> ~2:1 in CV (CV ∝ √D)
    kw = dict(Lx=0.8, Ly=0.8, dx=dx, D=Dxx, D_yy=Dyy, chi=1.0, Cm=1.0,
              ionic_model="ttp06", dt=0.02, stim_width=0.05, stim_amplitude=-52.0)

    mesh_x = create_cardiac_mesh(**kw)
    Nx, Ny = mesh_x.mask.shape
    tx, Vx = run_monodomain(mesh_x, t_end=35.0, save_every=0.5, device=DEV)
    cv_x = _cv(tx, Vx, (Nx // 4, Ny // 2), (3 * Nx // 4, Ny // 2), dx)

    mesh_y = create_cardiac_mesh(**kw)
    bottom = np.zeros(mesh_y.mask.shape, dtype=bool)
    bottom[:, :5] = True                                # bottom-edge stim -> +y front
    mesh_y.stimuli[0]["mask"] = bottom & mesh_y.mask
    ty, Vy = run_monodomain(mesh_y, t_end=35.0, save_every=0.5, device=DEV)
    cv_y = _cv(ty, Vy, (Nx // 2, Ny // 4), (Nx // 2, 3 * Ny // 4), dx)

    assert np.isfinite(cv_x) and np.isfinite(cv_y), (cv_x, cv_y)
    assert cv_x > cv_y, (cv_x, cv_y)
    assert 1.5 < cv_x / cv_y < 2.6, cv_x / cv_y     # ~sqrt(Dxx/Dyy)=2, generous
