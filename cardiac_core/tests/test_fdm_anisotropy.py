"""Regression tests for the monodomain FDM anisotropic cross-derivative (Audit #4).

The cardinal4 stencil discretizes ``div(D·∇V) = Dxx·V_xx + 2·Dxy·V_xy + Dyy·V_yy``.
The cross term must use ``cxy = 1/(2·dx·dy)`` (the factor 2 from 2·Dxy) and the diagonal
signs of the validated bidomain builder (NE +, NW −, SE −, SW +).

Before the fix the mono builder used ``cxy = 1/(4·dx·dy)`` (half) and the opposite signs,
so for a bilinear field ``V = x·y`` (∂²V/∂x∂y = 1) the interior gave ``−Dxy`` instead of
the correct ``+2·Dxy``. Isotropic (Dxy = 0) operators are unchanged (no diagonal terms).
"""
import numpy as np
import torch

from cardiac_core.mesh.structured import StructuredGrid
from cardiac_core._monodomain import FDMDiscretization

NX = NY = 12
DX = 0.1


def _build(dxy: float):
    """FDM with constant isotropic Dxx=Dyy=1 and constant cross term Dxy, chi=Cm=1."""
    grid = StructuredGrid.create_rectangle(DX * (NX - 1), DX * (NY - 1), NX, NY, device='cpu')
    Dxx = torch.ones(NX, NY, dtype=torch.float64)
    Dyy = torch.ones(NX, NY, dtype=torch.float64)
    Dxy = torch.full((NX, NY), dxy, dtype=torch.float64)
    return FDMDiscretization(grid, D_field=(Dxx, Dxy, Dyy), chi=1.0, Cm=1.0)


def _idx(i, j):
    return i * NY + j


def test_cross_derivative_bilinear():
    """V = x·y (V_xy = 1) → interior L·V = 2·Dxy (was −Dxy before the fix)."""
    dxy = 0.3
    fdm = _build(dxy)
    x = np.arange(NX) * DX
    y = np.arange(NY) * DX
    V = (x[:, None] * y[None, :]).astype(np.float64)   # V[i,j] = x_i·y_j
    V_flat = torch.tensor(V.reshape(-1), dtype=torch.float64)
    # apply_diffusion returns L·V/(chi·Cm); chi=Cm=1 here.
    out = fdm.apply_diffusion(V_flat).reshape(NX, NY).numpy()
    interior = out[1:-1, 1:-1]
    assert np.allclose(interior, 2.0 * dxy, atol=1e-10), (
        f"interior L·V should be 2·Dxy={2 * dxy}, got {interior.min()}..{interior.max()}"
    )


def test_cross_derivative_coefficients():
    """Interior off-diagonals pin the bidomain sign/magnitude convention exactly."""
    dxy = 0.3
    fdm = _build(dxy)
    L = fdm.L.to_dense().numpy()
    cxy = 1.0 / (2.0 * DX * DX)   # dy == dx
    k = _idx(5, 5)
    assert np.isclose(L[k, _idx(6, 6)], +dxy * cxy, atol=1e-12)   # NE +
    assert np.isclose(L[k, _idx(4, 6)], -dxy * cxy, atol=1e-12)   # NW −
    assert np.isclose(L[k, _idx(6, 4)], -dxy * cxy, atol=1e-12)   # SE −
    assert np.isclose(L[k, _idx(4, 4)], +dxy * cxy, atol=1e-12)   # SW +


def test_isotropic_unchanged():
    """Dxy = 0 → no diagonal cross-terms (isotropic operator identical to pre-fix)."""
    fdm = _build(0.0)
    L = fdm.L.to_dense().numpy()
    k = _idx(5, 5)
    for di, dj in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        assert L[k, _idx(5 + di, 5 + dj)] == 0.0
