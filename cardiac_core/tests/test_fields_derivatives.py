"""Phase-2 (analysis.fields): the operator toolkit grad/div/curl/laplacian.

Steps 2.1 (collocated central grad/div/curl + ghost-mirror boundary) and 2.2 (staggered
div=-grad* -> compact 5-point laplacian). Pure-tensor tests; no simulation needed.
"""

import math

import pytest
import torch

from cardiac_core.fields.derivatives import grad, div, curl, laplacian


def _grid(Nx=12, Ny=10, dx=0.05, dy=0.05, device='cpu'):
    x = torch.linspace(0, dx * (Nx - 1), Nx, dtype=torch.float64, device=device)
    y = torch.linspace(0, dy * (Ny - 1), Ny, dtype=torch.float64, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return xx, yy, dx, dy


def _interior(a, w=2):
    return a[w:-w, w:-w]


# ======================================================================
# Step 2.1 — grad / div / curl (collocated central) + boundary
# ======================================================================

class TestFirstOrder:
    def test_grad_linear_exact(self):
        xx, yy, dx, dy = _grid()
        f = 3.0 * xx + 2.0 * yy
        g = grad(f, dx, dy)
        assert g.shape == (12, 10, 2)
        assert torch.allclose(_interior(g[..., 0]), torch.full_like(_interior(g[..., 0]), 3.0), atol=1e-10)
        assert torch.allclose(_interior(g[..., 1]), torch.full_like(_interior(g[..., 1]), 2.0), atol=1e-10)

    def test_grad_batched_time_axis(self):
        xx, yy, dx, dy = _grid()
        f = torch.stack([3.0 * xx + 2.0 * yy, xx * 0 + 5.0])   # (2, Nx, Ny)
        g = grad(f, dx, dy)
        assert g.shape == (2, 12, 10, 2)
        assert torch.allclose(_interior(g[0, ..., 0]), torch.full_like(_interior(g[0, ..., 0]), 3.0), atol=1e-10)
        assert torch.allclose(_interior(g[1, ..., 0]), torch.zeros_like(_interior(g[1, ..., 0])), atol=1e-10)

    def test_curl_of_gradient_zero(self):
        torch.manual_seed(1)
        xx, yy, dx, dy = _grid(16, 14)
        f = torch.sin(4 * xx) * torch.cos(3 * yy) + 0.5 * xx * yy
        c = curl(grad(f, dx, dy), dx, dy)
        assert torch.allclose(_interior(c), torch.zeros_like(_interior(c)), atol=1e-9)

    def test_div_of_known_field(self):
        # F = (x, y) -> div = 2 everywhere (interior)
        xx, yy, dx, dy = _grid()
        F = torch.stack([xx, yy], dim=-1)
        d = div(F, dx, dy)
        assert torch.allclose(_interior(d), torch.full_like(_interior(d), 2.0), atol=1e-10)

    def test_curl_of_rotational_field(self):
        # F = (-y, x) -> curl = dFy/dx - dFx/dy = 1 - (-1) = 2 (interior)
        xx, yy, dx, dy = _grid()
        F = torch.stack([-yy, xx], dim=-1)
        c = curl(F, dx, dy)
        assert torch.allclose(_interior(c), torch.full_like(_interior(c), 2.0), atol=1e-10)

    def test_boundary_is_noflux(self):
        # A field symmetric about the left edge node has zero true normal derivative there.
        # face_mirror (whole-sample) reproduces 0; numpy one-sided would be nonzero.
        xx, yy, dx, dy = _grid()
        f = (xx - xx[0, 0]) ** 2          # symmetric about x=0 -> dV/dx|_edge = 0
        g = grad(f, dx, dy)
        assert torch.allclose(g[0, :, 0], torch.zeros_like(g[0, :, 0]), atol=1e-12)   # left edge normal deriv
        assert torch.allclose(g[-1, :, 0], torch.zeros_like(g[-1, :, 0]), atol=1e-12)  # right edge
        # a one-sided forward difference would NOT be zero:
        onesided = (f[1, :] - f[0, :]) / dx
        assert onesided.abs().max() > 1e-6

    def test_mask_noflux_no_blowup(self):
        # An interior hole: gradient stays finite at the rim; masked nodes are NaN.
        xx, yy, dx, dy = _grid(14, 14)
        f = torch.sin(3 * xx) * torch.cos(2 * yy)
        mask = torch.ones(14, 14, dtype=torch.bool)
        mask[6:9, 6:9] = False
        g = grad(f, dx, dy, mask=mask)
        assert torch.isnan(g[7, 7, 0])                       # masked centre -> NaN
        live = g[mask]
        assert torch.isfinite(live).all()                    # no blow-up on the live domain
        d = div(grad(f, dx, dy, mask=mask), dx, dy, mask=mask)
        assert torch.isfinite(d[mask]).all()

    def test_invalid_boundary_mode_raises(self):
        xx, _, dx, dy = _grid()
        with pytest.raises(ValueError):
            grad(xx, dx, dy, boundary_mode='wrap')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
    def test_device_preserved(self):
        xx, yy, dx, dy = _grid(device='cuda')
        g = grad(3 * xx + 2 * yy, dx, dy)
        assert g.device.type == 'cuda'
        assert torch.allclose(_interior(g[..., 0]), torch.full_like(_interior(g[..., 0]), 3.0), atol=1e-10)


# ======================================================================
# Step 2.2 — staggered laplacian (compact 5-point, not wide)
# ======================================================================

class TestLaplacian:
    def test_laplacian_is_compact_5pt(self):
        torch.manual_seed(2)
        xx, yy, dx, dy = _grid(15, 13, 0.05, 0.05)
        f = torch.sin(3 * xx) * torch.cos(2 * yy) + 0.3 * xx ** 2
        lap = laplacian(f, dx, dy)
        # reference compact 5-point on the interior
        ref = ((f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx ** 2
               + (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dy ** 2)
        assert torch.allclose(lap[1:-1, 1:-1], ref, atol=1e-9)

    def test_laplacian_not_wide(self):
        # On a checkerboard, the compact 5-point is large; the collocated div(grad) WIDE stencil
        # (couples i±2, same parity) is ~0. They must differ -> proves the staggered fix.
        Nx, Ny, dx = 12, 12, 1.0
        ii, jj = torch.meshgrid(torch.arange(Nx), torch.arange(Ny), indexing='ij')
        checker = ((ii + jj) % 2).double() * 2 - 1      # +1/-1 checkerboard
        lap_compact = laplacian(checker, dx, dx)
        lap_wide = div(grad(checker, dx, dx), dx, dx)   # collocated div(grad) = wide stencil
        assert _interior(lap_compact).abs().min() > 1.0          # compact sees the oscillation
        assert _interior(lap_wide).abs().max() < 1e-9            # wide stencil is blind to it

    def test_laplacian_analytic_neumann(self):
        # lap(x^2 + y^2) = 4 in the interior (Neumann/face_mirror box).
        xx, yy, dx, dy = _grid(20, 18, 0.05, 0.05)
        f = xx ** 2 + yy ** 2
        lap = laplacian(f, dx, dy)
        assert torch.allclose(_interior(lap), torch.full_like(_interior(lap), 4.0), atol=1e-9)

    def test_divergence_theorem_exact(self):
        # Sum of the compact laplacian over the WHOLE no-flux box == 0 (all boundary face flux 0)
        # to machine precision (B5 telescoping) — the free exact self-check.
        torch.manual_seed(3)
        xx, yy, dx, dy = _grid(16, 16, 0.05, 0.05)
        f = torch.sin(2 * xx) * torch.cos(3 * yy) + 0.2 * xx ** 2 - 0.1 * yy ** 2
        lap = laplacian(f, dx, dy)
        assert abs(lap.sum().item() * dx * dy) < 1e-10

    def test_laplacian_mask_nan(self):
        xx, yy, dx, dy = _grid(14, 14)
        f = torch.sin(3 * xx) * torch.cos(2 * yy)
        mask = torch.ones(14, 14, dtype=torch.bool)
        mask[6:9, 6:9] = False
        lap = laplacian(f, dx, dy, mask=mask)
        assert torch.isnan(lap[7, 7])
        assert torch.isfinite(lap[mask]).all()
        # masked no-flux: sum over the live domain still telescopes to ~0
        assert abs(lap[mask].sum().item() * dx * dy) < 1e-9

    def test_laplacian_matches_fdm5pt_solver(self):
        # The load-bearing gate: my staggered laplacian must equal the engine's OWN assembled
        # FDM 5-point diffusion operator (cardinal4, iso-uniform), NOT a hand-rolled stencil.
        # apply_diffusion(V) = (1/(chi*Cm)) div(D grad V) = div(D_eff grad V) = D_eff * lap(V).
        from cardiac_core import monodomain, Grid, ConductivityConfig
        g = Grid(24, 18, 0.05)
        cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
        stim = {'region': (lambda x, y: x < 0.06), 'start_time': 1.0,
                'duration': 2.0, 'amplitude': -52.0}
        sim = monodomain(g, 'ttp06', cond, stim)
        sp = sim._engine.spatial
        assert getattr(sp, '_is_iso_uniform', False)          # FDM-5pt path
        r = sim.run(5.0, save_every=1.0)
        Vm = r.Vm[-1]                                          # a snapshot with a wavefront
        D_eff = float(r.conductivity.D_eff.reshape(-1)[0])     # uniform scalar

        engine_diff = sp.grid.flat_to_grid(sp.apply_diffusion(sp.grid.grid_to_flat(Vm)))
        mine = D_eff * laplacian(Vm, r.dx, r.dy)
        # interior match to the engine's own operator (rel < 1e-6)
        ei, mi = _interior(engine_diff), _interior(mine)
        assert torch.allclose(mi, ei, rtol=1e-6, atol=1e-8), \
            f"max rel diff {((mi - ei).abs() / (ei.abs() + 1e-12)).max().item():.2e}"
