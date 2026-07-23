"""analysis.fields: fields.integrals — region/line integrals and the theorem cross-checks.

region_integral, net_flux (divergence theorem), circulation (Stokes), winding_number,
conduction_time (the slowness-vs-velocity trap), activated_area, and the isochrone family
(wavefront_length ∮ds, global_curvature ∮κds → Gauss–Bonnet 2π).
"""

import math

import numpy as np
import pytest
import torch

from cardiac_core.fields import integrals as I
from cardiac_core.fields.derivatives import grad, curl
from cardiac_core.fields.lat_fields import bundle_from_lat


def _grid(Nx=40, Ny=36, dx=0.05, dy=0.05):
    x = torch.arange(Nx, dtype=torch.float64) * dx
    y = torch.arange(Ny, dtype=torch.float64) * dy
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return xx, yy, dx, dy


# ======================================================================
# region_integral + net_flux (divergence theorem)
# ======================================================================

class TestRegionAndFlux:
    def test_region_integral_constant(self):
        _, _, dx, dy = _grid()
        f = torch.full((40, 36), 3.0, dtype=torch.float64)
        mask = torch.zeros(40, 36, dtype=torch.bool)
        mask[5:15, 5:15] = True                       # 10x10 nodes
        got = I.region_integral(f, dx, dy, over=mask)
        assert got.item() == pytest.approx(3.0 * 100 * dx * dy)

    def test_region_integral_time_axis(self):
        _, _, dx, dy = _grid()
        f = torch.ones(4, 40, 36, dtype=torch.float64)
        out = I.region_integral(f, dx, dy)
        assert out.shape == (4,)
        assert torch.allclose(out, torch.full((4,), 40 * 36 * dx * dy, dtype=torch.float64))

    def test_net_flux_divergence_theorem(self):
        # F = (x, y) -> div F = 2. Over an INTERIOR sub-region the staggered div is exactly 2, so
        # net_flux == 2 * area to machine precision (the free B5 check).
        xx, yy, dx, dy = _grid()
        F = torch.stack([xx, yy], dim=-1)
        region = torch.zeros(40, 36, dtype=torch.bool)
        region[8:24, 8:22] = True                     # interior block, away from the domain edge
        area = int(region.sum()) * dx * dy
        assert I.net_flux(F, dx, dy, region).item() == pytest.approx(2.0 * area, abs=1e-9)

    def test_net_flux_source_sign(self):
        # A localized outward (source) field -> positive net efflux through an enclosing region.
        xx, yy, dx, dy = _grid()
        cx, cy = 20 * dx, 18 * dy
        F = torch.stack([xx - cx, yy - cy], dim=-1)   # radial source about the centre
        region = torch.zeros(40, 36, dtype=torch.bool)
        region[10:30, 10:26] = True
        assert I.net_flux(F, dx, dy, region).item() > 0     # source inside -> efflux positive


# ======================================================================
# circulation (Stokes), winding, conduction_time (slowness trap)
# ======================================================================

class TestLineIntegrals:
    def test_circulation_stokes(self):
        # v = (-y, x) -> curl v = 2 -> circulation == 2 * area over an interior region.
        xx, yy, dx, dy = _grid()
        v = torch.stack([-yy, xx], dim=-1)
        region = torch.zeros(40, 36, dtype=torch.bool)
        region[8:24, 8:22] = True
        area = int(region.sum()) * dx * dy
        assert I.circulation(v, dx, dy, region).item() == pytest.approx(2.0 * area, abs=1e-9)

    def test_winding_number_spiral(self):
        Nx = Ny = 31
        i, j = torch.meshgrid(torch.arange(Nx, dtype=torch.float64),
                              torch.arange(Ny, dtype=torch.float64), indexing='ij')
        phase = torch.atan2(j - 15.3, i - 15.3)       # one +1 rotor
        assert I.winding_number(phase) == 1

    def test_conduction_time_is_delta_T(self):
        # radial LAT: conduction_time(center-ish, edge) == LAT(edge) - LAT(a) exactly.
        Nx = Ny = 41
        dx = 0.05
        i, j = torch.meshgrid(torch.arange(Nx, dtype=torch.float64),
                              torch.arange(Ny, dtype=torch.float64), indexing='ij')
        r_cm = torch.sqrt((i - 20) ** 2 + (j - 20) ** 2) * dx
        lat = r_cm / 0.05                             # CV = 50 cm/s
        a, b = (18, 20), (34, 20)
        assert I.conduction_time(lat, a, b) == pytest.approx((lat[34, 20] - lat[18, 20]).item())

    def test_conduction_time_slowness_not_velocity(self):
        # THE trap: line-integrating grad(LAT) (slowness) recovers ΔT; the closed-loop integral of a
        # gradient is ~0 (curl-free), while the `velocity` field is NOT curl-free -> not ΔT.
        Nx = Ny = 41
        dx = 0.05
        i, j = torch.meshgrid(torch.arange(Nx, dtype=torch.float64),
                              torch.arange(Ny, dtype=torch.float64), indexing='ij')
        r_cm = torch.sqrt((i - 20) ** 2 + (j - 20) ** 2) * dx
        lat = r_cm / 0.05
        gT = grad(lat, dx, dx)                        # ∇T (slowness)
        loop_gradT = I.region_integral(curl(gT, dx, dx), dx, dx)   # ∮∇T·dl = ∬curl(∇T) ≈ 0
        assert abs(loop_gradT.item()) < 1e-6
        # the velocity field ∇T/|∇T|^2 has nonzero curl -> its loop integral is NOT ~0
        vel = bundle_from_lat(lat, dx, dx)['velocity']
        vel = torch.nan_to_num(vel, nan=0.0)
        loop_vel = I.region_integral(curl(vel, dx, dx), dx, dx)
        assert abs(loop_vel.item()) > abs(loop_gradT.item())


# ======================================================================
# activated area + isochrone family (Gauss–Bonnet)
# ======================================================================

class TestAreaAndIsochrone:
    def test_activated_area_monotone(self):
        # A wave sweeping in +x -> activated area non-decreasing in t.
        Nx, Ny, dx = 30, 10, 0.05
        T = 16                                        # 2*(T-1)+1 = 31 > Nx -> fully activated by the end
        V = torch.full((T, Nx, Ny), -85.0, dtype=torch.float64)
        for t in range(T):
            V[t, : 2 * t + 1, :] = 20.0
        area = I.activated_area(V, dx, dx, threshold=-40.0)
        assert area.shape == (T,)
        assert torch.all(area[1:] >= area[:-1] - 1e-12)
        assert area[-1].item() == pytest.approx(Nx * Ny * dx * dx)   # fully activated at the end

    def test_wavefront_length_circle(self):
        # radial LAT: the T=level isochrone is a circle of radius r=level*CV -> perimeter 2πr.
        Nx = Ny = 61
        dx = 0.05
        i, j = torch.meshgrid(torch.arange(Nx, dtype=torch.float64),
                              torch.arange(Ny, dtype=torch.float64), indexing='ij')
        r_cm = torch.sqrt((i - 30) ** 2 + (j - 30) ** 2) * dx
        lat = r_cm / 0.05                            # CV=50 cm/s ; LAT=level -> r=level*0.05
        level = 8.0                                  # r = 0.4 cm
        L = I.wavefront_length(lat, level, dx, dx)
        assert L == pytest.approx(2 * math.pi * (level * 0.05), rel=0.03)

    def test_global_curvature_gauss_bonnet(self):
        # A single convex closed isochrone -> net turning ∮κ ds ≈ 2π.
        Nx = Ny = 61
        dx = 0.05
        i, j = torch.meshgrid(torch.arange(Nx, dtype=torch.float64),
                              torch.arange(Ny, dtype=torch.float64), indexing='ij')
        r_cm = torch.sqrt((i - 30) ** 2 + (j - 30) ** 2) * dx
        lat = r_cm / 0.05
        gc = I.global_curvature(lat, 8.0, dx, dx)
        assert gc == pytest.approx(2 * math.pi, rel=0.05)
