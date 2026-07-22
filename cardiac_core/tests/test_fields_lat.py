"""Phase-4 (analysis.fields): LAT-based fields — velocity/direction/speed/curvature/vorticity,
divergence gating, and the shared winding-loop-sum primitive.

Unit-tests the geometry on synthetic planar/radial LAT (known |∇T| and curvature), then the full
r.fields.* stack on a real sim; Step 4.4 tests the winding primitive + the phase_singularities
refactor onto it.
"""

import math

import numpy as np
import pytest
import torch

from cardiac_core import monodomain, Grid, ConductivityConfig
from cardiac_core import analysis
from cardiac_core.fields import VectorField
from cardiac_core.fields.derivatives import winding_loop_sum
from cardiac_core.fields.lat_fields import bayly_gradient, bundle_from_lat


def _cond():
    return ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)


# ======================================================================
# Step 4.1 — Bayly / Savitzky-Golay gradient + quality
# ======================================================================

class TestBaylyGradient:
    def test_planar_gradient_and_quality(self):
        Nx, Ny, dx = 24, 18, 0.05
        i = torch.arange(Nx, dtype=torch.float64).reshape(Nx, 1)
        lat = (i * 1.0).expand(Nx, Ny).contiguous()      # 1 ms per x-index -> |dT/dx| = 20 ms/cm
        Tx, Ty, quality, ok = bayly_gradient(lat, dx, dx)
        w = 3
        assert torch.allclose(Tx[w:-w, w:-w], torch.full_like(Tx[w:-w, w:-w], 20.0), rtol=1e-6)
        assert torch.allclose(Ty[w:-w, w:-w], torch.zeros_like(Ty[w:-w, w:-w]), atol=1e-9)
        assert torch.allclose(quality[w:-w, w:-w], torch.zeros_like(quality[w:-w, w:-w]), atol=1e-9)
        assert ok[w:-w, w:-w].all()

    def test_nan_no_bleed(self):
        Nx, Ny, dx = 24, 18, 0.05
        i = torch.arange(Nx, dtype=torch.float64).reshape(Nx, 1)
        lat = (i * 1.0).expand(Nx, Ny).contiguous()
        lat[10:14, 8:12] = float('nan')                  # a non-activating block
        Tx, Ty, quality, ok = bayly_gradient(lat, dx, dx)
        assert not ok[11, 10].item()                     # inside the block -> not usable
        assert not ok[9, 10].item()                      # 1-node halo (window touches NaN) -> gated
        assert ok[3, 3].item()                           # far away -> fine, no bleed
        assert torch.isfinite(Tx[3, 3])


# ======================================================================
# Step 4.2 — velocity/direction/speed/curvature + gating (synthetic radial LAT)
# ======================================================================

class TestLatGeometry:
    def _radial_lat(self, Nx=41, Ny=41, dx=0.05, cv_cms=50.0):
        cx, cy = 20, 20
        i, j = torch.meshgrid(torch.arange(Nx, dtype=torch.float64),
                              torch.arange(Ny, dtype=torch.float64), indexing='ij')
        r_cm = torch.sqrt((i - cx) ** 2 + (j - cy) ** 2) * dx
        lat = r_cm / (cv_cms / 1000.0)                   # ms  (CV in cm/ms = cv_cms/1000)
        return lat, dx, (cx, cy)

    def test_speed_matches_known_cv(self):
        lat, dx, _ = self._radial_lat(cv_cms=50.0)
        b = bundle_from_lat(lat, dx, dx)
        assert b['speed'][30, 20].item() == pytest.approx(50.0, rel=0.03)   # < 3%

    def test_direction_points_outward(self):
        lat, dx, (cx, cy) = self._radial_lat()
        b = bundle_from_lat(lat, dx, dx)
        d = b['direction']
        assert d[30, 20, 0].item() > 0.95      # east of centre -> +x
        assert abs(d[30, 20, 1].item()) < 0.1
        assert d[20, 30, 1].item() > 0.95      # north of centre -> +y

    def test_curvature_of_expanding_front(self):
        # κ = ∇·n̂ = 1/r for a radial front. At r=0.5 cm -> κ=2 /cm.
        lat, dx, _ = self._radial_lat()
        b = bundle_from_lat(lat, dx, dx)
        r_cm = 10 * dx                          # node (30,20) is 10 cells from centre
        assert b['curvature'][30, 20].item() == pytest.approx(1.0 / r_cm, rel=0.15)

    def test_collision_is_gated_not_blown_up(self):
        # Two counter-propagating planar fronts meet at the middle -> |∇T|->0 there -> gated NaN.
        # Nx odd so the collision peak sits EXACTLY on node mid (a symmetric SG window -> Tx=0).
        Nx, Ny, dx = 41, 12, 0.05
        i = torch.arange(Nx, dtype=torch.float64).reshape(Nx, 1)
        mid = (Nx - 1) // 2                              # = 20, integer
        lat = (float(mid) - (i - mid).abs()).expand(Nx, Ny).contiguous()   # peak on the collision line
        b = bundle_from_lat(lat, dx, dx)
        assert b['mask'][mid, 6].item()                 # collision flagged
        assert math.isnan(b['speed'][mid, 6].item())    # not 1/|∇T|=inf
        assert torch.isfinite(b['speed'][5, 6])         # away from the collision, fine

    def test_planar_curvature_near_zero(self):
        Nx, Ny, dx = 24, 18, 0.05
        i = torch.arange(Nx, dtype=torch.float64).reshape(Nx, 1)
        lat = (i * 1.0).expand(Nx, Ny).contiguous()
        b = bundle_from_lat(lat, dx, dx)
        w = 3
        assert torch.allclose(b['curvature'][w:-w, w:-w],
                              torch.zeros_like(b['curvature'][w:-w, w:-w]), atol=1e-6)


# ======================================================================
# Step 4.2 (integration) — r.fields.* on a real sim
# ======================================================================

class TestLatFieldsOnSim:
    def test_speed_matches_cv_hook(self):
        g = Grid(80, 12, 0.02)
        stim = {'region': (lambda x, y: x < 0.04), 'start_time': 1.0,
                'duration': 2.0, 'amplitude': -52.0}
        r = monodomain(g, 'ttp06', _cond(), stim).run(26.0, save_every=0.5)
        sp = r.fields.speed
        assert sp.shape == (80, 12)
        assert isinstance(r.fields.velocity, VectorField)
        # mid-tissue speed agrees with the two-point CV hook (same canonical LAT), ~15%
        cv = r.cv(x1=15, x2=45, y=6)
        assert not math.isnan(cv)
        mid = sp[18:42, 6]
        mid = mid[torch.isfinite(mid)]
        assert mid.numel() > 0
        assert mid.median().item() == pytest.approx(cv, rel=0.15)

    def test_lat_fields_shapes_and_direction(self):
        g = Grid(60, 12, 0.02)
        stim = {'region': (lambda x, y: x < 0.04), 'start_time': 1.0,
                'duration': 2.0, 'amplitude': -52.0}
        r = monodomain(g, 'ttp06', _cond(), stim).run(16.0, save_every=0.5)
        assert r.fields.curvature.shape == (60, 12)
        assert r.fields.vorticity.shape == (60, 12)
        assert r.fields.quality.shape == (60, 12)
        assert r.fields.mask.dtype == torch.bool
        d = r.fields.direction.x
        d = d[torch.isfinite(d)]
        assert d.mean().item() > 0.5           # propagation is mostly +x


# ======================================================================
# Step 4.4 — shared winding-loop-sum primitive + phase_singularities refactor
# ======================================================================

class TestWinding:
    def _spiral_phase(self, Nx=31, Ny=31):
        cx, cy = 15.3, 15.3   # off-node so no plaquette straddles the exact singularity
        i, j = torch.meshgrid(torch.arange(Nx, dtype=torch.float64),
                              torch.arange(Ny, dtype=torch.float64), indexing='ij')
        return torch.atan2(j - cy, i - cx)     # a +1 winding around the centre

    def test_primitive_charge_pm1(self):
        phase = self._spiral_phase()
        loop = winding_loop_sum(phase)
        charge = loop / (2 * math.pi)
        assert charge.abs().max().item() == pytest.approx(1.0, abs=0.05)   # exactly one ±1 tip
        assert (charge.abs() > 0.5).sum().item() == 1

    def test_phase_singularities_matches_reference(self):
        # The refactored analysis.phase_singularities (now on the shared atan2 primitive) must match
        # the historical modulo-wrap computation to a tight atol (atan2 re-associates at ULP).
        phase = self._spiral_phase()
        got = analysis.phase_singularities(phase)

        def _wrap(d):
            return (d + torch.pi) % (2 * torch.pi) - torch.pi
        d1 = _wrap(phase[1:, :-1] - phase[:-1, :-1])
        d2 = _wrap(phase[1:, 1:] - phase[1:, :-1])
        d3 = _wrap(phase[:-1, 1:] - phase[1:, 1:])
        d4 = _wrap(phase[:-1, :-1] - phase[:-1, 1:])
        ref = (d1 + d2 + d3 + d4) / (2 * torch.pi)
        assert torch.allclose(got, ref, atol=1e-9)
