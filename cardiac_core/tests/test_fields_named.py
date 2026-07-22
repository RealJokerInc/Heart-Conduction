"""Phase-3 (analysis.fields): the Fields accessor + VectorField + Vm/φ_e named fields.

Steps 3.1 (accessor + wrapper + lazy cache), 3.2 (voltage_gradient/voltage_flux/source_sink),
3.3 (electric_field/current_flux). The load-bearing gate is source_sink == the engine's OWN
FDM diffusion operator (uniform + a masked scar), which FAILS if raw D (not D_eff) is used.
"""

import numpy as np
import pytest
import torch

from cardiac_core import monodomain, bidomain, Grid, ConductivityConfig, create_cardiac_mesh
from cardiac_core.fields import VectorField
from cardiac_core.fields.derivatives import div, grad, diffusion_term


def _cond():
    return ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)


def _stim():
    return {'region': (lambda x, y: x < 0.06), 'start_time': 1.0,
            'duration': 2.0, 'amplitude': -52.0}


def _mono(Nx=24, Ny=18, mask=None):
    g = Grid(Nx, Ny, 0.05, mask=mask)
    return monodomain(g, 'ttp06', _cond(), _stim())


# ======================================================================
# Step 3.1 — accessor + VectorField + lazy cache
# ======================================================================

class TestAccessorAndCache:
    def test_fields_lazy_cache(self):
        r = _mono().run(4.0, save_every=1.0)
        a = r.fields.source_sink
        b = r.fields.source_sink
        assert a is b                       # per-name cached (same object)
        assert r.fields is r.fields         # accessor memoized (cached_property)

    def test_fresh_run_fresh_cache(self):
        sim = _mono()
        r1 = sim.run(4.0, save_every=1.0)
        r2 = sim.run(4.0, save_every=1.0)
        assert r1.fields is not r2.fields   # a new result -> a fresh accessor (mutate the sim, not the result)

    def test_vectorfield_wrapper(self):
        v = torch.tensor([[3.0, 4.0]], dtype=torch.float64)   # (1,2)
        vf = VectorField(v)
        assert torch.allclose(vf.x, torch.tensor([3.0], dtype=torch.float64))
        assert torch.allclose(vf.y, torch.tensor([4.0], dtype=torch.float64))
        assert vf.magnitude.item() == pytest.approx(5.0)
        assert vf.angle.item() == pytest.approx(np.arctan2(4.0, 3.0))
        assert vf.components is v


# ======================================================================
# Step 3.2 — voltage_gradient / voltage_flux / source_sink
# ======================================================================

class TestVmFields:
    def test_shapes(self):
        r = _mono().run(4.0, save_every=1.0)
        T = r.times.shape[0]
        assert r.fields.voltage_gradient.components.shape == (T, 24, 18, 2)
        assert r.fields.voltage_flux.components.shape == (T, 24, 18, 2)
        assert r.fields.source_sink.shape == (T, 24, 18)

    def test_source_sink_matches_fdm5pt_diffusion(self):
        # THE gate: source_sink == the engine's OWN assembled FDM diffusion operator, using D_eff.
        # (Raw D would be off by chi*Cm=1400 and fail hard.)
        sim = _mono()
        sp = sim._engine.spatial
        assert getattr(sp, '_is_iso_uniform', False)
        r = sim.run(5.0, save_every=1.0)
        Vm = r.Vm[-1]
        engine = sp.grid.flat_to_grid(sp.apply_diffusion(sp.grid.grid_to_flat(Vm)))
        mine = r.fields.source_sink[-1]
        w = 2
        ei, mi = engine[w:-w, w:-w], mine[w:-w, w:-w]
        assert torch.allclose(mi, ei, rtol=1e-6, atol=1e-8), \
            f"max rel {(mi - ei).abs().max().item():.2e}"

    def test_source_sink_matches_solver_masked_scar(self):
        # The flagship consumers (source_sink_mismatch, fig4c) run on MASKED geometry — the
        # no-flux hole rim is exactly where a wrong stencil would diverge from the engine.
        mask = np.ones((24, 18), dtype=bool)
        mask[10:14, 8:12] = False
        sim = _mono(mask=mask)
        sp = sim._engine.spatial
        r = sim.run(6.0, save_every=1.0)
        Vm = r.Vm[-1]
        engine = sp.grid.flat_to_grid(sp.apply_diffusion(sp.grid.grid_to_flat(Vm)))
        mine = r.fields.source_sink[-1]
        both = torch.isfinite(engine) & torch.isfinite(mine)
        both[:2, :] = both[-2:, :] = both[:, :2] = both[:, -2:] = False   # drop the outer ring
        assert both.sum() > 0
        assert torch.allclose(mine[both], engine[both], rtol=1e-6, atol=1e-8)

    def test_div_voltage_flux_consistent_with_source_sink(self):
        # A smooth analytic field: collocated div(D*grad V) and the staggered source_sink are the
        # SAME operator at two stencils -> agree to O(h^2) in the interior (NOT 1e-10 — the plan's
        # "identity" is the staggered self-consistency, checked in test_source_sink_matches_*).
        x = torch.linspace(0, 1, 40, dtype=torch.float64)
        y = torch.linspace(0, 1, 36, dtype=torch.float64)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        V = torch.sin(2 * xx) * torch.cos(2 * yy)
        dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); D = 1.3
        ss = diffusion_term(V, D, dx, dy)
        collo = div(D * grad(V, dx, dy), dx, dy)
        w = 3
        a, b = ss[w:-w, w:-w], collo[w:-w, w:-w]
        assert torch.allclose(a, b, rtol=3e-2, atol=1e-3)

    def test_source_sink_bidomain_raises(self):
        r = bidomain(Grid(16, 12, 0.05), 'ttp06', _cond(), _stim()).run(2.0, save_every=1.0)
        with pytest.raises(ValueError, match="monodomain"):
            _ = r.fields.source_sink

    def test_source_sink_anisotropic_raises(self):
        g = Grid(16, 12, 0.05)
        cond = ConductivityConfig.anisotropic(3.0, 1.0, fiber_angle=0.4, chi=1400.0)
        r = monodomain(g, 'ttp06', cond, _stim()).run(2.0, save_every=1.0)
        assert r.conductivity.is_anisotropic
        with pytest.raises(ValueError, match="[Aa]nisotropic"):
            _ = r.fields.source_sink


# ======================================================================
# Step 3.3 — electric_field / current_flux (bidomain)
# ======================================================================

class TestBidomainFields:
    def test_efield_monodomain_raises(self):
        r = _mono().run(2.0, save_every=1.0)
        with pytest.raises(ValueError, match="bidomain"):
            _ = r.fields.electric_field

    def test_current_flux_bidomain(self):
        r = bidomain(Grid(16, 12, 0.05), 'ttp06', _cond(), _stim()).run(3.0, save_every=1.0)
        cf = r.fields.current_flux
        T = r.times.shape[0]
        assert cf.components.shape == (T, 16, 12, 2)
        ef = r.fields.electric_field
        assert ef.components.shape == (T, 16, 12, 2)
        # current_flux = -sigma_e * grad(phi_e); electric_field = -grad(phi_e); sigma_e>0 so they
        # point the same way (positive dot product where the field is non-trivial).
        dot = (cf.x * ef.x + cf.y * ef.y)
        assert torch.nansum(dot) >= 0


# ======================================================================
# operator toolkit bound to the result
# ======================================================================

def test_fields_derivatives_toolkit():
    r = _mono(20, 16).run(4.0, save_every=1.0)
    D = r.fields.derivatives
    g = D.grad(r.Vm)
    assert isinstance(g, VectorField)
    assert g.components.shape == (r.times.shape[0], 20, 16, 2)
    lap = D.laplacian(r.Vm)
    assert lap.shape == (r.times.shape[0], 20, 16)
    # div(grad) via the toolkit returns a scalar of the right shape
    assert D.div(g).shape == (r.times.shape[0], 20, 16)
    assert D.curl(g).shape == (r.times.shape[0], 20, 16)
