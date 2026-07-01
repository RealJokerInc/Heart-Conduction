"""LBM flat-wall boundary modes (boundary_conduction_speedup, productized in Phase 3).

- Every mode is rest-neutral (uniform field is a no-op) + mass-conserving.
- Named-mode / alpha-endpoint equivalences: neumann==hbb, combined(1)==hbb,
  combined(0)==specular_samecell; and the modes genuinely differ on a non-uniform field.
- API validation: D2Q9-only, unknown mode rejected, default is neumann, replay preserves it.
- Oblique anisotropy (D_xy != 0) is rejected by lbm() (#40).
"""
import numpy as np
import pytest
import torch

from cardiac_core._lbm.collision.bgk import bgk_collide
from cardiac_core._lbm.streaming.d2q9 import stream_d2q9
from cardiac_core._lbm.boundary.neumann import apply_neumann_d2q9
from cardiac_core._lbm.boundary.wall_modes import apply_wall_overlay
from cardiac_core._lbm.state import recover_voltage
from cardiac_core._lbm.simulation import LBMSimulation
from cardiac_core.ionic import TTP06Model
from cardiac_core import lbm, create_cardiac_mesh

NX, NY = 21, 15
V_REST = -85.0
DX, DT, D = 0.025, 0.005, 1e-3


def _setup():
    # Use the engine's own rectangular bounce masks (_make_rect_masks), not
    # precompute_bounce_masks (which returns all-False on a full periodic domain).
    sim = LBMSimulation(NX, NY, DX, DT, D, TTP06Model(device='cpu'), lattice='d2q9')
    return sim.w, sim.bounce_masks, sim.omega


def _raw_step(f, V, mode, alpha, w, masks, omega):
    R = torch.zeros(NX, NY, dtype=torch.float64)   # diffusion only (no ionic)
    f = bgk_collide(f, V, R, DT, omega, w)
    f_star = f.clone()
    f = stream_d2q9(f)
    f = apply_neumann_d2q9(f, f_star, masks)
    f = apply_wall_overlay(f, f_star, mode, alpha, NX, NY)
    return f, recover_voltage(f)


@pytest.mark.parametrize("mode,alpha", [
    ('neumann', 1.0), ('hbb', 1.0), ('specular_neighbour', 1.0),
    ('specular_samecell', 0.0), ('combined', 0.0), ('combined', 0.5), ('combined', 1.0),
])
def test_rest_noop_and_mass(mode, alpha):
    """Uniform field stays uniform (rest-neutral) and mass is conserved over 60 steps."""
    w, masks, omega = _setup()
    V = torch.full((NX, NY), V_REST, dtype=torch.float64)
    f = w[:, None, None] * V[None, :, :]
    m0 = f.sum().item()
    dmax = 0.0
    for _ in range(60):
        f, V = _raw_step(f, V, mode, alpha, w, masks, omega)
        dmax = max(dmax, (V - V_REST).abs().max().item())
    assert dmax < 1e-9, f"{mode}(a={alpha}) not rest-neutral: max|V-Vrest|={dmax}"
    assert abs(f.sum().item() - m0) < 1e-9, f"{mode}(a={alpha}) mass drift"


def _run_f(mode, alpha, n=25):
    """Run a left-edge-driven front n steps so the distribution goes off-equilibrium
    (at step 1 from an equilibrium IC the modes can't differ)."""
    w, masks, omega = _setup()
    V = torch.full((NX, NY), V_REST, dtype=torch.float64)
    V[:2, :] = 0.0   # left-edge high → front propagates right with strong gradients
    f = w[:, None, None] * V[None, :, :]
    for _ in range(n):
        f, V = _raw_step(f, V, mode, alpha, w, masks, omega)
    return f


def test_neumann_equals_hbb():
    assert torch.equal(_run_f('neumann', 1.0), _run_f('hbb', 1.0))


def test_combined_alpha1_equals_hbb():
    assert torch.equal(_run_f('combined', 1.0), _run_f('hbb', 1.0))


def test_combined_alpha0_equals_samecell():
    assert torch.equal(_run_f('combined', 0.0), _run_f('specular_samecell', 0.0))


def test_modes_actually_differ():
    assert not torch.equal(_run_f('specular_samecell', 0.0), _run_f('hbb', 1.0))


# ---------------------------------------------------------------- API-level
def _mesh():
    return create_cardiac_mesh(0.4, 0.2, 0.02, D=1e-3, chi=1.0)


def test_lbm_boundary_requires_d2q9():
    with pytest.raises(ValueError, match="d2q9"):
        lbm(_mesh(), boundary='combined', lattice='d2q5')


def test_lbm_boundary_unknown_rejected():
    with pytest.raises(ValueError, match="boundary must be one of"):
        lbm(_mesh(), boundary='bogus', lattice='d2q9')


def test_lbm_default_boundary_is_neumann():
    assert lbm(_mesh(), lattice='d2q9')._engine.boundary == 'neumann'


def test_lbm_combined_selectable_and_replayed():
    sim = lbm(_mesh(), lattice='d2q9', boundary='combined', alpha=0.3)
    assert sim._engine.boundary == 'combined' and sim._engine.alpha == 0.3
    sim.reset()   # build_kwargs must replay boundary + alpha
    assert sim._engine.boundary == 'combined' and sim._engine.alpha == 0.3


def test_lbm_rejects_oblique_Dxy():
    """Oblique anisotropy (D_xy != 0) is out of scope for LBM → ValueError (#40)."""
    m = _mesh()
    m.D_xy = np.full_like(m.D_xy, 1e-4)
    with pytest.raises(ValueError, match="D_xy"):
        lbm(m, lattice='d2q9')
