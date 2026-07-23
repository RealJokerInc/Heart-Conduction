"""LBM flat-wall boundary modes.

- Every mode is rest-neutral (uniform field is a no-op) + mass-conserving.
- Named-mode / alpha-endpoint equivalences: neumann==hbb, combined(1)==hbb,
  combined(0)==specular_samecell; and the modes genuinely differ on a non-uniform field.
- API validation: D2Q9-only, unknown mode rejected, default is neumann, replay preserves it.
- Oblique anisotropy (D_xy != 0) is rejected by lbm().
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
from cardiac_core.run import run_lbm, simulate

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
    ('neumann', 1.0), ('hbb', 1.0), ('specular_nextcell', 1.0),
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


def test_lbm_default_boundary_is_lattice_aware():
    # d2q9 defaults to the HBB flat-wall baseline; d2q5 to generic neumann.
    assert lbm(_mesh(), lattice='d2q9')._engine.boundary == 'hbb'
    assert lbm(_mesh(), lattice='d2q5')._engine.boundary == 'neumann'


def test_lbm_hbb_requires_d2q9():
    # hbb joined the D2Q9-only flat-wall family; explicit hbb on d2q5 must raise
    with pytest.raises(ValueError, match="d2q9"):
        lbm(_mesh(), boundary='hbb', lattice='d2q5')


def test_lbm_combined_selectable_and_replayed():
    sim = lbm(_mesh(), lattice='d2q9', boundary='combined', alpha=0.3)
    assert sim._engine.boundary == 'combined' and sim._engine.alpha == 0.3
    sim.reset()   # build_kwargs must replay boundary + alpha
    assert sim._engine.boundary == 'combined' and sim._engine.alpha == 0.3


def test_ncs_scs_aliases():
    """Standard abbreviations resolve to canonical mode names."""
    for alias, canon in (('ncs', 'specular_nextcell'), ('scs', 'specular_samecell')):
        assert lbm(_mesh(), lattice='d2q9', boundary=alias)._engine.boundary == canon


def test_lbm_rejects_oblique_Dxy():
    """OBLIQUE anisotropy (D_xy != 0) is a REAL numerics limitation, not a wiring gap:
    mrt_collide_d2q9 discards D_xy (p_xy_eq=0); supporting it needs the moment-space
    rotation of s_jx/s_jy. So lbm() raises on a *documented* limitation. This covers the
    OBLIQUE case specifically (not per-axis anisotropy, which does work — see below).
    MUST be REPLACED with a positive CV-along-fiber test if the rotation is ever
    implemented; do NOT keep the raise merely to keep this test green.
    """
    m = _mesh()
    m.D_xy = np.full_like(m.D_xy, 1e-4)
    with pytest.raises(ValueError, match="oblique"):
        lbm(m, lattice='d2q9')


# ------------------------------- MRT / per-axis-anisotropic wall modes
def test_lbm_anisotropic_boundary_runs():
    """Per-axis-anisotropic D (D_xx != D_yy, D_xy = 0 → MRT) + a specular wall must
    construct + run finite. The old `collision != 'bgk'` guard wrongly blocked this."""
    for boundary, kw in (('ncs', {}), ('combined', {'alpha': 0.3})):
        m = create_cardiac_mesh(0.4, 0.2, 0.02, D=1e-3, D_yy=5e-4, chi=1.0)
        sim = lbm(m, lattice='d2q9', boundary=boundary, **kw)
        assert sim._engine.collision == 'mrt'
        res = sim.run(6, 6)
        assert torch.isfinite(res.Vm[-1]).all()


def test_mrt_wall_rest_neutral():
    """The raw MRT+wall step is rest-neutral (uniform field no-op) + mass-conserving
    over 40 steps — proving the overlay works on MRT (not asserting a rejection)."""
    from cardiac_core._lbm.step import lbm_step_d2q9_mrt_wall
    sim = LBMSimulation(NX, NY, DX, DT, D, TTP06Model(device='cpu'),
                        lattice='d2q9', collision='mrt', D_yy=D * 0.5, boundary='ncs')
    w, masks = sim.w, sim.bounce_masks
    V = torch.full((NX, NY), V_REST, dtype=torch.float64)
    f = w[:, None, None] * V[None, :, :]
    R = torch.zeros(NX, NY, dtype=torch.float64)
    m0 = f.sum().item()
    dmax = 0.0
    for _ in range(40):
        f, V = lbm_step_d2q9_mrt_wall(
            f, V, R, DT, w, sim.s_e, sim.s_eps, sim.s_jx, sim.s_q,
            sim.s_pxx, sim.s_pxy, masks, sim.boundary, sim.alpha, NX, NY, s_jy=sim.s_jy)
        dmax = max(dmax, (V - V_REST).abs().max().item())
    assert dmax < 1e-9, f"MRT wall not rest-neutral: max|V-Vrest|={dmax}"
    assert abs(f.sum().item() - m0) < 1e-8, "MRT wall mass drift"


# --------------------------------------------------- one-shot API parity
def _wall_mesh():
    """Left-edge-stim mesh whose front reaches the top/bottom walls (where the modes act)."""
    return create_cardiac_mesh(0.5, 0.3, 0.02, D=1e-3, chi=1.0)


def test_run_lbm_forwards_boundary():
    """A wall mode requested through the one-shot run_lbm must take effect (was silently HBB)."""
    _, v_scs = run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='scs', dt=0.005)
    _, v_hbb = run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='hbb', dt=0.005)
    assert (v_scs[-1] - v_hbb[-1]).abs().max() > 1e-2


def test_run_lbm_alpha_effective():
    """alpha is forwarded through run_lbm (combined-mode blend differs by alpha)."""
    _, v_a = run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='combined', alpha=0.2, dt=0.005)
    _, v_b = run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='combined', alpha=0.8, dt=0.005)
    assert (v_a[-1] - v_b[-1]).abs().max() > 1e-3


def test_run_lbm_rejects_bad_boundary():
    with pytest.raises(ValueError):
        run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='bogus', dt=0.005)


def test_simulate_matches_run_lbm():
    """run_lbm ≡ simulate(engine='lbm') for identical args (LBM is RNG-free → bit-identical)."""
    _, v_run = run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='scs', alpha=1.0, dt=0.005)
    res = simulate(_wall_mesh(), 8, 8, engine='lbm', lattice='d2q9',
                   boundary='scs', alpha=1.0, dt=0.005)
    assert torch.equal(v_run, res.Vm)
