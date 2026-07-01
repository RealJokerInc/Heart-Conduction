"""Contract-matrix stress harness (engine_consolidation API-consistency PLAN, Phase 0).

WRITTEN FIRST (before any fix), per the post-mortem lesson: the contract of
{entry point} x {engine} x {param} x {physics} is enumerated as data with an
expected verdict + audit finding ID, so a never-considered cell surfaces as a
standing xfail — not an absence — and each fix flips a cell that already exists.

Cell.status drives the marker (see `_marks`):
  * 'to_fix'   -> xfail(strict=True). The cell FAILS today; when its Phase-1..4 fix
                 lands the cell XPASSes and strict=True FAILS the suite, FORCING the
                 executor to flip status -> 'landed' in that same phase (no deferred
                 cleanup). This is what stops the matrix rotting into a green suite.
  * 'deferred' -> xfail(strict=False). Genuinely deferred (C2 oblique capability,
                 C7 boundary-Dxy truncation); body never passes; permanent xfail.
  * 'landed'   -> no marker (live assert). Already-correct behavior, regression-locked.

NOTE (execution refinement vs PLAN Step 0.1): the plan's Cell tuple listed
(entry,engine,param,physics,expected,match,finding_id,status,note,exc). A generic
data-only dispatch can't reconstruct arbitrary calls, so each Cell carries a `run`
callable (the honest way to encode "assert the change / raise / warn"); `exc` stays
the last, sole-defaulted field (namedtuple right-aligns defaults). Documentary
`entry/engine/param/physics` live in the pytest id; `note` folded into `run` docstrings.

Run: /home/norepinephrine/.conda/envs/heart-conduction/bin/python -m pytest \
        cardiac_core/tests/test_api_contract.py -q
"""
import os
import tempfile
import warnings
from collections import namedtuple

import numpy as np
import pytest
import torch

from cardiac_core import monodomain, bidomain, lbm, create_cardiac_mesh
from cardiac_core.run import run_lbm, simulate
from cardiac_core.io import save_result, load_result


# --------------------------------------------------------------------- helpers
def _tiny(**kw):
    """~11x6 mesh, chi=1 so D is an effective diffusivity (avoids the firewall block)."""
    return create_cardiac_mesh(0.2, 0.1, 0.02, D=1e-3, chi=1.0, **kw)


def _wall_mesh():
    """Larger mesh whose default left-edge stim drives a front to the top/bottom walls
    (where the wall modes act), so hbb vs specular diverge at dt=0.005 (~70 mV)."""
    return create_cardiac_mesh(0.5, 0.3, 0.02, D=1e-3, chi=1.0)


def _masked_hole_mesh(**kw):
    """11x11 mesh with a False interior hole (I1)."""
    mask = np.ones((11, 11), dtype=bool)
    mask[4:7, 4:7] = False
    return create_cardiac_mesh(0.2, 0.2, 0.02, D=1e-3, chi=1.0, mask=mask, **kw)


def _oblique(mesh):
    mesh.D_xy = np.full_like(mesh.D_xy, 1e-4)
    return mesh


def _not_none(x):
    assert x is not None
    return x


# --- behavioral runs (each FAILS today; PASSES once its fix lands) -----------
def _run_lbm_boundary_takes_effect():
    """P1: a wall mode requested through the one-shot run_lbm must take effect."""
    _, v_scs = run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='scs', dt=0.005)
    _, v_hbb = run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='hbb', dt=0.005)
    assert (v_scs[-1] - v_hbb[-1]).abs().max() > 1e-2


def _simulate_matches_run_lbm():
    """P1 invariant: simulate(engine='lbm') ≡ run_lbm for identical args (RNG-free → bit-identical)."""
    _, v_run = run_lbm(_wall_mesh(), 8, 8, lattice='d2q9', boundary='scs', alpha=1.0, dt=0.005)
    res = simulate(_wall_mesh(), 8, 8, engine='lbm', lattice='d2q9',
                   boundary='scs', alpha=1.0, dt=0.005)
    assert torch.equal(v_run, res.Vm)


def _bidomain_sigma_ratio_warn():
    """S3: on the declarative path, sigma_i/sigma_e win → sigma_ratio is silently ignored."""
    from cardiac_core import Grid, ConductivityConfig
    g = Grid(Nx=11, Ny=6, dx=0.02, dy=0.02)
    cond = ConductivityConfig.bidomain(sigma_i=1.74, sigma_e=6.25, chi=1.0)
    bidomain(g, 'ttp06', cond, sigma_ratio=10.0)


def _lbm_aniso_wall_runs():
    """C1: per-axis-anisotropic (MRT) + a specular wall must run (guard was over-conservative)."""
    sim = lbm(_tiny(D_yy=5e-4), lattice='d2q9', boundary='ncs')
    assert sim._engine.collision == 'mrt'


def _lbm_oblique_cv_correct():
    """C2 capability (DEFERRED, Audit #46): oblique fibers need moment-space rotation.
    Today the construction raises; asserts the not-yet-true capability → permanent xfail."""
    lbm(_oblique(_tiny()), lattice='d2q9')  # raises today
    raise AssertionError("C2 oblique CV-along-fiber not implemented (Audit #46)")


def _c7_boundary_dxy_truncation():
    """C7 (DEFERRED, paired to C2): mono/bidomain drop D_xy at the wall (interior-correct only)."""
    raise AssertionError("C7 boundary-Dxy truncation — deferred, paired to C2/Audit #46")


def _lbm_weights_mode_exposed():
    """C5: uniform_8 connectivity must be selectable via lbm()."""
    _not_none(lbm(_tiny(), lattice='d2q9', weights_mode='uniform_8'))


def _lbm_masked_hole_bounces():
    """I1: a masked interior hole must produce bounce-back on its rim (not just the rect edges)."""
    sim = lbm(_masked_hole_mesh(), lattice='d2q9')
    bm = sim._engine.bounce_masks  # dict: direction -> (Nx,Ny) bool
    interior_flagged = any(bool(bm[a][3:8, 3:8].any()) for a in bm)
    assert interior_flagged, "no interior bounce mask → the hole conducts (rect-edges only)"


def _with_and_factory_immutable():
    """I2: with_() child and the caller's mesh must not be mutated by stimulate()."""
    m = _tiny()
    n0 = len(m.stimuli)
    p = monodomain(m)
    c = p.with_(dt=0.02)
    c.stimulate(lambda x, y: x < 0.03)
    assert len(m.stimuli) == n0, "factory aliased the caller's mesh"
    assert len(p._data.stimuli) == n0, "with_ child mutated the parent"


def _dtype_roundtrips():
    """I3: a saved float32 Vm must reload as float32, not promoted to float64."""
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    vm32 = torch.zeros((2, 4, 3), dtype=torch.float32)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'r.npz')
        save_result(path, times, Vm=vm32)
        _, V, _, _ = load_result(path)
        assert V.dtype == torch.float32


# ----------------------------------------------------------------- the matrix
# Cell: exc is LAST (namedtuple right-aligns the single default onto exc, not note).
Cell = namedtuple(
    "Cell",
    "fid entry engine param physics expected run match status exc",
    defaults=["", "to_fix", ValueError],  # -> match, status, exc
)

CONTRACT = [
    # --- HIGH: the 5 silent-wrong-result bugs -------------------------------
    Cell('P1', 'run_lbm', 'lbm', 'boundary', 'iso', 'effect', _run_lbm_boundary_takes_effect,
         '', 'landed'),
    Cell('P1', 'simulate', 'lbm', 'parity', 'iso', 'effect', _simulate_matches_run_lbm,
         '', 'landed'),
    Cell('S1', 'bidomain', 'bidomain', 'boundary-ncs', 'iso', 'raise',
         lambda: bidomain(_tiny(), boundary='ncs'), 'boundary'),
    Cell('S1', 'lbm', 'lbm', 'boundary-bath', 'iso', 'raise',
         lambda: lbm(_tiny(), boundary='bath'), 'boundary must be one of', 'landed'),
    Cell('C1', 'lbm', 'lbm', 'boundary-ncs', 'aniso-mrt', 'effect', _lbm_aniso_wall_runs,
         '', 'landed'),
    Cell('I1', 'lbm', 'lbm', 'masked-hole', 'masked', 'effect', _lbm_masked_hole_bounces),
    Cell('I2', 'with_', 'monodomain', 'immutability', 'iso', 'effect', _with_and_factory_immutable),
    # --- HIGH: cross-engine capability / oblique ----------------------------
    Cell('C2', 'lbm', 'lbm', 'oblique-raise', 'oblique', 'raise',
         lambda: lbm(_oblique(_tiny()), lattice='d2q9'), 'oblique|Audit #46', 'landed'),
    Cell('C2', 'lbm', 'lbm', 'oblique-capability', 'oblique', 'effect',
         _lbm_oblique_cv_correct, '', 'deferred'),
    Cell('C3', 'monodomain', 'monodomain', 'ionic-phas13', 'iso', 'effect',
         lambda: _not_none(monodomain(_tiny(), ionic_model='phas13'))),
    Cell('C3', 'bidomain', 'bidomain', 'ionic-mhas13', 'iso', 'effect',
         lambda: _not_none(bidomain(_tiny(), ionic_model='mhas13'))),
    # --- MED: silent-degrade -> warn ----------------------------------------
    Cell('S2', 'lbm', 'lbm', 'alpha-inert', 'iso', 'warn',
         lambda: lbm(_tiny(), boundary='neumann', alpha=0.3), 'alpha'),
    Cell('S3', 'bidomain', 'bidomain', 'sigma_ratio', 'iso', 'warn',
         _bidomain_sigma_ratio_warn, 'sigma_ratio'),
    Cell('C4', 'lbm', 'lbm', 'lattice-override', 'aniso', 'warn',
         lambda: lbm(_tiny(D_yy=5e-4), lattice='d2q5'), 'lattice'),
    # --- MED: exposure ------------------------------------------------------
    Cell('C5', 'lbm', 'lbm', 'weights_mode', 'iso', 'effect', _lbm_weights_mode_exposed),
    Cell('C6', 'monodomain', 'monodomain', 'stencil', 'iso', 'effect',
         lambda: _not_none(monodomain(_tiny(), stencil='moore8_iso'))),
    Cell('S4', 'bidomain', 'bidomain', 'splitting', 'iso', 'effect',
         lambda: _not_none(bidomain(_tiny(), splitting='godunov'))),
    Cell('S4', 'monodomain', 'monodomain', 'theta-mismatch', 'iso', 'raise',
         lambda: monodomain(_tiny(), theta=0.5), 'bidomain|theta'),
    Cell('I3', 'io', 'na', 'dtype', 'float32', 'effect', _dtype_roundtrips),
    # --- MED: deferred (paired to C2) ---------------------------------------
    Cell('C7', 'monodomain', 'monodomain', 'oblique-wall', 'oblique', 'effect',
         _c7_boundary_dxy_truncation, '', 'deferred'),
]


def _marks(c):
    if c.status == 'landed':
        return ()
    return pytest.mark.xfail(reason=c.fid, strict=(c.status == 'to_fix'))


@pytest.mark.parametrize(
    "c",
    [pytest.param(c, id=f"{c.fid}:{c.entry}:{c.engine}:{c.param}:{c.physics}", marks=_marks(c))
     for c in CONTRACT],
)
def test_contract(c):
    """Each cell either takes effect, raises a validated error, or warns — never silently degrades."""
    if c.expected == 'raise':
        with pytest.raises(c.exc, match=c.match or None):
            c.run()
    elif c.expected == 'warn':
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            with pytest.warns(UserWarning, match=c.match or None):
                c.run()
    else:  # 'effect' — run() performs its own assertion
        c.run()
