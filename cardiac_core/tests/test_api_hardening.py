"""Silent-failure hardening for the public factories.

One named test per finding (S1, I1, I2, S2, S3, C4, I3), plus the S3 negative case
and the S1 valid-input case. These document each individual fix; the contract matrix
(test_api_contract.py) is the systemic guard that forces the fix via strict-xfail.
"""
import os
import tempfile
import warnings

import numpy as np
import pytest
import torch

from cardiac_core import monodomain, bidomain, lbm, create_cardiac_mesh, Grid, ConductivityConfig
from cardiac_core.io import save_result, load_result


def _mesh(**kw):
    return create_cardiac_mesh(0.2, 0.1, 0.02, D=1e-3, chi=1.0, **kw)


# --- S1: bidomain boundary validation ---------------------------------------
def test_bidomain_rejects_bad_boundary():
    for bad in ('ncs', 'insualted'):   # an LBM mode + a typo — both were silently 'insulated'
        with pytest.raises(ValueError, match='boundary'):
            bidomain(_mesh(), boundary=bad)


def test_bidomain_boundary_bath_valid():
    bidomain(_mesh(), boundary='bath')       # valid values still construct (no over-narrowing)
    bidomain(_mesh(), boundary='insulated')
    bidomain(_mesh())                        # None → data.boundary default


# --- I1: LBM masked interior hole -------------------------------------------
def test_lbm_masked_hole_nonconducting():
    mask = np.ones((11, 11), dtype=bool)
    mask[4:7, 4:7] = False
    sim = lbm(create_cardiac_mesh(0.2, 0.2, 0.02, D=1e-3, chi=1.0, mask=mask), lattice='d2q9')
    bm = sim._engine.bounce_masks
    assert any(bool(bm[a][3:8, 3:8].any()) for a in bm), "hole rim not flagged (rect-edges only)"
    res = sim.run(6, 6)
    assert torch.isfinite(res.Vm[-1]).all()   # no periodic leak at the outer walls


# --- I2: with_() and factory immutability -----------------------------------
def test_with_immutable_stimuli():
    p = monodomain(_mesh())
    n0 = len(p._data.stimuli)
    c = p.with_(dt=0.02)
    c.stimulate(lambda x, y: x < 0.03)
    assert len(p._data.stimuli) == n0, "with_ child mutated the parent"


def test_factory_mesh_not_aliased():
    m = _mesh()
    n0 = len(m.stimuli)
    a = monodomain(m)
    b = monodomain(m)
    a.stimulate(lambda x, y: x < 0.03)
    assert len(m.stimuli) == n0, "factory aliased the caller's mesh"
    assert len(b._data.stimuli) == n0, "two sims from one mesh cross-contaminated"


# --- S2 / S3 / C4: silent-degrade -> warn -----------------------------------
def test_lbm_alpha_inert_warns():
    with pytest.warns(UserWarning, match='alpha'):
        lbm(_mesh(), boundary='neumann', alpha=0.3)


def test_bidomain_sigma_ratio_ignored_warns():
    g = Grid(Nx=11, Ny=6, dx=0.02, dy=0.02)
    cond = ConductivityConfig.bidomain(sigma_i=1.74, sigma_e=6.25, chi=1.0)
    with pytest.warns(UserWarning, match='sigma_ratio'):
        bidomain(g, 'ttp06', cond, sigma_ratio=10.0)


def test_bidomain_sigma_ratio_no_warn_when_only_ratio():
    """Legacy mesh path (no sigma_i/sigma_e) USES sigma_ratio → must NOT warn."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        bidomain(_mesh(), sigma_ratio=10.0)
    assert not any('sigma_ratio' in str(x.message) for x in w)


def test_lbm_lattice_override_warns():
    with pytest.warns(UserWarning, match='lattice'):
        lbm(_mesh(D_yy=5e-4), lattice='d2q5')   # anisotropic → forced to d2q9/MRT


# --- I3: result dtype round-trip --------------------------------------------
def test_load_result_dtype_roundtrip():
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    vm32 = torch.zeros((2, 4, 3), dtype=torch.float32)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'r.npz')
        save_result(path, times, Vm=vm32)
        _, V, _, _ = load_result(path)
        assert V.dtype == torch.float32


# --- capability exposure (C3, C5, C6, S4) -----------------------------------
def test_ionic_registry_all_engines():
    """Every advertised ionic model constructs on ALL THREE engines (shared registry, C3)."""
    for name in ('ttp06', 'ord', 'phas13', 'mhas13', 'paci'):
        assert monodomain(_mesh(), ionic_model=name) is not None
        assert bidomain(_mesh(), ionic_model=name) is not None
        assert lbm(_mesh(), ionic_model=name) is not None


def test_lbm_weights_mode_exposed():
    sim = lbm(_mesh(), lattice='d2q9', weights_mode='uniform_8')
    assert 'uniform' in type(sim._engine.lattice).__name__.lower()


def test_monodomain_stencil_and_boundary_mode_exposed():
    assert monodomain(_mesh(), stencil='moore8_iso') is not None       # stencil vocab
    assert monodomain(_mesh(), boundary_mode='face_mirror_iso') is not None  # boundary_mode vocab


def test_bidomain_splitting_exposed():
    assert bidomain(_mesh(), splitting='godunov') is not None


def test_solver_knob_engine_mismatch():
    """S4: a knob valid on engine A raises a VALIDATED error (not a bare TypeError) on B."""
    with pytest.raises(ValueError, match='bidomain'):
        monodomain(_mesh(), theta=0.5)
    with pytest.raises(ValueError, match='monodomain'):
        bidomain(_mesh(), diffusion_solver='crank_nicolson')
    with pytest.raises(ValueError, match='monodomain'):
        bidomain(_mesh(), linear_solver='pcg')
