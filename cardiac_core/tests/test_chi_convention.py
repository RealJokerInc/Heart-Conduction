"""chi/D convention: `D_xx` is RAW; the membrane-effective diffusivity is
`D_xx/(χ·Cm)`, computed identically by every engine.

- `create_cardiac_mesh` / any raw mesh stores raw D_xx + real chi.
- The declarative path (`_build_mesh_data`) also stores raw D_xx so the factory
  division recovers the intended effective D (no double-divide).
- The old default `(D=0.001, chi=1400)` was conduction-blocked (eff 7.1e-7);
  the new default `D=1.4` gives eff 1e-3.
"""
import warnings
import numpy as np
import pytest

from cardiac_core import Grid, ConductivityConfig, monodomain, lbm, create_cardiac_mesh, Stim


def _stim(g):
    return Stim.from_region(g, (lambda x, y: x < 0.06),
                            start_time=1.0, duration=2.0, amplitude=-80.0)


def test_lbm_factory_divides_raw_D():
    """Legacy raw mesh (D=1.4, chi=1400) → LBM engine sees effective 1e-3."""
    m = create_cardiac_mesh(1.0, 0.5, 0.05, D=1.4, chi=1400.0)
    assert np.isclose(lbm(m)._engine.D, 1.4 / 1400.0, rtol=1e-9)


def test_declarative_lbm_no_double_divide():
    """Declarative path must land on cond.D_eff, not D_eff/(χ·Cm)."""
    g = Grid(40, 20, 0.05)
    cond = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    sim = lbm(g, 'ttp06', cond, _stim(g))
    assert np.isclose(sim._engine.D, float(cond.D_eff), rtol=1e-9)


def test_chi_one_treats_D_as_effective():
    """chi=1.0 → raw == effective (chip convention); LBM engine gets D verbatim."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # in-band ⇒ no warning
        m = create_cardiac_mesh(1.0, 0.5, 0.05, D=1e-3, chi=1.0)
    assert np.isclose(lbm(m)._engine.D, 1e-3, rtol=1e-9)


def test_default_mesh_propagates():
    """Default create_cardiac_mesh (D=1.4, chi=1400 → eff 1e-3) must NOT block."""
    sim = monodomain(create_cardiac_mesh(1.0, 0.5, 0.05))
    snaps = list(sim.snapshots(20.0, save_every=5.0))
    V = snaps[-1].Vm
    V = np.asarray(V.cpu() if hasattr(V, 'cpu') else V)
    assert V.max() > -20.0, f"no propagation (Vmax={V.max()})"
    assert (V > -40).sum() > V.size // 4


def test_effective_band_warning():
    """Effective D (0.001) at default chi=1400 → eff 7.1e-7, out of band → warns."""
    with pytest.warns(UserWarning, match="effective diffusivity"):
        create_cardiac_mesh(1.0, 0.5, 0.05, D=0.001, chi=1400.0)
