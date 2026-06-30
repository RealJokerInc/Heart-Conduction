"""Phase 2 — chip mesh + Parker anisotropic targets (fast, no sim)."""
import numpy as np

from tuner.chip import (
    chip_mesh, wavelength_mm, points_per_wavelength,
    PARKER_NRVM, PARKER_HIPSC, ANISOTROPY_RATIO,
)


def test_chip_mesh_shape_and_anisotropy():
    m = chip_mesh(domain_mm=16.0, dx_mm=0.1, D_long=0.001, D_trans=0.00025)
    assert m.mask.shape == (161, 161)
    assert np.isclose(m.dx, 0.01)
    assert np.allclose(m.D_xx, 0.001)
    assert np.allclose(m.D_yy, 0.00025)
    assert np.allclose(m.D_xy, 0.0)


def test_chip_mesh_default_D_ratio():
    """Default D_trans = D_long / ratio²  (CV_T = CV_L/ratio, CV ∝ √D)."""
    m = chip_mesh(domain_mm=4.0, dx_mm=0.1, D_long=0.004)
    assert np.allclose(m.D_yy, 0.004 / ANISOTROPY_RATIO ** 2)


def test_wavelength_and_ppwl():
    assert abs(wavelength_mm(9.33, 350.0) - 32.655) < 0.1     # NRVM, > 16 mm chip
    assert abs(wavelength_mm(5.2, 350.0) - 18.2) < 0.1        # hiPSC, > 16 mm chip
    assert points_per_wavelength(9.33, 350.0, 0.1) > 25       # dx resolves λ finely


def test_parker_targets_anisotropic():
    assert PARKER_NRVM.cv_longitudinal == 9.33
    assert abs(PARKER_NRVM.cv_transverse - 9.33 / 2.0) < 1e-9
    assert PARKER_HIPSC.cv_longitudinal == 5.2
    assert abs(PARKER_HIPSC.cv_transverse - 2.6) < 1e-9
    # dvdt_max_upper must be 120, not the default 60 (else cell_fitter rejects MHAS13)
    assert PARKER_NRVM.dvdt_max_upper == 120.0
    assert PARKER_HIPSC.dvdt_max == 110.0
