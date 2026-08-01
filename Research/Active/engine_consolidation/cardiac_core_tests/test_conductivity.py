"""Tests for cardiac_core.conductivity.ConductivityConfig — the chi/Cm firewall.

The for_monodomain D must equal the reference effective diffusivity to machine precision and
be Cm-INDEPENDENT (= sigma_eff/chi).
"""

import math

import pytest

from cardiac_core import ConductivityConfig

# Reference effective diffusivity for sigma_i=1.74, sigma_e=6.25, chi=1400 (Cm-independent).
_D_EFF_REF = 0.0009721973895941354


class TestConductivityArithmetic:
    """Always-on, no engine needed — the load-bearing firewall guard."""

    def test_arithmetic_gate(self):
        for Cm in (1.0, 2.0):
            cfg = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0, Cm=Cm)
            mono = cfg.for_monodomain()
            # for_monodomain D is Cm-INDEPENDENT and equals the reference D_eff.
            assert abs(mono['D'] - _D_EFF_REF) < 1e-12, f"Cm={Cm}: D={mono['D']}"
            assert mono['chi'] == 1.0
            assert mono['Cm'] == Cm
            # D_eff (true physical) scales as 1/Cm.
            assert math.isclose(cfg.D_eff, _D_EFF_REF / Cm, rel_tol=1e-12)
            # for_bidomain components are sigma/(chi*Cm).
            bd = cfg.for_bidomain()
            assert math.isclose(bd['D_i'], 1.74 / (1400.0 * Cm), rel_tol=1e-12)
            assert math.isclose(bd['D_e'], 6.25 / (1400.0 * Cm), rel_tol=1e-12)
            assert bd['Cm'] == Cm

    def test_isotropic_and_lbm(self):
        cfg = ConductivityConfig.isotropic(0.001, chi=1400.0, Cm=1.0)
        assert math.isclose(cfg.sigma_eff, 0.001)
        assert math.isclose(cfg.D_eff, 0.001 / 1400.0, rel_tol=1e-12)
        # for_lbm emits fully-scaled D_eff + real Cm (Form B).
        assert math.isclose(cfg.for_lbm()['D'], cfg.D_eff, rel_tol=1e-12)
        assert cfg.for_lbm()['Cm'] == 1.0

    def test_sigma_eff_harmonic_collapse(self):
        cfg = ConductivityConfig.bidomain(1.74, 6.25)
        expected = 1.74 * 6.25 / (1.74 + 6.25)
        assert math.isclose(cfg.sigma_eff, expected, rel_tol=1e-15)

    def test_anisotropic_tensor_matches_sigma_to_D(self):
        # Mirror LBM sigma_to_D for a 30-degree fiber.
        angle = math.radians(30.0)
        sl, st, chi, Cm = 1.74, 0.435, 1400.0, 1.0
        cfg = ConductivityConfig.anisotropic(sl, st, angle, chi=chi, Cm=Cm)
        Dxx, Dyy, Dxy = cfg.D_eff
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        sxx = st + (sl - st) * cos_a ** 2
        syy = st + (sl - st) * sin_a ** 2
        sxy = (sl - st) * cos_a * sin_a
        scale = 1.0 / (chi * Cm)
        assert math.isclose(Dxx, sxx * scale, rel_tol=1e-12)
        assert math.isclose(Dyy, syy * scale, rel_tol=1e-12)
        assert math.isclose(Dxy, sxy * scale, rel_tol=1e-12)

    def test_emitter_errors(self):
        # for_bidomain requires sigma_i/sigma_e.
        with pytest.raises(ValueError):
            ConductivityConfig.isotropic(0.001).for_bidomain()
        # An empty config has no conductivity data.
        with pytest.raises(ValueError):
            _ = ConductivityConfig().sigma_eff
