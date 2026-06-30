"""
Tests for config.py — Parameter registry, scaling, bounds.
"""

import pytest
import torch


class TestConfig:
    """Phase I: Config and parameter registry tests."""

    def test_tuning_targets_defaults(self):
        """TuningTargets has PHAS13-appropriate defaults."""
        from tuner.config import TuningTargets
        t = TuningTargets()
        assert t.apd_90 == 350.0
        assert t.cv_longitudinal == 15.0
        assert t.v_rest == -74.0

    def test_tuning_config_defaults(self):
        """TuningConfig defaults to MHAS13 / tier 2 (current target; was phas13/tier1)."""
        from tuner.config import TuningConfig
        c = TuningConfig()
        assert c.ionic_model == 'mhas13'
        assert c.tier == 2

    def test_registry_tier1(self):
        """Tier 1 has 6 params."""
        from tuner.config import get_params_for_tier
        t1 = get_params_for_tier(1)
        assert len(t1) == 6
        assert 'g_Na' in t1
        assert 'g_Kr' in t1

    def test_registry_tier2(self):
        """Tier 2 has 10 params (6 + 4)."""
        from tuner.config import get_params_for_tier
        t2 = get_params_for_tier(2)
        assert len(t2) == 10
        assert 'kNaCa' in t2

    def test_registry_tier3(self):
        """Tier 3 has 14 params (10 + 4)."""
        from tuner.config import get_params_for_tier
        t3 = get_params_for_tier(3)
        assert len(t3) == 14
        assert 'g_f' in t3
        assert 'V_leak' in t3

    def test_bounds_tensor_shape(self):
        """Bounds tensor is (2, n_params)."""
        from tuner.config import get_bounds_tensor
        b = get_bounds_tensor(1)
        assert b.shape == (2, 6)
        assert (b[0] < b[1]).all()  # Lower < upper

    def test_param_names_ordered(self):
        """Param names match registry order."""
        from tuner.config import get_param_names, get_params_for_tier
        names = get_param_names(1)
        assert len(names) == 6
        assert names == list(get_params_for_tier(1).keys())

    def test_apply_scaling(self):
        """Scaling factors applied correctly."""
        from tuner.config import apply_scaling, PHAS13_REGISTRY
        from cardiac_sim.ionic.phas13 import PHAS13Parameters

        p = PHAS13Parameters()
        original_gNa = p.g_Na

        apply_scaling(p, {'g_Na': 0.5})
        assert p.g_Na == pytest.approx(original_gNa * 0.5)

        # Restore
        apply_scaling(p, {'g_Na': 1.0})
        assert p.g_Na == pytest.approx(PHAS13_REGISTRY['g_Na'].published)

    def test_scaling_roundtrip(self):
        """theta_to_dict and dict_to_theta are inverses."""
        from tuner.config import theta_to_dict, dict_to_theta
        theta = torch.tensor([0.5, 0.8, 1.2, 0.9, 1.1, 0.7], dtype=torch.float64)
        d = theta_to_dict(theta, tier=1)
        theta_back = dict_to_theta(d, tier=1)
        assert torch.allclose(theta, theta_back)

    def test_apply_scaling_unknown_param(self):
        """apply_scaling raises on unknown parameter."""
        from tuner.config import apply_scaling
        from cardiac_sim.ionic.phas13 import PHAS13Parameters
        p = PHAS13Parameters()
        with pytest.raises(ValueError, match="Unknown parameter"):
            apply_scaling(p, {'nonexistent': 1.0})
