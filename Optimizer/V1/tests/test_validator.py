"""
Tests for validator.py — Validation suite.
"""

import pytest
import torch


class TestValidator:
    """Phase VI: Validator tests."""

    def test_validation_result_structure(self):
        """ValidationResult has correct structure."""
        from tuner.validator import ValidationResult, ValidationCheck

        r = ValidationResult()
        r.checks.append(ValidationCheck(name="test", passed=True))
        r.checks.append(ValidationCheck(name="test2", passed=False))

        assert r.n_passed == 1
        assert r.n_total == 2
        assert not r.all_passed
        assert "1/2" in r.summary()

    def test_validation_check_fields(self):
        """ValidationCheck stores all fields."""
        from tuner.validator import ValidationCheck

        c = ValidationCheck(
            name="apd_check", passed=True, value=350.0,
            target=350.0, tolerance=10.0, message="OK"
        )
        assert c.name == "apd_check"
        assert c.passed
        assert c.value == 350.0
