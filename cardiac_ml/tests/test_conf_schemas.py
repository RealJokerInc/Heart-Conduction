"""Tests for cardiac_ml/conf_schemas.py."""
from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cardiac_ml.conf_schemas import TrainingConfig, TrackingConfig, _register


def test_training_config_defaults():
    """TrainingConfig has sensible defaults for all non-required fields."""
    c = TrainingConfig(epochs=10, optimizer={"_target_": "torch.optim.Adam", "lr": 1e-3})
    assert c.seed == 42
    assert c.device == "cuda"
    assert c.dtype == "float64"
    assert c.ode_method == "dopri5"
    assert c.phase_name is None  # Optional, no default phase


def test_training_config_node_fields_accept_overrides():
    """NODE-specific fields accept explicit values."""
    c = TrainingConfig(
        epochs=500,
        optimizer={"_target_": "torch.optim.Adam", "lr": 5e-4},
        phase_name="A1",
        ode_method="dopri5",
        ode_adjoint=False,
    )
    assert c.phase_name == "A1"
    assert c.ode_adjoint is False


def test_tracking_config_defaults():
    c = TrackingConfig()
    assert c.enabled is True
    assert c.experiment_name == "default"
    assert c.tracking_uri == "./mlruns"
    assert c.checkpoint_every == 50


def test_register_populates_config_store():
    """_register() installs schemas under training/schema and tracking/schema."""
    _register()
    cs = ConfigStore.instance()
    # ConfigStore.repo is a nested dict; exact API is private. Verify via
    # load() which succeeds only if the schema is present.
    training_schema = cs.load("training/schema.yaml")
    tracking_schema = cs.load("tracking/schema.yaml")
    assert training_schema is not None
    assert tracking_schema is not None


def test_register_idempotent():
    """Calling _register() twice does not raise."""
    _register()
    _register()  # second call must be a no-op (not an error)
