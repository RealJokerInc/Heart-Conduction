"""An ionic_model override must survive reset()/stimulate()/with_().

Before the fix, build_kwargs omitted ionic_model, so reset()/stimulate() rebuilt
the engine from the MESH DEFAULT ('ttp06'), silently discarding an 'ord' override.
"""
from cardiac_core import monodomain, create_cardiac_mesh
from cardiac_core.ionic import ORdModel


def _model_cls(sim):
    return type(sim._engine._ionic_model).__name__


def _mesh():
    # mesh default ionic_model is 'ttp06'; we override with 'ord'
    return create_cardiac_mesh(1.0, 0.5, 0.05)


def test_reset_preserves_string_override():
    sim = monodomain(_mesh(), ionic_model='ord')
    assert _model_cls(sim) == 'ORdModel'
    sim.reset()
    assert _model_cls(sim) == 'ORdModel'   # was 'TTP06Model' before the fix


def test_stimulate_preserves_override():
    sim = monodomain(_mesh(), ionic_model='ord')
    sim.stimulate(lambda x, y: x < 0.06)
    assert _model_cls(sim) == 'ORdModel'


def test_reset_preserves_instance_override():
    sim = monodomain(_mesh(), ionic_model=ORdModel(device='cpu'))
    assert _model_cls(sim) == 'ORdModel'
    sim.reset()
    assert _model_cls(sim) == 'ORdModel'


def test_with_preserves_override():
    sim = monodomain(_mesh(), ionic_model='ttp06')
    sim2 = sim.with_(ionic_model='ord')
    assert _model_cls(sim2) == 'ORdModel'
    assert _model_cls(sim) == 'TTP06Model'   # receiver untouched (immutable)
