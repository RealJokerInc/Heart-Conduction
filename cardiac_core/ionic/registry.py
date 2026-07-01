"""Single shared ionic-model builder used by all three engines (C3).

Before this, mono/bidomain/lbm each had their own resolver: mono forwarded `cell_type`,
bidomain/lbm did not, and mono/bidomain accepted only ttp06/ord (lbm also accepted
phas13/mhas13/paci) — an asymmetry that made `monodomain(ionic_model='phas13')` raise.

`build_ionic_model` branches on CONSTRUCTOR CAPABILITY: TTP06/ORd take a `CellType`
(string -> enum here); PHAS13/MHAS13/paci are device-only (their ctors take no cell_type).
Default `cell_type='ENDO'` matches every engine's current default (TTP06/ORd ctors default
ENDO; bidomain/lbm pass no cell_type), so delegating is behavior-preserving (goldens hold).
`'paci'` is a same-class alias of PHAS13Model.
"""
from .base import IonicModel, CellType
from .ttp06 import TTP06Model
from .ord import ORdModel
from .phas13 import PHAS13Model
from .mhas13 import MHAS13Model

# TTP06/ORd accept a CellType; PHAS13/MHAS13/paci are device-only.
_CELLTYPE_MODELS = {'ttp06': TTP06Model, 'ord': ORdModel}
_DEVICE_ONLY_MODELS = {'phas13': PHAS13Model, 'paci': PHAS13Model, 'mhas13': MHAS13Model}


def build_ionic_model(name, cell_type='ENDO', device='cuda'):
    """Build an IonicModel from a name (or return an existing instance unchanged).

    Parameters
    ----------
    name : str | IonicModel
        Model name ('ttp06', 'ord', 'phas13', 'mhas13', 'paci') or a pre-built instance
        (e.g. a tuner-scaled model) — returned as-is.
    cell_type : str | CellType
        Applied ONLY to TTP06/ORd (string is upper-cased into the CellType enum). Ignored
        for the device-only models. Default 'ENDO' matches every engine's current default.
    device : str
        Compute device.
    """
    if isinstance(name, IonicModel):
        return name
    key = name.lower()
    if key in _CELLTYPE_MODELS:
        ct = getattr(CellType, cell_type.upper()) if isinstance(cell_type, str) else cell_type
        return _CELLTYPE_MODELS[key](cell_type=ct, device=device)
    if key in _DEVICE_ONLY_MODELS:
        return _DEVICE_ONLY_MODELS[key](device=device)
    raise ValueError(
        f"Unknown ionic model: {name!r} (available: ttp06, ord, phas13, mhas13, paci)"
    )
