"""Single source of truth for the 47-column TTP06 trace layout.

Mirrors TraceData.N_COLUMNS in single_cell_generator.py. Every v2 HDF5 file
writes COLUMN_NAMES and COLUMN_UNITS into its file-level attrs, so loaders
can introspect without importing this module.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

N_COLUMNS = 47

COLUMN_NAMES: Tuple[str, ...] = (
    "Vm",
    "I_stim",
    "dt",
    "K_i",
    "Na_i",
    "Ca_i",
    "Ca_SR",
    "Ca_ss",
    "m",
    "h",
    "j",
    "r",
    "s",
    "d",
    "f",
    "f2",
    "fCass",
    "Xr1",
    "Xr2",
    "Xs",
    "RR",
    "I_ion",
    "clamp_mask",
    "m_inf", "h_inf", "j_inf", "r_inf", "s_inf",
    "d_inf", "f_inf", "f2_inf", "fCass_inf",
    "Xr1_inf", "Xr2_inf", "Xs_inf",
    "m_tau", "h_tau", "j_tau", "r_tau", "s_tau",
    "d_tau", "f_tau", "f2_tau", "fCass_tau",
    "Xr1_tau", "Xr2_tau", "Xs_tau",
)

COLUMN_UNITS: Tuple[str, ...] = (
    "mV",
    "pA/pF",
    "ms",
    "mM", "mM", "mM", "mM", "mM",
    "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
    "pA/pF",
    "-",
    "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
    "ms", "ms", "ms", "ms", "ms", "ms", "ms", "ms", "ms", "ms", "ms", "ms",
)

assert len(COLUMN_NAMES) == N_COLUMNS
assert len(COLUMN_UNITS) == N_COLUMNS

COLUMN_GROUPS: Dict[str, List[int]] = {
    "Vm": [0],
    "stim": [1],
    "dt": [2],
    "concentrations": [3, 4, 5, 6, 7],
    "gates": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "RR": [20],
    "I_ion": [21],
    "clamp_mask": [22],
    "gate_inf": list(range(23, 35)),
    "gate_tau": list(range(35, 47)),
}


def column_groups_json() -> str:
    """JSON string for the `column_groups` file attr (HDF5 can't store nested dicts)."""
    return json.dumps(COLUMN_GROUPS, separators=(",", ":"))


# Range checks for write-time validation.
# (lower, upper) inclusive. None means unchecked.
COLUMN_BOUNDS: Dict[str, Tuple[float | None, float | None]] = {
    "Vm": (-100.0, 80.0),
    "K_i": (0.0, None),
    "Na_i": (0.0, None),
    "Ca_i": (0.0, None),
    "Ca_SR": (0.0, None),
    "Ca_ss": (0.0, None),
    "m": (-1e-6, 1.0 + 1e-6),
    "h": (-1e-6, 1.0 + 1e-6),
    "j": (-1e-6, 1.0 + 1e-6),
    "r": (-1e-6, 1.0 + 1e-6),
    "s": (-1e-6, 1.0 + 1e-6),
    "d": (-1e-6, 1.0 + 1e-6),
    "f": (-1e-6, 1.0 + 1e-6),
    "f2": (-1e-6, 1.0 + 1e-6),
    "fCass": (-1e-6, 1.0 + 1e-6),
    "Xr1": (-1e-6, 1.0 + 1e-6),
    "Xr2": (-1e-6, 1.0 + 1e-6),
    "Xs": (-1e-6, 1.0 + 1e-6),
}


DATASET_VERSION = "v2"
