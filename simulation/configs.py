# -*- coding: utf-8 -*-
"""Experiment configurations for the storage-tank simulator.

Each config is a plain dict with five sections — geometry, rule, pipes,
boundary, sim. The DEFAULT below documents every supported field. Named
configs are produced by `make({...overrides...})`, which deep-merges the
overrides into DEFAULT.

To add a new tunable: extend DEFAULT, then teach the engine (`tanks_vec.run`)
about the new option. To run a new combination: add a new named config below
or pass overrides to `experiment.run_experiment(make({...}))`.

The intent is that >90% of new experiments are config edits here, not engine
edits in tanks_vec.py.
"""

from __future__ import annotations

import copy
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DEFAULT: dict[str, Any] = {
    "name": "default",
    "description": "",
    "tags": [],
    "geometry": {
        "type": "line",            # 'line' | 'point_cluster' | 'custom'
        "Nx": 80,
        "Ny": 50,
        # custom_inlet_cells / custom_outlet_cells: lists of (x, y), used only
        # if type == 'custom'
        "custom_inlet_cells": None,
        "custom_outlet_cells": None,
    },
    "rule": {
        "type": "constant",        # 'constant' | 'gradient'
        "threshold": 45.0,
        "max_volume": 100.0,
        "max_pump": 10.0,          # used by 'constant' (John's effective val)
        "gradient_k": 0.08,        # used by 'gradient'
        "damping_cap": True,       # John's quarter-gap clamp
    },
    "pipes": {
        "directionality": "one_way",   # 'one_way' (John) | 'bidirectional'
        "connectivity": "moore8",      # 'moore8' (only option for now)
    },
    "boundary": {
        "type": "zero_pad",         # 'zero_pad' | 'reflect_y' | 'reflect_all'
    },
    "sim": {
        "steps": 4000,
        "record_history": False,
        "snap_every": 100,
        "sample_cols": (3, 8, 18, 30, 45, 60),  # for per-column LAT plot
    },
}


def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def make(overrides: dict[str, Any], *, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a config by deep-merging `overrides` into `base` (default: DEFAULT).

    Use `base=BASELINE` (or any other named config) to start from that variant
    instead of the schema default. Always deep-merges, so passing
    `{"rule": {"threshold": 10}}` only changes the threshold field, leaving
    `rule.type` and others intact.
    """
    cfg = copy.deepcopy(base if base is not None else DEFAULT)
    _deep_update(cfg, overrides)
    return cfg


def resolve_geometry(geom: dict) -> tuple[list, list]:
    """Translate a geometry dict to (inlet_cells, outlet_cells) lists."""
    Nx, Ny = geom["Nx"], geom["Ny"]
    t = geom["type"]
    if t == "line":
        return ([(0, y) for y in range(Ny)],
                [(Nx - 1, y) for y in range(Ny)])
    if t == "point_cluster":
        # John's original ids [703, 705, 706, 707] under id = x*Ny+y, Ny=50
        return ([(14, 3), (14, 5), (14, 6), (14, 7)],
                [(Nx - 1, y) for y in range(Ny)])
    if t == "custom":
        return (list(geom["custom_inlet_cells"]),
                list(geom["custom_outlet_cells"]))
    raise ValueError(f"unknown geometry type: {t!r}")


# ---------------------------------------------------------------------------
# Predefined experiments
# ---------------------------------------------------------------------------

BASELINE = make({
    "name": "baseline",
    "description": "John's original setup: line source, constant rule, one-way pipes, zero-pad BC",
    "tags": ["baseline", "line", "constant", "one_way", "zero_pad"],
})

GRADIENT = make({
    "name": "gradient",
    "description": "Same geometry, Fickian gradient rule (k * (V_src - V_dst)) instead of John's source-limited rule",
    "tags": ["line", "gradient", "one_way", "zero_pad"],
    "rule": {"type": "gradient"},
})

BIDIRECTIONAL = make({
    "name": "bidirectional",
    "description": "John's constant rule but with V_src > V_dst gate dropped — both pipe directions fire when their source is above threshold",
    "tags": ["line", "constant", "bidirectional", "zero_pad"],
    "pipes": {"directionality": "bidirectional"},
})

GRADIENT_BIDIRECTIONAL = make({
    "name": "gradient_bidirectional",
    "description": "Fickian gradient rule with bidirectional pipes — both A→B and B→A fire when their source is above threshold; signed k·(V_src−V_dst) gives signed net flux",
    "tags": ["line", "gradient", "bidirectional", "zero_pad"],
    "rule": {"type": "gradient"},
    "pipes": {"directionality": "bidirectional"},
})

REFLECT_Y = make({
    "name": "reflect_y",
    "description": "John's rule with reflection BC on y boundaries (corners=5, edges=8 effective channels)",
    "tags": ["line", "constant", "one_way", "reflect_y"],
    "boundary": {"type": "reflect_y"},
})

REFLECT_ALL = make({
    "name": "reflect_all",
    "description": "John's rule with reflection BC on all boundaries (translation-invariant operator at the wall)",
    "tags": ["line", "constant", "one_way", "reflect_all"],
    "boundary": {"type": "reflect_all"},
})

LONG_RUN_CONSTANT = make({
    "name": "long_run_constant",
    "description": "Steady-state probe for John's rule, 8000 steps, history recorded for activity analysis",
    "tags": ["line", "constant", "one_way", "zero_pad", "long_run"],
    "sim": {"steps": 8000, "record_history": True},
})

LONG_RUN_GRADIENT = make({
    "name": "long_run_gradient",
    "description": "Steady-state probe for gradient rule, 8000 steps, history recorded",
    "tags": ["line", "gradient", "one_way", "zero_pad", "long_run"],
    "rule": {"type": "gradient"},
    "sim": {"steps": 8000, "record_history": True},
})

JOHN_RADIAL = make({
    "name": "john_radial",
    "description": "John's exact original setup: point-cluster source, full 2000 steps, his rule",
    "tags": ["point_cluster", "constant", "one_way", "zero_pad"],
    "geometry": {"type": "point_cluster"},
    "sim": {"steps": 2000},
})


# Index of named configs for `experiment.run_experiment_by_name`
REGISTRY = {
    cfg["name"]: cfg
    for cfg in (BASELINE, GRADIENT, BIDIRECTIONAL, GRADIENT_BIDIRECTIONAL,
                REFLECT_Y, REFLECT_ALL,
                LONG_RUN_CONSTANT, LONG_RUN_GRADIENT, JOHN_RADIAL)
}
