"""
Optimizer V1 — tuned-parameter records + Lab-preset export (storage).

Tier 1 (canonical research artifact): full JSON records under Optimizer/V1/presets/.
A record holds θ_ionic multipliers + per-engine tissue params (monodomain D_long/
D_trans; LBM D + MRT rates) + the mesh dx/dt context + targets + validation +
provenance. Physical D is the source of truth; LBM MRT rates are derived per
dx/dt (BGK τ cannot represent anisotropy).

Tier 2 (consumable Lab preset, Lab/presets/{name}.yaml) is exported in Phase 5
via export_lab_preset (needs the _SCHEMA.md extension).

See PLAN § Parameter & Preset Storage.
"""
import json
import os
from typing import Optional

PRESETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "presets")
)

_VALID_BASELINES = ("nrvm", "hipsc")


def make_record(name: str, baseline: str, theta_ionic: dict, tissue: dict,
                targets: dict, validation: Optional[dict] = None,
                provenance: Optional[dict] = None,
                ionic_model: str = "mhas13",
                domain_mm: float = 16.0, dx_mm: float = 0.1) -> dict:
    """Assemble a Tier-1 record (the canonical fit artifact).

    Parameters
    ----------
    theta_ionic : dict   param_name -> multiplier (on published values)
    tissue : dict        per-engine tissue params, e.g.
        {"monodomain": {"D_long":.., "D_trans":.., "dt_ms":..},
         "lbm": {"D_long":.., "D_trans":.., "collision":"mrt",
                 "s_jx":.., "s_jy":.., "dx_mm":.., "dt_ms":..}}
    targets, validation, provenance : dict (provenance e.g. date/git_sha filled by caller)
    """
    if baseline not in _VALID_BASELINES:
        raise ValueError(f"baseline must be one of {_VALID_BASELINES}, got {baseline!r}")
    return {
        "name": name,
        "baseline": baseline,
        "ionic_model": ionic_model,
        "theta_ionic": dict(theta_ionic),
        "mesh": {"domain_mm": domain_mm, "dx_mm": dx_mm},
        "tissue": tissue,
        "targets": targets,
        "validation": validation or {},
        "provenance": provenance or {},
    }


def save_record(record: dict, name: str = None, presets_dir: str = PRESETS_DIR) -> str:
    """Write a record to {presets_dir}/{name}.json. Returns the path."""
    name = name or record["name"]
    os.makedirs(presets_dir, exist_ok=True)
    path = os.path.join(presets_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=False)
    return path


def load_record(name: str, presets_dir: str = PRESETS_DIR) -> dict:
    """Load {presets_dir}/{name}.json."""
    path = os.path.join(presets_dir, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def list_records(presets_dir: str = PRESETS_DIR) -> list:
    """List record names (without .json) in presets_dir."""
    if not os.path.isdir(presets_dir):
        return []
    return sorted(f[:-5] for f in os.listdir(presets_dir) if f.endswith(".json"))


def to_sim_kwargs(record: dict, engine: str) -> dict:
    """Materialize cardiac_core sim kwargs for `engine` from a record.

    Returns the per-axis effective D + dt to feed chip_mesh + a run_* call.
    (LBM derives its MRT rates from D inside LBMSimulation at the given dx/dt.)
    """
    tissue = record["tissue"]
    if engine not in tissue:
        raise KeyError(f"record has no tissue params for engine {engine!r} "
                       f"(have: {sorted(tissue)})")
    t = tissue[engine]
    out = {
        "ionic_model": record["ionic_model"],
        "theta_ionic": record["theta_ionic"],
        "dx_mm": record["mesh"]["dx_mm"],
        "domain_mm": record["mesh"]["domain_mm"],
        "D_long": t["D_long"],
        "D_trans": t["D_trans"],
        "dt": t.get("dt_ms"),
    }
    return out
