"""
Optimizer V1 — LBM planar-wave chip baseline + Lab-preset hand-off (Phase 5).

Loads a tuned record, builds the anisotropic LBM chip mesh (D2Q9-MRT via the
record's per-axis D), drives a planar wave (left-edge stim), measures CV + λ,
and exports a Tier-2 Lab preset. This is the contract handed to the
geometry_induced_reentry application (which then adds the obstacle sweeps).

Run (GATED full 161² chip): conda run -n heart-conduction python Optimizer/V1/run_chip_baseline_lbm.py
"""
import torch

from cardiac_core import run_lbm, analysis
from cardiac_core.ionic import MHAS13Model, PHAS13Model, TTP06Model

from tuner.config import apply_scaling
from tuner.chip import chip_mesh, wavelength_mm
from tuner.presets import load_record, export_lab_preset, PRESETS_DIR, LAB_PRESETS_DIR

_MODELS = {"mhas13": MHAS13Model, "phas13": PHAS13Model,
           "paci": PHAS13Model, "ttp06": TTP06Model}


def baseline_lbm(record, *, domain_mm=None, dx_mm=None, t_end=200.0,
                 save_every=1.0, device=None, export=True, lab_dir=LAB_PRESETS_DIR):
    """Planar-wave LBM baseline on the chip from a tuned record.

    Returns {cv, wavelength_mm, n_saves, preset_path}. Uses the record's LBM
    tissue (per-axis D) if present, else its monodomain D.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mesh_meta = record["mesh"]
    domain_mm = domain_mm or mesh_meta["domain_mm"]
    dx_mm = dx_mm or mesh_meta["dx_mm"]

    tissue = record["tissue"].get("lbm") or record["tissue"]["monodomain"]
    D_long, D_trans = tissue["D_long"], tissue["D_trans"]
    dt = tissue.get("dt_ms") or 0.01

    model = _MODELS[record["ionic_model"]](device=device)
    if record["theta_ionic"]:
        apply_scaling(model.params, record["theta_ionic"])

    mesh = chip_mesh(domain_mm=domain_mm, dx_mm=dx_mm,
                     D_long=D_long, D_trans=D_trans,
                     ionic_model=record["ionic_model"], dt=dt)   # left-edge planar stim
    times, V = run_lbm(mesh, t_end=t_end, save_every=save_every,
                       ionic_model=model, dt=dt, device=device)
    if not torch.is_tensor(times):
        times = torch.as_tensor(times, dtype=torch.float64)

    Nx, Ny = mesh.mask.shape
    cv = float(analysis.conduction_velocity(
        V, times, dx=dx_mm / 10.0, x1=Nx // 4, x2=3 * Nx // 4, y=Ny // 2,
        threshold=-30.0))
    apd = record["targets"].get("apd_90", 350.0)
    lam = wavelength_mm(cv, apd) if cv == cv else float("nan")

    preset_path = export_lab_preset(record, engine="lbm", lab_dir=lab_dir) if export else None
    return {"cv": cv, "wavelength_mm": lam, "n_saves": int(V.shape[0]),
            "preset_path": preset_path}


def main():  # pragma: no cover — gated full 161² chip run
    for baseline in ("nrvm", "hipsc"):
        rec = load_record(f"chip_{baseline}", presets_dir=PRESETS_DIR)
        out = baseline_lbm(rec, t_end=200.0)
        print(baseline, out)


if __name__ == "__main__":  # pragma: no cover
    main()
