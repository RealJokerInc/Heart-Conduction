"""Integrator error budget: Euler vs dopri5 vs simulator truth.

Diagnostic answering: does forward-Euler inference at dt=0.01ms accumulate
meaningful truncation error vs the same model integrated tightly (dopri5)?
And how does either compare to the simulator ground truth?

Three outcomes:
 (1) Euler ~= dopri5 ~= truth  -> integrator swap / new design is wasted effort
 (2) Euler != dopri5, both != truth -> f_theta capacity is the bottleneck
 (3) Euler != dopri5, dopri5 == truth -> integrator fix (Option 1 or g_phi) pays off

Run:
    conda run -n heart-conduction python -m Surrogate.diagnostics.integrator_error_budget
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torchdiffeq import odeint

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Surrogate"))

from surrogate.model.node import IonicNODE
from surrogate.model.stage1 import IonicStage1
from surrogate.data.preprocessor import V3Preprocessor
from surrogate.training.node_rollout import INIT_CONC
from surrogate.training.loss_normalization import _RANGES


H5_PATH = "/media/HDD/norepinephrine/surrogate_data/raw/tier01.h5"
HELD_OUT_BCL = 2000
WINDOW_MS = 300.0
DT_MS = 0.01


def _resolve_ckpt() -> Path:
    """Pick the latest Hydra best.pt (cardiac_ml harness). Falls back to the
    legacy multi_bcl_002 path for pre-harness runs if no Hydra output exists.
    """
    hydra_candidates = sorted(glob.glob(str(REPO / "outputs/*/*/best.pt")))
    if hydra_candidates:
        return Path(hydra_candidates[-1])
    legacy = REPO / "Surrogate/runs/multi_bcl_002/best.pt"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(
        "No checkpoint found under outputs/*/*/best.pt or "
        "Surrogate/runs/multi_bcl_002/best.pt"
    )


def _load_stage1_state(ckpt: dict, stage1: IonicStage1) -> None:
    """Dual-format loader: v3 wrapper OR cardiac_ml flat state dict."""
    if "stage1_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["stage1_state_dict"])
    else:
        stage1_sd = {
            k[len("stage1."):]: v
            for k, v in ckpt.items()
            if k.startswith("stage1.")
        }
        assert stage1_sd, (
            "Checkpoint has no stage1.* keys and no 'stage1_state_dict' wrapper; "
            "unknown format"
        )
        stage1.load_state_dict(stage1_sd)


def load_trajectory(h5_path: str, bcl: int, window_ms: float, dt: float):
    """Load first `window_ms` of BCL=`bcl` protocol from tier1. Returns preprocessed dict."""
    key = f"steady_bcl{bcl}_dt{dt:.2f}/data"
    n = int(window_ms / dt)
    with h5py.File(h5_path, "r") as f:
        raw = torch.tensor(f[key][:n, :], dtype=torch.float64)
    return V3Preprocessor().process_segment(raw)


def run_euler(node: IonicNODE, z0, V_traj, dt):
    """Forward Euler at native dt, using V(t) at each step."""
    T = V_traj.shape[0]
    zs = torch.empty(T, z0.shape[-1], dtype=torch.float64, device=z0.device)
    z = z0.clone()
    zs[0] = z
    for i in range(T - 1):
        z = node.euler_step(z, V_traj[i], dt)
        zs[i + 1] = z
    return zs


def run_dopri5(node: IonicNODE, z0, V_traj, t_grid, t_eval,
               rtol=1e-8, atol=1e-10):
    """Integrate f_theta with tight-tolerance dopri5. Ground truth of the vector field."""
    node.set_v_trajectory(V_traj, t_grid)
    try:
        z_out = odeint(node, z0, t_eval, method="dopri5", rtol=rtol, atol=atol)
    finally:
        node.clear_v_trajectory()
    return z_out  # (len(t_eval), carried_dim)


def decode_ionic(stage1: IonicStage1, z):
    """Map carried_state z -> decoded ionic_states (13 gates + CaSR)."""
    ionic = z[..., : stage1.ionic_dim]
    return stage1.ionic_state_decoder(ionic)


def rmse(a, b):
    return torch.sqrt(((a - b) ** 2).mean()).item()


def per_dim_rmse(a, b):
    return torch.sqrt(((a - b) ** 2).mean(dim=0))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    ckpt_path = _resolve_ckpt()
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Held-out BCL: {HELD_OUT_BCL}, window: {WINDOW_MS} ms, dt: {DT_MS} ms")
    print()

    # --- Load trajectory ---
    seg = load_trajectory(H5_PATH, HELD_OUT_BCL, WINDOW_MS, DT_MS)
    V_traj = seg["Vm"].to(device)                        # (T,)
    ionic_true = seg["ionic_states"].to(device)          # (T, 14)
    conc_true = seg["concentrations"].to(device)         # (T, 4)
    T = V_traj.shape[0]
    print(f"Trajectory: T={T} steps, V range [{V_traj.min():.1f}, {V_traj.max():.1f}] mV")

    # --- Build model ---
    stage1 = IonicStage1(scaffold=True).to(dtype=torch.float64, device=device)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    _load_stage1_state(ckpt, stage1)
    # Rest-attractor invariant: re-pin decoder bias (load_state_dict overwrote it).
    if hasattr(stage1, "pin_rest_bias"):
        stage1.pin_rest_bias()
    stage1.eval()
    node = IonicNODE(stage1).to(device)
    epoch = ckpt.get("epoch", "unknown")
    val_loss = ckpt.get("val_loss")
    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "unknown"
    print(f"Loaded checkpoint at epoch {epoch}, val_loss={val_loss_str}")

    # --- Initial state (matches training convention) ---
    z0 = torch.zeros(stage1.carried_dim, dtype=torch.float64, device=device)
    z0[stage1.ionic_dim:] = INIT_CONC.to(device)

    # --- Time grids ---
    dt_vec = torch.full((T,), DT_MS, dtype=torch.float64, device=device)
    t_grid = torch.cat([torch.zeros(1, dtype=torch.float64, device=device),
                        dt_vec.cumsum(0)])   # (T+1,)
    t_eval_full = t_grid[:T]                  # (T,) one sample per data step

    # --- Run Euler ---
    print("Running Euler at dt=0.01ms...")
    with torch.no_grad():
        z_euler = run_euler(node, z0, V_traj, DT_MS)         # (T, carried)
    ionic_euler = decode_ionic(stage1, z_euler)              # (T, 14)
    conc_euler = z_euler[:, stage1.ionic_dim:]

    # --- Run dopri5 at tight tolerance ---
    print("Running dopri5 (rtol=1e-8, atol=1e-10)...")
    with torch.no_grad():
        z_dopri = run_dopri5(node, z0, V_traj, t_grid, t_eval_full,
                             rtol=1e-8, atol=1e-10)          # (T, carried)
    ionic_dopri = decode_ionic(stage1, z_dopri)
    conc_dopri = z_dopri[:, stage1.ionic_dim:]

    # --- Sanity check shapes ---
    assert ionic_euler.shape == ionic_dopri.shape == ionic_true.shape, \
        f"{ionic_euler.shape} {ionic_dopri.shape} {ionic_true.shape}"

    # --- Aggregate errors ---
    print()
    print("=== Aggregate RMSE (over T x 14 ionic dims) ===")
    print(f"  Euler  vs dopri5 : {rmse(ionic_euler, ionic_dopri):.5f}   "
          "(integrator truncation)")
    print(f"  dopri5 vs truth  : {rmse(ionic_dopri, ionic_true):.5f}   "
          "(model capacity)")
    print(f"  Euler  vs truth  : {rmse(ionic_euler, ionic_true):.5f}   "
          "(total inference error)")
    print()

    # --- Per-dim breakdown ---
    lbl = ["m", "h", "j", "r", "s", "d", "f", "f2", "fCass", "Xr1", "Xr2", "Xs",
           "RR", "CaSR"]
    e_vs_d = per_dim_rmse(ionic_euler, ionic_dopri)
    d_vs_t = per_dim_rmse(ionic_dopri, ionic_true)
    e_vs_t = per_dim_rmse(ionic_euler, ionic_true)
    print("=== Per-dim RMSE ===")
    print(f"{'dim':>6}  {'E-D':>9}  {'D-T':>9}  {'E-T':>9}")
    for i, name in enumerate(lbl):
        print(f"{name:>6}  {e_vs_d[i].item():9.5f}  {d_vs_t[i].item():9.5f}  "
              f"{e_vs_t[i].item():9.5f}")
    print()

    # --- NRMSE: normalized by physiological range (compare to Session 27 v3) ---
    rng = _RANGES["ionic_states"]
    range_denom = (rng["max"] - rng["min"]).to(device=ionic_true.device,
                                                dtype=ionic_true.dtype)     # (14,)
    nrmse_e_t = per_dim_rmse(ionic_euler, ionic_true) / range_denom
    nrmse_d_t = per_dim_rmse(ionic_dopri, ionic_true) / range_denom
    print("=== NRMSE (% of physiological range) vs v3 baseline ===")
    print(f"{'dim':>6}  {'Euler-T %':>11}  {'dopri5-T %':>11}")
    for i, name in enumerate(lbl):
        print(f"{name:>6}  {100 * nrmse_e_t[i].item():11.2f}  "
              f"{100 * nrmse_d_t[i].item():11.2f}")
    print(f"  CaSR target: < 20%   (Session 27 v3 baseline: 27.4%)")
    print()

    # --- Time-resolved error (5ms windows) ---
    print("=== Time-resolved total-dim RMSE (integrator truncation only) ===")
    bin_ms = 5.0
    bin_steps = int(bin_ms / DT_MS)
    n_bins = T // bin_steps
    t_mid = (torch.arange(n_bins, device=device) + 0.5) * bin_ms
    err_e_d = (ionic_euler - ionic_dopri).reshape(n_bins, bin_steps, -1).pow(2).mean(dim=(1, 2)).sqrt()
    err_d_t = (ionic_dopri - ionic_true).reshape(n_bins, bin_steps, -1).pow(2).mean(dim=(1, 2)).sqrt()
    err_e_t = (ionic_euler - ionic_true).reshape(n_bins, bin_steps, -1).pow(2).mean(dim=(1, 2)).sqrt()
    print(f"{'t (ms)':>7}  {'E-D':>9}  {'D-T':>9}  {'E-T':>9}")
    # Print at 0-5, 5-10, 20-25, 100-105, 200-205, 295-300
    pick_ms = [2.5, 7.5, 22.5, 102.5, 202.5, 297.5]
    for tm in pick_ms:
        i = min(int(tm / bin_ms), n_bins - 1)
        print(f"{t_mid[i].item():7.1f}  {err_e_d[i].item():9.5f}  "
              f"{err_d_t[i].item():9.5f}  {err_e_t[i].item():9.5f}")
    print()

    # --- Save artifacts for optional plotting ---
    out_dir = REPO / "Surrogate/diagnostics/artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "V_traj": V_traj.cpu(),
        "ionic_true": ionic_true.cpu(),
        "ionic_euler": ionic_euler.cpu(),
        "ionic_dopri": ionic_dopri.cpu(),
        "conc_true": conc_true.cpu(),
        "conc_euler": conc_euler.cpu(),
        "conc_dopri": conc_dopri.cpu(),
        "dt": DT_MS,
        "bcl": HELD_OUT_BCL,
        "val_loss": val_loss,
        "ckpt_path": str(ckpt_path),
        "nrmse_euler_vs_truth": nrmse_e_t.cpu(),
        "nrmse_dopri_vs_truth": nrmse_d_t.cpu(),
    }, out_dir / "integrator_error_budget_v4.pt")
    print(f"Saved artifacts to {out_dir}/integrator_error_budget_v4.pt")


if __name__ == "__main__":
    main()
