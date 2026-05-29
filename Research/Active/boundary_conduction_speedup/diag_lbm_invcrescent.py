"""
LBM verification of the HBB ≡ face_mirror sign-lock.

Re-runs the case 7 inverse-crescent setup but in LBM V1 with D2Q9 + bounce-back.

Hypothesis (from monodomain case 7): face_mirror sign-locks to forward crescent.
An imposed inverse crescent (boundaries advanced 1 column ahead of interior) is
eaten by the per-column source-effect deficit within ~2 columns of propagation.

Per the corrected HBB ≡ face_mirror analysis, LBM with bounce-back should
exhibit the SAME sign-lock — i.e., the imposed inverse crescent should also
be eaten within a small number of columns.

Setup:
  - LBM V1, D2Q9 canonical lattice + bounce-back (HBB)
  - NX=41, NY=21, dx=0.025, dt=0.02 ms (standard LBM dt)
  - TTP06 EPI ionic
  - IC: cols 1,2,3 all rows at +30 mV; col 4 only at j=0 and j=NY-1 (inv crescent)
  - Sync window: 5 steps (0.1 ms) — re-clamp the inverse crescent shape after
    each LBM step. f populations re-equilibrated as w * V_clamped.
  - After release: full LBM evolution until t_end=25 ms.

Output:
  data/case8_lbm_d2q9_apfirst_invcrescent.h5
  /V (T, NX, NY) float64, /t (T,) float64, attrs lattice/weights_mode/etc.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch
import h5py

LBM_ROOT = Path("/home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1")
sys.path.insert(0, str(LBM_ROOT))

from src.simulation import LBMSimulation
from ionic.ttp06.model import TTP06Model
from ionic.base import CellType


# ---------- config ----------
NX, NY = 41, 21
DX = 0.025                  # cm
DT = 0.02                   # ms
D = 0.001                   # cm² / ms
LATTICE = "d2q9"
WEIGHTS_MODE = "canonical"  # canonical D2Q9 weights (4/9, 1/9, 1/36)
V_AP_TRIG = 30.0            # mV
SYNC_STEPS = 5              # 0.1 ms sync window at dt=0.02
T_END = 25.0                # ms

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "case8_lbm_d2q9_apfirst_invcrescent.h5"


def clamp_inv_crescent(V: torch.Tensor) -> torch.Tensor:
    """Apply the inverse-crescent clamp to V in-place: cols 1,2,3 all rows,
    plus col 4 only at the two y-boundary rows."""
    for col in (1, 2, 3):
        V[col, :] = V_AP_TRIG
    V[4, 0]      = V_AP_TRIG
    V[4, NY - 1] = V_AP_TRIG
    return V


def main():
    device = torch.device("cpu")
    ionic = TTP06Model(cell_type=CellType.EPI, device=device)
    V_rest = float(ionic.V_rest)

    sim = LBMSimulation(
        Nx=NX, Ny=NY, dx=DX, dt=DT, D=D,
        ionic_model=ionic, Cm=1.0,
        lattice=LATTICE, weights_mode=WEIGHTS_MODE,
    )

    # IC: rest everywhere, then impose inverse-crescent clamp.
    V_init = torch.full((NX, NY), V_rest, dtype=sim.dtype, device=device)
    V_init = clamp_inv_crescent(V_init)
    sim.V = V_init
    # Re-equilibrate f to match V (f = w * V, mass-consistent).
    sim.f = sim.w[:, None, None] * sim.V[None, :, :]

    n_steps = int(round(T_END / DT))
    print(f"Grid: {NX} × {NY} = {NX*NY} cells   dx={DX} cm   dt={DT} ms   "
          f"t_end={T_END} ms ({n_steps} steps)")
    print(f"Lattice: {LATTICE} / {WEIGHTS_MODE}   V_rest={V_rest:.3f} mV   "
          f"V_AP_TRIG=+{V_AP_TRIG} mV   sync_steps={SYNC_STEPS}")

    V_hist = np.empty((n_steps + 1, NX, NY), dtype=np.float64)
    t_hist = np.empty(n_steps + 1, dtype=np.float64)
    V_hist[0] = sim.V.cpu().numpy()
    t_hist[0] = 0.0

    t0 = time.time()
    for k in range(1, n_steps + 1):
        sim.step()
        if k <= SYNC_STEPS:
            # Re-clamp the inverse crescent shape; re-equilibrate f.
            sim.V = clamp_inv_crescent(sim.V.clone())
            sim.f = sim.w[:, None, None] * sim.V[None, :, :]
        V_hist[k] = sim.V.cpu().numpy()
        t_hist[k] = k * DT
    elapsed = time.time() - t0

    x_coords = (np.arange(NX) * DX).astype(np.float64)
    y_coords = (np.arange(NY) * DX).astype(np.float64)

    with h5py.File(OUT_FILE, "w") as f:
        f.create_dataset("V", data=V_hist, compression="gzip", compression_opts=4)
        f.create_dataset("t", data=t_hist)
        f.create_dataset("x", data=x_coords)
        f.create_dataset("y", data=y_coords)
        a = f.attrs
        a["engine"] = "LBM_V1"
        a["lattice"] = LATTICE
        a["weights_mode"] = WEIGHTS_MODE
        a["boundary_treatment"] = "halfway_bounce_back"
        a["physics"] = "ttp06_apfirst_invcrescent_lbm"
        a["dx"] = DX
        a["dt"] = DT
        a["D"] = D
        a["V_AP_TRIG"] = V_AP_TRIG
        a["V_rest"] = V_rest
        a["NX"] = NX
        a["NY"] = NY
        a["t_end"] = T_END
        a["n_steps"] = n_steps
        a["sync_steps"] = SYNC_STEPS
        a["sync_cols_all_rows"] = np.array([1, 2, 3], dtype=np.int64)
        a["sync_cols_bdry_only"] = np.array([4], dtype=np.int64)
        a["inverse_crescent"] = True

    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n  wrote {OUT_FILE.name}  ({size_mb:.1f} MB)  "
          f"V range [{V_hist.min():.3f}, {V_hist.max():.3f}] mV  elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
