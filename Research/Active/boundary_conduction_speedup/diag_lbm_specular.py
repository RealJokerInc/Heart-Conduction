"""
LBM with specular reflection at top/bottom walls — test inverse-crescent bias.

Hypothesis (user): an "elastic" wall reflection (specular) preserves tangential
momentum at the wall. Diagonal populations emitted from boundary cells get
their y-velocity flipped and traverse the wall horizontally — ending up at
the EAST/WEST NEIGHBOUR rather than back at the source cell.

Prediction (mine, to be tested): specular ≡ face_mirror_iso for the deficit
question. Gives ZERO deficit, NOT inverse crescent. The wave at the boundary
fires at the SAME time as in the interior.

If the test shows inverse crescent → user was right; specular biases toward
boundary speedup.

Setup:
  - LBM V1 D2Q9 canonical lattice + BGK collision
  - Top wall (j=NY-1) and bottom wall (j=0): SPECULAR for diagonals,
    HBB for cardinals. East/west walls (i=0, i=NX-1): standard HBB.
    Corners: HBB (simplification).
  - NX=41, NY=21, dx=0.025, dt=0.02 ms, D=0.001
  - TTP06 EPI
  - IC: V[col=0, :] = 0 mV (sub-threshold stim at leftmost wall); V[else] = V_rest
  - Run 25 ms. No clamp, no sync window — natural propagation under
    specular BC.

Compare:
  - case3 monodomain fm: +486 µs forward LAT shift (deficit)
  - case8 LBM HBB invc test: ~+73 µs asymptotic forward (after eaten lead)
  - case9 (this): natural propagation under specular. Expected ~0 LAT shift.

Output (filename pattern: case{N}_lbm_d2q9_{weights_mode}_{bc_mode}_natural.h5):
  data/case9_lbm_d2q9_canonical_specular_natural.h5
  data/case10_lbm_d2q9_canonical_hbb_natural.h5
  data/case11_lbm_d2q9_uniform_8_specular_natural.h5
  data/case12_lbm_d2q9_uniform_8_hbb_natural.h5
  data/case13_lbm_d2q9_canonical_horizontal_natural.h5
  data/case14_lbm_d2q9_uniform_8_horizontal_natural.h5
  data/case_weighted_a{α}_b{β}_g{γ}_{weights_mode}.h5  (weighted BC)
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
from src.collision.bgk import bgk_collide
from src.streaming.d2q9 import stream_d2q9
from src.boundary.neumann import apply_neumann_d2q9
from src.state import recover_voltage
from src.solver.rush_larsen import compute_source_term, ionic_step
from src.lattice import D2Q9
from ionic.ttp06.model import TTP06Model
from ionic.base import CellType


# ---------- config ----------
NX, NY = 41, 21
DX = 0.025
DT = 0.02
D = 0.001
V_STIM = 0.0
T_END = 25.0

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Selected via CLI arg: 'canonical' (case 9) or 'uniform_8' (case 11)
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--weights", default="canonical",
                     choices=["canonical", "uniform_8"])
_parser.add_argument("--bc", default="specular",
                     choices=["specular", "hbb", "horizontal", "weighted"],
                     help="boundary treatment at top/bottom walls")
_parser.add_argument("--alpha", type=float, default=None,
                     help="HBB weight in weighted BC (α + β + γ = 1)")
_parser.add_argument("--beta", type=float, default=None,
                     help="Specular weight in weighted BC")
_parser.add_argument("--gamma", type=float, default=None,
                     help="Horizontal weight in weighted BC")
_args, _ = _parser.parse_known_args()
WEIGHTS_MODE = _args.weights
BC_MODE = _args.bc
if BC_MODE == "weighted":
    if _args.alpha is None or _args.beta is None or _args.gamma is None:
        raise SystemExit("--bc weighted requires --alpha, --beta, --gamma (must sum to 1)")
    ALPHA, BETA, GAMMA = _args.alpha, _args.beta, _args.gamma
    if abs(ALPHA + BETA + GAMMA - 1.0) > 1e-9:
        raise SystemExit(f"weights must sum to 1, got α+β+γ = {ALPHA+BETA+GAMMA}")
    _case_id = f"case_weighted_a{ALPHA:.2f}_b{BETA:.2f}_g{GAMMA:.2f}_{WEIGHTS_MODE}"
    OUT_FILE = OUT_DIR / f"{_case_id}.h5"
else:
    ALPHA = BETA = GAMMA = None
    _case_id = {
        ("canonical", "specular"):   "case9",
        ("canonical", "hbb"):        "case10",
        ("uniform_8", "specular"):   "case11",
        ("uniform_8", "hbb"):        "case12",
        ("canonical", "horizontal"): "case13",
        ("uniform_8", "horizontal"): "case14",
    }[(WEIGHTS_MODE, BC_MODE)]
    OUT_FILE = OUT_DIR / f"{_case_id}_lbm_d2q9_{WEIGHTS_MODE}_{BC_MODE}_natural.h5"


def apply_specular_top_bottom_d2q9(f: torch.Tensor, f_star: torch.Tensor,
                                    NX: int, NY: int) -> torch.Tensor:
    """Specular reflection at top/bottom walls for D2Q9 — diagonals only,
    non-corner cells only. Caller MUST run apply_neumann_d2q9 (HBB) first
    so corners and east/west walls are correctly bounced; this function
    OVERWRITES the diagonal slots at top/bottom non-corner cells with the
    specular-traversal values.

    Top wall (j=NY-1), non-corner cells i in [1, NX-2]:
      - f_5 (NE) at (i, top) → f_8 (SE) at (i+1, top)
      - f_6 (NW) at (i, top) → f_7 (SW) at (i-1, top)

    Bottom wall (j=0), non-corner cells:
      - f_8 (SE) at (i, bot) → f_5 (NE) at (i+1, bot)
      - f_7 (SW) at (i, bot) → f_6 (NW) at (i-1, bot)

    Slicing carefully:
      - Top NE→SE: source i in [1, NX-3], dest i+1 in [2, NX-2] (keep both
        ends away from corners)
      - Top NW→SW: source i in [2, NX-2], dest i-1 in [1, NX-3]
    """
    # TOP WALL — diagonal traversal
    # f_5 (NE) at (i, top): for i in [1, NX-3], write to f[8, i+1, top]
    f[8, 2:NX - 1, NY - 1] = f_star[5, 1:NX - 2, NY - 1]
    # f_6 (NW) at (i, top): for i in [2, NX-2], write to f[7, i-1, top]
    f[7, 1:NX - 2, NY - 1] = f_star[6, 2:NX - 1, NY - 1]

    # BOTTOM WALL — diagonal traversal
    # f_8 (SE) at (i, bot): for i in [1, NX-3], write to f[5, i+1, bot]
    f[5, 2:NX - 1, 0] = f_star[8, 1:NX - 2, 0]
    # f_7 (SW) at (i, bot): for i in [2, NX-2], write to f[6, i-1, bot]
    f[6, 1:NX - 2, 0] = f_star[7, 2:NX - 1, 0]

    return f


def lbm_step_specular(f: torch.Tensor, V: torch.Tensor, R: torch.Tensor,
                      dt: float, omega: float, w: torch.Tensor,
                      bounce_masks_full: dict, NX: int, NY: int) -> tuple:
    """One D2Q9-BGK step with HBB everywhere first (overrides periodic
    wraparound, handles corners + east/west walls correctly), then
    SPECULAR overwrites top/bottom non-corner diagonals."""
    f = bgk_collide(f, V, R, dt, omega, w)
    f_star = f.clone()
    f = stream_d2q9(f)
    f = apply_neumann_d2q9(f, f_star, bounce_masks_full)
    f = apply_specular_top_bottom_d2q9(f, f_star, NX, NY)
    V = recover_voltage(f)
    return f, V


def apply_horizontal_redirect_top_bottom_d2q9(f: torch.Tensor, f_star: torch.Tensor,
                                                NX: int, NY: int) -> torch.Tensor:
    """Horizontal-redirect BC at top/bottom walls (non-corner cells).

    Replaces both HBB and specular for the diagonal slots at top/bottom.
    Cardinals (f_3 at top, f_4 at bottom) still use HBB (applied by caller).

    Rule: outgoing diagonal at the wall has its y-component absorbed, and
    its mass lands at the adjacent cell in the pure-horizontal slot.

    Top wall (j=NY-1), non-corner i in [1, NX-2]:
      - C's pre-stream f_5 (NE)  →  ADD to f_1 (E) at east neighbour (i+1, NY-1)
      - C's pre-stream f_6 (NW)  →  ADD to f_2 (W) at west neighbour (i-1, NY-1)
      - C's f_7, f_8 slots: ZERO out (caller has set them via HBB to C's own
        pre-stream f_5, f_6; we remove those since the same mass goes to
        neighbours via redirect — keeping both would double mass).

    Bottom wall (j=0), symmetric:
      - C's pre-stream f_7 (SW)  →  ADD to f_2 (W) at west neighbour
      - C's pre-stream f_8 (SE)  →  ADD to f_1 (E) at east neighbour
      - C's f_5, f_6 slots at bottom: ZERO out (HBB fills they get from
        C's own f_7, f_8 are now redundant with the redirect).

    Mass conservation: each pre-stream off-grid-bound diagonal goes to
    exactly ONE place — the adjacent cell's cardinal slot. Net effect:
    diagonal mass at top/bottom walls moves laterally along the wall
    in the cardinal slots, instead of bouncing back (HBB) or traversing
    in diagonal slots (specular).
    """
    # TOP WALL — zero HBB diagonal fill at top, redirect to cardinals
    f[7, 1:NX - 1, NY - 1] = 0
    f[8, 1:NX - 1, NY - 1] = 0
    # f_5 (NE) at top non-corner → ADD to east neighbour's f_1 (E)
    # source i in [1, NX-2], dest i+1 in [2, NX-1] (east-corner CAN be dest)
    f[1, 2:NX, NY - 1] += f_star[5, 1:NX - 1, NY - 1]
    # f_6 (NW) at top non-corner → ADD to west neighbour's f_2 (W)
    f[2, :NX - 2, NY - 1] += f_star[6, 1:NX - 1, NY - 1]

    # BOTTOM WALL — symmetric
    f[5, 1:NX - 1, 0] = 0
    f[6, 1:NX - 1, 0] = 0
    # f_8 (SE) at bottom non-corner → ADD to east neighbour's f_1
    f[1, 2:NX, 0] += f_star[8, 1:NX - 1, 0]
    # f_7 (SW) at bottom non-corner → ADD to west neighbour's f_2
    f[2, :NX - 2, 0] += f_star[7, 1:NX - 1, 0]

    return f


def lbm_step_horizontal(f: torch.Tensor, V: torch.Tensor, R: torch.Tensor,
                         dt: float, omega: float, w: torch.Tensor,
                         bounce_masks_full: dict, NX: int, NY: int) -> tuple:
    """One D2Q9-BGK step with horizontal-redirect at top/bottom non-corner cells."""
    f = bgk_collide(f, V, R, dt, omega, w)
    f_star = f.clone()
    f = stream_d2q9(f)
    f = apply_neumann_d2q9(f, f_star, bounce_masks_full)
    f = apply_horizontal_redirect_top_bottom_d2q9(f, f_star, NX, NY)
    V = recover_voltage(f)
    return f, V


def apply_weighted_top_bottom_d2q9(f: torch.Tensor, f_star: torch.Tensor,
                                     alpha: float, beta: float, gamma: float,
                                     NX: int, NY: int) -> torch.Tensor:
    """Weighted BC family on the (HBB, specular, horizontal) simplex.

    Each pre-stream diagonal mass at a top/bottom non-corner cell distributes
    across three destinations with weights α + β + γ = 1:
      α → HBB destination       (C's own anti-direction slot, same cell)
      β → specular destination  (adjacent cell's same-quadrant diagonal slot)
      γ → horizontal destination (adjacent cell's pure cardinal slot)

    Mass conservation: exact by construction (α + β + γ = 1).

    Vertex correspondence:
      (1, 0, 0)  →  pure HBB                  (≡ face_mirror in PDE)
      (0, 1, 0)  →  pure specular             (≡ face_mirror_iso in PDE)
      (0, 0, 1)  →  pure horizontal redirect  (novel; sustained inverse crescent)

    At each top non-corner cell C at (i, NY-1), i ∈ [1, NX-2]:
      Outgoing f_5 (NE):  α to C's f_7  | β to east's f_8  | γ to east's f_1
      Outgoing f_6 (NW):  α to C's f_8  | β to west's f_7  | γ to west's f_2

    Symmetric at bottom (j=0):
      Outgoing f_7 (SW):  α to C's f_5  | β to west's f_6  | γ to west's f_2
      Outgoing f_8 (SE):  α to C's f_6  | β to east's f_5  | γ to east's f_1

    This function OVERWRITES the diagonal slots at top/bottom non-corner cells
    (replacing whatever HBB wrote there), then ADDS the γ horizontal share to
    adjacent cells' cardinal slots.
    """
    # ─── TOP WALL non-corner (j = NY-1, i ∈ [1, NX-2]) ───
    # f[7, i, NY-1] = α · own f_5  +  β · east's f_6
    # f[8, i, NY-1] = α · own f_6  +  β · west's f_5
    f[7, 1:NX-1, NY-1] = (alpha * f_star[5, 1:NX-1, NY-1]
                          + beta * f_star[6, 2:NX, NY-1])
    f[8, 1:NX-1, NY-1] = (alpha * f_star[6, 1:NX-1, NY-1]
                          + beta * f_star[5, 0:NX-2, NY-1])
    # γ horizontal share — ADD to adjacent cardinal slots
    f[1, 2:NX, NY-1] += gamma * f_star[5, 1:NX-1, NY-1]
    f[2, 0:NX-2, NY-1] += gamma * f_star[6, 1:NX-1, NY-1]

    # ─── BOTTOM WALL non-corner (j = 0, i ∈ [1, NX-2]) ───
    # f[5, i, 0] = α · own f_7  +  β · west's f_8
    # f[6, i, 0] = α · own f_8  +  β · east's f_7
    f[5, 1:NX-1, 0] = (alpha * f_star[7, 1:NX-1, 0]
                       + beta * f_star[8, 0:NX-2, 0])
    f[6, 1:NX-1, 0] = (alpha * f_star[8, 1:NX-1, 0]
                       + beta * f_star[7, 2:NX, 0])
    # γ horizontal share at bottom
    f[1, 2:NX, 0] += gamma * f_star[8, 1:NX-1, 0]
    f[2, 0:NX-2, 0] += gamma * f_star[7, 1:NX-1, 0]

    return f


def lbm_step_weighted(f: torch.Tensor, V: torch.Tensor, R: torch.Tensor,
                       dt: float, omega: float, w: torch.Tensor,
                       bounce_masks_full: dict, alpha: float, beta: float,
                       gamma: float, NX: int, NY: int) -> tuple:
    """One D2Q9-BGK step with weighted BC family at top/bottom non-corner cells."""
    f = bgk_collide(f, V, R, dt, omega, w)
    f_star = f.clone()
    f = stream_d2q9(f)
    f = apply_neumann_d2q9(f, f_star, bounce_masks_full)
    f = apply_weighted_top_bottom_d2q9(f, f_star, alpha, beta, gamma, NX, NY)
    V = recover_voltage(f)
    return f, V


def main():
    device = torch.device("cpu")
    ionic = TTP06Model(cell_type=CellType.EPI, device=device)
    V_rest = float(ionic.V_rest)

    sim = LBMSimulation(
        Nx=NX, Ny=NY, dx=DX, dt=DT, D=D,
        ionic_model=ionic, Cm=1.0,
        lattice="d2q9", weights_mode=WEIGHTS_MODE,
    )

    # IC: V[col 0] = V_stim (leftmost wall column), V[else] = V_rest.
    # f = w·V (rest equilibrium).
    V_init = torch.full((NX, NY), V_rest, dtype=sim.dtype, device=device)
    V_init[0, :] = V_STIM
    sim.V = V_init
    sim.f = sim.w[:, None, None] * sim.V[None, :, :]

    # Use the full bounce_masks for HBB everywhere. Specular overwrites
    # top/bottom non-corner diagonals after the HBB pass.
    bounce_masks_full = sim.bounce_masks

    n_steps = int(round(T_END / DT))
    print(f"Grid: {NX} × {NY}   dx={DX} cm   dt={DT} ms   t_end={T_END} ms ({n_steps} steps)")
    print(f"Lattice: D2Q9 {WEIGHTS_MODE} + BGK   V_rest={V_rest:.3f} mV   V_stim={V_STIM} mV")
    print(f"Top/bottom walls: {BC_MODE.upper()}.  East/west walls: HBB.  Corners: HBB.")

    V_hist = np.empty((n_steps + 1, NX, NY), dtype=np.float64)
    t_hist = np.empty(n_steps + 1, dtype=np.float64)
    V_hist[0] = sim.V.cpu().numpy()
    t_hist[0] = 0.0

    t0 = time.time()
    f = sim.f
    V = sim.V
    for k in range(1, n_steps + 1):
        I_stim = torch.zeros(NX, NY, device=device, dtype=sim.dtype)
        I_ion = sim.ionic_model.compute_Iion(V.reshape(-1), sim.ionic_states)
        R_flat = compute_source_term(I_ion, I_stim.reshape(-1), sim.Cm)
        R = R_flat.reshape(NX, NY)
        if BC_MODE == "specular":
            f, V = lbm_step_specular(
                f, V, R, sim.dt, sim.omega, sim.w,
                bounce_masks_full, NX, NY,
            )
        elif BC_MODE == "horizontal":
            f, V = lbm_step_horizontal(
                f, V, R, sim.dt, sim.omega, sim.w,
                bounce_masks_full, NX, NY,
            )
        elif BC_MODE == "weighted":
            f, V = lbm_step_weighted(
                f, V, R, sim.dt, sim.omega, sim.w,
                bounce_masks_full, ALPHA, BETA, GAMMA, NX, NY,
            )
        else:  # hbb baseline
            f = bgk_collide(f, V, R, sim.dt, sim.omega, sim.w)
            f_star = f.clone()
            f = stream_d2q9(f)
            f = apply_neumann_d2q9(f, f_star, bounce_masks_full)
            V = recover_voltage(f)
        sim.ionic_states = ionic_step(sim.ionic_model, V.reshape(-1), sim.ionic_states, sim.dt)
        V_hist[k] = V.cpu().numpy()
        t_hist[k] = k * DT
    elapsed = time.time() - t0

    x_coords = (np.arange(NX) * DX).astype(np.float64)
    y_coords = (np.arange(NY) * DX).astype(np.float64)

    with h5py.File(OUT_FILE, "w") as fp:
        fp.create_dataset("V", data=V_hist, compression="gzip", compression_opts=4)
        fp.create_dataset("t", data=t_hist)
        fp.create_dataset("x", data=x_coords)
        fp.create_dataset("y", data=y_coords)
        a = fp.attrs
        a["engine"] = "LBM_V1_custom"
        a["lattice"] = "d2q9"
        a["weights_mode"] = WEIGHTS_MODE
        a["boundary_treatment_y"] = BC_MODE
        a["boundary_treatment_x"] = "halfway_bounce_back"
        a["physics"] = f"ttp06_natural_{BC_MODE}_y"
        if BC_MODE == "weighted":
            a["alpha"] = ALPHA
            a["beta"] = BETA
            a["gamma"] = GAMMA
        a["dx"] = DX
        a["dt"] = DT
        a["D"] = D
        a["V_stim"] = V_STIM
        a["V_rest"] = V_rest
        a["NX"] = NX
        a["NY"] = NY
        a["t_end"] = T_END
        a["n_steps"] = n_steps
        a["stim_col"] = 0

    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n  wrote {OUT_FILE.name}  ({size_mb:.1f} MB)  "
          f"V range [{V_hist.min():.3f}, {V_hist.max():.3f}] mV  elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
