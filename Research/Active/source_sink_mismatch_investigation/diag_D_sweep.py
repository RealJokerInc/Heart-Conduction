"""
D-sweep (LBM): how does the electrotonic foot-in-cells and the expansion
source-sink scale with the diffusion coefficient? CV in cm/s is not tracked
(it's a dx/dt labeling choice); the foot in CELLS is the order parameter.

Geometry: thin strand (4 cells) -> abrupt wide expansion at x=3 cm.
Per D: foot-ahead (cells), wave reached-x (block?), and the downstream
lateral-fill delay (corner LAT - center LAT) = source-sink strength.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/norepinephrine/Documents/Heart-Conduction")
sys.path.insert(0, str(REPO / "LBM/Engine_V1"))
from src.collision.bgk import bgk_collide
from src.streaming.d2q9 import stream_d2q9
from src.boundary.masks import precompute_bounce_masks
from src.boundary.neumann import apply_neumann_d2q9
from src.state import recover_voltage
from src.solver.rush_larsen import compute_source_term, ionic_step
from src.diffusion import tau_from_D
from src.lattice import D2Q9
from ionic.ttp06.model import TTP06Model
from ionic.base import CellType

LX, LY, DX = 6.0, 4.0, 0.025
NX = int(round(LX / DX)) + 1
NY = int(round(LY / DX)) + 1
DT = 0.02
T_END = 130.0
SAVE_EVERY = 0.5
YC = LY / 2
W_STRAND = 0.05          # half-width of thin strand (4 cells)
W_WIDE = 1.5
EXPAND_X = 3.0
VTHR = -40.0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strand_expansion():
    ix = torch.arange(NX, device=DEV).view(NX, 1).double() * DX
    iy = torch.arange(NY, device=DEV).view(1, NY).double() * DX
    hw = torch.where(ix < EXPAND_X,
                     torch.tensor(W_STRAND, device=DEV, dtype=torch.float64),
                     torch.tensor(W_WIDE, device=DEV, dtype=torch.float64))
    return torch.abs(iy - YC) <= hw


def lbm_run(domain, D):
    lat = D2Q9(); solid = ~domain
    obm = precompute_bounce_masks(domain, lat); bm = {}
    for a in range(1, 9):
        m = torch.zeros(NX, NY, dtype=torch.bool, device=DEV); ex, ey = lat.e[a]
        if ex == 1: m[-1, :] = True
        if ex == -1: m[0, :] = True
        if ey == 1: m[:, -1] = True
        if ey == -1: m[:, 0] = True
        bm[a] = obm[a] | m
    ionic = TTP06Model(cell_type=CellType.EPI, device=DEV); V_rest = float(ionic.V_rest)
    w = torch.tensor(lat.w, dtype=torch.float64, device=DEV)
    omega = 1.0 / tau_from_D(D, DX, DT, cs2=lat.cs2)
    V = torch.full((NX, NY), V_rest, dtype=torch.float64, device=DEV)
    V[:2, :] = 20.0; V[solid] = V_rest
    f = w[:, None, None] * V[None, :, :]
    states = ionic.get_initial_state(n_cells=NX * NY)
    Iz = torch.zeros(NX * NY, dtype=torch.float64, device=DEV)
    n = int(round(T_END / DT)); siv = int(round(SAVE_EVERY / DT))
    fr = [V.cpu().numpy().astype(np.float32)]; tt = [0.0]
    for k in range(1, n + 1):
        I_ion = ionic.compute_Iion(V.reshape(-1), states)
        R = compute_source_term(I_ion, Iz, 1.0).reshape(NX, NY)
        f = bgk_collide(f, V, R, DT, omega, w); fs = f.clone()
        f = stream_d2q9(f); f = apply_neumann_d2q9(f, fs, bm)
        f[:, solid] = w[:, None] * V_rest; V = recover_voltage(f)
        states = ionic_step(ionic, V.reshape(-1), states, DT)
        if k % siv == 0:
            fr.append(V.cpu().numpy().astype(np.float32)); tt.append(k * DT)
    return np.array(tt), np.array(fr), V_rest


def lat_field(V, times):
    above = V >= VTHR; ever = above.any(0); idx = np.argmax(above, axis=0)
    idxc = np.clip(idx, 1, len(times) - 1)
    v1 = np.take_along_axis(V, idxc[None], 0)[0]; v0 = np.take_along_axis(V, (idxc - 1)[None], 0)[0]
    t = np.asarray(times); denom = np.where(v1 == v0, 1.0, v1 - v0)
    lat = t[idxc - 1] + (VTHR - v0) * (t[idxc] - t[idxc - 1]) / denom
    lat[idx == 0] = times[0]; lat[~ever] = np.nan
    return lat


def analyze(times, V, V_rest):
    x = np.arange(NX) * DX; jc = int(round(YC / DX))
    # foot: snapshot with strand front near x=1.5 cm
    rows = V[:, :, jc]
    fronts = np.array([x[np.where(r >= VTHR)[0].max()] if (r >= VTHR).any() else np.nan for r in rows])
    k = int(np.nanargmin(np.abs(fronts - 1.5))); r = rows[k]; xf = fronts[k]
    foot_cells = (x[(x > xf) & (r >= V_rest + 2)].max() - xf) / DX if (r >= V_rest + 2)[x > xf].any() else 0.0
    lat = lat_field(V, times)
    reached = x[np.isfinite(lat[:, jc])].max() if np.isfinite(lat[:, jc]).any() else 0.0
    blocked = reached < LX - 0.3
    # lateral-fill spread downstream of expansion: across the channel cross-section
    # at x=5 cm, max-min LAT (corners activate late under radial collapse)
    ix5 = int(5.0 / DX)
    col = lat[ix5, :]; col = col[np.isfinite(col)]
    delay = (col.max() - col.min()) if col.size else np.nan
    return foot_cells, reached, blocked, delay


def main():
    print(f"D-sweep, strand={2*W_STRAND/DX:.0f} cells -> wide, expansion at x={EXPAND_X}cm\n")
    print(f"{'D':>8} {'foot(cells)':>12} {'reached_x(cm)':>14} {'block?':>8} {'lat-fill delay(ms)':>20}")
    for D in (0.001, 0.002, 0.004, 0.008, 0.016):
        t, V, vr = lbm_run(strand_expansion(), D)
        foot, reached, blocked, delay = analyze(t, V, vr)
        print(f"{D:>8.4f} {foot:>12.1f} {reached:>14.2f} {str(blocked):>8} {delay:>20.1f}", flush=True)


if __name__ == "__main__":
    main()
