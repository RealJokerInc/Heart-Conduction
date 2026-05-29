"""Exhaustive enumeration of symmetric mass-conserving wall-diagonal
redirect rules for D2Q9.

At a top-wall cell, the two outgoing diagonals f_5 (NE) and f_6 (NW) each
need a destination. By x-mirror symmetry, f_6 mirrors whatever f_5 does.
So every symmetric rule is parameterized by:
  - slot5: which of the 9 velocity slots f_5's mass lands in
  - dx5:   destination cell offset (-1 west, 0 same, +1 east)
f_6 -> x-mirror(slot5) at -dx5. Bottom wall is the y-mirror.

The cardinal f_3 (N) is left to HBB (handled by apply_neumann).

Classification by destination y-sign:
  -y slot (S,SW,SE)  -> mass leaves wall into interior  (reflection)
  x  slot (E,W)      -> mass stays on wall row           (horizontal trap)
  +y slot (N,NE,NW)  -> mass re-hits wall                (specular_up trap)

For each rule: no-stim diffusion artifact (Δwall, mass drift) +
with-stim TTP06 crescent (LAT@col38, wall precharge).

Output: data/wall_enumeration.txt
"""
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, "/home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1")
from src.simulation import LBMSimulation
from src.collision.bgk import bgk_collide
from src.streaming.d2q9 import stream_d2q9
from src.boundary.neumann import apply_neumann_d2q9
from src.state import recover_voltage
from src.solver.rush_larsen import compute_source_term, ionic_step
from ionic.ttp06.model import TTP06Model
from ionic.base import CellType

NX, NY = 41, 21
DX, DT, D = 0.025, 0.02, 0.001
ionic = TTP06Model(cell_type=CellType.EPI, device=torch.device("cpu"))
V_rest = float(ionic.V_rest)

E = {0:(0,0),1:(1,0),2:(-1,0),3:(0,1),4:(0,-1),5:(1,1),6:(-1,1),7:(-1,-1),8:(1,-1)}
NAME = {0:'0',1:'E',2:'W',3:'N',4:'S',5:'NE',6:'NW',7:'SW',8:'SE'}
XMIR = {0:0,1:2,2:1,3:3,4:4,5:6,6:5,7:8,8:7}
YMIR = {0:0,1:1,2:2,3:4,4:3,5:8,6:7,7:6,8:5}


def ysign_class(slot):
    ey = E[slot][1]
    if ey < 0:
        return "leaves(-y)"
    if ey == 0:
        return "stays(x)"
    return "rehits(+y)"


def apply_general(f, fs, slot5, dx5):
    slot6 = XMIR[slot5]
    dx6 = -dx5
    i = np.arange(1, NX - 1)
    # TOP wall
    f[7, 1:NX - 1, NY - 1] = 0.0
    f[8, 1:NX - 1, NY - 1] = 0.0
    for donor, slot, dxx in [(5, slot5, dx5), (6, slot6, dx6)]:
        dest = i + dxx
        valid = (dest >= 0) & (dest <= NX - 1)
        f[slot, torch.as_tensor(dest[valid]), NY - 1] += fs[donor, torch.as_tensor(i[valid]), NY - 1]
    # BOTTOM wall (y-mirror): outgoing diagonals are f_8 (SE), f_7 (SW)
    f[5, 1:NX - 1, 0] = 0.0
    f[6, 1:NX - 1, 0] = 0.0
    slot8 = YMIR[slot5]
    slot7 = YMIR[slot6]
    for donor, slot, dxx in [(8, slot8, dx5), (7, slot7, dx6)]:
        dest = i + dxx
        valid = (dest >= 0) & (dest <= NX - 1)
        f[slot, torch.as_tensor(dest[valid]), 0] += fs[donor, torch.as_tensor(i[valid]), 0]
    return f


def run(slot5, dx5, physics, n_stim, t_end):
    sim = LBMSimulation(Nx=NX, Ny=NY, dx=DX, dt=DT, D=D, ionic_model=ionic,
                        Cm=1.0, lattice="d2q9", weights_mode="canonical")
    V = torch.full((NX, NY), V_rest, dtype=sim.dtype)
    if n_stim > 0:
        V[:n_stim, :] = 0.0
    f = sim.w[:, None, None] * V[None, :, :]
    vs0 = float(V.sum())
    n_steps = int(round(t_end / DT))
    traj = []
    tarr = []
    for kk in range(1, n_steps + 1):
        if physics == "ttp06":
            I_ion = sim.ionic_model.compute_Iion(V.reshape(-1), sim.ionic_states)
            R = compute_source_term(I_ion, torch.zeros(NX * NY, dtype=sim.dtype), sim.Cm).reshape(NX, NY)
        else:
            R = torch.zeros(NX, NY, dtype=sim.dtype)
        f = bgk_collide(f, V, R, sim.dt, sim.omega, sim.w)
        fs = f.clone()
        f = stream_d2q9(f)
        f = apply_neumann_d2q9(f, fs, sim.bounce_masks)
        f = apply_general(f, fs, slot5, dx5)
        V = recover_voltage(f)
        if physics == "ttp06":
            sim.ionic_states = ionic_step(sim.ionic_model, V.reshape(-1), sim.ionic_states, sim.dt)
        if kk % 5 == 0:
            traj.append(V.cpu().numpy().copy())
            tarr.append(kk * DT)
    return V.cpu().numpy(), float(V.sum()) - vs0, np.array(traj), np.array(tarr)


def lat(Vt, t, thr=-40.0):
    a = Vt >= thr
    if not a.any():
        return float('nan')
    kk = int(np.argmax(a))
    if kk == 0:
        return t[0]
    return t[kk - 1] + (thr - Vt[kk - 1]) * (t[kk] - t[kk - 1]) / (Vt[kk] - Vt[kk - 1])


def main():
    L = []
    L.append("EXHAUSTIVE ENUMERATION of symmetric wall-diagonal redirect rules")
    L.append("f_5(NE) -> (slot @ cell-offset); f_6(NW) x-mirrored; bottom y-mirrored.")
    L.append("")
    hdr = "%-18s %-12s %14s %12s %12s %12s" % (
        "rule", "y-class", "dWall_noStim", "massDrift", "LAT_c38_us", "precharge")
    L.append(hdr)
    L.append("-" * 86)
    for slot5 in range(9):
        for dx5 in (-1, 0, 1):
            tag = "f5->%s@%+d" % (NAME[slot5], dx5)
            yc = ysign_class(slot5)
            try:
                Vn, driftn, _, _ = run(slot5, dx5, "diffusion", 0, 15.0)
            except Exception as e:
                L.append("%-18s %-12s   ERROR: %s" % (tag, yc, str(e)[:40]))
                continue
            dwall = Vn[:, 0].mean() - V_rest
            if (not np.isfinite(dwall)) or abs(driftn) > 1e3 or abs(dwall) > 1e3:
                L.append("%-18s %-12s %14s %12.1e %12s %12s" % (
                    tag, yc, "UNSTABLE", driftn, "-", "-"))
                continue
            Vs, drifts, traj, tarr = run(slot5, dx5, "ttp06", 1, 25.0)
            latb = lat(traj[:, 38, 0], tarr)
            latc = lat(traj[:, 38, NY // 2], tarr)
            diff = (latb - latc) * 1000
            k1 = int(np.argmin(np.abs(tarr - 1.0)))
            pre = traj[k1, 20, 0]
            L.append("%-18s %-12s %14.4f %12.1e %12.1f %12.2f" % (
                tag, yc, dwall, driftn, diff, pre))
    out = Path(__file__).parent / "data" / "wall_enumeration.txt"
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
