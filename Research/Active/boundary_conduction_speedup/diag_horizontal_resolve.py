"""Clean, self-contained resolution of the contradiction:
single-step shows redirect changes wall V; long manual loop showed 0.

Use the EXACT production lbm_step_horizontal from diag_lbm_specular, plus an
HBB control, uniform IC, no stim, real omega. Print wall V every 20 steps.
Write all output to a file (terminal rendering has been unreliable).
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, "/home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1")
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import LBMSimulation
from src.collision.bgk import bgk_collide
from src.streaming.d2q9 import stream_d2q9
from src.boundary.neumann import apply_neumann_d2q9
from src.state import recover_voltage
from ionic.ttp06.model import TTP06Model
from ionic.base import CellType
from diag_lbm_specular import (
    apply_horizontal_redirect_top_bottom_d2q9,
    lbm_step_horizontal,
)

NX, NY = 41, 21
DX, DT, D = 0.025, 0.02, 0.001
ionic = TTP06Model(cell_type=CellType.EPI, device=torch.device("cpu"))
V_rest = float(ionic.V_rest)

OUT = []


def emit(s):
    OUT.append(s)


def run_production_step(bc):
    """Use the production lbm_step_horizontal (or HBB equivalent)."""
    sim = LBMSimulation(Nx=NX, Ny=NY, dx=DX, dt=DT, D=D, ionic_model=ionic,
                        Cm=1.0, lattice="d2q9", weights_mode="canonical")
    w = sim.w
    V = torch.full((NX, NY), V_rest, dtype=sim.dtype)
    f = w[:, None, None] * V[None, :, :]
    bounce = sim.bounce_masks
    R = torch.zeros(NX, NY, dtype=sim.dtype)
    emit(f"  omega={float(sim.omega):.6f}  tau={1.0/float(sim.omega):.6f}")
    emit(f"  step  V(wall,c20)   V(sub,c20)   wall_mean   mass_sum")
    emit(f"  {0:4d}  {float(V[20,0]):+11.5f}  {float(V[20,1]):+11.5f}  "
         f"{float(V[:,0].mean()):+10.5f}  {float(V.sum()):+12.3f}")
    for k in range(1, 1251):
        if bc == "horizontal":
            f, V = lbm_step_horizontal(f, V, R, sim.dt, sim.omega, w, bounce, NX, NY)
        else:  # hbb
            f = bgk_collide(f, V, R, sim.dt, sim.omega, w)
            fs = f.clone()
            f = stream_d2q9(f)
            f = apply_neumann_d2q9(f, fs, bounce)
            V = recover_voltage(f)
        if k in (40, 250, 500, 1250):
            emit(f"  {k:4d}  {float(V[20,0]):+11.5f}  {float(V[20,1]):+11.5f}  "
                 f"{float(V[:,0].mean()):+10.5f}  {float(V.sum()):+12.3f}")
    return float(V[:, 0].mean()) - V_rest


def run_with_stim(bc, n_steps=1250):
    """Col-0 stim IC, TTP06, production step. Check wall pre-charge claim."""
    from src.solver.rush_larsen import compute_source_term, ionic_step
    sim = LBMSimulation(Nx=NX, Ny=NY, dx=DX, dt=DT, D=D, ionic_model=ionic,
                        Cm=1.0, lattice="d2q9", weights_mode="canonical")
    w = sim.w
    V = torch.full((NX, NY), V_rest, dtype=sim.dtype)
    V[0, :] = 0.0
    f = w[:, None, None] * V[None, :, :]
    bounce = sim.bounce_masks
    emit(f"  step  V(wall,c20)  V(ctr,c20)  V(wall,c10)  V(wall,c30)")
    for k in range(1, n_steps + 1):
        I_ion = sim.ionic_model.compute_Iion(V.reshape(-1), sim.ionic_states)
        R = compute_source_term(I_ion, torch.zeros(NX * NY, dtype=sim.dtype),
                                sim.Cm).reshape(NX, NY)
        if bc == "horizontal":
            f, V = lbm_step_horizontal(f, V, R, sim.dt, sim.omega, w, bounce, NX, NY)
        else:
            f = bgk_collide(f, V, R, sim.dt, sim.omega, w)
            fs = f.clone(); f = stream_d2q9(f); f = apply_neumann_d2q9(f, fs, bounce)
            V = recover_voltage(f)
        sim.ionic_states = ionic_step(sim.ionic_model, V.reshape(-1),
                                      sim.ionic_states, sim.dt)
        if k in (25, 50, 100, 250, 500, 1250):
            emit(f"  {k:4d}  {float(V[20,0]):+10.4f}  {float(V[20,NY//2]):+10.4f}  "
                 f"{float(V[10,0]):+10.4f}  {float(V[30,0]):+10.4f}")


emit("=== PRODUCTION lbm_step_horizontal, uniform IC, NO STIM, R=0 (1250 steps) ===")
emit("\n--- HBB control ---")
d_hbb = run_production_step("hbb")
emit(f"  final Δwall = {d_hbb:+.5f} mV")
emit("\n--- horizontal redirect ---")
d_hor = run_production_step("horizontal")
emit(f"  final Δwall = {d_hor:+.5f} mV")

emit("\n\n=== PRODUCTION step, COL-0 STIM, TTP06 — is the -69mV wall pre-charge real? ===")
emit("\n--- HBB ---")
run_with_stim("hbb")
emit("\n--- horizontal ---")
run_with_stim("horizontal")

LOG = Path(__file__).parent / "data" / "resolve_log.txt"
LOG.write_text("\n".join(OUT) + "\n")
print(f"wrote {LOG}")
