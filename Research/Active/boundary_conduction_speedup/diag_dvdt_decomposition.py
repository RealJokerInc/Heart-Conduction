"""
dV/dt boundary-vs-center diagnostic — 4 monodomain V5.4 simulations.

Tests whether the boundary deficit under (moore8_uniform + face_mirror) appears
*instantaneously* from step 1 (operator-level structural deficit) versus
develops over the ramp-up window. Two BC modes × two physics regimes = 4 cases.

All simulations:
  - Stencil: moore8_uniform (deficit-producing 9-pt)
  - Grid:    NX=41, NY=21, dx=0.025 cm
  - IC:      V[i=1, :] = V_stim;  V[else] = V_rest
  - dt=0.01 ms, log every step (no save_every undersampling)
  - No clamp anywhere — natural evolution from IC

Cases:
  1. face_mirror      + diffusion only   t_end=50 ms
  2. face_mirror_iso  + diffusion only   t_end=50 ms
  3. face_mirror      + diffusion+TTP06  t_end=25 ms
  4. face_mirror_iso  + diffusion+TTP06  t_end=25 ms

Output:
  data/case1_fm_diff.h5   — face_mirror      + diffusion only
  data/case2_fmi_diff.h5  — face_mirror_iso  + diffusion only
  data/case3_fm_ttp06.h5  — face_mirror      + diffusion+TTP06
  data/case4_fmi_ttp06.h5 — face_mirror_iso  + diffusion+TTP06

Each HDF5:
  /V        (T, NX, NY) float64   — voltage every step
  /t        (T,)        float64   — time [ms]
  /x        (NX,)       float64   — x coords [cm]
  /y        (NY,)       float64   — y coords [cm]
  attrs: stencil, boundary_mode, physics, dx, dt, D, V_stim, V_rest,
         NX, NY, t_end, n_steps

Usage:
  python diag_dvdt_decomposition.py            # run all 4 cases
  python diag_dvdt_decomposition.py 1          # run case N only (1..4)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch
import h5py

ENGINE = Path("/home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4")
sys.path.insert(0, str(ENGINE))

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation


# ---------- config ----------
LX, LY = 1.0, 0.5             # cm
DX = 0.025                    # cm  (square grid required for moore8)
NX = int(round(LX / DX)) + 1  # 41
NY = int(round(LY / DX)) + 1  # 21
DT = 0.01                     # ms
D = 0.001                     # cm² / ms
V_STIM = 0.0                  # mV (depolarized strip in column 0, leftmost wall)
V_REST = -86.2                # mV (TTP06 EPI default)
STENCIL = "moore8_uniform"
STIM_COL = 0                  # i index of activator strip (leftmost wall column)

# Case 5 — synchronous cold start (cols 1+2+3 clamped at AP-firing V)
SYNC_COLS  = (1, 2, 3)
V_AP_TRIG  = 30.0             # mV  (well above threshold)
SYNC_STEPS = 10               # number of steps to hold the clamp (0.1 ms at dt=0.01)

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CASES = [
    # (idx, name, boundary_mode,    physics,             t_end_ms,  out_file)
    (1, "fm_diff",   "face_mirror",     "diffusion",         50.0,  "case1_fm_diff.h5"),
    (2, "fmi_diff",  "face_mirror_iso", "diffusion",         50.0,  "case2_fmi_diff.h5"),
    (3, "fm_ttp06",  "face_mirror",     "ttp06",             25.0,  "case3_fm_ttp06.h5"),
    (4, "fmi_ttp06", "face_mirror_iso", "ttp06",             25.0,  "case4_fmi_ttp06.h5"),
    (5, "fm_ttp06_synchap_cols123",
                     "face_mirror",     "ttp06_synchap",     25.0,  "case5_fm_ttp06_synchap_cols123.h5"),
    (6, "fm_ttp06_apfirst_cols123",
                     "face_mirror",     "ttp06_apfirst",     25.0,  "case6_fm_ttp06_apfirst_cols123.h5"),
    (7, "fm_ttp06_apfirst_invcrescent",
                     "face_mirror",     "ttp06_apfirst_invcrescent",
                                                              25.0,  "case7_fm_ttp06_apfirst_invcrescent.h5"),
]


def make_fdm(boundary_mode: str) -> tuple[FDMDiscretization, StructuredGrid]:
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(
        grid, D=D, chi=1.0, Cm=1.0,
        stencil=STENCIL, boundary_mode=boundary_mode,
    )
    return fdm, grid


def initial_V_2d() -> torch.Tensor:
    """(NX, NY) tensor: column i=STIM_COL at V_stim, everything else at V_rest."""
    V = torch.full((NX, NY), V_REST, dtype=torch.float64)
    V[STIM_COL, :] = V_STIM
    return V


def run_diffusion_only(boundary_mode: str, t_end: float):
    """Hand-rolled forward-Euler loop over fdm.apply_diffusion.

    No ionic model, no stimulus, no clamp. Pure dV/dt = (1/(chi·Cm))·L·V.
    """
    fdm, grid = make_fdm(boundary_mode)
    V_2d = initial_V_2d()
    V_flat = V_2d.reshape(-1).clone()

    n_steps = int(round(t_end / DT))
    # Pre-allocate trajectory: (n_steps + 1, NX, NY) — include t=0 frame
    V_hist = np.empty((n_steps + 1, NX, NY), dtype=np.float64)
    t_hist = np.empty(n_steps + 1, dtype=np.float64)
    V_hist[0] = V_2d.numpy()
    t_hist[0] = 0.0

    for k in range(1, n_steps + 1):
        dV = fdm.apply_diffusion(V_flat)
        V_flat = V_flat + DT * dV
        V_hist[k] = V_flat.reshape(NX, NY).numpy()
        t_hist[k] = k * DT

    x_coords = (np.arange(NX) * DX).astype(np.float64)
    y_coords = (np.arange(NY) * DX).astype(np.float64)
    return t_hist, V_hist, x_coords, y_coords


def run_ttp06(boundary_mode: str, t_end: float):
    """MonodomainSimulation with TTP06 ionic + diffusion.

    NO stimulus protocol — we poke state.V[i=STIM_COL,:] = V_stim once before running.
    Gates stay at their resting values at t=0; they respond as V evolves.
    """
    fdm, grid = make_fdm(boundary_mode)
    sim = MonodomainSimulation(
        spatial=fdm,
        ionic_model='ttp06',
        stimulus=None,           # no current injection
        dt=DT,
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='forward_euler',
        cell_type='EPI',
    )
    # Poke IC: V[i=STIM_COL, :] = V_STIM
    V_init = sim.get_voltage().reshape(NX, NY).clone()
    V_init[STIM_COL, :] = V_STIM
    sim.set_voltage(V_init.reshape(-1))

    n_steps = int(round(t_end / DT))
    V_hist = np.empty((n_steps + 1, NX, NY), dtype=np.float64)
    t_hist = np.empty(n_steps + 1, dtype=np.float64)
    V_hist[0] = sim.get_voltage().reshape(NX, NY).cpu().numpy()
    t_hist[0] = 0.0

    for k in range(1, n_steps + 1):
        sim.step(DT)
        V_hist[k] = sim.get_voltage().reshape(NX, NY).cpu().numpy()
        t_hist[k] = k * DT

    x_coords = (np.arange(NX) * DX).astype(np.float64)
    y_coords = (np.arange(NY) * DX).astype(np.float64)
    return t_hist, V_hist, x_coords, y_coords


def run_ttp06_synchap(boundary_mode: str, t_end: float):
    """TTP06 with synchronous cold-start AP at cols 1+2+3.

    IC: V[1,2,3, :] = V_AP_TRIG (+30 mV), V[else] = V_rest.
    Clamp window: for the first SYNC_STEPS, force V[1,2,3, :] = V_AP_TRIG
    after each Strang step. After window, release — natural TTP06 evolution.

    No upstream column ever experiences a sub-threshold ramp-up → cannot
    inherit any LAT shift from face_mirror echoes. The diagnostic question
    is whether col 4 (the first downstream column charging from V_rest)
    develops a crescent purely from per-step operator deficit at col 4.
    """
    fdm, grid = make_fdm(boundary_mode)
    sim = MonodomainSimulation(
        spatial=fdm,
        ionic_model='ttp06',
        stimulus=None,
        dt=DT,
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='forward_euler',
        cell_type='EPI',
    )
    V_init = sim.get_voltage().reshape(NX, NY).clone()
    for col in SYNC_COLS:
        V_init[col, :] = V_AP_TRIG
    sim.set_voltage(V_init.reshape(-1))

    n_steps = int(round(t_end / DT))
    V_hist = np.empty((n_steps + 1, NX, NY), dtype=np.float64)
    t_hist = np.empty(n_steps + 1, dtype=np.float64)
    V_hist[0] = sim.get_voltage().reshape(NX, NY).cpu().numpy()
    t_hist[0] = 0.0

    for k in range(1, n_steps + 1):
        sim.step(DT)
        if k <= SYNC_STEPS:
            V_state = sim.get_voltage().reshape(NX, NY).clone()
            for col in SYNC_COLS:
                V_state[col, :] = V_AP_TRIG
            sim.set_voltage(V_state.reshape(-1))
        V_hist[k] = sim.get_voltage().reshape(NX, NY).cpu().numpy()
        t_hist[k] = k * DT

    x_coords = (np.arange(NX) * DX).astype(np.float64)
    y_coords = (np.arange(NY) * DX).astype(np.float64)
    return t_hist, V_hist, x_coords, y_coords


def run_ttp06_apfirst(boundary_mode: str, t_end: float):
    """TTP06 strict AP-first variant — diffusion frozen during sync window.

    IC: V[1,2,3, :] = V_AP_TRIG (+30 mV), V[else] = V_rest.
    Sync window (first SYNC_STEPS): IONIC SOLVER ONLY (no diffusion at all
    anywhere). Cols 1-3 clamped at V_AP_TRIG after each ionic step so the
    AP cascade locks in y-uniformly. Cols 0 and 4+ stay at V_rest because
    ionic at V_rest is a no-op (cells already at gate equilibria).

    After sync window: full Strang (ionic + diffusion). Col 4 charges
    from V_rest with cols 1-3 in plateau (y-uniform). The "diffusion
    starts only when AP starts" condition is now strict.
    """
    fdm, grid = make_fdm(boundary_mode)
    sim = MonodomainSimulation(
        spatial=fdm,
        ionic_model='ttp06',
        stimulus=None,
        dt=DT,
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='forward_euler',
        cell_type='EPI',
    )
    V_init = sim.get_voltage().reshape(NX, NY).clone()
    for col in SYNC_COLS:
        V_init[col, :] = V_AP_TRIG
    sim.set_voltage(V_init.reshape(-1))

    ionic_solver = sim.splitting.ionic_solver

    n_steps = int(round(t_end / DT))
    V_hist = np.empty((n_steps + 1, NX, NY), dtype=np.float64)
    t_hist = np.empty(n_steps + 1, dtype=np.float64)
    V_hist[0] = sim.get_voltage().reshape(NX, NY).cpu().numpy()
    t_hist[0] = 0.0

    for k in range(1, n_steps + 1):
        if k <= SYNC_STEPS:
            ionic_solver.step(sim.state, DT)
            sim.state.t += DT
            V_state = sim.get_voltage().reshape(NX, NY).clone()
            for col in SYNC_COLS:
                V_state[col, :] = V_AP_TRIG
            sim.set_voltage(V_state.reshape(-1))
        else:
            sim.step(DT)
        V_hist[k] = sim.get_voltage().reshape(NX, NY).cpu().numpy()
        t_hist[k] = k * DT

    x_coords = (np.arange(NX) * DX).astype(np.float64)
    y_coords = (np.arange(NY) * DX).astype(np.float64)
    return t_hist, V_hist, x_coords, y_coords


def run_ttp06_apfirst_invcrescent(boundary_mode: str, t_end: float):
    """TTP06 strict AP-first with INVERSE-CRESCENT clamp shape.

    IC and sync-window clamp:
      interior rows (j ∈ [1, NY-2]):   cols 1,2,3 at V_AP_TRIG
      boundary rows (j=0 and j=NY-1):  cols 1,2,3, AND 4 at V_AP_TRIG
                                       (one extra column ahead at the wall)

    Sync window: ionic-only (no diffusion) for SYNC_STEPS, clamping the
    pattern above after each ionic step. After window: full Strang.

    Hypothesis being tested:
      We're imposing an inverse crescent (boundaries advanced 1 column).
      face_mirror has a structural source-effect deficit at every column
      charging from rest at the wall. Will the deficit eat the artificial
      lead and flip the wavefront back to forward crescent over enough
      propagation distance — or does the imposed lead persist?
    """
    fdm, grid = make_fdm(boundary_mode)
    sim = MonodomainSimulation(
        spatial=fdm,
        ionic_model='ttp06',
        stimulus=None,
        dt=DT,
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='forward_euler',
        cell_type='EPI',
    )

    # IC: cols 1,2,3 all rows at AP_TRIG; col 4 at j=0 and j=NY-1 only
    V_init = sim.get_voltage().reshape(NX, NY).clone()
    for col in SYNC_COLS:           # (1, 2, 3)
        V_init[col, :] = V_AP_TRIG
    V_init[4, 0]      = V_AP_TRIG   # extra cell at top boundary
    V_init[4, NY - 1] = V_AP_TRIG   # extra cell at bottom boundary
    sim.set_voltage(V_init.reshape(-1))

    ionic_solver = sim.splitting.ionic_solver

    n_steps = int(round(t_end / DT))
    V_hist = np.empty((n_steps + 1, NX, NY), dtype=np.float64)
    t_hist = np.empty(n_steps + 1, dtype=np.float64)
    V_hist[0] = sim.get_voltage().reshape(NX, NY).cpu().numpy()
    t_hist[0] = 0.0

    for k in range(1, n_steps + 1):
        if k <= SYNC_STEPS:
            ionic_solver.step(sim.state, DT)
            sim.state.t += DT
            V_state = sim.get_voltage().reshape(NX, NY).clone()
            for col in SYNC_COLS:
                V_state[col, :] = V_AP_TRIG
            V_state[4, 0]      = V_AP_TRIG
            V_state[4, NY - 1] = V_AP_TRIG
            sim.set_voltage(V_state.reshape(-1))
        else:
            sim.step(DT)
        V_hist[k] = sim.get_voltage().reshape(NX, NY).cpu().numpy()
        t_hist[k] = k * DT

    x_coords = (np.arange(NX) * DX).astype(np.float64)
    y_coords = (np.arange(NY) * DX).astype(np.float64)
    return t_hist, V_hist, x_coords, y_coords


def save_h5(out_path: Path, t, V, x, y, *, boundary_mode, physics, t_end):
    with h5py.File(out_path, "w") as f:
        f.create_dataset("V", data=V, compression="gzip", compression_opts=4)
        f.create_dataset("t", data=t)
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        a = f.attrs
        a["stencil"] = STENCIL
        a["boundary_mode"] = boundary_mode
        a["physics"] = physics
        a["dx"] = DX
        a["dt"] = DT
        a["D"] = D
        a["V_stim"] = V_STIM
        a["V_rest"] = V_REST
        a["NX"] = NX
        a["NY"] = NY
        a["t_end"] = t_end
        a["n_steps"] = len(t) - 1
        a["stim_col"] = STIM_COL
        if physics in ("ttp06_synchap", "ttp06_apfirst", "ttp06_apfirst_invcrescent"):
            a["sync_cols"] = np.array(SYNC_COLS, dtype=np.int64)
            a["v_ap_trigger"] = V_AP_TRIG
            a["sync_steps"] = SYNC_STEPS
            a["diffusion_during_sync"] = (physics == "ttp06_synchap")
            a["inverse_crescent"] = (physics == "ttp06_apfirst_invcrescent")


def run_case(idx, name, boundary_mode, physics, t_end, out_file):
    print(f"\n[case {idx}] {name}  bc={boundary_mode}  physics={physics}  "
          f"t_end={t_end} ms  ({int(t_end/DT)} steps)", flush=True)
    t0 = time.time()
    if physics == "diffusion":
        t, V, x, y = run_diffusion_only(boundary_mode, t_end)
    elif physics == "ttp06":
        t, V, x, y = run_ttp06(boundary_mode, t_end)
    elif physics == "ttp06_synchap":
        t, V, x, y = run_ttp06_synchap(boundary_mode, t_end)
    elif physics == "ttp06_apfirst":
        t, V, x, y = run_ttp06_apfirst(boundary_mode, t_end)
    elif physics == "ttp06_apfirst_invcrescent":
        t, V, x, y = run_ttp06_apfirst_invcrescent(boundary_mode, t_end)
    else:
        raise ValueError(physics)
    elapsed = time.time() - t0
    out_path = OUT_DIR / out_file
    save_h5(out_path, t, V, x, y,
            boundary_mode=boundary_mode, physics=physics, t_end=t_end)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"           wrote {out_path.name}  ({size_mb:.1f} MB)  "
          f"V range [{V.min():.3f}, {V.max():.3f}] mV  elapsed {elapsed:.1f}s")


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Grid: {NX} × {NY} = {NX*NY} cells   dx={DX} cm   dt={DT} ms")
    print(f"IC: V[i={STIM_COL}, :] = {V_STIM} mV ; V[else] = {V_REST} mV")
    print(f"Stencil: {STENCIL}    Output: {OUT_DIR}")
    if sel is None:
        for case in CASES:
            run_case(*case)
    else:
        idx = int(sel)
        for case in CASES:
            if case[0] == idx:
                run_case(*case)
                break
        else:
            raise SystemExit(f"unknown case {idx}")


if __name__ == "__main__":
    main()
