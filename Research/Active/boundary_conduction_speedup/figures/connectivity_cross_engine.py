"""Cross-engine validation of the connectivity-mediated boundary deficit.

Demonstrates that Moore-8 connectivity creates a boundary deficit in three
different model classes, and that cardinal-only / iso-with-bounce-back
fully eliminates it.

3-row × 3-col figure:
  Row 1 (Storage Tank, John's setup): R1 (moore8 uniform) | R2 (cardinal4) | R5 (moore8 iso)
  Row 2 (Monodomain V5.4):  moore8_uniform+face_mirror | cardinal4+face_mirror | moore8_iso+face_mirror_iso
  Row 3 (LBM V1):           D2Q9 uniform_8 | D2Q5 | D2Q9 canonical

All under matched line-stim geometry (left-edge inlet, propagation in +x).
Each panel uses its own colorbar (different state variables / scales per row).

Output:
  figures/connectivity_cross_engine.png
  figures/connectivity_cross_engine.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- engine paths ----------
ROOT = Path(__file__).resolve().parents[4]
SIMULATION = ROOT / "simulation"
ENGINE_V54 = ROOT / "Monodomain" / "Engine_V5.4"
LBM_V1 = ROOT / "LBM" / "Engine_V1"

# ---------- output ----------
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Row 1: storage tank
# ============================================================================

def run_storage_tank_row():
    sys.path.insert(0, str(SIMULATION))
    from configs import GRADIENT, resolve_geometry
    import tanks_vec as tv

    inlet, outlet = resolve_geometry(GRADIENT["geometry"])
    rule = GRADIENT["rule"]
    pipes = GRADIENT["pipes"]
    bc = GRADIENT["boundary"]
    geom = GRADIENT["geometry"]
    Nx, Ny = geom["Nx"], geom["Ny"]

    cases = [
        ("R1: moore8 uniform",  "moore8",     "deficit"),
        ("R2: cardinal-4",       "cardinal4",  "baseline"),
        ("R5: moore8 iso 4:1",   "moore8_iso", "partial"),
    ]
    results = []
    for label, conn, _ in cases:
        out = tv.run(
            Nx=Nx, Ny=Ny, mode=rule["type"],
            steps=GRADIENT["sim"]["steps"],
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"],
            max_volume=rule["max_volume"],
            max_pump=rule["max_pump"],
            gradient_k=rule["gradient_k"],
            directionality=pipes["directionality"],
            boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            connectivity=conn,
            threshold_gate=True,
            record_isochrone=True,
            record_history=False,
        )
        results.append((label, out["iso"]))
    sys.path.remove(str(SIMULATION))
    return results, Nx, Ny


# ============================================================================
# Row 2: monodomain V5.4
# ============================================================================

def run_monodomain_row():
    sys.path.insert(0, str(ENGINE_V54))
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation

    LX, LY = 1.0, 0.5
    DX = 0.025
    NX, NY = 41, 21
    DT = 0.02
    T_END = 12.0    # mid-propagation, before wave reaches right wall
    SAVE_EVERY = 0.5
    snapshot_t = 8.0  # ms

    cases = [
        ("moore8_uniform + face_mirror",     "moore8_uniform", "face_mirror",      "deficit"),
        ("cardinal4 + face_mirror",          "cardinal4",      "face_mirror",      "baseline"),
        ("moore8_iso + face_mirror_iso",     "moore8_iso",     "face_mirror_iso",  "fix"),
    ]
    results = []
    for label, stencil, bc, _ in cases:
        grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
        fdm = FDMDiscretization(
            grid, D=0.001, chi=1.0, Cm=1.0,
            stencil=stencil, boundary_mode=bc,
        )
        proto = StimulusProtocol()
        proto.add_stimulus(
            region=lambda x, y: x < 0.05,
            start_time=0.0, duration=2.0, amplitude=-52.0,
        )
        sim = MonodomainSimulation(
            spatial=fdm, ionic_model='ttp06', stimulus=proto,
            dt=DT, splitting='strang', ionic_solver='rush_larsen',
            diffusion_solver='forward_euler', cell_type='EPI',
        )
        times, V_hist = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
        V_field = V_hist.reshape(len(times), NX, NY)
        # snapshot at snapshot_t
        idx = int(round(snapshot_t / SAVE_EVERY))
        idx = min(idx, len(times) - 1)
        snap = V_field[idx]
        results.append((label, snap))
    sys.path.remove(str(ENGINE_V54))
    return results, NX, NY, snapshot_t


# ============================================================================
# Row 3: LBM V1
# ============================================================================

def run_lbm_row():
    sys.path.insert(0, str(LBM_V1))
    from src.simulation import LBMSimulation
    from ionic.ttp06.model import TTP06Model
    from ionic.base import CellType

    Nx, Ny = 41, 21
    dx = 0.025
    dt = 0.02
    D = 0.001
    n_steps = int(round(12.0 / dt))    # 12 ms, matches monodomain row

    cases = [
        ("D2Q9 uniform_8",  "d2q9", "uniform_8", "deficit"),
        ("D2Q5",            "d2q5", "canonical", "baseline"),
        ("D2Q9 canonical",  "d2q9", "canonical", "fix"),
    ]
    results = []
    for label, lattice, weights_mode, _ in cases:
        ionic = TTP06Model(cell_type=CellType.EPI, device=torch.device('cpu'))
        sim = LBMSimulation(
            Nx=Nx, Ny=Ny, dx=dx, dt=dt, D=D,
            ionic_model=ionic, Cm=1.0,
            lattice=lattice, weights_mode=weights_mode,
        )
        # Line stim at x=0 column for 2 ms
        stim_mask = torch.zeros(Nx, Ny, dtype=torch.bool)
        stim_mask[0:2, :] = True   # 2 columns wide, full y
        sim.add_stimulus(stim_mask, start=0.0, duration=2.0, amplitude=-52.0)
        # Run until snapshot time
        for _ in range(n_steps):
            sim.step()
        V_field = sim.V.numpy()  # shape (Nx, Ny)
        results.append((label, V_field))
    sys.path.remove(str(LBM_V1))
    return results, Nx, Ny


# ============================================================================
# Plot
# ============================================================================

print("[run] storage tank row...", flush=True)
tank_results, tNx, tNy = run_storage_tank_row()
print("[run] monodomain row...", flush=True)
mono_results, mNx, mNy, snapshot_t = run_monodomain_row()
print("[run] LBM row...", flush=True)
lbm_results, lNx, lNy = run_lbm_row()

print("\n[plot] building 3x3 figure...", flush=True)
fig, axes = plt.subplots(
    3, 3, figsize=(16, 13),
    constrained_layout=True,
)
# Reserve a left strip (4 % of fig width) for the row labels so the
# constrained-layout engine doesn't push panels into the same column.
fig.get_layout_engine().set(
    w_pad=0.04, h_pad=0.04, hspace=0.03, wspace=0.03,
    rect=(0.035, 0.0, 0.965, 1.0),
)

# Column titles
col_titles = ["DEFICIT (Moore-8 + face_mirror)",
              "BASELINE (cardinal-4)",
              "FIX (iso + bounce-back)"]
for c, title in enumerate(col_titles):
    axes[0, c].set_title(title, fontsize=11, fontweight='bold', pad=8)

# Row 1: storage tank — isochrones (shared colorbar)
tank_vmin = min(np.nanmin(np.where(iso < 0, np.nan, iso.astype(float)))
                for _, iso in tank_results)
tank_vmax = max(np.nanmax(np.where(iso < 0, np.nan, iso.astype(float)))
                for _, iso in tank_results)
levels = np.linspace(tank_vmin, tank_vmax, 12)
cs_last = None
for c, (label, iso) in enumerate(tank_results):
    ax = axes[0, c]
    iso_plot = iso.astype(float)
    iso_plot[iso_plot < 0] = np.nan
    cs = ax.contourf(iso_plot, levels=levels, cmap='viridis', extend='neither')
    ax.contour(iso_plot, levels=levels[::2], colors='k', linewidths=0.4, alpha=0.5)
    cs_last = cs
    ax.set_xlabel("x (column)")
    if c == 0:
        ax.set_ylabel("y (row)")
    ax.set_aspect('equal')
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='0.5', alpha=0.9))
fig.colorbar(cs_last, ax=axes[0, :].tolist(), shrink=0.75,
             label="LAT step", pad=0.015)

# Row 2: monodomain — V field at snapshot_t (shared colorbar)
mono_vmin = min(snap.min() for _, snap in mono_results)
mono_vmax = max(snap.max() for _, snap in mono_results)
im_mono = None
for c, (label, snap) in enumerate(mono_results):
    ax = axes[1, c]
    im = ax.imshow(snap.T, origin='lower', extent=[0, 1.0, 0, 0.5],
                   cmap='inferno', vmin=mono_vmin, vmax=mono_vmax,
                   aspect='auto')
    im_mono = im
    ax.set_xlabel("x (cm)")
    if c == 0:
        ax.set_ylabel("y (cm)")
    # Place label in upper-right (resting region, dark) so it doesn't
    # overlap the depolarised wavefront on the left.
    ax.text(0.98, 0.98, label, transform=ax.transAxes,
            va='top', ha='right', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='black',
                      edgecolor='none', alpha=0.7))
fig.colorbar(im_mono, ax=axes[1, :].tolist(), shrink=0.75,
             label="V (mV)", pad=0.015)

# Row 3: LBM — V field at end (shared colorbar)
lbm_vmin = min(V.min() for _, V in lbm_results)
lbm_vmax = max(V.max() for _, V in lbm_results)
im_lbm = None
for c, (label, V) in enumerate(lbm_results):
    ax = axes[2, c]
    im = ax.imshow(V.T, origin='lower', extent=[0, 1.0, 0, 0.5],
                   cmap='inferno', vmin=lbm_vmin, vmax=lbm_vmax,
                   aspect='auto')
    im_lbm = im
    ax.set_xlabel("x (cm)")
    if c == 0:
        ax.set_ylabel("y (cm)")
    ax.text(0.98, 0.98, label, transform=ax.transAxes,
            va='top', ha='right', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='black',
                      edgecolor='none', alpha=0.7))
fig.colorbar(im_lbm, ax=axes[2, :].tolist(), shrink=0.75,
             label="V (mV)", pad=0.015)

# Row labels — placed in the left strip reserved by `rect` above, so they
# don't collide with the y-axis tick labels of the leftmost column.
for y, name in [(0.83, "Storage tank"),
                (0.50, "Monodomain V5.4"),
                (0.17, "LBM V1")]:
    fig.text(0.018, y, name, rotation=90, va='center', ha='center',
             fontsize=12, fontweight='bold')

fig.suptitle(
    "Connectivity-mediated boundary deficit across model classes\n"
    "Same fewer-neighbours mechanism in all three; eliminated by "
    "cardinal-only or diagonal-aware bounce-back (LBM-style).\n"
    f"Storage tank: line stim, 4000 steps   |   "
    f"Monodomain V5.4: TTP06 EPI, snapshot t={snapshot_t} ms   |   "
    f"LBM V1: TTP06 EPI, snapshot t=12 ms",
    fontsize=12,
)

png_path = OUT_DIR / "connectivity_cross_engine.png"
pdf_path = OUT_DIR / "connectivity_cross_engine.pdf"
fig.savefig(png_path, dpi=180, bbox_inches='tight')
fig.savefig(pdf_path, dpi=180, bbox_inches='tight')
print(f"[plot] saved {png_path}")
print(f"[plot] saved {pdf_path}")

# Numerical summary
print("\n=== max boundary asymmetry per panel (qualitative bridge proof) ===")
for label, iso in tank_results:
    iso_f = iso.astype(float)
    iso_f[iso_f < 0] = np.nan
    valid = ~np.isnan(iso_f)
    cols_full = valid.all(axis=0)
    if cols_full.any():
        spread = (np.nanmax(iso_f[:, cols_full], axis=0)
                  - np.nanmin(iso_f[:, cols_full], axis=0)).max()
    else:
        spread = float('nan')
    print(f"  STORAGE TANK  {label:<30}  max LAT spread = {spread:.2f} steps")

for label, snap in mono_results:
    i_mid = mNx // 2
    dev = abs(snap[i_mid, 0] - snap[i_mid, mNy // 2])
    print(f"  MONODOMAIN    {label:<30}  |V[top]-V[ctr]| at x_mid = {dev:.4f} mV")

for label, V in lbm_results:
    i_mid = lNx // 2
    dev = abs(V[i_mid, 0] - V[i_mid, lNy // 2])
    print(f"  LBM           {label:<30}  |V[top]-V[ctr]| at x_mid = {dev:.4f} mV")
