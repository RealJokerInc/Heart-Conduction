"""
Plot dV/dt traces at boundary vs center for all 4 dV/dt diagnostic cases.

Reads HDF5 files from data/ and produces:
  figures/dvdt_traces.png   — 2x4 grid: V(t) over dV/dt(t), one column per case
  figures/dvdt_deviation.png — single-panel overlay of dV/dt deviation across
                                all 4 cases (the key "does equalization happen?"
                                plot)

Question driving the plot:
  Does the boundary/center dV/dt asymmetry equalize after the deficit forms
  (i.e., does lateral diffusion catch up the boundary cell), or does the
  imbalance persist through AP firing? Compare diffusion-only vs +TTP06,
  face_mirror vs face_mirror_iso.
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    ("case1_fm_diff.h5",   "fm  + diff",   "C0"),
    ("case2_fmi_diff.h5",  "fmi + diff",   "C2"),
    ("case3_fm_ttp06.h5",  "fm  + ttp06",  "C3"),
    ("case4_fmi_ttp06.h5", "fmi + ttp06",  "C1"),
]
I_COL = 2          # column one downstream of stim strip (i=1)
J_BDRY = 0
# J_CTR will be NY // 2 (loaded from file)


def load_case(fname):
    with h5py.File(DATA_DIR / fname, "r") as f:
        t = f["t"][:]
        V = f["V"][:]
        attrs = {k: f.attrs[k] for k in f.attrs}
    return t, V, attrs


# Pre-load all 4
loaded = []
for fname, label, color in CASES:
    t, V, attrs = load_case(fname)
    NY = attrs["NY"]
    j_ctr = NY // 2
    V_bdry = V[:, I_COL, J_BDRY]
    V_ctr  = V[:, I_COL, j_ctr]
    # Central differences via np.gradient (handles endpoints)
    dVdt_bdry = np.gradient(V_bdry, t)
    dVdt_ctr  = np.gradient(V_ctr,  t)
    loaded.append({
        "fname": fname,
        "label": label,
        "color": color,
        "t": t,
        "V_bdry": V_bdry,
        "V_ctr": V_ctr,
        "dVdt_bdry": dVdt_bdry,
        "dVdt_ctr": dVdt_ctr,
        "attrs": attrs,
        "j_ctr": j_ctr,
    })


# ============================================================
# Figure 1 — V(t) over dV/dt(t), one column per case
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True,
                         sharex='col')

for col, d in enumerate(loaded):
    t = d["t"]
    # Top row: V(t)
    ax_v = axes[0, col]
    ax_v.plot(t, d["V_bdry"], color='crimson', lw=1.2,
              label=f"boundary (j=0)")
    ax_v.plot(t, d["V_ctr"], color='royalblue', lw=1.2, ls='--',
              label=f"center (j={d['j_ctr']})")
    ax_v.set_title(f"{d['label']}\n({d['fname']})", fontsize=10)
    ax_v.set_ylabel("V (mV)")
    ax_v.grid(alpha=0.3)
    ax_v.legend(fontsize=8, loc='best')

    # Bottom row: dV/dt(t)
    ax_d = axes[1, col]
    ax_d.plot(t, d["dVdt_bdry"], color='crimson', lw=1.2,
              label="boundary (j=0)")
    ax_d.plot(t, d["dVdt_ctr"], color='royalblue', lw=1.2, ls='--',
              label=f"center (j={d['j_ctr']})")
    ax_d.set_xlabel("t (ms)")
    ax_d.set_ylabel("dV/dt (mV/ms)")
    ax_d.grid(alpha=0.3)
    ax_d.axhline(0, color='gray', lw=0.5)

fig.suptitle(
    f"dV/dt diagnostic — moore8_uniform stencil, NX×NY=41×21, "
    f"column i={I_COL} (one downstream of stim strip at i=1)\n"
    f"IC: V[i=1, :] = 0 mV, V[else] = −86.2 mV. dt=0.01 ms. "
    f"No clamp, no current injection — initial-condition only.",
    fontsize=11,
)
out_path = OUT_DIR / "dvdt_traces.png"
fig.savefig(out_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {out_path}")


# ============================================================
# Figure 2 — dV/dt deviation overlay (the key plot)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

# Top-left: deviation V(t)
ax = axes[0, 0]
for d in loaded:
    dev_V = d["V_bdry"] - d["V_ctr"]
    ax.plot(d["t"], dev_V, color=d["color"], lw=1.3, label=d["label"])
ax.set_xlabel("t (ms)")
ax.set_ylabel("V[boundary] − V[center]   (mV)")
ax.set_title(f"V deviation (column i={I_COL})")
ax.axhline(0, color='gray', lw=0.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='best')

# Top-right: dV/dt deviation
ax = axes[0, 1]
for d in loaded:
    dev_dVdt = d["dVdt_bdry"] - d["dVdt_ctr"]
    ax.plot(d["t"], dev_dVdt, color=d["color"], lw=1.3, label=d["label"])
ax.set_xlabel("t (ms)")
ax.set_ylabel("dV/dt[boundary] − dV/dt[center]   (mV/ms)")
ax.set_title(f"dV/dt deviation (column i={I_COL})")
ax.axhline(0, color='gray', lw=0.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='best')

# Bottom-left: same V deviation, zoomed to early-ramp window
ax = axes[1, 0]
for d in loaded:
    dev_V = d["V_bdry"] - d["V_ctr"]
    ax.plot(d["t"], dev_V, color=d["color"], lw=1.3, label=d["label"])
ax.set_xlabel("t (ms)")
ax.set_ylabel("V[boundary] − V[center]   (mV)")
ax.set_title("V deviation — zoomed to 0–5 ms (ramp-up window)")
ax.set_xlim(0, 5)
ax.axhline(0, color='gray', lw=0.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='best')

# Bottom-right: same dV/dt deviation, zoomed
ax = axes[1, 1]
for d in loaded:
    dev_dVdt = d["dVdt_bdry"] - d["dVdt_ctr"]
    ax.plot(d["t"], dev_dVdt, color=d["color"], lw=1.3, label=d["label"])
ax.set_xlabel("t (ms)")
ax.set_ylabel("dV/dt[boundary] − dV/dt[center]   (mV/ms)")
ax.set_title("dV/dt deviation — zoomed to 0–5 ms (ramp-up window)")
ax.set_xlim(0, 5)
ax.axhline(0, color='gray', lw=0.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='best')

fig.suptitle(
    "Does lateral diffusion equalize the boundary deficit?  "
    "(deviation = boundary − center, column i=2)\n"
    "fmi cases ≡ 0 by construction; fm cases tell the story.",
    fontsize=12,
)
out_path = OUT_DIR / "dvdt_deviation.png"
fig.savefig(out_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {out_path}")


# ============================================================
# Numerical summary — key timepoints
# ============================================================
print()
print("Numerical summary at column i=2:")
print("=" * 90)
print(f"{'case':<14} {'t (ms)':>8} {'V_bdry':>10} {'V_ctr':>10} {'ΔV':>9} "
      f"{'dV/dt_b':>9} {'dV/dt_c':>9} {'Δdvdt':>9}")
print("-" * 90)
for d in loaded:
    label = d["label"].replace("  ", " ")
    t = d["t"]
    for t_pick in [0.01, 0.1, 1.0, 5.0, 25.0]:
        if t_pick > t[-1]:
            continue
        k = int(round(t_pick / 0.01))
        if k >= len(t):
            k = len(t) - 1
        dv_b = d["dVdt_bdry"][k]
        dv_c = d["dVdt_ctr"][k]
        print(f"{label:<14} {t[k]:8.2f} {d['V_bdry'][k]:+10.3f} "
              f"{d['V_ctr'][k]:+10.3f} {d['V_bdry'][k] - d['V_ctr'][k]:+9.3f} "
              f"{dv_b:9.3f} {dv_c:9.3f} {dv_b - dv_c:+9.3f}")
    print()
