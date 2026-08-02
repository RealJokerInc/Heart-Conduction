"""Source of truth for `cardiac_core/tutorials/05_4_numerical.ipynb`
(Chapter 5.4 — "Tuning by hand: trust your number (numerical hygiene)").

Edit THIS file, not the notebook, then regenerate:
    python cardiac_core/tutorials/_build/build_05_4_numerical.py
Emits the `.ipynb` (plain JSON) and, with `--script PATH`, a flat `.py` for headless verification.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(os.path.dirname(_HERE), "05_4_numerical.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Chapter 5.4 — Numerical Convergence & Trust

This is the last notebook of the series, and it is the one that keeps the other three honest. So far you
have tuned conductivity, geometry, and ion channels and read a conduction velocity off each run as if it
were the truth. It usually is — but only if you ran it on a fine enough **grid** and a small enough
**time step**. Get those wrong and the simulation will hand you a number that *looks* physiological and
*is* an artifact. A hand-tuner who doesn't check this will "fix" a healthy model to match a wrong number.

This is not physiology, but it is the difference between a tuned number you can trust and one you can't.

**What you'll learn**

1. **Grid spacing `dx` is a trap** — too coarse a grid makes a perfectly healthy wave look **slow, even
   blocked**. It is a numerical artifact, and it is the real reason models sometimes show "phantom"
   conduction block. Refine `dx` until the velocity stops moving, and tune only there.
2. **Time step `dt` is about accuracy, not stability** — the default solver never blows up, but too big
   a `dt` shifts the velocity a few percent. Know how much your number can wander.
3. **Engines are not interchangeable numbers** — monodomain, bidomain, and LBM give *different* absolute
   velocities for the *same* tissue. Compare an engine to **itself**, never one engine's number to
   another's.

**Runtime**: about a minute and a half (many short runs, including some deliberately fine ones). On Colab
add ~a minute the first time for the install.
"""),

(C, """# Installs cardiac_core if this environment doesn't already have it (e.g. a fresh Colab runtime).
# If it is already installed, this does nothing.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("cardiac_core") is None:
    print("Installing cardiac_core - this takes about a minute, please wait...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "cardiac-core[viz] @ git+https://github.com/Zimmerman-Research-Group/CardiacCore.git"],
        check=True,
    )
    print("Installed.")
else:
    print("cardiac_core is already available.")
"""),

(C, """import cardiac_core as cc
import matplotlib.pyplot as plt   # only used to plot the numbers we measure

print("cardiac_core is ready")
"""),

(M, """---
## 1. The grid-spacing trap

`cc.Grid(Nx, Ny, dx)` lays the tissue out as a lattice of points spaced `dx` cm apart, and the
simulation only ever knows the voltage *at those points*. The wavefront's upstroke — the sharp jump from
resting to excited — is physically only about half a millimetre to a millimetre wide. If your grid points
are spaced too far apart, only one or two of them sit on that jump, and the solver simply cannot draw the
front sharply enough. The velocity it reports comes out **wrong**.

To see it, we measure the *same* tissue at four grid spacings, from coarse (`dx = 0.04` cm) to fine
(`0.005` cm). We hold everything else fixed — same 1 cm strip, same conductivity, same time step, and a
stimulus defined by a fixed *physical* width (`x < 0.03` cm) so it injects the same charge whatever the
grid. Only `dx` changes.
"""),

(C, """def cv_at_dx(dx):
    \"\"\"Conduction velocity of the SAME 1 cm strip, discretised at spacing dx.\"\"\"
    Nx = round(1.0 / dx) + 1                       # keep the strip 1.0 cm long
    Ny = round(0.2 / dx) + 1                        #             and 0.2 cm wide
    g    = cc.Grid(Nx, Ny, dx)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25)
    stim = cc.Stim.from_region(g, lambda x, y: x < 0.03,        # fixed 0.03 cm physical width
                               start_time=1.0, duration=2.0, amplitude=-80.0)
    r = cc.monodomain(g, "ttp06", cond, stim, dt=0.05).run(t_end=38.0, save_every=0.5)
    return r.cv(x1=round(0.3 / dx), x2=round(0.7 / dx), y=Ny // 2)   # CV over x = 0.3..0.7 cm

spacings = [0.04, 0.02, 0.01, 0.005]
dx_cv = [cv_at_dx(dx) for dx in spacings]

for dx, cv in zip(spacings, dx_cv):
    print(f"dx = {dx:.3f} cm  ({round(1.0/dx)+1:>3} points):  CV = {cv:.1f} cm/s")
"""),

(C, """plt.figure(figsize=(5, 4))
plt.plot(spacings, dx_cv, "o-", color="darkorange", ms=9)
plt.axhline(dx_cv[-1], ls=":", color="0.6")
plt.annotate("coarse: ~20% too slow\\n(looks like sick tissue!)", (0.04, dx_cv[0]),
             xytext=(0.028, dx_cv[0] - 8), fontsize=9, color="crimson",
             arrowprops=dict(arrowstyle="->", color="crimson"))
plt.annotate("converged", (0.006, dx_cv[-1] + 0.6), fontsize=9, color="0.4")
plt.xlabel("grid spacing dx (cm)   --   coarse (left)  to  fine (right)")
plt.ylabel("conduction velocity (cm/s)")
plt.title("Refine dx until CV stops changing, then trust it")
plt.gca().invert_xaxis()          # coarse on the left, fine on the right
plt.tight_layout()
plt.show()
"""),

(M, """Read the plot right-to-left as you *coarsen* the grid. At the fine end the velocity has **converged**
— `dx = 0.01` and `0.005` cm agree at about **59 cm/s**, so refining further buys nothing. But coarsen to
`dx = 0.04` cm and the reported velocity collapses to **~47 cm/s** — a 20% error out of nowhere. Coarsen
further still and the wave can fail to propagate at all: a **phantom conduction block** that is pure grid
artifact, not biology.

This is the single most dangerous trap in by-hand tuning. If you had measured on the coarse grid and
seen 47 cm/s, you might have "corrected" a perfectly healthy model by cranking up its conductivity — or
worse, concluded the tissue was diseased. **The fix is the discipline the plot shows: refine `dx` until
the velocity stops moving, and do all your tuning in that converged regime.** (The other notebooks in
this series ran at `dx = 0.01` cm, on the converged side, for exactly this reason.)
"""),

(M, """---
## 2. The time step is about accuracy

The grid spacing can make a run flat-out wrong. The time step `dt` is gentler: the default solver
(Crank–Nicolson) is *unconditionally stable*, so a big `dt` will never make the simulation blow up. What
it costs you is **accuracy** — a coarse `dt` shifts the velocity by a few percent. Same strip, fixed fine
grid, only `dt` changing:
"""),

(C, """def cv_at_dt(dt):
    g    = cc.Grid(101, 31, 0.01)                  # a fixed, fine grid
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25)
    stim = cc.Stim.from_region(g, lambda x, y: x < 0.03, start_time=1.0, duration=2.0, amplitude=-80.0)
    return cc.monodomain(g, "ttp06", cond, stim, dt=dt).run(t_end=24.0, save_every=0.5).cv(30, 70, 15)

for dt in [0.2, 0.1, 0.05, 0.02]:
    print(f"dt = {dt:<5} ms:  CV = {cv_at_dt(dt):.1f} cm/s")
"""),

(M, """From a coarse `dt = 0.2` ms to a fine `dt = 0.02` ms the velocity climbs from **~54 to ~59 cm/s** and
then **settles** — `dt = 0.05` and `0.02` ms already agree. Nothing crashed at `dt = 0.2`; it was stable
but a few percent off. The lesson mirrors the grid: pick a `dt`, halve it, and if the number barely moves
you are accurate enough. (A bigger `dt` is the cheapest way to speed up a long run — just check it hasn't
moved your answer.)
"""),

(M, """---
## 3. Engines are not interchangeable numbers

`cardiac_core` ships three solvers — `monodomain`, `bidomain`, and `lbm` — and they make *different
approximations*. Run the *identical* tissue through all three and read off the velocity:
"""),

(C, """g    = cc.Grid(101, 31, 0.01)
cond = cc.ConductivityConfig.bidomain(1.74, 6.25)
stim = cc.Stim.from_region(g, lambda x, y: x < 0.03, start_time=1.0, duration=2.0, amplitude=-80.0)

cv_mono  = cc.monodomain(g, "ttp06", cond, stim, dt=0.1  ).run(24.0, 0.5).cv(30, 70, 15)
cv_bidom = cc.bidomain(  g, "ttp06", cond, stim, dt=0.1  ).run(24.0, 0.5).cv(30, 70, 15)
cv_lbm   = cc.lbm(       g, "ttp06", cond, stim, dt=0.005).run(24.0, 0.5).cv(30, 70, 15)

print(f"monodomain : {cv_mono:.1f} cm/s")
print(f"bidomain   : {cv_bidom:.1f} cm/s")
print(f"lbm        : {cv_lbm:.1f} cm/s")
"""),

(M, """Same σ, same strip — three different numbers: monodomain ~**57**, bidomain ~**58**, and LBM ~**65**
cm/s. None is "wrong"; they are different numerical schemes, and LBM in particular runs faster for the
same conductivity because of how it represents diffusion.

The rule that follows is simple and strict: **compare like with like.** If you tuned a model on
monodomain, judge every change against monodomain. Never conclude a drug slowed conduction because your
LBM number is lower than someone's monodomain number — you would be measuring the solver, not the drug.
The same goes for `dx` and `dt`: only differences taken *at the same settings* mean anything.
"""),

(M, """### Try it yourself

1. **Watch the block appear.** In section 1, change `spacings` to `[0.08, 0.06, 0.04, 0.02]`. On the
   coarsest grid the velocity may drop dramatically or `r.cv(...)` may return `nan` — the wave failing to
   cross is the phantom block, produced by nothing but too few grid points.
2. **Halve and compare.** In section 2, run a single `dt` you like, then run half that value. If the
   two velocities agree to within a fraction of a percent, your `dt` is fine; if not, keep halving.
"""),

(M, """---
## Recap — and the end of the series

- **`dx` (grid spacing) can make a healthy wave look slow or blocked** — a numerical artifact, the true
  source of "phantom" conduction block. Refine until CV stops changing; tune only in that converged
  regime.
- **`dt` (time step) is an accuracy knob**, not a stability one (the default solver never blows up):
  a coarse `dt` shifts CV a few percent. Halve it and check the number holds.
- **Different engines give different absolute velocities** for the same tissue — compare an engine to
  itself, and only ever compare numbers taken at matched `dx`, `dt`, and solver.

That closes the capstone. Across these four notebooks you turned every raw knob `cardiac_core` offers and
watched it move the wave: **conductivity, D, χ and Cm set the speed** (5.1); **fibres, scars and stimuli
set the shape and whether it launches** (5.2); **the ion channels from Chapter 2 set speed, block and
wavelength** (5.3); and **the grid and time step decide whether any of those numbers can be believed**
(5.4). That is tuning by hand — and it is the intuition every automated optimiser is only ever trying to
reproduce.
""")
]


def _src(text):
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def build(script_path=None):
    cells = []
    for i, (kind, text) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": _src(text)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {NB_PATH}  ({len(CELLS)} cells, {n_code} code)")
    if script_path:
        code = "\n".join(text for kind, text in CELLS if kind == "code")
        with open(script_path, "w") as f:
            f.write('import matplotlib\nmatplotlib.use("Agg")   # headless for verification\n\n')
            f.write(code)
        print(f"wrote {script_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script", metavar="PATH", default=None,
                    help="also write a flat .py of every code cell, for headless verification")
    build(script_path=ap.parse_args().script)
