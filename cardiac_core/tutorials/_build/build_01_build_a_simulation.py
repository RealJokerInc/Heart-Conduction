"""Source of truth for `cardiac_core/tutorials/01_build_a_simulation.ipynb`.

Edit THIS file, not the notebook — a `.py` diff is reviewable where an `.ipynb` diff is not — then
re-run it to regenerate the notebook in place:

    python cardiac_core/tutorials/_build/build_01_build_a_simulation.py

Emits both:
  - the `.ipynb` (written as plain JSON on purpose, so that authoring needs no `nbformat`
    dependency)
  - `--script PATH`: a flat `.py` of every CODE cell concatenated, so the exact reader-facing code
    can be executed and verified before shipping. This is how the lesson is regression-checked
    until the `nbconvert --execute` gate exists.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(os.path.dirname(_HERE), "01_build_a_simulation.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Build Your First Cardiac Simulation

By the end of this notebook you will have built a piece of virtual heart tissue, placed an electrode
on it, watched an electrical wave cross it, and measured how fast that wave travelled — then run the
identical experiment on all three simulation engines.

**What you'll learn**

1. **The grid** — how to describe a piece of tissue
2. **The tissue properties** — how well it conducts
3. **The stimulus** — where you put the electrode
4. **The simulation** — putting those three together and running it
5. **Measuring** — getting a number out
6. **The three engines** — monodomain, bidomain and LBM on the same tissue

**What you need**: a Python environment with `cardiac_core` installed. Nothing else — every number and picture below
is produced by code you run here.

**Runtime**: about a minute and a half. The three engine runs are the slow part.
"""),

(C, """import cardiac_core as cc
import numpy as np
import matplotlib.pyplot as plt

print("cardiac_core is ready")
"""),

(M, """---
## 1. The grid — where the tissue lives

A simulation needs a shape to happen in. Here that shape is a **grid**: a rectangular sheet of
points ("nodes"), evenly spaced. Three numbers describe it:

- `Nx` — how many nodes along the long axis
- `Ny` — how many nodes along the short axis
- `dx` — the spacing between neighbouring nodes, **in centimetres**

Let's build a strip 2 cm long and 0.5 cm wide, with nodes every 0.01 cm — 100 microns, roughly the
length of one real heart cell.
"""),

(C, """g = cc.Grid(201, 51, 0.01)      # 201 nodes long, 51 wide, spaced 0.01 cm apart

print(f"tissue is {g.Lx} cm long and {g.Ly} cm wide")
print(f"{g.n_dof} nodes in total")
"""),

(M, """**Why 201 nodes and not 200?** Because the length is made of the *gaps* between nodes, not the
nodes themselves: `Lx = dx * (Nx - 1)`. Two hundred gaps of 0.01 cm make 2 cm, and two hundred gaps
need two hundred and one nodes — the same reason a 2-metre fence with posts every metre needs three
posts, not two.

So to get a tissue of an exact size, use `Nx = round(Lx/dx) + 1`.
"""),

(M, """---
## 2. The tissue properties — how well it conducts

Heart tissue conducts in two places at once: **inside** the cells, through the gap junctions
connecting each cell to the next, and **outside** them, through the fluid in between. Those two
pathways have different conductivities — `sigma_i` and `sigma_e`, in mS/cm.

Healthy human ventricular muscle is roughly `sigma_i = 1.74`, `sigma_e = 6.25`. You will meet these
two numbers constantly.
"""),

(C, """cond = cc.ConductivityConfig.bidomain(1.74, 6.25)   # healthy human ventricle

print(f"effective conductivity: {cond.sigma_eff:.3f} mS/cm")
"""),

(M, """Turn those numbers down and the tissue becomes more weakly coupled: each cell has a harder time
exciting its neighbour, and the wave travels more slowly. That's the first exercise at the end.
"""),

(M, """---
## 3. The stimulus — where you put the electrode

Nothing happens until you excite the tissue. A **`Stim`** says exactly that: *where* the electrode
sits, *when* it turns on, and *how hard* it pushes.

The clearest way to build one is to name a place. `cc.Stim.boundary(g, "left")` lays a strip of
electrode along the left-hand edge — what you'd do to launch a wave travelling left to right.
"""),

(C, """stim = cc.Stim.boundary(
    g, "left",
    start_time=1.0,      # turn on at t = 1 ms
    duration=2.0,        # hold for 2 ms
    amplitude=-52.0,     # strength; NEGATIVE means depolarizing, i.e. it excites the tissue
)

print(f"{stim.n_nodes()} nodes will be stimulated, at t = {stim.times()} ms")
"""),

(M, """A `Stim` is not magic — underneath it is a map of which nodes the electrode touches. You can look at
that map directly, and it's a good habit: it's how you catch a stimulus that landed somewhere you
didn't intend.
"""),

(C, """fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(11, 2.4))

ax_all.imshow(stim.mask.T, origin="lower", aspect="auto", cmap="Greys",
              extent=[0, g.Lx, 0, g.Ly])
ax_all.set_title("the whole tissue")
ax_all.set_xlabel("x (cm)")
ax_all.set_ylabel("y (cm)")

ax_zoom.imshow(stim.mask.T, origin="lower", aspect="auto", cmap="Greys",
               extent=[0, g.Lx, 0, g.Ly])
ax_zoom.set_xlim(0, 0.1)                 # zoom into the leftmost millimetre
ax_zoom.set_title("the same thing, zoomed into the left edge")
ax_zoom.set_xlabel("x (cm)")

plt.show()
"""),

(M, """The left panel is honest but nearly empty, and that is the lesson: the electrode is a **thin strip**,
two nodes wide, against 2 cm of tissue. Zoomed in, on the right, you can actually see it. Stimulating
a small part of a large tissue is the normal case.

That `.T` transposes the map before drawing. `cardiac_core` stores fields as `(x, y)`, while images
are drawn as `(row, column)` — transposing puts x across the page where you expect it. You'll see the
same `.T` on every picture in this notebook.

Other places to put the electrode, all built the same way:

```python
cc.Stim.boundary(g, "right")       # also "top", "bottom"
cc.Stim.point(g, (0.5, 0.25))      # a small blob at an (x, y) point, in cm
cc.Stim.center(g)                  # a blob in the middle
```
"""),

(M, """---
## 4. The simulation — putting the pieces together

You now have three ingredients: a **grid**, its **conductivity**, and a **stimulus**. A simulation is
those three plus an **ionic model** — the description of the machinery inside a single heart cell.
We'll use `"ttp06"` (ten Tusscher 2006), a standard human ventricular model.

The arguments always come in the same order: *where*, *what kind of cell*, *how conductive*,
*stimulated how*.
"""),

(C, """sim = cc.monodomain(g, "ttp06", cond, stim, dt=0.05)

r = sim.run(t_end=40.0, save_every=0.5)     # simulate 40 ms, keep a frame every 0.5 ms

print(f"Vm has shape {tuple(r.Vm.shape)}  =  (frames, x nodes, y nodes)")
print(f"frames span t = {float(r.times[0]):.1f} to {float(r.times[-1]):.1f} ms")
"""),

(M, """Two arguments there are worth knowing.

`dt` is the **time step** — how far the simulator advances on each internal tick. Smaller is more
accurate and slower. `save_every` is different: it controls how often a frame is *kept for you*.
The simulation still takes every `dt` step internally; you just don't need 800 frames to see a wave.

The result `r` holds `r.Vm`, the membrane voltage of every node at every saved frame, and `r.times`,
the time of each frame.
"""),

(M, """---
## 5. Look at it

`r.Vm` is just numbers, so let's draw three moments and watch the wave move.
"""),

(C, """fig, axes = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True)

for ax, t_want in zip(axes, [5.0, 15.0, 30.0]):
    frame = int(np.argmin(np.abs(np.asarray(r.times) - t_want)))
    im = ax.imshow(np.asarray(r.Vm[frame]).T, origin="lower", aspect="auto",
                   extent=[0, g.Lx, 0, g.Ly], vmin=-90, vmax=40, cmap="inferno")
    ax.set_title(f"t = {float(r.times[frame]):.1f} ms")
    ax.set_ylabel("y (cm)")

axes[-1].set_xlabel("x (cm)")
fig.colorbar(im, ax=axes, label="membrane voltage (mV)")
plt.show()
"""),

(M, """Bright is depolarized (excited, near +20 mV), dark is resting (near -85 mV). The bright band is the
**wavefront**, and it moves left to right — away from the electrode you placed. That is conduction:
each patch of tissue exciting the patch next to it.
"""),

(M, """---
## 6. Measure it — conduction velocity

A picture is good; a number is better. **Conduction velocity** is how fast the wavefront travels,
and it's the single most-used measurement in this field.

You measure it the way you would on a real tissue: note when the wave arrives at one place, note
when it arrives at another, and divide the distance by the time difference. `r.cv()` does exactly
that between two points — given as node indices along x, on a chosen row `y`.
"""),

(C, """cv = r.cv(x1=20, x2=100, y=25)     # between node 20 and node 100, along the middle row

print(f"conduction velocity = {cv:.1f} cm/s")
"""),

(M, """Around 60 cm/s, which is the right ballpark for healthy human ventricular muscle.

Node 20 is 0.2 cm along and node 100 is 1.0 cm along, so this measures across the middle 0.8 cm.
Both points sit well inside the tissue, deliberately: right next to the electrode the wave is still
being pushed, and measuring there tells you about your electrode rather than about the tissue.
"""),

(M, """---
## 7. The same tissue, a different engine — bidomain

Everything so far used the **monodomain** engine, which tracks one voltage per node: the membrane
voltage. It's fast, and it's the right default.

**Bidomain** tracks the inside and the outside of the cells as two separate, coupled systems. That
costs noticeably more time per step, and buys you `phi_e` — the **extracellular potential**, the
signal an electrode sitting *outside* the tissue would actually record. Every ECG and every MEA
recording is a measurement of `phi_e`, so this is the engine to use when you care about what a
real recording would show.

The call is identical apart from the name. Same grid, same conductivity, same stimulus.
"""),

(C, """sim_bi = cc.bidomain(g, "ttp06", cond, stim, dt=0.05)
r_bi = sim_bi.run(t_end=40.0, save_every=0.5)

cv_bi = r_bi.cv(x1=20, x2=100, y=25)
print(f"bidomain conduction velocity = {cv_bi:.1f} cm/s")
print(f"and it also gives us phi_e, with shape {tuple(r_bi.phi_e.shape)}")
"""),

(M, """You may see a yellow `SolverConvergenceWarning` from the bidomain solver. It is informational — it
reports how tightly the extracellular solve converged, and the result is fine.
"""),

(C, """frame = int(np.argmin(np.abs(np.asarray(r_bi.times) - 15.0)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), sharex=True)

im1 = ax1.imshow(np.asarray(r_bi.Vm[frame]).T, origin="lower", aspect="auto",
                 extent=[0, g.Lx, 0, g.Ly], vmin=-90, vmax=40, cmap="inferno")
ax1.set_title(f"membrane voltage at t = 15 ms")
ax1.set_ylabel("y (cm)")
fig.colorbar(im1, ax=ax1, label="Vm (mV)")

im2 = ax2.imshow(np.asarray(r_bi.phi_e[frame]).T, origin="lower", aspect="auto",
                 extent=[0, g.Lx, 0, g.Ly], cmap="RdBu_r")
ax2.set_title("extracellular potential at the same moment")
ax2.set_xlabel("x (cm)")
ax2.set_ylabel("y (cm)")
fig.colorbar(im2, ax=ax2, label="phi_e (mV)")

plt.show()
"""),

(M, """Look at how the two panels line up. The wavefront in the top panel sits at about x = 0.75 cm, and in
the bottom panel that is exactly where `phi_e` flips sign: **positive (red) ahead of the front, in
the tissue that hasn't been excited yet, negative (blue) behind it**, in the tissue that already has.

That sign flip is why an extracellular recording looks the way it does. An electrode sitting at a
fixed spot doesn't see this whole map — it sees one point of it, over time. As the front sweeps past
underneath, that point goes from ahead-of-the-wave to behind-it, so the electrode records a swing
from positive to negative. That is the deflection you see on an MEA trace or an ECG.
"""),

(M, """---
## 8. ... and LBM

The third engine, **LBM** (lattice Boltzmann), reaches the same physics by a completely different
numerical route. It takes smaller time steps but each one is cheap, and it's the engine used in this
project for boundary-effect work.

Again: same grid, same conductivity, same stimulus.
"""),

(C, """sim_lbm = cc.lbm(g, "ttp06", cond, stim, dt=0.01)
r_lbm = sim_lbm.run(t_end=40.0, save_every=0.5)

cv_lbm = r_lbm.cv(x1=20, x2=100, y=25)
print(f"LBM conduction velocity = {cv_lbm:.1f} cm/s")
"""),

(M, """---
## 9. Three engines, one tissue

You have now run the same experiment three ways.
"""),

(C, """print(f"{'engine':<12}{'CV (cm/s)':>12}   gives you")
print("-" * 52)
print(f"{'monodomain':<12}{cv:>12.1f}   membrane voltage")
print(f"{'bidomain':<12}{cv_bi:>12.1f}   membrane voltage + phi_e")
print(f"{'LBM':<12}{cv_lbm:>12.1f}   membrane voltage")
"""),

(M, """The three numbers are close but not identical, because the three engines discretize the same
equations differently. Compare an engine against itself across conditions — a control run and a drug
run on monodomain — rather than comparing one engine's absolute number against another's.

**Which to reach for**

- **monodomain** — your default. Fastest, and it answers most questions.
- **bidomain** — when you need `phi_e`: extracellular recordings, defibrillation, bath and edge
  effects. Slower, because it solves a second coupled system every step.
- **LBM** — a different numerical route to the same physics; used here for boundary-effect studies.
"""),

(M, """---
## Try it yourself

**1. Weaker coupling.** Halve both conductivities and re-measure CV:

```python
cond_weak = cc.ConductivityConfig.bidomain(0.87, 3.125)
```

Rebuild the monodomain simulation with `cond_weak`, run it, and compare. Weaker coupling should be
slower — but is it half as fast? (It isn't. CV falls roughly with the *square root* of conductivity,
so halving it slows the wave by about 30%, not 50%.)

**2. Move the electrode.** Change `"left"` to `"bottom"` and re-run. Which way does the wave travel
now, and does `r.cv(x1=20, x2=100, y=25)` still measure anything sensible? (Think about which
direction that call measures along.)

**3. A different starting point.** Replace the boundary stimulus with `cc.Stim.center(g)` and look at
the snapshots. How does the wave's shape differ from the flat front you got from the edge?
"""),

(M, """---
## Recap

- A **`Grid`** is the tissue's shape: node counts plus spacing in cm, with `Lx = dx * (Nx - 1)`.
- A **`ConductivityConfig`** is how well it conducts, from `sigma_i` and `sigma_e` in mS/cm.
- A **`Stim`** is where the electrode is, when it fires, and how hard — built by naming a place, and
  inspectable as a mask.
- **Grid + ionic model + conductivity + stimulus** makes a simulation; `.run()` gives a result
  holding `r.Vm` and `r.times`.
- **`r.cv()`** measures conduction velocity between two points, which for healthy ventricle lands
  around 60 cm/s.
- **monodomain / bidomain / LBM** take identical arguments; bidomain additionally gives you `phi_e`.

**Where next**: the same four ingredients carry through everything else — applying a drug, pacing
repeatedly, carving a scar and watching conduction fail. The one you already know how to change is
the stimulus, so start there.
""")
]


def _src(text):
    """Split a cell body into the line list an .ipynb expects (each line keeps its \\n)."""
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]        # last line carries no trailing newline
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
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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
