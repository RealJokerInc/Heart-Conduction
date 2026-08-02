"""Source of truth for `cardiac_core/tutorials/01_intro.ipynb` (Chapter 1 — "Cardiac Core Intro").

Edit THIS file, not the notebook — a `.py` diff is reviewable where an `.ipynb` diff is not — then
re-run it to regenerate the notebook in place:

    python cardiac_core/tutorials/_build/build_01_intro.py

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
NB_PATH = os.path.join(os.path.dirname(_HERE), "01_intro.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Chapter 1 — Cardiac Core Intro

Welcome. `cardiac_core` is a Python library for simulating the electrical activity of heart tissue —
the wave of excitation that spreads across the heart with every beat. This first notebook is a gentle
tour: by the end you will have installed the library, met the handful of pieces a simulation is built
from, watched a single heart cell fire, and drawn a movie of a wave crossing a strip of tissue.

**What you'll learn**

1. **Install & import** — get the library and load it
2. **The object landscape** — the small set of pieces you assemble into a simulation
3. **A single cell** — run one heart cell and look at its action potential
4. **A picture and a movie** — turn a tissue simulation into a snapshot and an inline video

Nothing here asks you to know Python beyond running a cell and changing a number. Later chapters go
deep on each piece; this one just introduces the cast.

**Runtime**: well under two minutes. On Google Colab, add about a minute the first time for the install.
"""),

(M, """---
## 1. Install and import the library

A **package** is a bundle of ready-made code someone else wrote that you can install once and then
use. `cardiac_core` is such a package. The cell below installs it *if* your environment doesn't
already have it — on a fresh Google Colab runtime that takes about a minute; on a machine where it is
already installed the cell does nothing. You only ever run it once per session.

The `[viz]` part pulls in the extra pieces needed to render videos (beat 4). Keep it.
"""),

(C, """# Installs cardiac_core if this environment doesn't already have it (e.g. a fresh Colab runtime).
# If it is already installed, this does nothing.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("cardiac_core") is None:
    print("Installing cardiac_core — this takes about a minute, please wait...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "cardiac-core[viz] @ git+https://github.com/Zimmerman-Research-Group/CardiacCore.git"],
        check=True,
    )
    print("Installed.")
else:
    print("cardiac_core is already available.")
"""),

(M, """Now bring the library into this notebook. `import cardiac_core as cc` loads it and gives it the
short nickname `cc`, so that instead of typing `cardiac_core.Grid(...)` every time you can just write
`cc.Grid(...)`. The alias `cc` is a convention used throughout these notebooks — every example below
assumes it.
"""),

(C, """import cardiac_core as cc

print("cardiac_core is ready")
"""),

(M, """---
## 2. The object landscape — what a simulation is made of

Before running anything, it helps to see the whole cast at once. A tissue simulation is assembled
from a small number of pieces. You don't need to memorise this — you'll meet each one by *doing*, in
this chapter and the next few. Think of it as a map of where you're headed.

| The piece | What it is | You'll get it from |
|---|---|---|
| **Geometry** | the shape of the tissue — a grid of points, with optional **masks** to carve out regions or scars | `cc.Grid(...)`, `cc.rectangle_mask`, `cc.circle_mask` |
| **Conductivity** | how well the tissue passes current from one place to the next | `cc.ConductivityConfig.bidomain(...)` |
| **Ionic model** | the machinery inside a single heart cell — what makes it fire; also runnable on its own | a name like `"ttp06"`; or `cc.single_cell(...)` for one cell |
| **Stimulus / pacing** | the electrode: *where* you poke the tissue, *when*, and *how hard* — and, for a train of beats, how often | `cc.Stim.boundary(...)`, `.point(...)`, `.center(...)` |
| **Engine** | the numerical method that actually solves the physics — three of them | `cc.monodomain(...)`, `cc.bidomain(...)`, `cc.lbm(...)` |
| **The simulation** | the assembled, runnable object | the engine call returns it; `.run()` steps it forward |
| **The result** | everything the run produced, plus tools to measure and draw it | `r = sim.run(...)` → `r.cv(...)`, `r.apd()`, `r.image()`, `r.video()` |

The recipe is always the same shape: **geometry + conductivity + ionic model + stimulus → an engine →
`.run()` → a result you measure and draw.** The rest of this notebook walks the two ends of that
recipe — first a single cell (the ionic model alone), then a whole tissue run rendered as a movie.
"""),

(M, """---
## 3. A single cell — the action potential

Start at the smallest scale: one heart cell, no tissue. When a heart cell is excited, its voltage
does something dramatic and characteristic — it shoots up, holds high for a couple of hundred
milliseconds, then falls back to rest. That voltage-versus-time shape is the **action potential**, the
fundamental signal of the heart.

`cc.single_cell` runs exactly one cell of a chosen ionic model. Here we use `"ttp06"` — the
**ten Tusscher–Panfilov (2006)** human-ventricular model (full citation in the References at the end) —
as an `EPI` (epicardial) cell. `pre_pace=2` runs two warm-up
beats first and throws them away, so the cell we record has settled into a steady rhythm rather than
being caught cold on its first-ever beat.
"""),

(C, """sc = cc.single_cell(
    "ttp06",            # the ionic model: ten Tusscher 2006 human ventricle
    celltype="EPI",     # an epicardial cell (the heart's outer layer)
    pre_pace=2,         # 2 warm-up beats, discarded, so the recorded cell has settled
    dt=0.05,            # simulation time step in ms — larger runs faster; 0.05 is plenty here
)

print(f"resting voltage : {sc.v_rest:.1f} mV")     # where it sits between beats
print(f"peak voltage    : {sc.v_peak:.1f} mV")     # the top of the spike
print(f"APD90           : {sc.apd(0.9):.1f} ms")   # how long the beat lasts (90% recovery)
"""),

(M, """Those three numbers are the vital statistics of an action potential. The cell rests near
-85 mV, spikes up to about +77 mV, and the whole beat lasts roughly 233 ms — that last number, the
**APD90** (action-potential duration to 90% recovery), is the single most-quoted measure of a heart
cell's beat and you will see it constantly.

Now draw the curve. `sc.trace()` plots the action potential for you — it is part of the library's
figure pipeline, so you get a properly labelled plot without touching matplotlib yourself.
"""),

(C, """sc.trace(xlim=(0, 450))   # draw the action potential; zoom to the first 450 ms where the beat lives
"""),

(M, """Read the shape left to right — four phases:

1. **Rest** — the flat line at the bottom, near -85 mV. The cell sits here, waiting, between beats.
2. **Upstroke** — the near-vertical jump. Sodium channels snap open and the voltage rockets up in about
   a millisecond. This is the moment the cell "fires."
3. **Plateau** — the long high shoulder. Calcium flowing in roughly balances potassium flowing out, and
   the cell holds near the top for ~200 ms. This plateau is what makes heart cells special; a nerve
   cell has none.
4. **Repolarisation** — the fall back to rest, as potassium channels win out and return the cell to its
   resting level, ready for the next beat.

That single curve is what every cell in the tissue simulations below is doing, everywhere, all the
time.
"""),

(M, """### Try it yourself

The heart wall isn't uniform: cells from its inner layer (`ENDO`, endocardium) and outer layer (`EPI`,
epicardium) beat for slightly different lengths of time. Change `celltype="EPI"` to `celltype="ENDO"`
in the cell above, re-run it, and compare the APD90. It shifts by about ten milliseconds — in this
model the endocardial cell comes out a touch *shorter*, near 223 ms.
"""),

(M, """---
## 4. A picture and a movie

The picture you just drew — the action potential — is the first kind of figure `cardiac_core` makes.
The second is a **movie of a wave**, and for that we need space: a single cell has no room for a wave
to travel, so we need a strip of tissue.

Here we build one. Don't worry about the details — **Chapter 3 explains every one of these lines.**
For now, read it as: make a small rectangle of tissue, give it realistic conductivity, put an
electrode along the left edge, hand all of that to the fast `monodomain` engine, and run it for 20 ms.
"""),

(C, """g = cc.Grid(101, 31, 0.01)                              # a 1.0 cm x 0.3 cm strip of tissue
cond = cc.ConductivityConfig.bidomain(1.74, 6.25)       # healthy human ventricle
stim = cc.Stim.boundary(g, "left",                      # electrode along the left edge
                        start_time=1.0, duration=2.0, amplitude=-52.0)

sim = cc.monodomain(g, "ttp06", cond, stim, dt=0.05)    # assemble the simulation
r = sim.run(t_end=20.0, save_every=0.5)                 # run 20 ms, keeping a frame every 0.5 ms

print("done — the wave has crossed the strip")
"""),

(M, """`r` is the **result** — the last row of the map in section 2. It holds the voltage of every point
at every saved moment, and it knows how to draw itself. Ask it for a video:
"""),

(C, """r.video()   # an inline movie of the excitation wave sweeping across the strip
"""),

(M, """Press play. A bright band — the **wavefront** — enters from the left edge (where the electrode
fired) and sweeps to the right. That travelling band is a heartbeat in miniature: each patch of tissue
excites the patch next to it, and the excitation propagates. Bright is excited (depolarised) tissue —
the upstroke-and-plateau you saw in section 3; dark is tissue still at rest.

You can also freeze a single instant instead of playing the whole thing. `r.image(at=12.0)` draws the
voltage everywhere at 12 ms:
"""),

(C, """r.image(at=12.0)   # a snapshot of the membrane voltage at t = 12 ms
"""),

(M, """The bright front sits partway across the strip — the wave launched at the left and, 12 ms later,
has reached about the middle. Everything to its left has already been excited; everything to its right
is still resting, waiting its turn.

One more view. Rather than a single moment, `r.image(what="activation")` colours each point by *when*
the wave arrived there — an **activation map**. The contour lines connect points the wave reached at
the same instant — here, straight and evenly spaced: the fingerprint of a flat front moving steadily:
"""),

(C, """r.image(what="activation")   # colour each point by when the wave reached it
"""),

(M, """The colours sweep smoothly from the left edge to the right, confirming what the movie showed: a
clean wave crossing at a steady pace. How *fast* it crossed — the conduction velocity — is a number
you can pull straight out of `r`, but measuring and interpreting it is the job of Chapter 3, so we'll
leave it here.

### Try it yourself

Move the electrode. Change `"left"` to `"bottom"` in the `cc.Stim.boundary(...)` line above and re-run
the three cells. Which way does the wave travel now? (With the electrode along the bottom edge, the
front sweeps upward instead of left-to-right — try the snapshot and the activation map again and watch
the pattern turn.)
"""),

(M, """---
## Recap

- `cardiac_core` builds a tissue simulation from a few pieces — **geometry, conductivity, an ionic
  model, and a stimulus** — which you hand to an **engine** and `.run()` to get a **result**.
- A single heart cell's **action potential** has four phases — rest, a sharp upstroke, a long plateau,
  and repolarisation — and `cc.single_cell(...).trace()` runs one cell and draws it.
- A tissue **result** draws itself: `r.video()` plays the excitation wave, and `r.image(...)` freezes a
  snapshot or maps when the wave arrived — no plotting code required.

**Where next**: Chapter 2 stays with the single cell and asks what each ion channel *does* to the
action-potential shape — turn one knob and watch the plateau stretch or the upstroke slow. Then
Chapter 3 opens up the tissue strip you just ran and explains every line of it.
"""),

(M, """---
## References

- **TTP06** (`ttp06`) — ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a human ventricular tissue model." *Am J Physiol Heart Circ Physiol* 291(3):H1088–H1100.
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
