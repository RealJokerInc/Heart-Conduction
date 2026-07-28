"""Source of truth for `cardiac_core/tutorials/03_2_bidomain.ipynb`
(Chapter 3.2 — "Tissue simulation, in depth: the bidomain engine & the extracellular field").

Edit THIS file, not the notebook — a `.py` diff is reviewable where an `.ipynb` diff is not — then
re-run it to regenerate the notebook in place:

    python cardiac_core/tutorials/_build/build_03_2_bidomain.py

Emits both the `.ipynb` (plain JSON, no `nbformat` dependency) and, with `--script PATH`, a flat `.py`
of every code cell for headless verification.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(os.path.dirname(_HERE), "03_2_bidomain.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Tissue simulation, in depth — 2: the bidomain engine & the extracellular field

Notebook 3.1 built a strip of tissue and ran it on the **monodomain** engine, measuring a conduction
velocity of about 58 cm/s. This notebook keeps that exact strip and swaps in the second engine,
**bidomain**. It computes the same wave, but it also tracks something monodomain throws away: the
electrical potential in the space *outside* the cells. That extra field is the basis of every ECG.

You do not need to have run 3.1 first — this notebook rebuilds the tissue from scratch in one cell.

**What you'll learn**

1. **What bidomain adds** — one potential vs two, and why the second one matters
2. **Running it** — the same four ingredients, a different engine factory
3. **The extracellular field `phi_e`** — the wavefront seen from outside the cells
4. **Comparing fairly** — bidomain and monodomain agree on speed here, and why

**Runtime**: about 10 seconds of computing (one bidomain run — a little slower than monodomain because
it solves an extra equation every step). On Colab, add about a minute the first time for the install.
"""),

(M, """---
## 0. Install and import
"""),

(C, """# Installs cardiac_core if this environment doesn't already have it (e.g. a fresh Colab runtime).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("cardiac_core") is None:
    print("Installing cardiac_core — this takes about a minute, please wait...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "cardiac-core[viz] @ git+https://github.com/RealJokerInc/cardiac-core.git"],
        check=True,
    )
    print("Installed.")
else:
    print("cardiac_core is already available.")
"""),

(C, """import cardiac_core as cc

print("cardiac_core is ready")
"""),

(M, """---
## 1. Rebuild the strip

Exactly the tissue from 3.1 — a 1.0 cm × 0.3 cm strip, healthy conductivity, a line electrode down the
left edge. If any of these three lines needs explaining, that is what notebook 3.1 is for; here we take
them as given and move to the engine.
"""),

(C, """g    = cc.Grid(101, 31, 0.01)                                          # 1.0 cm x 0.3 cm strip
cond = cc.ConductivityConfig.bidomain(1.74, 6.25)                     # healthy human ventricle
stim = cc.Stim.boundary(g, "left", amplitude=-52.0, start_time=1.0, duration=2.0)  # left-edge electrode

print("strip rebuilt")
"""),

(M, """---
## 2. One potential, or two?

The **monodomain** engine from 3.1 tracks a single number at each node: the **transmembrane voltage**
`Vm`, the difference in potential across the cell membrane. To do that with one number it has to
*assume* something about the world outside the cells — specifically that the extracellular space is a
perfect, uniform ground. That assumption is cheap and, for a wave running down a well-insulated strip,
almost exactly right.

The **bidomain** engine drops the assumption. It tracks **two** potentials at every node:

- **`Vm`** — the transmembrane voltage, same as before;
- **`phi_e`** — the potential in the extracellular space, the conducting fluid and tissue *around and
  between* the cells.

`phi_e` is not an accounting detail — it is a real, measurable voltage. An electrode sitting in the
bath, or the leads of an ECG on the body surface, measure exactly this extracellular potential. The
price is speed: bidomain solves an extra equation for `phi_e` every step, so it runs a few times slower
than monodomain. You reach for it when the world outside the cells matters — a surrounding bath, tissue
edges, defibrillation — and otherwise stay on the faster monodomain.
"""),

(M, """---
## 3. Run it

Same four ingredients — grid, ionic model, conductivity, stimulus — handed to `cc.bidomain` instead of
`cc.monodomain`. Everything else about the call is identical.
"""),

(C, """sim = cc.bidomain(g, "ttp06", cond, stim, dt=0.05)   # the ONLY change from 3.1: bidomain, not monodomain
r = sim.run(t_end=30.0, save_every=0.5)

print("done — bidomain run complete")
"""),

(M, """---
## 4. The voltage looks the same

First look at the transmembrane voltage `Vm`, exactly as we did for monodomain in 3.1 — a snapshot at
12 ms:
"""),

(C, """r.image(at=12.0)   # transmembrane voltage Vm at t = 12 ms
"""),

(M, """This is the same picture you saw in 3.1: a bright excited region behind the front, resting tissue
ahead, the front sitting a little past the middle at 12 ms. On a simple insulated strip like this,
bidomain and monodomain compute essentially the *same* `Vm` — which is the whole reason monodomain's
shortcut is trustworthy here. So what did the extra computation buy us?
"""),

(M, """---
## 5. The new field — `phi_e`

The payoff is a second field the monodomain run never produced: `phi_e`, the extracellular potential.
On a monodomain result `r.phi_e` is simply `None`; on a bidomain result it is a full voltage field you
can draw. Ask `r.image` for it with `field="phi_e"`. It is a small signed field — only a couple of tens
of millivolts, dwarfed by the 100-plus-mV swing of `Vm` — so we let the colour scale auto-fit to its
own range (`Gradient.autoscale()`) to bring the structure out:
"""),

(C, """r.image(at=12.0, field="phi_e", gradient=cc.Gradient.autoscale())   # extracellular potential at 12 ms
"""),

(M, """Look at the front. The extracellular potential makes a sharp **step** right where the wave is: the
tissue the wave has already passed sits *low* (dark, negative), the tissue still ahead of it sits *high*
(bright, positive), and the two are separated by a steep jump at the wavefront. The active front behaves
like a tiny travelling battery, pushing the outside potential up in front of itself and pulling it down
behind. As that jump sweeps past a fixed point, an electrode there records a swing from high to low —
and *that* deflection, summed over the whole heart, is what an ECG draws. Monodomain, tracking only
`Vm`, has no `phi_e` to show it; bidomain does, which is exactly why it exists.
"""),

(M, """---
## 6. Comparing fairly

Measure the conduction velocity the same way as 3.1 — same call, same columns, same row — so the two
engines can be compared honestly:
"""),

(C, """cv = r.cv(x1=20, x2=80, y=15)
print(f"bidomain conduction velocity = {cv:.1f} cm/s")
"""),

(M, """About **60 cm/s** — within a whisker of monodomain's 58 cm/s on the identical strip. That
agreement is the point: with a well-insulated strip and no surrounding bath, the two engines see the
same wave travelling at the same speed, and monodomain gets it for a fraction of the cost. Bidomain
starts to *disagree* — and starts to earn its extra expense — only when the extracellular world stops
being a simple ground: a conducting bath drawing current off the tissue edge, a defibrillation shock
applied from outside, boundaries that load the wave. Absent those, the honest move is the cheap one.

### Try it yourself

The `phi_e` step travels with the wave. Change `at=12.0` to `at=6.0` and then `at=18.0` in the
`field="phi_e"` cell and re-run it. Early on the step sits in the left third of the strip; later it has
moved most of the way across — you are watching the source of the ECG signal sweep down the tissue.
"""),

(M, """---
## Recap

- **Monodomain** tracks one potential (`Vm`) and assumes a perfect ground outside the cells;
  **bidomain** tracks two (`Vm` **and** `phi_e`, the extracellular potential) and pays for it in speed.
- On a plain insulated strip the two engines compute the **same `Vm` and nearly the same CV**
  (≈ 60 cm/s bidomain vs ≈ 58 cm/s monodomain), so monodomain's shortcut is safe.
- Bidomain's payoff is **`phi_e`** — the field a real electrode measures, showing the wavefront as a
  travelling step in the extracellular potential (the seed of the ECG). Reach for bidomain when the
  world outside the cells matters (a bath, an edge, a shock); otherwise use the faster monodomain.

**Where next**: Notebook 3.3 runs the same strip on the third engine — **LBM** — which solves the same
physics by an entirely different numerical route, and comes out at a slightly different speed. That
difference, and what to do about it, is the lesson.
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
