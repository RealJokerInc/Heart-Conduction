"""Source of truth for `cardiac_core/tutorials/04_1_pacing_restitution.ipynb`
(Chapter 4.1 — "Pacing a Cell: Rate and Restitution").

Edit THIS file, not the notebook — a `.py` diff is reviewable where an `.ipynb` diff is not — then
re-run it to regenerate the notebook in place:

    python cardiac_core/tutorials/_build/build_04_1_pacing_restitution.py

Emits both:
  - the `.ipynb` (written as plain JSON on purpose, so that authoring needs no `nbformat`
    dependency)
  - `--script PATH`: a flat `.py` of every CODE cell concatenated, so the exact reader-facing code
    can be executed and verified before shipping.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(os.path.dirname(_HERE), "04_1_pacing_restitution.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Chapter 4.1 — Pacing a Cell: Rate and Restitution

Your heart doesn't beat once — it beats *again and again*, and it speeds up and slows down. This
chapter is about **rate**: what happens to a single heart cell when you pace it faster and faster.

The headline result is one of the most important facts in cardiac electrophysiology: **the faster you
pace a cell, the shorter each beat becomes.** A cell driven at rest takes its time; a cell driven hard
cuts every beat short. That rate-dependence is called **restitution**, and by the end of this notebook
you will have measured the restitution curve yourself.

**What you'll learn**

1. **Pacing and BCL** — driving a cell at a fixed cycle length
2. **APD shortens with rate** — pace faster, and each action potential gets briefer
3. **The restitution curve** — plotting beat duration against recovery time
4. **Effective rate and capture** — why a cell can't keep up past a certain speed

This chapter stays with the **single cell** from Chapter 1 — no tissue yet. Chapter 4.2 takes pacing
into tissue and does something dramatic with it.

**Runtime**: about 80 seconds — the restitution sweep runs several settled simulations back to back.
On Google Colab, add about a minute the first time for the install.
"""),

(M, """---
## Setup — install and import

Same first cell as always: install `cardiac_core` if this environment doesn't have it, then import it
under the short name `cc`.
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
         "cardiac-core[viz] @ git+https://github.com/RealJokerInc/cardiac-core.git"],
        check=True,
    )
    print("Installed.")
else:
    print("cardiac_core is already available.")
"""),

(C, """import cardiac_core as cc
import matplotlib.pyplot as plt   # for the two restitution figures at the end

print("cardiac_core is ready")
"""),

(M, """---
## 1. Pacing and BCL

To **pace** a cell is to poke it with a stimulus over and over on a fixed schedule. The gap between
pokes is the **basic cycle length** (**BCL**) — the number of milliseconds from one beat to the next.
It is just the pacing rate written as a period instead of a frequency:

| BCL | Beats per minute | What it feels like |
|---|---|---|
| 1000 ms | 60 bpm | a calm resting heart |
| 500 ms | 120 bpm | brisk exercise |
| 300 ms | 200 bpm | a dangerously fast arrhythmia |

`cc.single_cell` paces a cell for you: pass `bcl=` for the cycle length and `pre_pace=` for a few
warm-up beats that are run and discarded, so the cell you actually record has *settled into* that
rhythm rather than being caught on its first-ever beat. The APD you read back is then the steady-state
beat *at that rate* — which is exactly what restitution is about.
"""),

(M, """---
## 2. Pace faster, and each beat gets shorter

Here is the whole experiment in one loop. We pace the same `"ttp06"` cell at a ladder of cycle
lengths — from a leisurely 600 ms down to a racing 250 ms — letting it settle at each rate, and for
every one we record two numbers:

- **APD90** — the action-potential duration, how long the beat lasts (`sc.apd(0.9)`).
- **DI** — the **diastolic interval**, the resting gap *before* that beat: `DI = BCL − APD`
  (`cc.di(bcl, apd)`). It is how much recovery time the cell got.

This is the runtime-heavy cell — several settled simulations in a row, roughly 80 seconds in total.
Let it run once.
"""),

(C, """bcls = [600, 450, 350, 300, 270, 250]   # cycle lengths in ms, slow -> fast

apds, dis, traces = [], [], {}
for bcl in bcls:
    sc = cc.single_cell(
        "ttp06", celltype="EPI",
        bcl=bcl, n_beats=1, pre_pace=2,   # settle for 2 beats at this rate, record the next
        dt=0.05,
    )
    apd = sc.apd(0.9)                      # APD90 of the settled beat
    di = cc.di(bcl, apd)                   # diastolic interval = BCL - APD
    apds.append(apd)
    dis.append(di)
    traces[bcl] = sc                       # keep the trace so we can draw it below
    print(f"BCL = {bcl:4d} ms   ->   APD90 = {apd:5.1f} ms   DI = {di:5.1f} ms")
"""),

(M, """Read down the `APD90` column: **226 → 214 → 198 → 191 → 187 → 183 ms.** As the cycle length
drops from 600 to 250 ms, every beat gets shorter — the cell shaves roughly 40 ms off its action
potential just by being driven harder. Meanwhile the diastolic interval (the recovery gap) collapses
far faster, from 374 ms down to 67 ms: at a fast rate the cell barely gets to rest before it is
kicked again.

Let's *see* the shortening. The loop kept each beat's trace, so we can overlay the slowest and fastest
of them on one plot:
"""),

(C, """slow, fast = traces[600], traces[250]

plt.figure(figsize=(7, 4))
plt.plot(slow.times, slow.V, label="BCL = 600 ms  (slow pacing)")
plt.plot(fast.times, fast.V, label="BCL = 250 ms  (fast pacing)")
plt.xlim(0, 350)
plt.xlabel("time (ms)")
plt.ylabel("membrane voltage (mV)")
plt.title("The same cell, paced slow vs. fast")
plt.legend()
plt.tight_layout()
plt.show()
"""),

(M, """Both beats launch together at the upstroke, but the fast-paced beat (orange) peels away and
returns to rest noticeably sooner — its plateau is cut short. Same cell, same machinery; the *only*
difference is how hard it was driven.
"""),

(M, """---
## 3. The restitution curve

Now the classic picture. Plot each beat's **APD** against the **diastolic interval that preceded it**.
This is the **restitution curve** — arguably the single most-studied relationship in cardiac
electrophysiology, because its shape predicts whether a heart will beat steadily or fall into chaos.
"""),

(C, """plt.figure(figsize=(7, 4))
plt.plot(dis, apds, "o-")
for bcl, di, apd in zip(bcls, dis, apds):
    plt.annotate(f"{bcl}", (di, apd), textcoords="offset points", xytext=(6, 6), fontsize=8)
plt.xlabel("diastolic interval  DI  (ms)   — recovery time before the beat")
plt.ylabel("APD90  (ms)   — how long the beat lasts")
plt.title("APD restitution curve  (labels = BCL in ms)")
plt.tight_layout()
plt.show()
"""),

(M, """Read it left to right. When the diastolic interval is **short** (left side — the cell had little
time to recover), the next action potential is **short**. Give the cell more recovery time (move right)
and the beat lengthens, until it flattens out toward its fully-rested duration. Every point on this
curve is one of the pacing rates from the loop; the labels are the BCL that produced it.

The *steepness* of this curve matters enormously. Where it rises steeply — the crowded left end, at
short DI — a small change in recovery time causes a large swing in the next beat's length. That
positive feedback is the seed of **alternans** (a long-short-long-short beat pattern) and, in tissue,
of the wave breakup you'll induce in Chapter 4.2. A flat restitution curve is a stable heart; a steep
one is a vulnerable one.
"""),

(M, """---
## 4. Effective rate and capture

There is a hard floor to how fast you can drive a cell. A stimulus that arrives while the cell is
still **refractory** — still repolarising from the previous beat, before it has recovered excitability
— simply fails to trigger a new action potential. The beat is *dropped*.

You can see the floor coming in the numbers above: at BCL = 250 ms the diastolic interval was already
down to just 67 ms. Push the cycle length lower still and the DI heads toward zero — the next stimulus
starts landing in the refractory tail of the last beat, and the cell can no longer answer every one.
When that happens the cell "captures" only every *other* stimulus (a 2:1 rhythm), so its **effective**
cycle length is twice what you asked for. The rate you *command* and the rate the cell actually *runs*
part ways.

This refractory floor is not a nuisance — it is a protection. It is the same mechanism that, in
Chapter 4.2, lets a well-timed stimulus land in a patch of still-recovering tissue and spin a wave
into a self-sustaining rotor.
"""),

(M, """### Try it yourself

**1. Push past the floor.** Add a very fast rate to the `bcls` list — change the first cell's list to
`bcls = [600, 450, 350, 300, 270, 250, 220]` and re-run. Watch the bottom of the printed table:
around this rate the diastolic interval collapses toward zero and the tidy APD-shortening trend breaks
down (the APD may even tick back *up*) — the first sign the cell is struggling to keep up, the edge of
capture.

**2. Change the cell type.** In the loop, change `celltype="EPI"` to `celltype="ENDO"`. Endocardial
cells have slightly different repolarisation, so the whole restitution curve shifts. Does the curve
move up or down?
"""),

(M, """---
## Recap

- **Pacing** drives a cell repeatedly at a fixed **basic cycle length (BCL)**; `cc.single_cell(bcl=…,
  pre_pace=…)` settles the cell at that rate and reports the steady-state beat.
- **APD shortens as BCL shortens** — over this sweep, APD90 fell from **226 ms at BCL 600** to
  **183 ms at BCL 250**, while the diastolic interval collapsed from **374 ms to 67 ms**.
- The **restitution curve** — APD plotted against the preceding diastolic interval — captures that
  rate-dependence in one line; its **steep** low-DI end is where alternans and, in tissue, wave
  breakup begin.
- There is a **refractory floor**: below some cycle length the cell can no longer capture every
  stimulus, and its effective rate diverges from the commanded one.

**Where next**: Chapter 4.2 takes pacing into a 2-D sheet of tissue. There, a first wave followed by a
second, well-timed stimulus — landing exactly in the recovering tail this chapter just described — can
break a wavefront and set a **reentrant rotor** spinning: an arrhythmia, built from scratch.
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
