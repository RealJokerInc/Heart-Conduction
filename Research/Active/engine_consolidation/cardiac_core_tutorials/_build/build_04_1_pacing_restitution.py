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

(M, """# Chapter 4.1 — Pacing & Restitution

Your heart doesn't beat once — it beats *again and again*, and it speeds up and slows down. This
chapter is about **rate**: what happens to a single heart cell when you pace it faster and faster.

The headline result is one of the most important facts in cardiac electrophysiology: **the faster you
pace a cell, the shorter each beat becomes.** A cell driven at rest takes its time; a cell driven hard
cuts every beat short. That rate-dependence is called **restitution**, and by the end of this notebook
you will have measured the restitution curve yourself.

**What you'll learn**

1. **Pacing and BCL** — driving a cell at a fixed cycle length
2. **APD shortens with rate** — pace faster, each beat briefer; seen in one beat and across a whole train
3. **The restitution curve** — plotting beat duration against recovery time
4. **Effective rate and capture** — drive the cell past its refractory floor and watch beats drop

This chapter stays with the **single cell** from Chapter 1 — no tissue yet. Chapter 4.2 takes pacing
into tissue and does something dramatic with it.

**Runtime**: about 3 minutes — the restitution sweep, three multi-beat pacing trains, and a fast-pacing
run, all settled simulations back to back. On Google Colab, add about a minute the first time for the
install.
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

(M, """### See the whole train

That overlay is two *single* beats. But pacing is a *train* of beats, and the real feel of "rate" is how
tightly those beats pack together. Here is the same cell recorded continuously at three rates, with a
**red dashed tick at every stimulus**:
"""),

(C, """T0  = 10.0                                   # single_cell fires its first stimulus at t = 10 ms
WIN = 1200.0                                 # a common time window (ms) to compare the rates in
train_bcls = [600, 350, 250]                 # slow, medium, fast

fig, axes = plt.subplots(len(train_bcls), 1, figsize=(8, 6), sharex=True)
for ax, bcl in zip(axes, train_bcls):
    n  = int(WIN / bcl) + 2                   # enough beats to fill the window
    sc = cc.single_cell("ttp06", celltype="EPI", bcl=bcl, n_beats=n, pre_pace=1, dt=0.05)
    ax.plot(sc.times, sc.V, color="C0")
    for k in range(n):                        # one tick per stimulus
        ax.axvline(T0 + k * bcl, color="crimson", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlim(0, WIN)
    ax.set_ylabel("V (mV)")
    ax.set_title(f"BCL = {bcl} ms   ({round(60000 / bcl)} bpm)", loc="left", fontsize=10)
axes[-1].set_xlabel("time (ms)    —    red dashed = stimulus")
fig.suptitle("The same cell, three pacing rates (each red tick is one stimulus)")
fig.tight_layout()
plt.show()
"""),

(M, """Top to bottom the beats crowd closer, because each stimulus lands sooner after the last. Watch
the *baseline between beats*: in the top (slow) panel the trace flattens out at rest for a long stretch
before the next tick; by the bottom (fast) panel that flat resting stretch has all but vanished — the
cell is yanked back up almost the instant it repolarises. That shrinking gap **is** the diastolic
interval collapsing, the very number that fell from 374 ms to 67 ms in the table. Push the rate higher
still and it runs out entirely — which is exactly where §4 goes.
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

(M, """### Watch a beat get dropped

Enough describing it — let's *drive* the cell past its floor and watch. We pace at **BCL = 170 ms**
(353 bpm), deep into the refractory zone, with a **modest, near-threshold stimulus**. That last detail
matters: during the refractory tail the cell's excitation threshold is *raised*, so a gentle pulse that
easily captures a rested cell now **fails outright** when it lands too soon — a cleanly dropped beat. (A
very strong shock could still force a diminished beat; that is the *relative* refractory period. A
near-threshold pulse tests excitability honestly.) Each stimulus is marked **green** if it fired a beat,
**red** if it was dropped:
"""),

(C, """T0 = 10.0                                      # single_cell fires its first stimulus at t = 10 ms
fast_bcl = 170                                 # ms — deep into the refractory zone (353 bpm)
n = 8
scf = cc.single_cell("ttp06", celltype="EPI", bcl=fast_bcl, n_beats=n,
                     pre_pace=0, dt=0.05, save_every=0.5,
                     stim_amplitude=-18.0)     # a modest, near-threshold test pulse

def fired(sc, ts):
    # Did the stimulus at time ts trigger a real upstroke? (a fast dV/dt spike within 8 ms of it)
    t, V = sc.times, sc.V
    dv = (V[1:] - V[:-1]) / (t[1:] - t[:-1])    # mV per ms
    m = (t[:-1] >= ts) & (t[:-1] <= ts + 8.0)
    return bool(m.any() and dv[m].max().item() > 20.0)

plt.figure(figsize=(8, 4))
plt.plot(scf.times, scf.V, color="C0")
for k in range(n):
    ts = T0 + k * fast_bcl
    plt.axvline(ts, color=("seagreen" if fired(scf, ts) else "crimson"), ls="--", lw=1.3)
plt.plot([], [], color="seagreen", ls="--", label="stimulus captured — a beat fires")
plt.plot([], [], color="crimson",  ls="--", label="stimulus dropped — cell still refractory")
plt.xlim(0, T0 + n * fast_bcl)
plt.xlabel("time (ms)")
plt.ylabel("membrane voltage (mV)")
plt.title("Paced at BCL = 170 ms — too fast: the cell captures only every other beat (2:1)")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()
"""),

(M, """Read it as a rhythm: **full beat — dropped — full beat — dropped.** Every red tick is a stimulus
that arrived while the cell was still repolarising; with its threshold raised, the gentle pulse can't
re-excite it — it raises at most a small bump on the falling tail of the previous beat, never a new
upstroke. Only the green ticks fire a real action potential. The cell has locked into **2:1 capture** —
one beat for every two stimuli — so
although you *commanded* 170 ms, it actually beats about every **340 ms**, half the rate you asked for.
Command and response have parted ways, exactly as this section opened. That same refractory gate, in a
2-D sheet, is what Chapter 4.2 exploits: a stimulus timed to fall in the recovering tail of a passing
wave doesn't merely drop — it can split the wavefront and seed a rotor.
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
- There is a **refractory floor**: below some cycle length a stimulus that lands in the refractory tail
  fails, and the cell drops into **2:1 capture** — driven at 170 ms it beats only every ~340 ms, half
  the commanded rate. We paced it there and watched the beats drop.

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
