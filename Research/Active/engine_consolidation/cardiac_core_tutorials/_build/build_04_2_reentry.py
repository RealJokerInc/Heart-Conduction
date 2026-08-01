"""Source of truth for `cardiac_core/tutorials/04_2_reentry.ipynb`
(Chapter 4.2 — "Pacing in Tissue: Voltage Clamp and Reentry").

Edit THIS file, not the notebook — a `.py` diff is reviewable where an `.ipynb` diff is not — then
re-run it to regenerate the notebook in place:

    python cardiac_core/tutorials/_build/build_04_2_reentry.py

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
NB_PATH = os.path.join(os.path.dirname(_HERE), "04_2_reentry.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Chapter 4.2 — Pacing in Tissue: Voltage Clamp and Reentry

Chapter 4.1 paced a single cell. Now we pace a whole **sheet** of tissue, and use pacing to do
something dramatic: **start an arrhythmia from scratch.**

Two ideas, one notebook:

1. **Voltage clamp** — the other way to drive tissue. So far you've *injected current* to pace. The
   complementary tool, the one behind the patch-clamp experiments cardiac cells are characterised with,
   is to *command the voltage* directly — pin a patch of tissue at a value you choose.
2. **Reentry** — a self-sustaining spinning wave (a **rotor**), the engine of the most dangerous
   arrhythmias. You'll induce one deliberately with a two-shock **S1–S2** protocol and confirm it
   mathematically.

**What you'll learn**

1. **Voltage clamp** — `cc.Stim(mask, clamp=…)` holds a region at a command voltage
2. **The S1–S2 protocol** — a planar wave, then a well-timed second stimulus in its recovering tail
3. **A rotor** — watch a wavefront break and curl into a spinning spiral
4. **Confirming reentry** — locating the rotor's core with a phase singularity

**Runtime**: about 75 seconds — the reentry simulation is the longest single run in these tutorials.
On Google Colab, add about a minute the first time for the install.
"""),

(M, """---
## Setup — install and import
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

print("cardiac_core is ready")
"""),

(M, """---
## 1. Voltage clamp — commanding the voltage

Every stimulus so far has *injected current*: `cc.Stim.boundary(g, "left", amplitude=-52)` pushes
charge into the tissue and lets the voltage respond. A **voltage clamp** does the opposite — it *holds*
a region at a voltage you name and lets whatever current is needed flow to keep it there. It is the
foundation of the patch-clamp rig used to characterise ion channels, and in `cardiac_core` it is one
keyword: `cc.Stim(mask, clamp=<mV>)`.

Here we take a small healthy sheet and clamp its left strip to **0 mV** for 40 ms — well above the
firing threshold. Watch what a *held* voltage does to the tissue around it.
"""),

(C, """g  = cc.Grid(80, 40, 0.02)                         # a 1.6 cm x 0.8 cm sheet
cond = cc.ConductivityConfig.bidomain(1.74, 6.25)   # healthy ventricle

clamp_region = cc.rectangle_mask(80, 40, 0.02, 0.0, 0.0, 0.1, g.Ly)   # a strip along the left edge
clamp = cc.Stim(clamp_region, clamp=0.0, start_time=1.0, duration=40.0)   # HOLD it at 0 mV for 40 ms

rc = cc.monodomain(g, "ttp06", cond, clamp, dt=0.1).run(t_end=60.0, save_every=1.0)
print("done — the left strip was held at 0 mV")
"""),

(M, """Compare a node *inside* the clamped strip with one far downstream; the two dotted verticals
bracket the 40 ms window during which the clamp is held:
"""),

(C, """rc.trace(at={"clamped strip (x=0.04 cm)": (2, 20),
             "downstream (x=0.90 cm)": (45, 20)},
         hline=(0.0, "command voltage"),
         vline=[(1.0, "clamp on"), (41.0, "clamp off")])   # bracket the 40 ms clamp window
"""),

(M, """The clamped node (first trace) is a **flat line pinned exactly at 0 mV** for the whole 40 ms
window — that is the clamp doing its job, holding the voltage no matter what. And because 0 mV is far
above threshold, that held strip pours current into its neighbours and **launches a travelling wave**:
the downstream node sits at rest until the wave arrives, then fires its own action potential. A
snapshot partway through shows the front leaving the clamped edge:
"""),

(C, """rc.image(at=8.0)   # voltage everywhere at t = 8 ms — a wave spreading from the clamped strip
"""),

(M, """So the two ways to drive tissue are: **inject current** (`amplitude=…`) and let the voltage
follow, or **clamp the voltage** (`clamp=…`) and let the current follow. Same tissue, two handles.

### Try it yourself

Change `clamp=0.0` to `clamp=-40.0` (right at the excitability threshold) and re-run. Does the strip
still launch a wave? Then try `clamp=-85.0` (the resting voltage) — now the strip is held at rest and
does nothing. The clamp *voltage* is the knob that decides whether the held region excites its
neighbours.
"""),

(M, """---
## 2. Reentry — inducing a rotor

Normally a wave crosses the tissue once and dies at the far edge, and the tissue waits for the next
beat from the pacemaker. **Reentry** is the catastrophe where a wave instead finds a way to *circle
back* and re-excite tissue that has just recovered — a wave that never leaves. A rotating reentrant
wave is a **rotor**, and rotors underlie the most dangerous arrhythmias (ventricular tachycardia and
fibrillation).

**How you start one — the S1–S2 protocol.** It takes two shocks:

- **S1**: a normal planar wave, launched from the left edge. It sweeps across, leaving behind it a
  refractory (recently-fired, not-yet-recovered) region that recovers *back-to-front* — the left
  recovers first.
- **S2**: a second, differently-placed shock, fired into that recovering tail at *just* the right
  moment. Where the tissue has recovered, S2 launches a new wave; where it is still refractory, S2
  fails. The new wave therefore has a **free end** — a broken wavefront — and a free end doesn't
  travel straight, it **curls**. That curl is the birth of a rotor.

**A note on scale (be honest about this).** In real healthy ventricle the reentry *wavelength*
(`CV × refractory period`, the length of tissue one wave occupies — you met its two ingredients in
Chapter 4.1) is over 10 cm: a rotor simply won't fit on a sheet you can simulate in seconds. So we
deliberately **shrink the wavelength** to make a rotor fit here — we weaken the coupling (a slower
wave) and shorten the action potential (a shorter refractory period). The *mechanism* is exactly the
real one; only the length scale is compressed onto a laptop.
"""),

(C, """g = cc.Grid(120, 120, 0.025)                          # a ~3 cm x 3 cm sheet

# Weak coupling -> a slow wave (short wavelength). 15% of healthy conductivity:
cond = cc.ConductivityConfig.bidomain(1.74 * 0.15, 6.25 * 0.15)

# S1: a planar wave from the left edge, fired at t = 1 ms.
s1 = cc.Stim.boundary(g, "left", start_time=1.0, duration=2.0, amplitude=-52.0)

# S2: a shock over the LOWER HALF of the sheet, fired at t = 180 ms into S1's recovering tail.
lower_half = cc.rectangle_mask(120, 120, 0.025, 0.0, 0.0, g.Lx, g.Ly / 2)
s2 = cc.Stim(lower_half, start_time=180.0, duration=2.0, amplitude=-52.0)

sim = cc.monodomain(g, "ttp06", cond, [s1, s2], dt=0.1)   # BOTH shocks, as a list

# Shorten the action potential -> shorter refractory period (the other half of the wavelength):
sim.scale_conductance("PCa", 0.4)    # weaken I_CaL (the plateau current)
sim.scale_conductance("GKs", 3.0)    # strengthen I_Ks  (a repolarising current)

r = sim.run(t_end=420.0, save_every=2.0)   # the long run — about a minute
print("done — now let's see what the two shocks produced")
"""),

(M, """The moment of truth. Play the movie:
"""),

(C, """r.video()   # the excitation wave over the whole 420 ms
"""),

(M, """Watch the sequence. The S1 wave sweeps left-to-right and fades. At 180 ms the S2 shock lights up
the lower half — but only the part that has recovered takes off, so the new wavefront has a **free
end** hanging in the middle of the sheet. That free end curls, wraps around, and by the end of the
movie it has closed into a **spinning spiral** — a rotor, turning around a core in the lower-left. The
wave is now feeding itself: it no longer needs the pacemaker. A still frame late in the run shows the
characteristic spiral:
"""),

(C, """r.image(at=400.0)   # a snapshot of the rotor, well after it has formed
"""),

(M, """---
## 3. Confirming the rotor — the phase singularity

The video is convincing, but we can pin the rotor down mathematically. At the exact centre of a
rotor — the pivot the spiral turns around — every phase of the action potential (resting, upstroke,
plateau, recovering) meets at a single point. That pivot is a **phase singularity**, and
`cardiac_core` finds it in two steps: `cc.phase_map` assigns every node a phase, and
`cc.phase_singularities` scans for the pivot as a **topological charge** — a value near **±1** marks a
rotor tip (the sign is the direction of rotation).
"""),

(C, """t_idx = len(r.times) - 15                 # a frame near the end, where the rotor is mature
phase = cc.phase_map(r.Vm, r.times, t_idx)
tips = cc.phase_singularities(phase)      # topological-charge map; |charge| ~ 1 = a rotor tip

strongest = float(tips.abs().max())
print(f"strongest phase singularity: |charge| = {strongest:.2f}")
print("a value near 1.0 confirms a rotor tip is present")
"""),

(M, """The strongest topological charge comes out at essentially **1.0** — a genuine phase
singularity, the mathematical fingerprint of the rotor core you watched form in the video. A planar
wave, or tissue that simply activated and recovered, would score near 0 everywhere. This number is how
an analysis pipeline detects and counts rotors automatically, without a human watching the movie.

### Try it yourself

The S2 timing is everything — it must land in the **vulnerable window**, the brief interval when part
of the tissue has recovered and part hasn't. In the reentry cell, change `start_time=180.0` on `s2`:

- Make it **too early** (`start_time=90.0`) — the tissue is still fully refractory, S2 does nothing.
- Make it **too late** (`start_time=280.0`) — the tissue has fully recovered, so S2 just launches a
  second clean planar wave, no free end.

Re-run and check `|charge|` each time: only a well-timed S2 (near 180 ms) drives it to ~1. That narrow
window is exactly why real arrhythmias are triggered by a premature beat falling at precisely the
wrong instant.
"""),

(M, """---
## Recap

- A **voltage clamp** (`cc.Stim(mask, clamp=<mV>)`) holds a region at a command voltage — the
  complement to injecting current. A depolarised clamp both pins its own nodes and drives its
  neighbours.
- **Reentry** is a wave that circles back and re-excites recovering tissue; a rotating one is a
  **rotor**, the engine of dangerous arrhythmias.
- The **S1–S2 protocol** induces one on purpose: a planar S1, then an S2 fired into S1's recovering
  tail (`cc.monodomain(..., [s1, s2], ...)`) breaks a wavefront so its free end curls into a spiral.
- A rotor fits on a small sheet only if you **shrink the wavelength** — weaker coupling (slower wave)
  and a shorter action potential (shorter refractory period), the two ingredients from Chapter 4.1.
- `cc.phase_map` + `cc.phase_singularities` **confirm** the rotor: a topological charge near **±1**
  marks its core — here **~1.0**.

**Where next**: Chapter 5 is the capstone — **tuning by hand**. You've now turned most of the knobs
(`ConductivityConfig`, `scale_conductance`, the stimulus, the grid). Chapter 5 turns them one at a
time, systematically, and measures exactly what each does to the wave: conduction velocity, wavefront
shape, block, and the wavelength you just used to fit a rotor onto a laptop.
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
