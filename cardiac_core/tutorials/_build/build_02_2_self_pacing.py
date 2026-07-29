"""Source of truth for `cardiac_core/tutorials/02_2_self_pacing.ipynb`
(Chapter 2A — "Self-pacing & the funny current").

Edit THIS file, not the notebook — a `.py` diff is reviewable where an `.ipynb` diff is not — then
re-run it to regenerate the notebook in place:

    python cardiac_core/tutorials/_build/build_02_2_self_pacing.py

Emits both the `.ipynb` (plain JSON, no `nbformat` dependency) and, with `--script PATH`, a flat `.py`
of every code cell for headless verification.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(os.path.dirname(_HERE), "02_2_self_pacing.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Self-pacing & the funny current

Every action potential so far has needed an **electrode**: we injected a stimulus to fire the cell.
But the heart doesn't wait to be poked — its pacemaker cells fire *by themselves*, over and over,
setting the rhythm of the whole organ. So do the human stem-cell-derived heart cells (**hiPSC-CMs**)
grown in dishes and on tissue chips, which is exactly why they are so useful in the lab: put them in a
dish and they beat spontaneously.

This short advanced notebook switches from the adult-ventricle `ttp06` model to **`paci`** (Paci et
al. 2013), a model of a spontaneously-active hiPSC cardiomyocyte. We'll watch it beat on its own with
no stimulus at all, meet the current behind that automaticity — the delightfully named **"funny"
current, `I_f`** — slow the rhythm by turning it down, and finally *silence* the beating altogether by
switching to the cell's **matured** form, `mhas13`.

**What you'll learn**

1. **Automaticity** — run a cell that fires repeatedly with `stim_amplitude=0` (no electrode)
2. **Diastolic depolarisation** — the slow drift toward threshold that *is* the pacemaker
3. **The funny current `g_f`** — scale it and watch the spontaneous rate change
4. **Maturation → quiescence** — meet `mhas13`, the matured hiPSC cell that sits silent until stimulated

**Runtime**: a couple of minutes (three hiPSC-model runs). These models are stiffer than `ttp06`, so we
use a larger step `dt=0.1` and a few seconds of simulated time to capture several spontaneous beats.
"""),

(M, """---
## Setup

Install `cardiac_core` if needed (a no-op if you already have it), then import it as `cc`.
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
## 1. A cell that beats on its own

The single change that makes this notebook different from Chapter 2 is one keyword:
**`stim_amplitude=0.0`** — no stimulus current at all. A `ttp06` cell given no stimulus would just sit
at rest forever. A `paci` cell doesn't: it fires anyway. We simulate 3.5 seconds so several beats fit,
at the larger step `dt=0.1` the stiffer model is happy with.
"""),

(C, """paci = cc.single_cell(
    "paci",                 # a spontaneously-active hiPSC cardiomyocyte model
    stim_amplitude=0.0,     # NO electrode — the cell fires on its own
    n_beats=1, bcl=3500,    # record one 3.5-second stretch
    dt=0.1,                 # larger step for the stiffer Paci model
)

print(f"most negative point (MDP) = {paci.v_rest:.0f} mV   peak = {paci.v_peak:.0f} mV")
paci.trace(xlim=(0, 3500))
"""),

(M, """No electrode, and yet the cell fires — **twice** in this window, entirely on its own. That is
**automaticity**. Look closely at what happens *between* the spikes: the voltage never sits still.
After each beat the cell relaxes to its most-negative point (the **maximum diastolic potential**, about
**-77 mV** here) and then immediately begins a slow, steady climb back up. That upward creep is
**diastolic depolarisation** — the pacemaker ramp. When it drags the voltage up to threshold (~-60 mV),
the next spike fires, and the cycle repeats.

The two upstrokes here sit about **1.6 seconds apart** (at ~590 ms and ~2230 ms), so this cell is
free-running at roughly **0.6 Hz — about 37 beats per minute**. The whole rhythm is set by *how steeply*
that diastolic ramp climbs: a steeper climb reaches threshold sooner and beats faster.
"""),

(M, """---
## 2. The funny current tunes the pace

What drives the diastolic ramp? Several currents contribute, but the emblematic one is the **"funny"
current, `I_f`**. Its story is worth a paragraph, because the name *is* the biology.

**Where the name comes from.** In the late 1970s, Dario DiFrancesco and colleagues, recording from
cardiac pacemaker tissue, found a current that broke the usual rule. Almost every channel opens when
the cell **depolarises** (gets *less* negative); this one did the opposite — it opened when the cell
**hyperpolarised**, i.e. right after a beat, when the voltage had fallen to its most negative. That was
so contrary to expectation that they simply called it the *funny* current, `I_f`, and the name stuck.

**The channel and its ions.** `I_f` flows through **HCN channels** — *hyperpolarisation-activated
cyclic-nucleotide-gated* channels — of which pacemaker cells use mainly **HCN4**. Two properties make it
a pacemaker current:

- It is a **mixed inward current of Na⁺ and K⁺**: when it opens, both ions cross, and the *net* flow is
  **depolarising** — it nudges the voltage back **up**.
- It opens on **hyperpolarisation**, so it switches on exactly when a beat ends and then carries the
  slow climb (the diastolic ramp) toward the next threshold. More `I_f` → steeper ramp → faster rate.

The *cyclic-nucleotide* part of the name is how the body sets its own heart rate: cAMP binds the
channel, and that is the lever adrenaline uses to speed you up and the vagus nerve to slow you down —
both act on `I_f`.

In `paci` its conductance is **`g_f`**. (Note the lower-case `g_` — the hiPSC models name their
channels differently from `ttp06`'s upper-case `G`; more on that below.) Let's turn it down to a
quarter and overlay the result on the free-running baseline.
"""),

(C, """slow = cc.single_cell("paci", stim_amplitude=0.0, n_beats=1, bcl=3500, dt=0.1,
                      conductances={"g_f": 0.25})   # 75% block of the funny current

cc.draw(cc.Trace({"baseline": (paci.times, paci.V),
                  "g_f x0.25": (slow.times, slow.V)},
                 xlabel="time (ms)", ylabel="Vm (mV)", xlim=(0, 3500))).show()   # .show() renders it even though a print() follows

print(f"diastolic low point:  baseline = {paci.v_rest:.0f} mV,  g_f-blocked = {slow.v_rest:.0f} mV")
"""),

(M, """With the funny current cut to a quarter, the diastolic ramp climbs more slowly, so the cell
takes **longer to reach threshold** — every beat is delayed. The first spontaneous beat slips from
~590 ms to ~925 ms, and the beat-to-beat interval stretches from about **1.64 s to 1.81 s** (roughly
**0.61 → 0.55 Hz**, ~37 → ~33 beats per minute). The cell also settles a little deeper between beats
(the printed diastolic low point drops from about **-77 mV to -79 mV**). Because each beat is a little
later than the last, the orange trace drifts steadily to the right of the blue one — a slower pacemaker,
drawn out in front of you. Turn the funny current down and the rhythm slows: the ramp *is* the rate.
"""),

(M, """---
## 3. Silencing the pacemaker: maturation and `mhas13`

Turning `g_f` *down* slowed the cell — but it never stops. Even a **complete** block (`{"g_f": 0.0}`)
leaves this `paci` cell crawling along, only a little slower. `I_f` is not the whole story: a young
hiPSC cell also has very little **`I_K1`**, the strong inward-rectifier potassium current that in an
adult cell pins the voltage to a firm resting floor. With almost no `I_K1` holding it down, the cell
drifts back up to threshold no matter what the funny current does. To truly silence it you must change
*both* — which is exactly what **maturation** does.

`cardiac_core` ships the matured cell as **`mhas13`**. It is the same `paci` cell with the two-step
recipe from the maturation literature (Verkerk 2019):

- **suppress the funny current** — `g_f = 0` (automaticity is a developmental trait, switched off), and
- **inject an adult inward rectifier** — a strong TTP06-style `I_K1` at the critical conductance.

Run it with no electrode — the same `stim_amplitude=0.0` that made `paci` beat — and compare the two:
"""),

(C, """mature = cc.single_cell(
    "mhas13",               # the MATURED hiPSC model: g_f = 0 AND an injected adult I_K1
    stim_amplitude=0.0,     # no electrode — exactly as the paci cell above
    n_beats=1, bcl=3500, dt=0.1,
)

cc.draw(cc.Trace({"paci (immature)":  (paci.times, paci.V),
                  "mhas13 (matured)": (mature.times, mature.V)},
                 xlabel="time (ms)", ylabel="Vm (mV)", xlim=(0, 3500))).show()

print(f"mhas13, no electrode:  resting V = {mature.v_rest:.0f} mV   (flat line = no spontaneous beats)")
"""),

(M, """**Nothing happens.** The matured cell sits flat at about **-85 mV** — no diastolic drift, no
spikes — while the immature `paci` cell beats twice in the same window. That flat line is
**quiescence**: a strong `I_K1` clamps the resting potential, and with no `I_f` to lift the cell off
that floor there is no pacemaker at all. It is not broken — give `mhas13` a real stimulus and it fires a
clean action potential like any ventricular cell (peak ~+58 mV) — it simply never fires *on its own*.

This is the whole reason `mhas13` exists, and it points straight at Chapter 3. **Tissue simulation needs
quiescent cells.** A sheet wired from self-firing `paci` cells would ignite everywhere at once, with no
resting tissue for a wave to travel *into*. So the division of labour is: **`paci` is the cell for
studying a rhythm; `mhas13` is the cell you build tissue from** — matured, silent, and ready to be
excited by a passing wave.
"""),

(M, """---
## A note on names: hiPSC models use lower-case `g_`

The name-validation you can trust from Chapter 2 lists *this* model's channels. Because `paci` names
its conductances lower-case (`g_Na`, `g_Kr`, `g_f`, …) rather than `ttp06`'s upper-case (`GNa`, `GKr`),
using a `ttp06`-style name here is a typo — and, as before, it is caught, not ignored. Run this to see
the funny-current cell's real vocabulary printed back at you:
"""),

(C, """try:
    cc.single_cell("paci", stim_amplitude=0.0, conductances={"g_F": 0.25})   # wrong case for Paci
except ValueError as e:
    print("Rejected (note the available names are the hiPSC lower-case set):\\n", e)
"""),

(M, """---
### Try it yourself

1. **Speed it up instead.** In the section-2 cell, change `{"g_f": 0.25}` to `{"g_f": 1.5}`. A
   *stronger* funny current makes the diastolic ramp steeper, so the cell reaches threshold sooner and
   beats *faster* — the mirror image of turning it down.
2. **Watch it longer.** Raise `bcl=3500` to `bcl=6000` in both cells to capture more beats and see the
   rate difference accumulate. (It costs a little more runtime — the run is proportional to the window.)
"""),

(M, """---
## Recap

- Some heart cells — real pacemaker cells, and hiPSC-CMs in a dish — **fire on their own**. Run one
  with `stim_amplitude=0.0` (no electrode) and it beats repeatedly: **automaticity**.
- The rhythm comes from **diastolic depolarisation**, the slow climb from the diastolic potential up to
  threshold between beats. Steeper climb → faster rate.
- The **funny current `I_f`** (conductance `g_f`) — carried by **HCN channels**, a mixed inward Na⁺/K⁺
  current that opens on hyperpolarisation — is the emblematic pacemaker current. Turn it down
  (`conductances={"g_f": 0.25}`) and the ramp flattens and the rate **slows**. (In *this* model,
  removing it entirely only slows the cell — other currents keep it beating; `I_f` tunes the rhythm, it
  is not the sole engine.)
- hiPSC models name their channels **lower-case `g_*`**; a wrong-case name is a typo, and it is caught.

**Where next**: you have now met the cell from two sides — what shapes one beat (Chapter 2) and what
sets the rhythm of many (here). Chapter 3 leaves the single cell behind and builds **tissue**, where
these cells are wired together and a beat becomes a travelling *wave*.
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
