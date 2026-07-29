"""Source of truth for `cardiac_core/tutorials/02_1_conductances.ipynb`
(Chapter 2 — "Conductances & AP morphology", main tour).

Edit THIS file, not the notebook — a `.py` diff is reviewable where an `.ipynb` diff is not — then
re-run it to regenerate the notebook in place:

    python cardiac_core/tutorials/_build/build_02_1_conductances.py

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
NB_PATH = os.path.join(os.path.dirname(_HERE), "02_1_conductances.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Conductances & AP morphology

In Chapter 1 you ran a single heart cell and saw its **action potential** — the rest, upstroke,
plateau, and repolarisation that make up one beat. That shape is not arbitrary. It is sculpted by a
handful of **ion channels**, each carrying a current that dominates a different part of the beat. Turn
one channel up or down and a specific feature of the AP changes: the upstroke, the notch, the plateau
height, how long repolarisation takes.

This notebook is a guided tour of the six channels that shape the ventricular action potential. For
each one we take the same healthy cell, scale that single channel, and overlay the result on the
baseline so you can *see* exactly which part of the beat that channel controls.

**What you'll learn**

1. **The conductance knob** — scale any channel with one keyword, `conductances={NAME: factor}`
2. **A channel-by-channel tour** — `GNa`, `Gto`, `PCa`, `GKr`, `GKs`, `GK1`, and the AP feature each one owns
3. **Why the knob is safe** — a mistyped channel name is caught, not silently ignored

Everything runs on one 0-D cell (no tissue), so it is quick. Nothing asks more of you than editing a
number and re-running a cell.

**Runtime**: about a minute (seven short single-cell runs). On Google Colab add ~a minute the first
time for the install.
"""),

(M, """---
## Setup

The cell below installs `cardiac_core` if this environment doesn't already have it (e.g. a fresh
Colab runtime), then imports it under the short alias `cc`. If it is already installed the install
step does nothing.
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
## 1. The baseline cell, and the knob that changes it

First, the reference beat. This is the same `cc.single_cell` call from Chapter 1: one `ttp06`
epicardial cell, paced once to settle it (`pre_pace=1`), recorded for one beat. We pace at 2 Hz
(`bcl=500`, a beat every 500 ms) and use `dt=0.05` ms — small enough to be accurate, large enough to
be quick. Keep the two settings `BCL` and `DT` in one place so every run below matches.
"""),

(C, """BCL, DT = 500.0, 0.05        # pace at 2 Hz; dt = 0.05 ms is accurate and quick

base = cc.single_cell(
    "ttp06", celltype="EPI",
    pre_pace=1,                 # one warm-up beat, discarded, so the cell has settled
    bcl=BCL, dt=DT,
)

print(f"baseline  APD90 = {base.apd(0.9):.0f} ms   peak = {base.v_peak:.0f} mV   rest = {base.v_rest:.0f} mV")
base.trace(xlim=(0, 400))
"""),

(M, """That is our control: a beat that rests near **-85 mV**, spikes to about **+76 mV**, and lasts
**APD90 ≈ 219 ms**.

**What is APD90?** It stands for *action-potential duration at 90% repolarisation* — the time from the
upstroke until the cell has fallen **90% of the way** back down to its resting voltage. In plain terms:
how long the beat lasts. It is the single most-quoted number for an action potential, so it is the one
to watch in every overlay below. (You will also see **APD50** once — the same idea, measured to the
50% point, i.e. the duration of the first half of the beat.)

Every overlay below compares a single-channel change against exactly this curve.

**The knob.** To change one channel, pass `conductances={NAME: factor}` to `single_cell`. The factor
is **multiplicative**: `0.5` halves that channel's maximum conductance, `1.5` boosts it by 50%. Two
design points worth knowing:

- The change is applied **before** the warm-up beat, so `pre_pace` settles the *changed* cell to *its
  own* steady state. You are comparing two cells that have each reached equilibrium — an honest
  comparison, not a control cell nudged once.
- The channel names are the model's own (`GNa`, `GKr`, `PCa`, …) — we'll meet each below.

To keep every comparison identical except for the one channel we change, here is a tiny helper. It
runs a cell with one channel scaled, overlays it on `base`, and prints the headline numbers. The only
thing you ever change when calling it is the `{NAME: factor}` dictionary.
"""),

(C, """def compare(title, conductances):
    \"\"\"Run a cell with one channel scaled, overlay it on the baseline, show the plot, print the numbers.\"\"\"
    scaled = cc.single_cell("ttp06", celltype="EPI", pre_pace=1, bcl=BCL, dt=DT,
                            conductances=conductances)
    # .show() displays the overlay inline (like plt.show()). Calling cc.draw()
    # inside a function and NOT showing it would render nothing.
    cc.draw(cc.Trace({"baseline": (base.times, base.V),
                      title: (scaled.times, scaled.V)},
                     xlabel="time (ms)", ylabel="Vm (mV)", xlim=(0, 400))).show()
    print(f"{title:12s}  APD90 = {scaled.apd(0.9):.0f} ms   peak = {scaled.v_peak:.0f} mV   rest = {scaled.v_rest:.0f} mV")
"""),

(M, """---
## 2. Phase 0 — `GNa`: the upstroke

The **fast sodium current** `I_Na` fires the cell. When the cell is excited, sodium channels snap
open and sodium floods in, driving the near-vertical **upstroke** (phase 0). Its conductance is
`GNa`. Cut it and the cell is harder to excite — the upstroke is weaker and doesn't overshoot as
high. (In tissue this same reduction *slows conduction*, the story of Chapter 5.)
"""),

(C, """compare("GNa x0.3", {"GNa": 0.3})   # 70% sodium block
"""),

(M, """The overshoot drops from about **+76 mV to +54 mV**: with less sodium rushing in, the spike
simply doesn't reach as high. The rest of the beat is barely touched — `GNa` owns the upstroke and
little else.
"""),

(M, """---
## 3. Phase 1 — `Gto`: the notch

Immediately after the spike, epicardial cells show a small dip — the **phase-1 notch**, giving the
"spike-and-dome" look. It is carved by the **transient outward potassium current** `I_to` (conductance
`Gto`), a brief outward current that flickers on right after the upstroke. More `I_to`, deeper notch.
"""),

(C, """compare("Gto x2.0", {"Gto": 2.0})   # double the transient outward current
"""),

(M, """Doubling `Gto` deepens the notch dramatically — the voltage right after the spike plunges from
about **+15 mV down to -11 mV** before the plateau recovers, and the peak itself is pulled down (76 →
64 mV). Turn `Gto` the other way (try `{"Gto": 0.0}`) and the notch vanishes entirely, leaving a
higher, smoother dome. The size of this notch is the main thing that distinguishes the heart's outer
(epicardial) cells from its inner (endocardial) ones.
"""),

(M, """---
## 4. Phase 2 — `PCa`: the plateau

The long high **plateau** (phase 2) is the signature of a heart cell. It is held up by the **L-type
calcium current** `I_CaL`: calcium flowing *in* roughly balances potassium flowing *out*, and the cell
lingers near the top for ~200 ms. Its knob is **`PCa`** — a *permeability*, not an ohmic conductance,
which is why it is named `PCa` and not "`GCaL`". Less calcium, lower and shorter plateau, shorter beat.
"""),

(C, """compare("PCa x0.5", {"PCa": 0.5})   # 50% L-type calcium block
"""),

(M, """Halving the calcium current collapses the plateau: **APD90 falls from 219 ms to 166 ms** and the
first-half duration **APD50 nearly halves** too (166 → 109 ms). The upstroke peak is untouched —
calcium doesn't fire the cell, it *sustains* it. Push the knob the other way (`{"PCa": 1.5}`) and the
plateau swells, stretching APD90 out to ~246 ms.
"""),

(M, """---
## 5. Phase 3 — `GKr`: repolarisation, and the sign flip

Repolarisation (phase 3, the fall back to rest) is driven by **delayed-rectifier potassium currents**
that switch on during the plateau and eventually win. The **rapid** one is `I_Kr` (conductance `GKr`).
Here the tour flips sign. Every change so far *reduced* a current and got a *smaller* beat; watch what
happens when we reduce this one: **the beat gets LONGER, not shorter.** With less repolarising current,
it takes more time to bring the voltage back down.
"""),

(C, """compare("GKr x0.5", {"GKr": 0.5})   # GKr halved — the rapid repolarising K current
"""),

(M, """Halving `I_Kr` **prolongs APD90 from 219 ms to 240 ms** — the tail of the beat drags out. Sit
with the sign for a moment: this is an *outward* current, and cutting it made the beat *longer*, because
this current's whole job is to *end* the beat. Reduce what ends the beat and the beat runs on. The next
channel does the same thing for the same reason.
"""),

(M, """---
## 6. Phase 3 again — `GKs`: the repolarisation reserve

`I_Kr` has a slower sibling: the **slow** delayed-rectifier `I_Ks` (conductance `GKs`). It also helps
repolarise, but it builds up over successive fast beats, so it matters most when the heart is racing —
it is the "**repolarisation reserve**" that keeps the plateau from running away at speed.
"""),

(C, """compare("GKs x0.5", {"GKs": 0.5})   # GKs halved — the slow repolarising K current
"""),

(M, """Halving `I_Ks` also prolongs the beat (**APD90 219 → 242 ms** here at 2 Hz). On its own the
effect looks a lot like `GKr` — another outward current, another longer beat. But the two share the
load: with `I_Kr` already reduced, losing `I_Ks` as well removes the backup and repolarisation nearly
stalls. Because `I_Ks` builds up over successive fast beats, this reserve matters most when the cell is
paced quickly.
"""),

(M, """---
## 7. Phase 4 — `GK1`: the resting potential

Between beats the cell sits at a steady, deeply negative **resting potential**. That floor is held by
the **inward-rectifier potassium current** `I_K1` (conductance `GK1`), which also sharpens the very
last, steep part of repolarisation as the cell drops back to rest.
"""),

(C, """compare("GK1 x0.5", {"GK1": 0.5})   # 50% inward-rectifier block
"""),

(M, """With `I_K1` halved the resting voltage drifts **up from -85.4 mV to -84.1 mV** — the floor is
less firmly held — and the final return to rest is a touch slower and less sharp (APD90 219 → 227 ms).
`I_K1` is the anchor: it sets where the cell waits and how crisply it gets back there.
"""),

(M, """---
## The six channels, at a glance

| Phase | Channel | Current | Scale it down and… |
|---|---|---|---|
| 0 upstroke | `GNa` | `I_Na` fast sodium | weaker upstroke, lower overshoot (peak 76 → 54 mV at 0.3×) |
| 1 notch | `Gto` | `I_to` transient outward K | shallower notch (deeper if scaled *up*: to -11 mV at 2×) |
| 2 plateau | `PCa` | `I_CaL` L-type calcium | lower, shorter plateau (APD90 219 → 166 ms at 0.5×) |
| 3 repol | `GKr` | `I_Kr` rapid K | **longer** beat (APD90 219 → 240 ms at 0.5×) |
| 3 repol | `GKs` | `I_Ks` slow K | **longer** beat (APD90 219 → 242 ms at 0.5×) |
| 4 rest | `GK1` | `I_K1` inward-rectifier K | resting V rises (-85.4 → -84.1 mV at 0.5×) |

The pattern to carry away: **inward** currents (`GNa`, `PCa`) *build* the action potential — reduce them
and it shrinks; **outward** potassium currents (`GKr`, `GKs`, `GK1`, and `Gto`) *end* it — reduce the
repolarising ones and the beat drags on *longer*. That sign flip — an outward current making the beat
**longer** when you cut it — is the single idea to carry out of this notebook.
"""),

(M, """---
## 8. The knob is safe: bad names are caught

One more thing worth trusting the tool for. If you mistype a channel name — a lower-case `g`, an
invented `GCaL` — `single_cell` does **not** quietly run the baseline and hand you a null result.
It **stops and tells you**, listing the real names. Run this and read the error:
"""),

(C, """try:
    cc.single_cell("ttp06", conductances={"gKr": 0.5})   # lower-case 'g' — a typo
except ValueError as e:
    print("Rejected, as it should be:\\n", e)
"""),

(M, """That guarantee is the whole reason to use the `conductances=` knob instead of reaching into the
model by hand: when a change appears to do nothing, you can trust it really did nothing — you didn't
just fat-finger the name. The error even prints the exact list of channels you *can* scale.
"""),

(M, """---
### Try it yourself

1. **Go the other way — turn a channel UP.** Every example above *reduced* a channel. In the `GNa`
   cell (section 2), change `{"GNa": 0.3}` to `{"GNa": 3.0}`, then `{"GNa": 8.0}`, and re-run. The
   upstroke steepens and the overshoot climbs — but watch *how high*: the peak marches from ~+76 mV to
   about **+95 mV, then +102 mV**, far above the ~+40 mV a real ventricular cell ever reaches, while the
   beat's *duration* barely moves (APD90 stays near 210 ms — `GNa` owns the upstroke and nothing after
   it). That runaway overshoot is the interesting part: the model faithfully computes whatever
   conductance you hand it, physiological or not. A knob that sails off into impossible territory is a
   property of the simulator, not a warning it will give you — deciding what is realistic is *your* job.
2. **Find calcium's dose-response.** In the `PCa` cell (section 4), change `{"PCa": 0.5}` to `0.25`,
   then `0.75`, then `1.5`. Watch APD90 (printed under each plot) rise and fall with the calcium
   current — more calcium, longer plateau.
3. **Combine two changes.** The knob takes more than one channel at once. In any `compare(...)` cell,
   try `{"GKr": 0.5, "GKs": 0.5}` — reduce both delayed rectifiers together. The prolongation is larger
   than either alone: cut one repolarising current and the other partly covers for it; cut both and there
   is no backup left.
4. **Break it on purpose.** In a `compare(...)` cell, mistype a name — `{"gto": 2.0}` or
   `{"GCaL": 0.5}` — and read the `ValueError`. The tour helper lets the error surface, so a typo never
   passes silently.
"""),

(M, """---
## Recap

- One keyword, **`conductances={NAME: factor}`**, scales any single ion channel of the cell. The
  factor is multiplicative (`0.5` = half = a 50% block), applied *before* pre-pacing so you compare two
  settled cells.
- Each channel owns a feature of the AP: **`GNa`** the upstroke, **`Gto`** the phase-1 notch, **`PCa`**
  the plateau, **`GKr`** and **`GKs`** the speed of repolarisation, **`GK1`** the resting potential.
- **Inward** currents (Na, Ca) build the beat; **outward** K currents end it — so reducing a
  repolarising potassium channel (`GKr`/`GKs`) makes the beat *longer*, not shorter: the sign flip.
- Mistyped channel names **raise** rather than silently doing nothing — so a change that appears to do
  nothing really did nothing.

**Where next**: Chapter 2A takes a different kind of cell — a stem-cell-derived one that **beats on its
own**, with no electrode — and meets the "funny" current that sets its rhythm. Then Chapter 3 opens up
tissue, where these same channels change not just the shape of a beat but the speed and safety of the
*wave* that carries it.
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
