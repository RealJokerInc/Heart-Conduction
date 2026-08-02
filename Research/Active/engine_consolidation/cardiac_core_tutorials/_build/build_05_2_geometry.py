"""Source of truth for `cardiac_core/tutorials/05_2_geometry.ipynb`
(Chapter 5.2 — "Tuning by hand: direction, geometry and launch").

Edit THIS file, not the notebook, then regenerate:
    python cardiac_core/tutorials/_build/build_05_2_geometry.py
Emits the `.ipynb` (plain JSON) and, with `--script PATH`, a flat `.py` for headless verification.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(os.path.dirname(_HERE), "05_2_geometry.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Chapter 5.2 — Direction, Geometry & Launch

Notebook 5.1 was about **how fast** the wave goes. This one is about **what shape** it takes and
**whether it launches at all**. Same method — turn one knob, run, look at the wave — but now the thing
we watch is the *wavefront* itself, drawn as an **activation map**: every point coloured by the moment
the wave reached it, with contour lines (isochrones) joining points that lit up together. The shape of
those contours tells you everything.

**What you'll learn**

1. **Fibre direction bends the wave** — with equal conductivity in every direction the front is a
   **circle**; with fibres (anisotropy) it is an **ellipse**, and the axis ratio is set by the
   conductivities.
2. **A scar makes the wave curve — and can stop it dead** — the front detours around inexcitable
   tissue, and a narrow enough gap **blocks** it entirely (a source–sink mismatch).
3. **A wave has to be launched** — a stimulus that is too *weak*, or fired on too *small* a patch,
   never starts a wave at all.

**Runtime**: about a minute (each map is one short tissue run). On Colab add ~a minute the first time
for the install.
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

print("cardiac_core is ready")
"""),

(M, """---
## 1. Isotropy — a round wave

To watch a wave's *shape* we stimulate a single spot in the **middle** of a square sheet and let the
excitation spread outward in every direction. `cc.Stim.center(g, radius=0.05)` does that — a small round
electrode at the centre. (The `radius=0.05` matters, and section 3 will show why: too small a spot never
launches a wave.)

First, **isotropic** tissue: `ConductivityConfig.isotropic(...)` gives the sheet the *same* conductivity
in every direction. With no preferred direction, the wave should spread as a perfect circle. We measure
how fast it travels outward along x and along y with `r.cv_between(...)`, and draw the activation map.
"""),

(C, """g    = cc.Grid(81, 81, 0.01)                       # a 0.8 cm x 0.8 cm square sheet
stim = cc.Stim.center(g, radius=0.05, start_time=1.0, duration=2.0, amplitude=-80.0)

cond_iso = cc.ConductivityConfig.isotropic(1.36)   # same conductivity in every direction
r_iso = cc.monodomain(g, "ttp06", cond_iso, stim, dt=0.1).run(t_end=25.0, save_every=0.5)

c = (40, 40)                                        # the centre node (where we stimulated)
cv_x = r_iso.cv_between(c, (60, 40))               # outward speed along x
cv_y = r_iso.cv_between(c, (40, 60))               # outward speed along y
print(f"outward CV along x = {cv_x:.1f} cm/s")
print(f"outward CV along y = {cv_y:.1f} cm/s")
"""),

(C, """r_iso.image(what="activation")   # colour each point by when the wave arrived; contours = isochrones
"""),

(M, """The two speeds are equal (~57.5 cm/s each) and the isochrones are **concentric circles**. Nothing in
the tissue prefers one direction, so the wave doesn't either. This is the reference shape; everything
below is a departure from it.
"""),

(M, """---
## 2. Anisotropy — an elliptical wave

Real muscle is built from **fibres**, and current flows more easily *along* a fibre than *across* it.
`ConductivityConfig.anisotropic(sigma_l, sigma_t, fiber_angle)` captures that with two conductivities —
`sigma_l` **along** the fibre (longitudinal) and `sigma_t` **across** it (transverse) — and an angle
setting the fibre direction (`0` = fibres along x, in radians). Here we make along-fibre conduction
4× the cross-fibre value and keep the fibres horizontal.
"""),

(C, """cond_aniso = cc.ConductivityConfig.anisotropic(1.74, 0.435, 0.0)   # sigma_l : sigma_t = 4 : 1, fibres along x
r_aniso = cc.monodomain(g, "ttp06", cond_aniso, stim, dt=0.1).run(t_end=28.0, save_every=0.5)

cv_L = r_aniso.cv_between(c, (60, 40))    # along the fibre (x) — longitudinal
cv_T = r_aniso.cv_between(c, (40, 60))    # across the fibre (y) — transverse
print(f"CV along fibre  (CV_L) = {cv_L:.1f} cm/s")
print(f"CV across fibre (CV_T) = {cv_T:.1f} cm/s")
print(f"ratio CV_L / CV_T = {cv_L / cv_T:.2f}   (sqrt of the conductivity ratio sqrt(4) = 2.0)")
"""),

(C, """r_aniso.image(what="activation")   # now an ellipse, stretched along the fibre direction (x)
"""),

(M, """The wave now travels about **twice as fast along the fibres as across them** (~69 vs ~33 cm/s), and
the isochrones are **ellipses** stretched along x. The axis ratio isn't the 4:1 of the conductivities —
it's the **square root**, ≈ 2:1, because (as notebook 5.1 showed) velocity goes as √σ. Fibre direction
is why a real wavefront in the heart is elliptical, not round, and why the same tissue can look like it
conducts at two different speeds depending on which way you measure.
"""),

(M, """### Try it yourself

Rotate the fibres. Change the angle in `anisotropic(1.74, 0.435, 0.0)` from `0.0` to `0.785`
(that's 45°, or π/4 radians) and re-run the two cells above. The ellipse tilts to 45° — the wavefront
follows the fibres wherever you point them. (There is one global fibre angle here; per-region fibre
fields are a heavier tool than this by-hand tour needs.)
"""),

(M, """---
## 3. A scar — curving and blocking

Dead tissue — a **scar** — doesn't conduct. You carve one by masking a region and setting its
conductivity to zero with `sim.set_conductivity(mask, D=0.0)`. The wave can't cross it, so it has to go
**around**. We build a wall most of the way across a strip, leaving a single narrow **isthmus** gap for
the wave to squeeze through, and fire from the left as usual.

Start with a **wide** gap (0.06 cm). Watch what the front does on the far side.
"""),

(C, """def strip_with_isthmus(gap_width):
    \"\"\"A strip blocked by a wall except for a gap of `gap_width` cm at mid-height.\"\"\"
    Nx, Ny, dx = 101, 61, 0.01
    g    = cc.Grid(Nx, Ny, dx)                          # 1.0 cm x 0.6 cm strip
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25)
    stim = cc.Stim.boundary(g, "left", start_time=1.0, duration=2.0, amplitude=-80.0)
    sim  = cc.monodomain(g, "ttp06", cond, stim, dt=0.1)
    lo, hi = 0.30 - gap_width / 2, 0.30 + gap_width / 2   # the gap spans lo..hi in y, around y=0.30
    wall_top = cc.rectangle_mask(Nx, Ny, dx, 0.44, hi,  0.54, 0.60)
    wall_bot = cc.rectangle_mask(Nx, Ny, dx, 0.44, 0.0, 0.54, lo)
    sim.set_conductivity(wall_top | wall_bot, D=0.0)      # the wall is inexcitable scar
    return sim.run(t_end=42.0, save_every=0.5)

r_pass = strip_with_isthmus(0.06)     # a wide-ish gap
r_pass.image(what="activation")
"""),

(M, """The flat front reaches the wall, funnels through the gap, and on the far side **re-expands as a
curved wave** — the isochrones fan out in arcs from the mouth of the isthmus, like ripples from a new
point source. The wave survived, but its shape was rewritten by the geometry.

Now make the gap **narrow** (0.02 cm) and run it again:
"""),

(C, """r_block = strip_with_isthmus(0.02)    # a narrow gap
r_block.image(what="activation")
"""),

(M, """This time the far side stays **dark** — the wave **blocked**. The left half activates and the front
reaches the gap, but it dies there and never re-ignites the tissue beyond.

Why? A **source–sink mismatch**. The sliver of tissue in a narrow gap is a small *source* of current; the
wide sheet waiting on the other side is a big *sink* that has to be charged up to threshold. Through a
wide gap the source is large enough to do it; through a narrow gap it isn't, and conduction fails. This
is not a bug or a resolution problem — it is real electrophysiology, and it is exactly how a strand of
surviving muscle through a scar can conduct in one beat and block in the next. (`cardiac_core` exposes the
underlying map as `r.fields.source_sink`, and `cc.safety_factor(r)` — below 1 means block — if you want
to quantify it.)
"""),

(M, """---
## 4. Launching a wave — strength and size

So far we've assumed the stimulus starts a wave. It doesn't always. Igniting tissue is itself a
source–sink problem: the stimulated patch has to drag its resting neighbours up to threshold before it
recovers. Two things decide whether it wins — **how hard** you stimulate and **how big** a patch you
stimulate.

**Strength first.** Fire the left edge of a plain strip at a few amplitudes and ask for the conduction
velocity — which comes back as **`nan`** ("not a number") when no wave ever forms (`amplitude` is in
µA/µF; more negative = stronger depolarising push).
"""),

(C, """import math

def launch_cv(amplitude):
    g    = cc.Grid(101, 31, 0.01)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25)
    stim = cc.Stim.boundary(g, "left", start_time=1.0, duration=2.0, amplitude=amplitude)
    cv   = cc.monodomain(g, "ttp06", cond, stim, dt=0.1).run(t_end=25.0, save_every=0.5).cv(20, 70, 15)
    return cv                                      # nan if no wave ever formed

for amp in [-40.0, -52.0]:
    cv = launch_cv(amp)
    verdict = "nothing - no wave" if math.isnan(cv) else f"WAVE at {cv:.0f} cm/s"
    print(f"amplitude {amp:>6}:  {verdict}")
"""),

(M, """At **-40** nothing happens — no wave forms, the push is too weak to reach threshold. At **-52** the
whole strip lights up and conducts at ~58 cm/s. Somewhere between the two is the **capture threshold**;
below it, no beat. (This is why a pacing electrode has a threshold, and why a failing one drops beats.)

**Size next.** Even a strong stimulus fails if the patch is too small — a tiny source can't charge
enough neighbours. Fire the *centre* of the sheet at a fixed, strong amplitude but vary the electrode
**radius**:
"""),

(C, """def center_launch_cv(radius):
    g    = cc.Grid(81, 81, 0.01)
    stim = cc.Stim.center(g, radius=radius, start_time=1.0, duration=2.0, amplitude=-52.0)
    return cc.monodomain(g, "ttp06", cc.ConductivityConfig.isotropic(1.36), stim, dt=0.1).run(20.0, 0.5).cv(50, 70, 40)

for rad in [0.02, 0.05]:
    cv = center_launch_cv(radius=rad)
    verdict = "nothing - no wave" if math.isnan(cv) else "WAVE"
    print(f"radius {rad} cm:  {verdict}")
"""),

(M, """A **0.02 cm** spot (the same strong -52 amplitude) launches **nothing** — the little patch dumps its
charge into the surrounding tissue faster than it can raise it to threshold. Widen it to **0.05 cm** and
the wave takes off. This is the same source–sink balance as the scar isthmus, seen from the other side:
a source that is too small for its sink simply cannot get a wave going. (It is also why section 1
insisted on `radius=0.05` for the centre stimulus.)
"""),

(M, """### Try it yourself

1. **Find the capture threshold.** Add more values to the amplitude list — try `[-40.0, -44.0, -48.0,
   -52.0]`. Where does the strip flip from "nothing" to "WAVE"? That crossover is the threshold for this
   tissue and electrode.
2. **Shrink the isthmus further.** In section 3, call `strip_with_isthmus(0.03)`. Does the wave still
   get through? Somewhere between 0.02 and 0.06 cm is the width where this scar flips from conducting to
   blocking — the same kind of threshold, in space.
"""),

(M, """---
## Recap

- **Isotropic tissue makes a circular wave; fibres make an elliptical one.** The axis ratio is the
  **square root** of the conductivity ratio (√σ again), and the fibre angle sets the ellipse's tilt.
- **A scar forces the wave to curve, and a narrow enough gap blocks it** — a source–sink mismatch, where
  a small strand of tissue can't charge the larger sink beyond it.
- **A wave must be launched**: too *weak* a stimulus (below the capture threshold) or too *small* a patch
  fails to start one at all — the same source–sink balance, at the point of ignition.

**Where next**: notebook 5.3 goes back to the ion channels of Chapter 2 — `GNa`, `GKr` — and shows what
turning *those* knobs does to the wave: slowing it, blocking it, and stretching its wavelength.
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
