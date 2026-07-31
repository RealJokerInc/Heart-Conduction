"""Source of truth for `cardiac_core/tutorials/03_3_lbm.ipynb`
(Chapter 3.3 — "Tissue simulation, in depth: the LBM engine, a different numerical route").

Edit THIS file, not the notebook — a `.py` diff is reviewable where an `.ipynb` diff is not — then
re-run it to regenerate the notebook in place:

    python cardiac_core/tutorials/_build/build_03_3_lbm.py

Emits both the `.ipynb` (plain JSON, no `nbformat` dependency) and, with `--script PATH`, a flat `.py`
of every code cell for headless verification.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(os.path.dirname(_HERE), "03_3_lbm.ipynb")

M, C = "markdown", "code"

CELLS = [

(M, """# Tissue simulation, in depth — 3: the LBM engine, a different numerical route

The last two notebooks ran the same strip of tissue on **monodomain** (≈ 58 cm/s) and **bidomain**
(≈ 60 cm/s). This one runs it on the third engine, **LBM** — the lattice-Boltzmann method — which
arrives at the same wave by completely different machinery. It comes out a little *faster* on the
clock — and understanding why that is fine, not a bug, is the first lesson here (a like-to-like rule
you'll lean on all through Chapter 5). The second is bigger: LBM lets you model the **discrete** physics
of a tissue *edge* correctly — something the continuum engines, by construction, cannot — and that is
where this lab's boundary research lives.

You do not need to have run 3.1 or 3.2 first — this notebook rebuilds the tissue in one cell.

**What you'll learn**

1. **A different route to the same physics** — what LBM does instead of solving the equation directly
2. **Running it** — the same ingredients, a smaller time step
3. **The same wave** — the activation map looks just like monodomain's
4. **A different number** — LBM's CV, and the like-to-like rule that makes engine numbers usable
5. **Why LBM earns its place** — modelling *discrete* edge effects the continuum engines can't reach, through the wall rule: slowdown, flat, or a real speedup
6. **A tunable family of edge models** — blending the wall rules (the α dial) to span the discrete edge effects real tissue might show, and the number β behind the effect's size and sign

**Runtime**: about a minute or two of computing — a short strip run, plus several small square-sheet
runs for the wall-rule sections. On Colab, add about a minute the first time for the install.
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

The same 1.0 cm × 0.3 cm strip, conductivity, and left-edge electrode as the previous two notebooks:
"""),

(C, """g    = cc.Grid(101, 31, 0.01)                                          # 1.0 cm x 0.3 cm strip
cond = cc.ConductivityConfig.bidomain(1.74, 6.25)                     # healthy human ventricle
stim = cc.Stim.boundary(g, "left", amplitude=-52.0, start_time=1.0, duration=2.0)  # left-edge electrode

print("strip rebuilt")
"""),

(M, """---
## 2. A different route to the same physics

Monodomain and bidomain solve the propagation equation **directly**: at each step they compute how
voltage diffuses from every node to its neighbours by evaluating the diffusion term of the equation.

**LBM** — the lattice-Boltzmann method — never writes that equation down. Instead it imagines little
packets of "stuff" sitting at each node, and at every step they **stream** to neighbouring nodes and
**collide** (mix and relax) when they arrive. Do that with the right bookkeeping and diffusion
*emerges* from the streaming and collision automatically — the same physics, reached from underneath by
a different set of rules. LBM comes from computational fluid dynamics, where this trick is prized for
running fast on parallel hardware.

The one practical difference you will feel: LBM needs a **smaller time step** to stay accurate — here
`dt=0.01` ms, versus `0.05` for monodomain. So we keep the run short.
"""),

(M, """---
## 3. Run it

Same four ingredients, the `cc.lbm` factory, and the smaller `dt`:
"""),

(C, """sim = cc.lbm(g, "ttp06", cond, stim, dt=0.01)   # LBM engine; note the smaller time step
r = sim.run(t_end=20.0, save_every=0.5)          # shorter run — dt=0.01 means more steps per ms

print("done — LBM run complete")
"""),

(M, """---
## 4. The same wave

Watch it first — the wave sweeping across the strip, exactly like the monodomain movie in 3.1:
"""),

(C, """r.video()   # an inline movie of the LBM excitation wave sweeping left to right
"""),

(M, """A bright band enters from the left edge and marches to the right, each patch of tissue exciting
the next — by eye, indistinguishable from the monodomain wave. Now the activation map, each node coloured
by when the wave reached it:
"""),

(C, """r.image(what="activation")   # colour each node by the time the wave arrived
"""),

(M, """Straight, evenly-spaced isochrones — the same flat, steady wavefront monodomain produced. By eye,
the propagation is indistinguishable: a clean planar wave marching left to right. LBM and the other two
engines agree on the *shape* of the wave and on everything qualitative about it. Where they differ is a
single number.
"""),

(M, """---
## 5. A different number — and the like-to-like rule

Measure the conduction velocity with the identical call used in 3.1 and 3.2:
"""),

(C, """cv = r.cv(x1=20, x2=80, y=15)
print(f"LBM conduction velocity = {cv:.1f} cm/s")
"""),

(M, """About **64 cm/s** — noticeably higher than monodomain's 58 and bidomain's 60 on the *identical*
tissue with the *identical* conductivity. Here that is roughly a 10% gap.

**This is not an error, and neither engine is "wrong."** The three engines discretise space and time
differently, and each discretisation carries its own small, systematic offset in the effective
diffusivity — so each reports the wave's speed on its own slightly-shifted ruler. LBM's ruler runs a
little fast relative to the finite-difference engines. (For the same reason, the O'Hara–Rudy model
`"ord"` is available only through LBM in `cardiac_core` — the engines are genuinely different machines,
not skins over one solver.)

The consequence is a rule worth taping to your monitor:

> **Compare like-to-like.** When you change something — halve the conductivity, add a drug, carve a
> scar — and want to know how the wave responds, run the *same engine* before and after and compare
> those two numbers. Never compare one engine's absolute CV against another's; the ruler changes out
> from under you. An engine is perfectly trustworthy against **itself**.

Chapter 5 turns exactly these knobs and measures their effect, and it obeys this rule throughout: every
comparison is an engine against itself.

### Try it yourself

See the rule in action. Halve the conductivity — change `cc.ConductivityConfig.bidomain(1.74, 6.25)`
to `cc.ConductivityConfig.bidomain(0.87, 3.125)` in the rebuild cell — and re-run down to the CV. The
LBM speed drops (weaker coupling → slower wave). That *drop*, LBM-vs-LBM, is a real and trustworthy
result; the raw 64-vs-58 gap against monodomain is not something to read anything into.
"""),

(M, """---
## 6. Where LBM earns its keep — modelling discrete edge effects

*(This is the point of using LBM at all, not a footnote. If you only came to run the engine you can skip
to the recap — but this is where LBM does something the other two engines cannot.)*

Here is what the continuum engines can't do. Monodomain and bidomain model tissue as a **smooth medium**,
so at a wall they impose the continuum *no-flux* condition — and for a wave running along a straight wall
that has exactly one right answer: a **flat front**. In 3.1 you saw that when a finite-difference grid
bows the front at the wall, that curvature is a **numerical artifact** — a stencil bug the iso-mirror fix
removes to recover the flat answer. In the continuum world, edge curvature is a thing to *correct away*.

But real cardiac tissue is **not** a smooth continuum. It is discrete cells wired to their neighbours —
**diagonally** included — and near an edge that discrete structure does something physical the continuum
equation has no way to express. Rebuilding it inside a finite-difference solver is hard: you'd have to
re-inject the very discreteness a continuum model exists to smooth over.

LBM is the natural tool for it. Remember what it actually does: every step it **streams** little packets
of "stuff" from each node to its neighbours — including along the **diagonals** — then lets them
**collide**. At a wall the packets that would have streamed *out* of the tissue have nowhere to go, and
**how you reinject them — above all the diagonal ones, which carry the push *along* the wall — is the
wall rule**: not a nuisance boundary condition but a genuine modelling choice about how discrete tissue
behaves at its edge. `cardiac_core` gives you three on the 9-direction `d2q9` lattice, through
`boundary=`:

| `boundary=` | How it reinjects the packet at the wall | The front it makes |
|---|---|---|
| `"hbb"` | bounces it **straight back** the way it came | wall **lags** the interior — a **forward crescent** (slowdown) |
| `"ncs"` | reflects it to the **next cell** along the wall | **flat** — the continuum answer |
| `"scs"` | reflects it but keeps its **diagonal push along the wall**, in the same cell | wall **leads** the interior — an **inverse crescent** (a real speedup) |

The middle rule, `ncs`, simply reproduces the flat continuum front the finite-difference engines work to
get. The outer two are what a *discrete* edge can do that a continuum cannot — they differ only in **how
the diagonal currents are reinjected**, and `scs`, by keeping that diagonal push *along* the wall,
honours the diagonal inter-cellular coupling at the edge and yields a genuine **boundary speedup**. (3.1
showed an edge *slowdown* that was a stencil artifact; 3.2 a physical *speedup* from bath loading; here
the wall rule alone produces *either*.) That speedup is this lab's headline discrete-tissue effect — for
the background on why discrete boundary effects matter, and why continuum solvers miss them, see **Li
Chang's presentation _Correct handling of discrete boundary effects_ (lab SharePoint)**.

Run the identical wave on a 1 cm × 1 cm sheet three times — same tissue, same electrode, only the wall
rule changes. (These modes require `lattice="d2q9"`; the simpler `d2q5` lattice supports only plain
`"neumann"` walls.)
"""),

(C, """gb    = cc.Grid(41, 41, 0.025)                                 # a 1 x 1 cm sheet; top & bottom are the walls
iso   = cc.ConductivityConfig.isotropic(1.4)                   # LBM wall modes need isotropic tissue, dx = dy
stimb = cc.Stim.boundary(gb, "left", amplitude=-80.0, start_time=1.0, duration=2.0)

# same tissue, same wave — only the WALL RULE differs:
r_hbb = cc.lbm(gb, "ttp06", iso, stimb, lattice="d2q9", boundary="hbb", dt=0.005).run(t_end=45.0, save_every=0.5)  # bounce-back
r_ncs = cc.lbm(gb, "ttp06", iso, stimb, lattice="d2q9", boundary="ncs", dt=0.005).run(t_end=45.0, save_every=0.5)  # next-cell specular
r_scs = cc.lbm(gb, "ttp06", iso, stimb, lattice="d2q9", boundary="scs", dt=0.005).run(t_end=45.0, save_every=0.5)  # same-cell specular

r_hbb.image(what="activation")   # bounce-back wall
"""),

(M, """**Bounce-back (`"hbb"`)** — the baseline. The isochrones bow **backward** at the top and bottom
walls: the wave reaches the edges a touch *later* than the middle — a **forward crescent**, a small
**boundary slowdown** (about **+51 µs** of wall-vs-centre lag here). Bouncing the packet straight back is
the bluntest possible wall: it returns the signal but adds nothing along the edge, so the wall charges a
little slower than the interior. This is the same forward crescent you met in 3.1, and the yardstick the
other two rules are measured against.
"""),

(C, """r_ncs.image(what="activation")   # next-cell specular wall
"""),

(M, """**Next-cell specular (`"ncs"`)** — dead straight. Reflecting the packet to the *neighbour one cell
along the wall* is exactly the bookkeeping a flat wall needs: the front stays vertical, wall-vs-centre
≈ **0 µs** — and, tellingly, it is **exactly 0 at every resolution you try**. That makes `ncs` the clean
**correctness anchor**: whatever the other rules do, this one proves the engine itself carries no
built-in edge bias, so the crescents are a real property of the wall rule, not a bug.
"""),

(C, """r_scs.image(what="activation")   # same-cell specular wall
"""),

(M, """**Same-cell specular (`"scs"`)** — now the wall **leads**. The isochrones bow strongly *forward* at
the top and bottom edges: the wave reaches the walls **earlier** than the interior — an **inverse
crescent**, a genuine **boundary speedup** (about **−808 µs** of lead here, far larger than hbb's lag).
The rule keeps the packet's push *along* the wall in the same cell, so the edge gets an extra tangential
shove its interior neighbours never see — and it runs ahead.

Together these are the lab's "**Three Wall Rules, Three Crescents**" (Li Chang's June 2026 progress
report): a forward-crescent slowdown, a flat neutral, and an inverse-crescent speedup, with crescent
*rates* κ ≈ **+29 / 0 / −304 µs/cm** — the same three wall personalities first shown in the May 2026
report (slides 11–12). Watch the same-cell-specular front live — the highlight of this notebook:
"""),

(C, """r_scs.video()   # the leading front — the wall running AHEAD of the interior
"""),

(M, """The wave bulges *forward* at the top and bottom walls, the edges pulling ahead of the middle —
the boundary speedup, in motion.
"""),

(M, """---
## 7. The boundary speedup, deeper — the α dial and the number β

Two knobs turn that speedup from a yes/no switch into something you can dial and predict.

### The α-blend — a tunable family of edge models

The three wall rules differ only in **how the diagonal currents are reinjected** at the edge. By
*modulating* that reinjection — continuously blending bounce-back with same-cell specular —
`boundary="combined"` turns them into a single **family of edge models**: a tunable tool for the range of
discrete edge effects real tissue might actually show. The dial is `alpha=`:

- **α = 1** is pure **HBB** — the slowdown,
- **α = 0** is pure **same-cell specular** — the speedup,
- and in between, the wall's crescent slides smoothly from one to the other, passing through a **flat
  front near α ≈ 0.91**.

So α sets the wall's curvature — its **sign and its degree** — almost **linearly**: one knob spanning
slowdown, flat, and speedup, the whole tunable family of edge behaviours in a single parameter. Try the
midpoint:
"""),

(C, """blend = cc.lbm(gb, "ttp06", iso, stimb, lattice="d2q9",
               boundary="combined", alpha=0.5, dt=0.005).run(t_end=45.0, save_every=0.5)
blend.image(what="activation")   # half bounce-back, half same-cell specular
"""),

(M, """At α = 0.5 the front sits *between* the two extremes — a partial inverse crescent, the wall
leading, but by less than pure `scs`. Slide α up toward 1 and the crescent flattens and then reverses to
a slowdown; slide it down to 0 and you recover the full speedup. (`alpha=` only means anything for
`boundary="combined"` — on the other wall modes it is inert.)

### The number β — how the wave's width compares to the grid

There is a second, deeper control, and it is not a keyword at all. LBM has a natural dimensionless number,

    β  =  D · dt / dx²

— its diffusion/relaxation number, built from the diffusivity `D` (which you set through
`ConductivityConfig`), the time step `dt`, and the spacing `dx` (which you set through `cc.Grid`). You
never type `β`; you *set* it implicitly, every time you choose those three.

β controls the **magnitude** of the same-cell-specular crescent — and, decisively, its **sign**. On the
canonical `d2q9` lattice the crescent **flips sign at β\\* = 1/12**: land on one side of that value and
the wall speeds up, land on the other and it slows down. In plain terms, what the wall does depends on
**how the wavefront's own width compares to the grid spacing** — the ratio the June 2026 report calls the
control parameter **dx/r\\*** (with `r* = D/CV`, the wavefront's electrotonic length).

### Is the speedup "real"?

**Yes.** This is not a numerical glitch to explain away — it is what **discreteness** does at a wall, and
precisely why you need an engine that *represents* discreteness to see it at all. Real cardiac tissue is
not a smooth continuum: it is genuinely built from discrete, **diagonally-coupled** cells, and the
same-cell-specular rule is the lattice honouring exactly that diagonal coupling at an edge — reinjecting
the diagonal currents so the edge keeps its tangential push. The boundary speedup is what that discrete
structure *produces* at a wall — invisible, by construction, to a continuum solver.

The resolution-dependence — the sign flip at β\\* = 1/12, the front changing character as the grid
spacing crosses the wavefront's own width — is not a reason to doubt the effect; it is one of the most
**interesting** things about it. The wall's behaviour is *set by* the wavefront-width-to-grid-spacing
ratio, and that is precisely why it matters: real tissue lives near `r*/dx ~ O(1)` (the electrotonic
length `r*` is about the size of a real cardiac cell, ~100–150 µm), the very regime where this crescent
is switched on. For the full treatment — the control parameter **dx/r\\*** and how the boundary speedup
flips sign with resolution — see **Li Chang's May and June 2026 progress reports**.
"""),

(M, """---
## Recap

- **LBM** reaches the same propagation physics by a different numerical route — streaming and colliding
  packets on a lattice rather than solving the diffusion equation directly — and needs a **smaller time
  step** (`dt=0.01`).
- On the identical strip it produces the **same flat wave** but a **higher CV** (≈ 64 cm/s vs
  monodomain's ≈ 58) — a systematic numerical offset, **not** an error.
- The takeaway rule: **compare like-to-like** — an engine against itself across conditions, never one
  engine's absolute number against another's.
- **LBM's real payoff is discrete edge physics the continuum engines can't reach.** At a wall the rule
  decides the curvature by *how it reinjects the diagonal currents*: `"hbb"` lags (slowdown), `"ncs"`
  stays flat (the continuum answer / correctness anchor), `"scs"` leads — a genuine **boundary speedup**
  from honouring the tissue's diagonal coupling. `boundary="combined", alpha=` blends them into one
  tunable family of edge models, and the dimensionless β = D·dt/dx² sets the effect's size and flips its
  sign at β\\* = 1/12. Background in Li Chang's *Correct handling of discrete boundary effects* (lab
  SharePoint); the full quantitative story in the lab's May/June 2026 reports.

**Where next**: that completes the tour of building tissue and the three engines. Chapter 4 puts the
tissue to work with **pacing** — driving it beat after beat to measure how it recovers between beats and
how fast you can push it before it starts to fail.
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
