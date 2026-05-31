# Mesh Builder — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Use **Fiji (ImageJ2)** as the drawing front-end. User draws tissue geometry in Fiji; a Python-side loader consumes the exported output(s) and produces a canonical mesh for all three active engines. Fiji was prompted by Zimmerman's boundary-speedup demo code (he uses Fiji for the same kind of work). Building in isolation first; engine handoff once the tool emits a clean mesh.

## Next Step
Design the Fiji→canonical-mesh contract: (a) what the user draws in Fiji (single image with region colors, or stack of binary layers), (b) how boundaries get labeled, (c) what format the loader consumes.

## Thread
- **2026-04-22 (scaffold)**: Question scaffolded. Driver is pipeline speed — current draw→mesh→simulation loop is the slowest step across geometry-driven research (pacemaking, Kleber, scar, surrogate training inputs). User flagged boundary labeling (stacking approach) as a known concern.
- **2026-04-22 (Fiji pivot)**: User pointed at Zimmerman's 2026-04-16 "Diffusion Speed Up Simulation" email — he built a tool that ingests a B&W image and iterates diffusion on it. Mesh loader in that code isn't the interesting part; what matters is that **Fiji is the drawing tool**. User wants to adopt Fiji, automate it via Python scripting, and build the loader. Anti-aliasing was a major pain with the previous `Builder/` (it had to scrub alpha AA and color-blend artifacts); Fiji needed to be checked.
- **2026-04-22 (Fiji validated)**: Fiji 2.16.0/IJ 1.54p installed at `~/Applications/Fiji/`. Empirically confirmed via headless Jython smoke test that `ByteProcessor.fill(PolygonRoi)` produces a **pure binary mask** — only 0 and 255, zero intermediate values. So polygon-fill anti-aliasing is OFF by default. Headless execution works (`fiji --headless --run script.py`). Jython is Python 2, no numpy (use Java APIs instead), but `pyimagej` allows Python-3 control via JVM subprocess if we want numpy on the Fiji side.

## Failed Approaches
*None yet.*

## Session Log
- **2026-04-22** — Scaffolded question. Existing `Builder/` (~1055 lines) is V5.4-targeted; core pain is post-hoc AA cleanup (`threshold_transparency` + `filter_small_groups`). Fiji pivot: downloaded Fiji, confirmed headless + binary-mask fill, pointed at Zimmerman email as the source. Ready to design the Fiji→loader→engine contract.
