---
name: sim-media
description: Produce standardized figures and videos from a simulation result — propagation video, APD map, activation isochrones — saved to convention-compliant media/ paths. Use to visualize a Lab experiment's output for a lab member.
argument-hint: "[Lab/{date}_{slug} folder | a result] (default: most recent experiment)"
---

# Standardized simulation media

Turn a `SimulationResult` into the canonical visuals via `cardiac_core.viz` (do NOT hand-roll matplotlib —
that's the whole point of standardizing). Functions (see `cardiac_core/API_CHEATSHEET.md` §7 / `cardiac_core/viz.py`):

```python
from cardiac_core import propagation_video, apd_map_figure, activation_isochrones
propagation_video(result, slug, bulk=True)        # mp4 (gif fallback)
apd_map_figure(result, slug, bulk=True)           # APD90 heatmap PNG
activation_isochrones(result, slug, bulk=True)    # activation-time contours PNG
```
- `bulk=True` → gitignored `media/lab/_sim_outputs/...` (regenerable; the normal case).
- `bulk=False` → committed `media/lab/...` — use only for a curated figure worth keeping in git.

## Steps

1. **Locate the experiment.** From `$ARGUMENTS` pick the `Lab/{date}_{slug}/` folder (default: the most
   recent in `Lab/NOTEBOOK.md`). Read its `run.py` for the `SLUG`.
2. **Get a result.** Run the experiment's `run.py` (it already calls the viz functions if `MAKE_MEDIA=True`
   — just run it and collect the printed media paths). If you need extra views not in the script (e.g.
   isochrones), generate a tiny script that rebuilds the result from the same PARAMETERS and calls the
   missing `cardiac_core.viz` function — do NOT duplicate the physics by hand; reuse the script's params.
3. **Report the paths.** Print the canonical `media/...` paths produced. Note they're on disk (viewable) and,
   if `bulk=True`, intentionally gitignored (regenerable by re-running).
4. **(optional) Curate.** If the scientist wants to KEEP a figure in git, re-render that one with `bulk=False`
   and tell them where it landed.

## Rules
- Only `cardiac_core.viz` functions (no bespoke plotting); only `API_CHEATSHEET.md` API to rebuild a result.
- Default to `bulk=True` (regenerable) — don't bloat git with videos. Promote to `bulk=False` only on request.
- Report the exact saved paths; never claim a figure you didn't write to disk.
