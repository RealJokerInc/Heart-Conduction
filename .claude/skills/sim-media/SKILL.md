---
name: sim-media
description: Produce standardized figures and videos from a simulation result — propagation video, APD map, activation isochrones — saved to convention-compliant media/ paths. Use to visualize a Lab experiment's output for a lab member.
argument-hint: "[Lab/{date}_{slug} folder | a result] (default: most recent experiment)"
---

# Standardized simulation media

Turn a `SimulationResult` into the canonical visuals via the built-in media API (do NOT hand-roll matplotlib —
that's the whole point of standardizing). See `cardiac_core/API_CHEATSHEET.md` §10.

**The one-liner** — full-frame, unlabelled, 1080p, standard colours:

```python
result.video(slug, bulk=True)                     # -> media/lab/_sim_outputs/videos/{date}/{slug}_NN.mp4
from cardiac_core import apd_map_figure, activation_isochrones
apd_map_figure(result, slug, bulk=True)           # APD90 heatmap PNG
activation_isochrones(result, slug, bulk=True)    # activation-time contours PNG
```

**Richer option** (`Video` + `Gradient` + `render`) when the default is not enough — colour control,
overlays, or a side-by-side comparison:

```python
from cardiac_core import Video, Gradient, render
result.video(slug, style="annotated", gradient=Gradient.rest_anchored(),
             isochrones=True, speed=20.0, bulk=True)     # 20 ms of sim per real second

# Comparison: a panel IS a Video; sharing ONE Gradient is what makes panels comparable.
render([Video.annotated(r_ctrl, gradient=g, label="control"),
        Video.annotated(r_drug, gradient=g, label="knockdown")],
       slug, bulk=True)                               # one shared colorbar + one time stamp
```
- `Gradient` presets: `physiological()` (default) · `rest_anchored()` · `zoom(span=8.0)` (make a
  few-mV artifact visible) · `diverging()` · `autoscale()`.
- Figure-only knobs raise on a bare clip instead of silently doing nothing — use `style="annotated"`.
- `Video(result).preview(t_ms=...)` renders ONE frame to PNG: check colours before a long encode.
- `bulk=True` → gitignored `media/lab/_sim_outputs/...` (regenerable; the normal case).
- `bulk=False` → committed `media/lab/...` — use only for a curated figure worth keeping in git.

## Steps

1. **Locate the experiment.** From `$ARGUMENTS` pick the `Lab/{date}_{slug}/` folder (default: the most
   recent in `Lab/NOTEBOOK.md`). Read its `run.py` for the `SLUG`.
2. **Get a result.** Run the experiment's `run.py` (it already calls the viz functions if `MAKE_MEDIA=True`
   — just run it and collect the printed media paths). If you need extra views not in the script (e.g.
   isochrones), generate a tiny script that rebuilds the result from the same PARAMETERS and calls the
   missing media call — do NOT duplicate the physics by hand; reuse the script's params.
3. **Report the paths.** Print the canonical `media/...` paths produced. Note they're on disk (viewable) and,
   if `bulk=True`, intentionally gitignored (regenerable by re-running).
4. **(optional) Curate.** If the scientist wants to KEEP a figure in git, re-render that one with `bulk=False`
   and tell them where it landed.

## Rules
- Only the built-in media API (`result.video` / `render` / `apd_map_figure` / `activation_isochrones`) —
  no bespoke plotting; only `API_CHEATSHEET.md` API to rebuild a result.
- Default to `bulk=True` (regenerable) — don't bloat git with videos. Promote to `bulk=False` only on request.
- Report the exact saved paths; never claim a figure you didn't write to disk.
