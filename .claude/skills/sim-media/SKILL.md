---
name: sim-media
description: Produce standardized figures and videos from a simulation result — propagation video, map figures (snapshot/activation/APD/any field), series traces (action potential, restitution), multi-panel comparisons — saved to convention-compliant media/ paths. Use to visualize a Lab experiment's output for a lab member.
argument-hint: "[Lab/{date}_{slug} folder | a result] (default: most recent experiment)"
---

# Standardized simulation media

Turn a `SimulationResult` into the canonical visuals via the built-in media API (do NOT hand-roll matplotlib —
that's the whole point of standardizing). See `cardiac_core/API_CHEATSHEET.md` §10.

**The one-liner** — full-frame, unlabelled, 1080p, standard colours:

```python
result.video(slug, bulk=True)                     # -> media/lab/_sim_outputs/videos/{date}/{slug}_NN.mp4
result.image(slug, bulk=True)                     # -> .../images/{date}/{slug}_NN.png (a Vm snapshot)
result.trace(slug, bulk=True)                     # -> .../images/{date}/{slug}_NN.png (Vm at a node)
```

**Figures — `r.image()` for maps, `r.trace()` for series.** Both follow the same rule as video:
drawing displays, naming a destination saves. Annotated by default (axes, colorbar, units), because
a still carries its meaning in the labels.

```python
result.image(slug, what="activation", bulk=True)   # LAT map + isochrone contours
result.image(slug, what="apd", bulk=True)          # APD90 map (needs a run >= 1 action potential)
result.image(slug, what="source_sink", bulk=True)  # ANY named field from result.fields
result.image(slug, at=12.0, bulk=True)             # the voltage snapshot nearest 12 ms

result.trace(slug, at={"edge": (0, 4), "centre": (20, 4)}, bulk=True)   # labelled series + legend
result.trace(slug, hline=(-40.0, "threshold"), bulk=True)               # a reference line
result.trace(slug, what="restitution", at=(20, 4), bulk=True)           # APD vs DI (multi-beat)
cc.single_cell("ttp06", pre_pace=5).trace(slug, bulk=True)              # the 0-D action potential

# Comparison: map panels sharing a Gradient AND a quantity share ONE colorbar.
from cardiac_core import Image, draw
draw([Image(r_ctrl, label="control"), Image(r_drug, label="knockdown")], slug, bulk=True)
```
- `at=` means a **TIME in ms** on `image()` and a **NODE** on `trace()`.
- `pdf`/`webp` need an explicit `path=` (the `media/` convention accepts png/jpg/jpeg/svg/gif).
- An APD map on a run shorter than one action potential is all-NaN and warns — use a longer run.

**Richer video** (`Video` + `Gradient` + `render`) when the default is not enough — colour control,
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
  Like every renderer here it writes NOTHING unless a destination is named — add `bulk=True` if the
  frame is meant to go in the notebook record. Without it the return value is a `str` reading
  `<image — not saved …>`, which must never be reported as a path.
- **A file is written only when a destination is named.** `bulk=`/`question=`/`root=`/`date=`
  (the `media/` convention) and `path=` all count; with none of them the render is returned in
  memory and only displays inline. For a Lab experiment ALWAYS pass `bulk=True` — the notebook
  record needs the file on disk.
- `bulk=True` → gitignored `media/lab/_sim_outputs/...` (regenerable; the normal case).
- `bulk=False` → committed `media/lab/...` — use only for a curated figure worth keeping in git.
- `path="somewhere.mp4"` → written exactly there, no `media/` tree. Use for one-off exports only,
  never for a recorded experiment.

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
