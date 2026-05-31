# Mesh Builder — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding
Direction settled 2026-04-22: **Fiji (ImageJ2) is the drawing front-end** — user draws geometry in Fiji; a Python-side loader converts Fiji's output into the canonical mesh format for Bidomain V1 / Monodomain V5.4 / LBM V1. Prompted by Zimmerman's Fiji-based boundary-speedup demo.

### Existing `Builder/` package (baseline)
- ~1055 lines total across `common/`, `MeshBuilder/`, `StimBuilder/`.
- V5.4-only consumer. Workflow: load image → `threshold_transparency` (scrub alpha AA) → `filter_small_groups` (scrub RGB blend artifacts) → configure CellGroups → export.
- Core pain: the image-processing stack exists **to fix anti-aliasing artifacts** that came from whatever painting tool was used upstream. That scrubbing is why the pipeline is slow.

### Fiji (installed + validated 2026-04-22)
- Version: **2.16.0 / ImageJ 1.54p**. Location: `~/Applications/Fiji/` (launcher: `~/Applications/Fiji/fiji`). Size: 838 MB. Downloaded as `fiji-latest-linux64-jdk.zip` (includes bundled JDK).
- **Anti-aliasing for polygon fill is OFF by default.** Smoke test: `ByteProcessor(64,64)` + `PolygonRoi` + `ip.fill(roi)` → whole-image histogram has nonzero bins at exactly `{0, 255}`, zero intermediate values. This eliminates the Builder's AA-scrub step at the source.
- Line drawing has a separate `Roi.setAntiAlias(boolean)` macro function; text has `setFont(..., "non-antialiased")`. Fill commands are already non-AA.
- **Headless execution works**: `~/Applications/Fiji/fiji --headless --run script.py 'arg="value"'`. Verified returning correct results.
- **Scripting options**:
  - **Jython** (in-process): Python 2 stdlib only; no numpy/scipy; can import any Java class from Fiji jars (`from ij import IJ`, `from ij.gui import PolygonRoi`, etc.). Good for Fiji-internal logic.
  - **PyImageJ** (Python 3): Launches JVM as subprocess; full numpy interop; `imagej.init(mode='headless')`. Good for the Python loader side.
  - **Groovy / BeanShell / ImageJ macro (.ijm)**: all supported.
- Macro recorder (`Plugins > Macros > Record`) logs GUI actions to a script — useful for discovery while prototyping.

## Key Decisions
- **2026-04-22**: Adopt Fiji as the drawing tool instead of extending the existing Builder UI. Anti-aliasing elimination at source is cleaner than post-hoc scrubbing; Zimmerman already uses it for adjacent work; headless automation is straightforward.

## Open Questions
- ~~Drawing front-end~~ — **settled 2026-04-22: Fiji.** Open sub-decisions below.
- **Fiji authoring convention**: single multi-color image (region = color) vs. Fiji stack (each slice = one binary mask for a region) vs. ROI-manager export (polygon coordinates as .zip)? Each has trade-offs for boundary labeling.
- **Python↔Fiji interop**: Jython macros invoked via `fiji --headless --run` (simple, no extra Python deps, Python 2 only inside Fiji) vs. PyImageJ (numpy interop on Python 3 side, but JVM startup cost and extra dep). Which for which stage?
- **Canonical mesh format**: does it live in `Builder/` or in a shared `cardiac_core/` package (ties to `engine_consolidation`)?
- **Boundary labeling**: user leaning toward **stacking** — needs concrete spec (what is a layer? how is the boundary between layers tagged? how does the loader resolve overlaps?).
- **Per-region conductivity tensors**: painted directly (e.g., color→D_xx/D_yy/D_xy lookup) or specified via side-car metadata JSON alongside the Fiji file?
- **Stimulus regions**: unified into the same drawing file as mesh regions (different layer/color) or separate file?
- **LBM boundary compatibility**: can one label set map onto bounce-back / anti-bounce / absorbing (LBM), Neumann / Dirichlet / mixed (FDM/FEM/FVM), and bath-coupled (bidomain)? Or do we need per-engine adapters that interpret the same canonical labels differently?

## Connections
- **Engines**: Bidomain V1, Monodomain V5.4, LBM V1 (all active engines are consumers).
- **Related research**:
  - `engine_consolidation` — canonical mesh format is a shared-code concern.
  - `boundary_conduction_speedup` — Kleber-boundary geometries need clean boundary labels.
  - `geometry_induced_pacemaking` — sharp-tipped hiPSC layouts are a primary drawing target.
- **Pipelines**: Builder (current implementation), Surrogate (training-data generator needs programmatic geometry), Optimizer (parameter sweeps over geometry families).
