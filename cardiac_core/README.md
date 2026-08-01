# cardiac_core

A unified Python API for cardiac electrophysiology simulation. One way to declare a simulation —
geometry, conductivity, stimulus, ionic model — and three interchangeable engines to run it on:
**monodomain** (finite-difference), **bidomain** (with extracellular potential), and **lattice Boltzmann**.

Built on PyTorch, float64 throughout, CPU or CUDA.

> **Status: private.** Shared with Cornell BME for internal research use — not a public release.

## Quick start

```python
import cardiac_core as cc

g    = cc.Grid(201, 51, 0.01)                          # a 2.0 x 0.5 cm strip, dx = 0.01 cm
cond = cc.ConductivityConfig.bidomain(1.74, 6.25)      # healthy human ventricle (mS/cm)
stim = cc.Stim.boundary(g, "left", start_time=1.0, duration=2.0, amplitude=-52.0)

sim = cc.monodomain(g, "ttp06", cond, stim, dt=0.05)
r   = sim.run(t_end=40.0, save_every=0.5)

print(f"conduction velocity = {r.cv(x1=20, x2=100, y=25):.1f} cm/s")   # ~59 cm/s
```

## What's inside

- **One construction API, three engines.** `cc.monodomain(...)`, `cc.bidomain(...)`, and `cc.lbm(...)`
  take the same `Grid` + `ConductivityConfig` + `Stim` + ionic-model name and return a result you analyze
  the same way. Length is **cm**, time **ms**, voltage **mV**.
- **Ionic models** — ten Tusscher–Panfilov 2006 (`ttp06`), O'Hara–Rudy (`ord`), and hiPSC-CM models
  (`paci` / `phas13` / `mhas13`).
- **Single-cell (0-D).** `cc.single_cell("ttp06", pre_pace=5)` for an action potential; apply a drug with
  `conductances={"GKr": 0.5}` (50% IKr block) — the same conductance vocabulary as the tissue-level
  `sim.scale_conductance(...)`.
- **Stimulus as an object** — `cc.Stim.boundary / point / center / from_region`, current injection or
  voltage clamp.
- **Analysis** — `r.cv()`, `r.lat()`, `r.apd()`, activation maps, restitution; plus the `r.fields.*` layer
  (source–sink, velocity, curvature, vorticity, …) and scalar EP metrics (wavelength, safety factor).
- **Media** — `r.video()`, `r.image()`, `r.trace()` render to a file or display inline (`.show()`), with a
  reusable `Gradient` for colour control.

## Documentation

- **`API_CHEATSHEET.md`** — the recipes (verbs): construct → run → record → analyze → visualize.
- **`API_OBJECTS.md`** — the object atlas (nouns): every public object's attributes + methods.

Tutorial notebooks (single cell → tissue → pacing/reentry → tuning) are maintained separately from this
package.

## Requirements / use

Pure-Python package. Requires `torch`, `numpy`, `scipy`, `scikit-image`, and `torch-dct` (the spectral
solver used by the default rectangular meshes). Then:

```python
import cardiac_core as cc
```

All tensors are float64; pass `device="cuda"` to run on GPU.
