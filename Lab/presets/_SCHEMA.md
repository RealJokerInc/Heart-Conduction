# Preset schema

A preset is a YAML at `Lab/presets/{name}.yaml` holding the editable parameters of an experiment.
`/sim-preset` saves / lists / loads them; `/sim-experiment "using preset {name}"` fills its `run.py`
template from the preset (params are baked INLINE into the generated script — it stays self-contained,
no runtime YAML dependency). Keys mirror the manifest + the `run-template.py` PARAMETERS block.

```yaml
name: <slug>                 # required
description: <one line>      # required
recipe: R1|R2|R3|R4          # which recipe shape (see sim-experiment/reference/recipes.md)
engine: monodomain|bidomain|lbm
ionic: ttp06|ord
geometry:
  length_cm: <float>
  width_cm:  <float>
  dx:        <float>
conductivity:
  mode: bidomain|isotropic
  sigma_i: <float>           # bidomain mode
  sigma_e: <float>           # bidomain mode
  sigma:   <float>           # isotropic mode (instead of sigma_i/e)
  chi: 1400.0
  Cm:  1.0
stimulus:
  region: left_edge|<desc>
  width_cm: <float>
  start_ms: <float>
  duration_ms: <float>
  amplitude: <float>         # µA/µF, negative = depolarizing
run:
  t_end_ms: <float>
  save_every_ms: <float>
measure: cv|apd|restitution|reentry
```

Units follow `cardiac_core/API_CHEATSHEET.md`: `sigma*` are raw conductivities (mS/cm), `D_eff` is derived.
