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
ionic: ttp06|ord|mhas13
ionic_scaling:               # optional: tuner θ multipliers on published conductances
  g_Na: <float>              #   (e.g. from the Optimizer chip fit), applied to model params
  # ... any PHAS13/MHAS13 registry param -> multiplier
geometry:
  length_cm: <float>
  width_cm:  <float>
  dx:        <float>
conductivity:
  mode: bidomain|isotropic|anisotropic_monodomain|anisotropic_lbm
  sigma_i: <float>           # bidomain mode
  sigma_e: <float>           # bidomain mode
  sigma:   <float>           # isotropic mode (instead of sigma_i/e)
  D_long:  <float>           # anisotropic modes: per-axis EFFECTIVE D (cm²/ms), chi=1
  D_trans: <float>           #   (NOT a sigma — outside the sigma firewall)
  collision: mrt             # anisotropic_lbm: D2Q9-MRT
  s_jx: <float>              # anisotropic_lbm: MRT flux rate 1/tau_xx (carries D_long)
  s_jy: <float>              # anisotropic_lbm: MRT flux rate 1/tau_yy (carries D_trans)
  dt_lbm: <float>            # anisotropic_lbm: dt the rates are valid at
  chi: 1400.0                # anisotropic modes use chi=1.0 (D already effective)
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
For the `anisotropic_*` modes, `D_long`/`D_trans` are **effective diffusivities (cm²/ms)** carried
with `chi=1.0` — do NOT pass them where a `sigma` is expected. `anisotropic_lbm` additionally carries
the D2Q9-MRT relaxation rates `s_jx`/`s_jy` (= 1/τ, valid at `dt_lbm`). These extended fields are
written by the Optimizer chip-fit (`Optimizer/V1/tuner/presets.export_lab_preset`).
