# Zimmerman Storage-Tank Simulation

## Provenance
- Source: John Zimmerman (PI), email "Storage Tank Code", 2026-04-24.
- Origin: Colab notebook https://colab.research.google.com/drive/19X48Z8hPbodYucLOVkGqPLJ5Ewc1oue3
- Mirror: Google Drive file `storagetanks.py` (id `1am6_fxRLSz2WMFM_nHR0acAE2qNUGw5g`).

## Why it's filed here
John sent this as a minimal setup where he observed a boundary speedup effect. It's
a discrete-tank reduction of the electrotonic-loading argument. We use it to study
how three orthogonal axes (boundary operator, pipe directionality, pump rate law)
control the per-column wavefront-arrival shape — see KNOWLEDGE.md § "Discrete-lattice
boundary effects" in the parent research question.

## Layout

```
simulation/
├── tanks_vec.py            engine — pure simulation, accepts feature flags
├── tanks_channel_states.py reference OOP impl (parity-tested against tanks_vec)
├── test_vec_matches.py     parity test (max|ΔV| ~ 5e-14, max|Δiso| = 0)
├── configs.py              experiment configs (DEFAULT + named REGISTRY)
├── experiment.py           orchestrator: config dict → self-contained run dir
│
├── storagetanks_original.py  verbatim Colab export (do not edit)
├── storagetanks.py           John's organised
│
├── run_isochrones.py       legacy 2×2 sweep driver
├── per_column_camel_toe.py legacy per-column LAT diagnostic
│
└── outputs/
    ├── isochrones.png            canonical 4-panel iso plot
    ├── isochrones.npz / *.txt
    ├── per_column_camel_toe.png
    ├── tank_simulation.mp4       John's original 2000-step run
    └── experiments/              ← all new experiment runs go here
        ├── INDEX.md              auto-appended log of every run
        └── {date}_{name}/        one self-contained dir per experiment
            ├── config.json
            ├── iso.npz
            ├── isochrone.png
            ├── per_column_lat.png
            ├── summary.txt
            └── metadata.json
```

## Running an experiment

### 1. Predefined config
```bash
~/.conda/envs/heart-conduction/bin/python experiment.py baseline
~/.conda/envs/heart-conduction/bin/python experiment.py bidirectional
~/.conda/envs/heart-conduction/bin/python experiment.py gradient
~/.conda/envs/heart-conduction/bin/python experiment.py reflect_y
~/.conda/envs/heart-conduction/bin/python experiment.py reflect_all
~/.conda/envs/heart-conduction/bin/python experiment.py all   # everything in REGISTRY
```

Available named configs are defined in `configs.py`; current registry: `baseline`,
`gradient`, `bidirectional`, `reflect_y`, `reflect_all`, `long_run_constant`,
`long_run_gradient`, `john_radial`.

### 2. Ad-hoc override
```python
from configs import BASELINE, make
from experiment import run_experiment

cfg = make({**BASELINE,
            "name": "low_threshold_test",
            "description": "What if threshold = 10?",
            "rule": {"threshold": 10.0}})
run_experiment(cfg)
```

### 3. Sweep
```python
from configs import BASELINE, make
from experiment import run_experiment

for k in (0.04, 0.06, 0.08, 0.10, 0.12):
    cfg = make({**BASELINE, "name": f"k_sweep_{k:.2f}",
                "rule": {"type": "gradient", "gradient_k": k}})
    run_experiment(cfg)
```

Each run produces a `outputs/experiments/{date}_{name}/` directory with the verbatim
config, raw data, plots, and summary. `INDEX.md` at the experiments root gets one
new line per run (date, name, description, headline metric, dir).

## When to edit code vs. config

| Change | Config edit? | Code edit? |
|--------|:---:|:---:|
| Sweep `gradient_k`, `threshold`, `max_pump`, etc. | ✓ | — |
| Switch line vs point-cluster source | ✓ | — |
| Toggle reflection BC, bidirectional pipes, damping cap | ✓ | — |
| Lower steps, change grid size, change sample columns | ✓ | — |
| Add a new pump rule (e.g. piecewise-linear, Hill) | ✓ (new option) | ✓ (engine branch) |
| Switch Moore→von Neumann connectivity | ✓ (slot exists) | ✓ (engine branch) |
| Add 3D / new geometry / new BC type | ✓ (extend schema) | ✓ |

The pattern: **tunable values → config; new types/categories → both**. The engine's
job is to expose enough optionality that 90% of new experiments are config-only.

## Config schema

See `configs.DEFAULT` for the authoritative documentation. Sections:

- **`geometry`** — `type` ∈ {`line`, `point_cluster`, `custom`}, plus `Nx`, `Ny`,
  optional `custom_inlet_cells` / `custom_outlet_cells` lists of `(x, y)` tuples.
- **`rule`** — `type` ∈ {`constant`, `gradient`}, plus `threshold`, `max_volume`,
  `max_pump`, `gradient_k`, `damping_cap`.
- **`pipes`** — `directionality` ∈ {`one_way`, `bidirectional`}, `connectivity`
  (currently only `moore8`).
- **`boundary`** — `type` ∈ {`zero_pad`, `reflect_y`, `reflect_all`}.
- **`sim`** — `steps`, `record_history`, `snap_every`, `sample_cols`.

## Adding a new feature

1. Decide whether it's a new *value* of an existing axis (config-only) or a new
   *axis* (config + engine).
2. If new value: extend the relevant section in `configs.DEFAULT`, teach
   `tanks_vec.run` about the new option.
3. If new axis: add a new section to `DEFAULT`, plumb it through `experiment.py` →
   `tanks_vec.run`.
4. Add a named config in `configs.py` that demonstrates the feature.
5. Run `experiment.py {new_name}`. Results land in `outputs/experiments/`.
6. Confirm INDEX.md has the new row.

## Parity test

```bash
~/.conda/envs/heart-conduction/bin/python test_vec_matches.py
```
OOP vs vec engine, both rules, point + line geometries; max|ΔV| should be ~5e-14
and max|Δiso| should be 0. Run after any nontrivial engine edit.

## Related

- Research question: `../Research/Active/boundary_conduction_speedup/` —
  `KNOWLEDGE.md` has the PDE framing and the three-axis decomposition;
  `IDEALOG.md` has the session trail.
- Parent experiments testing the same effect in physical PDEs:
  `../Bidomain/Engine_V1/experiments/triangle_merger/`,
  `../Bidomain/Engine_V1/experiments/conductivity_sweep/`,
  `../Bidomain/Engine_V1/experiments/anisotropic_test/`.
