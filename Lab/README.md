# Lab — simulation experiments

This is where lab members' cardiac **simulation experiments** live. Each experiment is built by the
`/sim-experiment` skill, which turns a plain-language description into a runnable `cardiac_core` script —
*after* you confirm a parameter manifest (the accountability gate; no runs without sign-off).

## Structure

```
Lab/
  NOTEBOOK.md                  ← master log: one row per experiment (auto-appended)
  presets/                     ← saved parameter sets (/sim-preset)  {name}.yaml
  _validate/smoke.py           ← cheatsheet canary (don't delete)
  {YYYY-MM-DD}_{slug}/         ← one folder per experiment
    MANIFEST.md                ← the confirmed parameters — the accountability record
    run.py                     ← the generated, parameterized script (edit the PARAMETERS block)
    outputs/                   ← (optional) your own saved files; standardized media goes to
                                  media/lab/_sim_outputs/{videos,images}/ (gitignored, regenerable)
```

## The skills

| Skill | Does |
|---|---|
| `/sim-experiment` | describe an experiment → manifest → confirm → generate `run.py` + log it |
| `/sim-preset` | save / list / load named parameter sets |
| `/sim-media` | standardized propagation video + CV/APD figures from a result |
| `/sim-notebook` | organize this notebook (index, summarize, compare experiments) |

All generated code uses **only** `cardiac_core/API_CHEATSHEET.md` — the maintained API reference.

## To run an experiment

```bash
conda run -n heart-conduction python Lab/{YYYY-MM-DD}_{slug}/run.py
```
