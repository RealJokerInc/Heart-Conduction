---
name: sim-preset
description: Save, list, and load named simulation parameter sets (presets) for Lab experiments. Lets lab members reuse a tuned setup (control vs knockdown, a σ-sweep, a standard chip geometry) across experiments without re-specifying everything.
argument-hint: "save <name> | list | load <name> | show <name>"
---

# Simulation parameter presets

Presets live at `Lab/presets/{name}.yaml` (schema: `Lab/presets/_SCHEMA.md`). They hold the editable
parameters of an experiment so a tuned setup can be reused. Presets are applied at **generation time** —
`/sim-experiment` bakes them INLINE into the generated `run.py`, so scripts stay self-contained.

Parse `$ARGUMENTS` for the mode:

## `save <name>`
Capture the CURRENT experiment's parameters (from the just-confirmed manifest, or an existing
`Lab/{date}_{slug}/MANIFEST.md` / `run.py` the user points to) into `Lab/presets/{name}.yaml`,
following `_SCHEMA.md`. Don't invent values — read them from the manifest. Confirm the file written.

## `list`
List every `Lab/presets/*.yaml` with its `name` + `description` (one line each). Read each file's header.

## `show <name>`
Print `Lab/presets/{name}.yaml` and a plain-language summary of what it sets.

## `load <name>` (or invoked by `/sim-experiment "using preset <name>"`)
Read `Lab/presets/{name}.yaml`, validate against `_SCHEMA.md`, and hand its parameters to
`/sim-experiment` to fill the `run.py` template — **skipping gather**, but STILL presenting the manifest
and honoring the double-check gate (the scientist confirms the preset-derived parameters before any run).
The user may override individual fields ("…but make dx 0.005").

## Rules
- Presets carry parameters only — never API calls. The generated script always uses
  `cardiac_core/API_CHEATSHEET.md`.
- A loaded preset still goes through `/sim-experiment`'s manifest + double-check gate. No preset bypasses the gate.
- `name` + `description` are required; keep `description` honest (what it represents).
