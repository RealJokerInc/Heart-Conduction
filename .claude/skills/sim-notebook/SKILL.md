---
name: sim-notebook
description: Organize the Lab simulation notebook — rebuild the master index from per-experiment manifests, summarize one experiment, or compare a series (e.g. a sigma sweep) side by side. Keeps Lab/NOTEBOOK.md in sync with the experiments.
argument-hint: "index | summary <slug> | compare <slug> <slug> ..."
---

# Lab notebook organization

The per-experiment `Lab/{date}_{slug}/MANIFEST.md` files are the **source of truth**; `Lab/NOTEBOOK.md` is
a generated index. This skill curates them. Parse `$ARGUMENTS` for the mode (default `index`).

## `index`
Rebuild `Lab/NOTEBOOK.md` from every `Lab/*/MANIFEST.md`:
1. Scan `Lab/*/` (skip `_validate`, `presets`, `_*`). For each, read `MANIFEST.md` → extract `Goal`,
   `Engine`, and the result/status line (the `— Confirmed … Result: …` footer, or status `built` if no result).
2. Rewrite the table in `Lab/NOTEBOOK.md` (keep its header) — one row per experiment, sorted by date desc:
   `| {date} | {slug} | {goal} | {engine} | {status} | {result} |`.
3. Flag stale/failed runs (status `built` with no result = never run; a NaN/0 result = failed — mark it).
Idempotent: re-running `index` reproduces the same table.

## `summary <slug>`
Print a plain-language summary of one experiment from its `MANIFEST.md` + result: what was asked, the
key parameters, and the outcome — readable by a lab member who didn't run it.

## `compare <slug> <slug> ...`
For a set of experiments (e.g. a σ-sweep / control-vs-knockdown series), emit a table whose columns are the
parameters that DIFFER across them plus the result — so the effect is visible at a glance:
```
| experiment | σ_i | … | result |
| control    | 1.74| … | 59.3 cm/s |
| knockdown  | 0.87| … | 41.x cm/s |
```
Read each `MANIFEST.md`; show only the fields that vary (+ always the result). If a listed experiment has no
result yet, say so (offer to run it via its `run.py`).

## Rules
- Manifests are the source of truth; `NOTEBOOK.md` is generated — never hand-edit notebook rows (the header
  warns of this). To change a row, edit the manifest and re-run `index`.
- Read-only over `cardiac_core` and the experiments; this skill only organizes `Lab/`.
- Don't invent results — only report what's in a `MANIFEST.md` (or offer to run the experiment to get one).
