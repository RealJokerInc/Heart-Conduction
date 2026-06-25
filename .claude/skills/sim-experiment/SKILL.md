---
name: sim-experiment
description: Turn a plain-language cardiac-simulation experiment description into a runnable cardiac_core script, behind a parameter-manifest confirmation gate. For lab members (cell-culture / tissue-chip, no sim background) running simulation experiments. Generates code, logs it, never runs before you confirm.
argument-hint: "[free-form description of the experiment you want to run]"
---

# Build a simulation experiment

Turn `$ARGUMENTS` (a free-form experiment description) into a runnable, logged `cardiac_core` experiment —
**after the scientist confirms a parameter manifest**. You generate the script; the scientist owns the go.

## Step 0 — Load the bundle (every run)

Read, in this skill's directory:
- `reference/run-template.py` — the script you fill.
- `reference/recipes.md` — goal → parameter recipes (R1–R4).
- `reference/manifest-template.md` — the manifest you present at the gate.

And the **only** API source you may generate against:
- `cardiac_core/API_CHEATSHEET.md` (repo root) — exact signatures. **Never invent API.**

## Step 1 — RECEIVE

Take the description from `$ARGUMENTS`; if empty, ask: "What do you want to find out about the tissue?"
Accept any shape — a sentence, a paragraph, a wet-lab protocol. Don't make them speak in code.

## Step 2 — INTERPRET (build the parameters)

- Match the description to the closest recipe in `reference/recipes.md` (R1 CV · R2 reentry · R3 restitution
  · R4 edge/bath). If nothing fits, ASK 1–2 clarifying questions to place it; only if still unplaceable, build
  the closest recipe and FLAG the assumption in the manifest (the gate is the safety net). (Matches `recipes.md`.)
- **Infer the engine, don't ask:** `monodomain` by default; `bidomain` when the experiment is about the
  surrounding bath / tissue edge / boundary loading; `lbm` ONLY if the scientist explicitly asks for
  lattice-Boltzmann (never auto-inferred). Record the one-line reason.
- Compute concrete params from the recipe + `API_CHEATSHEET.md`: `Nx = round(Lx/dx) + 1`, `Ny = round(Ly/dy) + 1`
  (so the grid spans exactly Lx × Ly — `Grid.Lx = dx*(Nx-1)`), stimulus dict, CV indices, and `t_end` long
  enough for the front (≈ 50 cm/s = 0.05 cm/ms; 1 cm ≈ 20 ms).
- Ask the scientist ONLY for genuine gaps (e.g. "what tissue size?" if unstated and no sensible default).
  Default silently for `dt`, `Cm`, χ, splitting, solvers.

## Step 3 — MANIFEST

Render `reference/manifest-template.md` filled in, as PLAIN TEXT, and show it. Include the *optional* block
only if the scientist gave that info. State any assumption you made.

## Step 4 — ⛔ DOUBLE-CHECK GATE (hard stop)

Ask: **"Confirm, or tell me what to change."** Then STOP and wait.
- Do NOT generate files, do NOT run anything, until the scientist explicitly confirms.
- If they correct a value, update the manifest and re-present. Loop until confirmed.
- This gate is the whole point — accountability, no runaway runs. (See Rules.)

## Step 5 — GENERATE (only after confirmation)

- `slug` = kebab-case of the goal (≤ 40 chars). `today` = the current date (`YYYY-MM-DD`) from the harness
  `currentDate` context — do NOT guess it; if unsure, ask. (It can advance mid-session.)
- `dir = Lab/{today}_{slug}/`. **If `dir` already exists, do NOT overwrite it** (that would destroy a prior
  confirmed `MANIFEST.md` — the accountability record). Suffix the slug (`{slug}-02`, `-03`, …) until `dir`
  is new.
- Write `dir/MANIFEST.md` = the confirmed manifest text, VERBATIM (the record).
- Write `dir/run.py` = `reference/run-template.py` adapted to this recipe — fill the PARAMETERS block, set the
  engine/geometry/stimulus/measurement for the recipe, fill `{TITLE}/{DATE}/{GOAL}/{SLUG}`. Use ONLY
  `API_CHEATSHEET.md` calls.
- Append one row to `Lab/NOTEBOOK.md`: `| {date} | {slug} | {goal} | {engine} | built | — |`.

## Step 6 — RUN (offer)

Ask "Run it now?" If yes:
```bash
conda run -n heart-conduction python Lab/{today}_{slug}/run.py
```
- VERIFY the result is sane before reporting it: a CV should be physiological (~tens of cm/s, not NaN/0);
  voltage should have activated. If it's NaN / 0 / blew up, say so and propose a fix (finer dx, longer t_end)
  — do NOT present a broken number as a result.
- Record the outcome BOTH ways:
  - **sane** → update `MANIFEST.md` + the `NOTEBOOK.md` row with the result, status `done`.
  - **NaN / 0 / blew up** → update them with the bad value + status **`failed`** (so `/sim-notebook` flags it
    and there's a failure trail — never leave a stale `built` row pretending nothing ran).
- Offer `/sim-media` to render the propagation video + figures.

## Rules

- **Never run before the scientist confirms the manifest (Step 4).** No exceptions. This is accountability,
  not ceremony — it prevents vibe-coding runoff.
- **Only `cardiac_core/API_CHEATSHEET.md` API.** If you can't build a parameter from the cheatsheet, ask or
  drop it — never hallucinate a signature.
- **The manifest is the record.** It is saved verbatim as `MANIFEST.md`; it must reflect exactly what `run.py`
  does.
- **One experiment = one `Lab/{date}_{slug}/` folder + one `NOTEBOOK.md` row.** The `MANIFEST.md` is
  CANONICAL (source of truth); the `NOTEBOOK.md` row is a best-effort convenience — if it ever drifts,
  `/sim-notebook index` rebuilds it from the manifests. Never overwrite an existing experiment's `MANIFEST.md`.
- **Verify before presenting.** A result you didn't sanity-check is not a result.
- Generate `.py` scripts (not notebooks). Leave the `cardiac_core` calls intact; expose tunables in the
  PARAMETERS block so the scientist can iterate without touching the API.
