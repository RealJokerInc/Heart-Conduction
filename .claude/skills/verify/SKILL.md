---
name: verify
description: Auto-detect engine from working directory or recent changes, run the right test suite, produce pass/fail report. Modes — quick (tests only), full (tests + artifacts + diff), pre-commit.
argument-hint: "[quick|full|pre-commit]"
---

# /verify — Engine Test Runner

## Usage

`/verify [quick|full|pre-commit]` — defaults to `quick` if no argument given.

## Step 1: Detect Affected Engine(s)

Run `git diff --name-only HEAD` and `git status --short` to find changed files. Match against the engine detection table:

| Engine | Path Indicator | Test Command |
|--------|---------------|--------------|
| Bidomain V1 | `Bidomain/Engine_V1/` | `cd Bidomain/Engine_V1 && conda run -n heart-conduction pytest tests/ -v` |
| Monodomain V5.4 | `Monodomain/Engine_V5.4/` | `cd Monodomain/Engine_V5.4 && conda run -n heart-conduction python test_phase7.py && python test_phase8.py` |
| LBM V1 | `Monodomain/LBM_V1/` | `cd Monodomain/LBM_V1 && conda run -n heart-conduction python -m pytest tests/ -v` |

If no changed files match any engine, ask the user which engine to verify. If multiple engines have changes, run all of them sequentially.

## Step 2: Run by Mode

### Mode: `quick` (default)

1. Run the test command(s) from the table above.
2. Capture stdout/stderr.
3. Parse results: count passed, failed, errored.
4. Skip to Step 3 (report).

### Mode: `full`

Everything in `quick`, plus:

1. **Artifact scan** — search changed files for debugging leftovers:
   - `breakpoint()`, `pdb.set_trace()`, `import pdb`
   - `print(` statements that look like debug output (ignore logging calls)
   - `TODO`, `FIXME`, `HACK` comments added in the diff
2. **Git diff summary** — run `git diff --stat` to show what changed and how much.
3. Include artifact findings in report.

### Mode: `pre-commit`

Everything in `full`, plus:

1. **dtype check** — grep changed `.py` files for `float32` or `torch.float32` usage. This project defaults to float64; any float32 must be intentional.
2. **V5.3 protection** — verify no files under `Monodomain/Engine_V5.3/` are modified. If any are, report FAIL immediately with a warning.
3. **Import check** — attempt to import the changed engine's main module to catch syntax errors before tests run.

## Step 3: Report

Print a structured report:

```
=== VERIFY REPORT ({mode}) ===
Engine:     {engine name(s)}
Mode:       {quick|full|pre-commit}

BUILD:      {PASS|FAIL} — {detail if fail}
TESTS:      {PASS|FAIL} — {passed}/{total} passed, {failed} failed, {errors} errors
ARTIFACTS:  {PASS|FAIL|SKIP} — {count of breakpoints/debug prints found, or SKIP if quick mode}
GIT STATUS: {CLEAN|DIRTY} — {number of uncommitted changes}

{If any FAIL, list the first 3 failing test names and their one-line error}
```

## Rules

- Always use `conda run -n heart-conduction` to ensure correct environment.
- Timeout: 5 minutes per engine. If tests hang, kill and report TIMEOUT.
- If a test produces GPU OOM, note it in the report but do not retry.
- Never modify test files or source code — this skill is read-only.
- For Monodomain V5.4, both test_phase7.py and test_phase8.py must pass for TESTS: PASS.
- Report exact counts. Do not summarize "some tests failed" — give numbers.
