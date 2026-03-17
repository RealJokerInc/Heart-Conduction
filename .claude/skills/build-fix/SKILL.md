---
name: build-fix
description: Fix test failures one at a time. Parse errors, prioritize by type (import->type->assertion->numerical), fix minimally, re-run, stop if fix introduces more errors.
argument-hint: "[optional: specific test file or engine]"
---

# /build-fix — Systematic Test Failure Repair

## Usage

`/build-fix [target]` — target is optional. Can be a specific test file path, an engine name (`bidomain`, `monodomain`, `lbm`), or omitted to auto-detect from recent changes.

## Step 1: Get Current Failures

If a target is provided, run tests for that target. Otherwise, run `/verify quick` logic to detect the engine and get the failure list.

Capture the full test output. Parse every failure into a structured list:

```
- Test name
- Error type (ImportError, TypeError, AssertionError, ValueError, numerical mismatch, etc.)
- Error message (one line)
- File and line number
```

## Step 2: Prioritize Errors

Sort failures by this priority (fix highest priority first — earlier fixes often resolve later ones):

| Priority | Error Type | Rationale |
|----------|-----------|-----------|
| 1 | `ImportError`, `ModuleNotFoundError` | Nothing runs until imports work |
| 2 | `SyntaxError` | Nothing runs until syntax is valid |
| 3 | `TypeError`, `AttributeError` | Wrong API calls block all logic |
| 4 | `ValueError`, `RuntimeError` | Logic errors in specific paths |
| 5 | `AssertionError` (non-numerical) | Test expectations may need updating |
| 6 | Numerical tolerance failures | Require careful analysis, fix last |

## Step 3: Fix Loop

For each failure, starting with highest priority:

### 3a. Diagnose

1. Read the failing test to understand what it expects.
2. Read the source code at the error location.
3. Identify the root cause. Common patterns:

| Symptom | Recovery Strategy |
|---------|-------------------|
| `ImportError: No module named X` | Check if module exists. If renamed, update import. If missing, check if it should be created or if the import path changed. |
| `ImportError: cannot import name X` | Read the source module — was the name changed? Check for typos. |
| `AttributeError: has no attribute X` | Read the class/module. Was the API renamed? Check the ABC in `improvement.md`. |
| `TypeError: unexpected keyword` | Read the function signature. Was the parameter renamed or removed? |
| `TypeError: missing required arg` | Read the function signature. Was a parameter added? |
| `AssertionError` (shape/type) | Compare expected vs actual. Check if dtype or tensor shape changed. |
| Numerical mismatch (atol/rtol) | Compare with V5.3 reference output. Check dtype (float32 vs float64). Check if algorithm changed. |
| `RuntimeError: CUDA` | Check tensor devices. Ensure all tensors on same device. |

### 3b. Fix Minimally

- Make the smallest change that fixes the error.
- Prefer fixing the source code over fixing the test, unless the test expectation is clearly wrong.
- Never change numerical tolerances without understanding why the values differ.
- Never refactor or clean up unrelated code during a fix.

### 3c. Re-run and Verify

1. Re-run the full test suite for the affected engine (not just the fixed test).
2. Compare failure count: before vs after.

### 3d. Guardrail Check — STOP if any of these are true:

- **More errors after fix**: The fix introduced new failures. Revert the change and report to the user.
- **Same error persists after 2 attempts**: The root cause is deeper than a simple fix. Stop and report what was tried.
- **Architectural change required**: The fix would require changing interfaces, class hierarchies, or module structure. Stop and report the needed changes.
- **V5.3 modification needed**: NEVER modify files under `Monodomain/Engine_V5.3/`. It is a validated baseline. If a fix seems to require V5.3 changes, stop and report.
- **More than 5 files need changes for one fix**: This suggests a design issue, not a bug. Stop and report.

### 3e. Next Error

If the fix succeeded and no guardrails triggered, move to the next failure in priority order. Repeat from Step 3a.

## Step 4: Final Report

After all fixes are applied (or a guardrail stops the loop), run the full test suite one final time and report:

```
=== BUILD-FIX REPORT ===
Engine:          {engine name}
Initial failures: {count}
Fixed:           {count}
Remaining:       {count}

FIXES APPLIED:
  1. {file}:{line} — {what was wrong} -> {what was changed}
  2. ...

REMAINING FAILURES (if any):
  1. {test name} — {error type} — {why it was not fixed}

GUARDRAILS TRIGGERED (if any):
  - {which guardrail} — {context}
```

## Rules

- One fix at a time. Never batch-fix multiple unrelated errors.
- Always re-run after each fix to catch regressions.
- Read before writing. Never guess at an API — read the source.
- Preserve backwards compatibility. If fixing an API, check all callers.
- For numerical issues, always check dtype first (this project uses float64).
- Git commit after each successful fix if the user asked for commits.
- If zero failures are found initially, report "All tests passing" and exit.
