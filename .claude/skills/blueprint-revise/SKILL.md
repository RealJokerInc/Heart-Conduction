---
name: blueprint-revise
description: Update an existing PLAN.md — preserve completed steps, incorporate new IDEALOG decisions, log mutations. Use after /audit findings or mid-implementation course corrections.
argument-hint: "[what changed — reason for revision]"
---

# /blueprint-revise

Update an existing PLAN.md without losing completed work. Triggered after /audit
findings, new IDEALOG decisions, or mid-implementation course corrections.

## Procedure

### 1. Precondition Check

- Locate PLAN.md in the active research question directory (or current working context).
- If PLAN.md does not exist, STOP and tell the user:
  > "No PLAN.md found. Run `/blueprint` first to create one."
- Read the argument — note the reason for revision (audit findings, new decisions, etc.).

### 2. Load Context

- Read **PLAN.md** in full. Note:
  - The `Created` date in the header.
  - Which steps are marked `[x]` (completed) vs `[ ]` (open).
  - Any existing Mutation Log entries.
- Read **IDEALOG.md** in full. Identify decisions, findings, and insights logged
  AFTER the PLAN.md `Created` date (compare Thread entry timestamps).
- Read **IDEALOG.md Failed Approaches** section — these are known dead ends.
  Do NOT re-introduce any approach listed there.
- If the argument mentions `/audit` results, read the audit output or the relevant
  IDEALOG section where audit findings were recorded.

### 3. Classify Each Step

Walk through every step in the existing PLAN.md and classify it:

| Classification | Condition | Action |
|----------------|-----------|--------|
| **COMPLETED** | Marked `[x]` | PRESERVE unchanged. Never modify without explicit user request. |
| **STILL VALID** | Open `[ ]`, not contradicted by new decisions | KEEP as-is. |
| **CONTRADICTED** | Open `[ ]`, but a newer IDEALOG decision invalidates or changes it | MODIFY — rewrite to match the new decision. |
| **OBSOLETE** | Open `[ ]`, no longer needed given new direction | Mark as SKIPPED in the Mutation Log. |

### 4. Add New Steps

- If IDEALOG contains settled decisions that imply work not captured in the current
  PLAN.md, add new steps in the appropriate phase/section.
- New steps follow the same format as existing ones (numbered, with validation criteria).

### 5. Extensive Change Check

- If >50% of open steps are being MODIFIED or SKIPPED, STOP and suggest:
  > "This revision touches >50% of remaining steps. Consider running `/blueprint`
  > fresh instead of patching the existing plan."
- Wait for user decision before proceeding.

### 6. Write Mutation Log

Append to (or create) the `## Mutation Log` section at the bottom of PLAN.md.
Every change gets a dated entry:

```markdown
## Mutation Log

**MUTATED {YYYY-MM-DD}**: Step X.Y MODIFIED — {reason from IDEALOG or audit}
**MUTATED {YYYY-MM-DD}**: Step X.Y SKIPPED — {reason}
**MUTATED {YYYY-MM-DD}**: Step X.Y ADDED — {reason from IDEALOG decision}
```

- Use today's date for all mutations in this revision pass.
- Reference the specific IDEALOG thread entry or audit finding that drives each change.

### 6b. Switch Bottom Pane to PLAN.md

**IMMEDIATELY after writing the revised PLAN.md**, switch the bottom pane. Do BOTH:
1. Direct Bash call to switch pane 2 to show PLAN.md with glow shell loop
2. Spawn a background Agent as safety net (same switch command, `run_in_background: true`)

If not in tmux, skip silently.

**Also**: when revising, preserve the archive step in Final Cleanup. If the original PLAN.md had an archive instruction, ensure it survives the revision.

### 7. Present Summary

Output a clear summary to the user:

```
## Revision Summary

**Reason**: {argument / trigger for revision}
**Steps preserved (completed)**: N
**Steps kept (still valid)**: N
**Steps modified**: N — {brief list}
**Steps skipped**: N — {brief list}
**Steps added**: N — {brief list}

{If extensive: "Note: >50% of open steps changed. Consider /blueprint fresh."}
```

### 8. STOP — Approval Gate

**Do NOT begin any implementation.** Wait for explicit user approval, same as
`/blueprint`. The user may:
- Approve the revised plan as-is.
- Request further modifications.
- Decide to run `/blueprint` fresh instead.

## Rules

1. **STOP after revision** — same approval gate as `/blueprint`. Never auto-implement.
2. **Never modify completed steps** without explicit user request.
3. **Always log mutations** — every change has a dated reason traced to IDEALOG or audit.
4. **Read Failed Approaches** — never re-introduce known dead ends from IDEALOG.
5. **Suggest fresh /blueprint** if revision is extensive (>50% of open steps changed).
6. **Preserve formatting** — match the existing PLAN.md style (headers, numbering, checklist format).
7. **One revision pass** — apply all changes atomically, don't do multiple partial updates.
