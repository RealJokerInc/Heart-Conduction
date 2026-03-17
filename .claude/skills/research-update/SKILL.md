---
name: research-update
description: Record a finding, decision, or status change for a research question. Updates KNOWLEDGE.md, README.md, and MASTER.md as needed. Use when you've learned something and want to capture it.
argument-hint: "[question name]: [what changed]"
---

# Research Update

Record a finding, decision, or status change without starting a full research session.

Input: $ARGUMENTS

### Project Structure Reference
```
MASTER.md                              — project dashboard (update on status changes)
Research/Active/{question}/            — README.md, KNOWLEDGE.md, literature/, papers/, figures/
Research/Complete/{question}/           — answered (read-only KNOWLEDGE.md)
Research/Knowledge/                     — promoted knowledge files
{Engine}/experiments/{experiment}/      — EXPERIMENT.md, run.py, outputs/
```
This skill updates Active questions only. Complete questions have their knowledge promoted to Research/Knowledge/.

---

## Step 1: Determine the Research Question

**Priority order for identifying which question to update:**

1. **If the input explicitly names a question** (e.g., `boundary_conduction_speedup: CV ratio is 1.128`), use that.

2. **If a `/research-resume` session is active in this conversation**, default to whichever question was resumed. Check the conversation history for the most recent resume session brief (look for "=== RESUMING:" in prior messages). The user shouldn't have to re-specify the question they're already working on.

3. **If the update mentions engine files or concepts that clearly map to one question**, infer it. For example, mentioning "MHAS13" or "Optimizer" implies `ionic_model_optimization`; mentioning "Kleber" or "boundary CV" implies `boundary_conduction_speedup`.

4. **If still ambiguous**, ask the user.

**Examples with inference:**
- `dVdt target should be 100 V/s not 25` → during an ionic_model_optimization session, update that question
- `boundary_conduction_speedup: CV ratio converged to 1.128` → explicit naming
- `decided to keep Formulation A in V5.4` → clearly engine_consolidation
- `mark tissue validation sub-question as complete` → ambiguous (could be hipsc or ionic), ask

## Step 2: Read Current State

Read only what's needed (not the full context like `/research-resume` does):

- `Research/Active/{question}/KNOWLEDGE.md` — to find where the update goes
- `Research/Active/{question}/README.md` — if the update affects status, criteria, or experiment table
- `Research/Active/{question}/IDEALOG.md` — also read if the update type is idea, failure, issue, or next-step

## Step 3: Classify the Update

| Type | What changes | Files affected |
|------|-------------|----------------|
| **Finding** | New result, measurement, or insight | KNOWLEDGE.md (Current Understanding) |
| **Decision** | A choice was made with rationale | KNOWLEDGE.md (Key Decisions table) |
| **Question resolved** | An open question was answered | KNOWLEDGE.md (move from Open Questions to Current Understanding) |
| **New question** | A new unknown emerged | KNOWLEDGE.md (Open Questions) |
| **Criterion met** | A completion criterion was satisfied | README.md (check the box) |
| **Sub-question status** | A sub-question changed status | README.md (Sub-Questions table) |
| **New experiment** | An experiment was created or completed | README.md (Experiments table) |
| **New engine reference** | A new file was created that's relevant to this question | README.md (Engine References table) |
| **Correction** | A previous finding was wrong or imprecise | KNOWLEDGE.md (fix in place, add note) |
| **Status change** | The question itself changed status | README.md (Status line), MASTER.md |
| **Idea** | New idea or exploration to track | IDEALOG.md (Thread section) |
| **Failure** | An approach that was tried and failed | IDEALOG.md (Failed Approaches) |
| **Issue** | Bug or problem discovered during implementation | IDEALOG.md (Thread section) |
| **Next-step** | Update the immediate next action | IDEALOG.md (Next Step field) |

## Step 4: Apply the Update

Edit the relevant file(s). Use the Edit tool for targeted changes — don't rewrite entire files.

### For findings and corrections:
Update the relevant section of KNOWLEDGE.md. If correcting a previous value, note the correction:
```markdown
CV ratio = 1.128 at dx=0.025 (corrected from earlier estimate of 1.0714 —
recalculated with face-based stencil, updated 2026-03-17).
```

### For decisions:
Add a row to the Key Decisions table:
```markdown
| {decision} | {choice} | {rationale} |
```

### For criteria met:
Change `- [ ]` to `- [x]` in README.md.

### For sub-question status:
Update the Sub-Questions table status column.

### For new experiments:
Add a row to the Experiments table in README.md:
```markdown
| {name} | {engine} | {result or "—"} | `{path}` |
```

### For new engine references:
When a new file is created during research that future sessions should read (new engine source file, new test, new EXPERIMENT.md, new knowledge file from a connected question), add it to the Engine References table in README.md:
```markdown
| `{path}` | {what it tells you} |
```

### For ideas:
Add a dated entry to the Thread section of IDEALOG.md:
```markdown
### YYYY-MM-DD — {brief title}
{description of the idea or exploration}
```

### For failures:
Add a dated entry to the Failed Approaches section of IDEALOG.md:
```markdown
- **{approach name}** (YYYY-MM-DD) — {what was tried and why it failed}
```

### For issues:
Add a dated entry to the Thread section of IDEALOG.md (as a discovered issue):
```markdown
### YYYY-MM-DD — ISSUE: {brief title}
{description of the bug or problem discovered}
```

### For next-step:
Update the Next Step field in IDEALOG.md with the new immediate action. Replace the existing content, don't append.

## Step 5: Cascade Updates

If the update is significant enough to affect the project dashboard:
- Update the question's row in `MASTER.md` (status, next step)

If a sub-question was completed:
- Check if ALL sub-questions are now complete
- If so, ask the user if the parent question should be marked complete (→ `/research-complete`)

## Step 6: Confirm

Report what was changed:
```
Updated boundary_conduction_speedup:
  KNOWLEDGE.md — corrected CV ratio to 1.128 in Current Understanding
  README.md — no changes needed
  MASTER.md — no changes needed
```

---

## Rules

- **Minimal reads.** This skill is for quick updates — don't read the entire research context. Read only KNOWLEDGE.md and README.md for the target question.
- **Edit, don't rewrite.** Use targeted edits to modify specific sections. Don't rewrite entire files for a one-line update.
- **Date significant changes.** For corrections or major findings, note the date: `(updated YYYY-MM-DD)`.
- **Don't create files.** This skill updates existing files. To create new questions, sub-questions, or experiments, use `/research-new` or `/research-resume`.
- **Cascade to MASTER.md only when status changes.** Minor findings don't need a MASTER.md update. Completion criteria being met, status changes, and major results do.
