---
name: research-complete
description: Complete a research question — finalize knowledge file, move to Complete/, promote to Knowledge/, update MASTER.md. Use when a research question has been fully answered.
argument-hint: "[question name or sub-question path]"
---

# Complete a Research Question

Move a research question from Active to Complete, finalize and promote its knowledge file.

Input: $ARGUMENTS

### Project Structure Reference
```
Research/Active/{question}/    → mv to → Research/Complete/{question}/
Research/Complete/{question}/KNOWLEDGE.md → cp to → Research/Knowledge/{question}.md
MASTER.md                     — move row from Active to Complete table
```
Lifecycle: Active → Complete (folder moves) + Knowledge promotion (copy KNOWLEDGE.md to Research/Knowledge/).

---

## Step 1: Identify the Question

Find the question in `Research/Active/`. If the input is a sub-question path (contains `/`), identify both parent and sub-question.

Read its:
- `README.md` — check completion criteria
- `KNOWLEDGE.md` — check current state
- `results/` — verify artifacts exist

## Step 2: Verify Completion Criteria

Read the completion criteria from `README.md`. For each criterion:

| # | Criterion | Met? | Evidence |
|---|-----------|------|----------|
| 1 | ... | Yes/No | ... |

If any criteria are NOT met, stop and report to the user:
- Which criteria are unmet
- What work remains
- Ask whether to proceed anyway (user may want to complete with a note about remaining items)

## Step 3: Handle Sub-Questions vs Top-Level

### Completing a sub-question:

1. Update the sub-question's README.md status to `Complete`
2. Update the parent's README.md sub-question table (status → Complete, add key finding)
3. Merge any sub-question findings into the parent's KNOWLEDGE.md
4. **Do NOT move the sub-question folder** — it stays inside the parent
5. Check: are ALL sub-questions now complete? If so, ask the user if the parent question is also complete.

### Completing a top-level question:

Continue to Step 4.

## Step 4: Finalize KNOWLEDGE.md

Read the current `KNOWLEDGE.md`. Ensure it has:

- [ ] **Current Understanding** section is complete (not tentative/partial language)
- [ ] **Key Decisions** section captures all decisions made during the research
- [ ] **Open Questions** section is either empty or explicitly notes things deferred to future work
- [ ] **Connections** section lists all engines, pipelines, and related questions that depend on this knowledge
- [ ] Sub-question findings are integrated (not just referenced)

If the KNOWLEDGE.md is sparse, offer to flesh it out by reading:
- All `literature/*.md` summaries
- All sub-question README.md findings
- The `results/` directory for key data points

The knowledge file should stand alone — someone reading it should understand the answer without needing to read any papers or results.

## Step 4b: Verify IDEALOG.md Exists

Check whether `Research/Active/{question_name}/IDEALOG.md` exists. If missing, warn the user ("IDEALOG.md not found — this question predates the three-document architecture") but proceed with completion.

## Step 5: Move to Complete

```bash
mv Research/Active/{question_name} Research/Complete/{question_name}
```

IDEALOG.md moves with the folder automatically as part of the directory move.

Update the README.md status line:
```
## Status: Complete (YYYY-MM-DD)
```

## Step 6: Promote Knowledge File

Copy the finalized KNOWLEDGE.md to the Knowledge directory:

```bash
cp Research/Complete/{question_name}/KNOWLEDGE.md Research/Knowledge/{question_name}.md
```

The Knowledge/ copy is the permanent reference. The Complete/ copy stays with the full research context (papers, results, proofs).

**IDEALOG.md is NOT promoted to Research/Knowledge/.** It stays in `Complete/{question_name}/` as a historical archive of the thinking trail.

## Step 7: Update MASTER.md

1. Remove the row from **Active Research** table
2. Add a row to **Complete Research** table:

```markdown
| [{question_name}](Research/Complete/{question_name}/) | {key answer summary} | [Knowledge file](Research/Knowledge/{question_name}.md) |
```

## Step 8: Update Dependent Research

Check if any Active questions reference this one in their KNOWLEDGE.md or README.md "Connections" section. If so, notify the user that those questions may benefit from the completed knowledge.

## Step 9: Update Skills (if needed)

If this completion affects other Active questions (e.g., removes a dependency or changes shared assumptions), note the impact for the user.

## Step 10: Report

Output:
- Confirmation of the move
- Final KNOWLEDGE.md summary (first 10 lines)
- Any Active questions that depend on this completed work
- Suggested next steps (often: a new question spawned by the findings)

---

## Rules

- **Never complete without checking criteria.** The user may be in a hurry, but completing prematurely leaves gaps in the knowledge base.
- **KNOWLEDGE.md must stand alone.** If reading it requires also reading papers, it's not done.
- **Sub-question completion ≠ parent completion.** Only complete the parent when all sub-questions are resolved.
- **Always promote to Knowledge/.** The whole point of completion is producing a permanent reference.
- **Date the completion.** Add `(YYYY-MM-DD)` to the status line — knowledge ages and the date helps judge relevance later.
