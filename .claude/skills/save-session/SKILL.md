---
name: save-session
description: Session-end cleanup agent. 5 jobs — snapshot to IDEALOG, reorganize KNOWLEDGE, cross-reference IDEALOG↔KNOWLEDGE, condense IDEALOG, update MASTER_KNOWLEDGE_INDEX. Comprehensive rewrite allowed.
argument-hint: "[optional: research question name]"
---

# Save Session

Session-end cleanup agent. Runs five editorial jobs to keep research documents healthy.

Input: $ARGUMENTS

### Project Structure Reference
```
Research/Active/{question}/KNOWLEDGE.md   — polished reference (Job 2)
Research/Active/{question}/IDEALOG.md     — thinking trail (Jobs 1, 3, 4)
MASTER_KNOWLEDGE_INDEX.md                 — project-root index book (Job 5)
```

---

## Step 0: Identify the Research Question

**If argument provided**: Find the matching folder in `Research/Active/`. Go to Step 1.

**If inferable from conversation**: Check conversation history for an active `/research-resume` session or a clear single-question focus. Use that question.

**If ambiguous or no research question is active**: Skip Jobs 1-4 entirely. Inform the user:
```
No active research question detected. Jobs 1-4 require a question context.
Would you like to run Job 5 only (update MASTER_KNOWLEDGE_INDEX.md)?
```
If the user confirms, jump to Job 5. Otherwise, stop.

---

## Job 1: Write Session Snapshot to IDEALOG.md

Read `Research/Active/{question}/IDEALOG.md`.

Append to the **Session Log** section (create the section if missing):

```markdown
### YYYY-MM-DD Session
**Worked on**: {summarize from conversation — what was the focus}
**Accomplished**: {concrete outcomes with evidence — measurements, files created, decisions made}
**Next**: {exact next step — precise enough to resume cold}
```

Derive all content from the conversation history. Do not ask the user to dictate the snapshot.

---

## Job 2: Comprehensive KNOWLEDGE.md Reorganization

Read `Research/Active/{question}/KNOWLEDGE.md` in full.

Perform a **full editorial pass**. This can be a comprehensive rewrite if the document has drifted. Apply all of the following:

- **Restructure**: Group related findings logically, not chronologically. Create or merge sections to match the current state of understanding.
- **Rewrite for clarity**: Polish rough session notes into clean reference prose. Every entry should be something you can look up cold months later and immediately understand.
- **Merge and deduplicate**: Combine overlapping entries. If the same measurement or finding appeared in multiple places during incremental updates, consolidate into one authoritative entry.
- **Consistent depth**: Ensure all sections have similar resolution. A 200-line section next to a 2-line section means one needs expansion or the other needs tightening.
- **Preserve all information**: Reorganization and rewriting is encouraged. Deleting validated findings is not — every result stays, just in better form.

Write the updated KNOWLEDGE.md using the Edit tool (or Write if the rewrite is extensive enough to warrant it).

---

## Job 3: Cross-Reference IDEALOG <-> KNOWLEDGE

Read both documents in full (re-read if needed after Job 2 edits). Check every entry against the other document:

| Check | If found | Action |
|-------|----------|--------|
| IDEALOG idea was validated this session | Finding should be in KNOWLEDGE.md | Add to KNOWLEDGE if missing (polished form) |
| KNOWLEDGE finding spawned new ideas during session | IDEALOG should reference the finding | Add link in IDEALOG thread |
| IDEALOG failed approach contradicts a KNOWLEDGE entry | Inconsistency | Flag to user in the final report — may need to correct KNOWLEDGE |
| IDEALOG "Current Direction" no longer matches KNOWLEDGE state | Stale direction | Update IDEALOG direction to reflect current understanding |
| KNOWLEDGE has findings not yet reflected in IDEALOG narrative | Gap in thinking trail | Note in report (informational, not always actionable) |

Graduate validated ideas from IDEALOG into KNOWLEDGE. Do not remove graduated entries from IDEALOG — they stay as part of the narrative trail.

---

## Job 4: Condense Verbose IDEALOG Entries

Re-read IDEALOG.md after Job 3 edits. Look for verbose thread entries (common after interactive `/reason` sessions or long working sessions):

- **Collapse verbose prose** into concise summaries. A 40-line exploration that concluded "face-based averaging is correct" becomes 5-8 lines capturing the question, key reasoning, and conclusion.
- **Preserve the narrative arc**: What led to what. The reader should still understand the chain of reasoning.
- **Keep all decisions and failed approaches intact**: Never lose negative knowledge. Failed approaches stay with their exact error or reasoning.
- **Update "Current Direction"** and **"Next Step"** to reflect session-end state (may already be correct after Job 3).

---

## Job 5: Update MASTER_KNOWLEDGE_INDEX.md

Read `MASTER_KNOWLEDGE_INDEX.md` from the project root. If it does not exist, create it with this structure:

```markdown
# Master Knowledge Index

> Index book: where knowledge lives, how questions connect.
> NOT a copy of findings — follow the links for detail.
> Updated by /save-session after each research session.

## Research Statement
{To be written by researcher}

## Knowledge Index

| Question | Status | One-Liner | Knowledge |
|----------|--------|-----------|-----------|

## Cross-References

{How questions connect to each other}
```

Then apply two checks:

1. **One-liner accuracy**: Does this question's row still accurately describe its current state? If not, update the one-liner. If the question has no row, add one.
2. **New cross-references**: Did this session reveal connections to other questions that are not yet indexed? If so, add them to the Cross-References section with links to both KNOWLEDGE.md files.

Keep edits lightweight — update pointers and connections, do not duplicate findings.

---

## Step Final: Report

Output a summary of what changed across all five jobs:

```
/save-session complete:
  IDEALOG.md — {what changed: snapshot added, entries condensed, direction updated, etc.}
  KNOWLEDGE.md — {what changed: sections reorganized, entries merged, new findings added, etc.}
  Cross-reference — {what was graduated, flagged, or updated}
  MASTER_KNOWLEDGE_INDEX.md — {what changed: one-liner updated, cross-references added, etc.}
```

If any inconsistencies were flagged in Job 3, list them separately with recommended actions.

---

## Rules

- **Derive session snapshot from conversation.** Do not ask the user to dictate what happened — read the conversation history.
- **Comprehensive rewrite is allowed for KNOWLEDGE.md.** This is the one skill that can do a full rewrite. The goal is a polished reference, not incremental patches.
- **Never delete validated findings.** Restructure, rewrite, merge — but every confirmed result must survive.
- **IDEALOG narrative stays intact.** Condense prose, but preserve the chain of reasoning and all decisions.
- **MASTER_KNOWLEDGE_INDEX.md is an index, not a summary.** One-liners and cross-reference links only. Do not duplicate findings.
- **No fallback question guessing.** If the question cannot be identified, offer Job 5 only. Do not guess.
- **Date all session log entries.** Use YYYY-MM-DD format.
