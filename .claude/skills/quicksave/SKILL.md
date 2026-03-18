---
name: quicksave
description: Quick checkpoint — summarize recent chat into IDEALOG.md. No editorial pass, no KNOWLEDGE rewrite, no cross-referencing. Use mid-session when you want to capture state fast.
argument-hint: "[optional: research question name]"
---

# Quick Save

Fast checkpoint. Summarize recent conversation into IDEALOG.md. No heavy editorial work — that's `/save-session`'s job.

Input: $ARGUMENTS

---

## Step 0: Identify the Research Question

Same logic as `/save-session`: argument > conversation context > ask.

---

## Step 1: Summarize Chat → IDEALOG.md

Read `Research/Active/{question}/IDEALOG.md`.

Append to the **Thread** section with a dated entry summarizing what was discussed:

```markdown
### YYYY-MM-DD: {brief title of what was discussed}
{2-4 sentences capturing the key ideas, decisions, or direction changes from the recent conversation}
```

If any approaches were rejected during the conversation, add them to **Failed Approaches**:

```markdown
- **{approach}** (YYYY-MM-DD) — failed because: {reason}
```

Update **Next Step** if the conversation changed what should happen next.

---

## Step 2: Confirm

```
/quicksave complete:
  IDEALOG.md — {what was added: thread entry, failed approach, next step update}
```

---

## Rules

- **Fast, not thorough.** This is a 30-second save, not a 5-minute editorial pass.
- **No KNOWLEDGE.md edits.** That's `/save-session` Job 2.
- **No cross-referencing.** That's `/save-session` Job 3.
- **No MASTER_KNOWLEDGE_INDEX.md updates.** That's `/save-session` Job 5.
- **Derive content from conversation.** Don't ask the user to dictate.
