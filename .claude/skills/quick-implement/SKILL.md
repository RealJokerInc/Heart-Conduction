---
name: quick-implement
description: Skip the full planning pipeline. Present a quick fix list, get approval, implement, verify, log. For small focused changes where /reason → /blueprint is overkill.
argument-hint: "[what to implement]"
---

# Quick Implement

Fast implementation path. No PLAN.md, no blueprint, no phases. Just do the thing.

Input: $ARGUMENTS

---

## Step 1: Analyze and Present Fix List

Read the relevant files to understand what needs to change. Present a concise fix list:

```
QUICK IMPLEMENT: {title}

Changes:
  1. {file} — {what changes}
  2. {file} — {what changes}
  3. ...

Tests: {which test command will verify this}

Ready? Say "begin" to proceed.
```

**WAIT for user approval.** Do not implement until the user says "begin", "go", "yes", or equivalent.

---

## Step 2: Implement

Make the changes. One file at a time, minimal diffs.

---

## Step 3: Verify

Run `/verify quick` for the affected engine. If no engine is affected (skill/doc changes), skip.

---

## Step 4: Log

Append a one-liner to the active research question's IDEALOG.md Thread section (if one is active):

```markdown
### YYYY-MM-DD: Quick implement — {title}
{One sentence: what was changed and why}
```

If no research question is active, skip logging.

---

## Rules

- **Always present the fix list first.** Never implement without showing what will change.
- **Wait for "begin".** The fix list is the approval gate — lighter than /blueprint but still requires user confirmation.
- **Minimal changes.** This is not a refactoring tool. Small, focused edits only.
- **Run /verify after.** Catch regressions immediately.
- **Log to IDEALOG.** Even quick changes should leave a trace.
- **Never use for multi-session work.** If it needs phases, use `/reason` → `/blueprint` instead.
