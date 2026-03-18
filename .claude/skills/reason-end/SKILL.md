---
name: reason-end
description: Tear down reasoning workspace. Calls /save-session (graduates NOTEBOOK to KNOWLEDGE), then wipes NOTEBOOK.md + WHITEBOARD.md, kills tmux viewer panes.
argument-hint: ""
---

# Reason End — Workspace Teardown

Tears down the tmux workspace and scratch files used during `/reason` sessions. The workspace is set up by `/research-resume` (tmux panes) and populated by `/reason` (NOTEBOOK.md, WHITEBOARD.md). This skill cleans up both.

---

## Step 1: Identify Active Question

Determine which research question was active (from conversation context or ask). This is needed to find the correct NOTEBOOK.md path:
- If research question active: `Research/Active/{question}/NOTEBOOK.md`
- If no research question (engine-only): `NOTEBOOK.md` in project root

---

## Step 2: Run /save-session

Automatically invoke `/save-session` for the active question. This ensures:
- NOTEBOOK.md findings are graduated to KNOWLEDGE.md (Job 6)
- Session snapshot is written to IDEALOG.md (Job 1)
- KNOWLEDGE.md is reorganized (Job 2)
- Cross-references are checked (Jobs 3, 4, 5)

Wait for `/save-session` to complete before proceeding.

---

## Step 3: Wipe NOTEBOOK.md

Only wipe the active question's NOTEBOOK.md, not all questions:

```bash
# If research question active:
rm -f "Research/Active/{question}/NOTEBOOK.md"

# If engine-only (no question):
rm -f NOTEBOOK.md
```

---

## Step 4: Wipe WHITEBOARD.md

```bash
rm -f WHITEBOARD.md
```

---

## Step 5: Kill Viewer Panes

```bash
if [ -n "$TMUX" ]; then
  PANE_COUNT=$(tmux list-panes | wc -l)
  if [ "$PANE_COUNT" -gt 1 ]; then
    for i in $(seq $((PANE_COUNT - 1)) -1 1); do
      tmux kill-pane -t "$i"
    done
  fi
fi
```

---

## Step 6: Confirm

Report what actually happened (don't claim actions that didn't occur):

```
/reason-end complete:
  /save-session: {summary from save-session}
  NOTEBOOK.md: {wiped (findings graduated) | not present — skipped}
  WHITEBOARD.md: {removed | not present — skipped}
  Panes killed: {count | 0 — not in tmux}
```

---

## Rules

- **Always run /save-session first.** This is not optional — NOTEBOOK.md findings must be graduated before wiping.
- **Only wipe the active question's NOTEBOOK.md.** Do not glob across all questions.
- Only kill panes if in tmux and panes exist. If not in tmux, just clean up files.
- Never kill pane 0 — that's Claude Code itself.
- Report accurately — if NOTEBOOK.md didn't exist, say "not present" not "wiped."
- This skill wipes NOTEBOOK.md and WHITEBOARD.md. All other document modifications are `/save-session`'s job.
