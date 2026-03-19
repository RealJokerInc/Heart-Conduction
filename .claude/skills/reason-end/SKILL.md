---
name: reason-end
description: Tear down reasoning workspace. Calls /save-session, then wipes WHITEBOARD.md, kills tmux viewer panes.
argument-hint: ""
---

# Reason End — Workspace Teardown

Tears down the tmux workspace and scratch files used during `/reason` sessions. The workspace is set up by `/research-resume` (tmux panes) and populated by `/reason` (WHITEBOARD.md). This skill cleans up both.

---

## Step 1: Run /save-session

Automatically invoke `/save-session` for the active question. This ensures:
- Session snapshot is written to IDEALOG.md (Job 1)
- KNOWLEDGE.md is reorganized (Job 2)
- Cross-references are checked (Jobs 3, 4, 5)

Wait for `/save-session` to complete before proceeding.

---

## Step 2: Wipe WHITEBOARD.md

```bash
rm -f "Research/Active/{question}/WHITEBOARD.md"
```

Note: `{question}` is determined from the same question context used for `/save-session` in Step 1.

---

## Step 3: Kill Viewer Panes

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

## Step 4: Confirm

Report what actually happened (don't claim actions that didn't occur):

```
/reason-end complete:
  /save-session: {summary from save-session}
  WHITEBOARD.md: {removed | not present — skipped}
  Panes killed: {count | 0 — not in tmux}
```

---

## Rules

- **Always run /save-session first.** This is not optional — session state must be saved before teardown.
- Only kill panes if in tmux and panes exist. If not in tmux, just clean up files.
- Never kill pane 0 — that's Claude Code itself.
- Report accurately — if WHITEBOARD.md didn't exist, say "not present" not "wiped."
- This skill wipes WHITEBOARD.md. All other document modifications are `/save-session`'s job.
