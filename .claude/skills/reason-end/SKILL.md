---
name: reason-end
description: Tear down tmux research workspace. Kills viewer panes, cleans up WHITEBOARD.md. Counterpart to /research-resume workspace setup.
argument-hint: ""
---

# Reason End — Workspace Teardown

Tears down the tmux research workspace created by `/research-resume`. This is separate from `/save-session` — save is a checkpoint (keep working), reason-end is a teardown (done for now).

---

## Step 1: Offer Save

Ask the user:
> "Want to run `/save-session` first before tearing down?"

If yes, invoke `/save-session` and wait for it to complete before proceeding.
If no (or user says "just end"), proceed directly to teardown.

---

## Step 2: Kill Viewer Panes

```bash
# Check if in tmux
if [ -n "$TMUX" ]; then
  # Kill viewer panes (panes 1 and 2, if they exist)
  # Count panes first — only kill if more than 1
  PANE_COUNT=$(tmux list-panes | wc -l)
  if [ "$PANE_COUNT" -gt 1 ]; then
    # Kill from highest index down to avoid renumbering issues
    for i in $(seq $((PANE_COUNT - 1)) -1 1); do
      tmux kill-pane -t "$i"
    done
  fi
fi
```

---

## Step 3: Clean Up WHITEBOARD.md

Delete the ephemeral whiteboard file:

```bash
rm -f WHITEBOARD.md
```

---

## Step 4: Confirm

Report:
```
Session ended.
  Panes killed: {count}
  WHITEBOARD.md: removed
```

---

## Rules

- Always offer `/save-session` first — but don't force it.
- Only kill panes if in tmux and panes exist. If not in tmux, just clean up WHITEBOARD.md.
- Never kill pane 0 — that's Claude Code itself.
- This skill does NOT modify KNOWLEDGE.md, IDEALOG.md, or any research documents. That's `/save-session`'s job.
