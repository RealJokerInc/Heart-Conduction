---
name: reason-end
description: Tear down reasoning workspace. Calls /save-session (graduates NOTEBOOK to KNOWLEDGE), then wipes NOTEBOOK.md + WHITEBOARD.md, kills tmux viewer panes.
argument-hint: ""
---

# Reason End — Workspace Teardown

Tears down the reasoning workspace created by `/research-resume`. Calls `/save-session` first to graduate NOTEBOOK.md findings to KNOWLEDGE.md, then wipes scratch files and kills viewer panes.

---

## Step 1: Run /save-session

Automatically invoke `/save-session` for the active research question. This ensures:
- NOTEBOOK.md findings are graduated to KNOWLEDGE.md (Job 6)
- Session snapshot is written to IDEALOG.md (Job 1)
- KNOWLEDGE.md is reorganized (Job 2)
- Cross-references are checked (Jobs 3, 4, 5)

Wait for `/save-session` to complete before proceeding.

---

## Step 2: Wipe NOTEBOOK.md

NOTEBOOK.md is owned by `/reason`. Now that `/save-session` has graduated findings to KNOWLEDGE.md, wipe it:

```bash
rm -f Research/Active/*/NOTEBOOK.md
```

---

## Step 3: Wipe WHITEBOARD.md

```bash
rm -f WHITEBOARD.md
```

---

## Step 4: Kill Viewer Panes

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

## Step 5: Confirm

```
/reason-end complete:
  /save-session: {summary from save-session}
  NOTEBOOK.md: wiped (findings graduated to KNOWLEDGE.md)
  WHITEBOARD.md: removed
  Panes killed: {count}
```

---

## Rules

- **Always run /save-session first.** This is not optional — NOTEBOOK.md findings must be graduated before wiping.
- Only kill panes if in tmux and panes exist. If not in tmux, just clean up files.
- Never kill pane 0 — that's Claude Code itself.
- This skill wipes NOTEBOOK.md and WHITEBOARD.md. All other document modifications are `/save-session`'s job.
