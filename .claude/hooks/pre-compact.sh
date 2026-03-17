#!/bin/bash
# PreCompact Hook — Save state before context compaction
#
# Logs the compaction event and reminds Claude to re-read
# PROGRESS.md and IDEALOG.md after compaction.

SESSIONS_DIR="$HOME/.claude/sessions"
mkdir -p "$SESSIONS_DIR"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
COMPACTION_LOG="$SESSIONS_DIR/compaction-log.txt"

# Log the compaction event
echo "[$TIMESTAMP] Context compaction triggered in $(pwd)" >> "$COMPACTION_LOG"

# Output reminder injected into Claude's context
echo "[PreCompact] Context compaction occurring. After compaction, re-read PROGRESS.md for the active engine and IDEALOG.md for the active research question before continuing work."

exit 0
