---
name: strategic-compact
description: Compaction decision guide. When to compact, what survives, pre-compact checklist. Suggest running /save-session first.
argument-hint: ""
---

# /strategic-compact — Compaction Decision Guide

When context is getting large, use this guide to decide whether to compact and how to prepare.

## Step 1: Should You Compact?

Consult this decision table based on the current transition:

| Transition | Compact? | Why |
|------------|----------|-----|
| Research --> Planning | **Yes** | Research context is bulky (paper text, search results); the plan distills it into a file |
| Planning --> Implementation | **Yes** | Plan is saved to file; free context for code reading and writing |
| Mid-implementation (same feature) | **No** | Losing variable names, file paths, partial state causes errors and rework |
| Between implementation sub-tasks | **Maybe** | Only if the sub-tasks are independent and state is saved to PROGRESS.md |
| Debugging --> Next feature | **Yes** | Debug traces, stack traces, and failed hypotheses pollute unrelated work |
| Engine A --> Engine B | **Yes** | Different codebases, different conventions, different context needed |
| After a failed approach | **Yes** | Clear dead-end reasoning so you don't re-try the same thing |
| Mid-debugging (same bug) | **No** | You need the full trace of what was tried and what was ruled out |
| Writing tests --> Writing code | **No** | Tests define the contract the code must satisfy |

**Rule of thumb**: Compact when crossing a phase boundary where the output is saved to a file. Do NOT compact when you are mid-task and state exists only in context.

## Step 2: Pre-Compact Checklist

Before compacting, complete every item:

1. **Save session state** — Run `/save-session` to persist the current conversation summary to IDEALOG.md. This is your recovery point.
2. **Commit completed work** — Any finished code changes should be committed to git. Uncommitted work risks being forgotten or half-applied after compaction.
3. **Update PROGRESS.md** — Mark completed tasks as DONE, note any in-progress tasks with enough detail to resume (file name, function name, what remains).
4. **Update MEMORY.md** — Add any discoveries, gotchas, or decisions that would be expensive to re-derive (e.g., "spectral solver needs odd-extension for DST-I", "chi*Cm=1.0 convention").
5. **Write the compact summary** — Use `/compact` with a clear summary that captures:
   - What was accomplished this session
   - What is the immediate next step
   - Any decisions made and why
   - Any gotchas or traps to avoid

## Step 3: What Survives vs What's Lost

| Survives Compaction | Lost After Compaction |
|--------------------|-----------------------|
| CLAUDE.md (always loaded) | File contents you read but didn't save anywhere |
| MEMORY.md (always loaded) | Variable names, tensor shapes, intermediate values |
| PROGRESS.md (if you re-read it) | Error messages and stack traces |
| IDEALOG.md (if you re-read it) | The "why" behind decisions (unless in MEMORY.md) |
| Git commits and their messages | Hypotheses you tried and rejected |
| The compact summary itself | Code structure you mapped out mentally |
| Files on disk | Which files you already read vs need to re-read |

## Step 4: Post-Compact Recovery

After compaction completes, follow the orientation protocol from CLAUDE.md:

1. Read `PROGRESS.md` for the relevant engine
2. Read the relevant section of IMPLEMENTATION.md (not the whole file)
3. Read MEMORY.md for gotchas
4. Read the compact summary that was preserved
5. Resume from the next incomplete task in PROGRESS.md

**Do NOT re-read entire documents speculatively.** Use line numbers from PROGRESS.md. Do NOT re-do work marked as done.

## When NOT to Compact

Even if context is large, do NOT compact if:

- You are mid-debugging and have built up a mental model of the bug
- You have uncommitted changes that depend on context to complete
- You are in the middle of a multi-file refactor where files reference each other
- The next task directly depends on understanding gained in this session that is NOT saved anywhere
