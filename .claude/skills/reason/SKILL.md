---
name: reason
description: Interactive reasoning buddy for planning and exploring ideas. Presents big-picture map, drills into details on demand, follows organic thinking flow. Writes settled decisions and failed approaches to IDEALOG.md on natural transitions. Can invoke /blueprint when ready to implement.
argument-hint: "[objective or topic to reason about]"
---

# Interactive Reasoning

Think through a problem interactively with the user. Present big-picture maps, drill into details on demand, follow organic flow.

Input: $ARGUMENTS

---

## Step 1: Load Context

Read these files in parallel (skip any that don't exist):

| File | Purpose |
|------|---------|
| `Research/Active/{question}/IDEALOG.md` | Current direction, failed approaches, session log |
| `Research/Active/{question}/KNOWLEDGE.md` | Settled knowledge, open questions |
| `Research/Active/{question}/README.md` | Status, completion criteria, engine references |

If the topic involves engine work, also read in parallel:

| File | Purpose |
|------|---------|
| `{Engine}/PROGRESS.md` | Current phase, what's done/in-progress |
| `{Engine}/IMPLEMENTATION.md` (relevant section only) | Phase specs, validation criteria |

If no research question maps to the topic, skip research files and work from the user's input + codebase.

**NOTEBOOK.md location:**
- If a research question is active: `Research/Active/{question}/NOTEBOOK.md`
- If no research question (engine-only work): `NOTEBOOK.md` in project root (same as WHITEBOARD.md)

---

## Step 2: Present Big Picture

**Always open with a visual ASCII map.** Scannable in 5 seconds. See KNOWLEDGE.md `/reason` Detailed Design for full template. Key elements:

- Title bar with objective
- WHY (1-2 sentences), SCOPE (engines, phases, effort)
- ASCII box diagram showing phases with arrows, step counts per phase
- HIGH risks only (not medium/low)
- BUILDS ON / FEEDS INTO (connections to existing work)
- AVOID section if IDEALOG.md has failed approaches
- Closing prompt: "Drill into a phase? Or discuss the big picture?"

Keep to 15 lines max. Include test counts per phase.

---

## Step 3: Follow the User's Lead (Zoom Model)

Three zoom levels. User navigates by drilling down or jumping organically.

| Level | Shows | Risk filter |
|-------|-------|-------------|
| **Big** (phases) | ASCII map, scope, HIGH risks, test counts, dependencies | HIGH only |
| **Middle** (steps within phase) | Goal, current state, steps, architecture changes, key questions | HIGH + MEDIUM |
| **Small** (file-level detail) | File paths, what changes, why, pseudocode, test commands, all risks | ALL |

**Middle-level template** (see KNOWLEDGE.md `/reason` Detailed Design for full examples):
- Goal, current state, numbered steps with size tags
- DEPENDS ON / UNLOCKS
- Architecture changes (NEW/MOD files)
- Key questions with references to KNOWLEDGE.md

**Small-level template**:
- FILE path with line range
- WHAT CHANGES (current vs new)
- WHY (reasoning for this step)
- REFERENCE (IMPLEMENTATION.md section, Research/ knowledge)
- VERIFY (exact pytest/python commands)
- RISK with mitigation

**Organic flow is equally valid.** If the user jumps topics ("what about the solver? actually wait, does the ionic model handle this?"), follow the jump. Do NOT say "you skipped Phase 2, let's go back." Discuss at whatever resolution they're at.

---

## Step 4: Track State Internally

Regardless of exploration order, track:

| State | Examples |
|-------|---------|
| **Settled** | Decisions made, approaches chosen |
| **Open** | Unresolved questions, unexplored phases |
| **Rejected** | Approaches discussed and ruled out |

This tracking drives write timing (Step 5) and `/blueprint` handoff (Step 8).

---

## Step 5: Write to IDEALOG.md on Natural Transitions

Do NOT write after every exchange. Accumulate in context, batch write on transitions.

| Trigger | What to write |
|---------|---------------|
| User settles on a decision | Decision + rationale |
| User rejects an approach | Failed approach entry |
| Topic shift in conversation | Conclusions from previous topic |
| User says "write that down" | Whatever was just discussed |
| PreCompact hook fires | Emergency dump of unsaved insights |
| User says "let's build this" | Final summary before `/blueprint` handoff |

**Target: ~3-4 writes per 30-minute session**, not 15-20.

Route writes to the appropriate file and section:

| Content | File | Section |
|---------|------|---------|
| Decisions, direction changes | IDEALOG.md | Thread (current entry) |
| Failed approaches | IDEALOG.md | Failed Approaches |
| High-res technical findings (exact commands, configs, test results, file paths, error messages) | NOTEBOOK.md | Append freely — no formatting pressure |

NOTEBOOK.md is scratch paper. Dump raw findings without worrying about polish. `/blueprint` reads it for detail. `/save-session` graduates worthy findings to KNOWLEDGE.md (but does NOT delete NOTEBOOK.md). Only `/reason-end` wipes NOTEBOOK.md (after calling `/save-session` first).

---

## Step 6: Trade-Off Analysis

When encountering design forks, present a visual comparison table:

```
  TRADE-OFF: {decision}
  +------------------+------------------+
  |  Option A        |  Option B        |
  +------------------+------------------+
  |  + advantage     |  + advantage     |
  |  + advantage     |  + advantage     |
  |  - disadvantage  |  - disadvantage  |
  |  - disadvantage  |  - disadvantage  |
  +------------------+------------------+
  RECOMMENDATION: {option} ({why})
```

User decides. Decision gets written to IDEALOG.md per Step 5.

---

## Step 7: Red Flags

Proactively check for domain-specific anti-patterns during reasoning:

| Anti-pattern | Why it matters |
|--------------|----------------|
| Steps without verification commands | Unverified code is untrusted code |
| Phases that can't be tested independently | Defeats phased delivery |
| Missing `float64` dtype handling | #1 numerical bug source in this project |
| Modifying V5.3 | Validated baseline -- never allowed |
| Code duplicated across engines | Should go in `cardiac_core/` |
| Missing EXPERIMENT.md backlinks | Breaks research traceability |
| No connection to KNOWLEDGE.md or completion criteria | Work without purpose |
| Skipping isotropic/scalar-first phasing | Violates simplest-case-first principle |

Flag these when spotted. Don't wait for the user to ask.

---

## Step 8: Mid-Session Checkpoint

If the user says "save" or "checkpoint" during a long `/reason` session, invoke `/save-session`. This graduates NOTEBOOK.md findings to KNOWLEDGE.md and snapshots the session to IDEALOG.md — without ending the reasoning session. NOTEBOOK.md stays intact for continued use.

This is especially important before compaction risk — if context is getting large, suggest a checkpoint.

---

## Step 9a: Handoff to /blueprint

When the user says "let's build this" (or equivalent):

1. Write any unsaved decisions/rejections to IDEALOG.md
2. Summarize what's settled vs still open
3. Invoke the `/blueprint` skill to generate PLAN.md

---

## Domain-Specific Phasing Philosophy

When proposing phases, follow this progression:

| Phase | Purpose | Domain example |
|-------|---------|----------------|
| 1 | Simplest case first | Isotropic / scalar / 2D / single cell |
| 2 | Full implementation | Anisotropic / tensor / 3D / tissue |
| 3 | Validation | Against V5.3, literature values, or analytical solutions |
| 4 | Optimization | GPU kernels, LUT acceleration, memory reduction |

Phases should be independently deliverable where possible. Each phase has its own tests.

---

## Step 10: Visualize on WHITEBOARD.md

When presenting diagrams, architecture maps, trade-off tables, or any visual that benefits from a persistent view, write them to `WHITEBOARD.md` in the project root. If a tmux workspace is active (set up by `/research-resume`), the bottom-right pane auto-refreshes with glow rendering.

Write to WHITEBOARD.md when:
- Presenting the big-picture ASCII map (Step 2)
- Showing trade-off comparison tables (Step 6)
- Drawing architecture diagrams during middle/small-level discussion
- Any visual the user might want to reference while talking

Overwrite the file each time (it's ephemeral, not accumulated). The current visual replaces the previous one.

---

## Rules

- **NEVER implement during /reason.** No creating files (except IDEALOG.md, NOTEBOOK.md, and WHITEBOARD.md writes). No editing source code or skill files. No `pip install`. No `mkdir` for new components. No writing PLAN.md. /reason is ONLY for thinking and discussion. Implementation requires `/blueprint` → user approval → execute. This is a hard gate, not a suggestion.
- **NEVER invoke /blueprint without explicit user approval.** The user must say "let's build this" or equivalent. Do not auto-trigger /blueprint because the design "looks ready."
- **Always open with the big picture.** Never start at detail level.
- **Follow the user's thinking.** Don't force hierarchy or sequential exploration.
- **Write sparingly.** IDEALOG.md captures decisions, not exploration.
- **Every small-level item needs a "Why."** Reasoning enables judgment calls on edge cases.
- **Read IDEALOG.md failed approaches before proposing plans.** Never re-suggest a known dead end.
- **Keep ASCII maps compact.** 15 lines max for big picture. Scannable in 5 seconds.
- **Reference, don't inline.** Point to KNOWLEDGE.md sections and IMPLEMENTATION.md rather than copying content.
- **Visualize on WHITEBOARD.md.** Write diagrams and maps there so they persist in the tmux viewer pane.
- **Allowed tools during /reason:** Read, Grep, Glob (for codebase exploration), Edit/Write (ONLY to IDEALOG.md, NOTEBOOK.md, and WHITEBOARD.md), Bash (ONLY for reading — `cat`, `ls`, `wc`, `git log`, tmux pane setup. NEVER for installing, creating, or modifying code).
- **NOTEBOOK.md is owned by /reason.** Created on first technical finding, wiped by `/reason-end`. Dump raw findings freely — no formatting pressure. `/save-session` graduates worthy findings to KNOWLEDGE.md but does not delete NOTEBOOK.md.
