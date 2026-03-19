# Research Environment Optimization — Knowledge File

> Reference material organized by topic. Each topic has optional subsections:
> Findings, Design, Reference, Decisions (ordered, skip what doesn't apply).

## Summary

This research question analyzed the `everything-claude-code` (ECC) framework and our own workflow to identify gaps in session continuity, planning, and negative knowledge capture. The core insight: **context is the scarcest resource** — every optimization should reduce wasted tokens and improve session persistence. We built a four-document architecture (KNOWLEDGE + IDEALOG + PLAN + WHITEBOARD), 18 skills covering research lifecycle, planning, engineering, and textbook workflows, and a tmux workspace integration with live-rendered markdown panes. NOTEBOOK.md was tried as a separate scratch pad but merged back into IDEALOG due to routing friction. The system is fully implemented and in daily use.

## ECC Analysis

### Findings

We organize by *domain* (research questions, engines, pipelines); ECC organizes by *tool type* (agents, commands, skills, rules). Ours is content-centric; theirs is infrastructure-centric.

Our research question lifecycle has no ECC equivalent — they have no knowledge accumulation, no completion lifecycle, no promotion. Our engine work is tracked across 3 persistent files (PROGRESS.md, IMPLEMENTATION.md, improvement.md) with a formal Orientation Protocol; ECC uses a single self-contained plan file.

**Patterns skipped**: Continuous learning/instinct system (over-engineered for solo research), multi-instance parallelization, 108 framework-specific skills (we only need Python/CUDA), extensive MCP server configs, security review agent, formalized agent library.

### Reference

**Skill gap analysis (pre-implementation state, kept for historical reference):**

| Domain | Pre-Implementation Coverage | Gap |
|--------|---------------------------|-----|
| Research lifecycle | 6 skills | None |
| Textbook | 2 skills | None |
| Multi-session planning | None | **High** — no compaction-proof step planning |
| Session persistence | None | **High** — no failed-approach capture |
| Verification | None | **Medium** — manual pytest per engine |
| Debugging / build-fix | None | **Medium** — no systematic error resolution |
| Compaction strategy | None | **Medium** — manual protocol only |

All gaps now closed by implemented skills.

**Session persistence comparison:**

| Aspect | Our System | ECC |
|--------|-----------|-----|
| Scope | Research *question* (persistent entity) | *Session* (ephemeral work unit) |
| State source | README + KNOWLEDGE + IDEALOG (version-controlled) | `~/.claude/sessions/*.tmp` (not version-controlled) |
| Captures failure | IDEALOG Failed Approaches | "What Did NOT Work" with exact errors |
| Next step | IDEALOG "Next Step" section | Required "Exact Next Step" section |
| Wrap-up | `/save-session` (5-job cleanup agent) | Separate `/save-session` command |

**What each system does better:**

| Strength | Our System | ECC |
|----------|-----------|-----|
| Knowledge accumulation | KNOWLEDGE.md per question, promoted on completion | Nothing — session files are ephemeral |
| Research lifecycle | 6 skills: inception through completion with formal criteria | No research lifecycle |
| Domain structure | Research questions linked to engines linked to experiments | Flat project structure |
| Negative knowledge | IDEALOG Failed Approaches (version-controlled, scoped) | Session files capture failures (ephemeral) |
| Hooks/automation | 1 hook (PreCompact) | 16 hooks |
| Paper management | Full PubMed pipeline with structured summaries | None |
| Experiment tracking | EXPERIMENT.md with backlinks to research questions | None |

### Decisions

| Decision | Rationale |
|----------|-----------|
| Adopt hooks selectively, not full ECC ecosystem | Domain-specific research project, not a general dev team |
| Keep CLAUDE.md as single file | Not yet at the pain point (~200 lines). Revisit if >400 lines |
| Don't restructure to match ECC directory layout | Research/Engines/Pipelines separation is purpose-built for cardiac EP |

## Document Architecture

### Findings

KNOWLEDGE.md was overloaded — it held reference material, process narrative, and scratch thinking. Negative knowledge (failed approaches) had no home. Routing friction between too many documents causes content to land in the wrong place. The solution: each document has a single resolution level and a single purpose.

### Design

**Four per-question documents:**

| Document | Resolution | Purpose | Lifecycle |
|----------|-----------|---------|-----------|
| **KNOWLEDGE.md** | High | Reference: facts, analysis, designs, comparisons | Accumulates, promoted to `Research/Knowledge/` on completion |
| **IDEALOG.md** | Low | Thinking trail: insights, failed approaches, session log | Living, archived on completion |
| **PLAN.md** | High (structured) | Cold-start agent execution steps | Created by `/blueprint`, steps checked off, archived |
| **WHITEBOARD.md** | Visual | ASCII diagrams, trade-off tables | Per-question (`Research/Active/{question}/`). Ephemeral, wiped by `/reason-end`, gitignored |

**Research question folder structure:**

```
Research/Active/{question}/
├── README.md           Status, criteria, sub-questions, literature, experiments
├── KNOWLEDGE.md        Reference (promoted on completion)
├── IDEALOG.md          Thinking trail (archived on completion)
├── WHITEBOARD.md       Ephemeral diagrams for tmux workspace (gitignored)
├── PLAN.md             Agent execution steps (created by /blueprint)
├── literature/         Paper summaries
├── papers/             PDFs
├── code_examples/      Reference implementations
└── results/            Simulation outputs
```

**Document interaction:**

```
  Papers, experiments   ┌──────────────┐
  analysis, designs ──▶ │ KNOWLEDGE.md │  ← /save-session (editorial pass)
                        │ (high-res    │  ← /research (paper summaries)
                        │  reference)  │
                        └──────────────┘

  Thinking trail        ┌──────────────┐
  "oh wait" moments ──▶ │ IDEALOG.md   │  ← /reason (writes on transitions)
  failed approaches     │ (low-res     │  ← /save-session (session snapshots)
  session snapshots     │  narrative)  │  ← /quicksave (fast checkpoint)
                        └──────┬───────┘
                               │ settled idea
                               ▼
  /blueprint converts   ┌──────────────┐
  ideas into steps  ──▶ │  PLAN.md     │  ← /blueprint (generates)
                        │ (cold-start  │  ← /audit (adversarial check)
                        │  agent steps)│  ← execution marks steps [x]
                        └──────────────┘

  README.md  ← umbrella (status, criteria, literature, experiments)
  MASTER.md  ← project-level tracking
  PROGRESS.md (engine) ← execution cursor (done/in-progress/next)
```

### Decisions

| Decision | Rationale |
|----------|-----------|
| KNOWLEDGE + IDEALOG + PLAN (three-doc architecture) | KNOWLEDGE = reference, IDEALOG = narrative, PLAN = machine execution. Single resolution per document. |
| KNOWLEDGE keeps high resolution | Detailed comparison tables, design rationale are reference material you look up later |
| PLAN.md is machine-targeted, not human-targeted | Every step self-contained with full context brief for cold-start `claude -p` execution |
| PLAN.md organized by phases, not flat steps | Phases are independently deliverable with own verification, exit criteria, cleanup, and commit point. Matches existing engine workflow. |
| MASTER_KNOWLEDGE_INDEX.md as index, not summary | Points to where knowledge lives and maps cross-question connections. Does NOT duplicate findings. Avoids drift. |
| NOTEBOOK.md merged into IDEALOG | Tried as separate scratch pad, merged back due to routing friction (strategic vs technical judgment faded in long conversations) |

## Skill Pipeline

### Design

**18 skills by category:**

| Category | Skills |
|----------|--------|
| Research lifecycle | `/research-new`, `/research-resume`, `/quicksave`, `/research-status`, `/research-complete`, `/research`, `/summarize-paper` |
| Planning & reasoning | `/reason`, `/blueprint`, `/blueprint-revise`, `/audit`, `/save-session`, `/reason-end`, `/quick-implement` |
| Engineering | `/verify`, `/build-fix`, `/strategic-compact` |
| Textbook | `/textbook-edit`, `/textbook-compile` |

**Hook:** PreCompact — logs compaction event, triggers emergency IDEALOG dump if `/reason` is active.

**Skill length risk** — instructions in long skill files fade from active context in long conversations. Observed threshold: ~150 lines. Above 200 lines, instruction compliance degrades significantly.

| Risk | Skill | Lines | Mitigation |
|------|-------|-------|------------|
| HIGH | `/research` | 287 | Consider splitting into sub-skills (discover, screen, acquire, summarize, file) |
| HIGH | `/reason` | 254 | Was split from `/blueprint`; could move examples to KNOWLEDGE.md |
| HIGH | `/research-new` | 221 | Template-heavy; could reference KNOWLEDGE.md for templates instead of inlining |
| HIGH | `/blueprint` | 210 | Already split `/blueprint-revise` out to avoid growth |
| MED | `/research-status` | 171 | Monitor |
| MED | `/save-session` | 169 | Monitor |
| OK | All others | <150 | Within threshold |

Rule: keep skills under 150 lines. If a skill needs more, split it or reference KNOWLEDGE.md for detailed templates/examples instead of inlining them.

**Planning pipeline:**

```
/reason "topic"
  │  Interactive: big-picture map → user drills down or jumps organically
  │  Writes to IDEALOG on transitions (~3-4 writes per 30min session)
  ├──→ User: "let's build this"
  ▼
/blueprint
  │  Reads IDEALOG (settled decisions, failed approaches) + codebase
  │  Generates PLAN.md with cold-start steps
  │  Asks: "Want adversarial audit?"
  ├──→ User: "yes"
  ▼
/audit
  │  Opus subagent reviews PLAN.md (read-only tools)
  │  Returns severity-sorted issues
  ▼
  Execute PLAN.md steps
```

**`/reason` zoom model:** Always opens with a scannable big-picture map (goal, scope, phases, risks, connections). User can drill down sequentially (big to middle to small) or jump organically — the agent follows the user's lead. It tracks settled vs open decisions internally and writes to IDEALOG on natural transitions (decisions settled, approaches rejected, topic shifts, explicit requests, PreCompact). One agent, not two modes.

**`/blueprint` PLAN.md structure:** Each step contains: Context Brief, Read First, Why, Implementation Spec (files + interfaces), Pseudocode, Test Spec, Checklist, Risk. Steps grouped into phases with shared Phase Context. Each phase has: verification commands, exit criteria, cleanup pass, commit point. Known failures from IDEALOG included so the agent does not retry dead ends.

**`/save-session` jobs:**
1. Session snapshot appended to IDEALOG Session Log
2. Full editorial pass on KNOWLEDGE.md (restructure, rewrite, deduplicate)
3. Cross-reference IDEALOG and KNOWLEDGE for consistency
4. Condense verbose IDEALOG entries (preserve narrative arc, decisions, failures)
5. Update MASTER_KNOWLEDGE_INDEX.md (one-liners and cross-references)

### Reference

**ECC planner comparison:**

| Aspect | ECC Planner | Our `/reason` |
|--------|-------------|---------------|
| Model | Opus (read-only) | Main conversation (full tools) |
| Output | One-shot text dump | Interactive zoom: big, middle, small |
| Interaction | "Accept Y/N" | Organic follow-the-user + structured drill-down |
| Persistence | Ephemeral (chat text) | IDEALOG.md (survives compaction) |
| Domain awareness | Generic software dev | Reads KNOWLEDGE, IDEALOG, engine PROGRESS.md |
| Negative knowledge | None | Reads IDEALOG Failed Approaches |
| Trade-off analysis | In architect agent (separate) | Integrated (visual comparison tables) |

**Complexity tiers (borrowed from ECC):**

| Tier | Pipeline | Model |
|------|----------|-------|
| trivial | implement, verify | haiku/sonnet |
| small | implement, verify, self-review | sonnet |
| medium | read context, implement, verify, cleanup | sonnet |
| large | read context, implement, verify, cleanup, `/audit` | opus |

**Domain-specific phasing (adapted from ECC):**
- Phase 1: Minimum viable (isotropic first, scalar before tensor)
- Phase 2: Core (full implementation)
- Phase 3: Validation (against V5.3 or literature)
- Phase 4: Optimization (GPU, LUT acceleration)

**Borrowed patterns from ECC:**
- Per-step "Why" field for agent judgment calls on edge cases
- Per-step risk assessment (Low/Medium/High) with level-appropriate filtering
- De-sloppify cleanup pass after implementation (removes debug artifacts, checks float64)
- Known failures section in plans so agents do not retry dead ends
- Author-bias elimination for `/audit` (separate agent, read-only tools)
- Architecture changes overview before diving into steps
- Testing strategy as first-class concern at every zoom level
- Success criteria with checkboxes feeding into README.md completion criteria
- Red flags checklist: steps without verification, phases that cannot be tested independently, missing float64, modifying V5.3, missing EXPERIMENT.md backlinks

### Decisions

| Decision | Rationale |
|----------|-----------|
| `/reason` writes on transitions, not every exchange | ~3-4 writes per 30min session. Minimizes disruption. |
| `/reason` uses big-middle-small zoom with organic flow | Structured opening, then follow the user. No forcing back into hierarchy. |
| `/reason` can invoke `/blueprint` | "Let's build this" triggers seamless transition from thinking to execution planning |
| `/save-session` is a comprehensive cleanup agent (5 jobs) | Without editorial cleanup, KNOWLEDGE degrades to chronological dump and cross-question connections stay invisible |
| `/quicksave` replaced `/research-update` | Simpler: dump chat summary to IDEALOG. No routing logic. `/save-session` handles the heavy editorial work. |
| `/audit` is standalone opt-in (renamed from `/review`) | `/blueprint` asks "Want adversarial audit?" — user decides. Can also be used on any document independently. |
| PLAN.md lives in research question folder | Scoped to question, version-controlled, co-located with KNOWLEDGE + IDEALOG. Not in a separate `./plans/` directory. |
| PLAN.md steps include full scaffolding | Context Brief, Implementation Spec, Pseudocode, Test Spec, Checklist — everything the agent needs to write code without guessing |

## Workspace Integration

### Findings

| Feature | tmux | zellij |
|---------|------|--------|
| Pane targeting | By index (safe, precise) | By focus direction (risky) |
| Read pane content | `capture-pane -t N -p` | Not available natively |
| Send to specific pane | `send-keys -t N` | Focused pane only |
| Close specific pane | `kill-pane -t N` (safe) | `close-pane` (focused — can kill Claude) |
| New pane cwd | Inherits | Must `cd` explicitly |

`watch --color` strips glow's ANSI codes. Shell loop with md5sum change detection works correctly.

Glow installed via `sudo snap install glow`. Custom style in `.glow-style.json` at project root (snap cannot access `~/.config/`). Removes `#` prefix from headers, adds color coding per heading level.

### Design

**Layout (50-25-25):**

```
┌──────────────────┬──────────────┐
│                  │ KNOWLEDGE.md │
│  Claude Code     │  (glow)      │
│                  ├──────────────┤
│                  │ WHITEBOARD.md│
│                  │  (glow)      │
└──────────────────┴──────────────┘
```

Setup triggered lazily by `/reason`. Teardown by `/reason-end`.

**Auto-refresh shell loop (md5sum change detection, prevents flicker):**

```bash
H=""; while true; do
  N=$(md5sum FILE | cut -d" " -f1)
  [ "$N" != "$H" ] && clear && glow -s .glow-style.json -w 70 FILE && H=$N
  sleep 2
done
```

**Multiplexer detection:**

```bash
if [ -n "$TMUX" ]; then echo "tmux"
elif [ -n "$ZELLIJ" ]; then echo "zellij"
else echo "none"; fi
```

### Decisions

| Decision | Rationale |
|----------|-----------|
| tmux only | Precise pane targeting by index, safe `kill-pane -t N`, native `capture-pane` |
| Dynamic glow width | Adapts to terminal size |
| `/reason` owns pane lifecycle | Setup on start, teardown on `/reason-end` |

## Open Questions

- PLAN.md for engine-only work (not tied to a research question): where does it live?
- How does PLAN.md relate to existing PROGRESS.md? Should step completion auto-update PROGRESS.md?
- `/reason` compaction recovery: how does an interactive session resume after compaction? IDEALOG has the written insights but conversational state is lost.
- Context files for different modes? Deferred until CLAUDE.md exceeds ~400 lines.

## Connections

- **Engines**: All (cross-cutting workflow concern)
- **Related research**: None directly; this is meta-research about the research process itself
- **Pipelines**: All (same workflow patterns apply)
