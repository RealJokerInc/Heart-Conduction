# Research Environment Optimization — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

Analyzed `everything-claude-code` (github.com/affaan-m/everything-claude-code), the most popular open-source Claude Code configuration framework. Key takeaway: **context is the scarcest resource** — every optimization should reduce wasted tokens and improve session continuity.

Also audited our own 9 existing skills (updated 2026-03-17). **Key finding**: our skills cover the research writing lifecycle comprehensively (6 skills from `/research-new` through `/research-complete`) but have **zero coverage for engineering workflow** — no planning, session persistence, verification, debugging, or compaction management skills.

### Skill Gap Analysis

| Domain | Our Coverage | Gap |
|--------|-------------|-----|
| Research lifecycle | 6 skills (new → resume → update → status → complete + paper pipeline) | None — comprehensive |
| Textbook | 2 skills (edit + compile) | None |
| Multi-session planning | None | **High** — engine phases, optimizer pipeline, cross-engine work all span sessions. No skill helps plan compaction-proof steps. |
| Session persistence | None | **High** — PROGRESS.md captures success but not failed approaches or dead ends. New sessions blindly retry what already failed. |
| Verification | None | **Medium** — pytest runs manually with different commands per engine. No standardized check. |
| Debugging / build-fix | None | **Medium** — no systematic error resolution loop with guardrails. |
| Compaction strategy | None (manual protocol in CLAUDE.md only) | **Medium** — no guidance on WHEN to compact proactively; no hook for state preservation. |

### Design Conflict Analysis (updated 2026-03-17)

Deeper comparison of ECC patterns against our existing architecture revealed two significant conflicts:

**1. `/blueprint` contradicts our 3-file planning architecture.**

Our system already solves compaction recovery with PROGRESS.md (state) + IMPLEMENTATION.md (specs) + improvement.md (ABCs), reassembled by the Orientation Protocol. ECC's `/blueprint` replaces all of that with a single plan file containing self-contained context briefs per step. These are two solutions to the same problem — adopting both creates confusion about the source of truth. Our Task system (TaskCreate/TaskUpdate) also conflicts with blueprint's markdown-only step tracking.

**2. `/save-session` + `/resume-session` overlap heavily with `/research-resume` + `/research-update`.**

| Aspect | Our System | ECC System |
|--------|-----------|------------|
| Scope | Research *question* (persistent entity) | *Session* (ephemeral work unit) |
| State source | README.md + KNOWLEDGE.md (version-controlled) | `~/.claude/sessions/*.tmp` (not version-controlled) |
| Captures success | Yes (KNOWLEDGE.md findings, decisions) | Yes (what worked, with evidence) |
| **Captures failure** | **No** | **Yes — "What Did NOT Work" with exact errors** |
| Next step | No explicit field | Required "Exact Next Step" section |
| File status tracking | No | Table of every file touched (done/in-progress/broken) |
| Wrap-up | Built into `/research-resume` Step 8 | Separate `/save-session` command |

**The real gap is negative knowledge.** Our entire research lifecycle (new → resume → update → complete) captures positive knowledge only. Nothing records "tried X, failed because Y — don't retry." This is the single most valuable thing ECC's session system does.

### Three-Document Architecture (finalized 2026-03-17)

**Key insight**: KNOWLEDGE.md was doing double duty — holding both research findings AND planning/process state. But the solution is NOT to strip KNOWLEDGE.md down to bare facts. KNOWLEDGE.md should keep its **high resolution** — detailed analysis, comparison tables, design rationale, templates. What moves out is only the **thinking trail** (the narrative of how we got there).

Briefly explored a four-document model (KNOWLEDGE + IDEALOG + WORKLOG + PLAN) but WORKLOG was unnecessary — failed approaches and session snapshots are part of the thinking trail and belong in IDEALOG. Three documents is sufficient.

#### The Three Documents (per research question)

| Document | What it is | Analogy | Contains | Lifecycle |
|----------|-----------|---------|----------|-----------|
| **KNOWLEDGE.md** | Reference manual | Look things up | Facts, detailed analysis, comparisons, designs, templates, decisions with full rationale — everything you'd reference later. **High resolution.** | Accumulates → promoted to `Research/Knowledge/` on completion |
| **IDEALOG.md** (new) | Research diary | Follow the story | Thinking trail: "oh wait" moments, branching insights, failed approaches, session snapshots, what to try next. **Low resolution** — condensed narrative you scan in 30 seconds. | Living during active work → archived on completion |
| **PLAN.md** (new) | Construction manual | Hand to a contractor | Self-contained implementation steps for cold-start agent execution. **High resolution** but structured for machines. | Created by `/blueprint` → steps checked off → archived when done |

#### The Key Distinction

Same information, different purpose:

**KNOWLEDGE.md**: `ECC's /plan spawns an Opus planner agent with read-only tools (Read/Grep/Glob). Output is ephemeral chat text. /blueprint runs a 5-phase pipeline (research → design → draft → review → register) and writes a persistent plans/*.md file. They're independent entry points, not sequential.` — Detailed reference you'd look up.

**IDEALOG.md**: `2026-03-17: Realized ECC's /plan and /blueprint serve different purposes — conversational thinking vs document generation. Our /reason should write to IDEALOG (persistent thinking trail), unlike ECC's ephemeral chat output. → maps cleanly to our IDEALOG → /blueprint → PLAN.md pipeline.` — The story of the insight.

#### Why This Resolves the ECC Conflicts

**Before**: ECC's `/blueprint` conflicted with our PROGRESS.md + IMPLEMENTATION.md. Now: **PLAN.md = ECC blueprint, exactly.** No overlap.

**Before**: ECC's `/save-session` conflicted with `/research-update`. Now: `/save-session` writes to **IDEALOG.md** (session snapshots). `/research-update` routes to KNOWLEDGE (findings/analysis) or IDEALOG (ideas/failures). No conflict.

#### IDEALOG.md Template

```markdown
# {Question Title} — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
{What we're currently pursuing and why — updated when direction changes}

## Next Step
{Precise enough to resume with zero context. Updated at session end.}

## Thread

### {date}: {insight or idea title}
{What the idea is, why it matters, what it implies.
Link to spawned questions, PLAN.md, or note if rejected.}

## Failed Approaches
- **{approach}** ({date}) — failed because: {exact error or reasoning}

## Session Log
{Brief snapshots at session boundaries — written by /save-session.}

### {date} Session
**Worked on**: {what}
**Accomplished**: {what, with evidence}
**Next**: {exact next step}
```

#### PLAN.md Template

```markdown
# Plan: {title}

Created: {date}
Engine(s): {Bidomain V1 | V5.4 | LBM V1 | Optimizer}
Research question: [{name}](README.md)
IDEALOG context: [Idea that motivated this](IDEALOG.md#{section})

## Objective
{One paragraph — what we're implementing and why}

## Steps

### Step 1: {title}
**Depends on**: None | Step N
**Engine**: {which engine}

#### Context Brief
{Everything a cold-start agent needs to execute this step.
File paths, conventions, gotchas. This survives compaction.}

#### Read First
- `{path}` — {why}

#### Tasks
- [ ] {concrete action}

#### Verify
```bash
{exact commands}
```

#### Exit Criteria
- {measurable}

### Step 2: ...

## Risks
- **HIGH**: {risk} — mitigation: {approach}

## Mutation Log
{When steps are skipped, split, inserted, or reordered during execution}
```

**PLAN.md is for the coding agent, not for the human.** The human discusses ideas in IDEALOG.md and chat. When an approach is settled, `/blueprint` generates PLAN.md from the conversation + IDEALOG context. The agent then executes PLAN.md steps cold.

#### Updated Research Question Folder Structure

```
Research/Active/{question}/
├── README.md           Status, criteria, sub-questions, literature, experiments
├── KNOWLEDGE.md        Reference: facts, analysis, designs, comparisons (promoted on completion)
├── IDEALOG.md          Thinking trail: insights, failures, session log (NEW)
├── PLAN.md             Cold-start agent execution steps (NEW, created by /blueprint)
├── literature/         Paper summaries
├── papers/             PDFs
├── code_examples/      Reference implementations
└── results/            Simulation outputs
```

#### How the Documents Interact

```
  Papers, experiments   ┌──────────────┐
  analysis, designs ──▶ │ KNOWLEDGE.md │  ← /research-update finding
                        │ (high-res    │  ← /research (paper summaries)
                        │  reference)  │
                        └──────────────┘

  Thinking trail        ┌──────────────┐
  "oh wait" moments ──▶ │ IDEALOG.md   │  ← /reason (writes on transitions)
  failed approaches     │ (low-res     │  ← /save-session (session snapshots)
  session snapshots     │  narrative)  │  ← /research-update idea/failure
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

### Revised Skills Plan (updated 2026-03-17)

**Extend existing skills:**
1. **Extend `/research-new`** — Create IDEALOG.md with template. KNOWLEDGE.md keeps high resolution (analysis, designs, comparisons) but narrative process stuff moves to IDEALOG.
2. **Extend `/research-resume`** — Read IDEALOG.md in addition to KNOWLEDGE.md. Briefing includes: "Current Direction" + "Next Step" (from IDEALOG), "What NOT to retry" (from IDEALOG Failed Approaches).
3. **Extend `/research-update`** — Route by type: `finding` (facts, analysis, designs) → KNOWLEDGE.md. `idea`, `failure`, `issue`, `next-step` → IDEALOG.md.
4. **Extend `/research-complete`** — Archive IDEALOG.md with the question folder in Complete/ (not promoted to Research/Knowledge/).

**New skills:**
4. **`/reason`** — Interactive planning agent. Reads IDEALOG, KNOWLEDGE, engine files. Discusses ideas conversationally. Writes to IDEALOG.md on **natural transitions** — not after every exchange. Write triggers: user settles on a decision, rejects an approach, shifts topics, says "write that down", or PreCompact fires. Batch related insights into single edits to minimize workflow disruption. Can invoke `/blueprint` when user says "let's build this."
5. **`/blueprint`** — Autonomous pipeline. Reads IDEALOG.md (settled approach, known failures to avoid) + codebase. Generates PLAN.md with self-contained steps. Asks "Want adversarial audit?" before finalizing.
6. **`/save-session`** — Session-end cleanup agent. Five jobs: (1) write session snapshot to IDEALOG.md Session Log, (2) organize KNOWLEDGE.md for clean reference lookup, (3) cross-reference IDEALOG ↔ KNOWLEDGE for consistency, (4) condense verbose IDEALOG entries, (5) update MASTER_KNOWLEDGE_INDEX.md. See detailed design below.
7. **`/audit`** — Adversarial review. Spawns Opus subagent with read-only tools. Challenges any document (PLAN.md, design, code). Opt-in, never auto-triggered.
8. **`/verify`** — Auto-detect engine, run test suite, produce report.
9. **`/build-fix`** — Systematic error resolution with guardrails.
10. **`/strategic-compact`** — Decision table for phase transitions + pre-compact checklist.

**Hook:**
11. **PreCompact** — Logs compaction event. If `/reason` session is active, triggers emergency dump of unsaved insights to IDEALOG.md. Reminds to re-read PROGRESS.md.

**Not implementing:**
- **`/aside`** — Investigated multi-turn sidebar with auto-cleanup. The disappearing-from-transcript behavior is hardcoded into Claude Code's `/btw` UI overlay — not exposed to skills, hooks, or any extensibility system. Use built-in `/btw` instead. (dropped 2026-03-17)
- **WORKLOG.md** — Briefly explored as a separate "lab notebook" for tactical details. Dropped because failed approaches and session snapshots are part of the thinking trail and belong in IDEALOG.md. Three documents is sufficient. (dropped 2026-03-17)

#### `/reason` Write Behavior (Key Design Detail)

The interactive `/reason` skill does NOT write after every chat exchange. It accumulates ideas in context and writes **batch updates** to IDEALOG.md on natural transitions:

| Trigger | What gets written |
|---------|-------------------|
| User settles on a decision | The decision + rationale |
| User rejects an approach | Failed approach entry |
| Topic shift in conversation | Summarize previous topic's conclusions |
| User says "write that down" | Whatever was just discussed |
| PreCompact hook fires | Emergency dump of unsaved insights |
| `/save-session` invoked | Session snapshot to Session Log section |
| User says "let's build this" | Trigger `/blueprint` to generate PLAN.md |

All writes go to **IDEALOG.md** — the single thinking-trail document. A typical 30-minute `/reason` session produces ~3-4 writes, not 15-20.

#### `/reason` Agent Presentation: Big → Middle → Small (Sequential + Organic)

The `/reason` agent uses a **zoom-level model** for presentation. It always opens with the big picture, then follows the user's lead — either drilling down sequentially or jumping organically.

**Rule 1: Always open with the big picture.**

When `/reason` starts, it presents a visual map — scannable in 5 seconds:

```
═══════════════════════════════════════════════════════
  PLAN: Add anisotropic diffusion to Bidomain V1
═══════════════════════════════════════════════════════

  WHY: Boundary speedup research requires fiber-direction-
       dependent CV measurement. Currently isotropic only.

  SCOPE: Bidomain V1 engine, 3 phases, ~5 sessions

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  Phase 1    │────▶│  Phase 2    │────▶│  Phase 3    │
  │  Tensor     │     │  Solver     │     │  Validation  │
  │  conductivity│     │  adaptation │     │  & CV meas.  │
  └─────────────┘     └─────────────┘     └─────────────┘
       3 steps             2 steps             3 steps

  RISKS: ● HIGH — spectral solver assumes isotropic
         ○ MED  — anisotropy ratio affects stability

  BUILDS ON: isotropic CV validation (54.3 cm/s confirmed)
  FEEDS INTO: boundary_conduction_speedup anisotropic sub-question

═══════════════════════════════════════════════════════
  Drill into a phase? Or discuss the big picture?
═══════════════════════════════════════════════════════
```

**Rule 2: Follow the user's lead after the opening.**

If user drills down sequentially (big → middle → small), present each level with the same visual structure — goal, current state, steps, dependencies, key questions.

If user jumps around organically ("what about the solver? actually wait, does the ionic model handle this?"), follow the jumps. Discuss at whatever resolution they're at. Don't force them back into the hierarchy.

**Rule 3: Track what's settled vs open internally.**

Regardless of whether the user explored via structured drill-down or organic jumping, the agent tracks:
- Which phases/steps have been discussed
- Which decisions are settled vs still open
- Which questions remain unresolved

This internal tracking drives when to write to IDEALOG.md (write on settled decisions, not on exploration) and feeds into `/blueprint` when the user says "let's build this."

**Middle-level presentation** (when user drills into a phase):

```
───────────────────────────────────────────────────────
  Phase 1: Tensor Conductivity
───────────────────────────────────────────────────────

  GOAL: Replace scalar D_i, D_e with tensor D_i, D_e

  CURRENT STATE:
    operators.py — scalar D * Laplacian
    parabolic.py — A_para built with scalar

  STEPS:
    1.1  Extend ConductivityField to hold tensors    [small]
    1.2  Rewrite FDM stencil for anisotropic case     [medium]
    1.3  Update parabolic operator assembly            [medium]

  DEPENDS ON: nothing
  UNLOCKS: Phase 2

  KEY QUESTION: face-based vs node-based tensor averaging?
    → KNOWLEDGE.md says face-based (bidomain_simulation.md)

───────────────────────────────────────────────────────
  Drill into a step? Or back to big picture?
───────────────────────────────────────────────────────
```

**Small-level presentation** (when user drills into a step):

```
───────────────────────────────────────────────────────
  Step 1.2: Rewrite FDM stencil for anisotropic case
───────────────────────────────────────────────────────

  FILE: cardiac_sim/simulation/diffusion/operators.py:45-120

  WHAT CHANGES:
    Current: 5-point stencil, scalar D
    New: 9-point stencil, tensor D = [[Dxx, Dxy], [Dxy, Dyy]]
         Adds cross-derivative terms

  REFERENCE:
    IMPLEMENTATION.md Phase 2 validation table
    Research/Knowledge/bidomain_simulation.md §FDM stencils

  VERIFY:
    pytest tests/test_phase2_fdm.py -v
    New test: anisotropic_stencil_symmetry

  RISK: Cross-derivative stencil can produce negative coefficients
    → Mitigation: check positive-definiteness

───────────────────────────────────────────────────────
  Questions? Modify? Or move on?
───────────────────────────────────────────────────────
```

**Organic flow example** — this is equally valid:

```
User: /reason "add anisotropic diffusion to Bidomain V1"
Agent: [presents big picture map]

User: "what about the spectral solver? can it handle anisotropy?"
Agent: [jumps to Phase 2 solver detail, discusses DST limitations]

User: "oh wait, does our conductivity field even support tensors?"
Agent: [jumps back to Phase 1 step 1.1, shows current ConductivityField API]

User: "let's just start with diagonal tensors, no cross-terms"
Agent: [WRITES to IDEALOG: decision to start with diagonal-only tensors]
       [adjusts plan: step 1.2 simplifies from 9-point to modified 5-point]

User: "okay what does phase 3 validation look like?"
Agent: [jumps to Phase 3, shows CV measurement approach]
```

The agent doesn't say "you skipped Phase 2, let's go back." It follows the thinking.

#### Borrowed from ECC (what they do well, adapted for us)

**Per-step reasoning ("Why")**: Every step at the small level includes WHY this step exists, not just what to do. Adopted directly from ECC's planner. This is critical for understanding impact if a step needs to be skipped or reordered.

**Per-step risk assessment**: Each step gets a risk tag (Low/Medium/High) with specific risk description, not just global project risks. At the big-picture level, only HIGH risks are shown. At middle level, HIGH + MEDIUM. At small level, all risks including LOW.

**Phasing philosophy**: Phases should be independently deliverable where possible:
- Phase 1: Minimum viable — smallest slice that demonstrates value
- Phase 2: Core — complete happy path
- Phase 3: Edge cases — error handling, robustness
- Phase 4: Optimization — performance, polish

Adapted for our domain: Phase 1 often means "isotropic first" or "scalar before tensor." Phase 2 is "full implementation." Phase 3 is "validation against V5.3 or literature." Phase 4 is "GPU optimization / LUT acceleration."

**Testing strategy as first-class concern**: Not an afterthought. At the big-picture level, the map shows "N tests" per phase. At the middle level, each phase lists what tests are needed. At the small level, each step has exact `pytest` commands.

**Success criteria with checkboxes**: Concrete "done" definition. Shown at the big-picture level. These feed directly into the research question's README.md completion criteria.

**Architecture changes section**: Before drilling into steps, the middle level shows what's being structurally added/modified. For our domain: "New file: `diffusion/anisotropic.py`" or "Modified: `operators.py` Laplacian assembly" or "New test: `test_anisotropic_stencil.py`."

**Trade-off analysis**: When `/reason` encounters a design fork (e.g., "face-based vs node-based averaging"), it presents:
```
  TRADE-OFF: Tensor averaging method
  ┌──────────────────┬──────────────────┐
  │  Face-based      │  Node-based      │
  ├──────────────────┼──────────────────┤
  │ ✓ Conservative   │ ✓ Simpler code   │
  │ ✓ Matches V5.3   │ ✓ Fewer ops      │
  │ ✗ More complex   │ ✗ Not conservative│
  │ ✗ Face indexing   │ ✗ Diverges at    │
  │   needed         │   boundaries     │
  └──────────────────┴──────────────────┘
  RECOMMENDATION: Face-based (matches existing convention)
```
User decides, decision gets written to IDEALOG.md.

**Red flags checklist**: `/reason` agent internally checks for anti-patterns during planning and flags them proactively. Adapted for our domain:
- Steps without verification commands
- Phases that can't be tested independently
- Missing float64 dtype checks (our #1 numerical bug source)
- Modifying V5.3 (validated baseline — never allowed)
- Steps that duplicate code across engines (should go in cardiac_core)
- Missing EXPERIMENT.md backlinks for new experiments
- Plans with no connection to KNOWLEDGE.md or research question criteria

#### ECC Planner Comparison

| Aspect | ECC Planner | Our `/reason` |
|--------|-------------|-------------|
| Model | Opus (read-only) | Main conversation (full tools) |
| Output | One-shot text dump | Interactive zoom: big → middle → small |
| Visual | Wall of text, no diagrams | ASCII diagrams, boxed sections, trade-off tables |
| Interaction | "Accept Y/N" | Organic follow-the-user + structured drill-down |
| Persistence | Ephemeral (chat text) | IDEALOG.md (survives compaction) |
| Resolution | One zoom level (everything at once) | Three zoom levels (big picture → phase → step) |
| Domain awareness | Generic software dev | Reads KNOWLEDGE.md, IDEALOG.md, engine PROGRESS.md |
| Negative knowledge | None | Reads IDEALOG Failed Approaches, avoids known dead ends |
| Per-step reasoning | Yes ("Why" field) | Yes (adopted from ECC) |
| Per-step risk | Yes (Low/Med/High) | Yes (adopted, with level-appropriate filtering) |
| Phasing philosophy | MVP → Happy path → Edge cases → Optimization | Adapted: Isotropic → Full → Validation → GPU/LUT |
| Trade-off analysis | In architect agent (separate) | Integrated into `/reason` (visual comparison tables) |
| Red flags | Generic (large functions, deep nesting) | Domain-specific (float64, V5.3, cardiac_core, backlinks) |
| Testing strategy | Section in plan | First-class at every zoom level |
| Success criteria | Checkboxes | Checkboxes that feed into README.md completion criteria |

#### Full Skill Pipeline

```
/reason "add anisotropic diffusion to Bidomain V1"
  │
  │  Interactive: opens with big-picture map
  │  User drills down or jumps organically
  │  Writes to IDEALOG.md on transitions (decisions, rejections, topic shifts)
  │  ~3-4 writes per 30min session
  │
  ├──→ User: "let's build this"
  │
  ▼
/blueprint
  │  Autonomous: reads IDEALOG (settled decisions, failed approaches) + codebase
  │  Generates PLAN.md with cold-start steps
  │  Asks: "Want adversarial audit?"
  │
  ├──→ User: "yes"
  ▼
/audit
  │  Opus subagent reviews PLAN.md
  │  Returns severity-sorted issues
  │  Issues folded into PLAN.md or accepted as risks
  │
  ▼
  Ready to execute (agent follows PLAN.md steps)
```

### `/save-session` Detailed Design (updated 2026-03-17)

Not a quick snapshot writer — a **cleanup agent** that runs at session end. Five jobs:

#### Job 1: Write Session Snapshot to IDEALOG.md

Append to the Session Log section:

```markdown
### {date} Session
**Worked on**: {what}
**Accomplished**: {what, with evidence}
**Next**: {exact next step}
```

This is the simple part — same as originally designed.

#### Job 2: Comprehensive KNOWLEDGE.md Reorganization

During a session, findings get added incrementally and ad-hoc. KNOWLEDGE.md drifts. `/save-session` does a **full editorial pass** — this can be a comprehensive rewrite if needed:

- **Restructure**: Group related findings logically, not chronologically. Create or merge sections to match the current state of understanding.
- **Rewrite for clarity**: Polish rough session notes into clean reference prose. Every entry should be something you can look up cold months later and immediately understand.
- **Merge and deduplicate**: Combine overlapping entries. If "CV ratio = 1.071" appeared in three places during incremental updates, consolidate into one authoritative entry.
- **Consistent depth**: Ensure all sections have similar resolution. A 200-line section next to a 2-line section means one needs expansion or the other needs tightening.
- **Preserve all information**: Reorganization and rewriting is fine. Deleting findings is not — every validated result stays, just in better form.

#### Job 3: Thorough IDEALOG ↔ KNOWLEDGE Cross-Reference

Full read of both documents, not just surface checks:

| Check | If found | Action |
|-------|----------|--------|
| IDEALOG idea was validated this session | Finding should be in KNOWLEDGE.md | Add to KNOWLEDGE (may already be there in rough form — polish it) |
| KNOWLEDGE finding spawned new ideas during session | IDEALOG should reference the finding | Add link in IDEALOG thread |
| IDEALOG failed approach contradicts a KNOWLEDGE entry | Inconsistency | Flag to user — may need to correct KNOWLEDGE |
| IDEALOG "Current Direction" no longer matches KNOWLEDGE state | Stale direction | Update IDEALOG direction to reflect current understanding |
| IDEALOG thread entries are verbose from mid-session discussion | Can be condensed | Collapse verbose entries into concise summaries preserving the narrative arc, decisions, and failed approaches |

#### Job 4: IDEALOG.md Cleanup

The thinking trail accumulates verbose entries during interactive `/reason` sessions. At session end, `/save-session` can:
- Collapse verbose thread entries into concise summaries
- Preserve the narrative arc (what led to what)
- Keep all decisions and failed approaches intact (never lose negative knowledge)
- Update "Current Direction" and "Next Step" to reflect session-end state

#### Job 5: Update MASTER_KNOWLEDGE_INDEX.md

`MASTER_KNOWLEDGE_INDEX.md` is a project-root **index book** — it points to where knowledge lives, not duplicates it. It tells you *where to look* and *how things connect*, not *what the findings are*.

**Why index, not summary**: A summary duplicates findings across two files (question KNOWLEDGE.md + MASTER_KNOWLEDGE_INDEX.md) and they drift apart. An index maintains pointers and connections only. `/save-session` Job 5 stays lightweight — just update the one-liner and check for new cross-references.

After finishing Jobs 1-4, `/save-session` checks:
- Does this question's one-liner in MASTER_KNOWLEDGE_INDEX.md still accurately describe its current state?
- Did this session reveal connections to other questions that aren't indexed yet?

If yes, update MASTER_KNOWLEDGE_INDEX.md — quick edits, not rewriting findings.

```markdown
# Master Knowledge

> Index book: where knowledge lives, how questions connect.
> NOT a copy of findings — follow the links for detail.
> Updated by /save-session after each research session.

## Research Statement
{Researcher's personal statement, goals, thesis direction — lives here only}

## Knowledge Index

| Question | Status | One-Liner | Knowledge |
|----------|--------|-----------|-----------|
| Boundary conduction speedup | Active | CV increases 7-13% at bath-coupled boundaries (Kleber effect) | [KNOWLEDGE](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) |
| Ionic model optimization | Active | BayesOpt multi-objective tuning, MHAS13 pilot APD=347ms | [KNOWLEDGE](Research/Active/ionic_model_optimization/KNOWLEDGE.md) |
| Bidomain simulation | Complete | FDM/FEM/FVM + 3-tier solver + Strang/RL/CN | [KNOWLEDGE](Research/Knowledge/bidomain_simulation.md) |
| ... | | | |

## Cross-References

{How questions connect to each other — the value-add of this document}

- **Boundary speedup ↔ Geometry-induced pacemaking**: Kleber effect at boundaries reduces electrotonic loading → may enable geometry-dependent pacemaker entrainment. [boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | [pacemaking](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md)
- **Ionic optimization ↔ Mature hiPSC-CM**: MHAS13 parameters from optimization feed directly into hiPSC tissue validation. [optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md) | [hipsc](Research/Active/mature_hipsc_cm_models/KNOWLEDGE.md)
- **Engine consolidation ↔ All engines**: Chi/Cm convention differences (Formulation A vs B) affect every engine. [consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md)
```

The agent reports what it changed:
```
/save-session complete:
  IDEALOG.md — added session snapshot, condensed 3 thread entries,
               updated Current Direction
  KNOWLEDGE.md — reorganized §Current Understanding (grouped tensor findings),
                  rewrote stencil section for clarity,
                  merged 2 duplicate convergence entries
  Cross-reference — graduated "face-based averaging is correct" from
                    IDEALOG idea to KNOWLEDGE §Key Findings,
                    flagged stale direction in IDEALOG (updated)
  MASTER_KNOWLEDGE_INDEX.md — updated boundary_conduction_speedup one-liner,
                         added new cross-reference: boundary ↔ pacemaking
```

#### Why This Matters

Without this cleanup step, KNOWLEDGE.md slowly degrades into a chronological dump, IDEALOG ideas that get validated never graduate, and cross-question connections are invisible. `/save-session` is the editorial agent that keeps all documents healthy — KNOWLEDGE.md as a polished reference, IDEALOG.md as a concise thinking trail, and MASTER_KNOWLEDGE_INDEX.md as the cross-cutting executive summary.

### `/blueprint` Detailed Design (updated 2026-03-17)

#### The Problem with Both Existing Systems

**Built-in Plan Mode** (Shift+Tab → `⏸ plan mode on`):
- It's a *permission mode*, not a planning tool. Entering it switches your entire session to read-only (no Bash, Edit, Write). You can't plan and implement in the same flow without toggling modes.
- The UI is excellent: a persistent plan document pinned in the top-right, blue text box, and you chat to refine it iteratively. The plan file lives in `~/.claude/plans/` (configurable via `plansDirectory` setting). It survives compaction.
- But: the plan document is generic markdown with no structure. No engine-awareness, no adversarial review, no self-contained context briefs, no verification criteria.

**ECC's `/plan` command**:
- Spawns a planner agent (read-only tools) that dumps a plan as chat text, then says "WAITING FOR CONFIRMATION" — accept/modify/reject. One-shot, not interactive.
- No persistent file, no special UI, no blue box. Just text in the conversation that gets compacted away.
- Includes risk assessment, complexity estimation, and agent-based analysis. But the output is ephemeral.

Neither alone gives us what we want. The built-in mode has the right UI but wrong workflow. ECC has the right discipline but wrong delivery.

#### Design: `/blueprint` Skill

**Core principle**: PLAN.md is written **for the machine, not the human.** The human's thinking is in IDEALOG.md. PLAN.md is a construction manual that a cold-start Claude agent can execute without any prior context — no IDEALOG needed, no conversation history, no "what were we thinking."

**Generation pipeline**: `/blueprint` reads IDEALOG.md (settled approach, known failures to avoid) + codebase (affected files, existing tests, IMPLEMENTATION.md) and generates PLAN.md autonomously.

#### PLAN.md Format (Machine-Targeted)

Each step in PLAN.md is designed to be executed as an isolated `claude -p` call or a fresh agent invocation. The step contains EVERYTHING the agent needs — no external context required.

```markdown
# PLAN: {title}

Created: {date}
Engine(s): {Bidomain V1 | V5.4 | LBM V1 | Optimizer}
Research question: [{name}](README.md)
Source: [IDEALOG.md](IDEALOG.md) — {which thread entry motivated this plan}

## Objective
{What we're implementing and why — 2-3 sentences}

## Success Criteria
- [ ] {measurable criterion — feeds into README.md completion criteria}
- [ ] All existing tests pass (no regressions)

## Architecture Changes
{Structural overview — what's being added/modified at the file level}
- NEW: `cardiac_sim/diffusion/anisotropic.py` — tensor conductivity field
- MOD: `cardiac_sim/diffusion/operators.py:45-120` — Laplacian assembly
- NEW: `tests/test_anisotropic_stencil.py` — validation suite

## Known Failures (from IDEALOG)
{Approaches already tried and failed — agent MUST NOT retry these}
- {approach} — failed because: {exact reason}

---

## Phase 1: {title}

**Goal**: {what this phase achieves — independently deliverable}
**Tier**: small | medium | large
**Estimated scope**: {N steps, ~N sessions}

### Phase Context
{Everything the agent needs to work on ANY step in this phase.
Conventions, current code state, what NOT to do.
Steps inherit this — they don't repeat it.}

### Step 1.1: {title}
**Model**: sonnet | opus

#### Read First
- `{path}:{lines}` — {what to look for}

#### Why
{Reasoning — helps the agent make judgment calls on edge cases}

#### Implementation Spec

**Files to create:**
- `{path}` — {purpose, what it exports}

**Files to modify:**
- `{path}:{lines}` — {what changes and why}

**Interfaces / Signatures:**
```python
class TensorConductivityField:
    """2D/3D tensor conductivity field for anisotropic diffusion."""
    D: torch.Tensor  # shape (Nx, Ny, 2, 2) for 2D, float64

    def get_face_averaged(self, axis: int) -> torch.Tensor:
        """Harmonic mean of tensor at cell faces along given axis."""
        ...
```

#### Pseudocode
```
for each interior node (i,j):
    D_face_x = harmonic_mean(D[i,j], D[i+1,j])
    D_face_y = harmonic_mean(D[i,j], D[i,j+1])
    Lxx = (D_face_x[0,0] * (Vm[i+1,j] - Vm[i,j]) - ...) / dx²
    Lyy = (D_face_y[1,1] * (Vm[i,j+1] - Vm[i,j]) - ...) / dy²
    Lxy = D[i,j][0,1] * (Vm[i+1,j+1] - Vm[i+1,j-1] - ...) / (4*dx*dy)
    L[i,j] = Lxx + Lyy + Lxy
```

#### Test Spec
- `test_anisotropic_stencil.py::test_diagonal_reduces_to_isotropic`
  - Setup: 64x64 grid, D = [[0.001, 0], [0, 0.001]]
  - Expected: matches isotropic Laplacian, rtol=1e-12
- `test_anisotropic_stencil.py::test_rotated_symmetry`
  - Setup: 64x64 grid, D rotated 45°, symmetric stimulus
  - Expected: solution symmetric under 90° rotation, rtol=1e-10

#### Checklist
- [ ] Read existing Laplacian assembly
- [ ] Create TensorConductivityField class
- [ ] Implement face averaging
- [ ] Modify stencil for tensor D
- [ ] Write tests
- [ ] Verify float64 throughout

#### Risk
{risk} — mitigation: {approach}

---

### Step 1.2: {title}
...

### Phase 1 Verification
```bash
conda run -n heart-conduction pytest tests/test_anisotropic_stencil.py -v
conda run -n heart-conduction pytest tests/ -v  # full regression
```

### Phase 1 Exit Criteria
- [ ] All new tests pass
- [ ] All existing tests pass (no regressions)
- [ ] float64 verified throughout
- [ ] Ready for git commit

### Phase 1 Cleanup
{De-sloppify after all steps in this phase:
remove debugging artifacts, stray prints, check float64,
ensure no V5.3 modifications}

**→ Commit point: git commit after Phase 1 passes**

---

## Phase 2: {title}
{Same structure: goal, context, steps, verification, exit criteria, cleanup, commit}

---

## Phase 3: {title}
...

---

## Final Cleanup (after all phases)
{Cross-phase de-sloppify: review ALL changes across phases,
verify EXPERIMENT.md backlinks, check for code duplication
that should go to cardiac_core}

## Mutation Log
{When phases/steps are skipped, split, inserted, or reordered during execution}
```

The phase structure means:
- Each phase is independently deliverable and committable
- The agent implements phase by phase, not step by step across phases
- Phase verification catches issues before moving on
- Phase cleanup prevents slop from accumulating across phases
- Natural commit points after each phase — rollback is per-phase, not per-step

**Phases are variable** — the number of phases, the number of steps per phase, and the structure within each phase are determined by `/blueprint` based on the scope and complexity of the work. A simple task might have 1 phase with 2 steps. A major engine feature might have 5 phases with 4-6 steps each. `/blueprint` decides the right decomposition based on IDEALOG context, codebase analysis, and complexity tier. There is no fixed number of phases or steps.

#### Key Design Patterns Borrowed from ECC

**1. Complexity tiers** — different steps get different treatment:

| Tier | Pipeline | Model |
|------|----------|-------|
| **trivial** | implement → verify | haiku/sonnet |
| **small** | implement → verify → self-review | sonnet |
| **medium** | read context → implement → verify → cleanup | sonnet |
| **large** | read context → implement → verify → cleanup → `/audit` | opus |

**2. De-sloppify pass** — a dedicated cleanup step AFTER all implementation steps. Don't constrain the implementer with "don't do X." Let it be thorough, then clean up. Removes: debugging artifacts, stray prints, float32 where float64 expected, overly defensive checks.

**3. Known failures section** — PLAN.md includes failures from IDEALOG.md so the executing agent doesn't retry dead ends. This is the bridge between human reasoning (IDEALOG) and machine execution (PLAN).

**4. Per-step "Why"** — every step explains its reasoning so the agent can make judgment calls on edge cases rather than following instructions blindly.

**5. Author-bias elimination for `/audit`** — the audit agent never generated the plan. Separate context, separate agent, read-only tools. Reviews for: completeness, dependency errors, missed edge cases, domain-specific anti-patterns.

**6. Architecture changes section** — structural overview before diving into steps. Shows what's being added/modified at the file level. Agent reads this first to understand the scope.

#### What Makes Ours Different from ECC's Blueprint

| Aspect | ECC Blueprint | Our PLAN.md |
|--------|--------------|-------------|
| Audience | Human developer following steps | Cold-start Claude agent executing autonomously |
| Context briefs | Brief ("what a fresh agent needs") | Comprehensive (conventions, known failures, current code state) |
| Known failures | Not included | Included from IDEALOG — agent must not retry |
| De-sloppify | Separate skill | Built into plan as final step |
| Complexity tiers | Not in blueprint (in autonomous-loops) | Built into plan — per-step tier + model assignment |
| Why per step | Not included | Required — agent needs reasoning for judgment calls |
| Domain awareness | Generic | Engine-specific (float64, chi=1.0, V5.3 protection, face-based stencils) |
| Cleanup checklist | Not included | Domain-specific cleanup pass (float64, backlinks, V5.3) |

#### Key Design Decisions for `/blueprint`

| Question | Decision | Rationale |
|----------|----------|-----------|
| Permission mode? | **No** — stay in normal mode | User hates toggling modes. Skill writes PLAN.md directly to the research question folder. |
| Where do plans live? | Inside the research question folder (`Research/Active/{question}/PLAN.md`) | Scoped to question, version-controlled, co-located with KNOWLEDGE + IDEALOG. Dropped ECC's `./plans/` convention — our questions have their own folders. |
| Relationship to PROGRESS.md? | **Complementary, not replacing.** Blueprint is the plan; PROGRESS.md tracks execution state. Blueprint step completion updates PROGRESS.md. | PROGRESS.md is our compaction recovery mechanism. Blueprint adds the "why" and "how" that PROGRESS.md lacks. |
| Relationship to IMPLEMENTATION.md? | Blueprint steps **link to** IMPLEMENTATION.md sections for validation criteria, rather than duplicating them. | Single source of truth for specs. Blueprint adds execution context (read-first, verify commands). |
| What about Task tool (TaskCreate/TaskUpdate)? | **Don't use both.** Blueprint's `- [ ]` / `- [x]` checkboxes replace TaskCreate for planned work. Tasks are still fine for ad-hoc work not covered by a blueprint. | Two tracking systems creates confusion. |
| Plan mutation mid-execution? | Supported — steps can be marked SKIPPED, SPLIT, or new steps INSERTED. Add `**MUTATED {date}**: {reason}` annotation. | Plans always change during execution. The annotation preserves why. |

#### Resolved Questions

- **plansDirectory UI display**: Dropped. PLAN.md lives in the research question folder, not `./plans/`. The built-in Plan Mode UI is nice-to-have but not load-bearing — we don't depend on it. (2026-03-17)
- **Adversarial review**: Separate skill (`/review`), opt-in. NOT auto-triggered. Blueprint Phase 4 simply asks: "Plan looks ready. Want adversarial review? (/review)" — user decides. (2026-03-17)
- **Blueprint vs PROGRESS.md relationship**: Deferred to workflow design phase. (2026-03-17)

#### Still Open (implementation details — implementer decides)

- Should `/blueprint` auto-detect existing PROGRESS.md and incorporate it? (Reasonable default: yes, read it for context but don't modify it)
- Blueprint + research question cross-linking: how tightly should PLAN.md reference README.md completion criteria? (Reasonable default: Success Criteria in PLAN.md should mirror the relevant README.md criteria)

### Full Structural Comparison: Our System vs ECC (updated 2026-03-17)

#### Top-Level Organization

```
OUR PROJECT                                 ECC
============                                ============
Heart-Conduction/                           everything-claude-code/
├── MASTER.md          (dashboard)          ├── CLAUDE.md        (project guidance, ~60 lines)
├── CLAUDE.md          (instructions)       ├── AGENTS.md        (agent orchestration rules)
├── Research/          (writing only)       ├── agents/          (25 agent definitions)
│   ├── Active/                             ├── commands/        (57 slash commands)
│   ├── Complete/                           ├── skills/          (108 skills)
│   ├── Backlog/                            ├── rules/           (coding rules by language)
│   └── Knowledge/                          │   ├── common/
├── Bidomain/Engine_V1/  (engine code)      │   ├── typescript/
├── Monodomain/Engine_V5.4/                 │   └── python/ ...
├── Monodomain/LBM_V1/                      ├── hooks/           (hooks.json config)
├── Optimizer/                              ├── scripts/         (Node.js runtime)
├── Surrogate/                              │   ├── hooks/       (hook implementations)
├── Builder/                                │   └── lib/         (shared utilities)
├── Engines/           (symlinks)           ├── mcp-configs/     (22 MCP servers)
├── Pipelines/         (symlinks)           ├── contexts/        (dev.md, review.md, research.md)
├── .claude/                                ├── manifests/       (install profiles)
│   ├── skills/        (9 → 16 after impl)  ├── examples/        (template CLAUDE.md files)
│   └── memory/        (MEMORY.md)          └── tests/           (997 tests)
├── MASTER_KNOWLEDGE_INDEX.md (NEW)
└── Research/Active/{q}/IDEALOG.md (NEW)
    Research/Active/{q}/PLAN.md    (NEW)
```

**Key difference**: We organize by *domain* (research questions, engines, pipelines). They organize by *tool type* (agents, commands, skills, rules). Ours is content-centric; theirs is infrastructure-centric.

#### What Gets Created: New Research Question

When we run `/research-new "boundary_conduction_speedup"`:

```
Research/Active/boundary_conduction_speedup/
├── README.md           Question, status, completion criteria, sub-questions,
│                       engine references, experiment table, literature table
├── KNOWLEDGE.md        Running synthesis: current understanding, key decisions,
│                       open questions, connections to engines/pipelines
├── literature/         Paper summaries (markdown, one per paper)
├── papers/             PDFs renamed to sanitized titles
├── code_examples/      Reference implementations from papers
└── results/            Simulation outputs, data, plots
```

**Also updates**: MASTER.md (adds row to Active Research table)

**Over its lifetime, a mature question accumulates**:
- README.md grows: literature table (8+ rows), experiment table, sub-question table, engine references
- KNOWLEDGE.md grows: findings, decisions, corrections, connections
- literature/ fills with paper summaries (each with "Connections to Our Models" section)
- papers/ fills with renamed PDFs
- Experiments live in `{Engine}/experiments/` (not here — Research/ = writing only)

ECC has **no equivalent**. They don't have a research question lifecycle. The closest analog is:
- A project's CLAUDE.md (static, no lifecycle)
- Session files in `~/.claude/sessions/` (ephemeral, per-session)
- `plans/` directory (per-task, no accumulation)

#### What Gets Created: New Engine Phase

When we start engine work (no skill yet — manual via PROGRESS.md):

```
{Engine}/
├── PROGRESS.md         Done/in-progress/next tasks, line numbers into improvement.md
├── IMPLEMENTATION.md   Phase specs, validation criteria tables
├── improvement.md      ABC interfaces, design decisions (1750+ lines)
├── README.md           Architecture overview
├── cardiac_sim/        Source code
├── tests/              Test suite
└── experiments/        Research experiments
    └── {experiment}/
        ├── EXPERIMENT.md   Backlinks to research question + MASTER.md
        ├── run.py          Experiment script
        └── outputs/        Results
```

ECC's equivalent:
```
plans/{project}-{objective}.md    Single plan file (blueprint)
```

**Key difference**: Our engine work is tracked across 3 persistent files (PROGRESS.md, IMPLEMENTATION.md, improvement.md) with a formal Orientation Protocol for recovery. ECC uses a single self-contained plan file where each step has its own context brief. Theirs is simpler; ours has more structure but more files to keep in sync.

#### Document Types Comparison

| Document | Our System | ECC | Notes |
|----------|-----------|-----|-------|
| **Project dashboard** | `MASTER.md` — research questions, engines, pipelines, all in one | `CLAUDE.md` — project guidance only, no status tracking | Ours is a living status board; theirs is static instructions |
| **Project instructions** | `CLAUDE.md` (~210 lines, single file) | `CLAUDE.md` (~60 lines) + `rules/common/*.md` (8 files) + `rules/{lang}/*.md` | They split instructions into modular rule files. We keep one file. |
| **Research synthesis** | `KNOWLEDGE.md` per question — high-res reference (facts, analysis, designs, comparisons) | No equivalent | Our most unique document. Nothing in ECC accumulates domain knowledge. |
| **Thinking trail** | `IDEALOG.md` per question (NEW) — low-res narrative of insights, failed approaches, session log | `~/.claude/sessions/*.tmp` (ephemeral) | Ours is version-controlled and accumulates. Theirs is disposable. |
| **Research status** | `README.md` per question — criteria, sub-questions, literature, experiments | No equivalent | |
| **Paper summaries** | `literature/{citation_key}.md` — structured with "Connections to Our Models" | No equivalent | |
| **Execution state** | `PROGRESS.md` per engine — done/in-progress/next, line references | `plans/*.md` with `- [x]` checkboxes | Similar function, different structure |
| **Implementation spec** | `IMPLEMENTATION.md` per engine — phase specs, validation tables | Within `plans/*.md` as exit criteria per step | Ours separates spec from tracking; theirs combines |
| **Design interfaces** | `improvement.md` per engine — ABCs, 1750+ lines | No equivalent (architecture decisions in CLAUDE.md or agent output) | |
| **Session state** | `IDEALOG.md` Session Log section (NEW) — written by `/save-session` | `~/.claude/sessions/YYYY-MM-DD-session.tmp` | Ours is scoped to research question, version-controlled. Theirs is global, ephemeral. |
| **Plans** | `PLAN.md` per question (NEW) — cold-start agent steps, created by `/blueprint` | `plans/{name}.md` — same concept | Direct equivalent. Ours is scoped to research question. |
| **Cross-session memory** | `MEMORY.md` — facts about user, project, feedback | `MEMORY.md` + session files + instinct system | Similar base, they layer more on top |
| **Agent definitions** | None (ad-hoc subagents) | `agents/*.md` — 25 agents with tool scoping, model selection | |
| **Skills** | `.claude/skills/*/SKILL.md` — 9 research + textbook skills | `skills/*/SKILL.md` — 108 skills (frameworks, patterns, workflows) | Same format, vastly different scale and focus |
| **Hooks config** | None (gap!) | `hooks/hooks.json` — 16 hooks across 6 event types | |
| **Experiment tracking** | `{Engine}/experiments/{name}/EXPERIMENT.md` — backlinks to research question | No equivalent | |
| **Textbook** | `Research/Knowledge/textbook/` — HTML source, style guide, audits, changelog | No equivalent | |

#### Lifecycle Comparison

**Our research lifecycle** (updated with new document architecture):
```
/research-new          → creates folder + README + KNOWLEDGE + IDEALOG
  ↓
/research-resume       → loads KNOWLEDGE + IDEALOG, presents briefing (incl. failed approaches)
  ↓
/reason                → interactive reasoning, writes insights to IDEALOG on transitions
  ↓
/blueprint             → generates PLAN.md from IDEALOG + codebase
  ↓
[work: experiments, papers, implementation]
  ↓
/research-update       → findings→KNOWLEDGE, ideas/failures→IDEALOG
/save-session          → session snapshot to IDEALOG Session Log
  ↓
/verify                → auto-detect engine, run tests
  ↓
/research-status       → staleness audit, coverage check
  ↓
/research-complete     → Active/ → Complete/, promote KNOWLEDGE to Knowledge/
```

**Documents touched at each step**:
- `/research-new`: Creates README.md, KNOWLEDGE.md, IDEALOG.md, empty dirs. Updates MASTER.md.
- `/research-resume`: Reads README, KNOWLEDGE, IDEALOG. Briefing shows: Current Direction, Next Step, What NOT to Retry.
- `/reason`: Reads KNOWLEDGE + IDEALOG + engine files. Writes to IDEALOG.md on transitions (decisions, rejections, topic shifts).
- `/blueprint`: Reads IDEALOG (settled approach) + codebase. Creates PLAN.md.
- `/research-update`: Routes by type — findings/analysis → KNOWLEDGE.md, ideas/failures/issues → IDEALOG.md.
- `/save-session`: (1) Session snapshot → IDEALOG Session Log, (2) Organize KNOWLEDGE.md, (3) Cross-reference IDEALOG ↔ KNOWLEDGE.
- `/research-complete`: Moves folder Active/ → Complete/. Copies KNOWLEDGE.md → Knowledge/. IDEALOG.md archived with question. Updates MASTER.md.

**ECC project lifecycle** (convention-driven, no formal lifecycle):
```
[start project]        → write CLAUDE.md, install rules
  ↓
/plan                  → planner agent outputs plan in chat
  ↓
/blueprint             → creates plans/{name}.md with steps
  ↓
[work: implement steps]
  ↓
/save-session          → creates ~/.claude/sessions/YYYY-MM-DD-session.tmp
  ↓
/resume-session        → loads session file, presents briefing
  ↓
/verify                → runs build + test + lint + diff
  ↓
/code-review           → agent reviews changes
```

**Documents touched at each step**:
- Project start: CLAUDE.md (manual), rules/*.md (installed)
- `/blueprint`: Creates plans/{name}.md
- `/save-session`: Creates session file (what worked, what failed, next step, file status)
- `/resume-session`: Reads session file, presents briefing
- No knowledge accumulation, no completion lifecycle, no promotion

#### What Each System Does Better

| Strength | Our System | ECC |
|----------|-----------|-----|
| **Knowledge accumulation** | KNOWLEDGE.md per question — findings synthesized over time, promoted to Knowledge/ on completion | Nothing. Session files are ephemeral snapshots, not accumulated wisdom. |
| **Research lifecycle** | 6 skills covering question inception through completion with formal criteria | No research lifecycle at all. |
| **Domain structure** | Research questions linked to engines linked to experiments with backlinks | Flat project structure, no domain modeling. |
| **Negative knowledge** | IDEALOG.md Failed Approaches section (NEW — was a gap) | Session files capture "What Did NOT Work" | Both now covered. Ours is version-controlled + scoped to question. |
| **Planning** | `/reason` interactive agent + `/blueprint` generates PLAN.md (NEW — was a gap) | `/reason` + `/blueprint` | Both now covered. Ours adds visual zoom + IDEALOG persistence. |
| **Session continuity** | `/research-resume` + IDEALOG.md Session Log (NEW) | `/save-session` + `/resume-session` | Both now covered. Ours integrates with research question lifecycle. |
| **Verification** | Manual pytest per engine (gap — `/verify` designed but not yet built) | `/verify` auto-detects and runs. |
| **Adversarial review** | `/audit` designed but not yet built | Opus subagent reviews plans. |
| **Hooks/automation** | PreCompact hook designed but not yet built | 16 hooks. |
| **Compaction recovery** | Orientation Protocol + PreCompact hook (designed, not built) | Strategic compact skill + PreCompact hook. |
| **Paper management** | Full PubMed pipeline with structured summaries | None. |
| **Experiment tracking** | EXPERIMENT.md with backlinks to research questions | None. |

### Patterns We're Evaluating

- **Dynamic context injection** — CLI aliases that load mode-specific system prompts (`claude-research`, `claude-engine`). Only useful if CLAUDE.md bloat becomes a problem.
- **Post-edit hooks** — auto-format with `ruff` after Python edits. May slow iteration.
- **Doc file guard** — hook to enforce Research/ = writing, Engines/ = code boundary.
- **Cost tracking hook** — log token usage per session. Nice-to-have, not a pain point yet.
- **SessionStart tmux/zellij layout** — Auto-launch a pane layout on session start via SessionStart hook. Research session layout: left = Claude Code, top-right = KNOWLEDGE.md (live preview), bottom-right = WHITEBOARD.md (diagrams/sketches). Uses `tmux new-session -d` (background, non-blocking). File watcher on right panes for live updates when Claude edits files.
- **`/draw` or `/sketch` skill** — Write ASCII diagrams, flow charts, architecture diagrams, data flow maps to WHITEBOARD.md. Live-updates in the right pane. "Draw the bidomain splitting architecture" → appears in your tmux pane. WHITEBOARD.md is ephemeral per-session (not version-controlled).
- **MASTER_KNOWLEDGE.md (textbook-oriented)** — A separate consolidated knowledge document (distinct from MASTER_KNOWLEDGE_INDEX.md) that synthesizes findings across all research questions into textbook-ready prose. Would feed into the textbook pipeline. Deferred — design when textbook work resumes.

### Patterns We're Skipping

- Continuous learning / instinct system — over-engineered for solo research.
- Multi-instance parallelization — not needed for single-researcher workflow.
- 108 framework-specific skills — we only need Python/CUDA.
- Extensive MCP server configs — our PubMed + Gmail MCPs are sufficient.
- Security review agent — simulation code, not a web app.
- Formalized agent library — our skill system handles delegation already.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Adopt hooks selectively, not the full ECC ecosystem | We're a domain-specific research project, not a general dev team. Cherry-pick what helps. |
| Keep CLAUDE.md as single file for now | Not yet at the pain point where modular rules are needed (~200 lines). Revisit if it exceeds 400 lines. |
| Don't restructure to match their directory layout | Our Research/Engines/Pipelines separation is purpose-built for cardiac EP work. |
| ~~Four-document~~ → Three-document architecture: KNOWLEDGE + IDEALOG + PLAN | Briefly explored four docs (adding WORKLOG). Dropped WORKLOG — failed approaches and session snapshots are part of the thinking trail and belong in IDEALOG. KNOWLEDGE keeps high resolution (facts, analysis, designs, comparisons). IDEALOG is low resolution (condensed narrative). PLAN is for cold-start agent execution. (finalized 2026-03-17) |
| KNOWLEDGE.md keeps high resolution | Originally considered stripping KNOWLEDGE down to bare facts. That loses too much — detailed comparison tables, design rationale, template formats are all reference material you'd look up later. KNOWLEDGE = everything you'd reference. IDEALOG = the story of how you got there. (decided 2026-03-17) |
| `/reason` writes on transitions, not every exchange | Accumulate in context, batch-write to IDEALOG.md on: decisions settled, approaches rejected, topic shifts, explicit "write that down", PreCompact. ~3-4 writes per 30min session. Minimizes workflow disruption. (decided 2026-03-17) |
| `/reason` uses big→middle→small zoom with organic flow | Always opens with visual big-picture map (scannable in 5 seconds). User can drill down sequentially OR jump organically — agent follows. No forcing back into hierarchy. Agent tracks settled vs open internally regardless of exploration order. (decided 2026-03-17) |
| `/reason` is one agent, not two modes | Structured drill-down and organic jumping are not separate modes — they're natural behaviors of the same agent. Start structured (big picture), then follow the user's lead. (decided 2026-03-17) |
| `/reason` can invoke `/blueprint` | When user says "let's build this" during a `/reason` session, `/reason` triggers `/blueprint` to generate PLAN.md from the settled approach. Seamless transition from thinking to execution planning. (decided 2026-03-17) |
| `/save-session` is a comprehensive cleanup agent | Five jobs: (1) session snapshot → IDEALOG, (2) full editorial pass on KNOWLEDGE.md, (3) cross-reference IDEALOG ↔ KNOWLEDGE, (4) condense verbose IDEALOG, (5) update MASTER_KNOWLEDGE_INDEX.md with cross-question findings and connections. Can take as long as needed. (decided 2026-03-17) |
| MASTER_KNOWLEDGE_INDEX.md as index book, not summary | Project-root index that points to where knowledge lives and maps connections between questions. Does NOT duplicate findings — one-liner per question + cross-reference links. Research statement lives here only. Lightweight to maintain — `/save-session` Job 5 just updates one-liners and cross-references. Avoids drift between duplicate copies. (decided 2026-03-17) |
| PLAN.md is machine-targeted, not human-targeted | ECC's blueprint is for human developers. Our PLAN.md is for cold-start Claude agents. Every step is self-contained with full context brief, known failures, per-step reasoning ("Why"), model/tool assignments, and domain-specific conventions. A fresh `claude -p` call can execute any step without reading prior steps or IDEALOG. (decided 2026-03-17) |
| Borrow ECC's vibe coding patterns for PLAN.md | Complexity tiers (trivial→large), de-sloppify cleanup pass, known failures section, author-bias elimination for /audit, architecture changes overview. Adapted from ECC's autonomous-loops and agentic-engineering skills. (decided 2026-03-17) |
| PLAN.md steps include full scaffolding for the agent | Each step has: Context Brief, Implementation Spec (interfaces/signatures), Pseudocode (algorithmic sketch), Test Spec (exact tests with setup/expected/tolerances), and Checklist (sequential to-do). This is more than a task list — it's everything the agent needs to write code without guessing. (decided 2026-03-17) |
| PLAN.md organized by phases, not flat steps | Phases are independently deliverable units with their own context, verification, exit criteria, cleanup, and commit point. Steps live within phases. Agent implements phase-by-phase: complete Phase 1 → verify → commit → Phase 2. Matches our existing engine workflow (Bidomain 6 phases, V5.4 9 phases). Phase context is shared across steps within the phase to avoid repetition. (decided 2026-03-17) |
| `/audit` as standalone opt-in skill (renamed from `/review`) | Adversarial Opus subagent. `/blueprint` asks "Want adversarial audit?" — user decides. Can also be used independently on any document. (decided 2026-03-17) |
| `/research-update` routes to correct document by type | `finding` (facts, analysis, designs) → KNOWLEDGE.md. `idea`, `failure`, `issue`, `next-step` → IDEALOG.md. (decided 2026-03-17) |

### Rollout Plan (decided 2026-03-17)

Implementation proceeds in 3 phases + 1 parallel track. Phase 0 is the biggest — it creates the foundation everything else depends on.

**Phase 0: Foundation** — Templates, retroactive IDEALOGs, KNOWLEDGE migration, CLAUDE.md rewrite.
- 0.1: Finalize IDEALOG.md and PLAN.md templates as standalone references (done — already in this document)
- 0.2: Backtrace 16 session `.jsonl` chat logs → create IDEALOG.md for all 6 active research questions. Map conversations to research questions by files touched, extract thinking trail (decisions, failed approaches, "oh wait" moments, branching insights). Each IDEALOG gets: Current Direction, Next Step, Thread entries (dated), Failed Approaches, Session Log.
- 0.3: Audit & split existing KNOWLEDGE.md files. For each of 6 active questions: read current KNOWLEDGE.md, identify process/narrative content that belongs in IDEALOG, move it (supplement what backtrace found), reorganize remaining KNOWLEDGE.md as clean high-res reference. Biggest: research_environment_optimization (~900 lines of mixed analysis + process).
- 0.4: Update CLAUDE.md with three-document architecture, new skill table, updated workflow instructions.

**Phase 1: Rewire Existing Skills** — All 4 must ship together to avoid inconsistent state.
- 1.1: `/research-new` — add IDEALOG.md creation
- 1.2: `/research-resume` — read IDEALOG, show failed approaches + direction + next step
- 1.3: `/research-update` — route by type (findings→KNOWLEDGE, ideas/failures→IDEALOG)
- 1.4: `/research-complete` — archive IDEALOG.md with question in Complete/

**Phase 2: Core New Skills** — The planning pipeline + cleanup agent + quality gate.
- 2.1: `/reason` — interactive reasoning buddy
- 2.2: `/blueprint` — autonomous PLAN.md generator
- 2.3: `/save-session` — comprehensive cleanup agent
- 2.4: `/audit` — adversarial review

**Phase 2b: Independent Skills** (parallel with Phase 2, no dependencies)
- 2b.1: `/verify` — auto-detect engine, run tests
- 2b.2: `/build-fix` — systematic error resolution
- 2b.3: `/strategic-compact` — compaction decision table

**Phase 3: Hook + Final Polish**
- 3.1: PreCompact hook (with emergency dump for active `/reason` sessions)
- 3.2: Final CLAUDE.md update with all new skills registered

## Open Questions

- PLAN.md for engine-only work (not tied to a research question): where does it live? Deferred — hasn't come up yet.
- How does PLAN.md relate to existing PROGRESS.md? Should step completion auto-update PROGRESS.md?
- `/reason` compaction recovery: how does an interactive `/reason` session resume after compaction? IDEALOG.md has the written insights, but the conversational state is lost. Needs design.
- Context files for different modes? Deferred until CLAUDE.md exceeds ~400 lines.

## Connections
- **Engines**: All (cross-cutting workflow concern)
- **Related research**: None directly; this is meta-research about the research process itself
- **Pipelines**: All (same workflow patterns apply)
