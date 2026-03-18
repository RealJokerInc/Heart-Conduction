# PLAN: Architecture Refinements — KNOWLEDGE restructure, README Future Work, skill updates, /blueprint iterative mode

Created: 2026-03-18
Engine(s): None (cross-cutting workflow)
Research question: [research_environment_optimization](README.md)
Source: [IDEALOG.md](IDEALOG.md) — "2026-03-18: Comprehensive session — architecture refinements"

## Objective
Four interconnected changes: (1) Restructure the research_environment_optimization KNOWLEDGE.md into per-topic sections (the other 5 are small and already organized), (2) Add "Future Work" section to README.md for deferred items, (3) Update skill dependency chain (reason, research-new, research-resume, research-complete, save-session), (4) Make /blueprint iterative (update existing PLAN.md instead of always creating fresh).

## Success Criteria
- [ ] research_environment_optimization KNOWLEDGE.md restructured with per-topic sections (Findings/Design/Reference/Decisions)
- [ ] All 6 active README.md files have a Future Work section
- [ ] `/save-session` Job 2 has per-topic structure guidance
- [ ] `/reason` Step 0 reads README.md Future Work when no topic given
- [ ] `/research-new` template includes Future Work section
- [ ] `/research-resume` briefing shows Future Work items
- [ ] `/research-complete` warns about unfinished Future Work items
- [ ] `/blueprint` detects existing PLAN.md and switches to update mode

## Architecture Changes
- MOD: `Research/Active/research_environment_optimization/KNOWLEDGE.md` — full restructure into per-topic sections
- MOD: `Research/Active/*/README.md` — add Future Work section to all 6
- MOD: `.claude/skills/save-session/SKILL.md` — add per-topic structure guidance to Job 2
- MOD: `.claude/skills/reason/SKILL.md` — Step 0 reads README.md Future Work
- MOD: `.claude/skills/research-new/SKILL.md` — add Future Work to README template
- MOD: `.claude/skills/research-resume/SKILL.md` — show Future Work in briefing
- MOD: `.claude/skills/research-complete/SKILL.md` — warn about unfinished Future Work
- MOD: `.claude/skills/blueprint/SKILL.md` — add iterative mode (detect existing PLAN.md)

## Known Failures (from IDEALOG)
- Previous `/save-session` runs did surface polish but didn't restructure — Job 2 lacks structural guidance
- NOTEBOOK.md routing friction — the IDEALOG/NOTEBOOK split required judgment calls that faded from context. Merged into single IDEALOG. Don't re-introduce document routing complexity.
- Skill guardrails fade in long conversations — instructions in skill files don't persist as global rules

---

## Phase 1: Restructure research_environment_optimization KNOWLEDGE.md

**Goal**: Transform the 1109-line flat dump into a clean per-topic reference document. Remove embedded templates, condense historical analysis, organize by topic with optional Findings/Design/Reference/Decisions subsections.
**Tier**: large

### Phase Context
The KNOWLEDGE.md has 5 content categories: Current Understanding (6 lines, keep), ECC Analysis (232 lines, condense), Document Architecture (142 lines, keep design, remove templates), Skill Designs (568 lines, condense — full detail lives in skill files), Workspace/Rollout/Decisions/Open Questions (116 lines, reorganize). Target: per-topic structure. Templates, evolution history, and rollout plan move to IDEALOG or are removed (already captured there).

### Step 1.1: Rewrite KNOWLEDGE.md
**Model**: opus

#### Read First
- `Research/Active/research_environment_optimization/KNOWLEDGE.md` — read in chunks (0-250, 250-500, 500-750, 750-1000, 1000-end)
- `Research/Active/research_environment_optimization/IDEALOG.md` — verify evolution history is already captured there

#### Why
The file is unusable as reference — 400 lines of templates, sections at different resolutions, no topic-based organization. Must be a document you can look up cold.

#### Implementation Spec
**Files to modify:** `Research/Active/research_environment_optimization/KNOWLEDGE.md` — full rewrite via Write tool

New structure:
```
## Summary (5-7 sentences)

## ECC Analysis
  ### Findings (condense 232→~50 lines: key tables, core philosophy)
  ### Reference (skill gap table, session persistence comparison — keep)
  ### Decisions (adopted/skipped/deferred — clean table, no evolution)

## Document Architecture
  ### Findings (KNOWLEDGE overload problem, negative knowledge gap)
  ### Design (KNOWLEDGE + IDEALOG + PLAN + WHITEBOARD, interaction diagram)
  ### Decisions (per-topic structure, NOTEBOOK merged, etc.)

## Skill Pipeline
  ### Design (18 skills by category, /reason→/blueprint→/audit pipeline)
  ### Reference (ECC planner comparison, complexity tiers, phasing philosophy)
  ### Decisions (key skill decisions — clean table)

## Workspace Integration
  ### Findings (tmux vs zellij, glow rendering, shell loop fix)
  ### Design (50-25-25 layout, dynamic width, md5sum change detection)
  ### Decisions (tmux only, /reason owns panes)

## Open Questions

## Connections
```

Rules:
- No embedded templates (IDEALOG, PLAN.md, MASTER_KNOWLEDGE_INDEX templates)
- No evolution history (strikethroughs, "originally planned X then Y")
- No rollout plan (historical, already executed)
- Deferred patterns move to README.md Future Work (Phase 2)
- Skill designs condensed to key design points — full detail is in skill files
- NOTEBOOK.md design section marked as historical one-liner

#### Checklist
- [ ] Read full file in chunks
- [ ] Write Summary
- [ ] Write ECC Analysis (Findings/Reference/Decisions)
- [ ] Write Document Architecture (Findings/Design/Decisions)
- [ ] Write Skill Pipeline (Design/Reference/Decisions)
- [ ] Write Workspace Integration (Findings/Design/Decisions)
- [ ] Write Open Questions + Connections
- [ ] Verify no templates remain
- [ ] Verify evolution history in IDEALOG (not lost)

#### Verify
```bash
wc -l Research/Active/research_environment_optimization/KNOWLEDGE.md
grep -c "^## " Research/Active/research_environment_optimization/KNOWLEDGE.md
grep -c "## Current Direction\|## Thread\|## Session Log\|## Failed Approaches" Research/Active/research_environment_optimization/KNOWLEDGE.md  # should be 0
```

#### Exit Criteria
- [ ] Per-topic structure with Findings/Design/Reference/Decisions subsections
- [ ] No embedded templates
- [ ] All reference material preserved (condensed, not deleted)

#### Risk
Information loss during condensation. Mitigation: verify all removed content exists elsewhere (templates in skill files, evolution in IDEALOG, rollout plan historical). Make git commit after this step for rollback safety.

---

### Phase 1 Verification
```bash
wc -l Research/Active/research_environment_optimization/KNOWLEDGE.md
grep -c "^## \|^### " Research/Active/research_environment_optimization/KNOWLEDGE.md
```

### Phase 1 Exit Criteria
- [ ] KNOWLEDGE.md restructured with per-topic sections
- [ ] glow renders cleanly in tmux pane (visual check)

### Phase 1 Cleanup
- Visual check of tmux KNOWLEDGE.md pane rendering

**→ Commit point: "Restructure KNOWLEDGE.md: per-topic sections (Findings/Design/Reference/Decisions)"**

---

## Phase 2: README.md Future Work + Deferred Patterns

**Goal**: Add "Future Work" section to all 6 active README.md files. Move deferred patterns from KNOWLEDGE.md to research_environment_optimization's README.md Future Work.
**Tier**: small

### Step 2.1: Add Future Work section to all 6 active README.md files
**Model**: sonnet

#### Read First
- `Research/Active/*/README.md` — check if any already have a Future Work section

#### Why
Deferred ideas need a persistent, per-question home. README.md is never purged and is read by `/reason`, `/research-resume`, and `/research-complete`.

#### Implementation Spec
**Files to modify:** All 6 `Research/Active/*/README.md`

For research_environment_optimization, populate with the deferred patterns:
```markdown
## Future Work
- SessionStart tmux layout hook (auto-setup on session start)
- `/draw` skill for WHITEBOARD.md visualization
- Cost tracking hook (token usage per session)
- Post-edit ruff formatting hook
- MASTER_KNOWLEDGE.md (textbook-oriented consolidated knowledge)
- Memory.md role in document architecture (always-loaded cheat sheet?)
- Skill guardrail persistence (rules that survive beyond skill invocation)
- IDEALOG.md format improvement (richer entries without losing scannability)
- WHITEBOARD.md persistent sections (Current Focus + To-Do above divider, variable Scratch below)
```

For the other 5 questions, add an empty section:
```markdown
## Future Work
{No deferred items yet.}
```

#### Checklist
- [ ] Add Future Work to research_environment_optimization README.md (populated)
- [ ] Add empty Future Work to boundary_conduction_speedup README.md
- [ ] Add empty Future Work to ionic_model_optimization README.md
- [ ] Add empty Future Work to engine_consolidation README.md
- [ ] Add empty Future Work to geometry_induced_pacemaking README.md
- [ ] Add empty Future Work to mature_hipsc_cm_models README.md

#### Verify
```bash
for d in Research/Active/*/; do echo "$d: $(grep -c 'Future Work' ${d}README.md)"; done
```

#### Exit Criteria
- [ ] All 6 README.md files have a Future Work section

#### Risk
None — additive section to existing files.

---

### Phase 2 Verification
```bash
for d in Research/Active/*/; do grep -l "Future Work" ${d}README.md; done | wc -l  # should be 6
```

### Phase 2 Exit Criteria
- [ ] All 6 README.md files have Future Work section
- [ ] Deferred patterns moved from KNOWLEDGE.md to README.md

### Phase 2 Cleanup
None.

**→ Commit point: "Add Future Work section to all 6 active README.md files"**

---

## Phase 3: Skill Dependency Chain Updates

**Goal**: Update 5 skills to support Future Work section and per-topic KNOWLEDGE.md structure.
**Tier**: medium

### Phase Context
5 skills need updates. All are markdown edits to skill files. They must all ship together for consistency.

### Step 3.1: Update `/save-session` Job 2 with per-topic structure guidance
**Model**: sonnet

#### Read First
- `.claude/skills/save-session/SKILL.md:54-66` — current Job 2

#### Why
Job 2 does surface polish because it has no target structure. Adding per-topic guidance ensures it maintains the Findings/Design/Reference/Decisions organization.

#### Implementation Spec
**Files to modify:** `.claude/skills/save-session/SKILL.md`

Add after the current Job 2 bullet list:
```markdown
**Target structure for KNOWLEDGE.md** — maintain on every editorial pass:

Each topic section has optional-but-ordered subsections:
1. **Findings** — what was discovered or validated
2. **Design** — what was built based on findings
3. **Reference** — comparison tables, technical detail for lookup
4. **Decisions** — final choices with rationale (no evolution history)

Rules:
- New findings slot into existing topic's Findings subsection
- If no existing topic fits, create a new topic section
- No templates — those live in skill files
- No evolution history — that's IDEALOG's job
- Condense, don't accumulate — tighten on each pass
```

#### Checklist
- [ ] Add per-topic structure guidance to Job 2
- [ ] Add maintenance rules

---

### Step 3.2: Update `/reason` Step 0 to read README.md Future Work
**Model**: sonnet

#### Read First
- `.claude/skills/reason/SKILL.md:15-22` — current Step 0

#### Why
When `/reason` has no topic, it should show Future Work items as potential things to work on, not just IDEALOG Current Direction.

#### Implementation Spec
**Files to modify:** `.claude/skills/reason/SKILL.md`

In Step 0, after the auto-resume logic, add:
```markdown
After loading IDEALOG.md, also read `Research/Active/{question}/README.md` for the **Future Work** section. If it has items, mention them as available topics:
> "Continuing {question}. Current direction: {from IDEALOG}. Also pending in Future Work: {list items}."
```

#### Checklist
- [ ] Add README.md Future Work read to Step 0
- [ ] Include Future Work items in the resume message

---

### Step 3.3: Update `/research-new` template to include Future Work
**Model**: sonnet

#### Read First
- `.claude/skills/research-new/SKILL.md` — find README template section

#### Why
New research questions should have the Future Work section from the start.

#### Implementation Spec
**Files to modify:** `.claude/skills/research-new/SKILL.md`

Add to the README.md template (after Literature table):
```markdown
## Future Work
{No deferred items yet.}
```

#### Checklist
- [ ] Add Future Work to README template in research-new

---

### Step 3.4: Update `/research-resume` to show Future Work in briefing
**Model**: sonnet

#### Read First
- `.claude/skills/research-resume/SKILL.md` — find briefing format

#### Why
When resuming a question, you should see what deferred items exist — they might be what you want to work on.

#### Implementation Spec
**Files to modify:** `.claude/skills/research-resume/SKILL.md`

Add to the session brief format (after "What NOT to retry"):
```
Future work:
  - {items from README.md Future Work section, if any}
```

#### Checklist
- [ ] Add Future Work to briefing format

---

### Step 3.5: Update `/research-complete` to warn about unfinished Future Work
**Model**: sonnet

#### Read First
- `.claude/skills/research-complete/SKILL.md` — find completion verification section

#### Why
Before completing a question, unfinished Future Work items should be flagged — they might need to be moved to a new question or explicitly dropped.

#### Implementation Spec
**Files to modify:** `.claude/skills/research-complete/SKILL.md`

Add a check step before the move:
```markdown
Check README.md Future Work section. If items remain:
- List them and ask the user: move to a different question, create a new question, or drop?
- Do not complete the question with unresolved Future Work items without explicit user approval.
```

#### Checklist
- [ ] Add Future Work check before completion

---

### Phase 3 Verification
```bash
grep -c "Future Work" .claude/skills/reason/SKILL.md .claude/skills/research-new/SKILL.md .claude/skills/research-resume/SKILL.md .claude/skills/research-complete/SKILL.md
grep -c "per-topic\|Findings.*Design.*Reference.*Decisions" .claude/skills/save-session/SKILL.md
```

### Phase 3 Exit Criteria
- [ ] All 5 skills updated
- [ ] Future Work integrated into the skill dependency chain

### Phase 3 Cleanup
- Verify no stale references in updated skills

**→ Commit point: "Update 5 skills: Future Work dependency chain + /save-session per-topic structure"**

---

## Phase 4: Create `/blueprint-revise` Skill

**Goal**: Create a separate skill for updating an existing PLAN.md instead of bloating `/blueprint` with iterative mode. Keeps both skills focused and under 210 lines.
**Tier**: small

### Phase Context
`/blueprint` is already 210 lines. Adding iterative mode would push it to 250+, hitting the zone where skill instructions fade from context in long conversations. Splitting into two skills: `/blueprint` creates, `/blueprint-revise` updates.

### Step 4.1: Create `/blueprint-revise` skill
**Model**: sonnet

#### Read First
- `.claude/skills/blueprint/SKILL.md` — understand the PLAN.md format it generates

#### Why
Plans get revised through multiple `/reason` → `/blueprint` → `/audit` cycles. A separate skill avoids bloating `/blueprint` and keeps each skill focused. Clear mental model: "create" vs "revise."

#### Implementation Spec
**Files to create:** `.claude/skills/blueprint-revise/SKILL.md` (~80-100 lines)

```yaml
---
name: blueprint-revise
description: Update an existing PLAN.md — preserve completed steps, incorporate new IDEALOG decisions, log mutations. Use after /audit findings or mid-implementation course corrections.
argument-hint: "[what changed — reason for revision]"
---
```

Body structure:
1. Read existing PLAN.md in full
2. Read IDEALOG.md for decisions/findings made AFTER the PLAN.md was created (compare dates)
3. Identify completed steps (`[x]`) — preserve unchanged
4. Identify open steps that need modification based on new IDEALOG content
5. Update open steps, add new steps if needed, mark obsolete steps as SKIPPED
6. Log all changes in Mutation Log: `**MUTATED {date}**: Step X.Y {MODIFIED|ADDED|SKIPPED} — {reason}`
7. Do NOT modify completed steps unless user explicitly requests it
8. Present summary of changes and STOP — wait for user approval

Rules:
- Never modify completed steps without explicit request
- Always log mutations
- STOP after revision — same approval gate as /blueprint

#### Checklist
- [ ] Create `.claude/skills/blueprint-revise/SKILL.md`
- [ ] Write YAML frontmatter
- [ ] Write revision logic (read existing, diff against IDEALOG, modify open steps)
- [ ] Write mutation logging format
- [ ] Write approval gate (STOP after revision)
- [ ] Verify skill file is under 100 lines

#### Verify
```bash
test -f .claude/skills/blueprint-revise/SKILL.md && echo "EXISTS"
wc -l .claude/skills/blueprint-revise/SKILL.md
```

#### Exit Criteria
- [ ] `/blueprint-revise` skill file created
- [ ] Under 100 lines
- [ ] Has approval gate (STOP after revision)

#### Risk
None — new file, no existing functionality affected.

---

### Step 4.2: Update CLAUDE.md skill table
**Model**: sonnet

#### Read First
- `CLAUDE.md` — find Planning & reasoning section of skill table

#### Implementation Spec
**Files to modify:** `CLAUDE.md`

Add `/blueprint-revise` to the skill table after `/blueprint`:
```markdown
| `/blueprint-revise` | Update existing PLAN.md — preserve completed steps, incorporate new IDEALOG, log mutations |
```

#### Checklist
- [ ] Add `/blueprint-revise` to CLAUDE.md skill table

---

### Phase 4 Verification
```bash
test -f .claude/skills/blueprint-revise/SKILL.md && echo "SKILL EXISTS"
grep "blueprint-revise" CLAUDE.md && echo "IN SKILL TABLE"
```

### Phase 4 Exit Criteria
- [ ] `/blueprint-revise` skill created
- [ ] CLAUDE.md skill table updated

### Phase 4 Cleanup
None.

**→ Commit point: "Create /blueprint-revise skill — update existing PLAN.md without bloating /blueprint"**

---

## Final Cleanup
- Run `/save-session` to test Job 2 with new per-topic structure guidance
- Verify all 18 skills load correctly
- Visual check of tmux KNOWLEDGE.md pane after restructure

## Mutation Log
{To be filled during execution}
