---
name: blueprint
description: Generate a machine-targeted PLAN.md from IDEALOG.md settled decisions and codebase analysis. Each step is self-contained for cold-start agent execution. Phases are independently deliverable with verification and commit points.
argument-hint: "[objective — what to implement]"
---

# Blueprint — PLAN.md Generator

Autonomous pipeline. Reads IDEALOG.md (settled approach, known failures) + codebase (affected files, existing tests). Outputs a machine-targeted PLAN.md with self-contained steps for cold-start agent execution.

Input: $ARGUMENTS

---

## Precondition: Verify IDEALOG.md

Find the research question folder for this objective. Check `Research/Active/` for a matching question.

Read `Research/Active/{question}/IDEALOG.md`. Verify:
- The **Thread** section has at least one settled decision (not just stubs)
- There is a **Current Direction** describing the settled approach

If IDEALOG.md is missing or has no settled decisions, stop and tell the user:
> "No settled approach found in IDEALOG.md. Use `/reason` first to explore the design space, then run `/blueprint` when you have a settled direction."

If IDEALOG.md has a Current Direction but the Thread is sparse, proceed with a warning — the plan will be based primarily on codebase analysis.

---

## Step 1: Gather Input

Read these sources in parallel:

### 1a. IDEALOG.md (required)
- **Current Direction** — the settled approach
- **Thread** — settled decisions with rationale
- **Failed Approaches** — approaches already tried and failed (agent MUST NOT retry these)
- **Next Step** — what the user expects to happen next

### 1b. KNOWLEDGE.md (if exists)
- `Research/Active/{question}/KNOWLEDGE.md` — reference findings, analysis, designs
- Extract anything load-bearing for implementation: algorithms, parameter values, validated results

### 1c. Codebase analysis
Read the files that will be affected. Determine:
- Existing file structure and interfaces
- Current test suite and how tests are run
- IMPLEMENTATION.md sections (if engine work — for validation criteria)
- PROGRESS.md (if exists — for current state, do not modify)
- Related code in other engines (to avoid duplication)

### 1d. README.md
- `Research/Active/{question}/README.md` — completion criteria (PLAN.md Success Criteria should mirror these)

---

## Step 2: Decompose into Phases and Steps

Break the objective into **phases** (independently deliverable, own verification + commit point) and **steps** within phases (sequential, each completable in one session). Phase count is variable — 1 phase with 2 steps for simple work, up to 5 phases with 4-6 steps each for major features. Dependencies flow forward: Phase N never depends on Phase N+1.

Each step must be self-contained: a cold-start agent can execute it with only its own sections + Phase Context.

Assign each step a complexity tier:

| Tier | Pipeline | Model |
|------|----------|-------|
| **trivial** | implement, verify | haiku/sonnet |
| **small** | implement, verify, self-review | sonnet |
| **medium** | read context, implement, verify, cleanup | sonnet |
| **large** | read context, implement, verify, cleanup, /audit | opus |

---

## Step 3: Write PLAN.md

Write the plan to `Research/Active/{question}/PLAN.md`.

### Header

```markdown
# PLAN: {title}

Created: {date}
Engine(s): {Bidomain V1 | V5.4 | LBM V1 | Optimizer | All | None}
Research question: [{question_name}](README.md)
Source: [IDEALOG.md](IDEALOG.md) — {which thread entry motivated this plan}

## Objective
{What we're implementing and why — 2-3 sentences}

## Success Criteria
- [ ] {measurable criterion — mirrors README.md completion criteria}
- [ ] All existing tests pass (no regressions)

## Architecture Changes
- NEW: `path/to/new_file.py` — {purpose}
- MOD: `path/to/existing.py:{lines}` — {what changes and why}

## Known Failures (from IDEALOG)
- {approach} — failed because: {exact reason}
```

### Per-phase structure

```markdown
## Phase N: {title}

**Goal**: {what this phase achieves — independently deliverable}
**Tier**: {small | medium | large}
**Estimated scope**: {description}

### Phase Context
{Everything the agent needs to work on ANY step in this phase.
Conventions, current code state, what NOT to do.
Steps inherit this — they don't repeat it.}
```

### Per-step structure (the key value-add)

Every step must include ALL sections below. A cold-start agent executes with zero prior context.

```markdown
### Step N.M: {title}
**Model**: sonnet | opus

#### Read First
- `{path}:{lines}` — {what to look for}

#### Why
{Reasoning — helps agent make judgment calls on edge cases}

#### Implementation Spec
**Files to create:** `{path}` — {purpose}
**Files to modify:** `{path}:{lines}` — {what changes}
**Interfaces / Signatures:** {key signatures with types}

#### Pseudocode
{Algorithmic sketch — enough to implement unambiguously}

#### Test Spec
- `{test_file}::{test_name}` — Setup: {setup}. Expected: {values, tolerances}

#### Checklist
- [ ] {sequential to-do items}

#### Verify
{Exact shell commands}

#### Exit Criteria
- [ ] {what must be true before moving on}

#### Risk
{What could go wrong} — mitigation: {approach}
```

### Per-phase closing

```markdown
### Phase N Verification
{Exact shell commands — pytest, manual checks}

### Phase N Exit Criteria
- [ ] All new tests pass
- [ ] All existing tests pass (no regressions)
- [ ] {phase-specific criteria}

### Phase N Cleanup
{De-sloppify checklist — see domain-specific items below}

**-> Commit point: git commit after Phase N passes**
```

### Plan closing

Include `## Final Cleanup` (cross-phase de-sloppify) and `## Mutation Log` (initially empty — populated during execution with `**MUTATED {date}**: Step X.Y {SKIPPED|SPLIT|INSERTED} — {reason}`).

### Domain-specific cleanup checklist

Include in every Phase Cleanup and Final Cleanup:
- float64 consistency — no float32 leaks (all tensors `torch.float64`)
- V5.3 not modified — `Monodomain/Engine_V5.3/` is read-only
- EXPERIMENT.md backlinks exist for new experiments in engine folders
- No code duplication across engines — shared logic belongs in `cardiac_core/`

---

## Step 4: Offer Audit

After writing PLAN.md, report:
- Number of phases, total steps, estimated complexity
- Any gaps or assumptions made due to sparse IDEALOG content

Then ask:
> "Want adversarial audit? (`/audit`)"

---

## Rules

- **PLAN.md is for the machine, not the human.** The human's thinking is in IDEALOG.md. PLAN.md is a construction manual that a cold-start agent can execute without conversation history.
- **Never duplicate IMPLEMENTATION.md content.** Link to it (`see IMPLEMENTATION.md Phase 3 validation table`). Single source of truth.
- **Include Known Failures.** Pull every failed approach from IDEALOG.md. The executing agent must not retry dead ends.
- **Every step gets a "Why" section.** Agents need reasoning to handle edge cases, not just instructions.
- **Phases are independently deliverable.** Each phase has verification, cleanup, and a commit point. If the session dies mid-plan, completed phases are safe.
- **Use `- [ ]` checkboxes, not TaskCreate.** Blueprint checkboxes replace task tracking for planned work.
- **Plan mutation is expected.** Steps can be marked SKIPPED, SPLIT, or new steps INSERTED during execution. Add `**MUTATED {date}**: {reason}` annotations.
- **Read PROGRESS.md if it exists** for current state context, but do not modify it. Blueprint and PROGRESS.md are complementary.
- **Success Criteria mirror README.md completion criteria.** The plan is done when the research question's criteria are met.
- **Conda environment is `heart-conduction`.** All verify commands use `conda run -n heart-conduction`.
