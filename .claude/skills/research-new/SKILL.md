---
name: research-new
description: Scaffold a new research question (or sub-question) with the standard folder structure, README, KNOWLEDGE.md, and IDEALOG.md. Use when starting a new investigation.
argument-hint: "[question name] or [parent/sub-question name]"
---

# New Research Question

Scaffold a new research question or sub-question with the standard structure.

Input: $ARGUMENTS

### Project Structure Reference
```
MASTER.md                              — project dashboard (update when adding questions)
Research/Active/{question}/            — active research (README.md, KNOWLEDGE.md, literature/, papers/, figures/)
Research/Complete/{question}/           — answered questions
Research/Backlog/{question}/            — parked questions
Research/Knowledge/                     — promoted knowledge files from Complete questions
{Engine}/experiments/{experiment}/      — code + outputs (EXPERIMENT.md with backlinks, run.py, outputs/)
Engines/cross_engine/{question}/        — cross-engine experiments
```
Research/ = writing only (no .py). Engines/ = code only.

---

## Step 1: Parse Input

Determine whether this is a **top-level question** or a **sub-question**:

- `"anisotropic boundaries"` → top-level in `Research/Active/`
- `"boundary_conduction_speedup/anisotropic_boundaries"` → sub-question under existing parent

If the input looks like a sub-question (contains `/`), verify the parent exists in `Research/Active/`. If it doesn't, ask the user whether to create the parent first.

## Step 2: Name the Folder

Convert the question name to a folder name:
- Lowercase, underscores, no special characters
- Descriptive but concise (2-4 words)
- Examples: `boundary_conduction_speedup`, `hipsc_cm_ionic_models`, `anisotropic_boundaries`

## Step 3: Ask Clarifying Questions

Before scaffolding, gather:

1. **The question** — one sentence, as specific as possible
2. **Why it matters** — what depends on the answer
3. **Which engines** — which simulation engines does this touch? (Bidomain V1, Monodomain V5.4, LBM V1, none yet)
4. **What "done" looks like** — concrete completion criteria
5. **Known sub-questions** — are there obvious sub-topics to create now?

If the user already provided enough context (e.g., from a conversation), infer the answers and confirm rather than asking.

## Step 4: Create Folder Structure

### For a top-level question:

```
Research/Active/{question_name}/
├── README.md
├── KNOWLEDGE.md
├── IDEALOG.md
├── literature/          # Paper summaries go here
├── papers/              # PDFs go here
├── code_examples/       # Reference implementations (if any)
└── results/             # Simulation outputs, data, plots
```

### For a sub-question:

```
Research/Active/{parent}/{sub_question_name}/
├── README.md
└── results/
```

Sub-questions are lightweight. They share the parent's `literature/`, `papers/`, and `code_examples/` unless they grow large enough to need their own.

## Step 5: Write README.md

### Top-level README:

```markdown
# {Question Title}

## Question
{One sentence — the specific thing we're trying to answer}

## Status: Active

## Why It Matters
{1-2 sentences — what depends on the answer, which engines/pipelines are affected}

## Engines
{List of engines this touches, with what role each plays}

## Completion Criteria
- [ ] {Concrete criterion 1}
- [ ] {Concrete criterion 2}
- [ ] ...

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| {sub if any} | — | — |

## Key Findings So Far
{Empty initially — updated as work progresses}

## Literature
| Paper | Summary | Key Insight |
|-------|---------|-------------|
| {empty initially} | | |

## Future Work
{No deferred items yet.}
```

### Sub-question README:

```markdown
# {Sub-Question Title}

**Parent**: [{parent name}](../README.md)

## Question
{One sentence}

## Status: Active

## Completion Criteria
- [ ] {criterion}

## Findings
{Empty initially}
```

## Step 6: Write KNOWLEDGE.md

Only for top-level questions (sub-questions contribute to the parent's KNOWLEDGE.md).

```markdown
# {Question Title} — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding
{What we know so far — even if just "nothing yet, starting investigation"}

## Key Decisions
{Decisions made based on this research, with rationale}

## Open Questions
{Things we still don't know}

## Connections
- **Engines**: {which engines implement or depend on this}
- **Related research**: {links to other research questions that interact}
- **Pipelines**: {Optimizer, Surrogate, Builder if relevant}
```

## Step 6b: Write IDEALOG.md

Only for top-level questions (sub-questions don't get their own IDEALOG.md).

```markdown
# {Question Title} — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
{What we're currently pursuing and why — same as README.md status initially}

## Next Step
{First step from completion criteria}

## Thread

## Failed Approaches

## Session Log
```

## Step 7: Update MASTER.md

Add a row to the Active Research table in `MASTER.md`:

```markdown
| [{question_name}](Research/Active/{question_name}/) | {engines} | Just started | {first step} |
```

If this is a sub-question, don't add to MASTER.md — update the parent's README.md sub-question table instead.

## Step 8: Update Parent (Sub-Questions Only)

If creating a sub-question, update the parent's `README.md`:
- Add a row to the Sub-Questions table
- If this is the first sub-question, create the Sub-Questions section

## Step 9: Report

Output:
- Path to the new folder
- The README.md content (for user review)
- Suggested first steps (literature search, engine experiment, etc.)
- Whether existing knowledge files or papers might be relevant (check `Research/Knowledge/` and `MASTER.md`)

---

## Rules

- **Never create a question without a clear completion criterion.** "Investigate X" is too vague. "Determine whether X causes Y, validated by Z" is concrete.
- **Top-level questions go in `Research/Active/`.** Never create directly in `Complete/` or `Backlog/`.
- **Sub-questions inherit parent resources.** Don't duplicate papers/ or literature/ unless the sub-question has material the parent doesn't.
- **Check for duplicates.** Before creating, scan Active/, Complete/, and Backlog/ for existing questions that might already cover this topic. If found, suggest extending the existing question instead.
- **Don't over-structure.** If the question is simple and has no obvious sub-topics, skip the sub-question scaffolding. It can be added later.
