---
name: research-status
description: Show the status of all research questions, update MASTER.md, and sync knowledge files. Use to get a project-wide research overview or to audit staleness.
argument-hint: "[optional: specific question name to drill into]"
---

# Research Status

Show and optionally update the status of all research questions.

Input: $ARGUMENTS

### Project Structure Reference
```
MASTER.md                              — project dashboard (source of truth for status)
Research/Active/{question}/            — README.md, KNOWLEDGE.md, literature/, papers/, figures/
Research/Complete/{question}/           — answered questions (KNOWLEDGE.md promoted to Knowledge/)
Research/Backlog/{question}/            — parked questions
Research/Knowledge/                     — promoted knowledge files (must sync with Complete sources)
{Engine}/experiments/                   — experiment code + outputs
```

---

## Mode A: Full Overview (no argument)

### 1. Read MASTER.md

Read `MASTER.md` at project root. This is the dashboard.

### 2. Scan the filesystem

Verify MASTER.md matches reality by scanning:

```
Research/Active/*/README.md
Research/Complete/*/README.md
Research/Backlog/*/README.md
```

For each question, extract:
- Status (from README.md `## Status:` line)
- Sub-questions and their statuses
- Last modified date of KNOWLEDGE.md (proxy for recent work)
- Number of papers in `papers/`
- Number of results in `results/`

### 3. Present the overview

```
ACTIVE RESEARCH (4)

  boundary_conduction_speedup          [3 sub-questions: 1 complete, 2 active]
    KNOWLEDGE.md last updated: 2026-03-10
    Papers: 8  |  Results: 3 dirs  |  Engines: Bidomain, LBM

  ionic_model_optimization             [no sub-questions]
    KNOWLEDGE.md last updated: 2026-03-14
    Papers: 8  |  Results: 0  |  Engines: V5.4, Optimizer

  hipsc_cm_ionic_models                [no sub-questions]
    KNOWLEDGE.md last updated: 2026-03-15
    Papers: 0  |  Results: 0  |  Engines: V5.4, Bidomain

  engine_consolidation                 [no sub-questions]
    KNOWLEDGE.md last updated: 2026-03-16
    Papers: 0  |  Results: 0  |  Engines: All

COMPLETE RESEARCH (5)

  bidomain_simulation                  Knowledge: Research/Knowledge/bidomain_simulation.md
  lbm_cardiac                          Knowledge: Research/Knowledge/lbm_cardiac.md
  scar_bc_validity                     Knowledge: Research/Knowledge/scar_bc_validity.md

BACKLOG (1)

  fetal_heart_development              Trigger: boundary speedup validated in 3D

WARNINGS:
  - hipsc_cm_ionic_models has no papers yet (consider /research to find literature)
  - engine_consolidation KNOWLEDGE.md is empty (consider writing initial synthesis)
```

### 4. Check for staleness

Flag questions where:
- KNOWLEDGE.md hasn't been updated in >14 days but status is Active
- Sub-questions are all complete but parent is still Active
- Papers exist but KNOWLEDGE.md doesn't reference their findings
- MASTER.md has entries not matching filesystem (orphaned or missing)

### 5. Check knowledge file sync

For each Complete question, verify:
- `Research/Knowledge/{name}.md` exists
- It's not older than `Research/Complete/{name}/KNOWLEDGE.md` (may need re-promotion)

### 6. Offer fixes

If any issues found:
- Stale KNOWLEDGE.md → offer to update based on recent results/papers
- Missing Knowledge/ promotion → offer to copy
- MASTER.md out of sync → offer to regenerate

---

## Mode B: Drill Into Specific Question (argument provided)

### 1. Find the question

Search in Active/, Complete/, and Backlog/ for a folder matching the argument.

### 2. Read everything

- README.md (question, status, criteria, sub-questions)
- KNOWLEDGE.md (current synthesis)
- All sub-question README.md files
- List all papers/ and results/

### 3. Present detailed status

```
BOUNDARY CONDUCTION SPEEDUP — Active

Question: Does conduction velocity increase at inert tissue boundaries?

Completion Criteria:
  [x] Isotropic CV ratio measured (1.071 → 1.131)
  [x] Bidomain validates Kleber effect
  [ ] Anisotropic boundary study
  [ ] 3D validation

Sub-Questions:
  [Complete] Isotropic CV ratio — 1.071 at dx=0.025, converges to 1.131
  [Active]   Anisotropic boundaries — fiber-parallel vs perpendicular effect
  [Active]   Bath loading arrhythmias — wavefront curvature at boundaries
  [Complete] Stencil comparison — 5pt vs Mehrstellen, mono vs bidomain

Literature (8 papers):
  bishop_2011_augmented_monodomain       → KNOWLEDGE: referenced
  bishop_2011_bath_loading_arrhythmias   → KNOWLEDGE: referenced
  rossi_2018_thickness_curvature         → KNOWLEDGE: NOT referenced (stale?)
  ...

Results:
  triangle_merger/         14 .pt files, 2 .json
  triangle_merger_quick/   10 .pt files, 1 .json
  anisotropic_test/        (directory exists)
  conductivity_sweep/      (directory exists)

Knowledge File:
  Last updated: 2026-03-10
  Sections: Current Understanding, Key Decisions, Open Questions, Connections
  Coverage: 6/8 papers referenced
```

### 4. Suggest actions

Based on the analysis:
- Papers not referenced in KNOWLEDGE.md → "Consider updating knowledge with findings from rossi_2018"
- Active sub-questions with no results → "anisotropic_boundaries has no results yet — ready to start?"
- All criteria met → "All completion criteria appear met — ready for /research-complete?"

---

## Rules

- **Read the filesystem, not just MASTER.md.** MASTER.md may be stale. The filesystem is ground truth.
- **Don't modify anything without asking.** This skill is read-only by default. Offer fixes but let the user approve.
- **Date awareness matters.** Convert "last modified" to relative time ("3 days ago", "2 weeks ago") for quick staleness assessment.
- **Knowledge coverage is the key metric.** Papers in the folder but not in KNOWLEDGE.md means findings aren't synthesized yet — this is the most common gap.
