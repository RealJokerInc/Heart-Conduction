# Monthly Report Pipeline

## Question
How do we reliably produce John Zimmerman's monthly lab report deck from project state (git history per active question, PROGRESS, KNOWLEDGE, IDEALOG) with minimal manual effort, while flagging missing imagery/proofs that should be created?

## Status: Active

## Why It Matters
Lab requires a monthly presentation deck following John Zimmerman's format guidelines. Multiple side projects (11+ active research questions, 3 engines, 3 pipelines) make manual tracking error-prone. A reliable AI-assisted pipeline lets monthly reporting happen in minutes instead of hours and ensures nothing important is omitted.

## Engines
None directly — meta/tooling project. Consumes state from all engines, pipelines, and active research questions.

## Tool Output Location
`Monthly_Notebook/` at repo root (decks, scout outputs, imagery gap lists, monthly summaries).

## Completion Criteria
- [x] John Zimmerman's deck guidelines extracted from his email and documented in `KNOWLEDGE.md` (required sections, slide count, length, style, tone, what gets included vs omitted) — done 2026-04-28
- [ ] Pipeline architecture designed: which skills exist, what each consumes/produces, how they chain
- [ ] Entry-point skill implemented (reason-style interactive orchestrator that walks through the deck build with the user)
- [ ] Sub-skills implemented:
  - **Scout** — sweep git log per active question, PROGRESS.md across engines, IDEALOG.md across questions, identify side-project activity
  - **Consolidate** — per-question/per-engine monthly summary in a uniform structure
  - **Assemble** — produce deck matching Zimmerman's format
  - **Imagery audit** — list figures that should exist but don't (missing proofs)
- [ ] First real monthly deck produced end-to-end and approved
- [ ] Imagery/proof gap list generated for the reporting period

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| — | — | — |

## Key Findings So Far
None yet — starting investigation.

## Literature
| Paper | Summary | Key Insight |
|-------|---------|-------------|
| (n/a — tooling project, no literature expected) | | |

## Future Work
None deferred yet.
