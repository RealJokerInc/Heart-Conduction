# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment

Conda environment `heart-conduction`, Python 3.11, PyTorch 2.10 (CUDA), scipy, torch_dct.

```bash
conda activate heart-conduction
```

GPU: NVIDIA RTX PRO 4500 Blackwell. All tensors default to float64.

## Running Tests

**Bidomain Engine V1** (pytest-based, 38+ tests across 6 phases):
```bash
cd Bidomain/Engine_V1
pytest tests/ -v                                    # All tests
pytest tests/test_phase2_fdm.py -v                  # Single phase
pytest tests/test_phase6c_boundary_cv.py::test_name  # Single test
```

**Monodomain Engine V5.4** (standalone scripts, not pytest):
```bash
cd Monodomain/Engine_V5.4
python test_phase7.py    # Builder integration (7 tests)
python test_phase8.py    # Per-node conductivity (7 tests)
```

**LBM V1** (pytest or standalone):
```bash
cd Monodomain/LBM_V1
python -m pytest tests/ -v
```

## Project Architecture

This is a **cardiac electrophysiology simulation** project with multiple engine implementations, a research corpus, and a neural surrogate pipeline.

### Active Engines

| Engine | Path | Status | Purpose |
|--------|------|--------|---------|
| **Bidomain V1** | `Bidomain/Engine_V1/` | 6 phases DONE (38+ tests) | Full bidomain equations, decoupled GS splitting, three-tier spectral/PCG/GMG elliptic solver. Ground truth for Surrogate training. |
| **Monodomain V5.3** | `Monodomain/Engine_V5.3/` | VALIDATED BASELINE | **Read-only.** Reference for V5.4 migration. Never modify. |
| **Monodomain V5.4** | `Monodomain/Engine_V5.4/` | 9 phases DONE (77 tests) | Full rewrite: pluggable FEM/FDM/FVM, 6 diffusion solvers, Strang/Godunov splitting, LBM (D2Q5/D3Q7). |
| **LBM V1** | `Monodomain/LBM_V1/` | 8 phases DONE (34 tests) | Lattice Boltzmann monodomain (D2Q5/D2Q9, BGK/MRT). Boundary speedup research. |

Legacy engines (V2–V5.2, Backup, Prototype) are in `Monodomain/_archive/` — do not modify.

### Supporting Components

| Component | Path | Purpose |
|-----------|------|---------|
| **Builder** | `Builder/` | Image-to-mesh conversion (PNG/SVG → StructuredGrid + stimulus). Integrated with V5.4. `Builder/MeshLibrary/` has geometry assets. |
| **Optimizer** | `Optimizer/` | Engine Tuner: BayesOpt pipeline for tuning ionic model parameters (TTP06/ORd) to target CV, APD, and restitution. V1 targets Monodomain V5.4 only. |
| **Surrogate** | `Surrogate/` | Neural operator pipeline (Ionic Transformer + Diffusion ResNet) replacing Bidomain V1 solver. Planning phase. |
| **Research** | `Research/` | Literature reviews organized by question (Q1–Q8). `Research/INDEX.md` is the entry point. Papers in `Research/papers/`. |
| **ResearchStatement** | `ResearchStatement/` | Grant materials. |
| **Images** | `Images/` | Centralized copies of all project images, organized by source (see subdirectories below). Originals remain in place. |
| **Videos** | `Videos/` | Centralized copies of all project videos, organized by source. |

### Shared Module Pattern

Ionic models (TTP06, O'Hara-Rudy) are copied across engines with identical interfaces:
- `ionic/base.py` — IonicModel ABC: `compute_Iion(V, states)`, `gate_inf()`, `gate_tau()`
- `ionic/lut.py` — Lookup table acceleration
- `ionic/ttp06/`, `ionic/ord/` — 18-state and 40-state models respectively

### Bidomain V1 Architecture (most active engine)

```
BidomainSimulation (orchestrator)
  → SplittingStrategy (Strang / Godunov)
    → IonicSolver (RushLarsen / ForwardEuler)
    → DecoupledBidomainDiffusionSolver
      → Step 1: Parabolic solve (A_para * Vm^{n+1} = rhs)
      → Step 2: Elliptic solve  (A_ellip * phi_e^{n+1} = L_i * Vm^{n+1})
        → LinearSolver: Tier 1 Spectral / Tier 2 PCG+Spectral / Tier 3 PCG+GMG
```

Key conventions:
- chi=1.0, Cm=1.0 in operators; D_i, D_e pre-scaled by chi*Cm
- Parabolic coupling: RHS uses `L_i * phi_e` (full coupling, NOT `theta * L_i * phi_e`)
- Elliptic solver auto-selected from BoundarySpec (Neumann→DCT, Dirichlet→DST, Mixed→PCG+GMG)

### Monodomain V5.4 Architecture

```
MonodomainSimulation (orchestrator)
  → SplittingStrategy (Strang / Godunov)
    → IonicSolver (RushLarsen / ForwardEuler)
    → DiffusionSolver
      → Explicit: ForwardEuler, RK2, RK4
      → Implicit: CrankNicolson, BDF1, BDF2 → LinearSolver (PCG / Chebyshev / FFT)
  → SpatialDiscretization (FEM / FDM / FVM — pluggable)
  → LBM path: LBMSimulation → Collision (BGK/MRT) + Streaming + BoundaryConditions
```

### LBM V1 Architecture (`Monodomain/LBM_V1/`)

Self-contained Lattice Boltzmann engine. Two-layer design: OOP classes for configuration, pure functions for `@torch.compile` kernel fusion.

```
LBMSimulation (coordinator, lattice-agnostic)
  → CollisionOperator (BGK single-tau / MRT multi-relaxation)
  → Streaming (stream_d2q5 / stream_d2q9 — pure functions)
  → BoundaryConditions (Neumann bounce-back / Dirichlet anti-bounce / absorbing equilibrium)
  → IonicSolver (Rush-Larsen standalone step)
  → Lattice (D2Q5 isotropic / D2Q9 full tensor — frozen dataclass singletons)
```

Key details:
- Research goal: demonstrate Kleber boundary speedup effect (reduced electrotonic loading at tissue edges)
- `sigma_to_D()` / `tau_from_D()` in `src/diffusion.py` handle LBM ↔ physical unit conversion
- State uses `(Nx, Ny)` grid convention, matching V5.4

## Three-Document Architecture

Each research question in `Research/Active/` uses three documents with distinct purposes:

| Document | Resolution | Purpose | Lifecycle |
|----------|-----------|---------|-----------|
| **KNOWLEDGE.md** | High | Reference: facts, analysis, designs, comparisons. Look things up. | Accumulates → promoted to `Research/Knowledge/` on completion |
| **IDEALOG.md** | Low | Thinking trail: insights, failed approaches, session log. Scan in 30s. | Living → archived with question on completion |
| **PLAN.md** | High (structured) | Cold-start agent execution steps. Created by `/blueprint`. | Created → steps checked off → archived |

**When to write where:**
- Findings, analysis, designs, decisions with rationale → **KNOWLEDGE.md**
- Ideas, failures, "oh wait" moments, session snapshots, next steps → **IDEALOG.md**
- Implementation steps for agent execution → **PLAN.md** (via `/blueprint`)

**`MASTER_KNOWLEDGE_INDEX.md`** at project root indexes all research questions and their KNOWLEDGE files.

### Planning Workflow

Pipeline: **`/reason`** (interactive thinking) → **`/blueprint`** (generate PLAN.md) → **`/audit`** (optional adversarial review) → execute PLAN.md steps.

`/reason` writes to IDEALOG.md on natural transitions (~3-4 writes per session, not after every exchange). When the approach is settled, `/blueprint` reads IDEALOG + codebase and generates a self-contained PLAN.md with cold-start steps.

## Permission Handling

- **DO NOT** request permissions for complete inline commands with embedded code
- Commands should use pattern-based permissions ending with `:*`
- If a command requires specific inline code, write to a temp file first, then run the file

## Plan Mode Usage

- Plan files contain **structured implementation steps only**
- Never write conversation transcripts, tool outputs, or raw session data to plan files

---

## Research & Textbook Workflows

Custom skills are available for research, planning, and engineering work (defined in `.claude/skills/`):

| Skill | When to use |
|-------|-------------|
| **Research lifecycle** | |
| `/research-new` | Create new research question (README + KNOWLEDGE + IDEALOG + dirs). Updates MASTER.md. |
| `/research-resume` | Resume work — loads KNOWLEDGE + IDEALOG, shows current direction, next step, what NOT to retry |
| `/research-update` | Route by type: findings/analysis → KNOWLEDGE.md, ideas/failures/issues → IDEALOG.md |
| `/research-status` | Staleness audit across all active questions |
| `/research-complete` | Active/ → Complete/, promote KNOWLEDGE to Knowledge/, archive IDEALOG |
| `/research` | Full PubMed pipeline: search → screen → acquire → summarize → file |
| `/summarize-paper` | Quick-summarize a single local PDF (stages 4-5 of `/research` only) |
| **Planning & reasoning** | |
| `/reason` | Interactive reasoning buddy — big→middle→small zoom, writes to IDEALOG on transitions |
| `/blueprint` | Generates machine-targeted PLAN.md from IDEALOG + codebase |
| `/audit` | Adversarial review via Opus subagent (opt-in) |
| `/save-session` | Session-end cleanup: snapshot→IDEALOG, organize KNOWLEDGE, cross-reference, update index |
| `/reason-end` | Tear down tmux workspace (kill viewer panes, clean WHITEBOARD.md) |
| **Engineering** | |
| `/verify` | Auto-detect engine, run test suite, produce pass/fail report |
| `/build-fix` | Systematic one-at-a-time error resolution with guardrails |
| `/strategic-compact` | Compaction decision table + pre-compact checklist |
| **Textbook** | |
| `/textbook-edit` | Writing or revising textbook content (reads style guide, changelog, audits first) |
| `/textbook-compile` | Building the textbook PDF from HTML source via Playwright |

**Key entry point**: `Research/INDEX.md` — master question map, debugging quick-reference, and citation registry. Read this first when doing any research-related work.

Research is organized by question (`Research/Active/` for in-progress, `Research/Complete/` for finished). Each question folder has README.md (status/criteria), KNOWLEDGE.md (reference), and IDEALOG.md (thinking trail). Papers live in `Research/papers/`. The textbook is in `Research/textbook/`.

---

## V5.4 Implementation Workflow

Multi-phase engine rewrite (9 phases, 77 validation tests). Sessions frequently hit compaction.

### Session Startup — Orientation Protocol

**Every session, before doing any work:**

1. Read `Monodomain/Engine_V5.4/PROGRESS.md` — current phase, done/in-progress/next
2. Read the relevant IMPLEMENTATION.md section for the current phase (not the whole file)
3. If implementing a file, read its ABC from `improvement.md` at the line number in PROGRESS.md
4. Check MEMORY.md for gotchas or prior failures

**Do NOT re-read entire documents speculatively.** Use line numbers from PROGRESS.md.

### Compaction Recovery

1. **Stop.** Don't continue blindly.
2. Run orientation protocol (PROGRESS.md first).
3. If mid-file when compacted, re-read that file to verify.
4. On "continue", pick up from next incomplete task in PROGRESS.md.
5. Do NOT re-do work marked as done.

### Task Management

- Use TaskCreate at phase start, mark in_progress/completed as work proceeds
- After each task: **update PROGRESS.md immediately**

### Implementation Rules

1. **Always check the ABC first** from `improvement.md` (line numbers in PROGRESS.md).
2. **Always check research references** in `IMPLEMENTATION.md § Summary of Key Research References`.
3. **V5.3 is ground truth for migrated code** — bitwise-identical output required.
4. **New code validates against IMPLEMENTATION.md criteria** (each phase has a validation table).
5. **One file at a time** — implement → validate → commit → update PROGRESS.md.
6. **Simple before complex** — ForwardEuler before CrankNicolson, BGK before MRT, isotropic before anisotropic.

### Cross-Reference Table

| What you need | Where to find it |
|---------------|-----------------|
| Current progress | `Engine_V5.4/PROGRESS.md` |
| Phase plan, validation criteria | `Engine_V5.4/IMPLEMENTATION.md` |
| ABC interfaces, design decisions | `Engine_V5.4/improvement.md` (use line numbers from PROGRESS.md) |
| High-level architecture | `Engine_V5.4/README.md` |
| Algorithm details | `Research/openCARP_FDM_FVM/01-04_*.md` |
| Reference implementations | `Research/code_examples/` |
| V5.3 validated code | `Engine_V5.3/` |
| All project images | `Images/` (subdirs by source engine/component) |
| All project videos | `Videos/` (subdirs by source engine/component) |

### What NOT To Do

- Don't re-read `improvement.md` in full (1750+ lines) — use line-number jumps
- Don't implement without reading the ABC
- Don't skip validation
- Don't modify V5.3
- Don't write code from memory after compaction — re-read source first
- Don't create new architectural patterns — ask the user if something seems missing
