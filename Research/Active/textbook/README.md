# Cardiac Computational Modeling Textbook

## Question
Is the *Cardiac Computational Modeling* textbook complete, correct against the engines, and pedagogically sound (Feynman-style, 5-layer complexity) across all four parts?

## Status: Active

## Why It Matters
The textbook is the single narrative documentation of the whole simulation stack (single-cell ionic → monodomain → bidomain → LBM). It doubles as a teaching artifact and grant/ResearchStatement material. Its usefulness depends on two things: equations that match the actual engine code, and writing accessible enough for the target readers. Chapter audits (2026-03-08) found the parts range from 1.9/5 (Part III, since rewritten) to 3.8/5 (Part II) — so there is a concrete, bounded backlog to close.

## Textbook Files (this folder)
**Migrated here 2026-07-02** from `Research/textbook/` — the full textbook source now lives alongside this question's tracking docs (README/KNOWLEDGE/IDEALOG). The `/textbook-edit` and `/textbook-compile` skills and `CLAUDE.md` point here. Paths below are relative to this folder.

| Asset | File |
|-------|------|
| HTML source (~12,300 lines) | `Bidomain_Textbook.html` |
| Rendered PDF | `Cardiac_Computational_Modeling.pdf` |
| Standalone Part IV PDF | `LBM_Textbook_Part_IV.pdf` |
| Website build | `Cardiac_Textbook_Website.html` + `website/` |
| Chapter index / registry | `INDEX.md` |
| Changelog (edits, newest first) | `CHANGELOG.md` |
| Style rules | `STYLE_GUIDE.md`, `SVG_FIGURES_SKILL.md` |
| Chapter audits (backlog source) | `audits/` |
| Reference implementations | `code_examples/` |

## Engines
Documents all three engines; equations are verified against their code.

| Part | Chapters | Engine documented |
|------|----------|-------------------|
| I Single-cell dynamics | 1–6 | Monodomain V5.4 ionic models (TTP06, ORd) |
| II Monodomain | 7–11 | Monodomain V5.4 (FDM/FEM/FVM, explicit/implicit solvers) |
| III Bidomain | 12–15 | Bidomain V1 (decoupled GS, three-tier elliptic solver) |
| IV Lattice-Boltzmann | 18–20 | LBM V1 (D2Q5/D2Q9, BGK/MRT) |
| Appendices | A–D | DEs, LinAlg, Numerical Analysis ("The Bridge"), PyTorch |

## Completion Criteria
- [ ] **Part II cross-ref cleanup** — Ch 10 figure numbers (Fig 9.x→10.x), eq (9.2)→(10.2), "Chapter 10's implicit methods"→Ch 11
- [ ] **Part II depth** — add BDF2 worked example (Ch 11); expand §11.6 linear-solver section (PCG/Chebyshev/FFT currently one paragraph each)
- [ ] **Part II length** — split/trim Ch 8 (~1100 lines covering FDM/FEM/FVM/BC — four mini-chapters)
- [ ] **Part IV Ch 18 restructure** — add "30-second preview" box at chapter open; trim §18.1 (Two Cases aside); front-load a worked example
- [ ] **Part IV numbering** — fix equation-numbering gaps + stale "Chapter 17"/"eq 17.x" cross-refs → 18/19
- [ ] **Part IV Ch 20 depth** — add worked example(s) for the three bidomain-LBM strategies (currently conceptual only)
- [ ] **Part III re-audit** — confirm the session-13b rewrite (Ch 12–17 → 12–15) actually closed the 1.9/5 issues; the existing Bidomain audit predates the rewrite
- [ ] **Reader-B accessibility** — provide a non-code reading path for the L5 implementation layer (inaccessible to no-programming readers)
- [ ] **Known content gaps** — Ch 4 SVGs → literature images; document engine-limited topics (Ch 6 chloride currents, Ch 18 MRT D3Q7, Ch 20 bidomain LBM) as "described, not yet in engine"
- [ ] **Rebuild** PDF + website from current HTML after edits land (`/textbook-compile`)

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| — | — | — |

## Key Findings So Far
- **Part quality (2026-03-08 audits):** Part II 3.8/5 (strong), Part IV 3.6/5 (good, length/ordering issues), Part III 1.9/5 → **rewritten** in session 13b (Ch 12–17 collapsed to 12–15; deleted nonexistent FGMRES+AMG architecture, restored Feynman style).
- **Engine verification:** Ch 5, 6, 10, 11, 19 fully verified equation-by-equation against engine code; Ch 1, 7, 8, 9 engine-box verified. See KNOWLEDGE.md verification table.
- **Reader-B lens:** the code/implementation (L5) layer is structurally inaccessible to physics/math readers with no programming; Ch 8 is the critical inflection point where PDEs first become matrices.

## Literature
This is a self-authored textbook, not a literature synthesis — its "sources" are the engine codebases (verification) and the vendored reference implementations in `code_examples/`. No PubMed papers filed under this question.

## Future Work
- Ch 20 bidomain LBM is architectural-only until Engine V5.4/LBM V1 implement a dual-lattice bidomain path.
- Ch 18 MRT for D3Q7 (7×7 M matrix) pending engine implementation.
