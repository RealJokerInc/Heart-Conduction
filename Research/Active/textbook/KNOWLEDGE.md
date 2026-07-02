# Cardiac Computational Modeling Textbook — Knowledge File

> This file is a running synthesis. Updated as the textbook is revised.
> Textbook source now lives in THIS folder (migrated 2026-07-02 from `Research/textbook/`); this file tracks its state, not its content.

## Current Understanding

The textbook is **fully drafted** across Parts I–IV plus four appendices (~12,300 lines of HTML). It renders to `Cardiac_Computational_Modeling.pdf` and a website. Content quality is uneven but bounded: three of four parts audit at B+ or better; the weakest (Part III) was rewritten in session 13b. The remaining work is a **known, enumerated backlog** from four 2026-03-08 audits — not open-ended authoring.

### Structure (from `INDEX.md`)

| Part | Chapters | Title | Status |
|------|----------|-------|--------|
| I | 1–6 | Single Cell Dynamics | Good; Ch 5 (TTP06), 6 (ORd) **Verified** |
| II | 7–11 | Tissue-Level Monodomain | 3.8/5; Ch 10, 11 **Verified** |
| III | 12–15 | Bidomain | Rewritten session 13b (was 12–17) |
| IV | 18–20 | Lattice-Boltzmann | 3.6/5; Ch 19 **Verified** |
| App | A–D | DEs / LinAlg / Numerical Analysis ("The Bridge") / PyTorch | A–C rewritten sessions 14–15 |

Note the chapter-number gap: Part III ends at Ch 15, Part IV starts at Ch 18 (old Ch 16–17 were deleted in the 13b rewrite and not renumbered).

## Engine Verification Status (from INDEX.md)

| Chapter | Engine Box | Equations | Basis |
|---------|-----------|-----------|-------|
| 1 HH | ✅ | — | IonicModel ABC method names |
| 5 TTP06 | ✅ | ✅ | 12 currents vs `ttp06/currents.py` |
| 6 ORd | ✅ | ✅ | 15 currents vs `ord/currents.py` |
| 7 Monodomain | ✅ | — | `SpatialDiscretization` ABC |
| 8 Spatial Disc | ✅ | — | FDM/FEM/FVM + BC impl |
| 9 Splitting | ✅ | — | Godunov/Strang/RushLarsen |
| 10 Explicit | ✅ | ✅ | RK2/RK4 bare-k matrix forms |
| 11 Implicit | ✅ | ✅ | BDF1/CN/BDF2 A_lhs/B_rhs forms |
| 19 LBM Mono | ✅ | ✅ | BGK/MRT, streaming, bounce-back, τ-D |

**Not yet verified equation-by-equation:** Part III (Ch 12–15) post-rewrite, Ch 18, Ch 20, appendices.

## Audit Backlog (from `audits/`, 2026-03-08)

Four audits scored the book against `STYLE_GUIDE.md` (Feynman, 5-layer: L1 ELI5 / L2 conceptual / L3 visual / L4 worked example / L5 implementation).

### Part II — Monodomain (Ch 7–11) — 3.8/5 · `MONODOMAIN_CHAPTER_AUDIT.md`
Strong part. Issues:
1. **Ch 8 too long** (~1100 lines = FDM+FEM+FVM+BC, four mini-chapters; overload by §8.5 FVM).
2. **Ch 10 wrong self-refs** — "Chapter 10's implicit methods" should be Ch 11; figures "Figure 9.x" should be 10.x; CFL cited as "(9.2)" should be "(10.2)". Copy-paste artifacts from restructuring.
3. **Ch 11 no BDF2 worked example** — BDF1 and CN get examples, BDF2 gets only math + table.
4. **§11.6 linear solvers too thin** — PCG/Chebyshev/FFT one paragraph each, despite being 60–80% of cost. Appendix C fills it, but the reader doesn't know that yet.

### Part IV — LBM (Ch 18–20) — 3.6/5 · `LBM_CHAPTER_AUDIT.md`
Ambitious, mostly successful. Issues:
1. **Ch 18 too long (~1300 lines), theory-front-loaded** — no pseudocode until §18.3 (~line 850). Fix: add a 10-line "30-second preview" box (the 6-step LBM loop, ≈ eq 19.11) before §18.1.
2. **§18.1 too long (~350 lines)** — trim the "Two Cases (rest vs flow)" subsection; only rest matters for cardiac EP.
3. **Equation-numbering gaps + stale cross-refs** — jumps/reuses; several "Chapter 17"/"eq 17.26" should be 18/19.
4. **Ch 20 no worked examples, no code** — pseudo-time/hybrid/dual-lattice described but never computed. §20.4 "worked example outline" computes nothing.
5. Moment-space §18.5 (the payoff) arrives ~1000 lines in — anchor BGK/moment examples earlier.

### Reader Profile B — no-computational-background reader · `READER_B_AUDIT.md`
Reader has neurophysiology + heat/mass transfer + linear algebra + vector calc, but **no programming, no numerical methods, no time-stepping-as-algorithm**.
- **L5 (implementation) layer is inaccessible throughout** — pseudocode, ABCs, `torch.roll`, factory functions are meaningless to this reader. Not harmful, but a whole layer is lost.
- **Ch 7 = 5.0/5** (physics matches their background — "it's just the heat equation with a source").
- **Ch 8 = 3.4/5, the critical inflection point** — first time a PDE becomes a matrix. FDM on-ramp good; **FEM weak form under-motivated** ("multiply by a test function and integrate" — why? never explained for a strong-form reader). Worked examples (5-node cable) are the moment of understanding.

### Part III — Bidomain (Ch 12–17) — 1.9/5 · `BIDOMAIN_CHAPTER_AUDIT.md` — ⚠️ LARGELY STALE
This audit **triggered** the session-13b rewrite; its findings are mostly already fixed. Two original failure modes:
1. Described a solver architecture (monolithic 2N×2N + FGMRES+AMG) that **does not exist** — code uses decoupled N×N SPD solves. → Fixed: rewrite deleted the fictional architecture (old Ch 16), Ch 15 now documents real Engine V1 classes.
2. Feynman style abandoned after Ch 12 (Ch 13–17 = reference-manual catalogue). → Fixed: Ch 13–15 rewritten with analogies, worked examples (5-node cable with numbers), face-based stencil.
**Action:** re-audit the rewritten Ch 12–15 to confirm closure — do NOT act on the pre-rewrite issue list.

## Key Decisions

- **Full migration executed 2026-07-02** — the entire textbook (61 tracked files, ~20 MB) was `git mv`'d from `Research/textbook/` into this question folder, history preserved. The `/textbook-edit` / `/textbook-compile` skills, `CLAUDE.md`, and `MASTER.md` were repointed here. The question folder is now self-contained (tracking docs + textbook source together), not a wrapper around a separate directory.
- **Backlog is prioritized mechanical → structural:** Part II cross-ref fixes (cheap, high-value) before Ch 18 restructure (larger) before Reader-B accessibility pass (design work) before Part III re-audit.
- **Four-appendix dependency chain** (session 14 decision): A = what to solve (DEs), B = tools (LinAlg), C = "The Bridge" (numerical analysis combining A+B), D = PyTorch. Each appendix does exactly one job.
- **Appendix C = method-by-method on one running 2D grid** (session 15): every numerical method (FE/BE/CN/DST/CG/PCG/Chebyshev) gets its own section + worked example on the same problem.

## Open Questions

- Do the rewritten Part III chapters (12–15) actually hit Feynman-style parity with Part II, or did the rewrite trade one set of gaps for another? (Needs re-audit.)
- Is a parallel non-code reading path worth the maintenance cost, or is a single "skip the engine boxes" convention enough for Reader-B?
- Should the Ch 15/18 chapter-number gap be closed (renumber 18–20 → 16–18), or is the gap harmless given cross-refs already point correctly?

## Connections
- **Engines**: documents Monodomain V5.4 (Parts I–II), Bidomain V1 (Part III), LBM V1 (Part IV); equations verified against their code.
- **Related research**: draws on completed [bidomain_simulation](../../Complete/bidomain_simulation/) and [scar_bc_validity](../../Complete/scar_bc_validity/) knowledge; Part IV overlaps [lbm_ep](../lbm_ep/).
- **Skills**: `/textbook-edit` (write/revise), `/textbook-compile` (build PDF via Playwright).
- **Style**: `STYLE_GUIDE.md` (Feynman + 5-layer), `SVG_FIGURES_SKILL.md`.
