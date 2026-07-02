# Cardiac Computational Modeling Textbook — Knowledge File

> This file is a running synthesis. Updated as the textbook is revised.
> Textbook source lives in THIS folder (migrated 2026-07-02 from `Research/textbook/`); this file tracks its state, not its content.

## Source of Truth (canonical files) — READ FIRST

**As of 2026-07-02 the canonical source is the split website chapters, NOT the old single file.**

- **Edit here:** `website/chapters/*.html` (`ch1`–`ch20`, `appendix-a`–`appendix-d`, `references`) + `website/toc.json`.
- **Whole-book render:** `Cardiac_Textbook_Website.html` (generated snapshot) and the multi-page `website/index.html`.
- **ARCHIVED / do not touch:** `_archive/monolithic_pre-fork_2026-07-02/` holds the old `Bidomain_Textbook.html` + its two PDFs. That file is **stale** — it still contains the deleted Part III "Schur Complement / Implementation Roadmap" chapters (fictional solver architecture) and only 2 appendices.
- **PDF:** 🚧 no working build — `html_to_pdf_v3.py` is missing from the repo. The former "official" PDF (`Cardiac_Computational_Modeling.pdf`) was built from the stale file and is archived with it.

### The fork (discovered & resolved 2026-07-02)
The book had **forked into two divergent copies**. A chapter-by-chapter content comparison established which was newer:

| Part | Verdict |
|------|---------|
| I (Ch 1–6) | Identical |
| II (Ch 7–11) | Identical except Ch 8 — split has corrected appendix cross-refs (→ App C) |
| III (Ch 12–17→12–15) | **Split much newer** — monolithic still had the deleted Schur/FGMRES Ch 16–17 |
| IV (Ch 18–20) | Identical |
| Appendices | **Split much newer** — split has A/B/C/D; monolithic had only A + PyTorch (~10k words of LinAlg + NumAn missing) |

Every divergence favored the split; the monolithic file had **no unique content worth keeping**. Direction of the fork: all post-Feb rewriting (13b Part III, 14/15 appendices) went into `website/chapters/`; the single-file + its PDF pipeline got orphaned. Decision: adopt the website as source of truth, archive the monolithic file, repoint the `/textbook-edit` + `/textbook-compile` skills and all tracking docs.

**Caveat found during the comparison:** INDEX.md claims Ch 18 "Quadrature First" and Ch 19 "Ω^NR/Ω^R" rewrites are done, but **neither is present in the current source** — tracking docs overstate Part IV completion. Verify before trusting.

## Current Understanding

The textbook is **fully drafted** across Parts I–IV plus four appendices. It renders to the website (`Cardiac_Textbook_Website.html` + multi-page site); the PDF build is currently blocked (missing script). Content quality is uneven but bounded: three of four parts audit at B+ or better; the weakest (Part III) was rewritten in session 13b. The remaining work is a **known, enumerated backlog** from four 2026-03-08 audits — not open-ended authoring.

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

## Full Chapter-by-Chapter Audit (2026-07-02) — `audits/FULL_CHAPTER_AUDIT_2026-07-02.md`

Fresh audit of the **canonical website source** (supersedes the 2026-03-08 part-level audits). Book overall ≈ **3.8/5**. Strongest: appendices A/C + Ch 10 (5-layer exemplars). Weakest: Ch 20 & Ch 2 (3.2). Part scores: I ≈3.8 · II ≈3.9 · III ≈3.75 · IV ≈3.7 · App A–D ≈4.45.

**The book is dragged down by three SYSTEMIC issues, not weak chapters:**
1. **Book-wide off-by-one stale cross-refs** — chapters were renumbered (Parts I–II +1; Part IV 17/18/19→18/19/20) but in-text refs never updated. ~40 bad refs; Ch 19 worst (24). Highest-value single fix (one mechanical sweep). Full table in the audit file.
2. **Figure drought** — 8 chapters have ZERO SVGs (Ch 2, 7, 9, 11, 14, 15, 19, 20); Part IV = 1 SVG / 2617 lines.
3. **Missing L4 worked examples** — none in all of Part I, plus Ch 7/9/11/15/20. The chapters that have them (8, 10, App C) score highest.

**Correctness bugs — HIGH (teach something wrong):** Ch 11 BDF2 stability (contradictory + wrong; BDF2 IS L-stable); Ch 8 FDM worked-example arithmetic (node-2 = +475, should be +262.5, breaks its own teaching point); Ch 13 block matrix (13.1)/Fig 13.1 sign (off-diagonals must be −Li). **MEDIUM:** App C.3 Chebyshev recurrence (2γ²→4γ²), App C.10 DCT coefficients non-reproducible, Ch 3 I_K1 direction, Ch 20 eq 20.2, Ch 19 D2Q9/D2Q5 mix, Ch 18 garbled closing paragraph. **LOW (systematic):** ORd 40→41, TTP06 17→18 + R' miscategorized, Ch 4 buffer numbers, Ch 6 current count, Ch 5 dup "Figure 4.1".

**Engine spot-checks:** all TTP06/ORd (Ch 5/6) and RK/BDF (Ch 10/11) equations verified against V5.4 source; Part III class names verified against Bidomain V1 (nits: §15.4 `get_L_i/get_L_e` don't exist; paths abbreviated). Prioritized fix backlog (3 tiers) in the audit file.

## Image / Figure Audit (2026-07-02) — `audits/IMAGE_AUDIT_2026-07-02.md`

All **31 figures are inline SVG** (zero `<img>`/raster/external → no broken links). Integrity: 19/31 clean. Correctness: 23/31 clean, **3 HIGH-severity math-visual errors**.
- **Systemic integrity #1 — LaTeX-in-SVG doesn't render:** figures with `$...$` inside `<svg><text>` show literal LaTeX under MathJax tex-svg (confirmed broken: Fig 5.1, Ch 10 ×3; at-risk: appendix labels). Fix = plain Unicode (Ch 6/8 already do this). Verify in compiled render.
- **Systemic integrity #2 — 8 figures mislabeled** (off-by-one, same disease as prose): Ch 5 "Fig 4.1"→5.2 (duplicates Ch 4's real 4.1), Ch 8 7.x→8.x, Ch 10 9.x→10.x, Ch 18 17.1→18.1.
- **HIGH correctness:** Fig 13.1 block-matrix sign (+L_i → −L_i; the Ch 13 prose table repeats it — reinforces the chapter-audit bug); Fig B.2 SPD bowl drawn concave-DOWN (inverted); Fig B.1 shear ê₂→(0.625,1) contradicts its matrix (2 1;0 1).
- **MEDIUM:** Fig 4.5 stacked bar/pie non-proportional; Fig 4.1 buffer legend vs prose; Fig 3.1 non-scale axes; Fig 1.2 annotations vs curves; Fig 4.4 reversed OUT arrows.
- **Best:** App C.1–C.3 (match worked examples exactly), the D2Q5/9-D3Q7 lattice stencils (all correct). Full per-figure table + 3-tier fix backlog in the audit file.

## Audit Backlog (from `audits/`, 2026-03-08 — largely superseded by the 2026-07-02 full audit above)

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

- **Canonical source switched to the website chapters (2026-07-02)** — see "Source of Truth" above. The monolithic `Bidomain_Textbook.html` was archived; `/textbook-edit` + `/textbook-compile` and the tracking docs were repointed to `website/chapters/`. Rationale: the two copies had forked and the website copy was confirmed newer everywhere they differed (Part III + appendices decisively; Parts I/II/IV identical).
- **Full migration executed 2026-07-02** — the entire textbook (61 tracked files, ~20 MB) was `git mv`'d from `Research/textbook/` into this question folder, history preserved. The `/textbook-edit` / `/textbook-compile` skills, `CLAUDE.md`, and `MASTER.md` were repointed here. The question folder is now self-contained (tracking docs + textbook source together), not a wrapper around a separate directory.
- **Backlog is prioritized mechanical → structural:** Part II cross-ref fixes (cheap, high-value) before Ch 18 restructure (larger) before Reader-B accessibility pass (design work) before Part III re-audit.
- **Four-appendix dependency chain** (session 14 decision): A = what to solve (DEs), B = tools (LinAlg), C = "The Bridge" (numerical analysis combining A+B), D = PyTorch. Each appendix does exactly one job.
- **Appendix C = method-by-method on one running 2D grid** (session 15): every numerical method (FE/BE/CN/DST/CG/PCG/Chebyshev) gets its own section + worked example on the same problem.

## Open Questions

- ~~Do the rewritten Part III chapters (12–15) hit Feynman-style parity with Part II?~~ **ANSWERED (2026-07-02 audit): yes** — Part III ≈3.75 vs Part II ≈3.9; no fictional monolithic framing survives. Residual: (13.1) sign bug + Ch 14/15 figure drought.
- Is a parallel non-code reading path worth the maintenance cost, or is a single "skip the engine boxes" convention enough for Reader-B?
- Should the Ch 15/18 chapter-number gap be closed (renumber 18–20 → 16–18), or is the gap harmless given cross-refs already point correctly?

## Connections
- **Engines**: documents Monodomain V5.4 (Parts I–II), Bidomain V1 (Part III), LBM V1 (Part IV); equations verified against their code.
- **Related research**: draws on completed [bidomain_simulation](../../Complete/bidomain_simulation/) and [scar_bc_validity](../../Complete/scar_bc_validity/) knowledge; Part IV overlaps [lbm_ep](../lbm_ep/).
- **Skills**: `/textbook-edit` (write/revise), `/textbook-compile` (build PDF via Playwright).
- **Style**: `STYLE_GUIDE.md` (Feynman + 5-layer), `SVG_FIGURES_SKILL.md`.
