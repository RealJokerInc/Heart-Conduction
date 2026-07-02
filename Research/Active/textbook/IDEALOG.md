# Cardiac Computational Modeling Textbook — Idea Log

> Thinking trail: how the textbook evolved, what was tried, where we are.
> Scan in 30 seconds. Full edit history is in `CHANGELOG.md` (this folder).

## Current Direction
Canonical source is now the **split website chapters** (`website/chapters/*.html`) — the monolithic `Bidomain_Textbook.html` was archived 2026-07-02 after a chapter-by-chapter comparison proved the website copy newer (see KNOWLEDGE "Source of Truth"). With the fork resolved, the active work is a **fresh chapter-by-chapter audit of the current source** (the user's request), then closing the audit backlog: Part II cross-ref/depth fixes → Part IV Ch 18 restructure + numbering → Part III re-audit → Reader-B accessibility pass. Rebuilding the missing PDF pipeline is a parallel to-do.

## Next Step
Full chapter-by-chapter audit is DONE (`audits/FULL_CHAPTER_AUDIT_2026-07-02.md`, book ≈3.8/5). Execute the Tier-1 fix backlog on `website/chapters/*.html`: (1) **book-wide cross-reference sweep** (~40 off-by-one stale refs — highest value); (2) three HIGH correctness bugs — Ch 11 BDF2 L-stability text, Ch 8 FDM worked-example arithmetic (+prose), Ch 13 (13.1)/Fig 13.1 sign; (3) notation/count unification (ORd 41, TTP06 18, R' recovery var, Ch 5 "Figure 4.1"→5.2). Then Tier 2 (App C.3/C.10 numeric fixes, figure drought) and Tier 3 (worked examples, Ch 8 split, §11.6).

## Thread

### 2026-07-02: Image/figure audit complete
Audited all 31 figures (all inline SVG; zero raster/external → no broken links) on integrity + correctness via 4 parallel agents. Wrote `audits/IMAGE_AUDIT_2026-07-02.md` (per-figure table + fix backlog). Two systemic integrity bugs: (1) `$...$` LaTeX inside `<svg><text>` doesn't render under MathJax tex-svg (Fig 5.1, Ch 10 ×3 confirmed; appendix labels at risk) — fix is plain Unicode as Ch 6/8 already do; (2) 8 figures mislabeled off-by-one (Ch 5 4.1→5.2 dup, Ch 8 7.x→8.x, Ch 10 9.x→10.x, Ch 18 17.1→18.1). Three HIGH correctness (math-visual) errors: Fig 13.1 block-matrix sign (confirms + reinforces the Ch 13 prose bug), Fig B.2 SPD bowl drawn inverted (concave-down), Fig B.1 shear disagrees with its own matrix. Plus MEDIUM quantitative-fidelity issues (Fig 4.5 non-proportional bar/pie, Fig 3.1 non-scale axes, Fig 4.1 buffer legend, Fig 1.2, Fig 4.4 arrows). Audit only — nothing fixed.

### 2026-07-02: Full chapter-by-chapter audit complete
Audited all 18 chapters + 4 appendices + refs against STYLE_GUIDE (Feynman 5-layer) on 4 dimensions (style/correctness/figures/connectiveness) via 6 parallel agents, on the canonical website source. Wrote `audits/FULL_CHAPTER_AUDIT_2026-07-02.md` (full scorecard + ranked findings + 3-tier fix backlog). Book ≈3.8/5. Three SYSTEMIC drags (not weak chapters): (1) book-wide off-by-one stale cross-refs from un-propagated renumbering (~40, Ch 19 worst); (2) figure drought (8 chapters w/ 0 SVGs); (3) missing L4 worked examples. HIGH correctness bugs: Ch 11 BDF2 stability (wrong), Ch 8 FDM arithmetic (wrong, +475→+262.5), Ch 13 block-matrix sign. Engine equation spot-checks all PASSED. **Part III open question answered: the 13b rewrite reached parity (3.75).** INDEX overstates Ch 18 "Quadrature First" (absent) but Ch 19 "Ω^NR" is actually present (earlier comparison miss). Nothing fixed yet — audit only.

### 2026-07-02: Reconciliation complete (checkpoint)
Finished the source-of-truth switch. Archived `Bidomain_Textbook.html` + its 2 PDFs → `_archive/monolithic_pre-fork_2026-07-02/` (git mv, history preserved, ARCHIVE_README added). Repointed `/textbook-edit` + `/textbook-compile` skills and all tracking docs (README/INDEX/KNOWLEDGE/CHANGELOG/MASTER) to `website/chapters/`; added a stale banner to WEBSITE_PLAN.md; recorded a `project-textbook-source-fork` memory. **Nothing committed yet.** Pending user decision: (a) launch the fresh chapter-by-chapter audit against `website/chapters/` (incl. the never-done Part III re-audit), and/or (b) commit the plumbing changes first. Two logged loose ends: missing `html_to_pdf_v3.py` (no PDF build), and INDEX overstating Part IV rewrites.

**2026-07-02 — Discovered the book had FORKED into two divergent copies; resolved in favor of the website.**
While setting up a chapter-by-chapter audit, found the documented "main file" `Bidomain_Textbook.html` (12,261 lines) disagreed with the tracking docs: it still had the deleted Part III Schur/FGMRES chapters (Ch 16–17) and only 2 appendices. Ran a 5-way parallel content comparison (one agent per Part/appendices) of the monolithic file vs the split `website/chapters/*.html`. Result: Parts I & IV byte-identical; Part II identical except Ch 8 appendix refs (split newer); Part III and the appendices **decisively newer in the split**. The rewrites (13b Part III, 14/15 four-appendix restructure) had all gone into the website source; the single-file + its PDF got orphaned. **Resolution (user directive):** website = source of truth. `git mv`'d `Bidomain_Textbook.html` + `Bidomain_Textbook.pdf` + `Cardiac_Computational_Modeling.pdf` into `_archive/monolithic_pre-fork_2026-07-02/` (with an ARCHIVE_README); repointed `/textbook-edit` + `/textbook-compile` and all tracking docs (README/INDEX/KNOWLEDGE/CHANGELOG/MASTER) to `website/chapters/`. Also discovered `html_to_pdf_v3.py` is missing entirely — no working PDF build until rebuilt. And INDEX overstates Part IV (claims Ch 18 "Quadrature First" / Ch 19 "Ω^NR" rewrites that aren't in the source).

**2026-07-02 — Textbook promoted to a tracked research question, then fully migrated in.**
Previously the textbook lived only in `Research/textbook/` with ad-hoc tracking (INDEX.md + CHANGELOG.md + audits/), so `/research-resume textbook` had no README/KNOWLEDGE/IDEALOG to load. Created `Research/Active/textbook/` and transferred the tracking state: status, engine-verification table, and the four-audit backlog synthesized into KNOWLEDGE.md; completion criteria into README.md. Then, on request, ran the **full migration**: `git mv`'d all 61 tracked textbook files (~20 MB, history preserved) from `Research/textbook/` into this folder and deleted the old dir. Repointed the `/textbook-edit` + `/textbook-compile` skills, `CLAUDE.md`, and `MASTER.md` at the new path. The folder is now self-contained (tracking docs + textbook source).

**Backlog snapshot at promotion** (see KNOWLEDGE.md for detail):
- Part II (3.8/5): Ch 8 too long; Ch 10 stale figure/eq refs; no BDF2 example; §11.6 solvers thin.
- Part IV (3.6/5): Ch 18 theory-front-loaded + no preview box; §18.1 too long; eq-numbering gaps + "Ch 17" stale refs; Ch 20 no worked examples.
- Reader-B: L5 code layer inaccessible to non-programmers; Ch 8 the inflection point; FEM weak-form under-motivated.
- Part III (1.9/5): audit is PRE-rewrite; largely fixed by session 13b — needs re-audit, not action on the old list.

## Failed Approaches

- **Monolithic bidomain solver architecture in Part III (FGMRES + AMG + SDIRK2 on a 2N×2N indefinite block system).** Written into old Ch 13–17, but it describes a solver that does not exist — Engine V1 uses decoupled N×N SPD solves. Scored the part 1.9/5 and would strand any reader who opened the code. Deleted in the session-13b rewrite (old Ch 16 removed entirely). Do not reintroduce monolithic-solver framing.
- **Reference-manual style for numerical chapters.** After Ch 12, the draft dropped analogies/worked examples/diagrams and became a method catalogue — the difficulty spiked with no bridge. The whole-book style guide (Feynman, 5-layer) exists specifically to prevent this; every new/revised section must carry L1 ELI5 + L4 worked example, not just L2/L5.
- **Appendix A as a catch-all** (DEs + LinAlg + numerical analysis crammed together). Split in session 14 into A/B/C/D with one job each; don't recombine.

## Session Log

- **2026-07-02 (later)** — Fork discovery + resolution: compared the two copies chapter-by-chapter, archived the stale monolithic file, switched canonical source to `website/chapters/`, repointed skills + docs. Flagged missing PDF build script and Part IV tracking-doc overstatement. No textbook *content* edited (structure/plumbing only). Audit of the current source is the next step.
- **2026-07-02** — Created this question as a tracking wrapper; wrote README/KNOWLEDGE/IDEALOG from existing INDEX.md + CHANGELOG.md + audits/. No textbook content edited yet.
- **Earlier history** (in `CHANGELOG.md`): session 15 (2026-03-09) Appendix C method-by-method rewrite + B visual overhaul; session 14 four-appendix restructure; session 13b Part III rewrite (12–17 → 12–15); Mar 13 website build + audit consolidation.
