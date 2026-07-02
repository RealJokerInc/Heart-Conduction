# Cardiac Computational Modeling Textbook — Idea Log

> Thinking trail: how the textbook evolved, what was tried, where we are.
> Scan in 30 seconds. Full edit history is in `CHANGELOG.md` (this folder).

## Current Direction
Textbook is fully drafted (Parts I–IV + appendices A–D) and rendered to PDF + website. The active work is **closing the audit backlog** from the 2026-03-08 chapter audits, in priority order: Part II cross-ref/depth fixes → Part IV Ch 18 restructure + numbering → Part III re-audit → Reader-B accessibility pass → recompile.

## Next Step
Part II cross-reference cleanup (cheapest, highest-value): fix Ch 10 figure numbers (Fig 9.x→10.x), eq (9.2)→(10.2), and the "Chapter 10's implicit methods"→Ch 11 self-reference in `bidomain_textbook.html`. Then add the missing BDF2 worked example in Ch 11.

## Thread

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

- **2026-07-02** — Created this question as a tracking wrapper; wrote README/KNOWLEDGE/IDEALOG from existing INDEX.md + CHANGELOG.md + audits/. No textbook content edited yet.
- **Earlier history** (in `CHANGELOG.md`): session 15 (2026-03-09) Appendix C method-by-method rewrite + B visual overhaul; session 14 four-appendix restructure; session 13b Part III rewrite (12–17 → 12–15); Mar 13 website build + audit consolidation.
