# Cardiac Computational Modeling Textbook — Idea Log

> Thinking trail: how the textbook evolved, what was tried, where we are.
> Scan in 30 seconds. Full edit history is in `CHANGELOG.md` (this folder).

## Current Direction
**NEW TRACK (2026-07-03): website UI/UX refresh + interactive figures.** Content-wise the book is done (5-phase
audit-remediation COMPLETE, 40 figs, 195-pp PDF). The open front is now the *website presentation*: it's a
competent-but-anonymous docs template with a real dark-mode figure bug and static/decorative figures. Ran a UI
audit (`audits/UI_AUDIT_2026-07-03.md`), decided a direction, and built an interactive proof-of-direction
(artifact `a514e90d`). Plan lives in `REFRESH_PLAN.md`. Direction (set while user away, PENDING CONFIRM):
explorable-explanation aesthetic · ~6–8 flagship interactive widgets + themeable static SVG for the rest ·
stay vanilla/no-build so the PDF pipeline survives (interactivity layered over a static SVG print fallback).

Canonical source remains the **split website chapters** (`website/chapters/*.html`); monolithic archived.

## Next Step (updated 2026-07-06 — Phases 1–2 DONE + committed)
**Phase 1 committed `31f8937`, Phase 2 committed `13f58e7` on branch `textbook-website-refresh`** (isolated from the
in-flight engine-tuner work). Phase 1: dark-mode figure bug FIXED at root (blanket `svg` override deleted; 62 color
values migrated to an 18-token `--fig-*` system via census map; verified colored+legible in dark on ch2/ch5/ch18);
serif×mono + arterial crimson + dropped-justify typography; orphaned cover+part pages wired into the SPA. Phase 2:
print-safe widget framework (`figures.js` classic→`window.mountFigures`+dynamic import; `figures/_canvas.js`+`fhn.js`),
FHN phase-plane widget LIVE in ch2 (drag IC, 4 sliders, limit-cycle readout, themeable), static SVG prints. PDF holds
at exactly 195pp both phases; 0 console errors throughout. Reusable harnesses: `verify_site.py`, `migrate_figure_colors.py`.

**REMAINING — Phase 3 (5–7 more widgets) + Phase 4 (identity/polish).** Each Phase-3 widget needs a figure-target +
physics-pedagogy decision, some non-obvious:
- ch10 has clean targets: **Fig 10.1 RK-slope → RK step-size widget (3.2)**; **Fig 10.3 stability-region → CFL widget (3.3)**.
- **AP shaper (3.1)**: ch3's only figure is Fig 3.1 "five phases + dominant currents" — the Mitchell–Schaeffer shaper
  reshapes AP/APD but does NOT show the 5 phases/currents, so wrapping it drops that annotation on screen (static
  fallback keeps it for print). DECISION NEEDED: wrap Fig 3.1 anyway / add a new interactive figure / place AP widget elsewhere.
- Nernst/GHK (3.4): no existing Nernst figure found — likely a NEW figure (scope beyond "wrap existing").
- Wavefront (3.5): hardest; 1-D monodomain cable; may split.
Prototype AP (Mitchell–Schaeffer + APD90) already exists in `plans/2026-07-03_refresh_prototype.html`.
(Blueprint hard-gate: no implementation until explicit approval.) Phase 1 = CSS+nav, no widgets: build
`verify_site.py` harness → add the expanded `--fig-*` token system (TWO scopes only: `:root`+`[data-theme=dark]`)
→ census-driven color migration (`migrate_figure_colors.py --census`→review→`--apply`) → delete blanket override
`style.css:810-812` (replaced by `.figure text{fill:var(--fig-axis)}`) → shell/type refresh → wire cover/part
pages. Then P2 figure-widget framework (classic `figures.js`→`window.mountFigures`+dynamic import; print-safe),
P3 6–8 widgets, P4 identity+PDF-regression. Durable prototype: `plans/2026-07-03_refresh_prototype.html`.
Deferred *content* items still open (separate track): Ch 8 split, Reader-B non-code path, Ch 4 SVGs, §18.1 trim.

## Thread

### 2026-07-03: Blueprint → PLAN.md, adversarial audit ×2 → CONVERGED
`/blueprint` turned `REFRESH_PLAN.md` + codebase analysis into a machine-targeted `PLAN.md` (4 phases: fix/re-shell
→ figure-widget framework → 6–8 widgets → identity+PDF-verify). Archived the completed audit-remediation plan →
`plans/2026-07-02_audit-remediation.md`. Then ran the /audit-revise loop the user asked for. **Round 1** (Opus
auditor, read-only): 2 HIGH + 4 MED + 8 LOW — and it *verified my line numbers/functions/physics were right*, but
caught two real errors: (HIGH-1) my 8-token figure palette can't express the **61 distinct hexes** the figures
actually use (census confirmed) → expanded to categorical(7)+ink(4)+stage+tint(6), migration made census-first;
(HIGH-2) I mis-stated the theme model as 3-scope `@media` — the site is really TWO JS-toggled scopes
(`:root`+`[data-theme=dark]`), and my instruction would've leaked dark colors into a light-toggled dark-OS view.
Plus MEDs: deleting `style.css:810` orphaned ch18 group-inherited text (added scoped `.figure text` default); the
verify leak-assertion was trivially always-true (replaced); app.js-IIFE vs figures.js-ES-module boundary (→ classic
`window.mountFigures` + dynamic import). Revised all 14. **Round 2**: CONVERGED — 0 crit/0 high, 2 med (fallback
double-render visibility; census ignored `fill="white"`×42) + 7 low, all folded in; auditor independently
re-derived the MED-1 text-default is sound (0 corpus cases of a hued group wrapping unfilled text). Persisted the
prototype into the repo (was ephemeral scratchpad). PLAN ready to execute; blueprint hard-gate holds — awaiting "go".

### 2026-07-03: Website UI audit + interactive-refresh direction + proof-of-direction artifact
User: "the website ui and images are horrible... currently the images are static and non interactive... make them
interactive... looks horrid. start with a ui audit, then a plan for a refresh rebuild." **Audited the live site**
via Playwright screenshots (light+dark, title/ch1/ch2/ch10) — not just the code. Verdict: light-mode book
typography is genuinely fine (~3.5/5), but two real bugs + a wasted opportunity drag it down. Wrote
`audits/UI_AUDIT_2026-07-03.md`. Key findings: **(HIGH)** dark-mode figure bug — `style.css:811-812`
`svg path{stroke:var(--text-secondary)}` blanket-repaints EVERY figure curve one grey (confirmed live: FHN
red/blue/green nullclines+trajectory all collapse to identical grey → figure meaningless in dark); **(HIGH)**
figures are hand-drawn static SVG w/ hardcoded coords+hex, ~460px, space-starved, zero interactivity — biggest
miss for a book about *dynamics*; **(MED)** anonymous docs-template identity, one crimson does everything;
**(MED)** cover `title.html` + `part-*.html` are ORPHANED — SPA never loads them (only PDF does), so no
landing/part dividers, `#title`→ch1 fallback. **Hard constraint:** `chapters/*.html` double as PDF source →
interactivity must be progressive enhancement over a static SVG fallback, stay vanilla.
**Asked 3 direction forks via AskUserQuestion; user was away** → proceeded on best-judgment defaults:
explorable-explanation aesthetic · ~6–8 flagship widgets + themeable static SVG rest · enhance-vanilla-in-place.
Because the complaint is *visual*, built a real **proof-of-direction artifact** (`a514e90d`) instead of prose:
new serif×mono identity, arterial-crimson accent, always-dark hero AP monitor, a live before/after of the
dark-mode bug, and TWO working physically-correct interactive widgets — FHN phase plane (drag IC, sliders,
live limit-cycle detection) + Mitchell–Schaeffer AP shaper (APD₉₀ live). Verified in both themes, 0 console
errors; figures stay colored+legible in dark (the fix, demonstrated). Wrote `REFRESH_PLAN.md` (design tokens +
`<figure data-widget>` architecture + flagship list + 4 phases). Prototype widget code (RK4, themeable canvas)
reusable in the real build. **Nothing in the actual `website/` touched yet** — audit + plan + prototype only.

### 2026-07-02: PLAN.md Phases 4–5 executed + committed (remediation COMPLETE)
**Phase 4** (commit d5e332a): authoring via 6 parallel per-chapter agents. Worked examples — ch1 Nernst, ch2 FHN, ch11 BDF2 (5×5 matrix + bootstrap), ch15 3-node elliptic solve, ch20 5-node pseudo-time relaxation (replacing the compute-nothing outline). 9 new SVG figures (2.1 FHN phase plane, 11.1 L-stable-vs-CN-ringing, 13.2 face stencil, 14.1 GS data-flow, 15.1 solver decision tree, 18.2 bell curve, 19.1 bounce-back, 20.1 dual-lattice, C.4 CG-vs-descent). Depth: §11.6 expanded + App C.12–14 links; ch8 reading-guide box; ch18 30-sec preview box; References → 18 entries. ch18 equations renumbered gapless 18.1→18.41 with ALL cross-chapter refs (ch19×8, ch20×1) updated — carefully verified each against ch18's new labels (BGK→18.9, τ-D D-form→18.26, τ-form→18.28, equilibrium→18.18). Fixed a pre-existing orphan </p> in ch19. Book now 40 figures; no $ in any SVG, tags balanced.
**Phase 5** (next commit): wrote `website/build/html_to_pdf.py` (assembles chapters in toc.json order + MathJax head + print CSS → Playwright → PDF; `--html-only` mode for dep-free assembly). Installed playwright + chromium in the env. Built `Cardiac_Computational_Modeling.pdf` = **195 pp A4, verified**: 0 raw-LaTeX leak (MathJax rendered), 0 stale Schur chapter, current Part III + appendices A–D. Reconciled INDEX (dropped "Quadrature First v4.0", kept Ω^NR). Un-blocked the `/textbook-compile` skill. **Everything the audits found — wrong content AND missing content — is now fixed, and the book builds to a correct PDF.**

### 2026-07-02: PLAN.md Phases 1–3 executed + committed
Worked through the audit-remediation blueprint. **Phase 1** (commit 2a6c541): 3 HIGH correctness bugs (Ch 11 BDF2 is L-stable; Ch 8 FDM node-2 arithmetic −190→−105 ⇒ +262.5 with rewritten "node 4" prose + conservation check; Ch 13 block-matrix off-diagonals +L_i→−L_i in table+Fig 13.1+caption) + MEDIUM (App C.3 Chebyshev 2γ²→4γ²; App C.10 DCT recomputed to [5,0.924,2,−0.383], inverse-verified; Ch 3 I_K1 above E_K; Ch 20 eq 20.2 tensor form; Ch 18 garbled paragraph; Ch 19 §19.5 reconciled to D2Q5) + notation (ORd 41, TTP06 18, R' recovery var, Ch 4 buffers, Ch 6 counts). **Phase 2** (commit 0008226): book-wide off-by-one sweep — ~50 cross-refs + 8 figure numbers; Parts I–II via agent, Part IV by hand (ch19 dual mapping: Ch18 refs→18.x, self-refs→19.x per its own eq labels); verified zero residual Chapter 16/17 / Figure 7.x/9.x/17.x. **Phase 3** (commit 46a8375): figure integrity — all `$…$`-in-SVG → Unicode (Fig 5.1, ch10×3, appendices); redraws — Fig B.1 shear→(1,1), Fig B.2 bowls flipped concave-up, Fig 4.5 proportional bar+pie, Fig 4.1 viewBox widened, Fig 3.1 linear axes, Fig 1.2 annotation, Fig 4.4 arrows. All verified (no `$` in any SVG, tags balanced, geometry checked). **Everything the audits found WRONG is now fixed.** Phases 4–5 (authoring + PDF pipeline) remain.

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

### 2026-07-02 Session
**Worked on**: `/research-resume textbook` → discovered the book had forked into two divergent copies; audited every chapter + every image; then executed the entire 5-phase audit-remediation blueprint.
**Accomplished**: (1) Confirmed `website/chapters/` is canonical; archived the stale monolithic `Bidomain_Textbook.html` + its PDFs; repointed skills + docs. (2) Full chapter-by-chapter audit (book ≈3.8/5) + image audit (31 figs) → `audits/{FULL_CHAPTER,IMAGE}_AUDIT_2026-07-02.md`. (3) `/blueprint` → 5-phase PLAN.md. (4) Executed ALL 5 phases (commits 7590635→5457c99): P1 correctness bugs (BDF2 L-stability, Ch8 FDM arithmetic, Ch13 signs, App-C Chebyshev/DCT, …) + notation; P2 book-wide cross-ref/figure-number sweep (~50 refs, zero residual); P3 figure integrity (LaTeX-in-SVG→Unicode) + B.1/B.2/4.5/3.1 redraws; P4 authoring (5 worked examples + 9 new figures → **40 total** + §11.6 + Ch18 preview + References→18); P5 PDF pipeline (`website/build/html_to_pdf.py` → **195-pp verified PDF**) + INDEX reconcile. Installed the `frontend-design` plugin.
**Next**: deferred items via `/textbook-edit` — Ch 8 full split, Reader-B non-code path, Ch 4 SVGs→literature images, §18.1 "Two Cases" trim; OR website-chrome redesign via `frontend-design`; OR `/research-complete`. Re-run `/textbook-compile` after any future edits.

- **2026-07-02 (later)** — Fork discovery + resolution: compared the two copies chapter-by-chapter, archived the stale monolithic file, switched canonical source to `website/chapters/`, repointed skills + docs. Flagged missing PDF build script and Part IV tracking-doc overstatement. No textbook *content* edited (structure/plumbing only). Audit of the current source is the next step.
- **2026-07-02** — Created this question as a tracking wrapper; wrote README/KNOWLEDGE/IDEALOG from existing INDEX.md + CHANGELOG.md + audits/. No textbook content edited yet.
- **Earlier history** (in `CHANGELOG.md`): session 15 (2026-03-09) Appendix C method-by-method rewrite + B visual overhaul; session 14 four-appendix restructure; session 13b Part III rewrite (12–17 → 12–15); Mar 13 website build + audit consolidation.
