# Cardiac Computational Modeling Textbook — Knowledge File

> Running synthesis of the textbook's state. Updated as the book is revised.
> Source lives in THIS folder (migrated 2026-07-02 from `Research/textbook/`); this file tracks its state, not its content.

## Source of Truth (canonical files) — READ FIRST

**Canonical source = the split website chapters, NOT any single file.**

- **Edit here:** `website/chapters/*.html` (`ch1`–`ch20`, `appendix-a`–`appendix-d`, `references`) + `website/toc.json`.
- **Whole-book renders:** the multi-page `website/index.html` + `app.js` (loads `chapters/` live — always current); the bundled `Cardiac_Textbook_Website.html` snapshot.
- **PDF (✅ working):** build with `website/build/html_to_pdf.py` (assembles chapters in `toc.json` order → MathJax `tex-svg` head + print CSS → Playwright headless Chromium). Current `Cardiac_Computational_Modeling.pdf` = **195 pp, A4**. `--html-only` gives a dep-free combined-HTML assembly. Needs `playwright` + chromium (installed in `heart-conduction` env) and network at render time (MathJax CDN).
- **ARCHIVED / never edit:** `_archive/monolithic_pre-fork_2026-07-02/` holds the old `Bidomain_Textbook.html` + its two PDFs — **stale** (deleted Part III "Schur Complement / Roadmap" chapters, only 2 appendices).

### The fork (discovered & resolved 2026-07-02)
The book had **forked into two divergent copies**. A chapter-by-chapter comparison established the split source is newer:

| Part | Verdict |
|------|---------|
| I (Ch 1–6) | Identical |
| II (Ch 7–11) | Identical except Ch 8 — split has corrected appendix cross-refs |
| III (Ch 12–17→12–15) | **Split much newer** — monolithic still had the deleted Schur/FGMRES Ch 16–17 |
| IV (Ch 18–20) | Identical |
| Appendices | **Split much newer** — split has A/B/C/D; monolithic had only A + PyTorch (~10k words of LinAlg + NumAn missing) |

Every divergence favored the split; the monolithic file had no unique content worth keeping. All post-Feb rewriting (13b Part III, 14/15 appendices) went into `website/chapters/`; the single-file + its PDF pipeline got orphaned. Resolution: adopted the website as canonical, archived the monolithic file, repointed `/textbook-edit` + `/textbook-compile` + all tracking docs.

## Current State (post-remediation, 2026-07-02)

The textbook is **fully drafted (Parts I–IV + appendices A–D) and the full audit-remediation backlog is CLOSED.** Book quality ≈ **3.8/5** pre-remediation; all correctness bugs, the book-wide cross-reference disease, and the figure-integrity issues are fixed, and the depth gaps (worked examples, figures) are largely filled. The book now has **40 SVG figures** (was 31) and **builds to a correct 195-pp PDF**.

> **Content is done; active work is now the WEBSITE** (refresh + interactive tooling) — see the "Website Refresh +
> Interactive Tooling" section below. Branch `textbook-website-refresh`: dark-mode figure system + identity + cover/part
> nav + a print-safe widget framework + the Ion-Current Playground are committed; the textbook figure-widgets (Phase 3)
> are paused. IDEALOG has the narrative trail.

### Structure

| Part | Chapters | Title | State |
|------|----------|-------|-------|
| I | 1–6 | Single Cell Dynamics | Good; Ch 5/6 engine-verified; Ch 1/2 now have worked examples + Ch 2 a phase-plane figure |
| II | 7–11 | Tissue-Level Monodomain | Strong (Ch 10 = 5-layer exemplar); Ch 11 BDF2 example + §11.6 expanded; Ch 8 reading-guide box |
| III | 12–15 | Bidomain | 13b rewrite reached parity (~3.75); Fig 13.1 signs fixed; +3 figures (13.2/14.1/15.1) + Ch 15 elliptic example |
| IV | 18–20 | Lattice-Boltzmann | Ch 18 gapless eqs + preview box + bell-curve fig; Ch 19/20 +figures; Ch 20 real worked example |
| App | A–D | DEs / LinAlg / Numerical Analysis / PyTorch | Crown of the book (A/C ≈4.5); App-C Chebyshev/DCT bugs fixed, +CG figure |

Chapter-number gap remains: Part III ends at Ch 15, Part IV starts at Ch 18 (old Ch 16–17 deleted in 13b, not renumbered — cross-refs point correctly, so harmless).

### Engine Verification
Ch 1, 5, 6, 7, 8, 9, 10, 11, 19 verified (engine box and/or equations) against V5.4 / LBM V1 code. Part III (Ch 12–15) class names verified against Bidomain V1 (nits noted below). Ch 18, 20, appendices not equation-by-equation verified against code (mostly kinetic-theory / general numerics, limited direct engine mapping).

## Website Refresh + Interactive Tooling (2026-07-03 → 07-10)

The book **content** is done; this is about the **website presentation + interactive tools**. All on branch
`textbook-website-refresh` (isolated from in-flight engine-tuner work). Aesthetic = explorable-explanation; **stay
vanilla / no build step** so the same `chapters/*.html` keep assembling into the 195-pp PDF. Interactivity is
**progressive enhancement over a static SVG fallback** — the loader (`figures.js`) is SPA-only and never injected by
`html_to_pdf.py`, so widgets never mount in print. PDF held at exactly **195 pp** through every change.

### Design — themeable figure system (Phase 1, commit `31f8937`)
- **Root-cause fix for the dark-mode figure bug**: deleted the blanket `[data-theme=dark] svg text/line/path` override
  (it repainted every figure curve one grey). Figures now carry themeable color via inline `style="…:var(--fig-*)"`.
- **Figure token system** — the site has exactly **two theme scopes** (`:root` light + `[data-theme="dark"]`; NO
  `@media prefers-color-scheme` — dark mode is JS-toggled). `--fig-*` palette (18 tokens): 7 categorical hues +
  ink/muted/faint/grid/stage + 6 panel tints. `--fig-axis` aliases `--fig-ink`. `.figure text{fill:var(--fig-axis)}`
  default covers group-inherited/unfilled text (replaces deleted line 810).
- **Color migration** (`website/build/migrate_figure_colors.py --census`→review `figure_color_map.json`→`--apply`):
  swept all **62 distinct color values** across 19 chapter SVGs → tokens. `var()` MUST be in a `style=` attribute,
  never a presentation attribute (`fill="var(--x)"` does not resolve). Idempotent.
- **Identity**: literary-serif × instrument-mono (labels/eyebrows/axis in mono), deepened arterial crimson
  `#c31d38`/`#ff5468`, left-aligned body (dropped justify). Wired the orphaned cover (`title.html`) + 5 part-divider
  pages into the SPA (were PDF-only); cover is the landing page.
- **Verification harness** `website/build/verify_site.py` — dual-theme Playwright screenshots + console-error capture
  + a print-safety assertion (loader absent from the PDF assembly; 0 `canvas.fig-widget` in print HTML).

### Design — figure-widget framework (Phase 2, commit `13f58e7`)
- `<figure class="fig" data-widget="NAME" data-params='…'>` with a `.fig-fallback` static SVG + a `.fig-controls` rail.
- **Module boundary**: `figures.js` is a **classic script** exposing `window.mountFigures` (so the IIFE `app.js` can
  call it); widget files are **ES modules** loaded via dynamic `import()`. `_canvas.js` holds shared helpers
  (`fit`/`cvar`/`rk4`/…). On successful mount the loader adds `.has-widget` → CSS hides the SVG on screen; print always
  shows the SVG. Reference widget: `figures/fhn.js` (FHN phase plane, LIVE in ch2).

### Design — Ion-Current Playground (AP-morphology explorer, commit `4c70236`)
Interactive tool: pick an ionic engine, tune per-current conductances, watch the single-cell AP change shape.
`chapters/playground.html` + `figures/ap-explorer.js` + an "Interactive Tools" sidebar section; NOT in `toc.json`
(kept out of the PDF). Widened via `.chapter-content:has(.tool-page)`.
- **Architecture** = precompute + knob grids (engines are PyTorch → can't run in browser). `website/build/gen_ap_traces.py`
  runs the REAL cardiac_core engines offline; every displayed trace is exact engine output; **sliders snap to grid
  levels** (no physically-invalid interpolation). Per engine: baseline + a **3-knob combinable grid** (5³) + **1-D
  isolated sweeps** for the rest. Output `data/ap_explorer/*.json` (8 configs, 924 KB; Vm as uint8 + base64).
- **Batched generation on CPU**: set each conductance field on `model.params` to an **(N,) tensor** (broadcasts inside
  `model.step`) → all cells of a config in one pacing loop. **`torch.compile` = 4.5M cell-steps/s** (16× over eager) →
  whole bank in minutes. Metrics computed on the **full-dt** trace (else the sub-ms upstroke aliases Vpeak/dV/dt).
- **UI**: engine + cell-type selector, AP plot (grey baseline vs red tuned), live metrics (APD₉₀/₅₀, dV/dt, V_rest/peak,
  + cycle length for PHAS13), Combine (grid) + Isolate (sweep) panels, current glossary with unfamiliar currents badged.

### Reference — the 4 cardiac_core ionic engines (verified against code)
Selected by string via `cardiac_core.ionic.registry.build_ionic_model(name, cell_type='ENDO', device)`.
Conductances are mutable `model.params` dataclass fields (tune = scale × default). FHN is NOT in the repo;
Mitchell–Schaeffer only in vendored `torchcor` — so the ch2/prototype pedagogical widgets are JS ports, not engines.

| name | model | states | beating | cell types | conductance knobs |
|------|-------|--------|---------|-----------|-------------------|
| `ttp06` (default) | ten Tusscher–Panfilov 2006 | 18 | paced | ENDO/EPI/M | GNa, PCa(ICaL), GKr, GKs, GK1, Gto, GpCa, GpK, GbNa, GbCa |
| `ord` | O'Hara–Rudy 2011 | 40 | paced | ENDO/EPI/M | GNa, GNaL(late Na), PCa, GKr, GKs, GK1, Gto, GKb, GpCa, Gncx, Pnak |
| `phas13` (="PHA-S") | Paci 2013 hiPSC-CM ("HIPSE"≈hiPSC) | 17 | **spontaneous** (I_f) | — | g_Na, g_CaL, g_Kr, g_Ks, g_K1, g_to, **g_f (funny)**, kNaCa, PNaK, g_pCa, g_bNa, g_bCa |
| `mhas13` | matured PHAS13 (g_f=0, TTP06 IK1) | 17 | paced | — | as PHAS13 minus funny |

Validated baselines (single-cell, BCL 1000, 12 beats): TTP06 APD90 236/236/293 ms (ENDO/EPI/M — M>Endo gradient
correct); ORd 258/239/377; PHAS13 CL 1634 ms / APD90 568; MHAS13 537. PHAS13's 12 currents include the "unfamiliar"
ones the tool demystifies: `I_f` (funny/HCN pacemaker, E_f=−17 mV), `I_NaCa` (exchanger), `I_NaK` (pump), `I_pCa`,
`I_bNa`/`I_bCa` (leaks). **Doc-gap**: `cardiac_core/API_CHEATSHEET.md §4` advertises only ttp06/ord.

## The 2026-07-02 Audit + Remediation (the authoritative reference)

Two fresh audits of the canonical website source (`audits/FULL_CHAPTER_AUDIT_2026-07-02.md`, `audits/IMAGE_AUDIT_2026-07-02.md`) — supersede the 2026-03-08 part-level audits. **All findings below are now RESOLVED** (commits 2a6c541 → 5457c99) unless marked DEFERRED.

### Scores (pre-remediation)
Book ≈3.8/5. Parts: I ≈3.8 · II ≈3.9 · III ≈3.75 · IV ≈3.7 · App A–D ≈4.45. Strongest: App A/C, Ch 10. Weakest: Ch 2, Ch 20 (3.2).

### Three systemic issues (all fixed)
1. **Book-wide off-by-one stale cross-refs** — chapters renumbered (Parts I–II +1; Part IV 17/18/19→18/19/20) but in-text refs never updated (~50 refs incl. 8 figure numbers; Ch 19 worst at 24). → **Fixed** (Phase 2), zero residual "Chapter 16/17" / "Figure 7.x/9.x/17.x". ch19's dual mapping (Ch18 refs→18.x, self-refs→19.x) done per its own eq labels.
2. **Figure drought** — 8 chapters had ZERO SVGs. → **Fixed** (Phase 4): 9 new figures; every formerly-empty chapter has ≥1.
3. **Missing L4 worked examples** — none in Part I, plus Ch 7/9/11/15/20. → **Largely fixed** (Phase 4): added to Ch 1, 2, 11, 15, 20.

### Content-correctness bugs (all fixed, Phase 1)
- **HIGH:** Ch 11 BDF2 stability (was contradictorily "A-stable / not L-stable"; BDF2 **is** L-stable — fixed box+table+amplification); Ch 8 FDM worked example (node-2 −190→−105 ⇒ +262.5, nodes 2 & 4 equal, prose rewritten + conservation check); Ch 13 block-matrix off-diagonals +L_i→−L_i (table + Fig 13.1 + caption) to match eq (14.1)/Alg 14.1.
- **MEDIUM:** App C.3 Chebyshev recurrence 2γ²→4γ²; App C.10 DCT recomputed to [5.000, 0.924, 2.000, −0.383] (inverse reconstructs [4,2,1,3]); Ch 3 I_K1 "just below E_K"→"above"; Ch 20 eq 20.2 → tensor form with Δτ/2·δ; Ch 18 garbled §18.5 closing paragraph; Ch 19 §19.5 reconciled to D2Q5 (removed phantom D2Q9 bounce-back diagonals).
- **LOW / notation:** ORd 40→41, TTP06 17→18, R' = recovery var (not concentration), Ch 4 buffer numbers unified (ATP/Mg 15% / Other 38%), Ch 6 current count clarified (15 distinct, 16 terms via NCX split).

### Image findings (all fixed, Phase 3)
All 31 figures were inline SVG (zero raster/external → no broken links). Fixes:
- **LaTeX-in-SVG doesn't render** under MathJax tex-svg — `$…$` inside `<svg><text>` showed literally. → All converted to plain Unicode (Fig 5.1, Ch 10 ×3, appendices). Book-wide invariant: **no `$` inside any SVG**.
- **HIGH math-visual errors:** Fig 13.1 sign (fixed w/ Phase 1); Fig B.2 SPD bowl was drawn concave-DOWN → flipped to open upward, minimum at bottom; Fig B.1 shear ê₂→(0.625,1) → corrected to (1,1) matching matrix [[2,1],[0,1]].
- **MEDIUM:** Fig 4.5 bar/pie made proportional; Fig 4.1 legend + viewBox widened (no clip); Fig 3.1 axes linearized; Fig 1.2 annotations reconciled to drawn curves; Fig 4.4 OUT arrows flipped outward.

### Engine spot-checks (passed)
All TTP06/ORd (Ch 5/6) and RK/BDF (Ch 10/11) equations match V5.4 source. Part III class names verified vs Bidomain V1. Known nits (not yet fixed): §15.4 lists `get_L_i()/get_L_e()` methods that don't exist (real: `get_parabolic_operators`/`get_elliptic_operator`); code paths abbreviated (real prefix `cardiac_sim/simulation/classical/`).

### Deferred (open, non-blocking)
- **Ch 8 full split** — only a reading-guide box added; still ~1100 lines.
- **Reader-B non-code path** — improved by new examples/figures, but no dedicated parallel path.
- **Ch 4 SVGs → literature images** — still hand-drawn SVG.
- **§18.1 "Two Cases (rest vs flow)" trim** — still present.
- **§15.4 engine-method-name nits** (above).

## Historical audits (2026-03-08, superseded)
Four part-level audits (`audits/{MONODOMAIN,LBM,READER_B,BIDOMAIN}_CHAPTER_AUDIT.md`) scored the book against `STYLE_GUIDE.md`. Part II 3.8, Part IV 3.6, Reader-B (L5 inaccessible to non-programmers; Ch 8 = inflection point; FEM weak-form under-motivated — since FIXED), Part III 1.9 (⚠️ **pre-13b-rewrite, obsolete** — it triggered the rewrite that deleted the fictional monolithic architecture). Retained for provenance; the 2026-07-02 audits above are current.

## Key Decisions
- **Canonical source = website chapters (2026-07-02)** — monolithic archived; skills + docs repointed. Rationale: two copies forked, website confirmed newer everywhere they differed.
- **PDF pipeline = `website/build/html_to_pdf.py`** (2026-07-02) — replaces the missing `html_to_pdf_v3.py`; assembles chapters via `toc.json` → Playwright. `--html-only` mode for dep-free assembly.
- **ch18 equations renumbered gapless-ascending 18.1→18.41** (Phase 4) — all in-file + cross-chapter refs (ch19 ×8, ch20 ×1) updated and individually verified against the new labels.
- **Four-appendix chain** (session 14): A = DEs, B = LinAlg, C = "The Bridge" (numerics on one running grid), D = PyTorch. Do not recombine.
- **Full migration** (2026-07-02): all 61 textbook files `git mv`'d into this question folder, history preserved.

## Open Questions
- ~~Do Part III (12–15) hit Feynman parity with Part II?~~ **ANSWERED: yes (~3.75 vs 3.9); no fictional framing survives; figures added.**
- Is a dedicated parallel non-code reading path worth the maintenance, or is "skip the engine boxes" enough for Reader-B? (Deferred.)
- Close the Ch 15→18 chapter-number gap (renumber 18–20→16–18), or leave it? (Cross-refs already correct, so harmless.)

## Connections
- **Engines**: documents Monodomain V5.4 (Parts I–II), Bidomain V1 (Part III), LBM V1 (Part IV); equations verified against their code.
- **Related research**: draws on completed [bidomain_simulation](../../Complete/bidomain_simulation/) and [scar_bc_validity](../../Complete/scar_bc_validity/); Part IV overlaps [lbm_ep](../lbm_ep/).
- **Skills**: `/textbook-edit` (write/revise → `website/chapters/`), `/textbook-compile` (build PDF via `website/build/html_to_pdf.py`). The installed `frontend-design` plugin fits a future website-chrome redesign.
- **Style**: `STYLE_GUIDE.md` (Feynman + 5-layer), `SVG_FIGURES_SKILL.md`.
