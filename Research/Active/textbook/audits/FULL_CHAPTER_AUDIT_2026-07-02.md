# Full Chapter-by-Chapter Audit — 2026-07-02

**Source audited:** the canonical split website source `website/chapters/*.html` (NOT the archived monolithic file).
**Rubric:** `STYLE_GUIDE.md` — Feynman 5-layer (L1 ELI5 · L2 conceptual · L3 visual+math · L4 worked example · L5 implementation) + the 10 "Needs Improvement" flags.
**Dimensions scored:** writing style · content correctness · graphs/figures · connectiveness.
**Method:** 6 parallel per-part audit agents; engine equations spot-checked against the real engine code where claimed. Supersedes the 2026-03-08 part-level audits.

---

## Scorecard

| Part | Ch | Title | Score |
|------|----|-------|-------|
| I | 1 | The Hodgkin–Huxley Revolution | 4.2 |
| I | 2 | From Neurons to Heart Cells | 3.2 |
| I | 3 | Anatomy of the Cardiac AP | 3.7 |
| I | 4 | Intracellular Calcium | 3.6 |
| I | 5 | ten Tusscher–Panfilov 2006 (TTP06) | 3.9 |
| I | 6 | O'Hara–Rudy 2011 (ORd) | 4.0 |
| II | 7 | The Monodomain Equation | 3.9 |
| II | 8 | Spatial Discretization | 3.8 |
| II | 9 | Operator Splitting | 3.8 |
| II | 10 | Explicit Diffusion Solvers | **4.2** |
| II | 11 | Implicit Diffusion Solvers | 3.7 |
| III | 12 | The Bidomain Model | 3.9 |
| III | 13 | From Equations to Matrices | 3.5 |
| III | 14 | Solving the Coupled System | 4.0 |
| III | 15 | Linear Solvers and Implementation | 3.6 |
| IV | 18 | LBM: Kinetic Theory to Computation | 4.0 |
| IV | 19 | LBM for Monodomain | 3.8 |
| IV | 20 | LBM for Bidomain | **3.2** |
| App | A | Differential Equations | **4.6** |
| App | B | Linear Algebra | 4.4 |
| App | C | Numerical Analysis: The Bridge | 4.5 |
| App | D | PyTorch | 4.3 |
| App | — | Key References | 3.0 |

**Part averages:** I ≈ 3.8 · II ≈ 3.9 · III ≈ 3.75 · IV ≈ 3.7 · Appendices A–D ≈ **4.45**.
**Book overall ≈ 3.8/5.** Strongest: appendices A/C and Ch 10 (5-layer exemplars). Weakest content chapters: Ch 20 and Ch 2 (3.2).

**Headline:** The prose and physics are consistently strong (three parts B+ or better; the Part III 13b rewrite reached parity at 3.75, up from 1.9). The book is dragged down by **three systemic issues that cut across all parts**, not by any single weak chapter.

---

## Cross-cutting findings (ranked by impact)

### 1. Systematic off-by-one stale cross-references — BOOK-WIDE, highest-value single fix
The book was renumbered at least twice (Parts I–II chapters +1; Part IV chapters 17/18/19 → 18/19/20) and the **in-text references were never updated**. This is the dominant connectiveness defect and is mechanical to fix in one sweep:

| Location | Stale ref | Should be |
|----------|-----------|-----------|
| Ch 5, Ch 6 | Rush–Larsen "(Chapter 8)" | Chapter 9 |
| Ch 8 | "Figure 7.1/7.2/7.3" | 8.x |
| Ch 8 | "sections 7.5–7.7"; "Godunov (Chapter 8)"; "Chapters 9 and 10 … time integrators" | 8.2–8.4; Ch 9; Ch 10 & 11 |
| Ch 10 | "Chapter 10's implicit methods"; "(Chapter 10)" ×2 | Chapter 11 |
| Ch 10 | "Figure 9.1/9.2/9.3"; "Chapter 7" (spatial) | 10.x; Chapter 8 |
| Ch 11 | "Explicit methods (Chapter 9)"; "CFL (equation 9.2)"; "Chapter 7" (K/M) | Ch 10; (10.2); Ch 8 |
| Ch 18 | "equation 17.2/17.3/17.4/17.14"; "Figure 17.1" | 18.x |
| Ch 19 | 16× "Chapter 17"/"17.x"; 8× "18.x" that are really this chapter's own 19.x eqs; "Chapter 19 develops three strategies" | 18.x; 19.x; Chapter 20 |
| Ch 20 | "monodomain equation (Chapter 18)"; "(equation 17.26)"; "Part IV (Chapters 17–19)" | Ch 19; 18.26; 18–20 |
| App D | "The FFT solver (Chapter 10)" | Chapter 11 §11.6 |

*(Appendix A already fixed its "Chapter 17" → "Part IV (Chapters 18–20)".)* Ch 19 is the worst offender (24 bad refs).

### 2. Figure drought — the "graphs" dimension, second-highest impact
Eight chapters have **zero SVGs** despite highly visual content: **Ch 2, 7, 9, 11, 14, 15, 19, 20**. Part IV is the worst — 1 stencil SVG across 2,617 lines. Highest-value missing figures:
- Ch 2: FHN phase-plane / nullclines; AP-shape comparison.
- Ch 11: L-stable-damping vs CN-ringing plot.
- Ch 13: face-based-stencil diagram (the one visual the chapter most needs).
- Ch 14: Gauss-Seidel data-flow (φe → parabolic → Vm → elliptic).
- Ch 15: three-tier solver decision flowchart.
- Ch 18: Gaussian/bell-curve + moment-decomposition figures.
- Ch 19: bounce-back reflection + fiber-anisotropy diagrams.
- Ch 20: dual-lattice architecture schematic (the "fish tank," invoked 3× but never drawn).
- App C: CG-vs-gradient zigzag (C.12); stiffness/two-timescale plot (C.8).

### 3. Missing L4 worked examples — pervasive
No numeric worked example in **Ch 1–6 (all of Part I), Ch 7, Ch 9, Ch 11 (BDF2), Ch 15, Ch 20**. The chapters that HAVE them (Ch 8, 10, App C) are the highest-scoring — direct evidence this is the lever. Ch 20's §20.4 "worked example outline" computes nothing.

---

## Content-correctness bugs (ranked by severity)

**HIGH — teaches something incorrect:**
1. **Ch 11 — BDF2 stability is wrong and self-contradictory.** ELI5 says "L-stable," math box says "technically not L-stable," table says "A-stable." BDF2 **is** L-stable (dominant root → 0 as z → −∞). The printed `R(z)=(4−1/(1+2z/3))`, |R(∞)|=1/3 is garbled (that form → 4, not 1/3). Fix all three to "L-stable" with a correct amplification factor.
2. **Ch 8 — FDM flagship worked example has an arithmetic error.** Node-2 bracket is written −190; its own expression `(-1)(-85)+2(-85)+(-1)(20)` = −105, giving +262.5, not the printed **+475**. This also breaks the prose claim "node 2 receives +475 … node 4 only half" — with the K they build, nodes 2 and 4 are **equal (+262.5)**.
3. **Ch 13 — block matrix (13.1)/Fig 13.1 sign inconsistency.** As printed (A12=A21=+Li, A22=−(Li+Le)) the elliptic row gives +(Li+Le)φe = +Li·Vm, contradicting eq (14.1)'s −(Li+Le)φe = Li·Vm. Off-diagonals must be **−Li** (or flip A22's sign). Isolated to the schematic; Algorithm 14.1 is correct.

**MEDIUM — reproducibility / physics slips:**
4. **App C.3 — Chebyshev recurrence bug.** Algorithm C.3 line 3 `ω_{k+1}=1/(1−δ²ω_k/(2γ²))` should be `/(4γ²)` (matches C.14's own prose `1/(1−ρ²ω_k/4)`); it's 2× off for k ≥ 2.
5. **App C.10 — DCT coefficients non-reproducible.** Printed [5.000, 1.577, 0.500, −0.224] don't come from a standard DCT-II of [4,2,1,3] (k=0 matches; k=1,2 don't). A hand-tracing reader can't reproduce it.
6. **Ch 3 — I_K1 direction slip.** "passes large outward current at voltages just below E_K" — outward K⁺ current flows **above** E_K, not below.
7. **Ch 20 — eq 20.2 inconsistency.** Writes "1/2" where 19.6/18.26 use "Δt/2", and feeds a tensor into a scalar formula.
8. **Ch 19 — D2Q9 vs D2Q5 mismatch (§19.5).** §19.5 says "the D2Q9 lattice used in this chapter" (and a different index convention) while the rest of Ch 19 uses D2Q5.
9. **Ch 18 — garbled closing paragraph** (line ~1294): "In the next chapter… In Section 18.5 above… moment space, —".

**LOW — notation / label / count inconsistencies (systematic, group-fix):**
- **ORd state count:** "40" in Ch 2 and Ch 7 vs the style-guide-canonical **41** (Ch 6 reconciles it correctly as "40 ionic + V stored separately" — propagate that phrasing).
- **TTP06 state count:** "17" in Ch 8 (L796, L870) vs **18** everywhere else; and Ch 5 miscategorizes R' as a "concentration" (it's the SR-release recovery variable → "5 concentrations + R'").
- **Ch 4 buffer breakdown mismatch:** Fig 4.1 "SR membranes" vs Fig 4.5/§4.6 "ATP/Mg²⁺"; "Other" 37% vs 38%.
- **Ch 6 current count muddled:** eq (6.1) has 16 terms, prose says "15", table "15 (+chloride)".
- **Ch 5 duplicate figure label:** §5.8 diagram labeled "Figure 4.1" (should be 5.2).
- **Ch 18 equation order:** 18.9/18.10 appear after 18.11/18.12; 18.27 before 18.17; 18.28 before 18.23–26 (no missing/reused numbers, just out-of-order defs).

---

## Per-part notes

**Part I (≈3.8).** Strong L1/L2 physiology storytelling; every TTP06/ORd equation spot-checked matches the V5.4 source. Weaknesses are structural: no worked examples anywhere, under-figured (Ch 2 = 0 figures, Ch 6 = 1 for 20 equations), and the Rush–Larsen "Chapter 8" cross-ref bug in Ch 5 & 6.

**Part II (≈3.9).** Pedagogically the strongest part; **Ch 10 is a genuine 5-layer exemplar (4.2)**. FEM weak-form motivation is now fixed. Held back by the Ch 8 arithmetic bug, the Ch 11 BDF2 stability error, the systematic off-by-one refs, and Ch 8's length (~1100 lines, still four mini-chapters).

**Part III (≈3.75).** The 13b rewrite is a clear success — reached parity with Part II, decoupled Gauss-Seidel is presented as THE method (Algorithm 14.1), no fictional monolithic/Schur framing survives as real, and named engine classes verify. Residual: the (13.1) sign bug and a figure drought (Ch 14 & 15 have zero SVGs; Ch 15 no worked example). Minor engine nits: §15.4 lists `get_L_i()/get_L_e()` methods that don't exist (real: `get_parabolic_operators`/`get_elliptic_operator`); abbreviated code paths (real prefix `cardiac_sim/simulation/classical/`).

**Part IV (≈3.7).** The strongest *prose* in the book (Ch 18's D2Q5 collision worked example is outstanding; Ch 19 is implementation-ready) — but crippled by the off-by-one ref failure (worst in the book), a near-total figure absence, and Ch 20's lack of any computed example. Ch 18 still lacks the "30-second preview" box and front-loads theory (moment space at ~74% depth).

**Appendices (A–D ≈4.45).** The crown of the book. The A→B→C→D "one job each" design holds; C is a genuine method-by-method masterclass on one running grid with real worked examples. Defects are narrow: the C.3 Chebyshev and C.10 DCT bugs, the App D "Chapter 10" FFT ref, an unfinished B.11 arithmetic, a missing Ch 11 §11.6 → App C.12–14 back-link, and a thin/bidomain-only References list.

---

## Prioritized fix backlog

**Tier 1 — cheap, high-value, do first**
1. **Global cross-reference sweep** (finding #1) — fix every stale chapter/figure/equation ref book-wide. One mechanical pass; biggest connectiveness win.
2. **Three HIGH correctness bugs** — Ch 11 BDF2 L-stability text; Ch 8 FDM arithmetic (+prose); Ch 13 (13.1)/Fig 13.1 sign.
3. **Notation/count unification** — ORd = 41, TTP06 = 18, R' = recovery var, Ch 4 buffer numbers, Ch 6 current count, Ch 5 "Figure 4.1"→5.2.

**Tier 2 — medium**
4. **App C.3 + C.10** numeric fixes; App D FFT ref; finish B.11.
5. **Figure drought** — add the ~10 highest-value SVGs (finding #2), prioritizing Ch 13 stencil, Ch 14 GS data-flow, Ch 15 decision tree, Ch 2 phase-plane.
6. **Ch 18** — add the "30-second preview" box; reorder 18.9/18.10/18.27/18.28; rewrite the garbled §18.5 closing paragraph.

**Tier 3 — larger / structural**
7. **Worked examples** — add L4 numeric examples to Ch 11 (BDF2), Ch 15, Ch 20, and at least one per Part I chapter.
8. **Ch 8 length** — split or add a reading-guide box (~1100 lines).
9. **§11.6** — expand PCG/Chebyshev/FFT beyond one paragraph each; link to App C.12–14.
10. **References** — add ionic/monodomain/FEM + the numerical/3B1B sources actually cited.

---

## Known-backlog status (vs 2026-03-08 audits)

| Prior issue | Status |
|-------------|--------|
| Ch 8 too long | STILL PRESENT (1102 lines) |
| FEM weak-form under-motivated | ✅ FIXED (§8.3 now has an ELI5 + IBP rationale) |
| Ch 10 stale figure/chapter refs | STILL PRESENT |
| Ch 11 no BDF2 worked example | STILL PRESENT |
| Ch 11 §11.6 too thin | STILL PRESENT |
| Ch 18 no "30-second preview" / theory-front-loaded | STILL PRESENT (§18.1 partially trimmed) |
| Ch 18 "Two Cases: rest vs flow" aside | STILL PRESENT |
| Ch 20 no worked example / no code | code ✅ ADDED; worked example STILL MISSING |
| Part III fictional monolithic architecture | ✅ RESOLVED (13b rewrite; sidebars only) |
| INDEX "Quadrature First v4.0" (Ch 18) | ❌ OVERSTATED — phrase absent from source |
| INDEX "Ω^NR/Ω^R notation" (Ch 19) | ✅ ACCURATE — present in §§19.2–19.4 |
