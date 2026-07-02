# Image / Figure Audit — 2026-07-02

**Scope:** all **31 inline SVG figures** in the canonical website source (`website/chapters/*.html`). There are **zero `<img>` raster images and zero external image files** — every figure is hand-coded inline SVG, so there are no broken-link risks. Eight chapters have no figures (Ch 2, 7, 9, 11, 14, 15, 19, 20).
**Dimensions:** INTEGRITY (well-formed, in-bounds, renders, correctly numbered) · CORRECTNESS (accurately depicts the science/math; proportions, axes, values, arrows match the text).
**Method:** 4 parallel figure-audit agents inspecting each SVG's markup (viewBox, coordinates, paths, text, arc angles) against its caption and referencing prose.

---

## Systemic integrity findings (ranked)

### 1. LaTeX-in-SVG labels don't render — cross-cutting, HIGHEST priority
Several figures put `$...$` LaTeX **inside `<svg><text>`** elements. The site's MathJax 3 `tex-svg` config (default `skipHtmlTags`, no `svg` handling) does **not** typeset math inside SVG `<text>` — browsers cannot place an HTML math container there — so those labels render as **literal `$I_{\text{Na}}$` / `$k_1$` / `$\begin{pmatrix}...$`**.
- **Confirmed broken:** Fig 5.1 (TTP06 map, 12 current labels), Ch 10 Figs 10.1/10.2/10.3 (RK2/RK4/stability — axis + legend labels).
- **At risk (verify in compiled output):** appendix figures A.1, B.1 (worst — a 2×2 `\begin{pmatrix}` stuffed into one text line), and others using `$...$`.
- **Fix (known — the authors already do it elsewhere):** Ch 6 and Ch 8 use plain Unicode text (`I_Na`, `k₁`) and render perfectly. Replace `$...$` in SVG `<text>` with plain Unicode across all figures, then confirm in the Playwright render.

### 2. Figure-number mislabels — the off-by-one renumber bug, in the figures too
Eight figures carry the wrong number (same disease as the prose cross-refs):

| File | Printed label | Should be |
|------|---------------|-----------|
| ch5.html §5.8 (calcium cycling) | **"Figure 4.1"** | **5.2** — and it **duplicates** Ch 4's real Fig 4.1 |
| ch8.html (5-node cable) | "Figure 7.1" | 8.1 |
| ch8.html (5/9-pt stencils) | "Figure 7.2" | 8.2 |
| ch8.html (FEM mesh) | "Figure 7.3" | 8.3 |
| ch10.html (RK2 slopes) | "Figure 9.1" | 10.1 |
| ch10.html (RK4 stages) | "Figure 9.2" | 10.2 |
| ch10.html (stability regions) | "Figure 9.3" | 10.3 |
| ch18.html (lattice stencils) | "Figure 17.1" | 18.1 |

### 3. Content overflow / clipping
- **Fig 4.1** (dyadic cleft): two labels placed at x=470 in a width-500 viewBox → they overflow the right edge and clip.

---

## Correctness findings (figure depicts wrong science/math)

### HIGH — the drawing is mathematically wrong
1. **Fig 13.1 — bidomain block-system signs (ch13).** Off-diagonals drawn **+L_i** with A22 = −(L_i+L_e) → row 2 reads −(L_i+L_e)φ_e = **−**L_i·V_m, contradicting eq (14.1)/Algorithm 14.1's **+**L_i·V_m. Off-diagonals must be **−L_i**. The **Ch 13 prose block table repeats the identical error** — figure + table agree with each other but both contradict Ch 14. (Same defect the chapter audit flagged; now confirmed in the figure.)
2. **Fig B.2 — SPD bowl vs semi-definite trough (appendix-b).** Both surfaces are drawn **concave-DOWN (domes)**, with the "unique minimum" dot at the apex of a dome. A positive-definite paraboloid must curve **UP** to a minimum at the bottom — the geometry is inverted and contradicts the "curves up in all directions" claim it's illustrating.
3. **Fig B.1 — unit-square → parallelogram (appendix-b).** The drawn shear encodes ê₂ → (0.625, 1), but the stated/labeled matrix (2 1; 0 1) requires ê₂ → (1, 1). The shear is under-drawn ~37%; the figure contradicts its own matrix. (Plus the oversized in-`<text>` pmatrix label from finding #1.)

### MEDIUM — quantitative fidelity / physics slips
4. **Fig 4.5 — calcium buffering (ch4).** The stacked bar is grossly non-proportional: the largest pool (Other, 38%) is drawn **shortest** (~18%); 15% and 12% get equal bars. The pie's "free Ca" wedge subtends ~17° for a labeled **1–2%**. Also disagrees with Fig 4.1 (below).
5. **Fig 4.1 — dyadic cleft buffer legend (ch4).** Legend says "SR membranes (~15%) · Other (~37%)", but §4.6 prose **and** Fig 4.5 say the 15% pool is **ATP/Mg²⁺** and Other is **38%**. Values sum to 99%.
6. **Fig 3.1 — five-phase AP (ch3).** Both axes are non-linearly spaced/mislabeled: voltage ticks −85/−20/+20/+40 sit at unequal spacings (top of scale stretched ~3×); time ticks read 0/50/150/250/350 at equal spacing (first interval 50 ms, rest 100 ms).
7. **Fig 1.2 — steady-state gating curves (ch1).** Curve shapes are right, but the annotated resting values (h_inf≈0.6, n_inf≈0.3 — squid −65 mV numbers) don't match the drawn curves at the marker (h_inf≈0.95, n_inf≈0.04). Marker, curves, and caption are mutually inconsistent.
8. **Fig 4.4 — NCX bidirectional (ch4).** Stoichiometry and net-current text are correct, but the two "OUT" arrows (forward Ca²⁺-out, reverse Na⁺-out) point **downward into the cytoplasm** instead of out across the membrane.

### LOW — cosmetic / schematic quibbles
- **Fig 12.1** — "D_eff = harmonic mean" label is loose (the chapter's D_eff = D_i·D_e/(D_i+D_e) is half the true harmonic mean).
- **Fig A.4** — split-wave copies drawn ~58% amplitude rather than ½.
- **Figs 10.1/10.2** — minor slope-ordering/placement quibbles vs the concave true curve.
- **Fig 1.3** — axis floors at −80 mV while prose quotes V_rest ≈ −85 mV (figure otherwise correctly squid-like, matching its caption).

---

## Per-figure verdict table (all 31)

| Figure (printed → correct) | Chapter | Integrity | Correctness |
|---|---|---|---|
| Fig 1.1 HH circuit | ch1 | OK | OK |
| Fig 1.2 gating curves | ch1 | OK | ISSUE (annotations vs curves) |
| Fig 1.3 nerve AP | ch1 | OK | OK (minor axis floor) |
| Fig 3.1 five-phase AP | ch3 | OK | ISSUE (non-scale axes) |
| Fig 4.1 dyadic cleft | ch4 | ISSUE (overflow + dup #) | ISSUE (buffer label) |
| Fig 4.2 spark→wave | ch4 | OK | OK |
| Fig 4.3 SERCA/PLB | ch4 | OK | OK |
| Fig 4.4 NCX | ch4 | OK | ISSUE (OUT arrows) |
| Fig 4.5 buffering | ch4 | OK | ISSUE (non-proportional) |
| Fig 5.1 TTP06 map | ch5 | ISSUE (LaTeX-in-SVG) | OK |
| Fig 4.1→**5.2** calcium cycling | ch5 | ISSUE (mislabel/dup) | OK |
| Fig 6.1 ORd map | ch6 | OK | OK |
| Fig 7.1→**8.1** 5-node cable | ch8 | ISSUE (mislabel) | OK |
| Fig 7.2→**8.2** FDM stencils | ch8 | ISSUE (mislabel) | OK |
| Fig 7.3→**8.3** FEM mesh | ch8 | ISSUE (mislabel) | OK |
| Fig 9.1→**10.1** RK2 | ch10 | ISSUE (mislabel + LaTeX) | OK (cosmetic) |
| Fig 9.2→**10.2** RK4 | ch10 | ISSUE (mislabel + LaTeX) | OK (cosmetic) |
| Fig 9.3→**10.3** stability | ch10 | ISSUE (mislabel + LaTeX) | OK |
| Fig 12.1 mono vs bidomain | ch12 | OK | OK (minor label) |
| **Fig 13.1 block system** | ch13 | OK | **ISSUE (HIGH: sign)** |
| Fig 17.1→**18.1** lattice stencils | ch18 | ISSUE (mislabel) | OK |
| Fig A.1 heat decay | appendix-a | OK (LaTeX caveat) | OK |
| Fig A.2 Laplacian detector | appendix-a | OK | OK |
| Fig A.3 rubber sheet | appendix-a | OK | OK |
| Fig A.4 wave split | appendix-a | OK | OK (minor amplitude) |
| **Fig B.1 transformation** | appendix-b | ISSUE (matrix label) | **ISSUE (HIGH: shear ≠ matrix)** |
| **Fig B.2 SPD bowl/trough** | appendix-b | OK | **ISSUE (HIGH: inverted curvature)** |
| Fig B.3 condition contours | appendix-b | OK | OK |
| Fig C.1 grid + stencil | appendix-c | OK | OK |
| Fig C.2 FE heatmap | appendix-c | OK | OK |
| Fig C.3 stability regions | appendix-c | OK | OK |

**Tally:** Integrity — 19/31 clean, 12 with issues (8 mislabeled numbers, 4+ LaTeX-render, 1 overflow; several overlap). Correctness — 23/31 clean, 8 with issues, of which **3 are HIGH** (Fig 13.1, B.2, B.1). The best figures are App C (C.1–C.3 match their worked examples exactly) and the D2Q5/D2Q9/D3Q7 lattice stencils (all velocity counts/weights correct).

---

## Fix backlog

**Tier 1 — cheap, high-value**
1. **Figure-number sweep** — fix the 8 mislabels (Ch 5 4.1→5.2, Ch 8 7.x→8.x, Ch 10 9.x→10.x, Ch 18 17.1→18.1); resolves the duplicate "Figure 4.1". Fold into the book-wide cross-reference sweep.
2. **LaTeX-in-SVG sweep** — replace `$...$` in every `<svg><text>` with plain Unicode (the Ch 6/Ch 8 pattern); verify in the compiled render. Fixes ~4 confirmed-broken + the at-risk appendix labels.
3. **Fig 13.1 sign** — redraw off-diagonals as −L_i (and fix the matching Ch 13 prose table) — same fix as the chapter-audit HIGH bug.

**Tier 2 — redraws (correctness)**
4. **Fig B.2** — flip the bowl/trough to concave-UP with the minimum at the bottom.
5. **Fig B.1** — redraw the parallelogram so ê₂ → (1,1), matching its matrix.
6. **Fig 4.5** — make the stacked bar and pie proportional to the stated percentages.
7. **Fig 4.1** — fix the buffer legend (ATP/Mg²⁺ 15% / Other 38%), and move the x=470 labels inside the viewBox.
8. **Fig 3.1** — linear, correctly-labeled axes.
9. **Fig 1.2** — reconcile the resting-value annotations with the drawn curves. **Fig 4.4** — flip the two "OUT" arrows outward.

**Tier 3 — cosmetic:** Fig 12.1 label wording, Fig A.4 amplitude, Fig 10.1/10.2 slope ordering.

**Cross-reference:** the figure findings reinforce two chapter-audit items — the Fig 13.1 sign bug (now confirmed in figure + prose table + eq mismatch) and the book-wide off-by-one renumbering (now shown to affect figure numbers, not just cross-refs).
