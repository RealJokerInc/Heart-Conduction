# PLAN: Textbook Audit Remediation

Created: 2026-07-02
Engine(s): None (documentation — HTML source under `website/chapters/`)
Research question: [textbook](README.md)
Source: [IDEALOG.md](IDEALOG.md) — the 2026-07-02 "Full chapter-by-chapter audit" + "Image/figure audit" thread entries
Audit references (single source of truth for exact locations — do NOT duplicate their tables here):
- [`audits/FULL_CHAPTER_AUDIT_2026-07-02.md`](audits/FULL_CHAPTER_AUDIT_2026-07-02.md)
- [`audits/IMAGE_AUDIT_2026-07-02.md`](audits/IMAGE_AUDIT_2026-07-02.md)

## Objective
Close the remediation backlog from the two 2026-07-02 audits of the cardiac-modeling textbook. Fix the correctness bugs (wrong math/physics), the book-wide off-by-one cross-reference/figure-number disease, the figure-integrity and figure-correctness defects, then the depth/structure gaps — all in the **canonical `website/chapters/*.html` source**. Order is highest-value / lowest-risk first (correctness + mechanical sweeps) → figure redraws → authoring → infrastructure.

## Success Criteria
- [ ] All 3 HIGH correctness bugs fixed (Ch 11 BDF2 stability, Ch 8 FDM arithmetic, Ch 13 block-matrix sign incl. Fig 13.1) + the MEDIUM numeric fixes (App C.3, C.10, Ch 3, Ch 20).
- [ ] Zero stale cross-references and zero mislabeled figure numbers remain (grep-clean per the checks below).
- [ ] Zero `$...$` LaTeX remains inside any `<svg><text>`; all figure labels render (verified visually).
- [ ] The 3 HIGH figure-correctness errors redrawn (Fig B.1, B.2, and Fig 13.1 signs) + the MEDIUM figure fixes.
- [ ] Notation/counts unified book-wide (ORd = 41, TTP06 = 18, R' = recovery var, Ch 4 buffers, Ch 6 current count).
- [ ] (Phase 4) Missing worked examples + §11.6 depth + highest-value new figures added; (Phase 5) PDF pipeline rebuilt and a correct PDF regenerated.
- [ ] Mirrors README completion criteria (Part II cross-ref/depth/length, Part IV restructure/numbering/depth, rebuild PDF pipeline, reconcile tracking-doc claims).
- [ ] No regressions — the multi-page site still loads every chapter; equations/figures still render.

## Architecture Changes
- MOD: `website/chapters/ch1.html … ch20.html`, `appendix-a/b/c/d.html`, `references.html` — content, cross-ref, figure, and SVG fixes per the audit docs.
- MOD (Phase 5): `Cardiac_Textbook_Website.html` (regenerated snapshot); `INDEX.md` (reconcile Part IV claims).
- NEW (Phase 5): `website/build/html_to_pdf.py` (or similar) — assemble `website/chapters/*` in `toc.json` order → single print HTML → Playwright → PDF (replaces the missing `html_to_pdf_v3.py`).
- Do NOT touch: `_archive/monolithic_pre-fork_2026-07-02/` (stale). `Monodomain/Engine_V5.3/` is read-only (N/A here but standing rule).

## Known Failures (from IDEALOG — do NOT repeat)
- **Editing `Bidomain_Textbook.html`** — it is the ARCHIVED, stale monolithic copy (deleted Part III Schur chapters, only 2 appendices). All edits go to `website/chapters/`.
- **Reintroducing the monolithic bidomain-solver framing** (2N×2N indefinite / FGMRES / AMG / Schur as the *real* method) — Engine V1 uses decoupled N×N SPD solves. Monolithic stays a labeled sidebar only.
- **Reference-manual style** — every revised section must keep L1 ELI5 + a L4 worked example, not just L2/L5.
- **Recombining appendices** into a catch-all — keep A/B/C/D one-job-each.
- **`$...$` LaTeX inside `<svg><text>`** — MathJax `tex-svg` does NOT typeset it; it renders literally. Use plain Unicode inside SVG (as Ch 6 / Ch 8 already do).

---

## Verification model (read once — applies to all phases)

There is **no pytest suite** for the HTML. Verification is:
1. **grep assertions** — specific stale strings must return zero (commands in each step).
2. **Visual render check** — open the multi-page site (`website/index.html`) which loads `chapters/` live; confirm the edited chapter renders (equations typeset, figures draw, labels legible). The PDF build is broken until Phase 5.
3. **Numeric re-derivation** — for the math fixes, recompute the corrected values (Python one-liners) and confirm they match the new prose.

Conda env for any Python: `conda run -n heart-conduction python ...`.

---

## Phase 1: Correctness fixes (HIGH + MEDIUM) + notation unification

**Goal**: Every place the book teaches something *wrong* is corrected. Independently deliverable; no figure redraws except Fig 13.1's signs (kept with its prose for atomicity).
**Tier**: medium
**Estimated scope**: 4 steps, ~8 chapter files touched, all small localized edits.

### Phase Context
Exact locations and the wrong-vs-right values are in `FULL_CHAPTER_AUDIT_2026-07-02.md` §"Content-correctness bugs" and `IMAGE_AUDIT_2026-07-02.md` §"Correctness". Read the specific bullet before each edit. These are surgical text edits — preserve surrounding prose voice. After each edit, re-derive the number to confirm.

### Step 1.1: Fix the three HIGH correctness bugs
**Model**: opus

#### Read First
- `FULL_CHAPTER_AUDIT_2026-07-02.md` bullets HIGH-1/2/3 — the exact wrong/right values.
- `website/chapters/ch11.html` — BDF2 section (ELI5, math box, comparison table).
- `website/chapters/ch8.html` — the FDM 5-node worked example (node-2 computation + the "node 4 only half" prose).
- `website/chapters/ch13.html` — the block-matrix prose table AND the Fig 13.1 SVG.

#### Why
These three actively teach incorrect facts (a wrong stability class, wrong arithmetic that breaks its own teaching point, a sign-inconsistent central equation). They are the highest-severity items and must be internally self-consistent after the fix.

#### Implementation Spec
**Ch 11 — BDF2 L-stability:** Make ELI5, math box, and comparison table all say **L-stable** (BDF2 is L-stable: dominant amplification-factor root → 0 as z → −∞). Replace the garbled `R(z)=(4−1/(1+2z/3))`, |R(∞)|=1/3 with a correct amplification statement (the two-step BDF2 update has |R|→0 as Re(z)→−∞). Keep the CN-vs-BDF2 contrast (CN is A-stable-not-L-stable; BDF2 is L-stable).
**Ch 8 — FDM worked example:** node-2 bracket is currently −190 → +475. Correct the bracket to `(-1)(-85)+2(-85)+(-1)(20) = -105`, giving **+262.5**. Fix the follow-on prose: with this K, nodes 2 and 4 are **equal (+262.5)** — replace the "node 2 gets +475 … node 4 only half" claim with the correct symmetric reading.
**Ch 13 — block-matrix sign:** In BOTH the prose block table and Fig 13.1's SVG `<text>`, change the off-diagonals A12 = A21 from **+L_i → −L_i** (with A22 = −(L_i+L_e) unchanged) so row 2 reads −(L_i+L_e)φ_e = +L_i·V_m, matching eq (14.1) / Algorithm 14.1.

#### Pseudocode
```
# Ch8 recompute to confirm:
node2 = 0.5 * ( (-1)*(-85) + 2*(-85) + (-1)*(20) )   # = 0.5 * (-105) = -52.5 ... (use the chapter's exact prefactor)
# Reconcile the printed prefactor so the worked result = +262.5 as the audit states; verify node2 == node4.
```
(Confirm the exact prefactor against the chapter; the audit's corrected target is +262.5 and node2 == node4.)

#### Test Spec
- Manual: after edit, `grep -n "475" ch8.html` → the stale +475 claim is gone; recomputed value matches prose.
- Manual: `grep -ni "not L-stable\|A-stable" ch11.html` in the BDF2 context → BDF2 no longer labeled A-stable/not-L-stable.

#### Checklist
- [ ] Ch 11 BDF2: ELI5 + math box + table all consistent on L-stable; amplification statement correct.
- [ ] Ch 8: node-2 arithmetic corrected; dependent prose ("node 4 half") rewritten to the symmetric truth.
- [ ] Ch 13: prose table A12/A21 → −L_i; Fig 13.1 SVG A12/A21 text → −L_i; both now match eq (14.1).

#### Verify
```bash
cd Research/Active/textbook/website/chapters
grep -n "475" ch8.html || echo "OK: stale +475 gone"
grep -ni "A-stable\|not L-stable" ch11.html
grep -n "L_i\|+ L\|- L" ch13.html   # eyeball the block-table + Fig 13.1 signs
```

#### Exit Criteria
- [ ] All three fixes internally self-consistent; recomputed numbers match prose; Fig 13.1 + Ch 13 table + eq (14.1) agree on signs.

#### Risk
Ch 8 prefactor ambiguity — mitigation: recompute from the chapter's own stencil definition, don't assume; make node2 == node4 the invariant. Ch 13 sign — mitigation: term-match against Algorithm 14.1 Stage 3 explicitly.

### Step 1.2: Appendix C numeric fixes
**Model**: opus

#### Read First
- `IMAGE_AUDIT`/`FULL_CHAPTER_AUDIT` MEDIUM bullets for App C.3 and C.10.
- `website/chapters/appendix-c.html` — Algorithm C.3 (Chebyshev) line 3; the C.10 DCT worked example.

#### Why
Both are reproducibility bugs in the "crown-jewel" appendix — a reader following the arithmetic gets a different answer than printed.

#### Implementation Spec
- **C.3 Chebyshev:** change `ω_{k+1}=1/(1−δ²ω_k/(2γ²))` → `/(4γ²)` (matches C.14's own prose `1/(1−ρ²ω_k/4)`).
- **C.10 DCT:** recompute the orthonormal DCT-II of the running vector (the audit found [4,2,1,3] → printed [5.000, 1.577, 0.500, −0.224] is wrong). Replace with the correctly computed coefficients and confirm the reader can reproduce them.

#### Pseudocode
```python
import numpy as np; from scipy.fft import dct
print(dct([4,2,1,3], type=2, norm='ortho'))  # use these values in C.10
```

#### Test Spec
- Recompute both; printed values must equal the computed ones.

#### Checklist
- [ ] C.3 denominator fixed (2γ²→4γ²); ω₂ still correct, k≥2 now correct.
- [ ] C.10 coefficients replaced with reproducible values (state the norm convention used).

#### Verify
```bash
conda run -n heart-conduction python -c "from scipy.fft import dct; print(dct([4,2,1,3],type=2,norm='ortho'))"
```

#### Exit Criteria
- [ ] Both examples reproduce by hand/Python exactly.

#### Risk
DCT normalization mismatch — mitigation: state which convention (ortho vs unnormalized) the chapter uses and make the numbers match THAT convention.

### Step 1.3: Minor physics / equation corrections
**Model**: opus

#### Read First
- Audit MEDIUM bullets for Ch 3 (I_K1), Ch 20 (eq 20.2), Ch 18 (garbled closing paragraph), Ch 19 (§19.5 D2Q9-vs-D2Q5).

#### Implementation Spec
- **Ch 3:** "large outward current at voltages just **below** E_K" → "just **above** E_K" (outward K⁺ flows above E_K).
- **Ch 20:** eq 20.2 — replace "1/2" with "Δt/2" to match 19.6/18.26; fix the tensor-fed-into-scalar form.
- **Ch 18:** rewrite the garbled closing paragraph (~line 1294: "In the next chapter… In Section 18.5 above… moment space, —") into a clean transition to Ch 19.
- **Ch 19 §19.5:** reconcile "the D2Q9 lattice used in this chapter" — the chapter uses D2Q5; fix the lattice name and the index convention so §19.5 matches the rest of Ch 19.

#### Checklist
- [ ] Ch 3 I_K1 direction fixed.
- [ ] Ch 20 eq 20.2 corrected.
- [ ] Ch 18 closing paragraph rewritten.
- [ ] Ch 19 §19.5 lattice/index reconciled to D2Q5.

#### Verify
```bash
grep -ni "just below E_K\|below \$E_K" ch3.html || echo "OK"
grep -ni "D2Q9" ch19.html   # confirm remaining D2Q9 mentions are intentional (Ch18 context) only
```

#### Exit Criteria
- [ ] Each corrected; no new inconsistency introduced.

#### Risk
Ch 19 lattice reconciliation could ripple — mitigation: confirm which lattice the chapter's equations actually use before renaming.

### Step 1.4: Notation & count unification (book-wide)
**Model**: opus

#### Read First
- Audit LOW/"notation" bullets; STYLE_GUIDE §"Cross-Reference Consistency" (canonical counts: TTP06 = 18 = 12 gates + 5 conc + R'; ORd = 41).

#### Implementation Spec
- **ORd count:** "40" in Ch 2 and Ch 7 → 41 (or adopt Ch 6's reconciled phrasing "40 ionic + V stored separately" consistently everywhere).
- **TTP06 count:** "17" in Ch 8 → 18; keep 18 everywhere.
- **Ch 5:** stop calling R' a "concentration" → "5 concentrations + R' (SR-release recovery variable)".
- **Ch 4 buffers:** make Fig 4.1, Fig 4.5, and §4.6 agree — ATP/Mg²⁺ ≈ 15%, Other ≈ 38% (Fig 4.1 currently says "SR membranes 15% / Other 37%").
- **Ch 6 current count:** reconcile "15" vs "16" vs "15 (+chloride)" — pick the count matching eq (6.1)'s terms and state the chloride caveat once.

#### Checklist
- [ ] ORd 41 consistent (Ch 2, 7 vs 6).
- [ ] TTP06 18 consistent (Ch 8).
- [ ] Ch 5 R' recategorized.
- [ ] Ch 4 buffer numbers unified across Fig 4.1 / Fig 4.5 / §4.6.
- [ ] Ch 6 current count unified.

#### Verify
```bash
cd Research/Active/textbook/website/chapters
grep -rni "40 (state\|40 variables\|40 ionic" ch2.html ch7.html
grep -rni "17 state\|17 ionic" ch8.html
```

#### Exit Criteria
- [ ] No conflicting state-counts or buffer numbers remain across chapters.

#### Risk
Over-eager replace hitting unrelated "40"/"17" — mitigation: match the full phrase, edit per-occurrence.

### Phase 1 Verification
```bash
cd Research/Active/textbook/website/chapters
grep -n "475" ch8.html || echo "ch8 OK"
grep -ni "A-stable" ch11.html    # BDF2 no longer mislabeled
conda run -n heart-conduction python -c "from scipy.fft import dct; print(dct([4,2,1,3],type=2,norm='ortho'))"
```
Open `website/index.html`, spot-check Ch 8, 11, 13, App C render with the new values.

### Phase 1 Exit Criteria
- [ ] 3 HIGH + 4 MEDIUM correctness fixes done and self-consistent.
- [ ] Notation/counts unified.
- [ ] Site still renders the edited chapters.

### Phase 1 Cleanup
- No `$...$` introduced inside any SVG (Fig 13.1 edit uses plain Unicode).
- Canonical source only — no edits leaked into the archived file.
- Re-derived numbers match prose.

**-> Commit point: git commit after Phase 1 passes**

---

## Phase 2: Book-wide cross-reference + figure-number sweep

**Goal**: Zero stale cross-references and zero mislabeled figure numbers. The single highest-value connectiveness fix.
**Tier**: medium
**Estimated scope**: 2 steps, mechanical but must avoid clobbering legitimate refs.

### Phase Context
The exact stale-ref tables are in `FULL_CHAPTER_AUDIT` §"Cross-cutting finding #1" and `IMAGE_AUDIT` §"finding #2" — treat them as the checklist. DANGER: legitimate "Chapter 7/8/9/10" refs also exist; match the **full phrase** from the audit table, not the bare number. Do per-file, verify each with grep after.

### Step 2.1: Parts I–II sweep (ch5, ch6, ch8, ch10, ch11, appendix-d)
**Model**: opus

#### Read First
- `FULL_CHAPTER_AUDIT` cross-ref table rows for Ch 5/6/8/10/11 + App D; `IMAGE_AUDIT` figure-number rows for Ch 5/8/10.

#### Implementation Spec (from the audit tables)
- Ch 5, Ch 6: Rush–Larsen "(Chapter 8)" → "(Chapter 9)".
- Ch 8: "Figure 7.1/7.2/7.3" → 8.x; "sections 7.5–7.7" → 8.2–8.4; "Godunov splitting (Chapter 8)" → Ch 9; "Chapters 9 and 10 … time integrators" → "10 and 11".
- Ch 10: "Chapter 10's implicit methods" / "(Chapter 10)" ×2 → Ch 11; "Figure 9.1/9.2/9.3" → 10.x; "Chapter 7" (spatial) → Ch 8.
- Ch 11: "Explicit methods (Chapter 9)" → Ch 10; "CFL (equation 9.2)" → "(10.2)"; "Chapter 7" (K/M matrices) → Ch 8.
- App D: "The FFT solver (Chapter 10)" → "Chapter 11 §11.6".
- Ch 5 figure: caption "Figure 4.1" (calcium cycling) → "Figure 5.2".

#### Checklist
- [ ] Each row above applied; each verified by grep.

#### Verify
```bash
cd Research/Active/textbook/website/chapters
grep -ni "Chapter 8" ch5.html ch6.html | grep -i "rush\|larsen" || echo "OK: RL ref fixed"
grep -n "Figure 7\." ch8.html || echo "ch8 fig OK"
grep -n "Figure 9\." ch10.html || echo "ch10 fig OK"
grep -ni "equation 9.2\|(9.2)" ch11.html || echo "ch11 CFL OK"
grep -n "Figure 4.1" ch5.html || echo "ch5 fig relabel OK"
```

#### Exit Criteria
- [ ] All Parts I–II stale prose refs + figure numbers resolved; grep-clean.

#### Risk
Clobbering a legitimate "Chapter 8/9/10" — mitigation: phrase-match; re-read each hit in context before editing.

### Step 2.2: Part IV sweep (ch18, ch19, ch20)
**Model**: opus

#### Read First
- `FULL_CHAPTER_AUDIT` cross-ref rows for Ch 18/19/20; `IMAGE_AUDIT` Ch 18 figure row. Ch 19 is the worst (24 bad refs).

#### Implementation Spec (from the audit tables)
- Ch 18: "equation 17.2/17.3/17.4/17.14" → 18.x; figure caption "Figure 17.1" → "18.1".
- Ch 19: the 16 "Chapter 17"/"17.x" → 18.x; the 8 "18.x" that are really this chapter's own equations → 19.x (per the audit's explicit mapping 18.1→19.1, 18.2→19.2, 18.10→19.10, 18.3→19.3, 18.8→19.8); "Chapter 19 develops three strategies" → "Chapter 20".
- Ch 20: "monodomain equation (Chapter 18)" → Ch 19; "(equation 17.26)" → "18.26"; "Part IV (Chapters 17–19)" → "18–20".

#### Checklist
- [ ] Ch 18 five 17.x refs + figure number fixed.
- [ ] Ch 19 all 24 refs fixed per the mapping (17→18 for foreign eqs; 18→19 for own eqs).
- [ ] Ch 20 three refs fixed.

#### Verify
```bash
cd Research/Active/textbook/website/chapters
grep -n "17\.\|Chapter 17\|Figure 17" ch18.html ch19.html ch20.html || echo "OK: no stale 17.x"
grep -n "Chapter 19 develop" ch19.html || echo "ch19 self-ref OK"
grep -n "Chapters 17" ch20.html || echo "ch20 range OK"
```

#### Exit Criteria
- [ ] Zero "17.x"/"Chapter 17" anywhere in Part IV; self-refs point to 19.x; end-of-Ch19 points to Ch 20.

#### Risk
Ch 19's dual mapping (some refs go 17→18, others 18→19) — mitigation: follow the audit's explicit per-equation mapping; verify each of the 8 self-refs individually.

### Phase 2 Verification
```bash
cd Research/Active/textbook/website/chapters
echo "--- residual stale patterns (should all be empty) ---"
grep -n "Figure 7\." ch8.html; grep -n "Figure 9\." ch10.html
grep -n "Chapter 17\|Figure 17\|17\.2\|17\.26" ch18.html ch19.html ch20.html
grep -ni "rush.*Chapter 8" ch5.html ch6.html
```
Open the site; click through Ch 5→6, 8, 10→11, 18→19→20 and confirm every cross-ref/figure number resolves.

### Phase 2 Exit Criteria
- [ ] All grep checks empty; figure numbers monotonic per chapter; no duplicate "Figure 4.1".
- [ ] Site renders; no broken in-text links.

### Phase 2 Cleanup
- Confirm no legitimate ref was clobbered (spot-check 5 random surviving "Chapter N" refs resolve correctly).

**-> Commit point: git commit after Phase 2 passes**

---

## Phase 3: Figure integrity + correctness redraws

**Goal**: Every figure renders (no literal LaTeX), fits its viewBox, and depicts correct math/science.
**Tier**: medium
**Estimated scope**: 3 steps over ~8 SVGs. Follow `SVG_FIGURES_SKILL.md`.

### Phase Context
Per-figure defects are in `IMAGE_AUDIT_2026-07-02.md` (per-figure table + fix backlog). RULE: all new/edited SVG text uses **plain Unicode** (`I_Na`, `k₁`, `φ_e`), never `$...$`. Preserve each SVG's viewBox; keep content inside bounds. After editing, open the site and eyeball the figure.

### Step 3.1: LaTeX-in-SVG → plain Unicode sweep
**Model**: opus

#### Read First
- `IMAGE_AUDIT` systemic finding #1 (confirmed-broken: Fig 5.1, Ch 10 ×3; at-risk: appendix labels). Ch 6/Ch 8 SVGs as the plain-Unicode reference pattern.

#### Why
MathJax `tex-svg` cannot typeset math inside `<svg><text>`, so these labels currently show raw `$...$`. Plain Unicode is the proven fix already used elsewhere in the book.

#### Implementation Spec
Find every `$...$` inside an `<svg>…</svg>` block and replace with Unicode: subscripts (₀₁₂ₑ), Greek (φ ω γ δ ρ λ Δ), `≈ ≤ ×`, italic vars as plain letters. Files: ch5.html (Fig 5.1), ch10.html (10.1/10.2/10.3), and any appendix SVG using `$` (appendix-a/b/c).

#### Pseudocode
```python
# detection: for each file, extract <svg>..</svg> blocks and flag any containing '$'
import re,glob
for f in glob.glob('website/chapters/*.html'):
    s=open(f).read()
    for m in re.findall(r'<svg.*?</svg>', s, re.S):
        if '$' in m: print(f, 'has $ in SVG')
```

#### Test Spec
- After edits, the detection script prints nothing.

#### Checklist
- [ ] Fig 5.1 current labels → Unicode.
- [ ] Ch 10 Figs axis/legend/stage labels → Unicode.
- [ ] Any appendix SVG `$` labels → Unicode.

#### Verify
```bash
cd Research/Active/textbook
conda run -n heart-conduction python -c "
import re,glob
bad=[f for f in glob.glob('website/chapters/*.html') for m in re.findall(r'<svg.*?</svg>',open(f).read(),re.S) if '\$' in m]
print('SVGs with \$:', bad or 'NONE')"
```
Open the site; confirm the labels typeset as real symbols.

#### Exit Criteria
- [ ] Zero `$` inside any SVG; all affected labels render.

#### Risk
Losing a subscript/Greek nuance in translation — mitigation: map against the equation the label references.

### Step 3.2: HIGH figure-correctness redraws (Fig B.1, Fig B.2)
**Model**: opus

#### Read First
- `IMAGE_AUDIT` HIGH-2 (Fig B.2) and HIGH-3 (Fig B.1). (Fig 13.1's signs were fixed in Step 1.1.)

#### Implementation Spec
- **Fig B.2 (appendix-b):** redraw so both surfaces curve **UP** — positive-definite paraboloid with the unique minimum at the BOTTOM; the semi-definite trough shows the flat/degenerate direction (a valley) with the min-line at the bottom. Currently both are concave-down domes.
- **Fig B.1 (appendix-b):** redraw the parallelogram so ê₂ → (1,1) to match the labeled matrix (2 1; 0 1) (currently ê₂ → (0.625,1), shear under-drawn ~37%). Move the in-`<text>` `\begin{pmatrix}` label out to plain Unicode or a caption.

#### Test Spec
- Geometry check: B.1 top-right vertex at ê₁+ê₂ = (2,0)+(1,1) = (3,1) in figure coords; B.2 min y at the bottom, curvature opens upward.

#### Checklist
- [ ] Fig B.2 concave-up, min at bottom.
- [ ] Fig B.1 shear matches its matrix; matrix label rendered as Unicode/caption.

#### Verify
Open the site; visually confirm the bowl opens up and the parallelogram shear matches (2 1; 0 1).

#### Exit Criteria
- [ ] Both figures depict the stated math correctly.

#### Risk
SVG path math is fiddly — mitigation: recompute vertex coordinates from the matrix before drawing; keep within viewBox.

### Step 3.3: MEDIUM figure fixes (Fig 4.5, 4.1, 3.1, 1.2, 4.4)
**Model**: opus

#### Read First
- `IMAGE_AUDIT` MEDIUM bullets 4–8.

#### Implementation Spec
- **Fig 4.5:** stacked-bar heights and pie-wedge angles proportional to the stated percentages (35/15/12/38; free-Ca ≈ 1–2% → ~4–7°).
- **Fig 4.1:** buffer legend → ATP/Mg²⁺ 15% / Other 38% (match §4.6 + Fig 4.5); move the x=470 labels inside the width-500 viewBox (or widen viewBox).
- **Fig 3.1:** linear, correctly-spaced voltage and time axes (equal ms per tick).
- **Fig 1.2:** reconcile the annotated resting values with the drawn curves (either move the V_rest marker or relabel to the drawn h_inf≈0.95 / n_inf≈0.04 at cardiac rest).
- **Fig 4.4:** flip the two "OUT" arrows to point outward across the membrane.

#### Checklist
- [ ] 4.5 proportional; 4.1 legend + in-bounds; 3.1 linear axes; 1.2 consistent; 4.4 arrows outward.

#### Verify
Open the site; measure/eyeball each corrected figure against its caption/percentages.

#### Exit Criteria
- [ ] Each figure quantitatively matches its labels/text.

#### Risk
Fig 4.1/4.5 must agree with the Step 1.4 buffer-number unification — mitigation: do 1.4 first; use the same numbers here.

### Phase 3 Verification
```bash
cd Research/Active/textbook
conda run -n heart-conduction python -c "
import re,glob
print('SVGs with \$:', [f for f in glob.glob('website/chapters/*.html') for m in re.findall(r'<svg.*?</svg>',open(f).read(),re.S) if '\$' in m] or 'NONE')"
```
Open the site; walk every edited figure.

### Phase 3 Exit Criteria
- [ ] Zero `$`-in-SVG; Fig B.1/B.2/13.1 correct; MEDIUM figures proportional/in-bounds; all render.

### Phase 3 Cleanup
- All SVG text is Unicode; every edited SVG stays within its viewBox.

**-> Commit point: git commit after Phase 3 passes**

---

## Phase 4: Depth & structure (Tier 3 authoring)

**Goal**: Close the pedagogical-depth gaps — worked examples, thin sections, missing figures, references. This is authoring, so steps are specified by intent + acceptance criteria rather than exact diffs.
**Tier**: large (worth `/audit` before executing)
**Estimated scope**: 4 steps; new prose + new SVGs. Each new/revised section MUST carry L1 ELI5 + L4 worked example (Known Failure: no reference-manual style).

### Phase Context
Follow STYLE_GUIDE (Feynman 5-layer) and SVG_FIGURES_SKILL. New SVGs use plain Unicode labels. Do not reintroduce monolithic-solver framing in any bidomain section.

### Step 4.1: Add missing worked examples (L4)
**Model**: opus
- **Ch 11:** a numeric BDF2 worked example (two-history-level RHS on a small grid).
- **Ch 15:** a tiny elliptic-solve worked example (don't just defer to Ch 11).
- **Ch 20:** replace the §20.4 "outline that computes nothing" with a real numeric bidomain-LBM step.
- **Part I:** at least one L4 example in the weakest chapters (e.g., a Nernst/Rush–Larsen gate update in Ch 1; an FHN excitation-cycle trace in Ch 2).
- **Verify:** each new example is reproducible by hand; numbers stated.
- **Exit:** the "missing worked example" audit items for Ch 11/15/20 + Part I closed.

### Step 4.2: Expand §11.6 + address Ch 8 length
**Model**: opus
- **§11.6:** expand PCG / Chebyshev / FFT beyond one paragraph each (motivation + when-to-use + cost); add cross-links to App C.12–C.14.
- **Ch 8:** add a "reading guide" box at the top (or split FVM/BC into a clearly-marked sub-part) to tame the ~1100-line length.
- **Exit:** §11.6 no longer one-paragraph-each; Ch 8 has a navigational on-ramp.

### Step 4.3: Add the highest-value missing figures
**Model**: opus
- Priority SVGs (from `FULL_CHAPTER_AUDIT` finding #2): Ch 13 face-based-stencil; Ch 14 Gauss-Seidel data-flow (φe→parabolic→Vm→elliptic); Ch 15 three-tier decision tree; Ch 2 FHN phase-plane; Ch 11 L-stable-damping vs CN-ringing; Ch 18 Gaussian/bell-curve; Ch 19 bounce-back reflection; Ch 20 dual-lattice architecture; App C.12 CG-vs-gradient zigzag.
- **Constraint:** plain-Unicode labels; correct math (unlike the B.1/B.2 originals).
- **Exit:** the eight zero-figure chapters each gain ≥1 figure; the priority list is drawn.

### Step 4.4: Ch 18 preview box + equation reorder + References
**Model**: opus
- **Ch 18:** add the 10-line "30-second preview" (6-step collide-stream loop) before §18.1; reorder the out-of-sequence equation defs (18.9/18.10 after 18.11/18.12; 18.27; 18.28).
- **References:** add the numerical-analysis + 3Blue1Brown sources actually cited in App B/C, plus ionic-model (TTP06/ORd) and monodomain/FEM refs; verify the footnote folder paths exist.
- **Exit:** Ch 18 opens with a preview; equations monotonic; References cover all parts.

### Phase 4 Verification
Open the site; confirm new examples render, new figures draw (Unicode labels), §11.6 expanded, Ch 18 preview present.

### Phase 4 Exit Criteria
- [ ] Worked examples added (Ch 11/15/20 + Part I); §11.6 expanded; ≥1 figure in each formerly-zero-figure chapter; Ch 18 preview + reorder; References expanded.
- [ ] Every revised section keeps L1 + L4 (no reference-manual regressions).

### Phase 4 Cleanup
- No `$`-in-SVG in any new figure; no monolithic-solver framing introduced; appendices stay A/B/C/D.

**-> Commit point: git commit after Phase 4 passes**

---

## Phase 5: Infrastructure — PDF pipeline + tracking-doc reconciliation

**Goal**: Restore a working PDF build from the website source and correct the overstated INDEX claims.
**Tier**: large
**Estimated scope**: 2 steps; new build script + doc edits + regenerated deliverables.

### Phase Context
`html_to_pdf_v3.py` is missing. The `/textbook-compile` skill documents the intended pipeline. Canonical source = `website/chapters/` in `toc.json` order.

### Step 5.1: Rebuild the PDF build script
**Model**: opus
- Write `website/build/html_to_pdf.py`: read `website/toc.json` for order → concatenate `website/chapters/*.html` fragments into one print HTML with the print `<head>`/MathJax/CSS (crib from `Cardiac_Textbook_Website.html` or `website/standalone.html`) → Playwright headless Chrome, wait for MathJax typeset → `page.pdf(...)` → `Cardiac_Computational_Modeling.pdf`.
- **Verify:** PDF builds; page count sane; spot-check Part III shows Ch 12–15 (not the archived Schur Ch 16/17) and appendices A/B/C/D.
- **Exit:** a correct, current PDF regenerated; `/textbook-compile` unblocked.

### Step 5.2: Reconcile INDEX Part IV claims + regenerate snapshot
**Model**: opus
- **INDEX.md:** remove/soften the "Quadrature First v4.0" claim for Ch 18 (phrase absent from source); keep "Ω^NR/Ω^R" for Ch 19 (confirmed present). Update the verification/status tables to reflect the 2026-07-02 audit.
- **Regenerate** `Cardiac_Textbook_Website.html` snapshot from the edited chapters.
- **Exit:** INDEX matches reality; whole-book render current.

### Phase 5 Verification
```bash
cd Research/Active/textbook
conda run -n heart-conduction python website/build/html_to_pdf.py   # builds PDF
conda run -n heart-conduction python -c "from pypdf import PdfReader; print(len(PdfReader('Cardiac_Computational_Modeling.pdf').pages),'pages')"
```

### Phase 5 Exit Criteria
- [ ] PDF builds from website source and shows current content; INDEX reconciled; snapshot regenerated.

### Phase 5 Cleanup
- New build script documented in `/textbook-compile`; no reference to the archived file in the build path.

**-> Commit point: git commit after Phase 5 passes**

---

## Final Cleanup (cross-phase de-sloppify)

- [ ] No `$...$` inside any `<svg><text>` anywhere in `website/chapters/`.
- [ ] No stale "Chapter 7/9/10/17" or "Figure 7.x/9.x/17.x" refs; figure numbers monotonic; no duplicate "Figure 4.1".
- [ ] State counts consistent book-wide (TTP06 = 18, ORd = 41).
- [ ] No edits leaked into `_archive/monolithic_pre-fork_2026-07-02/`.
- [ ] No monolithic bidomain-solver framing introduced; appendices remain A/B/C/D.
- [ ] Re-run the full-audit grep checks (Phases 1–3 Verify blocks) — all clean.
- [ ] Update `CHANGELOG.md` with the remediation entry; update `KNOWLEDGE.md` audit-backlog status (mark closed items); tick README completion criteria.

1. Archive the completed plan:
```bash
mkdir -p Research/Active/textbook/plans
cp Research/Active/textbook/PLAN.md "Research/Active/textbook/plans/$(date +%Y-%m-%d)_textbook-audit-remediation.md"
```

2. Revert the bottom tmux pane from PLAN.md back to WHITEBOARD.md:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/textbook/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/textbook/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Mutation Log
_(populated during execution: `**MUTATED {date}**: Step X.Y {SKIPPED|SPLIT|INSERTED} — {reason}`)_
