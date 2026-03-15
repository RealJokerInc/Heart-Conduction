# Experimental Evidence: Boundary CV in Cardiac Tissue — An Honest Assessment

## Critical Finding

**The direct experiment has not been done.** No published study measures spatially-resolved conduction velocity along the edge of a clean insulating obstacle in cardiac tissue and compares it to CV in the interior.

This document catalogs what evidence *does* exist, with honest caveats about what each study proves and what it does not.

---

## The Experimental Gap

The experiment we need: optical mapping (≤50 μm/pixel) of a planar wavefront propagating past a photolithographically-defined rectangular obstacle in an NRVM monolayer, with CV measured at multiple points along the boundary vs. in the interior.

### Why this experiment hasn't been done:
1. **Resolution limits:** Optical mapping: ~100–350 μm/pixel. Boundary effects concentrate within 1–3 pixels.
2. **Edge exclusion practice:** Researchers routinely exclude boundary regions from CV analysis because homogeneity cannot be verified near edges (Kléber & Rudy, Physiol Rev 2004).
3. **Clinical irrelevance:** Boundary speedup has no clinical consequence; funding targets slowing (reentry, block).
4. **Technical difficulty:** Creating a clean insulating boundary (not ablation damage, not ischemic scar) requires engineered monolayers.

---

## Indirect Evidence (Honest Assessment)

### A. Cabo et al. (1994) — Isthmus Diffraction, Sheep Epicardium
*Circ Res* 75:1014–1028. PMID: 7525101

**What they measured:** CV at the exit of narrow isthmuses (200–600 μm) in epicardial tissue.

**What it shows:** Convex wavefront expansion at isthmus exit causes CV slowdown proportional to curvature. The v(κ) relationship is confirmed for κ > 0 (convex, slowing side).

**What it does NOT show:** CV along a flat or concave boundary (κ ≤ 0). This is an isthmus/expansion geometry, not boundary-parallel propagation. Speedup (κ < 0) is *inferred* by linear extrapolation of the measured v(κ) relationship, never directly measured.

**Relevance to debate:** Confirms v(κ) law works in cardiac tissue — but only on the slowing side.

---

### B. Cabo et al. (2006) — Spiral Pinning, NRVM Monolayers
*Circulation*. DOI: 10.1161/CIRCULATIONAHA.105.598631

**What they measured:** Average CV around the full perimeter of circular obstacles (0.6–2.6 mm) after spiral wave attachment. CV_tip = circumference / cycle length.

**What it shows:** Average CV increases linearly with obstacle size. Larger obstacle → less curvature → faster average speed.

**What it does NOT show:** Spatially-resolved CV at different positions around the obstacle. The measurement is perimeter-averaged — a single number per obstacle size. You cannot distinguish speedup-along-edge from slowdown-at-corners from this data.

**Relevance to debate:** Consistent with curvature mechanism but cannot isolate boundary effect.

---

### C. Kucera, Kléber, Rohr (2000) — Designer Cultures
*Circ Res* 86:453–461. PMID: 10881747

**What they measured:** CV in patterned NRVM strands with and without periodic side branches.

**What it shows:** Branches create current sinks (equivalent to positive curvature), reducing CV by 63%. Pure geometry controls CV.

**What it does NOT show:** Anything about boundary-parallel propagation. This is about branching/source-sink, not obstacle edges.

**Relevance to debate:** Confirms geometry controls CV in cardiac tissue. No direct boundary relevance.

---

### D. Orozco et al. (2022) — RF Ablation Lesions, Isolated Hearts
*Front Physiol*. DOI: 10.3389/fphys.2022.794761

**What they measured:** CV in concentric zones around ablation lesions using high-density optical mapping.

**What it shows:** CV **decreases** 31–33% at the lesion border (0.36 m/s vs 0.52 m/s sham). No speedup observed.

**Why it's confounded:** RF ablation creates tissue damage, edema, and inflammatory gradients — not a clean insulating boundary. The slowdown is likely thermal injury, not a geometric effect.

**Relevance to debate:** Closest to a boundary CV measurement, but impossible to separate geometric from damage effects. Does show: no evidence of speedup near an obstacle, even when tissue damage might elevate ionic concentrations.

---

### E. Roth (1991) — APD Near Insulating Boundaries, Frog Heart
*Circ Res* 68:162–173. PMC2447675

**What they measured:** Action potential duration near a nonconducting boundary in frog cardiac tissue.

**What it shows:** APD shortens ~10% near the boundary due to electrotonic current diffusion.

**What it does NOT show:** Conduction velocity. APD ≠ CV. Shorter APD does not imply faster propagation.

**Relevance to debate:** Confirms electrotonic effects exist at boundaries (not ionic buildup), but does not measure CV.

---

### F. Foerster, Müller, Hess (1988) — BZ Chemical Waves
*Science* 241:685–687. DOI: 10.1126/science.241.4866.685

**What they measured:** Full v(κ) curve including both κ > 0 (slowing) and κ < 0 (speedup) in the Belousov-Zhabotinsky chemical reaction.

**What it shows:** Linear v(κ) = v₀ − Dκ confirmed with independently-measured D. The speedup side IS directly measured here.

**What it does NOT show:** Anything about cardiac tissue. The BZ reaction is a chemical excitable medium. It obeys the same reaction-diffusion PDE, but biological tissues have cellular structure, gap junctions, and ion channels that could modify the behavior.

**Relevance to debate:** Strongest evidence that v(κ) works on BOTH sides of κ = 0 — but in a non-cardiac system. Whether cardiac tissue deviates from BZ at κ < 0 is an open experimental question.

---

## Counter-Evidence Search: Ionic Buildup → Speedup

Comprehensive search across PubMed, PMC, AHA Journals, Frontiers for experimental or theoretical support of the hypothesis that ionic accumulation at no-flux boundaries causes CV speedup.

**Result: Zero papers found.**

| Mechanism searched | Finding | Supports hypothesis? |
|---|---|---|
| K⁺ accumulation at boundaries | Occurs at ischemic borders → causes SLOWING | No |
| Bath-loading effect | Electrical shunting (not ionic); requires bidomain; purely computational | No |
| Current reflection | Cable equation effect; transient; perpendicular approach, not along-boundary | Closest match, wrong mechanism |
| Any major author (Kléber, Rohr, Fast, Cabo, de Bakker) | All discuss curvature/geometry; none propose ionic buildup at boundaries | No |

---

## The Definitive Experiment (Proposed)

**Setup:** NRVM monolayer with photolithographically-defined rectangular obstacle.
**Mapping:** Optical, ≤50 μm/pixel.
**Protocol:** Planar wave past obstacle. Measure CV at:
1. Multiple points along flat boundary (parallel to propagation)
2. Around corners (concave wavefront zone)
3. Interior far from obstacle (reference)

**Expected results (from theory):**
- Flat edges: CV = v₀ (square-infarct redundancy)
- Convex corners: CV > v₀ (concave focusing → speedup)
- Interior: CV = v₀ (reference)

**Feasibility:** High. Kucera/Rohr demonstrated patterned cultures in 2000. Modern CMOS cameras achieve 20–50 μm/pixel. The experiment is straightforward — it has simply not been done for this geometry.

---

## Honest Bottom Line

The v(κ) relationship is mathematically derived from the monodomain PDE and experimentally confirmed on the slowing side (κ > 0) in cardiac tissue and on both sides in the BZ reaction. The speedup side (κ < 0) in cardiac tissue is a reasonable extrapolation but is NOT directly measured.

The ionic buildup hypothesis has zero support — experimental, computational, or theoretical.

The boundary CV experiment needed to fully settle this question has not been published.
