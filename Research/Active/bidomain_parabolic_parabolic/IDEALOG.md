# Bidomain Parabolic-Parabolic Formulation — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Research complete. Three-phase implementation plan settled: ε-regularization experiment → Cattaneo implementation → boundary validation. Ready for /blueprint when user approves.

## Next Step
Phase 1: Implement ε-regularization as numerical experiment (modify spectral solver denominator, ~5 lines). Run Kleber boundary test with ε sweep.

## Thread

### 2026-03-20 — Origin
- Observed triangular (not parabolic) wavefront artifacts at tissue-bath boundaries during boundary speedup research
- Root cause: elliptic equation propagates extracellular information instantaneously
- This is non-physical — real extracellular space has finite diffusion speed
- The parabolic-parabolic formulation adds ∂Ve/∂t to make both equations parabolic
- Goal: implement PP variant in Bidomain V1 to get physically accurate boundary artifacts

### 2026-03-20 — Literature Research Complete (4 parallel agents)

**Critical finding: "PP bidomain" is NOT what we thought.**
- The "parabolic-parabolic" bidomain in literature (Bourgault 2009, Pavarino & Scacchi 2011, Colli Franzone) is the ORIGINAL (φ_i, φ_e) variable form of the same physics as PE. Adding the two equations cancels dv/dt → elliptic constraint is hidden inside. Same physics, different variables.
- There is NO published ε·∂φ_e/∂t regularization used as a computational method. Bendahmane & Karlsen 2006 use it only as a proof technique (ε→0 recovers true bidomain). Nobody simulates with it.
- The ONLY published model with finite extracellular propagation speed is the **Cattaneo/hyperbolic bidomain** (Rossi & Griffith 2017, Chaos 27:093926). Replaces Ohm's law with τ·dJ/dt + J = -σ∇V, giving a telegraph equation with genuine finite propagation.

**Two types of "instantaneous" in the bidomain:**
1. SPATIAL: φ_e adjusts GLOBALLY when Vm changes anywhere (elliptic = Green's function has global support). THIS causes the triangular artifact. Both ε-regularization and Cattaneo fix this.
2. LOCAL: At each point, Im_i = -Im_e (conservation of charge). This is physically correct. Cm·dVm/dt provides the real time delay. No continuum model should "fix" this.

**Implementation findings:**
- ε-regularization is trivial: change spectral denominator from `eigenvals` to `ε/dt + eigenvals`. Same DCT/DST/FFT. ~5 lines. Bonus: no null-space pinning needed for ε > 0.
- Cattaneo needs 2nd-order time stepping (leapfrog/Newmark), dIion/dt computation, flux state variables. ~100+ lines. But physically motivated and gives genuinely finite propagation speed.
- Both fix the triangular artifact because both make φ_e adjust locally rather than globally.

**Alternative models investigated:**
- EMI (Tveito et al.): cell-resolved, finite coupling via membrane RC, but 300-3000x cost. Impractical for boundary studies.
- KNM (Jaeger & Tveito 2023): cell-based, 1% of bidomain cost, but extracellular still instantaneous within compartments.
- Extended bidomain/tridomain: still elliptic extracellular. Doesn't fix our issue.
- LBM bidomain (Belmiloudi 2019): dual-lattice approach, but without time-delay extension, same elliptic φ_e.

**Key papers:**
- Rossi & Griffith 2017 (Chaos 27:093926) — hyperbolic bidomain, Cattaneo flux, THE paper to build from
- Bourgault et al. 2009 — PP well-posedness, bidomain operator
- Pavarino & Scacchi 2011 — PP vs PE preconditioners (SIAM J Sci Comp)
- Jaeger & Tveito 2022 — EMI model derivation and comparison
- Bendahmane & Karlsen 2006 — ε-regularization as proof technique only

**Settled decision: Three-phase approach**
1. Quick ε-regularization experiment (numerical experiment, not physical model) to confirm artifact source
2. Cattaneo implementation (the physically justified formulation)
3. PE vs ε-reg vs Cattaneo comparison at Kleber boundary (novel — nobody has done this)

### 2026-03-20 — Source Code Analysis & Novelty Assessment

**Rossi & Griffith DID implement hyperbolic bidomain** — confirmed in their Bidomain.cpp:
- Three equation types in one class: `ParabolicEllipticBidomain` (τ=0), `ParabolicEllipticHyperbolic` (τ_i=τ_e≠0), `ParabolicParabolicHyperbolic` (τ_i≠τ_e)
- The naming confirms our derivation: when τ_i=τ_e, φ_e equation is STILL elliptic (`ParabolicEllipticHyperbolic`)
- `BidomainWithBath.cpp` also has the hyperbolic implementation — directly relevant for our work
- They ran ONE bidomain test (virtual electrode phenomenon, Figure 12) — a single 2ms snapshot. Their conclusion: "remains unclear, necessitates further investigation"

**Citation analysis (12 papers, 9 years):** Zero follow-up implementations. Rossi's own 2018 paper reverted to standard parabolic bidomain. The hyperbolic bidomain was abandoned after one figure.

**What has never been done (confirmed gaps):**
- Hyperbolic bidomain propagation/CV/wavefront shape — never
- Hyperbolic bidomain at tissue-bath boundaries — never
- LBM-based hyperbolic bidomain — never
- FDM/spectral hyperbolic bidomain — never (only FEM)
- Systematic (τ_i, τ_e) parameter study — never

**LBM–Cattaneo correspondence discovered:** LBM's Chapman-Enskog expansion naturally produces Cattaneo dynamics (τ_LBM·∂J/∂t + J = -D∇u). This means dual-lattice LBM is the natural discretization for hyperbolic bidomain — each lattice's relaxation parameter directly maps to the physical Cattaneo τ. No pseudo-time iteration needed. This would be a completely novel numerical approach.

**Our unique contributions would be:**
1. First comprehensive hyperbolic bidomain simulation (propagation, CV, arrhythmia)
2. First tissue-bath boundary analysis with finite extracellular propagation
3. First FDM/spectral implementation (vs their FEM)
4. First LBM hyperbolic bidomain (exploiting the Cattaneo correspondence)
5. Resolution of the triangular wavefront artifact question

### Work plan (revised after literature research)
1. ε-regularization numerical experiment — add shift to spectral denominator, sweep ε, test if triangle→parabola
2. Cattaneo hyperbolic bidomain implementation — telegraph equation, leapfrog/Newmark, dIion/dt
3. Three-way comparison at Kleber boundary: PE vs ε-reg vs Cattaneo (novel contribution)
4. Characterize ε and τ sensitivity — what values produce physically reasonable wavefront shapes?
5. Cross-reference with boundary speedup research — does Kleber ratio change?

## Failed Approaches

## Session Log
| Date | What happened |
|------|--------------|
| 2026-03-20 | Research question scaffolded. Motivation: triangular artifact from elliptic instantaneous propagation. |
| 2026-03-20 | Deep literature research: 4 parallel agents on PP formulation, Cattaneo derivation, LBM correspondence. Deep dive into Rossi & Griffith source code and citation analysis. |
