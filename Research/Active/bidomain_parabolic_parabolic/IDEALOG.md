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
| 2026-04-23 | Endgame reframed: target is a dual-evolving bidomain LBM. User established they want 2 independent dynamical DOFs, not 1-DOF-with-memory. Pulled Rossi-Griffith 2017 + ESAIM M2AN 2013 + Bishop-Plank 2011 PDFs; wrote literature summaries. Wrote `HYPERBOLIC_HYPERBOLIC_ANALYSIS.md` — the 6 `(τ_i − τ_e)` terms that distinguish hyperbolic-hyperbolic from hyperbolic-only in BeatIt. |

## 2026-04-23 — The 1-DOF-with-memory vs 2-DOF-dual-evolving Insight

### Observation
Rossi-Griffith's "ParabolicParabolicHyperbolic" (τ_i ≠ τ_e) is structurally 2-unknown (Q, V_e) but physically **1-DOF-with-memory**. In his final equations (eqs 13, 14), only V has second-order time derivatives; V_e has no ∂V_e/∂t or ∂²V_e/∂t² anywhere. V_e is algebraically/differentially determined by V at each instant, just with V's time derivatives now on the source side instead of V alone. The naming is misleading.

### Why this matters
A genuinely dual-evolving bidomain — each potential evolving under its own PDE, coupled through membrane current — is **not** what Rossi's formulation produces at the continuum PDE level. The dual-lattice LBM approach, however, would naturally implement 2-DOF dual evolution because each lattice carries its own distribution function and kinetic relaxation. This means our LBM endgame is architecturally bolder than Rossi's FEM — potentially a novel contribution.

### Is 2-DOF mathematically/conceptually correct?
- Mathematically: yes, no objection. Any set of coupled PDEs with appropriate BCs is well-posed regardless of DOF count.
- Conceptually: depends on which physical assumption you relax. Standard bidomain's "V_e is elliptic" is a CONSEQUENCE of the quasi-static charge-neutrality assumption, not a fundamental truth. Drop quasi-staticity and you get 2 DOFs.
- Three honest paths: full Maxwell (ns timescale, wrong regime for cardiac), phenomenological extracellular capacitance (ansatz, needs defense), lattice-scale freedom (2-DOF emerges from LBM kinetics rather than being imposed).

### The key mathematical operation Rossi performed (the "collapse")
Rossi's **eq 5** — the quasi-static charge conservation `∇·(J_i + J_e) = 0` — is the step that collapses the implicitly 2-DOF Cattaneo flux system into the 1-DOF-with-memory bidomain. His eqs 1-2 (Cattaneo fluxes) are genuinely 2-DOF. Imposing eq 5 forces the flux divergences to be equal-and-opposite, coupling the two equations into a single dynamical unknown. Without eq 5, the Cattaneo flux equations evolve independently, coupled only through the membrane source I_m.

### Candidate 2-DOF reformulation for dual-lattice LBM
Keep Cattaneo eqs 1-2. Replace eq 5 with INDEPENDENT compartmental charge conservation with finite storage: `C_i ∂V_i/∂t + ∇·J_i = -I_m`, `C_e ∂V_e/∂t + ∇·J_e = +I_m`. Each (V_c, J_c) pair is then a first-order telegraph system — exactly what LBM BGK's Chapman-Enskog recovers at the Navier-Stokes level. In the limit C_i, C_e → 0 we recover standard quasi-static bidomain, so it's a consistent extension.
See `TWO_DOF_FORMULATION.md` for the derivation.

### Correction (later same day): the phenomenological C_c was not rigorous
User pushed back: "collapse happens in necessity to satisfy charge conservation, not a mathematical trick. We need a better physically correct constraint to prevent the collapse." Correct objection. My earlier framing conflated universal continuity (`∂ρ/∂t + ∇·J = 0`, unbreakable) with quasi-static neutrality (`∂ρ/∂t ≈ 0`, an approximation). Eq 5 follows from both. Simply "replacing" eq 5 with an ansatz `ρ_c = C_c V_c` is phenomenological — need to defend C_c separately.

The honest 2-DOF formulation is **Poisson-Nernst-Planck (PNP)**: track compartmental charge density `ρ_c` as an explicit dynamical variable; get V_c from Poisson `-∇·(ε_c ∇V_c) = ρ_c`; evolve ρ_c via continuity `∂ρ_c/∂t + ∇·J_c = S_c`. Quasi-static bidomain is the `ρ_c → 0` projection of PNP. Allowing ρ_c ≠ 0 gives 2 DOFs for real reasons — no ansatz needed.

Crucial timescale argument: the dielectric Debye time `τ = ε/σ ≈ 1 ns` is NOT the relevant physical timescale for PNP-vs-bidomain divergence. The dominant mechanism is **ion redistribution via diffusion over cellular scales**: `τ_diff = L²/D_ion ≈ (10 μm)² / (10⁻⁹ m²/s) ≈ 0.1 ms`. Same order as AP upstroke. **The quasi-static reduction loses physically relevant dynamics at exactly the timescale we care about.** This is the scientifically defensible motivation for the 2-DOF direction — not "finite numerical capacitance" but "ion redistribution physics that bidomain averages away."

Next step: literature survey on PNP-cardiac. Mori, Peskin, and others have worked on this. Need to know what's been tried before committing to a formulation. See TWO_DOF_FORMULATION.md §10+ for corrected PNP derivation once written.

### PNP survey done (same day) — see `literature/PNP_LANDSCAPE_SURVEY.md`
Key findings:
1. **Homogenized PNP-bidomain already exists:** Okada-Sugiura-Hisada 2013 (Phys Rev E, cardiac, paywalled) and Whiteley 2020 (Math Med Biol, general, PDF in hand). Whiteley starts from microscale PNP with Debye layers → homogenizes → recovers standard PE bidomain as valid under normal cardiac conditions. The quasi-static form is structurally robust to homogenization of PNP.
2. **Where PNP adds DOFs: ion concentrations.** Each ion species per compartment becomes tissue-level dynamical. V_c stays elliptic-coupled. Not the "2-DOF-in-V" we originally imagined, but 2-DOF-in-ions.
3. **Cell-resolved PNP exists** at nano-scale (Mori 2008 PNAS, Jæger-Tveito 2023 and 2025). Preserves Debye-layer physics but not continuum-scalable.
4. **KNP-EMI framework** (Tveito group): electroneutral NP + cell-resolved EMI. This is the state-of-the-art answer to "how do we add PNP to bidomain affordably?"
5. **No LBM-PNP for cardiac.** Existing LBM-PNP is all electrokinetics. Clear gap.

Next: acquire Okada 2013 PDF (institutional access). If Okada's "rational bidomain" is what we want, the formulation is settled and we proceed to implementation. If not, we know the specific gap to fill.
