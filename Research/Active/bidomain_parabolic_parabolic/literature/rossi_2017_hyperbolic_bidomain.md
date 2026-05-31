---
paper: rossi_2017_hyperbolic_bidomain
title: "Incorporating Inductances in Tissue-Scale Models of Cardiac Electrophysiology"
authors: "Simone Rossi, Boyce E. Griffith"
year: 2017
journal: "Chaos: An Interdisciplinary Journal of Nonlinear Science"
doi: "10.1063/1.5000706"
pmid: "28964127"
pmcid: "PMC5585078"
arxiv: "1706.08490"
pdf: ../papers/rossi_griffith_2017_hyperbolic_bidomain.pdf
questions: [bidomain_parabolic_parabolic, lbm_ep, boundary_conduction_speedup]
---

## Key Findings

- **Constitutive law, not PDE, is modified.** Replacing Ohm's law `J = -σ∇V` with the Cattaneo relation `τ ∂J/∂t + J = -σ∇V` gives intracellular and extracellular fluxes their own time dynamics. Eliminating J then produces a **hyperbolic bidomain** with genuine finite propagation speed — `c_s = √(σ / (τ·χ·C_m))`. The standard model has infinite propagation speed (compact-support initial data becomes non-compact instantly for any t > 0).
- **Critical τ_e vs τ_i asymmetry.** The φ_e equation is multiplied by `(τ_e − τ_i)`. When τ_i = τ_e, the φ_e equation collapses to the standard elliptic form `∇·D_i∇V + ∇·(D_e+D_i)∇V_e = 0`. **Only τ_i ≠ τ_e produces a truly dual-evolving bidomain** (what Rossi's source code calls `ParabolicParabolicHyperbolic`). The τ_i = τ_e case is `ParabolicEllipticHyperbolic` — the Vm eq becomes hyperbolic but the Ve eq stays elliptic.
- **Linear vs nonlinear divergence.** Linear analysis (Scott, Kaplan-Trujillo) says inductances are negligible. Rossi & Griffith show this holds for the **piecewise-linear McKean model** (CV decreases monotonically with τ — linear theory works). For **nonlinear ionic models** (Aliev-Panfilov, Fenton-Karma, TTP06, Grandi-atrial), CV *increases* for small-to-moderate τ. Linear analysis fails during the nonlinear upstroke because `∂I_ion/∂t` is large and its time derivative enters the hyperbolic equations (the fingerprint of eliminating J with dynamics).
- **Mesh convergence.** Accurate CV requires ~25 μm (single-myocyte scale) for fiber-transverse propagation in both parabolic and hyperbolic formulations. The hyperbolic model does not make mesh requirements worse.
- **Arrhythmia differences.** Spiral wave breakup occurs after 2 stimuli in the hyperbolic case vs 3 in the parabolic case — qualitatively different arrhythmia inducibility. Tested in anatomically detailed 3D left atrium.
- **Virtual electrode phenomenon.** Tested in bidomain (the only bidomain figure in the paper — Fig 12, one 2 ms snapshot). Authors conclude effect "remains unclear, necessitates further investigation." This is the only published hyperbolic bidomain simulation.

## Method

- **Derivation approach**: Start from the discretized cable circuit (Fig 1) with explicit axial inductances `L_i`, `L_e`, capacitance `C_m`, ionic current `I_ion`, membrane shunt resistance. The inductor voltage drop produces the `τ ∂J/∂t` term in the continuum limit. The relaxation times are physical inductance/resistance ratios: `τ = (L_i + L_e)/(R_i + R_e)` in 1D.
- **Quasi-static coupling preserved**: Eq 5 imposes `∇·(J_i + J_e) = 0` exactly (charge conservation between compartments), and eq 6 imposes the pointwise compartment balance `I_t = -∇·J_i = ∇·J_e`. The Cattaneo modification does not touch these conservation laws — only the flux-voltage relation.
- **Full PDE system** (eqs 13, 14 in paper):
  ```
  Vm:  τ_i C_m ∂²V/∂t² + C_m ∂V/∂t - ∇·D_i∇V - ∇·D_i∇V_e = -I_ion - τ_i ∂I_ion/∂t
  Ve:  (τ_e - τ_i) C_m ∂²V/∂t² + ∇·D_i∇V + ∇·(D_e+D_i)∇V_e = (τ_i - τ_e) ∂I_ion/∂t
  ```
- **First-order system**: Introduce Q = ∂V/∂t; solve for (Q, V_e) coupled via a 2×2 block system, then V is reconstructed via `V^{n+1} = V^n + dt·Q^{n+1}` (SBDF1) or SBDF2 extrapolation. Appendix C.
- **Spatial discretization**: P1 (linear) finite elements, isoparametric trilinear on hex meshes.
- **Time discretization**: IMEX-RK, first-order ARS(1,1,1) or second-order H-CN(2,2,2) (implicit-CN for diffusion, Heun for reaction). `dt ∈ [0.0025, 0.125]` ms.
- **Ionic models tested**: McKean (piecewise-linear, analytical solution), Aliev-Panfilov, Fenton-Karma (4 parameter sets), ten Tusscher-Panfilov (TTP06, 20 vars), Grandi-atrial (57 vars).
- **Geometries**: 1D cables, 2D sheets (12×12 cm), 3D anatomic left atrium (~3.5M elements).
- **Open-source code**: https://github.com/rossisimone/beatit (BeatIt, C++, uses libMesh + PETSc).

## Key Equations / Results

- **Cattaneo fluxes (eqs 1, 2):**
  ```
  τ_i ∂J_i/∂t + J_i = -σ_i ∇V_i
  τ_e ∂J_e/∂t + J_e = -σ_e ∇V_e
  ```
- **Full bidomain after eliminating J (eqs 13, 14):** see above.
- **Hyperbolic monodomain (eq 21)** — equal anisotropy ratio D_e = λD_i:
  ```
  τ C_m ∂²V/∂t² + C_m ∂V/∂t - ∇·D∇V = -I_ion - τ ∂I_ion/∂t
  ```
  with `τ = τ_i + λ(τ_e - τ_i)/(λ+1)`. This is the classical **telegraph equation** with reaction.
- **Characteristic propagation speed (eqs 34, 35):**
  ```
  c_s = √(σ / (τ·χ·C_m))
  ```
  In the McKean linear analysis, the front speed is `v = √(σk/(χ·C_m²)) · f(α, μ)` where `μ = τk/C_m` parameterizes the ratio between relaxation time and characteristic reaction time.
- **TTP06 numerical result**: Peak CV enhancement occurs at τ ≈ 0.1–0.2 ms. Larger τ reverses the trend.
- **Cost comparison (2D, TTP06)**: Hyperbolic is **3% cheaper** overall than parabolic. Reaction is 7% cheaper (same ionic solve but no Vm coupling inside), diffusion is 15% more expensive (bigger system for Q), but fewer CG iterations offset that.
- **Spiral waves (Fenton-Karma)**: With τ = 0.5 ms, spiral breakup occurs after 2 stimuli; parabolic requires 3. Qualitatively different arrhythmia onset.

## Connections to Our Models

### Relevant Engine Components

- **Bidomain V1** (`Bidomain/Engine_V1/`): this paper's equations 13–14 map directly onto the solver architecture specified in `HYPERBOLIC_BIDOMAIN_MAPPING.md` — a new `HyperbolicBidomainSolver` that extends the current Gauss-Seidel decoupled solver with SBDF1/SBDF2 time stepping, an auxiliary Q = ∂V/∂t field, and storage of Iion + dIion/dt.
- **LBM V1** (`LBM/Engine_V1/`): the Chapman-Enskog expansion of LBM's BGK collision naturally produces Cattaneo-type flux dynamics `τ_LBM ∂J/∂t + J = -D∇u` — see KNOWLEDGE.md §"LBM-Cattaneo Correspondence". Rossi's τ_i ≠ τ_e regime maps directly onto a dual-lattice LBM where each lattice has its own relaxation parameter. The Cattaneo "correction" usually minimized in LBM practice becomes the *desired* physics here.

### Agreements

- The derivation starts from charge conservation (eqs 5, 6), which is identical to what KNOWLEDGE.md identifies as the "local inter-domain coupling" constraint that must be preserved. Rossi & Griffith explicitly preserve this — the Cattaneo modification is to the constitutive law, not the conservation laws. This matches our physical reasoning.
- Their "ParabolicEllipticHyperbolic" (τ_i = τ_e > 0) vs "ParabolicParabolicHyperbolic" (τ_i ≠ τ_e) naming matches our derivation: the Ve equation is elliptic iff τ_i = τ_e. This is the critical subtlety for our use case.
- They arrive at the same telegraph-equation monodomain reduction we derived (eq 21 vs KNOWLEDGE.md §"The monodomain reduction").

### Disagreements or Gaps

- **They only did one bidomain experiment** — virtual electrode phenomenon, one 2 ms snapshot (Fig 12). No CV measurement, no propagating wave, no arrhythmia. The ParabolicParabolicHyperbolic case (which is what we actually want) is essentially unvalidated.
- **No tissue-bath boundary tests.** They do not simulate the Kleber boundary speedup effect or examine whether finite extracellular propagation alters wavefront curvature at tissue-bath interfaces. **This is the gap we intend to fill.**
- **No systematic (τ_i, τ_e) parameter sweep** — all their bidomain work uses nominal τ values without studying how CV, wavefront shape, or boundary artifacts scale with τ_e − τ_i.
- **FEM only** — no FDM, no spectral, no LBM implementation. We would be the first to do any of these.
- **No reported CV value for bidomain hyperbolic.** The 3% cost advantage is for the **monodomain** hyperbolic; the bidomain cost is not quantified.

### Actionable Insights

- **High priority: Use τ_i, τ_e as the primary control.** τ_i = τ_e reduces to a "decorated" PE bidomain (only Vm equation is hyperbolic). Set τ_i ≠ τ_e from day one — this is the only regime where our "dual-evolving" goal holds.
- **High priority: Sweep τ around 0.1–0.5 ms.** Rossi found peak CV enhancement at τ ≈ 0.1–0.2 ms for TTP06. Start the Kleber boundary experiment with τ_i = 0.1, τ_e ∈ {0.05, 0.1, 0.15, 0.2, 0.3} ms. Compare wavefront curvature and edge/bulk CV ratio against the standard PE bidomain.
- **High priority: Use TTP06 (already in our engines).** Rossi validated hyperbolic behavior with TTP06; we have TTP06 in both Bidomain V1 and LBM V1. No ionic-model porting needed.
- **Medium priority: Track Iion, dIion/dt in state.** Our current state only stores Vm, phi_e, ionic_states. The hyperbolic RHS requires Iion(x,t) and ∂Iion/∂t to be persistent fields (not recomputed per step). See `HYPERBOLIC_BIDOMAIN_MAPPING.md` §3.4 for the state-field additions.
- **Medium priority: Steal BeatIt's assembly expressions, not the FEM discretization.** The 2×2 block RHS from Bidomain.cpp L904–919 (SBDF1) and L879–901 (SBDF2) translates directly to our FDM (identity mass matrix) with sign flips for the stiffness (our L_i is negative semi-definite; their K_i is positive definite). The mapping is worked out in `HYPERBOLIC_BIDOMAIN_MAPPING.md` §3.8.
- **Medium priority: Reuse spectral solver for the Q sub-problem.** The Q operator `A_QQ = (1 + τ_i/cdt)·C_m·I + cdt·(-L_i)` has the same eigenvectors as the Laplacian, only a shifted eigenvalue formula. No modification of `SpectralSolver` needed — just build a new eigenvalue array in the hyperbolic solver. See mapping doc §4.4.
- **Low priority: Record NFE / ODE steps per timestep.** Rossi uses IMEX-RK; we use IMEX-SBDF. Our stiffness behavior may differ — track the effective number of Newton iterations when we build the system, especially during the upstroke where `∂Iion/∂t` is large.

## Limitations / Caveats

- **No physical measurement of τ for cardiac tissue.** Rossi uses τ = 0.4 ms calibrated to match Fenton-Karma parabolic CV. Clapham & DeFelice's 1 Hz resonance in embryonic heart membranes is the only experimental indication, and it concerns the *membrane* — not the axial inductance that generates τ. The values Rossi used are phenomenological.
- **Only one bidomain figure.** The actual bidomain behavior with finite extracellular propagation is essentially unstudied. Our project would be the first to produce propagation/CV/boundary results in this regime.
- **Cost savings do not transfer directly.** Their 3% cost advantage is for **monodomain**. The bidomain cost structure is dominated by the elliptic solve in the standard PE case; in the hyperbolic case the φ_e equation becomes parabolic (if τ_e ≠ τ_i) and arguably cheaper — but no reported numbers.
- **Citation analysis shows the approach was abandoned.** Across 12 subsequent papers over 9 years, no one implemented hyperbolic bidomain as a practical solver. Rossi's own 2018 follow-up reverted to standard parabolic-elliptic bidomain. Either the community deemed it not worth the complexity, or nobody had a compelling use case. We have the compelling use case (boundary artifacts + LBM natural fit), but we should be aware the direction is untravelled.
- **Insulation BC only.** Rossi's boundary conditions are `D·N·∇V = 0` (Neumann/insulation). No bath coupling is derived for the hyperbolic formulation. We would need to extend the BC treatment for the Kleber boundary experiment.
