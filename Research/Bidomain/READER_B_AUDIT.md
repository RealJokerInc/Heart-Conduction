# Full Textbook Audit — Reader Profile B

**Reader Profile**: Intro neurophysiology, heat & mass transfer, linear algebra, basic vector (multi) calculus. **No computational background** — no programming, no numerical methods, no prior exposure to FDM/FEM/FVM, no concept of "time-stepping as an algorithm."
**Scope**: Parts II (Ch 7–11), III (Ch 12–17), IV (Ch 18–20)
**Evaluated against**: `STYLE_GUIDE.md` (Feynman style, 5-layer complexity)
**Date**: 2026-03-08
**Comparison baseline**: Reader A (previous audits — student with monodomain knowledge, intro lin alg, no numerical analysis)

---

## What This Reader Knows and Doesn't Know

### Has
- **Neurophysiology**: Action potentials, ion channels (Na⁺, K⁺, Ca²⁺), membrane capacitance, gap junctions, the idea that cardiac tissue conducts electrical waves. Understands depolarization/repolarization qualitatively.
- **Heat & mass transfer**: The diffusion equation ∂T/∂t = α∇²T, Fick's law J = -D∇C, boundary conditions (Neumann = insulated, Dirichlet = fixed temperature), steady-state vs transient, control volume analysis. May know the Boltzmann distribution from thermodynamics.
- **Linear algebra**: Matrices, eigenvalues/eigenvectors, Ax = b, matrix-vector products, symmetric/positive-definite, maybe condition number conceptually.
- **Vector calculus**: Gradient, divergence, curl, Gauss's divergence theorem, integration by parts in higher dimensions, partial derivatives, chain rule.

### Does NOT Have
- **Any programming experience.** "Pseudocode" is not a language they read. "Function signatures," "ABC," "torch.roll" are meaningless.
- **Numerical methods.** FDM, FEM, FVM are entirely new. "Discretization" is an unfamiliar concept — they've always worked with continuous PDEs.
- **Time-stepping as an algorithm.** Forward Euler, RK2, RK4, Crank-Nicolson — these are new. They've seen ODEs in math class but never "stepped" through one computationally.
- **Iterative solvers.** PCG, Chebyshev, multigrid, CFL condition — all new. They know Gaussian elimination to solve Ax=b but not iterative methods.
- **GPU/parallel computing.** "Trivially parallelizable," "zero-synchronization," "memory coalescing" — not in their vocabulary.
- **Software engineering patterns.** "Solver ABC," "factory function," "operator pattern" — foreign.

### Key Difference from Reader A
Reader A had monodomain knowledge and some computational intuition — they knew what a "mesh" was, what "solving a system" meant in practice, and could read pseudocode. Reader B comes from the **physics/math side** only. The PDE itself is familiar territory; turning it into something a computer can execute is the new frontier.

---

## Part II — Tissue-Level Monodomain Modeling (Ch 7–11)

### Chapter 7 — The Monodomain Equation — Score: 5.0/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | Light bulbs and wires — immediately clear. The physiology here is exactly what this reader studied in neurophysiology. |
| L2 Feynman | **5** | "Voltage change = reaction + diffusion." This reader *already knows this* from heat & mass transfer — it's the heat equation with a source term. The chapter makes the connection explicit ("just Ohm's law in continuous form"). |
| L3 3B1B | **4** | Swimming-pool analogy for BCs is effective. Missing a current-flow diagram, but the text is so clear it barely matters. |
| L4 Worked Example | **4** | The BC interpretations use physical terms this reader understands (insulated surface = Neumann). |
| L5 Implementation | **5→3** | The Engine V5.4 boxes are irrelevant to this reader — they can't read code. But the mathematical specification is clean enough to follow without the code. **Adjusted score reflects that L5 is less useful, not harmful.** |

**Reader B experience:** This is the chapter where Reader B thinks "I know this! This is just the heat equation on cardiac tissue!" The light-bulb analogy is unnecessary — they already understand diffusion — but it doesn't condescend. The conductivity tensor (§7.3) is familiar from heat transfer (anisotropic thermal conductivity). BCs are second nature. **This reader enters Ch 8 with high confidence.**

**Reader B score: 4.8/5** (vs Reader A: 4.6) — slightly higher because the physics content matches their background perfectly.

---

### Chapter 8 — Spatial Discretization — Score: 3.4/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **4** | "How do you turn a continuous PDE into something a computer can solve?" is the right question. The "each node looks at its neighbors" framing for FDM is clear. But the fundamental concept — "the computer can't handle continuous functions, so we sample at discrete points" — deserves a dedicated ELI5 that the chapter doesn't have. It jumps to "place nodes on a regular grid" without explaining *why* discretization is necessary at all. |
| L2 Feynman | **3** | FDM is accessible: Taylor expansion → stencil → matrix. This reader knows Taylor expansions. FVM is accessible: "flux balance across a control volume" is exactly what they did in heat & mass transfer. **FEM is the problem.** "Multiply by a test function and integrate" is not in their toolkit. The weak form is presented as a mathematical manipulation (integration by parts), which they can follow step-by-step, but the *motivation* ("why would you do this?") is never explained to someone who has always worked with strong-form PDEs. |
| L3 3B1B | **5** | The SVG stencil diagrams, hat functions, and control volumes are exactly the visual aids this reader needs. They've never seen a "stencil" before — the diagram makes it concrete. |
| L4 Worked Example | **5** | The 5-node cable and 4-element FEM examples are outstanding. Building the actual K matrix with numbers is the bridge between "I know the PDE" and "I see how the computer represents it." |
| L5 Implementation | **2** | Engine boxes reference `SpatialDiscretization` subclasses, sparse matrix storage (COO/CSR), and code interfaces. This reader cannot parse any of it. The practical "how do I code this?" layer is inaccessible. |

**Reader B experience:** This chapter is the **critical inflection point**. Reader B has never seen a PDE turned into a matrix equation before. The FDM section is the best on-ramp: "approximate the derivative → build a matrix → the matrix encodes the physics." The worked example (5-node cable) is the moment of understanding: "Oh! Each row of K represents one node's relationship to its neighbors."

But FEM (§8.4) is a significant stumbling block. Reader B knows integration by parts but has never encountered test functions or the variational framework. The text says "multiply the PDE by a test function" — but *why*? For Reader A (who may have seen FEM briefly), this is a recap. For Reader B, it's a new mathematical framework introduced mid-chapter without motivation.

FVM (§8.5) is actually more accessible to this reader than to Reader A, because control-volume analysis is a standard tool in heat & mass transfer.

The chapter's length (~1100 lines) is a bigger problem for Reader B than Reader A, because every section introduces a fundamentally new computational concept. By §8.7 (BCs), Reader B has processed three entirely new numerical methods plus the concept of discretization itself. Cognitive overload is severe.

**Reader B score: 3.4/5** (vs Reader A: 4.0) — lower because the computational concepts are all new, FEM motivation is missing, and the implementation layer is opaque.

---

### Chapter 9 — Operator Splitting — Score: 3.8/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | The stir-fry analogy works perfectly for any reader. |
| L2 Feynman | **4** | "Solve reaction and diffusion separately." This reader understands the physics of both terms. The *why* is clear: the reaction is local (one cell), diffusion couples neighbors. What's less clear: *why is this computationally advantageous?* The chapter says "each substep is simpler" — but Reader B doesn't yet know what "simpler" means computationally. They haven't internalized that "solving a coupled system is harder than solving two smaller systems." |
| L3 3B1B | **3** | Still no diagrams. A timeline showing the substep sequence would help any reader. |
| L4 Worked Example | **3** | The Rush-Larsen example computes exponential integration for a gate variable. Reader B can follow the math (they know exponentials, ODEs) but has never seen "Forward Euler" as an algorithm, so the comparison "FE error = X, RL error = Y" lacks a baseline. The chapter assumes the reader already knows Forward Euler from Ch 10 — but Ch 10 comes *after* Ch 9. **Ordering issue.** |
| L5 Implementation | **1** | `SplittingStrategy`, `GodunovSplitting`, `StrangSplitting` mean nothing to Reader B. |

**Reader B experience:** The concept of splitting is accessible — it's a standard technique in applied math that this reader may have seen for coupled ODEs. But the *computational* motivation ("this is faster because...") requires understanding what "solving a linear system" costs, which Ch 8 only hinted at. The Rush-Larsen section assumes Forward Euler knowledge that hasn't been introduced yet.

**Reader B score: 3.8/5** (vs Reader A: 4.4) — lower because the computational motivation is implicit and the FE/RL comparison requires background not yet provided.

---

### Chapter 10 — Explicit Diffusion Solvers — Score: 4.2/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | "Heading north at 60 mph" for Forward Euler. GPS/speedometer for RK. These analogies work for everyone. |
| L2 Feynman | **5** | The progression FE → "why go beyond?" → RK2 → RK4 is excellent. The "predictor-corrector" explanation is clear. The Simpson's rule connection for 1/6 weights would resonate with this reader (they know numerical integration from calculus). |
| L3 3B1B | **5** | The SVG diagrams showing slope evaluations are exactly what this reader needs to understand time-stepping visually. The stability region plot is beautiful — though "stability" as a numerical concept is new, the visual (inside = safe, outside = blow up) is immediate. |
| L4 Worked Example | **5** | dV/dt = -2V carried across FE, RK2, RK4 with exact error comparison. This reader can follow every arithmetic step. The concrete demonstration that "more evaluations = less error" is convincing. |
| L5 Implementation | **2** | `DiffusionSolver` ABC, `apply_diffusion(V)` operator pattern — foreign. But the math is self-contained enough that implementation details aren't needed to understand the methods. |

**Reader B experience:** This is where time-stepping as a concept clicks. Reader B has solved ODEs analytically in calculus — "given dy/dt = f(y), find y(t)." Now they're learning that a computer approximates this step by step. The GPS analogy bridges the gap. The worked examples are the anchor: they can verify every number with pencil and paper.

The CFL condition is a new idea but well-motivated: "too big a step → blowup." The h² scaling argument is accessible from dimensional analysis (which they know from heat & mass transfer).

The "bare k" convention (§10.3b, 10.4b) is confusing for this reader because they don't know about "what the code computes" vs "what the textbook writes." There's no code in their mental model.

**Reader B score: 4.2/5** (vs Reader A: 4.8) — slightly lower because the implementation layer is opaque and "stability" is a new concept, but the core teaching is excellent.

---

### Chapter 11 — Implicit Diffusion Solvers — Score: 3.0/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | "Steering a boat by looking where you're going, not where you are." Excellent for any reader. |
| L2 Feynman | **3** | The "why implicit?" argument (CFL wall → need larger timesteps) makes sense only if Reader B understood the CFL condition in Ch 10. The conceptual explanation is clear. But "solving a linear system at every timestep" is presented as the "cost" — Reader B doesn't know what this costs. They know Gaussian elimination is O(n³), but they don't know that sparse matrices make it cheaper, or what "iterative" means. The chapter assumes this knowledge without providing it. |
| L3 3B1B | **2** | Zero diagrams. Same problem as Reader A, but worse — Reader B has no visual intuition for what A_lhs and B_rhs look like as matrices, what "SPD" means geometrically, or what "CG converging" looks like. |
| L4 Worked Example | **3** | The CN 5-node example is accessible: Reader B can see A_lhs = I + 0.01K, B_rhs = I - 0.01K and verify the matrix arithmetic. BDF2 has no worked example — same gap as before. |
| L5 Implementation | **1** | `CrankNicolsonSolver`, `BDF1Solver`, `get_diffusion_operators(dt, scheme)` — completely opaque. The `step()` method description is pseudocode that Reader B cannot parse. |

**New issue for Reader B — §11.6 (Linear Solvers):**

§11.6 is **critically inadequate** for this reader. It mentions PCG, Chebyshev, and FFT/DCT in single paragraphs. Reader B has never heard of any of these methods. They know Gaussian elimination (O(n³), impractical for large systems) but the chapter never makes this connection:

> "You know Gaussian elimination from linear algebra. It works perfectly — but for a million-node cardiac mesh, it would take years. Iterative methods find a good-enough answer in seconds by making successive approximations, each one closer to the true solution. Think of it like tuning a guitar: you don't solve for the exact frequency mathematically — you pluck, listen, and adjust."

This bridging paragraph is entirely missing. The student goes from "we must solve Ax = b" to "PCG converges in O(√κ) iterations" with no intermediate step.

**Reader B score: 3.0/5** (vs Reader A: 3.8) — significantly lower because the linear solver machinery is completely new and unexplained, and the chapter's "price" (solving Ax = b iteratively) is never motivated for a reader who doesn't know what iterative solving means.

---

### Part II Summary for Reader B

| Chapter | Reader A Score | Reader B Score | Delta | Key Reason |
|---------|---------------|---------------|-------|------------|
| 7 — Monodomain Equation | 4.6 | **4.8** | +0.2 | Physics is exactly in Reader B's wheelhouse |
| 8 — Spatial Discretization | 4.0 | **3.4** | -0.6 | Discretization itself is new; FEM unmotivated; implementation opaque |
| 9 — Operator Splitting | 4.4 | **3.8** | -0.6 | Computational motivation implicit; FE assumed before taught |
| 10 — Explicit Solvers | 4.8 | **4.2** | -0.6 | Time-stepping is new but well-taught; implementation layer lost |
| 11 — Implicit Solvers | 3.8 | **3.0** | -0.8 | Linear solvers completely unexplained; "solving Ax=b iteratively" is a black box |
| **Part II Average** | **4.0** | **3.4** | **-0.6** | |

**Grade: B** (vs Reader A's A-)

---

## Part III — The Bidomain Extension (Ch 12–17)

### Chapter 12 — The Bidomain Model — Score: 4.0/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | Balloon for divergence, glass dish for BCs. But more importantly: the *physiology* (§12.1) — intracellular current through gap junctions, extracellular current through interstitial fluid, transmembrane current through ion channels — is **exactly what this reader studied in neurophysiology**. They don't need the analogy; they already know the physics. The analogy still helps because it maps familiar physiology to the mathematical formalism. |
| L2 Feynman | **5** | Conservation of charge (∇·J_i + ∇·J_e = 0) is Kirchhoff's current law, which this reader knows. "No capacitor in the extracellular space" → no time derivative → elliptic constraint: this chain of reasoning maps perfectly to heat transfer (steady-state = Laplace equation = no ∂T/∂t). The "slave variable" characterization connects to their math background. |
| L3 3B1B | **3** | Same as before — Figure 12.1 is schematic, not geometric. |
| L4 Worked Example | **1** | Same gap — no concrete computation. |
| L5 Implementation | **1** | Engine boxes reference wrong codebase. |

**Reader B experience:** Chapter 12 is where this reader **shines**. The bidomain physics maps directly to their neurophysiology knowledge. They understand two conducting media (intracellular/extracellular), they know about anisotropy ratios from physiology courses, they understand the conservation principle from physics. The "unequal anisotropy ratio" argument for why monodomain fails is immediately convincing.

§12.5 (building the bidomain system) is entirely accessible — it's substituting Ohm's law into conservation equations and rearranging. This is standard applied math.

The parabolic-elliptic distinction (§12.6) maps to transient vs steady-state in heat transfer — a connection the text doesn't make but this reader would catch independently.

**Reader B score: 4.0/5** (vs Reader A: 3.6) — higher because the physics is familiar and the math is within their toolkit. Still penalized for zero worked examples and wrong implementation references.

---

### Chapter 13 — Spatial Discretization (Bidomain) — Score: 1.8/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **2** | Same issues as Reader A. The weak form arrives without motivation. For Reader B, this is even worse — they struggled with the weak form in Ch 8.4 and now it reappears for a harder system. |
| L2 Feynman | **1** | "Saddle-point problem," "condition number 10⁸–10¹⁰," "preconditioning essential" — Reader B has never heard any of these terms. Reader A at least had some computational context. Reader B is completely lost. They know Ax = b from linear algebra, but "indefinite block system" is not a concept they've encountered. "Condition number" was maybe mentioned once in lin alg — it wasn't important for analytical problem-solving. |
| L3 3B1B | **3** | The block matrix SVG exists but doesn't help Reader B understand what "indefinite" means. |
| L4 Worked Example | **2** | Same gap. |
| L5 Implementation | **1** | Irrelevant. |

**Reader B experience:** This is the **wall**. Reader B understood the bidomain *physics* beautifully in Ch 12. Now they're told "we discretize it" — but they struggled with discretization in Ch 8 and never fully grasped FEM. Now the discretization produces a "2×2 block saddle-point system" that is "indefinite" with "condition number 10⁸–10¹⁰." Every single computational term is foreign. The chapter makes no attempt to connect back to what Reader B knows (e.g., "This 2×2 block system is like having two coupled systems of equations from your linear algebra course — one for each domain").

**Reader B score: 1.8/5** (vs Reader A: 2.4) — even lower because the computational concepts that Reader A partially understood are completely opaque to Reader B.

---

### Chapters 14–17 — Abbreviated Scoring

These chapters progressively worsen for Reader B because they layer computational concept upon computational concept:

| Chapter | Reader A | Reader B | Key Issue for Reader B |
|---------|----------|----------|----------------------|
| 14 — Time Integration | 1.2 | **0.8** | Operator splitting for bidomain requires understanding splitting error, which requires understanding time-stepping schemes, which Reader B barely knows from Ch 10. "Semi-implicit" is a computational concept with no physics analog. |
| 15 — Parabolic Solvers | 2.4 | **1.2** | PCG, Chebyshev semi-iteration, spectral methods — Reader B has never heard of any iterative solver. The entire chapter is about *how to solve Ax=b efficiently*, which is a computational topic Reader B has zero background in. They know eigenvalues/eigenvectors from lin alg, so the spectral decomposition is followable in principle, but "preconditioning" and "Krylov subspace" are meaningless. |
| 16 — Elliptic Solvers | 1.0 | **0.5** | AMG (Algebraic Multigrid), Schur complement, FGMRES — every single word is computational jargon. Reader B cannot extract a single useful concept from this chapter. The Schur complement is a linear algebra concept they might know abstractly, but its computational significance ("reduces a 2N×2N system to an N×N system") requires understanding why 2N×2N is worse than N×N — which is a computational cost argument Reader B can't evaluate. |
| 17 — Implementation Roadmap | 1.4 | **0.5** | A chapter about code architecture for a reader who can't read code. |

---

### Part III Summary for Reader B

| Chapter | Reader A | Reader B | Delta |
|---------|----------|----------|-------|
| 12 — Bidomain Model | 3.6 | **4.0** | +0.4 |
| 13 — Spatial Discretization | 2.4 | **1.8** | -0.6 |
| 14 — Time Integration | 1.2 | **0.8** | -0.4 |
| 15 — Parabolic Solvers | 2.4 | **1.2** | -1.2 |
| 16 — Elliptic Solvers | 1.0 | **0.5** | -0.5 |
| 17 — Implementation Roadmap | 1.4 | **0.5** | -0.9 |
| **Part III Average** | **1.9** | **1.5** | **-0.4** |

**Grade: F** (vs Reader A's D+)

The pattern is stark: Reader B scores *higher* on Ch 12 (physics) and *lower* on everything else (computation). The gap between "understanding the physics" and "understanding the computation" is even wider for this reader.

---

## Part IV — Lattice-Boltzmann Methods (Ch 18–20)

### Chapter 18 — LBM: From Kinetic Theory to Computation — Score: 3.6/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **4** | Same quality as before. The ion-cataloging analogy for velocity space works. |
| L2 Feynman | **5** | **This is where Reader B's background pays off.** The Maxwell-Boltzmann distribution (§18.1) is something they may have encountered in thermodynamics or heat & mass transfer. Phase space, equilibrium distributions, the Gaussian as maximum entropy — these are concepts from statistical mechanics that a heat & mass transfer student has at least touched. The "three physical claims" (conservation, zero flux, isotropic variance) map directly to moment analysis in transport phenomena. Chapman-Enskog analysis as "deriving macroscopic equations from microscopic dynamics" connects to their continuum mechanics education. |
| L3 3B1B | **3** | Same single diagram. Still needs more. |
| L4 Worked Example | **4** | The BGK and moment-space worked examples are fully accessible — they're just arithmetic that Reader B can verify. |
| L5 Implementation | **1** | `torch.roll`, `LBMState`, `CollisionOperator(ABC)` — foreign language. |

**Reader B experience:** This is paradoxically the chapter where Reader B does **better** than Reader A in §18.1–18.2 but **worse** in §18.3–18.5.

§18.1–18.2 (kinetic theory, Boltzmann equation, BGK, Chapman-Enskog): Reader B recognizes the physics from heat & mass transfer. The Maxwell-Boltzmann distribution is familiar. The concept "macroscopic transport equations emerge from microscopic particle dynamics" may have been mentioned in their courses. The Boltzmann equation itself (streaming + collision) has a physical clarity that resonates. Reader B thinks: "I know this from statistical thermodynamics!"

§18.3 (discretization onto a lattice): Here Reader B hits the same wall as in Ch 8 — "the computer can't handle continuous velocity space, so we sample at discrete points." But the Gauss-Hermite quadrature framework (§18.4) is actually *more* accessible to Reader B than to Reader A, because Reader B knows numerical integration from calculus. "Replace an integral with a weighted sum at specific nodes" is a concept they've used, even if they haven't coded it.

§18.5 (moment space): The M·f transformation is matrix algebra that Reader B follows easily. The decode → relax → re-encode round trip is clear mathematically. But the "why this matters for implementation" argument (controlling relaxation rates independently) is a computational concern that Reader B doesn't fully appreciate.

**Reader B score: 3.6/5** (vs Reader A: 3.8) — slightly lower overall because the implementation layer is completely lost, but §18.1–18.2 actually scores higher for this reader.

---

### Chapter 19 — LBM for Monodomain — Score: 3.4/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **4** | "Faucet/drain" for source term works. "Stiff spring" for τ-D inversion works. |
| L2 Feynman | **5** | The Ω^NR / Ω^R decomposition connects to this reader's understanding of reaction-diffusion from heat & mass transfer. The "why slower relaxation = faster diffusion" insight box would resonate — it maps to "lower resistance → higher conductivity." The anisotropy explanation (§19.4) connects to directional thermal conductivity, which they know. |
| L3 3B1B | **3** | Same missing diagrams. |
| L4 Worked Example | **5** | MRT anisotropy example and stimulus example are fully accessible — pure arithmetic. |
| L5 Implementation | **1** | `LBMSimulation`, `BGKCollision`, `MRTCollision`, `D2Q5` frozen dataclasses, `torch.roll` — completely inaccessible. The 6-step algorithm box (eq 19.11) is readable as *math* but the LaTeX pseudocode (`\text{IonicModel.compute\_Iion}`) mixes code and math in a way that confuses Reader B. |

**New issue for Reader B — §19.5 (Boundary Conditions):**

The bounce-back BC section is actually **more accessible** to Reader B than the classical BC implementations in Ch 8.7. Why? Because "a ball bounces off a wall" is physically intuitive, and the formula f_ī = f_i* is a single equation — no stiffness matrix modification, no ghost nodes, no assembly step. For a reader who struggled with Ch 8.7's BC implementation, bounce-back is refreshingly simple.

However, §19.7 (Memory Layout and GPU Acceleration) is a complete loss for Reader B. "Structure of Arrays," "memory coalescing," "warp processes adjacent nodes" — this is GPU architecture, not physics or math.

**Reader B score: 3.4/5** (vs Reader A: 4.4) — significantly lower because the implementation layer (which scored 5 for Reader A) is inaccessible.

---

### Chapter 20 — LBM for Bidomain — Score: 3.2/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | Fish tanks, heated plates, hammer vs pen — all work universally. |
| L2 Feynman | **5** | **This reader gets the core concept better than Reader A.** "Parabolic = time-evolving, elliptic = steady-state constraint" is exactly the transient/steady-state distinction from heat transfer. "Add a fictitious time derivative and march to steady state" is a technique they may have seen in iterative solution of Laplace's equation (common in heat & mass transfer courses). The "evolution vs constraint" tension is crystalline for this reader. |
| L3 3B1B | **2** | Same missing diagrams. |
| L4 Worked Example | **1** | Same gap. |
| L5 Implementation | **1** | `DualLatticeBidomainLBMSolver` — meaningless. |

**Reader B experience:** Chapters 20.1–20.2 are surprisingly strong for this reader. The steady-state relaxation technique (pseudo-time) maps directly to methods they've seen in heat transfer for solving Laplace's equation on complex geometries. The bidomain physics is clear from Ch 12. The challenge (parabolic + elliptic coupled system) is stated in terms they understand.

§20.3 (hybrid solver) and §20.4 (dual-lattice) are conceptually clear but implementation-opaque. Reader B understands *what* each strategy does but cannot evaluate the computational trade-offs ("pseudo-time takes 100 iterations" — is that a lot? They have no reference point).

**Reader B score: 3.2/5** (vs Reader A: 3.0) — slightly higher because the physics resonates more, offsetting the lost implementation layer.

---

### Part IV Summary for Reader B

| Chapter | Reader A | Reader B | Delta | Key Reason |
|---------|----------|----------|-------|------------|
| 18 — LBM Theory | 3.8 | **3.6** | -0.2 | Kinetic theory is familiar, but implementation lost |
| 19 — LBM Monodomain | 4.4 | **3.4** | -1.0 | The entire implementation layer (L5 = 5 for Reader A) is invisible |
| 20 — LBM Bidomain | 3.0 | **3.2** | +0.2 | Physics resonance offsets computational opacity |
| **Part IV Average** | **3.7** | **3.4** | **-0.3** | |

**Grade: B** (vs Reader A's B+)

---

## Cross-Part Analysis: Where Reader B Struggles vs Thrives

### Where Reader B Scores HIGHER Than Reader A

| Topic | Why |
|-------|-----|
| Ch 7 (Monodomain PDE) | The PDE is the heat equation with a source — exactly in Reader B's education |
| Ch 12 (Bidomain physics) | Two-domain physiology is their neurophysiology background |
| §18.1–18.2 (Kinetic theory) | Maxwell-Boltzmann, Chapman-Enskog are from statistical thermodynamics |
| §20.1–20.2 (Elliptic challenge, pseudo-time) | Steady-state relaxation is standard in heat transfer courses |

### Where Reader B Scores LOWER Than Reader A

| Topic | Why | Gap Size |
|-------|-----|----------|
| Ch 8 (FDM/FEM/FVM) | Discretization is entirely new | -0.6 |
| Ch 11 (Implicit solvers, linear solvers) | "Solving Ax=b iteratively" is a black box | -0.8 |
| Ch 15 (PCG, Chebyshev, Spectral) | Never heard of iterative solvers | -1.2 |
| Ch 16 (AMG, Schur complement, FGMRES) | Every word is computational jargon | -0.5 |
| Ch 17 (Implementation roadmap) | Can't read code | -0.9 |
| Ch 19 (LBM implementation) | GPU architecture, memory layout, code mapping — all lost | -1.0 |

### The Pattern

Reader B's scores form a **two-tier structure**:
- **Physics/math chapters**: Score well (3.4–5.0). Reader B understands the PDEs, the conservation laws, the boundary conditions, the physical motivation.
- **Computational method chapters**: Score poorly (0.5–3.0). The gap is not in understanding *what* needs to be computed, but *how* the computation works at an algorithmic level.

The textbook's L5 layer ("Implementation Detail") is designed for someone who can read pseudocode and map it to engine code. Reader B cannot access this layer at all. Since L5 typically accounts for 20% of each chapter's score, Reader B starts with a structural 1-point penalty on every chapter that has significant L5 content.

---

## Overall Scores — Full Comparison

| Chapter | Reader A | Reader B | Delta |
|---------|----------|----------|-------|
| 7 — Monodomain Equation | 4.6 | **4.8** | +0.2 |
| 8 — Spatial Discretization | 4.0 | **3.4** | -0.6 |
| 9 — Operator Splitting | 4.4 | **3.8** | -0.6 |
| 10 — Explicit Solvers | 4.8 | **4.2** | -0.6 |
| 11 — Implicit Solvers | 3.8 | **3.0** | -0.8 |
| 12 — Bidomain Model | 3.6 | **4.0** | +0.4 |
| 13 — Spatial Disc (Bidomain) | 2.4 | **1.8** | -0.6 |
| 14 — Time Integration | 1.2 | **0.8** | -0.4 |
| 15 — Parabolic Solvers | 2.4 | **1.2** | -1.2 |
| 16 — Elliptic Solvers | 1.0 | **0.5** | -0.5 |
| 17 — Implementation Roadmap | 1.4 | **0.5** | -0.9 |
| 18 — LBM Theory | 3.8 | **3.6** | -0.2 |
| 19 — LBM Monodomain | 4.4 | **3.4** | -1.0 |
| 20 — LBM Bidomain | 3.0 | **3.2** | +0.2 |
| | | | |
| **Part II Average** | **4.0** | **3.4** | **-0.6** |
| **Part III Average** | **1.9** | **1.5** | **-0.4** |
| **Part IV Average** | **3.7** | **3.4** | **-0.3** |
| **Full Textbook Average** | **3.0** | **2.5** | **-0.5** |

---

## Recommendations Specific to Reader B

### R1: Add a "Why Discretize?" Section Before Chapter 8 (Critical)

Reader B has never discretized a PDE. They need a 1-page section (in Ch 7 or as a Ch 8 preamble) explaining:

> "You know how to write the diffusion equation as a PDE. You can solve it analytically for simple geometries (a slab, a cylinder) using separation of variables. But the heart is not a slab — it is an irregularly shaped, anisotropic, heterogeneous medium with nonlinear ionic currents at every point. No analytical solution exists. The only path forward is to let a computer approximate the solution at thousands of discrete points, marching forward in time step by step. This is called *numerical simulation*, and the rest of this textbook teaches you how it works."

This single paragraph bridges the gap between "I know the physics" and "I need a computer."

### R2: Motivate the Weak Form in §8.4 for a Non-Computational Reader

Add before the test-function machinery:

> "FDM works beautifully on regular grids, but the heart is not regular — it has curved surfaces, varying wall thickness, and fiber orientation that changes continuously. To handle these geometries, we need a method that doesn't require a rectangular grid. The Finite Element Method achieves this by recasting the PDE in a different mathematical form — one that asks 'does the equation hold *on average* over each region?' instead of 'does it hold at every point?' This weaker requirement allows irregular meshes and curved boundaries."

### R3: Bridge "Ax = b" Between Linear Algebra and Computation (§11.6)

Reader B knows Gaussian elimination. They need:

> "In your linear algebra course, solving Ax = b meant row reduction — O(n³) operations. For a cardiac mesh with 10⁶ nodes, that's 10¹⁸ operations — billions of years on any computer. Iterative methods are the escape: instead of finding the exact answer, they find an approximate answer that improves with each iteration. After 10–30 iterations (seconds of computation), the answer is accurate to machine precision. The key insight: for sparse matrices (where each row has only 5–10 nonzero entries), each iteration costs only O(n), not O(n²)."

### R4: Reduce Implementation Layer Weight for This Audience

The style guide's Layer 5 (Implementation Detail) serves readers who can write code. For Reader B, the engine boxes and pseudocode are noise — not harmful, but consuming space that could be used for more physics/math content. Consider:
- Making engine boxes collapsible (in the HTML) or visually distinct ("For implementers only")
- Adding a "math-only" reading path that skips implementation sections

### R5: Exploit Reader B's Heat & Mass Transfer Background

Several connections are missed that would help this reader enormously:

| Textbook Concept | Heat & Mass Transfer Analog | Where to Add |
|------------------|-----------------------------|--------------|
| CFL condition (Ch 10) | Explicit Euler stability for heat equation — same formula | §10.1 insight box |
| Crank-Nicolson (Ch 11) | Trapezoidal rule for transient heat conduction | §11.3 opening |
| Pseudo-time relaxation (Ch 20) | Iterative solution of Laplace equation for steady-state temperature | §20.2 opening |
| Monodomain equation (Ch 7) | Heat equation with volumetric source term | §7.2 (already implicit, make explicit) |
| Bidomain elliptic eq (Ch 12) | Laplace/Poisson equation for steady-state potential | §12.6 |
| FVM (Ch 8.5) | Control volume analysis for conservation equations | §8.5 opening (already good) |
| Bounce-back BC (Ch 19.5) | Reflection condition at insulated boundary | §19.5.1 |

Adding one-sentence connections ("This is the cardiac analogue of the CFL condition you saw for the explicit heat equation") would anchor each concept in familiar territory.
