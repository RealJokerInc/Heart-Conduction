# Bidomain Chapters Audit (Ch 12–17)

**Perspective**: Student with monodomain knowledge, intro linear algebra, no bidomain or heavy numerical analysis background.
**Scope**: Chapters 12–17 of `bidomain_textbook.html` vs. actual implementation in `Bidomain/Engine_V1/`.
**Evaluated against**: `STYLE_GUIDE.md` (Feynman style, 5-layer complexity, 3B1B visual intuition)
**Date**: 2026-03-08

---

## Executive Summary

Part III has two independent failure modes that compound into an unreadable result:

1. **The content describes a solver architecture that doesn't exist in the code.** The textbook presents a monolithic 2N×2N indefinite block system solved by FGMRES + AMG. The code uses decoupled N×N SPD solves. A student reading these chapters and opening the code would be completely lost.

2. **The writing abandons the Feynman style after Chapter 12.** Chapter 12 follows the layered-complexity principle beautifully (physical analogies → equations → interpretation). Chapters 13–17 deteriorate into a reference-manual catalogue of methods with no narrative thread, no physical analogies, no worked examples, and no visualizations. The difficulty spikes from "accessible intro" to "graduate numerics seminar" between Ch 12.6 and Ch 13.2 with no bridge.

The combined effect: the student understands *what* the bidomain equations are (Ch 12 works), but has no idea *how* to solve them (Ch 13–17 fail), and the "how" they're taught doesn't match the code anyway.

---

## Part A: Readability, Flow, and Pedagogical Quality

### Chapter-by-Chapter Scoring

Each chapter is scored 1–5 on each Style Guide layer:
- **L1 (ELI5/Analogy)**: Physical analogy or plain-language intro before hard concepts
- **L2 (Feynman/Conceptual)**: "Why does this work?" explained without notation overload
- **L3 (3B1B/Visual)**: Diagrams, SVGs, geometric intuition alongside equations
- **L4 (Worked Example)**: Concrete numbers, step-by-step arithmetic
- **L5 (Implementation)**: Pseudocode/code-level detail matching the engine

Score: 5=exemplary, 4=good, 3=adequate, 2=weak, 1=missing/failing

---

### Chapter 12 — The Bidomain Model — Score: 3.6/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | Balloon analogy for divergence (12.2), rubber sheet for elliptic (12.6), glass dish for BCs (12.9), altitude for null space. Excellent. |
| L2 Feynman | **4** | "No capacitor in extracellular space" (12.5) is a gem. Conservation identity insight box is well-placed. Minor: repeats the "no time derivative" point 3 times. |
| L3 3B1B | **3** | Figure 12.1 (monodomain vs bidomain boxes) is decent but schematic, not geometric. Missing: diagram of current flowing in two domains, visual of φ_i → Vm + φ_e substitution. |
| L4 Worked Example | **1** | Zero worked examples in the entire chapter. The anisotropy ratio table (12.7) gives numbers, but no computation is performed. The student never plugs σ_i, σ_e values into the PDEs to see what comes out. |
| L5 Implementation | **2** | Engine boxes reference V5.4 (wrong codebase). No pseudocode, no code snippets showing how the engine represents Di, De, Vm, φ_e. |

**Flow issues:**
- **12.4 (Monodomain Review) breaks the narrative.** We're mid-derivation of bidomain. Sections 12.1–12.3 build momentum: "here's why we need two domains." Then 12.4 suddenly reviews monodomain. This belongs *before* 12.1, as a brief recap, or inline as a "compare this to..." paragraph within 12.5. Placing it as its own section stops the story dead.
- **12.7 (When Bidomain Reduces to Monodomain) kills momentum.** Section 12.6 just explained why the parabolic–elliptic couple is hard. Then 12.7 says "but sometimes you don't need it." This deflates the narrative at the worst moment. Move this to a sidebar or to the end of the chapter.
- **12.8 (Conductivity Tensors and Fiber Architecture) is premature.** Rotation matrices and transmural fiber rotation are implementation details that belong in Ch 13 or Ch 17. At this point the student hasn't even seen the discrete system. The 3×3 tensor rotation equation is the most intimidating equation in Ch 12, and it's not needed for understanding the bidomain PDE.
- **12.9 (BCs) is extensive but premature.** The student hasn't seen the discrete block system yet. Discussing how BCs "modify the stiffness matrices and the block system" refers to content that doesn't arrive until Ch 13.4. This creates a dangling forward reference that frustrates the reader.
- **Repetition**: The "no time derivative → constraint" explanation appears in 12.5 (inline), 12.5 (insight box), 12.6 (paragraph 2), and 12.6 (insight box). Four times for the same idea. Once well-stated is enough.

**What works well:**
- 12.1–12.3 are among the best writing in the entire textbook. The physical picture is clear, the motivation is compelling, and the unequal-anisotropy argument is crisp.
- The conservation identity insight box (12.2) is a model of how to present a key result.
- The "slave variable" characterization of φ_e is excellent intuition.

---

### Chapter 13 — Spatial Discretization — Score: 2.4/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **2** | Glass-dish analogy in 13.4 is good, but 13.1–13.3 have zero analogies. The weak form (13.2) arrives with no ELI5 whatsoever — just test functions and H^1(Ω). |
| L2 Feynman | **2** | 13.1 ("call the assembly routine twice") is clear but thin. 13.2–13.3 are pure mathematical exposition with no "why does this work?" explanation. "Saddle-point problem" is introduced as a label, never explained. |
| L3 3B1B | **3** | Figure 13.1 (block matrix SVG) is the only diagram. It shows colored boxes but doesn't visualize what the numbers look like, or what "indefinite" means geometrically. Missing: stencil diagram for bidomain FDM, energy-surface visualization for saddle-point. |
| L4 Worked Example | **2** | The 5-node cable example (13.4.3) is a good start but only demonstrates null-space pinning. It never solves the actual bidomain system. K_i and K_e matrices are shown symbolically, never with numbers plugged in. |
| L5 Implementation | **2** | Engine box says "call assembly twice" — correct but trivially obvious. Doesn't mention the face-based stencil, doesn't show code for building L_i and L_e. |

**Flow issues:**
- **13.2 (Weak Form) is a massive, unannounced difficulty spike.** The student has seen FEM in Ch 8 with gentle hand-holding (hat functions, assembly, worked example). Section 13.2 drops them into bilinear forms with $v \in H^1(\Omega)$, integration by parts of coupled equations, and a 2×2 block semi-discrete system — all in under 30 lines. There is no ELI5 bridge ("imagine testing whether the equations hold at each node..."), no diagram, no worked example. This violates Style Guide rule #1: "Math-first opening — jumps into equations without physical motivation."
- **13.3 introduces "saddle-point" as a buzzword.** The term appears and is immediately followed by implications (CG can't be used, condition number 10^8–10^10, preconditioning essential). For the target student, every single one of these terms needs explanation. What's a condition number? Why 10^8? What is preconditioning? The monodomain chapters never needed these concepts because the system was always SPD. The student is drowning.
- **13.4 repeats 12.9 almost verbatim.** The three canonical BC scenarios, the glass-dish analogy, the null-space insight box — all appeared in 12.9, and they appear again in 13.4. This violates Style Guide rule #5: "Repeated content — same concept explained in two chapters without cross-reference." Either 12.9 or 13.4 should be cut; the other should cross-reference.
- **13.4 is 180+ lines of BC combinatorics.** Four subsections (13.4.1–13.4.4), each with equations and tables. The reader is overwhelmed with BC permutations before seeing a single solved example. The student thought they were learning spatial discretization — suddenly they're deep in boundary condition theory.
- **No transition from "here's the matrix" to "here's how we solve it."** The chapter ends with a summary table and forward pointers. The student has a 2×2 block system and no idea what to do with it. Chapter 14 should immediately answer this, but instead it opens with a discussion of operator splitting theory.

**What works well:**
- 13.1's opening paragraph connecting back to Ch 8 is a good anchor.
- The block definitions table in 13.3 (A11 = M/Δt + K_i, etc.) is a clear reference.
- The SVG block matrix diagram is visually clean.

---

### Chapter 14 — Time Integration and Operator Splitting — Score: 1.8/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **1** | Zero physical analogies in the entire chapter. No "imagine..." paragraph for any method. Compare to Ch 9's "stir-fry" analogy for operator splitting — nothing equivalent exists here. |
| L2 Feynman | **2** | 14.1's insight box (monodomain vs bidomain splitting) is the one bright spot. The rest is method cataloguing: "here's CN, here's SDIRK2, here's IMEX, here's RKC." No unifying narrative, no guiding question answered. |
| L3 3B1B | **1** | Zero diagrams. No stability region plots. No comparison of CN oscillations vs SDIRK2 damping. No visual for the splitting procedure applied to bidomain. Chapter 10 had SVG slope diagrams for RK2 and RK4 — nothing equivalent here. |
| L4 Worked Example | **1** | Zero worked examples. Not a single number is computed. The SBDF2 equation is the most complex single equation in the bidomain chapters, and it arrives without any concrete computation showing what it does to actual voltage values. |
| L5 Implementation | **1** | Engine box references V5.4 explicit solvers (wrong for bidomain). No pseudocode for any method. The actual bidomain time step algorithm — the single most important piece of information for a student trying to write code — is completely absent. |

**Flow issues:**
- **The chapter is a catalogue, not a narrative.** It presents five time integration methods (CN, SDIRK2, IMEX-Euler, SBDF2, RKC) one after another. There is no story connecting them. Compare to Ch 10–11 in Part II, which follow the arc: "FE is simple but limited → CFL forces small steps → implicit removes the CFL wall → here's how implicit works." Ch 14 has no equivalent arc. It's a menu with no recommendation.
- **14.1 contradicts itself in the opening.** The chapter intro says "the same splitting framework applies" and immediately follows with "but the diffusion sub-step is fundamentally different." This whiplash confuses the reader. Is it the same or different? (Answer: the splitting is the same; only the diffusion solve changes. But the text doesn't resolve this tension clearly.)
- **SDIRK2 is dropped cold (14.2).** The student knows Forward Euler, CN, BDF from Part II. SDIRK2 introduces multi-stage implicit RK methods, Butcher tableaux (implicitly), L-stability, and the specific coefficient γ = 1 - √2/2 — all without motivation. No ELI5, no "imagine taking two smaller implicit steps instead of one big one." Then the code doesn't use SDIRK2 anyway. Dead content.
- **IMEX (14.3) introduces four new acronyms in one section.** IMEX, SBDF, DIRK, IMEX-RK. Each is a complex method family. The student is hit with SBDF2's equation — the most notation-dense equation in Part III — without seeing a single worked example. Compare the explicit solvers chapter (Ch 10): RK2 gets a diagram, a worked example, and step-by-step matrix forms. SBDF2 gets... nothing.
- **14.4 ("Why Explicit Fails") is the best-placed section but arrives too late.** This section clearly explains the one key insight: the elliptic equation is a constraint, so at least one linear solve per step is unavoidable. This should come *first* in the chapter (or at the end of Ch 13), because it's the motivation for everything else. Instead it's buried as section 4 of 5.
- **The complete bidomain time step algorithm is never shown.** Chapter 9 gives boxed algorithms for Godunov and Strang splitting applied to monodomain. Chapter 14 should give the equivalent for bidomain — "at each step: (1) ionic update, (2) parabolic solve, (3) elliptic solve" — in a clear boxed format. This is the single most important missing piece.

**What works well:**
- The insight box distinguishing monodomain splitting error from bidomain splitting error (14.1) is genuinely illuminating.
- The "elliptic equation cannot be marched in time" argument (14.4) is clear and well-stated.
- The "spectrum of explicitness" insight box (14.5) is a useful mental model.

---

### Chapter 15 — Parabolic Solvers — Score: 3.0/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **2** | No new analogies. PCG, Chebyshev, and spectral methods were already introduced in Part II / Appendix A. The chapter doesn't add any new physical intuition. |
| L2 Feynman | **3** | Warm-starting explanation (15.1) is practical and well-motivated. Chebyshev as "no sync points" (15.2) is clear. Spectral as "transform → divide → inverse transform" (15.3) is well-structured. |
| L3 3B1B | **3** | Comparison tables for PCG vs Chebyshev and solver selection are useful. But no new diagrams. Compare to Appendix A which has worked CG examples with geometric interpretations. |
| L4 Worked Example | **1** | No worked examples. The spectral solver gets a step-by-step algorithm but no actual numbers. A 4×4 grid example showing the DCT solve for a bidomain parabolic sub-step would solidify understanding. |
| L5 Implementation | **3** | Engine boxes describe PCGSolver and ChebyshevSolver classes with reasonable detail. But they reference V5.4 files, not the bidomain engine. |

**Flow issues:**
- **This chapter is redundant.** The intro says "the A11 block looks exactly like the monodomain stiffness-plus-mass matrix." If it looks exactly the same, why does it need its own chapter? The monodomain solver chapters (Ch 11, Appendix A) already cover PCG, Chebyshev, and spectral methods in detail. This chapter should either (a) be eliminated and replaced with a brief note ("the parabolic solver is identical to monodomain — see Ch 11"), or (b) focus exclusively on what's *different* about the bidomain parabolic solve (answer: nothing, really — which is the whole point).
- **15.3 is exhaustive where it should be concise.** Four sub-sections (DCT, DST, Mixed, FFT) cover every spectral variant. The student doesn't need all four — they need to know that DCT handles Neumann, DST handles Dirichlet, and mixed BCs combine them. The FFT subsection (periodic BCs) is rarely relevant to cardiac simulation and adds bulk.
- **The spectral denominator (15.1) uses σ-notation.** χ·Cm appears in the formula, but the code's spectral solver uses 1/dt (Formulation B). A student trying to verify the formula against `spectral.py` will be confused immediately.

**What works well:**
- The warm-starting table (cold start vs warm start iteration counts) is concrete and useful.
- The final solver comparison table (DCT/DST/FFT/PCG/Chebyshev) with complexity, sync points, and best-use scenarios is a well-structured decision aid.

---

### Chapter 16 — Elliptic Solvers: The Schur Complement — Score: 1.4/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **1** | Zero physical analogies. No ELI5 for saddle-point, Schur complement, AMG, or Krylov methods. Every concept is introduced at Layer 3 (math) or Layer 5 (library names). |
| L2 Feynman | **1** | The Schur complement insight box says "it represents the effective equation for φ_e after eliminating Vm" — this is the one conceptual sentence in the entire chapter. Everything else is definitions and formulas. |
| L3 3B1B | **1** | Zero diagrams. A chapter about block preconditioners, multigrid hierarchies, and Krylov subspaces has no visualizations at all. Multigrid is inherently visual (coarse/fine grids), yet no multigrid diagram appears. The saddle-point structure has a natural geometric picture (the energy surface), yet no energy-surface plot appears. |
| L4 Worked Example | **1** | Zero worked examples. The Schur complement formula S = A22 - A21·A11^{-1}·A12 is given without any numerical demonstration. A 4×4 example would make this concrete. |
| L5 Implementation | **1** | Recommends FGMRES(30) + Block LDU + AMG — none of which exist in the code. Lists library names (AMGX, Hypre, PyAMG, Trilinos) that the student will never use with this engine. |

**Flow issues:**
- **This is the worst chapter in Part III and arguably in the entire textbook.** It violates nearly every rule in the Style Guide:
  - Starts with jargon, not physics (rule #1: "Math-first opening")
  - No ELI5 for any concept (rule #2: "Missing ELI5")
  - Equations without derivation (rule #3: "Disconnected equations")
  - No worked example (rule #4)
  - Stops at Layer 3 for most content (rule #9)
  - No visuals (rule #10)
- **The chapter describes a solver stack that doesn't exist.** This is both a content-accuracy problem (see Critical issue C1) and a pedagogical problem: the student is learning how to solve a problem using tools they will never touch. The entire learning effort is wasted.
- **AMG gets one paragraph.** AMG is one of the most important algorithms in computational science. A one-paragraph treatment that doesn't explain how V-cycles work, what "algebraically coarsening" means, or why smooth errors are cheap to handle — this is name-dropping, not teaching. The student comes away knowing the acronym but not the idea.
- **Krylov methods (16.4) is a dictionary.** Three paragraphs, each defining one method (MINRES, GMRES, FGMRES). No intuition for what a Krylov subspace is, no explanation of why minimizing the residual over a growing subspace converges, no comparison of how they handle indefiniteness differently. The target student has only seen CG from Appendix A. MINRES and GMRES are significant extensions that need genuine explanation.
- **16.5 (Null-Space) is the one adequate section.** Pin-one-node and DCT zero-mode handling are clearly explained with practical advice. This section could be extracted and placed in a better chapter.

**What works well:**
- The Schur complement physical interpretation ("effective equation for φ_e after eliminating Vm") is a good idea — it just needs more development.
- Section 16.5 on null-space handling is clear and actionable.

---

### Chapter 17 — Implementation Roadmap — Score: 1.6/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **1** | No analogies. |
| L2 Feynman | **2** | The monodomain vs bidomain comparison table is a decent conceptual summary. |
| L3 3B1B | **2** | Tables are adequate but there are no diagrams showing the engine architecture. |
| L4 Worked Example | **1** | No worked examples, no code snippets, no pseudocode. |
| L5 Implementation | **1** | Describes an architecture (FGMRES+AMG+SDIRK2) that doesn't exist. Points to no actual files. A student trying to follow this "roadmap" would get lost immediately. |

**Flow issues:**
- **Only 1.5 pages long.** For a chapter titled "Implementation Roadmap," this is remarkably thin. Compare to the monodomain implementation coverage in Part II, which weaves engine connections throughout each chapter.
- **Recommends a solver stack no one has built.** The recommendation box (FGMRES(30), Block LDU, AMGX) sounds authoritative but points to vapor. The student reads it as "this is what to build" and then finds... it doesn't exist.
- **No file paths, class names, or function signatures from the actual bidomain engine.** The entire point of Layer 5 (Implementation Detail) is to map math to code. This chapter maps math to imaginary code.

---

### Overall Part III Scoring Summary

| Chapter | L1 | L2 | L3 | L4 | L5 | Avg | Style Grade |
|---------|----|----|----|----|-----|-----|-------------|
| 12: Bidomain Model | 5 | 4 | 3 | 1 | 2 | 3.0 | B- (strong start, no worked example) |
| 13: Spatial Disc. | 2 | 2 | 3 | 2 | 2 | 2.2 | D+ (difficulty spike, weak form cold) |
| 14: Time Integration | 1 | 2 | 1 | 1 | 1 | 1.2 | F (catalogue, no narrative, no examples) |
| 15: Parabolic Solvers | 2 | 3 | 3 | 1 | 3 | 2.4 | C- (redundant with Part II) |
| 16: Elliptic Solvers | 1 | 1 | 1 | 1 | 1 | 1.0 | F (jargon wall, nonexistent code) |
| 17: Roadmap | 1 | 2 | 2 | 1 | 1 | 1.4 | F (vaporware roadmap) |
| **Part III Average** | **2.0** | **2.3** | **2.2** | **1.2** | **1.7** | **1.9** | **D** |

For comparison, Part II (monodomain) scores roughly 4.0 average across all layers. The drop from Part II to Part III is dramatic and would be jarring to any reader who enjoyed the monodomain chapters.

---

### The Narrative Arc Problem

The deepest structural issue is that **Part III has no narrative arc**. Part II tells a clear story:

> *Here's the PDE (Ch 7) → here's how to turn it into matrices (Ch 8) → here's how to split the problem (Ch 9) → here are explicit solvers (Ch 10) → here are implicit solvers (Ch 11)*

Each chapter answers one question, and the chapters build on each other sequentially. A student finishing Ch 11 can implement a monodomain solver from scratch.

Part III's arc is broken:

> *Here's the PDE (Ch 12, good) → here's the block matrix (Ch 13, difficulty spike) → here are many time steppers (Ch 14, catalogue) → here are parabolic solvers (Ch 15, redundant) → here are elliptic solvers (Ch 16, wrong architecture) → here's a roadmap (Ch 17, vaporware)*

The student finishes Ch 17 and **cannot implement a bidomain solver**. They don't know the complete algorithm. They don't know what the code does. They've been taught solver machinery (FGMRES, AMG, Schur complement) that the engine doesn't use.

The arc should be:

> *Here are the PDEs and why they're coupled (Ch 12) → here's how to turn them into two matrices (Ch 13) → here's the complete algorithm: split them and solve sequentially (Ch 14, with boxed algorithm) → here are the solver options for each sub-problem (Ch 15) → here's how the code puts it all together (Ch 16)*

This gives the student: understanding → discretization → **the algorithm** → solver details → implementation. The missing piece is "the algorithm" — a clear, boxed, numbered procedure that says exactly what happens at each time step.

---

## Part B: Content Accuracy Issues (Code vs. Textbook)

### CRITICAL Issues

#### C1. Monolithic solver (textbook) vs decoupled solver (code)

**Location**: Chapters 13–17 (pervasive)

The entire solver narrative is built around a 2N×2N block system solved monolithically. The code uses **decoupled N×N SPD solves**: Gauss-Seidel sequential, semi-implicit, Jacobi parallel, IMEX-SBDF2, and explicit RKC — none described in the textbook. FGMRES, MINRES, AMG, block LDU preconditioners, and Schur complements don't exist in the code.

**Fix**: Rewrite Ch 14 around the decoupled approach. Present the complete Gauss-Seidel algorithm as the primary method. Present the monolithic approach (Ch 16) as "the literature alternative" in a sidebar.

#### C2. Formulation mismatch: σ-notation (textbook) vs D-notation (code)

**Location**: Every equation in Ch 12–15

The textbook writes χ·Cm·∂Vm/∂t and uses χ·Cm in the spectral denominator (15.1). The code uses Formulation B (D = σ/(χ·Cm)) where the parabolic operator uses 1/dt scaling. Every matrix equation has different coefficients from the code.

**Fix**: Add a prominent "Convention Translation" box at the start of Ch 12. Optionally rewrite all discrete equations in Formulation B.

#### C3. Chapter 17 describes a nonexistent architecture

**Location**: Chapter 17 (all of it)

Recommends FGMRES(30) + Block LDU + AMG V-cycles + SDIRK2. None exist. The actual engine is `Bidomain/Engine_V1/` with five decoupled strategies, three-tier spectral/PCG elliptic solver, and CN with θ parameter.

**Fix**: Rewrite to describe the actual Engine_V1 architecture with real class names, file paths, and factory functions.

---

### MAJOR Issues

#### M1. Face-based stencil not explained
Ch 13 implies the monodomain FDM stencil is reused. The code had to switch to a face-based symmetric stencil because the ghost-node approach breaks the elliptic equation (no identity term → asymmetric → non-SPD → PCG fails). This hard-won lesson is completely absent.

#### M2. No boxed algorithm for the complete bidomain time step
The most important missing piece. Chapter 9 has boxed Godunov and Strang algorithms for monodomain. The bidomain equivalent — ionic step → parabolic solve → elliptic solve — is never shown.

#### M3. "Saddle-point" unexplained for target audience
"Symmetric but indefinite" is stated as fact without geometric intuition, concrete example, or analogy. The target student has never encountered indefinite systems.

#### M4. IMEX equations use wrong scaling and wrong structure
The SBDF2 equation uses χ·Cm/Δt scaling (σ-notation) and presents Vm and φ_e in one monolithic equation. The code uses decoupled Formulation B with 1/dt scaling.

#### M5. Three-tier elliptic solver strategy undocumented
The code's auto-selection (spectral → PCG+spectral → PCG+GMG) based on BoundarySpec properties is the engine's most elegant design pattern. Not mentioned anywhere.

#### M6. No worked example for the full bidomain system
The monodomain chapters have excellent worked examples (5-node cable, 4-node FEM mesh). Part III has one partial example (null-space pinning in 13.4.3) but never shows the complete solve.

#### M7. BoundarySpec → solver selection logic undocumented
The code's BoundarySpec protocol determines which solver tier is used. The textbook discusses BCs in theory but never connects them to solver-selection logic.

---

### MODERATE Issues

| # | Issue | Location |
|---|-------|----------|
| Mo1 | **5 cross-reference errors**: Ch 14 intro says "Chapter 14" (should be 13), Ch 13.4.4 says "Ch 7.10" (nonexistent), Ch 13.4.4 says "Chapter 15" for null-space (should be 16), Ch 14.1 says "Chapter 8" (should be 9), Eq 15.1/15.2 appear in Ch 14 not Ch 15 | Multiple |
| Mo2 | **SDIRK2 taught but not used in code.** γ = 1-√2/2, Butcher tableaux, two implicit stages — all for a method the engine doesn't implement | Ch 14.2 |
| Mo3 | **Engine V5.4 connection boxes reference wrong codebase.** Spectral solver is `spectral.py` not `fft.py`, state is `BidomainState` not monodomain State, diffusion stepping is in `solver/diffusion_stepping/` (new directory) | Ch 12–16 |
| Mo4 | **φ_i substitution step skipped.** Goes from ∇·(D_i·∇φ_i) to eq (12.4) without showing ∇·(D_i·∇(Vm+φ_e)) expansion | Ch 12.5 |
| Mo5 | **Conservation law addition not shown.** "Add both conservation laws" → jumps to eq (12.5) without intermediate algebra (4 lines) | Ch 12.5 |
| Mo6 | **Null-space: 4 strategies given equal weight, only 2 used.** Pin-one-node and DCT zero-mode are in the code; integral constraint and deflation are not | Ch 16.5 |
| Mo7 | **"70–85% runtime" claim unjustified.** Why is the elliptic solve the bottleneck? No conditioning analysis, no comparison to parabolic solve cost | Ch 16 intro |

---

### MINOR Issues

| # | Issue | Location |
|---|-------|----------|
| m1 | K_i (textbook) vs L_i (code) naming inconsistency | Ch 13 |
| m2 | Equations (15.1), (15.2) appear in Ch 14, not Ch 15 | Ch 14, line 6993 |
| m3 | Block definitions table mixes time-stepped (A11) and static (A12) forms | Ch 13.3 |
| m4 | "Chapter 10" should be "Chapter 11" for implicit solvers | Ch 14, line 6897 |
| m5 | D_eff harmonic mean written as scalar but D_i, D_e are tensors | Ch 12.7 |

---

## Part C: Structural Recommendations

### R1. Restructure Part III as a 4-chapter arc

**Current** (6 chapters, broken arc):
> Ch 12 (PDEs) → Ch 13 (Block matrix) → Ch 14 (Time steppers catalogue) → Ch 15 (Parabolic solvers) → Ch 16 (Elliptic solvers) → Ch 17 (Roadmap)

**Proposed** (4 chapters, clear arc):

| New Ch | Title | Content |
|--------|-------|---------|
| **12** | The Bidomain Equations | Keep 12.1–12.6, cut 12.7–12.8 to sidebars. Add worked example: plug in σ_i, σ_e values and show what the coupling term looks like numerically. |
| **13** | From PDEs to Matrices | Keep 13.1, simplify 13.2 (cut or sidebar the weak form), keep 13.3. Replace 13.4 with a brief cross-ref to 12.9. **Add**: face-based stencil explanation. **Add**: full worked example — 5-node cable with numerical K_i, K_e, and the complete decoupled solve. |
| **14** | Solving the Bidomain System | **New core chapter.** Open with "why explicit fails" (currently 14.4). Then: the decoupled approach (boxed algorithm for GS). Then: semi-implicit, Jacobi, IMEX-SBDF2, RKC as variations. Comparison table. Worked example for each. Sidebar: the monolithic alternative (briefly). |
| **15** | Linear Solvers and Implementation | Merge current Ch 15 (SPD solvers) with useful parts of Ch 16 (null-space handling). Add: three-tier elliptic solver strategy. Add: how BoundarySpec drives solver selection. Rewrite Ch 17 as the final section: map concepts to actual Engine_V1 classes with file paths and pseudocode. |

This reduces 6 chapters to 4, eliminates redundancy, and gives every chapter a clear purpose.

### R2. Add the missing algorithm box

The single highest-impact improvement. A boxed "Algorithm 14.1: Decoupled Gauss-Seidel Bidomain Step" showing:

```
Given: Vm^n, φ_e^n, ionic_states^n

1. IONIC STEP (same as monodomain):
   Vm^* ← Rush-Larsen(Vm^n, ionic_states^n, dt)

2. PARABOLIC STEP — solve for Vm^{n+1}:
   A_para · Vm^{n+1} = b_para
   A_para = (1/dt)·I − θ·L_i          [N×N, SPD]
   b_para = (1/dt)·Vm^* + (1−θ)·L_i·Vm^* + L_i·φ_e^n

3. ELLIPTIC STEP — solve for φ_e^{n+1}:
   A_ellip · φ_e^{n+1} = b_ellip
   A_ellip = −(L_i + L_e)              [N×N, SPD after pinning]
   b_ellip = L_i · Vm^{n+1}
```

Then show the semi-implicit, Jacobi, and RKC variants as modifications.

### R3. Restore the Feynman style in Ch 13–15

For every hard concept, add the missing ELI5:

| Concept | Missing Analogy | Suggested |
|---------|-----------------|-----------|
| Saddle-point system | No analogy given | "Imagine a mountain pass between two peaks. Walking uphill in one direction but downhill in the perpendicular direction — that's an indefinite system." |
| Schur complement | "Effective equation after eliminating Vm" (too brief) | "If you solve the first equation for Vm as a function of φ_e, and substitute back, you get a single equation for φ_e alone. That single equation is the Schur complement." |
| Decoupled vs monolithic | Not discussed | "Solving both equations simultaneously is like untangling two knotted ropes at once. Decoupling is like cutting one rope, straightening it, then using it to straighten the other." |
| Null-space | Altitude analogy exists (good) | Keep as-is |
| AMG | Not explained at all | "Imagine a blurry photograph and a sharp one. Smooth out the sharp errors first (cheap), then fix the fine details. Multigrid repeats this at multiple resolutions." |
| Three-tier solver selection | Not discussed | "If the problem is simple (isotropic, Cartesian), use the fastest tool (DCT). If it's mildly complex, use the fast tool as a warm-up for an iterative solver (PCG+spectral). If it's fully complex, fall back to brute force (PCG)." |

### R4. Add worked examples to every chapter

| Chapter | Missing Example | What to Show |
|---------|-----------------|--------------|
| Ch 12 | Numerical PDE evaluation | Plug σ_i=0.17, σ_e=0.62, χ=1400, Cm=1.0 into eq (12.4). Show the coupling term ∇·(D_i·∇φ_e) is ~25% of the total diffusive flux. |
| Ch 13 | Full bidomain solve | 5-node cable: build K_i, K_e with numbers. Show the 10×10 block system. Solve via decoupled GS. Show Vm and φ_e values. |
| Ch 14 | Time step comparison | Same 5-node cable: advance one step with CN decoupled, then with semi-implicit. Compare Vm values. Show the splitting error. |
| Ch 15 | Spectral solve | 4×4 grid: DCT of a RHS vector, spectral division, IDCT. Show actual numbers at each step. |

### R5. Convention translation box

Add at the start of Ch 12, before any equations:

> **Two notations for the same physics.** The bidomain literature uses two equivalent formulations:
> - **Conductivity form (σ)**: Uses σ_i, σ_e with explicit χ·Cm. Standard in journal papers.
> - **Diffusivity form (D)**: Uses D_i = σ_i/(χ·Cm), absorbing χ·Cm. Used in Engine V1 code.
>
> This chapter derives equations in σ-form (matching the literature). The engine connection boxes show the D-form equivalent. To convert: replace χ·Cm·∂Vm/∂t with ∂Vm/∂t, replace σ with D·χ·Cm, and set M = I (lumped mass).

---

## Summary Checklist

| # | Type | Severity | Issue | Status |
|---|------|----------|-------|--------|
| — | **READABILITY** | — | — | — |
| F1 | Flow | CRITICAL | Part III has no narrative arc — catalogue of methods, not a story | Open |
| F2 | Flow | CRITICAL | Complete bidomain algorithm never shown (no boxed procedure) | Open |
| F3 | Style | CRITICAL | Feynman style abandoned after Ch 12 — no ELI5 in Ch 13–17 | Open |
| F4 | Style | MAJOR | Difficulty spike at Ch 13.2 (weak form) with no bridge | Open |
| F5 | Style | MAJOR | Ch 14 is a catalogue, not a narrative — 5 methods, no arc | Open |
| F6 | Style | MAJOR | Ch 16 is a jargon wall — saddle-point, Schur, AMG, GMRES unexplained | Open |
| F7 | Style | MAJOR | Zero worked examples in Ch 14, 16, 17 | Open |
| F8 | Style | MAJOR | Zero diagrams in Ch 14, 16 | Open |
| F9 | Style | MODERATE | 12.9 and 13.4 repeat the same BC content near-verbatim | Open |
| F10 | Style | MODERATE | 12.6 repeats "no time derivative" explanation 4 times | Open |
| F11 | Style | MODERATE | 12.7–12.8 break momentum (monodomain reduction + fiber tensors) | Open |
| F12 | Style | MODERATE | Ch 15 is redundant with Part II solver chapters | Open |
| — | **ACCURACY** | — | — | — |
| C1 | Code | CRITICAL | Monolithic (textbook) vs decoupled (code) solver mismatch | Open |
| C2 | Code | CRITICAL | σ-notation (textbook) vs D-notation (code) formulation mismatch | Open |
| C3 | Code | CRITICAL | Ch 17 describes nonexistent architecture (FGMRES+AMG+SDIRK2) | Open |
| M1 | Code | MAJOR | Face-based stencil (critical for bidomain) not explained | Open |
| M2 | Code | MAJOR | Decoupled GS algorithm — the actual engine algorithm — not documented | Open |
| M3 | Code | MAJOR | "Saddle-point" / "indefinite" not explained for target audience | Open |
| M4 | Code | MAJOR | IMEX equations wrong scaling + wrong structure vs code | Open |
| M5 | Code | MAJOR | Three-tier elliptic solver strategy undocumented | Open |
| M6 | Code | MAJOR | No worked example for full bidomain block system + decoupled solve | Open |
| M7 | Code | MAJOR | BoundarySpec → solver selection logic undocumented | Open |
| Mo1 | Ref | MODERATE | 5 cross-reference errors | Open |
| Mo2 | Code | MODERATE | SDIRK2 taught but not used in code | Open |
| Mo3 | Code | MODERATE | Engine boxes reference wrong codebase (V5.4 not Engine_V1) | Open |
| Mo4 | Math | MODERATE | φ_i substitution step skipped in derivation | Open |
| Mo5 | Math | MODERATE | Conservation law addition not shown (4 intermediate lines) | Open |
| Mo6 | Code | MODERATE | Null-space strategies: equal weight to unused ones | Open |
| Mo7 | Code | MODERATE | "70–85% runtime" claim unjustified | Open |
| m1 | Name | MINOR | K_i (textbook) vs L_i (code) naming inconsistency | Open |
| m2 | Num | MINOR | Eq (15.1), (15.2) appear in Ch 14, not Ch 15 | Open |
| m3 | Fmt | MINOR | Block definitions table mixes time-stepped and static forms | Open |
| m4 | Ref | MINOR | "Chapter 10" should be "Chapter 11" for implicit solvers | Open |
| m5 | Math | MINOR | D_eff harmonic mean written as scalar but D_i, D_e are tensors | Open |

**Total**: 3 critical flow + 3 critical accuracy + 9 major + 10 moderate + 5 minor = **30 issues**
**Part III average style score**: **1.9 / 5.0** (vs ~4.0 for Part II)
