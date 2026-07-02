# Monodomain Chapters Audit (Ch 7–11)

**Perspective**: Student with basic PDE intuition, intro linear algebra, no numerical analysis background.
**Scope**: Chapters 7–11 of `bidomain_textbook.html` (Part II: Tissue-Level Monodomain Modeling)
**Evaluated against**: `STYLE_GUIDE.md` (Feynman style, 5-layer complexity, 3B1B visual intuition)
**Date**: 2026-03-08

---

## Executive Summary

Part II is **strong** — dramatically better than Part III. The Feynman-style writing is consistent, analogies are plentiful and well-chosen (light bulbs, stir-fry, GPS, swimming pools), worked examples use real numbers, and the layered-complexity principle is followed throughout. A student reading Ch 7–11 sequentially will genuinely understand how to go from the monodomain PDE to a working simulation.

That said, Part II is not without issues. The main problems are:

1. **Ch 8 is too long.** At ~1100 lines, it covers FDM, FEM, FVM, and BCs — effectively four mini-chapters. By the time the student reaches FVM (§8.5), cognitive overload has set in. The material is good, but the sheer volume dilutes the excellent worked examples.

2. **Cross-chapter self-references are occasionally wrong.** Chapter 10 refers to "Chapter 10's implicit methods" when it means Chapter 11. Figure numbering in Ch 10 uses "Figure 9.x" instead of "Figure 10.x." The CFL equation is referenced as "equation (9.2)" instead of "(10.2)." These are copy-paste artifacts from a restructuring.

3. **Ch 11 has no worked examples for BDF2.** BDF1 gets an example via analogy to Ch 11.2's matrix form. CN gets a proper 5-node worked example. BDF2 gets the math and a comparison table — but no concrete computation showing the two-history-level RHS in action.

4. **The linear solver section (§11.6) is too thin.** PCG, Chebyshev, and FFT/DCT are each covered in a single paragraph. For the student who just learned they need to solve Ax=b at every timestep (60–80% of cost!), this is a major gap. Appendix A.10 fills some of this, but the student doesn't know that yet.

**Overall Part II score: 3.8/5.0** — a solid B+. Compared to Part III's 1.9/5.0, this is a different textbook.

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

### Chapter 7 — The Monodomain Equation — Score: 4.6/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | "Bundle of straws" for fiber anisotropy. "Light bulb" for reaction-diffusion coupling. "Swimming pool" for boundary conditions. Every hard concept has a physical analogy. |
| L2 Feynman | **5** | The progression from "current leaks between cells" → Ohm's law → conductivity tensor → full PDE is flawless. The reader understands *why* each term exists before seeing it in math. |
| L3 3B1B | **4** | The swimming-pool BC section has good intuitive framing. Missing: a diagram showing current flow through tissue with fiber orientation (the "bundle of straws" deserves a figure). |
| L4 Worked Example | **4** | The BC section walks through Neumann/Dirichlet/Robin with physical interpretation. Missing: a small numerical example showing the actual PDE evaluated at one grid point with real cardiac parameters. |
| L5 Implementation | **5** | Clear mapping to Engine V5.4's `SpatialDiscretization` ABC. The operator-splitting preview foreshadows Ch 9 cleanly. |

**Flow assessment:**
- **Excellent narrative arc.** 7.1 (physical picture) → 7.2 (the PDE) → 7.3 (conductivity and anisotropy) → 7.4 (BCs). The student never feels lost.
- **The BC section (7.4) is well-placed.** It introduces BC *concepts* before the student encounters BC *implementation* in Ch 8. This is good pedagogical design — concepts first, mechanics later.
- **Smooth on-ramp.** Coming from Part I (ionic models), the student has V_m and I_ion. Chapter 7 says "now let's connect cells together" — the most natural possible transition.

**Minor issues:**
- §7.3 introduces the full 3D conductivity tensor with rotation matrices. For a 2D-first audience, this could be gently flagged as "you can skip this for 2D simulations."
- The phrase "equation (7.5)" is referenced in later chapters but appears to be the semi-discrete form from Ch 8, not Ch 7. Cross-reference inconsistency.

---

### Chapter 8 — Spatial Discretization — Score: 4.0/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **4** | Good opening ("how do you turn a continuous PDE into a system a computer can solve?"). The rubber-band analogy for FEM stiffness is implicit but present. Missing: a direct ELI5 for "weak form" — the text jumps to test functions without a bridge analogy. |
| L2 Feynman | **4** | FDM explained as "each node looks at its neighbors." FVM explained as "conservation over boxes." FEM is the weakest conceptually — the text goes from "hat functions" to integration-by-parts quickly. |
| L3 3B1B | **5** | SVG stencil diagrams for FDM (5-point, 9-point). Hat function diagram for FEM. FVM control volume diagram. The comparison table (§8.6) is a model of how to summarize trade-offs visually. |
| L4 Worked Example | **5** | 5-node 1D cable for FDM with actual matrix assembly. 4-element FEM mesh with explicit stiffness computation. These are among the best worked examples in the entire book. |
| L5 Implementation | **4** | Engine boxes map to `SpatialDiscretization` subclasses. FDM and FEM assembly steps correspond to code. Missing: explicit mention of how the code stores sparse matrices (COO → CSR). |

**Flow assessment:**
- **Strong internal structure.** FDM (simplest) → FEM (most general) → FVM (conservation-based) → comparison table → BCs. The ordering respects the "simple before complex" principle.
- **The comparison table (§8.6) is a highlight.** After 800+ lines of three methods, the student needs a summary. The table delivers exactly that, with clear "use when..." guidance.
- **BC sections (§8.7) are thorough but long.** Four subsections (8.7.1–8.7.4) covering Neumann, Dirichlet, Robin, and implementation details. At this point in the chapter, the student has been reading for a long time. Consider: the BC *concept* was already introduced in §7.4. The BC *implementation* here could be more concise.

**Major issue — Chapter length:**
- At ~1100 lines, this is the longest chapter in Part II. A student reading front-to-back will hit cognitive fatigue around FVM (§8.5). The material quality doesn't drop — the reader's capacity does.
- **Recommendation**: Consider splitting into Ch 8A (FDM + FEM, 600 lines) and Ch 8B (FVM + BCs + Comparison, 500 lines). Or, if the chapter must stay unified, add a "reading guide" box at the start: "If you're implementing FDM, read §8.1–8.3 and §8.7. For FEM, add §8.4. For FVM, §8.5."

**Minor issues:**
- The FEM section (§8.4) is good but could use a clearer ELI5 for "weak form." The current opening ("multiply the PDE by a test function and integrate") is correct but not intuitive. Something like: "Instead of demanding the PDE hold at every point, we ask: does it hold *on average* over each element? This weaker requirement is easier to satisfy and leads naturally to a matrix equation."
- The phrase "mass matrix" appears without an analogy. "Mass" in the FEM context means "how much each node contributes to the integral" — a one-sentence explanation would help.

---

### Chapter 9 — Operator Splitting: Divide and Conquer — Score: 4.4/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | The stir-fry analogy is perfect: "Cook the rice and the vegetables separately, then combine." The student immediately grasps why splitting is useful. |
| L2 Feynman | **5** | Godunov vs. Strang explained as "cook in sequence" vs. "cook, re-cook, combine." The symmetry argument for Strang's second-order accuracy is crisp. |
| L3 3B1B | **3** | No diagrams. The stir-fry analogy is verbal only — a simple timeline diagram showing the substep sequence (R → D vs. R/2 → D → R/2) would reinforce the concept. Missing: visual showing how splitting error manifests (e.g., wavefront position shift). |
| L4 Worked Example | **4** | Rush-Larsen worked example with actual exponential-integration computation. The comparison between FE and RL for a fast gate is concrete. Missing: a splitting-error example (same problem with Godunov vs. Strang, showing the 1st-order vs. 2nd-order error). |
| L5 Implementation | **5** | Engine boxes map directly to `SplittingStrategy`, `GodunovSplitting`, `StrangSplitting`, `RushLarsenSolver`. The code flow (reaction step → diffusion step) is explicit. |

**Flow assessment:**
- **Tightly structured.** 9.1 (why split?) → 9.2 (Godunov) → 9.3 (Strang) → 9.4 (Rush-Larsen for the reaction substep). Each section builds on the last.
- **Rush-Larsen (§9.4) is well-motivated.** The student understands from Ch 5–6 that gates have fast and slow time constants. RL's "use the exponential solution for the fast part" is a natural payoff.
- **Good length.** At ~143 lines, this chapter respects the reader's time. It says what needs to be said and stops.

**Minor issues:**
- §9.1 opens with "After the reaction substep updates the ionic variables, the diffusion substep must advance V_m." This is actually the intro for explicit solvers (Ch 10), not for operator splitting itself. The chapter intro should be about *why* we split, not about what the substeps do.
- The transition from Ch 9 to Ch 10 is slightly abrupt. A brief closing paragraph ("Now that we know *how* to split, the next two chapters tackle the diffusion substep itself — first with explicit methods, then implicit.") would smooth the handoff.

---

### Chapter 10 — Explicit Diffusion Solvers — Score: 4.4/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | "Forward Euler is the computational equivalent of 'heading north at 60 mph.'" The GPS/speedometer analogy for RK methods is extended brilliantly across FE → RK2 → RK4. |
| L2 Feynman | **5** | "RK2 is a rough draft followed by a revision." "RK4 checks the speedometer four times." Simpson's rule connection for the 1/6 weights is genuine insight. The "predictor-corrector" naming is explained, not just stated. |
| L3 3B1B | **5** | Three SVG diagrams: RK2 slope visualization (Fig 9.1), RK4 four-stage diagram (Fig 9.2), stability region plot (Fig 9.3). Each diagram directly illustrates the corresponding equation. The stability region plot shows all three methods overlaid — excellent for comparison. |
| L4 Worked Example | **5** | The dV/dt = -2V example is carried across all three methods with exact error comparison (FE: 0.01873, RK2: 0.00127, RK4: 0.000001). The CFL worked example uses real cardiac parameters. Both are exemplary. |
| L5 Implementation | **4** | Engine box explains the `DiffusionSolver` ABC and `apply_diffusion(V)` operator pattern. The "bare k" convention is documented. Missing: explicit pseudocode for the RK4 step (the equations are there, but a 6-line pseudocode block would cement the implementation). |

**Flow assessment:**
- **Model narrative arc.** FE (simplest, limited) → "why go beyond?" (motivation) → RK2 (one correction) → RK4 (four corrections) → stability analysis (when does each fail?) → comparison table (choose your method) → practical guidance (when explicit methods shine). This is textbook-perfect progression.
- **The comparison table (§10.6) mirrors §8.6's approach** — consistent pedagogical pattern across chapters. The student recognizes the format and knows how to read it.
- **§10.7 (When Explicit Methods Shine) is a great closer.** It doesn't just dump the reader into implicit methods. It says "here's when explicit is the right call" — respecting the method they just learned before saying "but sometimes you need something else."

**Issues:**
- **Figure numbering is wrong.** Figures are labeled "Figure 9.1", "Figure 9.2", "Figure 9.3" — they should be "Figure 10.1", "10.2", "10.3". This is a copy-paste artifact from when Ch 10 was Ch 9.
- **Cross-reference error.** The CFL warning box (after eq 10.2) says "This is exactly why Chapter 10's implicit methods exist" — it should say "Chapter 11." The §10.6 comparison table also says "Chapter 10" for implicit methods.
- **Cross-reference error.** §11.1 references "Chapter 9" for explicit methods and "equation (9.2)" for CFL — should be "Chapter 10" and "equation (10.2)."
- **"Bare k" convention** is explained well but only after the standard form (10.3/10.4). A student encountering (10.3b) may be confused about why the Δt placement changed. Consider swapping: present the bare-k form first (since that's what the code uses), then note the standard form as an equivalent.

---

### Chapter 11 — Implicit Diffusion Solvers — Score: 3.8/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | "Steering a boat in rapids" for CFL instability. "Peeking ahead" for implicit methods. "CN averages start and destination." "BDF2 is CN's tougher cousin." Every method gets an analogy. |
| L2 Feynman | **4** | The "why implicit?" argument is well-made with the h² scaling calculation. The CN "ringing artifact" explanation is honest and practical. BDF2's "two history levels" is clear. Missing: a Feynman-style explanation of *why* implicit methods are unconditionally stable (the "peeking ahead" analogy is good, but the mathematical mechanism — evaluating at the future state forces the stability function to be bounded — is never connected back to the analogy). |
| L3 3B1B | **2** | Zero diagrams in the entire chapter. Missing: stability region diagram comparing implicit methods (a mirror of Fig 10.3 for explicit). Missing: a visual showing the A_lhs and B_rhs matrix structures (block pattern, sparsity). Missing: diagram of CN oscillation ("ringing") at a sharp wavefront. This is the weakest visual chapter in Part II. |
| L4 Worked Example | **3** | CN gets a decent worked example (5-node cable, FDM lumped mass). BDF1 gets a simplification note (M=I case). BDF2 gets nothing — the two-history RHS is never computed with real numbers. For a method that requires bootstrapping (first step with BDF1), a worked example showing both the bootstrap step and the first BDF2 step would be very valuable. |
| L5 Implementation | **4** | Engine box clearly maps `CrankNicolsonSolver`, `BDF1Solver`, `BDF2Solver` to the ABC. The `get_diffusion_operators(dt, scheme)` interface is described. The BDF2 bootstrap logic (auto-fallback to BDF1 for first step) is mentioned. Missing: how the linear solver selection works (PCG vs Chebyshev vs FFT). |

**Flow assessment:**
- **Good opening motivation.** §11.1 connects directly to Ch 10's CFL limitation with concrete numbers (h=0.1mm → Δt ≤ 0.004ms). The student understands *why* they need implicit methods before seeing the first equation.
- **BDF1 → CN → BDF2 progression is natural.** Simplest implicit (BDF1) → better accuracy (CN) → better stability (BDF2). Each method is presented as fixing a limitation of the previous one.
- **The comparison table (§11.5) continues the pattern** from §8.6 and §10.6. Excellent consistency.
- **§11.6 (Linear Solvers) is the weakest section in Part II.** Three paragraphs for three solvers. PCG gets one paragraph. Chebyshev gets one paragraph. FFT/DCT gets one paragraph. For the student who was just told "60–80% of wall-clock time is spent here," this is frustratingly thin. They know *that* they need a linear solver, but not *how* any of them work. Appendix A.10 covers PCG in depth, but the student doesn't know to go there — and even if they did, Chebyshev and FFT are still unexplained.

**Minor issues:**
- The opening paragraph of §11.1 says "Explicit methods (Chapter 9)" — should be Chapter 10.
- BDF2's insight box claims "BDF2 is A-stable but not L-stable" then says the stability function satisfies |R(∞)| = 1/3. This is actually the definition of L-stability (|R(∞)| → 0 as z → -∞). The text contradicts itself. [Correction: BDF2 is indeed A-stable and the |R(∞)| = 1/3 < 1 means it strongly damps but technically the BDF2 stability function formula given appears incorrect — this needs verification.]
- The "precomputed once" insight box for CN (after eq 11.4) is important practical info but is styled as a warning (yellow). It's not a warning — it's an optimization note. Should be `insight-engine` (blue).

---

## Part B: Cross-Cutting Issues

### B1. The Chapter 8 Length Problem

Chapter 8 is 1100+ lines — more than Ch 7 + Ch 9 + Ch 10 combined. It covers three spatial discretization methods (FDM, FEM, FVM), a comparison, and boundary conditions. Each section is individually well-written, but the aggregate length is a readability problem.

**Impact**: A student reading front-to-back will experience fatigue. The excellent FVM section (§8.5) and BC sections (§8.7) are underserved because the reader is already exhausted from FDM and FEM.

**Recommendation**: Add a reading guide at the chapter opening:
> "This chapter covers three spatial discretization methods. You only need one to get started. **If you're implementing a regular-grid FDM code** (like Engine V5.4's default), read §8.1–8.3 and §8.7. **For unstructured meshes**, add §8.4 (FEM). **For conservation-preserving codes**, §8.5 (FVM). Section §8.6 compares all three."

### B2. Figure Numbering Errors (Ch 10)

All three figures in Chapter 10 are labeled "Figure 9.x" instead of "Figure 10.x":
- "Figure 9.1" → should be "Figure 10.1" (RK2 diagram)
- "Figure 9.2" → should be "Figure 10.2" (RK4 diagram)
- "Figure 9.3" → should be "Figure 10.3" (Stability regions)

This is a copy-paste artifact from a chapter renumbering. Easy fix.

### B3. Cross-Reference Errors

| Location | Says | Should Say |
|----------|------|------------|
| §10.1 (CFL warning box) | "Chapter 10's implicit methods" | "Chapter 11" |
| §10.6 | "Chapter 10 does not cover" (referring to adaptive stepping) | Ambiguous self-reference — clarify |
| §11.1 opening | "Explicit methods (Chapter 9)" | "Chapter 10" |
| §11.1 | "equation (9.2)" for CFL | "equation (10.2)" |
| Ch 7 INDEX entry | "equation (7.5)" in cross-refs | Verify: is (7.5) actually in Ch 7 or Ch 8? |

### B4. The Implicit Chapter Needs Visuals

Chapter 11 is the only chapter in Part II with zero SVG diagrams. Every other chapter has at least one (Ch 7: none explicitly, but BC conceptual figures via text; Ch 8: multiple stencil/hat function SVGs; Ch 9: none but short; Ch 10: three excellent SVGs).

For a chapter about methods that are *harder* than the explicit ones — and that dominate real-world simulation cost — the lack of visuals is a missed opportunity.

**Recommended additions:**
1. **Stability region diagram** for BDF1, CN, BDF2 (mirror of Ch 10's Fig 10.3 for explicit methods). Show the entire left half-plane being stable.
2. **Matrix structure diagram** showing A_lhs and B_rhs sparsity patterns for a small 5×5 example. The student has seen matrix diagrams in Ch 8 — continue that visual language.
3. **CN ringing diagram** showing a 1D wavefront with and without the 2Δt oscillation artifact. This would make the A-stable vs. L-stable distinction concrete.
4. **BDF2 bootstrap timeline** showing: step 0 (initial condition) → step 1 (BDF1 bootstrap) → step 2+ (BDF2 with two history vectors). Simple horizontal timeline with arrows.

### B5. Linear Solver Gap (§11.6)

§11.6 is the thinnest section in Part II relative to its importance. The student learns that 60–80% of simulation time is spent solving Ax=b, then gets three paragraphs.

**What's missing:**
- How PCG works (even at ELI5 level — "CG finds the minimum of a bowl-shaped function, each step slides downhill")
- Why Jacobi preconditioning helps (just dividing by the diagonal — 1 sentence)
- How Chebyshev avoids inner products (the zero-synchronization property needs more than 1 sentence for a GPU novice)
- How DCT diagonalizes the Laplacian (Appendix A.9 covers this, but the student needs a forward pointer with a 2-sentence preview)

**Recommendation**: Either expand §11.6 to ~2 pages with ELI5 + key equations for each solver, or add explicit forward pointers: "For a visual explanation of how CG works, see Appendix A.10. For the DCT trick, see Appendix A.9."

### B6. Transition Between Parts II and III

The transition from Ch 11 to Ch 12 (Part II → Part III) is one of the strongest in the book. Chapter 12.1–12.3 explicitly builds on the monodomain foundation:
- "In the monodomain model, we make a simplifying assumption..."
- "But real cardiac tissue violates this assumption..."
- The monodomain vs. bidomain comparison diagram (Fig 12.1)

This is exactly how a multi-part textbook should work. The student finishes Part II feeling competent, and Part III opens by saying "you know enough to understand why we need more."

---

## Part C: Narrative Arc Assessment

### The Story of Part II

Part II tells a coherent story in five acts:

1. **Ch 7** — *The Setup*: "Here is the monodomain PDE. Here's what each term means physically."
2. **Ch 8** — *The Toolbox*: "Here are three ways to turn that PDE into a matrix equation."
3. **Ch 9** — *The Strategy*: "Don't solve everything at once. Split the problem."
4. **Ch 10** — *The Simple Path*: "Compute the diffusion step explicitly. Fast, simple, limited."
5. **Ch 11** — *The Robust Path*: "When explicit fails, solve implicitly. Harder, but unconditionally stable."

This arc works. Each chapter answers a question raised by the previous one:
- Ch 7: "How do cells communicate?" → The PDE
- Ch 8: "How does a computer represent this?" → Discretization
- Ch 9: "The PDE has two parts (reaction + diffusion). Do we solve them together?" → Splitting
- Ch 10: "How do we solve the diffusion part?" → Explicit methods
- Ch 11: "What if explicit isn't enough?" → Implicit methods

The student never asks "why am I reading this?" — the motivation is always clear.

### Comparison to Part III's Narrative Arc

| Aspect | Part II (Ch 7–11) | Part III (Ch 12–17) |
|--------|-------------------|---------------------|
| Story | Clear 5-act structure | Ch 12 is strong, Ch 13–17 are a catalogue |
| Analogies | Every chapter, multiple per chapter | Ch 12 only; Ch 13–17 nearly zero |
| Worked examples | FDM assembly, RK2/RK4 with numbers, CN 5-node | Zero in Ch 13–16 |
| Diagrams | 5+ SVGs across Part II | 2 SVGs total in Part III |
| Difficulty ramp | Gentle, consistent gradient | Cliff at Ch 13.2 (weak form) |
| Implementation mapping | Every chapter has engine boxes | Ch 15–17 describe architecture that doesn't exist |
| Comparison tables | §8.6, §10.6, §11.5 — consistent format | §11.5 equivalent missing in Part III |
| Chapter length | Well-balanced (except Ch 8) | Unbalanced (Ch 12 = 381 lines, Ch 16 = 155 lines) |

---

## Summary Checklist

### Critical Issues (must fix)
| # | Issue | Location | Fix |
|---|-------|----------|-----|
| C1 | Figure numbering: "Figure 9.x" → "Figure 10.x" | Ch 10 (3 figures) | Rename |
| C2 | Cross-reference: "Chapter 10" → "Chapter 11" for implicit | §10.1, §10.6 | Fix refs |
| C3 | Cross-reference: "Chapter 9" → "Chapter 10" for explicit | §11.1 | Fix refs |

### Major Issues (should fix)
| # | Issue | Location | Fix |
|---|-------|----------|-----|
| M1 | Ch 11 has zero diagrams | Entire Ch 11 | Add stability region, matrix structure, CN ringing, BDF2 timeline diagrams |
| M2 | §11.6 (linear solvers) is too thin for its importance | §11.6 | Expand or add forward pointers to Appendix A.9–A.10 |
| M3 | BDF2 has no worked example | §11.4 | Add: BDF1 bootstrap step → first BDF2 step with 5-node cable |
| M4 | Ch 8 is too long (~1100 lines) | Entire Ch 8 | Add reading guide box at chapter start |
| M5 | FEM "weak form" has no ELI5 | §8.4 | Add 2-sentence analogy before test functions appear |

### Moderate Issues (nice to fix)
| # | Issue | Location | Fix |
|---|-------|----------|-----|
| m1 | "Bare k" convention explained after standard form | §10.3, §10.4 | Consider presenting bare-k first since that's what code uses |
| m2 | BDF2 insight box may have incorrect stability claim | §11.4 insight | Verify L-stability claim and |R(∞)| formula |
| m3 | CN "precomputed once" box styled as warning | After eq 11.4 | Change to insight-engine (blue) |
| m4 | Ch 9 opens with diffusion substep text, not splitting motivation | §9.1 intro | Rewrite intro to focus on "why split?" |
| m5 | No diagram for "bundle of straws" anisotropy analogy | §7.3 | Add simple fiber-orientation SVG |
| m6 | "Mass matrix" in FEM section has no analogy | §8.4 | Add 1-sentence explanation ("how much each node weighs in the integral") |
| m7 | §11.1 references "equation (9.2)" for CFL | §11.1 | Should be "equation (10.2)" |

### Minor Issues (cosmetic)
| # | Issue | Location | Fix |
|---|-------|----------|-----|
| s1 | §7.3 could flag 3D tensor content as skippable for 2D | §7.3 | Add "For 2D simulations, skip to §7.4" note |
| s2 | Ch 9 → Ch 10 transition is abrupt | End of Ch 9 | Add 2-sentence bridge paragraph |
| s3 | §10.7 engine box says "Chapter 10" for implicit (self-ref) | §10.7 | Fix to "Chapter 11" |

---

## Overall Scores

| Chapter | L1 | L2 | L3 | L4 | L5 | Average | Grade |
|---------|----|----|----|----|----|---------| ------|
| 7 — Monodomain Equation | 5 | 5 | 4 | 4 | 5 | **4.6** | A |
| 8 — Spatial Discretization | 4 | 4 | 5 | 5 | 4 | **4.4** | A- |
| 9 — Operator Splitting | 5 | 5 | 3 | 4 | 5 | **4.4** | A- |
| 10 — Explicit Solvers | 5 | 5 | 5 | 5 | 4 | **4.8** | A+ |
| 11 — Implicit Solvers | 5 | 4 | 2 | 3 | 4 | **3.6** | B |
| **Part II Average** | **4.8** | **4.6** | **3.8** | **4.2** | **4.4** | **4.0** | **A-** |

For comparison: Part III average = **1.9/5.0** (D+)

---

## Recommendations

### R1: Fix the Three Cross-Reference Errors (30 min)
Simple find-and-replace. "Chapter 9" → "Chapter 10" for explicit, "Chapter 10" → "Chapter 11" for implicit, "Figure 9.x" → "Figure 10.x", "equation (9.2)" → "equation (10.2)".

### R2: Add Diagrams to Chapter 11 (2–3 hours)
Four SVGs:
1. Implicit stability regions (BDF1, CN, BDF2) — mirror of Ch 10's explicit stability plot
2. A_lhs / B_rhs sparsity patterns for a 5-node example
3. CN ringing at a steep wavefront (1D Vm plot showing 2Δt oscillations)
4. BDF2 bootstrap timeline (horizontal, 3 timesteps)

### R3: Add BDF2 Worked Example (1 hour)
5-node 1D cable, lumped mass (M=I). Show:
- Step 0: V^0 = [initial]
- Step 1 (BDF1 bootstrap): A_lhs · V^1 = V^0, solve
- Step 2 (BDF2): A_lhs · V^2 = 4·V^1 - V^0, solve
The student sees the two-history mechanism in action.

### R4: Expand §11.6 or Add Forward Pointers (1 hour)
At minimum, add: "For a visual, step-by-step explanation of how the conjugate gradient method works, see Appendix A.10. For how the DCT diagonalizes the Laplacian on regular grids, see Appendix A.9."
Better: expand each solver paragraph to include a 2-sentence ELI5 + the key equation.

### R5: Add Reading Guide to Chapter 8 (15 min)
A boxed paragraph at the start of §8.1 directing FDM-only, FEM-only, and FVM-only readers to the relevant sections. This doesn't require restructuring — just a navigation aid.
