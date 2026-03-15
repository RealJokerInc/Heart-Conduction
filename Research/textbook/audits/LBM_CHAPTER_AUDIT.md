# LBM Chapters Audit (Ch 18–20)

**Perspective**: Student with monodomain knowledge (Part II), intro linear algebra, no kinetic theory or statistical mechanics background.
**Scope**: Chapters 18–20 of `bidomain_textbook.html` (Part IV: Lattice-Boltzmann Methods)
**Evaluated against**: `STYLE_GUIDE.md` (Feynman style, 5-layer complexity, 3B1B visual intuition)
**Date**: 2026-03-08

---

## Executive Summary

Part IV is **ambitious and mostly successful** — it attempts something genuinely hard (teaching LBM from kinetic theory to a student who has only seen FDM/FEM) and largely pulls it off. Chapter 18 is a tour de force of layered exposition, building from phase space through the Boltzmann equation to the lattice-Boltzmann algorithm in a coherent narrative. Chapter 19 cleanly maps the abstract machinery onto cardiac electrophysiology. Chapter 20 is a well-structured but shallow survey of bidomain LBM strategies.

The main problems are:

1. **Chapter 18 is massively long (~1300 lines) and front-loaded with theory.** The student doesn't see a single line of algorithmic pseudocode or a concrete "here's what the computer does" until §18.3 (line ~850+). The first two sections (§18.1–18.2) are pure kinetic theory — beautifully written, but the student who came here to learn "how does LBM solve the diffusion equation?" has to wade through the Maxwell-Boltzmann distribution, the full collision integral, and Chapman-Enskog analysis before seeing a single lattice. This is the opposite of the "ELI5 first" principle.

2. **The equation numbering has significant gaps and cross-reference errors.** Equations jump from (18.10) to (18.11), skip (18.15)–(18.16) in the running text, then reuse numbers inconsistently. Several cross-references point to "equation 17.x" instead of "18.x" or "19.x" — artifacts of chapter renumbering. Chapter 19 references "Chapter 17" and "equation 17.26" when it means Ch 18 and eq (18.26).

3. **Chapter 20 has no worked examples and no code-level implementation detail.** The three strategies (pseudo-time, hybrid, dual-lattice) are described conceptually but never computed with numbers. The student finishes Ch 20 understanding *what* the strategies are but having no idea how to implement any of them. The "worked example outline" in §20.4 says "suppose..." but never actually computes anything.

4. **The moment-space section (§18.5) is excellent but arrives late.** By the time the student reaches the MRT worked example — the pedagogical payoff of the entire chapter — they've been reading for 1000+ lines. The BGK worked example and the moment-space worked example should come earlier to anchor the theory.

**Overall Part IV score: 3.6/5.0** — a solid B+, matching Part II's quality in writing but with structural length/ordering problems. Far above Part III's 1.9.

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

### Chapter 18 — The Lattice-Boltzmann Method: From Kinetic Theory to Computation — Score: 3.8/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **4** | Good analogies scattered throughout: "catalog every ion and record its velocity" for phase space, "5 well-chosen points on a steep hill capture more than 1000 in the flat plains" for quadrature, "all dials locked together" for BGK. Missing: a single opening ELI5 that says "LBM is like a crowd of people at an intersection — some walk north, some south, they shuffle and regroup at each step, and the crowd's density naturally diffuses." The chapter opens with a dense paragraph about FDM vs kinetic theory instead. |
| L2 Feynman | **5** | Exceptional. The "three physical claims" decomposition of the Maxwell-Boltzmann (zeroth/first/second order) is pedagogical brilliance. The "BGK was always operating in moment space — it just didn't need to make the round trip explicit" insight is a genuine eureka moment. The Chapman-Enskog explanation ("the macroscopic PDEs are emergent, not fundamental") is the kind of perspective-shifting statement that defines great teaching. |
| L3 3B1B | **3** | The D2Q5/D2Q9/D3Q7 lattice SVG (Figure 17.1) is excellent — clear, color-coded, with weights annotated. But that's the only diagram in 1300 lines. Missing: a visual of the Gaussian bell curve in velocity space with quadrature nodes overlaid (§18.1/18.4 would benefit enormously). Missing: a streaming diagram showing f_i moving between nodes. Missing: a collision-before-and-after visualization showing distributions relaxing toward equilibrium. |
| L4 Worked Example | **4** | The BGK worked example (§18.5) with actual numbers (f = [0.340, 0.175, ...], τ = 0.053) is excellent. The moment-space round-trip showing density conservation and flux decay is the best worked example in Part IV. The τ-D computation (§18.4) with cardiac parameters is concrete. Missing: a worked example in §18.3 showing the actual collision-then-stream for a small 3-node or 5-node lattice — the student needs to see distributions physically move between nodes. |
| L5 Implementation | **3** | The variable reference table (§18.3) is helpful. The "collide then stream" two-step decomposition is clear. But there's no pseudocode for the LBM loop, no code-level implementation until Chapter 19's engine box. The connection to Engine V5.4 is absent in Ch 18 itself — the engine box appears only in Ch 19. |

**Flow assessment:**

- **The "Quadrature First" structure is intellectually elegant but pedagogically backwards.** The chapter starts with thermodynamic equilibrium (§18.1), then the Boltzmann equation and BGK (§18.2), then the discrete lattice equation (§18.3), then the quadrature derivation of weights (§18.4), then moment space (§18.5). This is the logical derivation order — start from first principles and build up. But it's not the learning order. A student coming from FDM/FEM expects: "Here's what the algorithm does" → "Here's why it works" → "Here's the deeper theory." Instead they get: "Here's 19th-century statistical mechanics" → "Here's the Boltzmann equation you'll never solve directly" → "Now, 800 lines later, here's the algorithm."

- **Recommendation: Add a "30-second preview" box at the chapter opening.** Before §18.1, insert a 10-line insight box showing the complete LBM algorithm in 6 steps (collide, stream, bounce-back, recover — basically equation 19.11 from Ch 19). Label each step with "we'll derive this in §18.x." The student sees the destination before the journey begins. This is standard Feynman-style teaching: "Let me show you the punchline, then I'll explain the setup."

- **§18.1 is too long.** The equilibrium section (~350 lines) covers phase space, velocity space, the Gaussian, zeroth/first/second order moments, the two cases (rest vs flow), and why the equilibrium is "the heart of LBM." The individual pieces are excellent, but the aggregate length means the student hits §18.2 (the Boltzmann equation) already somewhat fatigued. Consider trimming the "Two Cases" subsection (rest vs flow) — for cardiac EP, only the rest case matters, and the flow case can be a brief aside.

- **§18.2 introduces the full collision integral (eq 18.8) then immediately discards it.** The five-line explanation of the collision integral notation ($f'f'_1 - ff_1$, scattering cross-section, solid angle integration) is accurate but serves no pedagogical purpose — the student will never use it. It exists only to motivate BGK. A single sentence ("The full collision integral is a 5-dimensional integral — computationally intractable") followed by the BGK simplification would suffice.

- **§18.3 finally delivers the algorithm — and it's clear.** The "Three Deliberate Choices" structure (discretize velocity, discretize time, lock grid to velocities) is excellent. The collision/streaming decomposition is well-motivated. The variable reference table is a helpful anchor. This section would benefit from appearing earlier in the chapter.

- **§18.4 (Quadrature) is the intellectual heart of the chapter and mostly works.** The "How Can Five Points Replace an Entire Plane?" insight box is a highlight. The weight derivation for D2Q5 (in the insight-math box) is the right level of detail — concrete enough to verify, general enough to generalize. The lattice naming convention section is efficient.

- **§18.5 (Moment Space) is the best section.** The "decode → relax → re-encode" round trip is the clearest explanation of MRT I've seen in any textbook. The table showing each moment's physical role ("why we need it") is excellent pedagogy. The worked example showing the same collision in both distribution space and moment space is the payoff the entire chapter builds toward. The "BGK was always operating in moment space" revelation is perfectly placed.

- **The "What Comes Next" closing paragraph (after §18.5) is garbled.** It contains a sentence fragment that reads like two different versions were merged: "the $f_i$ are our fundamental variables, and the BGK collision operator relaxes all of them at the same rate $1/\tau$. But there is a more revealing way to look at the same information. In Section 18.5 above, we transformed the distributions into **moment space**, — first by unlocking the moment-space relaxation rates to achieve anisotropic diffusion (Multiple Relaxation Time)..." This needs rewriting.

**Minor issues:**
- Equation (18.11) references "equation 17.2" — should be (18.2).
- The BGK equation (18.12) drops the force term without explicitly marking this as a simplification specific to diffusion/cardiac problems. A student encountering other LBM literature will see the force term and wonder why it's missing.
- The figure is labeled "Figure 17.1" — should be "Figure 18.1."
- Several references in §18.4 say "equation 17.x" where they mean "18.x."

---

### Chapter 19 — Lattice-Boltzmann Methods for Monodomain — Score: 4.2/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **4** | "Think of the source S as a faucet (or drain) at each lattice node" for the ionic injection. "A stiff spring returns to rest quickly but doesn't transmit waves far" for the τ-D inversion. Missing: an opening analogy for "how LBM solves the monodomain equation" — the chapter dives straight into the target PDE. |
| L2 Feynman | **5** | The Ω^NR / Ω^R decomposition is a brilliant pedagogical move — separating diffusion (non-reactive collision) from reaction (source injection) makes the operator splitting crystal clear. The "why slower relaxation means faster diffusion" insight box is genuinely illuminating. The BGK→MRT transition is seamless. |
| L3 3B1B | **3** | No new diagrams in the entire chapter. The comparison table (§19.8) is excellent but not visual. Missing: a figure showing the 6-step LBM-EP algorithm as a flowchart. Missing: a diagram showing bounce-back at a boundary (ball bouncing off wall). Missing: a before/after diffusion front showing voltage spreading through the lattice. |
| L4 Worked Example | **5** | The MRT anisotropy worked example (§19.4) with real cardiac parameters (D_l = 0.001, D_t = 0.00025, τ_xx = 0.053, τ_yy = 0.017) is excellent. The stimulus worked example (§19.6 insight box) with actual I_stim = -52 μA/μF showing the tiny voltage change per step is concrete and correct. The τ-D calculation with cardiac values is repeated for reinforcement. |
| L5 Implementation | **5** | The engine box (§19.7) maps directly to `LBMSimulation`, `BGKCollision`, `MRTCollision`, `LBMState`, `D2Q5`, `D3Q7`. The 6-step algorithm box (eq 19.11) is a complete pseudocode-level specification. The bounce-back precomputation pseudocode (§19.5.3) is implementation-ready. The comparison table (§19.8) includes GPU suitability — practical for the reader choosing a method. |

**Flow assessment:**

- **Excellent narrative arc.** 19.1 (target PDE) → 19.2 (splitting) → 19.3 (BGK with source) → 19.4 (MRT for anisotropy) → 19.5 (BCs) → 19.6 (complete algorithm) → 19.7 (implementation) → 19.8 (comparison). Each section answers the next natural question.

- **The Ω^NR / Ω^R notation is a major pedagogical innovation.** By explicitly separating the collision into "diffusion part" and "reaction part," the chapter makes the operator-splitting structure visible at the collision level. This connects Ch 19 back to Ch 9 (operator splitting) in a way that reinforces both chapters.

- **§19.5 (Boundary Conditions) is thorough and well-structured.** Five subsections covering full-way bounce-back, half-way bounce-back, boundary identification, irregular geometries, and comparison with classical methods. The comparison table (§19.5.5) is a highlight. The corner node explanation is clear.

- **§19.8 (LBM vs Classical) comparison table continues the pattern** from §8.6, §10.6, §11.5. By now the student recognizes this format instantly. The "When to use LBM" / "When to use classical" closing paragraph is practical and honest.

- **The chapter intro references "Chapter 17" — should be Chapter 18.** This appears in the opening paragraph and several equation references throughout.

**Issues:**

- **Cross-reference errors are pervasive.** The chapter intro says "Chapter 17 built the LBM machinery." Section 19.2 says "the Godunov splitting framework of Chapter 8" — should be Chapter 9. Section 19.3 references "equation 17.26" (should be 18.26), "equation 18.11" inconsistently, "equation 18.2" (should be 19.2), "Chapter 17" in multiple places. These are all chapter-renumbering artifacts.

- **Equation (19.2) defines S with χ·C_m in the denominator, but the source term discussion references "equation 18.2" as if that's where S was defined.** Equation 18.2 is the Maxwell-Boltzmann distribution. This is a cross-reference error — should reference (19.2) itself.

- **Missing: a streaming + bounce-back diagram.** The bounce-back section (§19.5) describes the concept verbally ("imagine a ball thrown at a wall") but never shows it. A simple diagram showing f_i streaming toward a wall and f_ī bouncing back would make the concept immediate. The opposite-pairs table is helpful but not a visual.

- **§19.2 references "equation 8.1" for Rush-Larsen** — should be equation (9.1).

---

### Chapter 20 — Lattice-Boltzmann Methods for Bidomain — Score: 3.0/5

| Layer | Score | Notes |
|-------|-------|-------|
| L1 ELI5 | **5** | The fish-tank analogy for bidomain (two connected tanks, one evolves, one is constrained) is excellent. The heated metal plate for pseudo-time is intuitive. The "hammer vs pen" for hybrid is memorable. The "two bookkeepers" for dual-lattice works. Every strategy gets a clear analogy. |
| L2 Feynman | **4** | The core mismatch (evolution vs constraint) is well-articulated. The pseudo-time idea ("invent a fictitious time and march until steady state") is clearly motivated. The hybrid rationale ("right tool for the job") is practical. Missing: a deeper explanation of *why* pseudo-time convergence works — what guarantees the steady state of (20.1) equals the solution of the elliptic equation? A 2-sentence argument would suffice ("the steady state sets ∂φ_e/∂τ = 0, which recovers exactly the original elliptic equation"). |
| L3 3B1B | **2** | Zero diagrams. Missing: a timeline diagram showing the dual-lattice coupling (Lattice 1 takes one step → Lattice 2 iterates to convergence → exchange → repeat). Missing: a convergence plot showing pseudo-time residual decreasing over sub-iterations. Missing: architecture comparison diagram (three boxes showing the three strategies side by side with data flow arrows). |
| L4 Worked Example | **1** | Zero computed examples. The "worked example outline" in §20.4 says "suppose... $V_m$ is known ($-85$ mV)... Lattice 1 takes one step... $V_m$ advances to $-84$ mV (approximately)... Lattice 2 then iterates..." — but no actual numbers are computed. No τ_e calculation. No source term assembly. No convergence residual shown. For a chapter that introduces three new solver strategies, the absence of concrete computation is a major gap. |
| L5 Implementation | **3** | The engine box pseudocode (§20.5) for `DualLatticeBidomainLBMSolver` is a good starting point. The 5-phase implementation roadmap is practical. But: the pseudocode is schematic (e.g., `gradient_phi_e` appears with no definition of how gradients are computed on the LBM lattice; `source_phi_e = -gradient(D_i @ gradient(V_m))` is hand-wavy). No actual Engine V5.4 code mapping — the class name given doesn't exist in the codebase. |

**Flow assessment:**

- **Good three-act structure.** 20.1 (the challenge) → 20.2–20.4 (three strategies) → 20.5 (outlook). Each strategy is presented as addressing the same problem in a different way.

- **§20.1 is well-written.** The "evolution vs constraint" framing is the clearest statement of the bidomain LBM challenge I've seen. The fish-tank analogy perfectly captures the asymmetry between Vm (evolves) and φ_e (constrained).

- **§20.2 (Pseudo-time) is the strongest strategy section.** The heated-plate ELI5, the fictitious time derivative, and the convergence criterion are all clearly explained. The τ_e formula (20.2) connects back to Chapter 18's machinery. Missing: a worked convergence example.

- **§20.3 (Hybrid) is too brief.** Five bullet points for the algorithm and one insight box. The data transfer challenge ("LBM lattice → classical solver") is mentioned but not addressed. How does one compute ∇V_m on the LBM lattice? How does the classical solver's mesh relate to the LBM grid? These are the questions a student would ask.

- **§20.4 (Dual-Lattice) is conceptually clear but implementation-light.** The coupling equations (20.3, 20.4) use "[coupling to φ_e]" and "[coupling to V_m + stimulus]" as placeholders — the actual source terms are never written out explicitly. For a textbook, this is too vague.

- **§20.5 (Performance and Outlook) is speculative.** Performance estimates ("5–20× speedup," "3–10× speedup") are unsupported by benchmarks or citations. The 5-phase implementation roadmap is practical but belongs in an engineering document, not a textbook chapter.

- **The closing paragraph claims "the full mathematical and algorithmic foundation for bidomain electrophysiology" is complete.** But the bidomain LBM is not implemented in Engine V5.4 (as noted in INDEX.md: "architectural research only"). The chapter is describing future work as if it were established knowledge.

---

## Part B: Cross-Cutting Issues

### B1. Chapter 18 Length Problem

At ~1300 lines, Chapter 18 is the longest chapter in the entire textbook. It covers:
- §18.1: Thermodynamic foundation (~350 lines)
- §18.2: Boltzmann equation and BGK (~200 lines)
- §18.3: Discrete LBM equation (~250 lines)
- §18.4: Quadrature, weights, lattices (~400 lines)
- §18.5: Moment space and MRT (~300 lines)

The quality is high throughout, but the student's cognitive capacity is finite. By §18.4, even an engaged reader is fatigued.

**Recommendation (structural):** Consider splitting Ch 18 into two chapters:
- **Ch 18A: "The LBM Algorithm"** — §18.3 (discrete equation), §18.4 (lattice geometry, weights, τ-D), and the BGK worked example from §18.5. This is the "how to do it" chapter. (~650 lines)
- **Ch 18B: "Why LBM Works: From Kinetic Theory to Macroscopic PDEs"** — §18.1 (phase space, equilibrium), §18.2 (Boltzmann, BGK, Chapman-Enskog), and the moment-space formalism from §18.5. This is the "why it works" chapter. (~650 lines)

This mirrors Part II's structure: Ch 10 (explicit solvers — how to do it) before the deeper Ch 11 (implicit solvers — the harder version). The student learns the algorithm first, then understands the theory.

**Recommendation (minimal):** If splitting is not desired, add a "30-second preview" box at the chapter opening showing the complete 6-step LBM algorithm from eq (19.11), with forward pointers to where each step is derived. This gives the student a map before the journey.

### B2. Cross-Reference Errors (Systematic)

Chapter renumbering has left dozens of incorrect cross-references. Key patterns:

| Pattern | Count | Fix |
|---------|-------|-----|
| "Chapter 17" → should be "Chapter 18" | ~8 occurrences in Ch 19–20 | Find/replace |
| "equation 17.x" → "equation 18.x" | ~12 occurrences in Ch 19 | Find/replace |
| "Figure 17.1" → "Figure 18.1" | 1 occurrence in Ch 18 | Rename |
| "Chapter 8" → "Chapter 9" (splitting) | 2 occurrences in Ch 19 | Fix |
| "Chapter 9"/"Chapter 10" → "Chapter 10"/"Chapter 11" (explicit/implicit) | 3 occurrences in Ch 19.8 table | Fix |
| "equation 8.1" → "equation 9.1" (Rush-Larsen) | 1 occurrence in Ch 19.2 | Fix |
| "equation 18.2" → "equation 19.2" (source term S) | 1 occurrence in Ch 19.3 | Fix |

### B3. Missing Diagrams

Part IV has exactly **one SVG diagram** in 2600+ lines of content (the D2Q5/D2Q9/D3Q7 lattice stencil figure). For chapters teaching a radically different computational paradigm, this is insufficient.

**Recommended additions (priority order):**

1. **Streaming diagram** (Ch 18.3): Show 5 nodes in a row. At time t, distributions f_1, f_2 have values at the center node. At time t+Δt, f_1 has moved one node right, f_2 one node left. Simple arrows. This makes streaming concrete.

2. **Collision visualization** (Ch 18.3): Bar chart showing 5 distributions (f_0...f_4) before collision (asymmetric) and after collision (closer to equilibrium). The "relaxation toward equilibrium" becomes visible.

3. **Bounce-back diagram** (Ch 19.5): A boundary node with f_i heading toward a wall and f_ī bouncing back. One image, replaces a paragraph of text.

4. **Gaussian bell curve with quadrature nodes** (Ch 18.4): A 2D bell curve (contour plot) with the 5 D2Q5 nodes marked as dots, showing why the center gets the largest weight. This is the visual the entire quadrature section is begging for.

5. **Dual-lattice coupling timeline** (Ch 20.4): Horizontal timeline showing one physical step: ionic → Lattice 1 step → Lattice 2 iterates (with sub-iteration count) → source exchange → next step. Shows the nested loop structure.

6. **LBM-EP 6-step algorithm flowchart** (Ch 19.6): Visual version of eq (19.11) as a circular flow diagram. Ionic → Source → Collide → Stream → Bounce-back → Recover → (loop).

### B4. Chapter 20 Worked Example Gap

Chapter 20 introduces three novel solver strategies and computes **zero** numbers. Compare with Ch 10 (which computes RK2 and RK4 to 6 decimal places on the same test problem) or Ch 18.5 (which carries a full BGK collision through both distribution and moment space).

**Recommendation:** Add a minimal worked example to §20.2 (pseudo-time):
- 3-node 1D domain, uniform conductivity
- Show: initial φ_e guess → one pseudo-time LBM iteration → residual → second iteration → residual decreasing
- The student sees the convergence mechanism concretely

Even a 20-line computation would transform this section from "I understand the idea" to "I could implement this."

### B5. The "What Comes Next" Paragraph (End of Ch 18)

The closing paragraph of §18.5 is broken — it reads like two overlapping drafts were merged. Current text:

> "We now have the complete theoretical framework... In the next chapter, we will put this machinery to work — the $f_i$ are our fundamental variables, and the BGK collision operator relaxes all of them at the same rate $1/\tau$. But there is a more revealing way to look at the same information. In Section 18.5 above, we transformed the distributions into **moment space**, — first by unlocking the moment-space relaxation rates to achieve anisotropic diffusion..."

This refers to "Section 18.5 above" from within Section 18.5 itself. The second half is describing what 18.5 already did. Needs a clean rewrite that previews Ch 19.

### B6. Chapter 19 Cross-References to Ch 9/10

The comparison table in §19.8 labels columns "Explicit (Ch 9)" and "Implicit (Ch 10)." These should be Ch 10 and Ch 11, respectively. The same error appears in §19.2 ("Chapter 8" for splitting → should be Ch 9) and §19.1 ("Chapters 9 and 10" for diffusion solvers → should be "Chapters 10 and 11").

---

## Part C: Narrative Arc Assessment

### The Story of Part IV

Part IV tells a three-act story:

1. **Ch 18** — *The Foundation*: "Here is a completely different way to solve PDEs. Instead of discretizing equations, we simulate particles. The macroscopic physics emerges automatically."
2. **Ch 19** — *The Application*: "Here is how this machinery solves the cardiac monodomain equation. Same physics, different paradigm."
3. **Ch 20** — *The Challenge*: "The bidomain equation has an elliptic constraint. Here are three ways to handle it."

This arc works well. The student is first equipped with the tool (Ch 18), then sees it in action on a familiar problem (Ch 19 — monodomain, which they already know from Part II), then faces the new challenge (Ch 20 — bidomain). The parallel with Part II → Part III (monodomain → bidomain) is intentional and effective.

### Comparison Across Parts

| Aspect | Part II (Ch 7–11) | Part III (Ch 12–17) | Part IV (Ch 18–20) |
|--------|-------------------|---------------------|---------------------|
| Story | Clear 5-act structure | Ch 12 strong, Ch 13–17 catalogue | Clear 3-act structure |
| Analogies | Multiple per chapter | Ch 12 only | Every chapter, especially Ch 20 |
| Worked examples | Excellent throughout | Zero in Ch 13–16 | Good in Ch 18–19, absent in Ch 20 |
| Diagrams | 5+ SVGs across Part II | 2 total in Part III | 1 total in Part IV |
| Difficulty ramp | Gentle, consistent | Cliff at Ch 13.2 | Steep in Ch 18.1–18.2, levels off |
| Chapter length | Well-balanced (except Ch 8) | Unbalanced | Ch 18 too long, Ch 20 too short |
| Engine mapping | Every chapter | Ch 15–17 wrong architecture | Ch 18 none, Ch 19 good, Ch 20 speculative |
| Comparison tables | §8.6, §10.6, §11.5 | Missing | §19.5.5, §19.8 |

---

## Summary Checklist

### Critical Issues (must fix)
| # | Issue | Location | Fix |
|---|-------|----------|-----|
| C1 | "Chapter 17" → "Chapter 18" systematic cross-reference error | Ch 19 (~8), Ch 20 (~4) | Find/replace |
| C2 | "equation 17.x" → "equation 18.x" | Ch 19 (~12 occurrences) | Find/replace |
| C3 | Figure labeled "Figure 17.1" | Ch 18 | Rename to "Figure 18.1" |
| C4 | Broken closing paragraph at end of §18.5 | Ch 18, last paragraph | Rewrite |

### Major Issues (should fix)
| # | Issue | Location | Fix |
|---|-------|----------|-----|
| M1 | Ch 18 is ~1300 lines with no algorithm preview | Ch 18 opening | Add 30-second preview box showing the 6-step LBM algorithm |
| M2 | Part IV has only 1 SVG diagram total | All chapters | Add streaming, collision, bounce-back, bell curve, timeline diagrams (see B3) |
| M3 | Ch 20 has zero worked examples | Entire Ch 20 | Add pseudo-time convergence example with 3 nodes |
| M4 | Ch 19.8 table says "Ch 9" / "Ch 10" for explicit/implicit | §19.8 | Fix to Ch 10 / Ch 11 |
| M5 | Ch 19.2 says "Chapter 8" for splitting | §19.2 | Fix to Chapter 9 |
| M6 | Dual-lattice coupling equations (20.3, 20.4) use placeholders | §20.4 | Write out explicit source terms |

### Moderate Issues (nice to fix)
| # | Issue | Location | Fix |
|---|-------|----------|-----|
| m1 | §18.2 full collision integral (eq 18.8) serves no pedagogical purpose | §18.2 | Reduce to 1 sentence + the BGK simplification |
| m2 | §18.1 "Two Cases" subsection (rest vs flow) is long for cardiac-only audience | §18.1 | Trim flow case to 1 paragraph, mark as optional |
| m3 | "equation 18.2" reference in §19.3 should be (19.2) | §19.3 | Fix |
| m4 | "equation 8.1" for Rush-Larsen should be (9.1) | §19.2 | Fix |
| m5 | §20.5 performance estimates are unsupported | §20.5 | Add citations or mark as "estimated" |
| m6 | §20.5 closing claims "full foundation is complete" but bidomain LBM is unimplemented | §20.5 | Soften language ("provides the foundation for future implementation") |
| m7 | Ch 18 equation numbering has gaps (no 18.15–18.16 in text, jump from 18.10 to 18.11) | Ch 18 | Audit equation numbers for gaps and renumber |

### Minor Issues (cosmetic)
| # | Issue | Location | Fix |
|---|-------|----------|-----|
| s1 | §18.2 should mark force-term dropping as cardiac-specific simplification | §18.2 | Add "(for diffusion problems — force term relevant for fluid flow)" |
| s2 | D2Q5 weight derivation is in an insight-math box, could be promoted to main text | §18.4 | Consider moving to main flow for this key derivation |
| s3 | §19.1 repeats the diffusion tensor definition from Ch 7 | §19.1 | Acceptable (Part IV should be somewhat self-contained) |

---

## Overall Scores

| Chapter | L1 | L2 | L3 | L4 | L5 | Average | Grade |
|---------|----|----|----|----|----|---------| ------|
| 18 — LBM: Kinetic Theory to Computation | 4 | 5 | 3 | 4 | 3 | **3.8** | B+ |
| 19 — LBM for Monodomain | 4 | 5 | 3 | 5 | 5 | **4.4** | A- |
| 20 — LBM for Bidomain | 5 | 4 | 2 | 1 | 3 | **3.0** | B- |
| **Part IV Average** | **4.3** | **4.7** | **2.7** | **3.3** | **3.7** | **3.7** | **B+** |

For comparison:
- Part II (Monodomain) = **4.0/5.0** (A-)
- Part III (Bidomain) = **1.9/5.0** (D+)
- Part IV (LBM) = **3.7/5.0** (B+)

---

## Recommendations

### R1: Fix Cross-Reference Errors (1 hour)
Systematic find-and-replace: "Chapter 17" → "Chapter 18" in Ch 19–20; "equation 17.x" → "equation 18.x" in Ch 19; "Chapter 8" → "Chapter 9" for splitting; "Chapter 9/10" → "Chapter 10/11" for explicit/implicit; "Figure 17.1" → "Figure 18.1".

### R2: Add Algorithm Preview to Chapter 18 Opening (30 min)
Before §18.1, add a boxed "30-second preview" showing the 6-step LBM-EP algorithm (from eq 19.11) with annotations: "Step 1 (ionic) — same as Part II. Steps 3–6 (collide, stream, bounce-back, recover) — this is LBM, derived in §18.3–18.5. The rest of this chapter explains *why* each step takes the form it does."

### R3: Add 4–6 SVG Diagrams (3–4 hours)
Priority order: streaming diagram (Ch 18.3), collision bar-chart (Ch 18.3), bounce-back diagram (Ch 19.5), Gaussian with quadrature nodes (Ch 18.4), dual-lattice timeline (Ch 20.4), LBM-EP flowchart (Ch 19.6).

### R4: Add Worked Example to Chapter 20 (1–2 hours)
Pseudo-time convergence example: 3-node 1D domain, show 3–5 iterations with decreasing residual. Demonstrate that the steady state matches the elliptic solution.

### R5: Fix Broken Closing Paragraph of §18.5 (15 min)
Rewrite to cleanly preview Chapter 19: "The machinery is complete. In Chapter 19, we apply it to the cardiac monodomain equation — adding an ionic source term to the collision, operator splitting for reaction and diffusion, and bounce-back boundary conditions for tissue edges."

### R6: Consider Splitting Chapter 18 (Optional, High-Impact)
If Chapter 18's length is a concern, split into "The LBM Algorithm" (§18.3–18.4 + BGK worked example) and "Why LBM Works" (§18.1–18.2 + moment space). The algorithm chapter comes first; the theory chapter deepens understanding for interested readers.
