# Textbook Writing Style Guide — "Feynman Style"

## Core Philosophy

The gold standard is Section 7.8 ("Choosing Your Method"). Every section should:

1. **Tell a story** — not list facts. The reader should feel guided through a narrative, not browsing a reference manual.
2. **ELI5 first, then add complexity** — Begin every hard concept with a physical analogy or plain-language paragraph. Only after the reader "gets it" intuitively should the math appear.
3. **Be honest about trade-offs** — No method/model is universally best. State clearly when something excels and when it fails.
4. **Stay grounded in the physical** — Every equation should be connected to something the reader can picture (current flowing, cells firing, waves propagating).
5. **Respect the reader's intelligence** — ELI5 does not mean dumbing down. It means removing unnecessary abstraction while keeping rigor.
6. **Final complexity matches implementation** — The deepest layer of each section must reach the level of detail needed to write actual code. Pseudocode, matrix forms, and algorithm steps should map directly to Engine V5.4 function signatures.

## The Layered Complexity Principle

Every technical concept should be presented in **ascending layers**, each building on the last. The reader chooses how deep to go; the author ensures every layer is self-consistent and that the final layer is implementation-ready.

### Layer 1 — ELI5 / Physical Analogy
Start with something anyone can picture. No math. Pure intuition.
- Example: "RK4 is like checking your GPS four times during a drive instead of once."
- This layer answers: **What is this, in one sentence?**

### Layer 2 — Feynman / Conceptual Explanation
Explain the *mechanism* in plain language with light technical vocabulary. Introduce the key idea without full notation.
- Example: "Instead of taking one big step, we sample the slope at several points and combine them for a better estimate."
- This layer answers: **Why does this work?**

### Layer 3 — 3B1B / Visual + Mathematical
Introduce the actual equations alongside diagrams (SVG, stability region plots, slope visualizations). Each equation gets a graphical counterpart so the reader can *see* what the math describes.
- Example: The four $k$-stages of RK4 shown as slope evaluations on a curve, with arrows.
- This layer answers: **What does the math look like, and what does each piece mean?**

### Layer 4 — Worked Example
Plug in real numbers (or simple test values). Walk through every arithmetic step. No skipping.
- Example: Apply RK2 to $f(t,y) = -2y$ with $y_0=1$, $\Delta t=0.1$ — compute $k_1$, $k_2$, final answer.
- This layer answers: **Can I reproduce this with pencil and paper?**

### Layer 5 — Implementation Detail
Pseudocode or code-level specifics that map directly to the engine. Include function signatures, data structures, and computational cost.
- Example: `k1 = dt * diffusion_operator(V_n)` with explicit loop structure matching `ExplicitSolver.step()`.
- This layer answers: **How do I code this?**

### Why This Matters
A section that stops at Layer 2 leaves the reader unable to implement. A section that starts at Layer 4 loses the reader who doesn't yet understand the concept. The full stack — ELI5 → Feynman → 3B1B → Worked Example → Implementation — ensures every reader finds their entry point and can follow the thread all the way to working code.

## Structural Patterns

### Section Opening
- **DO**: Start with a motivating question or physical scenario ("Real hearts are not rectangular...")
- **DON'T**: Start with a definition or equation ("The weak form is defined as...")

### Introducing Math
- **DO**: One plain-language paragraph → then the equation → then physical interpretation
- **DON'T**: Equation → wall of variable definitions → next equation

### Worked Examples
- **DO**: Use small, concrete numerical examples (5-node cable, 4-node mesh) with actual numbers
- **DON'T**: Leave the reader to "verify as an exercise"

### Comparison/Decision Sections
- **DO**: Use a comparison table, then 1-2 sentences per option on when each is used in practice
- **DON'T**: Write multi-paragraph arguments for each option without a summary

### Insight Boxes
- `insight-intuition` (green): Physical analogies, "what this means" explanations
- `insight-warning` (yellow): Important caveats, "when to use" / "when NOT to use"
- `insight-math` (purple): Mathematical subtleties for advanced readers
- `insight-engine` (blue): Connection to Engine V5.4 implementation

## Voice and Tone

| DO | DON'T |
|----|-------|
| "Think of cardiac tissue like a bundle of straws" | "The tissue exhibits anisotropic conduction" (as an opener) |
| "This is just Ohm's law in continuous form" | "Applying the constitutive relation" |
| "The crucial step is..." | "It can be shown that..." |
| "Check: every row sums to zero" | "The reader may verify that..." |
| "This matters because..." | "Note that..." (without explaining why) |
| "Let us see this concretely" | "We now derive..." (without motivation) |
| Active voice: "FEM assembles the matrix" | Passive: "The matrix is assembled by FEM" |

## Equation Formatting Rules

1. **Number every important equation** that will be referenced later: `(7.1)`, `(7.2)`, etc.
2. **Multi-line for long equations**: Use `\begin{aligned}` when an equation exceeds ~60 characters of LaTeX.
3. **Underbrace for meaning**: Use `\underbrace{...}_{\text{label}}` to annotate what each term represents.
4. **No orphan equations**: Every equation must have at least one sentence before it (setup) and one after (interpretation).

## Cross-Reference Consistency

- Ionic current totals: "equation (5.1) for TTP06" and "equation (6.1) for ORd"
- Monodomain PDE: always "equation (7.1)"
- Semi-discrete system: always "equation (7.5)"
- State variable counts: TTP06 = 18 (12 gates + 5 concentrations + R'), ORd = 41 (29 gates + 8 concentrations + CaMKII + Vm + release states)
- Surface-to-volume ratio: $\chi$ consistently
- Membrane capacitance: $C_m$ consistently

## What "Needs Improvement" Means

A section needs improvement if it has ANY of these:
1. **Math-first opening** — jumps into equations without physical motivation
2. **Missing ELI5** — a hard concept with no plain-language bridge
3. **Disconnected equations** — formula appears without showing where it came from
4. **No worked example** — especially for matrix/discretization sections
5. **Repeated content** — same concept explained in two chapters without cross-reference
6. **Inconsistent notation** — same quantity with different symbols in different chapters
7. **Wall of text** — >4 paragraphs without a table, figure, insight box, or equation to break it up
8. **Missing "why"** — tells the reader WHAT but not WHY it matters
9. **Stops at Layer 2** — concept is explained but no equations, no worked example, no implementation path
10. **Missing visual** — equation-heavy section with no SVG diagram, stability plot, or graphical interpretation
