# Bidomain Textbook — Changelog

All major edits to `bidomain_textbook.html`, newest first.

## 2026-02-24 (session 8)

### Chapter 17.3 and 17.4 restructuring

**Section 17.3 — Cleaned up**
- Moved "Core Strategy of LBM" from 17.2 into 17.3 as indented roadmap box
- Added "From Continuous to Discrete: Three Deliberate Choices" bridge (17.2 → 17.3)
- Rewrote discrete LBE as $f_i(t+\Delta t) - f_i(t) = \text{collision}$ (eq 17.13)
- Added combined streaming form (eq 17.16) after Step 1 + Step 2
- Removed premature $f_i^{\text{eq}} = w_i\phi$ formula — replaced with forward pointer to 17.4
- Moved DdQq naming convention, lattice choice guidance, and Memory vs Physics box to 17.4

**Section 17.4 — Rewritten narrative frame**
- Deleted stale re-introduction of Maxwell-Boltzmann ("But what IS this equilibrium?")
- New opening: "Returning to the Gaussian" — one paragraph bridging 17.1 → 17.2 → 17.3 → now
- New subsection: "The Three Claims Become Three Constraints" — eqs (17.3)→(17.18), (17.4)→(17.19), (17.5)→(17.20) with explicit continuous-to-discrete mapping
- Quadrature section tightened to reference the three constraints
- feq subsection rewritten as "Closing the Loop" — connects back to 17.1 and 17.3
- $c_s$ dual-role paragraph moved from deleted opening to τ-D section
- DdQq naming, lattice tables, SVG figure, weights, derivation boxes all kept

**Section 17.5 header fixed** from mislabeled "17.4" to "17.5"

**Equation renumbering**: 17.1–17.44 (plus 17.7a), no gaps, all cross-references updated

---

### Chapter 17.1 and 17.2 fixes

**Section 17.1 — Three Physical Claims alignment**
- Wrapped zeroth/first/second order claims in styled `<div>` with `<h4>` subheadings for consistent visual alignment

**Section 17.2 — Restructured and expanded**
- Added total derivative form (eq 17.7a) with Eulerian vs Lagrangian explanation
- Moved BGK simplification BEFORE "Why the Boltzmann Equation Is Almost Never Solved" and Chapman-Enskog
- Expanded Chapman-Enskog analysis: zeroth order → conservation (Euler), first order → dissipative terms (viscous stress, diffusion) with explicit underbraces
- Connected f^eq back to 17.1's three claims (eqs 17.3–17.5)
- Equations: (17.7)–(17.12) plus new (17.7a)

---

## 2026-02-21 (session 7)

### Chapter 17 major restructuring: New 17.1 + Moment Space (17.5)

**New Section 17.1 — Thermodynamic Foundation: Phase Space and the Equilibrium State**
- Position space → velocity space → phase space conceptual progression
- Distribution function $f(\mathbf{x}, \mathbf{c}, t)$ introduced with physical intuition
- Maxwell-Boltzmann equilibrium as the Gaussian in velocity space (eq 17.2)
- Three physical claims of the Gaussian: conservation (zeroth), zero flux (first), isotropic variance (second)
- Two cases: system at rest (diffusion) vs system with flow (Navier-Stokes)
- Bridge paragraph: why the equilibrium is the heart of LBM
- Equations (17.1)–(17.6)

**Section renumbering**
- Old 17.1 (Boltzmann Equation) → 17.2
- Old 17.2 (Lattice-Boltzmann Equation) → 17.3
- Old 17.3 (Maxwellian/Quadrature) → 17.4
- New Section 17.5: Moment Space (draft inserted)
- All equations renumbered sequentially (17.1–17.44)
- Cross-references in Ch 18–19 updated

**New Section 17.5 — Moment Space: Physical Decomposition of the Collision Operator**
- Why the collision operator hides its structure
- Moments dictated by PDE conservation laws (zeroth, first, second order)
- Building M: each row is a conservation law (D2Q5 and D3Q7 explicit matrices)
- The round trip: decode → relax → re-encode
- BGK as special case (all dials locked; M⁻¹M cancels)
- feq vs moment space: destination vs journey insight
- Worked example: one D2Q5 collision in both distribution and moment space

**Chapter 17 ending cleaned up**
- Removed duplicate "Equilibrium Distribution Function" and "Lattice Requirements" sections
- Replaced "Bridge to Cardiac Electrophysiology" with "What Comes Next" pointing to moment space + appendix

### PDF regenerated
- Main: `Cardiac_Computational_Modeling.pdf` — 2550 MathJax containers, 5.8 MB
- Standalone: `LBM_Textbook_Part_IV.pdf` — 801 MathJax containers, 1.9 MB

## 2026-02-19 (session 6, continued — part 3)

### Figure 17.1 upgrade: Multi-panel research-quality SVG
- **Replaced old D2Q5-only SVG** (200×200 viewBox, crude) with new 920×320 three-panel figure
- **Panel (a) D2Q5**: Center node (blue), 4 cardinal neighbors (orange), weight labels, direction indices
- **Panel (b) D2Q9**: Cardinal (dark arrows) + diagonal (vermillion arrows), all 9 weight labels
- **Panel (c) D3Q7**: Isometric 3D projection with axis labels, face-centered neighbors, depth cues
- **Colorblind-safe Okabe-Ito palette**: Blue #0072B2 (center), Orange #E69F00 (cardinal), Vermillion #D55E00 (diagonal)
- **SVG_FIGURES_SKILL.md**: Created comprehensive protocol for research-quality SVG figure generation

### Equations 17.14–17.15: Vector notation added
- **Eq 17.14**: Added vector form $\sum_i w_i \mathbf{c}_i = \mathbf{0}$ alongside Einstein notation
- **Eq 17.15**: Added outer product form $\sum_i w_i \mathbf{c}_i \otimes \mathbf{c}_i = c_s^2 \mathbf{I}$ alongside Einstein notation
- **Outer product explained**: Added description of $\otimes$ notation

### Notation fix: f^(0) → f^eq
- Changed $f^{(0)}$ to $f^{\text{eq}}$ in equation 17.12 (f^(0) was never introduced)

### PDF regenerated
- Main: `Cardiac_Computational_Modeling.pdf` — 2321 MathJax containers, 5.4 MB
- Standalone: `LBM_Textbook_Part_IV.pdf` — 572 MathJax containers, 1.5 MB

## 2026-02-19 (session 6, continued — part 2)

### Section 17.3 rewrite: Gaussian quadrature framework
- **New narrative**: Section 17.3 restructured around the central idea that weights and velocities arise from Gauss-Hermite quadrature on the Maxwell-Boltzmann Gaussian
- **New subsection "The Continuous Equilibrium Is a Gaussian"**: Maxwell-Boltzmann distribution (eq 17.12), 3B1B-style bell curve intuition, two cases (gas at rest vs gas with flow)
- **New subsection "Discretizing the Bell Curve: Gauss-Hermite Quadrature"**: Quadrature nodes = discrete velocities, quadrature weights = lattice weights, paired by construction
- **Exactness conditions**: Isotropy conditions (17.13–17.15) reframed as quadrature exactness requirements (normalization, symmetry, variance)
- **Weight intuition**: Each weight now explained as "how much of the bell curve this node represents" — diagonal weights smaller because Gaussian decays with distance from peak
- **New insight box**: "Why Diagonal Weights Are Smaller" — Gaussian decay explanation
- **feq connected to Gaussian**: Diffusion equilibrium ($w_i \phi$) = centered Gaussian at quadrature nodes; fluid flow equilibrium = Taylor expansion of shifted Gaussian
- **Moments NOT introduced**: Saved for MRT chapter; conditions called "exactness requirements" not "moments"
- **32 markdown bold artifacts** (`**text**`) converted to `<strong>` HTML tags across all of Ch 17

### PDF regenerated
- Main: `Cardiac_Computational_Modeling.pdf` — 2289 MathJax containers, 5.3 MB
- Standalone: `LBM_Textbook_Part_IV.pdf` — 540 MathJax containers, 1.4 MB

## 2026-02-19 (session 6, continued)

### Chapter 17 v3 fixes: Formatting, consistency, variable audit
- **Introduced $f_i^*$ notation**: Added explicit definition ("post-collision distribution") after eq 17.8, added to variable reference table
- **D2Q9 velocity table**: Expanded from 3-row compact form to 9 individual rows with integer components (1, 0, -1), matching D2Q5 convention
- **D3Q7 velocity table**: Replaced equation-only form with full 7-row table using same integer convention
- **Weight set intuition**: Added physical descriptions for each weight set — what rest/cardinal/diagonal weights represent at equilibrium
- **Fixed insight-math artifact**: Removed stream-of-consciousness text; replaced with clean derivation of D2Q5 weights
- **Restructured feq section**: Renamed "Why D2Q5 Suffices..." to "The Equilibrium Distribution Function" — introduces feq generally before discussing lattice requirements
- **Fixed SVG artifact**: Removed invalid `<arrow>` HTML tag from D2Q5 diagram
- **Div balance verified**: 384/384

### PDF regenerated
- Main: `Cardiac_Computational_Modeling.pdf` — 2247 MathJax containers, 5.1 MB
- Standalone: `LBM_Textbook_Part_IV.pdf` — 498 MathJax containers, 1.3 MB

## 2026-02-19 (session 6)

### Chapter 17 v3: Complete rewrite following Perumal & Dass review paper
- **New structure**: 3 sections following kinetic theory → LBM derivation → lattice geometry
- **Section 17.1 — The Boltzmann Equation**: Continuous Boltzmann equation in differential form, LHS (free-streaming/external forces) vs RHS (collision integral), explains collision integral complexity (5D integral), BGK simplification, Chapman-Enskog → Navier-Stokes connection
- **Section 17.2 — LBM Single Relaxation Time**: Presents unified BGK equation, splits into collision + streaming steps, explains every variable in component form (f_i, f_i^eq, τ, Δt, c_i), DXQY naming convention, complete variable reference table
- **Section 17.3 — Weight Lattice & Discrete Velocity Set**: Isotropy conditions, weight derivation, D2Q5 lattice with SVG diagram, lattice speed of sound c_s, τ-D relationship
- **Equations**: (17.1)–(17.25), including BGK approximation, equilibrium distribution, isotropy conditions, τ-D relationship
- **Features**: 1 SVG diagram (D2Q5 stencil), 8 insight boxes, variable reference table
- **Size**: 467 lines (old Ch 17 v2 was 1080 lines — leaner and more focused)
- **Ch 18 cross-refs updated**: eq 17.16→17.9 (streaming), eq 17.17→17.23 (τ-D), Section 17.5→17.2, Section 17.10→17.2

### PDF regenerated
- Main: `Cardiac_Computational_Modeling.pdf` — 2202 MathJax containers, 5.1 MB
- Standalone: `LBM_Textbook_Part_IV.pdf` — 453 MathJax containers, 1.2 MB

## 2026-02-13 (session 5, continued — part 2)

### Chapter 18 refactored: Removed redundancy with expanded Ch 17
- **Sections reduced**: 8 → 7 (18.1–18.7), total lines ~480 → ~280
- **Removed redundant re-introductions** of distribution functions, D2Q5/D3Q7 lattices, BGK collision, streaming, τ-D relationship, MRT, and bounce-back — all now thoroughly covered in Ch 17
- **Added cross-references** to Ch 17 for foundational concepts (~15 cross-refs)
- **Focused on cardiac-specific content**: ionic source term coupling (eq 18.1), BGK with source (eq 18.2), MRT with source (eq 18.3 boxed), 5-stage LBM-EP algorithm (eq 18.4 boxed)
- **Kept unique content**: worked example with cardiac parameters, comparison table (Explicit vs Implicit vs LBM), GPU acceleration, Engine V5.4 insight box
- **Equations renumbered**: (18.1)–(18.4) instead of (18.1)–(18.10) — removed equations that duplicated Ch 17 content

### Cross-reference fixes
- Ch 19: "bidomain equations from Chapter 17" → "from Chapter 12" (correct source)
- Ch 19: "equation 18.6" → "equation 17.17" (τ-D relationship moved to Ch 17)
- Ch 7: "Lattice Boltzmann-based solvers (Chapter 11)" → "(Chapters 17–18)"
- Ch 19: "Part III (Chapters 17–19)" → "Part IV (Chapters 17–19)"

### PDF regenerated
- Main: `Cardiac_Computational_Modeling.pdf` — 2235 MathJax containers, 5.4 MB
- Standalone: `LBM_Textbook_Part_IV.pdf` — 486 MathJax containers, 1.5 MB

## 2026-02-13 (session 5, continued)

### Chapter 17 v2: Major rewrite addressing user feedback
- **17.1 rewritten**: Full derivation from diffusion PDE → 1D specialization → centered differences → Forward Euler → weighted average (not dropped out of nowhere)
- **Added SVG diagrams**: D2Q5 lattice diagram (center node with 4 cardinal arrows + rest), D3Q7 lattice diagram (3D extension), 1D redistribution diagram, collision-streaming 3×3 grid diagram, cardiac LBM state diagram
- **New Section 17.4**: The Distance Matrix — $\|\mathbf{e}_i\| \cdot \Delta x$, lattice velocity $c = \Delta x/\Delta t$, speed of sound $c_s = c/\sqrt{3}$
- **17.5 rewritten**: Collision and streaming axis independence — East/West independent of North/South, f₀ never moves, explicit multi-step explanation with SVG
- **New Section 17.7**: What Gets Streamed in Cardiac LBM — voltage distributions (not momentum), ionic state variables NOT streamed (node-local ODE), operator splitting connects to Chapter 8
- **17.8 rewritten**: Conservation in Cardiac LBM vs. Navier-Stokes — Vm is conserved (not momentum), flux slaved to ∇Vm via Fick's law, source term (ionic current) as body force analogue
- **Overall**: Reduced math-first feel, added physical intuition throughout, 28 SVG elements, 34 equation references
- **File grew**: ~730 lines → ~1080 lines (Ch 17 alone)

### PDF regenerated
- Main: `Cardiac_Computational_Modeling.pdf` — 2319 MathJax containers, 5.6 MB
- Standalone: `LBM_Textbook_Part_IV.pdf` — 570 MathJax containers, 1.7 MB

## 2026-02-13 (session 5)

### Major restructuring: Part IV Lattice-Boltzmann Methods
- **Renamed textbook** from "Computational Cardiac Electrophysiology" to "Cardiac Computational Modeling"
- **Created Part IV** — dedicated section for all LBM content (3 chapters)
- **New Chapter 17**: "Why LBM? — Discovering a New Paradigm Through the Heat Equation"
  - 10 sections covering: heat equation redistribution, distribution functions, direction matrix,
    BGK collision, streaming, averaging/flux conservation, eigen-angle matrix (anisotropic diffusion),
    transformation matrix M (MRT), Chapman–Enskog bridge, cardiac diffusion bridge
  - Equations (17.1)–(17.18): FD as weighted average, equilibrium, voltage recovery, isotropy condition,
    direction matrices E for D2Q5/D3Q7, BGK collision, streaming, τ-D relationship, mass conservation,
    flux/Fick's law, eigendecomposition, D_xx/D_yy from fiber angle, M matrix, MRT collision, bounce-back
  - Feynman-style narrative: story of discovering LBM through solving the heat equation
- **Chapter 18**: LBM for Monodomain (moved from old Ch 11, renumbered)
  - All 10 equations renumbered (11.x → 18.x), 8 sections renumbered
- **Chapter 19**: LBM for Bidomain (expanded from old Ch 17)
  - 5 sections: bidomain challenge, pseudo-time relaxation, hybrid LBM–classical, dual-lattice architecture, performance
  - Equations (19.1)–(19.4): pseudo-time augmented equation, phi_e collision, Vm lattice collision, phi_e lattice collision
  - Feynman-style analogies: two fish tanks, heating a metal plate, right tool for the job
  - Architecture comparison table (Dual-Lattice vs Hybrid vs Full Classical)
- **Renumbered Part III**: old Ch 12–16 → Ch 11–15, old Ch 18 → Ch 16 (Implementation Roadmap)
- **Removed old Ch 11** from Part II and **old Ch 17** from Part III
- **Updated TOC** with complete Part IV listing
- **Created standalone LBM textbook** (LBM_Textbook_Part_IV.pdf) with Ch 17–19 + Appendix A
- Research papers read: Rapaka (MICCAI 2012), Corre & Belmiloudi (2016), Campos et al. (2016), Perumal & Dass (2015)

### PDF regenerated
- Main: `Cardiac_Computational_Modeling.pdf` — 2294 MathJax containers, 5.4 MB, 19 chapters + appendices
- Standalone: `LBM_Textbook_Part_IV.pdf` — 545 MathJax containers, 1.5 MB, Ch 17–19 + Appendix A

## 2026-02-13 (session 4)

### Ch 11: Full expansion — Lattice-Boltzmann Methods for Monodomain
- Expanded from ~113 lines (3 thin sections, no numbered equations) to ~480 lines (8 sections, 10 equations)
- Added eq (11.1): voltage recovery V = Σ f_i
- Added eq (11.2): equilibrium distribution f_eq_i = w_i · V
- Added Section 11.2: D2Q5/D3Q7 lattice tables with weights, velocities, opposite directions
- Added eq (11.3): BGK collision f*_i = f_i - ω(f_i - f_eq_i) + Δt·w_i·S
- Added eq (11.4): source term S = -(I_ion + I_stim)/(χ·Cm)
- Added eq (11.5): streaming f_i(x + e_i) ← f*_i(x)
- Added eq (11.6): τ-D relationship (boxed): τ = 0.5 + 3·D·Δt/Δx²
- Added eq (11.7): MRT transformation matrix M (5×5 for D2Q5) with moment interpretation
- Added eq (11.8): MRT collision (boxed): f* = f - M⁻¹·S·(Mf - Mf_eq) + Δt·w·S
- Added eq (11.9): bounce-back BC
- Added eq (11.10): complete LBM-EP timestep (boxed 5-stage algorithm matching LBMSimulation.step())
- Added Section 11.5: bounce-back boundary conditions
- Added Section 11.6: complete LBM-EP timestep with worked example (5×5 grid)
- Added Section 11.7: comparison table (Explicit vs Implicit vs LBM-BGK) with 7 aspects
- Verified all equations against Engine V5.4 code: d2q5.py, d3q7.py, collision.py, state.py, monodomain.py
- Each section follows layered complexity: ELI5 → conceptual → math → code form

### PDF regenerated
- 2024 MathJax containers (up from 1891), 4.7 MB

## 2026-02-12 (session 3, continued)

### Ch 9: Restructure — final equations match code form
- Added eq (9.3b): discrete matrix form for RK2 using bare-k convention matching `RK2Solver` code
- Added eq (9.4b): discrete matrix form for RK4 using bare-k convention matching `RK4Solver` code
- Removed Section 9.8 (pseudocode) — replaced by (9.3b)/(9.4b) which show the code form directly
- Removed convention-note insight box (no longer needed)
- Condensed verbose engine insight box to short reference to DiffusionSolver ABC

### Ch 10: Full expansion — Implicit Diffusion Solvers
- Expanded from ~120 lines (3 thin sections, eqs 10.1–10.3) to ~330 lines (6 sections, eqs 10.1–10.6)
- Added Section 10.2: BDF1/Backward Euler with ELI5, continuous form (10.1), boxed matrix form (10.2)
- Added Section 10.3: Crank–Nicolson with ELI5, continuous form (10.3), boxed matrix form (10.4), worked example
- Added Section 10.4: BDF2 with ELI5, continuous form (10.5), boxed matrix form (10.6), bootstrap note
- Added Section 10.5: Three-method comparison table (BDF1/CN/BDF2) with A_lhs, B_rhs, stability columns
- Added Section 10.6: Linear solvers (PCG, Chebyshev, FFT) with engine insight box
- Each method follows layered progression: ELI5 → continuous → boxed A_lhs/B_rhs matrix form matching code
- Engine insight box documents CrankNicolsonSolver/BDF1Solver/BDF2Solver class structure

### PDF regenerated
- 1891 MathJax containers (up from 1809), 4.4 MB

## 2026-02-12 (session 3)

### Ch 9: Complete rewrite — Explicit Diffusion Solvers
- Expanded from 118 lines (3 thin sections) to 636 lines (8 sections: 9.1–9.8)
- Added RK2/Heun's method: equation (9.3), SVG slope diagram (Figure 9.1), worked example
- Added RK4 classical: equation (9.4), SVG four-stage diagram (Figure 9.2), worked example
- Added stability regions: equations (9.5)–(9.6), SVG stability region plot (Figure 9.3)
- Added cost vs. accuracy comparison table (Section 9.6)
- Added implementation pseudocode matching Engine V5.4 class names (Section 9.8)
- Added insight-math box explaining k-convention difference (math: dt baked in; code: bare operator)
- Full 5-layer treatment per STYLE_GUIDE.md layered complexity principle

### Engine V5.4 cross-reference audit (all engine insight boxes)
- **Ch 1**: Fixed method names `compute_ionic_current()`→`compute_Iion()`, `update_state_variables()`→`step()`
- **Ch 5**: Fixed file path `ttp06.py`→`ttp06/` package (model.py, currents.py, gating.py, calcium.py, parameters.py)
- **Ch 6**: Fixed method `update_state_variables()`→`step()`, state count "41"→"40 ionic + V stored separately"
- **Ch 7**: Fixed interface `DiffusionOperator`→`SpatialDiscretization`, method `apply_diffusion(V)`/`get_diffusion_operators(dt, scheme)`, state count "41"→"40 + V"
- **Ch 8**: Rewrote entire engine box: described `SplittingStrategy` ABC, `GodunovSplitting`, `StrangSplitting`, `RushLarsenSolver` with its specific 5-step order of operations
- **Ch 9**: Rewrote pseudocode to match actual Engine V5.4 classes (`DiffusionSolver(ABC)`, `ForwardEulerDiffusionSolver`, `RK2Solver`, `RK4Solver`)

### Ionic equation verification against Engine V5.4 code
- **Eq (5.3) ICaL (TTP06)**: Fixed three errors — added 15 mV voltage shift (V→V-15), corrected activity coefficients (0.25 on intracellular, none on extracellular), changed notation from ḡ to P_CaL
- **Eq (5.10) INaCa (TTP06)**: Fixed two errors — added missing α=2.5 factor on reverse term, corrected saturation from constant 1/(K_sat+1) to voltage-dependent 1/(1+k_sat·exp((γ-1)FV/RT))
- **Section 5.4 Gto values**: Corrected epicardial from 0.073→0.294, endocardial from 0.0→0.073
- **Eq (6.7) IKr (ORd)**: Fixed Ko normalization from 5.0→5.4
- **Eq (6.8) IKs (ORd)**: Added missing KsCa calcium-sensitivity factor with explanatory text
- **Eq (6.9) IK1 (ORd)**: Fixed Ko factor from √(Ko/5.0) to √Ko (no normalization in ORd)
- **Section 6.8 heterogeneity table**: Expanded from 4 rows to 8 rows with correct scaling factors matching code (e.g., EPI GKs = 1.4×, not same as ENDO)
- **Section 6.4**: Added engine insight box noting chloride currents (IClCa, IClb) not implemented in Engine V5.4

### STYLE_GUIDE.md update
- Added "Layered Complexity Principle" (ELI5 → Feynman → 3B1B → Worked Example → Implementation)
- Added principle #6: "Final complexity matches implementation"
- Added "Needs Improvement" criteria #9 (Stops at Layer 2) and #10 (Missing visual)

## 2026-02-12 (sessions 1-2)

### Systematic audit and fixes (all chapters)
- **Equation numbering fixed (critical)**: Ch 9 had (8.x)→(9.x), Ch 10 had (9.x)→(10.x), Ch 12 had (11.x)→(12.x), Ch 13 had (12.x)→(13.x), Ch 14 had (13.x)→(14.x), Ch 15 had \tag{4.x}→\tag{15.x}
- **Figure numbering fixed**: Ch 12 Figure 1.1→12.1
- **Cross-references fixed**: Ch 13 refs to old eq numbers, Ch 16 ref to block system, Ch 18 "Chapters 4-6"→"Chapters 14-16"
- **CSS class fixed**: Ch 15 `equation-box`→`equation-block` (5 instances)
- **Notation consistency**: Ch 10 equation (10.2) changed from $\mathbf{L}$ to $\mathbf{K}$/$\mathbf{M}$ to match Ch 7 conventions
- **ELI5 intros added**: Ch 8.1 (stir-fry analogy for operator splitting), Ch 9.1 (driving analogy for Forward Euler), Ch 10.1 (boat-in-rapids analogy for implicit methods)
- Created STYLE_GUIDE.md, CHANGELOG.md, INDEX.md tracking documents

### Ch 7: Section 7.9 — Full derivation of equation (7.5)
- Replaced the 3-paragraph section with a 6-step derivation: continuous PDE → node sampling → diffusion matrix → mass matrix → load vector → assembly
- Added origin table mapping each matrix term to its source
- Added intuition box explaining eq (7.5) in plain language
- Kept the full-timestep walkthrough and Engine V5.4 insight box
- Updated TOC entry: "From Ionic Model to Full System" → "From PDE to Matrix System"

### Ch 7: ELI5 additions throughout
- 7.1: Added "row of light bulbs connected by wires" analogy before Ohm's law
- 7.2: Added "voltage change = reaction + diffusion" plain-language paragraph before PDE
- 7.3: Added "bundle of straws" anisotropy analogy + explanation of why a tensor is needed
- 7.6 (FEM weak form): Added 2-paragraph ELI5 explaining why weak form is needed (kinks in hat functions) and what integration by parts does physically

### Ch 6: Complete rewrite to match Ch 5 format
- Expanded from 4 sections (9,626 chars) to 10 sections (49,425 chars)
- Added: 6.1 (41 state variables), 6.2 (current inventory), 6.3–6.5 (currents by function), 6.6 (CaMKII), 6.7 (visual map), 6.8 (heterogeneity), 6.9 (calcium handling), 6.10 (comparison table)
- 20 numbered equations (6.1)–(6.20)
- Fixed equation overflow in 6.2: single-line → `\begin{aligned}` two-line layout
- Removed duplicate IKb from background currents
- Fixed equation numbering conflicts: Ch 7 (6.x) → (7.x), Ch 8 Rush-Larsen (7.1) → (8.1)

### Ch 7: Spatial discretization rewrite (7.4–7.9)
- Replaced 17,862 chars with 40,905 chars
- Added worked examples: 5-node 1D FDM cable, 4-node 2-triangle FEM mesh, FVM 1D example
- Added physical interpretations for every matrix operation
- Added Section 7.8 (Choosing Your Method) with 10-row comparison table
- Added Section 7.9 with F(t) explanation and full timestep walkthrough

## Earlier sessions (pre-2026-02-12)

### Ch 5: TTP06 expansion
- Expanded to 8 sections covering all 12 currents with equations (5.1)–(5.17)
- Added state variable tables, visual activity map, transmural heterogeneity, calcium handling

### Ch 4: Calcium chapter expansion
- Expanded to 8 sections (880 lines) covering dyadic cleft, CICR, RyR, SERCA/PLB, NCX, buffering, T-tubules, frontiers
- SVG diagrams included (user prefers literature images — replacement pending)

### Ch 7.4–7.8: Initial spatial discretization content
- First version of FDM, FEM, FVM sections (later rewritten with worked examples)

### Appendix A: Visual Guide to Differential Equations
- 7 sections (A.1–A.7) covering heat equation, Laplacian, PDE families, parabolic/elliptic/hyperbolic, connection to bidomain

### Part III: Bidomain chapters 12–18
- Initial content for all bidomain chapters
- Chapter 12: physical motivation
- Chapters 13–17: equations, discretization, time integration, solvers, LBM
- Chapter 18: implementation roadmap
