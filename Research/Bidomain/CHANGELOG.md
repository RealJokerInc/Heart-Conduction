# Bidomain Textbook — Changelog

All major edits to `bidomain_textbook.html`, newest first.

## 2026-03-09 (session 15) — Appendix C Complete Rewrite + B Visual Overhaul

### Design Decision
Appendix C restructured from concept-organized (9 sections) to **method-by-method on a 2D grid** (14 sections). Each numerical method gets its own section with a worked example on the same running problem. Iterative methods expanded: CG, PCG, and Chebyshev each get dedicated sections per user request.

### Appendix C Rewrite (C.1–C.14)
- **C.1**: 3×3 grid setup, explicit 9×9 Laplacian assembly, SVG Figure C.1 (grid + 5-point stencil), boundary node stencils, mass matrix aside
- **C.2**: Truncation error (kept, condensed)
- **C.3**: Forward Euler — dedicated section, hot-spot worked example on 3×3 grid, SVG Figure C.2 (before/after heatmap)
- **C.4**: Stability — CFL violation blowup demo ($\Delta t = 0.5$ gives $-9$ at center), eigenvalue analysis, 2D CFL formula, SVG Figure C.3 (stability regions)
- **C.5**: Backward Euler — dedicated section, same grid, builds $A = I - \Delta t DL$ explicitly (9×9), worked example with solution
- **C.6**: Crank-Nicolson — dedicated section, FE vs BE vs CN comparison table on same grid
- **C.7**: Lax Equivalence (kept, brief)
- **C.8**: Operator Splitting — expanded: Godunov and Strang algorithms (boxed), explanation of why each sub-problem is easier
- **C.9**: Three solver families comparison table (direct/spectral/iterative)
- **C.10**: Transform methods — orchestra analogy, DCT/DST, eigenvalue formulas, 1D worked example, limitations
- **C.11**: Iterative methods intro — the iteration loop, warm-starting, 4-point 1D running example setup
- **C.12**: Conjugate Gradient — energy minimization, Algorithm C.1 (boxed), 2-iteration worked example on 4×4 system with full arithmetic, convergence bound
- **C.13**: PCG — preconditioning concept ("reshape the bowl"), Jacobi preconditioner, Algorithm C.2 (boxed), cost vs benefit table
- **C.14**: Chebyshev — GPU bottleneck motivation, three-term recurrence, Algorithm C.3 (boxed), CG-then-Chebyshev workflow, comparison table

### Appendix B Visual Overhaul
- **B.1**: Softened vector definition to dual arrow+list (3B1B alignment)
- **B.2**: Added SVG Figure B.1 (unit square → parallelogram transformation)
- **B.7**: Added 2×2 eigenvalue worked example
- **B.8**: Added SVG Figure B.2 (SPD bowl vs semi-definite trough)
- **B.9**: Added SVG Figure B.3 (round vs elongated contour plots with descent paths)
- **B.11**: Added numerical 4×4 Schur complement worked example

### Appendix A Fix
- **A.7**: "Chapter 17" → "Part IV (Chapters 18–20)"

### Cross-Reference Updates
- ch8.html: equation (C.7) → (C.5), Appendix C.8 → C.10
- ch15.html: Appendix C.9 → C.13
- appendix-b.html: Section C.8 → C.10, Section C.9 → C.12

### Equation Renumbering
Old → New: C.3 (FE) kept, C.4 (CFL) kept, C.5 (eigenvalues, was C.7), C.6 (BE, was C.4), C.7 (CN, was C.5), C.8 (DST, was C.8), C.9 (2D sum, was C.9), C.10–C.12 (iterative, was C.10–C.12)

---

## 2026-03-09 (session 14) — Four-Appendix Restructure: A(DEs), B(LA), C(Bridge), D(PyTorch)

### Motivation
Linear algebra and numerical analysis concepts (SPD, eigenvalues, condition number, null space, CFL, Chebyshev) were used throughout Parts II–IV without proper introduction. The three appendices should form a clean dependency chain: A = what we want to solve (DEs), B = the tools (LA), C = how to combine them (numerical analysis as the bridge).

### Design Principle
Each appendix does ONE job. A is pure differential equations. B is pure linear algebra (no cardiac content, no discretization). C is "The Bridge" — it shows how A + B combine to produce computable solutions. The discretization story (functions→vectors→matrices) lives in C.1, not B.1, because discretization IS the bridge.

### Structural Changes
- **Appendix A**: Trimmed to A.1–A.7 (PDE types only). A.8–A.10 migrated to C.
- **Appendix B** (NEW): Pure Linear Algebra — 12 sections (B.1–B.12), 3B1B style. B.1 vectors/spaces, B.2 matrices as transformations, B.7 eigenvalues (abstract, no Laplacian formulas), B.8 SPD (no cardiac table), B.11 block matrices/Schur (abstract, no bidomain). Equations (B.1)–(B.8). No B.13 cross-reference table.
- **Appendix C** (NEW): Numerical Analysis: The Bridge — 9 sections (C.1–C.9). Narrative arc: C.1 discretization leap (PDE→matrix), C.2 truncation error, **C.3 time-stepping → Ax=b** (the climactic section where A meets B), C.4 stability with Laplacian eigenvalue formulas, C.5 Lax, C.6 stiffness, C.7 direct vs iterative, C.8 transform methods (DCT/DST), C.9 CG/PCG/Chebyshev. Equations (C.1)–(C.12).
- **Appendix D**: Old PyTorch appendix renumbered B.x→D.x. Content unchanged.

### Cross-Reference Updates
- Ch 8: Neumann eigenvalues → eq (C.7), Dirichlet → eq (C.8), transforms → Appendix C.8
- Ch 15: PCG theory → Appendix C.9
- toc.json, INDEX.md, CHANGELOG.md updated

## 2026-03-08 (session 13b) — Complete Part III Rewrite: Ch 12–15 (formerly 12–17)

### Motivation
Bidomain chapter audit scored Part III at 1.9/5.0 (D+) with 30 issues. Two critical failures: (1) solver architecture described (FGMRES+AMG+SDIRK2) does not exist in the code — Engine V1 uses decoupled N×N SPD solves; (2) Feynman style abandoned after Ch 12 — no ELI5, no worked examples, no diagrams in Ch 13–17.

### Structural Changes
- **Restructured from 6 chapters to 4**: old Ch 12–17 → new Ch 12–15
- **Deleted**: Ch 16 (Schur/AMG/FGMRES — nonexistent architecture), Ch 17 (vaporware roadmap)
- **Merged**: old Ch 15 (parabolic solvers) + useful parts of Ch 16 (null-space) + Ch 17 (roadmap) → new Ch 15

### Ch 12 Fixes
- §12.7 (monodomain reduction) and §12.8 (fiber tensors) converted to **sidebars** (insight boxes) — stops momentum kill
- §12.9 renamed §12.7, trimmed, forward ref fixed (Ch 16→Ch 15)
- Added **convention translation box** at chapter start: σ-form (literature) vs D-form (code)
- Fixed Engine V5.4 reference → Engine V1
- Removed redundant BC content (was duplicated in §13.4)

### Ch 13 "From Equations to Matrices" (rewritten)
- §13.1 rewritten with **road network analogy** (two overlaid networks)
- §13.2 block system: updated definitions to D-form ($1/\Delta t \cdot I - \theta \cdot L_i$), added **ELI5 for why L_i appears everywhere**
- §13.3 **NEW**: face-based stencil explanation (why ghost-node mirror breaks elliptic, how face-based fixes it)
- §13.4 FEM weak form moved to brief note (was full derivation blocking flow)
- §13.5 **worked example**: 5-node cable with numerical L_i, L_e, A_para, A_ellip values
- Removed 180-line §13.4 BC subsections (was near-verbatim repeat of §12.9)
- Updated all engine boxes to Engine V1

### Ch 14 "Solving the Coupled System" (NEW core chapter)
- §14.1 **"Why Explicit Fails"** moved to opening (was §14.4) — puppet-on-strings analogy
- §14.2 **Algorithm 14.1** — boxed Decoupled Gauss-Seidel Bidomain Step (the missing algorithm)
- §14.3 **Four more strategies**: Semi-Implicit, Jacobi, IMEX-SBDF2, Explicit RKC — each as modification of Algorithm 14.1
- §14.4 **Comparison table**: all 5 strategies with cost, stability, temporal order, best-for
- §14.5 **Worked example**: advance 5-node cable one step with GS strategy
- §14.6 Operator splitting (Godunov/Strang) with bidomain splitting error insight
- Monolithic alternative moved to **sidebar** (brief, for literature context)
- **Deleted**: SDIRK2 section (not in code), monolithic FGMRES content (not in code)

### Ch 15 "Linear Solvers and Implementation" (merged from old 15+16+17)
- §15.1 Parabolic sub-problem: brief cross-ref to Ch 11 (identical to monodomain)
- §15.2 Elliptic sub-problem: **null-space handling** (pinning + spectral zero-mode), **pinning = choosing sea level** analogy
- §15.3 **Three-tier automatic solver selection** (spectral → PCG+spectral → PCG) — **NEW**, matches Engine V1 auto-selection logic
- §15.4 **Engine V1 architecture**: real class names, file paths, factory strings — replaces vaporware Ch 17
- Deleted: Schur complement, AMG library table, FGMRES, block LDU, all nonexistent architecture

### Cross-Cutting Fixes
- Fixed 5+ cross-reference errors (Ch 14→13, Ch 16→15, nonexistent §7.10)
- Updated Table of Contents for Part III (4 chapters, new titles)
- Fixed Ch 18 intro reference (Ch 13–16 → Ch 8–15)
- All equation numbers within new chapters follow Ch.N convention
- Formulation B (D-form) used consistently in all discrete equations

### Audit Issues Addressed
- F1 (narrative arc): ✅ — clear 4-chapter story: equations → matrices → algorithm → solvers+code
- F2 (missing algorithm): ✅ — Algorithm 14.1 boxed procedure
- F3 (Feynman style): ✅ — ELI5 analogies in every section (puppet, road network, rope untangling, sea level, tool selection)
- F4 (difficulty spike): ✅ — weak form moved to brief note, ELI5 bridge added
- F5 (Ch 14 catalogue): ✅ — restructured as variations of one base algorithm
- F6 (Ch 16 jargon wall): ✅ — deleted; useful null-space content moved to §15.2
- F7 (no worked examples): ✅ — worked examples in §13.5 and §14.5
- F8 (no diagrams): ✅ — SVG block diagram retained, face-based stencil described
- F9 (BC repetition): ✅ — §13.4 BC section deleted, §12.7 is single source
- F10 (repeated "no time derivative"): ✅ — done in session 13a
- F11 (§12.7-12.8 momentum kill): ✅ — converted to sidebars
- F12 (Ch 15 redundant): ✅ — parabolic section is now a brief cross-ref
- C1 (monolithic vs decoupled mismatch): ✅ — decoupled approach is now the core
- C2 (σ vs D notation mismatch): ✅ — convention translation box at Ch 12 start
- C3 (nonexistent architecture): ✅ — Ch 17 deleted, replaced with real Engine V1 mapping
- M1 (face-based stencil): ✅ — new §13.3
- M2 (no boxed algorithm): ✅ — Algorithm 14.1
- M3 (saddle-point unexplained): ✅ — monolithic approach moved to sidebar, not central
- M4 (IMEX wrong scaling): ✅ — SBDF2 in D-form with correct coefficients
- M5 (three-tier solver): ✅ — new §15.3
- M6 (no full worked example): ✅ — §13.5 and §14.5
- M7 (BoundarySpec logic): ✅ — described in §15.3
- Mo1 (cross-ref errors): ✅ — all fixed
- Mo2 (SDIRK2 dead content): ✅ — removed
- Mo3 (wrong codebase): ✅ — all Engine V1
- Mo4 (φ_i substitution skipped): ✅ — done in session 13a
- Mo5 (conservation addition skipped): ✅ — done in session 13a
- Mo6 (null-space equal weight): ✅ — only pinning and spectral zero-mode (used by code)
- Mo7 (runtime claim): ✅ — removed unjustified claim
- m1 (K_i vs L_i): ✅ — consistently L_i in Ch 13–15
- m2 (equation numbering): ✅ — equations numbered within correct chapters
- m3 (block table mixing): ✅ — clean separation in new table
- m4 (Ch 10→11 ref): ✅ — fixed
- m5 (D_eff scalar/tensor): ✅ — simplified in sidebar

## 2026-03-08 (session 13a) — Rewrite §12.5–12.6: Foundation Fix for Bidomain Understanding

### Motivation
Audit identified two critical failures: (1) the bidomain equations were derived correctly but each term's physical meaning was never explained after the φ_i→Vm+φ_e substitution; (2) the "how to solve coupled parabolic+elliptic equations" question was never posed or answered.

### Changes to §12.5 "Building the Bidomain System"
- **Show all intermediate algebra explicitly**: the φ_i = Vm + φ_e substitution now shows ∇φ_i = ∇Vm + ∇φ_e and the linearity-of-divergence step (was skipped)
- **Elliptic derivation**: conservation identity → Ohm's law → φ_i substitution → collect φ_e terms (was a single jump)
- **Term-by-term physical dictionary**: each term in (12.4) and (12.5) labeled with underbrace and explained physically:
  - ∇·(D_i∇Vm) = "propagation" / "source"
  - ∇·(D_i∇φ_e) = "coupling"
  - ∇·((D_i+D_e)∇φ_e) = "response"
- **Steady-state heat analogy** (insight box): elliptic eq = steady-state temperature with Vm as heat source
- **"Why no time derivative?"** (warning box): no extracellular capacitor → no charge storage → no ∂φ_e/∂t
- **Numerical worked example**: Clerc 1976 parameters, wavefront estimates showing propagation ~6.8 mV/ms, coupling ~5%, elliptic balance
- **"Coupling is small but essential"** insight box: 5% correction → qualitatively different physics

### Changes to §12.6 "The Parabolic–Elliptic Couple" (retitled "A New Kind of Problem")
- **Explicit coupled-ODE comparison**: "you know dx/dt=f, dy/dt=g — bidomain is harder because the second equation is 0=g, a constraint"
- **Heater-and-AC analogy** for bidirectional coupling
- **Three-strategy preview table**: solve together (monolithic), advance-then-equilibrate (decoupled), advance-explicitly (semi-implicit) — maps to actual Engine_V1 strategies
- **"No memory" insight**: elliptic depends only on current Vm, enabling decoupled solvers
- **Removed**: 4× repetition of "no time derivative" → consolidated into one clear warning box in §12.5
- **Removed**: the redundant "why the elliptic equation cannot be marched in time" insight box (same content now in §12.5 warning box)
- **Added**: forward pointer to Ch 14 for detailed strategy development

### Net effect
- §12.5: ~50 lines → ~150 lines (all intermediate steps + physical dictionary + worked example)
- §12.6: ~60 lines → ~100 lines (coupled-system framing + strategy preview)
- Total chapter grew by ~250 lines but delivers fundamentally deeper understanding

---

## 2026-03-07 (session 12) — Major Structural Restructuring: Mirror Monodomain in Bidomain

### Motivation
The bidomain section (Part III) didn't mirror Part II's clean pedagogical flow: **Equation → Spatial Discretization → Splitting → Solvers**. This restructuring aligns both parts, splits solvers by type, and adds boundary condition introductions.

### Structural changes

**Part II — Monodomain (was Ch 7–10, now Ch 7–11)**
- **Ch 7 split**: Sections 7.1–7.3 stay in Ch 7 (PDE intro + conductivity). New 7.4 "Boundary Conditions: What Happens at the Edge?" added with swimming pool analogy, Neumann/Dirichlet/Robin types, forward pointers.
- **New Ch 8 "Spatial Discretization"**: Old sections 7.4–7.10 extracted and renumbered to 8.1–8.7 (incl. subsections 8.7.1–8.7.4).
- **Ch 8→9**: "Operator Splitting: Divide and Conquer" renumbered.
- **Ch 9→10**: "Explicit Diffusion Solvers" renumbered.
- **Ch 10→11**: "Implicit Diffusion Solvers" renumbered.

**Part III — Bidomain (was Ch 11–16, now Ch 12–17)**
- **Ch 11+12 merged → Ch 12 "The Bidomain Model"**: Old Ch 11 (Why Bidomain) and Ch 12 (Bidomain Equations) merged into single chapter. Old 11.1–11.3 → 12.1–12.3, old 12.1–12.4 → 12.4–12.7, old 12.5 → 12.8. New 12.9 "Boundary Conditions: Two Domains, Two Choices" added with three canonical scenarios (isolated/grounded bath/current injection).
- **BC formulation moved**: Old 12.6 (BC formulation detail) → new Section 13.4 in Ch 13 "Spatial Discretization", with subsections 13.4.1–13.4.4.
- **Ch 15 split into two chapters**:
  - **New Ch 15 "Parabolic Solvers: The A₁₁ Block"**: PCG (15.1), Chebyshev (15.2), Spectral (15.3) with subsections 15.3.1–15.3.4.
  - **New Ch 16 "Elliptic Solvers: The Schur Complement"**: Saddle-point (16.1), Block preconditioners (16.2), AMG (16.3), Krylov (16.4), Null-space (16.5).
- **Ch 16→17**: "Implementation Roadmap" renumbered.

**Part IV — LBM (was Ch 17–19, now Ch 18–20)**
- Ch 17→18, Ch 18→19, Ch 19→20. All equations and sections renumbered.

### New content written
- **Section 7.4** "Boundary Conditions: What Happens at the Edge?" (~48 lines): Swimming pool analogy, three BC types, why Neumann is default, forward pointers to Ch 8 and Ch 12.
- **Ch 8 header + intro**: Spatial Discretization chapter with intro paragraph.
- **Section 12.9** "Boundary Conditions: Two Domains, Two Choices" (~47 lines): Three canonical scenarios (isolated/grounded bath/current injection), null-space insight box, forward pointers to Ch 13 and Ch 16.
- **Ch 15 header + intro**: Parabolic Solvers chapter with intro about SPD A₁₁ block.
- **Ch 16 header + intro**: Elliptic Solvers chapter with intro about saddle-point systems.

### Equation renumbering (143 equations, all unique)
- (7.4)–(7.9) → (8.1)–(8.6), (8.1) → (9.1)
- (9.x) → (10.x), (10.x) → (11.x)
- (11.1)–(11.2) → (12.1)–(12.2), (12.1)–(12.4) → (12.3)–(12.6)
- (12.5)–(12.7) → (13.3)–(13.5)
- (17.x) → (18.x), (18.x) → (19.x), (19.x) → (20.x)

### Cross-references updated
- All "Section X.Y", "Chapter X", "equation (X.Y)" references updated throughout
- Forward pointers in new BC sections verified
- Figure 14.1 → Figure 13.1

### Verification
- Div balance: 491/491
- Equation labels: 143 total, no duplicates
- Chapters: 1–20 + A + B + References, no duplicates
- Section continuity: verified for all 20 chapters (no gaps)
- Website build: 29 fragments, 5 TOC entries
- Standalone build: 812 KB, 29 chapters embedded
- Total lines: ~12,262 (was ~12,124)

### Implementation
- `restructure.py`: Phase 2 structural surgery (split/merge/move operations)
- `renumber_v3.py`: Phase 3 surgical renumbering with title-based chapter matching to avoid collisions

---

## 2026-03-07 (session 11) — Bidomain Section Restructuring

### New sections added
- **A.8 "From Physical Space to Transform Space"** (~60 lines): Orchestra analogy, heat equation in Fourier space, FFT. Equations (A.7).
- **A.9 "Cosine and Sine Transforms"** (~130 lines): Guitar string analogy, DCT→Neumann, DST→Dirichlet, eigenvalue formulas (A.8)–(A.10), 2D extension table, when DCT/DST can combine, 4-point worked example.
- **A.10 "Iterative Methods and the Conjugate Gradient"** (~140 lines): Energy minimization (A.11), PCG pseudocode, convergence bound (A.12), 5-node worked example, warm-starting.
- **7.10 "Boundary Conditions for the Monodomain"** (~190 lines): Neumann/Dirichlet/Robin BC types with physical meaning, FDM/FEM/FVM implementation, ghost-node mirror (7.6), modified stiffness matrix (7.7), Dirichlet row elimination (7.8), mixed BC worked example (7.9).
- **12.6 "Boundary Conditions for the Bidomain"** (~192 lines): Three canonical scenarios (isolated/grounded bath/current injection), equations (12.5)–(12.7), 10×10 block system worked example, null-space pinning.

### Expanded sections
- **15.3–15.8** (was 15.3–15.5, 77 lines → ~310 lines): New 15.3 PCG (warm-starting detail, convergence criterion), 15.4 Chebyshev (Gershgorin bounds, PCG vs Chebyshev table), 15.5 Spectral Solvers with 4 subsections (DCT/Neumann, DST/Dirichlet, mixed DCT+DST, FFT/periodic, comparison table), equation (15.1). Old 15.3→15.6 AMG, 15.4→15.7 Krylov, 15.5→15.8 Null-Space (expanded with DCT handling).
- **12.3 "The Parabolic–Elliptic Couple"** (~40 lines added): Scaffolding for Part II→III transition, rubber sheet analogy for elliptic equations, insight box "Why the elliptic equation cannot be marched in time," forward pointer to splitting.
- **14.1 "Operator Splitting Applied to Bidomain"** (~30 lines added): Clarification that splitting error is reaction↔diffusion (not parabolic↔elliptic), insight box contrasting monodomain vs bidomain splitting.

### Verification
- Div balance: 486/486
- Equation labels: 143 total, no duplicates
- New equations: (A.7)–(A.12), (7.6)–(7.9), (12.5)–(12.7), (15.1)
- Website build: 28 fragments, 5 TOC entries
- Standalone build: 803 KB, 28 chapters embedded
- Total lines: ~12,124 (was ~11,043)

---

## 2026-02-27 (session 10, continued — part 2)

### Appendix B — B.7 rewrite + new B.9, expanded to 13 sections (B.1–B.13)
Based on audit of Engine V5.4 PyTorch usage patterns against Appendix B coverage:
- **B.7 "Sparse Matrices" rewritten**: Cleaner narrative structure, COO format explanation with coalescing worked example, sparse tensor inspection (`.indices()`, `.values()`, `.is_sparse`, `._nnz()`), numbered 5-operation catalogue (sparse×dense via `torch.sparse.mm`, scalar×sparse, sparse+sparse, sparse−sparse for implicit time-stepping, dense conversion), `speye` helper, 5-point Laplacian assembly, Engine V5.4 insight box
- **B.9 "Advanced Operations for Scientific Computing" (NEW)**: Eight subsections bridging "PyTorch as NumPy replacement" to "PyTorch as simulation backend":
  - `torch.where` with TTP06 α_h/β_h gating kinetics example (both-branches-evaluated warning)
  - `torch.roll` with D2Q5 LBM streaming (5-line GPU streaming step)
  - `torch.einsum` with FEM batched outer products (`'ei,ej->eij'`)
  - `scatter_add_` with element-to-global FEM assembly + Gershgorin bounds
  - In-place solver workspaces: simplified PCG class (`.copy_()`, `.add_(p, alpha=α)`, `.sub_()`, `.mul_().add_()`)
  - NumPy interop: `.detach().cpu().numpy()` canonical chain
  - `torch.meshgrid` with `indexing='ij'` (always-use-ij warning)
  - `torch.fft` (fft/ifft/fft2/fftfreq) for spectral solvers
- **Renumbered**: Old B.9 (Autograd) → B.10, B.10 (Neural Networks) → B.11, B.11 (Engine V5.4) → B.12, B.12 (ML) → B.13
- **Cross-references updated**: Mapping table "B.1–B.10" → "B.1–B.11", B.9→B.10 in table, B.13 neural network ref B.10→B.11
- **Chapter intro updated**: Added "and the intermediate patterns that bridge them to production simulation code"
- Website and standalone rebuilt (appendix-b.html = 1,165 lines, standalone = 724 KB)

### Navigation bug fixes (app.js + standalone)
- Fixed chapter number extraction regex: `/(\d+|A)/` → `/(\d+|[A-Z])(?:\s|$)/` (supports all appendix letters)
- Fixed section anchor regex: `/(\d+\.\d+|A\.\d+)/` → `/(\d+\.\d+|[A-Z]\.\d+)/` (supports B.x sections)
- Added conditional rendering for empty shortNum (References entry no longer shows stray ".")
- Applied same three fixes to standalone HTML's embedded JavaScript (rebuild_standalone.py only updates data, not JS)

## 2026-02-27 (session 10, continued)

### Appendix B — Introduction to PyTorch (v2 rewrite, 730 lines, B.1–B.12)
Rewrote as ground-up tutorial ("Intro to PyTorch book" style) instead of Engine V5.4 feature tour:
- **B.1 "Installation and First Tensor"**: import torch, scalar/vector/matrix/3D, .shape, .item()
- **B.2 "Arithmetic"**: Scalar/vector/matrix ops, element-wise vs dot product vs matrix multiply (@)
- **B.3 "Tensors and Shapes"**: Dimension table, reshaping (view/reshape), slicing, broadcasting rules
- **B.4 "Data Types and Precision"**: float32 vs float64, why EP needs float64 (cancellation argument)
- **B.5 "Common Operations Cheat Sheet"**: Creating tensors, reductions, clamp/where, stack/cat, in-place ops, math functions
- **B.6 "Linear Algebra"**: linalg.solve, eig, norm, det, inv; worked Ax=b example; solve-vs-inv insight
- **B.7 "Sparse Matrices"**: COO format, sparse_coo_tensor, 5-pt Laplacian, sparse.mm, speye, scalar-sparse
- **B.8 "GPU Acceleration"**: torch.device, .to(device), write-once-run-anywhere, transfer minimization
- **B.9 "Automatic Differentiation"**: Computation graph, requires_grad, backward, torch.no_grad()
- **B.10 "Neural Networks"**: nn.Module, nn.Sequential, training loop, I_Na surrogate worked example
- **B.11 "Connecting to Engine V5.4"**: Summary table mapping concepts → engine files. Backend + LUT boxes
- **B.12 "ML Meets Cardiac Modeling"**: PINNs, neural ODE surrogates, ML classification (survey only)
- Fixed build_website.py appendix slug and subsection anchor regexes for multi-letter appendices

### Chapter 18 — Ω convention fix, physical grounding, χ fix
- **Ω convention change**: Removed Δt from Ω^NR and Ω^R definitions; now rates consistent with eq 17.11. Update: f_i* = f_i + Δt(Ω^NR + Ω^R). Affected: eqs 18.3, 18.8, 18.11
- **χ fix**: Substep 1 ODE now has χ on left side matching eq (7.1)
- **Section 18.2 restructured**: Foregrounds Ω^NR/Ω^R decomposition (diffusion vs reaction)
- **Physical motivation**: Extensive explanation of why Ω^NR = diffusion (flux relaxation → ∇·(D∇V_m)) and Ω^R = reaction (isotropic source injection → ionic currents)

### Chapter 18.5 — Expanded boundary conditions (42 → 252 lines)
- **18.5.1 "Full-Way Bounce-Back"**: Original content refined, D2Q9 opposite-pair table added
- **18.5.2 "Half-Way Bounce-Back"**: Second-order accuracy, wall at cell face, eq (18.10b)
- **18.5.3 "Identifying Boundary Nodes"**: Binary tissue mask, precomputed boundary links, pseudocode
- **18.5.4 "Handling Irregular Tissue Geometries"**: Staircase adequacy for EP, interpolated bounce-back for sub-grid accuracy
- **18.5.5 "Comparison with Classical BC Methods"**: LBM vs FDM vs FEM comparison table, Engine V5.4 box

### Presentation — LBM_Monodomain_D2Q9.pptx
- 8-slide D2Q9 summary (18.1–18.6 + summary), pure white background, MathJax-rendered equations
- Created with pptxgenjs + mathjax-node

---

## 2026-02-27 (session 10)

### Chapter 18 — Complete rewrite following Campos et al. structure

**Structural overhaul**: Rewrote Chapter 18 from 282 lines to 721 lines, following the structure of the Campos et al. (2015) LBM-for-cardiac-electrophysiology paper while building on all notation and machinery established in Chapter 17.

- **New 18.1 "The Monodomain Equation as a Target PDE"**: Restates eq 7.1 using established notation ($V_m$, $\mathbf{w}$, $\mathbf{D}$), introduces diffusion tensor decomposition (eq 18.1) with fiber direction, $D_l$, $D_t$
- **New 18.2 "Operator Splitting: Reaction Then Diffusion"**: Full splitting workflow — ODE substep with Rush-Larsen (cross-ref Ch 8), then LBM diffusion substep. Source term $S$ (eq 18.2) properly motivated by splitting
- **Revised 18.3 "BGK Collision with Source Term"**: Same collision equation (now eq 18.3) but better connected to Ch 17 quadrature ($f_i^{eq} = w_i V_m$ from eq 17.27)
- **Expanded 18.4 "MRT for Anisotropic Tissue"**: Four sub-sections:
  - Why BGK fails for anisotropy (recap from 17.5)
  - **Tensor τ-D derivation** (eqs 18.4–18.6): $\tau_{\alpha\beta} = \frac{\Delta t}{2}\delta_{\alpha\beta} + \frac{3D_{\alpha\beta}\Delta t}{\Delta x^2}$
  - Off-diagonal problem: two approaches (rotated basis vs modified equilibrium, eq 18.7)
  - **Ghost-moment rates and Gram-Schmidt**: explains how ghost rows of M are constructed via Gram-Schmidt orthogonalization, why $s_3 = s_4 = 1.0$ is standard, Chapman-Enskog order analysis
  - Full MRT collision with source (eq 18.8, boxed), relaxation matrix (eq 18.9)
  - Worked example with $D_l = 0.001$, $D_t = 0.00025$, $\theta = 0$
- **New 18.5 "Bounce-Back Boundary Conditions"**: Bounce-back rule (eq 18.10), no-flux interpretation, D2Q5 index pairings, corner node handling
- **Revised 18.6 "Complete LBM-EP Algorithm"**: 6-stage boxed algorithm (eq 18.11, now includes bounce-back step), swap-streaming optimization discussion
- **Revised 18.7 "Implementation"**: SoA memory layout, swap-streaming, sparse/indirect addressing, Engine V5.4 insight box preserved
- **18.8 "LBM vs Classical Methods"**: Comparison table kept and enhanced

**Equation registry**: 11 equations (18.1–18.11), all labels unique, all cross-references valid. No broken refs from Ch 19+.

**Notation consistency**: All notation uses established conventions ($V_m$, $\mathbf{w}$, $\mathbf{D}$, $I_{\text{ion}}$, $\mathbf{e}_i$/$\mathbf{c}_i$ distinction from Ch 17).

---

## 2026-02-27 (session 9)

### Chapter 17.4 — Complete "Quadrature First" rewrite

**Pedagogical overhaul**: Replaced the old "three constraints → solve for weights" framing with "Quadrature First" — weights $w_i$ and velocities $\mathbf{c}_i$ come FROM Gauss-Hermite quadrature applied to the Gaussian, not from solving constraint equations.

- **Removed eqs 17.18–17.20** (the three constraint equations). They were consequences of quadrature, not sources — removing them fixes the causal direction
- **Renumbered all equations**: 17.21→17.18 through 17.44→17.41 (−3 shift to close the gap). Chapter now has eqs 17.1–17.41 + 17.7a
- **New subsection 1: "From Integrals to Sums"** — recalls the three integrals from 17.1, introduces quadrature formula (eq 17.18): $\int p(\mathbf{c}) G(\mathbf{c})\,d\mathbf{c} \approx \sum w_i p(\mathbf{c}_i)$. Explains $p(\mathbf{c})$ as a placeholder for whatever you're integrating
- **New subsection 2: "Lattice Velocities vs. Physical Velocities"** — distinguishes $\mathbf{e}_i$ (integer lattice vectors) from $\mathbf{c}_i = \mathbf{e}_i \cdot \Delta x/\Delta t$ (physical velocities with units)
- **New insight box: "How Can Five Points Replace an Entire Plane?"** — Gaussian decay means most of velocity space contributes negligibly
- **New subsection 3: "The Discrete Equilibrium"** — eq 17.27 ($f_i^{\text{eq}} = w_i\phi$) for diffusion, with physical explanation of why each direction carries fraction $w_i$
- **New subsection 4: "Gauss-Hermite Quadrature"** — formal name for the technique, "Nodes and Weights Are a Married Pair" insight box, diffusion (2nd-order) vs fluid flow (4th-order) exactness
- **Replaced "Closing the Loop"** with "The Navier-Stokes Equilibrium" (eq 17.28 only)
- **Updated Lattice Requirements** cross-references to new equation numbers
- Subsections 5–10 (Naming Convention through Speed of Sound) unchanged in content, only equation numbers shifted

### Website and formatting fixes
- Fixed dark mode contrast for inline `background:#f8f8f8` elements
- Converted three discretization choices (17.3) from inline `<strong>1.</strong>` to proper `<ol><li>` list
- Rebuilt standalone website (629 KB) and both PDFs (6.7 MB + 2.8 MB)

---

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
