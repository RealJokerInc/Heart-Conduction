# Boundary Conduction Speedup — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
CV ratio confirmed at 1.0714 (dx=0.025cm) converging toward theoretical 1.131. Isotropic case fully characterized: mechanism (8-link argument chain), mesh convergence, triangle merger wavefront, stencil comparison, and conductivity sweep all complete. Anisotropic boundary study is active — initial 2:1 anisotropy test shows sharper triangle (2.27cm edge lead vs 1.62cm isotropic), confirmed by eikonal prediction. Still need fiber-parallel vs perpendicular systematic study, 3D validation, and tissue thickness study.

## Next Step
Anisotropic boundary study: systematic fiber-parallel vs fiber-perpendicular analysis at the bath-coupled boundary. The initial 2:1 anisotropy experiment ran successfully but needs a full parameter sweep across fiber orientations.

## Thread

### 2026-03-13: Research restructured from domain-based to question-driven
The Research folder was reorganized from domain-based directories (Bidomain/, LBM/, FHD/) to question-driven folders (Q1 through Q7). Boundary conduction speedup became Q5. This was motivated by the workflow being question-driven: "does CV increase at boundaries?" rather than "what does the bidomain engine do?". Papers were re-filed by research question, INDEX.md created as the master question map.

### 2026-03-15: Mehrstellen 9-point stencil implementation for curved wavefront accuracy
Implemented the Mehrstellen (isotropic 9-point, O(h^4)) stencil to resolve whether the 5-point stencil's truncation error was contaminating the Kleber wavefront shape. This was needed because the boundary speedup produces curved wavefronts where stencil isotropy matters. Eight implementation steps, 16 tests passing. The stencil turned out to affect absolute CV (~4% lower, 47.1 vs 49.1 cm/s) but not the relative Kleber effect -- both stencils produce identical wavefront shapes within 0.05cm.

### 2026-03-15: Triangle merger experiment — the "merger" does not happen
Ran the full triangle merger pipeline (3 configs: monodomain Mehrstellen, bidomain 5pt, bidomain Mehrstellen) on a 50x8cm domain for 800ms. Key surprise: the triangular wavefront is the steady state, not a transient that merges. The edge-center lead saturates at 1.65-1.70cm by t~300-450ms and remains constant. The "triangle merger" terminology was misleading. The Kleber ratio of 1.131 describes the transient speedup during wavefront shape establishment; once the chevron forms, edge and center propagate at equal velocity. The effect is encoded in the accumulated lead distance, not in a persistent CV difference.

### 2026-03-15: Monodomain control produces perfectly flat wavefront
The monodomain Mehrstellen config on the same grid produced exactly 0.000cm deviation from flat. This definitively isolates the Kleber effect as a bidomain boundary coupling phenomenon (asymmetric BCs: Neumann intracellular + Dirichlet extracellular), not a numerical artifact of the stencil or grid.

### 2026-03-15: GPU 4x speedup confirmed for bidomain simulations
Bidomain triangle merger on GPU (RTX PRO 4500 Blackwell): 6.0ms per step vs 23.4ms on CPU, 4.0x speedup. Total pipeline dropped from ~75min to ~26min. This makes parameter sweeps practical.

### 2026-03-15 (approx): Conductivity sweep — edge lead scales with sqrt(D_eff)
Five configurations tested: 0.5x iso, 1x iso (baseline), 2x iso, 4x iso, and 4x sigma_i only. Scaling both sigma_i and sigma_e uniformly preserves the Kleber ratio (all at 1.131) with edge lead growing as sqrt(D_eff). The 4x-sigma_i-only config is the key result: increasing sigma_i to 6.96 while holding sigma_e at 6.25 raises the theoretical Kleber ratio from 1.131 to 1.454, and the edge lead jumps to 4.48cm (vs 1.45cm baseline). This confirms the boundary speedup is governed by the sigma_i/sigma_e ratio, not just absolute conductivity.

### 2026-03-15 (approx): Anisotropic test — 2:1 ratio produces sharper triangles
First anisotropic test with 2:1 conductivity ratio (longitudinal:transverse). Edge lead increased from 1.62cm (isotropic) to 2.27cm (anisotropic), a 40% increase. The sharper triangle is consistent with the eikonal prediction: lower transverse conductivity means the wavefront curves less easily, so the boundary speedup accumulates a larger geometric distortion. This is the precursor to the full anisotropic study.

### 2026-03-16: Research reorganized from Q-numbers to Active/Complete/Backlog
The Q-number naming (Q5_boundary_conduction_speedup) was replaced with status-based paths (Active/boundary_conduction_speedup). MASTER.md became the project dashboard. Experiments got their own directories inside engine folders with backlinks to the research question. The gap identified: experiments (scripts, parameters, outputs) had no standardized home -- they lived in engine test files with no connection to the research question that motivated them.

### 2026-03-16: Realized experiments need a home between "hypothesis" and "knowledge"
The research workflow goes Hypothesis -> Script -> Run -> Outputs -> Analysis -> Finding -> Knowledge. Each step was living in a different place with no links. Experiments directory structure was created inside engine folders, cross-linked to research questions via EXPERIMENT.md files. This solved the problem of the conductivity sweep and triangle merger results being detached from the boundary speedup question.

## Failed Approaches
- **"Triangle merger" framing** (2026-03-15) — failed because: the triangular wavefront does not merge. It IS the steady state. Edge-center lead saturates at 1.65-1.70cm and remains constant through 800ms. The experiment was designed expecting two triangular deformations from opposite edges to interact, but instead the chevron shape is simply the equilibrium between boundary speedup and wavefront curvature. The terminology was corrected but the experiment name was kept for historical continuity.
- **Late-time CV ratio as Kleber measurement** (2026-03-15) — failed because: at steady state, all y-rows advance at the same velocity (ratio ~1.0). The curvature-induced diffusive correction exactly compensates the boundary speedup. The Kleber ratio must be measured during the transient growth phase (t=25-200ms) before wavefront curvature develops, or from the accumulated edge lead distance. The existing phase 6C tests on the smaller 150x40 grid at dx=0.025cm are the correct measurement approach.
- **5-point stencil sufficiency for boundary effect** (2026-03-15) — not a hard failure but a resolved concern: the 5pt stencil overestimates absolute CV by ~4% compared to Mehrstellen at dx=0.05cm, but both produce identical Kleber wavefront shapes. The stencil choice affects absolute accuracy, not the relative boundary physics.

## Session Log
