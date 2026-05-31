---
paper: wu_2024_transolver
title: "Transolver: A Fast Transformer Solver for PDEs on General Geometries"
authors: "Wu H, Luo H, Wang H, Wang J, Long M"
year: 2024
journal: "ICML 2024 (Spotlight)"
doi: "arxiv:2402.02366"
pmid: ""
pdf: ../papers/transolver_2024.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Physics-Attention mechanism** — adaptively partitions the discretized domain into **learnable slices** of flexible shape, where mesh points in similar physical states are grouped into the same slice. Attention is then computed over slice-tokens rather than individual mesh points. Linear cost in sequence length (vs quadratic for full attention).
- **Designed for "general geometries"** — unstructured meshes, irregular domains, industrial CAD geometries. Not tied to regular Cartesian grids.
- **Linear computational complexity** in the number of discretization points. Scalable to large meshes (cardiac whole-heart mesh sizes, e.g., millions of points).
- **Validated on 6 standard PDE benchmarks + industrial applications** (cars, airfoils). Broad coverage.
- **ICML 2024 Spotlight**: strong peer review, high-trust paper.
- **Open-source code**: `github.com/thuml/Transolver`. THUML (Tsinghua Machine Learning) lab, actively maintained.

## Method
- **Physics-Attention core**: instead of attending over all N mesh points (O(N²)), group mesh points into M slices (M ≪ N), compute attention over slice-tokens (O(M²) or O(NM) depending on formulation).
- **Learnable slice assignment**: each mesh point is soft-assigned to one or more slices based on its learned physical state embedding. Points with similar physics → same slice. Makes the grouping data-driven rather than mesh-topology-driven.
- **Physics-aware tokens**: each slice is summarized into a token that captures the physical state of the points it contains. Attention computes correlations between these physics-aware tokens rather than between raw mesh points.
- **Decoder**: slice-level attention output is broadcast back to individual mesh points, preserving the full field output.
- **Training**: supervised on PDE solution pairs (input function, output function). Standard MSE or relative L² loss.
- **Geometry-general**: same architecture works on structured grids, unstructured meshes, point clouds. The slicing abstracts away the mesh topology.
- **BC handling**: not explicitly detailed in abstract. Likely via boundary-indicator features on mesh points. Need PDF read.

## Connections to Our Models

### Relevant Engine Components
**Phase-B architecture reference**, not Phase-A. Our Phase A target is Bidomain V1 on a structured 2D Cartesian grid — PDE-Transformer or CNO is a better fit. Transolver becomes relevant when/if we extend to:
- Patient-specific unstructured cardiac meshes (biventricular, whole-heart).
- Point-cloud-based representation for irregular tissue geometries.
- Scaling to very large meshes where even Swin attention is too expensive.

### Agreements
- **Linear-cost attention** is essential at scale. Same design principle as Swin (Poseidon, PDE-Transformer). Different implementation mechanism — physics-attention vs shifted-window — but convergent in goal.
- **Geometry-general** — if we ever want to publish across structured + unstructured validation, a single backbone is valuable.
- **ICML 2024 Spotlight + active maintenance** — high-trust, low-adoption-friction.
- **Open-source code** at `github.com/thuml/Transolver` — direct adoption path.

### Disagreements or Gaps
- **Unstructured geometry is overkill for Bidomain V1** — we have regular Cartesian grids. Transolver's geometry-generality is wasted overhead for Phase A.
- **Physics-attention's slicing mechanism is not obviously well-suited for elliptic problems.** The slice clustering groups points by physical state; for elliptic, all points matter globally (Green's function has global support). Slice-based attention may miss the long-range coupling that elliptic solutions require.
- **BC handling not explicit in abstract** — although general-geometry solvers typically need to handle BCs as features, not architecturally.
- **Benchmarks are primarily fluid/structural mechanics, not cardiac.** Transfer to cardiac EP is plausible but untested.
- **No multigrid structure** — unlike UGrid/MGCNN, Transolver doesn't exploit the multiscale hierarchy that's natural for elliptic PDEs. For the elliptic-only task UGrid is likely better.
- **Single-field focus in demonstrated benchmarks** — multi-field coupled problems (like our V_m + φ_e bidomain) not the primary showcase.

### Actionable Insights
- **LOW (Phase A) — Defer.** Not competitive with PDE-Transformer or UGrid for structured-grid bidomain.
- **MEDIUM (Phase B) — Revisit when scaling to unstructured meshes.** If we extend past Bidomain V1's Cartesian grid to biventricular anatomy, Transolver is the leading candidate for the mesh-general surrogate.
- **MEDIUM — Cross-verify linear-cost claim** on cardiac-sized domains (512² — 1024² structured grids, or equivalent unstructured). The constant factor matters.
- **LOW — Benchmark against PDE-Transformer on structured bidomain** as a sanity check that mesh-general doesn't hurt on the easy structured case.
- **LOW — Code at `thuml/Transolver`** is clean and well-documented. If Phase B arrives, this is where to start.

## Limitations / Caveats
- **Structured grid is Transolver's weakest case by design** — its strengths (unstructured mesh handling) don't apply there.
- **Slicing may miss elliptic's global coupling** — for sharp-wavefront V_m this is fine, for smooth φ_e this could be a weakness.
- **BC handling not specified** — likely via mesh-point features, but details matter for Neumann-dominant bidomain.
- **Fluid/structural benchmarks, not reaction-diffusion or cardiac** — transfer untested.
- **Multi-field coupled dynamics** — the dual-field bidomain use case is not the primary design target.
- **Linear cost claim has constants** that matter at specific mesh sizes — must benchmark on our problem specifically.
