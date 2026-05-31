---
paper: eshaghi_2025_nows
title: "NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers"
authors: "Eshaghi MS, Anitescu C, Valizadeh N, Wang Y, Zhuang X, Rabczuk T"
year: 2025
journal: "arXiv preprint (Nov 2025)"
doi: "arxiv:2511.02481"
pmid: ""
pdf: ../papers/nows_2025.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **The clinically-defensible deployment pattern**: neural operator provides an initial guess; classical Krylov solver (CG, GMRES) refines to tolerance. **"Preserves the stability and convergence guarantees of the underlying numerical algorithms"** — the key safety property for bidomain deployment.
- **Up to 90% wall-clock reduction** vs cold-started classical solvers. Concrete, published, credible number.
- **Works with CG and GMRES explicitly** — CG is what Bidomain V1 uses for the elliptic solve (PCG). Direct drop-in.
- **Addresses the "surrogates are unreliable out-of-distribution" critique head-on** — OOD inputs just mean the warm-start is worse, so the classical solver takes more iterations. **Worst case: same wall-time as cold-started classical solver. No catastrophic failure mode.**
- **Integrates with existing discretizations** — doesn't require re-meshing or architectural changes to the classical solver beyond accepting the initial guess.
- **Generic pattern, not tied to a specific surrogate architecture** — could wrap our dual-tower CNO, a UGrid output, a PDE-Transformer output, anything. NOWS is the *deployment architecture*, not a specific model.

## Method
- **Core idea**: a neural operator is trained to map (RHS, BCs, coefficients) → approximate solution. That approximate solution is fed as the initial guess `x₀` to a Krylov method solving `Ax = b`. The Krylov method then iteratively reduces the residual to a specified tolerance.
- **Why it works**: Krylov convergence depends on `‖x₀ − x_true‖`. A good warm-start drastically reduces this distance, so the Krylov method needs far fewer iterations to reach tolerance.
- **Training**: the neural operator is trained end-to-end on classical-solver ground-truth solutions. Loss is MSE on solution fields (or residual norm, depending on variant).
- **Deployment**: offline-trained operator is a drop-in `x₀ = NN(A, b, …)` call at the beginning of each classical solve. No runtime adaptation.
- **Fallback**: if NN's output is bad (worst case: random noise), Krylov methods still converge, just with more iterations. Never produces wrong answers — only faster or slower correct answers.
- **BC handling**: inherits whatever the underlying classical solver handles. The NN just provides a starting point; the classical method's BC machinery does the actual enforcement.
- **Iteration-count reduction + wall-clock reduction** both reported. Paper reports specific numbers (up to 90%).

## Connections to Our Models

### Relevant Engine Components
**The deployment architecture for the whole hybrid bidomain surrogate.** Whatever we build for the neural elliptic step (dual-tower CNO, PDE-Transformer, UGrid, etc.), we wrap it with NOWS: `NN produces x₀ → Bidomain V1's existing PCG refines to tolerance → final φ_e`. This preserves all of Bidomain V1's existing safety properties (convergence, residual tolerance, numerical stability) and only accelerates the solve.

### Agreements
- **Safety-first deployment story** exactly matches our preconditioner-first analysis. NOWS generalizes: any NN that produces a reasonable approximation can serve as a warm-start, regardless of whether it's preconditioner-structured or full-inverse-structured.
- **PCG-compatible** — directly usable with Bidomain V1's existing solver infrastructure.
- **"Up to 90% runtime reduction"** is the kind of credible, bounded claim we'd want to cite.
- **Graceful OOD degradation** — for a clinical application where patient-specific tissue configs will always be OOD vs training, this is essential.
- **Architecture-agnostic** — frees us to pick the surrogate backbone separately from the deployment story.

### Disagreements or Gaps
- **Very new paper (Nov 2025)** — implementation maturity is early, code repo not mentioned in abstract.
- **PDE classes not specified in abstract** — need PDF read to know whether they've tested anything close to bidomain elliptic (anisotropic Poisson with mixed BCs).
- **Benchmark baselines unclear** — "classical iterative solvers" is vague. Is the baseline cold-start CG? Cold-start GMRES? Preconditioned variants? The 90% reduction is relative to something specific.
- **Training-data-driven**: the NN requires supervised training on classical solver outputs. Same data-cost issue as supervised FNO/CNO/PDE-Transformer.
- **Single-field focus implied** — paper is about warm-starting Ax=b for linear systems. Our bidomain has coupled V_m + φ_e; the NN would either warm-start just the elliptic step or both fields in sequence. Latter is straightforward; former is essentially what we're already planning.

### Actionable Insights
- **HIGH — Adopt NOWS as the explicit deployment architecture** for the hybrid bidomain surrogate. Phrase the whole project as "neural warm-start for Bidomain V1's PCG solver" in any write-up. This framing:
  1. Preserves existing V1 safety properties (convergence, tolerance).
  2. Gives a clean success metric (iterations saved, wall-clock reduced).
  3. Makes the clinical-deployment story trivial — nothing changes about V1's correctness, only speed.
- **HIGH — Target 50-90% PCG iteration reduction** as our primary speedup metric. Beats abstract "our NN predicts X MSE" as a deliverable.
- **HIGH — Update Bidomain V1's `DecoupledBidomainDiffusionSolver`** to accept an optional initial-guess callback that the NN provides. Minimal engineering change; preserves all existing functionality.
- **MEDIUM — Cite NOWS in methods** as the deployment pattern. Gives us an explicit reference rather than reinventing the framing.
- **MEDIUM — Benchmark against NOWS's numbers**: 90% reduction on their test cases. If we hit ≥50% on bidomain elliptic, we're competitive; if ≥90%, we're state-of-the-art.
- **LOW — Wait for code release.** If the authors release code, adopt it; if not, the pattern is simple enough to reimplement.

## Limitations / Caveats
- **PDE classes not specified** in abstract — may not include our anisotropic mixed-BC Poisson specifically.
- **"Up to 90%"** is a best-case claim — average-case numbers may be lower. Need PDF for distribution of speedup across problems.
- **Very recent paper (Nov 2025)** — early in its maturity curve, code and reproducibility not yet established.
- **Supervised training required** — data-generation cost still applies.
- **Architecture-agnostic wrapper** — doesn't replace the need for a good surrogate backbone. Bad NN + NOWS = marginally-better-than-cold-start but not revolutionary.
- **BC handling inherited from classical solver** — the NN doesn't need to get BCs perfect; the PCG refinement handles that. This is actually a FEATURE: BC correctness falls back to the classical solver, reducing architecture burden on the NN.
- **No convergence-rate proof for the warm-start speedup** — the 90% is empirical, not proven. Theoretical justification would be: `‖x₀ − x*‖` is smaller for NN warm-start, and Krylov convergence is linear in log(`‖x₀ − x*‖`/tolerance). So speedup is essentially log-scale with warm-start quality.
