---
paper: raonic_2023_cno
title: "Convolutional Neural Operators for robust and accurate learning of PDEs"
authors: "Raonic B, Molinaro R, De Ryck T, Rohner T, Bartolucci F, Alaifari R, Mishra S, de Bezenac E"
year: 2023
journal: "NeurIPS 2023"
doi: "arxiv:2302.01178"
pmid: ""
pdf: ../papers/cno_2023_raonic.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Rehabilitates CNNs as proper neural operators.** Before CNO, CNN-based PDE solvers were "believed to be inconsistent in function space" and largely dismissed in favor of FNO/DeepONet. CNO proves otherwise — it is discretization-invariant **by design**, preserving its continuous-function-space interpretation even under discretization.
- **Universality theorem for CNOs**: proves CNOs can approximate operators arising in PDEs to arbitrary accuracy. Brings CNNs to architectural parity with FNO and DeepONet on the theoretical foundations side.
- **Beats FNO on non-periodic benchmarks** across a diverse set of PDEs including multi-scale problems. The critical point for us: FNO's periodic-BC structural limitation is genuinely a performance hit, not just a theoretical nuisance, and CNO is the CNN-family alternative that works.
- **Open-source code** at `github.com/bogdanraonic3/ConvolutionalNeuralOperator`, plus a ETH Zurich fork at `github.com/camlab-ethz/ConvolutionalNeuralOperator`.
- **CamLab (Mishra group, ETH Zurich)**: same group as Poseidon. Consistent methodological lineage; high-trust.

## Method
- **Core insight**: a standard CNN acts *on sampled function values*, not on the underlying function. Under grid refinement, the same CNN weights applied to denser samples produce different "operators" — i.e., the CNN is discretization-dependent.
- **CNO's fix**: treat convolution layers as discretizations of continuous integral operators. Use appropriate interpolation / anti-aliasing filters to ensure the discrete CNN's behavior converges to a well-defined continuous operator as resolution increases.
- **Architecture**: U-Net-style encoder-bottleneck-decoder with convolutional blocks modified to be discretization-aware. Skip connections preserve high-resolution info.
- **Aliasing prevention**: explicit filter design before downsampling ensures the coarse representation is a consistent projection of the fine one. Standard CNNs skip this and introduce aliasing artifacts.
- **Training**: supervised on PDE solution pairs. Standard MSE loss.
- **Benchmarks**: diverse set including elliptic (Poisson, Helmholtz), parabolic (heat, Navier-Stokes-like), and multi-scale problems. Non-periodic BCs explicitly included.

## Connections to Our Models

### Relevant Engine Components
**Foundation paper for any CNN-based dual-tower design for bidomain.** If we go the "dual CNN towers with cross-attention" route (the current direction), each tower should be a CNO rather than a vanilla U-Net — the discretization-invariance guarantee comes at minimal extra cost and prevents grid-resolution brittleness.

### Agreements
- **CNN operating on structured grids is the right architecture class for our bidomain elliptic step.** CNO rehabilitates this choice with theoretical backing.
- **Non-periodic BCs handled correctly** — CNO's architecture doesn't assume periodicity (no Fourier layers).
- **Universality theorem** gives us a formal "this can represent what we need" argument for publications.
- **Discretization invariance**: if we later want to evaluate on a different grid resolution (coarser for draft runs, finer for validation), CNO preserves behavior — plain CNN would not.
- **Multi-scale handling**: CNO is demonstrated on multi-scale PDEs. Bidomain has multi-scale character (wavefront width ~1 mm, domain ~10 cm = 100× scale separation). Relevant.

### Disagreements or Gaps
- **Not multigrid-structured**: UGrid's V-cycle is arguably better for elliptic problems specifically (multigrid is the classical gold standard for elliptic). CNO is more general-purpose — trades elliptic specialization for flexibility.
- **No explicit convergence guarantees under iterative rollout**: like FNO and DeepONet, CNO is a direct approximation. Errors do not self-correct through a classical fallback.
- **Supervised training**: requires ground-truth PDE solutions. Same data-cost issue as most neural operators. Residual-loss unsupervised training (Greenfeld, UGrid) would be preferable.
- **BC injection mechanism underspecified in abstract**: need PDF read to confirm how Dirichlet / Neumann / mixed BCs are encoded in inputs. Likely: BC values as boundary-indicator channels, similar to Lan 2023.
- **Anisotropic tensor operators not emphasized**: bidomain's D_i + D_e is anisotropic. CNO demonstrations are on scalar-coefficient PDEs. Extension plausible but not demonstrated.

### Actionable Insights
- **HIGH — Use CNO backbone for each tower of the dual-tower design.** Gives us: (a) BC-correct inheritance (no FNO problem), (b) discretization invariance, (c) universality theorem for paper defensibility, (d) open-source code.
- **HIGH — Cite the universality theorem** in any write-up to justify CNN choice over FNO/DeepONet.
- **HIGH — Cross-check against PDE-Transformer**: PDE-Transformer's architecture is transformer-based rather than CNN-based. We have a clean A/B comparison available: CNO-based dual-tower vs PDE-Transformer-style channels-as-tokens. Benchmark both on our bidomain task and pick the winner.
- **MEDIUM — Start from camlab-ethz fork** of the code (Mishra group's maintained branch, also used in Poseidon/scOT).
- **MEDIUM — Multi-scale handling is a genuine strength for cardiac**: the 100× scale separation between wavefront width and domain size is exactly the regime where CNO is validated.
- **LOW — Expect to re-derive convergence on anisotropic operators** if we need formal guarantees for our tensor-D bidomain case.

## Limitations / Caveats
- **Not a multigrid-native architecture**: for the elliptic step specifically, UGrid's V-cycle may outperform. Empirical question.
- **Direct approximation, no iterative refinement**: without a PCG wrap, errors don't self-correct.
- **BC injection details in paper, not abstract**: need PDF read to verify Neumann / mixed BC handling.
- **Supervised training required** — data-generation cost, like all supervised neural operators.
- **Anisotropic tensor extension untested** — plausible but not empirically validated.
- **Universality theorem gives existence, not rate of approximation**: we know CNO can approximate our operator, but not how much capacity is needed.
