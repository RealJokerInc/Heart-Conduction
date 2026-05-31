---
paper: mccabe_2023_autoregressive_stability
title: "Towards Stability of Autoregressive Neural Operators"
authors: "McCabe M, Harrington P, Subramanian S, Brown J"
year: 2023
journal: "Transactions on Machine Learning Research (TMLR), Nov 2023"
doi: "arxiv:2306.10619"
pmid: ""
pdf: ../papers/mccabe_2023_autoregressive_stability.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Formal characterization of autoregressive failure modes** in neural PDE operators. Identifies specific architectural operations (normalization layers, nonlinear activations without bounded gradients, feedback loops) that **induce uncontrolled error growth** during long rollouts.
- **Architectural surgery fixes stability** without retraining from scratch or increasing parameter/memory budget. Modifications target the specific identified failure modes.
- **Demonstrated on three distinct PDE systems** — Navier-Stokes, rotating shallow water, high-resolution global weather. The failure modes + fixes generalize across PDE families.
- **"Significantly lower errors for long-term forecasts"** — stability is preserved without qualitative divergence far past the rollout horizons at which alternatives fail.
- **Complements PDE-Refiner** (Lippe 2023): Lippe attacks error growth via refinement-at-each-step (adds compute); McCabe attacks it via architecture-level stability (no inference-time cost).

## Method
- **Analysis**: formal treatment of how operations within neural operators amplify or damp errors during autoregressive iteration. Specific culprits: (a) un-normalized nonlinearities that can amplify magnitudes, (b) certain normalization choices (BatchNorm over feature dims), (c) feedback connections whose gradient structure doesn't contract.
- **Fix categories**:
  1. Replace amplifying nonlinearities with bounded alternatives where stability matters.
  2. Careful normalization placement — often removing or replacing BN/LN at specific layers.
  3. Structural damping on feedback paths that would otherwise amplify.
- **PDE benchmarks**: Navier-Stokes (fluid flow, standard benchmark), rotating shallow water (atmospheric), high-res global weather (real-world challenging).
- **Output**: stable rollouts over horizons 2–10× longer than baselines before divergence.

## Connections to Our Models

### Relevant Engine Components
Directly relevant to the **dual-tower bidomain surrogate** — specifically the outer autoregressive loop where V_m → φ_e → V_m compounds over 30K time steps. If the dual-tower suffers from the failure modes this paper characterizes, the fixes are architectural and cheap.

### Agreements
- **Long-horizon rollout stability is a first-class design concern**, not an afterthought. Reinforces the case for careful architecture choice over raw parameter count.
- **Architectural fixes preferable to inference-time fixes.** McCabe's surgery adds no per-step cost; PDE-Refiner's does. For our 30K-step bidomain rollout, per-step cost multiplies heavily — prefer McCabe's approach first, then add Lippe's refinement only if needed.
- **Stability generalizes across PDE families** — the identified failure modes are architectural, not PDE-specific. Our bidomain should benefit from the same fixes.

### Disagreements or Gaps
- **Architecture-specific**: the paper analyzes neural-operator architectures (FNO, transformer-based). Our planned dual-tower CNN may have different failure-mode structure — some of McCabe's fixes may not apply directly. **Adaptation required**: re-derive the stability analysis for V-cycle CNN or dual-tower cross-talk architectures.
- **Fluid/weather benchmarks, not cardiac.** Their PDEs have different characteristics (globally-coupled turbulence vs locally-propagating wavefronts). The dominant instability sources may differ — cardiac wavefront dispersion may not be the same failure mode as fluid turbulence divergence.
- **Does not address BC-induced errors.** The stability analysis is interior-focused. Boundary-induced drift (FNO's periodic-BC issue) is a separate failure mode not addressed by this paper.
- **Coupled-field dynamics not analyzed.** Single-field autoregressive iteration. Our bidomain has V_m + φ_e coupling at each step — stability analysis for the coupled iteration would require extension.
- **No convergence proofs** — the paper shows empirical stability over longer horizons, but doesn't prove bounded error. Asymptotic behavior still unknown.

### Actionable Insights
- **HIGH — Read BEFORE building the dual-tower autoregressive loop.** Pre-screen architecture choices against this paper's identified failure modes. Specifically: avoid BatchNorm in the cross-talk layer, prefer bounded activations (GELU, Swish) over unbounded (raw linear), place residual connections carefully.
- **HIGH — Audit our planned dual-tower against McCabe's checklist**: normalize at residual outputs not inputs, avoid amplifying feedback paths, use spectral normalization where stability-critical.
- **MEDIUM — Combine with PDE-Refiner** if architectural fixes aren't enough. McCabe-style architecture first, Lippe-style refinement wrap only if drift persists.
- **MEDIUM — Generalize the analysis to dual-tower cross-talk.** The paper analyzes single-tower architectures. The cross-talk in our design is a new feedback path whose stability properties aren't established. Worth a focused stability analysis (or a direct experiment) before deploying.
- **LOW — Code availability**: check whether authors released fixes as library. If yes, adopt directly.

## Limitations / Caveats
- **Neural-operator-specific analysis** — fixes may not transfer cleanly to our CNN V-cycle or dual-tower architectures without adaptation.
- **Fluid/weather PDE benchmarks only.** Cardiac EP has distinct characteristics (sharp wavefronts, operator splitting, coupled V_m/φ_e); empirical validation needed.
- **Empirical, not proof-based.** Stability is demonstrated by extended-rollout benchmarks, not proven as a theorem. Distribution shift could still trigger divergence.
- **Single-field assumption** — does not address coupled-field iteration stability directly.
- **No BC-instability coverage.** Periodic-BC-induced boundary errors (the kind FNO produces) are a separate issue outside this paper's scope.
- **Architecture-level fixes are "non-trivial to apply correctly"** — the analysis requires understanding the specific failure modes, not just mechanical replacement. Blindly copying McCabe's fixes without the analysis risks missing the actual issue.
