---
paper: koehler_2024_apebench
title: "APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs"
authors: "Koehler F, Niedermayr S, Westermann R, Thuerey N"
year: 2024
journal: "NeurIPS 2024"
doi: "arxiv:2411.00180"
pmid: ""
pdf: ../papers/apebench_2024_koehler.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **46 distinct PDEs across 1D, 2D, and 3D** — the most comprehensive autoregressive PDE-emulator benchmark to date. Covers reaction-diffusion (Fisher-KPP, Gray-Scott, Swift-Hohenberg, Allen-Cahn, and likely more), heat equation, Poisson, linear and nonlinear advection-diffusion-reaction systems.
- **Explicit focus on *autoregressive* emulators and *long-rollout* stability metrics** — exactly the regime where cardiac surrogates fail (AGATA, PDE-Refiner all document this). APEBench is where we'd empirically measure drift, dispersion, and energy drift over long horizons.
- **"Unique identifier for PDE dynamics that directly relates to the stability criteria of classical numerical methods"** — the benchmark provides a formal way to characterize each PDE's numerical difficulty, enabling apples-to-apples comparisons across architectures.
- **`pip install apebench`** — lowest-friction adoption possible. Drop-in benchmark.
- **Open-source**: `github.com/tum-pbs/apebench`. TUM-PBS (Thuerey lab, same as PDE-Transformer).
- **External validation story for our project** — running our bidomain surrogate on APEBench's RD tasks gives a cardiac-independent accuracy/stability measure.

## Method
- **Benchmark scope**: 46 PDEs × 1D/2D/3D × multiple parameter regimes × multiple initial conditions. Test set design pressures autoregressive rollout stability rather than one-shot accuracy.
- **Metrics**: multiple per-PDE metrics plus a unifying "dynamics identifier" that ties the PDE to classical numerical-stability criteria (CFL, Péclet, etc.). Enables cross-PDE comparisons that standard L² loss doesn't.
- **Distribution**: Python package, pip-installable. Standard ML benchmark format (PyTorch DataLoader-compatible).
- **Included PDEs** (abstract mentions 46 across dimensions; specific enumeration requires PDF): explicitly includes reaction-diffusion systems which are the closest non-cardiac analog to our bidomain V_m dynamics.
- **Reference implementations**: various architectures benchmarked (FNO, UNet, CNN, transformer-based).

## Connections to Our Models

### Relevant Engine Components
**External-validity instrument.** APEBench is where we run our hybrid bidomain surrogate on cardiac-independent test cases to demonstrate that the architecture generalizes beyond the specific Bidomain V1 problem we trained on. Essential for publication credibility.

### Agreements
- **Autoregressive rollout stability is a first-class concern** — aligns with our own concerns about 30K-step bidomain rollouts.
- **Reaction-diffusion systems are a legitimate adjacent benchmark domain.** Fisher-KPP's wave-propagation dynamics are direct analogs to cardiac wavefronts; Gray-Scott produces Turing-like patterns similar to re-entry; Allen-Cahn gives sharp-interface dynamics like upstrokes.
- **Non-periodic BCs** handled in the benchmark (implicit from PDE selection: Fisher-KPP, Allen-Cahn typically use Dirichlet or Neumann).

### Disagreements or Gaps
- **No cardiac-specific PDE included** — the benchmark is general PDE emulation, not cardiac. Cardiac-specific artifacts (wavefront sharpness, stiff ionic coupling, operator splitting) may not be captured.
- **Abstract doesn't enumerate the exact 46 PDEs** — need to inspect the package to verify which reaction-diffusion systems are present.
- **Benchmarks are 1D-2D-3D general-purpose** — our Bidomain V1 is 2D-specific. Running APEBench tests is either (a) on a separate set of grids, or (b) requires adapting our architecture to other grid sizes.
- **Focus on "autoregressive emulators"** — our hybrid bidomain surrogate is not purely autoregressive in the PDE sense (classical ionic step integrated exactly, only elliptic learned). Some APEBench metrics may not directly apply.
- **Metric design focuses on stability and long-rollout accuracy** — good match for us. But doesn't directly benchmark preconditioner use or iteration count savings (which are the more "hybrid-friendly" metrics).

### Actionable Insights
- **HIGH — Install APEBench and run our dual-tower bidomain surrogate against Fisher-KPP, Gray-Scott, Allen-Cahn.** Immediate cross-domain sanity check. `pip install apebench` → benchmark loop → report.
- **HIGH — Cite APEBench in any write-up** as the standard-bearer for autoregressive PDE benchmarks; our compliance is publication-hygiene.
- **MEDIUM — Use APEBench's "dynamics identifier" formalism** to position the bidomain elliptic step in the PDE-difficulty landscape. Gives us a principled way to argue our problem is harder/easier than other benchmarks.
- **MEDIUM — Check whether APEBench includes cardiac or neural-axon PDEs** (FitzHugh-Nagumo, Hodgkin-Huxley). If not, propose adding them as a contribution.
- **LOW — Contribute cardiac bidomain as an APEBench PDE**: if we're publishing anyway, contributing our problem back to the benchmark extends the benchmark's utility and boosts our paper's impact. Worth considering after Phase A lands.

## Limitations / Caveats
- **No cardiac PDEs included** as of 2024 publication. We'd use APEBench for adjacent-domain validation, not direct cardiac accuracy.
- **46 PDEs**: the exact list requires inspecting the package. Some may be trivially simple (linear advection) and not informative.
- **Autoregressive-emulator focus** may not align perfectly with our hybrid classical+neural setup — some metrics may penalize our architecture for not being purely autoregressive.
- **1D/2D/3D general-purpose**: tests are on generic grid shapes, not cardiac geometries.
- **NeurIPS 2024 paper**: package maturity is early. Expect some rough edges, possibly breaking changes.
- **No BC handling evaluation explicitly**: need to check which BC types are tested and whether our Neumann-dominant case is represented.
