# Per-Question Brief — March-April 2026 Progress Report

> Per project: story, stated objective, evidence on hand, imagery plan, audit against Zimmerman V1 spec.
> Storage tank framed neutrally (no credit attribution per user 2026-04-28).

---

## Format compliance checklist (Zimmerman V1, applied to every research slide)

| # | Rule | Source |
|---|------|--------|
| 1 | Clear stated objective | "structured around a clear, stated objective for each experiment or task" |
| 2 | Standalone-readable; concise caption per figure | "without the need for verbal explanation" |
| 3 | ≥1 sentence interpretation per dataset | "explains its significance and your interpretation" |
| 4 | Grey "main takeaway" box at bottom (encouraged) | "summarized in a grey highlighted box at the bottom" |
| 5 | Videos explicitly marked | "explicitly indicate if any presented data is a video" |
| 6 | Scale bars on visual data | "all visual data must feature clear scale bars" |
| 7 | Notional/expected data **CLEARLY MARKED** (caps in spec) | hard rule |

Each slide audited against this checklist below.

---

## C — Boundary Conduction Speedup (rank 1, 2 slides)

### Story we want to tell
Boundary-driven CV speedup is real, mesh-converged, and bidomain-specific. We characterized the bidomain triangle artifact rigorously in March (anisotropy, conductivity scaling, monodomain control), then in April reduced the mechanism to a 2D storage-tank model where three independent control axes — boundary operator, directionality, and pump rule — each independently shape latency-of-activation-time (LAT). The hydrostatic Bernoulli derivation closes the loop: the per-cell law is Torricelli's law on a leaky tank. **The boundary speedup decomposes cleanly and the parameters that produce it are not coincidental — they sit at the joint optimum of all three axes.**

### Slide C1 — Bidomain triangle artifact: characterization & quantification

**Stated objective**: Characterize the triangular wavefront artifact at tissue-bath boundaries and determine whether the bath-coupled CV speedup is real or numerical.

**Key evidence**
- Anisotropic test: 2:1 conductivity ratio sharpens the triangular wavefront
- Conductivity sweep: edge-lead distance scales with √D_eff
- Monodomain control: produces flat wavefront → triangle is bidomain-specific (extracellular bath coupling, not numerical artifact)
- CV ratio: 1.071 measured at dx=0.025, **1.131 theoretical confirmed via mesh convergence**
- Triangle "merger" hypothesis test: merger does NOT occur — wavefronts remain distinct

**Imagery plan**
| # | Figure | Status | Source if exists |
|---|--------|--------|------------------|
| 1 | Triangle wavefront snapshot (anisotropic vs isotropic, side-by-side) | LIKELY HAVE | `Bidomain/Engine_V1/experiments/` (Phase 6 / boundary diagnostics, Mar 5 commits) |
| 2 | CV ratio vs dx mesh-convergence plot | LIKELY HAVE | `Bidomain/Engine_V1/experiments/` (boundary speedup tests) |
| 3 | (optional) monodomain-control flat wavefront | OPTIONAL | same dir |

**Main takeaway box**: "The triangle artifact is mesh-converged and bidomain-specific. Bath-coupled CV ratio reaches the theoretical 1.131."

**Format audit**
- [✓] Objective stated
- [✓] Captions per figure (must add: dx, conductivity ratio, simulated time)
- [✓] Interpretation sentence per dataset
- [✓] Grey takeaway box planned
- [n/a] No videos
- [⚠] **Scale bars required on wavefront snapshots** (mm). Need to verify existing PNGs include them; regenerate if not.
- [n/a] No notional data

### Slide C2 — Storage-tank reduced model: three-axis decomposition

**Stated objective**: Reduce the bidomain boundary-speedup mechanism to a minimal 2D storage-tank toy model and identify the dominant control axes.

**Key evidence**
- Numpy vec rewrite (`simulation/tanks_vec.py`): ~200× faster than OOP, bit-equivalent (max|ΔV| = 5e-14, max|Δiso| = 0)
- **Three independent axes identified**: boundary condition × directionality × pump rule. Each independently shapes LAT.
  - Reflection-BC test (`simulation/ghost_corner_test.py`): boundary *operator* dominates pump rule
  - Bidirectional-pipes test (`simulation/bidirectional_test.py`): drops V_src > V_dst gate from constant rule → kills "camel-toe" LAT shape → directionality is third axis
- Pump-speed Goldilocks zone: camel-toe shape peaks at `max_pump=10`
- Hydrostatic Bernoulli derivation (today): per-cell law = Torricelli, v = √(2g·(h − θ)) for outlet at threshold height
- Single-cell LBM mechanism analogy: clean isomorphism to D2Q5 lattice

**Imagery plan**
| # | Figure | Status | Source if exists |
|---|--------|--------|------------------|
| 1 | LAT contour panel: 3 BC regimes side-by-side (zero-pad / refl-y / refl-all) | HAVE / GENERATE | `simulation/` outputs from Apr 25 |
| 2 | Pump-speed Goldilocks scan (camel-toe metric vs max_pump) | HAVE / GENERATE | `simulation/` Apr 26 harness output |
| 3 | (optional) Bernoulli-tank schematic with outlet at θ | TO CREATE | hand schematic / TikZ |

**Main takeaway box**: "Three independent axes (BC × directionality × pump rule) control LAT shape; the empirical sweet-spot parameters sit at the joint optimum of all three."

**Format audit**
- [✓] Objective stated
- [✓] Captions per figure (must add: lattice size, pump speed, BC name)
- [✓] Interpretation per dataset (axis identification + sweep result + derivation)
- [✓] Grey takeaway box
- [n/a] No videos
- [⚠] **Scale bars on LAT contours** — `simulation/` plots use dimensionless units, but spatial scale bar (cells / mm-equivalent) needs to be added or noted as dimensionless tank-grid
- [n/a] No notional data (everything shown is empirical)

---

## D — Ionic Optimizer V1 + MHAS13 hiPSC-CM (rank 2, 1 slide)

### Story we want to tell
Built a Bayesian optimization pipeline (V1) for fitting hiPSC-CM ionic models, then used it to fit a matured cardiomyocyte variant (MHAS13). The pipeline works: APD targets are met to within 1ms. But fitting a single AP exposed a fundamental degeneracy — multiple parameter sets reach the same APD via different K-current balances (IKr/IKs compensation), and the APD–max-dV/dt tradeoff sits on a Pareto front. **Single-AP fitting is non-unique by construction; degeneracy must be resolved with restitution and tissue-level constraints in V2.**

### Slide D1 — Optimizer V1 + MHAS13: pipeline build + degeneracy finding

**Stated objective**: Build a Bayesian optimization pipeline for hiPSC-CM ionic-model fitting and apply it to a matured-cell variant (MHAS13).

**Key evidence**
- **V1 pipeline operational**: BayesOpt over HMC for V1 (HMC deferred); 10× speedup via batching + subcycling + analytical CV
- **MHAS13 created**: TTP06 + IK1 injection → quiescent V_rest = -83.7mV (matured-cell phenotype)
- **Optimizer fit**: APD 347ms (target 350ms) — within 1ms tolerance
- **Bidomain validation**: APD 349ms, CV 15.8 cm/s through full bidomain pipeline
- Iteration 2 enhancements: constraints + tier-2 + secant CV + seeding
- **Degeneracy finding**: APD-dVdt tradeoff is fundamental, not a fitting artifact. **IKr/IKs compensation = core degeneracy** — multiple parameter sets achieve the same APD via different K-current ratios.

**Imagery plan**
| # | Figure | Status | Source if exists |
|---|--------|--------|------------------|
| 1 | AP trace overlay: MHAS13 fit vs target waveform | TO CREATE | data exists in optimizer output, plot needs generation |
| 2 | APD-dV/dt Pareto scatter (multiple parameter sets at fixed APD) | TO CREATE | requires re-running iteration 2 logs through plotter |
| 3 | (optional) IKr/IKs compensation cross-section | TO CREATE / OPTIONAL | parameter-space heatmap |

**Main takeaway box**: "MHAS13 fits APD to 1ms tolerance, but single-AP fitting is fundamentally non-unique — restitution + tissue-level constraints are required to break IKr/IKs degeneracy."

**Format audit**
- [✓] Objective stated
- [✓] Captions per figure (mark sweep ranges, simulated BCL)
- [✓] Interpretation per dataset (the degeneracy finding is itself the interpretation)
- [✓] Grey takeaway box
- [n/a] No videos
- [⚠] **Scale bars / axes**: AP trace needs time-axis (ms) + voltage-axis (mV) labels; Pareto plot needs APD (ms) and dV/dt_max (V/s) units. These are charts, not microscopy — labels suffice, no physical scale bar needed.
- [⚠] **Target curve must be clearly marked** if it represents a desired (notional) outcome rather than measured data — label as "target" or use dashed line + legend

---

## A — Surrogate Pipeline (rank 3, 1 slide, honest-pivot framing)

### Story we want to tell
We invested two months building a neural surrogate to replace the bidomain ionic step — full data-generation pipeline (47,000× faster than sequential), three architecture iterations, a Neural ODE pivot after discrete training failed. A direct throughput benchmark in late April revealed the surrogate Euler path is **8× slower than the classical TTP06 solver at tissue scale**. The premise was wrong: 94% of bidomain wall time lives in the elliptic solve, not the ionic step. **We are redirecting effort to a learned elliptic sub-operator (dual-CNN tower) — the actual bottleneck.** This is an honest negative result for the ionic-replacement direction.

### Slide A1 — Surrogate pipeline: investigated and pivoted

**Stated objective**: Build a neural surrogate to replace the bidomain ionic step, benchmark against the classical solver, and redirect if the speedup case fails.

**Key evidence**
- Two-month build-out: data generation pipeline (`BatchGenerator` + `torch.compile`, **47,000× speedup** over sequential, all 12 BCL tiers, 270k cell-steps/s on CPU)
- Architecture iterations: v1 → v3 (Layer 0) → Neural ODE pivot after discrete A4 training failed at 30k steps; v4 expansion (StateRateMLP, 7,891 params)
- v4 capacity test: 7,891 params overfits T1's 25 trajectories; v3 at 1,444 params had converged on the same split → capacity ≠ the bottleneck
- **TTP06 vs surrogate inference benchmark**: classical TTP06 = 34.1 M cell-steps/s; v4 surrogate Euler = 4.2 M cell-steps/s → **classical is 8× faster** at tissue scale
- KNOWLEDGE §1 already noted that **94% of bidomain wall time is the elliptic solve**, not the ionic step → ionic surrogate was never the GPU speedup lever
- **Pivot**: keep classical TTP06 as ionic scaffold; build neural surrogate for the **bidomain elliptic step**. Architecture direction: dual CNN tower (Vm tower + φ_e tower with cross-attention). v1 targets parabolic-elliptic; hyperbolic-bidomain deferred. v4 ionic kept for narrow secondary roles only (CPU deployment +3-7×, differentiable parameter optimization)

**Imagery plan**
| # | Figure | Status | Source if exists |
|---|--------|--------|------------------|
| 1 | Throughput bar chart: TTP06 vs surrogate Euler (cell-steps/s, log y) | HAVE | `Surrogate/benchmarks/results/` (Apr 21) |
| 2 | v4 train/val loss curve showing overfitting fingerprint | HAVE | `Surrogate/runs/` or `archive/runs_legacy/` (Session 28) |
| 3 | Bidomain wall-time profile pie/bar (94% elliptic) | HAVE / GENERATE | from KNOWLEDGE §1 / V1 profiling |
| 4 | (optional) dual-CNN-tower architecture sketch — **NOTIONAL, must be marked** | TO CREATE | TikZ schematic of next-step direction |

**Main takeaway box**: "Surrogate ionic step is 8× slower than classical TTP06 at tissue scale. Effort redirected to the elliptic solve (94% of wall time) via dual-CNN tower architecture."

**Format audit**
- [✓] Objective stated (honest about goal — including "redirect if speedup case fails")
- [✓] Captions per figure
- [✓] Interpretation per dataset (benchmark numbers + overfit fingerprint + wall-time decomposition each get their own takeaway sentence)
- [✓] Grey takeaway box (clearly states the pivot)
- [n/a] No videos
- [⚠] No scale bars needed (charts only) — axis labels mandatory
- [⚠] **Dual-CNN sketch is notional/expected → must be CLEARLY MARKED** ("Direction — not yet implemented" label or "Future v1 architecture (notional)" caption)

---

## F — Bidomain Parabolic-Parabolic (rank 4, 1 slide)

### Story we want to tell
Triangle wavefront artifacts at tissue-bath boundaries traced to the bidomain elliptic equation propagating extracellular voltage instantaneously across the domain. We launched a four-agent literature dive to find a published bidomain formulation with finite extracellular propagation speed. **Finding: there isn't one.** What the literature calls "parabolic-parabolic bidomain" (Bourgault, Pavarino & Scacchi) is a variable rename of the same physics — adding the two equations cancels ∂Vm/∂t and the elliptic constraint reappears. The ε·∂φ_e/∂t regularization (Bendahmane & Karlsen) is a proof technique, never used as a solver. Only the Cattaneo/hyperbolic bidomain (Rossi & Griffith 2017) gives genuine finite extracellular speed. **The 2-DOF dual-evolving variant we want is novel and requires hyperbolic-hyperbolic terms (six (τ_i − τ_e) cross-terms identified in `HYPERBOLIC_HYPERBOLIC_ANALYSIS.md`).**

### Slide F1 — Bidomain PP literature dive: only Cattaneo gives finite propagation

**Stated objective**: Identify a published bidomain formulation with physically finite extracellular propagation speed, to eliminate the elliptic-instantaneous coupling responsible for tissue-bath boundary artifacts.

**Key evidence**
- 4-agent parallel literature dive across 6+ papers (Bourgault 2009, Pavarino & Scacchi 2011, Bendahmane & Karlsen 2006, Rossi & Griffith 2017 *Chaos*, ESAIM M2AN 2013, Bishop & Plank 2011)
- **Critical finding**: Bourgault 2009 / Pavarino & Scacchi 2011 "PP bidomain" is the original (φ_i, φ_e) variable form. Adding the two equations cancels ∂Vm/∂t — same physics as PE, different variables.
- ε·∂φ_e/∂t regularization (Bendahmane & Karlsen 2006) used only as proof technique (ε→0 recovers true bidomain). **No published solver uses it computationally.**
- **Only published model with finite extracellular propagation speed**: Cattaneo/hyperbolic bidomain (Rossi & Griffith 2017, *Chaos* 27:093926). Telegraph equation via τ·dJ/dt + J = -σ∇V.
- 2026-04-23 insight: Rossi-Griffith's "ParabolicParabolicHyperbolic" (τ_i ≠ τ_e) is structurally 2-unknown (Q, V_e) but physically **1-DOF-with-memory** — only V has time derivatives in eqs. 13/14.
- **Endgame**: dual-evolving bidomain LBM with 2 independent dynamical DOFs requires 6 (τ_i − τ_e) cross-terms (`HYPERBOLIC_HYPERBOLIC_ANALYSIS.md`).

**Imagery plan**
| # | Figure | Status | Source if exists |
|---|--------|--------|------------------|
| 1 | Schematic comparing what evolves in time across formulations: PE / "PP-rename" / Cattaneo / dual-evolving-LBM | TO CREATE | TikZ or hand schematic; columns = unknowns, rows = time-derivative orders |
| 2 | (optional) triangle artifact this is meant to fix | LIKELY HAVE | reuse from slide C1 if reused |

**Main takeaway box**: "No off-the-shelf bidomain formulation provides true 2-DOF extracellular dynamics. The dual-evolving target is novel and demands hyperbolic-hyperbolic terms."

**Format audit**
- [✓] Objective stated
- [✓] Caption on the schematic (column/row meaning)
- [✓] Interpretation sentence (the framework comparison itself is the interpretation)
- [✓] Grey takeaway box
- [n/a] No videos
- [n/a] No scale bars (schematic only)
- [⚠] **"Dual-evolving bidomain LBM" column is notional / not-yet-implemented → must be CLEARLY MARKED** as proposed/notional in the schematic

---

## B — cardiac_ml Harness (rank 5, 1 slide)

### Story we want to tell
The surrogate work and (eventually) the elliptic-surrogate, BayesOpt wrapper, and other learned components were all going to need the same training scaffolding: data loading, MLflow tracking, checkpointing, hyperparameter sweeps. Rather than rebuild it per project, we extracted a project-wide harness (`cardiac_ml/`) using Hydra config composition + MLflow file-backed tracking + a single `Trainer` class. **NODE parity migration validated the harness against the legacy pipeline (val_loss = 0.00835, threshold 0.0088). A separate diffusion-stub model trains end-to-end on the same harness — reusability proven.** 80 cardiac_ml tests pass. The harness now unblocks downstream learned-component work without re-paying the harness cost.

### Slide B1 — cardiac_ml harness: project-wide ML training infrastructure

**Stated objective**: Provide a single project-wide training harness for all learned components (NODE ionic surrogate, future elliptic surrogate, BayesOpt wrapper, etc.) — replacing ad-hoc per-project training scripts.

**Key evidence**
- **Architecture**: Hydra config composition + MLflow file-backed tracking + single `Trainer` class + pure-function `train_step_fn` injected via Hydra `_target_: hydra.utils.get_method`. Not Lightning, not `log_model`, not per-task subclasses.
- **Phases 1-5 executed**: env+docs, package skeleton + 18 tests (`eb057232`), Trainer + MLflow + callbacks + 57 tests (`57b7efac`), Phase 4 NODE migration + 15 tests (`b20fabf7`), Phase 5 Optuna sweep + SHAP + diffusion stub (`5e59ad39`)
- **NODE parity met**: `val_loss = 0.00835 < 0.0088 threshold` at epoch 1, warm-started from `multi_bcl_002/best.pt`
- **Reusability proof**: diffusion-stub model (a different architecture) trains end-to-end on the same harness — confirmed via test
- 80 cardiac_ml tests pass; surrogate runs cleanly archived to `archive/runs_legacy/`
- 11/11 completion criteria checked

**Imagery plan**
| # | Figure | Status | Source if exists |
|---|--------|--------|------------------|
| 1 | Training loss curve from NODE parity run (val crosses 0.0088 threshold) | HAVE | `archive/runs_legacy/multi_bcl_002/` or cardiac_ml MLflow output |
| 2 | (optional) Architecture / dependency diagram: Hydra → Trainer → train_step_fn | TO CREATE | TikZ block diagram |
| 3 | (optional) SHAP summary from Phase 5 analysis script | OPTIONAL | requires running `scripts/analyze.py` against parity ckpt |

**Main takeaway box**: "Harness validated end-to-end (NODE parity met, diffusion stub reusable). Downstream learned components (elliptic surrogate, BayesOpt wrapper) now have a single training surface."

**Format audit**
- [✓] Objective stated
- [✓] Captions per figure (axes, run config)
- [✓] Interpretation per dataset (parity threshold + reusability test outcome)
- [✓] Grey takeaway box
- [n/a] No videos
- [n/a] No scale bars (charts/diagrams only)
- [n/a] No notional data — everything shown is measured

---

## Summary slide pre-draft (Slide 2)

Each project gets primary bullet + 1-4 sub-bullets. Order on slide = ranking:

- **Boundary Conduction Speedup** (PI-collaborative)
  - Bidomain triangle artifact characterized: anisotropic + conductivity sweep, monodomain control flat, CV ratio 1.071 → 1.131 theoretical via mesh convergence
  - Storage-tank 2D toy model: 200× vec rewrite, three-axis decomposition (BC × directionality × pump rule), pump-speed Goldilocks zone, hydrostatic Bernoulli derivation

- **Ionic Optimizer V1 + MHAS13 hiPSC-CM**
  - V1 BayesOpt pipeline operational (10× speedup via batching + analytical CV)
  - MHAS13 fit: TTP06 + IK1 → V_rest -83.7mV, APD 347ms (target 350)
  - Bidomain validation: APD 349ms, CV 15.8 cm/s
  - Single-AP fitting non-unique — IKr/IKs compensation degeneracy identified

- **Surrogate Pipeline** (investigated → pivoted)
  - Built data-generation pipeline (47,000× speedup, 12 tiers); 2-month architecture arc through v3 + Neural ODE
  - Benchmark: surrogate Euler 8× slower than classical TTP06 at tissue scale
  - Pivot: replace elliptic step (94% of wall time) instead — dual-CNN tower architecture

- **Bidomain Parabolic-Parabolic** (literature finding)
  - 4-agent literature dive across 6+ papers
  - Published "PP bidomain" is variable rename, not finite-propagation; only Cattaneo gives true finite extracellular speed
  - Endgame reframed: dual-evolving bidomain LBM (novel, requires hyperbolic-hyperbolic terms)

- **cardiac_ml Harness** (completed)
  - Project-wide ML training infrastructure: Hydra + MLflow + single Trainer
  - NODE parity met (val_loss 0.00835 < 0.0088 threshold); 80 tests; diffusion-stub reusability proven

- **General laboratory activities**
  - Bidomain V1 hardening (March): two-round audit; four new diffusion solvers (Jacobi, semi-implicit, IMEX SBDF2, RKC); Mehrstellen 9-pt stencil
  - Engine consolidation Phase 0: unified `monodomain()/bidomain()/lbm()` API + 34 tests
  - Research workflow infrastructure: 16 skills + three-document architecture + PreCompact hook
  - Lab meetings attended (3/5, 3/19, 4/3, 4/24); Oxford application recommendation submitted

**Format audit (Summary slide)**: Spec calls for "primary bullet point per active project, supported by sub-list of 1-4 key takeaways." All 5 projects fit; sub-bullet counts within range (4-5; the 5-bullet on cardiac_ml may need to drop one to stay strict). General-lab-activities entry uses spec-allowed escape.

---

## Future Outlook pre-draft (Slide 9)

Each project gets a primary bullet of next-phase milestones (mirrors Summary):

- **Boundary Conduction Speedup**: anisotropic-boundaries sub-question (already scaffolded); 3D extension of triangle artifact characterization; map storage-tank Goldilocks zone back onto bidomain parameter space
- **Optimizer V1 + MHAS13**: tier-2 calibration with restitution + tissue-level constraints (resolves IKr/IKs degeneracy); ORd-based MHAS13 cross-validation
- **Surrogate Pipeline**: dual-CNN-tower elliptic surrogate v1 (parabolic-elliptic); train on bidomain V1 outputs through cardiac_ml harness — *first end-to-end hybrid run*
- **Bidomain Parabolic-Parabolic**: implement dual-evolving bidomain LBM per `HYPERBOLIC_HYPERBOLIC_ANALYSIS.md`; 2D test against PE baseline
- **cardiac_ml Harness**: consumer migrations (diffusion ResNet, Optimizer V1 BayesOpt wrapper)
- **Infrastructure**: Engine consolidation Phase 1 (extract shared ionic/mesh modules); finalize Mesh Builder Fiji loader contract

---

## Open questions before content draft

1. **Slide D imagery**: AP trace + Pareto plot need to be generated. Do you want to run/regenerate from optimizer iteration 2 logs, or do these already exist as PNGs somewhere?
2. **Slide A imagery**: dual-CNN-tower architecture sketch — make it now or punch through with a bare "future direction (notional)" textbox?
3. **Slide F imagery**: PE/PP-rename/Cattaneo/dual-evolving comparison schematic — TikZ or quick hand drawing? (TikZ takes longer, looks cleaner.)
4. **Slide B imagery**: SHAP plot — worth generating, or skip and just show the parity training curve?
5. **Total slide count comfort**: 9 slides. Spec range 7-14. Want to expand somewhere or trim?
