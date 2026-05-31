# April 2026 Progress Report — Slide Outline (Draft 1)

> Reporting period: 2026-03-01 → 2026-04-30 (extended at user request)
> Format: Zimmerman Lab Progress Report V1 (4/20/2026 spec)
> Template: `ZimmermanLab_DefaultSlides.pptx` (4:3, 10×7.5in)
> Project ranking (per user 2026-04-28): C > D > A > F > B

## Slide structure (9 slides)

### Slide 1 — Title
- **Researcher**: Charley Chang
- **Period**: March-April 2026
- (Layout: `7_Title Slide` from template)

### Slide 2 — Summary
*Each active project = primary bullet, 1-4 sub-bullets of key takeaways.*

- **Boundary Conduction Speedup** (rank 1, PI-collaborative)
  - Bidomain triangle wavefront artifact characterized: anisotropic + conductivity sweep, CV ratio 1.071 → 1.131 theoretical via mesh convergence
  - PI's storage-tank toy model rebuilt + analyzed: three independent axes (BC × directionality × pump rule), pump-speed Goldilocks zone, hydrostatic Bernoulli derivation

- **Ionic Optimizer V1 + MHAS13 hiPSC-CM** (rank 2)
  - V1 BayesOpt pipeline operational (10× speedup via batching + analytical CV)
  - MHAS13 created: TTP06 + IK1 → quiescent V_rest=-83.7mV, APD 347ms vs 350 target
  - APD-dVdt tradeoff identified as fundamental model property; IKr/IKs compensation degeneracy

- **Surrogate Pipeline** (rank 3, investigated → pivoted)
  - Two-month build-out: data gen (47,000× speedup, 12 tiers), Neural ODE pivot, v4 capacity test
  - Benchmark revealed surrogate Euler 8× slower than classical TTP06 at tissue scale
  - Pivot: replace elliptic step (94% of bidomain wall time), not ionic — dual CNN tower architecture

- **Bidomain Parabolic-Parabolic** (rank 4, literature finding)
  - 4-agent literature dive: published "PP bidomain" is variable rename, not finite-propagation
  - Only Cattaneo/hyperbolic gives true finite extracellular speed; endgame reframed to dual-evolving bidomain LBM

- **cardiac_ml Harness** (rank 5, completed)
  - Project-wide ML training infrastructure (Hydra + MLflow + single Trainer pattern)
  - 80 tests, NODE parity met (val_loss=0.00835 < 0.0088 threshold)

- **General laboratory activities**
  - Bidomain V1 hardening (March): two-round audit, 4 new diffusion solvers (Jacobi, semi-implicit, IMEX SBDF2, RKC), Mehrstellen 9-pt stencil
  - Engine consolidation Phase 0: unified `monodomain()`/`bidomain()`/`lbm()` API, 34 tests
  - Research workflow infrastructure: 16 skills + 3-document architecture + PreCompact hook
  - Lab meetings attended (3/5, 3/19, 4/3, 4/24); rec letter for Oxford application submitted

### Slide 3 — Boundary Speedup: Bidomain Triangle Artifact (rank 1, slide 1 of 2)
- **Objective**: Characterize the triangular wavefront artifact at tissue-bath boundaries and quantify the CV speedup it produces.
- Anisotropic test: 2:1 conductivity ratio → sharper triangles. Conductivity sweep: edge-lead distance scales with √D_eff.
- Monodomain control: flat wavefront → triangle is bidomain-specific (extracellular bath coupling)
- Bath-coupled CV ratio: **1.071 measured** at dx=0.025, **1.131 theoretical** confirmed via mesh convergence study
- Triangle "merger" experiment: hypothesized merger does NOT occur — wavefronts remain distinct
- **[FIGURE NEEDED]** triangle wavefront snapshot (anisotropic vs isotropic)
- **[FIGURE NEEDED]** CV ratio vs dx convergence plot
- **Main takeaway box**: The triangle artifact is a real, mesh-converged bidomain phenomenon driven by extracellular bath loading — not a numerical artifact.

### Slide 4 — Boundary Speedup: PI's Storage-Tank Toy Model (rank 1, slide 2 of 2)
- **Objective**: Reduce the bidomain boundary-speedup question to a minimal mechanism using John's storage-tank analogy and identify the dominant control axes.
- John (4/24) shared 2D storage-tank Colab → rebuilt as numpy vec (`simulation/tanks_vec.py`, ~200× faster, bit-equivalent to OOP)
- **Three independent control axes identified**: boundary condition × directionality × pump rule. Each independently shapes LAT.
  - Reflection-BC test: boundary *operator* dominates pump rule
  - Bidirectional pipes: directionality is the third axis (kills "camel-toe" LAT shape)
- Pump-speed Goldilocks zone: camel-toe peaks at `max_pump=10` (matches John's empirical sweet spot)
- Hydrostatic Bernoulli derivation: John's per-cell law = Torricelli's law on a tank with outlet at threshold height; single-cell LBM analogy clean
- **[FIGURE NEEDED]** LAT contour plot for the three boundary regimes (zero-pad / refl-y / refl-all)
- **[FIGURE NEEDED]** pump-speed Goldilocks scan (camel-toe metric vs max_pump)
- **Main takeaway box**: The boundary speedup decomposes into three independent axes; John's chosen rule sits at the joint optimum of all three.

### Slide 5 — Optimizer V1 + MHAS13 Matured hiPSC-CM (rank 2)
- **Objective**: Build a Bayesian optimization pipeline to fit hiPSC-CM ionic models to target electrophysiology, using a quiescent matured-cell variant as test case.
- **Optimizer V1** designed and implemented: BayesOpt selected over HMC for V1; 10× speedup via batching + subcycling + analytical CV
- **MHAS13** (matured hiPSC-CM via TTP06 + IK1 injection): quiescent at V_rest=-83.7mV
  - Optimizer pipeline: APD 347ms (target 350ms)
  - Bidomain pipeline validation: APD 349ms, CV 15.8 cm/s
- Iteration 2 added constraints + tier-2 + secant CV + seeding
- **Key finding**: APD-dVdt tradeoff is a fundamental model property, not a fitting artifact. **IKr/IKs compensation = core degeneracy** — multiple parameter sets reach same APD via different K-current balance.
- **[FIGURE NEEDED]** AP trace (MHAS13 vs target)
- **[FIGURE NEEDED]** APD-dVdt Pareto front
- **Main takeaway box**: Single-AP fitting is fundamentally non-unique; degeneracy must be resolved with restitution + tissue-level constraints.

### Slide 6 — Surrogate Pipeline: Investigated and Pivoted (rank 3)
- **Objective**: Build a neural surrogate to replace the bidomain ionic step; benchmark against classical solvers; redirect if not justified.
- Two-month build-out: TTP06 data generation (`BatchGenerator` + `torch.compile`, **47,000× speedup** vs sequential, 12 tiers)
- Architecture iterations v1 → v3 → Neural ODE pivot (after discrete A4 training failed at 30k steps)
- v4 capacity test: 7,891-param model overfits T1's 25 trajectories; v3 at 1,444 params had converged on the same split
- **TTP06-vs-surrogate inference benchmark (Apr 21)**: surrogate Euler path **8× slower than classical TTP06** at tissue scale. The ionic step was never the bottleneck — 94% of bidomain wall time is the elliptic solve.
- **Pivot**: keep classical TTP06 as ionic scaffold, build neural surrogate for the **bidomain elliptic step** instead. Architecture direction: dual CNN tower (Vm + φ_e) with cross-attention. Parabolic-elliptic v1 first; hyperbolic deferred.
- **[FIGURE NEEDED]** TTP06 vs surrogate throughput bar chart (cell-steps/sec)
- **[FIGURE NEEDED]** v4 train/val loss curve showing overfitting fingerprint
- **Main takeaway box**: Ionic surrogate work demonstrated honest negative result — classical solver wins on the ionic step. Effort redirected to the actual 94%-of-wall-time bottleneck (elliptic).

### Slide 7 — Bidomain Parabolic-Parabolic: Literature Finding (rank 4)
- **Objective**: Identify whether a published bidomain formulation provides physically finite extracellular propagation speed (to eliminate elliptic instantaneous coupling).
- 4-agent parallel literature dive (Bourgault 2009, Pavarino & Scacchi 2011, Bendahmane & Karlsen 2006, Rossi & Griffith 2017, ESAIM M2AN 2013, Bishop & Plank 2011)
- **Critical finding**: published "parabolic-parabolic bidomain" is variable rename of the same physics — adding the two equations cancels ∂Vm/∂t and the elliptic constraint reappears hidden inside. ε·∂φ_e/∂t regularization exists only as a *proof technique* (Bendahmane & Karlsen) — never used as a computational method.
- The only published model with finite extracellular propagation speed: **Cattaneo/hyperbolic bidomain** (Rossi & Griffith 2017, Chaos 27:093926) — telegraph equation via τ·dJ/dt + J = -σ∇V
- **2026-04-23 insight**: Rossi-Griffith "ParabolicParabolicHyperbolic" (τ_i ≠ τ_e) is structurally 2-unknown (Q, V_e) but physically **1-DOF-with-memory** — only V has time derivatives. To get 2 independent dynamical DOFs, need the 6 (τ_i − τ_e) terms identified in `HYPERBOLIC_HYPERBOLIC_ANALYSIS.md`. Endgame: dual-evolving bidomain LBM.
- **[FIGURE NEEDED]** comparison diagram: PE vs PP (variable rename) vs Cattaneo vs dual-evolving — what evolves in time
- **Main takeaway box**: No off-the-shelf formulation gives true 2-DOF extracellular dynamics — the dual-evolving variant we want is novel and requires hyperbolic-hyperbolic terms.

### Slide 8 — cardiac_ml Harness: Project-Wide ML Training Infrastructure (rank 5)
- **Objective**: Provide a single project-wide training harness for all learned components (ionic NODE, diffusion ResNet, future BayesOpt wrapper) — replace ad-hoc per-project training scripts.
- Architecture: Hydra config composition + MLflow file-backed tracking + single `Trainer` class + pure-function `train_step_fn` (`_target_: hydra.utils.get_method`)
- Phases 1-5 executed: env, package skeleton, Trainer + MLflow + callbacks, NODE parity migration, Optuna sweep + SHAP + diffusion stub
- **NODE parity met**: `val_loss=0.00835 < 0.0088` threshold at epoch 1 (warm-started from `multi_bcl_002/best.pt`)
- Reusability proven via diffusion stub test (a different model trains end-to-end on the same harness)
- 80 cardiac_ml tests pass; surrogate runs cleanly archived to `archive/runs_legacy/`
- **[FIGURE NEEDED]** training loss curve from parity run (NODE on T1)
- **[FIGURE NEEDED]** SHAP summary plot (sample output from analysis script)
- **Main takeaway box**: Training infrastructure unblocks all downstream learned-component work (hybrid bidomain elliptic surrogate, optimizer wrapper) without re-paying the harness cost.

### Slide 9 — Future Outlook
*Strategic mirror of the Summary slide. Primary bullets per ongoing project, upcoming milestones.*

- **Boundary Conduction Speedup**: anisotropic boundaries study (sub-question already scaffolded); 3D extension; map storage-tank Goldilocks zone back onto bidomain parameter space
- **Optimizer V1 + MHAS13**: tier-2 calibration with restitution + tissue-level constraints; ORd-based MHAS13 variant for cross-validation
- **Surrogate Pipeline**: implement dual-CNN-tower elliptic surrogate v1 (parabolic-elliptic); train on bidomain V1 outputs via cardiac_ml harness
- **Bidomain Parabolic-Parabolic**: implement dual-evolving bidomain LBM per `HYPERBOLIC_HYPERBOLIC_ANALYSIS.md`; first 2D test against PE baseline
- **cardiac_ml Harness**: consumer migrations (diffusion ResNet, BayesOpt wrapper for Optimizer V1)
- **Infrastructure**: complete Engine consolidation Phase 1 (extract shared ionic/mesh); finalize Mesh Builder Fiji loader contract

---

## Imagery audit summary

**Figures we have (need to verify and locate):**
- Triangle wavefront snapshots (Bidomain V1 Phase 6 / Boundary speedup early experiments)
- CV ratio convergence plots (Boundary speedup)
- Storage-tank LAT contours (in `simulation/` outputs, generated April)
- Pump-speed Goldilocks scan (April harness output)
- TTP06 vs surrogate benchmark (`Surrogate/benchmarks/results/`)
- v4 train/val loss curve (Surrogate Session 28)
- NODE parity training curve (cardiac_ml `archive/runs_legacy/`)

**Figures we MUST CREATE before submission:**
- AP trace MHAS13 vs target (slide 5) — likely have raw data, need clean comparison plot
- APD-dVdt Pareto front (slide 5) — may need to regenerate from optimizer iteration 2 outputs
- PE/PP/Cattaneo/dual-evolving comparison diagram (slide 7) — schematic, hand-drawn or TikZ
- SHAP summary plot (slide 8) — run `scripts/analyze.py` against parity checkpoint

**[FIGURE NEEDED]** total: 11 across 6 research slides. Each must have scale bars where applicable. Notional/expected data must be marked.

---

## Open questions before drafting content

1. Approve 9-slide allocation (C×2, D×1, A×1, F×1, B×1)? Or move F to bullet-only?
2. Storage-tank work on slide 4 — frame as "John's tool extended" or "joint analysis"? Tone affects credit attribution.
3. Slide 6 surrogate framing — comfortable with explicit "8× slower" / "honest negative result" language? Or soften?
4. Should slide 9 mention the monthly report pipeline scaffolded today, or omit (it's meta)?
