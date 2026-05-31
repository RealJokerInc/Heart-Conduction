# March-April 2026 — Scout Output

> Reporting period extended at user request 2026-04-28.
> Raw activity inventory from git log + active question IDEALOGs.
> Friction observations logged in `../../Research/Active/monthly_report_pipeline/IDEALOG.md`.

## Reporting period
2026-03-01 → 2026-04-28 (will extend to 2026-04-30 at submission)

---

## Active project candidates (consolidated March-April)

### A. Surrogate pipeline — two-month arc with multiple pivots
**March (build-out)**
- Mar 13: Architecture docs begin (two-component neural surrogate for bidomain)
- Mar 19: Session 7 — ionic architecture major pivot
- Mar 20: Session 8 — training strategy + pipeline audit
- Mar 21-22: **Data generation pipeline operational** — `BatchGenerator` with `torch.compile` (47,000× speedup vs sequential), 12 tiers, 270K steps/s on CPU; gate_inf/tau as columns (23→47 cols, post-hoc 0% overhead vs 40%); chunked processing prevents 1.2 TB OOM
- Mar 23-24: Architecture v2 + blueprint (Sessions 12-13)
- Mar 26-30: Model implementation + architecture refinement
- Mar 30-31: Architecture v3 from Layer 0 (Sessions 15-16)

**April (training + pivots)**
- Apr 2-3: v3 complete; full training pipeline Phases 1-6 (data cache, trainer, CLI, shard streaming)
- Apr 4: dt curriculum + TBPTT + warm restarts
- Apr 6: A4 discrete training failed at 30k steps → **Neural ODE pivot**. IonicNODE with `odeint_adjoint`
- Apr 9: NODE pivot validated, dense MLP replaces VoltageAttention (commit `8f191f77` SHA-pin)
- Apr 19-20 (Sessions 27-28): v4 implementation — 7,891 params **overfits T1** (v3 at 1,444 params worked on same split)
- Apr 21 (Session 29): TTP06 benchmark — surrogate Euler 8× **slower** than classical solver. **Major pivot: hybrid bidomain surrogate** — keep classical TTP06 ionic, replace **elliptic step** (94% of wall time). Dual CNN tower architecture proposed (Vm + φ_e + cross-attention). v4 ionic kept for CPU deployment + parameter optimization roles only.

### B. cardiac_ml harness — completed in April
- Apr 19: Phases 1-3 (env, package skeleton, Trainer + MLflow + callbacks). 4 audit rounds + 3 blueprint-revise passes. 57 tests
- Apr 20: Phase 4 NODE parity **MET** at epoch 1 (`val_loss=0.00835 < 0.0088` threshold). Phase 5: Optuna sweep + SHAP + diffusion stub reusability. Surrogate runs archived. **80 tests**, 11/11 criteria. Project-wide harness ready for downstream consumers (diffusion ResNet, BayesOpt wrapper)

### C. Boundary conduction speedup — diagnostics → PI-collaborative storage-tank model
**March (boundary diagnostics)**
- Mar 5: Boundary diagnostic data/images, mixed spectral solver (DCT+DST), spiral wave script, S1-S2 example
- Mar 15: Anisotropic test — 2:1 conductivity ratio produces sharper triangular wavefronts at boundaries. Conductivity sweep — edge lead scales with √D_eff. Monodomain control gives flat wavefront (so triangle is bidomain-specific). Triangle "merger" experiment showed merger does NOT happen.
- (Result so far: bath-coupled CV ratio 1.071 → theoretical 1.131 confirmed via mesh convergence)

**April (PI-collaborative)**
- Apr 24: John shared 2D storage-tank toy model (Colab); set up `simulation/` at repo root. Tank model exhibits boundary speedup
- Apr 25: Numpy vec rewrite (~200× faster than OOP, bit-equivalent). **Three-axis decomposition**: BC × directionality × pump rule each independently shape LAT. Reflection-BC test — boundary operator dominates pump rule. Bidirectional pipes test — directionality is third axis (kills "camel toe" shape)
- Apr 26: Experiment harness (`simulation/configs.py` + `experiment.py`); pump-speed Goldilocks zone (camel-toe peak at max_pump=10)
- Apr 28 (today): Hydrostatic Bernoulli derivation of John's per-cell law (Torricelli); single-cell LBM mechanism analogy; axiom restructure

### D. Optimizer V1 + Mature hiPSC-CM (MHAS13)
- Mar 14: Triangle merger visualizations + isochrone map
- Mar 15: **Optimizer V1 pipeline designed and first implementation** (BayesOpt for V1, HMC deferred). 10× speedup via batching + subcycling + analytical CV. PHAS13 (Paci2013 renamed) added
- Mar 16: **MHAS13 created** — matured hiPSC-CM via TTP06 IK1 injection. Quiescent V_rest=-83.7mV. APD 347ms (target 350) via optimizer pipeline; APD 349ms / CV 15.8 cm/s through bidomain pipeline. First optimizer run — APD matches but dVdt reveals tension
- Mar 17: Iteration 2 — constraints + tier 2 + secant CV + seeding. APD-dVdt tradeoff = fundamental model property. **IKr/IKs compensation = core degeneracy**. Maturation pathway validated through bidomain pipeline

### E. Bidomain V1 hardening — March audit + new solvers
- Mar 5: Phase 6 complete; GPU-native spectral solver
- Mar 8: **Two-round audit cleanup pass** — fixed critical bugs (dt guards, spectral eigenvalues, PCG, Chebyshev), medium-severity (ionic ordering, types, state clone), R2-H1 (FDM cross-derivative factor of 2 + sign pattern). D_eff validation tests use exact discrete eigenfunctions
- Mar 8: **Diffusion solver suite expansion** — added Jacobi parallel, semi-implicit (Forward Euler parabolic + implicit elliptic), IMEX SBDF2 (2nd-order BDF self-starting), explicit RKC (Chebyshev-stabilized, no parabolic solve). Renamed `decoupled.py → decoupled_gs.py`
- Mar 13: Fixed PCG premature termination + Dirichlet boundary RHS in bidomain solvers
- Mar 14: **Mehrstellen 9-point stencil** added to FDM discretization + spectral eigenvalues; wired through `BidomainSimulation` solver factories. Validated bidomain-insulated matches monodomain.

### F. Bidomain parabolic-parabolic — literature breakthrough → endgame reframe
- Mar 20: Question scaffolded — observed triangular wavefront artifacts at tissue-bath boundaries during boundary speedup work. Root cause: elliptic propagates extracellular info instantaneously
- Mar 20: **4 parallel literature agents → critical finding**: published "PP bidomain" is NOT what we thought. The Bourgault 2009 / Pavarino & Scacchi 2011 "PP" is the original (φ_i, φ_e) form — same physics as PE, different variables. There is **no published ε·∂φ_e/∂t regularization used as a computational method** (only proof technique). The only published model with finite extracellular propagation speed is the **Cattaneo/hyperbolic bidomain** (Rossi & Griffith 2017)
- Apr 23: **Endgame reframed** to dual-evolving bidomain LBM. Pulled Rossi-Griffith 2017, ESAIM M2AN 2013, Bishop-Plank 2011. **Insight**: Rossi-Griffith's "ParabolicParabolicHyperbolic" (τ_i ≠ τ_e) is structurally 2-unknown but physically **1-DOF-with-memory** — only V has time derivatives. User wants 2 independent dynamical DOFs, requiring the 6 (τ_i − τ_e) terms identified in `HYPERBOLIC_HYPERBOLIC_ANALYSIS.md`

### G. Research environment optimization — completed in March
- Mar 11-19: Major infrastructure work
  - Three-document architecture (KNOWLEDGE / IDEALOG / WHITEBOARD) crystallized
  - 16 skills implemented: `/reason`, `/blueprint`, `/save-session`, `/audit`, `/verify`, `/build-fix`, `/strategic-compact`, `/quicksave`, `/quick-implement`, `/reason-end`, `/blueprint-revise`, etc.
  - PreCompact hook, settings.json polish
  - tmux workspace integration, glow style, dynamic glow width, per-question WHITEBOARD
  - Skill catalog in KNOWLEDGE.md, parallel startup, background writes catalog
- Status: implementation complete, real-world testing ongoing

### H. Engine consolidation — Phase 0 in March
- Mar 17: **Phase 0 complete** — unified file format + `monodomain()` / `bidomain()` / `lbm()` API wrapper. 34 tests
- Phase 1 (extract shared ionic/mesh) not started

### I. Mesh builder — April scaffold + Fiji pivot
- Apr 22: Question scaffolded. Fiji 2.16.0 installed and validated headless: pure binary-mask polygon fill (no anti-aliasing). Tied to John's 4/16 "Diffusion Speed Up" email

### J. LBM-EP — April reopen
- Apr 19: Reopened (was complete `lbm_cardiac` 2026-03-16). New framing: production-quality solver assessment (anisotropy, boundary artifacts, tuning). Scaffolding only

### K. Monthly report pipeline (this question)
- Apr 28: Scaffolded; Zimmerman format V1 spec extracted

### L. LBM V1 — Feb-Mar finish
- Mar 5: All 8 phases complete, 24/24 tests pass. Committed but most work was Feb

---

## Inactive in reporting period
- engine_consolidation (Phase 1+ since March, no further activity)
- geometry_induced_pacemaking (scaffolded, no work)

## General laboratory activities (candidate)
- **Weekly lab meetings** (Friday 10:30) — paper discussions attended (1/9, 2/19, 3/5, 3/19, 4/3, 4/24)
- **PI collaboration on storage-tank model** (could fold into project C or break out)
- **PI grant — bidomain simulation models** (Jan 20 one-pager, ongoing context)
- **Oxford application** — recommendation letter request resolved Feb 22, application due March 3
- **MEMORY infrastructure** — research_environment_optimization implementation
- **Monthly report pipeline scaffold** — this work itself

## PI-visible context (relevant to deck framing)
- John's emails this period: "Structure Dependent Speed Up" (2/12), "Pressure Gradient" (2/24), "Geometry Dependent Action Potential Conduction" (3/25, Flavio Fenton reference), "Great Review on Cardiac Conduction" (4/1), "V?" + "Ionic Surrogate Architecture" exchange (4/2-3, John asked for PPT/PDF explainer), "Diffusion Speed Up Simulation" (4/16, John shared his Colab tool), "Storage Tank Code" (4/24, John's tank simulator)
- John's repeated theme: he wants visual PowerPoint/PDF explainers, geometry-dependent conduction is his anchor question

---

## Triage call needed (you decide)

12 candidates → must fit 7-14 slides total. My recommendation:

**Headline projects (1-2 slides each, on Summary + research slides)**
- C. Boundary conduction speedup (PI's core interest — foreground this) — 2 slides
- A. Surrogate pipeline (long arc, ends with hybrid pivot — high interest) — 2 slides
- D. Optimizer + MHAS13 (PI grant material from Jan) — 1 slide
- F. Bidomain parabolic-parabolic (literature breakthrough) — 1 slide

**Secondary projects (1 slide each, on Summary)**
- E. Bidomain V1 hardening (technical infrastructure, March) — 1 slide
- B. cardiac_ml harness (completed deliverable) — 1 slide

**General lab activities (Summary slide bullet, no own slide)**
- G. Research environment optimization (meta tooling)
- K. Monthly report pipeline scaffold
- I. Mesh builder + J. LBM-EP scaffolds → "infrastructure prep for next month"
- Lab meetings + PI collaborations

**Slide budget: Title (1) + Summary (1) + C×2 + A×2 + D×1 + F×1 + E×1 + B×1 + Future (1) = 11 slides** ✓ in 7-14 range

Open calls for you:
1. Approve / adjust the headline vs secondary triage
2. Should H (engine consolidation Phase 0) get its own bullet or fold into general infrastructure?
3. L (LBM V1 phases done) — was March 5 commit only; mostly Feb work. Mention or omit?
4. The textbook is gitignored; was it active in March-April? If so, where does it go?
