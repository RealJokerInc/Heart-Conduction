# Ionic Model Parameter Optimization — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

Tuning ionic model parameters (MHAS13/TTP06/ORd) to match target CV and APD requires **multi-objective optimization** because fitting to a single action potential waveform is fundamentally non-unique. Many parameter sets produce identical APs but yield different tissue-level behavior (CV, membrane resistance, restitution).

### Pipeline status (updated 2026-03-17)

The Optimizer V1 pipeline is implemented end-to-end across both engines:
- **Cell fitter**: batched BayesOpt (qLogNEHVI), 10-param tier 2, hard constraints on dVdt/Vpeak/Vrest
- **Tissue fitter**: analytical CV∝√D_eff warm-start + secant refinement, engine-agnostic (mono or bidomain)
- **Bidomain support**: `tissue_runner_bidomain.py` wraps BidomainSimulation with D_eff→(D_i,D_e) decomposition at fixed ratio D_e/D_i=3.597
- **Best result** (iteration 2): APD=352ms (0.6% err), CV_L=14.6 (2.4%), CV_T=7.6 (1.7%), dVdt=106V/s (constrained <120)
- **Monodomain vs bidomain**: D_eff values agree within 6% (0.000447 mono vs 0.000422 bidomain), confirming D_eff≈D_i·D_e/(D_i+D_e) is a good approximation for insulated tissue

## ⚑ ARCHITECTURE DECISION 2026-07-02 — ionic + conduction MUST be tuned JOINTLY (never sequential)

**Decision (user):** ionic-engine tuning and conduction tuning must **never be sequential** —
they must be **joint / "parallel"**, because *ionic parameters must be fit with respect to the
whole tissue chip* (conduction, geometry, grid resolution), not a 0-D cell in isolation.

**What forced it.** The 2026-07-02 sequential chip fit produced garbage tissue records
(`D_long=D_trans=0.004` fallback, `CV=nan`) — a real failure, caused by the secant's ×4-up-bump
bug (since fixed → bracket-DOWN) *and* the frozen-θ structural problem below. (An earlier reading
of this as an r*/dx "source-sink discretization block → CV_T unreachable" was the seed of THE
MISTAKE; that interpretation is withdrawn — r*/dx is SCS-specific.)

**Why sequential is structurally broken.** `CV ~ sqrt(D·excitability(θ))`, `r*=D/CV`. The V1
pipeline fits θ→(APD,dV/dt) on a 0-D cell, then freezes θ and secants **D alone**→CV. For a
slow target the frozen-θ secant can only push D **down**, shrinking r* into the block. The lever
that hits slow CV *without* collapsing r* is **lower G_Na + higher D** — but **G_Na is shared**:
it sets dV/dt (cell objective, stage 1) AND source strength → CV/r* (tissue objective, stage 2).
No single stage can trade them. (Groenendaal 2015: single-objective ionic fitting is non-unique;
Pouranbarani 2019 fits AP **and** tissue CV jointly.)

**The fix (valid part):** one optimization over θ_ionic **+** D_long/D_trans, each candidate
evaluated on cell (APD/dV/dt) AND cable (CV_L/CV_T) on ONE model; the secant stage is removed, D
becomes a decision variable. This operationalizes the "Joint refinement" criterion. **⚠ The
architecture's `r*=D/CV ≥ k·dx` "resolvability constraint" is the piece that turned out WRONG
(THE MISTAKE) — it is SCS-specific, not a general fit gate. Everything else in
[`JOINT_TUNING_ARCHITECTURE.md`](../../../Optimizer/V1/JOINT_TUNING_ARCHITECTURE.md) that hangs
off r*/dx (feasibility gate, "resolution shell", the three-lock framing) is likewise suspect and
should be re-derived without it.**

**Load-bearing implementation findings (surfaced by audit, 2026-07-10):**
- **Two ionic backends (the P-1 blocker).** V1 measures dV/dt/APD on `cardiac_sim` (V5.4, via
  `cell_runner`/`batch_ionic`) but CV on `cardiac_core` (`cc_runner`). A cardiac_core kinetics axis moves
  CV but not the V5.4-measured dV/dt → unidentifiable. Joint tuning must first unify both onto cardiac_core.
- **The tissue solver uses the gate HOOKS, not `step()`.** cardiac_core monodomain's `RushLarsenSolver`
  drives the model via `compute_Iion`/`compute_gate_steady_states`/`compute_gate_time_constants`; it never
  calls `MHAS13Model.step()` (which carries extra Cai-dependent ICaL `constf1/constfCa` the hooks lack). So
  Na-kinetic multipliers must live in the hooks, and the 0-D cell driver must use the same hook path.
- **Method: constrained scalarization on a GP emulator** (extend `joint_refiner.py`), NOT tissue-in-the-loop
  (sims dominate cost); block region masked, not penalty-smoothed; dt accuracy-bounded (implicit CN, no CFL wall).

## ⚑⚑ THE MISTAKE (2026-07-10) — `r*/dx≥3` was wrongly applied; the last fit's "INFEASIBLE" is VOID

The joint fit baked in **`r*/dx ≥ 3`** (r* = D/CV) as a HARD FEASIBILITY constraint. That rule
is **SPECIFIC to the LBM specular-same-cell (SCS) wall / wavefront-curvature regime**
([[boundary_conduction_speedup]] / source_sink_mismatch_investigation) — it is **NOT** a general
resolution requirement for a monodomain or HBB CV fit. Applying it outside SCS was the error,
and it produced *every* "infeasible" verdict: of 4000 D-solved candidates, **37 actually hit
CV_T=2.6** and 538 hit CV_L=5.2, but they were discarded by the bogus filter. The dx /
coarse-dx / "excitability floor" excursion was all downstream of the same mistake.

**Nuance — do NOT over-correct.** r*/dx is not universally irrelevant; it is a real concern *for
SCS*. The correction is contextual: the **next run uses HBB**, where r*/dx does not apply, so we
drop it *there* — this is not a permanent rejection of r*/dx.

**WITHDRAWN conclusions (artifacts of the bogus constraint — do NOT cite as facts):**
"conductance-only is INFEASIBLE for CV_T=2.6"; "kinetics is NECESSARY (but not sufficient)";
"CV_T=2.6 is the wall / a physical excitability floor robust to dx"; the three locks (dV/dt /
g_Na floor / dx-refinement) framed as necessary; the P1a feasibility-map "gate = infeasible"
(its feasibility was *defined* by r*/dx≥3); and `presets/chip_hipsc_joint_INFEASIBLE.json` (VOID).

**Corrected next step:** re-run the joint fit on the **LBM engine + HBB wall** (not monodomain,
not SCS), **no r*/dx** (`require_resolved=False`, already wired), targeting only the real EP
numbers — CV_L=5.2, CV_T=2.6, APD=350, dV/dt, 2:1 anisotropy — and take whatever (θ, kinetics,
D) reaches them. Honest open question kept: whether a plain low-D conduction block exists in
LBM+HBB (if so, lower excitability is the lever) — but do NOT assert infeasibility until it is
run on the right engine without the filter.

## ⚑ Engine Tuner V2 — what is BUILT and VALID (branch `engine-tuner-v2-joint`, 2026-07-10)

The machinery is implemented + tested; only the r*/dx-driven *conclusions* were wrong.

**P-1 backend unification (VALID).** `tuner/cell_runner_cc.py` drives the cell AP as a 0-D
`run_monodomain` on a uniform strip → the SAME hook-based Rush-Larsen path the tissue-CV
runner uses, so dV/dt (cell) and CV (tissue) come from ONE cardiac_core model → a kinetics
axis is identifiable. **Parity finding:** the V5.4↔cardiac_core delta is PACING HISTORY,
not a formulation delta — APD Δ 9.35%@6 beats → 0.67%@20; V_rest matches 0.06% throughout.
Port is faithful; parity ≤1% at steady state. cardiac_core is the single reference.

**P0 secant + diagnostics (VALID; the r*/dx "resolution shell" is SCS-only, not a fit gate).**
The secant bracket-DOWN fix (`cc_runner.fit_D_for_cv`) brackets D *down* into the propagating
window and returns (NaN,NaN) on a genuine block — never the old fake `D=0.004`. Diagnostic
finding that stands: at low CV the failure mode is a real conduction *block* (source too weak),
and the high-D NaN is **sink overload** (`no_capture`, Vmax sub-threshold), not
over-depolarization. `cv_estimator.resolved_cv` (dx-ladder + r*/dx check) exists but its r*/dx
gate is SCS-specific — do not use it as a feasibility filter for HBB/monodomain fits.

**P1.5 kinetics (DONE).** MHAS13 gained per-instance Na knobs `tau_m/h/j_scale`,
`v_half_shift`, applied in the gate HOOKS (not `step()`), identity-safe (cardiac_core green,
PHAS13 untouched). τ_m genuinely moves CV and decouples dV/dt:CV (dV/dt more sensitive).

**P2 joint fit (DONE).** `tuner/decision_space.py` (one `apply(vector)→(model,mesh)`,
D_trans FREE) + `tuner/joint_fit.py` (constrained scalarization on a GP emulator).
Three load-bearing engineering findings: **(1)** the block region must be MASKED (NaN
never enters a CV GP target — fixes joint_refiner's 50.0-penalty cliff); **(2)** candidates
must **SOLVE D on the emulator** (bisection) to hit each CV target — random D never lands on
the thin "hits both CV" manifold; **(3)** GP inputs must be **NORMALIZED** (D~1e-4 vs
g_Na~1 span 4 orders → the RBF cannot fit CV otherwise). The fit SURFACES infeasibility
(names the binding lock) instead of a fake fit. For the real high-dim fit, train the CV GP
on a reduced CV-relevant feature set (g_Na + kinetics + D), not the full vector (open-Q8).

**P3 hand-off (VALID).** `chip.chip_mesh` exposes dx as a knob; the Tier-1 record schema gains a
`kinetics` block + per-axis D; the pre-existing `export_lab_preset(engine="lbm")` KeyError on
monodomain-only records is fixed. (The former "resolved 0.02 mm default" was an r*/dx artifact —
dx is just a knob now.) `run_joint_fit.py` + `--smoke` exercise the whole pipeline end-to-end.

**Production run (2026-07-10) — result VOID (see THE MISTAKE).** A full run was executed
(`run_joint_fit.py`, monodomain, 16 axes, 49 seeded points). It returned "INFEASIBLE", but that
verdict is an ARTIFACT of the r*/dx≥3 filter and is withdrawn — 37/4000 candidates *did* reach
CV_T=2.6. No valid θ* was produced; the record is void. The re-run (LBM+HBB, no r*/dx) is the
next step.

### The core degeneracy problem

The key degeneracy is **IKr/IKs compensation** during the AP plateau: these currents overlap temporally and can trade off against each other without changing the AP shape. This means single-AP fitting leaves IKr and IKs individually unconstrained, even though their ratio matters for rate-dependent behavior and drug response.

Breaking the degeneracy requires richer target data:
- **Multi-rate pacing** (restitution curves) -- probes rate-dependent kinetics
- **Paired waveforms** (control + selective channel block) -- isolates individual currents
- **Stochastic stimulation** (randomly timed pulses) -- probes nonlinear dynamics
- **Tissue-level CV** -- constrains GNa and the diffusion coefficient

### Method landscape

| Method | Strengths | Weaknesses | Best for |
|--------|-----------|------------|----------|
| **NSGA-II** (Pouranbarani 2019) | Pareto front shows explicit tradeoffs; handles 16 params; no gradients needed | No uncertainty quantification; 10,000 evaluations | Exploring AP vs CV tradeoffs |
| **Bayesian HMC** (Nieto Ramos 2023) | Full posterior distributions; identifies compensatory mechanisms; noise diagnostic | Needs differentiable model; tested on 5-13 params only | Parameter uncertainty quantification |
| **ABC-SMC** (Nieto Ramos 2023) | Likelihood-free; handles model inadequacy | Wider posteriors than HMC; computationally expensive | When likelihood is intractable |
| **GP Emulator** (Coveney 2021) | 10^5x speedup after training; tissue-level calibration | 500 simulations for training; 5-param model only | Restitution-based calibration |
| **GA** (Groenendaal 2015) | Simple, robust, handles discontinuities | No uncertainty; no gradients | Baseline rough search |
| **Gradient PO** (Zhang 2024) | Fast convergence for small param sets; two-waveform protocol | Local optima; needs smooth objective | Fine-tuning 6 conductances |

### Parameter subsets

Typically 6-16 maximal conductance scaling factors are tuned, not all model parameters:
- **CV-dominant**: GNa, D (diffusion coefficient)
- **APD-dominant**: GCaL, GKr, GKs, GK1
- **Plateau shape**: GCaL, GKr, GKs (the compensatory trio)
- **Resting potential**: GK1, GNaK

Pouranbarani 2019 tuned 16 conductances in TTP06 (our exact model) and found that adding membrane resistance (Rm) as an optimization objective improved intercellular accuracy by 89-96% at a cost of 20-30% worse AP RMSE.

### Key insight from Groenendaal 2015

Single-AP fitting with 9 parameters: parameters converge to wrong values despite near-perfect AP match (SSE < 0.01 mV^2). Adding stochastic stimulation drops prediction error by ~1 order of magnitude. Adding voltage-clamp data drops it by another order. Iterative GA refinement gains yet another order. Total: ~3 orders of magnitude improvement over naive single-AP fitting.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary method | BayesOpt with qNEHVI (Optimizer V1) | Multi-objective, sample-efficient, GPU-native via BoTorch |
| Objectives | APD + CV (dual) | Captures both cellular and tissue-level behavior |
| Parameter space | Maximal conductance scaling factors (6-16) | Standard approach across all reviewed literature |
| Target data | hiPSC-CM experimental measurements | MHAS13 model provides baseline; Optimizer V1 tunes to targets |
| Differentiability | PyTorch-based ionic models enable gradient computation | Opens path to HMC if BayesOpt proves insufficient |
| dVdt handling | Hard constraint (dvdt<120 V/s), NOT objective | Objective caused infeasibility at 60 V/s; 120 matches MHAS13 physiology |
| Tier level | Tier 2 (10 params) for production | Tier 1 (6) had 4/6 at bounds; tier 2 adds kNaCa, PNaK, g_pCa, VmaxUp |
| CV warm-start | Two-point secant (not analytical √D) | Secant uses real sim data; converges to <3% in 2-3 steps |
| Bidomain tissue | D_eff optimization with fixed D_e/D_i=3.597 | Reduces 2-unknown problem to 1-unknown; ratio from cv_shared.py |
| Reproducibility | seed=42 in config, passed to Sobol + torch | Ensures identical runs |

### Iteration 2 findings (2026-03-17)

**dV/dt constraint is essential but must be tuned carefully.** Setting dvdt_max<60 V/s with MHAS13 (baseline 132 V/s) made 72/74 candidates infeasible — the constraint was too tight. Relaxing to dvdt_max<120 V/s (physiological for TTP06 IK1 on Paci) gave 41/74 feasible, with final dV/dt=106 V/s.

**Tier 2 (10 params) helps but several still hit bounds.** g_Kr=3.0, kNaCa=2.5, g_pCa=2.5, VmaxUp=2.0, g_K1=0.3, g_to=0.3 all at their limits. The extra tier 2 params (kNaCa, PNaK, g_pCa, VmaxUp) participate in the solution, confirming they add value. PNaK=0.5 (downscaled) is a new knob not available in tier 1.

**Secant CV refinement is dramatically better than Newton.** CV_long: 50.6% initial overshoot → 2.4% in 3 secant steps. CV_trans: 20.5% → 1.7% in 2 steps. The two-point secant uses actual simulation data for the derivative estimate rather than the √D analytical approximation.

**The APD-dVdt tradeoff is a fundamental model property.** MHAS13 cannot simultaneously achieve APD=350ms and dV/dt=25V/s with conductance scaling alone. The hiPSC-CM model inherently has fast sodium kinetics relative to its AP morphology. Achieving dV/dt=25 V/s would require modifying the Na channel kinetics (gating time constants), not just the conductance.

## Open Questions

- Should dV/dt target be revised to match MHAS13's physiological range (~80-130 V/s) rather than the original 25 V/s target?
- Can multi-rate pacing (CL=500, 1000, 1500) break the remaining parameter degeneracies?
- Is membrane resistance (Rm) a better tissue-level proxy than direct CV measurement?
- Would a restitution-curve emulator (Coveney approach) provide sufficient accuracy?
- Can HMC scale to 10+ conductances given that Nieto Ramos only tested up to 13?

## Engine gotcha: cardiac_core mesh `chi` convention (effective-D meshes need chi=1.0)

**Effect (discovered 2026-06-30 during chip-fit implementation).** Building a chip
mesh via `create_cardiac_mesh(D=<effective diffusivity ~1e-3>, ...)` with the
**default `chi=1400`** produced **no propagation**: the stimulus over-depolarized the
source nodes to a non-physical Vmax ≈ 80–123 mV but the wave never launched (CV = NaN).
Isotropic *and* anisotropic, every engine config, was affected — it is a mesh-assembly
issue, independent of the anisotropy/MRT work. Setting **`chi=1.0`** → clean
propagation, CV = 59 cm/s (TTP06 at D=0.001).

**Why.** The monodomain FDM operator solves `χ·Cm·∂V/∂t = ∇·(D·∇V)`
(`cardiac_core/_monodomain/.../fdm.py`): the stiffness Laplacian `L` is built from the
mesh `D` **alone**, and `χ·Cm` appears only in the mass/time term. So the
**membrane-effective diffusivity is `D/(χ·Cm)`, not `D`.** `create_cardiac_mesh` stores
its `D` argument directly into `D_xx` (documented as a "diffusion coefficient") yet
defaults `chi=1400` — so an *effective* `D≈1e-3` is silently divided ~1400× →
effective diffusivity ≈ 7e-7 → CV ∝ √D drops ~√1400 ≈ 37× → below the source–sink
launch threshold (hence the pooled-but-non-propagating stimulus).

**Rule.** This is the Formulation-A/B firewall (`cardiac_core/conductivity.py`):
- Pass an **effective diffusivity** (cm²/ms) → set **`chi=1.0`, `Cm=1.0`** so `D/(χ·Cm)=D`. ← what `cc_runner`/`chip.chip_mesh` do.
- Pass a **raw conductivity** σ (mS/cm) → keep `chi=1400` so the effective diffusivity is σ/(χ·Cm). ← what `ConductivityConfig.bidomain(...)` does.

Documented in the `create_cardiac_mesh` docstring (`cardiac_core/file_format.py`).
LBM is unaffected by this (it maps D→τ directly, no χ division).

## Lateral boundary-speedup dimensionless number as a dt guide (2026-06-30)

Added a fourth tunable to the chip fit — **dt** — guided by the **lateral
boundary-speedup** dimensionless number from `boundary_conduction_speedup`. This is
**distinct** from the source-sink `dx/r*` number: the source-sink one is
geometry/CV-driven and comes in later (the reentry application); the boundary one is
BC + dt-driven and governs the **side-wall isochrone crescent**.

**The number** (`tuner/chip.boundary_number`, matches the LBM `tau_from_D`):
```
   β = D·dt/dx²     τ = 0.5 + β/c_s²  (c_s² = 1/3, so τ = 0.5 + 3β)
```
The regime is **BC-SPECIFIC** (bcs KNOWLEDGE 2026-06-25), and this is the load-bearing
caveat the 2026-06-30 "final check" surfaced:
- **HBB** (halfway bounce-back): **forward (slow-down) at ALL τ; NO speed-up ever**;
  |C| grows with τ.  ← **this is the BC cardiac_core's LBM actually runs.**
- **same-cell specular**: the only rule that FLIPS — τ≲0.67 inverse/**speed-up**,
  τ≈0.75 flat, τ≳0.84 forward.  **NOT implemented in cardiac_core's `_lbm`.**
- **neighbour-cell ("zero")**: flat (C≈0) at all τ.

CV is **inert** to it (GNa×0.25→×24 never flipped the crescent), so with D pinned by
the CV fit and dx fixed by the chip, **dt is the dial** for τ — but under the chip's
HBB wall, dt only scales a *forward slow-down*; **the wall speed-up is unreachable
until the specular/α-blend BC is wired into the engine** (see Engine gap below).

**Why a guide, not a fitted target.** There is **no converged scalar metric for the
curvature degree** yet (the crescent is read visually off the isochrone front —
`feedback_visual_front_over_derived_metric`), so dt is not optimized against a
target. The fit instead **records** which regime (D, dt, dx) lands in, per axis, and
the LBM baseline reports it at the actual run dt. Pick dt against the regime you want.

**Coupling caveat (why it can't be a clean post-hoc knob).** dt is shared — it sets
ionic-integration accuracy (APD, dV/dt) and CV (numerical), not just the wall
crescent. So any (θ_ionic, D) set is only valid at the dt it was fit at; changing dt
to chase curvature silently invalidates the EP match. For now dt stays at the fit
value and the guide is informational; a future curvature metric would let dt be
fit jointly (outer-dt / inner-re-secure-EP).

**Concrete chip numbers.** At the chip default (dx=0.1 mm, dt=0.01 ms) both Parker
baselines sit at **τ≈0.51** (hiPSC D≈2.5e-5: τ≈0.508; NRVM D≈8e-5: τ≈0.524). Under
the chip's actual **HBB** wall this is a (weak) *forward* crescent — **no speed-up**;
raising dt only *strengthens* the forward slow-down. (If/when specular is wired, τ≈0.51
*would* be deep inverse/speed-up — but that BC does not exist in the engine yet.)
Because β∝D, a **shared dt puts the two baselines at different τ** → dt is per-baseline.
Stored in each Tier-1 record under `boundary` (now carries `bc`, default `hbb`).

## Engine gap: cardiac_core LBM is HBB-only — the wall speed-up is unreachable (2026-06-30)

**Diagnosis (the "final check").** The chip-fit tuner uses the cardiac_core functional
API **correctly** (`run_lbm`); it is **not** miswired. The lateral wall speed-up is
unreachable because of an **engine gap**, not tuner misuse:
- `run_lbm()` has **no BC parameter** for the LBM path; `lbm()` ignores `mesh.boundary`
  for LBM (uses it only for the mono/bidomain `BoundarySpec`) and builds
  `LBMSimulation` with no wall argument; `LBMSimulation` is **HBB-only**
  (`apply_neumann_d2q9` = full bounce-back).
- The **specular / α-blend wall op (`apply_combined`) exists only in the
  `boundary_conduction_speedup` research scripts** — never ported into any engine
  `src` (not cardiac_core `_lbm`, not `LBM/Engine_V1/src`).
- HBB is forward at **all** τ → no dt produces a speed-up. So the boundary-speedup
  knob is presently moot on the chip.

**To implement correctly (engine work, mirrors the Phase-0 MRT pattern):** (1) port
`apply_combined` into `cardiac_core/_lbm/boundary/` + upstream `LBM/Engine_V1/src`;
(2) add a `boundary='hbb'|'specular'|'alpha'` (+`alpha`) selector to
`LBMSimulation.__init__`; (3) plumb it through `lbm()` + `run_lbm()`. Then
`boundary='specular'` makes the guide's speed-up regime reachable and dt controls a
real wall curvature. **Ionic-engine coupling** stands: the LBM step integrates
Rush-Larsen with the *same* `self.dt`, so moving dt to shift τ coarsens APD/upstroke —
bounding the usable dt range.

### RESOLVED 2026-07-02 — cardiac_core now handles specular, verified on the anisotropic chip

The engine gap above is **closed** (cardiac_core commits `40cd2ca` run_lbm/simulate
forward boundary+alpha [P1]; `1dda8f6` MRT/per-axis-anisotropic wall modes [C1]; plus
Phase 3–5 hardening). Verified:
- `wall_modes.py` = full family `neumann/hbb/specular_nextcell/specular_samecell/combined`
  (+`normalize_mode` accepting `ncs`/`scs`); `LBMSimulation(boundary=, alpha=)` selector;
  **collision-gate removed** — the overlay is post-stream (collision-agnostic), so a new
  `lbm_step_d2q9_mrt_wall` applies specular **on the MRT path** (step() line 218). So the
  previously-fatal **anisotropic (MRT) + specular** combo now works.
- Plumbed all the way out: `run_lbm(boundary=, alpha=)` forwards to `lbm()`.
- **Smoke (anisotropic MRT chip via `run_lbm`, D_long=1e-3/D_trans=5e-4):** specular runs
  without ValueError and **changes the field by 1.38 mV, concentrated at the wall rows**
  (1.38 wall vs 0.68 interior); `combined(α=0.5)` is intermediate (0.67) — correct
  α-blend monotonicity. 67 cardiac_core LBM/wall/MRT tests green.
- The guide (`chip.boundary_number`) now accepts the engine's canonical WALL_MODES names
  so the same `boundary=` string feeds both `run_lbm` and the guide (9 chip tests green).

**So the wall speed-up IS now reachable on the anisotropic chip** — set
`run_lbm(boundary='specular_samecell')` (or `'combined'`+α). It remains an *opt-in*: the
default chip baseline still runs HBB (neutral); a dedicated specular run is a deliberate
wall-curvature study. The dt guide's specular regime (τ≲0.67 speed-up) is now physical.

## Connections
- **Engines**: Monodomain V5.4 (simulation backend for optimization), Bidomain V1 (CV validation)
- **Related research**: hipsc_cm_ionic_models (MHAS13 is the tuning target), boundary_conduction_speedup (boundary effects must be accounted for in tissue-level CV fitting)
- **Pipelines**: Optimizer V1 (BayesOpt pipeline using V5.4)
