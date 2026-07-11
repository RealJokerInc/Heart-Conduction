# Joint Ionic + Conduction Tuning — Architecture

> ⚑⚑ **CORRECTION (2026-07-11) — READ FIRST.** This doc makes **`r*/dx ≥ 3`** a load-bearing HARD
> constraint (the "resolution shell", the feasibility gate, "lock-3", the resolved-not-fit split).
> **That is WRONG.** `r*/dx≥k` is **SCS-specific** (LBM specular-same-cell wall / wavefront
> curvature) — NOT a general resolution requirement for a monodomain or HBB CV fit. Enforcing it in
> the joint fit produced a **false "INFEASIBLE"** (37/4000 candidates actually hit CV_T=2.6, all
> filtered out). Everything below that hangs off r*/dx (the resolution shell, the feasibility gate,
> the three-lock "necessary" framing, the "CV_T unreachable" conclusion) is **withdrawn**. The
> code already supports dropping it (`joint_fit.refine_joint_cc(require_resolved=False)`). The
> corrected plan: re-run on **LBM + HBB with no r*/dx**, targeting only CV_L/CV_T/APD/dV/dt/2:1. The
> non-r*/dx machinery (backend unification, GP emulator, D-solve, constrained scalarization) is
> sound. See `Research/Active/ionic_model_optimization/KNOWLEDGE.md` → "THE MISTAKE".

Created: 2026-07-02
Revised: 2026-07-02 (architecture deep-dive: attack-all fit, FIT/RESOLVE split, constraint graph)
Revised: 2026-07-10 (audit iter 1 → fixes: emulator-accelerated method [joint_refiner], implicit-solver/no-CFL,
         r*/dx≈3 ladder floor, anisotropy relations are SOFT not hard-tied, kinetics sequenced as P1.5)
Revised: 2026-07-10 (audit iter 2 → fixes: P-1 backend unification [cell=cardiac_sim vs tissue=cardiac_core],
         real kinetics hook [phas13/gating.py, shared], emulator open-risks named, g_Na floor ≤0.17,
         slow-corner dx≈0.02mm, dV/dt target CONTESTED vs README, free-D_trans caveat)
Owner: `ionic_model_optimization` (build) — applies to Engine Tuner → cardiac_core
Supersedes: the V1 sequential cell→tissue fit (`run_chip_fit.fit_chip_baseline`)
Revised: 2026-07-10 (audit iter 3 → CONVERGED: 0 critical / 0 high. All iter-1/2 fixes verified true
         against code; folded 4 residual minors: P-1 batching-throughput caveat, cc_runner None→NaN
         guard, dx budget 0.02mm reconciled, tissue-backend framing)
Status: ARCHITECTURE — AUDIT-CONVERGED (0 crit/0 high over 3 iters); pre-implementation. Blueprint next.

> **Decision (user, 2026-07-02):** ionic-engine tuning and conduction tuning must **never be
> sequential**. They must be tuned **jointly ("attack all at once")**, because ionic parameters
> must be fit *with respect to the whole tissue chip* (conduction, geometry, resolution) — not to
> a 0-D cell in isolation.

---

## 1. The failure that forced this

Fitting the two Kit-Parker baselines (`run_chip_fit`, 2026-07-02) produced **garbage tissue
records**: `D_long = D_trans = 0.004` (secant fallback), `CV = nan`, on BOTH baselines. A cable
D-sweep at the chip config (`dx = 0.1 mm`, `dt = 0.02 ms`, fitted NRVM θ):

```
   D (cm²/ms)   CV(cm/s)
   1.0e-3       nan     (above the usable window)
   1.0e-4       6.90    ✓
   5.0e-5       5.00    ✓   r* = D/CV ≈ 100 µm = dx  → r*/dx ≈ 1
   2.5e-5       nan     ✗   r*/dx < 1  → source-sink discretization block
```

Propagation survives only in a narrow window (~5e-5–1e-4, CV ~5–7 cm/s). Below it,
`r* = D/CV < dx` → the wavefront's electrotonic foot is sub-grid → **conduction block** (the
`dx/r*` control parameter from `source_sink_mismatch_investigation` /
`boundary_conduction_speedup`). The chip targets straddle/fall below the window — hiPSC
**CV_T = 2.6 cm/s is unreachable at dx = 0.1 mm**.

## 2. Why the sequential architecture *causes* it

V1 pipeline (`fit_chip_baseline`): Stage 1 (cell) BayesOpt `θ → APD, dV/dt` on a 0-D cell (no
conduction); Stage 2 (tissue) **θ frozen**, secant on **D alone** → CV on a 1-D cable.

CV is a function of **both** θ and D, and `r* = D/CV`:

```
   CV  ~  sqrt( D · excitability(θ) )        r* = D / CV
```

For a slow target the frozen-θ secant can only push **D down**, shrinking r* into the block. The
lever that hits slow CV *without* collapsing r* is **lower G_Na + higher D** — but **G_Na is
shared**: it sets dV/dt (a *cell* objective, stage 1) **and** source strength → CV/r* (a *tissue*
objective, stage 2). No single stage can trade them. Sequential is not a shortcut — it is
*structurally incapable* of reaching the slow, low-D regime. (Groenendaal 2015: single-objective
ionic fitting is non-unique; Pouranbarani 2019 fits AP **and** tissue CV jointly.)

## 3. The organizing idea — one equation, attack all *physical* arguments at once

Everything reduces to making this hold:

```
   CV_target   =   Φ( θ_ionic ,  D ,  dx ,  dt ,  Cm ;   model ,  observables )
     └─LHS─┘        └──────────── knobs ────────────┘     └── structure ──┘
```

The system is **coupled**, so the correct object is not a *ladder* of single-argument attacks —
it is the **full coupled decision space, fit jointly** (high-dimensional). Factoring into
independent stages (V1) was the bug. But the arguments are not all the same *kind*, and the
architecture must encode that (next section).

## 4. FIT vs RESOLVE — the physical / numerical split (the resolution shell)

```
   PHYSICAL params → FIT jointly :  θ_ionic  +  Na kinetics  +  D_long, D_trans  +  Cm
   NUMERICAL params → RESOLVE    :  dx , dt
```

**Do not put dx/dt in the free decision vector.** The discretized measurement is
`CV_num = CV_phys + ε(θ,D,dx,dt)`, with ε → 0 as dx,dt → 0. If the optimizer may move dx to hit
the target, it hits it **using ε** — a grid-error fudge that evaporates when the real chip runs.

**But the CV *calculation* SHOULD be dx/dt-aware** — it must *remove* ε, not ignore it. The right
form is a **convergence-aware estimator**: run a small dx-ladder (fixed θ, D, dt; vary *only* dx)
and extrapolate the trend to the resolved limit. CV then *varies with dx* (uses several) yet dx is
**never a decision variable** — it moves to **cancel** ε, not to **achieve** a target. The
honest/fudge line is exactly *whether dx moves to remove ε or to exploit it*.

```
   CV_hat(θ,D)  =  extrapolate[ CV_num(·,dx₁), CV_num(·,dx₂), CV_num(·,dx₃) ]  → CV_resolved
```

Precedent + honesty: the project already reasons this way via **mesh convergence** — the Bidomain
Kléber ratio is quoted as `1.0714 @ dx=0.025` *converging toward* the analytic target
`1.131 = √((σᵢ+σₑ)/σₑ)` under refinement (it is ~5% short at that dx, not a converged value). What's
new/**proposed** here is the *automatic* extrapolating estimator; it is not existing code, and it is
only valid where the trend is monotone/smooth (see floor below).

**Why this is legitimate now (and wasn't before): the tissue dimension is known.** The chip is a
*real* 16 mm tissue, so `dx = L/N` and `r* ≈ 130–160 µm` are physical scales → `r*/dx` is a
*physical resolvability ratio* you can compute and certify. Under-resolution is detectable; the
honest grid has a right answer (`dx ≲ r*/3 ≈ 45 µm`). CV stops being "a number I tuned" and becomes
a physical prediction.

**Limits of the estimator (why it doesn't rescue the slow corner):**
- You **cannot extrapolate *through* a block** — below `r*/dx ≈ 1`, `CV_num → nan` (a bifurcation,
  not a finite offset); there is nothing to extrapolate.
- **The usable floor is `r*/dx ≈ 3`, not 1.** Between 1 and 3 the wave propagates but the curvature
  coefficient is still *corrupted and even sign-inverts*, and CV0 sags with dx (source_sink S0b:
  62→49 cm/s) — so rungs in `1 < r*/dx < 3` extrapolate a *wrong trend*. Every ladder rung must sit
  at `r*/dx ≳ 3`.
- ε does **not** collapse onto one function of r*/dx (source-sink: same r*/dx, different CV by the
  CV-route vs dx-route) → ε is multi-dimensional. Extrapolation is only valid along a **controlled
  single-variable path** (vary dx alone, θ/D/dt fixed), not across the (θ,D) space.
- **The solver is implicit** (`run_monodomain` defaults to `crank_nicolson`, unconditionally
  stable) — so there is **no CFL blow-up wall**; `dt` is bounded by *accuracy*, not stability, and
  gets its own (cheaper) convergence ladder (*proposed — no dt-ladder exists in code yet, and
  `config.py`'s `dt` comment still says "CFL: dx²/4D", stale for the implicit solver*). NB: the
  high-D `nan` in the §1 sweep is therefore **not** a CFL blow-up — it is CN
  accuracy / over-depolarization (warned of in `config.py`) or a CV-measurement failure (distinct
  from the *low-D* wave-death that `cc_runner`'s chi-comment warns of). Settling it is a P0
  discriminator (§9).
- Any correction is only as good as its ε-model; an optimizer **hunts the residual**. Extrapolate
  where ε is smooth and cheap to model; **resolve** where it's discontinuous/untrustworthy. The
  known tissue dimension tells you which side of that line you're on.

## 5. The decision space — what's in the vector

`θ_ionic` is the **ionic** slice: today, tier-N *multiplicative scaling factors* on maximal
conductances/flux rates (`PHAS13_REGISTRY`: g_Na, g_CaL, g_Kr, g_Ks, g_K1, g_to [t1] + kNaCa, PNaK,
g_pCa, VmaxUp [t2] …), applied as `params.g_Na *= θ['g_Na']`. It is **separate** from `D` (mesh
diffusion), `Cm` (membrane), and the numerical dx/dt.

**Conductance scaling is amplitude-only, and that is the core limit.**

```
   I_Na(t) = g_Na · m³(t)·h(t)·j(t) · (V − E_Na)
             └amplitude┘  └── shape/timing: τ_m,τ_h,τ_j — FROZEN ──┘

   dV/dt_max  ∝  peak I_Na  ∝ g_Na · (gate product at peak)
   CV-source  ∝  I_Na to charge the downstream sink ∝ g_Na · (gate-trajectory integral)
```

`dV/dt ∝ g_Na` (linear) and `CV ∝ √(D·g_Na)` (sublinear), so the two are not on a literal ray — with
`D` free the reachable (dV/dt, CV) set is a 2-D region. The point is directional: **amplitude scaling
provides no direction that moves dV/dt without moving CV's excitability term** (both go through the
*same* `g_Na·gate-trajectory`); only the *ratio* dV/dt : charge-to-sink is what we need to change,
and that ratio is fixed by kinetics, which scaling can't touch. Real numbers (verified against
`config.py`: g_Na published 3.6712302, scaling bounds (0.5, 2.0)): the slow-hiPSC corner needs
≈0.17× ≈ 0.62 absolute — *below* the 0.5-multiplier floor. So two things are both true and must be
stated together: (a) admitting 0.62 absolute needs the floor lowered to **≤ 0.17 (≈0.15)** — a 0.2
floor still excludes it (§9 lock-2); but (b) widening **alone, with dV/dt still pinned at 110,
doesn't help** — dropping g_Na drops dV/dt in lockstep. **It is the dV/dt target that must also move
(§9 lock-1); the g_Na bound and dV/dt target are a coupled pair.**

**Kinetics unlock it.** Adding Na-kinetic axes (scale `τ_m`; if needed `τ_h/τ_j`, `V_half` shifts)
reshapes `I_Na(t)` → decouples peak-rate (dV/dt) from charge-to-sink (CV). Cost is real and specific:
the Na kinetics live in **`cardiac_core/ionic/phas13/gating.py`** as module functions
(`INa_m_tau`, `INa_m_inf`, `INa_h_*`, `INa_j_*`) — **mhas13 imports them from phas13** (mhas13 has no
`gating.py`), and the ABC hooks are `compute_gate_time_constants` / `compute_gate_steady_states`
(`ionic/base.py`). Two consequences: (i) editing `phas13/gating.py` **also mutates PHAS13**
(shared-module coupling) → the τ multipliers must be *parameterized per-model instance*, not
hard-edited (or mhas13 gets its own gating override); (ii) this is a model change, not a registry
row. It stays within `cardiac_core/ionic/` — **read-only `Monodomain/Engine_V5.3/` is never touched.**

**⚑ FOUNDATIONAL BLOCKER — two ionic backends.** The pipeline currently measures its two observable
families on **different model implementations**: the cell fit (`cell_runner`/`batch_ionic`) runs the
**`cardiac_sim` (V5.4)** phas13 model; the tissue path's target runner (`cc_runner`) uses
**`cardiac_core`** (today `joint_refiner` still rides the legacy `tissue_runner`/`cardiac_sim` — the
rewire to `cc_runner` is part of P2, §Migration).
So dV/dt/APD and CV come from *different code*. A cardiac_core kinetics axis would shift emulated CV
yet leave the V5.4-measured dV/dt **unchanged** → the very axis meant to decouple them is *invisible
to the objective that identifies it*, and warm-starting θ across stages is only partial (each backend
has its own non-`PHAS13_REGISTRY` defaults). **Unifying both observables onto one backend
(cardiac_core) is a prerequisite for any joint/kinetics fit — step P-1 (§9).**

**The vector is heterogeneous** — ionic conductances, ionic kinetics, tissue D, (optionally) Cm:
distinct subsystems, distinct "apply" paths, today in disconnected homes (`PHAS13_REGISTRY`;
`TISSUE_PARAMS` — **live**, used by `tissue_fitter`/`joint_refiner` but not yet by `cc_runner`;
kinetics: none). **Architectural ask: one decision-space registry** — each axis declares
`{subsystem, bounds, apply_fn}` behind a single `apply(vector) → (scaled_model, mesh)` — **which
presupposes the P-1 backend unification** (otherwise "one apply()" spans two models).

## 6. Identifiability — the constraint graph + observables (pin the high-dim fit)

High reachability (attack-all) ⇒ **under-determined**. Pin it two ways:

```
   pin the fit  =  ASSERT known-true relationships   +   MEASURE new observables
                   (constraint graph — cheap/free)       (restitution, block — expensive)
```

**Provide boundaries for parameter *pairs* we know are true.** Our physical knowledge is low-order
(two/three-body laws), so pairwise/triple constraints are exactly where certainty lives:

| Confidence | Relationship (pair/triple) | Encode as |
|---|---|---|
| Definitional (exact) | Resolvability: `r* = D/CV ≥ k·dx` (k≈3 to *resolve* source-sink) | hard inequality |
| Definitional (exact) | `λ = CV·APD` | identity (diagnostic; a derived quantity, not a fit knob) |
| Numerical convergence — **not** a decision constraint | dx, dt → resolved via the §4 ladders (solver is implicit CN → **no CFL stability wall**; dt is accuracy-bounded) | resolution shell, not the optimizer |
| Physiological (moderate) | IKr/IKs ratio bound; g_K1 ↔ V_rest; Rm ↔ conductances | inequality bounds |
| Leading-order / assumption → **SOFT** | `CV ∝ √D`; 2:1 anisotropy ratio (≈2.1±0.8, cross-construct); **and everything derived from them: `D_trans = D_long/ratio²`, `CV_T = CV_L/ratio`** | soft prior / warm-start ONLY |

- **The key correction (self-consistency):** `D_trans = D_long/ratio²` and `CV_T = CV_L/ratio` are
  **not** "by construction" — they are *derived from* `CV∝√D` and the 2:1 ratio, both of which sit in
  the soft row. By the doc's own "stiffness = confidence" rule they must be **soft too.** So
  **`D_trans` should be a *free* decision variable fit to `CV_T`**; the 2:1 relation is a
  *warm-start*, not an equality tie. ⇒ **no free dimension reduction** from tying — an earlier
  "12→11" claim is withdrawn. **Caveat (teeth):** freeing `D_trans` only *matters* once `CV_T` is an
  *independent* target. Today `PARKER` **derives** `CV_T = CV_L/ratio` and `chip_mesh` hard-codes
  `D_trans = D_long/ratio²` (`chip.py:25,29,48`), so a free `D_trans` fit to a *derived* `CV_T` just
  relands near `D_long/4`. The freedom is real only when `CV_T` becomes an independent measured
  target — see open-Q7.
- **Inequality bounds carve** the search to the physically-real submanifold.
- **Stiffness must match confidence.** Hard-encoding a merely-approximate relation *fences the true
  solution out of the search* — worse than not constraining. Only the definitional rows are hard;
  everything touching `CV∝√D` or the anisotropy ratio stays soft.

**Observables must grow with the decision space.** `{θ + kinetics + D + Cm}` against only `{APD,
dV/dt, CV_L, CV_T}` is wildly under-determined. Every kinetic axis needs an observable that
constrains kinetics — **CV-restitution** (multi-rate; also breaks the IKr/IKs degeneracy — two for
one), upstroke *shape* (not just peak dV/dt), selective block. The more you can *assert* (constraint
graph), the less you must *measure*.

## 7. Method — constrained scalarization, evaluated on a surrogate (extend `joint_refiner.py`)

**Formulation** (the *what*):

```
   minimize   aggregate AP-morphology error
   over       { θ_ionic, (kinetics), D_long, D_trans, (Cm) }     ← physical, high-dim; D_trans FREE
   s.t.       CV_L, CV_T within tol      (measured at converged dx via the §4 ladder)
              r*/dx ≥ k   (per-axis, r* = D/CV_resolved; transverse is the binding axis —
                           D_trans < D_long ⇒ smaller r*.  Do NOT encode r*_trans = r*_long/2:
                           that assumes the withdrawn CV∝√D + 2:1 tie, §6)
              V_rest, V_peak, dV/dt bounds
              + soft warm-starts: 2:1 anisotropy, CV∝√D  (priors, NOT hard ties — §6)
```

Constrained scalarization beats 4-objective qNEHVI for **identifiability/reporting** reasons — it
**surfaces infeasibility explicitly** ("no (θ,D) hits CV_T at r*/dx≥k within bounds → refine dx /
revise dV/dt / add kinetics") instead of returning silent dominated compromises. It reuses
`cell_fitter._check_constraints` **only for the AP/cell bounds** (converged, `dvdt_max_upper`,
`v_peak_max`, `v_rest_range`); the **CV-tolerance and r*/dx constraints are new code.** (The
hypervolume saving over qNEHVI is minor and is *not* the reason — see cost.)

**Evaluation** (the *how* — the real cost lever): **the dominant cost is simulations, not the
acquisition.** Direct *tissue-in-the-loop* with a per-candidate dx-ladder (~3 rungs × 2 axes ≈ 6
sims/candidate × hundreds of evals × 12–15 dims) is **intractable** — an earlier draft wrongly
called it "cheaper." `Optimizer/V1/tuner/joint_refiner.py` provides the **pattern** to fix this — a
**GP emulator** trained on real sims, optimized on the surrogate, top-~5 validated on real sims —
and this design **extends** it (rewire off `tissue_runner.run_cv_measurement` to `cc_runner`; add the
r*/dx constraint; train on resolved CV; NSGA-II → constrained scalarization). But it is **not
"already solved"** — the emulator carries three open risks the build must design against:

- **Cost is not free.** `_build_training_data` already runs ~3 sims/point (1 cell + 2 CV) → ~150 sims
  for 50 points *today*; adding the §4 resolved-CV ladder (≥3 rungs × 2 axes) makes it **~300+
  tissue sims at *finer* dx** (each far costlier). The "sim cost lives in the shell" (§8) is where the
  budget actually goes — it must be budgeted, not waved off. Warm-start the emulator from the saved
  cell fits; consider active-learning refill instead of a fixed 50-point design.
- **Out-of-support queries.** `joint_refiner` trains on a *thin* manifold (Pareto-front × local D
  perturbations) yet the constrained search runs the **full box** — the GP is queried outside its
  training support, worse at 12–15 dims. Training design must cover the searched region (open-Q8).
- **The block is a non-stationary cliff for the GP.** `joint_refiner` maps blocked CV to a flat
  `50.0` penalty → an RBF `SingleTaskGP` sees a discontinuity **exactly at the CV_T=2.6 block edge**,
  and the r*/dx constraint is computed as `D/CV_emulated` from that *least-trustworthy* region.
  The infeasible region should be **masked/classified** (a feasibility classifier), not smoothed by a
  penalty the GP interpolates through (open-Q8). **Interface note:** on the `cc_runner` rewire,
  blocked propagation returns **`NaN`** (not `joint_refiner`'s current `None`), so the inherited
  `cv if cv else 50.0` idiom breaks (`NaN` is truthy → `abs(NaN−target)=NaN` poisons the GP) — guard
  with `isfinite`.

## 8. The three-leg architecture

```
   ┌ RESOLUTION SHELL ─ numerics (dx,dt) → convergence-extrapolated (§4 ladders) ───────────┐
   │  anchored to the KNOWN physical tissue scale (r*/dx a physical ratio; honest dx exists) │
   │   ┌ CONSTRAINT GRAPH ─ hard: r*/dx≥k, definitional ; soft: √D, 2:1 (warm-start) ─────┐  │
   │   │   ┌ PHYSICAL JOINT FIT — {θ, kinetics, D_long, D_trans(free), (Cm)}, high-dim ─┐  │  │
   │   │   │   method: constrained scalarization, run ON A GP EMULATOR of tissue CV    │  │  │
   │   │   │   emulator TRAINED on mesh-adequate ladder sims; survivors VALIDATED real  │  │  │
   │   │   │   pinned by:  constraint graph (cheap)  +  rich observables (dear)         │  │  │
   │   │   └──────────────────────────────────────────────────────────────────────────┘  │  │
   │   └──────────────────────────────────────────────────────────────────────────────────┘  │
   └────────────────────────────────────────────────────────────────────────────────────────┘
   sim cost lives in the SHELL (training-data ladder), NOT the inner fit — via the emulator (§7)
```

**Design goal:** attack *all the physical knobs at once*; **resolve** (don't fit) the numerics,
anchored to the known physical scale; **maximize what you can honestly assert** (constraint graph)
so you **minimize what you're forced to measure** (observables); and keep the expensive tissue sims
(with their dx-ladders) in *emulator training*, out of the inner loop. Reachability from attack-all;
identifiability from constraints-first, observables-second; tractability from the surrogate.

## 9. Audit verdict + the three locks (necessary-but-not-sufficient)

Adversarial audit (Opus, 2026-07-02/07-10): joint tuning is the right direction but **one of coupled
fixes**; slow hiPSC CV_T=2.6 at dx=0.1 mm is unreachable under current constraints. There is a
**foundational prerequisite (P-1)** and then **three locks that must open together:**

**P-1 (prerequisite): unify the ionic backend.** The cell fit runs `cardiac_sim`/V5.4; the tissue fit
runs `cardiac_core` (§5 blocker). Until both observables come from ONE model, θ/kinetics cannot be
jointly identified. Port `cell_runner`/`batch_ionic` onto the cardiac_core ionic model. Everything
below assumes P-1 is done.

1. **dV/dt target — CONTESTED, must be reconciled (not simply lowered).** Pinned at 110 it forbids
   the G_Na↓ trade (at dV/dt≈110, g_Na≈0.83× → CV_T=2.6 at D≈1.6e-5 → r*/dx≈0.62 → block). *But* the
   open README criterion (README:28) says revise MHAS13 dV/dt to **~100 V/s** (up from an old 25),
   and MHAS13 is **by construction the *Matured* hiPSC model** (native ~110–132 V/s; `PARKER` presets
   pin 110). So "lower dV/dt to reach slow CV" pushes *opposite* to the project's own criterion and
   possibly *outside MHAS13's identifiable range*. This is the sharpest open tension: either revise
   the target down (contradicting README + risking MHAS13's range) **or** conclude MHAS13-matured is
   the wrong base model for a slow-upstroke target (→ the "change the model" attack, §strategy). To
   be resolved on data (P1a), not asserted here.
2. **Excitability range / kinetics** — widen g_Na floor **to ≤0.17 (≈0.15)** [not 0.2 — 0.17× ≈ 0.62
   absolute sits below a 0.2 floor] AND/OR add Na kinetics (conductance scaling alone can't decouple
   dV/dt from CV — needs gating τ, §5; requires P-1 for identifiability).
3. **dx / resolution** — for faithful source-sink physics the reentry campaign needs `r*/dx ≳ 3`. At
   the *slow* corner r* = D_trans/CV_T ≈ 1.6e-5/2.6e-3 ≈ **62 µm** (not the fast 130–160 µm), so k=3
   needs **dx ≲ ~20 µm (≈0.02 mm)** — note 0.03 mm gives only r*/dx≈2, below the k=3 floor. Refine
   chip dx from 0.1 mm → **~0.02 mm** (≈25× more cells). Treat r*/dx as a *precondition*.

**Sequenced plan** (P-1 is a hard prerequisite; kinetics is a *model change*, scheduled as its own
gated step — not smuggled into P1):
- **P-1** — **backend unification** (blocker): port `cell_runner`/`batch_ionic` off `cardiac_sim`
  (V5.4) onto the **cardiac_core** ionic model so dV/dt/APD and CV are measured on one model. Verify
  parity against the current cell fit. **Throughput caveat:** `batch_ionic` evaluates M candidates in
  one batched step with per-cell `(M,14)` conductances, but cardiac_core models carry *scalar*
  `self.params` — a naive port loses the batched cell eval (the README "10× speedup"). Either
  serialize candidates or extend cardiac_core to per-node conductances. *(Without P-1, kinetics is
  unidentifiable and warm-start is partial — §5.)*
- **P0** — cheap CPU discriminators: fix the secant ×4-up-bump (bracket *down* into the window —
  may rescue CV_L); re-sweep the cable at **hiPSC θ** (not NRVM); settle the high-D `nan` (CN
  accuracy/over-depolarization vs CV-measurement artifact — decides the true window width, §4).
- **P1a** — **conductance-only feasibility map** (no model change): with the lock-1 dV/dt question
  parameterized and the g_Na floor widened (lock-2, ≤0.17), sweep (g_Na, D) at dx ∈ {0.1, 0.05,
  0.03, 0.02 mm} and *plot* the feasible region for (CV_T=2.6, dV/dt, r*/dx≥3). **Gate:** feasible ⇒
  A+B suffices, skip to P2; infeasible ⇒ kinetics required, go to P1.5.
- **P1.5** — **kinetics model change** (only if P1a infeasible): add tunable Na-kinetic multipliers
  (`τ_m` scale; if needed `τ_h/τ_j`, `V_half`) as **per-model-instance parameters** applied around
  `cardiac_core/ionic/phas13/gating.py`'s `INa_*` functions — *without hard-editing the shared
  phas13 module* (mhas13 imports it); register the new axes; add CV-restitution as an identifying
  observable (§6). *(Keeps V5.3 untouched.)*
- **P1b** — re-run the feasibility map with the kinetic axes (answers open-Q1 for the kinetics
  branch, which P1a cannot).
- **P2** — build the constrained-scalarization joint fit on the **GP emulator** (extend
  `joint_refiner.py`; rewire it to `cc_runner`; add the r*/dx constraint + resolved-CV training data;
  mask the block region, §7) — **only in the regime P1a/P1b proves feasible**; warm-start θ from the
  saved cell-only fits (`presets/chip_{nrvm,hipsc}.json`).
- **P3** — refine `chip.py` mesh dx per P1 (slow corner ⇒ **~0.02 mm**); hand the (heavier) grid +
  resolved r*/dx to the reentry campaign.

## 10. Open questions (post-audit)

1. Does joint tuning + revised dV/dt + widened g_Na reach CV_T=2.6 at **conductance-only**, or is
   Na-kinetics genuinely required? (**P1a** answers conductance-only; **P1b** the kinetics branch.)
2. Minimal kinetic parameterization that decouples dV/dt from CV — `τ_m` scale alone, or +
   inactivation shift? What's the smallest identifiable set?
3. Observable set to match the expanded decision space — CV-restitution sufficient, or also
   upstroke-shape / block?
4. Extrapolating CV estimator — how many dx rungs, and how to detect "finest rung still blocked"
   (→ auto-refine or report infeasible)?
5. dx budget the reentry campaign can inherit — the slow corner needs **~0.02 mm** (≈25× cells; 0.03
   mm gives only r*/dx≈2 < k=3, §9 lock-3). Global ~0.02 mm, or per-baseline (NRVM tolerates coarser)?
6. Cm as a decision variable — real lever, but entangled with the effective-D convention (D = σ/(χ
   Cm)); include or hold fixed?
7. Is the **2:1 anisotropy *target*** itself right (cross-construct assumption, ≈2.1±0.8)? With
   `D_trans` free (§6), we fit CV_L and CV_T independently and *report* the achieved ratio — should
   the ratio be a soft target, or left to fall out? (Distinct from the withdrawn hard-tie question.)
8. Emulator design (extending `joint_refiner`): training-point count / active-learning refill, and
   how the r*/dx-infeasible region is represented to the surrogate (masked vs penalized)?

## Migration / impact

- **Primary superseded/extended module is `tuner/joint_refiner.py`** (the existing GP-emulator +
  NSGA-II over {θ, D_long, D_trans}), **not** just `fit_chip_baseline`. Extend it: rewire from the
  legacy `tissue_runner.run_cv_measurement` to **`cc_runner`** (+ the §4 dx-ladder estimator); add
  the `r*/dx` resolvability constraint; train the emulator on **resolved** CV; swap bare NSGA-II for
  the constrained-scalarization formulation (§7).
- **Evaluation kernel** (candidate → outputs) is **`cell_runner`** (AP → APD, dV/dt, V_rest/peak) +
  **`cc_runner`** (tissue → resolved CV_L/CV_T). `cell_fitter` is the BO *loop*; we reuse only its
  `_check_constraints`. `fit_chip_baseline`'s cell-then-secant is retired.
- Reconciles the still-open criterion **"Joint refinement (GP emulator + NSGA-II)"** — the GP
  emulator is kept; **NSGA-II → constrained scalarization**. Update the README criterion wording to
  match.
- New pieces: unified decision-space registry (`{subsystem, bounds, apply_fn}` + constraint graph);
  the convergence-aware CV estimator; the Na-kinetics gate multipliers (P1.5).
- **Decision vector (canonical count):** 10 conductance/flux-rate scales (incl. exchanger kNaCa,
  pump PNaK, pump g_pCa, uptake VmaxUp — not all conductances) + 2 diffusion (D_long, D_trans, both
  *free*) = 12 core; +~3 kinetics if P1.5 fires; +1 if `Cm` is admitted (open-Q6). No dimension is
  removed by tying (§6).
- Records/presets gain joint (θ, kinetics, D_long, D_trans, Cm) provenance + achieved r*/dx + the dx
  ladder used. Also fix the pre-existing `export_lab_preset(engine="lbm")` KeyError on
  monodomain-only records (write the engine that exists, or skip if absent).
- The V1 sequential result (best APD 352 / CV_L 14.6, single-rate) stands as a historical cell-fit
  baseline, fit at a coarser regime where the block did not bite.
