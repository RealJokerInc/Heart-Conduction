# PLAN: Engine Tuner → cardiac_core multi-engine (mono + bidomain + LBM)

Created: 2026-06-29
Revised: 2026-06-29 (audit iters 1–4 → CONVERGED; then domain correction: anisotropy restored)
Engine(s): Optimizer (build) + cardiac_core (integration layer) + Monodomain V5.4 / Bidomain V1 / LBM V1 (backends)
Research question: [Ionic Model Optimization](README.md) — *build owner*
Driving application: [Geometry-Induced Reentry](../geometry_induced_reentry/README.md) — *first customer*
Source: [IDEALOG.md](IDEALOG.md) §2026-06-29 (cross-plan) + [../geometry_induced_reentry/IDEALOG.md](../geometry_induced_reentry/IDEALOG.md) §2026-06-29 + [Optimizer/improvement.md](../../../Optimizer/improvement.md)
MASTER: [MASTER.md](../../../MASTER.md)

> **This is a SHARED CROSS-PLAN.** The *build* (a tuner that drives all three engines) is owned here. The *application* (fit a Kit Parker tissue-chip EP set, then run reentry sweeps on LBM) is owned by `geometry_induced_reentry`. They meet at the cardiac_core integration layer.

> **PLAN-ONLY GATE (user, 2026-06-29):** Do NOT execute any phase yet. This document is the deliverable. Execution waits for explicit "go".

> **AUDIT-DRIVEN REALITY CHECK (iter 1):** cardiac_core's high-level OO `CardiacSimulation` methods (`add_pacing`, `scale_conductance`, `compute_cv`, …) are **stubbed (`raise NotImplementedError`, `cardiac_core/api.py`)**. The **functional** API (`cardiac_core/run.py`: `run_monodomain/run_bidomain/run_lbm/simulate` + `SimulationResult.cv()/.apd()/.lat()/.restitution()` + `cardiac_core/analysis.py`) **is real and is what this plan uses.** The functional API takes `ionic_model` as a string with no θ/D hook → **Phase 0 adds the seam.** Everything depends on Phase 0.

> **DOMAIN CORRECTION (user, 2026-06-29):** the fit is **ANISOTROPIC (~2:1)**, NOT isotropic. Aligned hiPSC-CM / NRVM engineered tissue is anisotropic by construction — **Bursac & Parker 2002** measured ratio ≈2.1 (CV_L 23.5→37.2, CV_T 18.1→9.2 cm/s); de Diego 2010 got 2.1±0.8. The prior isotropic simplification (from the iter-1 audit) was wrong physics and is reverted. The tuner's existing fields (`cv_longitudinal/cv_transverse`, `TISSUE_PARAMS['D_long']/['D_trans']`) are already anisotropic — so mono/bidomain need no special-casing. The one real constraint the isotropic detour was hiding: **cardiac_core's `lbm()` wrapper rejects non-isotropic D** (`ValueError`, scalar only) — but **LBM V1 D2Q5 itself supports per-axis τ** (`improvement.md` §6 table). So Phase 0 also **extends the cardiac_core LBM wrapper to per-axis D**.

## Objective
Extend the Engine Tuner (`Optimizer/V1`, today fits MHAS13 + anisotropic diffusion on monodomain & bidomain) so its engine layer is **cardiac_core's functional API**, adding **LBM** as a first-class **per-axis-anisotropic** backend. Then use it to fit a **Kit Parker tissue-chip EP set** — both NRVM (CV_L 9.33 cm/s) and hiPSC-CM (CV_L 5.2 cm/s) baselines, each with **CV_T ≈ CV_L/2** — on a **chip-sized mesh (L = 16 mm, dx = 0.1 mm)**, cross-validated across all three engines, producing a reusable parameter set + preset that the reentry campaign runs on LBM.

## Success Criteria
- [ ] **cardiac_core tuning seam (Phase 0):** a fit can call cardiac_core with (a) a θ_ionic-scaled MHAS13 model (via `apply_scaling`, the module-level fn in `tuner.config`) and (b) **per-axis** diffusion — mesh `D_xx/D_yy` (mono), `D_i/D_e` per axis (bidomain), `tau_long/tau_trans` (LBM, **after extending the wrapper to per-axis D**); the call returns V(t) tensors. *(jointly owned with `engine_consolidation`.)*
- [ ] The **same `TuningConfig` runs on `engine ∈ {monodomain, bidomain, lbm}`** through cardiac_core's **functional** API (no use of the stubbed OO methods).  *(mirrors ionic_model_optimization → "Cross-engine validation (V5.4 vs Bidomain vs LBM)")*
- [ ] A tuned θ* (ionic + anisotropic `D_long/D_trans`) reproduces target **CV_L and CV_T ≤2%** on monodomain & bidomain via cardiac_core (anisotropy ratio CV_L/CV_T ≈ 2.0–2.1), and on **LBM (MRT) ≤2% per axis after recalibration** (expect ~35% raw offset first; LBM anisotropy needs MRT — Phase 0 Step 0.2).
- [ ] Chip-EP param sets fit for **both** NRVM (CV_L 9.33) and hiPSC-CM (CV_L 5.2) baselines (CV_T = CV_L/2), each with λ = CV·APD **measured** on the chip mesh.
- [ ] A clean **planar wave** runs on the **LBM** chip mesh at target CV with measured CV + wavelength.  *(mirrors geometry_induced_reentry → "Baseline" completion criterion)*
- [ ] **Tuned parameters + presets are persisted and reloadable** — `Optimizer/V1/presets/*.json` (full records) + `Lab/presets/chip_*.yaml` (consumable); a record round-trips to reproduce a run on any engine. *(see § Parameter & Preset Storage)*
- [ ] All existing Optimizer V1 tests pass (no regressions).

> **Dropped (audit iter 1):** the former "CV parity ≤0.5% vs current runners" criterion. The current `tissue_runner.py` is a hand-rolled explicit-Euler 1D cable, NOT `MonodomainSimulation`; cardiac_core uses Strang + Crank-Nicolson + PCG. We **re-measure CV on the cardiac_core path and fit the physical CV target**, using the old runner only as a loose (~few-%) cross-check.

## Architecture Changes
- **NEW (Phase 0):** a **cardiac_core tuning seam** — MOD `cardiac_core/run.py` (+ `api.py` factories) so a caller can pass a **pre-built (tuner-scaled) `IonicModel` instance** and **per-axis diffusion**: mono/bidomain already accept this (mesh `D_xx/D_yy`; bidomain `D_i/D_e`), **but LBM needs engine-level MRT work** — `LBMSimulation` is BGK-scalar-D only and `lbm()` raises `ValueError` on non-isotropic D; anisotropy needs multi-relaxation (per-axis rates from `tau_tensor_from_D`) via the existing `lbm_step_d2q9_mrt` (or by wiring `mrt_collide_d2q5`). See Phase 0 Step 0.2. θ_ionic via `apply_scaling` (module fn in `tuner.config`); do **NOT** use `scale_conductance` (stubbed *and* current-name-keyed — can't express `kNaCa/PNaK/g_pCa/VmaxUp/V_leak`).
- **NEW (design decision):** the tuner's engine layer is **cardiac_core's functional API**, NOT three bespoke `EngineAdapter` subclasses and NOT the stubbed OO `CardiacSimulation`. `improvement.md` §1–4 SUPERSEDED; reuse §5 (cross-engine validator concept) + §6 (per-axis anisotropy) + the LBM stability discussion (NOT its τ formula — see Known Failures).
- NEW: `Optimizer/V1/tuner/cc_runner.py` — single cardiac_core-backed runner: `run_1d_cable` (CV, per direction), `run_2d_tissue` (CV_L/CV_T + tissue APD), `run_s1s2`. Engine-dispatch by `config.engine`. **No `run_single_cell`** (single-cell stays in `cell_runner.py`).
- NEW: `Optimizer/V1/tuner/chip.py` — chip-mesh + **anisotropic (2:1)** Parker target presets (NRVM / hiPSC-CM), λ = CV·APD helper, points-per-λ check.
- MOD: `Optimizer/V1/tuner/config.py:37-106` — add `engine='lbm'`, `anisotropy_ratio: float=2.0`, chip-mesh fields (`domain_mm`, `dx_mm`), per-engine `dt` (`dt`, `dt_lbm`), `baseline` enum. **KEEP the existing anisotropic fields** `cv_longitudinal/cv_transverse` + `TISSUE_PARAMS` (`D_long/D_trans`) — the chip fit uses them directly (no isotropic flag, no `D_trans=D_long`).
- MOD: `Optimizer/V1/tuner/tissue_fitter.py` — dispatch through `cc_runner`, keep the existing **dual-axis** secant fit (CV_L→D_long, CV_T→D_trans); LBM tissue params via `tau_tensor_from_D` → MRT rates `s=1/τ` (BGK `tau_from_D` is isotropic-only).
- MOD: `Optimizer/V1/tuner/tissue_runner.py`, `tissue_runner_bidomain.py` — **shim** over `cc_runner`, kept until `test_tissue.py` migrated (do NOT delete — `test_tissue.py` imports `run_cv_measurement` at lines 15 & 36).
- NEW: `Optimizer/V1/tuner/cross_engine.py` — cross-engine validator over cardiac_core (per axis); reports mono↔bidomain and mono↔LBM CV deltas for both axes.
- NEW: `Optimizer/V1/tuner/presets.py` — tuned-parameter **record** save/load/list + Lab-preset export + `to_sim_kwargs`. *(see § Parameter & Preset Storage)*
- NEW: `Optimizer/V1/presets/` — canonical tuned-parameter records (JSON), one per fit.
- MOD: `Lab/presets/_SCHEMA.md` — extend to carry tuned ionic scalings + MHAS13 + **per-axis** LBM `tau_long/tau_trans` (and `D_long/D_trans`); current schema has none of these.
- NEW (application hand-off): a saved chip-EP preset consumable by `geometry_induced_reentry`.

## Known Failures (do NOT retry)
*From ionic_model_optimization IDEALOG:*
- **dV/dt target 25 V/s** — MHAS13 fast Na kinetics; range 80–130 V/s. Use ~100–120.
- **dV/dt hard constraint at 60 V/s** — only 2/74 feasible; use 120 (`dvdt_max_upper=120`).
- **Tier 1 only (6 params)** — 4/6 hit bounds; use tier 2 (10 params).
- **Newton-based CV refinement** — 50.6% overshoot; use **two-point secant**.
- **Analytical √D warm-start alone** — insufficient; secant on top required.

*From geometry_induced_reentry IDEALOG:*
- **Treating MacQueen's "~5 mm spatial period" as λ** — small-geometry artifact; λ = CV·APD. Fit/sweep against CV·APD.
- **Tuning adult TTP06 down to chip CV** — use **MHAS13** (mature quiescent hiPSC-CM).

*Established (audit iter 1–4):*
- **`scale_conductance` for θ_ionic** — stubbed AND current-name-keyed; can't express NCX/pump/SERCA/leak. Use `apply_scaling` (module fn in `tuner.config`), passed through the Phase 0 seam as a pre-scaled instance.
- **≤0.5% CV parity to the hand-rolled runner** — impossible across the integrator change. Re-measure & fit the physical target.
- **`improvement.md`'s `tau_from_D` prose** (lines 203 & 252: `tau = 0.5 + D/(cs²·dt)`) is **WRONG**. Correct (`LBM/Engine_V1/src/diffusion.py`): **`tau = 0.5 + D·dt/(cs²·dx²)`**, cs²=1/3.

*Domain correction (user, 2026-06-29):*
- **Do NOT assume isotropic.** Aligned hiPSC-CM/NRVM engineered tissue is **~2:1 anisotropic** (Bursac & Parker 2002 ratio ≈2.1; de Diego 2010 2.1±0.8). Fit CV_L **and** CV_T (≈CV_L/2). The transverse CV is NOT "fabricated" — it's `CV_L / anisotropy_ratio`, grounded in the corpus.
- **LBM anisotropy is engine-level MRT work, NOT a wrapper tweak.** `LBMSimulation` is BGK-scalar-D only; `mrt_collide_d2q5` exists but is **unwired** (no `lbm_step_d2q5_mrt`; only `lbm_step_d2q9_mrt`). Use **MRT** with per-axis rates from `tau_tensor_from_D` (+ `check_stability_tensor`) — **NOT `tau_from_D` twice** (BGK has one `omega`). `improvement.md` §6 is V2 *design*, not code.

---

## Parameter & Preset Storage
*(Persist tuned parameters AND presets so fits are reusable + reproducible across both questions and the wet-lab workflow. Cross-cutting — implemented across Phases 3–5.)*

**Two tiers, one source of truth.**

### Tier 1 — Tuned-parameter RECORD (canonical research artifact)
- **Location:** `Optimizer/V1/presets/{name}.json` (NEW dir). Written by `tuner/presets.py::save_record(result, name)` at end of fit (Phase 3); enriched with cross-engine results (Phase 4).
- **Schema (JSON, ANISOTROPIC, per-engine dt):**
```jsonc
{
  "name": "chip_nrvm", "baseline": "nrvm|hipsc", "ionic_model": "mhas13",
  "theta_ionic": { "g_Na": 0.83, "g_CaL": 1.12, "kNaCa": 0.9, ... },   // MULTIPLIERS on published values
  "mesh": { "domain_mm": 16.0, "dx_mm": 0.1 },
  "tissue": {
    "monodomain": { "D_long": 4.2e-4, "D_trans": 2.1e-4, "dt_ms": 0.02 },     // cm²/ms — canonical physical (CV_L/CV_T ≈ 2:1)
    "bidomain":   { "D_i_long": ..., "D_i_trans": ..., "D_e_long": ..., "D_e_trans": ..., "sigma_ratio": 3.597, "dt_ms": 0.02 },
    "lbm":        { "D_long": ..., "D_trans": ..., "collision": "mrt", "mrt_rates": {...}, "dx_mm": 0.1, "dt_ms": 0.05 }  // MRT relaxation rates DERIVED from per-axis D via tau_tensor_from_D at this dx/dt (BGK τ cannot represent anisotropy)
  },
  "targets": { "cv_longitudinal": 9.33, "cv_transverse": 4.67, "apd_90": 350, "dvdt_max": 110 },
  "validation": { "cv_long": 9.30, "cv_trans": 4.65, "tissue_apd": 348,
                  "cross_engine": { "mono_vs_bidomain_pct": 3.1, "mono_vs_lbm_pct": 34.8 } },
  "provenance": { "date": "...", "git_sha": "...", "plan": "PLAN.md", "tuner_version": "V1",
                  "ionic_published_ref": "PHAS13_REGISTRY" }
}
```
- **Targets carry both axes** (`cv_longitudinal`, `cv_transverse = cv_longitudinal/anisotropy_ratio`) — these are the existing `TuningTargets` fields (no key translation needed).
- **Why canonical per-axis D + per-engine dt:** physical `D_long/D_trans` are engine-independent; bidomain `D_i/D_e` and LBM MRT rates are *derived* and meaningless without `dx/dt`; LBM may need a **larger dt** to keep the relaxation off the 0.5 floor at chip CV — so each engine carries its own `dt`. Store the physical D's as the source of truth; recompute the LBM MRT rates on load via `tau_tensor_from_D` then `s = 1/τ` (mono/bidomain consume D directly — no τ).
- **API (`tuner/presets.py`):** `save_record(result, name)`, `load_record(name) -> dict`, `list_records()`, `to_sim_kwargs(record, engine) -> dict`.

### Tier 2 — Lab PRESET (consumable projection)
- **Location:** `Lab/presets/{name}.yaml` — the existing `/sim-preset` store (schema `Lab/presets/_SCHEMA.md`). Written by `tuner/presets.py::export_lab_preset(record, engine)`.
- **Schema extension required (MOD `Lab/presets/_SCHEMA.md`):** add `mhas13` to `ionic:`; an `ionic_scaling:` block (θ_ionic multipliers); and **per-axis** LBM knobs (`conductivity.D_long/D_trans` + `tau_long/tau_trans`). Keep presets **parameters-only, no API calls** (existing rule).
- **Naming:** `chip_{baseline}` → `chip_nrvm`, `chip_hipsc`.

### Consumption paths
- Tuner re-validation / cross-engine: `load_record` → `to_sim_kwargs(record, engine)`.
- Reentry sweeps (LBM, primary): `/sim-experiment "using preset chip_hipsc"`, or `load_record('chip_hipsc')` → cardiac_core LBM run (per-axis D).
- Wet-lab reuse: `/sim-preset load chip_nrvm`.

> **Phase placement:** `save_record` → Phase 3; record enrichment → Phase 4; `_SCHEMA.md` extension + `export_lab_preset` + planar-wave baseline → Phase 5.

---

## Phase 0: cardiac_core tuning seam + LBM anisotropy via MRT (PREREQUISITE)

**Goal**: cardiac_core can run a *parametrized, anisotropic* simulation — caller injects a θ_ionic-scaled MHAS13 instance + per-axis diffusion — and get V(t) back on all three engines. Mono/bidomain anisotropy is a small seam (Step 0.1); **LBM anisotropy is genuine engine-level MRT work (Step 0.2)**. Without this, nothing downstream can tune.
**Tier**: large
**Estimated scope**: Step 0.1 small (instance pass-through + per-axis mesh for mono/bidomain). **Step 0.2 medium-large ENGINE work** (LBM is BGK-scalar-only today; per-axis needs MRT). Jointly owned with `engine_consolidation`.

### Phase Context
- Functional entry points are real: `run_monodomain/run_bidomain/run_lbm/simulate` return `(times, V[, phi_e])`; `SimulationResult` exposes `.cv()/.apd()/.lat()/.restitution()`.
- mono/bidomain **sims** accept an `IonicModel` instance and their api.py factories forward `ionic_model or data.ionic_model` straight through (api.py:1072, 1186); their meshes carry per-axis `D_xx/D_yy` (mono) / `D_i/D_e` (bidomain) — so **mono/bidomain anisotropy works once a per-axis mesh is built**.
- **LBM is the hard one (verified against engine source):** `LBMSimulation.__init__` takes a **scalar `D`**, one `tau = tau_from_D(D,…)`, one `omega`, **BGK-only** (`cardiac_core/_lbm/simulation.py` ≈ `LBM/Engine_V1/src/simulation.py:55-102`). Step fns are `lbm_step_d2q5_bgk`, `lbm_step_d2q9_bgk`, **`lbm_step_d2q9_mrt`** — **no `lbm_step_d2q5_mrt`**. A D2Q5-MRT primitive `mrt_collide_d2q5` exists (`collision/mrt/d2q5.py`) but is **unwired**. The `lbm()` wrapper raises `ValueError("…Use D2Q9 MRT for anisotropic diffusion")` (api.py:~1365); the LBM factory `.lower()`-rebuilds the model from a string (api.py:1338/1353), discarding an instance.
- **Anisotropic LBM requires MRT, not BGK** (BGK has a single `omega` — `tau_from_D` twice is meaningless). Per-axis relaxation rates come from **`tau_tensor_from_D(D_xx, D_yy, D_xy, dx, dt)`** + stability via **`check_stability_tensor`** (`diffusion.py:37,63`). `improvement.md` §6 is V2 *design intent*, not code; its §4 ("D2Q5 **with MRT** suffices") + the wrapper guard agree.
- The validated scaling mechanism is `apply_scaling(params, theta)` (module-level in `tuner.config`, `config.py:131-153`). **The TUNER calls it and passes the scaled instance**, so cardiac_core needs no Optimizer import.

### Step 0.1: Instance pass-through + per-axis D for mono/bidomain
**Model**: opus
#### Read First
- `cardiac_core/run.py:93-228` — `run_*` signatures.
- `cardiac_core/api.py:~1025-1401` — mono/bidomain instance forward (1072/1186); LBM `.lower()` rebuild (1338/1353); LBM guard (~1365). api.py:~700-1024 = stubbed OO methods (not used).
- `cardiac_core/file_format.py` — `CardiacMeshData` (per-axis `D_xx/D_yy/D_xy` arrays); **`create_cardiac_mesh` is SCALAR-D only** (hard-sets `D_xx=D_yy=D`) → a per-axis mesh must build `CardiacMeshData` directly or extend the helper.
- `Optimizer/V1/tuner/config.py:131-153` — `apply_scaling`.
- `cardiac_core/ionic/mhas13/model.py` — the **vendored** MHAS13 params object (NOT the V5.4 copy).
#### Why
mono/bidomain already support per-axis D and instances; the only gaps are a per-axis mesh builder + the LBM factory's instance handling. Land these first so mono/bidomain anisotropy works while LBM-MRT (0.2) proceeds in parallel.
#### Implementation Spec
**Files to modify:** `cardiac_core/api.py` (LBM factory instance guard), `cardiac_core/file_format.py` (per-axis mesh: extend `create_cardiac_mesh` to accept `D_xx/D_yy` or add a `CardiacMeshData` builder).
**Design:** accept a pre-built `IonicModel` instance on all three factories (mono/bidomain already do; patch the LBM factory's `.lower()`/`model_map` rebuild to use a passed instance). Build per-axis meshes (`D_xx=D_long`, `D_yy=D_trans`, `D_xy=0`).
#### Pseudocode
```
model = ionic_model if isinstance(ionic_model, IonicModel) else build(ionic_model)   # tuner pre-scaled it
mesh  = CardiacMeshData(..., D_xx=D_long, D_yy=D_trans, D_xy=0)   # create_cardiac_mesh is scalar-only → build directly
times, V = run_monodomain|run_bidomain(mesh, t_end, ionic_model=model, ...)   # both consume per-axis mesh D
# LBM factory: if isinstance(ionic_model, IonicModel): use it (skip .lower()/model_map rebuild)
```
#### Test Spec
- `cardiac_core/tests/test_param_seam.py::test_scaled_instance_changes_apd` (create) — instance with `g_Kr`↓ lengthens APD.
- `::test_mono_mesh_anisotropy_changes_cv` — mono `D_xx>D_yy` → CV_L>CV_T (ratio ~`sqrt(D_xx/D_yy)`).
- `::test_bidomain_anisotropy` — bidomain per-axis → CV_L>CV_T.
- `::test_lbm_instance_runs` — LBM accepts an instance (isotropic D), runs, finite CV (anisotropy → 0.2).
#### Checklist
- [ ] Per-axis `CardiacMeshData` builder (or `create_cardiac_mesh` extended).
- [ ] LBM factory accepts an instance (no `.lower()` crash).
- [ ] mono/bidomain anisotropy verified; no change to V5.3.
#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_param_seam.py -v
```
#### Exit Criteria
- [ ] mono/bidomain run with a scaled instance + per-axis mesh D (CV_L>CV_T); LBM runs with a scaled instance (isotropic).
#### Risk
`create_cardiac_mesh` scalar-only is the snag. Mitigation: build `CardiacMeshData` directly in `chip.chip_mesh` (it carries per-axis arrays) — don't fight the helper.

### Step 0.2: LBM per-axis anisotropy via D2Q9-MRT (engine work)
**Model**: opus
**Tier**: large — the **biggest single item in the plan**. Three sub-steps: **0.2a** wire the MRT path into `LBMSimulation`; **0.2b** route the `lbm()` wrapper to MRT on anisotropic D; **0.2c** validate against an analytic anisotropic-diffusion benchmark. Do them in order — 0.2c gates trust in the whole LBM branch.

#### Read First
- `cardiac_core/_lbm/simulation.py` ≈ `LBM/Engine_V1/src/simulation.py:55-203` — `LBMSimulation.__init__` (scalar `D`; selects `self._step_fn` = BGK only; `tau = tau_from_D(D, dx, dt, cs2=self.lattice.cs2)`, `self.omega = 1/tau`); `step()` calls `self._step_fn(self.f, self.V, R, self.dt, self.omega, self.w, self.bounce_masks)` (line 162). **Note the existing TODO at lines 89-91**: "BGK only — if a collision selector is added later, `weights_mode` plumbing must be re-checked for MRT".
- `LBM/Engine_V1/src/step.py:44-58` — **`lbm_step_d2q9_mrt(f, V, R, dt, w, s_e, s_eps, s_jx, s_q, s_pxx, s_pxy, bounce_masks, s_jy=None)`** — note the signature DIFFERS from the BGK step fns (no single `omega`; six named moment rates + optional `s_jy`). There is **no `lbm_step_d2q5_mrt`**.
- `LBM/Engine_V1/src/collision/mrt/d2q9.py:57-119` — `mrt_collide_d2q9(...)`. Its docstring gives the Chapman-Enskog mapping: **`D_xx = cs2·(1/s_jx − 0.5)·dt`, `D_yy = cs2·(1/s_jy − 0.5)·dt`** (lattice units); `s_jy=None → s_jx` (isotropic). Per-axis is ALREADY built in.
- `LBM/Engine_V1/tests/test_phase2.py:41-131` — standard **free-rate values** (`s_e=s_eps=s_q=s_pxx=s_pxy=1.0`; variants 1.2/1.1/1.3/0.9/0.8). `LBM/Engine_V1/tests/test_phase5.py:144-261` — **already calls `mrt_collide_d2q9` with `s_jy` (anisotropic collision is tested at the collision level)** — reuse for the *collision wiring/free-rate* reference ONLY. **⚠ Do NOT copy its s-mapping:** `test_5v4/5v5` compute `s = 1/(0.5 + D/(cs²·dt))` with `dx=dt=1` (a degenerate form that drops the `dx²`). The authoritative mapping is **`s = 1/τ` with `τ` from `tau_tensor_from_D(D_long, D_trans, 0, dx, dt, cs2)`** — never the docstring's lattice-unit form.
- `LBM/Engine_V1/src/diffusion.py:37-77` — `tau_tensor_from_D(D_xx, D_yy, D_xy, dx, dt, cs2) -> (τ_xx, τ_yy, τ_xy)` (relaxation **times**); `check_stability_tensor(...) -> (is_stable, τ_min)`.
- `cardiac_core/api.py` (lbm wrapper) — the `is_isotropic` check + `ValueError("LBM BGK currently supports isotropic D only. Use D2Q9 MRT…")` (~1358-1369); `D = float(data.D_xx.flat[0])`; `LBMSimulation(... lattice=lattice)` construction (~1380).
- `LBM/Engine_V1/src/collision/mrt/d2q5.py:37` — `mrt_collide_d2q5` (the unwired D2Q5-MRT primitive; only relevant if the D2Q5 fallback is pursued — NOT the recommended path).

#### Why
LBM is the **primary** reentry engine; it must honor the 2:1 anisotropy. BGK has a single relaxation (`omega`) → isotropic only. The D2Q9-MRT collision already encodes per-axis diffusion via separate `s_jx`/`s_jy`; the ONLY gap is that `LBMSimulation` never selects the MRT step. So this is wiring + a wrapper route + a correctness benchmark — not new numerics.

#### Risk (whole step)
Biggest, riskiest item. (1) **Mapping correctness** — the collide docstring writes `D = cs2·(1/s − 0.5)·dt` (lattice units, dx=1); the physical mapping must use `tau_tensor_from_D(…, dx, dt)` so it stays consistent with the working BGK path (`tau_from_D(D, dx, dt, cs2)`). 0.2c's analytic benchmark is the gate that catches any dx²-scaling slip. (2) **Stability** — the slow transverse axis sits near the τ=0.5 floor at chip CV; `dt_lbm` may need raising (recorded per-engine). (3) **`weights_mode`/MRT** — the MRT moment matrix assumes canonical D2Q9 (`cs2=1/3`); reject `collision='mrt'` with `weights_mode='uniform_8'` (D2Q9_uniform `cs2=0.75`). (4) **`D_xy≠0`** (oblique fibers) needs the moment-space rotation the collide docstring defers to "Phase 8" — **out of scope**; restrict to axis-aligned (`D_xy=0`) and raise otherwise. Mitigation: D2Q9-MRT only (proven `lbm_step_d2q9_mrt` + tested collision); benchmark before trusting tuned CV; the vendored `cardiac_core/_lbm/` is cardiac_core's source of truth — coordinate with `engine_consolidation`.

---
**Sub-step 0.2a — wire MRT into `LBMSimulation`** (`cardiac_core/_lbm/simulation.py`; mirror upstream `LBM/Engine_V1/src/simulation.py`). NOTE: `step.py` already defines `lbm_step_d2q9_mrt` + imports `mrt_collide_d2q9` (byte-identical upstream↔vendored). The real import gap is in **`simulation.py`**, whose import block currently has only `from .step import lbm_step_d2q5_bgk, lbm_step_d2q9_bgk` and `from .diffusion import tau_from_D` — add `lbm_step_d2q9_mrt`, `tau_tensor_from_D`, `check_stability_tensor` there.
- `__init__`: add `collision: str = 'bgk'`, `D_yy: Optional[float] = None` (existing `D` becomes `D_xx`; `D_yy=None` → isotropic), and free-rate kwargs `s_e=1.0, s_eps=1.0, s_q=1.0, s_pxx=1.0, s_pxy=1.0`.
- Guards: `collision='mrt'` requires `lattice='d2q9'` and `weights_mode='canonical'` (raise otherwise); `D_xy` is implicitly 0 (axis-aligned only).
- MRT setup: `τ_xx, τ_yy, _ = tau_tensor_from_D(D_xx, D_yy or D_xx, 0.0, dx, dt, cs2=self.lattice.cs2)`; `self.s_jx, self.s_jy = 1.0/τ_xx, 1.0/τ_yy`; `ok, τ_min = check_stability_tensor(...)`; `if not ok: raise`. Store the 5 free rates.
- `step()` dispatch (the signatures differ): keep the BGK call as-is; for MRT:
```
self.f, self.V = lbm_step_d2q9_mrt(self.f, self.V, R, self.dt, self.w,
                                   self.s_e, self.s_eps, self.s_jx, self.s_q,
                                   self.s_pxx, self.s_pxy, self.bounce_masks, s_jy=self.s_jy)
```
- Tests (create `cardiac_core/tests/test_lbm_anisotropy.py`, mirror upstream `LBM/Engine_V1/tests/`):
  - `::test_mrt_isotropic_matches_bgk` — `collision='mrt'`, `D_yy=D_xx` → CV within a few % of the BGK baseline at the same D (regression guard).
  - `::test_mrt_anisotropic_cv` — `D_xx=2·D_yy` → CV_L/CV_T ≈ `sqrt(2)` (eikonal: CV ∝ √D).
  - `::test_mrt_guards` — `collision='mrt'` + `lattice='d2q5'` or `weights_mode='uniform_8'` raises; unstable τ-tensor raises.
- Checklist: [ ] collision selector + per-axis D in `__init__`; [ ] `s=1/τ` via `tau_tensor_from_D` (NOT the docstring/test_phase5 lattice-unit form); [ ] `step()` MRT dispatch; [ ] `simulation.py` imports `lbm_step_d2q9_mrt`/`tau_tensor_from_D`/`check_stability_tensor`; [ ] guards; [ ] upstream `LBM/Engine_V1` kept in parity.

**Sub-step 0.2b — route the `lbm()` wrapper to MRT** (`cardiac_core/api.py`).
- Replace the `is_isotropic`→`ValueError` block: if anisotropic (`D_xx ≠ D_yy`, `D_xy≈0`) → force `lattice='d2q9'`, `collision='mrt'`, pass `D=data.D_xx.flat[0]`, `D_yy=data.D_yy.flat[0]` to `LBMSimulation`; keep the isotropic → BGK path unchanged (back-compat). Raise on `D_xy≠0` (oblique fibers, out of scope).
- **Round-trip:** anisotropy is carried by the mesh (`data.D_xx/D_yy`), which this same wrapper re-reads on rebuild → it auto-detects MRT. So `build_kwargs` need NOT carry `D`/`D_yy`; passing `collision` is optional/redundant. Don't thread D through `build_kwargs`.
- **Ordering dependency:** an anisotropic *fit* run passes an `IonicModel` instance, which hits the `ionic_name.lower()` crash (api.py:1353) until **Step 0.1** lands the LBM-factory instance fix. 0.1 MUST precede 0.2b (the phase order enforces this; both edit the same `lbm()` block — coordinate, don't conflict).
- Test: `::test_lbm_wrapper_anisotropic_runs` — a per-axis `CardiacMeshData` through `run_lbm` returns finite V, CV_L>CV_T, no ValueError; `::test_lbm_wrapper_isotropic_unchanged` — isotropic mesh still uses BGK.

**Sub-step 0.2c — analytic anisotropic-diffusion benchmark (the correctness gate).**
- Pure diffusion, **no ionics** (`R=0`): initialize a Gaussian blob, evolve N steps, fit second moments. Theory: `σ²_x(t) = σ²_0 + 2·D_xx·t`, `σ²_y(t) = σ²_0 + 2·D_yy·t`.
- Test `::test_mrt_recovers_D_tensor` — recovered `D_xx, D_yy` from the moment growth match the input `D_long, D_trans` within ~5%. **This is the test that proves `s_jx/s_jy → D` (incl. the dx²/dt scaling) is correct** — do not trust tuned CV until it passes.
- **CRITICAL: the benchmark MUST use `dx ≠ dt`** (e.g. `dx=0.025 cm, dt=0.01 ms`). The correct mapping `τ = 0.5 + D·dt/(cs²·dx²)` (`tau_tensor_from_D`) and the wrong lattice-unit mapping `τ = 0.5 + D/(cs²·dt)` (the `d2q9.py` docstring form, also used by `test_phase5`'s `test_5v4/5v5`) **coincide exactly when dx=dt** — a `dx=dt=1` benchmark passes for BOTH and proves nothing. Only `dx≠dt` discriminates them.
- Checklist: [ ] benchmark uses `dx≠dt`; [ ] recovers both D's ≤5%; [ ] documents the `dx,dt,cs2` used.

---
#### Verify (whole step)
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_lbm_anisotropy.py LBM/Engine_V1/tests/ -v
```
#### Exit Criteria
- [ ] 0.2a: `LBMSimulation` runs D2Q9-MRT; isotropic-MRT matches BGK; guards fire.
- [ ] 0.2b: `lbm()` routes anisotropic meshes to MRT (no ValueError); isotropic unchanged.
- [ ] 0.2c: analytic benchmark recovers `D_xx,D_yy` ≤5% — the `s→D` mapping is proven.
- [ ] Existing LBM tests pass; vendored `cardiac_core/_lbm/` + upstream `LBM/Engine_V1` in parity; V5.3 untouched.

### Phase 0 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ LBM/Engine_V1/tests/ -v
```
- [ ] Steps 0.1+0.2 pass; mono/bidomain + **LBM(MRT)** all run anisotropic via cardiac_core; existing tests pass; V5.3 untouched.
**-> Commit point**

---

## Phase 1: cardiac_core-backed runner (functional API; add LBM, anisotropic)

**Goal**: One runner that measures CV_L/CV_T/APD on `engine ∈ {monodomain, bidomain, lbm}` via cardiac_core's **functional** API + `analysis`, using the Phase 0 seam. CV re-measured and sane, LBM per-axis path functional.
**Tier**: large

### Phase Context
- Use `run_monodomain/run_lbm/run_bidomain` + `analysis.conduction_velocity` / `SimulationResult.cv()`. **Do NOT** call the stubbed OO methods.
- Today's `tissue_runner.py` (hand-rolled explicit Euler, threshold −30, probes `Nx//4`/`3Nx//4`) and `tissue_runner_bidomain.py` are the *legacy* path — becoming shims, not deleted.
- **Anisotropic:** per-axis `D_long/D_trans`; measure CV along x (longitudinal cable) and y (transverse cable).
- **θ_ionic representation:** the **tensor** form (multipliers, ordered by tier) is canonical inside the fit loop — `cc_runner.run_1d_cable` and `tissue_fitter.fit_tissue` both take the tensor (matches legacy `run_cv_measurement`). Convert tensor→dict via `theta_to_dict(theta, tier)` ONLY for `apply_scaling` and the JSON record. `apply_scaling/theta_to_dict/dict_to_theta` are **module-level functions** in `tuner.config`, NOT `TuningConfig` methods. `cell_fitter.fit_cell` returns a `CellFitResult` (Pareto front of θ tensors) → `select_best` picks one (Step 3.1).
- Conventions: float64; `device='cuda'`; ionic=MHAS13; V5.3 read-only.

### Step 1.1: `cc_runner.run_1d_cable` (per-axis CV) over the functional API
**Model**: opus
#### Read First
- `cardiac_core/run.py` (`run_monodomain/run_lbm/run_bidomain`, `simulate`, `SimulationResult`).
- `cardiac_core/analysis.py` — `conduction_velocity(V, times, dx, x1, x2, y, threshold=-20)` (x1/x2 are INTEGER indices).
- `cardiac_core/grid.py`, `cardiac_core/file_format.py` — building a strip mesh with per-axis D (no `cable()` helper; build `CardiacMeshData` directly — `create_cardiac_mesh` is scalar-D only).
- `Optimizer/V1/tuner/tissue_runner.py` (whole) — legacy CV protocol to *loosely* cross-check (threshold −30, probes 1/4 & 3/4).
#### Why
CV_L and CV_T are the tissue-fit objectives; both must be measured consistently via cardiac_core so the dual-axis secant loop is stable. Parity to the old runner is NOT a gate — only a sanity cross-check.
#### Implementation Spec
**Files to create:** `Optimizer/V1/tuner/cc_runner.py`.
**Interfaces:**
```python
def run_1d_cable(theta_ionic: torch.Tensor, D: float, config: TuningConfig, *, axis: str='long') -> float  # CV cm/s along axis
def run_2d_tissue(theta_ionic: torch.Tensor, D_long: float, D_trans: float, config) -> dict  # {cv_long, cv_trans, tissue_apd}
def run_s1s2(theta_ionic, D_long, D_trans, config, s1_cl, di_values) -> list[tuple[float,float]]
def _run(theta_ionic, D_peraxis, config, *, mesh):  # dispatch by config.engine via run_monodomain/bidomain/lbm + Phase-0 seam
```
- `axis='long'` builds an x-cable with `D_xx=D`; `axis='trans'` builds a y-cable with `D_yy=D`. θ_ionic→dict via `theta_to_dict` for `apply_scaling`.
#### Pseudocode
```
run_1d_cable(theta_ionic, D, config, axis):
  model = build('mhas13'); apply_scaling(model.params, theta_to_dict(theta_ionic, config.tier))   # pass instance
  mesh  = strip CardiacMeshData along axis, len=config.cable_length_cm, dx=config.dx_cm, D on that axis
  if config.engine == 'bidomain':
      times, V, _phi = run_bidomain(mesh, t_end, ionic_model=model, sigma_ratio=3.597, dt=config.dt)  # 3-tuple
  else:                                                                   # monodomain / lbm (per-axis via mesh)
      times, V       = run_<engine>(mesh, t_end, ionic_model=model, dt=dt_for(config.engine))
  return analysis.conduction_velocity(V, times, dx=config.dx_cm, x1=N//4, x2=3*N//4, y=0, threshold=-20)
  # NOTE: x1/x2 INTEGER node indices; threshold=-20 = analysis default (legacy used -30 → loose cross-check only); run_bidomain returns a 3-tuple
```
#### Test Spec
- `tests/test_cc_runner.py::test_monodomain_cv_sane` (create) — CV_L within ~10% of legacy `run_cv_measurement` (loose cross-check, NOT a 0.5% gate).
- `tests/test_cc_runner.py::test_lbm_anisotropic_cv` — CV_L and CV_T finite, CV_L > CV_T for D_long > D_trans.
- `tests/test_cc_runner.py::test_bidomain_cv_sane` — within ~10% of legacy bidomain runner.
#### Checklist
- [ ] `_run` dispatch (mono/bidomain/lbm) via functional API + Phase-0 seam.
- [ ] `run_1d_cable` measures CV along the requested axis via `analysis.conduction_velocity`.
- [ ] Threshold/probe positions documented + engine-consistent.
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_cc_runner.py -v
```
#### Exit Criteria
- [ ] Mono/bidomain CV within ~10% of legacy (sanity); LBM per-axis CV finite, anisotropic.
#### Risk
cardiac_core strip-mesh + Strang/CN may need a smaller dt than the explicit runner for a clean upstroke. Mitigation: set dt from CFL; don't assume the legacy dt.

### Step 1.2: Wire `engine='lbm'` + anisotropic tissue dispatch into config & tissue_fitter
**Model**: opus
#### Read First
- `Optimizer/V1/tuner/config.py:37-106` — `TuningConfig`, `TISSUE_PARAMS` (`D_long/D_trans`).
- `Optimizer/V1/tuner/tissue_fitter.py` (whole) — the **existing dual-axis** secant fit + bidomain D_eff decomposition.
- `LBM/Engine_V1/src/diffusion.py:29-34` — **correct** `tau_from_D(D, dx, dt, cs2=1/3) = 0.5 + D*dt/(cs2*dx*dx)`.
#### Why
The dual-axis secant fit is the validated CV fit (keep it). Widen to a third engine whose per-axis knobs are `τ_long/τ_trans`, derived from `D_long/D_trans` via the correct `tau_from_D`.
#### Implementation Spec
**Files to modify:**
- `config.py`: add `engine='lbm'`, `anisotropy_ratio: float=2.0`, `domain_mm`, `dx_mm`, `dt_lbm`, `baseline`. Widen `TISSUE_PARAMS['D_long']`/`['D_trans']` lower bounds (→ `5e-6`/`2.5e-6`) for chip CV. KEEP `cv_longitudinal/cv_transverse`.
- `tissue_fitter.py`: keep the existing dual-axis secant (CV_L→D_long, CV_T→D_trans). For `engine=='lbm'`, record the **per-axis MRT relaxation rates** from `tau_tensor_from_D(D_long, D_trans, 0, dx_cm, dt_lbm)` (NOT `tau_from_D` twice — BGK has one ω); validate with `check_stability_tensor`.
#### Pseudocode
```
fit_tissue(theta_t, config, targets):
  D_long  = secant on D until CV_L→targets.cv_longitudinal  (cc_runner.run_1d_cable axis='long')
  D_trans = secant on D until CV_T→targets.cv_transverse     (axis='trans')
  if engine=='lbm': tau_xx,tau_yy,_ = tau_tensor_from_D(D_long,D_trans,0,dx,dt_lbm); s_jx,s_jy = 1/tau_xx,1/tau_yy; assert check_stability_tensor(D_long,D_trans,0,dx,dt_lbm)[0]  # MRT RATES=1/τ, NOT BGK τ; expect ~35% offset → Phase 4
  return TissueFitResult(D_long, D_trans, cv_long_achieved, cv_trans_achieved[, tau_long, tau_trans])
```
#### Test Spec
- `tests/test_cc_runner.py::test_lbm_dual_axis_converges` — targets CV_L=9.33, CV_T=4.67; both |err|<5% within 5 iters/axis.
- `tests/test_cc_runner.py::test_tau_from_D_formula` — `tau_from_D` matches `0.5 + D*dt/(cs2*dx*dx)`.
#### Checklist
- [ ] `engine='lbm'` accepted; per-axis D bounds set; per-engine dt added.
- [ ] dual-axis secant preserved; Newton NOT used; LBM via `tau_tensor_from_D` → rates `s=1/τ` (not `tau_from_D` twice).
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_cc_runner.py -v
```
#### Exit Criteria
- [ ] LBM per-axis tissue fit converges (within offset caveat).
#### Risk
`tau ≤ 0.51` for the slow transverse axis at dx=0.1 mm. Mitigation: raise `dt_lbm` until both τ ≥ ~0.55; log per-engine dt in the record.

### Phase 1 Verification
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/ -v
```
### Phase 1 Exit Criteria
- [ ] New tests pass; all existing V1 tests pass (no regressions).
- [ ] CV_L/CV_T sane + anisotropic on all three engines via cardiac_core.
### Phase 1 Cleanup
- float64 consistency; V5.3 untouched.
- **Shim** `tissue_runner.py`/`tissue_runner_bidomain.py` over `cc_runner`, **preserving the `run_cv_measurement` signature** so `test_tissue.py` (imports at lines 15 & 36) keeps passing; touch it only if the signature must change. Do NOT delete.
- No engine logic in the tuner that cardiac_core already provides.

**-> Commit point: git commit after Phase 1 passes**

---

## Phase 2: Chip mesh + Kit Parker dual-baseline targets (anisotropic 2:1)

**Goal**: Encode "simulate the entire Parker chip" — mesh (L=16 mm, dx=0.1 mm) + **anisotropic (2:1)** EP targets for both NRVM and hiPSC-CM — and verify the discretization resolves λ = CV·APD. Surface the λ-vs-chip tension honestly.
**Tier**: small

### Phase Context
- Chip footprint: 25 mm coverslip; usable square ≈16 mm; dx=0.1 mm → 161² grid; 1 mm obstacle = 10 cells.
- **Parker EP (provenance):** NRVM CV_L 9.33 and **hiPSC-CM CV_L 5.2 cm/s are BOTH from MacQueen 2018** (engineered model ventricles; apex→base Ca wavefront = the longitudinal direction). The 5.2 figure is the *3-D engineered-ventricle* hiPSC value, distinct from *monolayer* hiPSC-CM CV ~20–44 cm/s (Herron). **Anisotropy ratio ≈2:1** from Bursac & Parker 2002 (≈2.1) / de Diego 2010 (2.1±0.8) → **CV_T = CV_L / anisotropy_ratio** (default 2.0; band 2.0–2.1). *Caveat:* the ratio is from NRVM/monolayer prep, applied here to MacQueen's 3-D engineered-ventricle CV_L — a cross-construct assumption; flag if it matters.
- **No APD in the Parker corpus** → APD/restitution from broader hiPSC lit (Shadrin APD80 424–471; Paci ~350); MHAS13 native APD≈349 ms anchors it.

### Step 2.1: `chip.py` — mesh + anisotropic target presets + λ check
**Model**: opus
#### Read First
- `cardiac_core/geometry.py` — `rectangle_mask`, `circle_mask`, `left_edge_mask`.
- `cardiac_core/file_format.py` — `CardiacMeshData` per-axis `D_xx/D_yy/D_xy`. NOTE `create_cardiac_mesh` is **scalar-D only** → `chip_mesh` builds `CardiacMeshData` directly.
- `geometry_induced_reentry/KNOWLEDGE.md` → "Parameter-Fitting Strategy" + wavelength table; `parker_lab/INDEX.md` (anisotropy ratio).
#### Why
Anchor the campaign to a real (anisotropic) chip; one source of truth for mesh + targets; expose the λ-vs-domain tension early.
#### Implementation Spec
**Files to create:** `Optimizer/V1/tuner/chip.py`.
**Interfaces:**
```python
# anisotropic: CV_T = CV_L / anisotropy_ratio (2.0; Bursac&Parker ≈2.1). dvdt_max_upper=120 so cell_fitter
# (default 60) doesn't reject MHAS13's ~110 V/s (Known Failure).
PARKER_NRVM  = TuningTargets(cv_longitudinal=9.33, cv_transverse=4.67, apd_90=350, dvdt_max=110, dvdt_max_upper=120, ...)
PARKER_HIPSC = TuningTargets(cv_longitudinal=5.2,  cv_transverse=2.60, apd_90=350, dvdt_max=110, dvdt_max_upper=120, ...)
def chip_mesh(domain_mm=16.0, dx_mm=0.1, D_long=None, D_trans=None, device='cuda') -> CardiacMeshData  # D_xx=D_long, D_yy=D_trans, D_xy=0
def wavelength_mm(cv_cm_s, apd_ms) -> float            # λ = CV·APD
def points_per_wavelength(cv, apd, dx_mm) -> float     # warn if < 25
```
#### Pseudocode
```
wavelength_mm(cv, apd): return (cv cm/s → mm/ms) * apd_ms       # 9.33 cm/s = 0.0933 mm/ms
chip_mesh: build CardiacMeshData directly (create_cardiac_mesh is scalar-only), Nx=Ny=round(domain_mm/dx_mm)+1, D_xx=D_long, D_yy=D_trans, D_xy=0
points_per_wavelength: wavelength_mm/dx_mm; warn (log) if < 25
```
#### Test Spec
- `tests/test_chip.py::test_mesh_shape` (create) — 161×161 at L=16, dx=0.1; `D_xx≠D_yy` when D_long≠D_trans.
- `tests/test_chip.py::test_wavelength` — `wavelength_mm(9.33,350)`≈32.7; `wavelength_mm(5.2,350)`≈18.2; ppλ ≥ 25.
- `tests/test_chip.py::test_targets_anisotropic` — presets set `cv_longitudinal` + `cv_transverse=cv_long/2`, `dvdt_max≈110`, `dvdt_max_upper=120` (NOT 60/25 — Known Failures).
#### Checklist
- [ ] Anisotropic NRVM + hiPSC presets (CV_L + CV_T=CV_L/2; dV/dt~110, upper 120).
- [ ] `chip_mesh` (per-axis D), `wavelength_mm`, `points_per_wavelength` implemented.
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_chip.py -v
```
#### Exit Criteria
- [ ] Targets + mesh + λ helpers exist; tests pass.
- [ ] λ-vs-chip tension logged to IDEALOG (see Risk).
#### Risk — **UNRESOLVED design issue (do not claim solved):**
λ = CV·APD ≈ **33 mm (NRVM)** / **18 mm (hiPSC)** at APD≈350 — both exceed the 16 mm chip. The chip-EP **fit is unaffected** (CV/APD are *local*). But the downstream reentry **d/λ obstacle sweep cannot be done by varying obstacle size on a 16 mm chip** at these APDs. **The reentry application (its own plan) must resolve this** — larger-than-physical-chip domain, or rapid-pacing/S2 to shorten APD (hence λ). This PLAN flags it; it does NOT resolve it.

### Phase 2 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_chip.py -v
```
- [ ] Tests pass; λ-vs-chip tension logged.
**-> Commit point**

---

## Phase 3: Fit both baselines (cell + tissue) via cardiac_core (anisotropic)

**Goal**: Produce θ* = (θ_ionic, `D_long/D_trans`) for NRVM and hiPSC-CM, each hitting CV_L and CV_T ≤2% and APD ≤5%. Persist each as a Tier-1 record.
**Tier**: medium

### Phase Context
- Cell fit (MHAS13, tier 2) validated (APD 352, 0.6%). **Reuse `cell_fitter.py` → `cell_runner.py`** for single-cell (no cardiac_core 0-D path). Tissue CV targets (both axes) change per baseline.
- dV/dt constraint 120 V/s (Known Failure: 60 too tight).

### Step 3.1: Run cell + tissue fit for each baseline (monodomain), save records
**Model**: opus
#### Read First
- `Optimizer/V1/tuner/cell_fitter.py`, `cell_runner.py`, `pipeline.py`.
- `Optimizer/V1/run_mhas13.py` — invocation template.
- § Parameter & Preset Storage.
#### Why
Monodomain is the cheapest engine; fit there, validate/port cross-engine in Phase 4.
#### Implementation Spec
**Files to create:** `Optimizer/V1/run_chip_fit.py` — loops `baseline ∈ {nrvm, hipsc}`, runs cell fit + dual-axis tissue fit (`cc_runner`, monodomain), saves θ* + Tier-1 record. Includes helper `select_best(cellres: CellFitResult) -> torch.Tensor` (lowest-scalarized-objective θ; weights `0.4·f₁+0.3·f₂+0.3·f₃` from `ARCHITECTURE.md`).
#### Pseudocode
```
for baseline in [nrvm, hipsc]:
    config  = TuningConfig(ionic_model='mhas13', tier=2, engine='monodomain', ...)
    targets = PARKER_NRVM if baseline=='nrvm' else PARKER_HIPSC      # carries cv_longitudinal + cv_transverse
    cellres    = cell_fitter.fit_cell(config, targets)               # CellFitResult: Pareto front of θ tensors
    theta_t    = select_best(cellres)                                # pick one θ tensor
    theta_dict = theta_to_dict(theta_t, config.tier)                # module fn (tuner.config); dict for the record
    Dres       = tissue_fitter.fit_tissue(theta_t, config, targets) # dual-axis secant → D_long, D_trans
    result = {theta_ionic: theta_dict,
              tissue:{monodomain:{D_long: Dres.D_long, D_trans: Dres.D_trans, dt: config.dt}},
              targets, validation}
    presets.save_record(result, f"chip_{baseline}")                 # Tier 1 JSON
```
*(Real signatures: `fit_cell(config, targets, ...)`, `fit_tissue(theta_ionic, config, targets, ...)` — tier/engine on `config`.)*
#### Test Spec
- `tests/test_pipeline_chip.py::test_chip_fit_smoke` (create) — 1 baseline, tiny budget; returns a result with finite CV_L/CV_T/APD + writes a JSON record (smoke).
#### Checklist
- [ ] `run_chip_fit.py` produces `θ*_nrvm`, `θ*_hipsc` (each with D_long, D_trans).
- [ ] Each fit calls `presets.save_record(...)` → `Optimizer/V1/presets/chip_{baseline}.json`.
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_pipeline_chip.py::test_chip_fit_smoke -v
# Full fit (GATED — run only on "go"): conda run -n heart-conduction python Optimizer/V1/run_chip_fit.py
```
#### Exit Criteria
- [ ] Smoke test passes; full-fit script ready (gated); records written.
#### Risk
Per-axis D bounds may not reach CV_T 2.6 cm/s. Mitigation: warm-start D ∝ CV² predicts both D's — assert each sits inside `TISSUE_PARAMS['D_long']`/`['D_trans']` bounds; widen lower bounds if needed.

### Phase 3 Verification / Exit / Cleanup
- [ ] Both baseline fit scripts ready; smoke tests green; per-axis D bounds verified; Tier-1 records written.
**-> Commit point**

---

## Phase 4: Cross-engine validation via cardiac_core (per axis)

**Goal**: Run each θ* on monodomain/bidomain/LBM through cardiac_core; confirm mono↔bidomain CV ≤~6% (both axes), characterize the mono↔LBM ~35% offset, and recalibrate LBM **per-axis τ** so LBM CV_L/CV_T ≤2%. Enrich records.
**Tier**: medium

### Phase Context
- Reuse `improvement.md` §5 validator *concept* + §6 per-axis anisotropy, via cardiac_core. mono→bidomain: per-axis `D_i,D_e` from `D_long/D_trans` + `sigma_ratio=3.597` (pass explicitly; cardiac_core default 3.59). mono→LBM: `tau_tensor_from_D` → MRT rates `s=1/τ`.
- Expected: mono↔bidomain <6%; mono↔LBM ~35% (memory 73.5/54.3).

### Step 4.1: `cross_engine.py` validator + per-axis LBM τ recalibration
**Model**: opus
#### Read First
- `Optimizer/improvement.md:208-253` — validator concept + conversion (ignore the wrong τ prose; use `tau_from_D`).
- `LBM/Engine_V1/src/diffusion.py` — `tau_tensor_from_D`, `check_stability_tensor` (per-axis MRT; scalar `tau_from_D` is for the isotropic check only).
- § Parameter & Preset Storage (record enrichment).
#### Why
LBM is the *primary* reentry engine; θ* must be re-expressed (per axis) so LBM reproduces the SAME physical CV_L/CV_T, not 35%-inflated.
#### Implementation Spec
**Files to create:** `Optimizer/V1/tuner/cross_engine.py` — `validate(record, config) -> {deltas per axis}`; `recalibrate_lbm(targets, config) -> (D_long, D_trans, s_jx, s_jy)` (per-axis secant on D; convert via `tau_tensor_from_D` then `s=1/τ`).
#### Pseudocode
```
validate(record, config):
   for eng in ['monodomain','bidomain','lbm']:
       cfg = replace(config, engine=eng)
       cv_long[eng]  = cc_runner.run_1d_cable(record.theta_ionic, record.D_long,  cfg, axis='long')
       cv_trans[eng] = cc_runner.run_1d_cable(record.theta_ionic, record.D_trans, cfg, axis='trans')
   return {mono_vs_bidomain_pct, mono_vs_lbm_pct}   # per axis; expect bidomain<6%, lbm ~+35% raw
recalibrate_lbm(targets, config):
   cfg = replace(config, engine='lbm')
   D_long  = secant on D until cc_runner CV(axis='long')  within 2% of targets.cv_longitudinal
   D_trans = secant on D until cc_runner CV(axis='trans') within 2% of targets.cv_transverse
   tau_xx,tau_yy,_ = tau_tensor_from_D(D_long, D_trans, 0, dx, dt_lbm); return D_long, D_trans, (1/tau_xx, 1/tau_yy)   # MRT RATES s=1/τ (tau_tensor returns TIMES)
```
#### Test Spec
- `tests/test_cross_engine.py::test_mono_bidomain_close` (create) — |Δ| ≤ 6% both axes.
- `tests/test_cross_engine.py::test_lbm_recalibration` — after recalibration, LBM CV_L/CV_T within 2%.
- `tests/test_cross_engine.py::test_offset_measured_not_hardcoded` — 35% measured on a pure-diffusion benchmark at our dx/dt, not hardcoded.
#### Checklist
- [ ] Validator reports per-engine, per-axis CV + deltas.
- [ ] LBM per-axis recalibration yields D/τ hitting CV_L/CV_T ≤2%.
- [ ] Enrich each record with `tissue.bidomain`, `tissue.lbm.{D_long,D_trans,tau_long,tau_trans,dx,dt}`, `validation.cross_engine`.
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_cross_engine.py -v
```
#### Exit Criteria
- [ ] mono↔bidomain ≤6% both axes; LBM recalibrated ≤2% both axes; records enriched.
#### Risk
35% is resolution-dependent. Mitigation: measure on a pure-diffusion benchmark at OUR dx/dt; never hardcode 1.35.

### Phase 4 Verification / Exit / Cleanup
- [ ] Cross-engine report for both baselines; per-axis LBM D/τ recorded with per-engine dt.
**-> Commit point**

---

## Phase 5: Hand-off to the reentry application (LBM planar-wave baseline)

**Goal**: Package each tuned chip-EP set as a reusable Lab preset, and demonstrate a clean **planar wave on the LBM chip mesh** (anisotropic) at target CV with measured CV + λ — satisfying the reentry question's "Baseline" criterion.
**Tier**: medium

### Phase Context
- Deliver a loadable preset + a baseline LBM run, NOT the obstacle sweeps (the reentry question's plan — blocked by the λ-vs-chip tension it must resolve).

### Step 5.1: Extend schema, export presets, LBM planar-wave baseline
**Model**: opus
#### Read First
- `cardiac_core/run.py` `run_lbm`; `cardiac_core/analysis.py` `conduction_velocity`, `wavefront_mask`.
- `Lab/presets/_SCHEMA.md` (extend) and `.claude/skills/sim-preset/SKILL.md`.
- § Parameter & Preset Storage.
#### Why
A frozen, named preset is the contract between the build and the reentry application; prevents re-deriving parameters.
#### Implementation Spec
**Files to create:** `Optimizer/V1/run_chip_baseline_lbm.py` — `load_record`, build anisotropic LBM chip mesh (`D_long/D_trans`), planar stim at left edge, run, measure CV + λ; export Lab preset(s).
**Files to modify:** `Lab/presets/_SCHEMA.md` — add `mhas13` ionic, `ionic_scaling:` block, **per-axis** LBM `conductivity.D_long/D_trans` + `tau_long/tau_trans` (keep parameters-only). Document `D` is **cm²/ms** (diffusion), distinct from the sigma firewall (mS/cm) — don't pass D where `sigma` is expected (`conductivity.py`).
**Interfaces:** `tuner/presets.py::export_lab_preset(record, engine='lbm') -> Lab/presets/chip_{baseline}.yaml`.
#### Pseudocode
```
for baseline in [nrvm, hipsc]:
   rec   = presets.load_record(f"chip_{baseline}")
   model = build('mhas13'); apply_scaling(model.params, rec.theta_ionic)
   mesh  = chip.chip_mesh(16, 0.1, D_long=rec.tissue.lbm.D_long, D_trans=rec.tissue.lbm.D_trans)
   times, V = run_lbm(mesh, t_end, ionic_model=model, dt=rec.tissue.lbm.dt)   # per-axis τ from mesh D
   cv = analysis.conduction_velocity(...); lam = chip.wavelength_mm(cv, apd_measured)
   presets.export_lab_preset(rec, engine='lbm')      # → Lab/presets/chip_{baseline}.yaml
```
#### Test Spec
- `tests/test_chip_baseline.py::test_planar_wave_lbm` (create) — small grid; planar front L→R; CV finite > 0; λ ≈ `wavelength_mm(target)`.
- `tests/test_presets.py::test_record_roundtrip` (create) — `save_record`→`load_record`→`to_sim_kwargs` reproduces the same CV_L/CV_T per engine (≤2%).
- `tests/test_presets.py::test_lab_preset_validates` — exported YAML validates against the extended `_SCHEMA.md`.
#### Checklist
- [ ] `_SCHEMA.md` extended (mhas13 + ionic_scaling + per-axis D/tau); parameters-only.
- [ ] `export_lab_preset` writes `Lab/presets/chip_nrvm.yaml`, `chip_hipsc.yaml`.
- [ ] Record round-trip test green.
- [ ] LBM planar wave runs; CV + λ measured + logged.
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_chip_baseline.py Optimizer/V1/tests/test_presets.py -v
```
#### Exit Criteria
- [ ] Reusable preset(s) exist; LBM planar-wave baseline measured.
- [ ] Update `geometry_induced_reentry/README.md` experiment table + check off its "Baseline" criterion.
#### Risk
LBM at dx=0.1 mm + low transverse relaxation rate near the stability floor. Mitigation: `check_stability_tensor` before running; use the per-engine `dt_lbm` recorded in Phase 4.

### Phase 5 Verification / Exit / Cleanup
- [ ] Preset + baseline delivered; reentry README updated; EXPERIMENT.md backlinks added (BOTH research questions + MASTER.md).
**-> Commit point**

---

## Final Cleanup
- float64 consistency across new modules (no float32 leaks).
- `Monodomain/Engine_V5.3/` untouched.
- EXPERIMENT.md backlinks exist for new experiments (link BOTH research questions + MASTER.md).
- No engine logic duplicated in the tuner that cardiac_core already provides.
- Update `MASTER.md` rows for both questions if status changed.
- Update `MASTER_KNOWLEDGE_INDEX.md` if KNOWLEDGE files gained sections.
- Archive the completed plan:
```bash
mkdir -p Research/Active/ionic_model_optimization/plans
cp Research/Active/ionic_model_optimization/PLAN.md "Research/Active/ionic_model_optimization/plans/$(date +%Y-%m-%d)_engine-tuner-cardiac-core-multi-engine.md"
```

## Mutation Log
**MUTATED 2026-06-29**: Header ADDED `Revised` line + "AUDIT-DRIVEN REALITY CHECK" note — audit iter 1 found cardiac_core OO methods stubbed; functional `run.py` API is the real path.
**MUTATED 2026-06-29**: Phase 0 ADDED ("cardiac_core tuning seam") — audit #2: no θ/D injection hook. Hard prerequisite, jointly owned with `engine_consolidation`.
**MUTATED 2026-06-29**: Dropped ≤0.5% CV-parity gate (audit #3) — hand-rolled explicit-Euler runner vs Strang/CN/PCG; re-measure & fit physical target.
**MUTATED 2026-06-29**: Test names, shim-not-delete, single-cell stays in `cell_runner`, `tau_from_D` corrected, per-engine dt, λ>chip flagged, hiPSC provenance (audit #5–#11). Pseudocode added to all steps. (audit #12 LOW.)
**MUTATED 2026-06-29 (audit iter 2)**: Phase 0 seam → pre-built `IonicModel` instance + mesh-D (no layering inversion); presets add `dvdt_max_upper=120`; Step 3.1 real signatures (`fit_cell`/`fit_tissue`); Step 1.1 integer CV indices + `-20` threshold; `.apd()` not `.apd_map()`.
**MUTATED 2026-06-29 (audit iter 3)**: stale `ionic_overrides=`/`D=` pseudocode purged; **LBM factory** (`.lower()` rebuild, api.py:1338/1353) called out as the one to patch; Step 4.1 `replace(config, engine=...)`; θ_ionic dict↔tensor coherence; LOW (`.apd_map` 2nd copy, `TISSUE_PARAMS`, test_tissue 15&36, `cv` key map, bidomain 3-tuple).
**MUTATED 2026-06-29 (audit iter 4 → CONVERGED)**: θ tensor canonical in fit loop (`run_1d_cable: torch.Tensor`, tensor→dict via `theta_to_dict` for scaling/record); `apply_scaling/theta_to_dict/dict_to_theta` de-qualified to module fns in `tuner.config`; `select_best` defined as a `run_chip_fit.py` helper. Audit verdict: 0 critical / 0 high, CONVERGED.

### Domain correction (user, 2026-06-29) — anisotropy restored
**MUTATED 2026-06-29**: **Reverted the isotropic simplification → ANISOTROPIC (~2:1).** Aligned hiPSC-CM/NRVM engineered tissue is anisotropic (Bursac & Parker 2002 ratio ≈2.1; de Diego 2010 2.1±0.8); the iter-1 "isotropic/fabricated-transverse" finding was wrong physics. Restored CV_L **and** CV_T (`= CV_L/anisotropy_ratio`, default 2.0) targets and per-axis `D_long/D_trans` (the tuner's *existing* fields — so mono/bidomain need no special-casing; removed the `isotropic` flag and `D_trans=D_long`). Record schema + presets back to per-axis.
**MUTATED 2026-06-29**: Phase 0 EXPANDED — the real constraint the isotropic detour hid: cardiac_core `lbm()` rejects non-isotropic D (`ValueError`, scalar only). *(This entry's original claim "LBM V1 D2Q5 supports per-axis τ → wrapper-only change" was REFUTED by audit iter 5 — see below.)*

### Audit iteration 5 (2026-06-29) — anisotropy reversion check, NOT CONVERGED → fixed
**MUTATED 2026-06-29**: **LBM-anisotropy mechanism CORRECTED (audit-5 HIGH×2).** Engine source disproved the "D2Q5 per-axis τ = wrapper change" claim: `LBMSimulation` is **BGK-scalar-only**, `mrt_collide_d2q5` is **unwired** (no `lbm_step_d2q5_mrt`; only `lbm_step_d2q9_mrt`), and `tau_from_D` twice is meaningless (BGK has one ω). Phase 0 **split into Step 0.1 (mono/bidomain anisotropy + instance pass-through, small) and Step 0.2 (LBM per-axis via MRT — real engine work in vendored `cardiac_core/_lbm/` + `LBM/Engine_V1`)**. Per-axis relaxation now via `tau_tensor_from_D` + `check_stability_tensor` (D2Q9-MRT recommended; or wire D2Q5-MRT). All `tau_from_D`-twice pseudocode (Steps 1.2/4.1) replaced; record schema LBM → `collision:mrt`+`mrt_rates`.
**MUTATED 2026-06-29**: `create_cardiac_mesh` is **scalar-D only** (audit-5 MEDIUM) — `chip_mesh`/strip mesh build `CardiacMeshData` directly with per-axis `D_xx/D_yy`. Read-First + pseudocode (Steps 1.1, 2.1) corrected.
**MUTATED 2026-06-29**: `improvement.md` re-cited as **V2 design intent, not implemented** (audit-5 MEDIUM); MRT requirement (its §4 + the wrapper guard) made explicit.
**MUTATED 2026-06-29**: LOW (audit-5) — anisotropy ratio band 2.0–2.1 (was 2.0±0.1, which rejected the cited 2.1); Phase 5 stability uses `check_stability_tensor`; flagged the cross-construct provenance (NRVM/monolayer ratio applied to MacQueen's 3-D ventricle CV_L).

### Audit iteration 6 (2026-06-29) — MRT-fix check → CONVERGED
**MUTATED 2026-06-29**: time→rate inversion FIXED (audit-6 MEDIUM) — `tau_tensor_from_D` returns relaxation **times** `(τ_xx,τ_yy,τ_xy)`, but MRT collide consumes **rates** `s=1/τ`. All pseudocode (Steps 0.2/1.2/4.1) now computes `s_jx,s_jy = 1/τ`; noted that the other D2Q9-MRT moment rates (`s_e,s_eps,s_q,s_pxx,s_pxy`) are free stability params (~1.0), only `s_jx/s_jy` carry D; `recalibrate_lbm` returns `(D_long,D_trans,s_jx,s_jy)`.
**MUTATED 2026-06-29**: LOW (audit-6) — `check_stability_tensor(...)[0]` (it returns `(is_stable, τ_min)` — bare assert was always truthy); residual "`tau_from_D` per axis" vocabulary (Architecture/storage/Phase-4/checklist) → `tau_tensor_from_D` → `s=1/τ`.
**CONVERGED 2026-06-29**: audit iteration 6 returned 0 critical / 0 high; auditor verdict CONVERGED (all engine-source claims verified accurate). Two audit arcs total — (A) iters 1–4 to converge the cross-plan structure under an isotropic assumption; (B) iters 5–6 after the user's anisotropy domain-correction, converging the per-axis/MRT mechanism. Plan is execution-ready pending the user's "go" gate. **Scope note:** Phase 0 Step 0.2 (LBM per-axis anisotropy via MRT) is genuine engine-level work — the single largest item — added because anisotropy is required and the LBM engine is BGK-only today.

### Step 0.2 expansion (user request, 2026-06-30)
**EXPANDED 2026-06-30**: Step 0.2 detailed into **three ordered sub-steps** grounded in verified engine source — **0.2a** wire MRT into `LBMSimulation` (collision selector + per-axis D + `step()` dispatch to `lbm_step_d2q9_mrt`, whose signature differs from BGK; `s_jx/s_jy = 1/τ` from `tau_tensor_from_D`; 5 free rates ~1.0 per `test_phase2.py`); **0.2b** route the `lbm()` wrapper (replace the `is_isotropic` ValueError with MRT routing on `D_xx≠D_yy`); **0.2c** analytic anisotropic-diffusion benchmark (σ²=2Dt per axis) as the correctness gate for the `s→D` mapping. Key finds embedded: `mrt_collide_d2q9` already encodes per-axis (`s_jx`→D_xx, `s_jy`→D_yy) and `test_phase5.py` already exercises it with `s_jy`; guards added for `weights_mode='uniform_8'` (MRT assumes canonical cs²=1/3), `lattice='d2q5'` (no `lbm_step_d2q5_mrt`), and `D_xy≠0` (oblique fibers → out of scope, deferred Phase-8 rotation). No mechanism change — elaboration of the iter-6-converged design.
**MUTATED 2026-06-30 (audit-7 on expanded 0.2)**: Engine wiring all verified accurate (signatures, S-vector free rates, vendored copy byte-identical, guards, f-init). Fixed: **(HIGH)** 0.2c benchmark now MANDATES `dx≠dt` — the correct mapping `τ=0.5+D·dt/(cs²·dx²)` and the wrong lattice-unit form `τ=0.5+D/(cs²·dt)` coincide at `dx=dt`, so a `dx=dt=1` benchmark (as in `test_5v4/5v5`) couldn't catch the scaling slip. **(MEDIUM)** Read-First warns NOT to copy `test_phase5`'s `s = 1/(0.5+D/(cs²·dt))` mapping (dx=dt=1-degenerate) — authoritative mapping is `s=1/τ` via `tau_tensor_from_D`. **(LOW)** import target corrected to `simulation.py` (step.py already has the MRT fns); 0.2b round-trips anisotropy via mesh `data.D_yy` (not build_kwargs); 0.1→0.2b ordering dependency made explicit (instance `.lower()` crash).
**MUTATED 2026-06-29**: Phases 1–5 + storage re-anisotropized — `cc_runner.run_1d_cable(..., axis=)` per direction; `fit_tissue` dual-axis secant (CV_L→D_long, CV_T→D_trans); LBM per-axis `τ_long/τ_trans`; cross-engine validates both axes; chip presets `cv_transverse=cv_long/2`.

### Implementation (2026-06-30) — branch `engine-tuner-cardiac-core`
**IMPLEMENTED**: all 6 phases coded + tested + committed (8 commits, ~30 new tests; no regressions — 32 upstream LBM + cardiac_core suites green).
- P0: `create_cardiac_mesh(D_yy=)`, `lbm()` instance seam, D2Q9-MRT in `LBMSimulation` (vendored+upstream) + `lbm()` routing; dx≠dt benchmark proves `s=1/τ` (≤8%).
- P1: `cc_runner.py` (CV via functional API + scaled instance, mono/bidomain/lbm; CV∝√D). **tissue_fitter rewire DEFERRED** (would perturb the passing legacy suite; cc_runner is the parallel new path).
- P2: `chip.py` (161² mesh + anisotropic Parker targets). P3: `presets.py` Tier-1 records + `run_chip_fit.py` (smoke; full BayesOpt fit GATED). P4: `cross_engine.py` (mono↔bidomain CV_T ~12%, mono↔lbm ~29%). P5: `export_lab_preset` + `_SCHEMA.md` ext + `run_chip_baseline_lbm.py`.
- **Finding**: effective-D meshes require **chi=1.0** (FDM divides by chi; chi=1400 silently kills propagation). Remaining: the gated full fit run + optional tissue_fitter rewire.
