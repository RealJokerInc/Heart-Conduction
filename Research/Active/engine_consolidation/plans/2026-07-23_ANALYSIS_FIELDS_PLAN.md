# PLAN: cardiac_core `analysis.fields` branch + scalar EP metrics + `single_cell`

Created: 2026-07-22
Engine(s): cardiac_core (analysis layer; no solver changes)
Research question: [engine_consolidation](../Research/Active/engine_consolidation/README.md)
Source: [ANALYSIS_FIELDS_DESIGN.md](./ANALYSIS_FIELDS_DESIGN.md) (API + per-field math + operator toolkit + cited
calculations) · [ANALYSIS_METHODS_PRIOR_ART.md](./ANALYSIS_METHODS_PRIOR_ART.md) (prior art + structured-grid defaults
+ recommended build order) · [engine_consolidation IDEALOG](../Research/Active/engine_consolidation/IDEALOG.md).

> **AUDIT: CONVERGED (2026-07-22, 5 rounds, Opus adversarial).** R1 1C/6H/10M/4L → R2 1C/1H/6M/6L → R3 0C/1H/1M/5L →
> R4 0C/1H/2M/1L → **R5 0C/0H/0M/0L (verdict CONVERGED)**. Each round's remaining finding was a narrow follow-on to
> the prior round's own fix (χ source, CV-family threshold flip, ionic-model identity threading), all closed with
> code-verified attributes + guarding tests. The plan is execution-ready; nothing else is outstanding. **This is a
> PLAN — implementation is a separate, explicit user go (hard gate).**

## Objective
Implement the user-facing analysis-fields layer designed in `ANALYSIS_FIELDS_DESIGN.md`: a canonical LAT, a
torch/on-device operator toolkit (`grad`/`div`/`curl`/`laplacian` + integrals) that honors the solver's boundary
mode and mask, the named physical fields (`r.fields.source_sink`/`velocity`/`curvature`/`vorticity`/…), and the
adjacent scalar EP metrics (`wavelength`/`erp`/consolidated `apd`) + a 0-D `single_cell` mode and a `safety_factor`.
Every operation is documented with its math + cited stencil in the design doc; this plan turns that into test-gated,
dependency-sorted code. **No solver code changes** — this is a pure analysis/wrapper layer, so the per-engine
integrity goldens stay bit-identical.

## Success Criteria
- [ ] One **canonical LAT** (interpolated crossing, single threshold, torch) that `r.lat()` + `r.cv()` +
      `r.cv_between()` + `r.radial_cv()` + eikonal + all LAT-based fields route through; the −20-nearest vs −40-interp
      split is gone (documented).
- [ ] `fields.derivatives` (`grad`/`div`/`curl`/`laplacian`) torch/on-device, honoring `boundary_mode` + `domain_mask`,
      with `div(grad)==laplacian` (staggered). **Exactness scoping:** the discrete *divergence-theorem face-flux*
      identity (`∬div = Σ boundary face flux`, B5 telescoping) is exact to ~1e-12; the B4 *polyline* contour flux is
      only O(h^1.5) on a staircase mask — these are DIFFERENT checks (do not claim 1e-12 for the polyline).
- [ ] `r.fields.<name>` accessor exposing `voltage_gradient`/`voltage_flux`/`source_sink`/`electric_field`/
      `current_flux`/`velocity`/`direction`/`speed`/`curvature`/`vorticity` (lazy, cached for the result's lifetime).
- [ ] `fields.integrals` (`conduction_time`/`net_flux`/`circulation`/`winding_number`/isochrone family/co-area/
      activated-area/region-load) with the Stokes/divergence-theorem consistency tests green.
- [ ] Scalar EP: `wavelength` (λ=CV·ERP, ÷1000 units, ERP default), consolidated `apd` (multi-beat baseline fixed in
      `apd_at` AND `restitution_curve` AND `apd_per_beat`), `di`, protocol-based `erp`.
- [ ] `single_cell()` 0-D via the shared per-node ionic step; `safety_factor` (Boyle–Vigmond) on top of `source_sink`.
- [ ] All existing cardiac_core tests pass (no regressions). **Golden discipline:** the per-engine *solver-integrity*
      goldens stay bit-identical (atol=0) — solvers are untouched; the *analysis* CV/LAT numbers WILL move when
      `r.lat()`/`r.cv()` flip to the canonical interp LAT — that move is a controlled, documented regolden logged in
      the Mutation Log (never silent; a user-visible decision per the #13/#14 deferral pattern).

## Architecture Changes
- MOD: `cardiac_core/run.py:20-41` (`SimulationResult` dataclass) — add **7 fields**:
  `domain_mask: Optional[Tensor]=None`, `boundary_mode: str="face_mirror"`, `Cm: Optional[float]=None`,
  `chi: Optional[float]=None`, `conductivity=None` (the resolved **effective** diffusion field(s) / σ tuples the
  fields need — see Step 1.1), `ionic_model: Optional[str]=None`, `cell_type: Optional[str]=None` (resolved model
  identity, for Phase-3/7 `I_ion` re-eval); new `fields` property returning a `Fields` accessor.
- MOD: **BOTH** result builders thread the new fields: the `simulate()`/`_collect` result build at `run.py:321`
  (has `sim`; `_collect` at run.py:97 returns raw tensors, the `SimulationResult(...)` is constructed at :321) AND
  `api.py:1348-1371` (`_result_from`, called from `self.run` at api.py:255/262/265 — pass `self` or the needed
  fields), AND the empty-run branch `api.py:1360`. Source: `sim._data` (D_xx/D_yy/D_xy or σ_i/σ_e, `chi`, `Cm`,
  `mask`), `sim.dx/dy` (api.py:1046/1051), and the sim's stored `boundary_mode` (see next).
- MOD: `cardiac_core/api.py:1469+` — the factory takes `boundary_mode` but does not persist it; **store it on the sim**
  (`self._boundary_mode`) so the result builder can read it (currently it only reaches the discretization).
- MOD: `cardiac_core/run.py:49-93` — repoint `r.lat()`/`r.cv()`/`r.cv_between()`/`r.radial_cv()` to the canonical LAT.
- NEW: `cardiac_core/fields/__init__.py` — `Fields` accessor (named cached fields) + `VectorField` wrapper.
- NEW: `cardiac_core/fields/derivatives.py` — `grad`/`div`/`curl`/`laplacian` (staggered adjoint core + boundary/mask)
  + one shared **wrapped-loop-sum** primitive (winding number) reused by vorticity/circulation/Gauss-Bonnet/PS.
- NEW: `cardiac_core/fields/integrals.py` — line/region integrals, marching-squares contours.
- MOD: `cardiac_core/analysis.py` — canonical `activation_time` path (interp, single threshold); `wavelength`,
  `di`, `safety_factor`; fix the `V_rest=trace[0]` baseline in `apd_at`(110) + `restitution_curve`(458) +
  `apd_per_beat`(738); refactor `phase_singularities`(356) onto the shared loop-sum. `erp` goes in a NEW module
  (not analysis.py — it must call `simulate`, and analysis is imported by run/api → circular-import; see Step 6.2).
- NEW: `cardiac_core/single_cell.py` — 0-D driver over the shared `cardiac_core.ionic` per-node step.
- NEW: `cardiac_core/protocols.py` — `erp` (S1S2 + bisection; runs sims; lives above the analysis layer).
- NEW tests: `test_fields_derivatives.py`, `test_fields_integrals.py`, `test_fields_named.py`,
  `test_canonical_lat.py`, `test_scalar_ep.py`, `test_single_cell.py`.

## Known Failures / gotchas (from IDEALOG + prior art)
- Nearest-save-frame LAT quantizes ∇T → flat-neighbor `|∇T|→0` singular CV. **Use interpolated LAT** (DESIGN § 1).
- **Collocated `div(grad)` ≠ compact Laplacian** — it's the wide 2h checkerboard stencil. Use the **staggered
  `div=−grad*`** pair (DESIGN § A8). Do NOT build the Laplacian as central-grad-then-central-div.
- **`numpy.gradient` one-sided edges are NOT no-flux** — a post-hoc ∇²V would imply a spurious boundary flux at the
  reduced-sink edge. Boundary MUST be the solver's ghost/mirror (`face_mirror` ≙ scipy `reflect`; DESIGN § A6).
- **conduction_time integrates SLOWNESS ∇T, not `velocity`** (`∫v·dl ≠ ΔT`; only ∇T is curl-free). DESIGN § 7.
- **A single map cannot separate CV_L/CV_T** — do not advertise anisotropy from one activation map (DESIGN § 8.5).
- Bidomain scar φ_e disagrees pcg_spectral-vs-pcg (~pre-existing M4) — the `current_flux`/`electric_field` fields
  inherit that; document, don't try to fix here.
- A green monodomain test says nothing about bidomain/LBM — test each field on every engine it claims to support.
- **Use `D_eff = D_raw/(χ·Cm)`, NOT raw D** for `voltage_flux`/`source_sink` — the solver steps `∇·(D_eff∇V)`
  (api.py:1703); raw D is off by χ·Cm. (R1-audit HIGH.)
- **`SimulationResult` is immutable** — do NOT design a cache invalidated by `scale_conductance`/`reset` (those are
  SIM methods; they produce a NEW result). The accessor cache is valid for the result's lifetime. (R1-audit HIGH.)
- **`front_metrics`/`fit_eikonal` stay as-is** (DESIGN says migrate LATER) — do NOT assert the new torch fields match
  the numpy path to 1e-6 (different interior+boundary stencils; infeasible). (R1-audit HIGH.)
- **Two result builders** (`run.py:321` + `api.py:1371` + empty-run `api.py:1360`) — thread both or `.run()`-path
  results are silently mask/conductivity-unaware. (R1-audit HIGH.)
- **The compact 5-point Laplacian matches only the FDM-5pt operator** — FEM/FVM/9-pt(`moore8`) won't; gate the
  solver-match test to FDM and document those as "5-point reconstructions." (R1-audit MED.)

---

## Phase 1: Foundations — carry mask/boundary_mode + canonical LAT

**Goal**: `SimulationResult` and `Grid` carry the `domain_mask` + `boundary_mode` the field ops need; one canonical
interpolated LAT that `r.lat()`/`r.cv()` route through. Independently deliverable (fixes the open LAT triple-def issue).
**Tier**: large
**Estimated scope**: 1 dataclass extension + 1 grid attr + 1 analysis function refactor + result-hook repoint + tests.

### Phase Context
`SimulationResult` (run.py:20) is a dataclass: `times, Vm, phi_e, dx, dy, ionic_states`. It carries **no
conductivity, Cm, mask, or boundary_mode** — the exact gap this phase fills, because Phase 3 (`source_sink=div(D∇V)`,
`current_flux=−σ∇φ_e`) and Phase 7 (SF numerator `Cm·ΔV`) NEED them. **Two builders** construct results and BOTH must
thread the new fields: `run.py:321` (`_collect`, gets `sim`) and `api.py:1348-1371` (`_result_from`, gets only
`dx,dy` — its callers `self.run` at api.py:255/262/265 must also pass `self`/the fields), plus the empty-run branch
`api.py:1360`. The sources live on the sim: `sim._data.mask` (api.py:1056 `sim.mask`), `sim._data.Cm` (api.py:1537),
the conductivity fields `sim._data.D_xx/D_yy/D_xy` OR `sigma_i/sigma_e` (see `_rebuild_with_conductivity` api.py:499),
`sim.dx/dy` (1046/1051), and `chi`. **`boundary_mode` is a factory arg (api.py:1469) NOT persisted** — store it as
`self._boundary_mode`. `Grid` (grid.py:20) has `Nx,Ny,dx,dy,mask`, no boundary_mode. Existing LAT:
`analysis.activation_time` (−20 mV, nearest, torch); `activation_time_interp` (−40 mV, linear interp, numpy);
`conduction_velocity` computes its OWN −20 nearest crossing; `cv_between`(666)/`radial_cv`(696) likewise. Canonical
LAT = torch, interpolated, single threshold. Do NOT delete `activation_time_interp` (source_sink_mismatch research
imports it) — alias it. **Golden discipline: solver-integrity goldens must stay atol=0; analysis CV/LAT numbers may
move (Step 1.3) — that is a documented regolden, not a silent one.**

### Step 1.1: Carry `domain_mask` + `boundary_mode` + `conductivity` + `Cm` + `chi` on `SimulationResult`
**Model**: opus
#### Read First
- `cardiac_core/run.py:20-95` — `SimulationResult` dataclass + analysis hooks.
- `cardiac_core/run.py:97-321` — `_collect` (result builder #1, has `sim`).
- `cardiac_core/api.py:1348-1371` — `_result_from` (result builder #2) + callers api.py:255/262/265.
- `cardiac_core/api.py:499-560` (`_rebuild_with_conductivity`) — the TWO conductivity representations (D_xx fields
  vs σ_i/σ_e); api.py:1046/1051/1056 (`sim.dx/dy/mask`); api.py:1469/1537 (`boundary_mode` arg, `data.Cm`); api.py:1703
  ("D_xx is RAW; effective = D/(χ·Cm)").
- `cardiac_core/grid.py:20-97`.
#### Why
The field ops must honor the SAME edge treatment, mask, AND conductivity/Cm the solver used, or `source_sink` is off
by χ·Cm (raw-vs-effective) and edge/mask ops apply the wrong ghost rule. The result is the only object the user has.
#### Implementation Spec
**Add to `SimulationResult`:** `domain_mask: Optional[Tensor]=None`, `boundary_mode: str="face_mirror"`,
`Cm: Optional[float]=None`, `chi: Optional[float]=None`, `conductivity=None`, `ionic_model: Optional[str]=None`,
`cell_type: Optional[str]=None` — where `conductivity` is a small resolved holder of the **effective** diffusion
field `D_eff = D_raw/(χ·Cm)` (mono/LBM) or the σ tuples (bidomain), plus raw D (both, so Phase 3 can pick). The
`ionic_model`/`cell_type` names let Phase 3/7 rebuild the model (`build_ionic_model(ionic_model, cell_type)`) and
re-evaluate `I_ion = model.compute_Iion(Vm, ionic_states)` for the reaction-identity + safety-factor — the immutable
result carries no model object, so it carries the identity to reconstruct one. Provide `_conductivity_from(data)`.
**⚠️ χ SOURCE (R2-critical):** χ is `sim._data.chi` — **there is NO `sim._chi`/`sim.chi`**. The solver forms
`chi_Cm = data.chi * data.Cm` (api.py:1671/1706/1859), and **declarative monodomain sets `data.chi=1`** (Form-A,
conductivity.py:123/134 — `for_monodomain` returns `'chi':1.0`, "engine chi=1 (inert)"). So compute
`D_eff = D_raw/(data.chi * data.Cm)` — a hard-coded
1400 fallback would be **1400× wrong** on the declarative path (chi=1). Cm comes from `data.Cm` (`sim.Cm` at
api.py:1069 already does this). **Isotropy:** `D_eff` here is the scalar/isotropic case; carry `D_xy`/`D_xx≠D_yy`
too and (Phase 3) assert-or-warn on anisotropy (`∇·(D∇V)` is a tensor contraction, not `D·∇²V`).
**Persist boundary_mode:** add `self._boundary_mode = boundary_mode` in the factory (api.py:1469 region) + a public
`sim.boundary_mode` read.
**Thread BOTH builders:** `_collect`→`simulate()` result build (run.py:321, has sim → pull all) and `_result_from`
(api.py:1371; add params; callers api.py:255/262/265 pass `self._data`, `self.boundary_mode`). Empty-run branch
(api.py:1360) too.
**Grid:** add `boundary_mode` kwarg (default `"face_mirror"`) + repr, for the Grid-construction path.
#### Pseudocode
```
# _conductivity_from(data): everything from `data` — no sim._chi
chi_Cm = data.chi * data.Cm                       # == the solver's factor (api.py:1671)
D_eff  = data.D_xx / chi_Cm                        # mono/LBM isotropic; also carry D_yy, D_xy, raw
# bidomain branch: keep sigma_i/sigma_e (no D_eff); source_sink on bidomain → clear error (Phase 3)
# builder (both paths):
res = SimulationResult(times, Vm, phi_e, dx=sim.dx, dy=sim.dy, ionic_states=ionic,
                       domain_mask = _as_tensor(sim.mask, dev, torch.bool) if not _all(sim.mask) else None,
                       boundary_mode = _resolve_boundary_mode(sim),   # see note: mono→_boundary_mode; bidomain→BoundarySpec
                       Cm = sim._data.Cm, chi = sim._data.chi,
                       conductivity = _conductivity_from(sim._data),  # np→tensor onto (dev, float64) inside
                       ionic_model = _resolved_ionic_name(sim),       # RESOLVED name actually run, NOT raw data.ionic_model
                       cell_type = (sim._data.group_cell_types[0] if sim._data.group_cell_types else "ENDO"))
# NOTE cell_type: CardiacMeshData has NO `cell_type` attr — it has `group_cell_types` (list); mirror api.py:1562.
# NOTE ionic name: prefer the resolved `ionic = ionic_model or data.ionic_model` (api.py:1509/1637; build_kwargs
#   ['ionic_model']) — with an explicit ionic_model= override on a legacy mesh, raw data.ionic_model is stale.
# NOTE model rebuild (Phase 3/7): build_ionic_model(name, cell_type, device=r.Vm.device) — the registry defaults
#   device='cuda' (registry.py:24) but the run may be cpu (api.py:1471) → PASS the result's device (Vm.device) or
#   compute_Iion raises a cross-device error.
# NOTE device/dtype: sim.mask and data.D_xx are NumPy (api.py:1054-1056) → convert to torch on the result's
#   device/float64 (the field ops are torch). NOTE bidomain: it has no `boundary_mode` arg (uses BoundarySpec
#   bath/insulated) → `_resolve_boundary_mode` maps that to a mode string; do NOT blindly getattr→"face_mirror".
```
#### Test Spec
- `test_canonical_lat.py::test_result_carries_context` — masked Grid run → `r.domain_mask is not None`,
  `r.boundary_mode=="face_mirror"`, `r.Cm==1.0`; full-rect run → `domain_mask is None`.
- `::test_declarative_chi_is_one` (**the R2-critical guard**) — a **declarative** `monodomain(...)` run has
  `r.chi==1.0` (Form-A) and `r.conductivity.D_eff == r.conductivity.D_raw/(r.chi*r.Cm)` to 1e-12 — NOT off by 1400.
  (A hard-coded-1400 implementation fails this.)
- `::test_both_builders_thread` — the `.run()` path (api `_result_from`) AND the `simulate()` path (run.py:321) both
  populate the new fields (guards the "silently None on the .run() path" bug).
- `::test_bidomain_result_has_sigma` — a bidomain run's `r.conductivity` exposes σ_i/σ_e (and `D_eff is None`).
- `::test_result_carries_model_identity` — an **EPI** mesh (`group_cell_types=['EPI']`) and/or an explicit
  `ionic_model=` override → `r.cell_type=="EPI"` and `r.ionic_model` is the RESOLVED name actually run (guards the
  "always-ENDO getattr" + "stale data.ionic_model" bugs); rebuilding `build_ionic_model(r.ionic_model, r.cell_type,
  device=r.Vm.device)` reproduces the run's model on the right device.
#### Checklist
- [ ] Add **7 fields**: `domain_mask`, `boundary_mode`, `Cm`, `chi`, `conductivity`, `ionic_model`, `cell_type`
  (defaults preserve back-compat).
- [ ] `_conductivity_from` helper (D_eff = raw/(χ·Cm); σ for bidomain; raw retained; np→tensor on run device/f64).
- [ ] `ionic_model` = RESOLVED name; `cell_type` = `group_cell_types[0]`; both feed a **device-aware**
  `build_ionic_model(name, cell_type, device=Vm.device)` in Phase 3/7.
- [ ] Persist `boundary_mode` on the sim (`_resolve_boundary_mode` for bidomain); thread BOTH builders + empty-run.
- [ ] `Grid.boundary_mode`.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/test_canonical_lat.py -q -k "context or builder or sigma"`
#### Exit Criteria
- [ ] Both build paths populate mask/boundary_mode/Cm/chi/conductivity; D_eff arithmetic correct; no existing test breaks.
#### Risk
Missing `_result_from` (api.py, not run.py) → `.run()`-path results silently mask/conductivity-unaware (a
silent-wrong on scar domains, not a crash, due to None defaults). Mitigation: the `test_both_builders_thread` test
exercises BOTH paths explicitly. Bidomain σ vs mono D_eff branch — `_conductivity_from` must handle both (mirror
`_rebuild_with_conductivity`'s dual representation).

### Step 1.2: Canonical interpolated LAT (torch, single threshold)
**Model**: opus
#### Read First
- `cardiac_core/analysis.py:19-58` (`activation_time`), `:501-534` (`activation_time_interp` — note the `denom`
  guard at :528), `:59-108` (`conduction_velocity`).
- DESIGN § 1 (LAT) + § "GATE" ; PRIOR_ART § 1 + § 8.1 (parabolic dV/dt refine).
#### Why
`r.lat()`/`r.cv()`/eikonal must agree on one LAT. The eikonal CV field needs a smooth sub-frame LAT (nearest-frame
staircases → `|∇T|→0`).
#### Implementation Spec
Extend `activation_time(V, times, threshold=-20.0, *, method="nearest")` — **default UNCHANGED in this step**
(`method="nearest", threshold=-20`, bit-identical to today, so viz/`radial_cv` do NOT move mid-phase). ADD an opt-in
`method="interp"` path (linearly-interpolated first-crossing, torch/on-device, NaN at non-activating nodes) — the
canonical LAT, used at −40. The default FLIPS to `method="interp", threshold=-40` in Step 1.3 (with the regolden +
viz handling). Add `max_dvdt_time(V, times)` (parabolic sub-frame peak of dV/dt) for probe reference.
#### Pseudocode
```
above = V >= threshold                      # (T,Nx,Ny) bool
first = above.int().argmax(0); ever = above.any(0)
Vk  = gather(V, first); Vk1 = gather(V, clamp(first-1, min=0))
denom = Vk - Vk1
frac  = torch.where(denom.abs() > eps, (threshold - Vk1)/denom, 0.0)   # keep the denom guard (analysis.py:528)
lat   = times[clamp(first-1,min=0)] + frac*(times[first]-times[clamp(first-1,min=0)])
lat[first==0] = times[0]                    # activated at t0 (before the frac math for those nodes)
lat[~ever]    = nan
# max_dvdt_time: dVdt = central diff along t; k=argmax; parabolic offset = 0.5*(y[k-1]-y[k+1])/(y[k-1]-2y[k]+y[k+1])
```
#### Test Spec
- `test_canonical_lat.py::test_interp_subframe` — synthetic trace crossing −40 between frames → LAT = analytic
  crossing < 1e-9 (torch). Uses an **inline finite-difference** for any gradient (the grad operator is Phase 2).
- `::test_nearest_reproduces_old` — `method="nearest"` bit-matches the old `activation_time` (−20) output.
- `::test_nonactivating_is_nan`; `::test_on_cuda` (skip if no cuda) — no numpy in the path.
- `::test_max_dvdt_parabolic` — a known upstroke → peak time to sub-frame accuracy.
#### Checklist
- [ ] Interp path fully torch, device-preserving, denom-guarded.
- [ ] `method="nearest"` exact regression.
- [ ] `max_dvdt_time` parabolic refine.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/test_canonical_lat.py -q`
#### Exit Criteria
- [ ] Interp LAT sub-frame + torch + NaN-safe + denom-guarded; nearest path unchanged.
#### Risk
`first==0` underflow at `first-1` → `clamp(min=0)` + set the `first==0` branch to `times[0]` explicitly. Flipping the
DEFAULT of `activation_time` to interp/−40 has BLAST RADIUS: `radial_cv`(696, **delegates** to `activation_time`),
`viz.py:78` (`activation_isochrones` calls it with defaults), and `test_analysis.py` no-arg/−20 cases (56/77).
(`apd_at`/`apd_map` do NOT call `activation_time` — `apd_at`:149 does its own crossing — so they are NOT in the blast
radius.) Step 1.3 does the deliberate flip + regolden; **here KEEP the default at −20/nearest** and expose interp
opt-in only, so nothing moves mid-1.2.

### Step 1.3: Route `r.lat()`/`r.cv()`/`r.cv_between()`/`r.radial_cv()` through the canonical LAT
**Model**: opus
#### Read First
- `run.py:49-93` (`cv`/`lat`/`cv_between`/`radial_cv` hooks), `analysis.py:59-108` (`conduction_velocity`),
  `:649-711` (`cv_between`, `radial_cv` — each has its OWN inline −20 nearest crossing).
- DESIGN § 13 (route the scalar CV family through the canonical LAT too).
#### Why
Close the split everywhere a casual user reaches CV — not just `conduction_velocity`. `cv_between` and `radial_cv`
each re-derive a −20 nearest crossing (analysis.py:666/696), so leaving them would keep a residual split.
#### Implementation Spec
**Flip `activation_time`'s DEFAULT** to `method="interp", threshold=-40` — the deliberate canonicalization. This
alone canonicalizes only `r.lat()` and `viz.activation_isochrones` (viz.py:78 calls `activation_time` with no args).
**⚠️ The scalar-CV family carries its OWN `threshold=-20.0` default and does NOT inherit `activation_time`'s** —
`conduction_velocity`(analysis.py:66), `cv_between`(:656), and `radial_cv`(:685, which passes `threshold` EXPLICITLY
to `activation_time` at :696). So flip **each of their own default thresholds to −40** AND route them through
`activation_time(method="interp")` — otherwise you get a NEW −20-interp-vs-−40-interp split (the exact bug this step
must eliminate). `conduction_velocity`/`cv_between` currently inline their crossing (:59-108/666) → replace with the
canonical `activation_time`. **Keep `threshold=`/`method=` as OVERRIDES** (callers can still pass −20), but the
DEFAULT is −40/interp everywhere. Document the change + the reentry limit (first-crossing LAT invalid → use phase) in
each docstring + the cheatsheet.
#### Test Spec
- `::test_lat_cv_agree` — same run: `r.cv(...)` (secant on canonical LAT) and an **inline-FD** eikonal `1/|∇T|`
  along that line agree to discretization error (< 3%). (grad operator is Phase 2 — use inline FD here.)
- `::test_cv_family_uses_canonical` — `conduction_velocity`/`cv_between`/`radial_cv` now default to **both** interp
  AND −40 (assert the DEFAULT crossing threshold is −40, not just method=interp — the guard against the residual
  −20-interp split); `method="nearest", threshold=-20` overrides reproduce the old value exactly.
- Regolden: run the FULL suite; the ANALYSIS CV/LAT goldens (`test_analysis.py`, any CV golden) WILL move — update
  them deliberately and log each in the Mutation Log; the SOLVER-integrity goldens (`test_integrity.py`) must be
  UNCHANGED (atol=0) — if one moves, STOP (that means a solver path was touched, which it must not be).
#### Checklist
- [ ] Four hooks + three analysis fns repointed; overrides preserved on all.
- [ ] Cheatsheet/docstrings updated (−40 interp canonical; −20 nearest available; reentry limit).
- [ ] Analysis goldens updated + logged; integrity goldens confirmed bit-identical.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q -k "lat or cv or analysis or integrity"`
#### Exit Criteria
- [ ] All CV entry points canonical; overrides intact; analysis goldens re-pinned+logged; integrity goldens atol=0.
#### Risk
Silently regoldening the ANALYSIS numbers hides a real change — log every moved value. An integrity-golden move means
a solver path was inadvertently touched → hard stop (this is a pure analysis layer).

### Phase 1 Verification
`conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q`
### Phase 1 Exit Criteria
- [ ] Result/Grid carry mask+boundary_mode; canonical interpolated LAT; hooks routed; no unintended golden moves.
### Phase 1 Cleanup
- float64 consistency (no float32 leak); torch-only LAT path (no numpy); V5.3 untouched; no cross-engine dup.
**-> Commit point: git commit after Phase 1 passes.**

---

## Phase 2: Operator toolkit — `fields.derivatives`

**Goal**: torch/on-device `grad`/`div`/`curl`/`laplacian` with the staggered `div=−grad*` core, honoring
`boundary_mode` + `domain_mask`; the discrete divergence theorem exact.
**Tier**: large
**Estimated scope**: 1 new module + the boundary/mask machinery + consistency tests.

### Phase Context
This is the machinery every named field composes from (DESIGN § "operator toolkit" + § "Calculations" A1–A8).
Accept `(Nx,Ny)` or `(T,Nx,Ny)` (leading dims absorbed). Vectors stored `(...,2)`. Default `boundary_mode`
`"face_mirror"` ≙ scipy `reflect` ghost pad. Internal `domain_mask`: sever-the-connection (mirror the live node
across a masked face; masked nodes → NaN). **Implement operators as fixed conv/slice-subtract on device.** Keep the
exact staggered pair for the Laplacian + integral tiers; average faces→nodes only for user-facing vector display.

### Step 2.1: Central-difference `grad`/`div`/`curl` + ghost-mirror boundary
**Model**: opus
#### Read First
- DESIGN § "Calculations" A1–A3, A6; PRIOR_ART § 4 (boundary modes) + § 8.3.
- `cardiac_core/grid.py` (spacing/mask access).
#### Why
The primitive operators; boundary MUST match the solver (ghost mirror), not one-sided edges.
#### Implementation Spec
**File:** `cardiac_core/fields/derivatives.py`.
**Signatures:** `grad(f, dx, dy, *, boundary_mode="face_mirror", mask=None) -> Tensor(...,2)`;
`div(F, dx, dy, ...) -> Tensor`; `curl(F, dx, dy, ...) -> Tensor`.
#### Pseudocode
```
pad f with ghost per boundary_mode (face_mirror: replicate edge = scipy 'reflect'); mask: mirror live node across
masked face, set masked nodes NaN afterward
Dx = (f[i+1]-f[i-1])/(2dx) ; Dy analogous            # A1 2nd-order central
grad = stack(Dx, Dy, dim=-1)
div  = Dx(Fx)+Dy(Fy) ; curl = Dx(Fy)-Dy(Fx)          # A2
```
#### Test Spec
- `test_fields_derivatives.py::test_grad_linear_exact` — `f=3x+2y` → grad=(3,2) to ~1e-10 interior.
- `::test_curl_of_gradient_zero` — `curl(grad(f))` ≈ 0 (< 1e-9) for random smooth f.
- `::test_boundary_is_noflux` — with `face_mirror`, the edge normal derivative of a mirror-symmetric field is 0
  (vs numpy one-sided which is nonzero) — the load-bearing distinction.
- `::test_mask_noflux` — a hole: gradient at the hole rim uses the mirror, masked nodes NaN; no blow-up.
#### Checklist
- [ ] Ghost pad per `boundary_mode`; sever-connection mask; masked→NaN.
- [ ] Batches over leading `(T,…)`; device/dtype preserved.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/test_fields_derivatives.py -q`
#### Exit Criteria
- [ ] grad/div/curl exact on polynomials; `curl(grad)≈0`; boundary is no-flux; mask honored.
#### Risk
Even/odd decoupling if you build `div(grad)` from these (that's Step 2.2's staggered fix). Keep collocated central
for user-facing grad/div/curl; do the Laplacian separately.

### Step 2.2: Staggered `div=−grad*` core → compact Laplacian + `laplacian()`
**Model**: opus
#### Read First
- DESIGN § "Calculations" A4 (⚠️ wide vs compact), A8 (staggered); PRIOR_ART § 4 (mimetic).
#### Why
`div(grad)` of collocated central ops is the wide checkerboard stencil, not the compact 5-point; only the staggered
adjoint pair makes `laplacian(V)=source_sink` = the solver's diffusion term AND the divergence theorem exact.
#### Implementation Spec
Add `laplacian(f, dx, dy, *, boundary_mode, mask)` = compact 5-point via the staggered forward-grad/backward-div,
and an internal `_grad_face`/`_div_face` adjoint pair used by `laplacian` + the region-integral flux.
#### Pseudocode
```
g_x[i+1/2] = (f[i+1]-f[i])/dx        # forward to faces (A8)
lap_x[i]   = (g_x[i+1/2]-g_x[i-1/2])/dx   # backward from faces = (f[i+1]-2f[i]+f[i-1])/dx²
laplacian  = lap_x + lap_y           # compact 5-point EXACT
# no-flux face at boundary/mask: zero the face flux (ghost f_ghost=f_edge ⇒ g=0)
```
#### Test Spec
- `::test_laplacian_is_compact_5pt` — matches `(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1]-4f[i,j])/h²` bit-close.
- `::test_laplacian_not_wide` — differs from collocated `div(grad)` on a checkerboard input (proves the fix).
- `::test_laplacian_analytic_neumann` — `∇²(x²+y²)=4` on a `face_mirror` Neumann box (interior exact; pins the
  no-flux face sign/scale).
- `::test_laplacian_matches_fdm5pt_solver` — **FDM 5-point path ONLY**: build a monodomain sim with the default FDM
  5-point stencil, take a Vm snapshot, and compare `laplacian(Vm)` against the engine's OWN assembled diffusion
  operator applied to that snapshot (extract it via the FDM discretization object, not a hand-rolled stencil), rel
  < 1e-6 interior. **Explicitly skip/xfail** FEM, FVM, and the `moore8`/9-point stencil (a DESIGN open decision) —
  the compact 5-point equals only the FDM-5pt operator; document that source_sink for those paths is "the 5-point
  reconstruction," not bit-identical to their operator.
#### Checklist
- [ ] Staggered pair; no-flux face at boundary/mask; masked→NaN.
- [ ] Solver-match test gated to the FDM-5pt path; extraction uses the discretization's own operator.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/test_fields_derivatives.py -q -k laplacian`
#### Exit Criteria
- [ ] Compact 5-point (not wide); matches the FDM-5pt solver operator; non-5pt paths documented as reconstructions.
#### Risk
Sign/scaling of the no-flux face — the analytic-Neumann test pins it. Extracting the solver's operator: use the FDM
discretization's assembled matrix/stencil, don't reimplement (a reimplementation that matches proves nothing).

### Phase 2 Verification / Exit / Cleanup — as Phase 1; **commit point**.

---

## Phase 3: Vm/φ_e named fields — `r.fields.<name>` (source_sink first)

**Goal**: the `Fields` accessor + the non-LAT named fields (`voltage_gradient`, `voltage_flux`, `source_sink`,
`electric_field`, `current_flux`), lazily cached.
**Tier**: medium

### Phase Context
DESIGN § "Named fields" + § 3 (source_sink = div(D_eff∇V), the SF numerator) + § 6 (bidomain). `Fields` is an
accessor on `SimulationResult` (`r.fields`) computing each named field once and caching it **on the accessor**.
**Cache model (corrected):** `SimulationResult` is an IMMUTABLE post-run snapshot — its Vm/LAT never change, so the
cache is simply valid for the result's lifetime; there is NO `reset()`/`scale_conductance` on the result (those are
methods on the SIM, api.py:321/761 — calling them produces a NEW run → a NEW result → a fresh accessor). Do NOT
invalidate against sim mutations. **Conductivity: use `r.conductivity.D_eff` (= D_raw/(χ·Cm)), NOT raw D** — the
solver steps `∇·(D_eff∇V)` (api.py:1703). `source_sink = div(voltage_flux)`, `voltage_flux = D_eff·grad(Vm)`, via the
**staggered** laplacian path so `div(voltage_flux)==source_sink` and it equals the solver's term. `current_flux=−σ∇φ_e`
(σ from `r.conductivity`), `electric_field=−∇φ_e` (bidomain only → clear error when `phi_e is None`).

### Step 3.1: `Fields` accessor + `VectorField` wrapper
**Model**: opus
#### Read First
- `run.py:20-46` (`SimulationResult` — immutable dataclass), `api.py:321` (`sim.reset`), `api.py:761`
  (`sim.scale_conductance`) — confirm these are SIM methods, not result methods.
#### Why
The accessor holds the lazily-computed fields; because the result is immutable the cache is trivially valid — no
invalidation logic (a common mis-design the audit flagged).
#### Implementation Spec
**File:** `cardiac_core/fields/__init__.py`. `class Fields` holding a ref to the result; per-name lazy memoization on
the accessor instance (dict cache). `VectorField` light wrapper over `(...,2)`: `.x`, `.y`, `.magnitude`, `.angle`,
`.components`. `SimulationResult.fields` = a property returning a **memoized** `Fields(self)` (same accessor each call
so the cache persists), e.g. via `functools.cached_property` on the result.
#### Test Spec
- `test_fields_named.py::test_fields_lazy_cache` — `r.fields.source_sink` twice returns the SAME cached tensor
  (identity); `r.fields is r.fields` (accessor memoized). NO reset test (result is immutable).
- `::test_fresh_run_fresh_cache` — a second `sim.run()` (or after `sim.scale_conductance`) returns a new result whose
  `fields` recomputes (different object) — documents the "mutate the sim, not the result" model.
- `::test_vectorfield_wrapper` — `.x/.y/.magnitude/.angle` correct on a known vector.
#### Verify / Exit / Risk — standard; do NOT add sim-mutation invalidation (the result can't see it).

### Step 3.2: `voltage_gradient`, `voltage_flux`, `source_sink`
**Model**: opus
#### Read First
- DESIGN § 3 (three routes) + § A8 (staggered); `api.py:1703` (D_xx RAW; D_eff=D/(χ·Cm)); Step 1.1's
  `r.conductivity` (D_eff + raw + σ).
#### Why
`source_sink` first — highest-value field (source–sink research) and the SF numerator (DESIGN § 3). MUST use D_eff or
it's off by χ·Cm and fails the solver-match test.
#### Implementation Spec
`voltage_gradient = grad(Vm)`; `voltage_flux = D_eff·grad(Vm)` (D_eff from `r.conductivity`, NOT raw D);
`source_sink = div(voltage_flux)` via the staggered laplacian core. Per-frame `(T,Nx,Ny[,2])`.
- **Isotropy guard (R2):** this scalar `D_eff·∇V` is the isotropic case; if `r.conductivity` carries `D_xy≠0` or
  `D_xx≠D_yy` (anisotropic), `∇·(D∇V)` is a tensor contraction `∂_i(D_ij ∂_j V)`, not `D·∇²V` → **raise/warn**
  (DESIGN § 3 scopes source_sink to isotropic). Do NOT silently mis-compute.
- **Bidomain guard (R2):** `source_sink=∇·(D∇V)` is a MONODOMAIN quantity; on a bidomain result `r.conductivity.D_eff`
  is None → **raise a clear error** ("source_sink is monodomain; for bidomain use current_flux/electric_field") —
  symmetric to the `electric_field`-on-monodomain error (Step 3.3).
- `I_ion` (for the reaction-identity check) = re-evaluate the ionic model on recorded states (`record=("ionic_states",)`).
#### Test Spec
- `::test_div_voltage_flux_is_source_sink` — `div(voltage_flux) == source_sink` to ~1e-10 (pure operator identity).
- `::test_source_sink_matches_fdm5pt_diffusion` — **FDM-5pt path**: equals the engine's diffusion term (extracted
  from the FDM discretization) on a snapshot, rel < 1e-6, using **D_eff** (this test FAILS if raw D is used — the
  guard against the χ·Cm error). Include a **masked/scar** case (a hole with no-flux rim) — the flagship consumers
  (`source_sink_mismatch`, `fig4c` infarct) run on masked geometry, exactly where the staggered laplacian could
  diverge from the engine's no-flux treatment.
- `::test_source_sink_reaction_identity` — **opt-in (needs `record=("ionic_states",)`, FDM path, AND `save_every≈dt`
  or evaluated OFF the upstroke)**: `source_sink ≈ Cm·∂V/∂t + I_ion` (route 3, DESIGN § 3), rel < 1e-2 (coarse
  `save_every` makes `Cm·∂V/∂t` on the upstroke poorly resolved — do NOT claim 1e-3 unless `save_every≈dt`).
  `xfail`/skip when states weren't recorded.
- `::test_source_sink_anisotropic_raises`, `::test_source_sink_bidomain_raises` — the two guards.
#### Verify / Exit / Risk — standard. **Load-bearing: `voltage_flux` uses `D_eff=D_raw/(χ·Cm)`, not raw D; isotropic
+ monodomain only (guards above)** — the FDM-diffusion match (incl. the masked case) is the correctness gate.

### Step 3.3: bidomain `electric_field`, `current_flux`
**Model**: opus
#### Implementation Spec
`electric_field = −grad(phi_e)`, `current_flux = −σ·grad(phi_e)`; raise `NotImplementedError`-style clear error when
`phi_e is None` (monodomain). Document the pre-existing pcg_spectral φ_e caveat (Known Failures).
#### Test Spec
- `::test_efield_monodomain_raises` — monodomain result → informative error.
- `::test_current_flux_bidomain` — on a bidomain run, `−σ∇φ_e` has expected sign/shape.
#### Verify / Exit — standard. **Phase 3 commit point.**

---

## Phase 4: LAT-based fields + divergence gating

**Goal**: `velocity`/`direction`/`speed` (Bayly), `curvature`, `vorticity`, with `|∇T|→0` divergence gating; migrate
`front_metrics` onto these.
**Tier**: large

### Phase Context
DESIGN § 2, § 4, § 5, § 7.1–7.4; PRIOR_ART § 7. Compute ∇LAT via the **Bayly 5×5 local-polynomial (Savitzky–Golay)
kernel** (DESIGN § A7), not raw FD; emit the fit **residual as a `quality` field**. `velocity=∇T/|∇T|²`,
`direction=∇T/|∇T|`, `speed=1/|∇T|`; `curvature=div(n̂)` (Osher–Sethian, DESIGN § A5); `vorticity=curl(velocity)`.
**Divergence gating ON by default:** emit `divergence=div(n̂)` + a `mask` flagging `|∇T|<floor`, high residual, and
`|div n̂|>thresh` (foci/collisions); never return `1/|∇T|` at a collision. All route through the canonical LAT
(Phase 1). Single map → no CV_L/CV_T (identifiability; don't expose). **Ordering note:** the isochrone-based
curvature cross-check needs marching squares → it lives in Phase 5.3, NOT here; Phase 4 validates curvature against
the analytic `CV=CV0−D·κ` fit only.

### Step 4.1: Bayly/SG gradient of LAT + `quality`
- Precompute the SG derivative kernels (DESIGN § A7: `[−2,−1,0,1,2]/(10h)` linear/quad, or the k×k 2-D fit) as conv
  buffers; `∇LAT` = conv; residual map. Route the LAT through the canonical interp LAT (Phase 1).
- Test: planar wave → `speed` matches the known CV (< 2%); curved front → residual localizes curvature; NaN-masked
  nodes propagate NaN (no bleed). Uses the Phase-2 operators + Phase-1 LAT (both already built).
### Step 4.2: `velocity`/`direction`/`speed`/`curvature`/`vorticity` named fields + gating
- Wire as `r.fields.<name>`; `divergence=div(n̂)` gate + `mask`; guard `|∇T|<floor`.
- **`vorticity = curl(velocity)` uses the Step-2.1 `curl` operator** (a LOCAL differential — already built in Phase 2;
  no dependency on the Step-4.4 loop-sum, which is the INTEGRAL/topological-charge primitive for circulation/PS).
- Test: `CV=CV0−D·κ` fit recovers CV0/D on a synthetic curved front (< 5%); collision/focus (`|∇T|→0`) → gated NaN,
  not blow-up; monodomain AND (where applicable) bidomain results. **Deferred to 5.3:** the div-n̂-vs-isochrone-
  curvature cross-check (needs marching squares).
### Step 4.3: `front_metrics`/`fit_eikonal` — keep as-is; wire new fields alongside (migration DEFERRED)
- DESIGN § "Relationship to existing code" says `front_metrics`/`fit_eikonal` **STAY as-is for now**; the torch
  re-expression is documented FUTURE work, NOT this phase (different interior+boundary stencils → cannot agree to
  1e-6; forcing that was an infeasible assertion). So: leave `front_metrics` untouched; add the new
  `r.fields.{speed,direction,curvature}` next to it; assert only **qualitative agreement** (same sign/order-of-
  magnitude on a planar + a curved case) and that `front_metrics`'s own outputs are UNCHANGED (regression). Add a
  docstring note pointing users to the new fields and marking `front_metrics` a compatibility path.
### Step 4.4: ONE shared wrapped-loop-sum primitive (winding number) + refactor `phase_singularities`
- **Consolidate** (DESIGN § 5, PRIOR_ART § 0.4): build a single `_winding_loop_sum` (2×2 plaquette,
  `W(x)=atan2(sin x,cos x)`, CCW+) in `fields/derivatives.py`; **refactor the existing `phase_singularities`
  (analysis.py:356) to call it** (output agreeing to a tight `atol` — the atan2 re-association differs at ULP, so
  regress within tolerance, NOT exact), and reuse it for `circulation`/`winding_number` (Phase 5.2) and Gauss–Bonnet
  (Phase 5.3). (`vorticity`=curl(velocity) uses the Step-2.1 differential curl, not this integral primitive.)
- Test: refactored `phase_singularities` matches the old output on a synthetic rotor **within tolerance** (an atan2
  loop-sum reassociated through the shared primitive can differ at ULP → use a tight `atol`, not exact equality); the
  primitive gives ±1 charge on a known ±2π loop.
**Phase 4 commit point.**

---

## Phase 5: `fields.integrals` + consistency tests

**Goal**: line/region integrals — `conduction_time`, `net_flux`, `circulation`, `winding_number`, isochrone family
(`wavefront_length`, `global_curvature`), with the Stokes/divergence-theorem cross-checks.
**Tier**: large

### Phase Context
DESIGN § "integrals" + § 7 + § "Calculations" B1–B7. Region/boundary = a mask (`over=`/`region=`); isochrone = a LAT
level set via marching squares (DESIGN § B6). **`conduction_time` integrates the SLOWNESS ∇T** (`∫∇T·dl=ΔT`), NOT
`velocity` — hard requirement. **Two distinct flux computations, do not conflate their tolerances:** (i) the
**B5 face-flux telescoping** sum `Σ boundary-face F·n̂` — this is what agrees with `∬div F` to ~1e-12 (the free
EXACT check, uses the staggered face fluxes from Phase 2); (ii) the **B4 polyline** `∮F·n̂ ds` on a marching-squares
contour — geometric, arc-length-weighted, only O(h^1.5) on a staircase mask. `circulation`/`winding` reuse the
Phase-4.4 shared loop-sum. Midpoint region quadrature (B2). The divergence-theorem/Stokes identities are the tests.

### Step 5.1: region integral + net_flux (the EXACT B5 check) + region-load/activated-area/state-fractions
- `region_integral(f, over=mask)` (B2 midpoint, `dA=dx·dy`); `net_flux(F, region=mask)` via **B5 face-flux
  telescoping** and assert `net_flux == region_integral(div F)` to ~1e-12 (EXACT — this is the free check, NOT the
  B4 polyline). Also (dropped-coverage add-back, DESIGN integrals table): `activated_area(t)=∬𝟙[V≥θ]dA`,
  `region_load = ∬ source_sink dA` (= ∬∇²V·D_eff) and `∬ I_ion dA` (opt-in, needs states), `state_fraction`
  (excited/refractory occupancy).
- Test: manufactured `F` → `∬div == Σ boundary face flux` to 1e-12; masked Ω respected; activated-area monotone in t.
### Step 5.2: line integrals — circulation, winding_number, conduction_time
- `circulation(v, loop)`=`∮v·dl` (B4) vs `∬curl v` (Stokes) agree to discretization error; `winding_number` REUSES
  the Phase-4.4 `_winding_loop_sum` (B7); `conduction_time(a,b)` = `T(b)−T(a)` (canonical LAT) with the `∫∇T·dl`
  **slowness** line form as a self-test — assert `∮∇T·dl≈0` (curl-free) on a closed loop, and `≈CL` around a reentry
  loop (the topological signature). **Guard:** conduction_time line-integrates `grad(LAT)`, NEVER the stored
  `velocity` (`∫v·dl≠ΔT`) — a test that feeds `velocity` and confirms it does NOT equal ΔT documents the trap.
### Step 5.3: isochrone family (marching squares) + the deferred curvature cross-check + co-area gate
- `isochrone(level/at_time)` via marching squares (B6, sub-pixel, saddle-handled); `wavefront_length=∮ds`;
  `global_curvature=∮κ ds` (Gauss–Bonnet ≈2π for a convex closed isochrone — the test); **isochrone-spacing CV ≡
  1/|∇T|** regression test (the geometric-vs-differential agreement); the **div-n̂ vs isochrone-curvature** cross-
  check deferred from Phase 4.2; and the **co-area identity** `dA/dt = ∮_{T=t}(1/|∇T|)ds = L(t)·⟨CV⟩` (DESIGN § 7 /
  PRIOR_ART § 7.5) as a whole-field consistency gate (independently compute the three, assert agreement).
**Phase 5 commit point.**

---

## Phase 6: Scalar EP metrics

**Goal**: `wavelength`, consolidated `apd`, `di`, protocol-based `erp`; fix the `apd_at` multi-beat baseline.
**Tier**: medium

### Phase Context
DESIGN § 8; PRIOR_ART § 5. `wavelength(cv, refractory, kind="erp")` → cm with `÷1000` unit fix; `kind="apd"` a
**warned** proxy; `cv_scope="local"|"global"`. **The `V_rest=trace[0]` baseline bug is in THREE functions** —
`apd_at`(analysis.py:110), `restitution_curve`(458), AND `apd_per_beat`(738); all three must switch to the per-beat
pre-upstroke diastolic V. Expose `activation="threshold"|"dvdt_max"`. `di(bcl,apd)` = both algebraic + measured.
**`erp` is a PROTOCOL** (S1S2 extrastimulus + capture-detection bisection) that RUNS sims → it lives in the NEW
`cardiac_core/protocols.py`, NOT `analysis.py` (analysis is imported by run/api; putting a `simulate` call there is a
circular-import hazard — api.py:1355 already notes an api↔run workaround). Also `erp_proxy=apd90` +
`post_repol_refractoriness=erp−apd90`.

### Step 6.1: `wavelength` + `di` + the three-function apd baseline fix
- `wavelength` in analysis.py; `di`; fix the baseline in `apd_at` AND `restitution_curve` AND `apd_per_beat`.
- Test: `λ=CV·ERP` units (cm/s·ms/1000=cm); `kind="apd"` warns; multi-beat APD baseline correct in ALL THREE
  functions (a 3-beat drifting-baseline trace → each beat's APD uses its own diastolic V, not `trace[0]`); existing
  single-beat `test_analysis.py` APD values UNCHANGED (regression — the fix reduces to the old value for beat 0).
### Step 6.2: protocol-based `erp` (new `protocols.py`)
- `cardiac_core/protocols.py::erp(...)` — S1S2 + bisection capture detection; imports `simulate` at call time
  (lazy, to avoid the circular import); clearly documented as "runs simulations," separate from the trace-analysis
  metrics. `erp_proxy`/`post_repol_refractoriness` helpers.
- Test: paced 1-D cable → ERP ≈ APD90 in healthy tissue; a reduced-excitability case → ERP > APD90 (PRR). Skip-mark
  if it's too slow for CI (gate behind a `slow` marker) — but keep a small-grid smoke case in the default suite.
**Phase 6 commit point.**

---

## Phase 7: `single_cell` 0-D + `safety_factor`

**Goal**: 0-D mode via the shared ionic step; Boyle–Vigmond safety factor on top of `source_sink`.
**Tier**: large

### Phase Context
DESIGN § 9 + § 3 (SF); PRIOR_ART § 6 (single-cell) + § 3 (SF_VB). **Route `single_cell` through the SAME per-node
ionic step the tissue reaction substep uses** (diffusion omitted) — closes the 0-D gap AND sidesteps the ORd
concentration-ordering bug. The shared step is the `cardiac_core.ionic` model interface (`ionic/base.py` ABC:
`get_initial_state`, `step(V, states, dt)`, `compute_Iion(V, states)`, `compute_gate_steady_states` /
`compute_gate_time_constants` for Rush–Larsen; `run()` is an existing 0-D driver looping `step`) — the SAME object the
tissue reaction substep calls; do NOT re-implement the ODE. FE+RL default; per-model dt/stimulus/pre-pace table (DESIGN § 9).
`safety_factor` = Boyle–Vigmond `[Cm·ΔV+Q_ion−Q_s]/Q_thr(t_A)` — the numerator is `∫_A source_sink dt` (from Phase 3;
Cm from `r.Cm`, Q_ion from I_ion = model re-eval on recorded states), `Q_thr(t_A)` from a one-time single-cell
calibration (which `single_cell` produces). **Shaw–Rudy SF is explicitly DEFERRED** (1-D-fiber-only historical
secondary; documented future work, NOT a step in this plan — Boyle–Vigmond is the shipped default).

### Step 7.1: `single_cell()` driver + `pre_pace`
#### Read First
- `cardiac_core/ionic/base.py` — IonicModel ABC: `get_initial_state`(:84), **`step(V, states, dt, ...)`(:101) = the
  shared per-node RL step the tissue reaction substep uses**, `compute_Iion`(:127), `compute_gate_steady_states`(:146)
  / `compute_gate_time_constants`(:166) [NOT `gate_inf`/`gate_tau`], and **`run(t_end, dt, ...)`(:207) — an EXISTING
  single-cell driver that already loops `self.step(...)`**. `cardiac_core/ionic/registry.py::build_ionic_model`(:24).
#### Implementation Spec
`cardiac_core/single_cell.py::single_cell(model="ttp06", *, dt=None, stimulus=..., n_beats=1, pre_pace=0, celltype=…)`
— **build on `IonicModel.run()`/`step()` (do NOT re-implement the ODE loop)**, adding the stimulus protocol, pre-pace,
and per-model defaults. Stimulus injected into dV/dt ONLY (never a concentration balance); FE+RL; per-model default
dt/stimulus/pre-pace (DESIGN § 9 table). Returns the V trace + final state (for the tissue init and the SF `Q_thr`
calibration). If `IonicModel.run()` already covers the loop, `single_cell` is a thin protocol/pre-pace wrapper over it.
#### Test Spec
- `test_single_cell.py::test_ttp06_ap` — physiological APD90 (~270–300 ms at BCL 1000, EPI); stable rest.
- `::test_cm_scaling` — Cm=2 halves the reaction dV (matches the V5.5 reaction-`/Cm` property).
- `::test_0d_vs_tissue_singlenode` — a 1×1 (or fully-decoupled) tissue node vs `single_cell` at matched dt agree to
  tolerance (the ORd concentration-ordering-bug guard — the whole point of sharing the step).
- `::test_prepace_steady_state` — pre_pace reduces beat-to-beat APD drift below a threshold.
### Step 7.2: `safety_factor` (Boyle–Vigmond) + `Q_thr` calibration
#### Implementation Spec
`Q_thr(t_A)` calibration curve from `single_cell` (min charge to trigger an AP vs pulse duration, ~linear — fit once
per model); `safety_factor` per node = `[Cm·ΔV + Q_ion − Q_s]/Q_thr(t_A)` over the activation interval A=[t_1%, t_Im0],
numerator ≡ `∫_A source_sink dt` (Phase 3 field); Cm from `r.Cm`; Q_ion from I_ion (model re-eval on recorded states →
needs `record=("ionic_states",)`). SF<1 ⇒ block.
#### Test Spec
- `::test_sf_propagating_cable` — a healthy cable → SF > 1 everywhere the wave passes.
- `::test_sf_block_site` — a source–sink-mismatch geometry (expansion / low-excitability) → SF < 1 at the block site.
- `::test_sf_needs_states` — clear error / skip when ionic_states weren't recorded.
**Phase 7 commit point.**

---

## Final Cleanup
- float64 everywhere (no float32 leak); torch-only field/LAT paths (no numpy in hot paths); V5.3 untouched; no
  cross-engine duplication (shared logic in `cardiac_core/`); every new field tested on EACH engine it claims.
- Update `cardiac_core/API_CHEATSHEET.md` with the new `r.fields.*` surface + canonical LAT note + `wavelength`/`erp`/
  `single_cell`/`safety_factor`; re-run the cheatsheet canary.
- Update `ANALYSIS_FIELDS_DESIGN.md` "Open decisions" → resolved where implemented; `README.md` completion criteria.
- Archive this plan: `mkdir -p Research/Active/engine_consolidation/plans && cp cardiac_core/ANALYSIS_FIELDS_PLAN.md
  "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_analysis-fields-branch.md"`.
- Confirm per-engine integrity goldens bit-identical (atol=0) — this is an analysis layer, solvers untouched.

## Mutation Log
- **AUDIT 2026-07-22 — Phase 1 (Opus adversarial, 0 critical / 0 high / 3 med / 4 low).** Verdict:
  threading unified through one `build_result_context`, `D_eff=D_raw/(χ·Cm)` correct-by-construction,
  interp/−40 flip well-contained; no reachable crash or silent-wrong. Actioned:
  - **MED-1 FIXED** — `r.cell_type` returned `group_cell_types[0]` even for bidomain/LBM, but those
    factories force ENDO (no `cell_type` to `build_ionic_model`) → a Phase-7 `I_ion` re-eval would
    rebuild the WRONG cell type. `_cell_type_of(data, engine_type)` now forces ENDO for bidomain/LBM,
    threads `group_cell_types[0]` only for monodomain. Guard: `test_bidomain_cell_type_is_endo_not_mesh_group`.
  - **MED-3 + LOW-2 FIXED** — the CV-family default flip (interp/−40) had NO regression guard (step-wave
    synthetics give the same value under nearest/−20 and interp/−40). Added `TestCVFamilyCanonical` (a
    SMOOTH two-slope trace where the two differ: interp→62.5, nearest→50 cm/s) pinning that
    `conduction_velocity`/`cv_between`/`radial_cv` DEFAULT to interp/−40 AND the `method="nearest",
    threshold=-20` override reproduces the historical value.
  - **MED-2 → routed to Phase 3** — legacy `create_cardiac_mesh→bidomain` leaves `sigma_i/e=None`, so
    `_conductivity_from` takes the mono branch (`D_eff` set, `is_bidomain=False`) on a real bidomain run.
    Fix belongs in Phase 3: key the `source_sink` bidomain-guard on **`r.phi_e is not None`** (robust),
    NOT `conductivity.is_bidomain`. Recorded here so Phase 3 implements it.
  - **LOW-1/3/4 deferred** (minor/pre-existing/harmless): 2-point CV builds the full LAT map + batch
    path re-converts conductivity per chunk (perf, not correctness); empty-run device asymmetry
    (api CPU vs run output_device — no data); `times`/`V` cross-device not guarded (builders co-locate).
- **REGOLDEN 2026-07-22 — Step 1.3 (default LAT flip to interp/−40): analysis CV/LAT VALUE
  assertions did NOT move; no golden edits were needed.** Why: the existing CV tests
  (`test_analysis.py`, `test_usability_fixes.py`, `test_run_contract.py`,
  `test_construction_api.py`) are either loose bands or LAT-*difference* metrics, and every
  synthetic trace is a uniform rest→plateau STEP wave — so the interp-vs-nearest shift is a
  global constant that cancels in every CV (a LAT difference). Only the ABSOLUTE `activation_time`
  values shifted (~0.4–0.6 ms earlier), which the `abs=1.0`/`rel≥0.15`/std/NaN assertions
  absorb. Semantics changed (interp/−40 canonical, nearest/−20 override) but no numeric golden
  required editing. **Solver-integrity goldens (`test_integrity.py`) confirmed bit-identical
  (atol=0)** — solvers untouched (pure analysis layer). New guarding tests: `test_canonical_lat.py`
  (17 tests: context on both build paths + interp LAT + max_dvdt).
- **INSERTED 2026-07-22 — `cardiac_core/_result_context.py`** (`Conductivity` holder +
  `build_result_context`): a NEW shared module (not in the original Architecture-Changes list),
  so the `.run()` and `simulate()` builders thread the analysis context through ONE code path
  instead of duplicating it. Also added `max_dvdt_time` to the top-level lazy exports.

### Phases 2–7 implementation notes (all test-gated; full suite green)
- **Phase 2 (operators).** Collocated `grad`/`div`/`curl` use a WHOLE-sample mirror boundary
  (normal derivative = 0 at a no-flux edge; `curl(grad)=0` interior); the staggered `laplacian`
  matches the engine's OWN assembled FDM-5pt operator to rel<1e-6 (`test_laplacian_matches_fdm5pt_solver`)
  and the divergence theorem is exact to 1e-10. `winding_loop_sum` added here (Phase-4 primitive).
- **Phase 3 (Vm/φ_e fields).** `source_sink` uses a NEW `diffusion_term` operator: the conservative
  ∇·(D∇V) with **HARMONIC face-averaged D** (matching the cardinal-4 solver) — so it equals
  `apply_diffusion` on BOTH uniform D and a **masked scar** (the flagship-consumer case). MED-2 fix
  from the audit: the bidomain guard keys on **`r.phi_e is not None`** (robust for legacy D-based
  bidomain), not `conductivity.is_bidomain`. `div(voltage_flux)==source_sink` is O(h²)-consistent
  (collocated-vs-staggered), NOT 1e-10 — the exact identity is the FDM-operator match test.
- **Phase 4 (LAT fields).** `bayly_gradient` = fixed 2-D Savitzky–Golay conv (`K_x = g_x⊗s_y`);
  quality = the SG smoothing residual; divergence gating (`|∇T|<floor`, window-touches-a-block,
  no-bleed) → `velocity`/`speed`/`direction` NaN at collisions. `phase_singularities` refactored
  onto `winding_loop_sum` (matches the old modulo-wrap output to atol). Step 4.3: `front_metrics`
  left untouched (no code change needed).
- **Phase 5 (integrals).** `net_flux`/`circulation` = `region_integral` of the staggered
  `_divergence_flux` / collocated `curl` (divergence theorem exact to 1e-9; Stokes to 1e-9);
  isochrones via `skimage.measure.find_contours` (marching squares, B6) → `wavefront_length`,
  `global_curvature` (Gauss–Bonnet ≈2π). The co-area identity is validated implicitly via the
  speed/isochrone consistency (not a separate test).
- **Phase 6 (scalar EP).** The `apd` diastolic-baseline fix uses an **upstroke-foot walk**
  (`_upstroke_foot_V`), NOT a min-over-interval — the latter wrongly catches the PREVIOUS beat's
  lower repolarization undershoot on a drifting baseline. Reduces to `trace[0]` for a clean single
  beat (regression-safe; existing APD tests unchanged). `erp` (new `protocols.py`) builds the S1
  train as explicit per-beat stimulus dicts + one S2 (not `bcl`/`num_pulses`); smoke test uses the
  fast `phas13` model.
- **Phase 7 (single_cell + SF).** `single_cell` drives the shared `IonicModel.step` (0d-vs-tissue
  node match confirmed — the ORd-ordering-bug guard); `Cm` rescales the per-step voltage update.
  `safety_factor` numerator = **∫ of the INWARD (positive) `source_sink` over the activation window**
  — because `∇·(D∇V) = Cm·dV/dt + I_ion` (the PDE), this IS the Boyle–Vigmond `Cm·ΔV+Q_ion`, so NO
  separately-recorded ionic states are needed (a simplification vs the plan's route-3). `Q_thr` from
  a `single_cell` `threshold_charge` bisection. Shaw–Rudy SF deferred (as planned).
