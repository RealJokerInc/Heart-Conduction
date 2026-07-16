# PLAN: cardiac_core usability fixes — P0 bugs + P1 stub-trap & cheatsheet

Created: 2026-07-16
Engine(s): cardiac_core (all three vendored engines)
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — "API usability audit ROUND 2" → the MERGED P0–P4 fix list in [API_USABILITY_AUDIT_2026-07-15.md](API_USABILITY_AUDIT_2026-07-15.md)

## Objective
Fix the correctness/crash bugs (P0) and the stub-method hallucination trap + cheatsheet gaps (P1) that the
two-round usability audit surfaced, so a scientist's first real run is correct and doesn't crash or silently
lie. Every change is test-gated and must keep the per-engine bit-identical integrity goldens exact.

## Success Criteria
- [ ] All audited bugs in scope (B1–B8, plus B2 wiring) fixed with a regression test each
- [ ] The NotImplementedError stub methods no longer advertise a working API (honest errors + de-worked docstrings); `scale_conductance`/`set_conductivity` implemented
- [ ] `API_CHEATSHEET.md` corrected (solver/`dt` guidance; drug-block route; `paci`; ORd-LBM-only; `record=`/`save_result`/`dominant_frequency`/`phase_map`; per-node-D; `fiber_angle` radians; bath cost; LBM offset)
- [ ] Full suite green (currently **218 passed / 2 xfailed**) with the new tests added
- [ ] Per-engine integrity goldens **bit-identical** (atol=0) at every phase

## Architecture Changes
- MOD: `cardiac_core/api.py:993` (`_result_from`) — set `times` device from `Vm` (B1)
- MOD: `cardiac_core/mesh/structured.py:68` + `:148` (`__post_init__`/`from_mask`) — guard `dy` at `Ny==1` (B5)
- MOD: `cardiac_core/analysis.py` (`apd_at` ~106–159) — window peak to current beat + last-crossing option (B3/B4)
- MOD: `cardiac_core/api.py` run entry (`_iter_snapshots`/`run` ~194/219, `_run_monodomain` ~764) — validate `record=` keys (B7)
- MOD: `cardiac_core/_monodomain/.../explicit/forward_euler.py` — `dt` stability guard (B6)
- MOD: `cardiac_core/_monodomain/.../classical/monodomain.py:127–137` — thread grid params into DCT/FFT construction (B2)
- MOD: `cardiac_core/mesh/structured.py:178` (`StructuredGrid.flat_to_grid`) — NaN-fill masked-out nodes; covers monodomain + bidomain output (Vm + phi_e); LBM has no `flat_to_grid` and is untouched (B8)
- MOD: `cardiac_core/api.py:293–560` — de-trap stubs; implement `scale_conductance`/`set_conductivity`
- MOD: `cardiac_core/API_CHEATSHEET.md` — correctness + solver/`dt` pass
- NEW: `cardiac_core/tests/test_usability_fixes.py` — regression tests for every fix

## Known Failures / Constraints (from IDEALOG + prior decisions)
- **Do NOT change the default solver/`dt`/lattice** — goldens depend on the default `crank_nicolson`+`pcg` path (mono) and d2q5/neumann (LBM, prior 2026-07-15 decision). Fixes must be additive/opt-in or touch only masked/error paths.
- **Do NOT delete stub method *signatures*** — downstream may reference the names; replace the body/docstring only.
- **Goldens are full-rectangle** (no mask) → B8's NaN-fill and B5's Ny=1 guard cannot touch them; verify anyway.
- **Big-bang deletion of engine-local ionic copies is out of scope** (breaks Surrogate/Optimizer — see prior thread).

---

## Phase 1: P0 crash & silent-wrong bugs (cheap, high-value)

**Goal**: Fix the six bugs that crash or silently corrupt results, each independently. Highest ROI in the plan.
**Tier**: medium
**Estimated scope**: ~6 small edits + ~6 focused tests in `cardiac_core/tests/test_usability_fixes.py`.

### Phase Context
All work is in `cardiac_core/`. Run tests with `/home/norepinephrine/.conda/envs/heart-conduction/bin/python -m pytest`.
Float64 everywhere. After each step run the targeted test AND the goldens: `tests/test_integrity.py`
(`test_{monodomain,bidomain,lbm}_matches_golden` + `test_originals_untouched`, all full-rectangle) + `tests/test_self_contained.py`
— confirm bit-identical (atol=0). Do not modify `Monodomain/Engine_V5.3/` (frozen). New tests go
in `cardiac_core/tests/test_usability_fixes.py` (create it) unless an existing file is the obvious home.

### Step 1.1: B1 — GPU device-mismatch in `_result_from`
**Model**: opus
#### Read First
- `cardiac_core/api.py:980–996` (`_result_from`) — the non-empty branch builds `times` with no `device=`; `Vm = torch.stack([s.Vm …])` inherits the snapshot device.
- `cardiac_core/run.py:91,99` (`_collect`) — already sets `device=dev`; the correct pattern.
- `cardiac_core/analysis.py:50` — `times[first_idx]` is where the mismatch raises (a CUDA `first_idx` indexing a CPU `times`).
#### Why
On `device="cuda"` the declarative `.run()` path returns `Vm` on CUDA but `times` on CPU, so *every* analysis/viz call crashes. One-line device fix unbreaks the entire GPU analysis surface. `_collect` (the `run_*`/`simulate` path) is already correct — only `_result_from` (the `CardiacSimulation.run()` path) is wrong.
#### Implementation Spec
**Modify** `api.py:993` → `times = torch.tensor([s.t for s in snaps], dtype=torch.float64, device=snaps[0].Vm.device)`. Keep the empty branch's `empty_t`/`empty_v` on the same device as each other (CPU is fine for empty).
#### Pseudocode
```
times = torch.tensor([s.t for s in snaps], dtype=torch.float64, device=snaps[0].Vm.device)
```
#### Test Spec
- `test_usability_fixes.py::test_result_times_device_matches_vm` — CPU `CardiacSimulation.run()`, assert `r.times.device == r.Vm.device`. Add `@pytest.mark.skipif(not torch.cuda.is_available())` variant on cuda asserting `.lat()`/`.cv()` don't raise.
#### Checklist
- [ ] Add `device=` to the `times` construction
- [ ] CPU test asserts device equality
- [ ] cuda-guarded test asserts analysis hooks run
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k device -q`
#### Exit Criteria
- [ ] `times`/`Vm` share a device; analysis works on cuda (or skips cleanly)
#### Risk
Empty-result device asymmetry — mitigation: keep empty times+Vm both CPU and assert it.

### Step 1.2: B8 — NaN-fill out-of-domain nodes
**Model**: opus
#### Read First
- `cardiac_core/mesh/structured.py:178` (`StructuredGrid.flat_to_grid`) — the ONE reconstruction site: when `domain_mask` is set it does `grid = torch.zeros(Nx,Ny); grid[domain_mask] = flat` (the 0.0 fill). Its docstring already promises "fills masked-out locations with fill_value" (never parameterized).
- Consumers (all go through `flat_to_grid`, so one fix covers them): `api.py:767` (monodomain Vm), `api.py:782–783` (bidomain Vm + phi_e), `api.py:762` (ionic states), `api.py:269/280` (`.state`/`.phi_e`).
- **LBM does NOT use `flat_to_grid`** — it works on the full `(Nx,Ny)` grid with bounce masks. `cardiac_core/tests/test_api_hardening.py:41` (`test_lbm_masked_hole_nonconducting`) asserts `torch.isfinite(res.Vm[-1]).all()`, so LBM masked nodes MUST stay finite — B8 must not touch LBM.
- `cardiac_core/tests/test_integrity.py` goldens (`test_{monodomain,bidomain,lbm}_matches_golden`) are **full-rectangle (no `domain_mask`)** → `flat_to_grid` takes the `reshape` branch → NaN-fill cannot touch them.
- `cardiac_core/analysis.py:41,44` — `above = V >= threshold` treats `NaN` as `False` (NaN → not activated → NaN lat), so NaN-filling is analysis-safe.
#### Why
Masked (out-of-domain) nodes come back as `0.0 mV` (> −20 mV threshold), so `lat/apd/cv` count dead tissue as "activated at t=0" — a **23% silent CV error** and 100% false-activation on every scar/fibrosis/irregular run (mono + bidomain). Filling those nodes with `NaN` makes all analysis auto-correct.
#### Implementation Spec
**Modify** `StructuredGrid.flat_to_grid` (`structured.py:178`): add a `fill_value=float('nan')` parameter and build `grid = torch.full((Nx,Ny), fill_value, device=…, dtype=flat.dtype)` before `grid[domain_mask] = flat` (was `torch.zeros`). One change fixes monodomain + bidomain output uniformly. Do NOT touch the LBM path. `grid_to_flat` (the inverse) already extracts `domain_mask` only, so NaN never re-enters computation. Confirm `save_result`/viz tolerate NaN.
#### Pseudocode
```
def flat_to_grid(self, flat, fill_value=float('nan')):
    if self.domain_mask is not None:
        grid = torch.full((self.Nx,self.Ny), fill_value, device=self._device, dtype=flat.dtype)
        grid[self.domain_mask] = flat; return grid
    return flat.reshape(self.Nx, self.Ny)
```
#### Test Spec
- `::test_masked_nodes_are_nan_mono` — masked-hole monodomain run: `torch.isnan(r.Vm[:, ~mask]).all()`, `r.lat()` NaN at masked nodes, full-rect run has **no** NaN.
- `::test_masked_nodes_are_nan_bidomain` — masked-hole bidomain run: masked Vm AND `phi_e` are NaN at masked nodes.
- Golden guard: `test_integrity.py` (all 4) + `test_self_contained.py` bit-identical.
- **LBM regression guard:** `test_api_hardening.py::test_lbm_masked_hole_nonconducting` still passes (LBM masked Vm stays finite).
#### Checklist
- [ ] `flat_to_grid` NaN-fills (via `fill_value` default); LBM untouched
- [ ] Mono + bidomain masked tests pass; full-rect goldens exact; LBM `isfinite` test green
- [ ] `save_result`/`propagation_video` smoke on a NaN-containing result
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k masked cardiac_core/tests/test_integrity.py cardiac_core/tests/test_api_hardening.py::test_lbm_masked_hole_nonconducting -q`
#### Exit Criteria
- [ ] Mono+bidomain masked nodes NaN; LBM finite; goldens exact
#### Risk
NaN-filling LBM would break `test_lbm_masked_hole_nonconducting` — mitigation: LBM has no `flat_to_grid`; the fix is structurally mono/bidomain-only. A test elsewhere asserting masked Vm==0 — mitigation: grep `tests/` for masked `== 0` (none found on Vm in the audit); NaN is the correct contract. (LBM masked-node pollution, if any, is separate future work.)

### Step 1.3: B3/B4 — `apd_at` peak-over-remaining + notch
**Model**: opus
#### Read First
- `cardiac_core/analysis.py:106–159` (`apd_at`) — `V_peak = trace[act_idx:].max()` (max over ENTIRE remaining trace = B3) and the first-crossing-after-peak repol search (spike-and-dome notch for low repol = B4).
#### Why
A later taller beat corrupts earlier beats' APD (got 1341 ms); APD30 lands on the TTP06 notch (7.4 ms). Both silently wrong — they break every restitution/alternans/morphology measurement.
#### Implementation Spec
**Modify `apd_at`:** (a) bound the peak search to the current beat (max between `act_idx` and the next upstroke / one BCL, not the whole tail); (b) repolarization uses the **last** crossing of `V_repol` within the beat (dome-aware), default `dome_aware=True`, keep a `first_crossing` fallback.
#### Pseudocode
```
end = next_upstroke_after(act_idx) or len(trace); V_peak = trace[act_idx:end].max()
V_repol = V_peak - repol*(V_peak - V_rest)
# dome-aware: last index in (peak_idx,end) where trace crosses below V_repol
```
#### Test Spec
- `::test_apd_multibeat_not_corrupted` — 2-beat trace, beat 2 taller; beat-1 APD from beat-1 peak.
- `::test_apd_notch_dome_aware` — spike-and-dome; APD30 = dome repol (~150+ ms) not the notch (~7 ms).
#### Checklist
- [ ] Peak bounded to current beat
- [ ] Dome-aware last-crossing; `first_crossing` fallback kept
- [ ] Both tests pass; `test_analysis.py` green
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k apd cardiac_core/tests/test_analysis.py -q`
#### Exit Criteria
- [ ] Multi-beat + dome APD correct; single-clean-AP unchanged
#### Risk
Default repol semantics could shift existing APD test values — mitigation: `test_analysis.py`'s synthetic APs are **monotonic/dome-free** (`test_apd_at_known` = plateau + slow single-slope repol, asserted `40 < apd < 70`; `restitution_curve` test = linear repol, asserts only `len(APD)>=1`/`len(DI)==len(APD)`), so dome-aware last-crossing == first-crossing and beat-windowing doesn't change them — the change is safe against the existing suite. Still re-run `test_analysis.py`; if any value shifts, confirm it's the physically-correct one and update with a comment.

### Step 1.4: B5 — `Grid(N,1)` ZeroDivisionError
**Model**: sonnet
#### Read First
- `cardiac_core/mesh/structured.py:62–70` (`__post_init__`, `self.dy = self.Ly/(self.Ny-1)`) and `:148` (`from_mask`).
#### Why
`cc.Grid(N,1,dx)` crashes at engine construction; guarding `Ny==1` lets 1-D cables build.
#### Implementation Spec
**Modify:** `self.dy = self.Ly/(self.Ny-1) if self.Ny > 1 else self.dx` (and symmetric `dx`/`Nx==1` guard if present). Both `__post_init__` and `from_mask`.
#### Test Spec
- `::test_grid_1d_cable` — `cc.Grid(101,1,0.02)` constructs; `cc.monodomain(Grid(101,1,0.02),"ttp06",cond,stim).run(...)` runs; `r.Vm.shape == (T,101,1)`.
#### Checklist
- [ ] Guard dy (+dx); both paths
- [ ] 1-D run + analysis works
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k 1d cardiac_core/tests/test_mesh_shared.py -q`
#### Exit Criteria
- [ ] Ny=1 constructs and runs; Ny≥2 unchanged
#### Risk
Low. Mitigation: assert Ny≥2 goldens unchanged.

### Step 1.5: B7 — validate `record=` keys
**Model**: sonnet
#### Read First
- `cardiac_core/api.py:170` (`run(record=…)`), `:219` (`_iter_snapshots`), `:764` (`want_ionic = "ionic_states" in record`).
#### Why
`record=("Vm","I_Kr")` runs and silently produces nothing. Validate so a typo/unsupported key raises.
#### Implementation Spec
**Modify** the `run()`/`_iter_snapshots` entry to validate `record` against `{"Vm","ionic_states"}` and `raise ValueError(f"unknown record key(s): …; supported: Vm, ionic_states")`.
#### Test Spec
- `::test_record_rejects_unknown` — `run(record=("Vm","I_Kr"))` → `ValueError`; `("Vm","ionic_states")` and `("Vm",)` work.
#### Checklist
- [ ] Validate at the single entry point
- [ ] Known keys pass; unknown raise
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k record -q`
#### Exit Criteria
- [ ] Unknown record keys raise; existing runs unaffected
#### Risk
An internal caller passes an unlisted key — mitigation: grep `record=` usages first.

### Step 1.6: B6 — forward_euler `dt` stability guard
**Model**: opus
#### Read First
- `cardiac_core/_monodomain/simulation/classical/solver/diffusion_time_stepping/explicit/forward_euler.py` — its own docstring gives the CFL limit: **`dt ≤ Cm·h²/(4·D_max)`** (NOT `dx²/4D` — the `Cm` factor matters). Needs `dx`, `D_max`, `Cm` from the `spatial`/discretization it holds.
#### Why
`forward_euler` at `dt > Cm·dx²/(4·D_max)` silently produces oscillating garbage (threshold crossed 3693×). A guard turns silent-wrong into a clear signal — and it's the solver scientists are pushed toward for speed.
#### Implementation Spec
**Modify:** on construction/first step, compute `dt_max = Cm · min(dx,dy)² / (4 · D_max)` from the spatial discretization's `dx/dy`, effective `D_max`, and `Cm`, and `warnings.warn(...)` (quoting the stable `dt_max`) if `dt > dt_max`. Default: warn (don't hard-raise — opt-in speed users). If `D_max`/`Cm` aren't directly on the solver, pull them from `self.spatial` (it exposes the diffusion operators / grid).
#### Test Spec
- `::test_forward_euler_stability_warns` — `diffusion_solver='forward_euler'`, `dt` above `dx²/4D` → `UserWarning` naming the limit; below → no warning, runs.
#### Checklist
- [ ] Compute `dt_max` from grid + D
- [ ] Warn (with the numeric limit) when exceeded
- [ ] Test both sides
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k euler -q`
#### Exit Criteria
- [ ] Over-limit forward_euler warns; CN/PCG default untouched (goldens exact)
#### Risk
A test uses over-limit forward_euler and now warns — mitigation: `pytest.warns`/filter; do not change CN default.

### Phase 1 Verification
`…/python -m pytest cardiac_core/tests/ -q` → **224 passed / 2 xfailed** (218 baseline + 6 new).
### Phase 1 Exit Criteria
- [ ] All 6 bug tests pass; full suite green; integrity goldens bit-identical (atol=0)
### Phase 1 Cleanup
- float64 consistency (no float32 leaks); V5.3 untouched; no cross-engine duplication; remove scratch prints.
**-> Commit point: `fix(cardiac_core): P0 usability bugs B1/B3/B4/B5/B6/B7/B8`**

---

## Phase 2: B2 — repair the fast spectral solver path

**Goal**: Make `linear_solver='dct'`/`'fft'` work through the factory (they currently `TypeError`), restoring the fast path that's the root of the runtime wall. Opt-in only — the default `pcg` is untouched.
**Tier**: medium
**Estimated scope**: thread grid params through one call site + a construction/agreement test.

### Phase Context
`_build_linear_solver` (`_monodomain/.../classical/monodomain.py:127–137`) does `return FFTSolver(**kwargs)` / `DCTSolver(**kwargs)`, but `FFTSolver.__init__` requires `nx,ny,dx,dy,dt,D[,chi,Cm,scheme]` and the caller doesn't pass them (N3's `TypeError`). Find where `_build_linear_solver` is called from the diffusion-solver construction and thread the grid/timestep/D through.

### Step 2.1: Thread grid params into DCT/FFT construction
**Model**: opus
#### Read First
- `_monodomain/.../classical/monodomain.py:127–137` (`_build_linear_solver`, the factory) and the **call site `monodomain.py:338`**: `linear = _build_linear_solver(linear_solver, tol=pcg_tol, max_iters=pcg_max_iter)` — it passes NO grid params, so `FFTSolver(**kwargs)`/`DCTSolver(**kwargs)` get empty kwargs → `TypeError`.
- DCT/FFT `__init__` in `.../linear_solver/fft.py:245+` — required args `nx,ny,dx,dy,dt,D,chi,Cm,scheme`. **Verify all of these are in scope at `monodomain.py:338`** (the enclosing diffusion-solver builder has the grid + `dt`; confirm effective `D` and `scheme` are reachable there — if not, thread them in from the caller).
#### Why
The fast O(N log N) spectral solve is dead through the public factory → every run falls back to slow PCG. Fixing the wiring is the single biggest performance lever and is golden-safe (opt-in; default `pcg` unchanged).
#### Implementation Spec
**Modify** the `monodomain.py:338` call (and `_build_linear_solver`'s signature) to pass `nx,ny,dx,dy,dt,D,chi,Cm,scheme` through to DCT/FFT. DCT ↔ Neumann (cardiac default BC), FFT ↔ periodic — keep DCT recommended. If `D`/`scheme` aren't in scope at 338, thread them from the enclosing builder's config.
#### Test Spec
- `::test_dct_solver_runs_and_matches_pcg` — small Neumann isotropic strip with `linear_solver='dct'` completes and its CV matches the `pcg` run within tolerance. `fft` on a periodic config constructs+runs.
#### Checklist
- [ ] Grid params reach DCT/FFT ctor
- [ ] `dct` runs; CV ≈ pcg within tol
- [ ] Default `pcg` path + goldens unchanged
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k solver cardiac_core/tests/test_self_contained.py -q`
#### Exit Criteria
- [ ] `dct`/`fft` selectable and correct; default untouched
#### Risk
DCT/FFT operator may not match CN bitwise (BC handling) — mitigation: CV-agreement tolerance, not bitwise; if fft needs periodic BC, restrict its test to a periodic mesh and document the requirement.

### Phase 2 Verification / Exit / Cleanup
`…/python -m pytest cardiac_core/tests/ -q` green; goldens exact.
**-> Commit point: `fix(cardiac_core): B2 — wire grid params into DCT/FFT linear solvers (fast path)`**

---

## Phase 3: P1 — de-trap the stub methods + implement the two highest-value ones

**Goal**: Stop the object advertising a large NotImplementedError surface as if it works; implement `scale_conductance` + `set_conductivity` (engine support exists).
**Tier**: large (Step 3.2 gets an `/audit`)
**Estimated scope**: docstring/error cleanup across ~14 stubs + 2 real implementations via the existing sim-rebuild machinery.

### Phase Context
Stubs live in `api.py:293–560` (`get_state`, `state_names`, `set_voltage`, `set_state`, `add_pacing`, `inject_current`, `clamp_voltage`, `add_clamp_protocol`, `release_clamp`, `scale_conductance`, `set_conductivity`, `scale_conductivity`, `set_parameter`, …). The class already has replay machinery (`with_()`/`reset()`/`stimulate()` rebuild the factory with `mesh=self._data` + `_build_kwargs`) — `scale_conductance`/`set_conductivity` reuse it.

### Step 3.1: Make every stub honest
**Model**: opus
#### Read First
- `api.py:131–135` (class note about planned methods) and each stub body (`raise NotImplementedError`).
#### Why
The stubs ship worked-example docstrings (`>>> sim.set_conductivity(scar_mask, D=0.0)  # scar`) that read as a working API — the #1 hallucination trap across the audit (blocked drug/scar/clamp tasks). Make the error informative and strip the misleading examples.
#### Implementation Spec
**Modify** each remaining stub → `raise NotImplementedError("<name> is not implemented; <real route>")`, and remove the `>>>` worked examples (replace with a one-line "not implemented — see …"). Keep signatures.
#### Test Spec
- `::test_stubs_have_informative_errors` — call a few (`get_state`, `clamp_voltage`); assert the message is non-empty and names the method/alternative.
#### Checklist
- [ ] All stubs: informative message + de-worked docstring
- [ ] Test asserts message quality
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k stub -q`
#### Exit Criteria
- [ ] No stub advertises a working example
#### Risk
Low.

### Step 3.2: Implement `scale_conductance` + `set_conductivity`  (**/audit this step**)
**Model**: opus
#### Read First
- `api.py` `with_()`/`reset()`/`stimulate()` — the rebuild-from-`_data`+`_build_kwargs` pattern.
- `api.py:1451` (lbm) + mono/bidomain ionic-instance handling — the factory accepts a pre-built `IonicModel` instance ("tuner-scaled … use as-is"); `_build_mesh_data`/`CardiacMeshData.D_xx` for per-node D.
#### Why
The two highest-value capabilities (drug/tuning + scar/fibrosis); the engine already supports a scaled instance and a per-node `D_field`.
#### Implementation Spec
- `scale_conductance(name, factor)` → rebuild the sim with an ionic model whose `params.<name>` is scaled (a `build_scaled_ionic(base, {name: factor})` helper that handles TTP06 `.params` vs ORd `params_override`, validating `name` and raising on unknown). In-place rebuild from t=0 (matches the advertised signature; document the rebuild).
- `set_conductivity(mask, D)` → rebuild `_data` with `D_xx/D_yy = D` on `mask` (0.0 → inexcitable scar) and reconstruct.
#### Pseudocode
```
def scale_conductance(self, name, factor):
    model = build_scaled_ionic(self._build_kwargs['ionic_model'], {name: factor})  # validate name
    self._engine = rebuild(mesh=self._data, ionic_model=model, **rest)
def set_conductivity(self, mask, D):
    data = replace(self._data, D_xx=where(mask,D,self._data.D_xx), D_yy=…)
    self._engine = rebuild(mesh=data, **kwargs); self._data = data
```
#### Test Spec
- `::test_scale_conductance_changes_apd` — `sim.scale_conductance('GKr',0.5)` then run → APD90 longer than baseline; unknown name raises.
- `::test_set_conductivity_scar_blocks` — `sim.set_conductivity(scar_mask, D=0.0)` → scar nodes never activate; wave routes around.
#### Checklist
- [ ] `scale_conductance` rebuilds with a scaled instance; validates name
- [ ] `set_conductivity` rebuilds with per-node D; D=0 blocks
- [ ] Both tests pass; goldens exact
- [ ] Run `/audit` on this step's diff
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k "conductance or conductivity" -q`
#### Exit Criteria
- [ ] Both methods work end-to-end; no golden drift
#### Risk
Rebuild-from-t=0 differs from a true mid-run mutation — mitigation: document it; centralize TTP06-vs-ORd scaling in one `build_scaled_ionic` helper (the audit found their interfaces differ).

### Phase 3 Verification / Exit / Cleanup — full suite green; goldens exact.
**-> Commit point: `feat(cardiac_core): de-trap stub methods; implement scale_conductance + set_conductivity`**

---

## Phase 4: P1 — cheatsheet correctness + solver/`dt` guidance

**Goal**: Make `API_CHEATSHEET.md` tell the truth and surface the #1 usability lever (solver/`dt`), so a cheatsheet-only scientist avoids the traps the audit hit.
**Tier**: small
**Estimated scope**: doc rewrite + a "docs execute" smoke test.

### Step 4.1: Rewrite/extend the cheatsheet
**Model**: opus
#### Read First
- `cardiac_core/API_CHEATSHEET.md` (all) and the audit report's cheatsheet-gap list (Round-1 T3 + Round-2 §"Capability gaps").
#### Why
The cheatsheet has real ERRORS (ORd-on-monodomain) and omits the working functions + the solver/`dt` escape every heavy task needed.
#### Implementation Spec
Add/fix: (a) **"Solver & dt"** — `diffusion_solver`/`linear_solver`/`dt` choices, CN-is-unconditionally-stable so raise `dt`, `dct` for speed (post-Phase-2), the per-step cost; (b) **drug/conductance block** — `scale_conductance` (post-Phase-3) + the model-instance route + `PCa`/`GKr`/`GNa`/`GKs`/`GK1`/`Gto` map, `PCa`≠`GCaL`; (c) `paci`/hiPSC + **ORd runs on LBM only**; (d) `record=("Vm","ionic_states")`, `save_result`/`load_result` (+ tuple order), `dominant_frequency`, `phase_map` (+ two-step `phase_singularities`); (e) per-node-D recipe via `CardiacMeshData`; (f) `fiber_angle`=radians; (g) bidomain-`bath` cost; (h) LBM CV offset ~+30–47%; (i) `TTP06Model(device='cpu')`.
#### Test Spec
- `::test_cheatsheet_examples_execute` — extract fenced ```python blocks and `exec` them (tiny/fast), asserting none raise (or a curated subset if full extraction is brittle).
#### Checklist
- [ ] All nine items added/corrected
- [ ] Cheatsheet code blocks execute
#### Verify
`…/python -m pytest cardiac_core/tests/test_usability_fixes.py -k cheatsheet -q`
#### Exit Criteria
- [ ] No documented call errors; the ORd-monodomain error is gone
#### Risk
Doc drifts again — mitigation: the execute-test is the canary.
**-> Commit point: `docs(cardiac_core): cheatsheet correctness + solver/dt guidance (usability audit)`**

---

## Phase 5 (OPTIONAL): P2 — analysis aggregates / per-beat / axes

**Goal**: Add the missing map/aggregate/per-beat/axis analysis forms so multi-beat and 2-D tasks stop being hand-rolled. Deliver only if Phases 1–4 land cleanly.
**Tier**: medium
### Step 5.1: Aggregate + per-beat + guards
**Model**: opus
#### Implementation Spec (each with a test)
- `dominant_frequency_map(V,times)` (batched rfft) + `r.df_map()`; DF frequency-resolution warning (B10).
- `conduction_velocity` gains a direction (`along='x'|'y'` or `cv_between((x1,y1),(x2,y2))`) + `r.radial_cv(center)`.
- Per-beat family: `apd_per_beat`/`cv_beat`/`capture`; `restitution_slope` (fit + DI\* where slope=1, guarding the div-by-zero B13).
- Distinguish block/no-activation from `nan` (warn or sentinel); warn on zero-node stimulus; `save_every`→CV quantization guard (B11).
#### Test Spec
- One focused test per helper in `test_usability_fixes.py`.
#### Exit Criteria
- [ ] New analysis helpers work; existing analysis unchanged; goldens exact.
**-> Commit point: `feat(cardiac_core): analysis aggregates + per-beat + axis helpers (P2)`**

---

## Future work (documented, NOT in this plan)
P3: per-node fiber-angle field (wire `fiber_field_transmural`) + per-node cell-type + `MID` registration; single-cell 0-D path; `cc.sweep`/`fit_conductivity`; rotor tooling (batched `phase_map`, obstacle-aware tip tracking, mid-run stimulus / phase-IC seeding); `cc.electrogram`/`pseudo_ecg`; disk checkpoint/resume; wire-or-remove dead `stim_amplitudes_e`. P4: LUT/fused ionic kernel to cut the ~13 ms/step overhead; faster bidomain-bath elliptic solve.

## Final Cleanup
- float64 consistency; V5.3 untouched; no cross-engine duplication; scratch prints removed.
- Archive this plan: `mkdir -p Research/Active/engine_consolidation/plans && cp Research/Active/engine_consolidation/PLAN.md "Research/Active/engine_consolidation/plans/2026-07-16_usability-fixes-p0-p1.md"`

## Mutation Log
**REVISED 2026-07-16 (audit round 1 — run inline; the spawned Opus auditor died on the session rate limit, so verification was done in the main loop against the code):**
- **[HIGH] Step 1.2 (B8) re-targeted + LBM-safed** — the fix site is `StructuredGrid.flat_to_grid` (`mesh/structured.py:178`, currently `torch.zeros`), NOT a vague "monodomain reconstruction"; it is shared by monodomain AND bidomain (Vm + phi_e via `api.py:767/782`), so one change covers both. **LBM has no `flat_to_grid`** and `test_api_hardening.py::test_lbm_masked_hole_nonconducting` asserts `isfinite().all()`, so B8 must not touch LBM — added an explicit LBM regression guard + a bidomain masked test. Goldens (`test_integrity.py`) are full-rect → the `reshape` branch → unaffected.
- **[MED] Step 1.6 (B6) formula corrected** — the CFL limit is `dt ≤ Cm·h²/(4·D_max)` (the plan had dropped the `Cm` factor); source in the forward_euler docstring. Pull `D_max`/`Cm` from `self.spatial`.
- **[LOW] Step 2.1 (B2) call site pinned** — `monodomain.py:338` `_build_linear_solver(...)` passes no grid params; verify `D`/`scheme` are in scope there before threading.
- **[LOW] Step 1.3 (B3/B4) risk resolved** — verified `test_analysis.py`'s synthetic APs are dome-free/monotonic, so the dome-aware change can't shift existing asserted values.
- **[LOW] Golden test named** — `test_integrity.py` (full-rectangle) is the bit-identical golden, now referenced in Phase Context + verify commands.
**REVISED 2026-07-16 (audit round 2 — inline residual-risk verification):**
- Verified the one risk the round-1 revision could hide — is `flat_to_grid` (B8) ever called *inside* a solve loop where a mid-computation NaN would corrupt the solve? **No:** all 6 callers are in `api.py` output/snapshot reconstruction; **zero** uses in `_monodomain`/`_bidomain` engine internals. NaN-fill is purely output-side → safe. No new issues.
- **CONVERGENCE:** round-1 = 1 HIGH / 1 MED / 3 LOW (all folded in, 0 CRITICAL) → round-2 = 0 new issues. Plan is SOUND for execution. **Caveat:** both rounds were run **inline in the main loop** (the spawned Opus auditor died on the session rate limit at 4:50am-ET reset), so this is not the usual fully-independent multi-agent audit; a belt-and-suspenders independent subagent audit can be run after the limit resets if desired.
