# PLAN: cardiac_core Foundation Cleanup → Boundary-Mode Build

**STATUS: ✅ COMPLETE (2026-07-01)** — all 3 phases shipped; 195 cardiac_core+mcp tests green;
commits `a3915d1` (P1) · `945350f` (P2) · `736296d` (P3) on `engine-tuner-cardiac-core`.
Deferred items live in the Findings Coverage table + the Phase Mutation Log entries.

Created: 2026-06-30
Engine(s): All (cardiac_core = mono/bidomain/lbm) + cardiac_mcp
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-06-30 "system audit + cleanup decisions" · findings in [CARDIAC_CORE_AUDIT.md](CARDIAC_CORE_AUDIT.md)

> **NOTE — phase numbering:** this PLAN's Phase 1–3 are the *foundation-cleanup track*. They are DISTINCT from
> the README's consolidation Phase 1–5 (mesh/stimulus unify → rewire → delete copies). Do not conflate.

## Objective
Tidy the cardiac_core foundation before building the (real, β/discreteness-driven) boundary-mode
work: fix the 4 confirmed blockers, batch the dead-code/docs/API-consistency cleanup, then lift the
LBM boundary modes (HBB / specular / α-blend) into the engine and expose them through the unified API
with tests. Fixing on clean ground, per the audit's fix-ordering. Hardened by two adversarial audit rounds (see Mutation Log).

## Success Criteria
- [ ] All 4 HIGH blockers fixed with regression tests (#4, cluster-#1, #1, #3)
- [ ] `cardiac_core` full suite green at every phase commit (goldens refreshed within Phase 1)
- [ ] `D_xx` = raw convention uniform across **all three** engines (mono via mass term; **LBM factory AND bidomain D_eff branch divide by χ·Cm**); no mesh runs at ~1400× different D
- [ ] Boundary modes (neumann/dirichlet/absorbing/hbb/specular_neighbour/specular_samecell/combined-α) selectable via `cardiac_core.lbm(boundary=…)`, each with a rest-no-op + mass-conservation test
- [ ] Dead code removed or explicitly marked; API docs reconciled to shipped signatures
- [ ] Every confirmed audit finding is either fixed or in the explicit **Findings Coverage** table (none silently dropped)
- [ ] All existing tests pass (no regressions)

## Architecture Changes
- MOD: `cardiac_core/_monodomain/…/discretization_scheme/fdm.py:558,673-695` — fix Dxy cross-derivative (sign + factor 2); mirror into frozen `Monodomain/Engine_V5.5/…/fdm.py`
- MOD: `cardiac_core/file_format.py:175,191-244` — `D` default 0.001→1.4 + docstring → single "raw, chi divides" convention + band guard (needs `import warnings`)
- MOD: `cardiac_core/api.py` — LBM factory divides `D_xx`/`D_yy` by `chi·Cm` (1375,1384,1392); **bidomain D_eff branch divides** (1240 isotropic, 1249-1254 anisotropic); persist `ionic_model` in build_kwargs (1132/1287/1416); one-shot empty-run guard (`run.py:84`)
- MOD: `cardiac_mcp/core.py:385-394` — sanitize token `date/slug` + `is_relative_to(LAB)` guard in `commit_experiment`
- MOD: `cardiac_core/tests/_integrity/` — refresh Vm goldens (mono+lbm) + `engine_src_sha.json` (mono+lbm) within Phase 1
- NEW: `cardiac_core/_lbm/boundary/` bc-mode registry + `boundary=`/`alpha=` on `LBMSimulation` + `step.py` dispatch (Phase 3)
- DEL: `cardiac_core/_bidomain/simulation/classical/solver/linear_solver/fft.py`, `_lbm/collision/mrt/d2q5.py`, FEM/TriangularMesh (Phase 2, per confirmed FEM-ditch). **NOT** `_monodomain/…/linear_solver/fft.py` (live) or `_lbm/lattice/d2q5.py` (live)

## Known Failures (from IDEALOG — do NOT retry)
- **Big-bang ionic/engine deletion** — breaks Surrogate/Optimizer + engine tests. Per-consumer, test-gated only.
- **Pinning engine `Cm=1` to feed an effective D** — silently breaks the reaction at Cm≠1 (Cm-trap). The real Cm must reach every engine.
- **Treating the α-blend / same-cell-specular speedup as a numerical artifact to remove** — PI has adopted it as REAL (β/discreteness). The τ/β dependence is the physics knob, not a bug.
- **Naming a solver package `monodomain/`** (no underscore) — shadows the public factory. Keep `_monodomain` etc.

---

## Phase 1: Foundation blockers

**Goal**: Fix the 4 HIGH findings, each test-gated, and refresh goldens so the phase commits GREEN. Independently deliverable; unblocks anisotropic + mesh-based work.
**Tier**: large
**Estimated scope**: 4 fixes + regression tests + a golden refresh, across mono FDM, file format + LBM/bidomain factories, api replay, MCP.

### Phase Context
`cardiac_core` is editable-installed; run tests with `conda run -n heart-conduction python -m pytest cardiac_core/tests -q`. All tensors float64. The vendored `_monodomain/_bidomain/_lbm` are the LIVING source; the frozen originals (`Monodomain/Engine_V5.5/`, etc.) are hashed by `test_integrity.py::test_originals_untouched` — editing them turns that test RED until goldens are refreshed (Step 1.5). The bidomain **cardinal4** FDM builder is the trusted anisotropy oracle (its eikonal anisotropic test is validated). **DECISION (user):** `CardiacMeshData.D_xx` is a RAW conductivity-like value; every engine yields physical effective D = `D_xx/(χ·Cm)`. Current state: **monodomain** already does this (χ·Cm in the mass term); **bidomain's sigma branch** divides but **its D_eff branch (api.py:1240,1249-1254) does NOT** (treats D_xx as effective); **LBM** does NOT (passes D_xx straight to the engine). So Step 1.2 must add the χ·Cm division to BOTH the LBM factory AND the bidomain D_eff branch. The precise invariant is **`effective = D_xx/(χ·Cm)` computed once per factory**. `create_cardiac_mesh` stores RAW D_xx + real chi. On the DECLARATIVE path (`_build_mesh_data`): the mono branch stores a chi-folded D_xx with chi=1 (invariant-correct at all Cm — leave it); the **bidomain branch sets sigma_i/sigma_e so the factory uses the SIGMA branch** (raw σ + real chi — unaffected by the D_eff-branch change); but the **LBM branch stores an already-effective D_xx with real chi** → once the LBM factory divides, that path DOUBLE-divides unless `_build_mesh_data`'s LBM branch is changed to store RAW D_xx (Step 1.2).

### Step 1.1: Fix the monodomain FDM anisotropic cross-derivative (wrong sign + half magnitude)
**Model**: opus

#### Read First
- `cardiac_core/_monodomain/simulation/classical/discretization_scheme/fdm.py:556-558` (cxy) and `:668-697` (diagonal entries); ctor default `chi=1400` at `:116`; `apply_diffusion` returns `L·V/(chi·Cm)` at `:238`
- `cardiac_core/_bidomain/simulation/classical/discretization/fdm.py:452-457,506-534` — the CORRECT (oracle) cross-derivative

#### Why
`div(D·∇V) = Dxx·V_xx + 2·Dxy·V_xy + Dyy·V_yy`. The `2·` and the stencil sign must survive. The mono builder uses `cxy=1/(4·dx·dy)` (drops the factor 2 → half) and diagonal signs NE`−`,NW`+`,SE`+`,SW`−` (negated vs the oracle). Net: for `V=x·y` (V_xy=1) the mono interior gives `−Dxy`; the correct value is `+2·Dxy` (empirically confirmed by the audit). Untested (#41), so the bug is silent. Isotropic (Dxy=0) runs are unaffected (diagonal weights are 0).

#### Implementation Spec
**Files to modify:** `cardiac_core/_monodomain/.../fdm.py`
- `:558` `cxy = 1.0/(4.0*dx*dy)` → `cxy = 1.0/(2.0*dx*dy)`
- `:673-695` flip the four diagonal signs to match bidomain: NE `w = d_xy*cxy`, NW `w = -d_xy*cxy`, SE `w = -d_xy*cxy`, SW `w = d_xy*cxy` (each still does `center -= w`).

Mirror the SAME fix into the frozen original `Monodomain/Engine_V5.5/…/fdm.py` (vendoring policy allows editing originals in place; keep in sync). **This turns `test_originals_untouched` RED for the monodomain hash until Step 1.5 refreshes it — expected.**

#### Pseudocode
```
cxy = 1/(2*dx*dy)
NE: w = +d_xy*cxy; add(k, NE, w); center -= w
NW: w = -d_xy*cxy; add(k, NW, w); center -= w
SE: w = -d_xy*cxy; add(k, SE, w); center -= w
SW: w = +d_xy*cxy; add(k, SW, w); center -= w
```

#### Test Spec
Build the discretization DIRECTLY (create_cardiac_mesh cannot produce D_xy≠0 — it only sets per-axis D_yy, D_xy=0):
- `cardiac_core/tests/test_fdm_anisotropy.py::test_cross_derivative_bilinear` — Setup: `grid = StructuredGrid.create_rectangle(1.1,1.1,12,12)`; hand-build `D_field=(Dxx,Dxy,Dyy)` with constant `Dxx=Dyy=1.0, Dxy=0.3` (Nx×Ny tensors); `fdm = FDMDiscretization(grid, D_field=D_field, chi=1.0, Cm=1.0)`; `V[i,j]=x_i·y_j` flattened; `out = fdm.apply_diffusion(V_flat)` (returns `L·V/(chi·Cm)`, chi=Cm=1). Expected: interior ≈ `2·0.3 = 0.6` (tol 1e-10). (Pre-fix yields `−0.3`.)
- `…::test_matches_bidomain_oracle` — same anisotropic `D_field` + a smooth `V`; build the bidomain `_build_laplacian` (cardinal path) on the same field; assert mono interior `L·V` matches the bidomain interior `L·V` (tol 1e-10). Keep Dxx/Dyy/Dxy spatially CONSTANT — harmonic-mean faces equal arithmetic only for uniform D, so the exact oracle match holds only for constant D.

#### Checklist
- [ ] Patch cxy + 4 signs in cardiac_core mono fdm
- [ ] Mirror into Engine_V5.5 fdm
- [ ] Write both regression tests (direct FDMDiscretization construction, chi=Cm=1, 3-tuple D_field)
- [ ] Full mono suite green except the expected mono src-sha (fixed in 1.5)

#### Verify
```
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_fdm_anisotropy.py -q
```

#### Exit Criteria
- [ ] Both new tests pass; isotropic operator bit-identical (Dxy=0 ⇒ diagonal weights 0)

#### Risk
Isotropic runs/goldens use Dxy=0 → unaffected by value; only the src-sha changes (Step 1.5). Mitigation: full suite in Step 1.5.

### Step 1.2: Unify the D_xx=raw / chi-divides convention across ALL engines + fix the blocked default (cluster #1)
**Model**: opus

#### Read First
- `cardiac_core/file_format.py:171-287` (create_cardiac_mesh signature, docstring, D_xx storage; imports at 9-11 — **no `warnings`**)
- `cardiac_core/api.py:1361-1395` (LBM factory D handling) · `:1228-1267` (bidomain **D_eff branch** — does NOT divide) · `:1205-1227` (bidomain sigma branch — already divides) · `:1084-1111` (mono factory — already correct via mass term)
- `cardiac_core/conductivity.py` `for_lbm()` — it feeds `_build_mesh_data`'s LBM branch (api.py:881); the `lbm()` factory itself reads `data.D_xx` (the fix stores RAW there so the factory divide recovers effective — no double-division)

#### Why
`D_xx` means "raw" to monodomain (χ·Cm in mass term) but "already effective" to the LBM factory AND the bidomain D_eff branch. Same mesh → up to ~1400× different physics (#21). Decision: `D_xx` is RAW everywhere; effective = `D_xx/(χ·Cm)`. So **both** the LBM factory and the bidomain D_eff branch must divide. The default `(D=0.001, chi=1400)` gives effective 7.1e-7 → conduction block (#2); under "raw", the physiological default is `D=1.4` (raw) → effective `1.4/1400 = 1e-3`.

#### Implementation Spec
**Files to modify:**
- `cardiac_core/api.py` LBM factory `:1375,1384,1392` — feed `LBMSimulation` effective D: `D_eff_xx = float(data.D_xx.flat[0])/(data.chi*data.Cm)`, `D_eff_yy` likewise.
- `cardiac_core/api.py` bidomain **D_eff branch** `:1240` — `D_eff = float(data.D_xx.flat[0])/(data.chi*data.Cm)`; anisotropic branch `:1249-1254` — divide `data.D_xx/D_yy/D_xy` by `data.chi*data.Cm` before applying `sigma_ratio`. (Leave the sigma branch `:1205-1227` and the mono factory untouched — both already correct. NOTE: this D_eff branch is reached ONLY by the legacy `create_cardiac_mesh`→bidomain path, since the declarative path sets sigma_i/sigma_e.)
- `cardiac_core/api.py` `_build_mesh_data` **LBM branch `:880-882`** — store RAW D_xx so the now-dividing LBM factory recovers effective: `emit = conductivity.for_lbm(); D = emit['D'] * (conductivity.chi * emit['Cm'])` (element-wise if a tuple), with `chi=conductivity.chi, Cm=emit['Cm']`. Cm-safe: real Cm reaches the reaction; the factory divide uses real χ·Cm. The declarative **bidomain** branch `:883-894` needs NO change (sigma branch); the **mono** branch `:877-879` needs NO change (chi=1 + chi-folded D_xx is invariant-correct; full Form-A→B unification is the deferred convergence task).
- `cardiac_core/file_format.py` — add `import warnings`; `:175` `D: float = 0.001` → `D: float = 1.4`; update the `CardiacMeshData.D_xx/D_yy/D_xy` field docstring `:25` ("effective…" → "RAW conductivity-like; effective = `D/(χ·Cm)`"); `:191-244` rewrite the create_cardiac_mesh docstring to ONE convention ("`D` is RAW; effective = `D/(χ·Cm)` in ALL engines; default `D=1.4,chi=1400 → 1e-3`; for an effective D pass it with `chi=1.0`"); before `return`, add band guard: `D_eff=D/(chi*Cm); if not (1e-4<=D_eff<=1e-1): warnings.warn(...)`.

#### Pseudocode
```
# LBM factory:      D_eff = float(data.D_xx.flat[0])/(data.chi*data.Cm); (D_eff_yy likewise)
# bidomain D_eff:   D_eff = float(data.D_xx.flat[0])/(data.chi*data.Cm)   # then D_i/D_e from sigma_ratio
# bidomain aniso:   D_xx_e = (data.D_xx/(data.chi*data.Cm)); ... then *(1+r)/r etc.
# create_cardiac_mesh: D_eff=D/(chi*Cm); if not(1e-4<=D_eff<=1e-1): warn("effective D={D_eff:.2e} outside band; D is RAW (eff=D/(χ·Cm)); pass chi=1.0 to treat D as effective")
```

#### Test Spec
- `cardiac_core/tests/test_chi_convention.py::test_all_engines_same_effective_D` — one raw mesh (D_xx, chi=1400); build mono + lbm + bidomain; assert none conduction-blocked and effective D matches (same order; LBM has a known dispersion offset — assert not-blocked + comparable, not bit-equal).
- `…::test_create_cardiac_mesh_default_propagates` — defaults → mono wave launches downstream, no block.
- `…::test_effective_band_warning` — `create_cardiac_mesh(..., D=0.001, chi=1400)` warns.

**Existing tests affected (LEGACY create_cardiac_mesh path used default chi=1400 with an already-effective small D — the OLD no-divide behavior). Grep found ~54 `create_cardiac_mesh(` sites across 9 `tests/*.py` — give each a disposition:**
- `test_file_format.py::test_default_rectangle:143,144` (`assert D_xx==0.001` AND `D_yy==0.001`) → assert `1.4`; `test_custom_physics` (D=0.002,chi=1200,Cm=0.8 → eff 2.1e-6) now emits the band warning (harmless unless pytest `-W error` — check config).
- `test_lbm.py::test_matches_direct:84-119` + default-chi builders (~lines 15,31,45,60,72,107) → set `chi=1.0` (raw==effective) so the wrapper matches the direct `LBMSimulation(D=…)` side. `test_integration.py::test_lbm`, `test_lbm_anisotropy.py:98,107`, `test_param_seam.py` likewise (also rewrite `test_param_seam.py`'s module docstring `:9-13`, which documents the OLD "D is already-effective, set chi=1.0" convention).
- `test_bidomain.py` — `test_matches_direct:155-175` (direct side uses D_eff; wrapper via create_cardiac_mesh) → wrapper mesh `create_cardiac_mesh(D=D_eff, chi=1.0)`; explicit `D=0.001` sims at `:16,55` → `chi=1.0` or `D=1.4`; the default-D bidomain sims at `:80,94,100` (`test_from_file`/`test_insulated_default`/`test_bath_override`) have structural-only asserts → benign (verify still pass, no edit needed).
- `test_monodomain.py:20,55`, `test_run.py` — mono default-D change is benign (shape-only), but confirm any explicit-`D=0.001` propagation/CV asserts.
- **Declarative-path tests to RE-RUN/VERIFY (must STILL propagate after the `_build_mesh_data` LBM-branch fix — a failure here is NOT expected):** `test_run_contract.py:53`, `test_construction_api.py:47-58,73,112`, `test_viz.py:15` (they call `lbm(Grid,…)`/`bidomain(Grid,…)` with `ConductivityConfig`).
- After updating, verify LBM τ stability at the intended effective D (`tau_from_D` not ≤0.5).

#### Checklist
- [ ] LBM factory + bidomain D_eff branch (iso + aniso) divide by χ·Cm
- [ ] `import warnings`; default `D=1.4`; single-convention docstring; band guard
- [ ] Update every default-chi test listed above; verify LBM stability
- [ ] `grep -rn "create_cardiac_mesh(" ` repo-wide for other callers relying on the old default (note in Mutation Log)
- [ ] Confirm chip pipeline (`cc_runner`/`chip.chip_mesh`, chi=1.0) unaffected (raw==effective there)

#### Verify
```
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_chi_convention.py cardiac_core/tests/test_file_format.py cardiac_core/tests/test_lbm.py -q
```

#### Exit Criteria
- [ ] All three engines produce the same effective D from one raw mesh (no ~1400× divergence); default mesh propagates; updated tests green; chip pipeline unchanged

#### Risk
The default-D change and the divide break pre-existing default-chi tests (enumerated above) — they are UPDATED, not just "unaffected". LBM at very small effective D can approach the τ=0.5 floor — verify stability after updating the tests to chi=1.0. Vm goldens for the default canonical sims change → refreshed in Step 1.5.

### Step 1.3: Persist ionic_model override across reset()/stimulate() (#1)
**Model**: opus

#### Read First
- `cardiac_core/api.py:231-248` (reset/with_), `:337-352` (stimulate→reset); build_kwargs `:1132` (mono, local var `ionic`), `:1287` (bidomain, local var `ionic`), `:1416` (lbm, local var `ionic_name`); `_build_ionic_model` isinstance check `_monodomain/…/monodomain.py:76`

#### Why
Factories resolve the model but never store it; `reset()` replays from `build_kwargs` (no ionic_model) → re-resolves to the mesh default → silently wrong model. `with_()` already works — inconsistent.

#### Implementation Spec
Add `ionic_model=<resolved>` to each factory's `build_kwargs`, **using that factory's own local variable name**: `ionic_model=ionic` in the mono (`:1132`) and bidomain (`:1287`) factories; `ionic_model=ionic_name` in the LBM factory (`:1416`). (Both str and instance replay fine — factories accept either.) Do NOT mutate `data.ionic_model`.

#### Test Spec
- `cardiac_core/tests/test_replay_ionic.py::test_reset_preserves_override` — `sim=monodomain(mesh_ttp06, ionic_model='ord')`; `sim.reset()`; assert rebuilt engine uses ORd.
- `…::test_stimulate_preserves_override` — same via `stimulate`. Test both a string and an instance override.

#### Checklist
- [ ] Correct per-factory variable in all three build_kwargs
- [ ] reset + stimulate tests on a mesh whose default ≠ override (str + instance)

#### Verify
```
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_replay_ionic.py -q
```

#### Exit Criteria
- [ ] reset()/stimulate()/with_() all replay the overridden ionic model

#### Risk
None material. Mitigation: instance-override test.

### Step 1.4: Sanitize commit_experiment path + add LAB containment guard (#3)
**Model**: opus

#### Read First
- `cardiac_mcp/core.py:356-408` (commit_experiment), `:104` (`_slugify`), `:330` (date regex `re.fullmatch(r"\d{4}-\d{2}-\d{2}")`), `:446-449` (run_experiment's `is_relative_to(LAB.resolve())` guard = the pattern to copy)

#### Why
`commit_experiment` reads `date, slug` from the keyless/forgeable token and builds `LAB/f"{date}_{slug}"` then `mkdir(parents=True)` with NO re-validation and NO containment check — unlike `run_experiment`. A forged token (`slug="../../.."`) escapes LAB.

#### Implementation Spec
`cardiac_mcp/core.py:385-394` — after `date, slug = params["date"], params["slug"]`: (1) `re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)` else raise GATE; (2) **re-slugify and use** — `slug = _slugify(slug)` (do NOT `assert slug == _slugify(slug)`: if `_slugify` isn't idempotent that would false-reject a legit slug); (3) after computing `d` (post dedup loop), before `mkdir`: `if not d.resolve().is_relative_to(LAB.resolve()): raise ValueError("GATE: path escapes Lab/")`.

#### Pseudocode
```
date = params["date"]; if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date): raise ValueError("GATE: bad date")
slug = _slugify(params["slug"])
d = LAB / f"{date}_{slug}"; ...dedup...
if not d.resolve().is_relative_to(LAB.resolve()): raise ValueError("GATE: path escapes Lab/")
d.mkdir(parents=True)
```

#### Test Spec
- `cardiac_mcp/tests/test_gate.py::test_commit_rejects_traversal_slug` — forge a token with `slug="../../../tmp/evil"` (build payload + keyless sig via the module's `_sign_payload`); `commit_experiment(token, confirmed=True)` → either raises OR the neutralized slug keeps the folder inside LAB; assert nothing is written outside LAB (check no path outside `LAB` exists).
- `…::test_commit_rejects_bad_date` — token `date="../.."` → ValueError.
- Regression: legit build_manifest→commit still succeeds.

#### Checklist
- [ ] date regex + re-slugify from token
- [ ] `is_relative_to(LAB)` guard before mkdir
- [ ] traversal + bad-date + legit-path tests

#### Verify
```
conda run -n heart-conduction python -m pytest cardiac_mcp/tests -q
```

#### Exit Criteria
- [ ] Forged traversal/date tokens cannot write outside LAB; legitimate commits unaffected

#### Risk
`is_relative_to` needs Py≥3.9 (env 3.11 ✓).

### Step 1.5: Refresh integrity goldens (so Phase 1 commits GREEN) (#15, #42)
**Model**: sonnet (mechanical)

#### Why
Step 1.1 edited the V5.5 original (mono src-sha RED) and Step 1.2 changed the default canonical sims' effective D (Vm goldens for mono AND lbm change). The pre-existing LBM src-sha is already stale (#15). Refresh all so the suite is green and the drift guard is meaningful again.

#### Implementation Spec
- Re-run `cardiac_core/tests/_integrity/make_goldens.py` — it regenerates ALL THREE Vm goldens (`golden_monodomain/bidomain/lbm.pt`) AND all three engine src hashes in `engine_src_sha.json` unconditionally. The bidomain golden WILL change (its canonical sim uses the default D + the divided D_eff branch).
- `test_integrity.py:24,48` — change the silent `pytest.skip()` on missing goldens to a hard fail / `xfail(strict=True)` (#42), and/or add a meta-test asserting all three `golden_*.pt` + `engine_src_sha.json` exist.
- Document the vendoring policy near the integrity test: editing originals in place is allowed; regenerate goldens in the same change.

#### Verify
```
conda run -n heart-conduction python -m pytest cardiac_core/tests -q   # FULL suite must be green here
```

#### Exit Criteria
- [ ] Full cardiac_core suite green (0 failures); missing-golden no longer silently skips

#### Risk
Only refresh goldens AFTER 1.1+1.2 are correct and their own regression tests pass — a premature refresh would bake in a bug. Mitigation: 1.1/1.2 tests gate this step.

### Phase 1 Verification
```
conda run -n heart-conduction python -m pytest cardiac_core/tests cardiac_mcp/tests -q   # expect 0 failures
```
### Phase 1 Exit Criteria
- [ ] 4 blocker fixes in with passing regression tests; goldens refreshed
- [ ] **Full cardiac_core + cardiac_mcp suites GREEN** (Phase 1 is independently deliverable)
- [ ] No isotropic-path value regressions (only src-sha/Vm goldens changed, and they are refreshed)

### Phase 1 Cleanup
- float64 in new tests; Engine_V5.5 mono fdm in sync with the cardiac_core copy; no new cross-engine duplication; V5.3 untouched

**→ Commit point: git commit after Phase 1 passes**

---

## Phase 2: Dead-code, docs & API-consistency tidy

**Goal**: Remove/mark the confirmed dead + unfinished surface, reconcile docs to shipped API, close low-risk footguns. Batchable, non-blocking.
**Tier**: medium
**Estimated scope**: ~15 LOW/MED findings in 5 mechanical steps. Grep for importers before every deletion.

### Phase Context
Each step independent; commit per step is fine. Prefer deleting confirmed-dead code. Keep the package importable (don't break any `__init__`). Goldens are already refreshed (Step 1.5) — deletions here shouldn't touch numerics; re-run the suite after each.

### Step 2.1: Resolve FEM/TriangularMesh (confirmed FEM-ditch) (#28)
Delete `cardiac_core/mesh/triangular.py` and `cardiac_core/_monodomain/…/discretization_scheme/fem.py`, and remove ALL their import/export sites: `cardiac_core/mesh/__init__.py:11,17`; `_monodomain/simulation/classical/monodomain.py:19` (top-level `from …fem import FEMDiscretization`); `_monodomain/__init__.py:13,21`; `_monodomain/…/discretization_scheme/__init__.py:12,21`. Keep FVM (structured-grid-native, per the FEM-ditch decision; see Findings Coverage for #35). `cardiac_core/grid.py:4-5` already asserts "FEM/TriangularMesh dropped" — that claim BECOMES true once the export is deleted (the defect #28 is the lingering export, not the claim) → no doc edit needed there, just delete the export. Grep for any other importer first. Test: `import cardiac_core` + full suite green; `TriangularMesh`/`FEMDiscretization` no longer importable.

### Step 2.2: Delete orphaned/dead code (#30, #44, #45, #31, #46)
- `cardiac_core/_bidomain/simulation/classical/solver/linear_solver/fft.py` (deprecated, wrong-normalization, orphaned — not in that dir's `__init__`) — delete after grep confirms no importer. **Do NOT delete `_monodomain/…/linear_solver/fft.py`** (live: DCT/FFT mono solver).
- `cardiac_core/_lbm/collision/mrt/d2q5.py` (unreachable; engine rejects mrt+d2q5) — delete + drop its import; keep the ValueError guard. **Do NOT touch `_lbm/lattice/d2q5.py`** (live, imported at `lattice/__init__.py:1`).
- `_lbm/simulation.py:98-101` — remove the resolved MRT/weights_mode TODO (first verify which `_step_fn` MRT actually uses; wire it if that's the real gap).
- `_lbm/simulation.py:247-253` `get_activation_times` redirect stub — delete (analysis.py owns it) or implement via `analysis.compute_activation_time`.
- `_lbm/collision/mrt/d2q9.py:63-68` — downgrade the "full anisotropic tensor" docstring to "diagonal (D_xx,D_yy) only; ignores D_xy" (#46).
Test: full suite green after each deletion.

### Step 2.3: Mark unimplemented public surface honestly (#7/#12, #13)
- `api.py:282-697` — the ~21 `CardiacSimulation` methods that `raise NotImplementedError`: strip the misleading `>>>` examples + add a class-docstring "planned/unimplemented" list, OR give each `NotImplementedError` a message pointing to the working alternative (e.g. `compute_cv` → `analysis.conduction_velocity`). Do NOT leave worked examples on bare stubs.
- `_bidomain/…/bidomain.py:195-198` — `pcg_gmg` selector: `warnings.warn("GMG unimplemented; falling back to PCG")` (or raise). Delete the never-imported `multigrid.py`/`pcg_gmg.py` stubs if GMG isn't coming.

### Step 2.4: Reconcile API docs to shipped code (#9,#10,#11,#24,#25,#26,#36,#43)
In `cardiac_core/API_CHEATSHEET.md` and `Research/Active/engine_consolidation/{API_REFERENCE.md, API_DESIGN.md, GLOSSARY.md}` and `cardiac_core/_monodomain/__init__.py`:
- #9: Quickstart imports → `from cardiac_core.ionic import TTP06Model, CellType`, `from cardiac_core.stimulus.protocol import StimulusProtocol`, real mask helpers.
- #10: remove non-existent factory kwargs (`Cm`/`ionic_solver`/`splitting`/`parabolic_solver`); add shipped `device=`.
- #11: `result.cv(x1,x2,y)` requires indices (not arg-free).
- #24/#25: drop "Vm rename pending" (shipped) and "77 tests" (→149/cheatsheet canary).
- #26: `_monodomain/__init__.py:1-6` docstring advertises the no-underscore path `from cardiac_core.monodomain import …` → fix to `_monodomain` (the actual issue is the docstring lines 1-6, NOT line 28).
- #20/#27: `cardiac_core/__init__.py:28` — the "`triggers _prepare_engine on use`" comment is stale (that mechanism was removed) → update to "imports the vendored _monodomain/_bidomain/_lbm packages on first use".
- #36: FE/RK CFL docstrings (`_monodomain/…/explicit/forward_euler.py:5,28-31`) omit χ → `dt ≤ χ·Cm·h²/(4·D_max)`; state assumed χ in the worked example.
- #43/#22: note `for_bidomain()` is a helper the factory doesn't call (or delete it — see Findings Coverage).
Spot-check each corrected example actually runs.

### Step 2.5: API-consistency footguns (#5,#6,#16,#17,#34)
- `run.py:84` (`_collect`) — guard the empty (zero-save) case, mirroring `_result_from` (#5).
- Ionic-model registry parity (#6): **extend** `_build_ionic_model` (mono/bidomain) to accept phas13/paci/mhas13 (they share the `IonicModel` ABC — additive, low-risk) so all engines take the same `ionic_model=` strings. (The `'paci'→PHAS13Model` alias correctness (#12/gap) is a SEPARATE investigation → Findings Coverage / next round, not gated here.)
- `api.py:1380-1387` — warn/raise when explicit `lattice='d2q5'` is overridden to d2q9/mrt for anisotropic D (#16); record the effective lattice in build_kwargs.
- `api.py:1066-1067,1180-1181,1332-1333` — raise if both positional mesh and `mesh=` given (#17).
- `cardiac_mcp/core.py:497` — escape `|`/newline in `goal` before writing the NOTEBOOK row (#34).

### Phase 2 Verification
```
conda run -n heart-conduction python -m pytest cardiac_core/tests cardiac_mcp/tests -q
```
### Phase 2 Exit Criteria
- [ ] No confirmed-dead module remains (or is explicitly marked); docs examples run; footguns raise/warn; suite green
### Phase 2 Cleanup
- Grep confirms deleted symbols have no importers; docs spot-checks pass; float64 consistency

**→ Commit point: git commit after Phase 2 passes**

---

## Phase 3: Boundary-mode build (the real feature)

**Goal**: Lift the LBM boundary modes into the engine as a first-class, selectable, tested concept and expose via the unified API — HBB / specular-neighbour / specular-samecell / combined-α (the real β-controlled curvature knob), plus wiring the orphaned dirichlet/absorbing. Folds in #29, #37, #38, #39, #40, #14.
**Tier**: large
**Estimated scope**: 1 design step + engine impl + API exposure + tests. Reference [BC_IMPLEMENTATION_AUDIT.md](../boundary_conduction_speedup/BC_IMPLEMENTATION_AUDIT.md) (kernels verified there; this productizes them).

### Phase Context
Kernels are proven in `Research/Active/boundary_conduction_speedup/diag_lbm_specular.py` (α-blend = `apply_combined_top_bottom_d2q9`, line 517; slot maps verified vs the D2Q9 lattice). Move them into `_lbm` behind a selector — do NOT re-derive the physics. Naming must be unambiguous (the BC audit found `--bc specular` collides across scripts): `hbb`, `specular_neighbour` (zero-bias), `specular_samecell` (inverse), `combined` (α-blend: α=1→hbb … α=0→samecell). `neumann` is an ALIAS of `hbb` (current default; keep bit-identical). Effect is REAL (PI); carry the flat-wall-only + τ/β-dependence notes in docstrings, not as artifact-disclaimers.

### Step 3.1: Design the LBM bc-mode selector (registry + dispatch + API surface)
**Model**: opus. Read `_lbm/step.py`, `_lbm/simulation.py`, `_lbm/boundary/*`, and BC_IMPLEMENTATION_AUDIT.md §1. Produce a short design note (IDEALOG/WHITEBOARD) settling: registry shape (name→appliers for d2q5/d2q9), how `boundary=` + `alpha=` thread `lbm()` → `LBMSimulation` → `step.py` dispatch (replacing the hardcoded `apply_neumann`), corner/east-west handling (HBB), mask interaction (flat top/bottom walls only; slanted → HBB fallback), and whether `combined` α is a sim attribute or per-step arg.

### Step 3.2: Implement the selector in `_lbm` (#29, #37, #38)
Add `boundary` + `alpha` params to `LBMSimulation.__init__`; add `_lbm/boundary/registry.py` mapping mode→applier; make `step.py` dispatch (wire existing `apply_dirichlet_*`/`apply_absorbing_*`; port `apply_neumann`(hbb)/`apply_specular_*`/`apply_combined_*` from the research script into `_lbm/boundary/`). Default `neumann` (≡ hbb) → bit-identical to today (goldens unchanged).

### Step 3.3: Expose via `cardiac_core.lbm(boundary=…, alpha=…)`
Thread `boundary`/`alpha` through the `lbm()` factory → `LBMSimulation`, and into `build_kwargs` (reset/with_ replay). Add to `API_CHEATSHEET.md`. Default `boundary='neumann'` preserves behavior.

### Step 3.4: Tests (#38, #39, #40)
- Per-mode **rest no-op** (`max|V−V_rest|≈0` on uniform field) + **mass conservation** for hbb/specular_neighbour/specular_samecell/combined(α∈{0,0.5,1}).
- **`boundary='neumann'` == `boundary='hbb'`** bit-identical (pins the "default unchanged" claim).
- α endpoints: `combined(α=1)` == `hbb`; `combined(α=0)` == `specular_samecell`.
- Deficit-ratio cross-check vs the monodomain Moore-8 numbers (2/3, 5/6) where applicable.
- Bidomain `test_bidomain.py:93-103` — strengthen: assert phi_e / edge-CV DIFFER between `boundary='insulated'` and `'bath'` (#39).
- LBM `D_xy≠0` ValueError test (#40).

### Step 3.5 (optional): FDM boundary Dxy Neumann treatment (#14)
Give FDM diagonal off-grid neighbours a `boundary_mode`-consistent ghost (mirror the off-grid axis, as `face_mirror_iso` does for Moore-8) instead of dropping them, OR raise/warn when `Dxy≠0` at a wall. Pairs with Step 1.1. Defer if it expands scope.

### Phase 3 Verification
```
conda run -n heart-conduction python -m pytest cardiac_core/tests -q
```
### Phase 3 Exit Criteria
- [ ] Boundary modes selectable via `lbm(boundary=…)`; each tested (rest no-op + mass); neumann==hbb + α endpoints asserted; bidomain bath≠insulated asserted; default unchanged (goldens green)
### Phase 3 Cleanup
- Naming unambiguous (no `specular` collision); docstrings carry flat-wall-only + τ/β notes; BC_IMPLEMENTATION_AUDIT.md updated; no dup between research script and `_lbm`

**→ Commit point: git commit after Phase 3 passes**

---

## Findings Coverage (every confirmed audit finding → disposition)
Fixed: #1→1.3 · #2/#8/#21→1.2 · #3→1.4 · #4/#41→1.1 · #15/#42→1.5 · #5/#6/#16/#17/#34→2.5 · #7/#12(stub)/#13→2.3 · #9/#10/#11/#20/#24/#25/#26/#27/#36/#43→2.4 · #22→2.4(note)/2.x(delete) · #28→2.1 · #30/#31/#44/#45/#46→2.2 · #29/#37/#38/#39/#40→Phase 3 · #14→3.5(optional).
**Deferred (explicit, not dropped):**
- #18 (`record=`/ionic_states in one-shots) · #19 (LBM save-cadence/epsilon parity) — API-parity polish; next round.
- #23 (anisotropic bidomain unreachable via `ConductivityConfig.anisotropic` — no sigma_i/sigma_e) — needs a new anisotropic-bidomain constructor; design decision, next round.
- #32 (keyless MCP token) · #33 (`confirmed=True` is model-settable) — LOW under the stdio-local trust model; soften docs / HMAC only if a remote transport is enabled.
- #35 (FVM drops all Dxy) — FVM is unreachable from the public API (no `scheme=`); add a 3-tuple/`Dxy≠0` guard if/when FVM is exposed.
- #12 `'paci'→PHAS13Model` alias correctness — verify vs published Paci AP; next round.
- The 12 **completeness gaps** in CARDIAC_CORE_AUDIT.md (analysis.py, io.py round-trip, LBM masked grid, geometry conventions, stimulus shape, device/dtype, bidomain steppers, second mesh loader, etc.) — the next audit round.

## Final Cleanup
- float64 across new code; V5.3 untouched; boundary kernels live once (`_lbm`)
- Update `CARDIAC_CORE_AUDIT.md` finding statuses + `KNOWLEDGE.md` (chi/D single-meaning; boundary modes productized)
- Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_cardiac-core-foundation-cleanup-boundary-mode-build.md"
```

## Mutation Log
- **AUDITED 2026-06-30 (Round 1)**: two adversarial auditors (code-fidelity + executor-readiness). Applied: bidomain D_eff branch must also divide (Step 1.2 — my "bidomain already divides" claim was FALSE for the D_eff path, api.py:1240/1249-1254); Phase-1 golden refresh moved IN as Step 1.5 so the phase commits green (1.1 reddens mono src-sha, 1.2 changes Vm goldens); Step 1.1 test uses a hand-built FDMDiscretization (create_cardiac_mesh can't make Dxy≠0); enumerated existing default-chi tests to update in 1.2 (test_file_format, test_lbm/test_matches_direct, test_integration, test_lbm_anisotropy, test_param_seam); per-factory var name in 1.3; re-slugify (not assert-equal) in 1.4; full FEM import-site list + exact fft.py path + "not the live d2q5/fft" caveats in 2.1/2.2; corrected #26 line ref; added the Findings Coverage/Deferred table; `import warnings`; phase-numbering note.
- **AUDITED 2026-06-30 (Round 2)**: verify-fixes auditor. Caught that the DECLARATIVE LBM path stores already-effective D_xx + real chi → the new factory divide would double-divide → added the `_build_mesh_data` LBM-branch raw-storage fix (declarative bidomain uses the sigma branch = unaffected; declarative mono is invariant-correct). Added `test_bidomain.py` + a per-file test disposition + a declarative-path re-run/verify list; corrected Step 1.5 (make_goldens regenerates all three goldens/SHAs unconditionally, bidomain golden changes); added #20/#27 to coverage; `test_file_format` :143+:144; harmonic-mean-constant-D note on the 1.1 oracle test; import-site line :21.
- **AUDITED 2026-06-30 (Round 3 — CONVERGED)**: verdict SOUND (0 blockers, 0 majors). Confirmed the chi/D fix is correct + complete on all live paths (save/load round-trips D_xx verbatim; reset/with_ replay consistent; `mesh/loader.py` is the one other effective-D loader but is dead/deferred; `run_*` delegate to factories). All 46 findings accounted for. Applied the 3 residual doc/enumeration minors: field docstring `file_format.py:25`, `test_param_seam` module docstring `:9-13`, test_bidomain default-D structural sims noted, for_lbm Read-First bullet reworded, count 54. **Convergence trajectory: R1 (5 blk/6 maj) → R2 (2/2) → R3 (0/0).**
- **EXECUTED 2026-06-30 (Phase 1 — COMPLETE, committed)**: Steps 1.1–1.5 done. 1.1 mono cross-derivative fixed (cardiac_core + V5.5 mirror) + `test_fdm_anisotropy` (3). 1.2 chi/D=raw: LBM factory + bidomain D_eff branch divide by χ·Cm; `_build_mesh_data` LBM stores raw; default `D=1.4`; band guard; `test_chi_convention` (5); updated test_file_format/test_lbm/test_bidomain/test_lbm_anisotropy/test_param_seam. 1.3 `ionic_model` in all 3 build_kwargs + `test_replay_ionic` (4). 1.4 MCP `commit_experiment` date-regex + re-slugify + `is_relative_to(LAB)` + 2 traversal tests. 1.5 goldens regenerated (all propagate, Vmax 114–118; old ones captured the BLOCKED default) + integrity skip→fail (#42). **Full suite: 179 passed / 0 failed** (was 148/1). MUTATION: Step 1.1 test 2 implemented as a direct coefficient-oracle (pins NE+/NW−/SE−/SW+ × cxy) instead of instantiating BidomainFDMDiscretization — equivalent, avoids cross-package grid fragility. DEFERRED: 9 cosmetic band-warnings on matches-direct fixtures that intentionally use tiny D at default chi (benign; tests pass).
- **EXECUTED 2026-06-30 (Phase 2 — COMPLETE, committed)**: 2.1 FEM/TriangularMesh deleted (2 files + all import/export sites; `Mesh` ABC KEPT for StructuredGrid; test_mesh_shared import trimmed). 2.2 deleted orphaned `_bidomain/.../linear_solver/fft.py` + `_lbm/collision/mrt/d2q5.py` (NOT the live mono fft / lattice d2q5); removed the resolved MRT-TODO comment + the `get_activation_times` stub; downgraded the MRT-d2q9 "full tensor" docstring (#46). 2.3 `pcg_gmg` now warns on PCG fallback (#13); `CardiacSimulation` docstring flags the planned/unimplemented methods (#7/#12). 2.5 `run.py::_collect` zero-save guard (#5); positional-mesh vs `mesh=` clash raises (#17); NOTEBOOK goal cell-escaped (#34). Docs: `cardiac_core/__init__` `_prepare_engine` comment (#20/#27), `_monodomain/__init__` underscore-path docstring (#26). **Full suite: 179 passed / 0 failed.** DEFERRED (noted): 2.4 Research/ API_REFERENCE/API_DESIGN/GLOSSARY/CHEATSHEET reconciliation (#9/#10/#11/#24/#25/#36/#43 — documentation-only, no shipped-code impact); #6 ionic-registry parity (behavior change; paci→PHAS13 alias unresolved, gap #12); #16 lbm lattice-override warning (minor).
- **EXECUTED 2026-06-30 (Phase 3 — boundary-mode build, COMPLETE, committed)**: 3.1 design settled (registry + overlay-after-neumann; names hbb/specular_neighbour/specular_samecell/combined). 3.2 new `_lbm/boundary/wall_modes.py` (kernels ported from diag_lbm_specular.py, slot maps verified) + `lbm_step_d2q9_bgk_wall` (step.py) + `boundary`/`alpha` params + validation + step() dispatch (simulation.py); **neumann/hbb route through the UNCHANGED bit-identical path → goldens safe**. 3.3 exposed via `cardiac_core.lbm(boundary=, alpha=)` (both branches; build_kwargs replays). 3.4 `test_lbm_boundary` (16): rest-noop+mass per mode, neumann==hbb, combined(1)==hbb, combined(0)==specular_samecell, modes-differ, D2Q9/unknown validation, default-neumann, reset-replay, oblique-Dxy reject (#40). **Full suite: 195 passed / 0 failed** (goldens bit-identical). Two TEST bugs found+fixed en route (precompute_bounce_masks → all-False on a full periodic domain, use engine masks; modes can't differ at step-1-from-equilibrium, run a front). DEFERRED: 3.5 FDM boundary-Dxy Neumann (#14, optional); #39 bidomain bath≠insulated assert (test-strengthening); wiring orphaned dirichlet/absorbing as selectable modes (#37 — need a bc value; the wall-mode axis is the feature). **ALL THREE PHASES DONE.**
_(future: `**MUTATED {date}**: Step X.Y {SKIPPED|SPLIT|INSERTED} — {reason}`)_
