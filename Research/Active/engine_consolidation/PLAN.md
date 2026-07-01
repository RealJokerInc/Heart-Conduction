# PLAN: cardiac_core API Consistency Hardening + Stress Harness

Created: 2026-07-01 · Revised: 2026-07-01 (audit round 1 → 5 blockers / 10 majors folded in; see Mutation Log)
Engine(s): All (cardiac_core api/run/io/mesh + `_lbm`/`_monodomain`/`_bidomain`)
Research question: [engine_consolidation](README.md)
Source: [API_CONSISTENCY_AUDIT.md](API_CONSISTENCY_AUDIT.md) (7 HIGH · 8 MED · 6 LOW) + [PHASE3_BOUNDARY_GAPS.md](PHASE3_BOUNDARY_GAPS.md) + [IDEALOG.md](IDEALOG.md) 2026-07-01

> Supersedes the 2-gap boundary-fix plan (archived). A 4-lens adversarial audit showed the two
> boundary gaps were symptoms of a class: the numerics are sound, but the API surface is fragile —
> capability that isn't exposed, one kwarg meaning different things per engine with no validation, and
> a few real silent-wrong-result bugs. This plan fixes them AND builds a contract-matrix stress harness
> so the class can't regress.

## Objective
Make the public API behave consistently at every entry point × engine × param × physics cell: each
either takes effect, or raises a *validated* error, or emits an explicit warning — never silently
degrades or means something different per engine. Then lock it with a parametrized stress harness whose
contract table is **written first** (Phase 0) so a never-considered cell surfaces as a standing xfail,
not an absence.

## Execution order (strictly ascending: 0 → 5)
**Phase 0 FIRST** (write the contract matrix, commit it with unfixed cells as `xfail`) → Phase 1 → 2 →
3 → 4 → **Phase 5 LAST** (flip the matrix's xfails to live asserts). Rationale: the post-mortem's #1
lesson is "write the contract before coding, or you test what you built, not what you promised." Phases
1–4 each turn specific matrix cells `xfail → xpass`; Phase 5 removes the xfail markers and adds the
cross-engine invariants. Each phase stays independently committable and keeps the suite green (xfail is
not a failure).

## Success Criteria (mirror the contract matrix)
- [ ] **Phase 0 contract table exists before any fix** — `test_api_contract.py::CONTRACT` enumerates every `{entry × engine × param × physics}` cell as `(expected: effect|raise|warn, match, finding_id, status: to_fix|deferred|landed, exc=ValueError)` (`exc` last so the namedtuple default is valid). `to_fix`→`xfail(strict=True)` (XPASS-on-landing FAILS the suite → forces an in-phase flip to `landed`); `deferred`→`xfail(strict=False)`; `landed`→no marker (live regression-lock assert). Suite green at Phase 0.
- [ ] The 5 silent-wrong-result HIGHs fixed: `run_lbm` parity (P1), bidomain `boundary` validation (S1), mrt+wall (C1), LBM masked grid (I1, **outer wall + interior hole both bounce**), `with_`/factory immutability (I2)
- [ ] Silent-degrades become explicit (warn/raise): `alpha`-inert (S2), `sigma_ratio`-ignored (S3), `lattice`-override (C4)
- [ ] Cross-engine capability unified/exposed where cheap: ionic registry via **one shared builder that branches on ctor capability** (C3), `weights_mode`/MRT knobs (C5), `stencil` (C6), **bidomain `splitting` exposure + validated engine-mismatch errors (S4)**, result dtype round-trip (I3)
- [ ] Oblique-LBM (C2) is left a **documented, validated, message-bearing limitation** (default) — it is a *real numerics gap* (moment-space rotation of `s_jx/s_jy`, Audit #46), NOT a wiring gap; **C7** (mono/bidomain boundary-Dxy truncation) is dispositioned *consistently with* C2 (all three engines decline oblique — no silent per-engine divergence)
- [ ] A `test_api_contract.py` stress harness parametrizes `{lbm, run_lbm, simulate, reset, with_} × {mono, bidomain, lbm} × {params} × {iso-BGK, per-axis-MRT, masked-grid, Cm≠1}` — all **15 HIGH/MED findings** (7 HIGH + 8 MED, incl. C7) are asserted cells; every `raise`/`warn` cell pins `match=`; deferred cells are explicit `xfail(reason=<ID>)`
- [ ] Default paths bit-identical (integrity goldens unchanged — **`test_integrity.py` run in every phase that touches a default path**: 2, 3, 4); all existing tests pass

## Architecture Changes
- MOD `cardiac_core/run.py::run_lbm` — forward `boundary`/`alpha` (P1)
- NEW `cardiac_core/_lbm/step.py::lbm_step_d2q9_mrt_wall` — signature MIRRORS `lbm_step_d2q9_mrt` (`w` right after `dt`): `(f, V, R, dt, w, s_e, s_eps, s_jx, s_q, s_pxx, s_pxy, bounce_masks, mode, alpha, Nx, Ny, s_jy=None)` — NOT the `omega`-based `bgk_wall` signature; + MOD `_lbm/simulation.py` (drop bgk-only guard, insert `mrt&&special` dispatch branch *before* the plain `mrt` branch) (C1)
- MOD `cardiac_core/api.py::bidomain` — validate `boundary` ∈ {bath,insulated,None} (S1, mirror LBM); `lbm` — **union**ed masked bounce_masks (I1), warn on `lattice` override (C4) + `alpha`-inert (S2); `bidomain` — warn on ignored `sigma_ratio` (S3); expose `splitting` (S4)
- MOD `cardiac_core/api.py::CardiacSimulation._resolve_mesh` — `copy.deepcopy` the `CardiacMeshData` branch (closes factory + `with_` + `reset` aliasing in one place) (I2)
- NEW `cardiac_core/ionic/registry.py::build_ionic_model(name, cell_type='ENDO', device='cuda')` — ONE shared name→instance builder that **branches on ctor capability** (`cell_type` only to TTP06/ORd; device-only to PHAS13/MHAS13/paci); default `cell_type='ENDO'` MATCHES every current engine default (goldens-safe); mono forwards its mesh-derived cell_type, bidomain/lbm delegate w/o cell_type → ENDO (unchanged); all three resolvers delegate (C3)
- MOD `lbm()`/`monodomain()`/`bidomain()` — surface `weights_mode`/MRT knobs (C5) + `stencil` (C6)
- MOD `cardiac_core/io.py::load_result` — round-trip stored dtype for `V` and `phi_e` via `torch.from_numpy` (I3)
- NEW `cardiac_core/tests/test_api_contract.py` — the contract matrix (Phase 0) + the stress harness (Phase 5)

## Known Failures / lessons (from the post-mortem + this audit — do NOT repeat)
- **Tested at the level BUILT, not USED.** Write tests from `{entry × engine × param × physics}` FIRST (Phase 0); drive the OUTERMOST surface (`run_lbm`/`simulate`), not the inner one you touched.
- **A guard + a test asserting its rejection turns an unexamined assumption into a "feature."** Every `raise` is a claim to justify. For the *oblique* raise (kept), the test must assert the **documented-limitation message**, not "physics unsupported," and must be REPLACED (not preserved) if the limitation is ever lifted.
- **Wall overlay is post-stream → collision-agnostic** (don't gate on bgk). BUT **MRT is NOT oblique-capable today**: `mrt_collide_d2q9` carries `s_pxy` as a *free stability rate* and its docstring says `D_xy` is NOT applied (`p_xy_eq=0`; needs moment-space rotation of `s_jx/s_jy`, Audit #46). The audit's "MRT is oblique-CAPABLE" was half-true — the tau helpers compute `tau_xy`, the collision kernel discards it. Per-axis (`Dxx≠Dyy, Dxy=0`) DOES work on MRT; oblique (`Dxy≠0`) does not.
- **`precompute_bounce_masks` uses periodic `torch.roll`** → on a full-border tissue mask it detects NO outer walls (documented in LBM V1 `DESIGN_AND_CHANGES.md:100`). Masked-grid support must UNION it with the rectangular outer-edge masks (or pad the mask), never replace.
- The numerics are SOUND (Cm≠1, build_kwargs replay, mesh round-trip) — do NOT churn them; this is surface work. Confirm goldens bit-identical after every default-path touch.

---

## Phase 0: Contract matrix (write FIRST, commit red-as-xfail)
**Goal**: enumerate the full `{entry × engine × param × physics}` contract as data BEFORE any fix, so unfixed/never-considered cells are visible xfails, not absences. **Tier**: small. Independently committable.
### Phase Context
Test runner: `/home/norepinephrine/.conda/envs/heart-conduction/bin/python -m pytest cardiac_core/tests -q` (conda is off the non-interactive PATH). float64. **There is NO `conftest.py` under `cardiac_core/tests/`** — define a local tiny-mesh helper, do not import a shared fixture.
### Step 0.1: `test_api_contract.py` skeleton + `CONTRACT` table
**Model**: opus
#### Implementation Spec
Create `cardiac_core/tests/test_api_contract.py` with:
- A `_tiny(**kw)` helper: `create_cardiac_mesh(0.2, 0.1, 0.02, chi=1.0, **kw)` (~11×6; `chi=1.0` avoids the firewall-bypass block, see IDEALOG 2026-06-30). Wall-divergence cells override to `dt=0.005`.
- A module-level `CONTRACT = [ Cell(entry, engine, param, physics, expected, match, finding_id, status, note, exc), … ]` (a `namedtuple(..., defaults=[ValueError])` — `exc` is LAST so the single default binds to `exc`, not `note`; namedtuple defaults are right-aligned, so a mid-tuple defaulted field is invalid) enumerating the cross-product. `expected ∈ {'effect','raise','warn'}`; `match` = a stable message substring (REQUIRED for `raise`/`warn`); `exc` = expected exception type for `raise` cells (default `ValueError` — every planned raise is a ValueError); `status ∈ {'to_fix','deferred','landed'}` (see `_marks` below). Cover at minimum every HIGH+MED finding — **all 15: P1,S1,S2,S3,S4,C1,C2,C3,C4,C5,C6,C7,I1,I2,I3** — as ≥1 row, PLUS the cross-engine invariant rows (§ below). **C2 gets TWO rows pre-declared here**: `C2-raise` (`expected='raise', match='oblique|Audit #46', status='to_fix'` — the raise exists TODAY but with a `'D_xy'` message; Phase 2 rewords it to match, which flips this cell `to_fix→landed`) and `C2-capability` (the oblique-CV-correct `effect` cell, `status='deferred'`, permanent `xfail(strict=False)` — body never runs). **C7 gets one `status='deferred'` row** (mono/bidomain oblique-wall truncation). The `lbm(boundary='bath')→raise` regression-lock is `status='landed'` (already correct today → live assert, no marker). So Phase 5 only removes `to_fix` markers, never *adds* cells.
- Build the parametrize argvalues so each cell's marks are injected from its data: `[pytest.param(c, id=f"{c.finding_id}:{c.entry}:{c.engine}:{c.param}:{c.physics}", marks=_marks(c)) for c in CONTRACT]`, where `_marks(c)` = `xfail(reason=c.finding_id, strict=True)` for `status=='to_fix'`; `xfail(reason=c.finding_id, strict=False)` for `status=='deferred'`; `()` (NO marker → live assert) for `status=='landed'`. `finding_id` in the param `id=` mechanically satisfies "finding ID appears in a test id."
- A single parametrized `test_contract(c)` dispatching on `c.expected`: `'raise'` → `pytest.raises(c.exc, match=c.match)`; `'warn'` → `with warnings.catch_warnings(): warnings.simplefilter('always'); pytest.warns(UserWarning, match=c.match)`; `'effect'` → assert the change (result differs / engine attr set / replay preserves). **Landing a fix = flip that finding's cells `status: 'to_fix' → 'landed'` IN THE SAME PHASE** — the `strict=True` XPASS forces it (an un-flipped marker fails the suite the moment the fix works), NOT deferred to Phase 5. `deferred` stays `xfail(strict=False)`. Phase 0 commits GREEN (`to_fix` cells xfail-as-expected-fail; `deferred` xfail; `landed` cells assert live and pass).
#### Cross-engine invariant rows (must be in CONTRACT)
- `run_X ≡ simulate(engine=X) ≡ X()` for identical args (P1 parity).
- A param valid on engine A **raises with a message** (not silently drops) on engine B. Concrete required cells: `bidomain(boundary='ncs')`→raise (S1); `lbm(boundary='bath')`→raise (regression-lock, already correct); `LBMSimulation(collision='mrt', weights_mode='uniform_8')`→raise; `lbm(lattice='d2q5', <anisotropic mesh>)`→warn (C4); `bidomain(theta=…)` / `monodomain(theta=…)` engine-mismatch → validated error (S4).
- `with_`/factory leaves the caller's mesh `_data` untouched (I2).
#### Verify
`… -m pytest cardiac_core/tests/test_api_contract.py -q` → green (all unfixed cells xfail).
#### Exit Criteria / Risk
- [ ] The table enumerates every HIGH/MED finding as a row; suite green; no cell is silently missing (each finding ID appears ≥once). Risk: over-broad cross-product → keep grids tiny, `t_end`≤2 ms except wall-divergence cells (`t_end`≥8 ms, `dt`=0.005).
**→ commit after Phase 0** ("contract-matrix-first; unfixed cells xfail").

---

## Phase 1: One-shot API parity (P1)
**Goal**: `run_lbm()`/`simulate()` forward `boundary`/`alpha`. **Tier**: small. Independently committable.
### Phase Context
The `lbm()` factory already accepts+validates `boundary`/`alpha` (api.py:1324-1325); `simulate()` forwards via `**kwargs` (run.py:281); only `run_lbm` (explicit signature, run.py:195-235, calls `lbm(...)` at 230-233) drops them. Wall modes act on the **top/bottom flat walls** (`y=0`, `y=Ny-1`; `_lbm/boundary/wall_modes.py`), so a left-edge stimulus diverges via its top/bottom corners — pin `dt=0.005` to keep the divergence margin large (~70 mV; at the default `dt=0.02` it collapses to ~1 mV).
### Step 1.1: `run_lbm` forwards boundary/alpha + one-shot tests
**Model**: opus
#### Read First
- `cardiac_core/run.py:195-235` (run_lbm), `:238-284` (simulate), `cardiac_core/api.py::lbm` signature (~1324)
#### Implementation Spec
Add `boundary: str = 'neumann', alpha: float = 1.0` to `run_lbm`'s keyword-only block; add `boundary=boundary, alpha=alpha` to its `lbm(...)` call; update the docstring "Forwarded to lbm()". No change to run_monodomain/run_bidomain (mono has no wall mode; bidomain already forwards its `boundary`).
#### Test Spec (`cardiac_core/tests/test_lbm_boundary.py`)
- `test_run_lbm_forwards_boundary` — `run_lbm(stim_mesh, 8, 8, lattice='d2q9', boundary='scs', dt=0.005)` vs `boundary='hbb'`: assert `(V_scs - V_hbb).abs().max() > 1e-2` (a *physically meaningful* divergence, not bare `not allclose`). stim mesh = `create_cardiac_mesh(0.5,0.3,0.02,D=1e-3,chi=1.0)`, stim column at the left edge (top/bottom corners abut the walls).
- `test_run_lbm_alpha_effective` — `boundary='combined', alpha=0.2` vs `alpha=0.8`, `dt=0.005` → `(Va - Vb).abs().max() > 1e-3`.
- `test_run_lbm_rejects_bad_boundary` — `boundary='bogus'` → `pytest.raises(ValueError)`.
- `test_simulate_matches_run_lbm` — both calls pass **identical** `lattice='d2q9', boundary='scs', alpha=1.0, dt=0.005, device='cpu'` AND identical `t_end`/`save_every` (the time axis comes from `snapshots(t_end, save_every)`; a mismatch length-mismatches the stacks and false-fails `torch.equal` for an unrelated reason); compare `simulate(engine='lbm', …).Vm` against `run_lbm(...)[1]` with `torch.equal` (LBM is RNG-free → bit-identical, not `allclose`). This IS the `run_X ≡ simulate(engine=X)` invariant → flip its Phase-0 xfail.
#### Verify
`… -m pytest cardiac_core/tests/test_lbm_boundary.py -q`
#### Exit Criteria / Risk
- [ ] A wall mode through `run_lbm` + `simulate` takes effect (>1e-2 field diff); bad mode raises; `simulate`≡`run_lbm` bit-identical. Risk: without pinned `dt=0.005` the margin collapses — the tests pin it.
### Phase 1 Cleanup / commit
float64; no dup forwarding. **→ commit after Phase 1.**

---

## Phase 2: MRT / anisotropic (per-axis) wall modes (C1)
**Goal**: wall modes work on the per-axis-anisotropic (MRT) path; drop the bgk-only guard. **Tier**: medium.
### Phase Context
`apply_wall_overlay` (wall_modes.py:60) acts on post-stream `f`/`f_star` → collision-agnostic. Rest-neutral on MRT: at rest `f=feq`, the overlay only remaps diagonal slots 5-8 from `f_star` diagonals, and canonical D2Q9 has `w5=w6=w7=w8=1/36` → identity remap regardless of `s_jx/s_jy` (verified: `max|V-V_rest|≈5e-13` over 40 steps). Per-axis = `Dxx≠Dyy, Dxy=0` (→MRT). **OBLIQUE `Dxy≠0` is NOT handled here and NOT in Phase 4** — it is a real numerics gap (Audit #46), so the oblique `raise` STAYS (Step 2.1 + 4.4).
### Step 2.1: `lbm_step_d2q9_mrt_wall` + drop guard + dispatch
**Model**: opus
#### Read First
- `_lbm/step.py:45-60` (`lbm_step_d2q9_bgk_wall` — takes `omega`), `:63-77` (`lbm_step_d2q9_mrt` — takes the 7 moment rates `s_e,s_eps,s_jx,s_q,s_pxx,s_pxy` + `s_jy=None`); `_lbm/simulation.py:88-92` (the `D2Q9_ONLY and lattice!='d2q9'` KEEP at 88-89; the `collision != 'bgk'` guard to DELETE at 90-92), `:218-237` (`step()` dispatch), `_lbm/boundary/wall_modes.py`
#### Implementation Spec
- Add `lbm_step_d2q9_mrt_wall(f, V, R, dt, w, s_e, s_eps, s_jx, s_q, s_pxx, s_pxy, bounce_masks, mode, alpha, Nx, Ny, s_jy=None)` — **argument order MIRRORS `lbm_step_d2q9_mrt` EXACTLY** (`w` immediately after `dt`, then the 6 moment rates, then `bounce_masks`; `mode/alpha/Nx/Ny` appended; `s_jy` last), verified against `_lbm/step.py:63-69`. Body: `mrt_collide → f_star=collide result → clone → stream_d2q9 → apply_neumann (bounce_masks) → apply_wall_overlay(mode, alpha) → recover`. **NOT `bgk_wall`'s `omega` signature** (under MRT `self.omega is None` — cloning the bgk_wall signature and passing `self.omega` crashes).
- DELETE the `boundary in D2Q9_ONLY and collision != 'bgk'` guard (simulation.py:90-92). KEEP the `D2Q9_ONLY and lattice!='d2q9'` validation (88-89) and the MRT `lattice=='d2q9' + weights_mode=='canonical'` requirement (135-141) — those still protect real invariants.
- `step()` dispatch order (insert the new branch **before** the existing plain-`mrt` branch): `if collision=='mrt' and boundary in D2Q9_ONLY: mrt_wall(...)`; `elif collision=='mrt': mrt(...)`; `elif boundary in ('neumann','hbb'): _step_fn(...)`; `else: bgk_wall(...)`. (Use `D2Q9_ONLY = ('specular_nextcell','specular_samecell','combined')`, already imported at simulation.py:22 — there is NO `SPECIAL_WALL` symbol; `WALL_MODES \ ('neumann','hbb') == D2Q9_ONLY` exactly.) Call `mrt_wall` positionally EXACTLY as `step()` calls `lbm_step_d2q9_mrt` today (`self.f, self.V, R, self.dt, self.w, self.s_e, self.s_eps, self.s_jx, self.s_q, self.s_pxx, self.s_pxy, self.bounce_masks, …, s_jy=self.s_jy`, simulation.py:222-226) — i.e. `w` is the 5th arg — then append `mode, alpha, self.Nx, self.Ny`. Do NOT reorder `w`.
- **Golden-safety**: the isotropic-BGK-default-neumann integrity sim must still route to `_step_fn` (it has `collision=='bgk'`, `boundary=='neumann'` → 3rd branch, unchanged). Re-run `test_integrity.py` to confirm.
#### Test Spec (`test_lbm_boundary.py`)
- `test_lbm_anisotropic_boundary_runs` — `create_cardiac_mesh(0.4,0.2,0.02,D=1e-3,D_yy=5e-4,chi=1.0)` (→MRT) + `boundary='ncs'` and `boundary='combined',alpha=0.3` → constructs, `snapshots(6,6)` finite, `_engine.collision=='mrt'`.
- `test_mrt_wall_rest_neutral` — raw `lbm_step_d2q9_mrt_wall` × 40 on uniform V, R=0 → `max|V-V_rest| < 1e-9` (voltage) AND mass drift `< 1e-8` (leave headroom; measured ~5.5e-11).
- `test_lbm_rejects_oblique_Dxy` (REWORD, keep the raise): assert `lbm(<mesh with Dxy≠0>, boundary='ncs')` raises with a message naming the **documented limitation** — `pytest.raises(ValueError, match='oblique|D_xy|not.*wired|Audit #46')`. The docstring/comment must state this is a REAL numerics limitation (moment-space rotation, Audit #46), tests the OBLIQUE case specifically (NOT per-axis anisotropy), and must be REPLACED (not preserved) if Audit #46 is ever implemented. Cross-ref: do NOT keep the `raise` merely to keep this test green.
#### Verify
`… -m pytest cardiac_core/tests/test_lbm_boundary.py cardiac_core/tests/test_lbm_anisotropy.py cardiac_core/tests/test_integrity.py -q`
#### Exit Criteria / Risk
- [ ] `lbm(per-axis-aniso, boundary='ncs')` runs; MRT wall rest-neutral; oblique still raises (documented); goldens bit-identical. Risk: MRT near τ=0.5 — use stable `D_yy=5e-4`.
**→ commit after Phase 2.**

---

## Phase 3: Silent-failure hardening (S1, I1, I2, S2, S3, C4, I3)
**Goal**: the real silent-wrong-result bugs + turn silent-degrades into explicit warn/raise. **Tier**: medium. **Seven findings → seven named tests** (the earlier "5 items" undercounted; S2+S3+C4 are three distinct warns).
### Phase Context
Each is a small, independent guard/wiring change. Add a NAMED test per finding that asserts the *previously-silent* path now warns/raises/behaves (each must FAIL pre-fix, PASS post-fix). Do NOT change numerics. **Run `test_integrity.py` at the end** — I1/I2/I3 all touch default-path code.
### Step 3.1: bidomain `boundary` validation (S1 — HIGH)
Validate `boundary` in `bidomain()` (bc resolution at api.py:1206; current mapping `=='bath'`→bath_coupled else insulated at 1217-1220) against `{'bath','insulated',None}`; raise `ValueError` otherwise (mirror LBM's check at `_lbm/simulation.py:85-87`). This is the exact current mapped set (`data.boundary` only ever holds `'insulated'`/`'bath'`, file_format.py:78) so it does NOT amputate a reachable mode. **NOTE**: `BoundarySpec.bath_coupled_edges` (mesh/boundary.py:132) is a real mode the factory does not expose today — record it as a **Deferred exposure item** (do not pretend the vocabulary is complete; the boundary_conduction_speedup Kleber work may want it).
- Test `test_bidomain_rejects_bad_boundary`: `pytest.raises(ValueError)` for `bidomain(mesh, boundary='ncs')` and `'insualted'` (typo); `bidomain(mesh, boundary='bath')` still yields a Dirichlet-phi_e (bath) spec.
### Step 3.2: LBM masked grid — UNION hole + outer walls (I1 — HIGH)
In `lbm()`, when `data.mask` is not all-True (`not data.mask.all()`): compute `hole = precompute_bounce_masks(torch.tensor(data.mask, device=…), lattice_obj)` (masks.py:10 — periodic `torch.roll`, so it flags ONLY the interior hole rim, NOT outer walls) **UNION** the rectangular outer-edge masks (`_make_rect_masks`-equivalent) → `bounce_masks[a] = hole[a] | rect_edge[a]`. **PREFER the "pad `data.mask` with a False border before rolling" variant** — it needs no lattice object and both lattices' cardinal indices coincide anyway; if instead you build a lattice object, `from cardiac_core._lbm.lattice import D2Q9, D2Q5` and use the one matching the branch's sim lattice (D2Q9 keys 5-8 are harmlessly ignored by a d2q5 neumann, but match to be safe). **Pass `bounce_masks=` to BOTH `LBMSimulation` construction sites** — the anisotropic/MRT branch (api.py:1416) AND the isotropic branch (api.py:1425); a hole in an anisotropic mesh must bounce too. Guard on `not data.mask.all()` so the all-True default rect path keeps using `_make_rect_masks` unchanged (golden-safe).
- Test `test_lbm_masked_hole_nonconducting`: `lbm(mesh_with_circular_hole)` (isotropic) AND `lbm(mesh_with_circular_hole, D_yy=…)` (anisotropic/MRT) → V inside the hole stays ~V_rest after a run **AND** V at the outer wall stays bounded/rest (the outer-wall assertion catches the periodic-roll trap; the aniso case guards the MRT construction branch).
### Step 3.3: `with_()`/factory immutability via `_resolve_mesh` (I2 — HIGH)
Root cause: `_resolve_mesh` returns the caller's `CardiacMeshData` by reference (api.py:816-820), so `with_` (api.py:254), `reset` (243), and even the FACTORY (`monodomain(m)` stores the caller's `m`) all alias it; `stimulate` then `self._data.stimuli.append(...)` (357) mutates the shared object. **Fix in ONE place**: in `_resolve_mesh`'s `CardiacMeshData` branch, `return copy.deepcopy(mesh)` (import `copy`). Device-safe: `_data` holds only numpy arrays + scalars + a `stimuli` list of dicts with numpy masks — NO torch/CUDA tensors and NO ionic instance (the instance lives in `_build_kwargs`, so it is still shared/preserved through `reset`). Leave the `str`-path branch (loads fresh from disk) unchanged. **Golden-safety**: `canonical_sim` (tests/_integrity/make_goldens.py) — if it builds from a path, the str-branch is untouched; if from an in-memory `CardiacMeshData`, the deepcopy runs but `deepcopy` of float64 numpy arrays is bit-identical → confirm via `test_integrity.py`. Name which branch it takes when implementing.
- Test `test_with_immutable_stimuli`: `p=monodomain(m); c=p.with_(dt=0.02); c.stimulate(region); assert len(p._data.stimuli)==len(m.stimuli)` (parent + original untouched).
- Test `test_factory_mesh_not_aliased`: `a=monodomain(m); b=monodomain(m); a.stimulate(region); assert len(b._data.stimuli)==len(m.stimuli)` (two sims from one mesh don't cross-contaminate; caller's `m` untouched).
### Step 3.4: silent-degrade → warn (S2, S3, C4) — three NAMED tests
(Add `import warnings` to `api.py` — it currently has none. Every warning message must contain a stable substring the test pins with `match=`.)
- S2 `test_lbm_alpha_inert_warns`: `lbm()` warns (`UserWarning`, message containing `'alpha'`) when `alpha != 1.0` and `boundary in ('neumann','hbb')` (alpha inert). `pytest.warns(UserWarning, match='alpha')`.
- S3 `test_bidomain_sigma_ratio_ignored_warns`: `bidomain()` warns (message matching `'sigma_ratio'`) when `sigma_ratio` is non-default AND `sigma_i/sigma_e` (or a conductivity carrying them) are present (the sigma branch wins → ratio ignored). Two-case test: fires when both present (`pytest.warns(UserWarning, match='sigma_ratio')`); does NOT fire when only `sigma_ratio` given (`with warnings.catch_warnings(record=True) as w: warnings.simplefilter('always'); bidomain(..., sigma_ratio=5.0); assert not any('sigma_ratio' in str(x.message) for x in w)`).
- C4 `test_lbm_lattice_override_warns`: `lbm()` warns (message matching `'lattice'`) when an explicit non-default `lattice` is overridden to d2q9/mrt by anisotropy (api.py:1412-1432). `pytest.warns(UserWarning, match='lattice')`.
### Step 3.5: result dtype round-trip (I3)
`io.load_result` (io.py:67, returns the 4-tuple `(times, V, phi_e, metadata)`) currently hardcodes `dtype=torch.float64` for `times`/`V`/`phi_e` (lines 94/95/99). Read the stored dtype for `V` and `phi_e` independently — prefer `V = torch.from_numpy(f['V']).to(dev)` (preserves the numpy dtype), `phi_e = torch.from_numpy(f['phi_e']).to(dev)` when present. Leave `times` float64 (analysis assumes it; the audit I3 names only `Vm`).
- Test `test_load_result_dtype_roundtrip`: `save_result(path, times, Vm=Vm32)` with a float32 `Vm`; `t, V, phi, meta = load_result(path); assert V.dtype == torch.float32`. (Note: `load_result` is a **path-based 4-tuple**, not `load_result(sim).Vm`.)
### Phase 3 Verification / Exit
`… -m pytest cardiac_core/tests cardiac_core/tests/test_integrity.py -q` — all green; each of the SEVEN findings has a named test. **Prove "fails pre-fix" mechanically** (the post-mortem's core point — "17 green tests proved nothing because none was shown to fail on the broken code"): for each, `git stash` the fix, run the new test → confirm RED, `git stash pop`. The Phase-3 named tests and their Phase-0 CONTRACT cells assert the same finding — have the cell delegate to (call) the named test, or keep the named test canonical and let the cell reference it, to avoid drift. **→ commit after Phase 3.**

---

## Phase 4: Capability exposure + consistency (C3, C5, C6, S4, C2)
**Goal**: expose the engine capability the factories hide, and unify cross-engine registries. **Tier**: large.
### Phase Context
These are additive (new kwargs / registry entries) — default behavior unchanged (run `test_integrity.py`). C2 (oblique LBM) is a documented limitation, NOT a ship target — see 4.4.
### Step 4.1: unify the ionic registry — ONE builder that branches on ctor capability (C3 — HIGH)
There is **no** single map today: three separate resolvers — LBM inline `model_map` (api.py:1379-1385, builds `cls(device=…)`), mono `_build_ionic_model(name, cell_type, device)` (`_monodomain/…/monodomain.py:54`, forwards `cell_type=`), bidomain `_resolve_ionic_model(name, device)` (`_bidomain/…/bidomain.py:126`). **TTP06/ORd ctors take `cell_type`; PHAS13Model/MHAS13Model ctors take `device` ONLY** (`ionic/phas13/model.py:60`, `ionic/mhas13/model.py:60`) — so "just extend `_build_ionic_model`" would call `PHAS13Model(cell_type=…)` → `TypeError`.
- NEW `cardiac_core/ionic/registry.py::build_ionic_model(name, cell_type='ENDO', device='cuda')`: `if isinstance(name, IonicModel): return name`; map `name.lower()` → class; **forward `cell_type` ONLY to TTP06/ORd** (converting the string to the enum via `getattr(CellType, cell_type.upper())` — the ctors take a `CellType` enum, per the existing mono resolver at `_monodomain/…/monodomain.py:79`), call `cls(device=…)` for PHAS13/MHAS13/paci. **Default `cell_type='ENDO'`, NOT 'EPI'** — every current engine default is ENDO (TTP06/ORd ctors default `CellType.ENDO`; bidomain/LBM pass no cell_type; mono derives it from `data.group_cell_types` else `'ENDO'`, api.py:1134). An 'EPI' default would flip the bidomain + LBM integrity goldens. `'paci'→PHAS13Model` is a genuine same-class alias (`ionic/__init__.py:21 from .phas13 import PHAS13Model as PaciModel`), so propagating it is correct/benign (resolves C8).
- Rewire mono `_build_ionic_model`, bidomain `_resolve_ionic_model`, and the lbm `model_map` to delegate to `build_ionic_model` (mono forwards its mesh-derived `cell_type`; **bidomain + LBM delegate WITHOUT `cell_type` → ENDO default, preserving current behavior**; `test_integrity.py` bidomain+lbm goldens are the guard). The only callers of the three resolvers are internal (verified) — no other consumer breaks.
- Test `test_ionic_registry_all_engines`: for `name in {'ttp06','ord','phas13','mhas13','paci'}`, `monodomain(ionic_model=name)`, `bidomain(ionic_model=name)`, `lbm(ionic_model=name)` all construct + `snapshots(2,2)` finite (this WOULD catch the `cell_type` TypeError). NOTE: PHAS13/MHAS13 on bidomain/LBM is **newly-enabled** capability (they were ttp06/ord-only) — the bidomain state/solver are model-agnostic (IonicModel ABC), so it should work; if a 17-state model surfaces an incompatibility, treat it as a real Phase-4 DISCOVERY, not a plan defect.
### Step 4.2: surface `weights_mode` + MRT knobs on `lbm()` (C5)
Add `weights_mode='canonical'` (+ optionally explicit `collision`/MRT moment-rate knobs) to `lbm()`; forward to `LBMSimulation`; add to build_kwargs. Enables the `uniform_8` connectivity column.
- Test `test_lbm_weights_mode_exposed`: `lbm(lattice='d2q9', weights_mode='uniform_8')._engine.lattice` is the uniform-8 D2Q9 variant.
### Step 4.3: surface `stencil` (C6) + bidomain `splitting` (S4)
- C6: add `stencil=` (mono: cardinal4/moore8_uniform/moore8_iso; bidomain: 5pt/mehrstellen), forward to the discretization ctor, add to build_kwargs. Consider surfacing mono `boundary_mode` (the FDM ghost choice, the mono analog of the LBM wall modes — the boundary_conduction_speedup work wants it) — if cheap, do it; else record as a Deferred exposure item.
- S4 (promoted from Deferred — same "capability exists but isn't exposed" class as C5/C6): expose `splitting=` on `bidomain()` (the downstream `BidomainSimulation(splitting=…)` already supports it — `_bidomain/…/bidomain.py:47`; the factory currently hides it → bare TypeError). Forward + add to build_kwargs. For the genuinely single-engine knobs (`theta` bidomain-only; `diffusion_solver`/`linear_solver` mono-only): the factories have NO `**kwargs` (audit line 39), so `monodomain(theta=…)` raises a bare `TypeError` at call-binding today. **Mechanism = add-and-reject**: add the cross-engine knob to the signature as `theta: float | None = None` (etc.) and `raise ValueError("theta is bidomain-only")` when it is set on the wrong engine — do NOT rely on a catch-all `**kwargs`. This is the S4 contract-matrix invariant ("knob valid on A raises *with a message* on B").
- Tests: `test_monodomain_stencil_exposed` (`monodomain(stencil='moore8_iso')` routes to the Moore-8 builder); `test_bidomain_splitting_exposed` (`bidomain(splitting='godunov')` constructs + runs); `test_solver_knob_engine_mismatch` (`monodomain(theta=…)` and `bidomain(diffusion_solver=…)` → `ValueError` with the engine named).
### Step 4.4 (decision point): oblique LBM (C2) — DEFAULT = documented limitation
**Reality check (do not skip):** oblique (`Dxy≠0`) is a **real numerics gap, not a wiring gap**. `mrt_collide_d2q9` (`_lbm/collision/mrt/d2q9.py:57-69`) carries `s_pxy` as a *free stability rate* with `p_xy_eq=0` — its docstring states "D_xy is NOT applied … needs the moment-space rotation of s_jx/s_jy — not implemented (Audit #46)." `tau_tensor_from_D`/`check_stability_tensor` compute `tau_xy`, but the collision kernel **discards** it. So threading `D_xy` through `LBMSimulation` (stop hardcoding `D_xy=0` at simulation.py:144,150) + dropping the factory `raise` (api.py:1402) would run an UNROTATED collision → a **silent-wrong** anisotropy (exactly the bug class this plan kills).
- **DEFAULT decision: DO NOT ship oblique.** Keep the `lbm()` oblique `raise`, but downgrade its message to a clear, documented "not-yet-wired: oblique fibers need moment-space rotation of s_jx/s_jy (Audit #46)". This makes C7 (mono/bidomain drop `Dxy` at the wall — interior-correct, boundary-truncated) **consistent**: all three engines decline full oblique fidelity, so there is no silent per-engine divergence. Document C7's disposition here (paired to C2). Optionally add a one-line `UserWarning` at the mono/bidomain oblique-D wall path ("Dxy truncated at boundary — interior-correct only"); if not done, record as a Deferred item.
- **ONLY IF** someone implements the Audit #46 moment-space rotation in a SEPARATE task: ship requires BOTH (a) `check_stability_tensor` passes AND (b) an **analytic fiber-direction CV check** — CV must be *fastest along* a π/4 fiber (a stable+plausible-CV run is NOT sufficient proof; unrotated collision can be stable and wrong). At that point DELETE `test_lbm_rejects_oblique_Dxy` and add `test_lbm_oblique_runs` (CV fastest along fiber). This is out of scope for this plan.
### Phase 4 Verification / Exit
`… -m pytest cardiac_core/tests cardiac_core/tests/test_integrity.py -q`; goldens bit-identical (additive kwargs, defaults unchanged). **→ commit after Phase 4.**

---

## Phase 5: Contract-matrix stress harness — flip xfails to live asserts (the keystone)
**Goal**: turn the Phase-0 `CONTRACT` table into live assertions now that Phases 1–4 have landed; add the cross-engine invariants; leave deliberately-deferred cells as explicit `xfail`. **Tier**: medium.
### Step 5.1: activate `test_api_contract.py`
**Model**: opus
#### Implementation Spec
- The `strict=True` on every `to_fix` cell has ALREADY forced its marker-removal in the phase that landed the fix (an unremoved marker XPASS-fails the suite). Phase 5 confirms zero stray `to_fix` xfails remain and that each is now a live `effect`/`raise(match=)`/`warn(match=)` assertion.
- Cells that remain deferred keep `xfail(strict=False)`: the **C2-capability** oblique-CV-correct cell (`xfail(reason='C2/Audit#46')`, body never runs) and **C7** mono/bidomain oblique-wall-truncation (`xfail(reason='C7')`, the divergence-declined). The **C2-raise** documented-limitation cell is NOT deferred — Phase 2's message reword already flipped it `to_fix→landed`, so it is a live `raise(match='oblique|Audit #46')` regression-lock assert here. Any C5/C6/S4 sub-cell deliberately not exposed is `deferred`.
- Assert the cross-engine invariants from Phase 0: `run_X ≡ simulate(engine=X) ≡ X()` (bit-identical where deterministic); the "valid-on-A-raises-with-message-on-B" cells (S1, S4); `with_`/factory leaves `_data` untouched (I2).
- float32/cuda cells: `pytest.mark.skipif(not torch.cuda.is_available())` for cuda; float32 cells run on CPU. Keep grids tiny (`_tiny`), `t_end`≤2 ms except wall-divergence cells.
#### Test Spec
- The suite IS the test. Verify: ≥1 live asserted cell per HIGH+MED finding (**15 findings**, incl. C7); every `raise`/`warn` cell pins `match=`; every deferred cell (C2-raise message, C2-capability, C7) is `xfail`/documented with a finding-ID reason (never a silent gap). Each finding ID appears in a test id (mechanically, via the param `id=`).
#### Verify
`… -m pytest cardiac_core/tests/test_api_contract.py -q` + full suite `cardiac_core/tests -q`.
#### Exit Criteria
- [ ] Every HIGH/MED finding has a live cell that would fail pre-fix and passes post-fix; deferred cells are `xfail(reason)`; the harness is green.
**→ commit after Phase 5.**

---

## Final Cleanup
- Update API_CONSISTENCY_AUDIT.md + PHASE3_BOUNDARY_GAPS.md "status" → resolved (with commits + which findings shipped vs deferred; note C2 stays a documented limitation pending Audit #46).
- Update KNOWLEDGE/IDEALOG: the API is now contract-tested across `{entry × engine × param × physics}`; note deferred cells (oblique LBM / Audit #46, C7 pairing, `bath_coupled_edges` exposure, mono `boundary_mode`, the low/latent C9/I4/I5/I6/S5/P2).
- Archive: `cp PLAN.md plans/$(date +%Y-%m-%d)_api-consistency-hardening-stress-harness.md`
- **Rollback rule (per phase)**: if a phase changes an integrity golden, the phase is WRONG — revert and re-approach; do NOT update the golden. Goldens are the frozen-behavior contract.

## Deferred / not in this plan (explicit)
Tracked in API_CONSISTENCY_AUDIT.md. **Promoted OUT of this list by the round-1 audit**: S4 → Step 4.3 (splitting exposure + validated engine-mismatch errors); C8 → Step 4.1 (paci alias, benign same-class). **Genuinely deferred (each a one-liner unless noted):**
- **C2 / Audit #46** (oblique LBM moment-space rotation) — NOT a one-liner; a real numerics task, its own future plan. Left a documented, message-bearing `raise` (Step 4.4).
- **C7** (mono/bidomain boundary-Dxy truncation) — dispositioned *paired to C2* in Step 4.4 (consistent decline); optional mono/bidomain warn is a one-liner if wanted.
- **`bath_coupled_edges` exposure** (bidomain) — real BoundarySpec mode the factory doesn't surface (noted in Step 3.1); Kleber work may want it.
- **mono `boundary_mode`** exposure (Step 4.3, if not done there).
- S5 (BGK CFL/τ guard), C9 (bidomain `getattr(state,'Cm',1.0)` → fail-loud), I4 (persist ionic_states), I5 (delete dead `mesh/loader.py`), I6 (np.str_ group labels), P2 (`record=` in one-shots) — each a one-liner; batch or fold into the relevant phase as a quick win.

## Mutation Log
_(populated during execution: `**MUTATED {date}**: Step X.Y {SKIPPED|SPLIT|INSERTED} — {reason}`)_

**REVISED 2026-07-01 (audit round 1 — 5 blockers / 10 majors folded in):**
- **B1/Step 4.4 (C2 oblique):** rewrote from "thread D_xy → ship if stable" to "DEFAULT documented limitation." `mrt_collide_d2q9` discards `D_xy` (docstring d2q9.py:63-69, `p_xy_eq=0`) → threading it is a silent-wrong no-op; oblique is real numerics (moment-space rotation, Audit #46), a separate task. Ship (if ever) requires an analytic fiber-direction CV check, not just stability. Corrected the "MRT is oblique-CAPABLE" lesson.
- **B2/Step 3.2 (I1 masked grid):** `precompute_bounce_masks` uses periodic `torch.roll` → NO outer-wall bounce on a full-border mask (LBM V1 DESIGN_AND_CHANGES.md:100). Changed "swap to precompute_bounce_masks" → "UNION hole rim with rect outer edges (or pad mask)"; test now asserts the OUTER wall too; specified `lattice_obj` sourcing; guard on `not data.mask.all()`.
- **B3/Step 4.1 (C3 ionic registry):** PHAS13/MHAS13 ctors are `device`-only (model.py:60); the old "extend `_build_ionic_model`" (which forwards `cell_type=`) would `TypeError`. Replaced with a NEW shared `ionic/registry.py::build_ionic_model` that branches on ctor capability; all three resolvers delegate. Confirmed `paci`=PHAS13Model same-class alias.
- **B4/Step 3.5 (I3 dtype):** the test `load_result(p).Vm` was unrunnable — `load_result(path)` returns a 4-tuple. Rewrote to the tuple API; read `f['V']`/`f['phi_e']` dtype via `torch.from_numpy`; leave `times` float64.
- **B5/Step 2.1 + 4.4 (oblique-reject test):** removed the Phase-2→4.4 contradiction. The oblique `raise` legitimately STAYS (real limitation), but its test now asserts the **documented-limitation message** and carries an explicit "replace-if-lifted, do-not-cargo-cult-the-guard" note.
- **M1/Step 4.3 (S4):** promoted from Deferred — bidomain `splitting` exposure + convert bare-TypeError engine-mismatches to validated ValueErrors; added contract-matrix cell.
- **M2/Step 4.4 (C7):** paired C7's disposition to C2 (both decline oblique → no silent divergence) instead of a bare "doc" deferral.
- **M3/Phase 3 (tests):** split the inline one-liners into SEVEN named per-finding Test Specs; corrected the "5 items" miscount to 7.
- **M4/Step 3.1 (S1):** kept the validated set = current mapped `{bath,insulated,None}` but recorded `bath_coupled_edges` as a Deferred exposure item (don't pretend the vocabulary is complete).
- **M5/Step 2.1 (mrt_wall signature):** spelled out the full 7-moment-rate signature; warned against cloning the `omega`-based `bgk_wall` signature (`self.omega is None` under MRT).
- **M6/Step 3.3 (I2):** moved the deep-copy from `with_`-only to `_resolve_mesh` (closes factory + with_ + reset aliasing); added the two-sims-from-one-mesh test; noted device-safety.
- **M7/Step 1.1:** pin `dt=0.005` (default `dt=0.02` collapses the hbb-vs-scs margin to ~1 mV); assert `abs().max() > 1e-2`; corrected the "stim near wall" wording (walls are top/bottom).
- **M8/Phase 0 (NEW):** added a contract-matrix-FIRST phase — write `CONTRACT` as data with expected verdicts + finding IDs, unfixed cells `xfail`, BEFORE Phase 1. Split the old Phase 5 into 0 (table) + 5 (activate). This is the central process fix for the post-mortem's root cause.
- **M9/Phase 0+5:** enumerated the required cross-engine RAISE/WARN cells explicitly (bidomain-boundary, lbm-boundary regression-lock, mrt+uniform_8, alpha-inert, sigma_ratio, lattice-override, solver-knob mismatch).
- **M10/Step 1.1:** `simulate ≡ run_lbm` now pins identical lattice/boundary/alpha/dt/device and uses `torch.equal` (LBM RNG-free), not `allclose`.
- **Minors:** added `test_integrity.py` to Phase 3 verify + golden-safety notes to Steps 2.1/3.2/3.3; noted no `conftest.py` exists (local `_tiny` helper); mass-drift threshold `<1e-8`; per-phase rollback rule.

**REVISED 2026-07-01 (audit round 2 — 1 blocker / 5 majors folded in):**
- **BLOCKER/Step 4.1 (ionic default):** shared `build_ionic_model` default was `'EPI'` → would flip bidomain + LBM integrity goldens (whole codebase defaults ENDO: TTP06/ORd ctors, bidomain/LBM pass none, mono derives from mesh). Changed default to `'ENDO'`; bidomain/LBM delegate w/o cell_type; `test_integrity` is the guard. (verified: ORd ctor `CellType.ENDO`, mono api.py:1134.)
- **MAJOR/Step 2.1 (mrt_wall signature):** the explicit signature put `w` after the moment rates while the dispatch text said "same positional args as mrt" — contradiction that would bind `self.w` into the `s_e` slot. Reconciled to mirror `lbm_step_d2q9_mrt` exactly (`w` 5th, right after `dt`) across Architecture + Step 2.1 signature + dispatch call. (verified against step.py:63-69.)
- **MAJOR/Step 3.2 (I1 both branches):** `bounce_masks=` must reach BOTH LBM construction sites (aniso/MRT api.py:1416 AND iso api.py:1425), else a masked anisotropic grid still leaks; added an aniso-hole test case. Also: prefer the pad-mask variant (no lattice object needed) / named the `D2Q9`/`D2Q5` import.
- **MAJOR/Phase 0+3.4 (warn cells spurious):** `pytest.warns(UserWarning)` passes on ANY UserWarning; added a `match` field to the CONTRACT tuple and `match=` on every `raise`/`warn` cell + `import warnings` in api.py + S3 `simplefilter('always')` two-case.
- **MAJOR/Phase 0+5 (xfail rot):** `strict=False` never forces the Phase-5 flip → matrix could stay green as permanent-limitation xfails (anti-pattern one level up). Split cells `to_fix` (`strict=True` → XPASS fails the suite, forcing the in-phase flip) vs `deferred` (`strict=False`); added the `pytest.param(marks=…)` recipe + `status` field.
- **MAJOR/Phase 0+5 (C7 enumeration):** C7 (MED) had a Phase-5 cell but was dropped from the Phase-0 list and the "14 findings" count — added C7 to the enumeration (→15) and pre-declared C2's two rows in Phase 0 so Phase 5 only removes markers.
- **Minors:** execution-order heading corrected ("strictly ascending 0→5"); parity test pins `t_end`/`save_every`; Step 3.3 names the goldens' mesh branch; S4 mechanism spelled out (add-and-reject, no `**kwargs`); Phase-3 "git stash → confirm RED" pre-fix gate + matrix/named-test de-dup note.

**REVISED 2026-07-01 (audit round 3 — 1 blocker / 1 major folded in; both in the Phase-0 harness mechanism):**
- **BLOCKER/Step 0.1 (`c.exc` undefined):** the `raise` dispatch used `pytest.raises(c.exc, …)` but `exc` was not a `Cell` field → `AttributeError` for every raise cell. Added `exc` as the tuple's LAST field with `defaults=[ValueError]` (namedtuple right-aligns defaults, so a mid-tuple defaulted field is invalid — corrected in round 4).
- **MAJOR/Step 0.1+Phase 5 (C2-raise had no representable status):** `status ∈ {to_fix,deferred}` both forced an xfail marker, but the documented-limitation `raise` PASSES after Phase 2 → would read XPASS, not a live assert. Added a third `status='landed'` (no marker → live regression-lock assert); made `C2-raise` a `to_fix` cell that Phase 2's message-reword flips `to_fix→landed`; declared the already-passing `lbm(boundary='bath')→raise` as `landed`. Phase 5 prose reconciled.
- **Minors (round-3 correctness lens):** Step 2.1 dispatch `SPECIAL_WALL` → `D2Q9_ONLY` (the real, imported symbol — `SPECIAL_WALL` does not exist); Step 4.1 registry adds the `getattr(CellType, cell_type.upper())` string→enum conversion (ctors take the enum); noted PHAS13/MHAS13-on-bidomain/LBM is newly-enabled capability (a Phase-4 failure = discovery, not plan defect).

**REVISED 2026-07-01 (audit round 4 — 1 blocker, mechanical):**
- **BLOCKER/Step 0.1 (namedtuple default ordering):** `Cell(…, exc, finding_id, status, note)` with `defaults=[ValueError]` binds the default to `note`, not `exc` (Python right-aligns namedtuple defaults) → import error or silently shifted rows. Moved `exc` to the LAST field: `Cell(…, finding_id, status, note, exc)`. Field refs are name-based (`c.exc`), so no other change. (Also added `c.param` to the pytest `id=` template for unique, self-documenting ids.)

**Audit trajectory:** R1 5 blk / 10 maj → R2 1 blk / 5 maj → R3 1 blk / 1 maj (correctness lens 0/0) → R4 1 blk / 0 maj (mechanical namedtuple reorder, fix self-verified) → **CONVERGED**. All findings grounded in code; blocker fixes verified against source.
