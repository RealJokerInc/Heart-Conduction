# cardiac_core + cardiac_mcp + docs — Adversarial System Audit

> **Backlinks:** [engine_consolidation README](./README.md) · [KNOWLEDGE](./KNOWLEDGE.md) ·
> [MASTER.md](../../../MASTER.md) · companion: [boundary_conduction_speedup/BC_IMPLEMENTATION_AUDIT.md](../boundary_conduction_speedup/BC_IMPLEMENTATION_AUDIT.md)
>
> **Created:** 2026-06-30. **Type:** adversarial multi-agent audit (find → refute → completeness).
> **Method:** 8 lanes fanned out over `cardiac_core/` + `cardiac_mcp/` + the API docs; every finding
> was handed to an independent skeptic that reopened the cited code and tried to refute it before it
> was allowed in; a completeness critic then hunted for un-audited surface. 64 agents, ~3.1M tokens.
> **Result:** 55 raised → **46 confirmed**, 9 rejected, 12 completeness gaps.
> Purpose: tidy the foundation **before** the boundary-mode implementation.

## Severity tally (post-verification)
`4 HIGH · 11 MEDIUM · 27 LOW · 4 INFO`. By lane: api_surface 9, loose_ends 8, tests_health 7,
docs_drift 6, conductivity_chi 5, numerics 5, mcp 4, engine_integrity 2.

---

## Executive summary — the systemic clusters

Most of the 46 collapse into **five root themes**:

1. **The file-format chi/D ambiguity (the firewall has a hole).** `CardiacMeshData` stores a raw
   `D_xx` field *and* a `chi`, but the three engines disagree on whether `chi` applies to `D_xx`:
   monodomain treats it as raw (effective = `D/(χ·Cm)`), LBM/bidomain treat `D_xx` as already
   effective. So **the same mesh runs at ~1400× different diffusivity depending on engine**, and the
   default `create_cardiac_mesh(D=0.001, chi=1400)` yields a **conduction-blocked** monodomain mesh
   with no guard. Findings **#2, #8, #21** (+ rejected "canonical goldens never test propagation").
   This is the `create_cardiac_mesh` API-debt from the earlier session, now shown to be a *file-format
   layer* hole in the Formulation-A/B firewall, not just one builder.

2. **A real numerics bug: monodomain anisotropic cross-derivative is wrong (#4, HIGH).** In
   `_build_laplacian_cardinal`, the Dxy term uses `cxy = 1/(4·dx·dy)` (should be `1/(2·dx·dy)` — the
   factor of 2 from `2·Dxy·V_xy` is dropped → **half magnitude**) **and** the four diagonal signs are
   the negatives of the validated bidomain builder (**wrong sign**). Oblique-fiber monodomain runs are
   quantitatively wrong, and it's **untested** (#41). Directly relevant to the anisotropic-boundary
   work. (Distinct from #14, the *boundary* Dxy drop, which my earlier BC audit flagged.)

3. **Replay/one-shot API paths diverge from the eager path** — silent-correctness + crash surface:
   `reset()/stimulate()` drop an `ionic_model=` override (#1, HIGH); one-shot `run_*/simulate()`
   crash on a zero-save run where eager `run()` degrades gracefully (#5); `record=`/`ionic_states`,
   save-cadence, and termination epsilon differ between `run()` and `run_*()` (#18, #19).

4. **Advertised-but-unimplemented surface.** ~21 documented public `CardiacSimulation` methods raise
   `NotImplementedError` *with worked `>>>` examples* (#7, #12); `pcg_gmg` silently downgrades to plain
   PCG (#13); FEM/TriangularMesh is declared "dropped" but fully present + exported (#28); orphaned LBM
   dirichlet/absorbing kernels + dead bidomain `fft.py` + dead D2Q5-MRT (#29, #30, #37, #44).

5. **Docs describe an aspirational API, not the shipped one** (#9, #10, #11, #24, #25, #43) — import
   lines that `ImportError`, factory kwargs that `TypeError`, `result.cv()` shown arg-free but
   requiring indices, stale test counts / "rename pending" that already shipped.

MCP is in reasonable shape for its stated **stdio-local** trust model: the keyless token and
`confirmed=True` footgun were **downgraded to LOW** (single trust domain), but the
**`commit_experiment` path traversal stayed HIGH** (#3) — unsanitized `date/slug` from the token,
no `is_relative_to(LAB)` guard, unlike `run_experiment`.

---

## HIGH (4) — fix first

| # | Finding | Location | Fix |
|---|---------|----------|-----|
| 1 | `reset()/stimulate()` silently revert an `ionic_model=` override to the mesh default (build_kwargs never stores it; `with_()` works — inconsistent) | `cardiac_core/api.py:1132/1287/1416` + reset 231-237 + stimulate 337-352 | set `data.ionic_model = ionic` after resolving, or add `ionic_model=ionic` to every factory's `build_kwargs` |
| 2 | `create_cardiac_mesh` default `(D=0.001, chi=1400)` → effective D = 7.1e-7 (~1400× low) → conduction-blocked mesh, no runtime guard | `cardiac_core/file_format.py:175,179,245-287` | warn when `D/(χ·Cm)` leaves a physiological band; resolve the file-format D meaning (cluster #1) |
| 3 | `commit_experiment` path traversal — `date/slug` from the (keyless, forgeable) token used unsanitized in the folder path, no `is_relative_to(LAB)` guard (unlike `run_experiment`) | `cardiac_mcp/core.py:385-396` | re-validate `date` regex + `_slugify(slug)` from the token; add `d.resolve().is_relative_to(LAB)` before `mkdir` |
| 4 | **Monodomain FDM cross-derivative has WRONG SIGN and HALF MAGNITUDE** vs the validated bidomain builder → oblique-fiber runs quantitatively wrong; untested | `cardiac_core/_monodomain/…/discretization_scheme/fdm.py:558,673-695` | `cxy = 1/(2·dx·dy)`; flip diagonal signs to match bidomain (NE `+d_xy·cxy`, NW/SE `−`, SW `+`); add a `V=x·y` anisotropic regression test to confirm empirically |

**#4 is the standout** — an actual physics error in the anisotropic operator, on the exact code path
the fiber-angle / anisotropic-boundary work will exercise. It should be confirmed with a targeted
regression (the fix's own test) and fixed before any anisotropic boundary experiments.

---

## MEDIUM (11)

| # | Finding | Location |
|---|---------|----------|
| 5 | One-shot `run_*/simulate()` crash (IndexError) on a run with zero save-points; eager `run()` guards it | `cardiac_core/run.py:84` (`_collect`) |
| 6 | Ionic-model support asymmetric: `lbm()` accepts paci/phas13/mhas13; `monodomain()/bidomain()` raise ValueError | `api.py:1346-1358` vs `_monodomain …/monodomain.py:83-88` |
| 7 | 21 documented `CardiacSimulation` methods raise `NotImplementedError` but carry worked `>>>` examples (probes/clamps/drug-block/on-object analysis) | `api.py:282-653` |
| 8 | LBM factory drops `data.chi` — no assertion that stored `chi` is consistent with `D_xx` being effective (cluster #1) | `api.py:1375-1395` |
| 9 | API_REFERENCE Quickstart import line `ImportError`s (`TTP06Model/CellType/StimulusProtocol/left_edge_region` not importable from `cardiac_core`) | `engine_consolidation/API_REFERENCE.md:26-33` |
| 10 | Documented factory kwargs (`Cm`/`ionic_solver`/`splitting`/`parabolic_solver`) don't exist on shipped factories → `TypeError` | `API_REFERENCE.md:222-267` |
| 11 | `SimulationResult.cv()` documented arg-free but requires positional `x1,x2,y` | `API_REFERENCE.md:297-349` |
| 12 | (dup-cluster of #7 from loose_ends lane) ~21 bare-stub public methods on the concrete wrapper class | `api.py:282-697` |
| 13 | `linear_solver='pcg_gmg'` silently downgrades to plain PCG (GMG unimplemented; multigrid.py/pcg_gmg.py are dead stubs) | `_bidomain/…/bidomain.py:195-198` |
| 14 | Boundary Dxy cross-derivative silently dropped at wall/mask interfaces in both FDM builders (also in my BC audit §2.4/3.4) | `_monodomain/…/fdm.py:668-695` |
| 15 | `test_originals_untouched` is RED — LBM src-hash baseline stale after the Jun-30 anisotropy commit (`bf8aa74`) legitimately edited the original LBM engine | `tests/_integrity/engine_src_sha.json:4`; `test_integrity.py:45-53` |

Note #15: the full suite is **148 passed / 1 failed** — the one failure is this stale golden, not a
numerics regression. Decide the vendoring policy (editing originals in place is apparently allowed for
the anisotropy work) and refresh `make_goldens.py`.

---

## LOW (27) — grouped

**API footguns/consistency (#16–20):** LBM silently overrides `lattice=` for anisotropic D (#16);
positional mesh sniff clobbers explicit `mesh=` (#17); `record=`/ionic_states missing from one-shots
(#18); LBM uses a different save-cadence/epsilon than mono/bidomain (#19); stale `_prepare_engine`
comment in `__init__` (#20).

**Conductivity/firewall (#21–23):** the chi/D ambiguity itself (#21, cluster #1);
`ConductivityConfig.for_bidomain()` is dead — the bidomain factory converts σ→D inline, bypassing the
"single-source" emitter (#22); anisotropic bidomain unreachable via `ConductivityConfig.anisotropic()`
(no `sigma_i/sigma_e`) (#23).

**Docs drift (#24–27):** "Vm rename pending" already shipped (#24); "77 tests" vs actual ~149 (#25);
`_monodomain/__init__` docstring advertises a broken no-underscore import path (#26); stale
`_prepare_engine` comment (#27, dup of #20).

**Dead/unfinished code (#28–31, #44–45):** FEM/TriangularMesh present despite "dropped" (#28);
orphaned LBM dirichlet/absorbing kernels (#29, #37); deprecated wrong-result bidomain `fft.py` (#30);
LBM `get_activation_times` redirect-only stub (#31); unreachable D2Q5-MRT kernel (#44); resolved-but-
lingering MRT/weights_mode TODO (#45).

**Numerics (#35–36, #46):** FVM silently drops all Dxy (2-tuple vs 3-tuple D_field, but unreachable
from public API — no `scheme=`) (#35); Forward-Euler/RK CFL docstrings omit χ → example off by ~1400×
(too-*small* dt, safe-but-inefficient) (#36); MRT-D2Q9 claims "full anisotropic tensor" but ignores
Dxy (`s_pxy` is a free stability param) (#46).

**Test gaps (#37–42):** untested LBM dirichlet/absorbing (#37); no LBM specular/combined mode or test
(#38 — this is exactly what the boundary-mode build will add); bidomain insulated-vs-bath tests are
smoke-only, never assert the modes differ (#39); LBM `D_xy≠0` rejection untested (#40); oblique
anisotropy (D_xy≠0) mono/bidomain path has zero coverage (#41 — pairs with bug #4); golden-integrity
tests `skip` silently when goldens absent (#42).

**MCP (#32–34):** keyless SHA-256 token is forgeable (#32, LOW under stdio-local); `confirmed=True` is
model-settable — no proof a human reviewed (#33, LOW, doc/footgun); unescaped `goal` breaks the
NOTEBOOK.md table + status matcher (#34).

## INFO (4)
`for_bidomain()` documented as the factory emitter but never called (#43); dead D2Q5-MRT kernel (#44);
resolved MRT TODO (#45); MRT-D2Q9 "full tensor" docstring overclaim (#46).

---

## Rejected / not-a-bug (9) — checked and cleared
- `boundary=` honored only by `bidomain()`; mono/lbm ignore the mesh boundary field → **intended** (mono/lbm have no bath BC).
- Canonical goldens never test propagation → true but the blocked-default trap is a *test-gap*, folded into #2/#42, not a separate bug.
- LBM BGK has no `τ>0.5` guard → `check_stability` exists; not wired, but not a live defect at shipped params.
- LBM specular/combined absent from engine → **correct by design** (research-script-only; documented in BC audit).
- `CardiacSimulation.ionic_states` dead stub → duplicate of #7/#12 framing.
- `create_cardiac_mesh` can't make `D_xy≠0` → true but folded into #40/#41.
- `run_experiment` `openWorldHint=False` → **correct** per MCP spec semantics (openWorld ≠ "runs code").
- `run_experiment` provenance substring check → weak but intended for the local trust model.
- GLOSSARY/API_DESIGN cite `Engine_V5.5` → not stale (that IS the frozen source the vendoring came from).

---

## Completeness gaps (12) — un-audited surface (next-round targets)
Not bugs, but areas the 8 lanes did **not** cover; each is a concrete follow-up check:
1. `analysis.py` — 16 public science-output functions only smoke-tested (APD-at-mid-repol correctness, phase_map dtype/range).
2. `io.py` `save/load_result` round-trip fidelity (unconditional float64/device cast; metadata type loss).
3. **LBM masked/non-rectangular tissue** — `api` never passes `bounce_masks` from `data.mask`; a hole would leak (no interior-mask bounce).
4. `geometry.py` mask/distance convention correctness (0.5-cell offset, edge-mask width).
5. `stimulus/protocol.py` `get_current` flat-`(n_dof,)` vs `(Nx,Ny)` shape ambiguity.
6. `StructuredGrid` masked flat↔grid round-trip + `fill_value` (masked cells NaN vs spuriously "activated").
7. `cardiac_mcp/server.py` resources/prompts + annotation honesty (only `core.py` was audited).
8. **Device/dtype cross-cutting** — no confirmed check that `device='cuda'`/float32 stays consistent end-to-end.
9. Bidomain steppers (`explicit_rkc`, `imex_sbdf2`, `semi_implicit`, `decoupled_jacobi`) + solvers (chebyshev, multigrid, pcg_spectral) — opened for none.
10. `mesh/loader.py` (`MeshData`/`load_mesh`) — a **second parallel mesh loader** never audited; does its D get the same chi firewall?
11. one-shot save-cadence/time parity across all three engines (beyond the #5 IndexError).
12. paci/phas13/mhas13 never validated vs reference AP; is `'paci' → PHAS13Model` (api.py:1350) a **wrong alias**?

---

## Proposed fix ordering

**Before the boundary-mode work (foundation blockers):**
- **#4** (mono cross-derivative sign/magnitude) — corrupts the anisotropic operator the boundary work uses. Confirm w/ regression, fix.
- **cluster #1** (#2/#8/#21) — the file-format chi/D ambiguity + blocked default. At minimum land the guard (#2) and the LBM `chi==1` assertion (#8); ideally settle the `D_xx` = effective-D convention repo-wide.
- **#1** (ionic_model override lost on reset/stimulate) — silent wrong-model runs.
- **#3** (MCP path traversal) — the one live security hole.

**Fold into the boundary-mode build (they *are* the boundary surface):**
- #29/#37/#38 (orphaned/absent LBM BC modes, no bc-mode selector), #14 (boundary Dxy drop), #39 (bidomain bath-vs-insulated assert), #40 (D_xy reject test). Building the boundary-mode API resolves these.

**Foundation-tidy (batchable, not blocking):**
- #5 (zero-save guard), #6 (ionic registry parity), #7/#12 (mark/strip stub methods), #13 (pcg_gmg warn), #15 (refresh golden), #28/#30/#31/#44/#45 (delete/mark dead code), docs cluster #9/#10/#11/#24/#25/#26 (reconcile docs to shipped), #34 (escape NOTEBOOK goal).

**Next audit round:** the 12 completeness gaps — especially #3 (LBM masked grid), #8 (device/dtype), #10 (second mesh loader), #12 (paci alias).
