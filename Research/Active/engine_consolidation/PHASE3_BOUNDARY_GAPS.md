# Phase-3 Boundary-Mode Productization Gaps + Post-Mortem

> **Backlinks:** [PLAN.md](./PLAN.md) (the cleanup plan whose Phase 3 shipped these gaps) ·
> [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md) · [BC_IMPLEMENTATION_AUDIT.md](../boundary_conduction_speedup/BC_IMPLEMENTATION_AUDIT.md) ·
> [IDEALOG](./IDEALOG.md) · [MASTER.md](../../../MASTER.md)
>
> **Created:** 2026-07-01. Two real gaps in the Phase-3 LBM boundary-mode feature (commit `736296d`)
> found *after* shipping — by the user, not the tests. This doc records the gaps (for the fix
> blueprint) **and** the post-mortem on why the plan, the tests, and the 3-round convergence audit all
> missed them. The fix is blueprinted separately.

---

## Gap A — the one-shot `run_lbm()` does NOT forward `boundary`/`alpha` (still HBB-only)

**Symptom.** `run_lbm(mesh, boundary='ncs')` raises `TypeError` (unexpected kwarg); with no kwarg it
silently runs the default HBB wall. The boundary modes are reachable ONLY via the declarative
`lbm()` factory and via `simulate(**kwargs)` — **not** via the explicit one-shot `run_lbm`.

**Root cause.** The new params were threaded into the `lbm()` factory + its `build_kwargs`
(`api.py`), but `cardiac_core/run.py::run_lbm` (line ~195-235) is a *separate* thin wrapper with an
**explicit** signature that forwards only `ionic_model, dt, lattice, device` and then calls `lbm(...)`
with just those. `simulate()` happens to work because it uses `**kwargs`; `run_lbm` doesn't.
This is precisely the **one-shot-vs-factory asymmetry the audit already flagged as #18/#19** — and
which the plan *deferred*. Adding a new factory param re-opened the exact asymmetry.

**Fix approach.** Add `boundary='neumann', alpha=1.0` to `run_lbm`'s signature and forward them to
`lbm(...)`. (`run_monodomain` needs nothing — mono has no boundary mode yet; `run_bidomain` already
forwards its `boundary`=bath/insulated.) Add a test that drives a wall mode **through `run_lbm`**, and
one through `simulate(engine='lbm', boundary=…)`, so the one-shot surface is covered.

---

## Gap B — wall modes are BGK-only and BLOCK anisotropy (per-axis fibers + specular is impossible)

**Symptom.** `lbm(aniso_mesh, boundary='ncs')` raises `ValueError` — anisotropic D (`D_xx != D_yy`)
forces `collision='mrt'`, and `LBMSimulation` guards `boundary in D2Q9_ONLY and collision != 'bgk'`
→ raise. So **fiber-anisotropic diffusion + any specular wall mode is unreachable** — exactly the case
needed for fiber-parallel vs fiber-perpendicular boundary studies (the open "Anisotropic boundary
study" criterion in boundary_conduction_speedup).

**Root cause.** The wall overlay (`apply_wall_overlay`) runs **post-stream** — it reflects diagonal
populations *after* collision+streaming, so it is **collision-agnostic**. The `collision == 'bgk'`
requirement was over-conservative: it codified the research script's BGK-only scope as a hard guard
without checking whether the overlay actually depends on the collision operator (it doesn't). Rest-
neutrality still holds on MRT: at rest `f = feq`, and the canonical D2Q9 diagonal weights are matched
(`w5=w6=w7=w8=1/36`) regardless of the anisotropic relaxation rates `s_jx/s_jy`. Worse, a test
(`test_lbm_rejects_oblique_Dxy`) was written that **asserts the rejection**, conflating two different
things: *oblique* fibers (`D_xy != 0`, genuinely out of scope — needs moment-space rotation) with
*per-axis* anisotropy (`D_xx != D_yy, D_xy = 0`, which SHOULD work on MRT). The test validated the
limitation instead of questioning it.

**Fix approach.** (1) Add `lbm_step_d2q9_mrt_wall(...)` (mrt collide → clone → stream → neumann →
`apply_wall_overlay`). (2) Remove the `collision != 'bgk'` guard in `LBMSimulation.__init__`.
(3) `step()` dispatch: `collision=='mrt' && special-boundary → mrt_wall`; else `mrt`; else
`bgk_wall`; else `_step_fn`. (4) The `lbm()` anisotropic branch already passes `boundary/alpha`, so it
just works once the guard is gone. (5) KEEP the oblique (`D_xy != 0`) rejection — that's a real
limitation. Add tests: per-axis-anisotropic mesh + `boundary='ncs'` runs + is rest-neutral;
`combined(α)` selectable under anisotropy; oblique `D_xy != 0` still rejected (but reword the test so
it tests the *oblique* case, not "anisotropy" broadly).

---

## Post-mortem — why the plan, the tests, AND the 3-round audit all missed this

The deepest root, in one line: **the feature was tested and audited at the level it was BUILT (the
`lbm()` factory, isotropic BGK), not at the level a USER reaches it (`run_lbm`/`simulate`; anisotropic
fibers). The contract — which entry points × which physics must work — was never written down, so
nothing checked against it.** Concretely:

**1. The plan under-specified the feature's surface.**
- Phase 3.3 said "expose via `cardiac_core.lbm(boundary=…, alpha=…)`" — it named *one* entry point.
  The one-shot layer (`run_lbm`/`simulate`) was simply outside the sentence, even though the plan's own
  Findings Coverage had *deferred* #18/#19 (the one-shot-vs-class asymmetry) — a known-lagging surface
  that a new param would re-expose. The connection was never made.
- Phase 3 Context asserted "the specular/combined modes are D2Q9-only flat-wall overlays (**BGK**)" —
  it baked the research's BGK scope in as a *premise*. The design step (3.1) settled naming/registry/
  dispatch but never examined the **collision axis** (is the overlay collision-dependent? it isn't).
  Anisotropic + boundary was never listed as a case to support or reject.

**2. The tests matched the implementation surface, not the contract.**
- 3.4 exercised `lbm(boundary=…)` and the kernels directly — the *same* entry point the feature was
  added to. `run_lbm` was never driven. → an entry-point tested itself; the sibling entry point
  (`run_lbm`) went untested and shipped broken.
- All tests used **isotropic D + BGK**. The MRT (anisotropic) path was never combined with a boundary
  mode. And `test_lbm_rejects_oblique_Dxy` asserted a rejection that *codified* the over-restriction —
  giving false confidence that "anisotropy + boundary is intentionally unsupported."
- Net: 17 green tests, high confidence, two whole quadrants of the use-matrix (one-shot API;
  anisotropic physics) never entered.

**3. The 3-round convergence audit couldn't catch it — by construction.**
- The plan-audit checked the PLAN against the **46 existing findings** + cold-start executability +
  internal consistency. These two gaps are **NEW** (introduced by the Phase-3 *design*), not in the 46
  — so there was no oracle to check them against.
- The auditors verified "Phase 3.3 exposes via `lbm(boundary=)`" was *consistent*; they did not ask
  "does the feature cover the full public API (run_lbm/simulate) and the full physics matrix
  (isotropic/anisotropic)?" — because the plan under-specified it, and completeness-of-a-not-yet-written
  feature wasn't in the audit's scope. The completeness critic hunted un-audited *existing* code.
- **Lesson:** auditing a plan for internal correctness ≠ auditing a feature spec for surface coverage.
  A convergent adversarial audit hardens what's written; it can't catch what was never specified.

**4. The self-inflicted amplifier.** Writing a guard (`special → requires bgk`) + a test that asserts
the rejection turned an *unexamined assumption* into a *documented "feature."* Guards and their tests
are how a wrong scope decision becomes load-bearing. Prefer, when unsure whether a restriction is
real: leave it unrestricted + test the case, rather than guard it + test the guard.

### What would have caught it (the fix for the process, not just the code)
- **Write the contract first.** For any new API param, enumerate {every public entry point} × {every
  relevant physics axis} as the test matrix *before* coding — here: `{lbm, run_lbm, simulate}` ×
  `{isotropic-BGK, per-axis-anisotropic-MRT}` × `{each mode}`. Tests derive from the matrix, not from
  the implementation.
- **Test the outermost user surface, not the inner one you just touched.** A per-mode test through
  `run_lbm`/`simulate` would have caught Gap A on day one.
- **Treat every `raise`/guard as a claim to justify.** "special requires bgk" should have forced the
  question "why?" — and the answer (the overlay is post-stream) dissolves the guard.
- **The plan should have a "surface coverage" line per feature phase**, not just "expose via X".

---

## Fix status
**RESOLVED 2026-07-01.** Both gaps shipped in the broader API-consistency hardening ([PLAN.md](./PLAN.md),
[API_CONSISTENCY_AUDIT.md](./API_CONSISTENCY_AUDIT.md)):
- **Gap A (P1)** — Phase 1 (commit `40cd2ca`): `run_lbm`/`simulate` now forward `boundary`/`alpha`;
  tested through BOTH one-shot surfaces.
- **Gap B (C1)** — Phase 2 (commit `1dda8f6`): added `lbm_step_d2q9_mrt_wall`, dropped the bgk-only
  guard, dispatch `mrt && D2Q9_ONLY → mrt_wall`. Per-axis-anisotropic (MRT) + specular walls now run.
  The `test_lbm_rejects_oblique_Dxy` test was REWORDED to assert the OBLIQUE case as a *documented*
  limitation (Audit #46), not "anisotropy broadly."

The post-mortem's process fix — write the contract matrix FIRST — was implemented as
`test_api_contract.py` (Phase 0, `1a65d3d`): every `{entry × engine × param × physics}` cell is an
asserted row, unfixed cells were `xfail(strict=True)` so each landing fix XPASS-forced its in-phase
flip to a live assert. The two gaps can no longer hide behind green tests.
