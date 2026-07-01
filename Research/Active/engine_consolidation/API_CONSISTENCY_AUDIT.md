# cardiac_core API Consistency & Fragility Audit (2026-07-01)

> **RESOLVED 2026-07-01** — all 7 HIGH + 8 MED fixed and locked by the contract-matrix stress
> harness (`cardiac_core/tests/test_api_contract.py`, 22 cells). SHIPPED via [PLAN.md](./PLAN.md)
> Phases 0–5 (commits `1a65d3d`→`9702bb7` on `engine-tuner-cardiac-core`): P1 (run_lbm/simulate
> forward boundary/alpha), S1 (bidomain boundary validation), C1 (MRT wall modes), I1 (masked-grid
> union bounce), I2 (`_resolve_mesh` deepcopy immutability), S2/S3/C4 (warn), C3 (shared ionic
> registry), C5/C6/S4 (weights_mode/stencil/splitting exposure + validated engine-mismatch), I3
> (dtype round-trip). **DEFERRED (documented xfail):** C2 oblique-LBM *capability* (real numerics —
> moment-space rotation, Audit #46; the *raise* is a shipped documented-limitation) and C7
> (mono/bidomain boundary-Dxy truncation, paired to C2). 217 passed / 2 xfailed; goldens bit-identical.
>
> **Backlinks:** [PHASE3_BOUNDARY_GAPS.md](./PHASE3_BOUNDARY_GAPS.md) (the two boundary gaps that
> triggered this) · [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md) · [PLAN.md](./PLAN.md) · [IDEALOG](./IDEALOG.md)
>
> **Why this exists.** Two shipped Phase-3 boundary gaps (`run_lbm` drops `boundary`; wall modes
> BGK-only) turned out to be symptoms of a *class*: the public API surface is inconsistently exposed.
> Four adversarial auditors swept it along one axis each (entry-point parity · cross-engine semantics ·
> physics-capability matrix · state integrity). This is the catalog; fixes + a stress-test harness are
> blueprinted in [PLAN.md](./PLAN.md).

## TL;DR — the pattern (and what is NOT broken)
The **numerics are sound**: Cm≠1 is correct on all three engines (no Cm-trap), `build_kwargs` replay
is complete (reset/with_ don't lose an override), and `CardiacMeshData` save/load round-trips the
RAW-D/chi firewall bit-identically (max|dV|=0). The fragility is entirely at the **API surface**:
1. **Capability exists but isn't exposed** (weights_mode, stencils, MRT knobs, oblique fibers).
2. **The same kwarg means different things per engine, with no validation** (`boundary`).
3. **A handful of real silent-wrong-result bugs** (masked grids, `with_` aliasing, `run_lbm`).

Root cause (post-mortem in PHASE3_BOUNDARY_GAPS.md): features were tested at the level they were
BUILT (one factory, happy-path physics), never against a written CONTRACT of {entry points} ×
{engines} × {params} × {physics}. Nothing checked the cross-product, so gaps hid behind green tests.

**Confirmed: 7 HIGH · 8 MEDIUM · 6 LOW.**

---

## Class 1 — Entry-point parameter parity (silent param drop)
| # | Sev | Finding | Location |
|---|-----|---------|----------|
| P1 | HIGH | `run_lbm` drops `boundary` **and** `alpha` → a wall mode requested through the named one-shot is silently HBB (`combined` would run at α=1 ≡ HBB) | `run.py:195-205,230-233` |
| P2 | INFO | `record=`/`ionic_states` absent from all `run_*`/`simulate`; declarative `stimulus=` unreachable via one-shots (#18) | `run.py:70-97,283` |

✅ `build_kwargs` on all 3 factories are replay-complete (mono `api.py:1148`, bidomain `:1309`, lbm `:1453`) — no silent-loss-on-reset. `run_monodomain`/`run_bidomain` are param-complete; only `run_lbm` is incomplete.

## Class 2 — Cross-engine semantics / naming collisions (silent-ignore)
| # | Sev | Finding | Location |
|---|-----|---------|----------|
| S1 | HIGH | **`boundary` OVERLOAD.** bidomain checks only `== 'bath'` (else insulated) with **no validation** → `bidomain(boundary='ncs')` or a typo `'insualted'` silently runs *insulated*; the mirror `lbm(boundary='bath')` correctly raises. `boundary` is the one param valid on two engines with incompatible vocabularies → the entire silent-failure surface (factories have no `**kwargs`, so every *other* misroute is a loud TypeError) | `api.py:1206,1217-1220` vs `_lbm/simulation.py:85-87` |
| S2 | MED | `alpha` silently inert on `boundary='neumann'`/`'hbb'` — `lbm(alpha=0.3)` at default boundary has zero effect, no warning | `_lbm/simulation.py:94` |
| S3 | MED | `sigma_ratio` silently ignored on the **declarative** bidomain path (sigma branch wins when `sigma_i/sigma_e` present) — `bidomain(cond=…, sigma_ratio=10)` no-ops, no warning | `api.py:1223,1263-1276` |
| S4 | MED | Solver-knob availability is asymmetric + undocumented: `theta` bidomain-only, `splitting`/`diffusion_solver`/`linear_solver` mono-only; bidomain supports `splitting` downstream but the factory hides it (TypeError) | `api.py:1047-1049,1164`; `_bidomain/…/bidomain.py:48` |
| S5 | LOW | `dt` — BGK path has **no** CFL/τ guard (MRT does); a diffusive-unstable `dt` yields NaNs silently | `_lbm/simulation.py:144-149` (MRT only) |

## Class 3 — Physics-capability matrix (raise-but-shouldn't / silent-degrade / unexposed / inconsistent)
| # | Sev | Finding | Location |
|---|-----|---------|----------|
| C1 | HIGH | **mrt × specular RAISES** — the `collision != 'bgk'` guard blocks anisotropic(MRT)+wall, but the overlay is post-stream/collision-agnostic (Gap B) | `_lbm/simulation.py:90-92` |
| C2 | HIGH | **oblique (Dxy≠0) LBM rejection despite MRT being oblique-CAPABLE** — `tau_tensor_from_D` returns `tau_xy`, `check_stability_tensor` does the 2×2 test, `mrt_collide_d2q9` carries `s_pxy`; only the factory `raise` + `LBMSimulation` hardcoding `D_xy=0` block it. Inconsistent: mono/bidomain carry a Dxy interior term; LBM rejects | `api.py:1402-1407`; `_lbm/simulation.py:144,150` |
| C3 | HIGH | **ionic-registry asymmetry** — `lbm()` accepts paci/phas13/mhas13; `monodomain()/bidomain()` raise ValueError (only ttp06/ord) (#6) | `_monodomain/…/monodomain.py:82-87`, `_bidomain/…/bidomain.py:132-139` vs `api.py:1379-1385` |
| C4 | MED | `lattice='d2q5'` **silently overridden** to d2q9/mrt for anisotropic D — no warning (#16) | `api.py:1412-1423` |
| C5 | MED | `weights_mode='uniform_8'` + MRT moment-rate knobs + `D_yy` **unexposed** via `lbm()` — the exact `uniform_8` connectivity column the boundary research needs is unselectable | `api.py` lbm signature |
| C6 | MED | mono `stencil∈{moore8_uniform,moore8_iso}` + bidomain `stencil='mehrstellen'` **unexposed** via factories (always cardinal4 / 5pt) | `api.py:1109/1122,1292` |
| C7 | MED | oblique Dxy **dropped at the wall** in mono cardinal4 + bidomain 5pt (interior-correct, boundary-truncated) — three behaviors for one physics vs LBM's outright reject | mono `fdm.py:670-697`, bido `fdm.py:512-534` |
| C8 | LOW | `'paci'→PHAS13Model` alias (paci/phas13 params byte-identical today → benign, but a latent trap; `__init__` shadows real `PaciModel`) (#12) | `api.py:1383`; `ionic/paci/__init__.py:7` |
| C9 | LOW | bidomain reaction uses `getattr(state,'Cm',1.0)` (silent fallback) vs mono's direct `state.Cm` (fail-loud) — inconsistent defensive posture | `_bidomain/…/rush_larsen.py:83` |

✅ Cm≠1 numerical correctness is SOUND on all three (Form-A mono / Form-B bido+lbm both divide the reaction by the real Cm; no mis-scaling).

## Class 4 — State integrity (round-trip / replay / device / masked grids)
| # | Sev | Finding | Location |
|---|-----|---------|----------|
| I1 | HIGH | **LBM masked grid silently wrong** — `lbm()` never passes `bounce_masks` from `data.mask` → `_make_rect_masks` (edges only) → an interior hole conducts. `precompute_bounce_masks(data.mask, lattice)` exists, unwired | `api.py:1416,1425`; `_lbm/simulation.py:164,185-188`; `_lbm/boundary/masks.py:10` |
| I2 | HIGH | **`with_()` breaks immutability** — child shares the parent's `_data` (aliased); `child.stimulate(...)` appends to `self._data.stimuli` → mutates the PARENT. `with_` is engine-immutable, not state-immutable | `api.py:247-254,357,820` |
| I3 | MED | Result dtype not round-tripped — `load_result` hardcodes float64; a saved float32 `Vm` reloads promoted to float64 | `io.py:94-99` |
| I4 | LOW | `ionic_states` cannot be persisted — `save_result` has no param; `_collect` never collects them | `io.py:17-64`, `run.py:77-94` |
| I5 | LOW | Dead parallel loader `mesh/loader.py::load_mesh` stores RAW D with no χ·Cm division (bypasses the firewall) — **not live** (unexported, api uses `load_cardiac_mesh`), a latent trap | `mesh/loader.py:86-102` |
| I6 | LOW | `group_labels`/`group_cell_types` reload as `np.str_`, not `str` — works for `==`/dict but breaks `type()==str`/JSON | `file_format.py` load |

✅ Verified round-tripping correctly (no finding): CardiacMeshData save/load (firewall survives, max|dV|=0); `stimulate()` accumulates (not resets); `reset()` faithful t→0; CUDA stays on-device (mono+lbm); a pre-built float32 / tuner-scaled ionic instance is preserved through `reset()`.

---

## Priorities
- **Silent-wrong-result bugs (fix first):** P1 (`run_lbm`), S1 (`boundary` validation), C1 (mrt+wall), I1 (masked grid), I2 (`with_` aliasing).
- **Consistency/exposure (fix next):** C3 (ionic registry), C2 (oblique LBM), C4/S2/S3 (silent-degrade → warn), C5/C6 (expose weights_mode/stencils), I3 (dtype round-trip).
- **Low/latent:** C7/C8/C9/I4/I5/I6, S4/S5, P2.

## The systemic fix (not just per-bug)
Build a **contract-matrix stress harness**: a parametrized test over `{lbm, run_lbm, simulate, reset,
with_} × {monodomain, bidomain, lbm} × {each param} × {iso-BGK, per-axis-MRT, oblique, masked-grid,
Cm≠1, cuda/float32}` — assert each cell either takes effect, or raises a *validated* error, never
silently degrades. Every finding above becomes one asserted cell. Then a wrong scope decision can't
hide behind green tests again. Blueprinted in [PLAN.md](./PLAN.md).
