# Bidomain Engine V1 — Code Audit Report

**Date:** 2026-03-08 (Round 1), 2026-03-08 (Round 2 + verification)
**Scope:** Full codebase audit — solvers, discretization, ionic models, tests

## Status Key
- [ ] Open
- [x] Fixed
- [~] Won't fix (documented reason)
- [!] False positive (verified incorrect)

---

## Round 1 — All Resolved

| ID | File | Issue | Status |
|----|------|-------|--------|
| A1 | `explicit_rkc.py` | w1 docstring cited wrong formula (code was correct) | [x] |
| A2 | `imex_sbdf2/gs/jacobi` | dt mismatch guard added to step() | [x] |
| A3 | `spectral.py` | Periodic eigenvalues fixed to discrete formula | [x] |
| A4 | `chebyshev.py` | set_eigenvalue_bounds() sentinel fix | [x] |
| A5 | `pcg.py` | Zero RHS with warm start now returns zero | [x] |
| B4 | `forward_euler/rush_larsen` | Concentration rates moved before gate updates | [x] |
| B5 | `pcg_spectral.py` | Rewrote: pAp guard, iters tracking, warm-start flag | [x] |
| B6 | `pcg.py` | Dtype check in workspace allocation | [x] |
| B7 | `state.py` | BidomainState.clone() added | [x] |
| B8 | `rush_larsen.py` | Deduplicated gate update logic | [x] |
| C1 | `imex_sbdf2.py` | Unused import removed | [x] |
| C2 | `explicit_rkc.py` | phi_e_frozen now cloned | [x] |
| C6 | `protocol.py` | Stimulus accumulates with += | [x] |
| C8 | 4 files | SimulationState → BidomainState type annotations | [x] |
| T1 | `test_helpers.py` | D_eff eigenfunctions/eigenvalues fixed (0.1% error) | [x] |
| T3 | 4 test files | Separate PCG instances per solver | [x] |
| T4 | 4 test files | Factory tests now call sim.run() | [x] |

---

## Round 2 — Verified Findings

### HIGH (1 confirmed)

#### R2-H1. FDM cross-derivative missing factor of 2 + wrong sign ★
**Severity:** HIGH — CONFIRMED by verification
**Files:** ALL FDM implementations across the project:
- `Bidomain/Engine_V1/cardiac_sim/simulation/classical/discretization/fdm.py:383`
- `Monodomain/Engine_V5.4/cardiac_sim/simulation/classical/discretization_scheme/fdm.py:247`
- `Monodomain/Engine_V4/diffusion.py:354`
- `Monodomain/Engine_V3/diffusion.py:351`
- `Research/openCARP_FDM_FVM/01_FDM_Stencils_and_Implementation.md:73-76`

**Issue:** TWO errors in the 9-point anisotropic stencil cross-derivative:

1. **Missing factor of 2:** `cxy = 1/(4·dx·dy)` should be `1/(2·dx·dy)`.
   The expansion `div(D·∇V) = Dxx·V_xx + 2·Dxy·V_xy + Dyy·V_yy` has a
   factor of 2 on the mixed derivative. Off-axis coupling was 50% too weak.

2. **Wrong sign pattern:** The diagonal neighbor signs were all negated.
   Correct: NE=+Dxy·cxy, NW=-Dxy·cxy, SE=-Dxy·cxy, SW=+Dxy·cxy.
   The code had NE=-Dxy·cxy etc, making diffusion faster cross-fiber
   than along-fiber (physically backwards).

The research doc (lines 73-76) had both errors; all implementations copied them.

**Impact:** Anisotropic diffusion with rotated fibers (Dxy ≠ 0) was
qualitatively wrong (inverted direction + 50% magnitude). Harmless for
isotropic or axis-aligned cases (Dxy = 0).

**Fix (Bidomain V1):** Changed `cxy = 1/(4dx·dy)` → `1/(2dx·dy)` and
flipped all four diagonal entry signs. Validated by `test_aniso_crossderiv.py`
(3 tests: stencil coefficients, matrix decomposition, directional diffusion).
No regression in isotropic test suites (19/19 pass).
Other engine versions (V3, V4, V5.4) and the research doc are NOT yet fixed.
**Status:** [x] (Bidomain V1 only)

---

### MEDIUM (open items from Rounds 1+2)

#### B1. FDM diagonal stencil breaks symmetry for heterogeneous Dxy
**File:** `fdm.py:434-455`
**Issue:** Diagonal neighbors use local `dxy[i,j]` only, not averaged with
neighbor. Cardinal directions correctly use harmonic mean. For heterogeneous
anisotropic tissue, L becomes asymmetric → PCG fails.
**Fix:** Use arithmetic mean: `d_xy_face = (dxy[i,j] + dxy[i+1,j+1]) / 2`.
**Status:** [ ]

#### B2. FDM incomplete boundary cross-derivative stencil
**File:** `fdm.py` boundary handling
**Issue:** At corner/edge nodes, missing diagonal neighbors produce partial
cross-derivative contributions. O(h) error at boundaries.
**Status:** [ ]

#### R2-M2. CFL check uses scalar D_i only (ignores anisotropy)
**File:** `semi_implicit.py:56-58`, `explicit_rkc.py:142-144`
**Issue:** `dt_cfl = dx²/(4·D_i)` uses scalar. Should use max eigenvalue of
D tensor for anisotropic conductivity.
**Status:** [ ]

#### R2-M3. RKC: unnecessary phi_e clone + 2 temps per stage
**File:** `explicit_rkc.py:171,185-189`
**Status:** [ ]

#### R2-M4. PCG returns clone, then copied into state (wasted alloc)
**File:** `decoupled_gs.py:96,109` and similar
**Status:** [ ]

#### R2-M5. IMEX SBDF2 Vm_prev clone per step (could buffer-swap)
**File:** `imex_sbdf2.py:111,125`
**Status:** [ ]

#### R2-M7. PCG preconditioner cache uses id(A) (unreliable after GC)
**File:** `pcg.py:108-114`
**Status:** [ ]

#### R2-M9. BidomainState.clone() shares mutable stimulus lists
**File:** `state.py:112-131`
**Status:** [ ]

#### R2-M1. FDM diagonal stencil doesn't average Dxy with neighbor
**File:** `fdm.py:434-455` (same as B1; consolidated)
**Status:** [ ] (see B1)

#### R2-M11. TTP06 `I_stim` vs ORd/ABC `Istim` naming inconsistency
**File:** `ttp06/model.py:215` vs `base.py:102`
**Status:** [ ]

#### R2-M13. TTP06 calcium.py return type hint wrong (5 vs 6 values)
**File:** `ionic/ttp06/calcium.py:180`
**Status:** [ ]

#### R2-M14. StructuredGrid.edge_masks recomputes on every access
**File:** `tissue_builder/mesh/structured.py:209-220`
**Status:** [ ]

#### R2-H5. pcg_spectral.py: forced warm start, no disable option
**File:** `solver/linear_solver/pcg_spectral.py`
**Issue:** `_has_warm_start = True` after first solve, no way to disable.
PCGSolver has `use_warm_start` param; this solver doesn't.
**Status:** [ ]

---

### LOW (verified / downgraded)

#### R2-C1. TTP06 LUT kwarg name mismatch (dead code)
**File:** `ionic/ttp06/model.py:303` → `ionic/lut.py:209`
**Issue:** `cell_type_is_endo` vs `celltype_is_endo`. Will crash if LUT mode
enabled, but `use_lut=True` is never passed anywhere in the codebase.
**Verification:** Confirmed mismatch. LUT path is completely dead code.
**Status:** [ ]

#### R2-C2. ORd jp gate initial state uses wrong steady-state value
**File:** `ionic/ord/parameters.py:369`, `ionic/ord/gating.py:94-96`
**Verification:** The RUNTIME code (`_update_gates`, `compute_gate_steady_states`)
correctly uses `h_inf` for jp, matching the original ORd C code where `jss = hss`.
The bugs are: (a) `gating.py:INa_jp_inf` is dead code with wrong formula
(`hsp_inf` instead of `h_inf`), (b) initial state uses 0.4347 (`hsp_inf`) instead
of 0.6804 (`h_inf`). Error decays exponentially with τ_jp ≈ 11.7ms, negligible
after ~50ms.
**Status:** [ ]

#### R2-H2. Strang splitting time tracking (standard simplification)
**Verification:** Both half-ionic steps see same `state.t`. This is standard
practice in cardiac simulators. Timing error is at most dt/2 = 0.01ms at
stimulus edges, well below physiological significance.
**Status:** [~] Won't fix — standard practice

#### R2-H7. TTP06 stimulus not in K+ dynamics (per original paper)
**Verification:** Original TTP06 paper omits I_stim from dKi/dt. All reference
implementations (CellML, openCARP, Myokit) do the same. ORd includes it as a
refinement. Effect is ~0.002 mM K+ drift per stimulus (~0.001%).
**Status:** [~] Won't fix — faithful to paper

#### R2-H4. Deprecated fft.py FFTSolver has wrong eigenvalues
**File:** `solver/linear_solver/fft.py:308-315,366`
**Issue:** Continuous eigenvalues + zeroed DC. Module is DEPRECATED and not
exported from __init__.py.
**Status:** [~] Won't fix — deprecated, not exported

#### R2-M8. Spectral eigenvalue cache never invalidated
**Status:** [~] Won't fix — parameters immutable by design

#### R2-M10. bidomain.py hardcodes dtype=float64
**Status:** [~] Won't fix — float64 required for cardiac accuracy

#### R2-M12. ORd divides dV by Cm, TTP06 doesn't
**Status:** [~] Won't fix — matches respective paper conventions

#### R2-M15. IMEX BDF2 A-matrix via dt-trick is non-obvious
**Status:** [~] Won't fix — comment exists, formula is standard

---

### FALSE POSITIVES (verified incorrect)

#### R2-H3. semi_implicit/explicit_rkc missing dt guard — NOT A BUG
Both solvers use the argument `dt` directly in their formulas. Pre-computed
coefficients (w0, w1, CFL check) are dt-independent. The argument dt is
mathematically correct for any value within stability bounds.

#### R2-H6. ORd ICaL GHK singularity threshold too tight — NOT A CONCERN
Float64 handles `x/(e^x - 1)` with full precision down to `|x| ~ 1e-12`.
At V = -0.001 mV, `x = 2·vfrt ~ -7.5e-5` — no numerical issue.

#### R2-M6. Chebyshev solver 2× SpMV cost — FALSE
Only 1 SpMV per iteration (line 273). The from-scratch residual `r = b - Ax`
uses the single SpMV result. No extra SpMV for convergence (fixed iter count).

---

### LOW — Code Quality (carried forward)

| ID | Issue | Status |
|----|-------|--------|
| C3 | semi_implicit/explicit_rkc: multiple allocations per step | [ ] |
| C4 | base.py (discretization): Cm/conductivity not in ABC | [ ] |
| C5 | bidomain.py: no guard for manual spectral + anisotropic | [ ] |
| C7 | fft.py: deprecated but importable | [~] |
| C9 | conductivity.py: D_sum only valid isotropic | [ ] |
| R2-L1 | `_chebyshev_Tpp` is dead code | [ ] |
| R2-L2 | Unused `import warnings` in semi_implicit.py | [ ] |
| R2-L3 | Dense apply_elliptic_pinning mutates input | [ ] |
| R2-L4 | chebyshev.py uses assert instead of raise | [ ] |
| R2-L5 | chebyshev.py inline sparse.mm vs sparse_mv helper | [ ] |
| R2-L6 | _evaluate_Istim allocates tensor every call | [ ] |
| R2-L7 | bidomain.py recomputes dx instead of using grid.dx | [ ] |
| R2-L8 | fdm.py chi=1.0 accepted without deprecation warning | [ ] |
| R2-L9 | lut.py magic number 1.0001 in clamp | [ ] |
| R2-L10 | Forward Euler sequential in-place gate update | [~] |
| R2-L11 | pcg.py returns x.clone() every solve (necessary) | [~] |
| R2-L12 | sparse_mv imported from PCG module | [ ] |

---

## TEST SUITE ISSUES

| ID | Issue | Status |
|----|-------|--------|
| T2 | BDF-T2 convergence lower bound 1.5 too weak | [ ] |
| T5 | test_phase4 parabolic test is circular | [ ] |
| T6 | Energy test allows 1% growth per step | [ ] |
| T7 | Missing coverage: Godunov, ForwardEuler, Dirichlet, anisotropic | [ ] |
