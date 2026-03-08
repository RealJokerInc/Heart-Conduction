# Bidomain Engine V1 — Code Audit Report

**Date:** 2026-03-08
**Scope:** Full codebase audit of all solver, discretization, ionic, and test code

## Status Key
- [ ] Open
- [x] Fixed
- [~] Won't fix (documented reason)

---

## CRITICAL / HIGH — Bugs That Produce Wrong Results

### A1. `explicit_rkc.py` — `w1` docstring cited wrong formula
**Severity:** HIGH (initially); actual code was correct
**File:** `solver/diffusion_stepping/explicit_rkc.py:135`
**Issue:** The code uses `w1 = T_s(w0) / T_s'(w0)`, which is correct for the
UNNORMALIZED recurrence (W_j tracks T_j(w0)*Y_j). The docstring incorrectly
cited the paper's normalized form `T_s'(w0) / T_s''(w0)`. Verified P'(0)=1.0.
**Fix:** Docstring corrected; code unchanged.
**Status:** [x]

### A2. `imex_sbdf2.py` — LHS/RHS `dt` mismatch
**Severity:** HIGH
**File:** `solver/diffusion_stepping/imex_sbdf2.py:105,119-120`
**Issue:** `step()` uses the argument `dt` for RHS scaling (`(1.0/dt) * state.Vm`)
but the LHS operator `A_bdf1`/`A_bdf2` was built with the constructor's `dt`.
If they ever differ, the linear system is internally inconsistent. Same issue
affects `decoupled_gs.py` and `decoupled_jacobi.py`.
**Fix:** Added dt validation guard in `step()` for all 3 solvers. Raises ValueError
if step dt doesn't match constructor dt.
**Status:** [x]

### A3. `spectral.py` — Periodic eigenvalues use continuous formula
**Severity:** HIGH
**File:** `solver/linear_solver/spectral.py:149-151`
**Issue:** Uses `k² = (2π·fftfreq)²` (continuous Laplacian eigenvalue) instead of
the discrete formula `(2/dx²)(1 - cos(2πk/N))`. Gives wrong results for
high-frequency modes with periodic BCs. Neumann/Dirichlet formulas are correct.
**Fix:** Changed to discrete formula `(2/dx²)(1 - cos(2πk/N))`.
**Status:** [x]

### A4. `chebyshev.py` — `set_eigenvalue_bounds()` is broken
**Severity:** HIGH
**File:** `solver/linear_solver/chebyshev.py:293`
**Issue:** Sets `_A_id = None`, which causes `_estimate_eigenvalues()` to overwrite
the manually set bounds on the next `solve()` call.
**Fix:** Added `_MANUAL_BOUNDS = object()` sentinel. `set_eigenvalue_bounds()` sets
`_A_id = _MANUAL_BOUNDS`, and `_estimate_eigenvalues()` checks `is _MANUAL_BOUNDS`.
**Status:** [x]

### A5. `pcg.py` — Zero RHS with warm start returns wrong answer
**Severity:** HIGH
**File:** `solver/linear_solver/pcg.py:170-176`
**Issue:** When `b_norm < 1e-14` and warm start is active, returns `x.clone()` where
`x` contains the previous solution. For `Ax = 0` with SPD A, the only solution
is `x = 0`.
**Fix:** Added `x.zero_()` before return in zero-RHS path.
**Status:** [x]

---

## MEDIUM — Design Issues That Cause Bugs Under Extended Use

### B1. `fdm.py` — Cross-derivative stencil doesn't use harmonic mean
**Severity:** MEDIUM
**File:** `discretization/fdm.py` (_build_laplacian diagonal stencil)
**Issue:** Cardinal directions use harmonic mean of D at faces, but diagonal
directions use only the local node's D_xy. For spatially varying fiber angles,
this produces an asymmetric L matrix, making the elliptic operator non-SPD.
**Status:** [ ]

### B2. `fdm.py` — Incomplete boundary cross-derivative stencil
**Severity:** MEDIUM
**File:** `discretization/fdm.py` (_build_laplacian boundary handling)
**Issue:** At corner/edge nodes, the mixed derivative d²u/(dx·dy) pattern is
missing some diagonal neighbors, but partial contributions are still added.
Introduces O(h) error at boundaries.
**Status:** [ ]

### B3. `bidomain.py` — Stimulus timing in Strang splitting
**Severity:** MEDIUM
**File:** `bidomain.py:113-114`
**Issue:** Both Strang half-ionic steps evaluate stimulus at `state.t` (time before
the step). Time is only advanced after all three sub-steps. For stimuli starting/
ending mid-step, second half-step should evaluate at `t + dt/2`.
**Status:** [ ]

### B4. `forward_euler.py` / `rush_larsen.py` — Gauss-Seidel ordering in ionic update
**Severity:** MEDIUM
**File:** `ionic_stepping/forward_euler.py:81`, `ionic_stepping/rush_larsen.py:96`
**Issue:** Concentration rates computed from OLD V but NEW gates (gates updated
in-place before concentration rate eval). Not pure Forward Euler.
**Fix:** Moved concentration rate computation before gate updates in both files.
**Status:** [x]

### B5. `pcg_spectral.py` — Multiple issues
**Severity:** MEDIUM
**File:** `solver/linear_solver/pcg_spectral.py`
**Issues:**
  a. `pAp` guard uses absolute 1e-30 instead of scale-relative threshold
  b. `last_iters` not set on `pAp` early break path
  c. O(n) warm-start check instead of O(1) flag
  d. Does not pass per-axis BCs to spectral preconditioner
**Fix:** Rewrote file with all 4 fixes + dtype check + zero-RHS handling.
**Status:** [x]

### B6. `pcg.py` — Workspace allocation doesn't check dtype changes
**Severity:** MEDIUM
**File:** `solver/linear_solver/pcg.py:99`
**Issue:** Checks shape and device but not dtype. Float32→float64 switch would
use float32 workspace, causing silent precision loss.
**Fix:** Added `self._r.dtype != dtype` to workspace allocation check.
**Status:** [x]

### B7. No `clone()` method on `BidomainState`
**Severity:** MEDIUM
**File:** `state.py`
**Issue:** No way to snapshot state for adaptive time stepping rollback.
**Fix:** Added `clone()` method that deep-copies mutable tensors (Vm, phi_e,
ionic_states) while sharing immutable data (coordinates, spatial).
**Status:** [x]

### B8. `rush_larsen.py` — Duplicated gate update logic
**Severity:** MEDIUM
**File:** `ionic_stepping/rush_larsen.py:89-93` vs `116-120`
**Issue:** `step()` inlines Rush-Larsen logic and `_update_gates()` has identical
code. Maintenance risk — updates to one may not propagate to the other.
**Fix:** `step()` now calls `self._update_gates()` instead of inlining.
**Status:** [x]

---

## LOW — Code Quality, Edge Cases, Inconsistencies

### C1. `imex_sbdf2.py` — Unused import of `sparse_mv` (line 28) [x]
### C2. `explicit_rkc.py` — `phi_e_frozen` is a reference, not a copy [x] (now cloned)
### C3. `semi_implicit.py` / `explicit_rkc.py` — Multiple allocations per step [ ]
### C4. `base.py` (discretization) — `Cm`/`conductivity` not in ABC [ ]
### C5. `bidomain.py` — No guard for manual `spectral` with anisotropic conductivity [ ]
### C6. `protocol.py` — `get_current()` overwrites instead of accumulating [x] (now uses +=)
### C7. `fft.py` — Deprecated but importable with no runtime warning [ ]
### C8. Type annotations say `SimulationState` instead of `BidomainState` [x] (4 files fixed)
### C9. `conductivity.py` — `D_sum` / `get_effective_monodomain_D` only valid isotropic [ ]

---

## TEST SUITE ISSUES

### T1. D_eff validation used wrong eigenfunctions/eigenvalues [x]
Replaced Gaussian variance with exact discrete eigenmode `cos(πk(2i+1)/(2N))`
and discrete eigenvalue `(2/dx²)(1-cos(πk/N))`. Precision: 0.1-0.4% (was 25%+).
### T2. BDF-T2 convergence lower bound of 1.5 too weak [ ]
### T3. Shared PCG instances in cross-check tests [x] (separate instances per solver)
### T4. Factory tests only check construction, never run a step [x] (now call sim.run())
### T5. test_phase4 parabolic test is circular (validates against own operators) [ ]
### T6. Energy test allows 1% growth per step [ ]
### T7. Missing coverage: Godunov, ForwardEuler, Dirichlet on new solvers [ ]
