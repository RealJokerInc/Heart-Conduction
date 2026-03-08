# LBM_V1 Improvement Document

Critical code review of the Lattice Boltzmann Method monodomain cardiac simulation.
All phases (0-8) complete, 24/24 tests passing.

---

## Table of Contents

1. [Bug List](#1-bug-list)
2. [File-by-File Analysis](#2-file-by-file-analysis)
3. [Cross-Cutting Concerns: Chi/Cm Formulation](#3-cross-cutting-concerns-chicm-formulation)
4. [Unit Convention Analysis](#4-unit-convention-analysis)
5. [Design Issues](#5-design-issues)

---

## 1. Bug List

### Major

| # | Location | Description | Impact |
|---|----------|-------------|--------|
| M1 | `src/collision/mrt/d2q5.py:32` | `_meq_coeff_D2Q5` energy coefficient is `-4/3` but should be `-2/3`. Derivation: `e_eq = [-4, 1, 1, 1, 1] . [V/3, V/6, V/6, V/6, V/6] = -4V/3 + 4V/6 = -2V/3`. | Wrong equilibrium target for energy moment. Does NOT break diffusion coefficient (conserved/flux moments are correct, tests pass), but relaxes energy toward wrong target. Affects stability margins and higher-order error terms at certain tau values. |
| M2 | `ionic/ttp06/model.py:303` | LUT call uses `cell_type_is_endo=` but `lut.py:209` signature is `celltype_is_endo=`. Keyword mismatch → TypeError when `use_lut=True` for non-default cell types. | Crash on a supported code path. No tests exercise LUT mode, so latent. |

### Minor

| # | Location | Description | Impact |
|---|----------|-------------|--------|
| m1 | `src/collision/bgk.py:19` | Docstring says `R = -(I_ion + I_stim) / (chi * Cm)`. Actual formula (rush_larsen.py:63) is `R = -(I_ion + I_stim) / Cm`. Chi was removed from source term during Phase 7 bug fix but docstring not updated. | Documentation-only. Could mislead future developers. |
| m2 | `src/simulation.py:58` | `self.chi = chi` stored but never referenced anywhere in the simulation loop. Constructor default `chi=140.0` suggests it should be used. Vestigial from pre-Phase-7 bug fix. | Dead code. Confusing API surface. |
| m3 | `src/simulation.py:144` | `I_stim_flat` passed to `ionic_step()` → `model.step()` where it modifies `V_new` that is immediately discarded. Ionic states are updated using original V, so I_stim has zero effect via this path. | Wasted computation + misleading code. |
| m4 | `src/solver/rush_larsen.py:35-39` | `compute_Iion()` called first, then `model.step()` internally recomputes I_ion. Double computation per time step. | ~10-15% wasted work in ionic evaluation. |
| m5 | `tests/test_phase6.py:142-148` | Comment references old formula `R = -(I_ion+I_stim)/(chi*Cm)`. Stale from before chi bug fix. | Stale documentation in tests. |
| m6 | `ionic/base.py` docstring | Says `Istim` units are "uA/uF"; TTP06 model uses "pA/pF". Numerically identical (1 uA/uF = 1 pA/pF) but inconsistent labeling. | Confusion risk only. |

---

## 2. File-by-File Analysis

### 2.1 `src/lattice/base.py`
Clean ABC. Defines Q, D, cs2, e, w, opposite. `D: int = 2` hardcoded — acceptable for V1 (2D only). Two tensor conversion utilities.

### 2.2 `src/lattice/d2q5.py`
Correct. Weights sum to 1.0 (1/3 + 4×1/6). Opposite map (0,2,1,4,3) verified. Velocity vectors standard.

### 2.3 `src/lattice/d2q9.py`
Correct. Weights sum to 1.0 (4/9 + 4×1/9 + 4×1/36). Opposite map verified. Direction ordering matches M matrix in MRT.

### 2.4 `src/diffusion.py`
Correct formulas:
- `sigma_to_D`: `D = sigma / (chi * Cm)` — standard monodomain scaling
- `tau_from_D`: `tau = 0.5 + D*dt/(cs2*dx^2)` — standard Chapman-Enskog
- `tau_tensor_from_D`: correct, off-diagonal has no 0.5 offset

**Design note**: Returns `(tau_xx, tau_yy, tau_xy)` but MRT uses `s = 1/tau`. Conversion done by caller (inline in tests). A `s_from_D()` helper would reduce error risk.

### 2.5 `src/collision/bgk.py`
Logic correct. `f* = f - omega*(f - w*V) + dt*w*R`. Equilibrium `f_eq = w_i * V` is correct for reaction-diffusion (u=0). Source distribution preserves `sum(f*) = V + dt*R`.

**Bug m1**: Docstring claims chi in denominator of R.

### 2.6 `src/collision/mrt/d2q5.py`
M matrix correct and full rank. Collision logic correct (transform → relax → inverse transform → add source).

**Bug M1**: Equilibrium energy coefficient wrong. The code has:
```python
_meq_coeff_D2Q5 = [1.0, 0.0, 0.0, -4.0/3.0, 0.0]
```
Should be:
```python
_meq_coeff_D2Q5 = [1.0, 0.0, 0.0, -2.0/3.0, 0.0]
```

Verification: `e = [-4,1,1,1,1]`, `f_eq = [V/3, V/6, V/6, V/6, V/6]`, `e_eq = -4V/3 + 4(V/6) = -2V/3`.

### 2.7 `src/collision/mrt/d2q9.py`
M matrix matches Lallemand & Luo (2000). Equilibrium coefficients `[1, -2, 1, 0, 0, 0, 0, 0, 0]` verified against `M @ (w * V)`.

**Design gap**: D_xy is controlled by `s_pxy` in standard MRT Chapman-Enskog, but the docstring incorrectly says to use "moment-space rotation (Phase 8)". The standard approach is `tau_pxy = 0.5 + D_xy*dt/(cs2*dx^2)`, `s_pxy = 1/tau_pxy`. Current implementation cannot produce off-diagonal diffusion.

### 2.8 `src/streaming/d2q5.py`, `src/streaming/d2q9.py`
Correct. Pull convention with `roll(shifts=+e_component)`. Sign convention verified (was fixed during Phase 3 per PROGRESS.md).

### 2.9 `src/state.py`
Correct. `recover_voltage = f.sum(dim=0)`. Initialization `f = w*V` is standard equilibrium.

### 2.10 `src/boundary/masks.py`
Correct for irregular domains. Well-documented limitation: full-grid rectangular domain produces no boundaries via `precompute_bounce_masks()` because periodic roll wraps. Separate `make_rect_bounce_masks()` needed.

### 2.11 `src/boundary/neumann.py`
Correct. Full-way bounce-back `f[opp[a]] = f_star[a]` preserves mass exactly.

### 2.12 `src/boundary/dirichlet.py`
Correct. Anti-bounce-back `f[opp[a]] = -f_star[a] + 2*w[a]*V_D` (Inamuro 1995).

**Minor**: Uses `w[a]` (outgoing) where some references use `w[opp[a]]` (incoming). Equivalent for D2Q5/D2Q9 (symmetric weights) but would differ for non-standard lattices.

### 2.13 `src/boundary/absorbing.py`
Correct. First-order absorbing BC: sets incoming to local equilibrium.

### 2.14 `src/step.py`
Clean composition: collide → clone → stream → BC → recover. Only Neumann BC wired in. Dirichlet/absorbing available but require manual loop.

### 2.15 `src/solver/rush_larsen.py`
Source term `R = -(I_ion + I_stim) / Cm` is correct (chi absorbed into D).

**Bug m3**: `ionic_step()` passes I_stim to `model.step()` but the resulting V_new is discarded. I_stim has no effect on ionic state update through this path.

**Bug m4**: I_ion computed twice (once explicitly, once inside `model.step()`).

### 2.16 `src/simulation.py`
Orchestrator logic correct. Operator splitting order: source → LBM step → ionic update (first-order Lie splitting).

**Bug m2**: `self.chi` stored but never used.
**Bug m3**: I_stim passed to ionic_step unnecessarily.

`_make_rect_masks()` correctly handles rectangular domain bounce masks by checking velocity direction components.

### 2.17 `ionic/base.py`
Clean ABC. `step()` signature uses `Istim` while TTP06 uses `I_stim` — inconsistent naming.

### 2.18 `ionic/lut.py`
Correct LUT with linear interpolation. Clamping `0` to `n_points - 1.0001` prevents OOB.

**Design concern**: Module-level `_lut_cache` (mutable global state) is a testing/multiprocessing hazard.

### 2.19 `ionic/ttp06/model.py`
Correct TTP06 implementation. `compute_Iion` returns pA/pF. `compute_concentration_rates` duplicates `update_concentrations()` logic (maintenance risk).

**Bug M2**: LUT keyword mismatch.

### 2.20 `ionic/ttp06/parameters.py`
Correct. Parameters match TTP06 publication. Cell-level `Cm = 0.185 uF` is distinct from tissue-level `Cm = 1.0 uF/cm^2` — both used correctly in their respective contexts.

### 2.21 `ionic/ttp06/currents.py`
Correct. GHK formulation with L'Hopital limit near V=15mV handled properly.

**Note**: Module-level `R, T, F` constants duplicate `TTP06Parameters.R, T, F`. Not a bug but dual definitions.

### 2.22 `ionic/ttp06/gating.py`
Correct. `safe_exp` with clamp at ±80 prevents overflow. Rush-Larsen exponential integrator standard.

### 2.23 `ionic/ttp06/calcium.py`
Correct. Concentration clamping (lines 249-254) prevents negative concentrations.

---

## 3. Cross-Cutting Concerns: Chi/Cm Formulation

### 3.1 The Monodomain PDE

Standard form:
```
chi * Cm * dV/dt = div(sigma * grad(V)) - chi * I_ion + I_stim_vol
```

Where `I_stim_vol` is volumetric stimulus current (uA/cm^3).

### 3.2 Division by chi*Cm

Dividing both sides by `chi * Cm`:
```
dV/dt = D * laplacian(V) + R
```

Where:
- `D = sigma / (chi * Cm)` — effective diffusion coefficient (cm^2/ms)
- `R = -(I_ion + I_stim) / Cm` — reaction source term (mV/ms)

### 3.3 Where Chi Appears in Code

Chi appears in **exactly one place**: `diffusion.py:sigma_to_D()` which computes `D = sigma / (chi * Cm)`. Once D is computed, chi is absent from all subsequent computations. This is correct.

**Vestigial**: `simulation.py` stores `self.chi = chi` but never uses it (Bug m2).

### 3.4 Where Cm Appears in Code

Tissue-level `Cm` (1.0 uF/cm^2) appears in:
1. `diffusion.py:sigma_to_D()` — `D = sigma / (chi * Cm)`
2. `rush_larsen.py:compute_source_term()` — `R = -(I_ion + I_stim) / Cm`

Cell-level `Cm` (0.185 uF, in `parameters.py`) is a **different quantity** used only inside TTP06 for current-to-concentration-flux conversion. These are physically distinct and both used correctly.

### 3.5 Dimensional Consistency

- I_ion from `compute_Iion()`: pA/pF = mV/ms (since pA = pF × mV/ms)
- R = -(I_ion)/Cm: (mV/ms) / (uF/cm^2)
- With Cm = 1.0: R has units mV/ms numerically, matching dV/dt

The dimensional bookkeeping works because the ionic model provides current in normalized form (pA/pF), making Cm = 1.0 act as a dimensionless scaling factor. This is correct but under-documented.

### 3.6 Historical Bug Context

Per MEMORY.md, the original code had `R = -(I_ion + I_stim) / (chi * Cm)` which made the stimulus 140× too weak (chi = 140). The fix (removing chi from R denominator) was correct because chi is already absorbed into D.

---

## 4. Unit Convention Analysis

| Quantity | Units | Location | Notes |
|----------|-------|----------|-------|
| V | mV | everywhere | Consistent |
| I_ion, I_stim | pA/pF | ionic model, source term | = uA/uF = mV/ms |
| Cm (tissue) | uF/cm^2 | simulation.py (default 1.0) | Monodomain scaling |
| Cm (cell) | uF | parameters.py (0.185) | TTP06 internal only |
| sigma | mS/cm | diffusion.py input | Standard cardiac |
| chi | 1/cm | diffusion.py input | Default 140 |
| D | cm^2/ms | LBM, diffusion.py | After chi*Cm scaling |
| dx | cm | LBM grid | Spatial resolution |
| dt | ms | LBM step | Temporal resolution |
| t | ms | everywhere | Consistent |
| tau | dimensionless | LBM relaxation | = 0.5 + D*dt/(cs2*dx^2) |

**Two-Cm ambiguity**: The coexistence of cell-level Cm (0.185 uF) and tissue-level Cm (1.0 uF/cm^2) is a confusion risk. These are physically different quantities that happen to share a name. The cell-level Cm converts membrane current to charge flux for concentration dynamics. The tissue-level Cm scales the PDE. Both are used correctly but the distinction is nowhere documented in the codebase.

---

## 5. Design Issues

### D1. No off-diagonal diffusion (D_xy) support
The MRT implementation has `s_pxy` as a "free" parameter but doesn't use it to encode D_xy. The standard Chapman-Enskog analysis for D2Q9 MRT shows D_xy is controlled by `s_pxy = 1 / (0.5 + D_xy*dt/(cs2*dx^2))`. The docstring incorrectly defers this to "moment-space rotation" when the standard approach is direct.

### D2. Step functions hardcoded to Neumann BC
`step.py` functions only use bounce-back (Neumann). Dirichlet and absorbing BCs exist but require manual simulation loops. Should accept a BC function parameter for composability.

### D3. No irregular domain support in simulation.py
`LBMSimulation` only supports rectangular domains via `_make_rect_masks()`. `precompute_bounce_masks()` for irregular domains exists but isn't wired into the orchestrator.

### D4. Ionic state storage shape mismatch
`ionic_states` is `(Nx*Ny, n_states)` flat while V is `(Nx, Ny)` grid-shaped. Requires reshape operations in `step()`. Storing as `(Nx, Ny, n_states)` would be more natural for the LBM paradigm.

### D5. No MRT in simulation.py
Only BGK step functions are wired into the orchestrator. MRT collision operators exist but aren't accessible through `LBMSimulation`.

### D6. No torch.compile actually applied
Code is documented as "torch.compile compatible" but no compiled kernels are created. The collide-stream-BC pipeline would benefit from fusion.

### D7. Global LUT cache
Module-level `_lut_cache` dict in `lut.py` is mutable global state. Hazardous for multi-device or concurrent simulation scenarios.

### D8. ionic_step design
The current `ionic_step()` function calls `model.step()` (which updates V and ionic states) then discards V. This is wasteful and obscures the LBM paradigm where V comes from distributions. A dedicated `step_ionic_only(V, ionic_states, dt)` that only updates gates and concentrations (without computing V_new) would be cleaner and ~10-15% faster.
