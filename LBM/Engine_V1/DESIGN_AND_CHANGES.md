# LBM_V1 — Design & Changes Document

> Comprehensive record of architecture, design decisions, bugs encountered, and fixes applied during the implementation of all 8 phases.

---

## 1. Architecture Overview

### Two-Layer Pattern

All computational code follows a two-layer design:

| Layer | Role | Example |
|-------|------|---------|
| **Layer 2** (pure functions) | Stateless, `torch.compile`-ready kernels | `collide_bgk_d2q5()`, `stream_d2q5()`, `apply_neumann_d2q5()` |
| **Layer 1** (stateful classes) | Owns tensors, coordinates workflow | `LBMSimulation` |

Layer 2 functions take and return tensors with no side effects. Layer 1 classes hold state (`self.f`, `self.V`, `self.ionic_states`) and call Layer 2 functions in sequence.

### File Structure

```
LBM_V1/
  src/
    lattice/
      base.py          # LatticeDefinition dataclass
      d2q5.py          # D2Q5: e, w, opposite, cs2
      d2q9.py          # D2Q9: e, w, opposite, cs2
    collision/
      bgk.py           # BGK collision (D2Q5 + D2Q9)
      mrt/
        d2q5.py        # MRT D2Q5 (3 moments)
        d2q9.py        # MRT D2Q9 (9 moments, Lallemand & Luo 2000)
    streaming/
      d2q5.py          # Pull streaming (torch.roll)
      d2q9.py          # Pull streaming (torch.roll, 2D diagonals)
    boundary/
      masks.py         # precompute_bounce_masks (irregular domains)
      neumann.py       # Bounce-back BC
      dirichlet.py     # Anti-bounce-back BC
      absorbing.py     # Equilibrium-incoming BC
    solver/
      rush_larsen.py   # ionic_step + compute_source_term
    diffusion.py       # tau_from_D (Chapman-Enskog)
    state.py           # create_lbm_state, recover_voltage
    step.py            # Fused step functions (collide+stream+BC+recover)
    simulation.py      # LBMSimulation orchestrator
  ionic/               # Copied from Engine_V5.3 (TTP06, ORd)
  tests/
    test_phase2.py     # Collision operators (6 tests)
    test_phase4.py     # Boundary conditions (4 tests)
    test_phase5.py     # Pure diffusion (5 tests)
    test_phase6.py     # Ionic coupling (3 tests)
    test_phase7.py     # Simulation orchestrator (3 tests)
    test_phase8.py     # Boundary speedup experiment (3 tests)
```

### Simulation Step Sequence

Each call to `LBMSimulation.step()`:

1. Compute `I_stim` from active stimulus protocols
2. Compute `I_ion = model.compute_Iion(V, ionic_states)`
3. Compute source term `R = -(I_ion + I_stim) / Cm`
4. LBM fused step:
   - Collide: `f_star = f - omega*(f - w*V) + dt*w*R`
   - Clone: `f_pre = f_star.clone()` (needed for bounce-back)
   - Stream: pull convention via `torch.roll`
   - Neumann BC: `f[opp[a]] = torch.where(mask[a], f_pre[a], f[opp[a]])`
   - Recover: `V = sum(f, dim=0)`
5. Update ionic states: `_, states = model.step(V, states, dt, I_stim)` (V_new discarded)
6. Advance `self.t += dt`

---

## 2. Key Design Decisions

### Pull Streaming Convention

**Decision:** Use pull convention: `f_post[a](x) = f_pre[a](x - e_a)`.

In `torch.roll`, `shift = +e_component`:
- East (e_x=+1): `torch.roll(f[1], shifts=+1, dims=0)` — pulls from x-1
- West (e_x=-1): `torch.roll(f[2], shifts=-1, dims=0)` — pulls from x+1

**Rationale:** Pull convention means each node gathers its own incoming distributions — naturally parallelizable and cache-friendly.

### f_star Clone for Bounce-Back

**Decision:** Save `f_star = f.clone()` before streaming, use `f_star[a]` in bounce-back formulas.

**Rationale:** After roll-based streaming, boundary nodes contain wrapped values from the opposite side of the domain (periodic artifact). The pre-streaming `f_star` holds the correct outgoing distribution needed for bounce-back: `f[opp[a]](x) = f_star[a](x)`.

### Manual Edge Masks vs precompute_bounce_masks

**Decision:** Two mask strategies:
1. `_make_rect_masks()` — for full-grid rectangular domains (manual edge marking)
2. `precompute_bounce_masks(domain_mask, lattice)` — for irregular domains with explicit outside nodes

**Rationale:** `precompute_bounce_masks` uses `roll(domain_mask, -e_a)` to find boundary nodes. On a full-grid (all True), rolling an all-True mask gives all-True neighbors, so `mask & ~neighbor = False` everywhere — no boundaries detected. Rectangular domains need manual edge masks instead.

### Separate s_jx / s_jy for Anisotropic MRT

**Decision:** MRT relaxation rates split into `s_jx` (row 3 of S diagonal) and `s_jy` (row 5) instead of a single `s_j`.

**Rationale:** Chapman-Enskog gives `D_xx = cs2 * (1/s_jx - 0.5) * dt` and `D_yy = cs2 * (1/s_jy - 0.5) * dt`. A single `s_j` forces isotropic diffusion. Cardiac tissue requires anisotropic D (fiber vs cross-fiber).

### Source Term: No chi in Denominator

**Decision:** `R = -(I_ion + I_stim) / Cm` (not `/ (chi * Cm)`).

**Rationale:** The monodomain equation is:
```
chi * Cm * dV/dt = sigma * lap(V) - chi * (I_ion + I_stim)
```
Dividing by `chi * Cm`:
```
dV/dt = [sigma/(chi*Cm)] * lap(V) - (I_ion + I_stim) / Cm
```
The chi is absorbed into `D = sigma/(chi*Cm)`. The source term is `-(I_ion + I_stim)/Cm`.

### torch.where for BC Application

**Decision:** Use `torch.where(mask, new_val, old_val)` instead of boolean indexing `f[mask] = val`.

**Rationale:** `torch.where` is a pure tensor operation compatible with `torch.compile` graph capture. Boolean indexing creates dynamic shapes that break compilation.

---

## 3. Bugs Encountered and Fixed

### Bug 1: Streaming Roll Shifts ALL Inverted (CRITICAL)

**Phase:** 3 (Streaming) — discovered during Phase 4 testing
**Severity:** Critical — broke all boundary conditions and non-symmetric diffusion

**Symptom:** Phase 4 Neumann conservation test showed 97% mass loss. Bounce-back was overwriting valid incoming distributions with zeros.

**Root Cause:** Misunderstanding of `torch.roll` semantics. `torch.roll(x, shifts=-1, dims=0)` gives `output[i] = input[i+1]` — content moves LEFT (toward -x), not right. The code had `shifts=-1` labeled as "east" when it was actually pulling from the east neighbor (correct for pull) but shifting the wrong direction.

The correct rule for pull streaming: **shift = +e_component**.
- East (e_x=+1): `shifts=+1` → `output[x] = input[x-1]` (pulls from west neighbor)
- West (e_x=-1): `shifts=-1` → `output[x] = input[x+1]` (pulls from east neighbor)

**Diagnosis Method:** Step-by-step trace of a single pulse. Placed mass at x=3, rolled with shifts=-1, found mass at x=2 (moved west, not east). Then traced the bounce-back: at north wall y=3, f[4] (south) had valid data that was overwritten by f_star[3] which was zero — because streaming had placed southward distributions at the north wall (all directions swapped).

**Fix:** Flipped ALL roll signs in `src/streaming/d2q5.py` and `src/streaming/d2q9.py`. Updated CONVENTIONS.md and IMPLEMENTATION.md tables.

**Why it was invisible before Phase 4:** Isotropic Gaussian diffusion tests (Phase 2/5) are symmetric — swapping all directions produces identical results. Only boundary conditions (which are direction-specific) exposed the error.

---

### Bug 2: precompute_bounce_masks Finds No Boundaries on Full Grids

**Phase:** 4 (Boundary Conditions)
**Severity:** High — Neumann tests passed trivially without actually testing Neumann BC

**Symptom:** With `domain_mask = torch.ones(Nx, Ny, dtype=torch.bool)`, all bounce_masks were empty (False everywhere).

**Root Cause:** `torch.roll` wraps periodically. Rolling an all-True mask by any offset gives an all-True result. The formula `boundary = mask & ~roll(mask, -e_a)` gives False everywhere when mask is uniformly True.

**Impact:** The original Neumann test (4-V1) passed with 0.0 drift — not because Neumann BC was working, but because periodic streaming naturally conserves mass. The test didn't test Neumann BC at all.

**Fix:** Created `make_rect_bounce_masks()` that manually marks edge nodes:
```python
if ex == 1:   m[-1, :] = True   # east wall
if ex == -1:  m[0, :] = True    # west wall
if ey == 1:   m[:, -1] = True   # north wall
if ey == -1:  m[:, 0] = True    # south wall
```
`precompute_bounce_masks()` retained for irregular domains where outside nodes are explicitly marked as False.

---

### Bug 3: Border-of-Outside-Nodes Causes Mass Loss

**Phase:** 4 (Boundary Conditions)
**Severity:** High — alternative rectangular mask strategy also failed

**Symptom:** Surrounding the domain with a 1-node border of False values (outside nodes) caused 99% mass loss even with bounce-back enabled.

**Root Cause:** Roll-based streaming pulls distributions from ALL directions at every node, including non-outgoing directions. At the west boundary (x=1), direction 1 (east) pulls from x=0 — an outside node with zeroed distributions. Bounce-back only handles outgoing directions (those pointing into the wall), not the non-outgoing directions that also read from outside.

**Fix:** Abandoned the border approach for rectangular domains entirely. The manual edge mask approach (Bug 2 fix) doesn't rely on outside nodes existing.

---

### Bug 4: MRT Single s_j Prevents Anisotropic Diffusion

**Phase:** 5 (Pure Diffusion)
**Severity:** Medium — isotropic tests passed, anisotropic test failed

**Symptom:** Phase 5 test 5-V4 set D_xx=0.2, D_yy=0.05 but measured var_x/var_y = 1.0000 (perfectly isotropic).

**Root Cause:** MRT relaxation diagonal was:
```python
S = [0, s_e, s_eps, s_j, s_q, s_j, s_q, s_pxx, s_pxy]
#                    ^^^         ^^^
#                   row 3       row 5 — SAME value
```
Row 3 controls j_x flux relaxation, row 5 controls j_y. Same `s_j` → same D in both directions.

**Fix:** Split into separate parameters:
```python
S = [0, s_e, s_eps, s_jx, s_q, s_jy, s_q, s_pxx, s_pxy]
```
Added `s_jy` parameter with default `None` (falls back to `s_jx` for isotropic case). Chapman-Enskog: `D_xx = cs2*(1/s_jx - 0.5)*dt`, `D_yy = cs2*(1/s_jy - 0.5)*dt`.

---

### Bug 5: Source Term Had Extra chi Factor (140x Too Weak)

**Phase:** 7 (Simulation Orchestrator)
**Severity:** Critical — no action potential could initiate

**Symptom:** Phase 7 planar wave test: V_max = -84.1 mV after stimulus. A -80 pA/pF stimulus for 2 ms should depolarize significantly, but V barely moved.

**Root Cause:** `compute_source_term` had:
```python
R = -(I_ion + I_stim) / (chi * Cm)    # chi = 140
```
This makes R 140x too small. Per the monodomain equation derivation (see Design Decision #5 above), chi is absorbed into D, not the source term.

**Diagnosis:** delta_V per step = dt * R = 0.01 * (-(-80))/(140*1) = 0.00571 mV/step. Over 200 steps (2ms): 1.14 mV total. Should be: 0.01 * 80/1 = 0.8 mV/step → 160 mV over 2ms.

**Fix:** Changed to `R = -(I_ion + I_stim) / Cm`. Updated function signature, docstring, and all callers.

---

### Bug 6: Phase 8 Simulation Time Too Short

**Phase:** 8 (Boundary Experiment)
**Severity:** Low — test configuration error

**Symptom:** test_8v3 assertion "Wave didn't reach far end: 0 nodes activated".

**Root Cause:** Domain = 200 nodes * 0.025 cm = 5.0 cm. At CV ~75 cm/s, wave needs ~66 ms to traverse. `t_end` was 50.0 ms.

**Fix:** Increased `t_end` from 50.0 to 70.0 ms.

---

## 4. Final Test Results

**24/24 tests pass across 6 test files.**

| Phase | Tests | Key Results |
|-------|-------|-------------|
| 2 | 6/6 | BGK + MRT collision correctness |
| 4 | 4/4 | Neumann conservation (drift < 1e-10), Dirichlet linearity, Absorbing reflection < 5%, Mixed BC |
| 5 | 5/5 | Gaussian variance err < 0.1%, MRT-BGK match < 1e-14, Anisotropic D ratio correct, 30-deg fiber rotation |
| 6 | 3/3 | Single-cell AP (peak=56.6mV, APD90=223ms), Rush-Larsen stable at dt=0.1ms, Source conservation |
| 7 | 3/3 | Planar wave CV=75.4 cm/s, D2Q5/D2Q9 match, Stimulus timing correct |
| 8 | 3/3 | Neumann uniform CV (ratio=1.0000), D2Q5/D2Q9 CV match, Full propagation CV=75.0 cm/s |

---

## 5. Key Findings from Phase 8

- **Neumann BC gives perfectly uniform CV** — no speedup or slowdown at tissue boundaries
- **D2Q9 has ~3% edge CV artifact** (O(dx^2) diagonal discretization error) — known limitation
- **D2Q5 has no such artifact** — preferred for boundary-sensitive studies
- **Dirichlet BC on V cannot produce Kleber boundary speedup** — it acts as a current sink that slows conduction (see BOUNDARY_SPEEDUP_ANALYSIS.md)

---

## 6. References

| Reference | Used For |
|-----------|----------|
| Rapaka et al. 2012 | D2Q5 BGK, Chapman-Enskog for diffusion |
| Campos et al. 2016 | D2Q9 MRT, bounce-back (Eq. 17), anti-bounce-back |
| Lallemand & Luo 2000 | D2Q9 MRT moment matrix and equilibrium |
| Dawson, Chen & Doolen 1993 | Original LBM reaction-diffusion bounce-back |
| Inamuro et al. 1995 | Anti-bounce-back for Dirichlet BC |
| ten Tusscher & Panfilov 2006 | TTP06 ionic model |
