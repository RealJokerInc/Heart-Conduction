# Phase 6: Boundary CV Cross-Validation Plan

> **Part 2 of the Bidomain Engine V1 implementation.**
> Systematically compares boundary conduction velocity effects across three
> numerical methods using identical parameters and measurement protocols.

---

## 1. Objective

Distinguish two fundamentally different boundary CV phenomena:

1. **D2Q9 lattice artifact** — O(dx^2) numerical error from diagonal bounce-back
   in the LBM D2Q9 stencil. Causes ~3% CV *slowdown* at edges. Vanishes with
   mesh refinement. Not physical.

2. **Kleber boundary speedup** — Physical effect from intracellular/extracellular
   boundary condition asymmetry at tissue-bath interfaces. The bath shorts the
   extracellular resistance, boosting effective diffusivity from D_eff to D_i.
   Causes ~13% CV *speedup* at edges. Requires bidomain equations.

These effects act in **opposite directions** (slowdown vs speedup), providing a
clean experimental separation.

---

## 2. Critical Prerequisite: Chi*Cm Convention

### The Bug

The bidomain engine's parabolic operator is:
```
A_para = chi_Cm/dt * I - theta * L_i
```
where `L_i` is built with diffusion coefficients `D_i = sigma_i / (chi_real * Cm)`.

This discretizes: `chi_Cm * dVm/dt = L_i * (Vm + phi_e)`

Dividing by chi_Cm: `dVm/dt = D_i / chi_Cm * Lap(Vm + phi_e)`

With the default `chi=1400, Cm=1.0`:
- Effective diffusion rate = D_i / 1400 = 8.86e-7 cm^2/ms
- Correct physical rate = D_i = 0.00124 cm^2/ms
- **1400x too weak — no wave will propagate**

### The Fix

Pass `chi=1.0, Cm=1.0` to `BidomainFDMDiscretization`. Then `chi_Cm = 1.0` and:
```
dVm/dt = D_i * Lap(Vm + phi_e)    (correct)
```

This works because `D_i = sigma_i / (chi_real * Cm_real)` already contains the
physical chi*Cm scaling. The `chi_Cm` parameter in the discrete operator is a
*numerical* scaling factor, not the physical chi. Setting it to 1.0 means "D values
are already physical diffusion coefficients," which they are.

### Consistency Check

| Engine | D used | chi_Cm in operator | Effective diffusion rate |
|--------|--------|-------------------|------------------------|
| LBM V1 | D_eff directly | N/A (in tau) | D_eff |
| V5.4 FDM | D | chi*Cm (default 1400) | D / chi_Cm |
| Bidomain V1 (WRONG) | D_i, D_e | 1400 | D / 1400 |
| Bidomain V1 (FIXED) | D_i, D_e | 1.0 | D |

For the ionic source term: `dVm/dt = -(I_ion + I_stim) / Cm`.
With `Cm = 1.0`, both engines give `dVm/dt = -(I_ion + I_stim)`.
LBM uses `R = -(I_ion + I_stim) / Cm` with `Cm = 1.0` — identical.

---

## 3. Standardized Parameters

All phases share these parameters to enable direct comparison.

| Parameter | Symbol | Value | Units | Notes |
|-----------|--------|-------|-------|-------|
| Grid x-nodes | Nx | 150 | — | Propagation direction |
| Grid y-nodes | Ny | 40 | — | Transverse direction |
| Spacing | dx = dy | 0.025 | cm | 250 um resolution |
| Domain length | Lx | 3.725 | cm | (Nx-1)*dx |
| Domain width | Ly | 0.975 | cm | (Ny-1)*dx |
| Time step | dt | 0.01 | ms | |
| Intra diffusion | D_i | 0.00124 | cm^2/ms | sigma_i/(chi*Cm) = 1.74/1400 |
| Extra diffusion | D_e | 0.00446 | cm^2/ms | sigma_e/(chi*Cm) = 6.25/1400 |
| Effective D | D_eff | 0.000970 | cm^2/ms | D_i*D_e/(D_i+D_e) |
| Ionic model | — | TTP06 endo | — | Human ventricular endocardial |
| Stimulus region | — | Left 5 columns | — | x < 5*dx = 0.125 cm |
| Stimulus start | — | 1.0 | ms | |
| Stimulus duration | — | 2.0 | ms | |
| Stimulus amplitude | — | -80.0 | uA/uF | Strong depolarizing |

### CV Measurement Protocol

1. Record voltage history at `save_every = 0.5 ms` intervals
2. Activation time = first snapshot where `Vm > -30 mV` at a node
3. Measure CV between x-indices `x1=30` and `x2=80` (nodes 0.75 cm to 2.0 cm)
   - Avoids stimulus region (x < 0.125 cm) and right-edge effects
4. Two y-locations per config:
   - **Center:** `y = Ny/2 = 20` (far from transverse boundaries)
   - **Edge:** `y = 1` (first interior row, adjacent to bottom boundary)
5. `CV = (x2 - x1) * dx / (t_act(x2) - t_act(x1))` in cm/ms
6. `CV_ratio = CV_edge / CV_center`

### Predicted CV_ratio by Configuration

| Config | Engine | BCs | Expected CV_ratio | Physical basis |
|--------|--------|-----|-------------------|----------------|
| A | LBM D2Q5 | Neumann | 1.00 +/- 0.02 | No artifact, no physics |
| B | LBM D2Q9 | Neumann | ~0.97 | Diagonal bounce-back artifact |
| C | Bidomain FDM | Insulated | 1.00 +/- 0.02 | No bath coupling |
| D | Bidomain FDM | Bath-coupled | ~1.13 | Kleber speedup |

---

## 4. Phase Structure

### Phase 6A: Convention Fix Verification

**Goal:** Confirm chi=1.0 gives correct operator scaling and diffusion rate.
Lightweight tests (seconds, no ionic model needed).

**Test script:** `tests/test_phase6a_convention.py`

| # | Test | What It Checks | Pass Criteria |
|---|------|---------------|---------------|
| 6A-T1 | Operator diagonal scaling | A_para diagonal at interior node with chi=1.0 vs chi=1400 | chi=1.0: diag ~ 1/dt = 100; chi=1400: diag ~ 140000 |
| 6A-T2 | L_i applied to quadratic | L_i * x^2 = 2*D_i at interior nodes | Relative error < 1% |
| 6A-T3 | Bidomain Gaussian diffusion | 100 diffusion-only steps, variance growth rate | sigma^2 grows as 2*D_eff*t, error < 5% |
| 6A-T4 | LBM Gaussian diffusion | Same Gaussian, same D_eff, LBM D2Q5 | Variance matches bidomain within 5% |

**Implementation notes:**
- 6A-T3 constructs DecoupledBidomainDiffusionSolver directly, bypasses ionic
- 6A-T3 uses a small grid (50x50) with the Gaussian centered, runs 100 steps
- 6A-T4 uses the LBM pure diffusion path (no ionic model), same grid and D
- Both use `sigma_0^2 = (5*dx)^2`, measure after t = 100*dt = 1.0 ms
- Expected: `sigma_final^2 = sigma_0^2 + 2*D_eff*1.0`

---

### Phase 6B: CV Calibration

**Goal:** Verify all three engines produce similar absolute interior CV
with D_eff before comparing boundary effects.

**Test script:** `tests/test_phase6b_calibration.py`

| # | Test | Engine + Config | Pass Criteria |
|---|------|----------------|---------------|
| 6B-T1 | LBM D2Q5 center CV | D2Q5, D_eff, Neumann | CV in [0.03, 0.15] cm/ms |
| 6B-T2 | LBM D2Q9 center CV | D2Q9, D_eff, Neumann | Within 5% of 6B-T1 |
| 6B-T3 | Bidomain insulated center CV | FDM, D_i+D_e, insulated, chi=1.0 | Within 15% of 6B-T1 |
| 6B-T4 | Calibration summary | Compare all three | Print table, flag outliers |

**Implementation notes:**
- Each config runs a full TTP06 wave simulation (Nx=150, Ny=40, t_end=40ms)
- CV measured at y=Ny/2 only (center)
- LBM configs use D_eff directly; bidomain uses D_i, D_e (harmonic mean = D_eff)
- If bidomain CV differs by >15%, the convention fix may be incomplete

**Expected runtime:** ~2-5 minutes per config (TTP06 with 4000 time steps on 6000 nodes)

---

### Phase 6C: Boundary CV Cross-Validation (Main Experiment)

**Goal:** Run all 4 configurations and measure boundary vs interior CV.
This is the core experiment that tests both the D2Q9 artifact hypothesis
and the Kleber speedup hypothesis.

**Test script:** `tests/test_phase6c_boundary_cv.py`

| # | Test | Config | Pass Criteria |
|---|------|--------|---------------|
| 6C-T1 | LBM D2Q5 boundary ratio | Config A: D2Q5 Neumann | 0.97 < ratio < 1.03 |
| 6C-T2 | LBM D2Q9 boundary ratio | Config B: D2Q9 Neumann | ratio < 1.00 (slowdown) |
| 6C-T3 | Bidomain insulated ratio | Config C: FDM insulated | 0.97 < ratio < 1.03 |
| 6C-T4 | Bidomain bath ratio | Config D: FDM bath-coupled | ratio > 1.05 (speedup) |
| 6C-T5 | Artifact vs Kleber direction | Compare B and D | B < 1.0 and D > 1.0 |
| 6C-T6 | Cross-engine calibration | A vs C center CV | Relative diff < 15% |

**Implementation notes:**
- Configs A/B reuse LBM_V1 `LBMSimulation` class
- Configs C/D use `BidomainSimulation` with chi=1.0, Cm=1.0
- All configs use identical Nx, Ny, dx, dt, TTP06, stimulus protocol
- CV measured at y=Ny/2 (center) and y=1 (edge)
- Command-line interface: `python test_phase6c_boundary_cv.py [a|b|c|d]`
  to run individual configs (useful if machine can only handle one at a time)

**Expected runtime:** ~10-20 minutes total (4 configs x 2-5 min each)

---

### Phase 6D: Mesh Convergence Study

**Goal:** Verify that the D2Q9 artifact vanishes with mesh refinement
(confirming it's O(dx^2) numerical) while the Kleber effect persists
(confirming it's physical).

**Test script:** `tests/test_phase6d_convergence.py`

Three resolutions:

| Level | dx (cm) | Nx | Ny | t_end (ms) | Notes |
|-------|---------|----|----|------------|-------|
| Coarse | 0.050 | 75 | 20 | 40 | Fast |
| Medium | 0.025 | 150 | 40 | 40 | Standard (= Phase 6C) |
| Fine | 0.0125 | 300 | 80 | 40 | Expensive |

| # | Test | What It Checks | Pass Criteria |
|---|------|---------------|---------------|
| 6D-T1 | D2Q9 artifact convergence | Config B at 3 resolutions | |1-ratio| shrinks ~4x per halving |
| 6D-T2 | Kleber convergence | Config D at 3 resolutions | ratio converges to ~1.13 |
| 6D-T3 | D2Q5 mesh independence | Config A at 3 resolutions | ratio stays ~1.00 |
| 6D-T4 | Bidomain insulated independence | Config C at coarse + medium | ratio stays ~1.00 |

**Implementation notes:**
- Fine resolution (300x80 = 24,000 nodes) may be very slow for bidomain PCG
- Fine bidomain is optional — skip if runtime exceeds 30 minutes
- Command-line: `python test_phase6d_convergence.py [coarse|medium|fine]`

**Expected runtime:**
- Coarse: ~30s per config
- Medium: ~3 min per config
- Fine: ~15 min (LBM) / ~60 min (bidomain, optional)

---

## 5. File Structure

```
Bidomain/Engine_V1/tests/
  cv_shared.py                    # Shared parameters and utilities
  test_phase6a_convention.py      # Chi*Cm fix verification (seconds)
  test_phase6b_calibration.py     # CV calibration across engines (minutes)
  test_phase6c_boundary_cv.py     # Main 4-config experiment (10-20 min)
  test_phase6d_convergence.py     # Mesh refinement study (30+ min)
```

---

## 6. Success Criteria Summary

### Primary Hypotheses

| ID | Hypothesis | Quantitative Criterion | Status |
|----|-----------|----------------------|--------|
| H1 | D2Q5 Neumann produces no boundary CV effect | CV_ratio = 1.00 +/- 0.03 | |
| H2 | D2Q9 Neumann produces edge *slowdown* (artifact) | CV_ratio < 1.00 | |
| H3 | Bidomain insulated produces no boundary CV effect | CV_ratio = 1.00 +/- 0.03 | |
| H4 | Bidomain bath-coupled produces edge *speedup* (Kleber) | CV_ratio > 1.05 | |
| H5 | D2Q9 artifact and Kleber are in opposite directions | H2 < 1.0 AND H4 > 1.0 | |
| H6 | D2Q9 artifact vanishes with refinement | |1-ratio_B| ~ O(dx^2) | |
| H7 | Kleber effect persists with refinement | ratio_D -> ~1.13 | |

### Prerequisite Checks

| ID | Check | Criterion |
|----|-------|-----------|
| P1 | Convention fix | A_para diagonal ~ 1/dt with chi=1.0 |
| P2 | Diffusion rate | Gaussian variance grows as 2*D_eff*t |
| P3 | CV calibration | All engines within 15% at domain center |

---

## 7. Execution Order

```
Phase 6A (seconds)   → Must pass before proceeding
Phase 6B (minutes)   → Must pass before proceeding
Phase 6C (10-20 min) → Main results
Phase 6D (30+ min)   → Confirmation / publication-quality evidence
```

Each phase is independently runnable. Phase 6C can be run one config at a time
via command-line arguments if the machine is resource-constrained.

---

## 8. Theoretical Reference

### Kleber Speedup Derivation

Interior:
```
sigma_eff = sigma_i * sigma_e / (sigma_i + sigma_e)
D_eff = sigma_eff / (chi * Cm) = 0.000970 cm^2/ms
```

Boundary (bath shorts extracellular):
```
sigma_boundary = sigma_i
D_boundary = sigma_i / (chi * Cm) = D_i = 0.00124 cm^2/ms
```

CV ratio:
```
CV_boundary / CV_interior = sqrt(D_boundary / D_eff)
                          = sqrt(D_i / D_eff)
                          = sqrt((D_i + D_e) / D_e)
                          = sqrt(0.00570 / 0.00446)
                          = sqrt(1.278) = 1.131
```

### D2Q9 Bounce-Back Artifact

At a Neumann boundary in D2Q9, the 4 diagonal distributions (f_5, f_6, f_7, f_8)
undergo bounce-back when their target is outside the domain. This bounce-back is
*exact* for the cardinal distributions but introduces O(dx^2) error for diagonals
because the diagonal bounce-back path doesn't perfectly align with the zero-flux
condition on a Cartesian boundary.

The net effect is a slight reduction in effective diffusivity at boundary rows,
causing ~3% CV slowdown. This error scales as dx^2 and vanishes in the continuum
limit. D2Q5 has no diagonal distributions and thus no such artifact.
