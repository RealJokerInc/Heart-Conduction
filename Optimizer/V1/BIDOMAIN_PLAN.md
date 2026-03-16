# Bidomain Optimizer Pipeline — Design Plan

## Overview

Extend the Optimizer V1 pipeline to support the Bidomain V1 engine alongside the existing Monodomain V5.4 engine. The cell fitter (single-cell BayesOpt) is engine-agnostic; only the tissue runner and tissue fitter change.

## What Changes, What Stays

| Component | Change | Notes |
|-----------|--------|-------|
| `config.py` | **Extend** | Add `engine`, `D_i`, `D_e`, `De_Di_ratio` fields |
| `batch_ionic.py` | None | Single-cell ODE, no tissue coupling |
| `cell_runner.py` | None | Uses batch_ionic, engine-agnostic |
| `cell_fitter.py` | None | Calls cell_runner, engine-agnostic |
| `metrics.py` | None | AP biomarker extraction from V(t) traces |
| `tissue_runner.py` | **Keep** as monodomain runner | Rename function for clarity |
| `tissue_runner_bidomain.py` | **New** | Wraps BidomainSimulation engine for CV measurement |
| `tissue_fitter.py` | **Adapt** | Engine-aware: uses D for mono, (D_i, D_e) for bidomain |
| `pipeline.py` | **Adapt** | Engine dispatch |
| `run_mhas13_bidomain.py` | **New** | Bidomain pipeline run script |

## Bidomain Engine Parameters

### Reference Values (from `Bidomain/Engine_V1/tests/cv_shared.py`)

```
σ_i = 1.74 mS/cm     →  D_i = σ_i/(χ·Cm) = 0.00124 cm²/ms
σ_e = 6.25 mS/cm     →  D_e = σ_e/(χ·Cm) = 0.00446 cm²/ms
χ   = 1400 cm⁻¹          (surface-to-volume ratio)
Cm  = 1.0 μF/cm²         (membrane capacitance)

D_eff = D_i·D_e/(D_i+D_e) = 0.000970 cm²/ms
De/Di ratio = 3.597

Validated CV (TTP06, insulated): 54.3 cm/s at dx=0.025, dt=0.01
```

### Convention: chi=1, Cm=1 in operators

The Bidomain V1 engine uses **Formulation B**: D = σ/(χ·Cm) pre-scaled.
- chi and Cm do NOT appear in Laplacian operators
- D_i, D_e are already physical diffusivities
- This matches the monodomain convention: D = σ/(χ·Cm)

### Bidomain vs Monodomain CV Relationship

For **insulated** tissue (Neumann on all boundaries):
```
CV_bidomain ≈ CV_monodomain(D_eff)    where D_eff = D_i·D_e/(D_i+D_e)
```

For **bath-coupled** tissue (Dirichlet phi_e at boundaries):
```
CV_boundary ≈ CV_center × √((D_i+D_e)/D_e)    (Kleber effect: ~7-13% speedup)
```

The optimizer tunes D_i and D_e subject to a ratio constraint.

## Config Changes

```python
@dataclass
class TuningConfig:
    # ... existing fields ...

    # Engine selection
    engine: str = 'monodomain'      # 'monodomain' | 'bidomain'

    # Bidomain conductivities (only used when engine='bidomain')
    D_i: float = 0.00124            # cm²/ms (intracellular)
    D_e: float = 0.00446            # cm²/ms (extracellular)
    De_Di_ratio: float = 3.597      # D_e/D_i ratio (constrained during fitting)
    bc_type: str = 'insulated'      # 'insulated' | 'bath'

    # Bidomain solver options
    splitting: str = 'strang'       # 'strang' | 'godunov'
    elliptic_solver: str = 'auto'   # 'auto' | 'spectral' | 'pcg_spectral' | 'pcg_gmg'
```

## Bidomain Tissue Runner

### Design

Use the Bidomain V1 engine's `BidomainSimulation` directly. This gives us:
- Decoupled Gauss-Seidel splitting (parabolic Vm → elliptic phi_e)
- Three-tier elliptic solver (spectral / PCG+spectral / PCG+GMG)
- Proper boundary condition handling

### Cable Geometry

For CV measurement, use a **narrow 2D strip** (not true 1D — bidomain needs 2D for the elliptic solve):

```
Nx = int(cable_length / dx) + 1     # e.g., 38 nodes at 1.5cm/0.04cm
Ny = 5                               # Narrow strip (5 nodes wide)
```

The CV is measured from activation times at probe points along the center row (y_center = Ny//2).

### Implementation: `tissue_runner_bidomain.py`

```python
def run_cv_measurement_bidomain(
    theta_ionic: torch.Tensor,
    D_i: float, D_e: float,
    config: TuningConfig,
    cable_length_cm: float = None,
    dx_cm: float = None,
    n_beats: int = 3,
) -> CVResult:
    """
    Measure CV using Bidomain V1 engine.

    1. Build StructuredGrid (narrow strip)
    2. Build BidomainConductivity(D_i, D_e)
    3. Build BidomainFDMDiscretization
    4. Build StimulusProtocol (left edge pacing)
    5. Build BidomainSimulation with MHAS13 ionic model
    6. Run and extract Vm activation times at probe points
    7. Compute CV = distance / (t_probe2 - t_probe1)
    """
```

### MHAS13 Availability in Bidomain Engine

The MHAS13 model needs to be available in `Bidomain/Engine_V1/cardiac_sim/ionic/`. Options:

**Option A: Copy phas13 + mhas13 packages** (recommended for V1)
- Copy `ionic/phas13/` and `ionic/mhas13/` from Monodomain to Bidomain
- Both share the same ABC (`IonicModel`) and utility functions (`safe_exp`, `rush_larsen`)
- Self-contained, no cross-engine imports

**Option B: Shared ionic package** (future V2)
- Extract ionic models to a shared `cardiac_sim_common/ionic/` package
- Both engines import from the shared package
- Requires refactoring import paths in both engines

For this implementation, use **Option A**.

## Tissue Fitter Adaptation

### Single-Variable Fitting (D_eff)

For bidomain, CV depends on D_eff = D_i·D_e/(D_i+D_e). Given the ratio constraint r = D_e/D_i:

```
D_eff = D_i · r / (1 + r)      →  D_i = D_eff · (1 + r) / r
                                    D_e = D_eff · (1 + r)
```

So the fitting procedure is:
1. Fix r = De_Di_ratio (default 3.597)
2. Measure CV at a reference D_eff
3. Analytically compute D_eff_target = (CV_target/CV_ref)² × D_eff_ref
4. Back-compute D_i, D_e from D_eff_target and r
5. Verify with one bidomain sim

This reduces bidomain tissue fitting to the **same 1D problem** as monodomain, just with the D↔(D_i,D_e) mapping on top.

### Anisotropy (Future)

For anisotropic tissue:
- D_i_long, D_i_trans with AR_i = D_i_long/D_i_trans
- D_e_long, D_e_trans with AR_e = D_e_long/D_e_trans
- Four parameters, two CV targets (CV_long, CV_trans)
- Not in this implementation; uses isotropic D_i, D_e

## Updated Code Structure

```
Optimizer/V1/
  tuner/
    __init__.py
    config.py                  # + engine, D_i, D_e, De_Di_ratio, bc_type
    metrics.py                 # unchanged
    batch_ionic.py             # unchanged (single-cell)
    cell_runner.py             # unchanged (single-cell)
    cell_fitter.py             # unchanged
    tissue_runner.py           # monodomain runner (existing, minor rename)
    tissue_runner_bidomain.py  # NEW: wraps BidomainSimulation
    tissue_fitter.py           # adapted: engine dispatch, D_eff mapping
    joint_refiner.py           # unchanged
    validator.py               # adapted: engine-aware CV verification
    pipeline.py                # adapted: engine dispatch
  run_mhas13.py                # monodomain pipeline (existing)
  run_mhas13_bidomain.py       # NEW: bidomain pipeline
```

## Implementation Steps

### Step 1: Copy PHAS13 + MHAS13 to Bidomain Engine
- Copy `ionic/phas13/` (6 files) and `ionic/mhas13/` (4 files) into `Bidomain/Engine_V1/cardiac_sim/ionic/`
- Update `Bidomain/.../ionic/__init__.py` to import MHAS13Model
- Verify: `BidomainSimulation(..., ionic_model=mhas13_instance, ...)` works

### Step 2: Create `tissue_runner_bidomain.py`
- Use BidomainSimulation API (from exploration report above)
- Build narrow 2D strip grid
- Pace from left edge, measure Vm activation at probe points
- Return CVResult

### Step 3: Extend `config.py`
- Add bidomain fields with defaults from cv_shared.py
- Add `engine` field

### Step 4: Adapt `tissue_fitter.py`
- Engine dispatch: mono → D fitting, bidomain → D_eff fitting with ratio constraint
- D_eff → (D_i, D_e) back-computation

### Step 5: Create `run_mhas13_bidomain.py`
- Same pipeline structure as `run_mhas13.py`
- Uses bidomain tissue runner
- Reports D_i, D_e, D_eff, CV

### Step 6: Test and Validate
- Verify bidomain CV with TTP06 matches cv_shared (54.3 cm/s)
- Run MHAS13 bidomain pipeline
- Compare mono vs bidomain CV/APD results

## Expected Results

| Metric | Monodomain (current) | Bidomain (expected) |
|--------|---------------------|-------------------|
| APD90 | 347 ms | ~347 ms (same ionic model) |
| dV/dt | 98 V/s | ~98 V/s (same ionic model) |
| CV_long | 15.8 cm/s | ~15 cm/s (target) |
| D | 0.000447 cm²/ms | D_eff ≈ 0.000447 cm²/ms |
| D_i | — | D_eff·(1+r)/r ≈ 0.000572 cm²/ms |
| D_e | — | D_eff·(1+r) ≈ 0.00206 cm²/ms |

The cell-level results (APD, dV/dt) should be **identical** — the ionic model doesn't know about the tissue solver. Only the D values change representation.

## Timeline Estimate

| Step | Files | Effort |
|------|-------|--------|
| Copy ionic models | 10 files + __init__.py | Small |
| tissue_runner_bidomain.py | 1 file (~100 lines) | Medium |
| config.py extension | Edits | Small |
| tissue_fitter.py adaptation | Edits (~30 lines) | Small |
| run_mhas13_bidomain.py | 1 file (~150 lines) | Medium |
| Testing | Verification runs | Medium |
