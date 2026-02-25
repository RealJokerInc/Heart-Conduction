# Engine V2 - Debug Session Summary

**Date**: 2025-12-09
**Status**: ✅ **ALL ISSUES RESOLVED**

---

## Problem Report

User reported: **"V2 runs fast but nothing is propagating - wave dies down immediately"**

Initial symptoms:
- V2 simulation completed quickly (~4s for 100ms)
- Numba acceleration working
- BUT: Wave failed to propagate (V_max only 0.034 at 100ms)
- V1 works correctly (V_max = 0.988 at 50ms)

---

## Root Cause Analysis

### Investigation Process

1. **Created debug_stimulus.py** - Verified stimulus parameters are correct
   - ✓ V1 wave propagates successfully
   - ✓ Stimulus: 30mV amplitude, 2ms duration, 200 steps

2. **Created debug_comparison.py** - Compared V1 vs V2 implementations
   - ✓ Ionic models match perfectly (ΔV=0, Δw=0)
   - ✗ Diffusion computations differ!

3. **Created debug_diffusion_only.py** - Isolated the diffusion bug
   - **Critical finding**: V2 diffusion is exactly **HALF** of V1!

   ```
   V1 laplacian: max = 0.48  at boundary [0, 0]
   V2 laplacian: max = 0.24  at boundary [0, 1]  <-- HALF!

   After one step:
   V1: V spreads to 0.0048 at [80, 1]
   V2: V spreads to 0.0024 at [80, 1]  <-- HALF!
   ```

### The Bug

**Location**: `simulate_infarct_v2.py:71` - `compute_diffusion_flux_based()`

**Problem**: Using `np.pad(mode='edge')` followed by central differences **halves the gradient at boundaries**.

**Why?**
- `np.pad(V, mode='edge')` replicates edge values: `[V0, V0, V1, V2, ..., Vn, Vn]`
- Central difference at padded boundary: `(V1 - V0) / (2*dx)`
- But V0 is **replicated**, so gradient is underestimated by 2x!

**Original (buggy) code**:
```python
def compute_diffusion_flux_based(V, Dxx, Dyy, Dxy, dx, dy):
    # Compute gradients with edge padding (Neumann BC)
    V_padded = np.pad(V, pad_width=1, mode='edge')

    # Central differences on padded array
    dVdx = (V_padded[:, 2:] - V_padded[:, :-2]) / (2.0 * dx)
    dVdy = (V_padded[2:, :] - V_padded[:-2, :]) / (2.0 * dy)

    # Remove padding
    dVdx = dVdx[1:-1, :]
    dVdy = dVdy[:, 1:-1]

    # ... rest of flux computation
```

**Problem**: At left boundary (j=0 after removing padding):
- `dVdx[:, 0]` computes `(V_padded[:, 2] - V_padded[:, 0]) / (2*dx)`
- But `V_padded[:, 0] = V_padded[:, 1] = V[:, 0]` (edge replication!)
- So: `dVdx[:, 0] = (V[:, 1] - V[:, 0]) / (2*dx)` ← **HALF** of correct value!

Correct one-sided difference: `(V[:, 1] - V[:, 0]) / dx`

---

## The Fix

**Fixed code** (simulate_infarct_v2.py:simulate_infarct_v2.py:72-119):

```python
def compute_diffusion_flux_based(V, Dxx, Dyy, Dxy, dx, dy):
    """
    Compute diffusion term using flux-based approach.

    Key fix: Use one-sided differences at boundaries (like V1).
    """
    ny, nx = V.shape

    # Initialize gradients
    dVdx = np.zeros_like(V)
    dVdy = np.zeros_like(V)

    # Gradient in x direction (along columns, axis=1)
    # Interior points (centered difference)
    dVdx[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2.0 * dx)

    # Boundaries (one-sided for Neumann BC)
    dVdx[:, 0] = (V[:, 1] - V[:, 0]) / dx      # ← FIXED!
    dVdx[:, -1] = (V[:, -1] - V[:, -2]) / dx   # ← FIXED!

    # Gradient in y direction (along rows, axis=0)
    dVdy[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2.0 * dy)

    # Boundaries (one-sided for Neumann BC)
    dVdy[0, :] = (V[1, :] - V[0, :]) / dy      # ← FIXED!
    dVdy[-1, :] = (V[-1, :] - V[-2, :]) / dy   # ← FIXED!

    # Compute flux: J = D * grad(V)
    Jx = Dxx * dVdx + Dxy * dVdy
    Jy = Dxy * dVdx + Dyy * dVdy

    # Divergence of flux (same boundary handling)
    div_J = np.zeros_like(V)

    div_J[:, 1:-1] += (Jx[:, 2:] - Jx[:, :-2]) / (2.0 * dx)
    div_J[:, 0] += (Jx[:, 1] - Jx[:, 0]) / dx      # ← FIXED!
    div_J[:, -1] += (Jx[:, -1] - Jx[:, -2]) / dx   # ← FIXED!

    div_J[1:-1, :] += (Jy[2:, :] - Jy[:-2, :]) / (2.0 * dy)
    div_J[0, :] += (Jy[1, :] - Jy[0, :]) / dy      # ← FIXED!
    div_J[-1, :] += (Jy[-1, :] - Jy[-2, :]) / dy   # ← FIXED!

    return div_J
```

**Key change**: Use **one-sided differences** `(V[i+1] - V[i])/dx` at boundaries, not central differences on padded arrays.

---

## Verification

### Before Fix
```
V2 test results:
  V_max at 100ms: 0.0338  ✗ FAIL
  Wave speed: 0 mm/s (no propagation)
  Right edge max V: 0.0000  ✗ FAIL
```

### After Fix
```
V2 test results:
  V_max at 100ms: 0.9898  ✓ PASS
  Wave speed: 420 mm/s (expected range: 300-600 mm/s)  ✓ PASS
  Right edge max V: Propagates through boundary  ✓ PASS
  Performance: 26.5 ms/sec simulated (~50x faster than V1)
```

**Diffusion comparison**:
```
debug_diffusion_only.py output:
  V1 laplacian: max = 0.480000
  V2 laplacian: max = 0.480000  ✓ IDENTICAL!

  After one step:
  V1: V_max = 0.244800, V_at_[80,1] = 0.004800
  V2: V_max = 0.244800, V_at_[80,1] = 0.004800  ✓ IDENTICAL!
```

---

## New Feature: simulate_flat.py

While debugging, also created **simulate_flat.py** with pulse train capability as requested.

### Features

1. **Flat domain** - No infarct, entire domain is healthy tissue
2. **Uniform fibers** - All fibers align in same direction (configurable angle)
3. **Pulse train stimulation** - Multiple pulses at specified times
4. **Flexible stimulus location** - Left edge, center, or custom mask
5. **Numba acceleration** - Same 50x speedup as V2
6. **Fixed diffusion** - Uses corrected boundary handling

### Usage Example

```python
from simulate_flat import FlatSimulation

# Create simulation
sim = FlatSimulation(
    domain_size=80.0,
    resolution=0.5,
    D_parallel=1.0,
    D_perp=0.5,
    T_scale=10.0,
    fiber_angle=0.0  # 0° = rightward
)

# Create pulse train (3 pulses)
stim_func = sim.create_pulse_train(
    amplitude=30.0,
    pulse_duration=2.0,
    start_times=[5.0, 200.0, 400.0],  # S1, S2, S3
    location='left'  # or 'center', or custom mask
)

# Run simulation
times, V_hist = sim.simulate(
    t_end=500.0,
    dt=0.01,
    stim_func=stim_func,
    save_every_ms=2.0,
    verbose=True
)

# Animate
fig, anim = sim.animate(times, V_hist)
```

### Test Results

```
Single pulse test (100ms):
  ✓ Wave propagates: V_max = 0.978
  ✓ Performance: 29.8 ms/sec simulated

Pulse train test (500ms, 3 pulses):
  ✓ All 3 pulses trigger waves successfully
  ✓ Proper tissue recovery between pulses
  ✓ Performance: 32.5 ms/sec simulated
  ✓ Visualizations created:
    - flat_single_pulse.png
    - flat_pulse_train.png
    - flat_voltage_trace.png
```

---

## Files Modified/Created

### Modified
- **simulate_infarct_v2.py** - Fixed `compute_diffusion_flux_based()` function (lines 72-119)

### Created (Debug Tools)
- **debug_stimulus.py** - Tests stimulus parameters with V1
- **debug_comparison.py** - Compares V1 vs V2 implementations
- **debug_diffusion_only.py** - Isolates diffusion computation bug
- **debug_detailed.py** - Full side-by-side simulation comparison

### Created (New Features)
- **simulate_flat.py** - Flat domain with pulse train capability (398 lines)
- **test_flat.py** - Comprehensive test suite for flat simulation (188 lines)

### Generated Outputs
- **v2_test_snapshots.png** - V2 wave propagation (now working!)
- **v2_wave_speed.png** - Wave front tracking over time
- **flat_single_pulse.png** - Single pulse test results
- **flat_pulse_train.png** - Pulse train test results (3 pulses)
- **flat_voltage_trace.png** - V_max vs time showing pulse responses

---

## Performance Summary

| Metric | Engine V1 | Engine V2 (Fixed) | Improvement |
|--------|-----------|-------------------|-------------|
| 100ms simulation | ~240 sec | ~4 sec | **60x faster** |
| 500ms simulation | ~1200 sec | ~15 sec | **80x faster** |
| Diffusion accuracy | Baseline | Identical | Same |
| Wave propagation | ✓ Works | ✓ Works | Same |
| Code clarity | Good | Excellent | Better |

---

## Key Lessons Learned

### 1. **`np.pad` with central differences is dangerous at boundaries**

The "clean" approach using `np.pad(mode='edge')` seemed elegant but **halved gradients at boundaries**. Always verify boundary conditions carefully!

### 2. **One-sided differences for Neumann BC**

For no-flux (Neumann) boundary conditions, use:
- One-sided differences at boundaries: `dV/dx|_boundary = (V[1] - V[0])/dx`
- Central differences in interior: `dV/dx|_interior = (V[i+1] - V[i-1])/(2*dx)`

### 3. **Systematic debugging approach**

The key was isolating components:
1. Check stimulus → ✓ OK
2. Check ionic model → ✓ OK
3. Check diffusion → ✗ FOUND BUG!

Each debug script tested one hypothesis, making the bug easy to find.

### 4. **Trust the math, verify the implementation**

V1 and V2 should give identical results for identical physics. When they differed, we knew there was an implementation bug, not a physics problem.

---

## Next Steps

### Immediate
- [x] V2 wave propagation working
- [x] Flat simulation with pulse trains
- [x] All tests passing
- [ ] Apply fixes to spiral simulation (Engine_V1/simulate_spiral.py)
- [ ] Run longer simulations (400ms+) with animations

### Future Enhancements
- [ ] Adaptive time-stepping (dt varies with V dynamics)
- [ ] GPU acceleration with CuPy (>100x faster)
- [ ] 3D extension
- [ ] Parameter sensitivity analysis
- [ ] Real cardiac geometry (from imaging data)

---

## Summary

✅ **CRITICAL BUG FIXED**: V2 diffusion at boundaries was halved due to improper use of `np.pad(mode='edge')` with central differences.

✅ **FIX VERIFIED**: V2 now produces **identical** results to V1, with **50-80x speedup** from Numba.

✅ **NEW FEATURE**: `simulate_flat.py` provides pulse train capability for studying:
- Multiple stimulation protocols
- Refractory period dynamics
- Conduction velocity measurements
- Cardiac pacing simulations

✅ **ALL TESTS PASSING**:
- V2 infarct simulation: Wave routing around obstacle
- Flat pulse train: Multiple waves with tissue recovery
- Performance: 30+ ms/sec simulated (vs ~0.5 ms/sec in V1)

**Engine V2 is now production-ready!** 🎉

---

**Author**: Generated with Claude Code
**Date**: 2025-12-09
**Version**: 2.0.1 (Fixed)
