# Voltage Bounds Investigation - Findings

**Date**: 2025-12-09
**Issue**: Voltage exceeding expected [0, 1] dimensionless range

---

## 🎯 User-Reported Issues

1. **V_min consistently at -121.2 mV** (expected: -85 mV)
2. **V_max around 1.137 dimensionless** (57.1 mV physical, expected: ≤40 mV)
3. **Concern**: -121.2 mV might be crash threshold

---

## 📊 Investigation Results

### Finding #1: -121.2 mV is the Safeguard Floor ✓

**Analysis**:
```
Safeguard (from Fix #2): V ≥ -mu2 + 0.01 = -0.29 (dimensionless)

Physical conversion:
V_phys = V_rest + V_norm * (V_peak - V_rest)
       = -85 + (-0.29) * 125
       = -85 + (-36.25)
       = -121.25 mV ≈ -121.2 mV
```

**Conclusion**:
- -121.2 mV is NOT a natural voltage value
- It's the ARTIFICIAL FLOOR from the safeguard
- If voltage reaches this consistently, physics is trying to push lower
- **Safeguard is preventing crashes, but underlying issue remains**

---

### Finding #2: V_max Overshoot CONFIRMED ⚠️

**Test Results**:
- Test with 200mV stimulus, 100ms propagation:
  - **V_max = 1.1142** (dimensionless)
  - **V_max = 54.3 mV** (physical)
  - **Overshoot: 14.3 mV above expected 40 mV peak**

- Peak occurred at t=70-80ms (during wave propagation)
- User reports V_max ~1.137 (57.1 mV), even higher

**Expected Behavior**:
- Aliev-Panfilov model: V should stay in [0, 1]
- Physical voltage: should stay in [-85, +40] mV
- **This is violated!**

---

### Finding #3: Safeguard Activation Status

**Different scenarios**:

| Stimulus | V_min Reached | At Floor? | V_max Reached | Overshoot? |
|----------|---------------|-----------|---------------|------------|
| 100 mV   | -85.2 mV      | ❌ No     | -64.6 mV      | ❌ No AP   |
| 200 mV   | -85.0 mV      | ❌ No     | 54.3 mV       | ✅ Yes     |
| User obs | -121.2 mV     | ✅ YES!   | 57.1 mV       | ✅ Yes     |

**Interpretation**:
- In some conditions, voltage DOES hit the -121.2 mV floor
- This happens in user's interactive simulation but not all test scenarios
- Likely related to:
  - Longer simulation times
  - Multiple stimuli
  - Specific spatial patterns
  - Parameter combinations

---

## 🔍 Root Cause Analysis

### Why V_max Overshoots Above 1.0?

**Hypothesis 1: Stimulus Overapplication**
```python
# Stimulus is added to dimensionless V:
I_stim = sim.ionic_model.physical_to_voltage(I_stim_physical)

# If stimulus is very strong (200mV), and applied during depolarization:
V = 0.5 + 0.2 (from 200mV stimulus) → V = 0.7
```
But this should normalize correctly. ❌ Unlikely

**Hypothesis 2: Ionic Current Formulation**
```python
I_ion = k * v * (v - a) * (v - 1.0) + v * w

# For v slightly > 1:
# First term: k * v * (v - a) * (v - 1) = positive (repolarizing)
# Second term: v * w = positive (also repolarizing)
```
Model should naturally limit V ≤ 1. ❌ Unlikely

**Hypothesis 3: Time Step Too Large** ⚠️ LIKELY
```python
# Explicit Euler: V_{n+1} = V_n + dt * dV/dt

# If dV/dt is very large (during rapid depolarization):
dV/dt ≈ 100 (fast upstroke)

# With dt = 0.01 ms:
V_{n+1} = V_n + 0.01 * 100 = V_n + 1.0

# Can easily overshoot if V_n = 0.3!
```
**This is the most likely cause.** ✅

**Hypothesis 4: Diffusion Adding Extra Current** ⚠️ POSSIBLE
```python
# Diffusion step BEFORE ionic step:
V += dt * div(D∇V)

# If neighbors have high V, diffusion adds positive flux
# Then ionic step adds stimulus
# Combined effect can push V > 1
```
Possible contributing factor. ⚠️

---

### Why V_min Hits Floor (-121.2 mV)?

**Hypothesis 1: Repolarization Undershoot** ⚠️ LIKELY
```python
# During repolarization, recovery variable w is large:
I_ion = k * v * (v - a) * (v - 1.0) + v * w
#                                      ^^^^^^ Large positive term

# If w gets too large, I_ion becomes very positive
# dV/dt = -I_ion → very negative
# V drops rapidly, potentially below 0
```

**Mechanism**:
1. AP upstroke: V → 1, w still small
2. Plateau: w starts growing (dwdt = epsilon * recovery_term)
3. Repolarization: V drops, but w keeps growing
4. **Overshoot**: w too large, pushes V below 0
5. **Safeguard**: Clamps at -0.29, preventing crash

**Why it's intermittent**:
- Depends on exact timing of w dynamics
- Spatial gradients can create local undershoots
- Multiple stimuli interactions can amplify effect

---

## 💡 Root Causes Identified

### Primary Issue: Time Step Too Large for Explicit Euler

**Evidence**:
1. V_max overshoots above 1.0 (rapid depolarization not captured)
2. V_min undershoots below 0 in some cases (rapid repolarization)
3. Overshoots increase during fast dynamics (wave propagation)

**Current time step**: dt = 0.01 ms

**CFL condition** for diffusion:
```
dt < dx² / (2 * D_max)
dx = 0.5 mm, D = 1.0 mm²/ms
dt_max = 0.25² / 2 = 0.125 ms
```
We're well below this (dt = 0.01), so diffusion is stable.

**But**: Ionic dynamics are much faster than diffusion!
- Upstroke: dV/dt ≈ 100 /ms during depolarization
- For explicit Euler stability: dt * |dV/dt| should be << 1
- dt * 100 = 0.01 * 100 = 1.0 ⚠️  TOO LARGE!

---

### Secondary Issue: Recovery Variable Dynamics

**Evidence**:
1. V_min undershoots occur during repolarization phase
2. Related to w (recovery variable) growing too large
3. Voltage-dependent epsilon makes w dynamics nonlinear

**Contributing factors**:
- epsilon0 = 0.01 (increased from 0.002 in Fix #1)
- Faster recovery means w grows faster
- Can overshoot and push V negative

---

## 🎯 Proposed Fixes

### Fix #1: Reduce Time Step (Immediate) ✅

**Change**: `dt = 0.01 ms → 0.005 ms` (half current value)

**Rationale**:
- Explicit Euler needs dt * |dV/dt| < 0.5 for stability
- Current: 0.01 * 100 = 1.0 (marginally unstable)
- Proposed: 0.005 * 100 = 0.5 (stable)

**Impact**:
- 2x slower simulation (but still faster than V1)
- Should eliminate overshoots
- More accurate dynamics

**Implementation**:
```python
# In InteractiveSimulation.__init__:
self.dt = 0.005  # Changed from 0.01
```

---

### Fix #2: Add Voltage Clamping at Correct Bounds (Required) ✅

**Current safeguard**: V ≥ -0.29 (prevents crash, but wrong physics)

**Correct bounds**: V ∈ [0, 1] (as per Aliev-Panfilov model)

**Implementation**:
```python
# In ionic_step_numba, AFTER computing new V:
v_new = v + dt * dVdt
V[i, j] = max(0.0, min(v_new, 1.0))  # Clamp to [0, 1]

# Also clamp w:
w_new = w_val + dt * dwdt
w[i, j] = max(0.0, w_new)  # w ≥ 0
```

**Rationale**:
- Aliev-Panfilov model designed for V ∈ [0, 1]
- Explicit Euler can overshoot, clamping ensures bounds
- Prevents both overshoot AND undershoot

**Physical interpretation**:
- V = 0 → -85 mV (rest)
- V = 1 → +40 mV (peak)
- Clamping at [0, 1] → physical bounds [-85, +40] mV ✓

---

### Fix #3: Adjust epsilon0 Back Down (Optional) ⚠️

**Current**: epsilon0 = 0.01
**Original**: epsilon0 = 0.002

**Consideration**:
- Fix #1 increased epsilon0 to 0.01 for "stability"
- But this makes w recover FASTER
- Faster w → larger w values → more undershoot

**Trade-off**:
- Lower epsilon0: Slower recovery, longer APD, less undershoot
- Higher epsilon0: Faster recovery, shorter APD, more undershoot

**Recommendation**: Try epsilon0 = 0.005 (compromise)

---

### Fix #4: Adaptive Time Stepping (Advanced, Optional)

**Idea**: Reduce dt when |dV/dt| is large

```python
dVdt_max = np.max(np.abs(dVdt))
if dVdt_max > 50:
    dt_local = 0.5 * dt  # Use smaller step during fast dynamics
else:
    dt_local = dt
```

**Pros**: Automatic stability
**Cons**: Complexity, variable time

**Recommendation**: Try Fixes #1 and #2 first

---

## 📋 Implementation Plan

### Immediate (Required):

1. **Fix #1**: Reduce dt to 0.005 ms
2. **Fix #2**: Clamp V to [0, 1] and w to [0, ∞)

### Optional (If issues persist):

3. **Fix #3**: Adjust epsilon0 to 0.005
4. **Fix #4**: Implement adaptive time stepping

---

## 🧪 Expected Outcomes

After Fixes #1 and #2:
- ✅ V_max should stay ≤ 1.0 (≤40 mV physical)
- ✅ V_min should stay ≥ 0.0 (≥-85 mV physical)
- ✅ No more -121.2 mV floor hits
- ✅ More accurate AP dynamics
- ⚠️  Simulation 2x slower (but still fast with Numba)

---

## ❓ Questions for User

1. **Accept 2x slower simulation** for correct physics?
   - Alternative: Keep current speed but with bounded errors

2. **Preferred epsilon0 value**?
   - 0.005 (compromise)
   - 0.002 (original, slower recovery)
   - 0.01 (current, faster recovery)

3. **Try adaptive time stepping** if simple fixes insufficient?

---

**Author**: Generated with Claude Code
**Date**: 2025-12-09
**Status**: ⏸️ **AWAITING USER CONFIRMATION TO PROCEED WITH FIXES**
