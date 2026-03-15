# Physics Engine Stability Debug Plan

**Date**: 2025-12-09
**Issue**: Numerical instability causing simulation crashes
**Symptoms**:
- V_min drops below -200mV after action potentials
- Simulation crashes due to values "blowing up"
- Unclear if V_max or V_min causes crash

---

## 🎯 Objectives

1. **Identify the root cause** of numerical instability
2. **Determine failure mode**: Which variable blows up first?
3. **Find stability bounds** for all state variables
4. **Propose fixes** with minimal physics changes
5. **Validate fixes** don't alter correct wave propagation

---

## 📋 Debug Plan (Step-by-Step)

### Phase 1: Characterize the Failure

**Goal**: Document exactly when and how the simulation fails

#### Test 1.1: Monitor All Variables During Crash
```python
# Track min/max of V, w, dV/dt, dw/dt over time
# Record the exact sequence of events leading to crash
# Identify: Does V or w blow up first?
```

**Outputs**:
- Time series plot of V_min, V_max, w_min, w_max
- First derivative bounds (dV/dt, dw/dt)
- Exact crash time and values

**Questions to answer**:
- Does V go negative first, or positive infinity?
- Does w become negative or unbounded?
- Is crash immediate or gradual?
- Does crash happen during stimulus or after?

---

#### Test 1.2: Spatial Analysis of Instability
```python
# Where does instability start?
# - At stimulus site?
# - At wave front?
# - In recovered tissue?
# - At boundaries?
```

**Outputs**:
- Heatmap showing where variables first exceed bounds
- Spatial distribution of extreme values
- Correlation with stimulus locations

---

#### Test 1.3: Parameter Sensitivity
```python
# Test with different parameters:
# - Smaller dt (0.005, 0.001 ms)
# - Larger dt (0.05, 0.1 ms)
# - Different stimulus amplitudes
# - Different T_scale values
```

**Outputs**:
- Stability boundary in (dt, amplitude) space
- Critical time step for stability
- Parameter combinations that crash vs. survive

---

### Phase 2: Isolate the Component

**Goal**: Determine which part of the model causes instability

#### Test 2.1: Ionic Model Only (No Diffusion)
```python
# Run simulation with D = 0 (pure reaction)
# Apply uniform stimulus
# Does it still blow up?
```

**Purpose**: Isolate whether ionic model itself is unstable

**Expected outcome**:
- If crashes → Ionic model has stability issue
- If stable → Problem is in diffusion or coupling

---

#### Test 2.2: Diffusion Only (No Reaction)
```python
# Run simulation with ionic terms = 0
# Apply stimulus and watch diffusion only
# Does it blow up?
```

**Purpose**: Isolate whether diffusion term causes instability

**Expected outcome**:
- If crashes → Diffusion implementation has bug
- If stable → Problem is in ionic model or coupling

---

#### Test 2.3: Operator Splitting Order
```python
# Current: Diffusion → Reaction
# Test: Reaction → Diffusion
# Test: Strang splitting: R(dt/2) → D(dt) → R(dt/2)
```

**Purpose**: Check if operator splitting order matters

---

### Phase 3: Analyze Ionic Model

**Goal**: Verify Aliev-Panfilov model is implemented correctly

#### Test 3.1: Phase Plane Analysis
```python
# Plot nullclines: dV/dt = 0 and dw/dt = 0
# Verify fixed points are correct
# Check if trajectories escape to infinity
```

**Check against literature**:
- Aliev & Panfilov (1996) original paper
- Expected APD90 ≈ 250ms
- Expected resting potential: V = 0 (dimensionless)
- Expected peak: V ≈ 1.0

**Look for**:
- Incorrect signs in equations
- Missing terms
- Wrong parameter values causing unbounded solutions

---

#### Test 3.2: Verify Epsilon Function
```python
# Epsilon controls recovery rate
# eps = epsilon0 + epsilon_rest * sigmoid + (mu1 * w) / (v + mu2)
#
# CRITICAL: Check if (v + mu2) can be zero or negative!
# This would cause division by zero or sign flip
```

**Specific checks**:
1. Can `v + mu2` become zero or negative?
2. Is sigmoid function numerically stable?
3. Can epsilon become negative? (would flip dw/dt direction!)
4. Can epsilon become huge? (would make w explode)

---

#### Test 3.3: Check for Unphysical Parameter Ranges
```python
# Current parameters:
# k = 8.0
# a = 0.15
# epsilon0 = 0.002
# mu1 = 0.2
# mu2 = 0.3
# epsilon_rest = 0.05
# V_threshold = 0.13
# k_sigmoid = 30.0
#
# Are these validated against literature?
```

**Compare to**:
- Original Aliev-Panfilov 1996 paper
- Recent validated implementations
- Known stable parameter sets

---

### Phase 4: Analyze Numerical Method

**Goal**: Check if time-stepping scheme is appropriate

#### Test 4.1: Time Step Stability Analysis
```python
# Forward Euler (current method) has stability limit:
# dt < 2 / max_eigenvalue
#
# For reaction-diffusion:
# - Diffusion: dt < dx^2 / (4 * D_max)
# - Reaction: dt < 1 / max|df/dv|
#
# Calculate actual stability limits
```

**Compute**:
- Diffusion stability limit: `dx^2 / (4 * D_parallel)`
- Reaction stability limit: Maximum Jacobian eigenvalue
- Compare to current dt = 0.01 ms

**Expected findings**:
- For dx = 0.5mm, D = 1.0 mm²/ms: dt < 0.0625 ms (OK)
- But reaction term might have tighter constraint!

---

#### Test 4.2: Alternative Time-Stepping
```python
# Test different schemes:
# 1. Backward Euler (implicit, unconditionally stable)
# 2. Crank-Nicolson (2nd order, more stable)
# 3. Rush-Larsen for ionic gating variables
# 4. Adaptive dt (reduce when variables change rapidly)
```

**Purpose**: See if implicit methods prevent blow-up

---

### Phase 5: Check for Implementation Bugs

**Goal**: Find coding errors that cause instability

#### Test 5.1: Boundary Condition Issues
```python
# We fixed one-sided differences earlier
# But check:
# - Are gradients computed correctly everywhere?
# - Do boundary cells have unphysical values?
# - Is tissue_mask applied correctly?
```

**Specific checks**:
- Print V, w at all boundaries
- Check if corner cells behave strangely
- Verify Neumann BC implementation

---

#### Test 5.2: Sign Errors in Equations
```python
# Current ionic model:
# dV/dt = -I_ion + I_stim
# I_ion = k*v*(v-a)*(v-1) + v*w
# dw/dt = eps * (-k*v*(v-a-1) - w)
#
# Verify signs match literature EXACTLY
```

**Double-check**:
- Is it `+v*w` or `-v*w`?
- Is it `(v-a-1)` or `(v-1-a)`?
- Is diffusion added or subtracted?

---

#### Test 5.3: Normalization Issues
```python
# We normalize voltage: V_physical → V_dimensionless
# V_norm = (V_phys - V_rest) / V_range
#
# Check:
# - Is stimulus normalized correctly?
# - Is V converted back correctly for display?
# - Are there overflow/underflow issues?
```

---

### Phase 6: Implement Safeguards

**Goal**: Add runtime checks without fixing root cause yet

#### Test 6.1: Add Value Clipping (Temporary)
```python
# After each step, clip values:
# V = np.clip(V, -0.5, 1.5)  # Allow some overshoot
# w = np.clip(w, 0.0, 2.0)
#
# Does this prevent crash?
# If yes → confirms which variable is unstable
# If no → problem is more fundamental
```

---

#### Test 6.2: Add Adaptive Time-Stepping
```python
# If |dV/dt| > threshold: reduce dt by factor of 2
# If |dV/dt| < threshold: increase dt
#
# Does adaptive dt prevent instability?
```

---

#### Test 6.3: Add Nan/Inf Detection
```python
# After each step:
# if np.any(np.isnan(V)) or np.any(np.isinf(V)):
#     print(f"NaN/Inf detected at t={t}")
#     print(f"V range: [{np.nanmin(V)}, {np.nanmax(V)}]")
#     print(f"w range: [{np.nanmin(w)}, {np.nanmax(w)}]")
#     # Save state and crash gracefully
```

---

## 🔍 Proposed Investigation Script

Create `debug_stability.py` that:

1. **Runs controlled experiments** with single stimulus
2. **Monitors all variables** every step
3. **Saves state** just before crash
4. **Generates diagnostic plots**:
   - V(t), w(t) for single point
   - V_min(t), V_max(t) for whole domain
   - dV/dt, dw/dt over time
   - Spatial heatmaps at crash time
5. **Tests all hypotheses** systematically
6. **Reports findings** with clear conclusions

---

## 🎯 Expected Findings (Hypotheses)

### Hypothesis 1: Division by Zero in Epsilon
**Suspect**: `eps = ... + (mu1 * w) / (v + mu2)`
- If v drops below -mu2 = -0.3, denominator becomes negative
- This flips sign of recovery term!
- Recovery becomes anti-recovery → w explodes → V crashes

**Test**: Plot min(v + mu2) over time. Does it hit zero?

---

### Hypothesis 2: Time Step Too Large for Stiff System
**Suspect**: Forward Euler with dt = 0.01 ms
- Ionic model might have stiff components
- Fast variables need smaller dt
- Especially after AP when recovery is rapid

**Test**: Run with dt = 0.001 ms. Does it stabilize?

---

### Hypothesis 3: Wrong Sign in Ionic Model
**Suspect**: Transcription error from literature
- Small sign error can make system unstable
- Need to verify EVERY term against original paper

**Test**: Compare to Aliev & Panfilov (1996) equation by equation

---

### Hypothesis 4: Voltage Normalization Bug
**Suspect**: Stimulus or conversion between physical/dimensionless
- If stimulus is double-applied in wrong units
- Or conversion has factor-of-2 error
- Could push V far outside valid range

**Test**: Print actual stimulus values applied to V (dimensionless)

---

### Hypothesis 5: Operator Splitting Error
**Suspect**: Order matters for stiff equations
- Diffusion → Reaction might cause overshoot
- Need Strang splitting for 2nd-order accuracy

**Test**: Try Reaction → Diffusion or Strang splitting

---

## 📊 Deliverables

After completing investigation:

1. **Debug Report** (`STABILITY_FINDINGS.md`):
   - Root cause identified
   - Diagnostic plots showing failure mode
   - Comparison to literature
   - Proposed fixes

2. **Fixed Implementation** (if needed):
   - Corrected ionic model parameters
   - Improved time-stepping
   - Boundary condition fixes
   - Value bounds enforcement

3. **Validation Tests**:
   - Prove fix doesn't break correct behavior
   - Show instability is eliminated
   - Verify wave properties unchanged

4. **Updated Interactive Simulation**:
   - Robust to arbitrary clicking
   - Graceful degradation instead of crash
   - Warning messages if approaching instability

---

## ⚠️ Important Notes

1. **Don't fix prematurely**: First UNDERSTAND the problem completely
2. **Compare to literature**: Aliev-Panfilov model is well-studied
3. **Preserve physics**: Fixes shouldn't change validated behavior
4. **Test thoroughly**: Make sure fix doesn't break V1 validation

---

## 🚦 Decision Points

**After Phase 1**:
- If V blows up → Focus on ionic model
- If w blows up → Focus on recovery term
- If both → Likely coupling issue

**After Phase 2**:
- If ionic-only crashes → Deep dive into Aliev-Panfilov
- If diffusion-only crashes → Check spatial discretization
- If only coupled crashes → Operator splitting issue

**After Phase 3**:
- Found parameter issue → Use validated parameters
- Found sign error → Fix immediately
- Found division by zero → Add safeguard

---

## 📝 Next Steps (Awaiting User Confirmation)

1. Do you want me to proceed with this debug plan?
2. Should I start with Phase 1 (characterize failure)?
3. Any specific symptoms you noticed that I should prioritize?
4. Should I create the `debug_stability.py` script first?

---

**Author**: Generated with Claude Code
**Date**: 2025-12-09
**Status**: ⏳ **AWAITING USER CONFIRMATION TO PROCEED**
