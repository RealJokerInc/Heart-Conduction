# Comprehensive Physics Engine Review

**Date**: 2025-12-09
**Reviewer**: Claude Code
**Scope**: Full analysis of cardiac electrophysiology simulation engine

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Model Choice Analysis](#2-model-choice-analysis-aliev-panfilov)
3. [Physiological Accuracy Issues](#3-physiological-accuracy-issues)
4. [Numerical Implementation Issues](#4-numerical-implementation-issues)
5. [Cardiac-Specific Physics](#5-cardiac-specific-physics)
6. [Recommendations](#6-recommendations)
7. [Implementation Plan](#7-implementation-plan)

---

## 1. Executive Summary

### Current Implementation

The engine uses the **Aliev-Panfilov (AP) model** - a phenomenological 2-variable model for cardiac electrophysiology with:
- Monodomain formulation for wave propagation
- Anisotropic diffusion with fiber orientation
- Numba-accelerated computation
- Explicit Euler time integration

### Overall Assessment

| Category | Rating | Notes |
|----------|--------|-------|
| Computational Efficiency | ★★★★★ | Excellent with Numba |
| Numerical Stability | ★★★★☆ | Good after recent fixes |
| Physiological Realism | ★★☆☆☆ | Limited by model choice |
| Extensibility | ★★★☆☆ | Moderate - hardcoded AP model |
| Clinical Relevance | ★★☆☆☆ | Qualitative only |

### Key Findings

**Strengths**:
1. Fast computation (Numba JIT)
2. Stable numerics (after fixes)
3. Correct wave propagation physics
4. Good anisotropic diffusion handling

**Critical Limitations**:
1. AP model lacks ionic current details
2. No calcium dynamics
3. Simplified AP morphology
4. Missing rate-dependent effects
5. 2D only (no transmural variation)

---

## 2. Model Choice Analysis: Aliev-Panfilov

### What is the Aliev-Panfilov Model?

A **phenomenological** (curve-fitting) model with 2 variables:

```
dV/dt = -k·V·(V-a)·(V-1) - V·w + I_stim + D∇²V
dw/dt = ε(V,w)·(-k·V·(V-a-1) - w)
```

Where:
- V ∈ [0,1]: normalized membrane voltage
- w: recovery variable (represents refractoriness)
- ε(V,w): voltage-dependent time scale

### Comparison with Other Models

| Model | Variables | Ion Currents | Compute Cost | Physiological Detail |
|-------|-----------|--------------|--------------|---------------------|
| **Aliev-Panfilov** | 2 | 0 | Very Low | Minimal |
| FitzHugh-Nagumo | 2 | 0 | Very Low | Minimal |
| Mitchell-Schaeffer | 2 | 0 | Very Low | Low |
| Fenton-Karma | 3 | 3 | Low | Moderate |
| Bueno-Orovio (TP06) | 4 | 4 | Moderate | Moderate |
| Ten Tusscher-Panfilov | 17 | 12 | High | High |
| O'Hara-Rudy | 41 | 15 | Very High | Very High |

### What AP Model Captures

✅ **Correctly models**:
- Wave propagation (conduction velocity)
- Spiral wave dynamics
- Reentry initiation/maintenance
- Basic refractory period
- Anisotropic conduction

❌ **Does NOT model**:
- Ion channel kinetics (Na⁺, K⁺, Ca²⁺)
- Action potential morphology details
- Drug effects
- Ischemia/hypoxia effects
- Alternans (APD alternation)
- Early/delayed afterdepolarizations (EADs/DADs)
- Memory effects
- Calcium dynamics

### When to Use AP Model

**Appropriate for**:
- Educational demonstrations
- Spiral wave pattern studies
- Computational method development
- Quick qualitative explorations
- Activation mapping visualization

**NOT appropriate for**:
- Drug screening simulations
- Clinical arrhythmia prediction
- Quantitative APD measurements
- Ischemia modeling
- Detailed cellular studies

---

## 3. Physiological Accuracy Issues

### Issue #1: No Ion Channel Representation ⚠️ CRITICAL

**Problem**: The AP model has NO explicit ion channels.

**Real cardiac cell**:
```
I_Na  = Fast sodium current (upstroke)
I_CaL = L-type calcium current (plateau)
I_Kr  = Rapid delayed rectifier K⁺ (repolarization)
I_Ks  = Slow delayed rectifier K⁺ (repolarization)
I_K1  = Inward rectifier K⁺ (resting potential)
I_to  = Transient outward K⁺ (notch)
I_NaK = Na⁺/K⁺ pump
I_NCX = Na⁺/Ca²⁺ exchanger
... and many more
```

**AP model**:
```
I_ion = k·V·(V-a)·(V-1) + V·w  // One phenomenological term
```

**Impact**:
- Cannot model drug effects (most drugs target specific channels)
- Cannot model mutations/channelopathies
- Cannot study ion concentration changes
- Phase plane doesn't match real cardiac dynamics

---

### Issue #2: Wrong AP Morphology ⚠️ MODERATE

**Real ventricular AP phases**:
```
Phase 0: Rapid upstroke (Na⁺ influx) - ~1 ms, dV/dt ≈ 200-400 V/s
Phase 1: Early repolarization (I_to) - notch
Phase 2: Plateau (Ca²⁺ influx balances K⁺ efflux) - ~200 ms
Phase 3: Repolarization (K⁺ efflux) - ~50 ms
Phase 4: Resting (-85 mV)
```

**AP model AP phases**:
```
- Smooth upstroke (slower than real)
- No phase 1 notch
- Rounded plateau (no spike-and-dome)
- Smooth repolarization
```

**Impact**:
- APD measurements not accurate
- Rate-dependent changes not realistic
- Cannot distinguish atrial/ventricular/Purkinje APs

---

### Issue #3: Missing Calcium Dynamics ⚠️ CRITICAL

**Real heart**:
```
Ca²⁺ is THE key intracellular messenger:
- Ca²⁺ triggers muscle contraction (excitation-contraction coupling)
- Ca²⁺ affects APD via I_CaL inactivation
- Ca²⁺ homeostasis critical for arrhythmias
- Ca²⁺ sparks can trigger EADs/DADs
```

**AP model**:
```
No calcium variable at all.
Cannot model:
- Contractile force
- Calcium alternans
- DADs (triggered activity)
- Calcium overload
```

**Impact**:
- Cannot link electrical to mechanical activity
- Cannot model triggered arrhythmias
- Cannot study heart failure (Ca handling defects)

---

### Issue #4: Restitution Properties ⚠️ MODERATE

**Restitution** = APD depends on preceding diastolic interval (DI)

**Real heart**:
```
APD = f(DI, previous APDs, Ca loading...)
- Complex multi-exponential shape
- Memory effects (preceding beats matter)
- Regional variations
- Critical for wave break and VF initiation
```

**AP model restitution**:
```
ε(V,w) = ε₀ + ε_rest·sigmoid(V) + μ₁·w/(V+μ₂)
- Simple dependence on current w only
- No memory effects
- Restitution slope may be wrong
```

**Impact**:
- Spiral wave stability may not match reality
- Fibrillation thresholds incorrect
- S1S2 protocols won't give accurate results

---

### Issue #5: Conduction Velocity Accuracy ⚠️ MINOR

**Current implementation**:
```python
CV_target = 500 mm/s (longitudinal)
CV_target = 250 mm/s (transverse)
D_parallel = 1.0 mm²/ms
D_perp = 0.5 mm²/ms
```

**Real values**:
```
Ventricular CV:
- Longitudinal: 400-700 mm/s
- Transverse: 150-300 mm/s
- Ratio: 2-3:1

Depends on:
- Gap junction density (Cx43)
- Cell diameter
- Tissue structure
- Ischemia/fibrosis
```

**Assessment**: Current values are reasonable for healthy ventricular tissue.

**Issue**: D and CV are not independently specified - they're coupled through:
```
CV ∝ √(D · max_dV/dt)
```

---

### Issue #6: Stimulus Current Modeling ⚠️ MINOR

**Current implementation**:
```python
# Dimensionless stimulus added directly to dV/dt
dVdt = -I_ion + stim  # stim in dimensionless units
```

**Real stimulation**:
```
- Current density (μA/cm²) through electrode
- Affected by cell capacitance: C_m · dV/dt = -I_ion + I_stim
- Make-and-break phenomena
- Virtual electrode effects
```

**Impact**:
- Threshold values not realistic
- Cannot model defibrillation accurately
- S1S2 timing protocols may have artifacts

---

## 4. Numerical Implementation Issues

### Issue #1: Explicit Euler Integration ⚠️ MODERATE

**Current method**:
```python
V_new = V + dt * dV/dt
w_new = w + dt * dw/dt
```

**Problems**:
- First-order accuracy (O(dt))
- Requires small dt for stability
- No adaptive stepping

**Better alternatives**:
| Method | Order | Stability | Cost |
|--------|-------|-----------|------|
| Forward Euler | 1 | Poor | Low |
| RK2 (Heun) | 2 | Better | 2× |
| RK4 | 4 | Good | 4× |
| Rush-Larsen | Varies | Excellent for ionic | 1-2× |
| CVODE | Adaptive | Excellent | Varies |

**Recommendation**: Implement Rush-Larsen or RK2 for ionic terms.

---

### Issue #2: Operator Splitting ⚠️ MINOR

**Current approach** (Godunov splitting):
```python
# Step 1: Diffusion
V += dt * div(D·∇V)

# Step 2: Reaction
ionic_step(V, w, dt, I_stim)
```

**Analysis**:
- First-order splitting (O(dt))
- Strang splitting (O(dt²)) would be better:
  ```
  Diffusion(dt/2) → Reaction(dt) → Diffusion(dt/2)
  ```

**Impact**: Minor accuracy loss, probably acceptable for AP model.

---

### Issue #3: Diffusion Discretization ⚠️ MINOR

**Current approach**: Central differences for flux-based divergence

```python
# Gradient (centered)
dVdx[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2.0 * dx)

# Flux
Jx = Dxx * dVdx + Dxy * dVdy

# Divergence
div_J[:, 1:-1] = (Jx[:, 2:] - Jx[:, :-2]) / (2.0 * dx)
```

**Issues**:
- Second-order accurate (acceptable)
- May have mild checkerboard instability
- One-sided BCs at edges correct for Neumann

**Assessment**: Acceptable for current mesh resolution.

---

### Issue #4: Voltage Clamping ⚠️ NEW ISSUE

**Current fix**:
```python
V[i, j] = max(0.0, min(v_new, 1.0))  # Clamp to [0, 1]
```

**Problem**: This artificially limits dynamics!

**Real AP model** should stay in [0,1] naturally through its cubic structure. If clamping activates frequently:
- Time step may be too large
- Parameters may be wrong
- Stimulus too strong

**Better approach**:
- Use adaptive time-stepping
- Let model fail loudly if bounds exceeded
- Monitor clamping frequency as diagnostic

---

### Issue #5: Missing Adaptive Time-Stepping ⚠️ MODERATE

**Problem**: Fixed dt cannot handle varying dynamics

```
During AP upstroke: dV/dt ≈ 100/ms → need dt ≈ 0.001 ms
During plateau:     dV/dt ≈ 0.1/ms → could use dt ≈ 0.1 ms
```

**Impact**:
- Either too slow (small dt everywhere)
- Or unstable (large dt during fast phases)

---

## 5. Cardiac-Specific Physics

### Issue #1: Fiber Architecture ⚠️ MODERATE

**Current implementation**:
```python
# Uniform fiber angle or simple radial variation
fiber_angles = np.zeros((ny, nx))  # Horizontal fibers
```

**Real heart**:
```
- Transmural rotation: +60° → -60° from endo to epi
- Regional variations (apex vs base)
- Sheet structure (laminae)
- Complex 3D organization
```

**Impact**:
- Wave patterns simplified
- Breakthrough dynamics wrong
- Reentry anchoring unrealistic

---

### Issue #2: Tissue Heterogeneity ⚠️ CRITICAL

**Current implementation**:
```python
# Binary tissue mask
tissue_mask: True = healthy, False = infarct (V=0)
```

**Real heart heterogeneity**:
```
1. Cell type variation:
   - Endocardial cells (longer APD)
   - M-cells (very long APD)
   - Epicardial cells (shorter APD, notch)

2. Gradient properties:
   - Apico-basal gradients
   - Transmural gradients
   - I_to expression gradients

3. Pathological regions:
   - Border zone (intermediate properties)
   - Fibrosis (reduced diffusion, not zero)
   - Ischemic zones (elevated [K]o, slow CV)
```

**Impact**:
- No transmural dispersion of repolarization
- Cannot model T-wave genesis
- Infarct border zone dynamics wrong

---

### Issue #3: Gap Junction Modeling ⚠️ MODERATE

**Current implementation**:
```python
# Diffusion coefficient as proxy for gap junctions
D_parallel = 1.0  # mm²/ms
```

**Real gap junctions**:
```
- Cx43 primary ventricular connexin
- Cx40/Cx45 in conduction system
- pH and Ca²⁺ dependent gating
- Heterogeneous distribution
- Remodeling in disease
```

**Missing features**:
- Gap junction uncoupling during ischemia
- Regional Cx43 density variations
- Conduction discontinuities

---

### Issue #4: Infarct Model Too Simplified ⚠️ MODERATE

**Current implementation**:
```python
# Infarct = zero conductivity
if not tissue_mask[i, j]:
    V[i, j] = 0.0
    w[i, j] = 0.0
```

**Real infarct/scar**:
```
1. Dense scar (weeks-months): Truly non-conductive
2. Border zone:
   - Reduced CV (partial uncoupling)
   - Altered APD (ion channel remodeling)
   - Zigzag conduction
3. Surviving bundles:
   - Strands of viable tissue in scar
   - Critical for reentry circuits
```

**Impact**:
- Reentry around infarct too simple
- No slow conduction zones
- Missing figure-of-eight reentry

---

### Issue #5: 2D Limitation ⚠️ MODERATE

**Current**: 2D only (xy plane)

**Real heart**:
```
- 3D structure critical
- Transmural breakthrough
- Intramural reentry
- Scroll wave dynamics
- Apex-base differences
```

**Impact**: Cannot study 3D arrhythmia mechanisms

---

## 6. Recommendations

### Tier 1: Critical Improvements (High Priority)

#### 1.1 Upgrade Ionic Model

**Recommendation**: Implement **Fenton-Karma** or **Bueno-Orovio (minimal)** model

**Fenton-Karma** (3 variables):
```
dV/dt = -J_fi - J_so - J_si + D∇²V
dv/dt = Θ(V_c - V)(1-v)/τv⁻ - Θ(V - V_c)v/τv⁺
dw/dt = Θ(V_c - V)(1-w)/τw⁻ - Θ(V - V_c)w/τw⁺

Where:
J_fi = fast inward (sodium-like)
J_so = slow outward (potassium-like)
J_si = slow inward (calcium-like)
```

**Benefits**:
- Still fast (3 variables)
- Includes "calcium-like" current
- Better AP morphology
- Adjustable restitution
- Can match multiple cell types

**Implementation effort**: Medium (1-2 days)

---

#### 1.2 Add Tissue Heterogeneity

**Recommendation**: Implement APD gradients

```python
# Parameter variation arrays (same shape as V)
APD_scale = np.ones((ny, nx))

# Transmural gradient (if simulating wall thickness)
for j in range(nx):
    transmural_pos = j / nx  # 0=endo, 1=epi
    APD_scale[:, j] = 1.2 - 0.4 * transmural_pos  # Endo longer

# Use in ionic model
epsilon_local = epsilon0 / APD_scale
```

**Implementation effort**: Low (hours)

---

#### 1.3 Implement Better Time Integration

**Recommendation**: Rush-Larsen or RK2 for ionic terms

```python
# Rush-Larsen for gating variables
# For dv/dt = (v_inf - v) / tau_v
v_new = v_inf - (v_inf - v) * exp(-dt / tau_v)

# Much more stable than forward Euler for stiff equations
```

**Implementation effort**: Low-Medium (hours-1 day)

---

### Tier 2: Important Improvements (Medium Priority)

#### 2.1 Realistic Fiber Architecture

**Recommendation**: Implement transmural rotation

```python
def fiber_angle(x, y, wall_thickness):
    # Transmural position (0=endo, 1=epi)
    trans_pos = ... # compute from geometry

    # Linear rotation from +60° to -60°
    angle = np.radians(60 - 120 * trans_pos)
    return angle
```

---

#### 2.2 Border Zone Implementation

**Recommendation**: Replace binary mask with conductivity field

```python
# Instead of binary mask
tissue_mask: bool  # True/False

# Use continuous conductivity
sigma_scale = np.ones((ny, nx))  # 0 to 1
sigma_scale[infarct] = 0.0
sigma_scale[border_zone] = 0.3  # Reduced but not zero

# In diffusion
D_effective = D_base * sigma_scale
```

---

#### 2.3 Add Restitution Monitoring

**Recommendation**: Track APD/DI for restitution curves

```python
class RestitutionMonitor:
    def __init__(self):
        self.activation_times = []
        self.recovery_times = []

    def record(self, t, V, threshold=0.5):
        # Detect activation (V crosses up through threshold)
        # Detect recovery (V crosses down through threshold)
        # Compute APD, DI pairs
```

---

### Tier 3: Nice-to-Have (Lower Priority)

#### 3.1 Adaptive Time-Stepping

```python
def adaptive_step(V, w, dt_max):
    dVdt = compute_dVdt(V, w)

    # Estimate required dt
    dt_required = 0.5 / np.max(np.abs(dVdt))

    # Use smaller of required and max
    dt = min(dt_required, dt_max)

    # Sub-step if needed
    n_substeps = int(np.ceil(dt_max / dt))
    dt_sub = dt_max / n_substeps

    for _ in range(n_substeps):
        V, w = step(V, w, dt_sub)

    return V, w
```

---

#### 3.2 Strang Splitting

```python
def step_strang(V, w, dt):
    # Half diffusion
    V = diffusion_step(V, dt/2)

    # Full reaction
    V, w = ionic_step(V, w, dt)

    # Half diffusion
    V = diffusion_step(V, dt/2)

    return V, w
```

---

#### 3.3 3D Extension

**Long-term**: Extend to 3D with proper scroll wave handling

---

## 7. Implementation Plan

### Phase 1: Quick Wins (1-2 days)

| Task | Effort | Impact |
|------|--------|--------|
| Add APD gradient parameters | 2 hours | High |
| Implement RK2 for reaction | 4 hours | Medium |
| Add restitution monitoring | 4 hours | Medium |
| Border zone conductivity | 2 hours | Medium |

### Phase 2: Model Upgrade (3-5 days)

| Task | Effort | Impact |
|------|--------|--------|
| Implement Fenton-Karma model | 2 days | Very High |
| Add cell-type parameters | 1 day | High |
| Validate against literature | 1-2 days | High |

### Phase 3: Advanced Features (1-2 weeks)

| Task | Effort | Impact |
|------|--------|--------|
| Transmural fiber rotation | 2 days | Medium |
| Adaptive time-stepping | 2-3 days | Medium |
| 3D extension framework | 1 week | High |

---

## 8. Validation Checklist

After implementing improvements, validate:

- [ ] **Single cell AP**: Compare V(t) to published AP traces
- [ ] **APD90**: Should be ~250-300 ms (ventricular)
- [ ] **Conduction velocity**: 400-700 mm/s longitudinal
- [ ] **Restitution curve**: APD vs DI should match literature
- [ ] **Spiral wave period**: ~100-200 ms typical
- [ ] **Wavelength**: CV × APD ≈ 100-200 mm
- [ ] **Numerical convergence**: Solution stable as dx, dt decrease

---

## 9. Summary Table

| Issue | Severity | Fix Difficulty | Recommended Action |
|-------|----------|----------------|-------------------|
| No ion channels | Critical | Medium | Upgrade to Fenton-Karma |
| Wrong AP morphology | Moderate | Medium | Better ionic model |
| No calcium | Critical | High | Fenton-Karma or higher |
| Poor restitution | Moderate | Low | Add APD gradients |
| Binary infarct | Moderate | Low | Continuous conductivity |
| 2D only | Moderate | High | Future 3D extension |
| Explicit Euler | Moderate | Low | Implement RK2 |
| No heterogeneity | Critical | Low | Add parameter gradients |
| Simple fibers | Moderate | Low | Transmural rotation |

---

## 10. Conclusion

The current engine is a **solid foundation** for qualitative cardiac wave propagation studies. It correctly implements:
- Anisotropic diffusion
- Numba acceleration
- Boundary conditions
- Basic wave dynamics

However, for **clinically relevant** or **physiologically accurate** simulations, the engine needs:

1. **Immediate**: Better ionic model (Fenton-Karma minimum)
2. **Short-term**: Tissue heterogeneity and improved numerics
3. **Long-term**: 3D extension and advanced features

The Aliev-Panfilov model is appropriate for **educational purposes** and **method development**, but should not be used for quantitative predictions about drug effects, arrhythmia mechanisms, or clinical applications.

---

**Author**: Generated with Claude Code
**Date**: 2025-12-09
**Status**: Complete Review
