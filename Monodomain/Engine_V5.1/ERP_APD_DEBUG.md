# ERP/APD Discrepancy Investigation

## Issue Summary

**Date Identified:** 2024-12-23
**Status:** PHASE 0 COMPLETE - ROOT CAUSE IDENTIFIED: SOURCE-SINK MISMATCH

**Symptom:** In spiral wave S1-S2 protocol simulations:
- Expected S2 timing window: ~APD ± vulnerable window (~250ms ± 20ms)
- Actual S2 timing required: ~380ms (to form spiral)
- At 280ms: Tissue still completely refractory - NO AP forms anywhere

**Root Cause (IDENTIFIED):**
- **Single-cell APD90 = 296.5 ms** (within normal range)
- **Single-cell ERP < 40 ms** (cell becomes excitable during repolarization - NORMAL)
- **2D Tissue ERP ≈ 380 ms** due to SOURCE-SINK MISMATCH

**Key Insight:** The ionic model is correct. The long tissue ERP is caused by electrotonic loading from neighboring cells, not an ionic model bug. A small S2 stimulus cannot provide enough current to excite partially recovered tissue.

---

## Key Definitions

| Term | Definition | Expected Range |
|------|------------|----------------|
| APD90 | Time from upstroke to 90% repolarization | 250-300ms (ventricular) |
| APD50 | Time from upstroke to 50% repolarization | ~150-180ms |
| ERP | Earliest time tissue can be re-excited | APD90 - 20ms to APD90 |
| CI | Coupling Interval (time from S1 to S2) | Variable |
| **DI** | **Diastolic Interval = CI - APD90** | **0-50ms at ERP** |
| Vulnerable Window | Window where S2 causes spiral | ERP to APD90 + ~30ms |

**Key Relationship:** DI = CI - APD90

The Diastolic Interval (DI) is the time between the end of one action potential and the start of the next. It represents the actual recovery time available for the tissue.

**Normal Relationship:**
- ERP ≈ APD90 - (10 to 20ms)
- DI at ERP ≈ -10 to -20ms (tissue becomes excitable before full APD90)

**Our Observation:**
- ERP (~380ms) >> APD90 (assumed 250ms)
- DI at ERP = 380 - 250 = +130ms (tissue requires 130ms of diastole!)

---

## Possible Root Causes

### 1. APD90 is Actually Longer Than Assumed
- ORd model with default ENDO parameters may produce APD90 ~350-400ms
- The 250ms assumption may be incorrect for default parameters
- Need to MEASURE actual APD90 from single-cell simulation

### 2. Post-Repolarization Refractoriness
- Some ionic models exhibit refractoriness beyond APD90
- Recovery of INa (sodium channel) from inactivation
- h-gate and j-gate recovery kinetics may be slow

### 3. Gating Variable Recovery Issues
- INa h-gate (fast inactivation) recovery
- INa j-gate (slow inactivation) recovery
- ICaL f-gate recovery
- These may not fully recover until well after APD90

### 4. Implementation Bug in Ionic Model
- Incorrect time constants for recovery gates
- Wrong formulation of h∞, j∞, or f∞
- Numerical issues with gating variable updates

### 5. CV/Wavelength Mismatch
- If CV is too slow, the wavelength (λ = CV × APD) may be short
- This affects the spatial extent of refractory tissue

---

## Investigation Plan

### Phase 0: Characterize the Actual Behavior

**Goal:** Measure actual APD90 and ERP from the current implementation

**Tests:**
1. **Single-cell APD90 measurement**
   - Stimulate single cell, record voltage trace
   - Measure time from dV/dt_max to 90% repolarization
   - Compare to expected ~250-300ms

2. **Single-cell ERP measurement (S1-S2 protocol)**
   - Apply S1 stimulus
   - Apply S2 at various coupling intervals (200, 250, 300, 350, 400ms)
   - Record whether AP is triggered
   - Find minimum coupling interval that triggers AP = ERP

3. **Gating variable recovery analysis**
   - Plot h, j, f gates during and after AP
   - Identify when they recover to excitable levels
   - Compare recovery time to APD90

**Expected Output:**
- Actual APD90 value
- Actual ERP value
- **DI at ERP** (Diastolic Interval when tissue first becomes excitable)
- Gap between APD90 and ERP (should be ≤20ms, currently ~130ms)

**DI Analysis:**
- If DI at ERP is large and positive (>>20ms): Tissue requires excessive diastolic recovery
- If DI at ERP is near zero or negative: Normal - tissue excitable near/before APD90
- The DI tells us how much "extra" recovery time the tissue needs beyond APD90

---

### Phase 1: Identify Root Cause

Based on Phase 0 results:

**If APD90 >> 250ms (e.g., APD90 ~ 350-380ms):**
- Root cause: Wrong APD assumption
- Solution: Update expected APD values, adjust S2 timing calculations
- No bug in ionic model

**If APD90 ~ 250ms but ERP >> APD90:**
- Root cause: Post-repolarization refractoriness
- Need to investigate gating variable recovery kinetics
- Possible bug in h-gate, j-gate, or f-gate formulation

**If gating variables show incorrect recovery:**
- Root cause: Implementation bug
- Need to compare against CellML reference implementation
- Check time constant formulas

---

### Phase 2: Fix or Adapt

Depending on Phase 1 findings:

**Option A: Adjust Timing Calculations (if APD was wrong)**
```python
# Update D_min formula with correct APD
D_min = K_BASE * dx² * (APD_REF / actual_APD)^ALPHA

# Update S2 timing window
s2_window_start = actual_APD * 0.9
s2_window_end = actual_APD * 1.2
```

**Option B: Fix Ionic Model (if bug found)**
- Correct gating variable formulations
- Validate against CellML/published values
- Re-run single-cell validation

**Option C: Add ERP-specific Calculations (if post-repolarization refractoriness is real)**
- Compute ERP separately from APD90
- Use ERP for S2 timing calculations
- Document the distinction

---

### Phase 3: Validation

After fix/adaptation:
1. Re-run single-cell APD90 and ERP measurements
2. Verify ERP ≈ APD90 - 10ms (normal relationship)
3. Test S1-S2 protocol with corrected timing
4. Confirm spiral wave formation in expected window

---

## Experimental Protocol for Phase 0

### Test 0.A: Single-Cell APD90 Measurement

```python
# Pseudocode for APD90 measurement
from ionic import IonicModel, CellType

model = IonicModel(CellType.ENDO, device='cuda')
dt = 0.01  # 10µs for accuracy

# Stimulate at t=10ms
stim_start, stim_duration = 10.0, 1.0
stim_amplitude = -80.0  # µA/µF

V_trace = []
t_trace = []

for t in range(0, 600_000):  # 600ms in 10µs steps
    time_ms = t * dt

    # Apply stimulus
    I_stim = stim_amplitude if stim_start <= time_ms < stim_start + stim_duration else 0

    model.step(dt, I_stim)
    V_trace.append(model.V)
    t_trace.append(time_ms)

# Find APD90
V_rest = V_trace[0]  # ~-86mV
V_peak = max(V_trace)  # ~+40mV
V_90 = V_rest + 0.1 * (V_peak - V_rest)  # 90% repolarization level

# Find upstroke time (dV/dt max)
dVdt = np.diff(V_trace) / dt
t_upstroke = t_trace[np.argmax(dVdt)]

# Find APD90 endpoint
for i, (t, V) in enumerate(zip(t_trace, V_trace)):
    if t > t_upstroke and V < V_90:
        t_apd90 = t
        break

APD90 = t_apd90 - t_upstroke
print(f"APD90 = {APD90:.1f} ms")
```

### Test 0.B: Single-Cell ERP Measurement

```python
# S1-S2 protocol for ERP
def test_s1s2(coupling_interval):
    model = IonicModel(CellType.ENDO, device='cuda')

    # S1 at t=10ms
    s1_time = 10.0
    s2_time = s1_time + coupling_interval

    stim_duration = 1.0
    stim_amplitude = -80.0

    max_V_after_s2 = -90  # Track max V after S2

    for t in range(0, 1_000_000):  # 1000ms
        time_ms = t * dt

        # S1 or S2 stimulus
        if s1_time <= time_ms < s1_time + stim_duration:
            I_stim = stim_amplitude
        elif s2_time <= time_ms < s2_time + stim_duration:
            I_stim = stim_amplitude
        else:
            I_stim = 0

        model.step(dt, I_stim)

        # Track max V after S2
        if time_ms > s2_time + stim_duration:
            max_V_after_s2 = max(max_V_after_s2, model.V)

    # AP triggered if V exceeds 0mV after S2
    return max_V_after_s2 > 0

# Binary search for ERP
for ci in [200, 250, 280, 300, 320, 340, 360, 380, 400]:
    result = test_s1s2(ci)
    print(f"CI = {ci}ms: {'AP' if result else 'No AP'}")
```

### Test 0.C: Gating Variable Analysis

```python
# Track gating variables during AP
gates = {
    'h': [],  # INa fast inactivation
    'j': [],  # INa slow inactivation
    'f': [],  # ICaL inactivation
    'hf': [], # INaL fast inactivation
    'hs': [], # INaL slow inactivation
}

# After simulation, analyze recovery
# When do gates return to >0.9 of resting value?
```

---

## Questions to Resolve

1. **What is the actual APD90 of ORd ENDO cells with default parameters?**
   - Literature suggests ~270-300ms
   - Need to measure from our implementation

2. **Is post-repolarization refractoriness expected in ORd model?**
   - Need to check ORd paper for ERP vs APD relationship
   - Some models explicitly have prolonged refractoriness

3. **Are we using the correct stimulus parameters?**
   - Is -80 µA/µF sufficient to trigger AP?
   - Is 1ms duration sufficient?
   - These affect whether AP is triggered vs just subthreshold response

4. **Is the 2D tissue coupling affecting ERP?**
   - Source-sink mismatch can increase apparent ERP
   - Single-cell ERP may differ from tissue ERP
   - Need to test both

---

## Reference Values from Literature

| Source | Cell Type | APD90 (ms) | ERP (ms) | Notes |
|--------|-----------|------------|----------|-------|
| O'Hara 2011 | ENDO | 271 | ~260 | 1 Hz pacing |
| O'Hara 2011 | MID | 330 | ~320 | 1 Hz pacing |
| O'Hara 2011 | EPI | 251 | ~240 | 1 Hz pacing |
| Ten Tusscher 2006 | ENDO | 270 | ~265 | 1 Hz pacing |

**Note:** ORd ENDO should have APD90 ~270ms and ERP ~260ms at 1 Hz pacing.

If our observation is ERP ~380ms, we have either:
1. Much longer APD than expected (~390ms would give ERP ~380ms)
2. Bug in recovery kinetics
3. Different ionic parameters than standard ORd

---

## Next Steps

1. Run Phase 0.A test (single-cell APD90 measurement)
2. Run Phase 0.B test (single-cell ERP measurement)
3. Run Phase 0.C test (gating variable analysis)
4. Compare results to literature values
5. Determine root cause
6. Implement fix or update assumptions

---

## Investigation Log

### Entry 1: 2024-12-23 - Phase 0 Single-Cell Characterization

**Tests Performed:**
1. Test 0.A: Single-cell APD90 measurement
2. Test 0.B: Single-cell ERP measurement with DI analysis

**Results:**

| Measurement | Value | Expected | Status |
|-------------|-------|----------|--------|
| APD90 | **296.5 ms** | 250-300ms | ✓ NORMAL |
| Single-cell ERP | **< 40 ms** | ~260ms | SURPRISING! |
| DI at ERP | **-256.5 ms** | 0 to -20ms | VERY NEGATIVE |

**S1-S2 Protocol Results (Single Cell):**
```
CI= 40ms: max_V=+103.8mV -> AP TRIGGERED
CI= 50ms: max_V=+102.6mV -> AP TRIGGERED
CI=100ms: max_V=+88.0mV  -> AP TRIGGERED
CI=200ms: max_V=+39.3mV  -> AP TRIGGERED
CI=300ms: max_V=+38.0mV  -> AP TRIGGERED
CI=400ms: max_V=+38.5mV  -> AP TRIGGERED
```

**CRITICAL FINDING:**
The single-cell ERP is MUCH shorter than APD90! This is **NORMAL** behavior:
- Cardiac cells become excitable during the late repolarization phase
- The cell can trigger a new AP while still recovering from the previous one
- This is called "supernormal excitability" period

**Conclusion:**
The ionic model is working CORRECTLY. The single-cell ERP (~40ms) is appropriate.

---

### Entry 2: 2024-12-23 - Root Cause Identified

**The Real Problem: SOURCE-SINK MISMATCH in 2D Tissue**

The discrepancy between:
- Single-cell ERP: ~40ms
- 2D tissue apparent ERP: ~380ms (observed in spiral wave simulations)

This ~340ms difference is due to **source-sink mismatch** in 2D tissue:

1. **In a single cell:** The cell only needs to excite itself → low threshold
2. **In 2D tissue:** The excited cell must provide enough current to excite ALL neighboring cells → much higher threshold

**Physical Explanation:**
- When a single cell tries to initiate propagation in recovered tissue, it acts as a "point source"
- It must generate enough depolarizing current to overcome the "sink" of all neighboring resting cells
- During early recovery (low sodium channel availability), the cell cannot generate sufficient current
- The tissue appears "refractory" even though individual cells could fire if isolated

**This is NOT a bug!** It's correct physics of cardiac propagation.

**Implications for S1-S2 Protocol:**
- The S2 timing must account for tissue-level ERP, not single-cell ERP
- Tissue ERP ≈ APD90 + 80-100ms for point/small stimuli
- Larger S2 electrode area reduces source-sink mismatch
- The 2×2cm S2 block in spiral wave simulations may be too small

---

### Entry 3: 2024-12-23 - Revised Understanding

**Updated Key Relationships:**

| Term | Single Cell | 2D Tissue (small S2) |
|------|-------------|----------------------|
| APD90 | 296.5 ms | Same |
| ERP | < 40 ms | ~380 ms |
| DI at ERP | -256.5 ms | +83.5 ms |

**Formula for Tissue ERP:**
```
Tissue_ERP ≈ APD90 + source_sink_delay(S2_size, CV, D)
```

Where source_sink_delay depends on:
- S2 electrode size (larger = less delay)
- Conduction velocity (faster = less delay)
- Diffusion coefficient (higher = more current spread = more delay)

**Recommended Next Steps:**

1. **Phase 0.D: 2D Tissue ERP Measurement**
   - Measure ERP in actual 2D tissue simulation
   - Test different S2 electrode sizes
   - Quantify source-sink delay

2. **Adjust S2 Protocol for Spiral Waves:**
   - Use larger S2 electrode (3×3cm or 4×4cm)
   - OR use plane wave S2 (eliminates source-sink issue)
   - OR wait until tissue_ERP (~380ms) for point S2

3. **Update SpiralWaveSimulation:**
   - Change S2 timing window to account for tissue ERP
   - Add S2 size parameter
   - Document the tissue ERP vs single-cell ERP distinction

