# Engine V5: Validation Protocol

## Overview

This document defines the validation tests to confirm the LRd07 model is behaving correctly. Each test has:
- **Target values** from experimental data or published models
- **Tolerance** for acceptable deviation
- **Test procedure**
- **Pass/Fail criteria**

Tests are organized in order of implementation (single cell → tissue).

---

## Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **V1** | Basic AP Parameters | Verify fundamental electrophysiology |
| **V2** | AP Morphology | Verify AP shape matches guinea pig data |
| **V3** | Ionic Currents | Verify individual current magnitudes |
| **V4** | Calcium Handling | Verify Ca2+ transient and SR dynamics |
| **V5** | Rate Dependence | Verify APD changes with pacing rate |
| **V6** | Restitution | Verify S1-S2 and dynamic restitution |
| **V7** | SR Release/Alternans | Verify LRd07-specific graded release and alternans |
| **V8** | Tissue Propagation | Verify conduction velocity |

---

## V1: Basic AP Parameters

### V1.1 Resting Membrane Potential

| Parameter | Target | Tolerance | Source |
|-----------|--------|-----------|--------|
| V_rest | -84 to -87 mV | ±3 mV | LRd94/07 papers |

**Test Procedure:**
1. Initialize model at steady state
2. Run 100 ms without stimulus
3. Measure average membrane potential

**Pass Criteria:** V_rest ∈ [-90, -80] mV

---

### V1.2 Action Potential Amplitude

| Parameter | Target | Tolerance | Source |
|-----------|--------|-----------|--------|
| V_peak | +35 to +45 mV | ±10 mV | Guinea pig data |
| AP Amplitude | 120-130 mV | ±15 mV | V_peak - V_rest |

**Test Procedure:**
1. Apply stimulus (80 µA/µF, 0.5 ms) - per MATLAB code
2. Measure peak voltage during AP

**Pass Criteria:** V_peak > +20 mV, Amplitude > 100 mV

---

### V1.3 Maximum Upstroke Velocity

| Parameter | Target | Tolerance | Source |
|-----------|--------|-----------|--------|
| dV/dt_max | 200-400 V/s | ±100 V/s | LR91, LRd94 |

**Test Procedure:**
1. Trigger AP with standard stimulus
2. Compute dV/dt during upstroke phase
3. Find maximum value

**Pass Criteria:** dV/dt_max ∈ [100, 500] V/s

---

### V1.4 Action Potential Duration

| Parameter | Target (BCL=400ms) | Tolerance | Source |
|-----------|---------------------|-----------|--------|
| APD30 | 20-50 ms | ±20 ms | LRd07 model |
| APD50 | 40-80 ms | ±30 ms | LRd07 model |
| APD90 | 100-150 ms | ±30 ms | LRd07 MATLAB implementation |

**Note:** LRd07 is a guinea pig ventricular model. Guinea pig APD is inherently
shorter (~100-150 ms) than human ventricular APD (~200-300 ms). The values above
reflect actual MATLAB implementation behavior with default parameters.

**Test Procedure:**
1. Pace at BCL = 400 ms for 10+ beats (steady state)
2. Measure APD at 30%, 50%, 90% repolarization
3. APD_X = time from V_peak to V_rest + X% * (V_peak - V_rest)

**Pass Criteria:** APD90 ∈ [80, 180] ms

---

## V2: AP Morphology

### V2.1 Phase 0 (Upstroke)

| Parameter | Target | Description |
|-----------|--------|-------------|
| Duration | < 2 ms | Time from threshold to V_peak |
| Shape | Near-vertical | Fast Na+ activation |

---

### V2.2 Phase 2 (Plateau)

| Parameter | Target | Description |
|-----------|--------|-------------|
| V_plateau | 0 to +20 mV | Plateau voltage |
| Duration | 100-200 ms | ICaL vs IKr/IKs balance |

**Note:** No Phase 1 notch expected - LRd07 MATLAB code does NOT include Ito.

---

### V2.3 Phase 3 (Repolarization)

| Parameter | Target | Description |
|-----------|--------|-------------|
| Rate | Smooth descent | No oscillations |
| Duration | 50-100 ms | Time from plateau to V_rest |

---

### V2.4 Phase 4 (Resting)

| Parameter | Target | Description |
|-----------|--------|-------------|
| V_rest | Stable | No drift |
| Duration | N/A | Maintained until next stimulus |

**Visual Validation:**
- Compare AP shape against published guinea pig AP traces
- No spike-and-dome morphology expected (no Ito)

---

## V3: Ionic Currents

### V3.1 Peak Current Magnitudes

| Current | Expected Peak | Phase | Notes |
|---------|--------------|-------|-------|
| I_Na | -200 to -400 µA/µF | 0 | Fast upstroke |
| I_CaL | -5 to -15 µA/µF | 2 | Plateau sustaining (GHK) |
| I_CaT | -0.5 to -2 µA/µF | 0-1 | Small early phase |
| I_Kr | 0.5-2 µA/µF | 3 | Repolarization |
| I_Ks | 0.5-3 µA/µF | 2-3 | Slow repolarization, Ca-dependent |
| I_K1 | 1-3 µA/µF | 4 | Resting potential |
| I_Kp | 0.1-0.5 µA/µF | 2 | Plateau |
| I_NaCa | -1 to +2 µA/µF | 2-3 | Bidirectional |
| I_NaK | 0.5-2 µA/µF | All | Constant pump |
| I_pCa | 0.1-0.5 µA/µF | 2-3 | Ca extrusion |

**Test Procedure:**
1. Run single AP simulation
2. Record all currents throughout AP
3. Identify peak magnitude and timing for each current
4. Plot current traces vs time

**Pass Criteria:** All currents within expected ranges

---

### V3.2 Current Balance at Rest

| Condition | Target |
|-----------|--------|
| Sum of all currents at rest | ≈ 0 µA/µF |
| Dominant current at rest | I_K1 (outward) |

**Test Procedure:**
1. At steady state rest (no stimulus)
2. Sum all ionic currents
3. Verify near-zero total

**Pass Criteria:** |I_total| < 0.1 µA/µF at rest

---

## V4: Calcium Handling

### V4.1 Calcium Transient Amplitude

| Parameter | Target | Tolerance | Source |
|-----------|--------|-----------|--------|
| [Ca]i_rest | 80-150 nM | ±50 nM | Experimental |
| [Ca]i_peak | 0.5-1.5 µM | ±0.5 µM | Guinea pig data |
| [Ca]i amplitude | 5-15x resting | Factor | Peak/Rest ratio |

**Test Procedure:**
1. Run AP at BCL = 400 ms (MATLAB default)
2. Measure [Ca]i_rest before stimulus
3. Measure [Ca]i_peak during AP
4. Compute ratio

**Pass Criteria:** Peak [Ca]i ∈ [0.3, 2.0] µM

---

### V4.2 Calcium Transient Timing

| Parameter | Target | Description |
|-----------|--------|-------------|
| Time to peak | 30-100 ms | After AP upstroke |
| Decay τ | 100-300 ms | Exponential decay constant |
| Return to baseline | < 400 ms | Full relaxation (BCL=400) |

---

### V4.3 SR Calcium Load

| Parameter | Target | Description |
|-----------|--------|-------------|
| [Ca]_JSR rest | 0.5-2.0 mM | Free JSR calcium |
| [Ca]_NSR rest | 1.0-3.0 mM | NSR calcium content |
| JSR depletion | 30-70% | During release |

---

### V4.4 Analytical Buffering Validation

| Test | Method |
|------|--------|
| Cubic solver | Verify Ca_i from ca_T matches MATLAB |
| Quadratic solver | Verify Ca_JSR from jsr_T matches MATLAB |
| Conservation | Total Ca = Free + Buffered |

---

## V5: Rate Dependence

### V5.1 APD vs BCL (Steady State)

| BCL (ms) | Expected APD90 (ms) | Notes |
|----------|---------------------|-------|
| 1000 | 250-320 | Slow pacing |
| 500 | 200-260 | Moderate |
| 400 | 180-240 | MATLAB default |
| 300 | 140-180 | Fast |
| 250 | 100-140 | Very fast, alternans possible |

**Test Procedure:**
1. Pace at each BCL for 50 beats (reach steady state)
2. Measure APD90 of last 5 beats
3. Plot APD90 vs BCL

**Pass Criteria:**
- APD shortens monotonically with decreasing BCL
- APD at BCL=300 < 70% of APD at BCL=1000

---

### V5.2 Rate-Dependent IKs

| BCL | Expected IKs Change |
|-----|---------------------|
| Fast | Increased (Ca accumulation enhances gKs) |
| Slow | Baseline |

**Note:** IKs is Ca-dependent in LRd07: `gks = GKsmax * (1 + 0.6/(1 + (38e-6/Ca_i)^1.4))`

---

## V6: Restitution

### V6.1 S1-S2 APD Restitution

**Protocol:**
1. Pace at S1 = 400 ms for 20 beats
2. Apply S2 at varying coupling intervals (CI)
3. CI = S1-S2 interval, DI = CI - APD90(S1)
4. Measure APD90 of S2 beat
5. Plot APD90(S2) vs DI

| DI (ms) | Expected APD90(S2) |
|---------|-------------------|
| 30 | 80-120 ms |
| 50 | 100-150 ms |
| 100 | 140-180 ms |
| 200 | 180-220 ms |

**Pass Criteria:**
- Monotonic increase of APD with DI
- Maximum restitution slope < 2.0
- Recovery to 90% of S1 APD within DI = 200 ms

---

### V6.2 Dynamic Restitution

**Protocol:**
1. Pace at progressively faster BCLs
2. BCL sequence: 500 → 400 → 350 → 300 → 275 → 250 ms
3. 50 beats at each BCL
4. Plot APD90 vs preceding DI

**Pass Criteria:**
- Smooth curve without discontinuities
- Alternans onset detectable at fast pacing

---

## V7: SR Release and Alternans (LRd07 Specific)

### V7.1 Graded SR Release

The LRd07 model uses a graded release mechanism:
```
Rel_ss = ICaL * alpha_Rel / (1 + (K_Relss/jsr)^qn)
tau_Rel = tau / (1 + 0.0123/jsr)
dRel/dt = -(Rel_ss + Rel) / tau_Rel
```

| Parameter | Target | Description |
|-----------|--------|-------------|
| Rel peak | Proportional to ICaL | Graded release |
| JSR dependence | Steep (qn=9) | Low JSR → reduced release |
| Release timing | Follows ICaL | Triggered by L-type current |

**Test Procedure:**
1. Run AP simulation
2. Plot Rel vs time alongside ICaL
3. Verify graded relationship

---

### V7.2 Alternans Onset

**Protocol:**
1. Start pacing at BCL = 400 ms
2. Gradually decrease BCL (350, 300, 275, 250 ms)
3. Monitor beat-to-beat APD and Ca variation
4. Identify alternans onset BCL

| Parameter | Target | Description |
|-----------|--------|-------------|
| Alternans onset BCL | 250-300 ms | CL where alternans begins |
| APD alternans | 10-40 ms | APD_long - APD_short |
| Ca alternans | Present | Beat-to-beat Ca variation |

**Pass Criteria:**
- Alternans should appear at fast pacing (BCL < 300 ms)
- APD and Ca alternans should be concordant
- Mechanism: SR load-dependent release creates instability

---

### V7.3 JSR Load Dependence

| Condition | Expected Behavior |
|-----------|-------------------|
| High JSR load | Large release, long APD |
| Low JSR load | Small release, short APD |
| Fast pacing | Alternating JSR load → alternans |

**Test:** Plot JSR content over consecutive beats at BCL=250ms

---

## V8: Tissue Propagation

### V8.1 Conduction Velocity

| Direction | Target CV | Tolerance |
|-----------|-----------|-----------|
| Longitudinal | 0.5-0.7 mm/ms | ±0.2 mm/ms |
| Transverse | 0.15-0.25 mm/ms | ±0.1 mm/ms |
| Anisotropy ratio | 2.5-4.0 | Longitudinal/Transverse |

**Test Procedure:**
1. Create 1D cable (100 cells)
2. Stimulate one end
3. Measure activation time at multiple points
4. Compute CV from distance/time

---

### V8.2 Wavefront Shape

| Configuration | Expected |
|---------------|----------|
| Isotropic 2D | Circular wavefront |
| Anisotropic 2D | Elliptical wavefront |
| Fiber orientation | Faster along fibers |

---

### V8.3 Spiral Wave

| Parameter | Target | Description |
|-----------|--------|-------------|
| Rotation period | 100-200 ms | Tip rotation time |
| Core size | 2-5 mm | Spiral tip path |
| Stability | Stable or meandering | Depends on restitution |

---

## Validation Test Matrix

| Test ID | Test Name | Phase | Priority |
|---------|-----------|-------|----------|
| V1.1 | Resting potential | 1 | Critical |
| V1.2 | AP amplitude | 1 | Critical |
| V1.3 | Upstroke velocity | 1 | Critical |
| V1.4 | APD90 | 1 | Critical |
| V2.1-4 | AP morphology (no notch) | 1 | High |
| V3.1 | Peak currents (11 currents) | 2 | High |
| V3.2 | Current balance | 2 | High |
| V4.1 | Ca transient amplitude | 2 | Critical |
| V4.2 | Ca transient timing | 2 | High |
| V4.3 | SR load | 2 | Medium |
| V4.4 | Analytical buffering | 2 | High |
| V5.1 | APD rate dependence | 3 | High |
| V5.2 | IKs Ca-dependence | 3 | Medium |
| V6.1 | S1-S2 restitution | 3 | High |
| V6.2 | Dynamic restitution | 3 | Medium |
| V7.1 | Graded SR release | 3 | High |
| V7.2 | Alternans onset | 3 | High |
| V7.3 | JSR load dependence | 3 | Medium |
| V8.1 | Conduction velocity | 4 | High |
| V8.2 | Wavefront shape | 4 | Medium |
| V8.3 | Spiral wave | 4 | Low |

---

## References

- [Luo & Rudy 1994](https://pubmed.ncbi.nlm.nih.gov/7514509/) - Original LRd model
- [Faber & Rudy 2000](https://models.cellml.org/exposure/4e3c9d09ee9f7a01c840dce8a213c5de) - Base for LRd07
- [Livshitz & Rudy 2007](https://pubmed.ncbi.nlm.nih.gov/17277017/) - LRd07 paper

---

*Updated: 2024-12-21*
*Matched to MATLAB code: Livshitz_LRd_CaMKII_2007*
