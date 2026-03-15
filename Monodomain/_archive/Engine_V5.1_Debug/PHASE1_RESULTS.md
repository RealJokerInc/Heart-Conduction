# Phase 1 Results: Single-Cell Baseline Measurements

**Date:** 2026-01-01
**Engine:** V5.1 Debug
**Cell Type:** ORd ENDO
**Device:** CUDA GPU

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| dt | 0.01 ms |
| Simulation duration | 600 ms (APD), 800 ms (ERP) |
| Stimulus amplitude | -80 µA/µF |
| Stimulus duration | 1.0 ms |
| Stimulus time (S1) | 10.0 ms |

---

## Test A: Action Potential Duration

### Results

| Metric | Measured | Expected (ORd ENDO) | Status |
|--------|----------|---------------------|--------|
| V_rest | -87.5 mV | -87 to -88 mV | ✅ PASS |
| V_peak | +52.4 mV | +35 to +42 mV | ⚠️ HIGH (+10-17 mV) |
| V_90 threshold | -73.5 mV | - | - |
| Upstroke time | 10.8 ms | - | - |
| APD50 | 229.9 ms | 150-180 ms | ⚠️ LONG (+50-80 ms) |
| APD90 | 300.0 ms | 270-300 ms | ✅ PASS |

### Analysis

- **V_rest:** Within expected range. Resting potential is correct.
- **V_peak:** 10-17 mV higher than typical ORd values. May indicate:
  - Higher INa conductance
  - Different initial conditions
  - Model variant differences
- **APD50:** Significantly longer than expected. Suggests:
  - Prolonged plateau phase
  - Different ICaL or IKr balance
- **APD90:** At upper end of expected range but within tolerance.

---

## Test B: Single-Cell ERP (S1-S2 Protocol)

### Configuration

- S1 stimulus: t = 10.0 ms
- S2 stimulus: t = S1 + CI (coupling interval)
- APD90 reference for DI calculation: 270.0 ms (default)
- Tested coupling intervals: 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500 ms

### Results

| CI (ms) | DI (ms) | Max V after S2 | AP Triggered |
|---------|---------|----------------|--------------|
| 200 | -70.0 | +71.8 mV | ✅ YES |
| 220 | -50.0 | +62.7 mV | ✅ YES |
| 240 | -30.0 | +50.8 mV | ✅ YES |
| 260 | -10.0 | +39.8 mV | ✅ YES |
| 280 | +10.0 | +30.4 mV | ✅ YES |
| 300 | +30.0 | +27.6 mV | ✅ YES |
| 320 | +50.0 | +32.9 mV | ✅ YES |
| 340 | +70.0 | +39.1 mV | ✅ YES |
| 360 | +90.0 | +42.0 mV | ✅ YES |
| 380 | +110.0 | +43.3 mV | ✅ YES |
| 400 | +130.0 | +44.0 mV | ✅ YES |
| 420 | +150.0 | +44.3 mV | ✅ YES |
| 440 | +170.0 | +44.5 mV | ✅ YES |
| 460 | +190.0 | +44.6 mV | ✅ YES |
| 480 | +210.0 | +44.7 mV | ✅ YES |
| 500 | +230.0 | +44.7 mV | ✅ YES |

### Key Finding

**Single-cell ERP < 200 ms**

- AP is triggered at ALL tested coupling intervals (200-500 ms)
- The cell becomes excitable during late repolarization (DI = -70 ms means 70 ms BEFORE APD90)
- This is consistent with "supernormal excitability" during the relative refractory period

### Observation: S2 Response Amplitude Pattern

Interesting pattern in max V after S2:
- **CI=200ms:** +71.8 mV (highest - during repolarization)
- **CI=280-300ms:** +27.6 to +30.4 mV (lowest - near APD90)
- **CI>300ms:** Gradually increases back to +44.7 mV

This suggests the cell is most excitable during late repolarization, less excitable right around APD90, then recovers.

---

## Test C: Gating Variable Recovery

### Configuration

Tracked gating variables:
- hf: INa fast inactivation
- hs: INa slow inactivation
- j: INa recovery from inactivation
- hL: INaL inactivation
- ff: ICaL fast voltage-dependent inactivation
- fcaf: ICaL Ca-dependent fast inactivation

Recovery threshold: 90% of initial (resting) value

### Results

| Gate | Current | Initial Value | Min Value | Recovery to 90% |
|------|---------|---------------|-----------|-----------------|
| hf | INa | ~1.0 | - | ❌ **NOT RECOVERED** |
| hs | INa | ~1.0 | - | ❌ **NOT RECOVERED** |
| j | INa | ~1.0 | - | ❌ **NOT RECOVERED** |
| hL | INaL | ~1.0 | - | ❌ **NOT RECOVERED** |
| ff | ICaL | ~1.0 | - | ✅ 292.4 ms after upstroke |
| fcaf | ICaL | ~1.0 | - | ✅ 255.9 ms after upstroke |

### Critical Finding

**INa gates (hf, hs, j, hL) do NOT recover to 90% within 600 ms simulation!**

This is anomalous because:
1. INa gate recovery typically occurs within 50-100 ms after APD90
2. Literature values for ORd show full INa recovery by ~350-400 ms post-stimulus
3. Yet the single-cell ERP test shows AP triggered at CI=200 ms

### Paradox

There's an apparent contradiction:
- Gates don't recover to 90% → should be refractory
- But AP is triggered at all CIs → cell is excitable

**Possible explanations:**
1. Partial gate recovery (e.g., 50-70%) is sufficient to generate AP
2. The 90% threshold is too strict for excitability assessment
3. ICaL-mediated depolarization can trigger AP even with reduced INa
4. Initial gate values may differ from typical resting values

---

## Summary of Findings

### Passed Criteria ✅

| Metric | Value | Status |
|--------|-------|--------|
| V_rest | -87.5 mV | Within 1 mV of expected |
| APD90 | 300.0 ms | Within 10% of expected |
| Single-cell excitability | ERP < 200 ms | Cell is excitable |

### Concerns ⚠️

| Issue | Observation | Severity |
|-------|-------------|----------|
| V_peak high | +52.4 mV vs expected +35-42 mV | Medium |
| APD50 long | 229.9 ms vs expected 150-180 ms | Medium |
| INa gates don't recover | hf, hs, j, hL < 90% at 600 ms | **HIGH** |

### Critical Question

**Why do INa gates not recover, yet the cell remains excitable?**

This needs further investigation in Phase 2:
1. Plot actual gate values during AP and recovery
2. Compare against CellML reference implementation
3. Determine minimum gate values needed for AP generation
4. Check if initial state values are correct

---

## Comparison: Single-Cell ERP vs Tissue ERP

| Metric | Single Cell | Tissue (observed) | Difference |
|--------|-------------|-------------------|------------|
| ERP | < 200 ms | ~380 ms | +180 ms |
| APD90 | 300 ms | ~300 ms | Same |
| ERP/APD90 ratio | < 0.67 | ~1.27 | 0.6 difference |

The ~180 ms difference between single-cell and tissue ERP is **larger than expected** from source-sink mismatch alone (typically +30-100 ms).

**This suggests the tissue ERP prolongation may be due to:**
1. Source-sink mismatch (expected: +30-100 ms)
2. PLUS additional factor related to INa gate recovery issues
3. In tissue, partial INa recovery may be insufficient to overcome electrotonic load

---

## Recommended Next Steps

### Phase 2: Detailed Gate Analysis

1. **Plot full gate traces** during 600 ms simulation
2. **Identify actual recovery values** (what % do gates reach?)
3. **Compare initial vs final gate values**
4. **Check gate formulas** against CellML reference

### Phase 2b: Minimum Excitability Test

1. **Determine minimum INa availability** needed to trigger AP
2. **Test with reduced stimulus** to find threshold
3. **Compare single-cell vs tissue threshold**

---

## Raw Data Location

Test output files:
- APD test: `tests/debug_apd_erp.py --test apd`
- ERP test: `tests/debug_apd_erp.py --test erp`
- Gates test: `tests/debug_apd_erp.py --test gates`

---

## References

- O'Hara T, et al. (2011). PLoS Comput Biol. (ORd model reference)
- Expected values from CellML repository and original publication
