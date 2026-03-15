# Engine V5: Step-by-Step Implementation Guide

## Overview

This document provides a detailed implementation roadmap for the LRd07 model. Each step is linked to specific validation tests from `VALIDATION.md`.

**Strategy:** Build incrementally, validate at each step, never move forward with failing tests.

---

## Implementation Phases

| Phase | Focus | Validation Tests |
|-------|-------|------------------|
| **Phase 1** | Core Parameters & Gating | V1.1, V1.3 |
| **Phase 2** | Ionic Currents | V3.1, V3.2, V1.2 |
| **Phase 3** | Basic AP Generation | V1.4, V2.1-V2.5 |
| **Phase 4** | Calcium Handling | V4.1-V4.4 |
| **Phase 5** | Rate Dependence | V5.1, V5.2 |
| **Phase 6** | CaMKII & Alternans | V7.1-V7.3 |
| **Phase 7** | Restitution | V6.1, V6.2 |
| **Phase 8** | GPU Kernel | Performance benchmarks |
| **Phase 9** | Tissue Simulation | V8.1-V8.3 |

---

## Phase 1: Core Parameters & Gating Kinetics

### Step 1.1: Physical Constants
**File:** `ionic/parameters.py`

```python
# Physical constants
R = 8314.0      # Gas constant [mJ/(mol·K)]
T = 310.0       # Temperature [K] (37°C)
F = 96485.0     # Faraday constant [C/mol]
RTF = R * T / F  # ~26.7 mV
FRT = F / (R * T)

# Extracellular concentrations [mM]
Na_o = 140.0
K_o = 5.4
Ca_o = 1.8

# Cell geometry
C_m = 1.0       # Membrane capacitance [µF/cm²]
V_myo = 0.68    # Myoplasm volume fraction
V_jsr = 0.0048  # JSR volume fraction
V_nsr = 0.0552  # NSR volume fraction
```

**Validation:** None yet (infrastructure)

---

### Step 1.2: Initial Conditions
**File:** `ionic/parameters.py`

```python
# State variable initial conditions
V_init = -84.624      # mV (resting potential)
m_init = 0.00136
h_init = 0.9814
j_init = 0.9905
d_init = 3.0e-6
f_init = 1.0
fCa_init = 1.0
b_init = 0.0         # ICaT activation
g_init = 1.0         # ICaT inactivation
xKr_init = 0.0
xs1_init = 0.0
xs2_init = 0.0
a_to_init = 0.0      # Ito activation (LRd07)
i_to_init = 1.0      # Ito inactivation (LRd07)
Na_i_init = 10.0     # mM
K_i_init = 145.0     # mM
Ca_i_init = 0.00012  # mM (120 nM)
Ca_jsr_init = 1.8    # mM
Ca_nsr_init = 1.8    # mM
```

**Validation:** → V1.1 (V_rest should be ~-84 mV)

---

### Step 1.3: INa Gating (m, h, j)
**File:** `ionic/gating.py`

Implement alpha/beta rate functions from MATLAB code:

```python
@numba.jit(nopython=True)
def alpha_m(V):
    dV = V + 47.13
    if abs(dV) < 1e-7:
        return 3.2
    return 0.32 * dV / (1.0 - exp(-0.1 * dV))

@numba.jit(nopython=True)
def beta_m(V):
    return 0.08 * exp(-V / 11.0)

# Similar for alpha_h, beta_h, alpha_j, beta_j
```

**Test:** Plot m_inf, h_inf, j_inf, tau_m, tau_h, tau_j vs V
- m_inf should be ~0 at rest, ~1 at depolarized
- h_inf, j_inf should be ~1 at rest, ~0 at depolarized

**Validation:** Verify curves match published LRd figures

---

### Step 1.4: ICaL Gating (d, f, fCa)
**File:** `ionic/gating.py`

```python
@numba.jit(nopython=True)
def d_inf(V):
    return 1.0 / (1.0 + exp(-(V + 10.0) / 6.24))

@numba.jit(nopython=True)
def tau_d(V):
    # From CellML - check for singularity at V = -10
    pass

@numba.jit(nopython=True)
def f_inf(V):
    return 1.0 / (1.0 + exp((V + 35.06) / 8.6)) + \
           0.6 / (1.0 + exp((50.0 - V) / 20.0))

@numba.jit(nopython=True)
def fCa_inf(Ca_i, Km_Ca=0.0006):
    return 1.0 / (1.0 + (Ca_i / Km_Ca)**2)
```

**Test:**
- d_inf(0) ≈ 0.83, d_inf(-40) ≈ 0.007
- f_inf(0) ≈ 0.06, f_inf(-40) ≈ 0.95
- fCa_inf should decrease as [Ca]i increases

---

### Step 1.5: ICaT Gating (b, g) - LRd95+
**File:** `ionic/gating.py`

```python
@numba.jit(nopython=True)
def b_inf(V):
    return 1.0 / (1.0 + exp(-(V + 14.0) / 10.8))

@numba.jit(nopython=True)
def g_inf(V):
    return 1.0 / (1.0 + exp((V + 60.0) / 5.6))
```

---

### Step 1.6: IKr Gating (xKr)
**File:** `ionic/gating.py`

```python
@numba.jit(nopython=True)
def xKr_inf(V):
    return 1.0 / (1.0 + exp(-(V + 21.5) / 7.5))

@numba.jit(nopython=True)
def tau_xKr(V):
    return 1.0 / (0.00138 * (V + 14.2) / (1 - exp(-0.123 * (V + 14.2))) +
                  0.00061 * (V + 38.9) / (exp(0.145 * (V + 38.9)) - 1))
```

---

### Step 1.7: IKs Gating (xs1, xs2)
**File:** `ionic/gating.py`

```python
@numba.jit(nopython=True)
def xs_inf(V):
    return 1.0 / (1.0 + exp(-(V - 1.5) / 16.7))

@numba.jit(nopython=True)
def tau_xs1(V):
    # From LRd95 - typically ~100-1000 ms
    pass

@numba.jit(nopython=True)
def tau_xs2(V):
    # Slower than tau_xs1
    pass
```

---

### Step 1.8: Ito Gating (a_to, i_to) - LRd07 Addition
**File:** `ionic/gating.py`

```python
@numba.jit(nopython=True)
def a_to_inf(V):
    return 1.0 / (1.0 + exp(-(V + 10.0) / 15.0))

@numba.jit(nopython=True)
def i_to_inf(V):
    return 1.0 / (1.0 + exp((V + 28.0) / 6.5))

@numba.jit(nopython=True)
def tau_a_to(V):
    return 1.0 / (25.0 * exp((V + 10.0) / 15.0) +
                  25.0 * exp(-(V + 10.0) / 15.0))

@numba.jit(nopython=True)
def tau_i_to(V):
    return 1.0 / (0.03 * exp((V + 28.0) / 6.5) +
                  0.03 * exp(-(V + 28.0) / 6.5))
```

**Test:** a_to_inf should activate at depolarized potentials, i_to_inf should inactivate

---

### Step 1.9: Rush-Larsen Integration Functions
**File:** `ionic/gating.py`

```python
@numba.jit(nopython=True)
def rush_larsen_update(x, x_inf, tau, dt):
    """Unconditionally stable gating update."""
    return x_inf - (x_inf - x) * exp(-dt / tau)
```

**Validation:** → V1.3 (fast gates enable fast upstroke)

---

## Phase 2: Ionic Currents

### Step 2.1: Fast Sodium Current (INa)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_Na(V, m, h, j, Na_i, G_Na=16.0, Na_o=140.0):
    E_Na = RTF * log(Na_o / Na_i)
    return G_Na * m**3 * h * j * (V - E_Na)
```

**Test:**
- At rest (V=-85, m≈0): I_Na ≈ 0
- Peak during upstroke: I_Na ≈ -200 to -400 µA/cm²

**Validation:** → V3.1 (peak I_Na), → V1.3 (enables fast upstroke)

---

### Step 2.2: L-Type Calcium Current (ICaL)
**File:** `ionic/currents.py`

**Option A - Ohmic (simpler, like current V4):**
```python
@numba.jit(nopython=True)
def I_CaL_ohmic(V, d, f, fCa, Ca_i, G_CaL=0.1):
    E_Ca = 0.5 * RTF * log(Ca_o / Ca_i)
    return G_CaL * d * f * fCa * (V - E_Ca)
```

**Option B - GHK (from MATLAB code):**
```python
@numba.jit(nopython=True)
def I_CaL_ghk(V, d, f, fCa, Ca_i, Na_i, K_i, P_Ca=5.4e-4):
    # Full GHK with Ca, Na, K permeability
    # Translate from MATLAB LRd07 code
    pass
```

**Decision:** Check which formulation MATLAB LRd07 uses.

**Test:**
- Peak I_CaL ≈ -5 to -15 µA/cm² during plateau
- Window current near V = -20 to 0 mV

**Validation:** → V3.1 (peak I_CaL), → V2.3 (sustains plateau)

---

### Step 2.3: T-Type Calcium Current (ICaT)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_CaT(V, b, g, Ca_i, G_CaT=0.05):
    E_Ca = 0.5 * RTF * log(Ca_o / Ca_i)
    return G_CaT * b * g * (V - E_Ca)
```

**Test:** Peak I_CaT ≈ -0.5 to -2 µA/cm² early in AP

---

### Step 2.4: Rapid Delayed Rectifier (IKr)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_Kr(V, xKr, K_i, G_Kr=0.02, K_o=5.4):
    E_K = RTF * log(K_o / K_i)
    # Rectification factor
    r_Kr = 1.0 / (1.0 + exp((V + 9.0) / 22.4))
    G_Kr_scaled = G_Kr * sqrt(K_o / 5.4)
    return G_Kr_scaled * xKr * r_Kr * (V - E_K)
```

---

### Step 2.5: Slow Delayed Rectifier (IKs)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_Ks(V, xs1, xs2, K_i, Na_i, G_Ks=0.02):
    E_Ks = RTF * log((K_o + 0.01833 * Na_o) / (K_i + 0.01833 * Na_i))
    return G_Ks * xs1 * xs2 * (V - E_Ks)
```

---

### Step 2.6: Inward Rectifier (IK1)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_K1(V, K_i, G_K1=0.75, K_o=5.4):
    E_K = RTF * log(K_o / K_i)
    # Time-independent with strong rectification
    alpha = 1.02 / (1.0 + exp(0.2385 * (V - E_K - 59.215)))
    beta = (0.49124 * exp(0.08032 * (V - E_K + 5.476)) +
            exp(0.06175 * (V - E_K - 594.31))) / \
           (1.0 + exp(-0.5143 * (V - E_K + 4.753)))
    K1_inf = alpha / (alpha + beta)
    G_K1_scaled = G_K1 * sqrt(K_o / 5.4)
    return G_K1_scaled * K1_inf * (V - E_K)
```

**Validation:** → V1.1 (maintains resting potential)

---

### Step 2.7: Transient Outward (Ito) - LRd07 Addition
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_to(V, a_to, i_to, K_i, G_to=0.4, K_o=5.4):
    E_K = RTF * log(K_o / K_i)
    return G_to * a_to**3 * i_to * (V - E_K)
```

**Test:** Peak I_to ≈ 2-10 µA/cm² immediately after upstroke

**Validation:** → V2.2 (creates phase 1 notch)

---

### Step 2.8: Plateau Potassium (IKp)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_Kp(V, K_i, G_Kp=0.0183, K_o=5.4):
    E_K = RTF * log(K_o / K_i)
    Kp = 1.0 / (1.0 + exp((7.488 - V) / 5.98))
    return G_Kp * Kp * (V - E_K)
```

---

### Step 2.9: Na/Ca Exchanger (INaCa)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_NaCa(V, Na_i, Ca_i, k_NaCa=2000.0, Km_Na=87.5,
           Km_Ca=1.38, k_sat=0.1, eta=0.35):
    exp_eta = exp(eta * V * FRT)
    exp_eta_1 = exp((eta - 1.0) * V * FRT)

    numerator = k_NaCa * (Na_i**3 * Ca_o * exp_eta -
                          Na_o**3 * Ca_i * exp_eta_1)
    denominator = ((Km_Na**3 + Na_o**3) * (Km_Ca + Ca_o) *
                   (1.0 + k_sat * exp_eta_1))
    return numerator / denominator
```

---

### Step 2.10: Na/K Pump (INaK)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_NaK(V, Na_i, I_NaK_max=1.5, Km_Nai=10.0, Km_Ko=1.5, K_o=5.4):
    sigma = (exp(Na_o / 67.3) - 1.0) / 7.0
    f_NaK = 1.0 / (1.0 + 0.1245 * exp(-0.1 * V * FRT) +
                   0.0365 * sigma * exp(-V * FRT))
    return I_NaK_max * f_NaK * (K_o / (K_o + Km_Ko)) * (Na_i / (Na_i + Km_Nai))
```

---

### Step 2.11: Sarcolemmal Ca Pump (IpCa)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_pCa(Ca_i, I_pCa_max=1.15, Km_pCa=0.0005):
    return I_pCa_max * Ca_i / (Km_pCa + Ca_i)
```

---

### Step 2.12: Background Currents (INab, ICab)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_Nab(V, Na_i, G_Nab=0.001):
    E_Na = RTF * log(Na_o / Na_i)
    return G_Nab * (V - E_Na)

@numba.jit(nopython=True)
def I_Cab(V, Ca_i, G_Cab=0.003):
    E_Ca = 0.5 * RTF * log(Ca_o / Ca_i)
    return G_Cab * (V - E_Ca)
```

---

### Step 2.13: Non-Specific Ca-Activated (InsCa)
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_nsCa(V, Ca_i, P_nsCa=0.000175, Km_nsCa=0.0012):
    # Only significant at high [Ca]i
    pass
```

---

### Step 2.14: Total Ionic Current
**File:** `ionic/currents.py`

```python
@numba.jit(nopython=True)
def I_ion_total(V, gates, concentrations, params):
    # Sum all currents
    I_total = (I_Na + I_CaL + I_CaT + I_Kr + I_Ks + I_K1 +
               I_to + I_Kp + I_NaCa + I_NaK + I_pCa +
               I_Nab + I_Cab + I_nsCa)
    return I_total
```

**Validation:** → V3.2 (current balance at rest ≈ 0)

---

## Phase 3: Basic AP Generation

### Step 3.1: Single Cell Model Class
**File:** `ionic/lrd07_model.py`

```python
class LRd07Model:
    def __init__(self, dt=0.005):
        self.dt = dt
        self.params = load_parameters()

    def initialize_state(self):
        """Return initial state dictionary."""
        pass

    def ionic_step(self, state, I_stim):
        """Update state for one time step."""
        pass
```

---

### Step 3.2: Voltage Integration

```python
def update_voltage(V, I_ion, I_stim, dt, C_m=1.0):
    dV = -(I_ion + I_stim) / C_m
    return V + dt * dV
```

---

### Step 3.3: Single Cell Simulation

```python
def run_single_cell(model, t_end=500.0, stim_amplitude=-80.0,
                    stim_start=10.0, stim_duration=1.0):
    """Run single cell AP simulation."""
    pass
```

**Test:**
1. Run 500 ms simulation with single stimulus
2. Verify AP is generated
3. Measure V_rest, V_peak, APD90

**Validation:** → V1.1, V1.2, V1.4 (basic AP parameters)

---

### Step 3.4: AP Shape Verification

**Test:**
1. Plot V(t) for one AP
2. Identify all phases (0-4)
3. Compare against published LRd07 figures

**Validation:** → V2.1-V2.5 (AP morphology)

---

## Phase 4: Calcium Handling

### Step 4.1: SERCA Uptake (Iup)
**File:** `ionic/calcium.py`

```python
@numba.jit(nopython=True)
def I_up(Ca_i, I_up_max=0.005, Km_up=0.00092):
    return I_up_max * Ca_i / (Ca_i + Km_up)
```

---

### Step 4.2: SR Transfer (Itr)
**File:** `ionic/calcium.py`

```python
@numba.jit(nopython=True)
def I_tr(Ca_nsr, Ca_jsr, tau_tr=180.0):
    return (Ca_nsr - Ca_jsr) / tau_tr
```

---

### Step 4.3: SR Leak (Ileak)
**File:** `ionic/calcium.py`

```python
@numba.jit(nopython=True)
def I_leak(Ca_nsr, Ca_i, K_leak=0.00026):
    return K_leak * (Ca_nsr - Ca_i)
```

---

### Step 4.4: CICR Release (Irel) - CRITICAL for Ca transient
**File:** `ionic/calcium.py`

This is the **most important** component for correct Ca2+ transient amplitude.

```python
@numba.jit(nopython=True)
def I_rel(Ca_jsr, Ca_i, d, G_rel=30.0, Km_rel=0.0008):
    """
    Calcium-induced calcium release from JSR.
    Updated formulation in LRd07.
    """
    # Trigger: depends on d-gate (L-type Ca activation)
    # Release: proportional to JSR load
    pass
```

**Critical Parameters:**
- G_rel: Release conductance
- Km_rel: Half-saturation for Ca release

**Validation:** → V4.1, V4.4 (Ca transient amplitude and CICR)

---

### Step 4.5: Calcium Buffering
**File:** `ionic/calcium.py`

```python
@numba.jit(nopython=True)
def beta_Ca_i(Ca_i, TRPN_tot=0.07, Km_TRPN=0.0005,
              CMDN_tot=0.05, Km_CMDN=0.00238):
    """Cytoplasmic buffering factor."""
    TRPN = TRPN_tot * Km_TRPN / (Km_TRPN + Ca_i)**2
    CMDN = CMDN_tot * Km_CMDN / (Km_CMDN + Ca_i)**2
    return 1.0 / (1.0 + TRPN + CMDN)

@numba.jit(nopython=True)
def beta_Ca_jsr(Ca_jsr, CSQN_tot=10.0, Km_CSQN=0.8):
    """JSR buffering factor (calsequestrin)."""
    CSQN = CSQN_tot * Km_CSQN / (Km_CSQN + Ca_jsr)**2
    return 1.0 / (1.0 + CSQN)
```

---

### Step 4.6: Concentration Updates
**File:** `ionic/calcium.py`

```python
def update_concentrations(state, currents, dt):
    # Na_i update
    I_Na_tot = I_Na + I_Nab + 3*I_NaK + 3*I_NaCa
    dNa_i = -I_Na_tot / (V_myo * F) * 1e-3

    # K_i update
    I_K_tot = I_Kr + I_Ks + I_K1 + I_to + I_Kp - 2*I_NaK + I_stim
    dK_i = -I_K_tot / (V_myo * F) * 1e-3

    # Ca_i update (with buffering)
    I_Ca_tot = I_CaL + I_CaT + I_Cab + I_pCa - 2*I_NaCa
    dCa_i_unbuffered = (-I_Ca_tot / (2*V_myo*F) * 1e-3 +
                        (V_jsr/V_myo) * I_rel - I_up + I_leak)
    dCa_i = beta_Ca_i * dCa_i_unbuffered

    # JSR update (with buffering)
    dCa_jsr = beta_Ca_jsr * (I_tr - I_rel)

    # NSR update
    dCa_nsr = I_up - I_tr * (V_jsr/V_nsr) - I_leak
```

---

### Step 4.7: Ca Transient Validation

**Test:**
1. Run AP at BCL = 1000 ms
2. Plot [Ca]i vs time
3. Measure:
   - Peak [Ca]i (target: 0.8-2.0 µM)
   - Time to peak (target: 50-150 ms)
   - Decay time constant

**Validation:** → V4.1, V4.2, V4.3 (calcium handling)

---

## Phase 5: Rate Dependence

### Step 5.1: Steady-State Pacing Protocol

```python
def run_steady_state_pacing(model, bcl, n_beats=50):
    """Run model to steady state at given BCL."""
    pass
```

---

### Step 5.2: APD vs BCL Curve

**Test:**
1. Run at BCLs: [2000, 1000, 500, 400, 350, 300] ms
2. 50 beats at each BCL
3. Measure APD90 of last beat
4. Plot APD90 vs BCL

**Validation:** → V5.1 (APD rate dependence)

---

## Phase 6: CaMKII & Alternans

### Step 6.1: CaMKII Signaling Module
**File:** `ionic/calcium.py`

```python
@numba.jit(nopython=True)
def CaMKII_activity(Ca_i, CaMKII_trap, CaMKII_bar=0.05,
                    Km_CaM=1.5e-6, alpha_CaMK=0.05, beta_CaMK=0.00068):
    """
    CaMKII activation/deactivation dynamics.
    Key for LRd07 alternans behavior.
    """
    # Bound fraction depends on [Ca]i
    CaM_bound = Ca_i / (Ca_i + Km_CaM)

    # Active CaMKII
    CaMKII_active = CaMKII_bar * CaM_bound * (1.0 + CaMKII_trap)

    # Trapping dynamics
    dCaMKII_trap = alpha_CaMK * CaMKII_active - beta_CaMK * CaMKII_trap

    return CaMKII_active, dCaMKII_trap
```

---

### Step 6.2: CaMKII Modulation of Currents

CaMKII affects:
- I_CaL (facilitation)
- I_rel (enhanced release)
- SERCA (enhanced uptake)

```python
def apply_CaMKII_modulation(params, CaMKII_active):
    """Modify parameters based on CaMKII activity."""
    # Enhanced I_CaL
    f_CaMK_CaL = 1.0 + 0.5 * CaMKII_active

    # Enhanced CICR
    f_CaMK_rel = 1.0 + 1.0 * CaMKII_active

    return f_CaMK_CaL, f_CaMK_rel
```

---

### Step 6.3: Alternans Detection

```python
def detect_alternans(apd_sequence, threshold=5.0):
    """Detect APD alternans from beat sequence."""
    alternans = []
    for i in range(1, len(apd_sequence)):
        diff = abs(apd_sequence[i] - apd_sequence[i-1])
        if diff > threshold:
            alternans.append((i, diff))
    return alternans
```

**Validation:** → V7.1, V7.2, V7.3 (CaMKII and alternans)

---

## Phase 7: Restitution

### Step 7.1: S1-S2 Protocol

```python
def s1_s2_protocol(model, s1_cl=1000, s1_beats=20,
                   s2_intervals=[50, 100, 150, 200, 300, 500]):
    """S1-S2 restitution protocol."""
    results = []
    for s2_ci in s2_intervals:
        # Run S1 train
        # Apply S2
        # Measure APD of S2
        results.append((s2_ci, apd_s2))
    return results
```

**Validation:** → V6.1 (S1-S2 restitution)

---

### Step 7.2: Dynamic Restitution Protocol

```python
def dynamic_restitution(model, bcl_sequence):
    """Dynamic restitution protocol."""
    pass
```

**Validation:** → V6.2 (dynamic restitution)

---

## Phase 8: GPU Kernel

### Step 8.1: Numba CUDA Kernel
**File:** `solvers/gpu_kernel.py`

```python
from numba import cuda

@cuda.jit
def lrd07_ionic_kernel_gpu(V, m, h, j, d, f, fCa, b, g,
                            xKr, xs1, xs2, a_to, i_to,
                            Na_i, K_i, Ca_i, Ca_jsr, Ca_nsr,
                            CaMKII_trap, I_stim, dt, params):
    """GPU kernel for LRd07 model."""
    i, jj = cuda.grid(2)
    if i < V.shape[0] and jj < V.shape[1]:
        # All computations in registers
        # Same math as CPU kernel
        pass
```

---

### Step 8.2: Memory Layout Optimization

```python
# Structure of Arrays (SoA) for coalesced access
state = {
    'V': np.zeros((ny, nx)),      # Contiguous
    'm': np.zeros((ny, nx)),      # Contiguous
    # ... etc
}
```

---

### Step 8.3: GPU Benchmarking

**Test:**
- Compare GPU vs CPU for various grid sizes
- Target: 50-100x speedup for large grids

---

## Phase 9: Tissue Simulation

### Step 9.1: Diffusion Operator
**File:** `tissue/diffusion.py`

Copy from Engine_V4, modify as needed.

---

### Step 9.2: 2D Simulation
**File:** `tissue/simulation.py`

```python
class CardiacSimulation2D:
    def __init__(self, ionic_model, diffusion_operator):
        pass

    def run(self, t_end):
        # Operator splitting:
        # 1. Diffusion step
        # 2. Ionic step
        pass
```

---

### Step 9.3: CV Measurement

**Validation:** → V8.1 (conduction velocity)

---

### Step 9.4: Spiral Wave Demo

**Validation:** → V8.3 (spiral wave)

---

## Implementation Checklist

| Step | Description | Validation | Status |
|------|-------------|------------|--------|
| 1.1 | Physical constants | - | ☐ |
| 1.2 | Initial conditions | V1.1 | ☐ |
| 1.3 | INa gating | - | ☐ |
| 1.4 | ICaL gating | - | ☐ |
| 1.5 | ICaT gating | - | ☐ |
| 1.6 | IKr gating | - | ☐ |
| 1.7 | IKs gating | - | ☐ |
| 1.8 | Ito gating | - | ☐ |
| 1.9 | Rush-Larsen | V1.3 | ☐ |
| 2.1 | INa | V3.1 | ☐ |
| 2.2 | ICaL | V3.1 | ☐ |
| 2.3 | ICaT | - | ☐ |
| 2.4 | IKr | - | ☐ |
| 2.5 | IKs | - | ☐ |
| 2.6 | IK1 | V1.1 | ☐ |
| 2.7 | Ito | V2.2 | ☐ |
| 2.8-2.13 | Other currents | V3.2 | ☐ |
| 2.14 | Total current | V3.2 | ☐ |
| 3.1 | Model class | - | ☐ |
| 3.2 | Voltage integration | - | ☐ |
| 3.3 | Single cell sim | V1.1-V1.4 | ☐ |
| 3.4 | AP shape | V2.1-V2.5 | ☐ |
| 4.1-4.6 | Calcium handling | V4.1-V4.4 | ☐ |
| 4.7 | Ca validation | V4.1 | ☐ |
| 5.1-5.2 | Rate dependence | V5.1 | ☐ |
| 6.1-6.3 | CaMKII/Alternans | V7.1-V7.3 | ☐ |
| 7.1-7.2 | Restitution | V6.1-V6.2 | ☐ |
| 8.1-8.3 | GPU kernel | Benchmarks | ☐ |
| 9.1-9.4 | Tissue sim | V8.1-V8.3 | ☐ |

---

## Next Steps

1. **Download MATLAB code** from [Rudy Lab](https://rudylab.wustl.edu/code-downloads/)
2. **Extract equations** for all currents and gates
3. **Begin Phase 1** implementation
4. **Validate incrementally** at each step

---

*Created: 2024-12-21*
