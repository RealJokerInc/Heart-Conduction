# Paci 2013 hiPSC-CM Model — PyTorch Implementation Plan

## Overview

Translate the Paci 2013 ventricular hiPSC-CM ionic model from Myokit (.mmt) to PyTorch, conforming to the V5.4 `IonicModel` ABC. The same implementation will work in Bidomain V1 (identical ABC).

**Source**: `Research/code_examples/hipsc_ionic_models/tailored_ipsc_models/paci-2013-ventricular.mmt`
**Target**: `Monodomain/Engine_V5.4/cardiac_sim/ionic/paci/`

## Model Summary

```
States:        16 (10 gates + 3 concentrations + 1 Ca release gate + 2 Beattie IKr)
               (V stored separately per V5.4 convention)
Currents:      12 (INa, ICaL, IKr, IKs, IK1, Ito, If, INaCa, INaK, IpCa, IbNa, IbCa)
V_rest:        -74.3 mV (from initial condition)
Spontaneous:   Yes (If + depolarized V_rest drives automaticity)
APD90:         ~350 ms (ventricular variant)
dvdt_max:      ~50 V/s
```

## Structural Mapping to V5.4 IonicModel ABC

### State Index Definition

```python
class StateIndex(IntEnum):
    # Concentrations (Forward Euler)
    Nai    = 0     # Intracellular Na+ (mM)
    Cai    = 1     # Intracellular Ca2+ cytoplasm (mM)
    CaSR   = 2     # SR Ca2+ (mM)

    # INa gates (Rush-Larsen)
    m      = 3     # Activation
    h      = 4     # Fast inactivation
    j      = 5     # Slow inactivation

    # ICaL gates (Rush-Larsen, except fCa which is conditional)
    d      = 6     # Activation
    f1     = 7     # Voltage inactivation 1
    f2     = 8     # Voltage inactivation 2
    fCa    = 9     # Ca-dependent inactivation

    # IKr gates (Rush-Larsen)
    Xr1    = 10    # Activation
    Xr2    = 11    # Inactivation

    # IKs gate (Rush-Larsen)
    Xs     = 12    # Activation

    # Ito gates (Rush-Larsen)
    q      = 13    # Inactivation
    r      = 14    # Activation

    # If gate (Rush-Larsen)
    Xf     = 15    # Activation

    # Ca release (Forward Euler, conditional)
    g_rel  = 16    # RyR inactivation gate

    N_STATES = 17
```

**Note**: The Beattie IKr formulation (2 extra states: open, active) is **excluded** from V1. The tailored-ipsc-models repo added it for drug studies — the standard Paci 2013 uses the Xr1/Xr2 IKr only. This reduces states from 19 to 17.

### Property Mapping

```python
gate_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13 gates
concentration_indices = [0, 1, 2, 16]  # Nai, Cai, CaSR, g_rel
```

The RyR gate (g_rel) is treated as a concentration (Forward Euler with conditional update) because its dynamics include voltage-conditional logic that prevents straightforward Rush-Larsen.

## File Structure

```
cardiac_sim/ionic/paci/
├── __init__.py          # Exports PaciModel
├── parameters.py        # StateIndex, PaciParameters dataclass, initial state
├── gating.py            # All gate inf/tau functions (13 gates)
├── currents.py          # All 12 current functions
├── calcium.py           # Concentration update equations
└── model.py             # PaciModel(IonicModel) orchestrator
```

---

## Implementation Phases

### Phase 1: Parameters & State (parameters.py)

**Deliverables**:
- `StateIndex` enum (17 states, as above)
- `PaciParameters` dataclass with all constants from the .mmt file
- `get_initial_state()` returning the published initial conditions
- `V_REST = -74.334e-3 * 1000` (convert from V to mV — see unit note below)

**Unit conversion**: The Paci .mmt file uses **SI units** (V, A/F, S/F, m³, s). Our V5.4 convention uses **cardiac units** (mV, pA/pF or μA/μF, nS/pF, mM, ms). All values must be converted during translation.

```
Paci .mmt (SI)          →  V5.4 (cardiac)
──────────────────          ──────────────
V in Volts              →  V in mV (×1000)
t in seconds            →  t in ms (×1000)
Conductance in S/F      →  varies by current
Current in A/F          →  A/F (same, dimensionless per membrane area)
Concentration in mM     →  mM (same)
```

**Key parameters** (converted to cardiac units):

| Parameter | Paci (SI) | Cardiac Units | Notes |
|-----------|-----------|---------------|-------|
| g_Na | 3671.23 S/F | 3671.23 (same units if A/F) | Check against INa magnitude |
| g_CaL | 8.636e-5 L/F/ms | Permeability, GHK | Dimensionally different from TTP06 |
| g_Kr | 29.867 S/F | 29.867 | With sqrt(Ko/5.4) |
| g_Ks | 2.041 S/F | 2.041 | With Ca-dependent k factor |
| g_to | 29.904 S/F | 29.904 | |
| g_f | 30.103 S/F | 30.103 | Funny current (NEW) |
| g_K1 | 28.149 S/F | 28.149 | With alpha/beta rectification |
| KNaCa | 4900 A/F | 4900 | |
| PNaK | 1.841 A/F | 1.841 | |
| g_pCa | 0.4125 A/F | 0.4125 | |
| g_bNa | 0.9 S/F | 0.9 | |
| g_bCa | 0.69264 S/F | 0.69264 | |

### Validation (Phase 1)

| Test | Criterion |
|------|-----------|
| P1-V1 | `StateIndex.N_STATES == 17` |
| P1-V2 | `PaciParameters` instantiates with all published values |
| P1-V3 | `get_initial_state()` returns tensor of shape (17,) with correct values |
| P1-V4 | V_REST ≈ -74.3 mV |
| P1-V5 | All concentration initial values match .mmt file |

---

### Phase 2: Gating Functions (gating.py)

**Deliverables**: `_inf(V)` and `_tau(V)` functions for all 13 gates.

Gate-by-gate translation from .mmt:

| Gate | Current | inf formula | tau formula | Special |
|------|---------|-------------|-------------|---------|
| m | INa | 1/(1+exp((V+48.97)/(-5.7))) | m from alpha/beta | Standard |
| h | INa | Conditional V < -40 mV | Conditional V < -40 mV | **Biphasic** |
| j | INa | Conditional V < -40 mV | Conditional V < -40 mV | **Biphasic** |
| d | ICaL | 1/(1+exp(-(V+11.1)/7.2)) | 1/(1+exp(-(V+11.1)/7.2))... | Standard |
| f1 | ICaL | 1/(1+exp((V+26)/3)) | Complex expression | **Ca-dependent scaling** |
| f2 | ICaL | 0.33+0.67/(1+exp((V+35)/4)) | Biexponential | Standard |
| fCa | ICaL | Ca-dependent | Fixed 2 ms | **Conditional update** |
| Xr1 | IKr | Sigmoid with Ca-dependent V_half | Complex | **Ca-dependent V_half** |
| Xr2 | IKr | 1/(1+exp((V+88)/50)) mV-form | 1/(k3+k4) | Standard |
| Xs | IKs | 1/(1+exp((V-19.9)/(-12.7))) | Complex biexp | Standard |
| q | Ito | 1/(1+exp((V+53)/13)) | 6.06+39.1/... | Standard |
| r | Ito | 1/(1+exp(-(V-22.3)/18.75)) | 2.75+14.4/... | Standard |
| Xf | If | 1/(1+exp((V+77.85)/5)) | 1900/(1+exp((V+15)/10)) | **NEW current** |

**Non-standard features requiring care**:

1. **h, j biphasic**: Different alpha/beta formulas above and below -40 mV. Use `torch.where()`:
   ```python
   def INa_h_inf(V):
       return torch.where(V < -40.0,
           # V < -40 mV branch
           ...,
           # V >= -40 mV branch
           ...
       )
   ```

2. **fCa conditional update**: Only updates when V < -60 mV OR fCa > fCa_inf:
   ```python
   # const_fCa = 0 if V > -60 and fCa_inf > fCa, else 1
   # Effectively: fCa gate freezes during depolarized phase
   ```
   This cannot use standard Rush-Larsen. Handle in `model.py step()`.

3. **f1 Ca-dependent tau scaling**:
   ```python
   # constf1 = 1 + 1433*(Cai - 50e-6) if (f1_inf > f1 and V > -60)
   # tau_f1_effective = tau_f1 * constf1
   ```

4. **Xr1 Ca-dependent V_half**:
   ```python
   V_half = -RTF/Q * log((1+Cao/2.6)^4 / (L0*(1+Cao/0.58)^4)) - 19
   # Q=2.3, L0=0.025 — V_half shifts with extracellular Ca
   ```

### Validation (Phase 2)

| Test | Criterion |
|------|-----------|
| P2-V1 | All 13 `_inf(V)` functions return values in [0, 1] for V in [-100, +50] mV |
| P2-V2 | All 13 `_tau(V)` functions return positive values for V in [-100, +50] mV |
| P2-V3 | `INa_m_inf(-74)` ≈ 0.103 (matches initial state from .mmt) |
| P2-V4 | `INa_h_inf(-74)` ≈ 0.787 (matches initial state) |
| P2-V5 | `If_Xf_inf(-74)` ≈ 0.101 (matches initial state) |
| P2-V6 | Biphasic h/j: no discontinuity at V = -40 mV (check with torch.autograd) |
| P2-V7 | All functions work on batched tensors (100,) without error |

---

### Phase 3: Current Functions (currents.py)

**Deliverables**: 12 current functions matching the .mmt formulas.

| Current | Function Signature | Key Feature |
|---------|-------------------|-------------|
| `I_Na` | (V, m, h, j, Nai, GNa, Nao) | Standard Ohmic |
| `I_CaL` | (V, d, f1, f2, fCa, Cai, gCaL, Cao) | **GHK formulation** |
| `I_Kr` | (V, Xr1, Xr2, Ki, GKr, Ko, Cao) | sqrt(Ko/5.4) + **Ca-dependent V_half in Xr1** |
| `I_Ks` | (V, Xs, Ki, Nai, GKs, Ko, Nao, Cai) | **Ca-dependent k factor** |
| `I_K1` | (V, Ki, GK1, Ko) | Alpha/beta rectification |
| `I_to` | (V, q, r, Ki, Gto, Ko) | Standard Ohmic |
| `I_f` | (V, Xf, Gf) | **NEW**: E_f = -17 mV (fixed) |
| `I_NaCa` | (V, Nai, Cai, KNaCa, Cao, Nao, ...) | Voltage-dependent exponential |
| `I_NaK` | (V, Nai, Ki, PNaK, Ko, Nao, ...) | Saturation kinetics |
| `I_pCa` | (Cai, GpCa) | Simple Michaelis-Menten |
| `I_bNa` | (V, Nai, GbNa, Nao) | Background leak |
| `I_bCa` | (V, Cai, GbCa, Cao) | Background leak |

**ICaL GHK formulation** (most complex):
```python
def I_CaL(V, d, f1, f2, fCa, Cai, gCaL=8.636e-5, Cao=1.8):
    F, R, T = 96485.3415, 8314.472, 310.0
    zfrt = 2 * V * F / (R * T)  # V in mV → need conversion
    # GHK: (Cai * exp(zfrt) - 0.341*Cao) / (exp(zfrt) - 1)
    # L'Hôpital at V ≈ 0: use 2*F*(Cai - 0.341*Cao)
    numerator = torch.where(
        torch.abs(V) > 0.01,
        4 * V * F / (R*T) * (Cai * safe_exp(zfrt) - 0.341*Cao) / (safe_exp(zfrt) - 1),
        2 * F * (Cai - 0.341 * Cao)
    )
    return gCaL * d * f1 * f2 * fCa * numerator
```

**I_f (funny current)** — NEW, not in TTP06:
```python
def I_f(V, Xf, Gf=30.10312):
    E_f = -17.0  # mV (fixed reversal potential)
    return Gf * Xf * (V - E_f)
```

### Validation (Phase 3)

| Test | Criterion |
|------|-----------|
| P3-V1 | `I_Na` at V=-20, m=1, h=1, j=1 produces large inward current (< -50 A/F) |
| P3-V2 | `I_CaL` GHK: no NaN at V=0 (L'Hôpital limit) |
| P3-V3 | `I_f` at V=-80 mV produces inward current (depolarizing, If reversal at -17 mV) |
| P3-V4 | `I_f` at V=0 mV produces outward current |
| P3-V5 | `I_K1` rectification: large at V < E_K, small at V >> E_K |
| P3-V6 | Sum of all currents at initial state ≈ 0 (resting equilibrium) |
| P3-V7 | All currents work on batched tensors |

---

### Phase 4: Calcium Handling (calcium.py)

**Deliverables**: Concentration update function matching .mmt calcium dynamics.

Paci has **2 calcium compartments** (vs TTP06's 3):
- Cytoplasm (Cai)
- SR (Ca_SR)
- No subspace (CaSS) — ICaL couples directly to cytoplasm

**Key differences from TTP06 calcium**:

| Feature | TTP06 | Paci 2013 |
|---------|-------|-----------|
| Compartments | 3 (cyt, SR, subspace) | 2 (cyt, SR) |
| RyR trigger | CaSS-dependent | ICaL d-gate triggered |
| RyR state | RR (recovery) | g (inactivation, conditional) |
| SERCA | Vmax_up / (1 + (Kup/Cai)²) | VmaxUp / (1 + Kup²/Cai²) |
| Release | Vrel × OO × (CaSR - CaSS) | (c_rel + a_rel×CaSR²/(b_rel²+CaSR²)) × d × g |

**g_rel conditional update** (most tricky part):
```python
# g_inf depends on Cai level:
g_inf = torch.where(Cai <= 0.00035,
    1 / (1 + (Cai/0.00035)**6),
    1 / (1 + (Cai/0.00035)**16)
)
tau_g = 2.0  # ms

# Conditional: gate FREEZES if g_inf > g AND V > -60 mV
const2 = torch.where((g_inf > g) & (V > -60.0), 0.0, 1.0)
dg = const2 * (g_inf - g) / tau_g
```

### Validation (Phase 4)

| Test | Criterion |
|------|-----------|
| P4-V1 | SERCA uptake increases with Cai (positive i_up at Cai > Kup) |
| P4-V2 | Release is triggered by ICaL activation (d > 0) |
| P4-V3 | g_rel conditional: stays frozen during depolarization (V > -60) |
| P4-V4 | Buffering factors are in (0, 1) for physiological Ca range |
| P4-V5 | Steady-state Cai (no stimulus) stays near initial value (0.018 μM) |

---

### Phase 5: Model Orchestrator (model.py)

**Deliverables**: `PaciModel(IonicModel)` class implementing all ABC methods.

```python
class PaciModel(IonicModel):
    name = "Paci2013"
    n_states = 17
    V_rest = -74.3  # mV

    def step(self, V, ionic_states, dt, I_stim=None):
        # 1. Extract states
        # 2. Compute 12 currents
        # 3. I_ion = sum (includes If — drives spontaneous beating)
        # 4. V_new = V - I_ion * dt (+ I_stim if paced)
        # 5. Rush-Larsen for 13 gates (with fCa and f1 conditionals)
        # 6. Forward Euler for Nai, Cai, CaSR, g_rel
        # 7. Assemble new state tensor
```

**Conditional gate handling in step()**:

```python
# Standard Rush-Larsen for most gates:
m_new = rush_larsen(m, m_inf, m_tau, dt)

# fCa: conditional update
fCa_inf = compute_fCa_inf(Cai)
const_fCa = torch.where((V > -60.0) & (fCa_inf > fCa), 0.0, 1.0)
fCa_new = fCa + const_fCa * (fCa_inf - fCa) / tau_fCa * dt  # Forward Euler, gated

# f1: Ca-dependent tau scaling
f1_inf = compute_f1_inf(V)
f1_tau = compute_f1_tau(V)
constf1 = torch.where((f1_inf > f1) & (V > -60.0),
    1.0 + 1433.0 * (Cai - 50e-6), 1.0)
f1_tau_eff = f1_tau * constf1
f1_new = rush_larsen(f1, f1_inf, f1_tau_eff, dt)
```

### Validation (Phase 5)

| Test | Criterion |
|------|-----------|
| P5-V1 | `PaciModel` instantiates, `n_states == 17` |
| P5-V2 | `step()` runs for 1 timestep (dt=0.01) without error |
| P5-V3 | `compute_Iion()` at resting state ≈ 0 (equilibrium) |
| P5-V4 | `compute_gate_steady_states()` returns (13,) tensor in [0,1] |
| P5-V5 | `compute_gate_time_constants()` returns (13,) positive tensor |
| P5-V6 | Single-cell and batch (100,) modes both work |
| P5-V7 | With I_stim pulse: V reaches +20 mV (spike), returns to rest |

---

### Phase 6: Validation Against Reference

**Compare our PyTorch Paci against the Myokit reference implementation.**

The `tailored_ipsc_models/` repo has Python scripts that run the Paci model via Myokit. We can:
1. Run the Myokit model for 10 beats at CL=1000 → save V(t) trace
2. Run our PyTorch model with identical parameters and dt
3. Compare V(t) traces — must match within numerical precision

```python
# Reference run (Myokit):
import myokit
m = myokit.load_model('paci-2013-ventricular.mmt')
p = myokit.Protocol()
p.schedule(amplitude, start, duration, period)
s = myokit.Simulation(m, p)
d = s.run(10000)  # 10 seconds
V_ref = d['membrane.V']

# Our implementation:
model = PaciModel(device='cpu', dtype=torch.float64)
V, states = model.get_initial_state(1), ...
for t in range(n_steps):
    V, states = model.step(V, states, dt, I_stim)
V_ours = ...

# Compare
assert torch.allclose(V_ours, V_ref, atol=0.5)  # mV tolerance
```

### Validation (Phase 6) — Critical Tests

| Test | Criterion |
|------|-----------|
| P6-V1 | **Spontaneous beating**: No stimulus → model beats spontaneously (If drives depolarization) |
| P6-V2 | **Spontaneous CL**: Cycle length = 1.0-1.5 s (literature: ~1 s for Paci ventricular) |
| P6-V3 | **AP morphology**: V_peak > +20 mV, V_rest ≈ -74 mV |
| P6-V4 | **APD90**: 300-400 ms (Paci 2013 reports ~350 ms) |
| P6-V5 | **dvdt_max**: 30-80 V/s (hiPSC-CM range) |
| P6-V6 | **Myokit comparison**: V(t) RMSE < 1 mV over 10 beats at CL=1000 |
| P6-V7 | **Paced at CL=1000**: Steady-state reached within 20 beats |
| P6-V8 | **Paced at CL=500**: APD shortens (rate-dependent) |
| P6-V9 | **GPU**: Model runs on CUDA, results match CPU within float64 precision |
| P6-V10 | **Tissue compatible**: Works with MonodomainSimulation (V5.4, 1D cable) |

---

## Implementation Order

```
Phase 1  ████░░░░░░░░░░░░░░  parameters.py (StateIndex, dataclass, initial state)
Phase 2  ░░░░████████░░░░░░  gating.py (13 gate inf/tau functions)
Phase 3  ░░░░░░░░░░████░░░░  currents.py (12 current functions + If)
Phase 4  ░░░░░░░░░░░░██░░░░  calcium.py (2-compartment Ca handling)
Phase 5  ░░░░░░░░░░░░░░████  model.py (PaciModel orchestrator)
Phase 6  ░░░░░░░░░░░░░░░░██  Validation against Myokit reference
```

One file at a time. Validate → commit → next.

## Total Test Count: 32 tests across 6 phases

## Dependencies

- `pip install myokit` (for reference comparison in Phase 6 only)
- No new dependencies for the model itself — pure PyTorch

## Unit Conversion Reference

The .mmt file uses SI units internally but the gating/current equations are written with V in mV for the sigmoid arguments (e.g., `V * 1000 + 48.97`). This means the .mmt stores V in Volts but the kinetic expressions internally convert to mV. Our implementation stores V in mV directly, so the `* 1000` factors in the .mmt become unnecessary.

```
.mmt:  V_half = (V * 1000 + 48.97)    → V is in Volts, convert to mV
Ours:  V_half = (V + 48.97)           → V is already in mV
```

This conversion must be applied consistently to every gating function. The conductance values may also need scaling — verify by checking that I_ion at resting state sums to ~0.
