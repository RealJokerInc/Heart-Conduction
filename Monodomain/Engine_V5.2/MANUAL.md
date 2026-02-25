# Engine V5.2 Technical Manual

Comprehensive documentation of all model components.

---

## Table of Contents

1. [Overview](#1-overview)
2. [State Variables](#2-state-variables)
3. [Ionic Currents](#3-ionic-currents)
4. [Gating Kinetics](#4-gating-kinetics)
5. [Numerical Integration](#5-numerical-integration)
6. [CaMKII Signaling](#6-camkii-signaling)
7. [Calcium Handling](#7-calcium-handling)
8. [Cell Type Variants](#8-cell-type-variants)
9. [Parameters Reference](#9-parameters-reference)
10. [File Structure](#10-file-structure)
11. [Calibration Pipeline](#11-calibration-pipeline)

---

## 1. Overview

### 1.1 Model Description

The O'Hara-Rudy 2011 (ORd) model is a detailed mathematical representation of the human ventricular myocyte action potential. It incorporates:

- **41 state variables** describing voltage, ion concentrations, and gating states
- **15 ionic currents** across the sarcolemma
- **CaMKII signaling** pathway affecting channel phosphorylation
- **4-compartment calcium handling** (cytosol, subspace, NSR, JSR)
- **3 cell type variants** (endocardial, epicardial, M-cell)

### 1.2 Governing Equation

The membrane potential V evolves according to:

```
Cm * dV/dt = -(Iion + Istim)
```

Where:
- `Cm = 1.0 µF/cm²` - Membrane capacitance
- `Iion` - Total ionic current (µA/µF)
- `Istim` - Stimulus current (µA/µF)

### 1.3 Physical Constants

| Constant | Value | Units | Description |
|----------|-------|-------|-------------|
| R | 8314.0 | mJ/(mol·K) | Gas constant |
| T | 310.0 | K | Temperature (37°C) |
| F | 96485.0 | C/mol | Faraday constant |
| R·T/F | 26.71 | mV | Thermal voltage |

### 1.4 Cell Geometry

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| L | 0.01 | cm | Cell length |
| rad | 0.0011 | cm | Cell radius |
| Vcell | 38.0 | pL | Cell volume |
| Vmyo | 25.84 | pL | Myoplasm volume (68% Vcell) |
| Vnsr | 2.10 | pL | Network SR (5.52% Vcell) |
| Vjsr | 0.18 | pL | Junctional SR (0.48% Vcell) |
| Vss | 0.76 | pL | Subspace (2% Vcell) |

---

## 2. State Variables

The model uses 41 state variables indexed by `StateIndex`:

### 2.1 Membrane Potential

| Index | Name | Initial | Units | Description |
|-------|------|---------|-------|-------------|
| 0 | V | -87.5 | mV | Transmembrane potential |

### 2.2 Ion Concentrations - Bulk

| Index | Name | Initial | Units | Description |
|-------|------|---------|-------|-------------|
| 1 | nai | 7.0 | mM | Intracellular Na⁺ |
| 2 | ki | 145.0 | mM | Intracellular K⁺ |
| 3 | cai | 1.0e-4 | mM | Intracellular Ca²⁺ |
| 4 | cansr | 1.2 | mM | Network SR Ca²⁺ |

### 2.3 Ion Concentrations - Subspace

| Index | Name | Initial | Units | Description |
|-------|------|---------|-------|-------------|
| 5 | nass | 7.0 | mM | Subspace Na⁺ |
| 6 | kss | 145.0 | mM | Subspace K⁺ |
| 7 | cass | 1.0e-4 | mM | Subspace Ca²⁺ |
| 8 | cajsr | 1.2 | mM | Junctional SR Ca²⁺ |

### 2.4 INa Gates (Fast Sodium)

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 9 | m | 0.0 | Activation |
| 10 | hf | 1.0 | Fast inactivation |
| 11 | hs | 1.0 | Slow inactivation |
| 12 | j | 1.0 | Recovery from inactivation |
| 13 | hsp | 1.0 | Phosphorylated slow inactivation |
| 14 | jp | 1.0 | Phosphorylated recovery |

### 2.5 INaL Gates (Late Sodium)

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 15 | mL | 0.0 | Activation |
| 16 | hL | 1.0 | Inactivation |
| 17 | hLp | 1.0 | Phosphorylated inactivation |

### 2.6 Ito Gates (Transient Outward K⁺)

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 18 | a | 0.0 | Activation |
| 19 | iF | 1.0 | Fast inactivation |
| 20 | iS | 1.0 | Slow inactivation |
| 21 | ap | 0.0 | Phosphorylated activation |
| 22 | iFp | 1.0 | Phosphorylated fast inactivation |
| 23 | iSp | 1.0 | Phosphorylated slow inactivation |

### 2.7 ICaL Gates (L-type Ca²⁺)

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 24 | d | 0.0 | Activation |
| 25 | ff | 1.0 | Fast voltage inactivation |
| 26 | fs | 1.0 | Slow voltage inactivation |
| 27 | fcaf | 1.0 | Fast Ca-dependent inactivation |
| 28 | fcas | 1.0 | Slow Ca-dependent inactivation |
| 29 | jca | 1.0 | Ca-dependent recovery |
| 30 | nca | 0.0 | Ca-dependent factor |
| 31 | ffp | 1.0 | Phosphorylated fast inactivation |
| 32 | fcafp | 1.0 | Phosphorylated Ca-dependent inactivation |

### 2.8 IKr Gates (Rapid Delayed Rectifier)

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 33 | xrf | 0.0 | Fast activation |
| 34 | xrs | 0.0 | Slow activation |

### 2.9 IKs Gates (Slow Delayed Rectifier)

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 35 | xs1 | 0.0 | Activation gate 1 |
| 36 | xs2 | 0.0 | Activation gate 2 |

### 2.10 IK1 Gate (Inward Rectifier)

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 37 | xk1 | 1.0 | Activation |

### 2.11 SR Release

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 38 | Jrelnp | 0.0 | Non-phosphorylated release state |
| 39 | Jrelp | 0.0 | Phosphorylated release state |

### 2.12 CaMKII

| Index | Name | Initial | Description |
|-------|------|---------|-------------|
| 40 | CaMKt | 0.0 | Trapped (autophosphorylated) CaMKII |

---

## 3. Ionic Currents

The model includes 15 distinct ionic currents. The total ionic current is:

```
Iion = INa + INaL + Ito + ICaL + ICaNa + ICaK + IKr + IKs + IK1
     + INaCa_i + INaCa_ss + INaK + INab + ICab + IKb + IpCa
```

### 3.1 Reversal Potentials

| Potential | Equation | Typical Value |
|-----------|----------|---------------|
| E_Na | RTF × ln(nao/nai) | +70 mV |
| E_K | RTF × ln(ko/ki) | -88 mV |
| E_Ca | 0.5 × RTF × ln(cao/cai) | +130 mV |
| E_Ks | RTF × ln((ko + PKNa×nao)/(ki + PKNa×nai)) | -85 mV |

Where PKNa = 0.01833 (Na permeability ratio for IKs).

---

### 3.2 INa - Fast Sodium Current

**File:** `ionic/currents.py:63-99`

**Function:** Rapid depolarization during phase 0 of action potential.

**Equation:**
```
INa = GNa × m³ × h × j × (V - E_Na)
```

**Gating:**
- m: Activation (fast, τ ~ 0.1 ms)
- h = 0.99×hf + 0.01×hs: Inactivation (weighted fast + slow)
- j: Recovery from inactivation (slow, τ ~ 100 ms)

**Phosphorylation (CaMKII):**
```
INa = (1 - fCaMKp) × INa_np + fCaMKp × INa_p
```
Where:
- INa_np uses h = 0.99×hf + 0.01×hs, j
- INa_p uses hp = 0.99×hf + 0.01×hsp, jp

**Parameters:**
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| GNa | 75.0 | mS/µF | Maximum conductance |
| nao | 140.0 | mM | Extracellular Na⁺ |

---

### 3.3 INaL - Late Sodium Current

**File:** `ionic/currents.py:105-137`

**Function:** Sustained Na⁺ influx during plateau, contributes to APD.

**Equation:**
```
INaL = GNaL × mL × hL × (V - E_Na)
```

**Gating:**
- mL: Activation (same kinetics as INa m gate)
- hL: Inactivation (τ = 200 ms, constant)

**Phosphorylation:**
```
INaL = (1 - fCaMKp) × GNaL × mL × hL × (V - E_Na)
     + fCaMKp × GNaL × mL × hLp × (V - E_Na)
```

**Parameters:**
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| GNaL | 0.0075 | mS/µF | Maximum conductance |
| GNaL_scale | 1.0/0.6 | - | ENDO/EPI scaling |

---

### 3.4 Ito - Transient Outward K⁺ Current

**File:** `ionic/currents.py:143-186`

**Function:** Early repolarization (phase 1 notch), especially in EPI cells.

**Equation:**
```
Ito = Gto × a × i × (V - E_K)
```

**Gating:**
- a: Activation
- i = AiF×iF + AiS×iS: Inactivation (weighted fast + slow)
- AiF = 1/(1 + exp((V-213.6)/151.2))

**EPI-specific modification:**
For epicardial cells, inactivation time constants are scaled:
```
delta_epi = 1.0 - 0.95/(1.0 + exp((V+70)/5))
tiF = tiF × delta_epi
tiS = tiS × delta_epi
```

**Parameters:**
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| Gto | 0.02 | mS/µF | Maximum conductance |
| Gto_scale | 1.0/4.0/4.0 | - | ENDO/EPI/M scaling |

---

### 3.5 ICaL - L-type Ca²⁺ Current

**File:** `ionic/currents.py:192-291`

**Function:** Primary Ca²⁺ influx, triggers SR release, maintains plateau.

**Formulation:** Goldman-Hodgkin-Katz (GHK) equation:

```
ICaL = PCa × d × (fv × (1-nca) + jca × fca × nca) × PhiCa_L
```

Where:
```
PhiCa_L = 4 × V × F²/RT × (cass × exp(2VF/RT) - 0.341×cao) / (exp(2VF/RT) - 1)
```

**Components:** ICaL actually carries three ions:
- ICaL: Ca²⁺ component (main)
- ICaNa: Na⁺ component (PCaNa = 0.00125 × PCa)
- ICaK: K⁺ component (PCaK = 3.574e-4 × PCa)

**Gating:**
- d: Activation
- fv = 0.6×ff + 0.4×fs: Voltage inactivation
- fca = Afcaf×fcaf + Afcas×fcas: Ca-dependent inactivation
- jca: Ca-dependent recovery
- nca: Modulates between voltage-only and Ca-dependent inactivation

**Afcaf voltage dependence:**
```
Afcaf = 0.3 + 0.6/(1 + exp((V-10)/10))
```

**Phosphorylation:** Increases PCa by 10%:
```
PCa_p = 1.1 × PCa
```

**Parameters:**
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| PCa | 0.0001 | cm/s | Ca²⁺ permeability |
| PCa_scale | 1.0/1.2/2.5 | - | ENDO/EPI/M scaling |

---

### 3.6 IKr - Rapid Delayed Rectifier K⁺ Current

**File:** `ionic/currents.py:297-328`

**Function:** Major repolarizing current, especially during phase 3.

**Equation:**
```
IKr = GKr × sqrt(ko/5.4) × xr × rKr × (V - E_K)
```

**Gating:**
- xr = Axrf×xrf + (1-Axrf)×xrs: Activation (weighted fast + slow)
- Axrf = 1/(1 + exp((V+54.81)/38.21))

**Rectification:**
```
rKr = 1/(1 + exp((V+55)/75)) × 1/(1 + exp((V-10)/30))
```

**Parameters:**
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| GKr | 0.046 | mS/µF | Maximum conductance |
| GKr_scale | 1.0/1.3/0.8 | - | ENDO/EPI/M scaling |

---

### 3.7 IKs - Slow Delayed Rectifier K⁺ Current

**File:** `ionic/currents.py:334-362`

**Function:** Rate-dependent repolarization reserve.

**Equation:**
```
IKs = GKs × KsCa × xs1 × xs2 × (V - E_Ks)
```

**Ca-dependent conductance:**
```
KsCa = 1.0 + 0.6/(1 + (3.8e-5/cai)^1.4)
```

**Gating:**
- xs1, xs2: Two sequential activation gates

**Parameters:**
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| GKs | 0.0034 | mS/µF | Maximum conductance |
| GKs_scale | 1.0/1.4/1.0 | - | ENDO/EPI/M scaling |

---

### 3.8 IK1 - Inward Rectifier K⁺ Current

**File:** `ionic/currents.py:368-395`

**Function:** Maintains resting potential, final repolarization.

**Equation:**
```
IK1 = GK1 × sqrt(ko) × xk1 × rk1 × (V - E_K)
```

**Rectification:**
```
rk1 = 1/(1 + exp((V + 105.8 - 2.6×ko)/9.493))
```

**K-dependent steady-state:**
```
xk1_inf = 1/(1 + exp(-(V + 2.5538×ko + 144.59)/(1.5692×ko + 3.8115)))
```

**Parameters:**
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| GK1 | 0.1908 | mS/µF | Maximum conductance |
| GK1_scale | 1.0/1.2/1.3 | - | ENDO/EPI/M scaling |

---

### 3.9 INaCa - Na⁺/Ca²⁺ Exchanger

**File:** `ionic/currents.py:401-507`

**Function:** Exchanges 3 Na⁺ for 1 Ca²⁺, net inward current during diastole.

**Formulation:** Allosteric kinetic model with 3 Na⁺ and 1 Ca²⁺ binding sites.

**Components:**
- INaCa_i: Cytosolic component (80%)
- INaCa_ss: Subspace component (20%)

```
INaCa = 0.8 × Gncx × allo × (JncxNa + 2×JncxCa)  [cytosol]
      + 0.2 × Gncx × allo × (JncxNa + 2×JncxCa)  [subspace]
```

**Ca activation:**
```
allo = 1/(1 + (KmCaAct/cai)²)
```

**Voltage dependence through:**
```
hna = exp(qna × V × F/RT)
hca = exp(qca × V × F/RT)
```

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Gncx | 0.0008 | Exchange rate |
| KmCaAct | 150e-6 mM | Ca activation |
| qna | 0.5224 | Na charge movement |
| qca | 0.167 | Ca charge movement |

---

### 3.10 INaK - Na⁺/K⁺ ATPase

**File:** `ionic/currents.py:513-575`

**Function:** Pumps 3 Na⁺ out, 2 K⁺ in; net outward current.

**Formulation:** Albers-Post kinetic model.

```
INaK = Pnak × (JnakNa + JnakK)
```

**Voltage-dependent Na affinity:**
```
Knai = Knai0 × exp(delta_eNa × V × F/(3RT))
Knao = Knao0 × exp((1-delta_eNa) × V × F/(3RT))
```

**Parameters:**
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| Pnak | 30.0 | µA/µF | Maximum pump rate |
| Knai0 | 9.073 | mM | Intracellular Na affinity |
| Knao0 | 27.78 | mM | Extracellular Na affinity |

---

### 3.11 Background Currents

**File:** `ionic/currents.py:581-622`

#### INab - Background Na⁺ Current
GHK formulation:
```
INab = PNab × F × vfrt × (nai×exp(vfrt) - nao)/(exp(vfrt) - 1)
```
PNab = 3.75e-10 cm/s

#### ICab - Background Ca²⁺ Current
GHK formulation:
```
ICab = PCab × 4F × vfrt × (cai×exp(2vfrt) - 0.341×cao)/(exp(2vfrt) - 1)
```
PCab = 2.5e-8 cm/s

#### IKb - Background K⁺ Current
```
IKb = GKb × xkb × (V - E_K)
xkb = 1/(1 + exp(-(V-14.48)/18.34))
```
GKb = 0.003 mS/µF

#### IpCa - Sarcolemmal Ca²⁺ Pump
```
IpCa = GpCa × cai/(0.0005 + cai)
```
GpCa = 0.0005 µA/µF

---

## 4. Gating Kinetics

### 4.1 General Formulation

All voltage-dependent gates follow Hodgkin-Huxley formalism:

```
dx/dt = (x_inf - x) / tau_x
```

Where:
- x_inf(V): Voltage-dependent steady-state
- tau_x(V): Voltage-dependent time constant

### 4.2 Safe Exponential Function

**File:** `ionic/gating.py:22-38`

To prevent numerical overflow:
```python
def safe_exp(x, limit=80.0):
    return exp(clamp(x, -limit, limit))
```

### 4.3 INa Gating Functions

**File:** `ionic/gating.py:41-103`

| Gate | x_inf | tau (ms) |
|------|-------|----------|
| m | 1/(1 + exp(-(V+39.57)/9.871)) | 1/(6.765×exp((V+11.64)/34.77) + 8.552×exp(-(V+77.42)/5.955)) |
| h | 1/(1 + exp((V+82.90)/6.086)) | - |
| hf | (same as h) | 1/(1.432e-5×exp(-(V+1.196)/6.285) + 6.149×exp((V+0.5096)/20.27)) |
| hs | (same as h) | scale/(0.009794×exp(-(V+17.95)/28.05) + 0.3343×exp((V+5.730)/56.66)) |
| j | (same as h) | 2.038 + 1/(0.02136×exp(-(V+100.6)/8.281) + 0.3052×exp((V+0.9941)/38.45)) |
| hsp | 1/(1 + exp((V+89.1)/6.086)) | 3.0 × tau_hs |
| jp | (same as h) | 1.46 × tau_j |

### 4.4 INaL Gating Functions

**File:** `ionic/gating.py:109-137`

| Gate | x_inf | tau (ms) |
|------|-------|----------|
| mL | 1/(1 + exp(-(V+42.85)/5.264)) | (same as INa m) |
| hL | 1/(1 + exp((V+87.61)/7.488)) | 200.0 (constant) |
| hLp | 1/(1 + exp((V+93.81)/7.488)) | 600.0 (3× hL) |

### 4.5 Ito Gating Functions

**File:** `ionic/gating.py:143-201`

| Gate | x_inf | tau (ms) |
|------|-------|----------|
| a | 1/(1 + exp(-(V-14.34)/14.82)) | 1.0515/(1/(1.2089×(1 + exp(-(V-18.4099)/29.3814))) + 3.5/(1 + exp((V+100)/29.3814))) |
| i | 1/(1 + exp((V+43.94)/5.711)) | - |
| iF | (same as i) | 4.562 + 1/(0.3933×exp(-(V+100)/100) + 0.08004×exp((V+50)/16.59)) |
| iS | (same as i) | 23.62 + 1/(0.001416×exp(-(V+96.52)/59.05) + 1.780e-8×exp((V+114.1)/8.079)) |
| ap | 1/(1 + exp(-(V-24.34)/14.82)) | (same as a) |

**EPI/M-cell delta_epi:**
```
delta_epi = 1.0 - 0.95/(1 + exp((V+70)/5))
```
Multiplies iF and iS time constants for epicardial cells only.

### 4.6 ICaL Gating Functions

**File:** `ionic/gating.py:207-297`

| Gate | x_inf | tau (ms) |
|------|-------|----------|
| d | 1/(1 + exp(-(V+3.940)/4.230)) | 0.6 + 1/(exp(-0.05×(V+6)) + exp(0.09×(V+14))) |
| f | 1/(1 + exp((V+19.58)/3.696)) | - |
| ff | (same as f) | 7.0 + 1/(0.0045×exp(-(V+20)/10) + 0.0045×exp((V+20)/10)) |
| fs | (same as f) | 1000 + 1/(3.5e-5×exp(-(V+5)/4) + 3.5e-5×exp((V+5)/6)) |
| fcaf | (same as f) | 7.0 + 1/(0.04×exp(-(V-4)/7) + 0.04×exp((V-4)/7)) |
| fcas | (same as f) | 100 + 1/(1.2e-4×exp(-V/3) + 1.2e-4×exp(V/7)) |
| jca | (same as f) | 75.0 (constant) |
| ffp | (same as f) | 2.5 × tau_ff |
| fcafp | (same as f) | 2.5 × tau_fcaf |

**nca gate (special):** Uses Forward Euler, not Rush-Larsen:
```
anca = 1/(k2n/km2n + (1 + Kmn/cass)^4)
dnca/dt = anca × k2n - nca × km2n
```
Where km2n = jca × 1.0, Kmn = 0.002 mM, k2n = 1000 /ms.

### 4.7 IKr Gating Functions

**File:** `ionic/gating.py:303-329`

| Gate | x_inf | tau (ms) |
|------|-------|----------|
| xr | 1/(1 + exp(-(V+8.337)/6.789)) | - |
| xrf | (same as xr) | 12.98 + 1/(0.3652×exp((V-31.66)/3.869) + 4.123e-5×exp(-(V-47.78)/20.38)) |
| xrs | (same as xr) | 1.865 + 1/(0.06629×exp((V-34.70)/7.355) + 1.128e-5×exp(-(V-29.74)/25.94)) |

### 4.8 IKs Gating Functions

**File:** `ionic/gating.py:335-355`

| Gate | x_inf | tau (ms) |
|------|-------|----------|
| xs1 | 1/(1 + exp(-(V+11.60)/8.932)) | 817.3 + 1/(2.326e-4×exp((V+48.28)/17.80) + 0.001292×exp(-(V+210)/230)) |
| xs2 | (same as xs1) | 1/(0.01×exp((V-50)/20) + 0.0193×exp(-(V+66.54)/31)) |

### 4.9 IK1 Gating Functions

**File:** `ionic/gating.py:361-376`

| Gate | x_inf | tau (ms) |
|------|-------|----------|
| xk1 | 1/(1 + exp(-(V + 2.5538×ko + 144.59)/(1.5692×ko + 3.8115))) | 122.2/(exp(-(V+127.2)/20.36) + exp((V+236.8)/69.33)) |

---

## 5. Numerical Integration

### 5.1 Rush-Larsen Method

**File:** `ionic/gating.py:382-406`

For gating variables following dx/dt = (x_inf - x)/tau, the exact solution over timestep dt is:

```python
def rush_larsen(x, x_inf, tau, dt):
    return x_inf - (x_inf - x) * exp(-dt/tau)
```

**Advantages over Forward Euler:**
1. **Unconditionally stable** for any dt
2. **Exact** for constant x_inf and tau
3. **Accurate** for varying x_inf, tau if they change slowly

**Usage:** Applied to all voltage-dependent gates (m, h, j, d, f, x, etc.)

### 5.2 Forward Euler

Used for:
1. **Concentrations** - Ion concentrations with buffering
2. **nca gate** - ICaL Ca-dependent inactivation factor
3. **CaMKt** - Trapped CaMKII fraction

```python
x_new = x + dt * dx_dt
```

### 5.3 Integration Order

**File:** `ionic/model.py:491-594`

The model step follows this sequence:

```
1. Compute CaMKII activation (from current state)
2. Compute all ionic currents (using current gates)
3. Update gates (Rush-Larsen)
4. Update voltage (Forward Euler)
5. Compute SR fluxes (Jrel, Jup)
6. Update concentrations (Forward Euler with buffering)
7. Update CaMKII (Forward Euler)
```

This order ensures:
- Currents use **old** gate values (current step)
- Gates are updated based on **old** voltage
- Voltage is updated based on computed currents
- Concentrations are updated based on computed currents and fluxes

### 5.4 Recommended Time Steps

| Simulation Type | dt (ms) | Notes |
|----------------|---------|-------|
| Single cell | 0.001-0.01 | High accuracy |
| Tissue | 0.01-0.02 | Balance accuracy/speed |
| Fast preview | 0.02-0.05 | May lose accuracy |

The diffusion CFL condition typically limits dt more than ionic kinetics.

---

## 6. CaMKII Signaling

**File:** `ionic/camkii.py`

### 6.1 Overview

CaMKII (Calcium/Calmodulin-dependent protein kinase II) modulates multiple targets:

| Target | Effect |
|--------|--------|
| INa | Faster recovery (jp, hsp gates) |
| INaL | Increased current (hLp gate) |
| Ito | Altered inactivation (iFp, iSp gates) |
| ICaL | Increased current, altered inactivation |
| RyR | Increased SR release (Jrelp) |
| SERCA | Increased uptake |

### 6.2 Activation States

```
CaMKb: Calmodulin-bound (transiently active)
CaMKt: Trapped (autophosphorylated, persistently active)
CaMKa: Total active = CaMKb + CaMKt
```

### 6.3 Equations

**Calmodulin binding:**
```
CaMKb = CaMKo × (1 - CaMKt) / (1 + KmCaM/cass)
```

**Total active:**
```
CaMKa = CaMKb + CaMKt
```

**Phosphorylation factor (blend between pathways):**
```
fCaMKp = 1 / (1 + KmCaMK/CaMKa)
```
Range: 0 (non-phosphorylated) to 1 (fully phosphorylated)

**Trapping dynamics:**
```
dCaMKt/dt = aCaMK × CaMKb × CaMKa - bCaMK × CaMKt
```

### 6.4 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| CaMKo | 0.05 | Total CaMKII fraction |
| KmCaM | 0.0015 mM | Ca/CaM binding affinity |
| KmCaMK | 0.15 | Half-activation for phosphorylation |
| aCaMK | 0.05 /ms | Trapping rate |
| bCaMK | 0.00068 /ms | Detrapping rate |

---

## 7. Calcium Handling

**File:** `ionic/calcium.py`

### 7.1 Compartment Model

```
[Subspace (vss)] ←→ [Cytosol (vmyo)] ←→ [NSR (vnsr)] ←→ [JSR (vjsr)]
      ↑                    ↓                              ↓
   ICaL, NCX           SERCA                           RyR Release
```

### 7.2 SR Release (Jrel)

**Function:** Calcium-Induced Calcium Release (CICR) via RyR.

**Trigger:** ICaL (inward Ca²⁺ current, stored as negative value)

**Equation:**
```
Jrel_inf = a_rel × bt × (-ICaL)⁺ / (1 + (cajsr_half/cajsr)^8)
tau_rel = bt / (1 + 0.0123/cajsr)
```

**Rush-Larsen update:**
```
Jrel_new = Jrel_inf - (Jrel_inf - Jrel) × exp(-dt/tau_rel)
```

**Phosphorylated pathway:** Uses btp = 1.25 × bt (25% slower kinetics).

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| a_rel | 0.5 | Release amplitude |
| bt | 4.75 ms | Time constant base |
| cajsr_half | 1.5 mM | Half-max JSR Ca |

### 7.3 SERCA Uptake (Jup)

**Function:** Pump Ca²⁺ from cytosol into SR.

**Equation:**
```
Jupnp = Jup_max × cai / (cai + Kmup)
Jupp = Jup_max × 2.75 × cai / (cai + Kmup - 0.00017)
Jup = (1 - fCaMKp) × Jupnp + fCaMKp × Jupp
```

**Leak:**
```
Jleak = 0.0039375 × cansr / nsrbar
Jup_net = Jup - Jleak
```

### 7.4 Diffusion Fluxes

| Flux | Equation | tau (ms) |
|------|----------|----------|
| JdiffCa | (cass - cai) / tau | 0.2 |
| JdiffNa | (nass - nai) / tau | 2.0 |
| JdiffK | (kss - ki) / tau | 2.0 |
| Jtr | (cansr - cajsr) / tau | 100.0 |

### 7.5 Calcium Buffering

Buffers reduce free Ca²⁺ changes. Buffer factor formula:

```
b_X = B_max × Km / (Km + Ca)²
buffer_factor = 1 / (1 + sum(b_X))
```

**Cytosolic buffers:**
| Buffer | B_max (mM) | Km (mM) |
|--------|------------|---------|
| Calmodulin (CMDN) | 0.05 | 0.00238 |
| Troponin (TRPN) | 0.07 | 0.0005 |

**Subspace buffers:**
| Buffer | B_max (mM) | Km (mM) |
|--------|------------|---------|
| SR membrane (BSR) | 0.047 | 0.00087 |
| Sarcolemmal (BSL) | 1.124 | 0.0087 |

**JSR buffer:**
| Buffer | B_max (mM) | Km (mM) |
|--------|------------|---------|
| Calsequestrin (CSQN) | 10.0 | 0.8 |

### 7.6 Concentration Updates

**Na⁺ balance:**
```
dnass/dt = -INa_ss × Acap/(F×vss) - JdiffNa
dnai/dt = -INa_i × Acap/(F×vmyo) + JdiffNa × vss/vmyo

INa_ss = ICaNa + 3×INaCa_ss
INa_i = INa + INaL + INab + 3×INaCa_i + 3×INaK
```

**K⁺ balance:**
```
dkss/dt = -ICaK × Acap/(F×vss) - JdiffK
dki/dt = -IK_i × Acap/(F×vmyo) + JdiffK × vss/vmyo

IK_i = Ito + IKr + IKs + IK1 + IKb - 2×INaK + Istim
```

**Ca²⁺ balance (with buffering):**
```
dcass/dt = bcass × (-ICa_ss × Acap/(2F×vss) + Jrel × vjsr/vss - JdiffCa)
dcai/dt = bcai × (-ICa_i × Acap/(2F×vmyo) - Jup × vnsr/vmyo + JdiffCa × vss/vmyo)
dcansr/dt = Jup - Jtr × vjsr/vnsr
dcajsr/dt = bcajsr × (Jtr - Jrel)

ICa_ss = ICaL - 2×INaCa_ss
ICa_i = IpCa + ICab - 2×INaCa_i
```

---

## 8. Cell Type Variants

**File:** `ionic/parameters.py:281-326`

### 8.1 Overview

Three cell types represent transmural heterogeneity:

| Type | Location | APD | Features |
|------|----------|-----|----------|
| ENDO | Subendocardium | Medium | Baseline parameters |
| EPI | Subepicardium | Short | Large Ito, fast repolarization |
| M_CELL | Mid-wall | Long | Large ICaL, small IKr |

### 8.2 Scaling Factors

| Parameter | ENDO | EPI | M_CELL | Description |
|-----------|------|-----|--------|-------------|
| GNaL_scale | 1.0 | 0.6 | 1.0 | Late Na current |
| Gto_scale | 1.0 | 4.0 | 4.0 | Transient outward K |
| PCa_scale | 1.0 | 1.2 | 2.5 | L-type Ca |
| GKr_scale | 1.0 | 1.3 | 0.8 | Rapid delayed rectifier |
| GKs_scale | 1.0 | 1.4 | 1.0 | Slow delayed rectifier |
| GK1_scale | 1.0 | 1.2 | 1.3 | Inward rectifier |
| GKb_scale | 1.0 | 0.6 | 1.0 | Background K |
| Gncx_scale | 1.0 | 1.1 | 1.4 | Na/Ca exchanger |
| Pnak_scale | 1.0 | 0.9 | 0.7 | Na/K pump |
| Jup_scale | 1.0 | 1.3 | 1.0 | SERCA uptake |
| Jrel_scale | 1.0 | 1.0 | 1.7 | SR release |
| cmdnmax_scale | 1.0 | 1.3 | 1.0 | Calmodulin buffering |

### 8.3 Ito delta_epi

For EPI cells only, Ito inactivation kinetics are slowed at depolarized potentials:

```python
delta_epi = 1.0 - 0.95/(1.0 + exp((V + 70.0)/5.0))
tau_iF *= delta_epi
tau_iS *= delta_epi
```

This is NOT applied to M_CELL despite also having Gto_scale = 4.0.

---

## 9. Parameters Reference

**File:** `ionic/parameters.py`

### 9.1 ORdParameters Class

All model parameters are encapsulated in the `ORdParameters` dataclass.

### 9.2 Parameter Override

To modify parameters for specific simulations:

```python
model = ORdModel(
    celltype=CellType.ENDO,
    params_override={
        'GKr_scale': 2.0,    # Double IKr (shorten APD)
        'PCa_scale': 0.5     # Halve ICaL (shorten APD)
    }
)
```

Common modifications for APD shortening:
| Parameter | Value | Effect |
|-----------|-------|--------|
| GKr_scale | 2.0-3.0 | Increase repolarizing current |
| GKs_scale | 2.0-3.0 | Increase repolarizing current |
| PCa_scale | 0.3-0.5 | Reduce plateau current |
| GNaL_scale | 0.3-0.5 | Reduce inward current |

### 9.3 Extracellular Concentrations

| Ion | Value | Units |
|-----|-------|-------|
| nao | 140.0 | mM |
| cao | 1.8 | mM |
| ko | 5.4 | mM |

---

## 10. File Structure

```
ionic/
├── __init__.py          # Module exports
├── model.py             # ORdModel class (main entry point)
├── gating.py            # Gating kinetics (x_inf, tau functions)
├── currents.py          # Current calculations (I_Na, I_CaL, etc.)
├── calcium.py           # Ca handling (Jrel, Jup, buffering)
├── camkii.py            # CaMKII signaling
└── parameters.py        # StateIndex, CellType, ORdParameters
```

### 10.1 Import Hierarchy

```
parameters.py ← (no dependencies)
gating.py ← (no dependencies)
currents.py ← gating.py
calcium.py ← (no dependencies)
camkii.py ← (no dependencies)
model.py ← parameters, gating, currents, calcium, camkii
```

### 10.2 Usage Example

```python
from ionic import ORdModel, CellType, StateIndex

# Create model
model = ORdModel(celltype=CellType.ENDO, device='cuda')

# Get initial state
state = model.get_initial_state()

# Run simulation
dt = 0.01  # ms
for step in range(50000):  # 500 ms
    t = step * dt
    Istim = -80.0 if 10.0 <= t < 11.0 else 0.0
    Istim_tensor = torch.tensor(Istim, device='cuda', dtype=torch.float64)
    state = model.step(state, dt, Istim_tensor)

    V = state[StateIndex.V].item()
    print(f"t={t:.1f} ms, V={V:.1f} mV")
```

---

---

## 11. Calibration Pipeline

### 11.1 Overview

The calibration module (`calibration/`) provides optimization-based parameter tuning
to find diffusion coefficients (D_L, D_T) that match target electrophysiological parameters.

### 11.2 2D Tissue ERP Measurement

**File:** `calibration/tissue_erp_2d.py`

The optimizer uses 2D tissue simulation to measure tissue ERP, which is essential
for capturing the combined effect of anisotropic diffusion on refractoriness.

**Geometry:**
```
┌─────────────────────────────────┐
│                                 │
│            probe_y              │  (ny-1, nx//2)
│               ↓                 │
│                                 │
│      [center]━━━━━━→ probe_x    │  (ny//2, nx-1)
│      S1, S2                     │
│                                 │
└─────────────────────────────────┘
```

**Protocol:**
1. Apply S1 stimulus at center → wave propagates to edges
2. Wait for repolarization
3. Apply S2 at center at decreasing intervals
4. **Single ERP** = minimum S1-S2 interval where **BOTH** probes activate

**Mesh Sizing:**
```
L_x = 1.5 × CV_x × ERP_est   (wavelength with margin)
L_y = 1.5 × CV_y × ERP_est

n_x = max(L_x / dx + 1, 50)  (minimum 50 cells)
n_y = max(L_y / dx + 1, 50)
```

**Usage:**
```python
from calibration import measure_tissue_erp_2d

result = measure_tissue_erp_2d(
    D_x=0.001,      # Longitudinal diffusion (cm²/ms)
    D_y=0.0005,     # Transverse diffusion (cm²/ms)
    dx=0.02,        # Mesh spacing (cm)
    dt=0.01,        # Time step (ms)
    cv_x_est=0.04,  # Estimated CV_L (cm/ms)
    cv_y_est=0.03,  # Estimated CV_T (cm/ms)
    erp_est=300.0,  # Estimated ERP for mesh sizing (ms)
    verbose=True
)

print(f"Tissue ERP = {result.erp} ms")  # Single ERP where both probes activate
print(f"ERP_x = {result.erp_x} ms")     # ERP at x-terminus only
print(f"ERP_y = {result.erp_y} ms")     # ERP at y-terminus only
```

### 11.3 Tiered Optimization

**File:** `calibration/optimizer.py`

The optimizer uses a tiered approach to find optimal D values:

**Tier 1: Fixed dt** (faster)
- Optimize [D_L, D_T] with dt = dt_default
- Accept if loss < threshold (0.1)

**Tier 2: Variable dt** (more flexible)
- If Tier 1 fails, optimize [D_L, D_T, dt]
- dt is regularized toward default (w_dt_reg = 0.01)
- D accuracy is prioritized over dt preference

**dt Bounds:**
```
dt_min = 0.001 ms (can go very small for stability)
dt_max = 0.5 × dx² / D_max (safety factor for explicit schemes)
```

### 11.4 Loss Function

```
L = w_cv_L × (CV_L - CV_L_target)² / CV_L²
  + w_cv_T × (CV_T - CV_T_target)² / CV_T²
  + w_apd × max(0, APD_min - APD)² / APD_min²  [penalty only]
  + w_erp_tissue × (ERP_tissue - ERP_target)² / ERP_target²
  + w_dt_reg × (dt - dt_default)² / dt_default²  [Tier 2 only]
```

**Default Weights:**
| Weight | Value | Description |
|--------|-------|-------------|
| w_cv_L | 1.0 | Longitudinal CV error |
| w_cv_T | 1.0 | Transverse CV error |
| w_apd | 0.5 | APD minimum constraint (one-sided) |
| w_erp_tissue | 5.0 | Primary objective (tissue ERP) |
| w_dt_reg | 0.01 | dt regularization (tiebreaker only) |

### 11.5 Quick Start

```python
from calibration import calibrate_diffusion

result = calibrate_diffusion(
    cv_longitudinal=0.06,    # 60 cm/s target
    anisotropy_ratio=3.0,    # D_L / D_T
    erp_tissue_target=320.0, # Target tissue ERP (ms)
    dx=0.01,                 # Mesh spacing (cm)
    dt_default=0.02,         # Default time step (ms)
    verbose=True
)

print(f"D_L = {result.D_longitudinal:.6f} cm²/ms")
print(f"D_T = {result.D_transverse:.6f} cm²/ms")
print(f"dt = {result.dt_used:.4f} ms")
print(f"Tier used: {result.tier_used}")
print(f"Tissue ERP = {result.erp_tissue:.1f} ms")
```

---

## References

1. O'Hara T, Virág L, Varró A, Rudy Y (2011). "Simulation of the Undiseased Human Cardiac Ventricular Action Potential: Model Formulation and Experimental Validation." PLoS Comput Biol 7(5): e1002061.

2. Rush S, Larsen H (1978). "A practical algorithm for solving dynamic membrane equations." IEEE Trans Biomed Eng 25(4):389-392.

3. ORd model C++ reference implementation: http://rudylab.wustl.edu/research/cell/code/
