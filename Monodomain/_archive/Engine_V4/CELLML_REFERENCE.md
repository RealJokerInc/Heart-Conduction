# CellML LRd94 Reference Parameters and Equations

## Unit System (CellML)
- Length: mm
- Time: ms
- Voltage: mV
- Capacitance: µF/mm² (Cm = 0.01)
- Conductance: mS/mm²
- Current density: µA/mm²
- Permeability: mm/ms
- Concentration: mM

## Unit Conversion to cm² basis
Since 1 cm² = 100 mm²:
- Conductance: mS/mm² × 100 = mS/cm²
- Capacitance: µF/mm² × 100 = µF/cm² (so Cm = 1.0 µF/cm²)

## Parameters (CellML values → cm² equivalent)

### Physical Constants
| Parameter | CellML Value | cm² Equivalent | Unit |
|-----------|--------------|----------------|------|
| R | 8314.5 | 8314.5 | mJ/(mol·K) |
| T | 310.0 | 310.0 | K |
| F | 96845.0 | 96845.0 | C/mol |
| Cm | 0.01 | 1.0 | µF/mm² → µF/cm² |

### Extracellular Concentrations
| Parameter | Value | Unit |
|-----------|-------|------|
| Nao | 140.0 | mM |
| Ko | 5.4 | mM |
| Cao | 1.8 | mM |

### Conductances (CellML → cm² × 100)
| Parameter | CellML (mS/mm²) | cm² (mS/cm²) |
|-----------|-----------------|--------------|
| g_Na | 0.16 | 16.0 |
| g_K_max | 2.82e-3 | 0.282 |
| g_K1_max | 7.5e-3 | 0.75 |
| g_Kp | 1.83e-4 | 0.0183 |
| g_Nab | 1.41e-5 | 0.00141 |
| g_Cab | 3.016e-5 | 0.003016 |

### L-type Ca Channel (GHK) Permeabilities
| Parameter | Value | Unit |
|-----------|-------|------|
| P_Ca | 5.4e-6 | mm/ms |
| P_Na | 6.75e-9 | mm/ms |
| P_K | 1.93e-9 | mm/ms |

### Activity Coefficients
| Parameter | Value |
|-----------|-------|
| gamma_Cai | 1.0 |
| gamma_Cao | 0.34 |
| gamma_Nai | 0.75 |
| gamma_Nao | 0.75 |
| gamma_Ki | 0.75 |
| gamma_Ko | 0.75 |

### Pump/Exchanger Parameters
| Parameter | CellML | cm² Equiv | Unit |
|-----------|--------|-----------|------|
| I_NaK | 1.5e-2 | 1.5 | µA/mm² → µA/cm² |
| K_mNai | 10.0 | 10.0 | mM |
| K_mKo | 1.5 | 1.5 | mM |
| I_pCa | 1.15e-2 | 1.15 | µA/mm² → µA/cm² |
| K_mpCa | 0.5e-3 | 0.5e-3 | mM |
| K_NaCa | 20.0 | 2000.0 | (scaled by area) |
| K_mNa | 87.5 | 87.5 | mM |
| K_mCa | 1.38 | 1.38 | mM |
| K_sat | 0.1 | 0.1 | - |
| eta | 0.35 | 0.35 | - |

### SR Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| G_rel_max | 60.0 | 1/ms |
| tau_tr | 180.0 | ms |
| K_mrel | 0.8e-3 | mM |
| K_mup | 0.92e-3 | mM |
| I_up | 0.005 | mM/ms |

### Ca Channel Gating
| Parameter | Value | Unit |
|-----------|-------|------|
| Km_Ca (f_Ca) | 0.6e-3 | mM |

## Key Equations

### I_Ca_L (GHK formulation)
```
# Driving force terms
exp_2VF = exp(2 * V * F / (R * T))
exp_VF = exp(V * F / (R * T))
VF_RT = V * F / (R * T)

# Calcium component
if |V| < 1e-4:
    i_CaCa = P_Ca * 4 * F * (gamma_Cai * Cai - gamma_Cao * Cao)
else:
    i_CaCa = P_Ca * 4 * F * VF_RT * (gamma_Cai * Cai * exp_2VF - gamma_Cao * Cao) / (exp_2VF - 1)

# Sodium component
if |V| < 1e-4:
    i_CaNa = P_Na * F * (gamma_Nai * Nai - gamma_Nao * Nao)
else:
    i_CaNa = P_Na * F * VF_RT * (gamma_Nai * Nai * exp_VF - gamma_Nao * Nao) / (exp_VF - 1)

# Potassium component
if |V| < 1e-4:
    i_CaK = P_K * F * (gamma_Ki * Ki - gamma_Ko * Ko)
else:
    i_CaK = P_K * F * VF_RT * (gamma_Ki * Ki * exp_VF - gamma_Ko * Ko) / (exp_VF - 1)

# Total (gated)
I_Ca_L = d * f * f_Ca * (i_CaCa + i_CaNa + i_CaK)
```

### f_Ca gate (instantaneous)
```
f_Ca = 1 / (1 + (Cai / Km_Ca)^2)
```
Note: Hill coefficient = 2

### d gate
```
d_inf = 1 / (1 + exp(-(V + 10) / 6.24))
tau_d = d_inf * (1 - exp(-(V + 10) / 6.24)) / (0.035 * (V + 10))
# Handle V = -10 singularity
```

### f gate
```
f_inf = 1 / (1 + exp((V + 35.06) / 8.6)) + 0.6 / (1 + exp((50 - V) / 20))
tau_f = 1 / (0.0197 * exp(-(0.0337 * (V + 10))^2) + 0.02)
```

## Initial Conditions
| Variable | Value | Unit |
|----------|-------|------|
| V | -84.624 | mV |
| m | 0.0 | - |
| h | 1.0 | - |
| j | 1.0 | - |
| d | 0.0 | - |
| f | 1.0 | - |
| Cai | 0.12e-3 | mM (120 nM) |
| Nai | 10.0 | mM |
| Ki | 145.0 | mM |
| Ca_JSR | 1.8 | mM |
| Ca_NSR | 1.8 | mM |

## Expected Output (Original LRd94)
- APD90: ~300 ms
- V_rest: -84.6 mV
- V_peak: ~40 mV (with GHK)
- [Ca2+]i peak: 500-1000 nM
