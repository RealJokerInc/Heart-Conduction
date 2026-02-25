# Engine V4: Luo-Rudy 1994 (LRd) Implementation Guide

## Overview

Replace the Fenton-Karma 3-variable phenomenological model with the full Luo-Rudy 1994
biophysical model featuring 13 state variables, 10 ionic currents, and GHK calcium dynamics.

**Reference**: Luo CH, Rudy Y. "A dynamic model of the cardiac ventricular action potential.
I. Simulations of ionic currents and concentration changes." Circ Res. 1994;74(6):1071-1096.

---

## Model Specifications

### State Variables (13)
| Variable | Description | Initial Value | Unit |
|----------|-------------|---------------|------|
| V | Membrane potential | -84.624 | mV |
| m | INa activation | 0.00136 | - |
| h | INa fast inactivation | 0.9814 | - |
| j | INa slow inactivation | 0.9905 | - |
| d | ICaL activation | 3.0e-6 | - |
| f | ICaL voltage inactivation | 1.0 | - |
| f_Ca | ICaL calcium inactivation | 1.0 | - |
| X | IK activation | 0.0057 | - |
| Na_i | Intracellular Na+ | 10.0 | mM |
| K_i | Intracellular K+ | 145.0 | mM |
| Ca_i | Intracellular Ca2+ | 0.00012 | mM |
| Ca_jsr | JSR Ca2+ | 1.8 | mM |
| Ca_nsr | NSR Ca2+ | 1.8 | mM |

### Ionic Currents (10)
| Current | Type | Description |
|---------|------|-------------|
| I_Na | Ohmic | Fast sodium current |
| I_CaL | **GHK** | L-type calcium current |
| I_K | Ohmic | Time-dependent potassium |
| I_K1 | Ohmic | Inward rectifier potassium |
| I_Kp | Ohmic | Plateau potassium |
| I_NaCa | Non-linear | Na+/Ca2+ exchanger |
| I_NaK | Non-linear | Na+/K+ pump |
| I_pCa | Michaelis-Menten | Sarcolemmal Ca2+ pump |
| I_Na_b | Ohmic | Background sodium |
| I_Ca_b | Ohmic | Background calcium |

### Key Parameters (from CellML)
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| R | 8314.0 | mJ/(mol·K) | Gas constant |
| T | 310.0 | K | Temperature (37°C) |
| F | 96485.0 | C/mol | Faraday constant |
| Cm | 1.0 | µF/cm² | Membrane capacitance |
| g_Na | 16.0 | mS/cm² | Na+ conductance |
| P_Ca | 5.4e-4 | cm/s | Ca2+ permeability |
| P_Na | 6.75e-7 | cm/s | Na+ permeability (ICaL) |
| P_K | 1.93e-7 | cm/s | K+ permeability (ICaL) |
| g_K | 0.282 | mS/cm² | K+ conductance |
| g_K1 | 0.6047 | mS/cm² | IK1 conductance |
| Na_o | 140.0 | mM | Extracellular Na+ |
| K_o | 5.4 | mM | Extracellular K+ |
| Ca_o | 1.8 | mM | Extracellular Ca2+ |

---

## Implementation Phases

### Phase 1: Parameters & Gating Kinetics ✅ COMPLETE
**Files**: `parameters_lrd.py`, `gating_lrd.py`
**Completed**: 2024-12

**Deliverables**:
- [x] LRdParams dataclass with all 60+ constants
- [x] LRdInitialConditions dataclass (13 state variables)
- [x] All alpha/beta rate functions for 8 gating variables
- [x] Numba-accelerated gating kernels
- [x] Rush-Larsen update functions
- [x] Unit test with plots: `gating_lrd_test.png`

**Validation Results**:
| Gate | Value at Rest (-84mV) | Expected |
|------|----------------------|----------|
| m_inf | 0.0018 | ~0 (closed) |
| h_inf | 0.981 | ~1 (open) |
| j_inf | 0.988 | ~1 (open) |
| d_inf(0mV) | 0.83 | High |
| f_inf(0mV) | 0.06 | Low |
| tau_m(-40mV) | 0.13 ms | Fast |
| tau_f(0mV) | 26.6 ms | Slow |

---

### Phase 2: Ionic Currents ✅ COMPLETE
**Files**: `currents_lrd.py`
**Completed**: 2024-12

**Deliverables**:
- [x] GHK flux equation with activity coefficients
- [x] All 10 ionic current functions (Numba-accelerated)
- [x] Reversal potential calculations (Nernst)
- [x] Helper functions for concentration updates
- [x] Unit test with I-V plots: `currents_lrd_test.png`

**Validation Results**:
| Reversal | Value | Expected |
|----------|-------|----------|
| E_Na | +70.5 mV | ~+70 mV |
| E_K | -87.9 mV | ~-90 mV |
| E_Ca | +128.4 mV | ~+130 mV |

**ICaL GHK Verification**: Non-linear I-V curve confirmed (curved, not straight)

---

### Phase 3: Calcium Dynamics ⬅️ NEXT
**Files**: `calcium_dynamics_lrd.py`

**Deliverables**:
- [ ] CICR (calcium-induced calcium release) from JSR
- [ ] SERCA uptake into NSR
- [ ] JSR-NSR transfer
- [ ] Cytoplasmic buffering (troponin, calmodulin)
- [ ] JSR buffering (calsequestrin)

**Validation**:
- Ca2+ transient shape during AP
- Peak [Ca2+]i ~ 1-2 µM
- Proper SR refilling dynamics

---

### Phase 4: Ionic Model Assembly
**Files**: `luo_rudy_1994.py`

**Deliverables**:
- [ ] LuoRudy1994Model class
- [ ] initialize_state() method
- [ ] ionic_step() with Numba kernel
- [ ] Rush-Larsen integration for stiff gates
- [ ] Interface matching FentonKarmaModel

**Validation**:
- Single-cell AP waveform
- APD90 ~ 200-250 ms
- dV/dt_max ~ 200-400 V/s
- Proper resting potential ~ -85 mV

---

### Phase 5: Single-Cell Testing
**Files**: `test_single_cell.py`

**Deliverables**:
- [ ] AP morphology plots
- [ ] All ionic currents during AP
- [ ] Gating variable dynamics
- [ ] Ca2+ transient
- [ ] Restitution curve (APD vs DI)
- [ ] Rate dependence (different BCLs)

**Validation**:
- Compare against Fig. 2-5 in Luo & Rudy 1994
- Verify physiological APD range

---

### Phase 6: Tissue Simulation
**Files**: `simulation_v4.py`

**Deliverables**:
- [ ] CardiacSimulation2D class with LRd model
- [ ] Copy diffusion.py from V3
- [ ] Operator splitting integration
- [ ] Stimulus protocols

**Validation**:
- Wave propagation from edge stimulus
- No numerical instability

---

### Phase 7: Tissue Testing
**Files**: `test_tissue.py`

**Deliverables**:
- [ ] Planar wave propagation
- [ ] CV measurement (target 0.5-0.7 mm/ms)
- [ ] S1-S2 spiral initiation
- [ ] Performance benchmarks

**Validation**:
- Physiological CV
- Stable long-duration runs

---

## File Structure

```
Engine_V4/
├── __init__.py              # Package init
├── IMPLEMENTATION.md        # This file
├── parameters_lrd.py        # Phase 1 ✅
├── gating_lrd.py            # Phase 1 ✅
├── gating_lrd_test.png      # Phase 1 validation plot
├── currents_lrd.py          # Phase 2 ✅
├── currents_lrd_test.png    # Phase 2 validation plot
├── calcium_dynamics_lrd.py  # Phase 3 (TODO)
├── luo_rudy_1994.py         # Phase 4 (TODO)
├── test_single_cell.py      # Phase 5 (TODO)
├── diffusion.py             # Phase 6 (copy from V3)
├── simulation_v4.py         # Phase 6 (TODO)
└── test_tissue.py           # Phase 7 (TODO)
```

---

## Technical Notes

### GHK vs Ohmic
The L-type calcium current uses the Goldman-Hodgkin-Katz equation because:
- Intracellular Ca2+ changes 100-1000x during AP
- Concentration ratio [Ca]o/[Ca]i can reach 20,000:1
- Ohmic approximation breaks down at extreme ratios

### Integration Method
- **Gating variables**: Rush-Larsen (exponential integrator) for stability
- **Concentrations**: Forward Euler with buffering
- **Voltage**: Forward Euler

### Time Step Requirements
- Minimum dt: 0.005 ms (conservative)
- Recommended dt: 0.01 ms (balanced)
- Maximum dt: 0.02 ms (may be unstable)

### Known Errata
- Original paper has error in f-gate steady-state equation
- Corrected in companion paper (Circ Res 74:1097-113, 1994)
- CellML version uses corrected equations

---

## References

1. Luo CH, Rudy Y. Circ Res. 1994;74(6):1071-1096 (Part I)
2. Luo CH, Rudy Y. Circ Res. 1994;74(6):1097-1113 (Part II)
3. CellML Model: models.cellml.org/e/81
4. Rudy Lab: rudylab.wustl.edu

---

## Progress Tracking

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| 1 | ✅ COMPLETE | 2024-12 | Parameters + Gating (validated) |
| 2 | ✅ COMPLETE | 2024-12 | Currents + GHK ICaL (validated) |
| 3 | ⬅️ NEXT | | Calcium dynamics (SR, buffering) |
| 4 | PENDING | | Model assembly |
| 5 | PENDING | | Single-cell tests |
| 6 | PENDING | | Tissue simulation |
| 7 | PENDING | | Tissue tests |

---

## Quick Start (Next Session)

To continue implementation:

```bash
cd "/Users/lemon/Documents/Python/Heart Conduction/Monodomain/Engine_V4"
python3.11 gating_lrd.py      # Verify Phase 1 still works
python3.11 currents_lrd.py    # Verify Phase 2 still works
```

**Next step**: Implement `calcium_dynamics_lrd.py` (Phase 3)
- CICR from JSR
- SERCA uptake
- Cytoplasmic buffering (troponin, calmodulin)
- JSR buffering (calsequestrin)
