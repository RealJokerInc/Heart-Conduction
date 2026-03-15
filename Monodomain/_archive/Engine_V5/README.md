# Engine V5: O'Hara-Rudy (ORd 2011) - GPU-Accelerated Implementation

## Status: PLANNING PHASE - Structure Updated for ORd Model

### Documentation
- **[VALIDATION.md](VALIDATION.md)** - Validation tests and pass/fail criteria
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Step-by-step implementation guide

---

## Model Selection: O'Hara-Rudy (ORd 2011)

We are implementing the **O'Hara-Rudy dynamic (ORd)** model for the undiseased human ventricular action potential and calcium transient.

**Key Features:**
- **Human ventricular myocyte** model (validated against human data)
- **41 state variables** (comprehensive calcium and sodium handling)
- **Subspace calcium compartment** (cass) for local Ca2+ signaling
- **CaMKII signaling** integrated into ion channel gating
- **Cell-type specific** parameters (endocardial, epicardial, M-cell)
- **Late sodium current (INaL)** for APD modulation

**Reference**: O'Hara T, Virag L, Varro A, Rudy Y. "Simulation of the Undiseased Human Cardiac Ventricular Action Potential: Model Formulation and Experimental Validation." PLoS Comput Biol. 2011;7(5):e1002061.

**Links:**
- [PLoS Comput Biol Article](http://www.ploscompbiol.org/article/info:doi/10.1371/journal.pcbi.1002061)
- [Rudy Lab Code Downloads](https://rudylab.wustl.edu)

---

## ORd Complete Specification (From Reference Code)

### Ionic Currents (15 total)

| Current | Description | Type | Key Parameter |
|---------|-------------|------|---------------|
| **INa** | Fast sodium | Ohmic, voltage-gated | GNa = 75 mS/uF |
| **INaL** | Late sodium | Ohmic, CaMKII-modulated | GNaL = 0.0075 mS/uF |
| **Ito** | Transient outward K+ | Ohmic, CaMKII-modulated | Gto = 0.02 mS/uF |
| **ICaL** | L-type calcium | **GHK**, CaMKII-modulated | PCa = 0.0001 cm/s |
| **ICaNa** | L-type Na component | GHK | PCaNa = 0.00125*PCa |
| **ICaK** | L-type K component | GHK | PCaK = 3.574e-4*PCa |
| **IKr** | Rapid delayed rectifier K+ | Ohmic with rectification | GKr = 0.046 mS/uF |
| **IKs** | Slow delayed rectifier K+ | Ohmic, **Ca-dependent** | GKs = 0.0034 mS/uF |
| **IK1** | Inward rectifier K+ | Instantaneous rectifier | GK1 = 0.1908 mS/uF |
| **INaCa_i** | Na/Ca exchanger (bulk) | Electrogenic | Gncx = 0.0008 |
| **INaCa_ss** | Na/Ca exchanger (subspace) | Electrogenic | 0.2*Gncx |
| **INaK** | Na/K pump | Electrogenic | Pnak = 30 uA/uF |
| **IKb** | Background K+ | Ohmic | GKb = 0.003 mS/uF |
| **INab** | Background Na+ | GHK | PNab = 3.75e-10 cm/s |
| **ICab** | Background Ca2+ | GHK | PCab = 2.5e-8 cm/s |
| **IpCa** | Sarcolemmal Ca pump | Michaelis-Menten | GpCa = 0.0005 uA/uF |

### State Variables (41 total)

**Membrane Potential (1)**
| Index | Variable | Description | Initial Value |
|-------|----------|-------------|---------------|
| 1 | v | Membrane potential (mV) | -87.5 |

**Ion Concentrations - Bulk (4)**
| Index | Variable | Description | Initial Value |
|-------|----------|-------------|---------------|
| 2 | nai | Intracellular Na+ (mM) | 7.0 |
| 4 | ki | Intracellular K+ (mM) | 145.0 |
| 6 | cai | Intracellular Ca2+ (mM) | 1.0e-4 |
| 8 | cansr | Network SR Ca2+ (mM) | 1.2 |

**Ion Concentrations - Subspace (4)**
| Index | Variable | Description | Initial Value |
|-------|----------|-------------|---------------|
| 3 | nass | Subspace Na+ (mM) | 7.0 |
| 5 | kss | Subspace K+ (mM) | 145.0 |
| 7 | cass | Subspace Ca2+ (mM) | 1.0e-4 |
| 9 | cajsr | Junctional SR Ca2+ (mM) | 1.2 |

**INa Gates (6)** - Fast and slow inactivation with CaMKII
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 10 | m | Activation | 0 |
| 11 | hf | Fast inactivation | 1 |
| 12 | hs | Slow inactivation | 1 |
| 13 | j | Recovery | 1 |
| 14 | hsp | Slow inactivation (CaMKII) | 1 |
| 15 | jp | Recovery (CaMKII) | 1 |

**INaL Gates (3)** - Late sodium current
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 16 | mL | Activation | 0 |
| 17 | hL | Inactivation | 1 |
| 18 | hLp | Inactivation (CaMKII) | 1 |

**Ito Gates (6)** - Transient outward current
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 19 | a | Activation | 0 |
| 20 | iF | Fast inactivation | 1 |
| 21 | iS | Slow inactivation | 1 |
| 22 | ap | Activation (CaMKII) | 0 |
| 23 | iFp | Fast inactivation (CaMKII) | 1 |
| 24 | iSp | Slow inactivation (CaMKII) | 1 |

**ICaL Gates (9)** - L-type calcium with CDI/VDI
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 25 | d | Activation | 0 |
| 26 | ff | Fast voltage inactivation | 1 |
| 27 | fs | Slow voltage inactivation | 1 |
| 28 | fcaf | Fast Ca inactivation | 1 |
| 29 | fcas | Slow Ca inactivation | 1 |
| 30 | jca | Ca-dependent recovery | 1 |
| 31 | nca | Ca/CaM binding | 0 |
| 32 | ffp | Fast voltage inact (CaMKII) | 1 |
| 33 | fcafp | Fast Ca inact (CaMKII) | 1 |

**IKr Gates (2)** - Rapid delayed rectifier
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 34 | xrf | Fast activation | 0 |
| 35 | xrs | Slow activation | 0 |

**IKs Gates (2)** - Slow delayed rectifier
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 36 | xs1 | Activation 1 | 0 |
| 37 | xs2 | Activation 2 | 0 |

**IK1 Gate (1)**
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 38 | xk1 | Activation | 1 |

**SR Release (2)** - CaMKII-modulated
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 39 | Jrelnp | Release (non-phosphorylated) | 0 |
| 40 | Jrelp | Release (phosphorylated) | 0 |

**CaMKII (1)**
| Index | Variable | Description | Initial |
|-------|----------|-------------|---------|
| 41 | CaMKt | Trapped CaMKII | 0 |

### Cell-Type Scaling Factors

| Parameter | Endo (0) | Epi (1) | M-cell (2) |
|-----------|----------|---------|------------|
| GNaL | 1.0 | 0.6 | 1.0 |
| Gto | 1.0 | 4.0 | 4.0 |
| PCa | 1.0 | 1.2 | 2.5 |
| GKr | 1.0 | 1.3 | 0.8 |
| GKs | 1.0 | 1.4 | 1.0 |
| GK1 | 1.0 | 1.2 | 1.3 |
| Gncx | 1.0 | 1.1 | 1.4 |
| Pnak | 1.0 | 0.9 | 0.7 |
| GKb | 1.0 | 0.6 | 1.0 |
| Jrel | 1.0 | 1.0 | 1.7 |
| Jup | 1.0 | 1.3 | 1.0 |
| cmdnmax | 1.0 | 1.3 | 1.0 |

### CaMKII Signaling

CaMKII (Ca2+/calmodulin-dependent protein kinase II) modulates multiple currents:

```
CaMKb = CaMKo * (1 - CaMKt) / (1 + KmCaM/cass)
CaMKa = CaMKb + CaMKt
dCaMKt/dt = aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt
```

**Parameters:**
- aCaMK = 0.05 (activation rate)
- bCaMK = 0.00068 (deactivation rate)
- CaMKo = 0.05 (total CaMKII)
- KmCaM = 0.0015 mM (CaM binding Kd)
- KmCaMK = 0.15 (CaMKII effect Kd)

**Phosphorylation fraction:**
```
fCaMKp = 1 / (1 + KmCaMK/CaMKa)
```

This scales the contribution of phosphorylated channel populations.

### SR Calcium Handling

**Diffusion fluxes (subspace ↔ bulk):**
```
JdiffNa = (nass - nai) / 2.0      [mM/ms]
JdiffK = (kss - ki) / 2.0         [mM/ms]
Jdiff = (cass - cai) / 0.2        [mM/ms]
```

**SR Release (RyR - Ryanodine receptor):**
```
Jrel_inf = a_rel * (-ICaL) / (1 + (1.5/cajsr)^8)
tau_rel = bt / (1 + 0.0123/cajsr)
```
- bt = 4.75 ms (time constant base)
- a_rel = 0.5 * bt (release gain)
- Steep JSR-dependence (n=8) provides graded release

**SERCA Uptake:**
```
Jupnp = 0.004375 * cai / (cai + 0.00092)      [non-phosphorylated]
Jupp = 2.75 * 0.004375 * cai / (cai + 0.00092 - 0.00017)  [phosphorylated]
Jup = (1-fJupp)*Jupnp + fJupp*Jupp - Jleak
```

**SR Transfer & Leak:**
```
Jtr = (cansr - cajsr) / 100.0     [NSR → JSR]
Jleak = 0.0039375 * cansr / 15.0  [NSR → cytosol]
```

### Calcium Buffering

**Cytoplasmic buffers:**
```
Bcai = 1 / (1 + cmdnmax*kmcmdn/(kmcmdn+cai)^2 + trpnmax*kmtrpn/(kmtrpn+cai)^2)
```
- cmdnmax = 0.05 mM (calmodulin)
- trpnmax = 0.07 mM (troponin)
- kmcmdn = 0.00238 mM
- kmtrpn = 0.0005 mM

**Subspace buffers:**
```
Bcass = 1 / (1 + BSRmax*KmBSR/(KmBSR+cass)^2 + BSLmax*KmBSL/(KmBSL+cass)^2)
```
- BSRmax = 0.047 mM (SR binding sites)
- BSLmax = 1.124 mM (sarcolemmal binding sites)
- KmBSR = 0.00087 mM
- KmBSL = 0.0087 mM

**JSR buffer (calsequestrin):**
```
Bcajsr = 1 / (1 + csqnmax*kmcsqn/(kmcsqn+cajsr)^2)
```
- csqnmax = 10.0 mM
- kmcsqn = 0.8 mM

---

## Key Parameters (From Reference Code)

### Physical Constants
| Parameter | Value | Units |
|-----------|-------|-------|
| F | 96485 | C/mol |
| R | 8314 | mJ/(mol*K) |
| T | 310 | K |

### Extracellular Concentrations
| Parameter | Value | Units |
|-----------|-------|-------|
| nao | 140 | mM |
| ko | 5.4 | mM |
| cao | 1.8 | mM |

### Cell Geometry
| Parameter | Value | Units |
|-----------|-------|-------|
| L | 0.01 | cm |
| rad | 0.0011 | cm |
| vcell | 3.8e-5 | uL |
| vmyo | 0.68 * vcell | uL |
| vnsr | 0.0552 * vcell | uL |
| vjsr | 0.0048 * vcell | uL |
| vss | 0.02 * vcell | uL |

### Stimulus
| Parameter | Value | Units |
|-----------|-------|-------|
| amp | -80 | uA/uF |
| duration | 0.5 | ms |

---

## File Structure

```
Engine_V5/
├── ionic/
│   ├── __init__.py
│   ├── ord_model.py          # Main ORd model class + state management
│   ├── gating.py             # All gating kinetics (INa, INaL, Ito, ICaL, IKr, IKs, IK1)
│   ├── currents.py           # All 15 ionic current functions
│   ├── calcium.py            # SR dynamics, diffusion fluxes, buffering
│   ├── camkii.py             # CaMKII signaling module
│   └── parameters.py         # ORd-specific parameters + cell-type scaling
├── solvers/
│   ├── __init__.py
│   ├── cpu_kernel.py         # Complete Numba CPU kernel
│   └── gpu_kernel.py         # Complete Numba CUDA kernel
├── tissue/
│   ├── __init__.py
│   ├── diffusion.py          # Anisotropic diffusion operator
│   └── simulation.py         # 2D tissue simulation
├── utils/
│   ├── __init__.py
│   ├── constants.py          # Physical constants (R, T, F)
│   └── validation.py         # APD, CV measurement tools
├── examples/
│   ├── __init__.py
│   ├── single_cell.py        # Basic AP simulation
│   ├── cell_types.py         # Endo/epi/M-cell comparison
│   ├── restitution.py        # APD restitution curve
│   └── rate_dependence.py    # Steady-state rate adaptation
├── O'Hara_ORd_MATLAB_2011/    # Original MATLAB source
├── O'Hara_ORd_Cpp_2011/       # Original C++ source
├── Livshitz_LRd_CaMKII_2007/  # Legacy LRd07 reference
├── README.md
├── IMPLEMENTATION.md
└── VALIDATION.md
```

**File purposes:**
- `gating.py`: Gate kinetics for all voltage-gated channels with fast/slow components
- `currents.py`: 15 ionic current functions with CaMKII modulation
- `calcium.py`: Subspace diffusion, SR release/uptake, buffering
- `camkii.py`: CaMKII activation dynamics and phosphorylation fractions
- `ord_model.py`: Main class integrating all components
- `parameters.py`: All model constants + cell-type scaling factors
- `cpu_kernel.py`: Fused Numba kernel for single-cell/tissue
- `gpu_kernel.py`: CUDA kernel for parallel tissue simulation

---

## Development Strategy: Accuracy First

### Phase 1: Accurate Single Cell (CPU)
1. Translate MATLAB/C++ ORd code to Python
2. Implement all 15 currents correctly (GHK for ICaL/INab/ICab)
3. Implement 41 state variables with proper indexing
4. Implement CaMKII signaling and modulation
5. Implement subspace calcium handling
6. Validate against reference code output

### Phase 2: Validation Suite
1. Compare against published ORd figures
2. APD for endo/epi/M-cells at BCL=1000ms
3. Ca2+ transient amplitude and kinetics
4. Rate-dependent APD adaptation
5. Cell-type specific AP morphology

### Phase 3: GPU Acceleration
1. Port validated CPU kernel to CUDA
2. Memory layout optimization (SoA)
3. Benchmark and profile
4. Verify GPU matches CPU results

### Phase 4: Tissue Simulation
1. 2D diffusion with anisotropy
2. Wave propagation with endo/epi layers
3. Transmural heterogeneity
4. Spiral wave dynamics

---

## Comparison: LRd07 vs ORd

| Feature | LRd07 | ORd |
|---------|-------|-----|
| Species | Guinea pig | Human |
| State variables | 18 | 41 |
| INaL (late sodium) | No | Yes |
| Subspace Ca2+ | No | Yes (cass) |
| Subspace Na+ | No | Yes (nass) |
| Cell types | Single | Endo/Epi/M |
| ICaL gating | d, f, fCa | d, ff, fs, fcaf, fcas, jca, nca |
| CaMKII | Basic | Full integration |
| Background currents | Ohmic | GHK (INab, ICab) |

---

## Resources

- [PLoS Comput Biol Article](http://www.ploscompbiol.org/article/info:doi/10.1371/journal.pcbi.1002061) - Original paper (Open Access)
- [Rudy Lab](https://rudylab.wustl.edu) - Official source code
- [CellML ORd Model](https://models.cellml.org/exposure/a7179d94365ff0c9c0e6eb7c6a787d3d) - CellML version

---

*Updated: 2024-12-21*
*Engine V5 Planning Document - O'Hara-Rudy (ORd 2011)*
