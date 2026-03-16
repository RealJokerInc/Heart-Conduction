---
paper: hipsc_cm_maturation_models
title: "hiPSC-CM Ionic Models: Maturation, Automaticity, and the Path to Quiescent Models"
authors: "Literature survey (multi-paper)"
year: 2013-2026
journal: "Multiple"
questions: [Q8]
---

## The Problem

All published hiPSC-CM ionic models beat spontaneously. This is biologically correct — immature hiPSC-CMs exhibit automaticity due to:
1. **Low IK1** (inward rectifier K+): V_rest sits at -74 mV instead of -85 mV, too close to threshold
2. **Presence of If** (funny current): provides a slow depolarizing drift between APs, as in sinoatrial node pacemaker cells

For tissue-level simulation and parameter optimization, spontaneous beating causes:
- Paced AP detection failures (spontaneous rhythm competes with pacing)
- Unreliable CV measurement (probe points activate from spontaneous depolarization before wavefront arrival)
- Inability to define a stable resting state for parameter calibration

**The goal**: a model with hiPSC-CM-like current complement and AP morphology, but **quiescent** (fires only when paced), representing a matured hiPSC-CM phenotype.

---

## Published hiPSC-CM Ionic Models

| Model | Year | States | Currents | Spontaneous | Key Feature | PMID |
|-------|------|--------|----------|-------------|-------------|------|
| **Paci 2013** | 2013 | 17 | 12 (incl. If) | Yes | Original ventricular + atrial hiPSC-CM model | 23722932 |
| **Paci 2018** | 2018 | 17 | 12 | Yes | Updated Ca handling, refined with experimental data | 29925535 |
| **Koivumäki 2018** | 2018 | 17+ | 12 | Yes | Detailed immaturity analysis; novel Ca handling model | 29467678 |
| **Kernik 2019** | 2019 | 15 | 14 (incl. INaL, ICaT) | Yes | Population variability from experiments; 7 currents refitted | 31278749 |
| **Paci 2020** | 2020 | 17 | 12 | Yes | Population of 477 in silico cells; calibrated with all-optical AP+CaT data | 32298635 |
| **Akwaboah 2021** | 2021 | ~15 | 5 GA-fitted + others | Yes | GA-based parameter fitting to hiPSC-CM recordings | 34220540 |
| **Forouzandehmehr 2021** | 2021 | 17+ | 12 | Yes | Electromechanical model (AP + contraction) based on Paci | 34825519 |
| **Tomek 2026** | 2026 | ~20 | 12+ Markov ICaL | Configurable | Markov ICaL for Timothy Syndrome; most recent model | (Sci Rep 2026) |

**None of these models are quiescent by default.** All beat spontaneously.

---

## What Makes hiPSC-CMs Beat Spontaneously

### Koivumäki 2018 — Systematic Immaturity Analysis

**Paper**: "Structural Immaturity of Human iPSC-Derived Cardiomyocytes: In Silico Investigation of Effects on Function and Disease Modeling"
**Journal**: Frontiers in Physiology, 2018. PMID 29467678.

Key findings:
- Built a novel in silico model with immature Ca handling (2-compartment SR, underdeveloped t-tubule system)
- Sensitivity analysis on 3000 model variants: **IK1 is -59% lower** in the spontaneously active subpopulation vs the full database
- **"hiPSC-CMs are functionally closer to prenatal CMs than adult CMs"**
- **"Single modifications do not solve this problem"** — the immaturity is multi-factorial
- The immaturity is primarily in **intracellular calcium handling** (small SR, no t-tubules), not just membrane currents
- However, for tissue-level electrophysiology (CV, APD), **membrane currents dominate**, so IK1/If modifications may suffice

### The Two Key Currents

**IK1 (inward rectifier K+)**:
- In adult ventricular myocytes: high IK1 → V_rest = -85 mV, stable resting potential, no spontaneous depolarization
- In hiPSC-CMs: low IK1 → V_rest = -74 mV, insufficient to maintain stable quiescence
- IK1 has strong inward rectification: large inward current at hyperpolarized potentials, small outward current near V_rest → "clamps" the resting potential

**If (funny current)**:
- Present in pacemaker cells (SA node) and immature hiPSC-CMs
- Hyperpolarization-activated: opens at negative potentials, carries inward (depolarizing) mixed Na+/K+ current
- Provides the slow diastolic depolarization between APs
- Absent or minimal in adult ventricular myocytes

---

## Approaches to Achieving Quiescence

### Approach 1: IK1 Injection (Dynamic Clamp)

The experimental gold standard for "maturing" hiPSC-CMs electrophysiologically.

**Bett et al. 2013** — "Electronic Expression of the Inward Rectifier in Cardiocytes Derived from Human Induced Pluripotent Stem Cells"
- PMID: 24069225 (PMC3851822)
- First demonstration of electronic IK1 injection via dynamic clamp in hiPSC-CMs
- Results: V_rest hyperpolarized to -80 mV, spontaneous activity eliminated, AP upstroke velocity increased
- Used a simple Kir2.1-based IK1 formulation

**Verkerk et al. 2019** — "Required GK1 to Suppress Automaticity of iPSC-CMs Depends Strongly on IK1 Model Structure"
- PMID: 31623886 (PMC6990378)
- **The key computational paper for our purposes**
- Compared 5 mature IK1 formulations (Bett, Dhamoon, Ishihara, O'Hara-Rudy, ten Tusscher) applied to the Paci 2013 model
- Defined **GK1,critical**: the minimal IK1 conductance that suppresses all spontaneous activity
- Findings:
  - GK1,critical varies 4-fold depending on IK1 model choice (0.8–3.3 nS/pF)
  - The **Ishihara IK1** formulation recommended for dynamic clamp (best physiological rectification)
  - The **ten Tusscher (TTP06) IK1** also works well and is simpler
  - Once automaticity is suppressed: V_rest = -80 to -83 mV, dV/dt_max increases, APD changes modestly
  - The native Paci IK1 has fundamentally different rectification properties from mature IK1

**Verkerk & Bhatt 2023** — "Injection of IK1 through dynamic clamp can make all the difference in patch-clamp studies on hiPSC-derived cardiomyocytes"
- PMID: 38152247 (PMC10751953)
- Comprehensive review of IK1 dynamic clamp in hiPSC-CMs
- Confirms IK1 injection is now standard practice in electrophysiology labs
- Lists multiple independent implementations confirming the approach

### Approach 2: Model-Level Current Scaling

Rather than injecting an external current, scale the existing model parameters:

**Upscale IK1 (g_K1 × 3–5)**:
- Hyperpolarizes V_rest toward -80 to -85 mV
- Provides enough holding current to prevent spontaneous depolarization
- The Paci 2013 IK1 is a simplified formulation; for more physiological behavior, could replace it with the TTP06 or Ishihara IK1 formulation

**Downscale If (g_f × 0 or × 0.1)**:
- Removes or reduces the pacemaker depolarization
- Adult ventricular myocytes have negligible If
- Setting g_f = 0 is biologically justified for a "matured" phenotype

**Combined effect**: the model becomes quiescent, requiring external stimulus to fire, while retaining all other hiPSC-CM characteristics (ICaL GHK formulation, Ca-dependent IKs, smaller INa than TTP06, etc.)

### Approach 3: Build/Use a Purpose-Built Matured Model

**Paci 2020** (PMID 32298635):
- Updated from Paci 2018, calibrated against all-optical AP + CaT recordings
- Population of 477 in silico hiPSC-CMs
- Still spontaneous, but provides a more validated parameter space
- **Could be a better base for maturation than Paci 2013** due to refined kinetics

**Kernik 2019** (PMID 31278749):
- 14 currents including INaL and ICaT (not in Paci)
- Population variability built in from experiments
- More currents = more degrees of freedom for maturation tuning
- C++ implementation (would need PyTorch port)

**Tomek 2026** (Scientific Reports, in press):
- Most recent model, incorporates Markov ICaL
- Designed for disease modeling (Timothy Syndrome)
- Configurable for spontaneous vs paced behavior
- Not yet widely validated

---

## Experimental Data on Matured hiPSC-CMs

What do real matured hiPSC-CMs look like electrophysiologically?

**Metabolic maturation** (Correia et al. 2017, PMC7437654 review):
- Matured hiPSC-CMs in fatty acid media show:
  - V_rest = -80.3 ± 0.6 mV (near adult)
  - dV/dt_max = 250 ± 18 V/s (approaching adult ~300 V/s)
  - Low spontaneous beating frequency → can be fully quiescent
  - These values approach adult ventricular myocyte properties

**3D Engineered Heart Tissue** (Lemme et al. 2018, MacQueen et al. 2019):
- 3D culture substantially matures hiPSC-CMs
- CV = 15–25 cm/s (vs adult 60 cm/s)
- APD90 = 250–400 ms
- Reduced or absent spontaneous beating

**Target values for a matured hiPSC-CM model**:

| Property | Immature (Paci 2013) | Matured (experimental) | Adult (TTP06) |
|----------|---------------------|----------------------|---------------|
| V_rest | -74 mV | -80 mV | -86 mV |
| dV/dt_max | 23 V/s | 50–250 V/s | 300 V/s |
| APD90 | 469 ms | 250–400 ms | 280 ms |
| CV | — | 15–25 cm/s | 60 cm/s |
| IK1 | Low | Moderate–High | High |
| If | Present | Reduced/Absent | Absent |
| Spontaneous | Yes | No (or very slow) | No |

---

## Recommendation for This Project

### Preferred Approach: Modify PHAS13 with IK1 Upscaling + If Suppression

**Rationale**:
1. Already implemented and validated (38 tests)
2. Verkerk 2019 provides computational validation that IK1 upscaling on the Paci 2013 model suppresses automaticity
3. Two parameter changes (g_K1, g_f) — minimal code modification
4. Retains all hiPSC-CM-specific current formulations
5. The resulting model can be further tuned by the optimizer to match matured hiPSC-CM targets

**Specific modifications**:
- `g_K1`: scale by 3–5× (from 0.0281492 to ~0.085–0.14), or replace IK1 formulation with TTP06/Ishihara
- `g_f`: set to 0 (fully matured) or scale by 0.1 (partially matured)
- Optionally increase `g_Na` modestly (1.5–2×) to boost dV/dt_max toward matured values

**Expected outcome**:
- V_rest: -80 to -83 mV
- No spontaneous beating
- APD90: ~350–450 ms (shortened from 469 ms due to higher IK1)
- dV/dt_max: ~30–50 V/s (modestly increased from 23 V/s)

### Alternative: Port Paci 2020 or Kernik 2019

If the matured PHAS13 variant proves insufficient (e.g., Ca handling too immature for restitution studies), consider porting:
- **Paci 2020**: same framework as PHAS13, updated kinetics, would be a drop-in replacement
- **Kernik 2019**: more currents (INaL, ICaT), better experimental grounding, but requires C++ → PyTorch port

---

## References

| Key | Citation | PMID | Relevance |
|-----|----------|------|-----------|
| paci_2013 | Paci M et al. Ann Biomed Eng 2013;41(11):2334-48 | 23722932 | Base model (our PHAS13) |
| koivumaki_2018 | Koivumäki JT et al. Front Physiol 2018;9:80 | 29467678 | Immaturity analysis, Ca handling |
| kernik_2019 | Kernik DC et al. J Physiol 2019;597(17):4533-64 | 31278749 | Population variability model |
| verkerk_2019 | Verkerk AO et al. Biophys J 2019;117:2303-15 | 31623886 | **IK1 suppression of automaticity** |
| paci_2020 | Paci M et al. Biophys J 2020;118:2596-611 | 32298635 | Updated model, population |
| akwaboah_2021 | Akwaboah AD et al. Front Physiol 2021;12:675867 | 34220540 | GA-fitted hiPSC-CM model |
| bett_2013 | Bett GC et al. Heart Rhythm 2013;10:1903-10 | 24069225 | First IK1 electronic expression |
| verkerk_2023 | Verkerk AO, Bhatt R. Front Physiol 2023;14:1326160 | 38152247 | IK1 dynamic clamp review |
| tomek_2026 | Tomek J et al. Sci Rep 2026 (in press) | — | Markov ICaL, Timothy Syndrome |
