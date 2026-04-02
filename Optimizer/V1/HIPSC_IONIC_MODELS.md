# hiPSC-CM Ionic Models — Survey & Implementation Priority

## Available Models (Downloaded)

Three hiPSC-CM specific ionic models have been downloaded to `Research/code_examples/hipsc_ionic_models/`:

### 1. Kernik 2019 (ClancyLab, UC Davis)

```
Path: hipsc_ionic_models/kernik_2019_ipsc_model/
Code: C++ (main_ipsc_baseline.cpp) + MATLAB plotting
Repo: github.com/ClancyLabUCD/IPSC-model
Paper: Kernik et al., "A computational model of induced pluripotent
       stem-cell derived cardiomyocytes incorporating experimental
       variability from multiple data sources"
       J Physiol, 2019. DOI: 10.1113/JP277724
```

**Currents**: INa, INaL, ICaL, ICaT, IKr, IKs, IK1, Ito, If, INaCa, INaK, IpCa, IbNa, IbCa, ISERCA
**States**: 15 state variables
**Key feature**: Parameterized from multiple experimental iPSC-CM datasets. Includes experimental variability distributions for each current.
**Language**: C++ with MATLAB analysis scripts

### 2. Akwaboah 2021 (Bhatt lab)

```
Path: hipsc_ionic_models/akwaboah_2021_hipsc_ga/
Code: Python (Jupyter notebook) + C++ + GA fitting scripts
Repo: github.com/Adakwaboah/hiPSC-CM_Computational_Model
Paper: Akwaboah et al., "An in silico hiPSC-Derived Cardiomyocyte
       Model Built With Genetic Algorithm"
       Front Physiol, 2021. DOI: 10.3389/fphys.2021.675867
```

**Currents**: INa, ICaL, IKr, IK1, If (5 currents fitted by GA to hiPSC-CM data)
**States**: Based on LRd model framework, refitted
**Key feature**: **GA fitting code included** for each current individually. Has Python code. Directly relevant to our optimizer.
**Language**: Python + C++

### 3. Paci 2013 (Tampere University) — via CardiacModelling tailored-ipsc-models

```
Path: hipsc_ionic_models/tailored_ipsc_models/
Code: Myokit (.mmt) + Python analysis scripts
Repo: github.com/CardiacModelling/tailored-ipsc-models
Paper: Paci et al., "Computational Models of Ventricular- and
       Atrial-Like Human Induced Pluripotent Stem Cell Derived
       Cardiomyocytes"
       Ann Biomed Eng, 2013. DOI: 10.1007/s10439-013-0833-3
Also:  Lei et al., "Tailoring mathematical models to stem-cell derived
       cardiomyocyte lines can improve predictions of drug-induced
       changes to their electrophysiology"
       Front Physiol, 2023. DOI: 10.3389/fphys.2023.1126695
```

**Currents**: INa, ICaL, IKr, IKs, IK1, Ito, If, INaCa, INaK, IpCa, IbNa, IbCa
**States**: 16 state variables (ventricular variant)
**Key feature**: Most widely cited hiPSC-CM model. Includes the **funny current (If)** which is absent from TTP06. The tailored-ipsc-models repo also contains code for fitting Paci/ORd models to specific iPSC cell lines.
**Language**: Myokit/Python

## Model Comparison: hiPSC-CM vs TTP06

```
┌──────────────┬─────────┬──────────┬──────────┬──────────┐
│ Current      │ TTP06   │ Paci2013 │ Kernik19 │ Akwaboah │
├──────────────┼─────────┼──────────┼──────────┼──────────┤
│ INa          │ ✓       │ ✓        │ ✓        │ ✓ (GA)   │
│ INaL         │ ✗       │ ✗        │ ✓        │ ✗        │
│ ICaL         │ ✓       │ ✓        │ ✓        │ ✓ (GA)   │
│ ICaT         │ ✗       │ ✗        │ ✓        │ ✗        │
│ IKr          │ ✓       │ ✓        │ ✓        │ ✓ (GA)   │
│ IKs          │ ✓       │ ✓        │ ✓        │ ✗        │
│ IK1          │ ✓       │ ✓        │ ✓        │ ✓ (GA)   │
│ Ito          │ ✓       │ ✓        │ ✓        │ ✗        │
│ If (funny)   │ ✗       │ ✓ ***    │ ✓ ***    │ ✓ (GA)   │
│ INaCa        │ ✓       │ ✓        │ ✓        │ ✗        │
│ INaK         │ ✓       │ ✓        │ ✓        │ ✗        │
│ IpCa         │ ✓       │ ✓        │ ✓        │ ✗        │
│ IpK          │ ✓       │ ✗        │ ✗        │ ✗        │
│ IbNa         │ ✓       │ ✓        │ ✓        │ ✗        │
│ IbCa         │ ✓       │ ✓        │ ✓        │ ✗        │
│ ISERCA       │ ✓       │ ✓        │ ✓        │ ✗        │
├──────────────┼─────────┼──────────┼──────────┼──────────┤
│ States       │ 18      │ 16       │ 15       │ varies   │
│ V_rest       │ -85 mV  │ -74 mV   │ -75 mV   │ ~-70 mV  │
│ Spontaneous  │ No      │ Yes ***  │ Yes ***  │ Yes      │
│ APD90        │ ~280 ms │ ~350 ms  │ ~400 ms  │ varies   │
│ dvdt_max     │ ~300V/s │ ~50 V/s  │ ~30 V/s  │ varies   │
└──────────────┴─────────┴──────────┴──────────┴──────────┘

*** Key hiPSC-CM features MISSING from TTP06:
  1. If (funny current) — drives spontaneous beating
  2. Spontaneous automaticity — hiPSC-CMs beat without stimulus
  3. Depolarized V_rest (-70 to -75 mV vs -85 mV)
  4. Smaller IK1 density
```

## Implementation Priority

### PRIORITY 1: Paci 2013 → PyTorch port (HIGH)

**Why**: Most widely cited, well-validated, has If current, closest to our existing TTP06 code structure (same Hodgkin-Huxley formalism). The Myokit .mmt file is already downloaded and contains the complete model with all equations and parameters.

**Plan**:
1. Translate `paci-2013-ventricular.mmt` to PyTorch following our `ionic/base.py` IonicModel ABC
2. Same structure as TTP06: `parameters.py`, `gating.py`, `currents.py`, `calcium.py`, `model.py`
3. Add the funny current (If) as a new current function
4. Validate single-cell AP against published Paci 2013 figures
5. Place in `Monodomain/Engine_V5.4/cardiac_sim/ionic/paci/`

**Effort**: ~1-2 days. The equations are all in the .mmt file — it's a translation job, not research.

### PRIORITY 2: Kernik 2019 (MEDIUM)

**Why**: More currents (INaL, ICaT), experimentally grounded variability distributions. But C++ code is harder to port and the model is more complex.

**Deferred to**: After Paci is working and validated.

### PRIORITY 3: Akwaboah 2021 GA Fitting Code (REFERENCE ONLY)

**Why**: The GA fitting pipeline for individual currents is directly relevant to our optimizer. Don't port the model itself — use the fitting methodology as a reference for our BayesOpt pipeline.

## What This Changes for the Optimizer

With a native hiPSC-CM model (Paci 2013), the optimizer targets become **much more natural**:

```
    WITH TTP06 (current plan):           WITH PACI 2013 (after port):
    ─────────────────────────            ───────────────────────────
    V_rest = -85 mV (forced)             V_rest = -74 mV (native!)
    dvdt_max = 150 V/s (compromise)      dvdt_max = ~50 V/s (native!)
    APD = 250 ms (needs tuning)          APD = ~350 ms (needs tuning down)
    CV = 25 cm/s (D reduction)           CV = 15 cm/s (D reduction,
                                                 natural GNa level)
    No If current                        If drives spontaneous beating
    No spontaneous beating               Spontaneous beating native

    The Paci model IS the hiPSC-CM ground truth.
    Tuning it = moderate scaling around published values.
    Tuning TTP06 = fighting the model's adult phenotype.
```

## Downloaded Optimization Reference Code

Also downloaded to `Research/code_examples/optimization_pipelines/`:

| Repo | Path | Purpose |
|------|------|---------|
| BoTorch | `optimization_pipelines/botorch_reference/` | BayesOpt framework (PyTorch). Examples in `tutorials/` |
| pymoo | `optimization_pipelines/pymoo_reference/` | NSGA-II and multi-objective. Examples in `pymoo/examples/` |

## Sources

- [Kernik 2019 GitHub](https://github.com/ClancyLabUCD/IPSC-model)
- [Akwaboah 2021 GitHub](https://github.com/Adakwaboah/hiPSC-CM_Computational_Model)
- [CardiacModelling tailored-ipsc-models](https://github.com/CardiacModelling/tailored-ipsc-models)
- [Paci 2013 CellML](https://models.cellml.org/w/sseveri/paci_hyttinen_aaltosetala_severi_2013)
- [Kernik 2019 CellML](https://models.cellml.org/e/805)
- [BoTorch GitHub](https://github.com/meta-pytorch/botorch)
- [pymoo GitHub](https://github.com/anyoptimization/pymoo)
