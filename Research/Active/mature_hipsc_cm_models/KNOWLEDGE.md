# Mature hiPSC-CM Ionic Models — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.
>
> **Scope**: This question covers the maturation of spontaneously-beating hiPSC-CM models
> into quiescent models suitable for tissue simulation. Immature/spontaneous hiPSC-CM
> models are a separate research question.

## Current Understanding

All published hiPSC-CM ionic models beat spontaneously due to high If (funny current) and low IK1 (inward rectifier). This makes them unsuitable for tissue simulation where quiescence is required for stimulus-driven propagation. The maturation pathway addresses this by injecting adult-like IK1 and suppressing the developmental If current.

### The maturation pathway

| Model | Base | States | V_rest | Spontaneous? | Key Change |
|-------|------|--------|--------|-------------|------------|
| **Paci 2013** | Original publication | 17 | ~-75 mV | Yes | Base hiPSC-CM model |
| **PHAS13** | Paci + hiPSC modifications | 17 | ~-75 mV | Yes | Renamed, backward compat alias `PaciModel` |
| **MHAS13** | PHAS13 + maturation | 17 | -83.7 mV | **No** | TTP06 IK1 at critical GK1, g_f=0 |

The critical insight from Verkerk 2019: IK1 injection at a specific conductance value makes hiPSC-CMs quiescent without distorting the action potential morphology. The If (funny current) is suppressed (g_f=0) because it is a developmental artifact responsible for automaticity.

### Validated results

| Metric | MHAS13 (V5.4) | MHAS13 (Bidomain) | Target |
|--------|---------------|-------------------|--------|
| APD | 347 ms | 349 ms | 350 ms |
| V_rest | -83.7 mV | -83.7 mV | ~-85 mV (adult) |
| CV | — | 15.8 cm/s | 15-25 cm/s (hiPSC tissue) |
| Spontaneous beating | None | None | Quiescent |

### Implementation details

- All three models live in `Monodomain/Engine_V5.4/cardiac_sim/ionic/` (paci/, phas13/, mhas13/)
- MHAS13 uses the TTP06 IK1 formulation with the Verkerk 2019 critical GK1 value
- Cell capacitance Cm = 0.0987 nF (preserved from original Paci model)
- 17 states: 13 gating variables + 4 ionic concentrations (same as Paci)
- Optimizer V1 achieved 10x speedup via batching, subcycling, and analytical CV

### Why not existing mature models?

| Model | States | Issue |
|-------|--------|-------|
| TTP06 | 18 | Adult ventricular — not hiPSC-CM phenotype |
| ORd | 41 | Adult ventricular — different current balance |
| Kernik 2019 | 15 | hiPSC-CM but spontaneously beating |
| Koivumäki 2018 | 29 | hiPSC-CM, atrial-like, spontaneous |
| Paci 2018 (updated) | 17 | Still spontaneous, improved Ca²⁺ handling |

All hiPSC-CM models are spontaneous. Our maturation approach (IK1 injection + If suppression) is generalizable to any of them.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| IK1 source | TTP06 IK1 formulation | Same framework, validated, correct rectification |
| GK1 value | Critical value from Verkerk 2019 | Minimum needed for quiescence |
| If handling | Set g_f=0 | Eliminates spontaneous depolarization |
| Cm | 0.0987 nF (Paci cell capacitance) | Preserved from original model |
| Naming | MHAS13 = Matured PHAS13 | Clear lineage |

## Open Questions

- Does MHAS13 produce realistic restitution curves (APD vs DI)?
- Would an ORd-based hiPSC variant (40 states) capture late sodium and CaMKII effects better?
- How does MHAS13 tissue CV compare to experimental hiPSC monolayer CV measurements?
- Can the maturation pathway be validated against fetal-to-adult cardiomyocyte data?
- Is there a more principled way to set GK1 than the Verkerk 2019 critical value?

## Connections
- **Engines**: V5.4 (ionic model host), Bidomain V1 (tissue validation)
- **Related research**: ionic_model_optimization (Optimizer V1 tunes MHAS13 parameters), fetal_heart_development (maturation mirrors development), immature_hipsc_cm_models (planned — spontaneously-beating models)
- **Pipelines**: Optimizer V1 (MHAS13 as tuning target)
