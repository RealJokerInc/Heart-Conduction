# Mature hiPSC-CM Ionic Models

## Question
How do we create non-spontaneously-beating (quiescent) hiPSC-CM ionic models suitable for tissue simulation, by maturing immature hiPSC-CM models via IK1 injection and If suppression?

## Status: Active

## Why It Matters
All published hiPSC-CM ionic models beat spontaneously due to high If and low IK1. Tissue simulation requires quiescent models (fires only when paced). The maturation pathway (Paci 2013 → PHAS13 → MHAS13) solves this by injecting TTP06 IK1 at the Verkerk 2019 critical conductance and suppressing If (g_f=0). This question focuses specifically on the maturation step — building quiescent models from spontaneously-beating bases. Immature/spontaneous hiPSC-CM models are a separate research question.

## Engines
- **Monodomain V5.4**: hosts PHAS13 and MHAS13 in `cardiac_sim/ionic/`
- **Bidomain V1**: MHAS13 validated through bidomain pipeline (APD=349ms, CV=15.8cm/s)
- **Optimizer V1**: MHAS13 as tuning target

## Completion Criteria
- [x] PHAS13 model implemented (Paci 2013 with hiPSC modifications)
- [x] MHAS13 model implemented (matured, quiescent, TTP06 IK1, g_f=0)
- [x] Single-cell validation: APD=349ms, V_rest=-83.7mV
- [x] Bidomain pipeline validation: APD=349ms, CV=15.8cm/s
- [x] Tissue propagation validation (2D, multiple pacing rates)
- [ ] ORd-based mature hiPSC variant (alternative maturation pathway)
- [ ] Restitution curve characterization

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| IK1 injection | Complete | Critical GK1 from Verkerk 2019 makes Paci quiescent |
| If suppression | Complete | g_f=0 eliminates spontaneous beating without affecting AP |

## Experiments

| Experiment | Engine | Result | Location |
|-----------|--------|--------|----------|
| MHAS13 single-cell | V5.4 | APD=347ms (target 350) | `Engines/monodomain_v5.4/experiments/` |
| MHAS13 bidomain | Bidomain V1 | APD=349ms, CV=15.8cm/s | `Engines/bidomain_v1/experiments/` |

## Engine References

Files to read when resuming work on this question:

| File | What it tells you |
|------|-------------------|
| `Monodomain/Engine_V5.4/cardiac_sim/ionic/mhas13/model.py` | MHAS13 model implementation |
| `Monodomain/Engine_V5.4/cardiac_sim/ionic/mhas13/parameters.py` | MHAS13 parameters (GK1, g_f, etc.) |
| `Monodomain/Engine_V5.4/cardiac_sim/ionic/phas13/model.py` | PHAS13 base model (pre-maturation) |
| `Monodomain/Engine_V5.4/cardiac_sim/ionic/base.py` | IonicModel ABC interface |
| `Monodomain/Engine_V5.4/tests/test_paci.py` | PHAS13/MHAS13 validation tests |
| `Optimizer/V1/HIPSC_IONIC_MODELS.md` | hiPSC-CM model survey and selection |
| `Optimizer/V1/PACI_IMPLEMENTATION.md` | Paci → PHAS13 implementation notes |
| `Research/Active/ionic_model_optimization/KNOWLEDGE.md` | Optimization context for MHAS13 tuning |

## Literature
| Paper | Summary | Key Insight |
|-------|---------|-------------|
| hipsc_cm_maturation_models | Multi-paper survey on hiPSC-CM maturation and quiescence | IK1 injection + If suppression pathway; model landscape |

## Future Work
{No deferred items yet.}

## Connected Research
- **[Geometry-induced pacemaking](../geometry_induced_pacemaking/)** — Uses PHAS13 (immature, spontaneous) as the base model; MHAS13 serves as negative control
