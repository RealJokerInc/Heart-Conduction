# Surrogate Pipeline: Neural Surrogate for Bidomain Simulation

## Question
Can a two-component neural surrogate (Ionic Transformer + Cross-Skip Coupled ResNet) replace the bidomain PDE solver with sufficient accuracy and meaningful speedup?

## Status: Active

## Why It Matters
The bidomain simulation is computationally expensive — the elliptic solve for phi_e requires iterative convergence (PCG/GMG) at every time step, dominating wall-clock time. A trained neural surrogate that mirrors the operator-splitting structure (ionic step → diffusion step) could replace the PDE solver with neural forward passes, enabling real-time or near-real-time cardiac simulation for optimization loops, parameter sweeps, and clinical applications.

## Engines
- **Bidomain V1** — primary training data source (full bidomain: Vm + phi_e fields)
- **Monodomain V5.4** — monodomain baseline data source (Vm-only, for Stage B1 validation)

## Completion Criteria
- [ ] Single-cell data generation pipeline (TTP06 ODE trajectories)
- [ ] Gate decoder training scaffold validated
- [ ] Ionic Transformer reproduces single-cell APs (APD error < 5ms, stable 5-beat rollout)
- [ ] Monodomain single-path ResNet matches V5.4 diffusion within tolerance
- [ ] Bidomain cross-skip ResNet matches Bidomain V1 diffusion (Vm + phi_e)
- [ ] End-to-end surrogate: CV within 5% of simulator, phi_e qualitatively correct
- [ ] Inference speedup measured and documented (wall time surrogate vs simulator)
- [ ] Kleber boundary speedup reproduced by surrogate (CV_ratio within 10% of simulator)

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| Ionic component (Transformer) | In progress | NODE pivot (2026-04-06) validated. IonicRateMLP + conc KAN, multi-BCL val=0.008. |
| Diffusion component (Cross-Skip ResNet) | Not started | — |

## Parallel / Related Questions
- [cardiac_ml_harness](../cardiac_ml_harness/) — project-wide training harness (Hydra + MLflow + Optuna + SHAP). Originated here in Session 26; broken out on 2026-04-19 because the harness scope is project-wide and blocks diffusion ResNet + Optimizer V1 work, not just the surrogate line.

## Key Findings So Far
Documentation phase complete (2026-03-12). Architecture designed:
- **Ionic Transformer**: Vm-only input → attention → latent state → universal MLP_ion → I_ion. Gate decoder as training scaffold (removed for production).
- **Cross-Skip Coupled ResNet**: dual conv paths (Vm, phi_e) with bidirectional 1×1 skip connections. Monodomain single-path baseline first.
- **Training strategy**: Stage A (ionic, single-cell) → B1 (mono diffusion) → B2 (bidomain cross-skip) → C (end-to-end fine-tuning)
- **Upgrade path** if phi_e accuracy insufficient: dilated conv → U-Net → local Transformer → FNO

## Engine References

| Resource | Path | Purpose |
|----------|------|---------|
| Surrogate README | `Surrogate/README.md` | Architecture overview, data flow, build order |
| Surrogate improvement.md | `Surrogate/improvement.md` | Full architecture spec (component details, interfaces) |
| Surrogate IMPLEMENTATION.md | `Surrogate/IMPLEMENTATION.md` | 7-phase implementation plan with validation |
| Surrogate PROGRESS.md | `Surrogate/PROGRESS.md` | Implementation progress tracker |
| Bidomain V1 source | `Bidomain/Engine_V1/cardiac_sim/` | Training data generator (bidomain solver) |
| Monodomain V5.4 source | `Monodomain/Engine_V5.4/cardiac_sim/` | Monodomain baseline data generator |
| Bidomain PROGRESS.md | `Bidomain/Engine_V1/PROGRESS.md` | Engine status (all phases done, 38+ tests) |

## Literature
| Paper | Summary | Key Insight |
|-------|---------|-------------|
| [morier_2025_agata_gnn](literature/morier_2025_agata_gnn.md) | [PDF](papers/learning_cardiac_electrophysiology_with_graph_neural_networks_for_fast_data_driven_personalised_predictions.pdf) | Autoregressive GNN, 12x speedup, but monodomain + MS ionic only. Closest competitor. |
| [centofanti_2025_fno_kol_cardiac](literature/centofanti_2025_fno_kol_cardiac.md) | — | FNO/KOL maps stimulus→AT/RT directly. Single-shot, not timestep sim. Validates FNO for cardiac EP. |
| [lydon_2025_pino_cardiac](literature/lydon_2025_pino_cardiac.md) | — | PINO with FNO backbone. Autoregressive rollouts, 10x resolution upscaling. PINN-adjacent. |
| [salvador_2024_lnode_cardiac](literature/salvador_2024_lnode_cardiac.md) | — | Latent Neural ODE for electromechanics. 300x speedup but 0D outputs (PV loops), not spatial fields. |
| [salvador_2025_lfldnet](literature/salvador_2025_lfldnet.md) | — | LFLDNet: CfC liquid NN replaces ODE solver for latent dynamics. 30x speedup on 3D monodomain (TTP06). Δt=10ms vs 0.1ms FEM. CfC gating directly addresses our A4 discrete-step error accumulation. |

## Future Work
- Variable grid sizes (currently fixed Nx × Ny)
- Boundary condition / infarct region masks as input channels
- Variable dt (temporal compression for additional speedup)
- ORd ionic model + cross-model latent comparison
- ML-directed optimization loop (surrogate guides parameter space exploration)
