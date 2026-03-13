# Surrogate Pipeline: Two-Component Neural Surrogate for Bidomain Simulation

Autoregressive surrogate model that replaces the bidomain PDE solver with two learned components — an **Ionic Transformer** and a **Cross-Skip Coupled ResNet** — mirroring the operator-splitting structure of the Bidomain Engine V1. Predicts (Vm, phi_e) fields at t+1 from a non-uniform temporal history window, achieving per-step speedup by replacing PCG solves and stencil operations with neural forward passes.

## Core Idea

The bidomain simulator splits each time step into two operations:
1. **Ionic step** (local, per-node): advance gate variables, compute I_ion, update Vm
2. **Diffusion step** (global, spatial): solve parabolic (Vm) + elliptic (phi_e) PDEs

The surrogate mirrors this exactly:
1. **Ionic Transformer** (per-node): voltage history → attention → gates → I_ion prediction
2. **Cross-Skip Coupled ResNet** (spatial 2D): dual conv paths with bidirectional skip connections

```
PER NODE (ionic prediction):

  Vm[t0..tm-1]  →  Attention  →  MLP  →  latent_state(t)     (model-agnostic)
                                               |
                                               +→ [Gate Decoder → gates]  (scaffold only)
                                               |
                                    Vm(tN) + latent_state(t)  →  MLP_ion  →  I_ion(tN+1)
                                                                  (universal)       |
                                    Vm_post_ionic = Vm(tN) + dt*(-I_ion + I_stim)/Cm

GLOBAL (diffusion prediction):

  Vm_post_ionic (Nx,Ny)  →  ResBlock_i ──(+cross)──→ ... ──→  Vm(tN+1)
                                  ↕ cross-skip                    (Nx,Ny)
  phi_e(tN)       (Nx,Ny) →  ResBlock_e ──(+cross)──→ ... ──→  phi_e(tN+1)

  Output fed back autoregressively for t+2
```

## Training Data Source

Bidomain Engine V1 (`Bidomain/Engine_V1/`) — 38/38 tests passing, Kleber boundary speedup validated.

| Simulator Output | Shape | Used By |
|-----------------|-------|---------|
| Vm | (Nx, Ny) per timestep | Both components |
| phi_e | (Nx, Ny) per timestep | Diffusion ResNet |
| ionic_states | (Nx, Ny, 18) for TTP06 | Gate decoder scaffold targets (training only) |
| I_ion | (Nx, Ny) per timestep | MLP_ion training targets |
| dt | 0.01 ms | Surrogate step size (matched) |

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Two components, serial | Ionic → Diffusion | Mirrors operator splitting; enables independent training |
| Voltage-only Transformer input | Vm history → gates (predicted, not input) | Simpler input; network infers ionic state from AP waveform |
| Gates as intermediate prediction | Attention → MLP → gates → MLP → I_ion | Interpretable; trainable against known TTP06 gate trajectories |
| Universal latent space | Abstract ionic state, not compressed gates | Cross-model comparison; model-agnostic representation |
| Universal MLP_ion | Same current calculator for all ionic models | Only Attention block is model-specific |
| Gate decoder as scaffold | Auxiliary training loss, removed for production | Bootstraps latent space; enables interpretability during dev |
| Cross-skip coupled ResNet | Bidirectional skip connections between Vm/phi_e paths | Learns asymmetric coupling (L_i·phi_e, L_i·Vm) at each block |
| Non-uniform temporal lookback | 300 pts, dense near t, sparse distant | Captures fast upstroke + slow plateau context |
| Fixed grid size | Nx × Ny fixed per model | Simplifies conv architecture; variable grids deferred |
| Same dt as simulator | 0.01 ms | Speedup from per-step compute, not temporal compression |
| Sequential training | Ionic → Diffusion → End-to-end | Decompose complexity; monitor each component independently |
| Monodomain first | Single conv path baseline → bidomain upgrade | Validates pipeline before adding phi_e coupling complexity |

## Architecture

```
Surrogate/
|
+-- surrogate/                              # Python package
|   |
|   +-- data/                               # Data generation & preprocessing
|   |   +-- single_cell_generator.py        # TTP06 ODE → (Vm, gates, I_ion) trajectories
|   |   +-- diffusion_generator.py          # Bidomain diffusion-only snapshots
|   |   +-- bidomain_generator.py           # Full bidomain simulation snapshots
|   |   +-- temporal_sampler.py             # Non-uniform 300-point lookback schedule
|   |   +-- dataset.py                      # PyTorch Datasets + DataLoaders
|   |
|   +-- models/                             # Model components
|   |   +-- gate_decoder.py                 # Training scaffold: latent → gates (per ionic model)
|   |   +-- ionic_transformer.py            # Vm history → Attention → MLP → latent_state
|   |   +-- mlp_ion.py                      # Universal: (latent, Vm) → I_ion
|   |   +-- diffusion_resnet.py             # Cross-skip coupled ResNet
|   |   +-- surrogate.py                    # Full composed model + history buffer
|   |
|   +-- training/                           # Staged training harness
|   |   +-- stage_a_autoencoder.py          # A1: gate autoencoder
|   |   +-- stage_a_transformer.py          # A2: ionic Transformer (single-cell)
|   |   +-- stage_b_diffusion.py            # B: diffusion ResNet (pure diffusion)
|   |   +-- stage_c_finetune.py             # C: end-to-end fine-tuning
|   |   +-- losses.py                       # Component-specific + combined losses
|   |   +-- metrics.py                      # Vm/phi_e/AT/CV error metrics
|   |
|   +-- inference/                          # Fast prediction
|       +-- predictor.py                    # Autoregressive rollout wrapper
|
+-- tests/
+-- README.md
+-- improvement.md
+-- IMPLEMENTATION.md
+-- PROGRESS.md
```

## Build Order

```
Step 1: Monodomain baseline                Step 2: Bidomain upgrade
(single conv path, Vm only)                (add phi_e, cross-skips)

  Ionic Transformer → single ResNet          Ionic Transformer → cross-skip coupled ResNet
  - Validates full pipeline end-to-end       - Vm path = Step 1's ResNet (no wasted work)
  - No phi_e, no coupling complexity         - Add phi_e path + bidirectional cross-skips
  - Monodomain Engine V5.4 as data source    - Bidomain Engine V1 as data source
```

## Sequential Training Strategy

```
Stage A: Ionic Transformer                Stage B: Diffusion ResNet
(single-cell TTP06 ODE data)              (pure diffusion, no ionic)

  Scaffolded: gate decoder + I_ion loss      Monodomain: single conv path
  Then remove scaffold, I_ion only          then Bidomain: add cross-skips
  TTP06 + ORd decoders train in parallel
         |                                         |
         +------------------+----------------------+
                            |
                   Stage C: End-to-End
                   (full bidomain data)
                   - Freeze → unfreeze schedule
                   - Autoregressive rollout loss
                   - Monitor error accumulation
```

## Diffusion ResNet Upgrade Path

If cross-skip coupled ResNet is insufficient for phi_e accuracy (due to global nature of elliptic solve):

| Upgrade | What it adds | Cost | When to try |
|---------|-------------|------|-------------|
| Dilated convolutions | Exponential receptive field growth | Minimal | First |
| U-Net bottleneck in joint block | Global context via downsampling | Moderate | If dilated conv insufficient |
| Local Transformer cross-attention | Rich W×W pairwise interactions | ~7x conv cost | If U-Net insufficient |
| FNO spectral layer | Global via FFT (matches simulator's spectral solver) | Moderate | Alternative to local Transformer |

## Future Extensions (Deferred)

- Variable grid sizes
- Boundary condition masks as input channels
- Infarct region masks
- Discretization scheme masks
- Variable dt (temporal compression)
- ORd ionic model + cross-model latent comparison
- ML-directed optimization loop

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.9+ | Runtime |
| PyTorch | 2.0+ | Tensor computation, model training |
| NumPy | 1.24+ | Array utilities |
| Bidomain Engine V1 | — | Training data generation (bidomain) |
| Monodomain Engine V5.4 | — | Training data generation (monodomain baseline) |

## Documentation

| Document | Contents |
|----------|----------|
| `README.md` | This file — overview, data flow, build order, upgrade path |
| `improvement.md` | Architecture spec — component details, cross-skip design, interfaces |
| `IMPLEMENTATION.md` | Phase-by-phase implementation plan with validation |
| `PROGRESS.md` | Living progress tracker |
