---
paper: morier_2025_agata_gnn
title: "Learning Cardiac Electrophysiology with Graph Neural Networks for Fast Data-driven Personalised Predictions"
authors: "Morier M, Rodríguez-Padilla J, Gallinari P, Sermesant M"
year: 2025
journal: "FIMH 2025 (LNCS 15672/15673)"
doi: "hal-05114524v3"
pmid: ""
pdf: ../papers/learning_cardiac_electrophysiology_with_graph_neural_networks_for_fast_data_driven_personalised_predictions.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- AGATA is an **autoregressive GNN** for simulating cardiac AP propagation, the closest existing work to our surrogate approach
- Uses **GATv2Conv attention layers** (3 layers) to process graph-structured cardiac meshes
- Trained on simple ellipsoid meshes, **generalizes to unseen geometries** (2D triangular LV endocardium, 3D tetrahedral biventricular mesh) without retraining
- Achieves **up to 12x speedup** vs FEM (Firedrake) on 3D ventricles (GPU inference)
- MAE of 0.016 globally (dimensionless, AP range 0-1), activation time MAE ~0.9ms on ellipsoid

## Method
- **Equations**: Monodomain only with **Mitchell-Schaeffer (MS) ionic model** — a simplified 2-variable phenomenological model (u, h). NOT biophysically detailed.
- **Architecture**: Autoregressive sliding window (T_w = 5 steps = 25ms at dt=0.25ms). Three GATv2Conv layers with node + edge attention. Concatenates window outputs → linear → sigmoid.
- **Data**: Ellipsoid mesh (5,360 nodes, 10,628 triangles). 5,360 simulations with random scar/gray zones. BCL=300ms, 5 stimuli over 1.7s. Split by stimulus location.
- **Training**: Bayesian hyperparameter optimization (Optuna). lr=2.9e-4, batch=2, dropout=0.335, k1=41, k2=36. RMSE loss. Adam optimizer.
- **Inference**: GPU (NVIDIA A40). CPU total ~49s (ellipsoid), ~99s (ventricles) including mesh-to-graph conversion; GPU inference alone is 0.5-0.6s.

## Key Equations / Results
- MS model: ∂_t u = div(D∇u) + hu(u-λ)(u_max - u)/τ_in - u/τ_out + J_stim
- Attention weights: α_ij = softmax(LeakyReLU(a^T [W_n(x_i^s + x_j^s) + W_e e_{i,j}]))
- Speedup table (seconds for 1.7s simulation):

| Mesh | AGATA GPU | FEM | Speedup |
|------|-----------|-----|---------|
| Ellipsoid | 46.5 | 302 | 6.5x |
| LV-E | 41.4 | 293 | 7.1x |
| Ventricles | 91.6 | 1159 | 12.7x |

- MAE by tissue type (dimensionless, AP range 0-1):

| Mesh | Healthy | Gray Zone | Scar | Global |
|------|---------|-----------|------|--------|
| Ellipsoid | 2.9e-5 | 3.1e-5 | 4.6e-5 | 3.0e-5 |
| LV-E | 0.007 | 0.016 | 0.024 | 0.008 |
| Ventricles | 0.005 | 0.008 | 0.015 | 0.006 |

## Connections to Our Models

### Relevant Engine Components
- Their approach replaces the monodomain PDE solver entirely — no operator splitting, no separate ionic/diffusion components
- Our design explicitly separates ionic (Transformer) from diffusion (ResNet), mirroring the simulator's operator splitting
- Their GNN operates on unstructured meshes; our CNN operates on structured grids (Nx × Ny)

### Agreements
- Autoregressive per-timestep prediction is viable for cardiac EP surrogate modeling
- Attention mechanisms capture relevant spatial/temporal features in AP propagation
- Training on simple geometries can generalize (they show ellipsoid → ventricles transfer)

### Disagreements or Gaps
- **Ionic model complexity**: MS is a 2-variable toy model. TTP06 has 18 states, ORd has 40. The information content in detailed ionic dynamics (restitution, rate adaptation, early afterdepolarizations) is vastly higher. Their approach may not scale to biophysically detailed models.
- **No bidomain**: They only predict Vm (monodomain). No phi_e field. The elliptic solve for phi_e is the computational bottleneck we're trying to surpass — they don't address this at all.
- **No operator splitting**: Their monolithic GNN learns both ionic and diffusion simultaneously. Our split architecture enables independent training on cheap data (single-cell ODE for ionic, pure diffusion for spatial).
- **Short lookback window**: 5 steps (25ms) vs our 300-point non-uniform lookback covering ~300ms. Their window may miss rate-dependent effects.
- **Speedup overhead**: Their total pipeline (mesh→graph→inference→graph→mesh) has significant overhead. Pure inference is 0.5s but total is 46-99s. Our structured grid avoids mesh conversion entirely.

### Actionable Insights
- **Geometry generalization** is a validated concept — they train on ellipsoids, infer on ventricles. If we later extend to variable grids, this precedent supports feasibility. Priority: low (deferred).
- **Sliding window vs non-uniform lookback**: Their 5-step window works for MS but may fail for detailed ionic models with longer memory (restitution). Validates our choice of 300-point non-uniform lookback. Priority: informational.
- **Speedup baseline**: 12x over FEM is the benchmark to beat. Our CNN on structured grids should be significantly faster than their GNN on unstructured meshes. Priority: medium (Phase 5 comparison).
- **Attention in graph space**: Their GATv2Conv attention is spatial, ours is temporal. Not directly comparable but confirms attention is effective for EP. Priority: informational.

## Limitations / Caveats
- Mitchell-Schaeffer is not suitable for clinical applications (no restitution, no rate adaptation)
- Larger errors in scar/gray zone regions (under-represented in training data)
- Generalization accuracy drops on meshes with different connectivity distributions (LV-E has different edge length distribution than training ellipsoid)
- dt=0.25ms is much coarser than our dt=0.01ms — they may miss fast upstroke dynamics
- No long-horizon stability analysis (they simulate 1.7s = 5 beats; we need 5000ms+)
