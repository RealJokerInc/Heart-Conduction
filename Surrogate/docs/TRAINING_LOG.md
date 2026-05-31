# Training Log — Ionic Surrogate v3

> Per-epoch metrics for all training runs. Updated after each phase.

---

## Run Configuration

| Property | Value |
|---|---|
| Model | IonicSurrogateV3, 1,534 inference + 243 scaffold params |
| GPU | NVIDIA RTX PRO 4500 Blackwell, 33.7 GB VRAM |
| Data | TTP06 T1 (8M train, 7.6M val timesteps), cached on SSD |
| Precision | float64 (model + data) |
| Optimizer | AdamW, cosine LR decay per phase, grad clip max_norm=1.0 |
| Loss | ionic_state MSE + concentration MSE (combined from B1 onward) |
| Rollout init | zeros (ionic latent) + resting concentrations [Na_i=10, K_i=138, Ca_i=0.0001, Ca_ss=0.0002] |
| No encoder | Model discovers latent through attention + MLP + scaffold decoder |

---

## Phase Summary

| Phase | Rollout | Batch | LR | Epochs | Best Val MSE | Wall Time | Checkpoint |
|---|---|---|---|---|---|---|---|
| B1 (ionic only) | 1 | 4096 | 5e-4 | 5 | 0.086 | 10 min | `runs/b1_t1only/` |
| B1 (ionic + conc) | 1 | 4096 | 5e-4 | 30 | **0.56** | 57 min | `runs/b1_combined2/` |
| B2 | 10 | 4096 | 5e-4 | 30 | **1.68** | 47 min | `runs/b2_test/` |

### Notes
- B1 ionic-only: first test without encoder, proved the model can discover latent from zeros + Vm
- B1 combined: added concentration loss, K_i~138 dominates MSE scale
- B2: unstable early (val spiked to 41K at epoch 0, 27K at epoch 4) due to autoregressive error compounding. Stabilized by epoch 16, clean convergence from epoch 20 onward.

### Deprecated runs (old encoder-based approach, no longer used)
- `runs/a1_real/`: Phase A1 with encoder, val_recon_mse=7.9e-5 (3 epochs)
- `runs/a_phases/`: A1→A2→A3 with encoder, A2 val_conc_mse=0.029, A3 val_cond_mse=1.78e-4

---

## Per-Epoch Detail

### B1 — Ionic Only (rollout=1, batch=4096, T1 only)

| Epoch | Train Loss | Val MSE | LR |
|---|---|---|---|
| 0 | 1.339 | 0.088 | 4.52e-4 |
| 1 | 0.050 | 0.087 | 3.27e-4 |
| 2 | 0.050 | 0.087 | 1.73e-4 |
| 3 | 0.049 | 0.086 | 4.77e-5 |
| 4 | 0.049 | **0.086** | 0 |

### B1 — Combined Ionic + Concentration (rollout=1, batch=4096, T1 only)

| Epoch | Train Loss | Val MSE | LR |
|---|---|---|---|
| 0 | 593.8 | 403.2 | 4.99e-4 |
| 1 | 337.7 | 197.3 | 4.95e-4 |
| 2 | 273.6 | 163.6 | 4.88e-4 |
| 3 | 257.5 | 192.3 | 4.78e-4 |
| 4 | 206.3 | 119.3 | 4.67e-4 |
| 5 | 258.1 | 162.3 | 4.52e-4 |
| 6 | 199.2 | 89.1 | 4.36e-4 |
| 7 | 119.8 | 50.9 | 4.17e-4 |
| 8 | 83.4 | 41.4 | 3.97e-4 |
| 9 | 60.1 | 20.4 | 3.75e-4 |
| 10 | 31.9 | 14.3 | 3.52e-4 |
| 11 | 21.4 | 8.6 | 3.27e-4 |
| 12 | 22.6 | 6.1 | 3.02e-4 |
| 13 | 10.0 | 4.6 | 2.76e-4 |
| 14 | 8.3 | 6.0 | 2.50e-4 |
| 15 | 5.5 | 4.5 | 2.24e-4 |
| 16 | 3.8 | 2.0 | 1.98e-4 |
| 17 | 2.4 | 1.6 | 1.73e-4 |
| 18 | 1.8 | 1.4 | 1.48e-4 |
| 19 | 1.4 | 1.2 | 1.25e-4 |
| 20 | 1.1 | 0.89 | 1.03e-4 |
| 21 | 0.90 | 0.76 | 8.27e-5 |
| 22 | 0.80 | 0.69 | 6.42e-5 |
| 23 | 0.72 | 0.66 | 4.77e-5 |
| 24 | 0.67 | 0.61 | 3.35e-5 |
| 25 | 0.63 | 0.60 | 2.16e-5 |
| 26 | 0.61 | 0.58 | 1.22e-5 |
| 27 | 0.60 | 0.57 | 5.46e-6 |
| 28 | 0.59 | **0.56** | 1.37e-6 |
| 29 | 0.58 | 0.56 | 0 |

### B2 — Rollout=10 (batch=4096, T1 only, from B1 checkpoint)

| Epoch | Train Loss | Val MSE | LR | Notes |
|---|---|---|---|---|
| 0 | 939.6 | 41623.4 | 4.99e-4 | Initial shock — model sees own errors for first time |
| 1 | 2693.1 | 34.5 | 4.95e-4 | |
| 2 | 785.8 | 13.0 | 4.88e-4 | |
| 3 | 364.5 | 11.7 | 4.78e-4 | |
| 4 | 4804.0 | 27215.8 | 4.67e-4 | Spike — one bad rollout destabilized weights |
| 5 | 6182.6 | 6.4 | 4.52e-4 | Recovery |
| 6 | 1434.2 | 9.9 | 4.36e-4 | |
| 7 | 1516.3 | 4.8 | 4.17e-4 | |
| 8 | 649.6 | 4.9 | 3.97e-4 | |
| 9 | 775.0 | 4.5 | 3.75e-4 | |
| 10 | 143.7 | 4.9 | 3.52e-4 | |
| 11 | 291.1 | 3.1 | 3.27e-4 | |
| 12 | 434.6 | 3.2 | 3.02e-4 | |
| 13 | 18.5 | 27.7 | 2.76e-4 | Late spike |
| 14 | 4.7 | 3.9 | 2.50e-4 | Stabilizing |
| 15 | 6.4 | 3.4 | 2.24e-4 | |
| 16 | 3.1 | 2.5 | 1.98e-4 | |
| 17 | 2.9 | 3.6 | 1.73e-4 | |
| 18 | 3.0 | 2.2 | 1.48e-4 | |
| 19 | 2.5 | 2.0 | 1.25e-4 | |
| 20 | 2.2 | 1.9 | 1.03e-4 | |
| 21 | 2.1 | 1.9 | 8.27e-5 | |
| 22 | 2.0 | 1.9 | 6.42e-5 | |
| 23 | 1.9 | 1.8 | 4.77e-5 | |
| 24 | 1.9 | 1.7 | 3.35e-5 | |
| 25 | 1.8 | 1.8 | 2.16e-5 | |
| 26 | 1.8 | 1.7 | 1.22e-5 | |
| 27 | 1.8 | 1.7 | 5.46e-6 | |
| 28 | 1.8 | 1.7 | 1.37e-6 | |
| 29 | 1.8 | **1.68** | 0 | |
