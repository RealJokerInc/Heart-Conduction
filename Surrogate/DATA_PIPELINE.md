# Data Pipeline — Ionic Surrogate v3

> How data flows from raw HDF5 through preprocessing to the training loop.

---

## Raw Data

| Property | Value |
|----------|-------|
| Location | `/media/HDD/surrogate_data/raw/` |
| Format | One HDF5 file per tier: `tier01.h5` through `tier12.h5` |
| Precision | float64 (simulator native) |
| Columns | 47 per timestep |
| Total size | ~608 GB across T1-T12 |

**47-column format** (matches `TraceData` in `single_cell_generator.py`):

| Columns | Content | Count |
|---------|---------|-------|
| 0 | Vm (mV) | 1 |
| 1 | I_stim (sign-flipped: positive = depolarizing) | 1 |
| 2 | dt (ms) | 1 |
| 3-20 | 18 ionic states [Ki, Nai, Cai, CaSR, CaSS, m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs, RR] | 18 |
| 21 | I_ion (pure ionic current, no stimulus) | 1 |
| 22 | clamp_mask (0.0 = free-running, 1.0 = clamped) | 1 |
| 23-34 | gate_inf (12 steady-state values, gate_indices order) | 12 |
| 35-46 | gate_tau (12 time constants in ms, gate_indices order) | 12 |

**HDF5 structure**: Each tier file contains named groups (one per protocol). Each group has a `data` dataset (T, 47) and metadata attributes (protocol_name, tier, cell_type, duration_ms, etc.).

---

## Preprocessing (V3Preprocessor)

`Surrogate/surrogate/data/preprocessor.py` — converts raw 47-col segments to named tensors.

**Input**: `(T, 47)` raw tensor.

**Output dict**:

| Key | Shape | Source |
|-----|-------|--------|
| `Vm` | (T,) | col 0 |
| `dt` | (T,) | col 2 |
| `I_stim` | (T,) | col 1 |
| `I_ion` | (T,) | col 21 |
| `clamp_mask` | (T,) | col 22 |
| `concentrations` | (T, 4) | cols [Nai, Ki, Cai, CaSS] — reordered to [Na_i, K_i, Ca_i, Ca_ss] |
| `gates` | (T, 12) | cols 5-16 (m through Xs, excludes RR) |
| `ionic_states` | (T, 14) | 13 gates (m-RR) + CaSR — scaffold decoder targets |
| `conductance_products` | (T, 5) | computed: G_Na(m^3hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1Xr2), G_Ks(Xs^2) |
| `E` | (T, 4) | computed: [E_Na, E_K, E_Ca, E_Ks] via Nernst from concentrations |
| `gate_inf` | (T, 12) | cols 23-34 |
| `gate_tau` | (T, 12) | cols 35-46 |

**Concentration reorder**: TTP06 stores [Ki, Nai, Cai, CaSR, CaSS] at state indices 0-4. V3 reorders to [Na_i, K_i, Ca_i, Ca_ss] via indices [1, 0, 2, 4]. CaSR dropped (tracked implicitly by ionic latent).

**Nernst computation**: `E_Na = RTONF * ln(140/Na_i)`, `E_K = RTONF * ln(5.4/K_i)`, `E_Ca = 0.5 * RTONF * ln(2.0/Ca_i)`, `E_Ks = RTONF * ln((5.4 + 0.03*140)/(K_i + 0.03*Na_i))`. Constants match `nernst.py`.

**Conductance products**: Precomputed from raw gate states. These are the 5 scalar open-probability products that drive the 5 major currents.

---

## Shard Storage

| Property | Value |
|----------|-------|
| Location | `/media/norepinephrine/Elements-ext4/surrogate_data/train/` (train), `.../val/` (val) |
| Format | `.pt` files (PyTorch tensors) |
| Precision | float32 |
| Shard size | ~200 MB each |
| Segment shape | `(N_segments, segment_length, 47)` per shard |

`ShardProcessor` in `storage.py` converts HDF5 to shards:
1. Extracts overlapping segments (50% overlap, stride = segment_length / 2)
2. Converts float64 → float32
3. Shuffles segments across protocols
4. Writes to numbered shard files

---

## DataLoader Design

Each phase needs a different DataLoader because input/target shapes differ.

### Phase A1: Random State Snapshots

```
Sample: random timestep from T1 trace
Input:  14 ionic state values (12 gates + RR + CaSR)
Target: same 14 values (autoencoder reconstruction)
Batch:  (B, 14)
```

No temporal structure. Sample uniformly across all T1 timesteps. Very fast — no sequential loading needed.

### Phase A2: Sequential Concentration Pairs

```
Sample: (state_t, state_{t+1}) pairs from T1 traces
Input:  carried_state_t (20,) = encoder(ionic_states_t) + conc_t
        Vm_t (scalar), dt_t (scalar)
Target: true_conc_{t+1} (4,) = [Na_i, K_i, Ca_i, Ca_ss] at t+1
Batch:  (B, 20) + (B,) + (B,) input, (B, 4) target
```

Requires consecutive timestep pairs. Encoder (from A1) maps true_14_states to 16-dim ionic latent.

### Phase A3: Carried State → Conductance Products

```
Sample: single timestep from T1 traces
Input:  carried_state (20,) = encoder(ionic_states) + conc
Target: true_5_products (5,) = conductance products at same timestep
Batch:  (B, 20) input, (B, 5) target
```

Single-step, no temporal structure. Encoder provides ionic latent; compression maps carried_state to conductance_latent; decoder maps to 5 products.

### Phase B-E: Sequential Segments with Rollout

```
Sample: contiguous segment of length rollout_length from a trace
Raw:    (rollout_length, 47) → V3Preprocessor → dict of named tensors
Batch:  (B, rollout_length, ...) — sequential within each sample
```

**Per-step data available during rollout**:

| Field | Shape per step | Used for |
|-------|---------------|----------|
| Vm | scalar | Model input |
| dt | scalar | Model input |
| I_stim | scalar | Vm update in free-running mode |
| I_ion | scalar | Phase D-E loss target |
| clamp_mask | scalar | Determines Vm update mode |
| ionic_states | (14,) | Phase B loss target + teacher forcing |
| concentrations | (4,) | Phase C loss target + teacher forcing |
| conductance_products | (5,) | Monitoring only (Phase B) |

**Rollout execution**: For each step t in the segment, the model receives (carried_state, Vm_t, dt_t, cond_lat_prev, conc_prev) and produces predictions. The carried_state is either the model's own output (autoregressive) or replaced via teacher forcing (scheduled sampling).

---

## Teacher Forcing Mechanism

**Encoder**: Temporary component trained in Phase A1. Maps `true_14_states → 16-dim ionic latent`. Used only during Phase B for scheduled sampling. Discarded after Phase B.

**Scheduled sampling at each rollout step**:
```python
p = scheduled_sampling_probability  # ramps from 0.1 (B1) to 1.0 (B5)
if random() < p:
    # Use model's own prediction (autoregressive)
    carried_state = model_output['carried_state']
else:
    # Teacher forcing: replace with ground truth
    ionic_latent = encoder(true_14_states_t)      # (16,)
    carried_state = cat([ionic_latent, true_conc_t])  # (20,)
```

**After Phase B**: Encoder is no longer needed. All subsequent phases are fully autoregressive (p=1.0).

**Conductance latent and concentration teacher forcing**: During teacher forcing steps, cond_lat_prev and conc_prev are also replaced with values derived from ground truth (compression of encoder output + true concentrations).

---

## Segment Windowing

Raw traces are very long (e.g., BCL=1000 x 20 beats = 2M steps at dt=0.01ms). Must be sliced into windows matching the rollout length.

| Rollout | Window size | Stride | Overlap |
|---------|------------|--------|---------|
| 1 (A phases) | 2 (pair) | 1 | 50% |
| 10 | 10 | 10 | none |
| 100 | 100 | 50 | 50% |
| 1000 | 1000 | 500 | 50% |
| 10000 | 10000 | 5000 | 50% |
| 100000 | 100000 | 50000 | 50% |

**Window initialization**: Each window starts with the model at its default initial state (ionic latent = zeros, concentrations = resting values). The first few steps of each window are effectively a "warm-up" where the model self-corrects. This is by design — the model must handle cold starts.

**Alternative** (for long rollouts B4+): Start from a mid-trace state using encoder to initialize the latent. This avoids wasting the first ~100 steps on warm-up. Only used if cold-start warm-up hurts convergence.

---

## Data Loading Strategy

### Shard-Based Loading (Phase B-E)

```
1. Load shard .pt file → (N, segment_length, 47) float32 tensor
2. Slice into windows of rollout_length
3. V3Preprocessor.process_segment() on each window
4. Collate into batch: (B, rollout_length, ...)
5. Transfer to GPU
```

### Random Sampling (Phase A)

Phase A needs random timesteps, not sequential segments. Two options:
1. **Pre-extract**: Build a flat tensor of all timestep snapshots from T1 shards. Shuffle once. Load sequentially.
2. **On-the-fly**: Load shard, sample random indices within segments. More I/O but no preprocessing step.

Option 1 preferred — T1 is only ~3.5 GB, fits in RAM.

### Validation Split

Split by **protocol name**, not by timestep:
- Train: BCL = {300, 500, 700, 1000, 1500} (T1), selected S1-S2 DIs (T2), selected dynamics (T3)
- Val: BCL = {400, 600, 800, 2000} (T1), remaining S1-S2 DIs (T2), remaining dynamics (T3)

This ensures the model is validated on unseen pacing rates, not just unseen timesteps from the same trace.
