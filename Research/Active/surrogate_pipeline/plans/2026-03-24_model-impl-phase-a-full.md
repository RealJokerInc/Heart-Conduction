# PLAN: Ionic Surrogate Model Implementation + Phase A Training

Created: 2026-03-24
Engine(s): None (standalone ML model)
Research question: [surrogate_pipeline](README.md)
Source: [IDEALOG.md](IDEALOG.md) — Session 12 (Architecture v2 settled)

## Objective
Implement the ionic surrogate model (v2 architecture: n×1 cross-attention + two-round split GELU + KAN Chebyshev readout) as a `torch.nn.Module`, build the data loading pipeline for Phase A training, and train the gate autoencoder to bootstrap the latent space. This plan covers model code, unit tests, data loading, and Phase A only. Phases B-D training is a separate blueprint.

## Success Criteria
- [ ] IonicSurrogate nn.Module matches architecture spec (642 inference params, 886 FLOPs)
- [ ] Gate decoder training scaffold validated
- [ ] Phase A autoencoder trains successfully on T1 data
- [ ] Latent space reconstruction error < 0.001 MSE on gate snapshots
- [ ] All new tests pass, all existing 32 data generation tests pass (no regressions)

## Architecture Changes
- NEW: `Surrogate/surrogate/model/__init__.py` — model subpackage
- NEW: `Surrogate/surrogate/model/ionic_surrogate.py` — full 3-stage model + scaffold
- NEW: `Surrogate/surrogate/model/chebyshev.py` — Chebyshev polynomial readout layer
- NEW: `Surrogate/surrogate/model/autoencoder.py` — Phase A gate autoencoder
- NEW: `Surrogate/surrogate/training/__init__.py` — training subpackage
- NEW: `Surrogate/surrogate/training/data_loader.py` — HDF5 → gate snapshot loaders
- NEW: `Surrogate/surrogate/training/train_phase_a.py` — Phase A training loop
- NEW: `Surrogate/tests/test_model.py` — unit tests for model
- NEW: `Surrogate/tests/test_phase_a.py` — integration tests for Phase A

## Known Failures (from IDEALOG)
- Temporal Transformer (300-pt history) — 1M× RL, buffer management nightmare
- Vm history buffer (any size) — stimulus artifacts, non-uniform schedule maze
- Learned Rush-Larsen — too constrained, forces HH exponential, independent dims
- Neural ODE — too unconstrained, multi-timescale learning hard
- GRU cell — gating mechanism adds cost without benefit over residual formulation
- 17×17 self-attention — 47× RL, overkill for cross-channel coupling
- Deep MLP cross-channel — overkill, single linear layer suffices
- I_stim as model input — biophysically wrong, doesn't match operator splitting

---

## Phase 1: Model Implementation

**Goal**: Build `IonicSurrogate` nn.Module with all 3 stages + scaffold decoder. Verify parameter counts, shapes, and contractivity.
**Tier**: medium
**Estimated scope**: 3 files (~300 lines), 1 test file (~200 lines)

### Phase Context
- Architecture v2 spec in KNOWLEDGE.md § "Ionic Surrogate — Design"
- All tensors float32 for ML (data generation was float64, converted during loading)
- Model operates on single samples (not batched) during inference, but training uses batches
- Forward pass: `(latent_prev, Vm, dt) → (latent_new, I_ion, gates_pred)`
- latent_prev shape: `(batch, 16)` or `(16,)` for single sample
- Vm, dt: `(batch,)` or scalar
- 16 latent dims, 8 attention dim, 8/8 split for GELU
- W_q applied per-dim: `latent_prev[k] * W_q[k, :]` (scalar × row), NOT a full linear layer
- Spectral norm via `torch.nn.utils.spectral_norm()` on both W_cc1 and W_cc2
- No bias on Q/K/V/W_out. Bias on W_cc1, W_cc2, b_vm, b, W_dec, b_dec
- KAN Chebyshev normalization bounds (z_min, z_max) are buffers (registered, not parameters)
- Xavier uniform init for all weight matrices except W_cc (Xavier + spectral_norm wrapper)
- Readout C, b_vm, b initialized to zeros
- Scaffold decoder weights initialized from Phase A autoencoder (but random init for testing)

### Step 1.1: ChebyshevReadout layer
**Model**: opus

#### Read First
- KNOWLEDGE.md:45-52 — KAN Chebyshev K=3 spec
- WHITEBOARD.md — Stage 3 diagram

#### Why
The Chebyshev readout is a self-contained component used by IonicSurrogate. Building it first enables isolated testing. The Chebyshev polynomial computation (recurrence relation) and normalization are easy to get wrong — test independently.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/__init__.py`, `Surrogate/surrogate/model/chebyshev.py`
**Interfaces / Signatures:**
```python
class ChebyshevReadout(nn.Module):
    """Per-dimension Chebyshev polynomial readout.

    Each of n_dims dimensions gets its own degree-K polynomial:
    φ_k(z_k) = Σ_{j=0}^{K} C[k,j] * T_j(z̃_k)

    Output: sum of per-dim polynomials + b_vm * Vm + b
    """
    def __init__(self, n_dims: int = 16, degree: int = 3):
        # C: (n_dims, degree+1) — trainable Chebyshev coefficients
        # b_vm: (1,) — Vm bypass weight
        # b: (1,) — bias
        # z_min, z_max: (n_dims,) — registered buffers, set via set_bounds()

    def set_bounds(self, z_min: Tensor, z_max: Tensor):
        """Set normalization bounds from Phase A latent statistics."""

    def forward(self, z: Tensor, Vm: Tensor) -> Tensor:
        """z: (batch, n_dims), Vm: (batch,) → I_ion: (batch,)"""
```

#### Pseudocode
```
def forward(z, Vm):
    # Normalize to [-1, 1]
    z_norm = 2 * (z - z_min) / (z_max - z_min + eps) - 1
    z_norm = z_norm.clamp(-1, 1)  # safety clamp for rollout drift

    # Chebyshev basis via recurrence: T0=1, T1=z, Tn=2z*T_{n-1} - T_{n-2}
    T = [ones_like(z_norm), z_norm]
    for j in range(2, degree+1):
        T.append(2 * z_norm * T[-1] - T[-2])
    T = stack(T, dim=-1)  # (batch, n_dims, degree+1)

    # Per-dim polynomial
    phi = (self.C * T).sum(dim=-1)  # (batch, n_dims)

    # Sum + Vm bypass
    I_ion = phi.sum(dim=-1) + self.b_vm * Vm + self.b
    return I_ion
```

#### Test Spec
- `test_model.py::test_chebyshev_shape` — Input (32, 16), Vm (32,). Output: (32,)
- `test_model.py::test_chebyshev_params` — C has 64 params, b_vm has 1, b has 1 = 66 total
- `test_model.py::test_chebyshev_zero_init` — All C=0 → I_ion = b_vm*Vm + b
- `test_model.py::test_chebyshev_linear_recovery` — With C[:,0]=w, C[:,1:]=0, recovers linear: Σ w_k (since T0=1)
- `test_model.py::test_chebyshev_cubic` — Single dim, C=[0,0,0,1] → T3(z̃) = 4z̃³-3z̃. Verify numerically.

#### Checklist
- [ ] Create `surrogate/model/__init__.py`
- [ ] Implement `ChebyshevReadout` with recurrence
- [ ] Register z_min, z_max as buffers (default: -1, +1)
- [ ] Add eps to normalization denominator (prevent div-by-zero)
- [ ] Add safety clamp after normalization
- [ ] Initialize C to zeros (start as zero readout)
- [ ] Write 5 unit tests

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py -v -k chebyshev
```

#### Exit Criteria
- [ ] All 5 chebyshev tests pass
- [ ] 66 parameters confirmed

#### Risk
z_min = z_max for a dim (constant latent dim) → division by zero. Mitigation: eps in denominator + clamp.

---

### Step 1.2: IonicSurrogate nn.Module
**Model**: opus

#### Read First
- KNOWLEDGE.md:17-54 — Full architecture spec (all 3 stages + scaffold)
- KNOWLEDGE.md:106-118 — Parameter table
- `surrogate/model/chebyshev.py` from Step 1.1

#### Why
This is the core model. Every detail matters: the per-dim query in Stage 1 is NOT a standard nn.Linear (it's a broadcast multiply), the spectral norm must wrap both W_cc layers, and the scaffold decoder must be detachable for production.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/ionic_surrogate.py`
**Interfaces / Signatures:**
```python
class IonicSurrogate(nn.Module):
    """3-stage ionic surrogate with scaffold decoder.

    Architecture v2: n×1 cross-attention + 2× split GELU + KAN Chebyshev.
    642 inference params, 886 FLOPs, 3.7× Rush-Larsen.
    """
    def __init__(self, latent_dim: int = 16, attn_dim: int = 8,
                 cheby_degree: int = 3, split: int = 8,
                 n_gates: int = 18, scaffold: bool = True):

    def forward(self, latent_prev: Tensor, Vm: Tensor, dt: Tensor
                ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            latent_prev: (batch, latent_dim) or (latent_dim,)
            Vm: (batch,) or scalar
            dt: (batch,) or scalar
        Returns:
            latent_new: (batch, latent_dim)
            I_ion: (batch,)
            gates_pred: (batch, n_gates) or None if scaffold=False
        """

    def inference_param_count(self) -> int:
        """Count params excluding scaffold."""

    def remove_scaffold(self):
        """Delete scaffold decoder for production."""
```

#### Pseudocode
```
def forward(latent_prev, Vm, dt):
    # Ensure batch dimension
    squeezed = latent_prev.dim() == 1
    if squeezed:
        latent_prev = latent_prev.unsqueeze(0)
        Vm = Vm.unsqueeze(0) if Vm.dim() == 0 else Vm
        dt = dt.unsqueeze(0) if dt.dim() == 0 else dt

    # Stage 1: n×1 cross-attention
    x = torch.stack([Vm, dt], dim=-1)        # (batch, 2)
    k = self.W_k(x)                           # (batch, 8)
    v = self.W_v(x)                           # (batch, 8)
    # Per-dim query: q[b,d,:] = latent_prev[b,d] * W_q[d,:]
    q = latent_prev.unsqueeze(-1) * self.W_q  # (batch, 16, 8)

    score = (q * k.unsqueeze(1)).sum(-1) * self.scale  # (batch, 16)
    gate = torch.sigmoid(score)
    target = v @ self.W_out                    # (batch, 16)
    latent_mid = latent_prev + gate * (target - latent_prev)

    # Stage 2: two-round split GELU
    s = self.split
    gated1 = F.gelu(latent_mid[:, :s]) * latent_mid[:, s:]
    latent_a = latent_mid + self.cc1(gated1)
    gated2 = F.gelu(latent_a[:, :s]) * latent_a[:, s:]
    latent_new = latent_a + self.cc2(gated2)

    # Stage 3: KAN Chebyshev readout
    I_ion = self.readout(latent_new, Vm)

    # Scaffold
    gates_pred = None
    if hasattr(self, 'decoder'):
        gates_pred = torch.sigmoid(self.decoder(latent_new))

    if squeezed:
        latent_new = latent_new.squeeze(0)
        I_ion = I_ion.squeeze(0)
        if gates_pred is not None:
            gates_pred = gates_pred.squeeze(0)

    return latent_new, I_ion, gates_pred
```

#### Test Spec
- `test_model.py::test_surrogate_shapes` — Batch (32,): latent_new (32,16), I_ion (32,), gates (32,18)
- `test_model.py::test_surrogate_single_sample` — No batch dim: latent_new (16,), I_ion scalar, gates (18,)
- `test_model.py::test_surrogate_param_count` — inference: 642, training: 948
- `test_model.py::test_surrogate_contractivity` — Stage 1 only (disable Stage 2 via zero W_cc): ||latent_mid - target|| < ||latent_prev - target|| for random inputs
- `test_model.py::test_surrogate_spectral_norm` — ||W_cc1.weight||₂ ≤ 1.0 + eps, same for W_cc2
- `test_model.py::test_surrogate_no_scaffold` — scaffold=False → gates_pred is None, param count = 642
- `test_model.py::test_surrogate_remove_scaffold` — remove_scaffold() deletes decoder, reduces param count
- `test_model.py::test_surrogate_gradient_flow` — backward pass completes without NaN/Inf for 10-step rollout
- `test_model.py::test_surrogate_deterministic` — Same input → same output

#### Checklist
- [ ] Implement Stage 1 with per-dim query broadcast
- [ ] Implement Stage 2 with spectral_norm on both cc layers
- [ ] Integrate ChebyshevReadout as Stage 3
- [ ] Add scaffold decoder (optional, default True)
- [ ] Handle both batched and unbatched inputs
- [ ] Xavier uniform init for W_q, W_k.weight, W_v.weight, W_out
- [ ] Zero init for readout C, b_vm, b
- [ ] Implement inference_param_count() and remove_scaffold()
- [ ] Write 9 unit tests

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py -v
```

#### Exit Criteria
- [ ] All 14 tests pass (5 chebyshev + 9 surrogate)
- [ ] 642 inference params, 948 training params confirmed
- [ ] Spectral norms ≤ 1.0

#### Risk
Per-dim query broadcast is non-standard — easy to get shapes wrong. Mitigation: explicit shape comments, assertion tests. Also: spectral_norm wrapping must happen in __init__, not later — check with test.

---

### Phase 1 Verification
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py -v
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py -v  # regression
```

### Phase 1 Exit Criteria
- [ ] All 14 new model tests pass
- [ ] All 32 existing data generation tests pass
- [ ] IonicSurrogate forward pass runs on GPU without error

### Phase 1 Cleanup
- [ ] float32 consistency — model uses float32, no float64 leaks
- [ ] No code duplication with existing surrogate/data/ modules
- [ ] Docstrings matching KNOWLEDGE.md parameter table

**-> Commit point: git commit after Phase 1 passes**

---

## Phase 2: Data Loading for Phase A

**Goal**: Load T1 gate snapshots from HDF5 for autoencoder training. Extract (18,) gate vectors from every timestep.
**Tier**: small
**Estimated scope**: 1 file (~100 lines), 2 tests

### Phase Context
- T1 data at `/media/norepinephrine/Elements-ext4/surrogate_data/raw/tier01.h5`
- TraceData columns: gates are 3:21 (18 states). See `single_cell_generator.py:50-62`.
- Phase A needs ONLY gate snapshots — no Vm, no dynamics. Just (18,) vectors.
- Convert float64 → float32 on load.
- T1 has 9 protocols × ~2M steps ≈ 18M vectors. Subsample to ~100K for Phase A.
- TraceStorage.load_trace() loads from HDF5.

### Step 2.1: Gate snapshot DataLoader
**Model**: sonnet

#### Read First
- `surrogate/data/storage.py:55-62` — load_trace API
- `surrogate/data/single_cell_generator.py:50-62` — column indices

#### Why
Phase A trains on individual gate vectors, not temporal segments. Simple loader: extract columns 3:21, convert to float32, subsample, return DataLoader.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/__init__.py`, `Surrogate/surrogate/training/data_loader.py`
**Interfaces / Signatures:**
```python
def load_gate_snapshots(tier: int = 1, max_samples: int = 100_000,
                        raw_dir: str = DEFAULT_RAW_DIR,
                        seed: int = 42) -> torch.Tensor:
    """Returns: (N, 18) float32 tensor of gate snapshots."""

def make_gate_dataloader(gates: torch.Tensor, batch_size: int = 4096,
                         val_frac: float = 0.1, seed: int = 42
                         ) -> Tuple[DataLoader, DataLoader]:
    """Split into train/val, return DataLoaders."""
```

#### Pseudocode
```
def load_gate_snapshots(tier, max_samples, raw_dir, seed):
    storage = TraceStorage(raw_dir)
    protocols = storage.list_protocols(tier)
    all_gates = []
    for name in protocols:
        trace = storage.load_trace(tier, name)
        gates = trace.data[:, 3:21]  # (T, 18)
        all_gates.append(gates)
    all_gates = cat(all_gates, dim=0)
    if len(all_gates) > max_samples:
        idx = randperm(len(all_gates), generator=Manual(seed))[:max_samples]
        all_gates = all_gates[idx]
    return all_gates.float()
```

#### Test Spec
- `test_phase_a.py::test_load_gate_snapshots` — Returns (N, 18) float32, N ≤ max_samples. Requires HDD.
- `test_phase_a.py::test_gate_values_physical` — Gates in [0,1] for gate dims (cols 6-17), concentrations > 0 (cols 0-5). Requires HDD.

#### Checklist
- [ ] Create `surrogate/training/__init__.py`
- [ ] Implement load_gate_snapshots
- [ ] Implement make_gate_dataloader (TensorDataset + random_split + DataLoader)
- [ ] Write 2 tests (mark with `@pytest.mark.skipif` if HDD not mounted)

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_phase_a.py -v -k "load_gate or gate_values"
```

#### Exit Criteria
- [ ] Gate snapshots load from T1 in < 30s
- [ ] float32 output

#### Risk
HDD not mounted. Mitigation: `pytest.mark.skipif` + synthetic data fallback for CI.

---

### Phase 2 Verification
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_phase_a.py -v -k "load_gate or gate_values"
```

### Phase 2 Exit Criteria
- [ ] Gate snapshot loading works
- [ ] DataLoader yields correct batch shapes

### Phase 2 Cleanup
- [ ] float32 consistency
- [ ] No duplication with ShardProcessor

**-> Commit point: git commit after Phase 2 passes**

---

## Phase 3: Phase A — Gate Autoencoder Bootstrap

**Goal**: Train gate autoencoder (18→16→18), transfer decoder to scaffold, compute latent bounds.
**Tier**: medium
**Estimated scope**: 2 files (~250 lines), 5 tests

### Phase Context
- KNOWLEDGE.md § "Phase A — Latent Space Bootstrap"
- Autoencoder: encoder Linear(18,16), decoder Linear(16,18). No activation on decoder output.
- Scaffold decoder applies sigmoid. AE decoder does not — weights transfer directly, scaffold adds sigmoid on top.
- Loss: MSE(recon, gates). Not MSE(sigmoid(recon), gates).
- Training: AdamW, lr=1e-3, wd=1e-5, batch=4096, cosine LR schedule.
- After training: (1) transfer decoder weights to scaffold, (2) compute z_min/z_max from encoder, (3) set ChebyshevReadout bounds.
- z_min/z_max: add 10% margin for rollout drift safety.

### Step 3.1: Gate autoencoder module
**Model**: sonnet

#### Read First
- KNOWLEDGE.md:317-332 — Phase A spec
- `surrogate/model/ionic_surrogate.py` from Phase 1

#### Why
Simple module. Must match scaffold decoder dimensions exactly for weight transfer.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/autoencoder.py`
**Interfaces / Signatures:**
```python
class GateAutoencoder(nn.Module):
    def __init__(self, n_states: int = 18, latent_dim: int = 16):
    def forward(self, gates: Tensor) -> Tuple[Tensor, Tensor]:
        """gates: (batch, 18) → (recon: (batch, 18), latent: (batch, 16))"""
    def transfer_decoder(self, surrogate: 'IonicSurrogate'):
        """Copy decoder weights to surrogate scaffold."""
    def compute_latent_bounds(self, gates: Tensor) -> Tuple[Tensor, Tensor]:
        """z_min, z_max with 10% margin from encoder output stats."""
```

#### Test Spec
- `test_phase_a.py::test_autoencoder_shapes` — (32, 18) → recon (32, 18), latent (32, 16)
- `test_phase_a.py::test_autoencoder_param_count` — 304 + 306 = 610

#### Checklist
- [ ] Implement GateAutoencoder (encoder + decoder, no activation)
- [ ] Implement transfer_decoder (copy weight + bias)
- [ ] Implement compute_latent_bounds with 10% margin
- [ ] Write 2 tests

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_phase_a.py -v -k autoencoder
```

#### Exit Criteria
- [ ] Shapes correct, param count = 610
- [ ] Decoder weight shape (16, 18) matches scaffold

#### Risk
None significant — simple linear autoencoder.

---

### Step 3.2: Phase A training script
**Model**: opus

#### Read First
- KNOWLEDGE.md:317-332 — Phase A spec
- KNOWLEDGE.md:384-397 — Training timeline
- `surrogate/training/data_loader.py` from Phase 2
- `surrogate/model/autoencoder.py` from Step 3.1

#### Why
Integrates everything: load T1, train AE, compute bounds, transfer to surrogate, save checkpoint. The training is simple (standard AE) but the integration matters for Phase B readiness.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/train_phase_a.py`
**Interfaces / Signatures:**
```python
def train_phase_a(
    raw_dir: str = DEFAULT_RAW_DIR,
    max_samples: int = 100_000,
    latent_dim: int = 16,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    n_epochs: int = 200,
    device: str = 'cuda',
    checkpoint_dir: str = 'checkpoints/phase_a',
) -> Tuple[GateAutoencoder, IonicSurrogate, dict]:
    """Train gate AE, transfer to surrogate, return metrics."""
```

#### Pseudocode
```
def train_phase_a(...):
    # Load
    gates = load_gate_snapshots(tier=1, max_samples=max_samples)
    train_loader, val_loader = make_gate_dataloader(gates, batch_size)

    # Build
    ae = GateAutoencoder(18, latent_dim).to(device)
    opt = AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=n_epochs)

    # Train loop
    best_val = inf
    for epoch in range(n_epochs):
        # train step: MSE(recon, batch)
        # val step
        # save best by val_loss
        # print every 20 epochs

    # Post-training
    ae.load_state_dict(best checkpoint)
    z_min, z_max = ae.compute_latent_bounds(all_gates.to(device))

    surrogate = IonicSurrogate(latent_dim=latent_dim, scaffold=True).to(device)
    ae.transfer_decoder(surrogate)
    surrogate.readout.set_bounds(z_min, z_max)

    # Save
    torch.save({...}, checkpoint_dir / 'phase_a_complete.pt')
    return ae, surrogate, metrics
```

#### Test Spec
- `test_phase_a.py::test_phase_a_smoke` — 1000 synthetic samples, 5 epochs. Loss decreases. Scaffold has non-zero weights. Bounds are set.
- `test_phase_a.py::test_phase_a_reconstruction` — 10K synthetic samples, 50 epochs. MSE < 0.01 (relaxed for speed).
- `test_phase_a.py::test_decoder_transfer` — AE decoder.weight == surrogate.decoder.weight after transfer.

#### Checklist
- [ ] Implement train_phase_a
- [ ] Create checkpoint directory automatically
- [ ] Cosine LR schedule
- [ ] Save best model by val loss
- [ ] Compute and set latent bounds
- [ ] Transfer decoder to surrogate
- [ ] Print progress every 20 epochs
- [ ] Write 3 tests (use synthetic data for speed — random uniform [0,1] for gates, random positive for concentrations)

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_phase_a.py -v
```

#### Exit Criteria
- [ ] Phase A trains to completion
- [ ] Val MSE < 0.001 on real T1 data (manual run, not in test)
- [ ] Surrogate scaffold initialized, bounds set
- [ ] Checkpoint saved

#### Risk
T1 HDD not mounted during test. Mitigation: tests use synthetic data. Real T1 training is a manual run, not automated test.

---

### Phase 3 Verification
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_phase_a.py -v
cd Surrogate && conda run -n heart-conduction python -m pytest tests/ -v  # full regression
```

### Phase 3 Exit Criteria
- [ ] All Phase A tests pass
- [ ] All 46+ tests pass (32 data gen + 14 model + 5 phase A)
- [ ] Manual run on T1 data: val MSE < 0.001
- [ ] Checkpoint at `checkpoints/phase_a/phase_a_complete.pt`

### Phase 3 Cleanup
- [ ] float32 consistency throughout training pipeline
- [ ] No hardcoded paths (use function arguments with defaults)
- [ ] Checkpoint dir auto-created

**-> Commit point: git commit after Phase 3 passes**

---

## Final Cleanup
- [ ] float32 consistency — all ML tensors float32 (no float64 leaks from data loading)
- [ ] V5.3 not modified
- [ ] No code duplication across engines
- [ ] Archive plan:
```bash
mkdir -p Research/Active/surrogate_pipeline/plans
cp Research/Active/surrogate_pipeline/PLAN.md "Research/Active/surrogate_pipeline/plans/$(date +%Y-%m-%d)_ionic-surrogate-model-phase-a.md"
```
- [ ] Revert tmux pane:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Mutation Log
(Initially empty — populated during execution)
