# Surrogate Pipeline — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Ionic surrogate architecture settled: n×1 cross-attention + split GELU cross-channel + linear readout. 705 FLOPs, 2.9× Rush-Larsen, 466 params. Pure autoregressive latent (no Vm history). Strategy: start at scalar HH (Level 0), iterate up. Next: training data generation and training strategy.

## Next Step
Design training data generator (TTP06 single-cell trajectories with varied pacing protocols) and training strategy (curriculum, scaffold annealing, rollout schedule).

## Thread

### 2026-03-11 — Session 1: Initial scaffold
Created Surrogate/ folder with README, improvement.md, IMPLEMENTATION.md, PROGRESS.md. Two-component architecture (ionic Transformer + dual-path ResNet).

### 2026-03-12 — Sessions 2-4: Architecture refinement
Refined to Vm-only Transformer input, cross-skip coupled ResNet, monodomain-first build order, universal latent space + universal MLP_ion, gate decoder as training scaffold. 18 design decisions logged.

### 2026-03-18 — Session 5: Research structure
Created research question in `Research/Active/surrogate_pipeline/`.

### 2026-03-19 — Session 6: Competitive landscape
Surveyed 5 surrogate approaches for cardiac EP. **No bidomain surrogates exist.** Closest: AGATA (autoregressive GNN, monodomain, Mitchell-Schaeffer, 12× speedup vs FEM). Our differentiators: only bidomain (phi_e), only biophysically detailed ionic (TTP06/ORd), only physics-aware architecture (operator splitting). Filed 4 literature summaries + AGATA PDF.

### 2026-03-19 — Session 7: Ionic architecture — major pivot

**Problem**: Original Transformer design was 200M FLOPs/node/step (1M× Rush-Larsen). The ionic step is 1% of bidomain cost — making it slower defeats the purpose.

**Exploration path** (each rejected for cause):
1. Transformer on 300-pt Vm history → 1M× RL, buffer management nightmare
2. History buffer with non-uniform schedule → only covers 3ms at dt, restitution impossible
3. Extended 512-pt buffer → coverage ok but O(n²) attention still expensive
4. Fourier decomposition of Vm history → interesting (sliding DFT is O(K)) but unnecessary once we found carried latent
5. Carried latent + short Vm window → hybrid, but Vm history has stimulus artifact problem
6. Pure autoregressive latent → sufficient (Rush-Larsen doesn't use history either)
7. GRU cell → works but gating mechanism unnecessary given residual formulation
8. Residual MLP → 7× RL, fine but no state-dependent gating
9. Learned Rush-Larsen → too constrained (assumes HH exponential, independent dims)
10. Neural ODE → too unconstrained (multi-timescale learning hard)
11. Micro-Transformer (17×17 self-attention over latent dims) → 47× RL, overkill
12. **n×1 cross-attention** (latent queries Vm) → state-dependent gating, 464 FLOPs

**Key insight**: n×1 cross-attention IS a 1D hybrid Transformer in QKV notation. Q=latent, K=V=Vm. Each latent dim queries voltage independently. Mathematically equivalent to learned Rush-Larsen but with state-dependent gating (gate depends on both Vm AND current latent value).

**Settled architecture (3 stages):**
- Stage 1 (n×1 cross-attention): Q=W_q·latent, K=W_k·[Vm,I_stim], V=W_v·[Vm,I_stim]. gate=σ(QK^T/√d). delta=gate·(VW_out - latent). Contractive by construction.
- Stage 2 (split GELU cross-channel): GELU(latent[:8])⊙latent[8:] → W_cc(8×16). Cheaper than linear 16×16 AND adds nonlinearity. Rank-8 bottleneck matches real coupling rank (~3 concentrations).
- Stage 3 (linear current readout): I_ion = w·latent + b_vm·Vm + b. No activation (I_ion is unbounded). Linear is physically correct — real current is sum of per-channel contributions.
- Scaffold: Gate decoder = single linear + sigmoid → 18 gates. Training only.

**Final numbers**: 673 FLOPs inference (2.8× RL), 1,761 FLOPs training (7.3× RL), 418/724 params.

**Simplification spectrum** (start simple, add complexity as needed):
- Level 0: Scalar HH (176 FLOPs, 0.7× RL) — σ(wV+b) target, const rate, Nernst current
- Level 1: + Vm-dependent rates (240 FLOPs, 1× RL)
- Level 2: + split GELU coupling (416 FLOPs, 1.7× RL)
- Level 3: Full design (673 FLOPs, 2.8× RL) — n×1 cross-attn + split GELU + linear readout
- Accuracy upgrades available: multi-head, stacked attention, nonlinear readout, explicit concentrations, Nernst current

**Adversarial review** (steelmanned neuro + math critics):
- HIGH: Stage 2 can break Stage 1 contractivity → mitigate with ε-scaling or spectral norm
- HIGH: Error accumulation over 100K+ steps uncharacterized → validate with long rollouts, gate scaffold grounds per-step
- MEDIUM: Ca handling is compartmental not gate-like → monitor Ca gate predictions, add explicit Ca (mod A5) if needed
- MEDIUM: No conservation laws → monitor drift, add penalty if needed
- LOW: "Reinvented HH" — yes, but learned for Ca imaging transfer. Uninterpretable latent — acceptable, gate decoder during dev.

## Failed Approaches

| Approach | Why it failed |
|----------|---------------|
| Temporal Transformer (300-pt history) | 200M FLOPs = 1M× Rush-Larsen. Buffer management, resampling, variable dt issues. |
| Vm history buffer (any size) | Stimulus artifacts contaminate history. Non-uniform schedule is a hyperparameter maze. At dt=0.01ms, even 300 points only covers 3ms. |
| Fourier decomposition of Vm | Sliding DFT is O(K) and clever, but unnecessary once carried latent eliminates the need for history. |
| Learned Rush-Larsen | Too constrained — forces HH exponential relaxation and independent dimensions. Can't represent Markov or Ca dynamics. |
| Neural ODE (dz/dt = MLP(z,Vm)) | Too unconstrained — multi-timescale learning is notoriously hard without structural priors. |
| GRU cell | Works but gating mechanism adds cost (10× RL) without clear benefit over residual formulation. |
| 17×17 self-attention over latent dims | 47× RL. Cross-channel coupling not worth the cost. n×1 + linear coupling achieves the same. |
| Deep MLP for cross-channel | Overkill. Single linear (or split GELU) layer suffices — real coupling is rank-3. |

## Session Log
| Date | Session | Work Done |
|------|---------|-----------|
| 2026-03-11 | 1 | Created scaffold documents |
| 2026-03-12 | 2-4 | Architecture design and refinement |
| 2026-03-18 | 5 | Research question scaffolded |
| 2026-03-19 | 6 | Competitive landscape survey — 5 papers, no bidomain surrogates |
| 2026-03-19 | 7 | Major ionic architecture pivot: Transformer → carried latent with n×1 cross-attention. Explored 12 architectures. Settled: 673 FLOPs, 2.8× RL. Adversarial review done. |
| 2026-03-20 | 8 | Training data (12 tiers), training strategy (4 phases), pipeline audit (12 issues found and fixed — W_out dims, parallel scan, gate decoder FLOPs, voltage clamp timing, spectral norm) |
| 2026-03-21 | 9 | Blueprint for data generation (4 phases, 32 tests). Two audit rounds (34 fixes total). Began implementation: Phase 1 Steps 1.1-1.3 done (package scaffold, SingleCellGenerator, protocols). 10/10 tests passing. Continuing through all phases. |

### 2026-03-19 Session 7 Snapshot
**Worked on**: Ionic surrogate ML architecture — detailed design from first principles
**Accomplished**: Complete architecture pivot from Transformer to carried-latent cross-attention. Explored 12 candidate architectures with cost analysis. Settled on 3-stage design (n×1 cross-attn + split GELU + linear readout) at 673 FLOPs / 2.8× RL. Defined simplification spectrum (Levels 0-3) and modification menu (accuracy/speed upgrades). Conducted adversarial review (math + neuro critiques). All documented in WHITEBOARD.
**Advanced ML upgrades identified** (document, don't implement yet):
- Our architecture IS a selective SSM (Mamba) — independently derived. Can borrow parallel scan for O(log N) training and ZOH discretization for exact exponential update (+16 FLOPs).
- KAN readout: replace linear w·z with learned splines φ_k(z_k). Captures m³·h·j without hidden layer (+47 FLOPs).
- MoE: 4 phase-specialized experts with top-1 routing. 4× capacity at +8 FLOPs (router only). Each expert learns one AP phase.

### 2026-03-20 — Session 8: Training data generation plan

Designed 7-tier training data hierarchy for TTP06 single-cell:
- Tiers 1-4: standard pacing protocols (steady-state, S1-S2, dynamic, random intervals)
- Tier 5: tissue-mimicking current injection (OU noise, ramps, sub-threshold blips, biphasic, random telegraph, real I_diff extracted from Bidomain V1 runs). Key insight: in tissue, cells don't see clean pacing pulses — they see smooth diffusion current from neighbors. Must train on these profiles for tissue generalization.
- Tier 6: voltage clamp (step, ramp, staircase, AP clamp, partial clamp). Uniquely valuable because I_ion errors don't propagate through Vm (voltage is fixed) → clean gradients. Gate decoder gets exact supervision targets (gate_inf at clamp voltage).
- Tier 7: concentration perturbation (K_o, Na_i, Ca_i variations). Simulates tissue concentration variability.

Variable dt: train with dt ∈ {0.005-0.1ms}, feed dt as 3rd input to model K = W_k·[Vm, I_stim, dt]. With ZOH discretization, dt is explicit in formula.

Data augmentation (on-the-fly): Vm noise, conductance scaling, stimulus variation, random initial conditions, multi-scale dt within traces.

Advanced: phase space coverage monitor, curriculum over complexity, adversarial protocol generation, interpolation ground truth.

Gap analysis revealed 8 coverage holes. Added 5 new tiers:
- Tier 8: Long-duration stability (200+ beats, 5-30s quiescence, long blank→burst, variable-length traces)
- Tier 9: Recovery from corruption (random gate perturbation, sudden Vm jumps → record recovery)
- Tier 10: Tissue-specific scenarios (boundary cells, infarct border, inert tissue interface, stimulus site, spiral wave tip). Key: these simulate reduced/asymmetric electrotonic loading.
- Tier 11: Combined stressors (random pacing + OU noise + hyperkalemia simultaneously)
- Tier 12: Celltype variants (TTP06 epi/endo/M, ×3 all Tier 1-3 protocols)

Increased Tier 4 random protocols from 10 → 200 with varying trace lengths (5-200 beats).

Added stitched protocol augmentation to Tier 11 (random protocols concatenated with LogUniform(1-30s) rest breaks, 500+ traces).

Data storage format settled:
- Raw: HDF5 per tier (float64, full metadata, archival)
- Training: pre-chunked .pt shard files (~200MB each, float32, pre-shuffled). Segments of (N, seg_len, 22) loaded directly to GPU. Zero parsing overhead.
- float32 for training (2× savings vs float64, ML doesn't need the precision)
- Train/val split by protocol (unseen pacing patterns), not by timestep
- Augmentation: conductance/stitching/dt offline, Vm noise on-the-fly
- ~500GB total dataset, loads 1-2 shards at a time into GPU

### 2026-03-20 — Session 8 (continued): Training strategy

Settled 4-phase training curriculum:
- Phase A: Gate autoencoder bootstrap (encoder+decoder on gate snapshots, establishes latent coordinate system, trains in minutes)
- Phase B: Simple dynamics (Tier 1 only, decoder frozen, teacher forcing → N=10 → N=100 rollout, λ_gate dominant)
- Phase C: Full dynamics (Tiers 1-4, decoder unfrozen, N=1000→100000 rollout, λ_gate annealed to 0, all celltypes + dt values)
- Phase D: Robustness (Tiers 5-12, scaffold removed, tissue injection + clamp + stress testing, 5-beat rollout)

Key decisions:
- AdamW with variable weight decay: 1e-5 (Phase A) → 1e-3 (Phase D). Increase when overfitting detected.
- Scheduled sampling to bridge teacher forcing → autoregressive gap (10%→100% model predictions)
- Gradual data mixing at phase transitions (90/10 ratio to avoid destabilization)
- Double rollout length each time (not 10× jumps) to manage error accumulation
- Cosine LR decay within phases, reset at transitions
- Checkpoint best model per phase

Estimated ~40 GPU-hours total on Blackwell.

### 2026-03-20 — Session 8 (continued): Pipeline audit

Full audit of KNOWLEDGE.md found 12 issues (2 critical, 3 high, 5 medium, 2 low). All fixed:

**Critical fixes:**
- W_out must be (8,16) not (8,1) — all dims need per-dim targets. Params: 466 (not 418). FLOPs: 705 (not 673).
- Parallel scan incompatible with Stage 2 (GELU is nonlinear, breaks affine recurrence). Sequential backprop feasible for tiny model + gradient checkpointing.

**High fixes:**
- dt added to K/V input: W_k, W_v now (3,8) not (2,8).
- Gate decoder FLOPs corrected: 324 (not 1,088). Was stale from old 3-layer MLP.
- Voltage clamp moved from Phase D to Phase C (scaffold still active → maximizes gate supervision).

**Medium fixes:**
- Training loop needs clamp mask for mixed free/clamped batches.
- I_stim input flagged as biophysically questionable — keep for now, evaluate in Phase B.
- Spectral norm chosen over ε-scaling for Stage 2 stability (Lipschitz bound, not output scaling).
- Added overfitting risk in Phase B (466 params vs ~180 training APs).
- Added encoder-dynamics latent mismatch monitoring.

**Downstream consequences:**
- No parallel scan → gradient checkpointing for long rollouts (O(√N) memory, 2× compute). Feasible.
- Larger W_out → 29% more params → overfitting risk in Phase B. Monitor train/val gap.
- Voltage clamp in Phase C → need clamp data earlier, training loop more complex in Phase C.

**Corrected numbers:** 705 FLOPs inference (2.9× RL), 1,029 FLOPs training (4.3× RL), 466/772 params.

**All three questions settled:**
1. ML structure: n×1 cross-attention + split GELU + linear readout (705 FLOPs, 2.9× RL, 466 params)
2. Training data: 12-tier protocol hierarchy + augmentation + .pt shard storage
3. Training strategy: 4-phase curriculum (bootstrap → simple → full → robust)

**Next**: Complete Phase 1 (HDF5 storage), then Phases 2-4 of data generation PLAN.md.
