# Surrogate Pipeline — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Architecture v2 settled and implemented: n×1 cross-attention + two-round split GELU (with RMSNorm) + KAN Chebyshev readout. 886 FLOPs, 3.7× Rush-Larsen, 642 params. I_stim removed from model input. Data generation T1-T12 complete. Model code (ChebyshevReadout + IonicSurrogate) implemented with 50/50 tests passing. RMSNorm added to Stage 2 corrections (both rounds) for stability. ARCHITECTURE_v2.md written as detailed design document.

## Next Step
Update PLAN.md to reflect RMSNorm addition and update IonicSurrogate tests for RMSNorm behavior. Then Phase A autoencoder training blueprint.

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
| 2026-03-21 | 9 | Full implementation + benchmarking. 8 source files, 32/32 tests, committed. BatchGenerator with torch.compile: 77.9M cell-steps/s at n=10K (47,227× vs CPU sequential). Padding strategy validated: replicate unique protos to fill batch, copies match bitwise. |
| 2026-03-22 | 10 | Added gate_inf + gate_tau columns (23→47 cols). Per-step computation was 40% overhead — fixed to post-hoc vectorized (0% overhead). Ran T1+T2 generation (8.6GB on HDD). T3-12 still needed. Current run_datagen.py doesn't use padding → small batches run at CPU-equivalent speed (~500 steps/s/cell instead of ~7800). Need padding optimization before T4 (200 protos would take 12+ hours without it). |
| 2026-03-22 | 11 | Overnight T4-12 generation crashed (OOM: 1.2TB buffer for 200 protos × 16M steps). Fixed with chunked processing (500K steps/chunk, ~0.9GB each). Verified chunking works — T4 test running at 23% (6/29 chunks done, no OOM, 6K cell-steps/s at n=5). MASTER.md updated with bidomain_parabolic_parabolic question. |
| 2026-03-23 | 12 | Architecture v2: I_stim removed, Stage 2 doubled (two-round split GELU), Stage 3 KAN Chebyshev K=3. Data gen fixes (searchsorted, incremental HDF5, naming). T1-T11 complete, T12 running. |
| 2026-03-24 | 13 | Architecture deep-dive: attention math walkthrough, GPU friendliness analysis (kernel launch overhead, torch.compile fusion, teacher-forced temporal parallelism). Blueprint for model impl (2 phases, 18 tests). Audited twice — fixed spectral_norm init ordering (CRITICAL), KNOWLEDGE.md column layout 18+6→12+12 (CRITICAL), 4 high issues. Clean re-audit: 0 critical. |
| 2026-03-26–30 | 14 | Model implementation: ChebyshevReadout (66 params) + IonicSurrogate (642/948 params). 18 model tests + 32 data gen tests = 50/50 passing. RMSNorm added to Stage 2. TikZ architecture diagram (ionic_surrogate_v2.tex). ARCHITECTURE_v2.md written. Rejected sigmoid output bounding, LayerNorm, BatchNorm. Keras deemed unsuitable. |

### 2026-03-23 — Session 12: Architecture v2 + T4-T12 data generation

**Data generation fixes and completion:**
- searchsorted fix for RandomIntervalPacing schedule: O(n_steps × log(n_beats)) instead of O(n_steps × n_beats). Schedule computation dropped from 13+ min to <10s for 200 protocols.
- Lazy ext/clamp tensor allocation: only allocate when protocols actually use I_ext or clamp. Saves 78GB for T4.
- Incremental HDF5 writes: per-protocol chunk data written immediately via resizable datasets instead of accumulating in memory. Peak RAM ~42GB instead of 1.2TB.
- Sequential protocol naming bug fix: all 50 stitched protocols were saving as 'stitched_dt0.01', overwriting each other. Fixed with index suffix.
- T12 tier bug: was writing to tier01/02/03 instead of tier12. Fixed.
- T1-T10 complete (569GB). T11 complete (18GB, 50/50 stitched protocols). T12 running (ENDO/M_CELL celltypes).

**Architecture v2 decisions:**
1. **I_stim removed from model input**: Gates respond to Vm only (biophysically correct). Matches operator splitting (ionic step sees only Vm). I_stim applied externally in Vm update. W_k, W_v: (3,8)→(2,8), -16 params.
2. **Stage 2 doubled to two rounds**: Two sequential split GELU rounds. Round 1 produces pairwise products (m·h), Round 2 produces products of products (m·h·j). Both W_cc spectral-normed. +144 params, +176 FLOPs.
3. **Stage 3 upgraded to KAN Chebyshev K=3**: Per-dim learned 1D functions via Chebyshev polynomial basis (T₀ through T₃). Captures per-dim nonlinearity (m³, Xs²) that linear readout can't. 16 dims × 4 coefficients = 64 params + b_vm + b = 66 total.
4. **Design principle settled**: "Biophysics-inspired, not biophysics-prescribed. Provide learnability, let the optimizer decide." Rejected Nernst-structured readout as over-prescriptive for first pass.

**Updated totals**: 642 inference params, 886 FLOPs (3.7× Rush-Larsen). Training: 948 params, 1210 FLOPs.

**Training concerns noted:**
- **Temporal imbalance**: ~70% of timesteps are diastole (I_ion≈0). MSE dominated by trivial resting-state prediction. Fix: phase-weighted loss (weight by |dVm/dt|) — implement in training loop.
- **Tier imbalance**: T4 is 90% of data by size. Mitigated by curriculum (Phase B=T1 only, C=T1-T4, D=all). Sharding normalizes across tiers.
- **Stiffness**: 1000× timescale ratio (m gate τ≈0.1ms vs I_Ks τ≈100ms). Per-dim gating handles this naturally (gate≈0.1 for fast, ≈0.0001 for slow — same principle as Rush-Larsen). Concern: sigmoid must span 1000× range, small weight changes flip between frozen and snapping. Gradient signal temporally mismatched in rollout training. Manageable because: 16 dims > 12 gates (room for timescale separation), dt as input, ZOH upgrade available if needed.
- **T1-T12 data generation complete**: 608GB total. T4=551GB (200 random protocols), T12=22GB (ENDO/M_CELL).

### 2026-03-24 — Session 13: Architecture deep-dive + blueprint

**Architecture discussions:**
- Walked through cross-attention math step-by-step (per-dim query broadcast, shared K/V, sigmoid gate, contraction toward target).
- W_q per-dim projection: each latent scalar × one row of W_q → 8-dim query. NOT a standard nn.Linear.
- Attention dim d=8: √8=2.83 scaling, 4× expansion from 2 inputs, balances capacity vs cost.
- GPU analysis: ~25-30 kernel launches per forward pass. torch.compile fuses to ~5-8. Teacher-forced training parallelizes across time (reshape T×B into single batch). Rollout and inference must stay sequential — not a bottleneck for tiny model. All ops are GPU-native, no CPU-only ops.
- Chebyshev recurrence Python loop (2 iterations) is NOT a CPU-GPU sync issue — PyTorch ops are async, torch.compile unrolls it.
- Inference in tissue: batch size = mesh size (10K+), kernel launch overhead amortized. Real speedup comes from replacing diffusion solve (94% of cost), not ionic step (6%).

**Blueprint created:** `PLAN.md` — 2 phases (ChebyshevReadout + IonicSurrogate), 18 tests total.

**Audit fixes (2 rounds):**
- CRITICAL: spectral_norm init must happen BEFORE wrapping (weight_orig gotcha). Fixed pseudocode.
- CRITICAL: KNOWLEDGE.md column layout was 18+6, actual code is 12+12. Fixed.
- HIGH: contractivity test renamed to Stage 1 only, full-model contractivity noted as open risk.
- HIGH: import cascade prevention — do NOT modify top-level surrogate/__init__.py.
- Added tests: set_bounds, constant_dim, out_of_bounds, import cascade, dtype assertions, idempotent remove_scaffold.
- Re-audit: 0 critical, 3 high (all "be aware" notes).

**Next**: Implement PLAN.md Phase 1 (ChebyshevReadout) then Phase 2 (IonicSurrogate).

### 2026-03-26–30 — Sessions 14: Model implementation + architecture refinement

**Model implementation (Phase 1 + Phase 2):**
- Implemented ChebyshevReadout (`chebyshev.py`): Chebyshev polynomial readout with set_bounds(), normalize-to-[-1,1], recurrence loop, b_vm and b terms. 66 params.
- Implemented IonicSurrogate (`ionic_surrogate.py`): Full 3-stage model. spectral_norm applied BEFORE any weight wrapping (critical ordering). remove_scaffold() strips decoder and latent→weight hooks for production. 642 inference / 948 training params.
- 18 model tests all pass: shape, contractivity (Stage 1), spectral norm, gradient flow, remove_scaffold idempotency, set_bounds, dtype assertions, constant dim, out-of-bounds, import cascade.
- 32 data generation tests still pass (50/50 total).

**RMSNorm addition to Stage 2:**
- Added inline RMSNorm after W_cc output in both Stage 2 rounds. Normalizes the correction to consistent RMS before residual add. Prevents quadratic blowup from split GELU product.
- Implementation: `corr = corr / (corr.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)` — inline, no learnable params, no module.
- Spectral norm + RMSNorm together: "belt and suspenders." SN bounds the operator (||W||_2 <= 1), RMSNorm bounds the input scale. Hard guarantee: ||correction|| <= sqrt(16) + ||b||.
- Zero additional parameters. 18/18 tests still pass after addition.

**Architecture decisions (rejected alternatives):**
- **Sigmoid output bounding rejected**: Vanishing gradients at saturation, breaks residual identity at initialization, triple sigmoid path (Stage 1 gate + sigmoid + sigmoid = three chained sigmoids), concentration variables not bounded in [0,1]. Stability comes from architecture (spectral norm + RMSNorm + contractive Stage 1) instead.
- **LayerNorm in attention rejected**: Single attention step (not stacked transformer), per-dim magnitude IS information for state-dependent gating (removing it defeats the purpose of per-dim query), 1/sqrt(8) scaling sufficient.
- **No LayerNorm replacing W_cc**: W_cc is needed for 8->16 dimension expansion. LayerNorm adds centering (unneeded) and learnable params (unneeded). RMSNorm after W_cc is cleaner.
- **BatchNorm rejected**: Batch statistics unstable for autoregressive inference (batch=1 during tissue simulation). RMSNorm is instance-level.
- **Chebyshev normalization vs RMSNorm**: Different purposes. Chebyshev normalization maps to [-1,1] for polynomial stability (interval mapping). RMSNorm normalizes scale (prevents quadratic blowup). Both needed.

**TikZ architecture diagram:**
- Publication-quality TikZ/LaTeX diagram of full 3-stage pipeline.
- Horizontal (left-to-right) flow with Stage 1 columns, Stage 2/3 top-to-bottom.
- Merge convention: two lines -> vertical/horizontal bar -> single arrow.
- Residual skip connections branch from inter-stage routing junctions.
- Symbols: circled-asterisk for concatenation, circled-plus for add, circled-times for Hadamard.
- Rush-Larsen equivalence boxes below each stage.
- Source: `Images/ionic_surrogate_v2.tex`, rendered: `Images/ionic_surrogate_v2-1.png`.

**ARCHITECTURE_v2.md:**
- Nature-style detailed writeup of full architecture with biophysical reasoning for every design decision.
- Covers: problem formulation, Stage 1 (cross-attention), Stage 2 (split GELU + RMSNorm + spectral norm), Stage 3 (Chebyshev), scaffold, stability analysis.
- Located at `Research/Active/surrogate_pipeline/ARCHITECTURE_v2.md`.

**Keras compatibility assessment:**
- Concluded Keras is not suitable for this model. Raw nn.Parameters (W_q is not a layer), spectral_norm hooks, inline RMSNorm, remove_scaffold() all fight Keras's layer-oriented API. Staying with pure PyTorch.

**Files created/modified:**
- NEW: `Surrogate/surrogate/model/__init__.py`
- NEW: `Surrogate/surrogate/model/chebyshev.py`
- NEW: `Surrogate/surrogate/model/ionic_surrogate.py` (updated with RMSNorm)
- NEW: `Surrogate/tests/test_model.py` (18 tests)
- NEW: `Images/ionic_surrogate_v2.tex` (TikZ source)
- NEW: `Images/ionic_surrogate_v2-1.png` (rendered diagram)
- NEW: `Research/Active/surrogate_pipeline/ARCHITECTURE_v2.md`
- NEW: `Surrogate/TIKZ_REFERENCE.md`

**Next**: Update PLAN.md for RMSNorm, then Phase A autoencoder training blueprint.

### 2026-03-19 Session 7 Snapshot
**Worked on**: Ionic surrogate ML architecture — detailed design from first principles
**Accomplished**: Complete architecture pivot from Transformer to carried-latent cross-attention. Explored 12 candidate architectures with cost analysis. Settled on 3-stage design (n×1 cross-attn + split GELU + linear readout) at 673 FLOPs / 2.8× RL. Defined simplification spectrum (Levels 0-3) and modification menu (accuracy/speed upgrades). Conducted adversarial review (math + neuro critiques). All documented in WHITEBOARD.
**Advanced ML upgrades identified** (document, don't implement yet):
- Our architecture IS a selective SSM (Mamba) — independently derived. Can borrow parallel scan for O(log N) training and ZOH discretization for exact exponential update (+16 FLOPs).
- KAN readout: replace linear w·z with learned splines φ_k(z_k). Captures m³·h·j without hidden layer (+47 FLOPs).
- MoE: 4 phase-specialized experts with top-1 routing. 4× capacity at +8 FLOPs (router only). Each expert learns one AP phase.

### 2026-03-20 — Session 8: Training data + strategy + pipeline audit

**Training data**: Designed 12-tier hierarchy (T1-T12) covering steady-state, S1-S2, dynamic, random intervals, tissue-mimicking injection, voltage clamp, concentration perturbation, long-duration stability, corruption recovery, tissue-specific scenarios, combined stressors, and celltype variants. Key insight: tissue cells see smooth diffusion current, not clean pacing pulses. Variable dt, on-the-fly augmentation. Storage: HDF5 raw + .pt shards for training. Full details in KNOWLEDGE.md.

**Training strategy**: 4-phase curriculum (A: autoencoder bootstrap, B: simple dynamics, C: full dynamics, D: robustness). AdamW, scheduled sampling, gradual data mixing, double rollout length. ~40 GPU-hours estimated. Full details in KNOWLEDGE.md.

**Pipeline audit** (12 issues found, all fixed):
- CRITICAL: W_out must be (8,16) not (8,1); parallel scan incompatible with Stage 2
- HIGH: dt added to K/V input; gate decoder FLOPs corrected; voltage clamp moved to Phase C
- MEDIUM: clamp mask, I_stim biophysics concern, spectral norm > epsilon-scaling, overfitting risk, encoder-dynamics mismatch
- Post-audit numbers (v1): 705 FLOPs, 2.9x RL, 466/772 params (later updated to v2: 886 FLOPs, 3.7x RL, 642/948 params)
