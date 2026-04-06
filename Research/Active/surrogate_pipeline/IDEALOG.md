# Surrogate Pipeline — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
**Considering pivot to Neural ODE approach.** The discrete autoregressive rollout at native dt (0.01ms, 30K steps) cannot converge despite: dt curriculum (A1-A3 worked, A4 stuck), TBPTT, warm restarts, min-max normalization, batch tuning. The error compounding problem is fundamental to discrete autoregressive training — not a hyperparameter issue.

What worked: dt curriculum A1→A3 (coarse to medium dt), model learns AP shape. What failed: A4 (native dt, 30K steps) — val stuck at ~720 for 155+ epochs.

Neural ODE would replace discrete steps with continuous dynamics: dz/dt = f(z, Vm). Adjoint method handles gradients. No error compounding. See Salvador 2024 (300x speedup on cardiac electromechanics).

## Next Step
Research Neural ODE approach for ionic surrogate. Key questions: (1) can torchdiffeq handle the multi-timescale stiffness (0.1ms to 100ms+)? (2) how to couple with the existing attention architecture? (3) what about the operator splitting structure (Stage 1 off-path, Stage 2 on-path)? The v3 attention + MLP architecture may still be the right dynamics model — just integrated with an ODE solver instead of discrete steps.

### 2026-04-06: A4 failed — considering Neural ODE pivot
A4 (dt=0.01ms, rollout=30K) ran 155 epochs with TBPTT(window=500) + warm restarts (T_0=50, T_mult=2). Val stuck at ~720, zero improvement after two full restart cycles. The discrete autoregressive approach hits a wall at native dt — 30K sequential steps compound errors faster than the model can learn to correct them. TBPTT with window=500 (5ms) may be too narrow — gradients can't reach early steps where errors originate. But expanding the window defeats the purpose (gradient chain gets long again). User suggests Neural ODE as the right abstraction for continuous ionic dynamics. The v3 attention mechanism (voltage-dependent gating) may still work as the dynamics function f(z, Vm) inside an ODE solver.

### 2026-04-04: dt curriculum + TBPTT + warm restarts
- **dt curriculum**: Instead of increasing rollout at fixed dt=0.01ms, fix coverage at 300ms and vary dt. Subsample existing T1 data: dt=3ms(r=100)→dt=1ms(r=300)→dt=0.1ms(r=3000)→dt=0.01ms(r=30000). Same AP, different resolution. Model learns shape first (cheap), refines later. Replaces entire rollout=1→10→100→1K→10K curriculum.
- **Phase rename**: A=Stage 1 (A1-A4 Half 1 ionic, B1-B4 Half 2 conductance), C=Stage 2, D=end-to-end. Cleaner mapping to what's being trained.
- **Min-max loss normalization**: Per-dim normalization to [0,1] using physiological ranges from ALL tiers. Replaces variance normalization. Every dim contributes equally — Ca_i (0.0001 mM) gets same weight as K_i (138 mM). Ranges with 10-20% safety margin.
- **Batch size lesson**: 32768 doesn't learn (too few optimizer steps). 4096 with scaled LR (4e-3) explodes. Batch=4096, LR=5e-4/1e-4 is the proven config. Don't chase GPU utilization.
- **dt curriculum results**: A1(dt=3ms) unstable but learned. A2(dt=1ms) val=1.15. A3(dt=0.1ms) val=0.92. Progressive improvement through the dt stages.
- **A4 struggle**: dt=0.01ms, rollout=30K. Train~870, val~840 at epoch 58. The 30K autoregressive steps compound errors. Grad norms in millions.
- **Truncated BPTT**: Forward pass runs all 30K steps. Gradients only through last 500 (5ms). Detach earlier steps. Model sees full trajectory but gradient chain is short and stable.
- **Cosine warm restarts**: T_0=50, T_mult=2. Multiple chances at the "breakthrough" we observe — model oscillates chaotically then suddenly drops. Warm restarts give periodic high-LR exploration to find these transitions.
- **Breakthrough dynamics**: Autoregressive training has tipping points. Bad predictions at early steps corrupt the whole rollout → incoherent gradients → chaos. Once early steps stabilize → positive feedback → rapid convergence. This explains the sudden loss drops we see.
- **Short BCL only**: AP shape is ~300ms regardless of BCL. Long BCL just adds boring diastole. Train on BCL=300-500.
- **Two-half training**: A phases (Half 1: attention+MLP+ionic decoder), B phases (Half 2: conductance compression+decoder, Half 1 frozen). Same dt curriculum for both.

### 2026-04-03: Encoder removed + first real training (B1→B2)
- **Encoder removed**: User insight — the model should discover its own latent, not have it imposed by an external encoder. The encoder was creating a mapping that the attention couldn't reproduce. Phases A1 (autoencoder) and A3 (encoder-fed conductance) eliminated. encoder.py deleted.
- **New phase order**: A2→B1→B2→B3→B4→B5→C→D→E (9 phases, was 11). B1 replaces A1 — attention+MLP+decoder co-discover latent from (zeros, Vm).
- **Gradient separation proven**: Per-dim attention structure means ionic_loss only backprops through W_q rows 0-15, conc_loss only through rows 16-19. W_k/W_v shared (same Vm physics). No gradient masking needed — architecture handles it naturally.
- **B1 trained** (rollout=1, T1, batch=4096, 30 epochs): val=0.56 combined (ionic+conc). Model learns voltage-dependent equilibrium from zeros.
- **B2 trained** (rollout=10, T1, batch=4096, 30 epochs): val=1.68. Early instability (val spiked to 41K at epoch 0) from autoregressive error compounding. Stabilized by epoch 16. First time model handles its own errors.
- **Decoder recalibration**: Froze attention+MLP, trained only decoders (283 params) at rollout=1. Val=1.14 in 13 epochs. Decoders now calibrated to stable latent.
- **Per-component loss logging**: ionic_state_mse, conc_mse, conductance_mse, grad_norm tracked separately. HDF5 training log with 14-column epoch schema on HDD.
- **Concentration dominates loss**: K_i~138 mM vs gates~0.5. Raw unnormalized MSE means conc MSE >> ionic MSE. May need normalization later.

### 2026-04-02: Training pipeline implementation (overnight session)
- **PLAN.md**: 6 phases, 13 steps. 2 audit rounds (20→10→~0 critical issues). All phases implemented + committed.
- **Data cache**: CacheBuilder preprocesses T1-T3+T12 from HDD → SSD .pt files. 24 GB total (T12 larger than expected: 21 protocols × 2M steps). 3 minutes to build from HDD.
- **HDD speed measured**: 7 MB/s direct I/O, 246 MB/s buffered (USB 3.0 WD Elements). Old 1.26 MB/s figure was wrong. HARDWARE_CONSTRAINTS.md updated.
- **Training verified**: A1 converged (val_recon_mse=7.9e-5, 3 epochs). A2 slow (val_conc_mse=0.029 after 10 epochs — needs more epochs). A3 good (val_cond_mse=1.78e-4, 10 epochs).
- **Files created**: 8 training modules (data_cache, datasets, encoder, phases, rollout, trainer, checkpoint, monitor, metrics, shard_loader), train.py CLI, training-monitor agent definition. 51 training tests.
- **Key fix**: model must be .double() (float64 project convention) — caught at first real training run.

### 2026-04-02: Pre-blueprint decisions
- **Data format**: Keep raw HDF5 on HDD as-is. Preprocess T1-T3 → `.pt` cache on SSD (~5.5 GB float32). T4 shard-streamed from HDD with double-buffering.
- **SSD freed**: Deleted unused venv (7.7 GB), purged pip cache (11 GB), conda cache (2.9 GB), removed tf-gpu-test env (6.2 GB). SSD now 47 GB free (was 20 GB).
- **Agentic training oversight**: Project-based Claude agent. Reads training_control.json + training_log.jsonl + phase_summary.json. Can autonomously: pause, reduce LR, transition phase, rollback checkpoint. Must ask user for: skip phase, adjust batch size, abort. Agent is temporary — discarded after training pipeline validated. Full LLM reasoning on logs, not just heuristic thresholds.

### 2026-04-02 Session 19-20: Implementation completion + training planning

**Implementation**: All 3 PLAN phases done. Phase 2: V3Preprocessor (47-col→named tensors, Nernst, conductance products). Phase 3: v2 archived, exports updated. 51/51 tests.

**Code refactoring**: Extracted VoltageAttention and ConductanceAttention as nn.Modules. nn.Sequential for MLPs. einsum for attention ops. Renamed: markov_mlp→ionic_mixing_mlp, comp_nonlinear→gate_conductance_mlp, w_alpha→ionic_mixing_logit, w_beta→gate_conductance_logit, full_decoder→ionic_state_decoder, comp_decoder→gate_conductance_decoder.

**Scaffold redesign**: ionic_state_decoder(16→14): 12 HH gates + RR + CaSR. gate_conductance_decoder(8→5): G_Na(m³hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1·Xr2), G_Ks(Xs²). No sigmoid on decoders (Ca_SR is unbounded). Linear only — weak decoder forces strong latent.

**Training strategy settled** (single loss per phase, no weighting):
- A1: Ionic autoencoder (14↔16). A2: Concentration attention training. A3: Gate conductance projection.
- B1-B5: Dynamics with rollout curriculum (1→10→100→1000→10000 steps). Scheduled sampling.
- C: Concentration dynamics. D: Stage 2 regression (frozen Stage 1). E: End-to-end.
- Data curriculum: T1 only (A,B1-B2) → +T2 (B3) → +T3 (B4) → +T4 (D,E) → T5-T12 after E works.
- T12 (celltypes) enters at Phase B.
- Initialization: zeros for ionic latent (model discovers own representation), real resting values for concentrations (Layer 0 physics). Not from TTP06 encoder output — avoids imprinting model assumptions.

**Docs**: ARCHITECTURE_v3.md polished (Markov→cross-dimensional, compression→gate conductance projection, alpha/beta→logit names, 15→14). Explainer + Mermaid diagram created.

### 2026-04-02 — Compression input change + architecture doc
**Decision**: Compression takes full carried_state (36→16) instead of ionic_state only (32→16). Gives compression access to concentration context (Ca_ss for fCass-dependent conductances). +80 params (1416 vs 1336 inference). Code + tests updated, 43/43 pass.
**Rationale**: Compression recomputes every step because attention structurally cannot compute cross-dim products (m³·h·j). The compression MLP has 2 GELU layers specifically for this. Carrying conductance_latent forward (inferring instead of recomputing) was considered and rejected — attention can't maintain accurate gate products.
**Also**: ARCHITECTURE_v3.md written (Nature-style, 304 lines, audited 0 critical). Python TikZ generator built (generate_v3_diagram.py). ORd T1 EPI complete (9/9 protocols, 12GB).

### 2026-04-01 Session 17-18
**Worked on**: v3 model implementation (PLAN.md Phase 1) + ORd data generation planning
**Accomplished**: Implemented all 4 v3 model files (nernst.py, stage1.py, stage2.py, ionic_surrogate_v3.py). 75/75 tests passing (43 model + 32 data gen). Blueprint with 3 phases, audited 3 rounds (17→8→3 issues, final: 0 critical). Stage 2 input normalization identified as must-fix (Ca_i vs K_i: 6 orders of magnitude). ORd data gen plan drafted (3 phases, audited: 3 critical about TTP06-hardcoded code). TTP06 column mapping confirmed: [Ki(0),Nai(1),Cai(2),CaSR(3),CaSS(4),m(5)..RR(17)]. Scaffold targets: 12 HH gates (not 13, not 18). N_GATES corrected from 13→12 (RR excluded).
**Next**: PLAN.md Phase 2 (preprocessor) + Phase 3 (v2 cleanup) → first training run → ORd plan fixes → ORd data generation

### 2026-04-01 — Session 17: Training viability + multi-model conditioning

**Trainability concern**: v3 full architecture is ~4950 inference params for a 200-param system (TTP06). 25× overparameterized. Will train (scaffolds + staged training + 608GB data), but excess capacity slows convergence. MLP 32→32 (2112 params) is entirely wasted on TTP06 (no Markov states — α learns ~0 everywhere).

**Resolution: start small, scale up.** Same architecture, smaller hyperparams for first run:
- ionic_state=16, MLP 16→16, compression 16→12→12→8, conductance_latent=8, Stage 2 queries=8
- ~1200 params total. Validate architecture works. Then scale to 32 for ORd.

**Multi-model conditioning insight**: The 32-dim architecture is large enough to learn BOTH TTP06 and ORd in one model, conditioned on a label token (model ID as extra attention input). TTP06 uses ~18 of 32 ionic dims; ORd uses all 32. Shared structure (attention, compression, readout) is the same — latent usage differs. This IS the universal ionic latent space from Session 7, realized through conditioning rather than separate models.

Benefits: richer latent for arc light fine-tuning (model has seen two "theories" of ionic dynamics), single architecture for all ionic models, natural curriculum (train TTP06 first, add ORd later).

**Training strategy update**: Phase 1 (Stage 1 isolated) → Phase 2 (Stage 2 regression, frozen Stage 1) → Phase 3 (end-to-end fine-tune). Stage 2 trains trivially in isolation (25→1 supervised regression, no temporal dependency).

**Stage 2 input normalization (must-fix):** Environment tokens have wildly different scales — K_i~138mM vs Ca_i~0.0001mM (6 orders of magnitude). Without normalization, Ca_i/Ca_ss tokens are invisible to attention (key magnitude ≈ 0). Fix: normalize all 9 environment inputs to ~[-1,1] using known physiological ranges before embedding. 9 shifts + 9 scales = 18 fixed constants from physiology, not learned params.

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
| External encoder for latent bootstrapping (A1) | Encoder imposes a latent space the model didn't discover. With rollout=1 from zeros, the model can't distinguish upstroke from repolarization at same Vm — but that's the correct limitation (history-dependent). The model should discover its own representation through dynamics training, not have it dictated by an encoder. Removed 2026-04-03. |
| Separate A2 phase for concentration attention | Unnecessary — concentration loss is naturally separated in the per-dim attention structure. Combined ionic+conc loss in B1 trains both paths simultaneously without gradient interference. W_k/W_v shared (same Vm), W_q/W_out rows naturally separated. |
| Teacher forcing via encoder in Phase B | Without an encoder, there's nothing to teacher-force with. Purely autoregressive from step 1. The contractive attention mechanism provides natural error correction (gate≈1 when far from target). |
| Batch=32768 for GPU utilization | Too few optimizer steps per epoch (49 batches for rollout=10). Model barely learns. LR scaling (8x) to compensate causes explosions. Batch=4096 is the sweet spot — enough steps, stable gradients. |
| Variance normalization (MSE/Var) | Treats all dims within a component equally. K_i (var=0.018) still drowns Ca_i (var=1e-9) within conc_mse. Per-dim min-max is better — every dim mapped to [0,1] independently. |
| Standalone decoder recalibration (B2_decoder) | Decoder trained on frozen latent becomes stale when latent shifts in next phase. Always co-train encoder and decoder. |
| Rollout curriculum at fixed dt=0.01ms (1→10→100→1K→10K) | Slow and expensive. dt curriculum (3ms→1ms→0.1ms→0.01ms at fixed 300ms coverage) achieves same temporal coverage much cheaper. Rollout=100 at dt=3ms sees the same AP as rollout=30K at dt=0.01ms. |
| Discrete autoregressive rollout at native dt (0.01ms, 30K steps) | Fundamentally limited by error compounding. Tried: dt curriculum (A1-A3 worked but A4 stuck at val~720), TBPTT(500), warm restarts, min-max normalization, batch tuning. The 30K-step gradient chain is too long even with truncation. Error at step 100 corrupts steps 101-30000 → incoherent gradients. Not a hyperparameter problem — it's a structural limitation of discrete autoregressive training for stiff multi-timescale systems. |

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
| 2026-03-30–31 | 15 | Architecture v3 redesign from Layer 0. Carried state = 36 (32 ionic + 4 explicit conc). Stage 1: attention(36)→split→ionic through MLP(32→32)+α mixing→compression(32→16)+β mixing; conc attention-only. Nernst at end of Stage 1. Learned residual mixing replaces spectral norm. Chebyshev dropped. Ca_SR dropped. |
| 2026-03-31–04-01 | 16-18 | Stage 2 design + v3 implementation. Bilinear→ψ factorization→Ohmic split→all scrapped→cross-attention settled. Blueprint 3 phases, audited 3 rounds. Implemented: nernst.py, stage1.py, stage2.py, ionic_surrogate_v3.py. 75/75 tests. N_GATES corrected 13→12. ORd plan drafted+audited. TTP06 column mapping confirmed. Multi-model conditioning planned. |

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

**Next**: Settle Stage 2 and Stage 3 redesign, then update model code + tests, then Phase A training blueprint.

### 2026-03-30–31 — Session 15-16: Architecture v3 from Layer 0

**Layer 0 reasoning framework**: Established three-layer priority: physical reality (Layer 0) → biophysics models (Layer 1) → architecture (Layer 2). Future arc light imaging goal means architecture must not be locked to TTP06/ORd assumptions.

**3-stage → 2-stage pivot**: Mapped physical tasks to architecture. Cross-state coupling within one dt is negligible (all coupling is temporal, accumulates over steps). Tasks A (state evolution) and C (current readout) are parallel reads of old latent. Old Stage 2 (split GELU coupling) eliminated. Critical path insight: only readout needs to be fast; Stage 1 hides behind diffusion.

**Stage 1 settled**: n×1 attention (attn_dim=4) + MLP (32→32→32) with learned α mixing (DeepSeek-inspired convex combination). Concentrations (4 explicit dims: Na_i, K_i, Ca_i, Ca_ss — Ca_SR dropped) split off AFTER attention, before MLP. Compression: 32→24→24→16 with learned β mixing. Nernst at end of pipeline. Pre-RMSNorm before MLP. GELU everywhere + residual = linear bypass.

**Key design decisions with reasoning**:
- Markov coupling: n×1 attn + MLP sufficient (splitting error O(dt²) ≈ 10⁻⁸). Full self-attention unnecessary.
- MLP geometry: 32→32 (no bottleneck). ALLOWS coupling, doesn't FORCE it. HH dims pass through via residual.
- Learned α replaces spectral norm + zero-init + gate-modulation. Convex combination = no amplification by construction.
- Concentrations: attention-only (no MLP). Slow Vm-dependent tracking. Self-regulation through own-value gating.
- Compression: 2 GELU layers for triple product composition (m·h in layer 1, m·h·j in layer 2).

**Stage 2 readout exploration**: Derived bilinear form from TTP06 current equations (conductance × driving force). Explored ψ factorization (384 FLOP matmul off critical path → 8 FLOPs on critical path via Horner's method). Proposed Ohmic/non-Ohmic split then SCRAPPED (Ohm's law is a model assumption — rectification, voltage-dependent conductance violate it). Surveyed 8 ML architectures. **Settled on cross-attention**: 16 conductance queries attend to 9 environment tokens [Vm, 4 E, 4 conc]. Output 16→1 linear (Kirchhoff). Budget generous: even MLP h=32 is only 1× ORd step.

**Current architecture (v3, updated dims):**
```
carried_state(t) = [ionic_state(32), concentrations(4)] = (36,)

  ├→ Stage 1 (off critical path):
  │    carried_state(t) → attention(Vm,dt) over 36 dims
  │    → SPLIT: ionic(32) + conc(4)
  │    → ionic: Pre-RMSNorm → MLP(32→32→32) → α mixing(32 params) → ionic(t+1)
  │    → conc: pass through directly → conc(t+1)
  │    → RECOMBINE: carried_state(t+1) (36,)
  │    → ionic(t+1) → compression → conductance_latent(t+1) (16,)
  │    → conc(t+1) → Nernst → reversal_potentials(t+1) (4,)
  │
  └→ Stage 2 (ON critical path):
       conductance_latent(t) as queries, [Vm, E(t), conc(t)] as keys/values → cross-attention → 16→1 → I_ion(t)
```

**Failed approaches (this session):**
- Softmax for cross-channel gating: conservation constraint doesn't match independent gate biology
- Dedicated cross-coupling stage (old Stage 2): cross-state coupling is temporal not instantaneous, no physical basis for a per-step coupling layer
- Full self-attention for Markov (Proposal A): more elegant but splitting error at dt=0.01ms is negligible, not physically justified over simpler MLP
- Ohmic/non-Ohmic split readout: Ohmic behavior is a Layer 1 model assumption, not Layer 0 ground truth. Rectification, voltage-dependent conductance, surface charge effects all violate Ohm's law. Scrapped to avoid baking in model assumptions.
- Bilinear readout with hand-crafted features: [1,Vm,Vm²,Vm³,E,conc] feature vector is arbitrary. ψ factorization was clever engineering but created an overly complex pipeline. Scrapped in favor of cross-attention which learns the routing naturally.
- Concentration decoder from ionic state: replaced by explicit concentration dims in carried_state. No decoder that can go wrong.
- E as separate branch: folded into Stage 1 pipeline (Nernst at end). Two branches, not three.

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
