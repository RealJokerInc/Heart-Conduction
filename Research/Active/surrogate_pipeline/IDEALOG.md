# Surrogate Pipeline — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
**Hybrid bidomain surrogate pivot (Session 29 late, 2026-04-21).** A benchmark of TTP06 compiled on GPU (n=10k: 34.1 M cell-steps/s) vs the v4 ionic surrogate Euler path (4.2 M cs/s) showed the ionic surrogate is 8× *slower* than the classical solver at tissue scale. KNOWLEDGE §1 had already noted that 94% of bidomain wall time is the elliptic solve, not the ionic step — the ionic surrogate was never the GPU speedup lever. Redirecting: keep classical TTP06 as the "ionic scaffold," build a neural surrogate for the bidomain **elliptic step** instead. Chosen architecture direction: dual CNN towers (one for Vm / intracellular domain, one for φ_e / extracellular) with cross-communication (Transformer attention or 1×1 cross-conv). Parabolic-elliptic first (v1); hyperbolic bidomain deferred to future Phase B.

**Deprioritized from critical path:** v4 ionic overfitting fix, Data v2 single-cell T1-T12 regen, Stage 2 conductance attention, `SegmentDataset` rest-start rewrite. Ionic v4 keeps secondary roles only (CPU deployment 3-7× win confirmed, differentiable coupling, parameter optimization).

## Next Step

**Design decisions outstanding for the dual-tower elliptic surrogate:**
1. Cross-talk mechanism — full attention / windowed (Swin) / linear / Perceiver bottleneck tokens / 1×1 cross-conv / FiLM / bottleneck-only attention. Default lean: 1×1 cross-conv at every level + full self-attention at V-cycle bottleneck only (compute-friendly).
2. Tower depth — V-cycle UGrid-style per tower vs plain encoder-decoder U-Net.
3. Output semantics — full-field (Vm_{n+1}, φ_e_{n+1}), residual-to-classical (one Jacobi sweep + NN correction), or learned preconditioner for PCG (preserves convergence guarantee, safest).
4. Input conditioning — each tower sees anisotropic D tensor fields (D_i, D_e), stimulus mask, boundary mask.

**Data pipeline (new):**
- Generate tissue-scale (Vm, φ_e) field trajectories from Bidomain V1 (the ground-truth simulator). Not T1-T12 single-cell data.
- Define eval metrics: CV error (< 5% target), APD90 error, Kleber boundary ratio, elliptic residual norm, PCG iterations saved (if preconditioner path). Anchor CV/APD to Niederer 2011 N-version benchmark.

**Literature to pull into `literature/`:**
- UGrid (Li 2024, arXiv 2408.04846) — CNN V-cycle neural multigrid, most directly adoptable.
- NPO (Cai 2025, arXiv 2502.01337) — neural preconditioner for Krylov, safer adoption path.
- Ziarelli 2025 (arXiv 2512.13765) — Vm→ECG forward, confirms no one co-solves φ_e as a field.
- Salvador 2025 CMAME (arXiv 2504.20479) — branched LNM for geometric variability (future Phase B).

**Ionic side (paused, not abandoned):**
- v4 overfitting fix (Session 28 options A/B/C) — only revisit if CPU ionic deployment story becomes a deliverable.
- Data v2 T1 generation partial (batch 1 of 5 on disk at `/media/shared/norepinephrine/surrogate_data_v2/raw/tier01_epi.h5`) — preserved, not deleted. Can resume if needed.
- INIT_CONC fix in `node_rollout.py:21` — still a 1-line correctness fix regardless of direction; do it when next touching that file.

### 2026-04-21 (Session 29 late): TTP06 benchmark + hybrid bidomain pivot

**Trigger**: ran a TTP06-vs-surrogate inference benchmark (`Surrogate/benchmarks/speed_ttp06_vs_surrogate.py`, results in `benchmarks/results/`).

**Numbers (dt=0.01 ms, 30k-step benchmark, Blackwell RTX PRO 4500):**
- GPU (torch.compile on):
  - n=10:   TTP06 65k cs/s  vs surrogate 47k cs/s  → 0.72× (surrogate slower)
  - n=100:  TTP06 653k      vs surrogate 473k     → 0.72×
  - n=1k:   TTP06 6.5M      vs surrogate 2.6M     → 0.40×
  - n=10k:  TTP06 34.1M     vs surrogate 4.2M     → **0.12× (surrogate 8× slower)**
- CPU (no compile): surrogate wins 3.4–7.4× across n=10..1000.

**Read**: TTP06 `model.step` fuses beautifully under `torch.compile` (~30 kernels into a few); per-step cost stays flat at ~150 μs up to n=1000. Our v4 `StateRateMLP + dzdt + Euler` has branching structure (sigmoid skip-blend, input_ref subtraction, Euler-add not compiled with dzdt), preventing fusion. At n=10k the MLP compute saturates and per-step cost explodes to 2356 μs. Implementation-level issue, not fundamental — but unless we spend engineering on CUDA-graphs / kernel fusion / CfC-style batched update, the gap persists.

**KNOWLEDGE §1 already flagged** that bidomain elliptic = 94% wall-time, ionic ~6%. The benchmark confirms the project was attacking the wrong bottleneck with the ionic surrogate. Correction: the surrogate's real value proposition lives in the diffusion/elliptic component, not in replacing TTP06.

**Research direction pivot (settled)**: hybrid bidomain surrogate.
- **Ionic step**: classical TTP06 (compiled, already near kernel-launch limit).
- **Diffusion/elliptic step**: neural surrogate (this is where the speedup lives).
- **Parabolic-elliptic first, hyperbolic deferred.** Hyperbolic bidomain (second-time-derivative, Maxwell-Cattaneo / Feng-Bova form) makes classical solvers even harder (CFL-tight, no elliptic shortcut) — attractive target for NN but requires (a) a hyperbolic simulator we don't have, (b) TTP06 reformulation, (c) new data pipeline. Parabolic-elliptic first to validate the architecture.

**Architecture direction**: dual CNN towers (INTRA for Vm / intracellular domain, EXTRA for φ_e / extracellular) with cross-communication. Physics-native — matches bidomain's two-domain structure. Cross-talk mechanism open: Transformer attention (full / windowed / linear / Perceiver bottleneck) or lightweight 1×1 cross-conv / FiLM. Strong lean: 1×1 cross-conv at every level + full self-attention at V-cycle bottleneck only (memory-feasible on 33 GB Blackwell).

**Literature gap analysis (delegated to general-purpose Agent, 2026-04-21)**:
- No published cardiac NN surrogate co-solves φ_e as a field. All 5 papers in our `literature/` are monodomain whole-PDE replacements.
- Mature learned-Poisson literature exists in CFD context (Greenfeld 2019, Hsieh 2019, Ozbay 2019, UGrid 2024, NPO 2025, neural multigrid) for the analogous 80%-of-wall-time pressure-projection bottleneck. **Zero cardiac adoption.**
- "Hybrid split-step" has no standard name in cardiac. Subagent suggested "learned elliptic sub-operator" to disambiguate from other senses of "hybrid."
- All learned-Poisson work is isotropic Laplacian; anisotropic tensor D (bidomain requires `∇·((D_i+D_e)∇φ_e)`) is unhandled anywhere.
- FNO is Neumann-BC-weak; bidomain matches CNN/U-Net/multigrid better.
- No simulation-surrogate benchmark suite exists — we'd define our own (CV error, APD90, Kleber boundary ratio, elliptic residual norm, PCG iterations saved if preconditioner path). Anchor CV/APD to Niederer 2011 N-version benchmark.

**Recommended output semantic (subagent): preconditioner first, not full inverse.** In a bidomain rollout, φ_e feeds back into the parabolic RHS every step. A learned preconditioner (NPO-style) preserves PCG's convergence guarantee — errors fail gracefully. A learned inverse (UGrid-style) is faster but lossy; phi_e errors compound into Vm errors into phi_e errors. Preconditioner is the safer v1.

**4 new papers to pull into `literature/`**:
- UGrid (Li 2024, arXiv 2408.04846) — CNN V-cycle neural multigrid, most adoptable architecture.
- NPO (Cai 2025, arXiv 2502.01337) — neural preconditioner for Krylov, condition+residual loss.
- Ziarelli 2025 (arXiv 2512.13765) — Vm→ECG forward surrogate, seq2seq attention, confirms field ignores φ_e.
- Salvador 2025 CMAME (arXiv 2504.20479) — branched LNM on biventricular, cited for "surrogates are geometry-specific" — relevant if Phase B extends past structured grid.

**Session-29-late artifacts**:
- `Surrogate/benchmarks/speed_ttp06_vs_surrogate.py` + `benchmarks/results/{gpu,cpu}.{json,log}` — the benchmark evidence.
- `Research/Active/surrogate_pipeline/WHITEBOARD.md` — current sketch is the dual-tower design.
- **Folder cleanup**: `Surrogate/` root now holds only `README.md` + 7 organized dirs (`surrogate/`, `datagen/`, `benchmarks/`, `diagnostics/`, `tests/`, `docs/`, `archive/`). 9 obsolete root scripts → `archive/scripts/`. 11 stale root markdown → `docs/`. `improvement.md` → `docs/`. `run_multi_bcl.py` → `benchmarks/` (kept; Session 25 parity oracle per MEMORY.md).
- Data v2 T1 partial gen on disk (batch 1 of 5 only, ~3.5 GB) — preserved, not deleted. Spec (`DATA_V2_SPEC.md`) + generator (`datagen/generate_t1_v2.py`) + schema module (`surrogate/data/schema.py`) all ready if we resume the ionic-side track.

**Failed approaches added this session**:
- **Chasing per-step ionic speed parity with TTP06 on GPU** — benchmark confirmed TTP06 compiled is 8× faster at tissue scale. Stop optimizing the MLP for kernel-launch efficiency; the gap is structural (branching + separate dzdt+Euler calls prevent fusion).
- **"Surrogate replaces the whole bidomain solver" framing** — the ionic half is already at kernel-launch limits on GPU. Only the elliptic half is the real bottleneck.
- **Full Data v2 regen now** — paused. Single-cell ionic data is only useful for ionic-side work, which is deprioritized. Partial batch 1 retained in case we resume.

### 2026-04-21 (Session 29): Data v2 audit + T1 regeneration spec

**Audit findings** (full matrix in DATA_V2_SPEC.md §Motivation):
- v1 data pipeline loads only T1/T2/T3/T12 of the 12 tiers on disk (~30 GB / 612 GB, 5% utilization). T4 (551 GB random pacing), T11 (18 GB stitched) entirely unused.
- `INIT_CONC = [Na_i=10, K_i=138, Ca_i=1e-4, Ca_ss=2e-4]` in `node_rollout.py:21` doesn't match simulator rest `[8.604, 136.890, 1.26e-4, 3.60e-4]`. `voltage_clamp_ss.py:36-40` already flagged this and uses V5.4's `_V54_REST` directly; fix never propagated to training. Creates systematic latent-rest mismatch.
- "Steady state" is BCL-dependent: at beat 20 of BCL=2000, CaSR=1.23; at BCL=300, CaSR=4.73. Frozen decoder bias assumes CaSR=3.64. The rest-attractor contract is self-inconsistent across BCLs.
- Two incompatible segment extractors in the repo: `run_multi_bcl.py` (beat-aligned, honors z0=rest) vs `SegmentDataset` (uniform stride, lands mid-AP). Harness + `train_node.py` use the broken one.
- Splits are at protocol-name granularity; with 8 DIs and 3 held out, losing one = 12.5% of the tier. No test set possible at v1 density.

**T1 v2 locked**:
- Grid: 35 BCLs = {200..300 @ 10ms} ∪ {350..1000 @ 50ms} ∪ {1100..2000 @ 100ms}.
- 50 beats/BCL (up from 20) — beats 15–49 are settled and usable. Extra beats capture slow CaSR/Na_i equilibration (60–120 s timescale).
- Two-axis split: across-BCL (23 train / 6 val / 6 test, regime-stratified) × within-BCL (beats 0–14 warmup / 15–39 train / 40–44 val / 45–49 test on train BCLs).
- Val BCLs: {220, 280, 450, 900, 1200, 1800}. Test BCLs: {240, 260, 550, 850, 1500, 1900}.
- Effective: 575 train + 325 val + 325 test segments (~20× the Session 25 oracle's 25).

**Design decisions baked into spec** (DATA_V2_SPEC.md):
1. Per-(tier, celltype) h5 files. `tier01_epi.h5`, `tier01_endo.h5`.
2. Short group names (`bcl200`, not `steady_bcl200_dt0.01`). Redundant axes live in file/group attrs.
3. Column schema promoted from `preprocessor.py` constants to file-level `column_names`/`column_units` attrs + single source of truth in new `surrogate/data/schema.py`.
4. Splits as editable sidecar JSON in `splits/tier{NN}_v2.json`. Loader reads sidecar, not h5, for split decisions.
5. Gzip-4 compression, chunks (65536, 47) — ~60% size reduction.
6. Quality flags (`capture_flag`, `alternans_flag`) computed at gen time.
7. Per-tier provenance log in `provenance/{tier}_genlog.json`.
8. Top-level `MANIFEST.yaml` idempotently regenerated on each tier completion.

**Rejected during audit** (see DATA_V2_SPEC.md alternatives):
- Pushing T1 past 35 BCLs — diminishing returns; BCL is a continuous scalar the model interpolates.
- Reorganizing by semantic class (`rest_start_steady`, etc.) now — deferred until we know all tier designs.
- Baking splits into h5 attrs — sidecar JSON wins on editability + version control.

**Open** (blocking nothing immediate):
- T2 DI grid spec after T1 generates.
- Rewrite `SegmentDataset` to read v2 schema + honor rest-start contract (Session 29+).
- Fix INIT_CONC in `node_rollout.py` (1-line; applies to both v1 and v2 training).

### 2026-04-20 (quicksave): Session 28 post-mortem + harness issues + training-regime pivot

**Overfitting framing**: earlier I called the Session 28 train/val gap a "BCL distribution" problem. Wrong — user pointed out v3 (1,444 params) converged on the same split. The 5.5× capacity bump (7,891 params) on 25 trajectories (5 BCLs × 5 beats) is memorising the training set. Classic fingerprint confirmed: smooth monotone train-loss descent, oscillating val-loss. IDEALOG / KNOWLEDGE / MEMORY entries corrected.

**cardiac_ml harness issues surfaced during Session 28** (ordered by severity):
1. **(fixed)** Factory didn't re-pin rest bias after `load_state_dict` — Step 3.0 patch added `pin_rest_bias()` call. Warm-starts before this would have silently broken the rest-attractor invariant.
2. **Checkpoint format asymmetry** — `ModelCheckpoint` saves `trainer.model.state_dict()` *flat* (`stage1.*` prefix), but the factory's `load_state_dict` path expects the v3-era `{"stage1_state_dict": ...}` wrapper. Checkpoints can't round-trip through the factory. Diagnostic has dual-format loader; a future warm-start-from-harness-ckpt would hit this.
3. **No NFE callback** — `GradNormMonitor` exists, no counter for `IonicNODE.nfe`. When training slowed after epoch 8 we couldn't confirm the "NFE ramp from stiff vector field" hypothesis. Fixable with a 10-line callback.
4. `train.log` left empty (only MLflow is written) — tail-a-run workflow is broken; use `tail mlruns/*/metrics/val_loss` instead.
5. CLI syntax trap: `experiment=ionic_node_smoke` not `+experiment=ionic_node_smoke` (the `+` is for NEW keys; experiment is already in defaults).

**Decision**: **bypass the harness for now**. Use `Surrogate/surrogate/training/train_node.py` (argparse) directly for remaining v4 training. Rationale: the overfitting-fix loop needs fast iteration across architecture shrinks + regularizer tweaks, and the harness adds 3–4 min of Hydra / MLflow overhead per run for no benefit when we're not sweeping. `train_node.py` is v4-compatible (uses `state_rate_mlp`, `node.nfe`, `stage1_state_dict` wrapper ckpts). Harness remains the path for sweeps / final-parity runs.

**Open question for next session**: which option to run first — shrink StateRateMLP to ~3K params (Option A, directly addresses overfitting), bump weight decay + λ_rest on current arch (Option B, keeps capacity hypothesis alive), or A→B sequence (Option C). Not yet committed.

### 2026-04-20 (Session 28): v4 implementation + first end-to-end training

**Worked on**: full PLAN execution — Phase 1 (architecture pivot), Phase 2 (rest attractor + z_ss grid), Phase 3 (first training run via cardiac_ml harness). New: input-centering hotfix that Session 27's PLAN did not anticipate.

**Implementation status (all merged to working tree):**
- Phase 1: `StateRateMLP` replacing `IonicRateMLP`+`conc_kan` (7,891 params, band [7800, 8100]), `IONIC_DIM` 16→20, `TTP06_REST_IONIC_STATE` constant, `pin_rest_bias()` method pinning decoder bias at rest. 143 Surrogate tests pass (1 xfail — legacy discrete rollout), 81 cardiac_ml tests pass. Negative grep `ionic_rate_mlp|conc_kan|MLP_HIDDEN|VoltageAttention|KANLayer` returns empty on `Surrogate/tests/`.
- Phase 2: `Surrogate/surrogate/data/voltage_clamp_ss.py` uses V5.4's `compute_gate_steady_states`, `compute_gate_time_constants`, `compute_concentration_rates` primitives (thin composition, no RHS reimplementation); Rush-Larsen for gates + Euler for CaSR/RR (dt=0.01 ms). Converges to the held-V fixed point within 2000 ms at `rel_tol=1e-4`. `z_ss_grid.pt` saved. `L_rest` regularizer added to `node_rollout` with `V_REST_MV=-85.23`, `LAMBDA_REST=1e-2`, exposed as `result['L_rest']`.
- Step 3.0: `cardiac_ml/model/ionic_node_factory.py` re-pins rest bias after `load_state_dict`. Dedicated test file `test_factory_rest_bias.py` covers cold-start + warm-start + v4 load.

**Session 28 insight — the input-centering hotfix.** First t1 launch (LR=1e-4, dopri5, adjoint=False, 30 epochs, 5 train BCLs × 5 beats, 4 val BCLs × 3 beats) produced val_loss = 1.88e9 at epoch 0 with only ~24 %/epoch reduction by epoch 2. Diagnosis: the gated full-path skip (`alpha = sigmoid(-5) ≈ 0.007`) receives the raw carried-state as input. With `INIT_CONC` containing `K_i = 138` and `Vm ≈ -85 mV`, the skip output magnitude hit ~`0.42 × 138 = 58/dim` before the `α=0.007` gate, producing ~`0.4/dim` rate contributions during dopri5 integration — enough to blow the integrated latent out of range and decoder predictions out of physiological bounds. `skip_logit = BETA_INIT` (Phase 1 hotfix) was necessary but not sufficient.

**Fix**: added a non-trainable `input_ref` buffer to `IonicStage1` (shape `(carried_dim + 1,)`). `dzdt` subtracts it before calling `state_rate_mlp`, so the skip path sees "deviation from rest input" — `[zeros(ionic_dim), INIT_CONC, V_REST_MV]` becomes exactly zero at rest. At rest with random weights, both deep and skip paths return zero → rate = 0 (verified in smoke: `L_rest ≈ 1.6e-17`, `rate.abs().max() = 0.0` at V=-85). Buffer is registered so `.double()` cascades correctly; uses `torch.get_default_dtype()` so float32 tests keep working.

**Second t1 launch** (same config, model now with `input_ref`):
- `train_loss` decreased cleanly: `6.57e4 → 5.20e4 → 3.80e4 → 2.06e4 → 1.38e4` (epochs 0–4, ~79 %/epoch reduction).
- `val_loss` oscillated in millions: `1.27e7 → 2.18e7 → 2.43e7 → 2.03e7 → 1.62e7 → 1.32e7 → 2.08e7 → 2.13e7 → 6.74e6 → 2.30e7` (epochs 0–9). Best epoch = 8 at 6.74e6.
- **Train/val gap ~1000× is overfitting, not a data-split issue.** Oracle `multi_bcl_002` (v3, 1,444 params) reached val=0.00838 on the *same* train/val split (incl. BCL=2000 in val). v4 at 5.5× the parameter count (7,891) memorises the 25-trajectory training set (5 BCLs × 5 beats each) and fails to generalise. The classic pattern is here: smooth monotone train-loss decrease and oscillating val-loss with no trend. Earlier framing in this entry that blamed "BCL distribution gap" was wrong — v3 handled the same gap fine; the difference is model capacity.
- `L_rest` term works as designed: settles to ~`5e-5` by epoch 4 — rest is a fixed point of the learned dynamics.

**Outcome**: v4 architecture and training infrastructure are functional; Success Criterion `val_loss < 0.02` is NOT met within the 30-epoch / 1e-4 LR budget. Criterion was written against the Session 25 oracle (v3, 1,444 params, 8 epochs, val=0.00838); v4's 5.5× larger capacity needs either more epochs, warm-start, or narrower training distribution to converge to that threshold. Training was killed at epoch 10 after noticeable NFE slowdown; `best.pt` corresponds to epoch 8 (val=6.74e6).

**Integrator error-budget on `best.pt`** (held-out BCL=2000, 300 ms window, `rtol=atol=1e-8` dopri5 reference): `Surrogate/diagnostics/artifacts/integrator_error_budget_v4.pt`.
- Aggregate Euler vs dopri5 = 0.0008 (truncation), dopri5 vs truth = **9.76** (capacity), Euler vs truth = 9.76.
- **CaSR NRMSE = 37.11 %** (v3 baseline 27.4 %). v4 at this checkpoint is *worse* than v3. Several fast gates are also in the 200–2100 % NRMSE regime — predictions are multiples of the physiological range out. This reflects the latent-explosion-during-long-integration pattern already diagnosed by the millions-scale val_loss, not a fundamental architectural flaw: the model is undertrained under the current regime.
- Time-resolved error grows monotonically: 0.51 → 0.79 → 1.88 → 7.34 → 12.14 → 14.28 at t = 2.5, 7.5, 22.5, 102.5, 202.5, 297.5 ms. Error accumulates across the integration horizon.
- Conclusion: the v4 CaSR-capacity improvement hypothesized in Session 27 is not yet validated empirically. Running the full 30-epoch budget (or a warm-start / curriculum regime) before re-measuring is the next actionable step.

**Settled next steps (to be opened as a new research thread), in rough priority:**
1. **Shrink StateRateMLP back toward v3 scale.** Current capacity budget (7,891 params) is the dominant driver of the overfitting; H_STATE_MLP 32→16 and 5→3 hidden layers brings the count back to ~2.5–3K, within spitting distance of v3's 1,444. The Session 27 blueprint's capacity bump was a hypothesis, not an empirical requirement — the CaSR bottleneck identified in the diagnostic motivated it, but until we re-measure CaSR NRMSE on a *converged* v4 checkpoint we can't validate that extra capacity closes that specific gap. Smaller-v4 first so we have a trained baseline, then widen if CaSR stays stuck.
2. **More data**, not a data-split change. T2/T3 caches already exist in `/media/HDD/norepinephrine/surrogate_data/raw/`; rebuild the cache pipeline to include them and the effective trajectory count per epoch goes up 3–12×. Orthogonal to (1) and complementary.
3. Weight decay bump (1e-4 → 1e-3) as a cheap regulariser. Dropout is out (banned in PLAN / Failed Approaches — corrupts the vector field during adjoint).
4. Curriculum over BCLs (short first, warm-start into long) is still on the table but deferred behind (1) because overfitting, not distribution, is the primary issue.
5. Investigate whether `ode_rtol=1e-3` is stable enough at epochs 15+ — NFE may be climbing as the vector field stiffens. Orthogonal to capacity.

**Failed approaches added this session**:
- **Zero-centered raw inputs through the skip path** — magnitude spikes (`K_i=138`, `Vm=±80`) propagate un-gated during dopri5 integration. Fixed by `input_ref` buffer (non-trainable centering).
- **v4 at 7,891 params on T1 only** (2026-04-20) — overfits 25 training trajectories. v3 (1,444 params) reached val=0.00838 on the same split. Fix: shrink the model toward v3 scale before claiming v4 capacity is needed.
- **cardiac_ml harness checkpoints being compatible with the factory's warm-start loader** (2026-04-20) — `ModelCheckpoint` writes flat `stage1.*` keys, factory expects `{"stage1_state_dict": ...}` wrapper. Round-trip broken; warm-start-from-harness-best.pt would KeyError.

### 2026-04-19 (Session 27): Integrator error budget + StateRateMLP arch pivot

**Worked on**: (1) Quantifying whether inference-time Euler error buildup justifies a learned integrator redesign. (2) Root-cause of model underfitting, per dim. (3) Unified rate-predictor architecture to replace `IonicRateMLP` + `conc_kan`.

**Questions raised and resolved**:

**Q1. Does forward-Euler at dt=0.01ms accumulate enough truncation error to justify a learned integrator head `g_φ(rate, V_t, V_tdt, dt) → Δz` trained against dopri5 single-step flow?**

*Resolution: No.* Diagnostic `Surrogate/diagnostics/integrator_error_budget.py` on held-out BCL=2000, 300 ms window, checkpoint `multi_bcl_002/best.pt` (val=0.0084):
- Euler vs dopri5 RMSE: **0.00161** (truncation)
- dopri5 vs truth RMSE: **0.34638** (model capacity)
- Euler vs truth RMSE: **0.34737** (total)
- Truncation is **215× smaller** than capacity error; ratio is **700×** at the upstroke (t=2.5 ms) and never drops below 150× across the trajectory. A learned integrator head would eliminate ~0.16 % of total error and leave 99.8 % untouched. Rejected as no-op. See Failed Approaches.

**Q2. Which dims dominate the model failure?**

*Resolution: CaSR (slow variable) dominates; RR/Xs also lag.* Per-dim NRMSE normalized to physiological range (`_RANGES` in `loss_normalization.py`):
- Fast gates (m, h, j, r, s, d, f, f2, fCass, Xr1): 4–14 % per dim
- Xr2: 4.2 % (best)
- Xs: 7.4 %
- RR: 10.8 %
- **CaSR: 27.4 % — contributes 0.075 to normalized MSE, 4–10× any other dim.**
- Pattern: fast gates converge to the ~10 % regime uniformly; slow variables (CaSR, RR, Xs) lag. Suggests a latent-allocation / rate-capacity issue on slow-timescale dynamics, not a uniform undercapacity.

**Q3. Is the linear decoder the CaSR bottleneck?**

*Resolution: No.* A free `Linear(16 → 14)` has **linear universality**: the 16-dim latent's linear span can contain any 14 target directions; `W` is a free rotation. CaSR failure is upstream — the latent doesn't encode a CaSR-tracking direction because the rate predictor can't produce it.

*Refinement of prior claim:* "linear decoder ensures the latent encodes info similar to the parameters" = the latent must **linearly span** the 14 observables, **not** that `z[k] = param[k]` (identity alignment). Identity alignment would require constraining W (diagonal, or identity-init + frozen, or orthogonality regularizer). The scaffold-discard contract (no nonlinearity to launder knowledge into) still holds — this is a clarification, not a reversal.

**Q4. Should the rate predictor take conc self-input back?**

*Resolution: Yes.* Current `conc_kan(17 → 4)` feeds only ionic latent + Vm; conc self-input was dropped because an earlier `conc_mlp(20 → 4)` blew up (Xavier init → 1e237 in ODE feedback loop). That IDEALOG fix bundled *two* changes: (a) drop conc self-input and (b) switch to no-cross-talk KAN. Only (b) was strictly required. Zero-init on the last layer plus separable structure makes conc self-input stable. Physics needs it: `I_NaCa` depends on Na_i × Ca_i; Nernst potentials depend on conc; `I_NaK` is Na_i-saturating. Restore the 4 self-dims to the rate input.

**Q5. Is a single KAN layer sufficient for conc dynamics?**

*Resolution: No.* Single KAN layer is **additively universal** (each output is `Σⱼ φⱼ(xⱼ)`) but cannot represent multiplicative cross-terms. Conc physics has unavoidable products:
- `I_NaCa ∝ exp(γVF/RT) · Na_i³ · Ca_o − (reverse) · Ca_i`  → V × Na_i × Ca_i product
- `I_CaL flux = d·f·f2·fCass · Ca_ss · exp(VF/RT)`           → gate products × conc × exp(V)
- `I_NaK`                                                     → saturating Na_i × Vm interaction

None of these decompose into univariate sums. Need multiplicative capacity: stacked KAN, KAN + MLP hybrid, or a GELU mixer feeding a KAN readout.

**Q6. Separate rate paths for ionic vs conc, or unified?**

*Resolution: Unified.* Path separation was scar tissue from the MLP feedback-loop explosion — fixed by zero-init + no-cross-talk, not by path separation. Physical coupling (Ca_i ↔ Na_i via NCX) is real; the model should be allowed to learn it. Kill `conc_kan`, extend the rate predictor to full-state input and full-state output.

**Q7. Depth/width of the rate predictor and activation structure at output?**

*Resolution: `Linear(25 → 32) + GELU → KAN(32 → 24, grid=5, order=3, spline zero-init)` — single hidden mixer + KAN readout head (Option B, simplified).*
- Cheaper than Option A (KAN as hidden `32 → 32` block): saves ~3K params because output dim is 24, not 32, so the KAN matrix is smaller.
- Simpler than an earlier β proposal (3 hidden Linear+GELU layers before the KAN): a single GELU mixer trusts the KAN readout's spline capacity (~6.9K params) to handle per-dim rate shaping. If slow vars still underfit, revisit depth.
- Zero-init on `spline_weight` + near-zero `base_weight` (KAN default) → rate field ≈ 0 at init → ODE stable at init (same trick as before, now at the readout).
- GELU mixer produces 32 learned features from the 25-dim input — sufficient for the ~20–30 distinct physical cross-product interactions in TTP06.

**Q8. Is 16-dim ionic latent enough, or expand?**

*Resolution: Expand to 20.* With 16 latent dims for 14 targets there is no slack — the model has no incentive to reserve a CaSR-tracking slot when 15 other dims fit easily and redundancy is free. +4 dims adds trivial param cost (+80 MLP input, +80 decoder output) and gives explicit slack for slow variables to claim a dedicated direction.

**Final design (StateRateMLP + latent 20, Option B-simplified)**:

```
CONSTANTS
  IONIC_DIM        = 20   (was 16)
  CONC_DIM         =  4
  CARRIED_DIM      = 24   (was 20)

STAGE 1 · dzdt (unified StateRateMLP)          Input: z(24) + Vm(1) = 25,  Output: 24
  h    = GELU(Linear(25 → 32))                                            832 params
  rate = KAN(32 → 24, grid=5, order=3, spline zero-init)                 6912 params
                                                                dzdt total: 7744

Downstream (updated dims, same structure):
  ionic_state_decoder:      Linear(20 → 14)    free W, no activation      294
  gate_conductance_linear:  Linear(24 →  8)    no bias                    192
  gate_conductance_mlp:     Linear(24 → 12 → 12 → 8)  GELU×2              560
  gate_conductance_logit:   (8,)                                            8
  gate_conductance_decoder: Linear(8 → 5)                                  45

TOTAL  ~8,843 training, ~8,504 inference.   vs current 2,407 / 2,124 → ~4× bigger.
```

**Decisions**:
- `StateRateMLP` replaces `IonicRateMLP` + `conc_kan`.
- `IONIC_DIM` 16 → 20; `CARRIED_DIM` 20 → 24.
- Conc self-input restored — rate input is the full carried_state + Vm = 25 dims.
- Decoder stays `nn.Linear(20 → 14)` — free W, no activation, linear-span contract preserved, scaffold still discardable.
- Zero-init KAN `spline_weight` (was the pattern for `conc_kan`; carry forward to the readout).
- `g_φ` learned integrator head **not** pursued (see Failed Approaches).
- CaSR loss reweighting **deferred** — possible optional fine-tune after arch change lands. Up-weighting before arch is cheaper-looking but CaSR already dominates the gradient (~58 % of total loss signal); more capacity is the real fix.

**Old checkpoints incompatible** — `state_dict` keys `ionic_rate_mlp.*` and `conc_kan.*` disappear. `multi_bcl_002/best.pt` etc. become unloadable against the new model. Fresh training from scratch required.

**Files to touch in implementation**:
- `Surrogate/surrogate/model/stage1.py` — constants, remove `IonicRateMLP` + `conc_kan`, add `StateRateMLP` class, rewire `dzdt`, update `_init_weights`.
- `Surrogate/surrogate/training/node_rollout.py` — verify `INIT_CONC` indexing auto-adapts (uses `stage1.ionic_dim:`, should be fine).
- `Surrogate/tests/*` — dim fixtures (16 → 20, 20 → 24).
- `Surrogate/surrogate/training/loss_normalization.py` — **no changes** (decoder output dims unchanged at 14 + 4 + 5).

**Diagnostic artifact**: `Surrogate/diagnostics/integrator_error_budget.py` + saved tensors at `Surrogate/diagnostics/artifacts/integrator_error_budget.pt` (V trajectory, Euler/dopri5/truth ionic states for plotting).

**Next**: Implementation pending user greenlight. Pipeline formalization (Session 26) continues in parallel on a separate tmux pane.

**Q9. What the trained model is actually learning (weight + trajectory analysis).**

*Observation (from `multi_bcl_002/best.pt` at val=0.0084):*

- `ionic_rate_mlp.fc3.weight` Frobenius norm = 0.22, max |w| = 0.04 — essentially at zero-init after 7 epochs.
- `fc1`, `fc2` weights well-trained (norms 4.24, 4.15); `fc2` singular values give effective rank 14/16.
- All 16 latent dims contribute to the decoder (column norms 0.32 to 0.62, mean 0.48) — no dead slots.
- Per-dim prediction std vs truth std on held-out BCL=2000: fast gates match shape (ratio 0.5-1.6); **RR and CaSR oscillate 7× more in prediction than truth** — model is injecting AP-frequency noise into quantities that should be near-flat.
- Prediction at t=0 (rest): m=-0.06 (truth 0.00), h=-0.19 (truth 0.74), CaSR=0.09 (truth 3.68). **At t=300ms (also rest): m=0.01, h=0.70** — near-perfect.
- `decoder(z=0) = bias ≈ arbitrary values`; z=0 is an unlabeled convention the model never learns to map to physiological rest.

*Resolution: the "frozen latent" framing was too cynical.* The rate field is small-magnitude but non-trivial; integrated over 30K steps it produces real latent motion. The model HAS learned AP-shape dynamics for fast gates. The failure modes are more specific:

1. **Wrong initial state.** `decoder(z=0) = bias` never trained to equal physiological rest. Upstroke error (RMSE 0.70 at t=2.5ms) is dominated by this initial-state bleed-through, not failure to model the upstroke itself.
2. **Single AP-oscillation signature applied uniformly.** The rate MLP learned one Vm-driven signature. Fast gates match by luck (they're supposed to oscillate at AP frequency). Slow gates (CaSR, RR) inherit spurious AP-frequency oscillation.
3. **CaSR DC offset.** `pred_mean = 2.48 mM` vs `truth_mean = 3.69 mM`. Latent never reaches a z where `decoder(z)[CaSR] ≈ 3.7` — the rate field for CaSR-feeding latent dims never drives to the correct steady state.

**Q10. Should we hard-code the latent to physical gate parameters (kill the scaffold)?**

*Resolution: No — the multi-model goal (TTP06 + ORd) is load-bearing and hard-coding is a one-way door.* TTP06 and ORd have fundamentally different state sets (18 vs 41 states, different Markov parameterizations); committing to one model's gate structure abandons the universal-latent promise. Also violates Layer 0 maxim (HH gating is a modeling choice, not physical reality) and kills the optical-mapping transfer path.

Evidence-based middle ground — *hybrid representation*: promote slow reservoirs (CaSR, plus ORd's jSR/nSR/CaMKt) out of the latent into the explicit named set alongside concentrations. Keep HH gates / Markov states as learned latent. Rationale: slow reservoirs have clear physical meaning in every ionic model and are the specific dims the discovered latent can't track; gate parameterizations vary across models and deserve latent flexibility.

**Q11. Does weight-clamping the scaffold decoder help the observed gradient issues?**

*Resolution: No.* The diagnosed pathology is gradient *underuse* at the rate MLP's `fc3`, not gradient *explosion* at the decoder. Clamping `W_d` would scale down `∂L/∂z = W_d^T @ residual`, making the rate MLP's gradient signal *smaller*, not larger — worsens the stuck-fc3 problem. Current decoder weight norm (1.94, max |w| ≈ 0.5) is not excessive; nothing needs clamping. If defensive regularization is ever needed, spectral norm on the decoder is the right tool — but I don't predict it changes outcomes here.

**Q12. Physics-informed attractors — enforce z=0 → physiological rest, and what else?**

*Resolution: Adopt two attractors; they compose into one regularizer.*

**A. Rest attractor (decoder bias + rate regularizer).**
- Initialize `ionic_state_decoder.bias = rest_ionic_state` (TTP06 rest values per dim) and **freeze**: `bias.requires_grad_(False)`.
- Semantic effect: latent becomes "deviation from rest". `decoder(z=0) = rest` by construction. The latent has a meaningful origin.
- Add rate attractor term: `L_rest = λ · || f_θ(z_rest, V_rest) ||²` where `z_rest = [zeros, INIT_CONC]`, `V_rest = -85.23 mV`. Ensures z=0 is a fixed point of the dynamics, not just a decoding convention.

**B. Voltage-clamp steady-state attractor (generalization).**
- For any V held constant, HH dynamics have a deterministic fixed point `z_ss(V)` with all gates at `g_∞(V)`.
- Regularizer: `L_vclamp = E_V ~ pacing || f_θ(z_ss(V), V) ||²`.
- Two ways to obtain `z_ss(V)`: precompute via simulator at V grid {-90, -60, -40, -20, 0, +20, +40, +60 mV}, integrating 500 ms to convergence (~30s offline); or mine from training data, using windows where V has been near-constant for >50 ms as empirical z_ss samples.
- Rest attractor is a special case (V = V_rest). The two terms collapse into one unified `L_vclamp` evaluated at a V grid including V_rest.
- Physical meaning: encodes the defining property of HH — for any constant V, the system relaxes to the V-dependent fixed-point manifold.

**C. Optional layer on top (Tier 2, consider if errors persist).**
- Contraction-toward-target: soft penalty on rate direction pointing away from the local steady state. `L_contract = λ · relu(-sign(z_ss(V) - z) · dz/dt).sum()`. Rush-Larsen's contraction principle as a soft constraint, not the rigid exponential form that was rejected before.
- Decoded-gate bounds for the 12 HH rows: `relu(-decoded).sum() + relu(decoded - 1).sum()`. Cheap fix for prediction excursions outside [0, 1].

**Skipped (Tier 3):**
- Ca conservation — hard to implement cleanly with CaSR unbounded.
- Cycle periodicity — useful later for multi-beat rollouts, not for single-AP training.

**Multi-model consideration:** rest state values differ between TTP06 and ORd. Store per-model rest constants; select at training time via model-ID input. Decoder bias is per-model; rate predictor is shared. Clean factoring.

**Implementation scope (to layer onto Session 27 StateRateMLP pivot):**
- `Surrogate/surrogate/model/stage1.py`: ~10 lines to initialize + freeze `ionic_state_decoder.bias`. Add `TTP06_REST_IONIC_STATE` constant (14 values).
- `Surrogate/surrogate/training/node_rollout.py`: add `L_rest` / `L_vclamp` term in loss computation. Pass (V_grid, z_ss_grid) in as a training-time constant. ~20 lines.
- `Surrogate/surrogate/data/`: small preprocessing step to compute z_ss(V) once from the TTP06 simulator at the V grid, cache to disk.

**Decisions (additive to prior Session 27 decisions):**
- Freeze `ionic_state_decoder.bias` at TTP06 physiological rest values; latent becomes "deviation from rest".
- Add voltage-clamp steady-state attractor regularizer `L_vclamp` with λ ≈ 1e-2 (tunable).
- Tier 2 attractors (contraction, gate bounds) optional; evaluate after Tier 1 lands.
- Hybrid explicit-slow-var proposal (promote CaSR out of latent) deferred — revisit after seeing if physics-informed attractors close the CaSR gap without it.
- Hard-coded latent (neural Rush-Larsen) rejected — violates multi-model universality goal and Layer 0 maxim; the evidence isn't strong enough to take a one-way door.

### 2026-04-07 (Session 25): Architecture refinements + first multi-BCL and T2 training
**Worked on**: Replaced VoltageAttention with IonicRateMLP, concentration KAN, dense landmarks, multi-BCL training, T2 restitution training, designed learned contractive step.
**Accomplished**:

1. **VoltageAttention proven redundant and removed**:
   - Mathematical proof: attention on scalar inputs collapses to 32 effective parameters (16 gate slopes + 16 target slopes). The 136-param attention machinery was an overparameterized factorization.
   - Empirical proof: trained weights showed alpha (mixing logit) stayed at init (0.007), MLP contributed ~3% of rate. Gates saturated to binary switches. Model solved loss=0.047 as a switched linear ODE.
   - Replaced with IonicRateMLP: dense MLP (17→16→16→16) with internal residual. 832 params. Achieved loss=0.022 with 5001 dense landmarks — BETTER than attention.
   - Archived to v2_archive/stage1_attention.py.

2. **Concentration KAN**:
   - conc_mlp failed: cross-communicated concentration dims (physically wrong), unstable ODE feedback loop.
   - Replaced with B-spline KAN layer: 17 inputs (ionic latent + Vm) → 4 outputs (conc rates). No cross-talk between concentration dims. Physics-aligned: each conc rate = sum of independent nonlinear functions of ionic dims + Vm.
   - Grid=5, order=3, 612 params.

3. **Dense landmarks crucial**: 5001 points (every 0.1ms over 500ms) >> 23 sparse landmarks. Forces smooth vector field everywhere, not just at waypoints. Loss improved 0.066 → 0.022.

4. **Multi-BCL training succeeded**: T1 train BCLs {300,500,700,1000,1500}, val BCLs {400,600,800,2000}. Variable-length segments per BCL. Val loss reached 0.008 — generalizes to unseen pacing rates.

5. **T2 (S1S2 restitution) training in progress**: Full 11,000ms protocols (10 S1 beats + S2). Initial chaos (T2 val oscillated 24→68→18→25) but stabilizing by epoch 4 (T2_val=0.92).

6. **Next architecture direction — learned contractive step (not yet implemented)**:
   - Current: MLP outputs raw dz/dt, Euler integration has worst-case dt stability (worse than Rush-Larsen which TTP06 uses).
   - Proposed: MLP outputs (target, blend) per dim. `z_new = z + blend*(target - z)` where blend = sigmoid(MLP output).
   - Contractive by construction: z always moves toward target.
   - dt enters through blend (model learns dt-dependent blending, not hardcoded exp(-rate*dt)).
   - Enables discrete training (BPTT) without ODE solver — contractive Jacobian prevents gradient explosion.
   - More general than learned Rush-Larsen: model can learn non-exponential decay.
   - NOT YET IMPLEMENTED. Current model still uses raw rate + odeint.

**Decisions**: VoltageAttention archived, IonicRateMLP is the new ionic rate path. KAN for concentrations. Dense landmarks as default. Multi-BCL as standard training regime.
**Next**: Monitor T2 convergence. Prototype learned contractive step. Consider T3+ tiers.

### 2026-04-16 (Session 26): Pipeline formalization — direction settled

**Worked on**: Analyzed the learned contractive step idea (rejected), then pivoted to the second topic — ML pipeline formalization with MLflow/Optuna/SHAP.

**Accomplished**:

1. **Contractive step formally rejected** (see Failed Approaches table). Both contractive and raw-rate formulations make the same frozen-neighbor approximation; Rush-Larsen's dt-independence only holds at the single-gate level. Current MLP can learn attractor dynamics implicitly. Added architectural complexity not justified.

2. **Pipeline formalization direction settled**:
   - **MLflow** is must-have (core organization layer), **Optuna + SHAP** are nice-to-have analysis overlays.
   - Full rewrite, not a wrapper — archive old `runs/` folder, start fresh.
   - **Hydra** chosen over plain YAML+dataclass: config composition with `_target_` instantiation eliminates factory code, `--multirun` enables native Optuna sweeps, CLI overrides come free.
   - **Project-wide scope**: lives at project root as `cardiac_ml/` package, not `Surrogate/`-specific. Reusable across ionic surrogate, future diffusion ResNet, bidomain cross-skip, optimizer BayesOpt work.
   - Model code stays where it is (`Surrogate/surrogate/model/`, future `Bidomain/...`, etc.); Hydra references it via `_target_: path.to.Class`.
   - MLflow: file-backed (`./mlruns/`), `log_artifact` with state_dict (not `log_model`, avoids pickle fragility for custom classes), per-epoch metrics, auto-log git SHA + dirty flag as tags.
   - Model-specific training logic stays near the model (e.g., `Surrogate/surrogate/training/node_rollout.py`), but USES the project-wide `cardiac_ml.Trainer` skeleton and callbacks.
   - Single `Trainer` class with overridable `_train_step` method rather than unrelated trainer classes per task.

3. **Proposed directory structure**:
```
Heart-Conduction/
  cardiac_ml/                  # NEW project-wide pkg
    training/ (trainer, callbacks, mlflow_logger)
    analysis/ (shap_utils)
    utils/
  conf/                         # Hydra config tree at root
    config.yaml
    model/ data/ training/ optimizer/ experiment/
  scripts/
    train.py    # @hydra.main
    sweep.py    # Optuna via hydra-sweeper
    analyze.py  # SHAP
  mlruns/                       # gitignored
  outputs/                      # Hydra working dirs, gitignored
  archive/runs_legacy/          # old runs/ preserved
```

**Decisions**:
- Hydra over plain YAML (composition + CLI overrides + Optuna plugin).
- Project-wide `cardiac_ml/` package, not Surrogate-specific.
- File-backed MLflow, `log_artifact` with state_dict, per-epoch metrics.
- Single flexible Trainer with overridable `_train_step`.
- Archive old runs/, clean start.

**Next**: Browse GitHub for existing Hydra+MLflow+Optuna ML research templates before committing to design. Then `/blueprint` the implementation.

### 2026-04-07: VoltageAttention Redundancy — Mathematical Proof + Weight Analysis
**Worked on**: Advisor critique of attention layer prompted deep analysis of VoltageAttention mechanism and trained weights.
**Accomplished**:

**Advisor's critique:**
1. Attention layer is redundant — a dense MLP would suffice.
2. If doing attention, need cosine similarity step — which would be meaningless here (binary output on scalar tokens).

**Mathematical proof that attention collapses to 32 scalars:**
- Q_i = z_i * W_q[i,:] and K = Vm * W_k are scalars embedded in R^4. Each of the 16 ionic dims has its own W_q row.
- Dot product reduces to: score_i = z_i * Vm * c_i where c_i = W_q[i,:] dot W_k (a learned constant per dim).
- After sigmoid: gate_i = sigma(c_i * z_i * Vm). This is a hard switch — not soft modulation.
- Cosine similarity would be even worse: sign(z_i * Vm) * const — binary output, no directional content.
- Target is linear in Vm: target_i = t_i * Vm (W_v * W_out product). Cannot represent sigmoid gate steady-states (nonlinear in Vm).
- Full equation per dim: dz_i/dt = sigma(c_i * z_i * Vm) * (t_i * Vm - z_i). Only 32 effective params (16 c_i + 16 t_i) out of 136 attention params.

**Empirical evidence from trained weights (runs/single_ap_001/best.pt):**
- Alpha (ionic_mixing_logit) barely moved from init: mean 0.0076 (init was 0.007). MLP contributes ~3% of rate. Effectively unused.
- Gates saturate to binary switches (values span 1e-13 to 1.0) — hard on/off, not soft modulation.
- At z=0 (resting): gate = 0.5 for all dims. Cannot distinguish resting states of different variables.
- The model solved ionic_state_mse to 0.047 using attention alone as a **switched linear ODE**.
- The MLP, residual bypass, and alpha mixing were all effectively unused — the architecture defaulted to its simplest mode.

**Conclusion:** VoltageAttention with scalar per-dim inputs is fundamentally limited. The n x 1 cross-attention mechanism, which seemed elegant in Session 7, collapses when each "token" is a single scalar. The nonlinear expressivity we expected from attention (state-dependent gating of voltage-dependent targets) reduces to a linear-in-Vm target with a bilinear gate. Replace VoltageAttention + ionic_mixing_mlp + residual_bypass with a single dense MLP(z_ionic, Vm) -> ionic_rate. Simpler, fewer params, can learn nonlinear V-dependent dynamics that the linear target cannot.

**Checkpoint note:** runs/single_ap_001/best.pt represents the attention-based architecture (ionic_state_mse=0.047). Superseded by this analysis — architecture will be replaced. Checkpoint retained for comparison.

**Decisions**: Replace VoltageAttention + ionic_mixing_mlp + residual_bypass with dense MLP. The attention mechanism is not wrong in principle — it's wrong for scalar inputs where the QKV structure collapses.
**Next**: Design replacement MLP architecture for ionic Stage 1.

### 2026-04-07: Architecture Refinements + First Single-AP Training
**Worked on**: Architecture refinements to IonicStage1 dzdt() and first single-AP training runs.
**Accomplished**:
- **MLP input changed**: z_mid to ionic_delta (attention rate). Rationale: MLP is a rate corrector, should see rates not states. z_mid has DC offset that wastes tiny MLP capacity. Alpha logit stays semantic (mixing on/off) instead of becoming a scale fixer.
- **rms_norm removed** from MLP input: delta magnitude is informative (small at rest, large during upstroke). Normalizing destroys this signal.
- **VoltageAttention now returns rate directly**: `gate*(target-z)` instead of `z + gate*(target-z)`. Eliminates add-then-subtract roundtrip in dzdt().
- **VoltageAttention shrunk to ionic-only (16 dims)**: Concentrations no longer go through attention. Clean separation.
- **Dedicated concentration path via B-spline KAN**: `conc_kan = KANLayer(17->4, grid=5, order=3)`. Input: ionic_latent(16) + Vm(1). No conc self-input, no cross-communication between concentration dims. Physics-aligned: each conc rate = sum of independent nonlinear functions of ionic dims and Vm. 612 params (544 spline + 68 base weights).
- **conc_mlp removed** — replaced by KAN. MLP was wrong because it cross-communicated concentration dims.
- Training runs (single AP, BCL=2000, 500ms window):
  - Phase A1 (ionic_state_mse only): 50.6 to 0.047 in 500 epochs. ~1.25s/epoch on GPU.
  - Adjoint backward diverged with random weights — switched to backprop-through-solver (adjoint=False).
  - dopri8 diverged — switched to dopri5 with rtol=1e-3, atol=1e-3.
  - Concentration training (conc_only phase, frozen ionic):
    - Shared attention: corrupted ionic weights (shared W_q/W_out).
    - MLP: unstable feedback through ODE (Xavier init exploded to 1e237). Zero last-layer init: stable but slow (0.65 to 0.43 in 500 epochs).
    - KAN with L1: stable, 176 to 37 in 500 epochs. Slow due to grad clipping.
    - KAN with L2 + cosine decay (LR 1e-2 to 1e-5): 147K to best ~5.6, settled ~10 in 500 epochs.
- Model params: 1988 inference (was 1408), 2271 total with scaffold.
**Decisions**: Concentration loss detached from ionic training (train separately, frozen ionic). "conc_only" phase added to node_rollout.py. L2 loss for concentrations (not normalized MSE).
**Next**: Investigate W_out rank-4 bottleneck. Conc training needs per-dim normalization. Scale to multiple BCLs.

### 2026-04-06–07: Neural ODE Pivot Implementation
**Worked on**: Full NODE pivot — research, architecture design, 4 audit rounds, implementation (4 phases), training strategy.
**Accomplished**:
- Researched LFLDNets (Salvador 2025) — CfC evaluated but rejected (structural prior). Chose unconstrained vector field `dz/dt = f_θ(z, V)`.
- Key insight: VoltageAttention's `z + gate*(target-z)` IS a linear attractor in ODE form: `dz/dt = gate*(target(V)-z)`. Rush-Larsen emerges without being designed in.
- Real win of NODE: gradient chain 200-1000 steps (dopri8) vs 30K discrete. Not "attractor emergence" — gradient tractability.
- Stiffness analysis: eigenvalue spread ~1000× during upstroke from scaffold-imposed gate tracking. Manageable on 300ms solve. Segmentation rejected (no ground truth z for mid-AP init). Dense upstroke landmarks (10/20 in first 5ms).
- Implemented: Phase 0 (archive), Phase 1 (stage1.py: dt removed, residual_bypass, dzdt, _compress, forward repurposed), Phase 2 (node.py: IonicNODE + dopri8 + euler_step), Phase 3 (node_rollout.py: odeint_adjoint training loop). 116 tests passing.
- 4 audit rounds (Opus): 46→21→7→0 critical/high issues. Fixed: forward/euler_step incompatibility, V_traj indexing, ~50→200-1000 NFE, PLANNED markers, Section 5 SUPERSEDED, stale refs.
- New TRAINING_STRATEGY.md for NODE training. Discrete docs archived.
**Next**: First NODE training run. Phase A1 on T1 data. See TRAINING_STRATEGY.md.

### 2026-04-02–06 Session
**Worked on**: Full training pipeline implementation and first training attempts for IonicSurrogateV3.
**Accomplished**: Built complete training pipeline (data cache, datasets, rollout engine, phase-aware trainer, checkpoint, monitor, metrics, shard loader, CLI, training agent — 44 tests). Removed encoder (model discovers own latent). Implemented dt curriculum (subsample existing data at coarse→fine dt), per-dim min-max loss normalization, truncated BPTT, cosine warm restarts. Trained through A1-A3 of dt curriculum (A3 val=0.92 at dt=0.1ms). A4 (native dt=0.01ms, 30K steps) failed — val stuck at ~720 for 155+ epochs. Core issue: latent state becomes unstable over long discrete autoregressive rollouts. Error at step N corrupts steps N+1 through 30K.
**Next**: Pivot to Latent Neural ODE approach. The v3 attention + MLP architecture may serve as the dynamics function f(z, Vm) inside an ODE solver (torchdiffeq). Adjoint method avoids backpropping through 30K discrete steps. Key challenge: multi-timescale stiffness (0.1ms to 100ms+).

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
| Neural ODE with plain MLP (dz/dt = MLP(z,Vm)) | Too unconstrained without structural priors (Session 7). **Reconsidered**: using existing attention+MLP as f_θ provides inherent contraction via attention gating. Now the active approach — see Current Direction and KNOWLEDGE.md Section 5b. |
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
| Adjoint method (odeint_adjoint) with random weights | Adjoint backward diverged during early training when weights are random (vector field is chaotic). Switched to backprop-through-solver (adjoint=False). Adjoint may work later once field is partially trained. |
| dopri8 for early NODE training | Diverged — too aggressive for poorly-trained vector field. dopri5 with loose tolerances (rtol/atol=1e-3) is stable. |
| Shared VoltageAttention for concentrations (20-dim attention) | W_q/W_out are shared — training conc dims corrupted ionic weights. Concentrations need a dedicated path. |
| conc_mlp for concentration rates | MLP cross-communicates concentration dimensions (each conc rate sees all other concs). Physics violation: concentration rates are independent functions of ionic state and Vm. Xavier init caused ODE feedback explosion (1e237). Zero last-layer init was stable but slow. |
| Normalized MSE for concentration loss | Catastrophic for out-of-range predictions: if pred >> true, normalization makes loss look small. L2 (raw MSE) is correct for concentrations. |
| VoltageAttention (n x 1 cross-attention on scalar inputs) | Collapses to 32 effective params (switched linear ODE) when operating on scalar per-dim inputs. Q_i = z_i * W_q[i,:], K = Vm * W_k are scalars in R^4 — dot product reduces to score_i = z_i * Vm * c_i (learned constant). Target is linear in Vm (t_i * Vm), cannot represent nonlinear sigmoid gate steady-states. Trained weights confirm: MLP correction unused (alpha stayed at init 0.007), gates saturate to binary switches. Replace with dense MLP(z_ionic, Vm) -> ionic_rate. |
| conc_mlp (dense MLP for concentrations) | Cross-communicated concentration dimensions (each conc rate sees all other concs). Physics violation: concentration rates are independent functions of ionic state and Vm. Xavier init caused ODE feedback explosion (1e237). Zero last-layer init was stable but slow. Replaced by B-spline KAN layer with no cross-talk between concentration dims. |
| Xavier init on last layer of rate MLP | ODE diverged on first integration step — initial vector field too large, solver immediately produces NaN/Inf. Fixed with zero-init on last layer (model starts as identity/zero-rate, gradually learns dynamics). Critical for any MLP used as ODE right-hand side. |
| Learned contractive step (target + rate + exponential integrator) | Analyzed in Session 26 (2026-04-16). Mathematically cleaner than raw rate + Euler — factored as `z_new = z + (1-exp(-rate*dt)) * (target - z)`. BUT: it's a constrained parametrization of what current IonicRateMLP can already learn implicitly. Both formulations make the same frozen-neighbor assumption; Rush-Larsen's dt-independence only holds locally (single gate with constant V), not system-wide. Current MLP can learn attractor dynamics by outputting 0 at fixed point. Added structure would give training-stability benefit and guaranteed contraction at large dt, but doesn't solve the original dt-dependence concern (both formulations have frozen-neighbor errors that scale with dt). Added complexity (2x output dims, softplus rate, new integrator) not justified when current formulation works at fixed training dt. |
| Learned `g_φ` integrator head (rate + dt → Δz trained against dopri5 single-step flow) | Rejected 2026-04-19 (Session 27) based on `Surrogate/diagnostics/integrator_error_budget.py` measurement on trained `multi_bcl_002/best.pt` (val=0.0084). Euler-vs-dopri5 RMSE was 0.00161; dopri5-vs-truth RMSE was 0.34638 — integrator truncation is **215× smaller** than model-capacity error on native dt=0.01ms, with 700× ratio at the upstroke and never better than 150× across the 300ms trajectory. A learned integrator head would eliminate 0.16% of total inference error and leave 99.8% untouched. The real bottleneck is rate-field capacity (CaSR at 27% NRMSE dominates the loss); fix is `StateRateMLP` pivot + latent expansion, not integrator redesign. Generalizable lesson: measure before redesigning integrators — the suspect may not be guilty. |
| Single KAN layer for conc-rate prediction | Analyzed in Session 27 (2026-04-19). A single KAN layer is additively universal (each output = `Σⱼ φⱼ(xⱼ)`) but cannot represent multiplicative cross-terms like `I_NaCa = exp(γVF/RT)·Na_i³·Ca_i` or `I_CaL flux = d·f·f2·fCass·Ca_ss·exp(VF/RT)`. These are Layer-0 physics requirements, not model approximations. Replaced by `Linear(25→32)+GELU → KAN(32→24)`: the GELU mixer owns cross-product capture, KAN owns per-dim rate shaping at readout. |
| Separate rate paths for ionic vs conc (`IonicRateMLP` + `conc_kan`) | Historical scar from a `conc_mlp(20→4)` that blew up due to Xavier init + ODE feedback loop → 1e237. The fix at the time bundled two changes: (a) drop conc self-input, (b) switch to no-cross-talk KAN. Only (b) was required for stability. Maintaining separate paths prevents the model from learning real Ca↔Na coupling via NCX, and forces asymmetric depth/width tuning for ionic vs conc rates. Replaced 2026-04-19 (Session 27) with a unified `StateRateMLP(z_full(24), Vm(1)) → dz/dt(24)`. |
| Chasing per-step ionic speed parity with TTP06 on GPU | Benchmark 2026-04-21 (Session 29 late): TTP06 compiled @ n=10k = 34.1M cell-steps/s; v4 surrogate Euler @ n=10k = 4.2M cs/s — **surrogate 8× slower**. TTP06's `model.step` fuses into a handful of kernels under `torch.compile`; our `StateRateMLP + dzdt + Euler` has branching structure (sigmoid skip-blend, input_ref subtraction, Euler add not compiled with dzdt) that prevents equivalent fusion. Closing the gap would require CUDA-graph capture or CfC-style batched closed-form update — engineering effort disproportionate to the tiny fraction of wall-time this represents (ionic ≈ 6% of bidomain; KNOWLEDGE §1). Wrong bottleneck. Don't optimize ionic MLP for GPU speed; the win is elsewhere. |
| "Neural surrogate replaces the whole bidomain" framing | Benchmark-grounded rejection 2026-04-21. The ionic half is already near kernel-launch limits on GPU with compiled TTP06. Only the elliptic half is slow (94% wall time per KNOWLEDGE §1). Correct framing: hybrid bidomain surrogate — classical ionic scaffold + neural elliptic/diffusion replacement. See Current Direction. |
| Full Data v2 T1-T12 regen at this time | Paused 2026-04-21 after T1 batch 1 of 5 completed (~3.5 GB on disk). Single-cell T1-T12 data is only useful for the ionic surrogate track, which is deprioritized pending hybrid-bidomain architecture validation. Spec (`DATA_V2_SPEC.md`), generator (`datagen/generate_t1_v2.py`), schema module (`surrogate/data/schema.py`), and partial h5 are all preserved. Resume if the ionic CPU-deployment path becomes a deliverable. |
| Hyperbolic bidomain as v1 surrogate target | Considered 2026-04-21 (Session 29 late). Maxwell-Cattaneo / Feng-Bova hyperbolic form (second time derivative on Vm and φ_e, finite propagation speed) is theoretically a better NN target — no elliptic shortcut, CFL crushes classical dt, NN's learned effective dt flexibility shines. Deferred to future Phase B. Reasons: (a) no hyperbolic simulator in project, (b) Bidomain V1's parabolic-elliptic data is ready now, (c) ionic scaffold (TTP06) is defined for parabolic V_m; reformulating in hyperbolic regime is extra work, (d) need to validate the dual-tower + cross-talk architecture on a simpler PDE first. |

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
| 2026-04-02–03 | 19-20 | Training pipeline built (6 PLAN phases). Data cache, datasets, encoder, phases, rollout, trainer, checkpoint, monitor, metrics, shard loader, CLI, training agent. 83 tests. Encoder-based A1 verified (val=7.9e-5). Then removed encoder — model discovers own latent. B1(r=1) val=0.073, B1.5(r=10) val=0.39 after 200 epochs. Variance normalization, then min-max. |
| 2026-04-04 | 21 | dt curriculum replaces rollout curriculum. Subsample T1 data at variable dt (3ms→1ms→0.1ms→0.01ms), fixed 300ms coverage. A1-A3 completed (A3 val=0.92). TBPTT + warm restarts implemented. Batch=32768 failed (too few steps), reverted to 4096. |
| 2026-04-04–06 | 22 | A4 (dt=0.01ms, r=30K) stuck at val~720 for 155 epochs despite TBPTT(500) + warm restarts. Discrete autoregressive at native dt fundamentally limited by error compounding. **Decision: pivot to Latent Neural ODE.** The latent becomes unstable over long rollouts — continuous ODE integration with adjoint gradients avoids discrete error accumulation. |
| 2026-04-07 | 23 | Architecture refinements: VoltageAttention returns rate (not state), ionic-only (16 dims), MLP takes ionic_delta (not z_mid), rms_norm removed. conc_mlp replaced by B-spline KAN (612 params). First training: ionic_state_mse 50.6→0.047 (500 epochs). Conc KAN: 147K→~5.6 best. Adjoint unstable→backprop-through-solver. dopri8→dopri5 (rtol/atol=1e-3). Params: 1988 inference, 2271 total. |
| 2026-04-07 | 24 | VoltageAttention proven mathematically redundant: collapses to 32 effective params (switched linear ODE) on scalar inputs. Advisor critique + weight analysis of best.pt. MLP/residual bypass unused. Decision: replace with dense MLP. Checkpoint runs/single_ap_001/best.pt superseded. |
| 2026-04-07 | 25 | VoltageAttention replaced by IonicRateMLP (832 params, loss=0.022 vs 0.047). conc_mlp replaced by B-spline KAN. Dense landmarks (5001 pts). Multi-BCL training val=0.008. T2 restitution training in progress (stabilizing at epoch 4). Learned contractive step designed (not yet implemented). |
| 2026-04-16 | 26 | Learned contractive step rejected (frozen-neighbor equivalence, complexity not justified). Pivot to pipeline formalization: project-wide `cardiac_ml/` package at root, Hydra + MLflow + Optuna + SHAP. Model code stays in place, referenced via `_target_`. File-backed MLflow, single flexible Trainer. Archive old `runs/`. Next: survey existing open-source templates, then `/blueprint`. |

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
