# Data Generation — Execution Plan

> Master document for tracking the actual data generation run.
> Updated after each step with timings, file sizes, and issues.

## Pre-Flight: Speed Benchmarks

Before generating ~1TB of data, benchmark TTP06 execution speed across configurations:

| Config | 10K steps (100ms) | Steps/s | Notes |
|--------|-------------------|---------|-------|
| CPU single | 6.35s | 1,576 | baseline (n=1 ≈ 1,637 from full beat) |
| CPU batch n=10 | 6.41s | 15,592 | 4.9× throughput, same wall time |
| CPU batch n=50 | 6.51s | 76,793 | 24× throughput |
| CPU batch n=100 | 6.62s | 151,131 | 48× throughput |
| CPU batch n=200 | 7.41s | 269,806 | **86× throughput** ← sweet spot |
| GPU single | 21.4s | 466 | 3.5× SLOWER than CPU (kernel launch overhead) |
| GPU batch | pending | pending | waiting for results |
| LUT | SKIPPED | — | engine bug in TTP06LUT.get_all_gating() |

**Key finding: CPU batch is the winner.** Near-linear scaling because PyTorch vectorizes element-wise ops. 200 protocols run in the same wall time as 2 protocols. GPU single-cell is 3.5× slower than CPU due to kernel launch overhead for scalar ops.

**Strategy: batch protocols as parallel "cells"** in model.step(). Each cell gets its own I_stim schedule. One step() call advances all protocols simultaneously. At batch=200 on CPU: ~270K steps/s total throughput.

**torch.compile GPU results (game changer):**

| Config | n=200 steps/s | n=1000 steps/s | vs CPU batch |
|--------|--------------|---------------|-------------|
| CPU batch (no compile) | 269,806 | — | 1× |
| GPU torch.compile | 1,565,830 | 7,795,815 | 5.8×–29× |

GPU torch.compile fuses ~50 tiny CUDA kernels into a single graph per step.
Wall time is constant (~1.28s for 10K steps) across n=50 to n=1000.

**Estimated generation time (GPU compiled, batch=1000):**
- 1 beat (100K steps) × 1000 protocols: ~13 seconds
- Tier 4 (200 protocols × 63s avg): ~3 minutes
- Full dataset (~6B steps): ~13 minutes
- With multi-dt (×5): ~65 minutes
- With all augmentations + celltypes: ~3-4 hours total

**GPU scaling limit (32GB VRAM Blackwell):**

| n_cells | Steps/s total | VRAM | Wall time (5K steps) |
|---------|--------------|------|---------------------|
| 1,000 | 7.8M | ~0 MB | 0.64s |
| 5,000 | 39.1M | 2 MB | 0.64s |
| 10,000 | 77.5M | 3 MB | 0.65s ← sweet spot |
| 20,000 | 141.5M | 6 MB | 0.71s |
| 50,000 | 124.1M | 16 MB | 2.01s |
| 200,000 | 166.0M | 64 MB | 6.02s |

VRAM is NOT the bottleneck (200K cells = 64MB / 32GB). Throughput plateaus at n=10K-20K.
Per-cell efficiency peaks at n=1K-5K (~7,800 steps/s/cell).

**At n=10,000 (77.5M steps/s):**
- Full dataset (~6B steps): ~77 seconds
- With multi-dt (×5): ~6.5 minutes
- With all augmentations + celltypes: ~30-60 minutes

**Full 1-beat benchmark (100K steps at dt=0.01ms):**

| Config | Time/beat/proto | Speedup vs CPU n=1 |
|--------|----------------|-------------------|
| CPU n=1 | 61.1s | 1× |
| CPU n=200 | 0.358s | 171× |
| GPU compile n=200 | 0.066s | 924× |
| GPU compile n=1000 | 0.013s | 4,771× |
| GPU compile n=10000 | 0.0013s | 47,227× |

**DECISION: GPU + torch.compile + batch=10,000**
All tiers batched together. Pad small tiers to fill batch. ~9 min compute + ~30-60 min I/O = ~1 hour total.

## Step 1: Smoke Test (~2 min)
- [ ] Verify external HDD mounted at `/media/norepinephrine/Elements-ext4/`
- [ ] Run Tier 1, single BCL=1000, 5 beats
- [ ] Verify HDF5 written, readable, correct shape (T, 23)
- [ ] Report: file size, wall time, steps/sec

**Result**: _pending_

## Step 2: Core Data — Tiers 1-3 (~30-60 min)
- [ ] Tier 1: 9 BCLs × 20 beats × EPI
- [ ] Tier 2: 8 DI values × EPI
- [ ] Tier 3: 4 dynamic protocols × EPI
- [ ] Verify: AP shapes, restitution curve, alternans

**Result**: _pending_

## Step 3: Random + Injection — Tiers 4-5 (~2-4 hours)
- [ ] Tier 4: 200 random interval protocols
- [ ] Tier 5: 10 injection profiles
- [ ] Monitor: progress every 20 protocols

**Result**: _pending_

## Step 4: Clamp + Perturbation — Tiers 6-7 (~1-2 hours)
- [ ] Tier 6: 9 clamp protocols + AP clamp
- [ ] Tier 7: 20 concentration combos × 3 BCLs

**Result**: _pending_

## Step 5: Stress Data — Tiers 8-12 (~2-4 hours)
- [ ] Tier 8: Long pacing (50 beats), quiescence (2-10s)
- [ ] Tier 9: 6 corruption recovery
- [ ] Tier 10: 6 tissue-specific scenarios
- [ ] Tier 11: 50 stitched protocols
- [ ] Tier 12: ENDO + M_CELL × Tiers 1-3

**Result**: _pending_

## Step 6: Multi-dt Sweep — Tiers 1-4 (~2-4 hours)
- [ ] dt ∈ {0.005, 0.02, 0.05, 0.1} for Tiers 1-4 (dt=0.01 already done)
- [ ] Verify: same AP shape at all dt values

**Result**: _pending_

## Step 7: Shard Conversion (~30-60 min)
- [ ] HDF5 → .pt shards (float32, segment_length=1000)
- [ ] Train/val split (hold out 10% of protocols)
- [ ] Verify: shard loads to GPU, correct dtype/shape

**Result**: _pending_

## Summary

| Step | Tier(s) | Status | Time | Size |
|------|---------|--------|------|------|
| Pre-flight | — | pending | — | — |
| 1 | 1 (smoke) | pending | — | — |
| 2 | 1-3 | pending | — | — |
| 3 | 4-5 | pending | — | — |
| 4 | 6-7 | pending | — | — |
| 5 | 8-12 | pending | — | — |
| 6 | 1-4 (multi-dt) | pending | — | — |
| 7 | shards | pending | — | — |
| **TOTAL** | | | | |
