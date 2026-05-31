# Hardware Constraints for Training

## GPU
- NVIDIA RTX PRO 4500 Blackwell
- 33.7 GB VRAM
- All computation must stay on GPU. No CPU fallback for forward/backward.

## Storage

| Device | Path | Capacity | Free | Speed |
|---|---|---|---|---|
| NVMe SSD | `/` (root) | 240 GB | ~47 GB (post-cleanup 2026-04-02) | ~3000 MB/s |
| USB HDD | `/media/HDD/` | 5.5 TB | 4.6 TB | **~7 MB/s** direct I/O, ~246 MB/s buffered (USB 3.0, WD Elements) |

The training data (608 GB TTP06, 12 GB ORd) lives on the HDD. Measured 2026-04-02: `dd bs=1M iflag=direct` → 7.1 MB/s; `dd bs=4M` → 244 MB/s (OS page cache). Previous measurements of 1.26 MB/s and 14 MB/s were inaccurate.

## The Problem

Our model processes ~640 bytes per sample per step. At batch=1024:
- Data needed per batch: 1024 × 640 bytes = 640 KB
- GPU forward+backward: ~1ms for our tiny model
- HDD time to load 640 KB: ~90ms at 7 MB/s (direct I/O)

The GPU is ~90× faster than the data pipeline. GPU utilization will be ~1% if we naively load from HDD per batch.

## Mitigation Strategy

### 1. Pre-load entire training tier into GPU VRAM

For Phase A and early Phase B, the data is small enough to fit in VRAM:

| Tier | Raw size (HDD) | Preprocessed float32 (VRAM) | Fits in 33.7 GB? |
|---|---|---|---|
| T1 (steady-state) | 3.5 GB | ~1.8 GB | Yes |
| T1+T2 | 8.6 GB | ~4.3 GB | Yes |
| T1-T3 | 10.9 GB | ~5.5 GB | Yes |
| T1-T4 | 562 GB | ~280 GB | NO — T4 is 551 GB |

Strategy:
- Phases A1-A3, B1-B3: Pre-load T1 (or T1-T3) preprocessed into GPU VRAM. Zero HDD access during training. GPU utilization ~100%.
- Phase B4+, C, D, E (when T4 needed): Use shard-based streaming (see below).

### 2. Pre-process and cache to SSD

We have ~47 GB free on SSD (post-cleanup). Preprocessed T1-T3+T12 in float32 is ~11 GB. Cache it:

```
/tmp/surrogate_cache/          # or a dedicated SSD path
├── t1_preprocessed.pt         # ~1.8 GB, pre-segmented
├── t2_preprocessed.pt         # ~2.5 GB
├── t3_preprocessed.pt         # ~1.2 GB
```

SSD speed: ~3000 MB/s. Loading a batch from SSD: <1ms. Problem solved for T1-T3.

### 3. Shard streaming for T4

T4 (random intervals, 551 GB) cannot fit in VRAM or SSD. Strategy:
- Pre-convert T4 to .pt shards (~200 MB each, float32) on HDD
- Load ONE shard into VRAM at a time (200 MB fits easily)
- Train on that shard until exhausted, then swap next shard
- Shard load time: 200 MB at ~7-246 MB/s = 1-30 seconds (buffered vs direct)
- Training on one shard: ~2000 segments × ~5 epochs = minutes
- Shard swap overhead: <1% of training time if shards are large enough

### 4. Prefetch with double-buffering

While training on shard N (on GPU), prefetch shard N+1 from HDD into CPU RAM in a background thread. When training finishes shard N:
- Swap: move shard N+1 from CPU to GPU (PCIe, ~10 GB/s, <1s)
- Start loading shard N+2 from HDD in background

This hides almost all HDD latency.

### 5. Pin memory for async transfers

Use `torch.cuda.Stream()` for async HDD→CPU→GPU transfers. PyTorch DataLoader with `pin_memory=True` and `num_workers=2` handles this automatically for shard-based loading.

## Implementation Priority

1. **Phase A-B3**: Pre-load T1-T3 to GPU. No data pipeline complexity needed. Just `data = torch.load('cache.pt').cuda()`.
2. **Phase B4+**: Implement shard streaming with prefetch. Required only when T4 enters training.
3. **Phases D-E**: Same shard streaming. Stage 2 training data is small (precomputed features) — fits in VRAM.

## Data Transfer Budget

| Operation | Time | When |
|---|---|---|
| Load T1 from HDD to VRAM (first time) | ~22s (buffered) to ~8 min (direct) | Once at training start |
| Load T1-T3 from SSD cache | ~2s | Once at training start |
| Swap one T4 shard (200MB, prefetched) | ~1s | Every few minutes during B4+ |
| Forward+backward per batch | ~1ms | Every batch |

The initial T1 load from HDD (4 min) is a one-time cost. After that, everything runs from VRAM or SSD cache.
