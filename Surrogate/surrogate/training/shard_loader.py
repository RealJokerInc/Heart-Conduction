"""T4 shard streaming: convert tier to shards, load with double-buffering.

T4 (551 GB) cannot fit in RAM or SSD. Strategy:
- Pre-convert to .pt shards (~200 MB each, float32) on HDD
- Load one shard at a time, extract segments, yield batches
- Double-buffer: prefetch shard N+1 while training on shard N
"""

import logging
import random as pyrandom
from pathlib import Path
from threading import Thread
from typing import Optional, Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.preprocessor import V3Preprocessor
from .datasets import SegmentDataset

logger = logging.getLogger(__name__)


def _load_trace_raw(raw_dir: str, tier: int, protocol_name: str) -> Tensor:
    """Load raw (T, 47) trace from HDF5."""
    import h5py
    path = Path(raw_dir) / f'tier{tier:02d}.h5'
    with h5py.File(path, 'r') as f:
        return torch.tensor(f[protocol_name]['data'][:], dtype=torch.float64)


def _list_protocols(raw_dir: str, tier: int) -> list[str]:
    """List protocol names in a tier."""
    import h5py
    path = Path(raw_dir) / f'tier{tier:02d}.h5'
    if not path.exists():
        return []
    with h5py.File(path, 'r') as f:
        return list(f.keys())


class ShardConverter:
    """Convert a tier's HDF5 protocols to preprocessed .pt shards on HDD."""

    def __init__(self, raw_dir: str, shard_dir: str, shard_size_mb: float = 200.0):
        self.raw_dir = Path(raw_dir)
        self.shard_dir = Path(shard_dir)
        self.shard_size_mb = shard_size_mb
        self.preprocessor = V3Preprocessor()

    def convert_tier(self, tier: int = 4) -> int:
        """Convert tier to preprocessed shards. Returns number of shards created."""
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        protocols = _list_protocols(str(self.raw_dir), tier)
        if not protocols:
            raise ValueError(f"No protocols found in tier {tier}")

        current_chunks: list[dict[str, Tensor]] = []
        current_bytes = 0
        shard_idx = 0

        for proto in protocols:
            logger.info(f"Processing tier {tier} / {proto}")
            raw = _load_trace_raw(str(self.raw_dir), tier, proto)
            processed = self.preprocessor.process_segment(raw)
            processed_f32 = {k: v.float() for k, v in processed.items()}

            proto_bytes = sum(v.nbytes for v in processed_f32.values())
            current_chunks.append(processed_f32)
            current_bytes += proto_bytes

            if current_bytes >= self.shard_size_mb * 1e6:
                self._write_shard(current_chunks, shard_idx)
                shard_idx += 1
                current_chunks = []
                current_bytes = 0

        if current_chunks:
            self._write_shard(current_chunks, shard_idx)
            shard_idx += 1

        logger.info(f"Created {shard_idx} shards in {self.shard_dir}")
        return shard_idx

    def _write_shard(self, chunks: list[dict[str, Tensor]], idx: int) -> None:
        keys = chunks[0].keys()
        merged = {}
        for k in keys:
            merged[k] = torch.cat([c[k] for c in chunks], dim=0)
        path = self.shard_dir / f'shard_{idx:04d}.pt'
        torch.save(merged, path)
        size_mb = sum(v.nbytes for v in merged.values()) / 1e6
        logger.info(f"Saved {path} ({merged[list(keys)[0]].shape[0]} timesteps, {size_mb:.0f} MB)")


class ShardStreamLoader:
    """Stream preprocessed shards from HDD with double-buffering.

    Yields batches of (segment_length, ...) tensors. Shuffles shard order
    each epoch. Prefetches next shard in background thread.
    """

    def __init__(self, shard_dir: str, segment_length: int, batch_size: int = 64,
                 device: str = 'cuda', stride: Optional[int] = None):
        self.shard_dir = Path(shard_dir)
        self.segment_length = segment_length
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.stride = stride

        self.shard_paths = sorted(self.shard_dir.glob('shard_*.pt'))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {self.shard_dir}")

        self._prefetch_data: Optional[dict] = None
        self._prefetch_thread: Optional[Thread] = None

    def __len__(self) -> int:
        return len(self.shard_paths)

    def _load_shard(self, path: Path) -> dict[str, Tensor]:
        """Load shard from disk (runs in background thread)."""
        return torch.load(path, weights_only=False)

    def _start_prefetch(self, path: Path) -> None:
        """Start loading next shard in background."""
        def _load():
            self._prefetch_data = self._load_shard(path)
        self._prefetch_thread = Thread(target=_load, daemon=True)
        self._prefetch_thread.start()

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield batches from all shards, one shard at a time."""
        paths = list(self.shard_paths)
        pyrandom.shuffle(paths)

        # Start prefetching first shard
        self._start_prefetch(paths[0])

        for i, path in enumerate(paths):
            # Wait for current shard to finish loading
            self._prefetch_thread.join()
            shard_data = self._prefetch_data

            # Start prefetching next shard
            if i + 1 < len(paths):
                self._start_prefetch(paths[i + 1])

            # Create dataset + dataloader from this shard
            dataset = SegmentDataset(shard_data, self.segment_length, stride=self.stride)
            if len(dataset) == 0:
                continue
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

            for batch in loader:
                yield {k: v.to(dtype=torch.float64, device=self.device)
                       for k, v in batch.items()}
