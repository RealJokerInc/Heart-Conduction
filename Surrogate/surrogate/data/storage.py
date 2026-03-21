"""HDF5 storage and .pt shard conversion for training data.

Two-layer storage:
- Raw HDF5 (float64, full metadata) — source of truth, archival
- .pt shards (float32, pre-chunked, pre-shuffled) — training speed
"""

import random
from pathlib import Path
from typing import List, Optional, Dict, Any

import h5py
import torch
import numpy as np

from .single_cell_generator import TraceData


class TraceStorage:
    """HDF5 storage for generated traces."""

    def __init__(self, base_dir: str = '/media/norepinephrine/Elements-ext4/surrogate_data/raw'):
        self.base_dir = Path(base_dir)
        self._check_storage()

    def _check_storage(self):
        """Verify storage directory is accessible."""
        # For external HDD, check mount point
        if '/media/' in str(self.base_dir):
            mount_point = Path('/media/norepinephrine/Elements-ext4')
            if not mount_point.is_dir():
                raise RuntimeError(
                    f"External HDD not mounted at {mount_point}. "
                    f"Mount it before running data generation."
                )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_trace(self, trace: TraceData, tier: int, protocol_name: str):
        """Append trace to tier HDF5 file."""
        path = self.base_dir / f'tier{tier:02d}.h5'
        with h5py.File(path, 'a') as f:
            if protocol_name in f:
                del f[protocol_name]  # overwrite if exists
            grp = f.create_group(protocol_name)
            grp.create_dataset('data', data=trace.data.cpu().numpy(),
                               dtype=np.float64)
            for k, v in trace.metadata.items():
                if v is None:
                    continue
                if isinstance(v, dict):
                    grp.attrs[k] = str(v)
                else:
                    grp.attrs[k] = v

    def load_trace(self, tier: int, protocol_name: str) -> TraceData:
        """Load a single trace."""
        path = self.base_dir / f'tier{tier:02d}.h5'
        with h5py.File(path, 'r') as f:
            data = torch.tensor(f[protocol_name]['data'][:],
                                dtype=torch.float64)
            metadata = dict(f[protocol_name].attrs)
        return TraceData(data=data, metadata=metadata)

    def list_protocols(self, tier: int) -> List[str]:
        """List all protocols in a tier file."""
        path = self.base_dir / f'tier{tier:02d}.h5'
        if not path.exists():
            return []
        with h5py.File(path, 'r') as f:
            return list(f.keys())

    def save_tier(self, traces: List[TraceData], tier: int):
        """Save all traces for a tier."""
        for trace in traces:
            self.save_trace(trace, tier, trace.metadata.get('protocol_name', 'unknown'))


class ShardProcessor:
    """Convert HDF5 raw traces to .pt training shards.

    Extracts fixed-length segments with 50% overlap, converts to float32,
    shuffles, and writes to .pt shard files (~200MB each).
    """

    def __init__(self, raw_dir: str = '/media/norepinephrine/Elements-ext4/surrogate_data/raw',
                 shard_dir: str = '/media/norepinephrine/Elements-ext4/surrogate_data/train',
                 segment_length: int = 1000,
                 shard_size_mb: float = 200.0):
        self.raw_dir = Path(raw_dir)
        self.shard_dir = Path(shard_dir)
        self.segment_length = segment_length
        self.shard_size_mb = shard_size_mb

        # Estimate segments per shard
        bytes_per_segment = segment_length * TraceData.N_COLUMNS * 4  # float32
        self._segments_per_shard = max(1, int(shard_size_mb * 1e6 / bytes_per_segment))

    def process_tier(self, tier: int) -> List[torch.Tensor]:
        """Convert all traces in a tier to segments."""
        storage = TraceStorage(str(self.raw_dir))
        protocols = storage.list_protocols(tier)
        all_segments = []
        for proto_name in protocols:
            trace = storage.load_trace(tier, proto_name)
            segments = self._extract_segments(trace)
            all_segments.extend(segments)
        return all_segments

    def process_all(self, tiers: Optional[List[int]] = None,
                    val_protocols: Optional[List[str]] = None):
        """Full pipeline: HDF5 → segments → shuffled shards.

        Args:
            tiers: which tiers to process (default: all found)
            val_protocols: protocol names held out for validation
        """
        val_protocols = val_protocols or []
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        val_dir = self.shard_dir.parent / 'val'
        val_dir.mkdir(parents=True, exist_ok=True)

        if tiers is None:
            # Find all tier files
            tiers = sorted([
                int(p.stem.replace('tier', ''))
                for p in self.raw_dir.glob('tier*.h5')
            ])

        train_segments = []
        val_segments = []

        for tier in tiers:
            storage = TraceStorage(str(self.raw_dir))
            protocols = storage.list_protocols(tier)
            for proto_name in protocols:
                trace = storage.load_trace(tier, proto_name)
                segments = self._extract_segments(trace)
                if proto_name in val_protocols:
                    val_segments.extend(segments)
                else:
                    train_segments.extend(segments)

        # Shuffle and write train shards
        random.shuffle(train_segments)
        self._write_shards(train_segments, self.shard_dir)

        # Write val shards (no shuffle needed)
        if val_segments:
            self._write_shards(val_segments, val_dir)

    def _extract_segments(self, trace: TraceData) -> List[torch.Tensor]:
        """Extract overlapping segments from a trace.

        Uses 50% overlap (stride = segment_length // 2).
        Note: ~2x segment count vs non-overlapping.
        """
        data = trace.data  # (T, 23)
        segments = []
        stride = max(1, self.segment_length // 2)
        for start in range(0, len(data) - self.segment_length + 1, stride):
            segments.append(data[start:start + self.segment_length])
        return segments

    def _write_shards(self, segments: List[torch.Tensor], output_dir: Path):
        """Stack segments, convert to float32, save as .pt shards."""
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = 0
        for i in range(0, len(segments), self._segments_per_shard):
            batch = segments[i:i + self._segments_per_shard]
            tensor = torch.stack(batch).to(torch.float32)  # (N, seg_len, 23)
            shard_path = output_dir / f'shard_{shard_idx:04d}.pt'
            torch.save(tensor, shard_path)
            shard_idx += 1
