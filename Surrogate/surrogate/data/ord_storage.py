"""ORd-specific HDF5 storage and shard processing.

Mirrors storage.py but uses ORdTraceData (101 columns) instead of
TTP06 TraceData (47 columns). Separate raw directories, separate shards.
"""

from pathlib import Path
from typing import List

import h5py
import torch
import numpy as np

from .ord_trace_data import ORdTraceData
from .storage import TraceStorage, ShardProcessor


class ORdTraceStorage(TraceStorage):
    """HDF5 storage for ORd traces (101 columns)."""

    def __init__(self, base_dir: str = '/media/HDD/surrogate_data/raw_ord'):
        # Skip parent __init__ to avoid TTP06 mount check
        self.base_dir = Path(base_dir)
        self._check_storage()

    def _check_storage(self):
        """Verify storage directory is accessible."""
        if '/media/' in str(self.base_dir):
            mount_point = Path('/media/HDD')
            if not mount_point.is_dir():
                raise RuntimeError(
                    f"External HDD not mounted at {mount_point}. "
                    f"Mount it before running data generation."
                )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_trace(self, trace: ORdTraceData, tier: int, protocol_name: str):
        """Append ORd trace to tier HDF5 file."""
        path = self.base_dir / f'tier{tier:02d}.h5'
        with h5py.File(path, 'a') as f:
            if protocol_name in f:
                del f[protocol_name]
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

    def load_trace(self, tier: int, protocol_name: str) -> ORdTraceData:
        """Load a single ORd trace — returns ORdTraceData, NOT TTP06 TraceData."""
        path = self.base_dir / f'tier{tier:02d}.h5'
        with h5py.File(path, 'r') as f:
            data = torch.tensor(f[protocol_name]['data'][:],
                                dtype=torch.float64)
            metadata = dict(f[protocol_name].attrs)
        return ORdTraceData(data=data, metadata=metadata)


class ORdShardProcessor(ShardProcessor):
    """Shard processor for ORd 101-column data."""

    def __init__(self, raw_dir: str = '/media/HDD/surrogate_data/raw_ord',
                 shard_dir: str = '/media/HDD/surrogate_data/train_ord',
                 segment_length: int = 1000,
                 shard_size_mb: float = 200.0):
        # Don't call super().__init__ — it uses TTP06's N_COLUMNS
        self.raw_dir = Path(raw_dir)
        self.shard_dir = Path(shard_dir)
        self.segment_length = segment_length
        self.shard_size_mb = shard_size_mb
        # Use ORd column count for size estimation
        bytes_per_segment = segment_length * ORdTraceData.N_COLUMNS * 4  # float32
        self._segments_per_shard = max(1, int(shard_size_mb * 1e6 / bytes_per_segment))

    def process_tier(self, tier: int) -> List[torch.Tensor]:
        """Process a single tier using ORdTraceStorage."""
        storage = ORdTraceStorage(str(self.raw_dir))
        protocols = storage.list_protocols(tier)
        all_segments = []
        for proto_name in protocols:
            trace = storage.load_trace(tier, proto_name)
            segments = self._extract_segments(trace)
            all_segments.extend(segments)
        return all_segments

    def process_all(self, val_fraction: float = 0.1):
        """Process all tiers using ORdTraceStorage (NOT TTP06 TraceStorage).

        Overrides parent which hardcodes TTP06 TraceStorage.
        """
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        all_segments = []
        for tier_file in sorted(self.raw_dir.glob('tier*.h5')):
            tier = int(tier_file.stem.replace('tier', ''))
            segments = self.process_tier(tier)
            all_segments.extend(segments)

        # Shuffle and split train/val
        import random
        random.shuffle(all_segments)
        n_val = max(1, int(len(all_segments) * val_fraction))
        val_segments = all_segments[:n_val]
        train_segments = all_segments[n_val:]

        # Write shards
        train_dir = self.shard_dir / 'train'
        val_dir = self.shard_dir / 'val'
        self._write_shards(train_segments, str(train_dir))
        self._write_shards(val_segments, str(val_dir))
