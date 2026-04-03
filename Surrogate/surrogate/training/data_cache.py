"""Data cache builder: HDF5 on HDD -> preprocessed .pt on SSD.

Reads raw 47-col HDF5 tiers via TraceStorage, applies V3Preprocessor,
splits by protocol name into train/val, and saves as .pt files on SSD.

Usage:
    cache = CacheBuilder('/media/HDD/surrogate_data/raw', '/tmp/surrogate_cache')
    cache.build_all(tiers=[1, 2, 3, 12])
    cache.compute_normalization_stats(tiers=[1, 2, 3])
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from ..data.preprocessor import V3Preprocessor

logger = logging.getLogger(__name__)

# Default validation protocols per tier (split by protocol name, not timestep).
# Train on most BCLs, validate on held-out BCLs.
DEFAULT_VAL_PROTOCOLS: dict[int, list[str]] = {
    1: [
        'steady_bcl400_dt0.01',
        'steady_bcl600_dt0.01',
        'steady_bcl800_dt0.01',
        'steady_bcl2000_dt0.01',
    ],
    2: [
        's1s2_di75_dt0.01',
        's1s2_di200_dt0.01',
        's1s2_di800_dt0.01',
    ],
    3: [
        'ramp_300to1000_dt0.01',
    ],
    12: [
        'steady_bcl400_dt0.01',
        'steady_bcl600_dt0.01',
        'steady_bcl800_dt0.01',
        'steady_bcl2000_dt0.01',
        's1s2_di75_dt0.01',
        's1s2_di200_dt0.01',
    ],
}


def _load_trace_data(raw_dir: str, tier: int, protocol_name: str) -> Tensor:
    """Load a single trace from HDF5, returning raw (T, 47) float64 tensor.

    Reads via h5py directly to avoid TraceStorage's stale mount-point check.
    """
    import h5py

    path = Path(raw_dir) / f'tier{tier:02d}.h5'
    with h5py.File(path, 'r') as f:
        data = torch.tensor(f[protocol_name]['data'][:], dtype=torch.float64)
    return data


def _list_protocols(raw_dir: str, tier: int) -> list[str]:
    """List all protocol names in a tier HDF5 file."""
    import h5py

    path = Path(raw_dir) / f'tier{tier:02d}.h5'
    if not path.exists():
        return []
    with h5py.File(path, 'r') as f:
        return list(f.keys())


class CacheBuilder:
    """Build preprocessed .pt cache from raw HDF5 tiers.

    Args:
        raw_dir: Path to HDD raw HDF5 directory (e.g., '/media/HDD/surrogate_data/raw').
        cache_dir: Path to SSD cache directory (e.g., '/tmp/surrogate_cache').
        val_protocols: Per-tier validation protocol names. Defaults to DEFAULT_VAL_PROTOCOLS.
    """

    def __init__(
        self,
        raw_dir: str = '/media/HDD/surrogate_data/raw',
        cache_dir: str = '/tmp/surrogate_cache',
        val_protocols: Optional[dict[int, list[str]]] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.val_protocols = val_protocols or DEFAULT_VAL_PROTOCOLS
        self.preprocessor = V3Preprocessor()

        if not self.raw_dir.exists():
            raise RuntimeError(
                f"Raw data directory not found: {self.raw_dir}\n"
                f"Is the HDD mounted? Check /media/HDD/"
            )

    def build_tier_cache(self, tier: int) -> dict[str, Path]:
        """Preprocess one tier -> train.pt + val.pt on SSD.

        Returns dict with 'train' and 'val' paths.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        protocols = _list_protocols(str(self.raw_dir), tier)
        if not protocols:
            raise ValueError(f"No protocols found in tier {tier}")

        val_protos = set(self.val_protocols.get(tier, []))

        # Validate protocol names
        for vp in val_protos:
            if vp not in protocols:
                logger.warning(
                    f"Val protocol '{vp}' not found in tier {tier}. "
                    f"Available: {protocols}"
                )

        train_chunks: dict[str, list[Tensor]] = {}
        val_chunks: dict[str, list[Tensor]] = {}

        for proto in protocols:
            logger.info(f"Processing tier {tier} / {proto}")
            raw = _load_trace_data(str(self.raw_dir), tier, proto)
            processed = self.preprocessor.process_segment(raw)

            target = val_chunks if proto in val_protos else train_chunks
            for key, tensor in processed.items():
                target.setdefault(key, []).append(tensor)

        paths = {}
        for split_name, chunks in [('train', train_chunks), ('val', val_chunks)]:
            if not chunks:
                continue
            merged = {k: torch.cat(v, dim=0).float() for k, v in chunks.items()}
            # Add metadata
            merged['_n_timesteps'] = torch.tensor(merged['Vm'].shape[0])
            merged['_tier'] = torch.tensor(tier)
            path = self.cache_dir / f'tier{tier:02d}_{split_name}.pt'
            torch.save(merged, path)
            paths[split_name] = path
            logger.info(
                f"Saved {split_name}: {path} "
                f"({merged['Vm'].shape[0]} timesteps, "
                f"{path.stat().st_size / 1e6:.0f} MB)"
            )

        return paths

    def build_all(self, tiers: Optional[list[int]] = None) -> dict[int, dict[str, Path]]:
        """Build cache for multiple tiers. Returns {tier: {split: path}}."""
        if tiers is None:
            tiers = [1, 2, 3, 12]
        results = {}
        for tier in tiers:
            results[tier] = self.build_tier_cache(tier)
        return results

    def is_cached(self, tiers: Optional[list[int]] = None) -> bool:
        """Check if all tiers are already cached on SSD."""
        if tiers is None:
            tiers = [1, 2, 3, 12]
        for tier in tiers:
            train_path = self.cache_dir / f'tier{tier:02d}_train.pt'
            if not train_path.exists():
                return False
        return True

    def compute_normalization_stats(self, tiers: Optional[list[int]] = None) -> dict:
        """Compute environment token normalization stats from cached train data.

        Stats for 9 tokens: [Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss].
        Saved to cache_dir/norm_stats.pt.
        """
        if tiers is None:
            tiers = [1, 2, 3]

        all_env = []
        for tier in tiers:
            data = self.load_tier(tier, 'train')
            env = torch.stack([
                data['Vm'],
                data['E'][:, 0],  # E_Na
                data['E'][:, 1],  # E_K
                data['E'][:, 2],  # E_Ca
                data['E'][:, 3],  # E_Ks
                data['concentrations'][:, 0],  # Na_i
                data['concentrations'][:, 1],  # K_i
                data['concentrations'][:, 2],  # Ca_i
                data['concentrations'][:, 3],  # Ca_ss
            ], dim=-1)
            all_env.append(env)

        all_env = torch.cat(all_env, dim=0)

        env_min = all_env.min(dim=0).values
        env_max = all_env.max(dim=0).values
        env_mean = all_env.mean(dim=0)
        env_std = all_env.std(dim=0)

        midpoint = (env_min + env_max) / 2
        halfrange = (env_max - env_min) / 2
        halfrange = halfrange.clamp(min=1e-8)

        stats = {
            'shift': midpoint,
            'scale': halfrange,
            'min': env_min,
            'max': env_max,
            'mean': env_mean,
            'std': env_std,
            'token_names': [
                'Vm', 'E_Na', 'E_K', 'E_Ca', 'E_Ks',
                'Na_i', 'K_i', 'Ca_i', 'Ca_ss',
            ],
        }
        path = self.cache_dir / 'norm_stats.pt'
        torch.save(stats, path)
        logger.info(f"Normalization stats saved to {path}")
        return stats

    def load_tier(self, tier: int, split: str = 'train') -> dict[str, Tensor]:
        """Load preprocessed tier from cache. Returns dict of named tensors."""
        path = self.cache_dir / f'tier{tier:02d}_{split}.pt'
        if not path.exists():
            raise FileNotFoundError(
                f"Cache not found: {path}. Run build_tier_cache({tier}) first."
            )
        return torch.load(path, weights_only=False)
