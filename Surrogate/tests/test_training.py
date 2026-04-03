"""Tests for the training pipeline: cache builder, datasets, encoder, phases, rollout, trainer."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


# ============================================================================
# Helpers
# ============================================================================

def _make_fake_47col(T=200, seed=42):
    """Create fake 47-col data with physiological-ish values."""
    rng = np.random.RandomState(seed)
    data = np.zeros((T, 47), dtype=np.float64)
    data[:, 0] = np.linspace(-85, 30, T)        # Vm
    data[:, 1] = 0.0                              # I_stim
    data[:, 2] = 0.01                             # dt
    data[:, 3] = 138.0                            # Ki
    data[:, 4] = 10.0                             # Nai
    data[:, 5] = 0.0001                           # Cai
    data[:, 6] = 1.5                              # CaSR
    data[:, 7] = 0.0002                           # CaSS
    for i in range(5, 18):
        data[:, 3 + i] = rng.uniform(0.01, 0.99, T)
    data[:, 21] = rng.uniform(-10, 5, T)         # I_ion
    data[:, 22] = 0.0                              # clamp_mask
    data[:, 23:35] = rng.uniform(0.0, 1.0, (T, 12))  # gate_inf
    data[:, 35:47] = rng.uniform(0.1, 50.0, (T, 12))  # gate_tau
    return data


def _make_fake_tier_h5(path, protocols, T=200):
    """Create a fake tier HDF5 file with named protocols."""
    with h5py.File(path, 'w') as f:
        for i, proto_name in enumerate(protocols):
            data = _make_fake_47col(T=T, seed=42 + i)
            grp = f.create_group(proto_name)
            grp.create_dataset('data', data=data, dtype=np.float64)
            grp.attrs['protocol_name'] = proto_name
            grp.attrs['protocol_tier'] = 1
            grp.attrs['cell_type'] = 'EPI'


# ============================================================================
# Phase 1: Cache Builder Tests
# ============================================================================

class TestCacheBuilder:

    def test_cache_builder_creates_files(self, tmp_path):
        """Cache builder creates train.pt and val.pt files."""
        from surrogate.training.data_cache import CacheBuilder

        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        cache_dir = tmp_path / 'cache'

        _make_fake_tier_h5(
            raw_dir / 'tier01.h5',
            ['steady_bcl300_dt0.01', 'steady_bcl400_dt0.01', 'steady_bcl500_dt0.01'],
        )

        cache = CacheBuilder(
            raw_dir=str(raw_dir),
            cache_dir=str(cache_dir),
            val_protocols={1: ['steady_bcl400_dt0.01']},
        )
        paths = cache.build_tier_cache(1)

        assert 'train' in paths
        assert 'val' in paths
        assert Path(paths['train']).exists()
        assert Path(paths['val']).exists()

    def test_cache_builder_split(self, tmp_path):
        """Val protocols go to val.pt, rest to train.pt."""
        from surrogate.training.data_cache import CacheBuilder

        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        cache_dir = tmp_path / 'cache'

        T = 100
        _make_fake_tier_h5(
            raw_dir / 'tier01.h5',
            ['proto_a', 'proto_b', 'proto_c'],
            T=T,
        )

        cache = CacheBuilder(
            raw_dir=str(raw_dir),
            cache_dir=str(cache_dir),
            val_protocols={1: ['proto_b']},
        )
        cache.build_tier_cache(1)

        train = cache.load_tier(1, 'train')
        val = cache.load_tier(1, 'val')

        # Train should have 2 protocols * T timesteps, val should have 1 * T
        assert train['Vm'].shape[0] == 2 * T
        assert val['Vm'].shape[0] == 1 * T

    def test_cache_builder_shapes(self, tmp_path):
        """All tensors have correct shapes and consistent T dimension."""
        from surrogate.training.data_cache import CacheBuilder

        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        cache_dir = tmp_path / 'cache'
        T = 150

        _make_fake_tier_h5(raw_dir / 'tier01.h5', ['proto_a'], T=T)

        cache = CacheBuilder(
            raw_dir=str(raw_dir),
            cache_dir=str(cache_dir),
            val_protocols={1: []},
        )
        cache.build_tier_cache(1)
        data = cache.load_tier(1, 'train')

        assert data['Vm'].shape == (T,)
        assert data['dt'].shape == (T,)
        assert data['I_ion'].shape == (T,)
        assert data['concentrations'].shape == (T, 4)
        assert data['ionic_states'].shape == (T, 14)
        assert data['conductance_products'].shape == (T, 5)
        assert data['E'].shape == (T, 4)
        assert data['gates'].shape == (T, 12)
        # float32 storage
        assert data['Vm'].dtype == torch.float32

    def test_normalization_stats(self, tmp_path):
        """Normalization stats have correct shape and reasonable values."""
        from surrogate.training.data_cache import CacheBuilder

        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        cache_dir = tmp_path / 'cache'

        _make_fake_tier_h5(raw_dir / 'tier01.h5', ['proto_a', 'proto_b'])

        cache = CacheBuilder(
            raw_dir=str(raw_dir),
            cache_dir=str(cache_dir),
            val_protocols={1: []},
        )
        cache.build_tier_cache(1)
        stats = cache.compute_normalization_stats(tiers=[1])

        assert stats['shift'].shape == (9,)
        assert stats['scale'].shape == (9,)
        assert (stats['scale'] > 0).all()
        assert (stats['max'] >= stats['min']).all()
        assert (tmp_path / 'cache' / 'norm_stats.pt').exists()

    def test_is_cached(self, tmp_path):
        """is_cached returns correct status."""
        from surrogate.training.data_cache import CacheBuilder

        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        cache_dir = tmp_path / 'cache'

        _make_fake_tier_h5(raw_dir / 'tier01.h5', ['proto_a'])

        cache = CacheBuilder(
            raw_dir=str(raw_dir),
            cache_dir=str(cache_dir),
            val_protocols={1: []},
        )

        assert not cache.is_cached(tiers=[1])
        cache.build_tier_cache(1)
        assert cache.is_cached(tiers=[1])
        assert not cache.is_cached(tiers=[1, 2])

    def test_val_protocol_warning(self, tmp_path, caplog):
        """Warning issued when val protocol not found in tier."""
        import logging
        from surrogate.training.data_cache import CacheBuilder

        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        cache_dir = tmp_path / 'cache'

        _make_fake_tier_h5(raw_dir / 'tier01.h5', ['proto_a'])

        cache = CacheBuilder(
            raw_dir=str(raw_dir),
            cache_dir=str(cache_dir),
            val_protocols={1: ['nonexistent_protocol']},
        )
        with caplog.at_level(logging.WARNING):
            cache.build_tier_cache(1)
        assert 'nonexistent_protocol' in caplog.text


# ============================================================================
# Phase 1: Dataset Tests
# ============================================================================

class TestDatasets:

    def _make_cached_data(self, T=500):
        """Create fake cached data dict matching CacheBuilder output format."""
        return {
            'Vm': torch.randn(T).float(),
            'dt': torch.full((T,), 0.01).float(),
            'I_stim': torch.zeros(T).float(),
            'I_ion': torch.randn(T).float(),
            'clamp_mask': torch.zeros(T).float(),
            'concentrations': torch.randn(T, 4).float(),
            'gates': torch.rand(T, 12).float(),
            'ionic_states': torch.randn(T, 14).float(),
            'conductance_products': torch.rand(T, 5).float(),
            'E': torch.randn(T, 4).float(),
            'gate_inf': torch.rand(T, 12).float(),
            'gate_tau': torch.rand(T, 12).float(),
        }

    def test_snapshot_dataset_shapes(self):
        from surrogate.training.datasets import SnapshotDataset

        data = self._make_cached_data(T=100)
        ds = SnapshotDataset(data)

        assert len(ds) == 100
        sample = ds[0]
        assert sample['ionic_states'].shape == (14,)
        assert sample['concentrations'].shape == (4,)
        assert sample['conductance_products'].shape == (5,)
        assert sample['Vm'].shape == ()
        assert sample['ionic_states'].dtype == torch.float64

    def test_pair_dataset_consecutive(self):
        from surrogate.training.datasets import PairDataset

        data = self._make_cached_data(T=100)
        ds = PairDataset(data)

        assert len(ds) == 99  # T - 1
        sample = ds[5]
        # t+1 concentration should be the next timestep
        expected = data['concentrations'][6].double()
        assert torch.allclose(sample['concentrations_t1'], expected)

    def test_segment_dataset_contiguous(self):
        from surrogate.training.datasets import SegmentDataset

        data = self._make_cached_data(T=500)
        ds = SegmentDataset(data, segment_length=50)

        sample = ds[0]
        assert sample['Vm'].shape == (50,)
        assert sample['ionic_states'].shape == (50, 14)
        assert sample['concentrations'].shape == (50, 4)
        # Verify contiguous slice
        expected_vm = data['Vm'][:50].double()
        assert torch.allclose(sample['Vm'], expected_vm)

    def test_segment_dataset_overlap(self):
        from surrogate.training.datasets import SegmentDataset

        T = 500
        seg_len = 100
        data = self._make_cached_data(T=T)
        ds = SegmentDataset(data, segment_length=seg_len)  # default 50% overlap

        # stride = 50, so (500 - 100) / 50 + 1 = 9 segments
        expected = (T - seg_len) // 50 + 1
        assert len(ds) == expected

    def test_segment_dataset_custom_stride(self):
        from surrogate.training.datasets import SegmentDataset

        data = self._make_cached_data(T=500)
        ds = SegmentDataset(data, segment_length=100, stride=100)  # no overlap

        expected = 500 // 100
        assert len(ds) == expected

    def test_segment_dataset_float64(self):
        from surrogate.training.datasets import SegmentDataset

        data = self._make_cached_data(T=200)
        ds = SegmentDataset(data, segment_length=50)
        sample = ds[0]
        for key, tensor in sample.items():
            assert tensor.dtype == torch.float64, f"{key} is {tensor.dtype}, expected float64"

    def test_merge_tier_datasets(self):
        from surrogate.training.datasets import SnapshotDataset, merge_tier_datasets

        d1 = self._make_cached_data(T=100)
        d2 = self._make_cached_data(T=200)
        ds1 = SnapshotDataset(d1)
        ds2 = SnapshotDataset(d2)

        merged = merge_tier_datasets([ds1, ds2])
        assert len(merged) == 300
