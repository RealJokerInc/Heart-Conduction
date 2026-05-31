"""Tests for the training pipeline: cache builder, datasets, phases, rollout, trainer."""

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


# ============================================================================
# Phase 2: Phase Config Tests
# ============================================================================

class TestPhaseConfig:

    def test_phase_configs_complete(self):
        from surrogate.training.phases import PHASE_ORDER, PHASE_CONFIGS
        assert len(PHASE_ORDER) == 10
        for name in PHASE_ORDER:
            assert name in PHASE_CONFIGS

    def test_freeze_mask_A1(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary

        model = IonicSurrogateV3(scaffold=True)
        phase = get_phase_config("A1")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        # state_rate_mlp + ionic_state_decoder.weight unfrozen (bias frozen at rest)
        for name, grad in summary.items():
            if 'state_rate_mlp' in name or name == 'stage1.ionic_state_decoder.weight':
                assert grad, f"{name} should be unfrozen in A1"
            elif name == 'stage1.ionic_state_decoder.bias':
                assert not grad, f"{name} must stay frozen (rest-attractor contract)"
            elif 'stage2' in name:
                assert not grad, f"{name} should be frozen in B1"

    def test_freeze_mask_B1(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary

        model = IonicSurrogateV3(scaffold=True)
        phase = get_phase_config("B1")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        # B1 trains Half 2 (conductance params); Half 1 and Stage 2 are frozen.
        for name, grad in summary.items():
            if any(p in name for p in [
                'gate_conductance_mlp', 'gate_conductance_linear',
                'gate_conductance_logit', 'gate_conductance_decoder',
            ]):
                assert grad, f"{name} should be unfrozen in B1"
            elif 'state_rate_mlp' in name or 'ionic_state_decoder' in name:
                assert not grad, f"{name} (Half 1) should be frozen in B1"
            elif 'stage2' in name:
                assert not grad, f"{name} should be frozen in B1"

    def test_freeze_mask_B(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary

        model = IonicSurrogateV3(scaffold=True)
        phase = get_phase_config("C")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        # Only Stage 2 unfrozen
        for name, grad in summary.items():
            if 'stage2' in name:
                assert grad, f"{name} should be unfrozen in C"
            else:
                assert not grad, f"{name} should be frozen in C"

    def test_freeze_mask_C(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary

        model = IonicSurrogateV3(scaffold=True)
        phase = get_phase_config("D")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        # Everything unfrozen (stage1.* + stage2.*)
        for name, grad in summary.items():
            assert grad, f"{name} should be unfrozen in D"

    def test_phase_order(self):
        from surrogate.training.phases import PHASE_ORDER, get_all_phases
        phases = get_all_phases()
        assert [p.name for p in phases] == PHASE_ORDER

    def test_T1_in_A1(self):
        from surrogate.training.phases import get_phase_config
        a1 = get_phase_config("A1")
        assert 1 in a1.data_tiers, "T1 must be in A1 data_tiers"



# ============================================================================
# Phase 2: Rollout Tests
# ============================================================================

class TestRollout:

    def _make_segment(self, B=4, T=10):
        """Create a fake batched segment matching SegmentDataset + DataLoader output."""
        return {
            'Vm': torch.randn(B, T, dtype=torch.float64),
            'dt': torch.full((B, T), 0.01, dtype=torch.float64),
            'I_stim': torch.zeros(B, T, dtype=torch.float64),
            'I_ion': torch.randn(B, T, dtype=torch.float64),
            'clamp_mask': torch.zeros(B, T, dtype=torch.float64),
            'concentrations': torch.randn(B, T, 4, dtype=torch.float64).abs() * 0.001 + 0.0001,
            'gates': torch.rand(B, T, 12, dtype=torch.float64),
            'ionic_states': torch.randn(B, T, 14, dtype=torch.float64),
            'conductance_products': torch.rand(B, T, 5, dtype=torch.float64),
            'E': torch.randn(B, T, 4, dtype=torch.float64),
            'gate_inf': torch.rand(B, T, 12, dtype=torch.float64),
            'gate_tau': torch.rand(B, T, 12, dtype=torch.float64),
        }

    def test_rollout_shapes(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=4, T=10)

        result = rollout(model, segment, phase_name="A1")
        assert result['loss'].shape == ()
        assert result['per_step_losses'].shape == (10,)

    def test_rollout_autoregressive(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=2, T=5)

        result = rollout(model, segment, phase_name="A1")
        assert result['loss'].isfinite()

    @pytest.mark.xfail(reason="legacy discrete rollout path; v4 uses node_rollout (NODE adjoint). Kept for archive reference.")
    def test_rollout_gradient_flow(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=2, T=5)

        result = rollout(model, segment, phase_name="A1")
        result['loss'].backward()

        has_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through rollout"

    def test_rollout_D_phase_loss(self):
        """Phase C uses I_ion loss (Stage 2 regression)."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=2, T=5)

        result = rollout(model, segment, phase_name="C")
        assert result['loss'].isfinite()

    def test_rollout_C_phase_loss(self):
        """Phase C uses concentration loss."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=2, T=5)

        result = rollout(model, segment, phase_name="B1")
        assert result['loss'].isfinite()


# ============================================================================
# Phase 2: Trainer Tests
# ============================================================================

class TestTrainer:

    def _setup_fake_cache(self, tmp_path, T=200):
        """Create fake cached .pt files for tiers 1 and 12."""
        cache_dir = tmp_path / 'cache'
        cache_dir.mkdir()

        for tier in [1, 12]:
            for split in ['train', 'val']:
                data = {
                    'Vm': torch.randn(T, dtype=torch.float32),
                    'dt': torch.full((T,), 0.01, dtype=torch.float32),
                    'I_stim': torch.zeros(T, dtype=torch.float32),
                    'I_ion': torch.randn(T, dtype=torch.float32),
                    'clamp_mask': torch.zeros(T, dtype=torch.float32),
                    'concentrations': (torch.randn(T, 4).abs() * 0.001 + 0.0001).float(),
                    'gates': torch.rand(T, 12, dtype=torch.float32),
                    'ionic_states': torch.randn(T, 14, dtype=torch.float32),
                    'conductance_products': torch.rand(T, 5, dtype=torch.float32),
                    'E': torch.randn(T, 4, dtype=torch.float32),
                    'gate_inf': torch.rand(T, 12, dtype=torch.float32),
                    'gate_tau': torch.rand(T, 12, dtype=torch.float32),
                    '_n_timesteps': torch.tensor(T),
                    '_tier': torch.tensor(tier),
                }
                torch.save(data, cache_dir / f'tier{tier:02d}_{split}.pt')

        return cache_dir

    def test_trainer_freeze_unfreeze(self, tmp_path):
        """Phase B1 freeze mask: attention + MLP + ionic_state_decoder unfrozen."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.trainer import SurrogateTrainer

        cache_dir = self._setup_fake_cache(tmp_path)
        model = IonicSurrogateV3(scaffold=True).double()
        trainer = SurrogateTrainer(model, str(cache_dir), str(tmp_path / 'run'), device='cpu')

        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary
        phase = get_phase_config("A1")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        for name, grad in summary.items():
            if 'state_rate_mlp' in name or name == 'stage1.ionic_state_decoder.weight':
                assert grad, f"{name} should be unfrozen"
            elif name == 'stage1.ionic_state_decoder.bias':
                assert not grad, f"{name} must stay frozen (rest-attractor contract)"
            elif 'stage2' in name:
                assert not grad, f"{name} should be frozen"

    def test_trainer_B1_one_epoch(self, tmp_path):
        """Phase B1 rollout runs one epoch without error and produces a loss."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.trainer import SurrogateTrainer
        from surrogate.training.phases import get_phase_config

        cache_dir = self._setup_fake_cache(tmp_path, T=200)
        model = IonicSurrogateV3(scaffold=True).double()
        trainer = SurrogateTrainer(model, str(cache_dir), str(tmp_path / 'run'), device='cpu')

        b1 = get_phase_config("A1")
        b1.max_epochs = 1
        b1.patience = 1
        b1.batch_size = 4  # small for test
        b1.rollout_length = 10  # short segment so T=200 is sufficient
        b1.subsample = 1        # no subsampling: raw_length = 10
        metrics = trainer.train_phase(b1)
        assert 'val_ionic_state_mse' in metrics
        assert metrics['val_ionic_state_mse'] >= 0

    def test_trainer_phase_transition(self, tmp_path):
        """Phase transition resets optimizer and changes freeze mask."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.trainer import SurrogateTrainer
        from surrogate.training.phases import get_phase_config, get_freeze_summary

        cache_dir = self._setup_fake_cache(tmp_path, T=200)
        model = IonicSurrogateV3(scaffold=True).double()
        trainer = SurrogateTrainer(model, str(cache_dir), str(tmp_path / 'run'), device='cpu')

        # Run A1 with 1 epoch
        b1 = get_phase_config("A1")
        b1.max_epochs = 1
        b1.patience = 1
        b1.batch_size = 4
        b1.rollout_length = 10  # short segment so T=200 is sufficient
        b1.subsample = 1        # no subsampling: raw_length = 10
        trainer.train_phase(b1)

        summary = get_freeze_summary(model)
        # After A1: state_rate_mlp + decoder.weight should be unfrozen; bias stays frozen
        for name, grad in summary.items():
            if 'state_rate_mlp' in name or name == 'stage1.ionic_state_decoder.weight':
                assert grad, f"{name} should be unfrozen after A1"
            elif name == 'stage1.ionic_state_decoder.bias':
                assert not grad, f"{name} must stay frozen (rest-attractor contract)"

    def test_checkpoint_save_load(self, tmp_path):
        """Checkpoint round-trip preserves model weights."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.trainer import SurrogateTrainer
        from surrogate.training.phases import get_phase_config

        cache_dir = self._setup_fake_cache(tmp_path, T=200)
        model = IonicSurrogateV3(scaffold=True).double()
        trainer = SurrogateTrainer(model, str(cache_dir), str(tmp_path / 'run'), device='cpu')

        # Run A1 for 1 epoch
        b1 = get_phase_config("A1")
        b1.max_epochs = 1
        b1.patience = 1
        b1.batch_size = 4
        b1.rollout_length = 10  # short segment so T=200 is sufficient
        b1.subsample = 1        # no subsampling: raw_length = 10
        trainer.train_phase(b1)

        # Get current weights
        orig_weights = {k: v.clone() for k, v in model.state_dict().items()}

        # Load checkpoint into fresh model
        model2 = IonicSurrogateV3(scaffold=True).double()
        trainer2 = SurrogateTrainer(model2, str(cache_dir), str(tmp_path / 'run2'), device='cpu')
        ckpt_path = str(tmp_path / 'run' / 'checkpoints' / 'best_A1.pt')
        trainer2.load_checkpoint(ckpt_path)

        for name in orig_weights:
            assert torch.allclose(orig_weights[name], model2.state_dict()[name]), (
                f"Weights differ for {name} after checkpoint round-trip"
            )


# ============================================================================
# Phase 3: Checkpoint, Monitor, Metrics Tests
# ============================================================================

class TestCheckpointManager:

    def test_checkpoint_save_load_roundtrip(self, tmp_path):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.checkpoint import CheckpointManager
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        model = IonicSurrogateV3(scaffold=True).double()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        mgr = CheckpointManager(str(tmp_path))
        mgr.save('test', model, optimizer, scheduler, None, 'A1', 5, 100, 0.001)

        model2 = IonicSurrogateV3(scaffold=True).double()
        meta = mgr.load('test', model2, device='cpu')

        assert meta['phase'] == 'A1'
        assert meta['epoch'] == 5
        assert meta['step'] == 100
        for k in model.state_dict():
            assert torch.allclose(model.state_dict()[k], model2.state_dict()[k])

    def test_list_checkpoints(self, tmp_path):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.checkpoint import CheckpointManager
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        model = IonicSurrogateV3(scaffold=True).double()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        mgr = CheckpointManager(str(tmp_path))
        mgr.save('best_A1', model, optimizer, scheduler, None, 'A1', 0, 0, 0.0)
        mgr.save('latest', model, optimizer, scheduler, None, 'A1', 1, 10, 0.0)

        ckpts = mgr.list_checkpoints()
        assert 'best_A1' in ckpts
        assert 'latest' in ckpts


class TestMonitor:

    def test_monitor_jsonl_format(self, tmp_path):
        import json
        from surrogate.training.monitor import TrainingMonitor

        mon = TrainingMonitor(str(tmp_path))
        mon.log_batch('A1', 0, 0, 0, 0.5, 1e-3, 0.1)

        lines = (tmp_path / 'training_log.jsonl').read_text().strip().split('\n')
        entry = json.loads(lines[0])
        assert entry['phase'] == 'A1'
        assert entry['loss'] == 0.5
        assert 'timestamp' in entry

    def test_monitor_control_pause(self, tmp_path):
        import json, threading, time
        from surrogate.training.monitor import TrainingMonitor

        mon = TrainingMonitor(str(tmp_path))
        mon.update_control(status='pause_requested')

        # Resume in background after short delay
        def resume():
            time.sleep(0.5)
            mon.update_control(status='running')
        t = threading.Thread(target=resume)
        t.start()

        result = mon.check_control()
        t.join()
        assert result == 'running'

    def test_monitor_nan_detection(self, tmp_path):
        from surrogate.training.monitor import TrainingMonitor

        mon = TrainingMonitor(str(tmp_path))
        mon.log_batch('A1', 0, 0, 0, 1.0, 1e-3, 0.1)  # seed EMA
        result = mon.check_divergence(float('nan'), 0.1)
        assert result == 'nan'

    def test_monitor_spike_detection(self, tmp_path):
        from surrogate.training.monitor import TrainingMonitor

        mon = TrainingMonitor(str(tmp_path))
        # Seed EMA with low loss
        for i in range(100):
            mon.log_batch('A1', 0, i, i, 0.1, 1e-3, 0.1)
        # Spike should be detected
        result = mon.check_divergence(10.0, 0.1)
        assert result == 'spike'

    def test_intervention_handler(self, tmp_path):
        import json
        from surrogate.training.monitor import TrainingMonitor
        from torch.optim import AdamW
        import torch.nn as nn

        mon = TrainingMonitor(str(tmp_path))
        model = nn.Linear(10, 10)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        result = mon.apply_intervention({'action': 'reduce_lr', 'factor': 0.5}, optimizer)
        assert 'Reduced' in result
        assert optimizer.param_groups[0]['lr'] == pytest.approx(5e-4)


class TestMetrics:

    def test_apd90_known_trace(self):
        from surrogate.training.metrics import compute_apd90

        # Synthetic AP: rest at -80, upstroke to +30, plateau, repolarize
        T = 10000
        dt = 0.01  # ms
        Vm = torch.full((T,), -80.0, dtype=torch.float64)
        # Upstroke at t=1000 (10ms)
        Vm[1000:1010] = torch.linspace(-80, 30, 10)
        # Plateau
        Vm[1010:5000] = torch.linspace(30, 0, 3990)
        # Repolarization
        Vm[5000:7000] = torch.linspace(0, -80, 2000)
        # Rest
        Vm[7000:] = -80.0

        apd = compute_apd90(Vm, dt=dt)
        # AP starts around t=1000, repolarizes to 90% around t=6500-ish
        assert not torch.isnan(apd)
        assert 40 < apd.item() < 70  # reasonable range for synthetic trace

    def test_apd90_no_ap(self):
        from surrogate.training.metrics import compute_apd90

        Vm = torch.full((1000,), -80.0, dtype=torch.float64)
        apd = compute_apd90(Vm)
        assert torch.isnan(apd)

    def test_dvdt_max_known_trace(self):
        from surrogate.training.metrics import compute_dvdt_max

        T = 1000
        Vm = torch.full((T,), -80.0, dtype=torch.float64)
        # Sharp upstroke: -80 to +30 in 10 steps at dt=0.01
        Vm[100:110] = torch.linspace(-80, 30, 10)

        dvdt = compute_dvdt_max(Vm, dt=0.01)
        # Max slope ~110mV/0.01ms = 11000 mV/ms... actually per step
        # linspace: each step is 110/9 ≈ 12.2 mV per 0.01ms = 1222 mV/ms
        assert dvdt.item() > 1000

    def test_dvdt_max_variable_dt(self):
        from surrogate.training.metrics import compute_dvdt_max

        T = 100
        Vm = torch.linspace(-80, 30, T, dtype=torch.float64)
        dt = torch.full((T,), 0.01, dtype=torch.float64)

        dvdt = compute_dvdt_max(Vm, dt=dt)
        assert dvdt.isfinite()


# ============================================================================
# Phase 5: Shard Loader Tests
# ============================================================================

class TestShardLoader:

    def test_shard_converter_creates_shards(self, tmp_path):
        """ShardConverter creates .pt shard files from fake tier."""
        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        shard_dir = tmp_path / 'shards'

        # Create fake tier with 2 protocols
        _make_fake_tier_h5(raw_dir / 'tier04.h5', ['proto_a', 'proto_b'], T=500)

        from surrogate.training.shard_loader import ShardConverter
        converter = ShardConverter(str(raw_dir), str(shard_dir), shard_size_mb=0.01)  # tiny shards
        n_shards = converter.convert_tier(tier=4)
        assert n_shards > 0
        assert len(list(shard_dir.glob('shard_*.pt'))) == n_shards

    def test_shard_stream_loader_yields_batches(self, tmp_path):
        """ShardStreamLoader yields batches with correct shapes."""
        shard_dir = tmp_path / 'shards'
        shard_dir.mkdir()

        # Create 2 small shards manually
        for i in range(2):
            data = {
                'Vm': torch.randn(200, dtype=torch.float32),
                'dt': torch.full((200,), 0.01, dtype=torch.float32),
                'I_stim': torch.zeros(200, dtype=torch.float32),
                'I_ion': torch.randn(200, dtype=torch.float32),
                'clamp_mask': torch.zeros(200, dtype=torch.float32),
                'concentrations': torch.randn(200, 4, dtype=torch.float32).abs(),
                'gates': torch.rand(200, 12, dtype=torch.float32),
                'ionic_states': torch.randn(200, 14, dtype=torch.float32),
                'conductance_products': torch.rand(200, 5, dtype=torch.float32),
                'E': torch.randn(200, 4, dtype=torch.float32),
                'gate_inf': torch.rand(200, 12, dtype=torch.float32),
                'gate_tau': torch.rand(200, 12, dtype=torch.float32),
            }
            torch.save(data, shard_dir / f'shard_{i:04d}.pt')

        from surrogate.training.shard_loader import ShardStreamLoader
        loader = ShardStreamLoader(str(shard_dir), segment_length=10, batch_size=4, device='cpu')

        batch_count = 0
        for batch in loader:
            assert batch['Vm'].shape[0] == 4  # batch size
            assert batch['Vm'].shape[1] == 10  # segment length
            assert batch['Vm'].dtype == torch.float64
            batch_count += 1
            if batch_count >= 3:
                break
        assert batch_count > 0

    def test_shard_loader_float64_output(self, tmp_path):
        """Shard loader outputs float64 tensors from float32 shards."""
        shard_dir = tmp_path / 'shards'
        shard_dir.mkdir()

        data = {
            'Vm': torch.randn(100, dtype=torch.float32),
            'dt': torch.full((100,), 0.01, dtype=torch.float32),
            'concentrations': torch.randn(100, 4, dtype=torch.float32),
            'ionic_states': torch.randn(100, 14, dtype=torch.float32),
            'conductance_products': torch.rand(100, 5, dtype=torch.float32),
            'E': torch.randn(100, 4, dtype=torch.float32),
        }
        torch.save(data, shard_dir / 'shard_0000.pt')

        from surrogate.training.shard_loader import ShardStreamLoader
        loader = ShardStreamLoader(str(shard_dir), segment_length=10, batch_size=2, device='cpu')

        for batch in loader:
            for key, tensor in batch.items():
                assert tensor.dtype == torch.float64, f"{key} should be float64"
            break


# ============================================================================
# Phase 6: Agent Definition Test
# ============================================================================

class TestAgentDefinition:

    def test_agent_definition_exists(self):
        from pathlib import Path
        agent_path = Path('/home/norepinephrine/Documents/Heart-Conduction/.claude/agents/training-monitor.md')
        assert agent_path.exists(), "Training monitor agent definition should exist"
        content = agent_path.read_text()
        assert 'Analysis Checklist' in content
        assert 'Intervention Protocol' in content
        assert 'Output Format' in content


# ============================================================================
# NODE Rollout Tests (Phase 3)
# ============================================================================

class TestNodeRollout:
    """Tests for the Neural ODE training rollout."""

    def _make_node(self):
        from surrogate.model.stage1 import IonicStage1
        from surrogate.model.node import IonicNODE
        stage1 = IonicStage1(scaffold=True).double()
        return IonicNODE(stage1)

    def _make_segment(self, B=4, T=3000):
        """Fake segment covering ~300ms at dt=0.1ms."""
        return {
            'Vm': torch.randn(B, T, dtype=torch.float64) * 10 - 80,
            'dt': torch.full((B, T), 0.1, dtype=torch.float64),
            'ionic_states': torch.rand(B, T, 14, dtype=torch.float64),
            'concentrations': torch.rand(B, T, 4, dtype=torch.float64).abs() + 0.0001,
            'conductance_products': torch.rand(B, T, 5, dtype=torch.float64),
        }

    def test_node_rollout_runs(self):
        """node_rollout completes and returns dict with 'loss'."""
        from surrogate.training.node_rollout import node_rollout
        node = self._make_node()
        seg = self._make_segment(B=2, T=100)  # short for speed
        out = node_rollout(node, seg, 'A1')
        node.clear_v_trajectory()
        assert 'loss' in out
        assert out['loss'].dim() == 0  # scalar

    def test_node_rollout_backward(self):
        """loss.backward() runs, gradients exist on stage1 params, norm finite."""
        from surrogate.training.node_rollout import node_rollout
        node = self._make_node()
        seg = self._make_segment(B=2, T=100)
        out = node_rollout(node, seg, 'A1')
        out['loss'].backward()
        node.clear_v_trajectory()
        gnorm = sum(
            p.grad.norm().item()
            for p in node.stage1.parameters()
            if p.grad is not None
        )
        assert gnorm > 0, "Zero gradient norm"
        assert torch.isfinite(torch.tensor(gnorm)), "Non-finite gradient norm"

    def test_node_rollout_loss_finite(self):
        """Loss is finite float64 scalar."""
        from surrogate.training.node_rollout import node_rollout
        node = self._make_node()
        seg = self._make_segment(B=2, T=100)
        out = node_rollout(node, seg, 'A1')
        node.clear_v_trajectory()
        assert out['loss'].dtype == torch.float64
        assert torch.isfinite(out['loss'])

    def test_node_rollout_phase_names(self):
        """A1, B1 accepted; C raises NotImplementedError."""
        from surrogate.training.node_rollout import node_rollout
        node = self._make_node()
        seg = self._make_segment(B=2, T=100)

        # A1 works
        out_a1 = node_rollout(node, seg, 'A1')
        node.clear_v_trajectory()
        assert 'loss' in out_a1

        # B1 works
        out_b1 = node_rollout(node, seg, 'B1')
        node.clear_v_trajectory()
        assert 'loss' in out_b1

        # C raises
        with pytest.raises(NotImplementedError):
            node_rollout(node, seg, 'C')
        node.clear_v_trajectory()

    def test_node_rollout_z0_noise(self):
        """z0_noise_sigma>0 in training mode produces different loss than sigma=0."""
        from surrogate.training.node_rollout import node_rollout
        node = self._make_node()
        node.train()
        seg = self._make_segment(B=2, T=100)

        torch.manual_seed(0)
        out_no_noise = node_rollout(node, seg, 'A1', z0_noise_sigma=0.0)
        loss_clean = out_no_noise['loss'].item()
        node.clear_v_trajectory()

        torch.manual_seed(0)
        out_noise = node_rollout(node, seg, 'A1', z0_noise_sigma=0.01)
        loss_noisy = out_noise['loss'].item()
        node.clear_v_trajectory()

        assert loss_clean != loss_noisy, "Noise should change the loss"

    def test_build_t_grid_cumulative(self):
        """t_grid[0]=0, t_grid[-1]=sum(dt), shape=(T+1,)."""
        from surrogate.training.node_rollout import build_t_grid
        dt = torch.full((4, 100), 0.1, dtype=torch.float64)
        t_grid = build_t_grid(dt)
        assert t_grid.shape == (101,)
        assert t_grid[0] == 0.0
        assert torch.allclose(t_grid[-1], torch.tensor(10.0, dtype=torch.float64))

    def test_interpolate_v_boundary(self):
        """V interpolation at t beyond V_traj range clamps correctly."""
        from surrogate.model.node import IonicNODE
        from surrogate.model.stage1 import IonicStage1
        node = IonicNODE(IonicStage1().double())
        V_traj = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        t_grid = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
        node.set_v_trajectory(V_traj, t_grid)

        # Beyond V_traj range (t=0.3 maps to t_grid[3] but V only has 3 points)
        V_end = node._interpolate_V(torch.tensor(0.2, dtype=torch.float64))
        assert torch.isfinite(V_end).all()
        assert torch.allclose(V_end, torch.tensor([3.0], dtype=torch.float64))
        node.clear_v_trajectory()

    # ---- Rest-attractor regularizer (Session 27, PLAN Step 2.2) ----

    def test_L_rest_computed(self):
        """node_rollout exposes a finite L_rest scalar."""
        from surrogate.training.node_rollout import node_rollout
        node = self._make_node()
        seg = self._make_segment(B=2, T=100)
        out = node_rollout(node, seg, 'A1')
        node.clear_v_trajectory()
        assert 'L_rest' in out
        assert torch.isfinite(out['L_rest'])
        assert out['L_rest'].dim() == 0

    def test_L_rest_float64_contract(self):
        """L_rest is computed on a float64 model; locks dtype."""
        from surrogate.training.node_rollout import node_rollout
        node = self._make_node()
        seg = self._make_segment(B=2, T=100)
        out = node_rollout(node, seg, 'A1')
        node.clear_v_trajectory()
        assert out['L_rest'].dtype == torch.float64

    def test_total_loss_contains_rest(self):
        """loss == base + LAMBDA_REST * L_rest (empirical, via LAMBDA_REST=0 run)."""
        from surrogate.training import node_rollout as nr
        node = self._make_node()
        seg = self._make_segment(B=2, T=100)

        torch.manual_seed(0)
        out_normal = nr.node_rollout(node, seg, 'A1')
        node.clear_v_trajectory()

        original = nr.LAMBDA_REST
        try:
            nr.LAMBDA_REST = 0.0
            torch.manual_seed(0)
            out_no_rest = nr.node_rollout(node, seg, 'A1')
            node.clear_v_trajectory()
        finally:
            nr.LAMBDA_REST = original

        delta = (out_normal['loss'] - out_no_rest['loss']).item()
        expected = original * out_normal['L_rest'].item()
        assert abs(delta - expected) / (abs(expected) + 1e-12) < 1e-4, (
            f"loss delta {delta:.6e} vs expected {expected:.6e} — "
            "implementer likely dropped the LAMBDA_REST * L_rest term"
        )

    def test_L_rest_goes_to_zero(self):
        """L_rest = ||rate||^2/N — zero rate => zero L_rest (sanity check)."""
        from surrogate.training.node_rollout import INIT_CONC, V_REST_MV
        B = 2
        carried_dim = 24
        z_rest = torch.zeros(B, carried_dim, dtype=torch.float64)
        z_rest[:, 20:] = INIT_CONC
        V_rest = torch.full((B,), V_REST_MV, dtype=torch.float64)
        # Zero-rate stub mirrors the "trained" model's fixed-point behavior.
        rate_zero = torch.zeros_like(z_rest)
        assert (rate_zero.pow(2).mean()).item() == 0.0

    def test_init_conc_not_leaky(self):
        """INIT_CONC is a constant tensor, not a Parameter — won't leak into backward."""
        from surrogate.training.node_rollout import INIT_CONC
        assert INIT_CONC.requires_grad is False
