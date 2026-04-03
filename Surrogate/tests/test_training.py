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


# ============================================================================
# Phase 2: Encoder Tests
# ============================================================================

class TestEncoder:

    def test_encoder_shape(self):
        from surrogate.training.encoder import TemporaryEncoder
        enc = TemporaryEncoder(n_ionic_targets=14, ionic_dim=16)
        x = torch.randn(8, 14)
        out = enc(x)
        assert out.shape == (8, 16)

    def test_encoder_unbatched(self):
        from surrogate.training.encoder import TemporaryEncoder
        enc = TemporaryEncoder()
        x = torch.randn(14)
        out = enc(x)
        assert out.shape == (16,)

    def test_encoder_differentiable(self):
        from surrogate.training.encoder import TemporaryEncoder
        enc = TemporaryEncoder()
        x = torch.randn(4, 14, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_make_carried_state(self):
        from surrogate.training.encoder import TemporaryEncoder, make_carried_state
        enc = TemporaryEncoder()
        ionic = torch.randn(4, 14)
        conc = torch.randn(4, 4)
        carried = make_carried_state(enc, ionic, conc)
        assert carried.shape == (4, 20)
        # Last 4 dims should be concentrations
        assert torch.allclose(carried[:, 16:], conc)


# ============================================================================
# Phase 2: Phase Config Tests
# ============================================================================

class TestPhaseConfig:

    def test_phase_configs_complete(self):
        from surrogate.training.phases import PHASE_ORDER, PHASE_CONFIGS
        assert len(PHASE_ORDER) == 11
        for name in PHASE_ORDER:
            assert name in PHASE_CONFIGS

    def test_freeze_mask_A1(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary

        model = IonicSurrogateV3(scaffold=True)
        phase = get_phase_config("A1")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        # Only ionic_state_decoder should be unfrozen (encoder is separate)
        for name, grad in summary.items():
            if 'ionic_state_decoder' in name:
                assert grad, f"{name} should be unfrozen in A1"
            else:
                assert not grad, f"{name} should be frozen in A1"

    def test_freeze_mask_B3(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary

        model = IonicSurrogateV3(scaffold=True)
        phase = get_phase_config("B3")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        # Stage 1 dynamics unfrozen, Stage 2 frozen
        for name, grad in summary.items():
            if any(p in name for p in ['voltage_attention', 'ionic_mixing_mlp', 'ionic_mixing_logit']):
                assert grad, f"{name} should be unfrozen in B3"
            elif 'stage2' in name:
                assert not grad, f"{name} should be frozen in B3"

    def test_freeze_mask_D(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary

        model = IonicSurrogateV3(scaffold=True)
        phase = get_phase_config("D")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        # Only Stage 2 unfrozen
        for name, grad in summary.items():
            if 'stage2' in name:
                assert grad, f"{name} should be unfrozen in D"
            else:
                assert not grad, f"{name} should be frozen in D"

    def test_freeze_mask_E(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.phases import get_phase_config, apply_freeze_mask, get_freeze_summary

        model = IonicSurrogateV3(scaffold=True)
        phase = get_phase_config("E")
        apply_freeze_mask(model, phase)
        summary = get_freeze_summary(model)

        # Everything unfrozen (stage1.* + stage2.*)
        for name, grad in summary.items():
            assert grad, f"{name} should be unfrozen in E"

    def test_phase_order(self):
        from surrogate.training.phases import PHASE_ORDER, get_all_phases
        phases = get_all_phases()
        assert [p.name for p in phases] == PHASE_ORDER

    def test_T12_in_B1(self):
        from surrogate.training.phases import get_phase_config
        b1 = get_phase_config("B1")
        assert 12 in b1.data_tiers, "T12 (celltypes) must be in B1 data_tiers"

    def test_encoder_phases(self):
        from surrogate.training.phases import get_all_phases
        phases = get_all_phases()
        # A1-B5 use encoder, C-E do not
        for p in phases:
            if p.name in ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5"]:
                assert p.uses_encoder, f"{p.name} should use encoder"
            else:
                assert not p.uses_encoder, f"{p.name} should NOT use encoder"


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

        result = rollout(model, segment, phase_name="B1")
        assert result['loss'].shape == ()
        assert result['per_step_losses'].shape == (10,)

    def test_rollout_teacher_forcing(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.encoder import TemporaryEncoder
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        encoder = TemporaryEncoder().double()
        segment = self._make_segment(B=2, T=5)

        # p=0.0 means all teacher forcing
        result = rollout(model, segment, encoder=encoder,
                         scheduled_sampling_p=0.0, phase_name="B1")
        assert result['loss'].isfinite()

    def test_rollout_autoregressive(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=2, T=5)

        # p=1.0 means fully autoregressive
        result = rollout(model, segment, encoder=None,
                         scheduled_sampling_p=1.0, phase_name="B1")
        assert result['loss'].isfinite()

    def test_rollout_gradient_flow(self):
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=2, T=5)

        result = rollout(model, segment, phase_name="B1")
        result['loss'].backward()

        # Check that some Stage 1 params got gradients
        has_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through rollout"

    def test_rollout_D_phase_loss(self):
        """Phase D uses I_ion loss."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=2, T=5)

        result = rollout(model, segment, phase_name="D")
        assert result['loss'].isfinite()

    def test_rollout_C_phase_loss(self):
        """Phase C uses concentration loss."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.rollout import rollout

        model = IonicSurrogateV3(scaffold=True).double()
        segment = self._make_segment(B=2, T=5)

        result = rollout(model, segment, phase_name="C")
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
        """Phase A1 freeze mask: only encoder + ionic_state_decoder unfrozen."""
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
            if 'ionic_state_decoder' in name:
                assert grad, f"{name} should be unfrozen"
            else:
                assert not grad, f"{name} should be frozen"

    def test_trainer_A1_one_epoch(self, tmp_path):
        """Phase A1 runs one epoch without error and produces a loss."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.trainer import SurrogateTrainer
        from surrogate.training.phases import get_phase_config

        cache_dir = self._setup_fake_cache(tmp_path, T=500)
        model = IonicSurrogateV3(scaffold=True).double()
        trainer = SurrogateTrainer(model, str(cache_dir), str(tmp_path / 'run'), device='cpu')

        phase = get_phase_config("A1")
        # Override max_epochs to just 1
        phase.max_epochs = 1
        phase.patience = 1

        metrics = trainer.train_phase(phase)
        assert 'val_recon_mse' in metrics
        assert metrics['val_recon_mse'] >= 0

    def test_trainer_B1_one_epoch(self, tmp_path):
        """Phase B1 rollout runs one epoch (need encoder from A1 first)."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.trainer import SurrogateTrainer
        from surrogate.training.phases import get_phase_config

        cache_dir = self._setup_fake_cache(tmp_path, T=200)
        model = IonicSurrogateV3(scaffold=True).double()
        trainer = SurrogateTrainer(model, str(cache_dir), str(tmp_path / 'run'), device='cpu')

        # Must create encoder first (normally done in A1)
        a1 = get_phase_config("A1")
        a1.max_epochs = 1
        a1.patience = 1
        trainer.train_phase(a1)

        b1 = get_phase_config("B1")
        b1.max_epochs = 1
        b1.patience = 1
        b1.batch_size = 4  # small for test
        metrics = trainer.train_phase(b1)
        assert 'val_ionic_state_mse' in metrics

    def test_trainer_phase_transition(self, tmp_path):
        """Phase transition resets optimizer and changes freeze mask."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.trainer import SurrogateTrainer
        from surrogate.training.phases import get_phase_config, get_freeze_summary

        cache_dir = self._setup_fake_cache(tmp_path, T=200)
        model = IonicSurrogateV3(scaffold=True).double()
        trainer = SurrogateTrainer(model, str(cache_dir), str(tmp_path / 'run'), device='cpu')

        # Run A1 with 1 epoch
        a1 = get_phase_config("A1")
        a1.max_epochs = 1
        a1.patience = 1
        trainer.train_phase(a1)

        # Run A2 with 1 epoch — should have different freeze mask
        a2 = get_phase_config("A2")
        a2.max_epochs = 1
        a2.patience = 1
        trainer.train_phase(a2)

        summary = get_freeze_summary(model)
        # After A2: voltage_attention should be unfrozen
        for name, grad in summary.items():
            if 'voltage_attention' in name:
                assert grad, f"{name} should be unfrozen after A2"

    def test_checkpoint_save_load(self, tmp_path):
        """Checkpoint round-trip preserves model weights."""
        from surrogate.model import IonicSurrogateV3
        from surrogate.training.trainer import SurrogateTrainer
        from surrogate.training.phases import get_phase_config

        cache_dir = self._setup_fake_cache(tmp_path, T=200)
        model = IonicSurrogateV3(scaffold=True).double()
        trainer = SurrogateTrainer(model, str(cache_dir), str(tmp_path / 'run'), device='cpu')

        # Run A1 for 1 epoch
        a1 = get_phase_config("A1")
        a1.max_epochs = 1
        a1.patience = 1
        trainer.train_phase(a1)

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
