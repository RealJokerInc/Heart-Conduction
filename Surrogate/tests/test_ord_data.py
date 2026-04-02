"""Tests for ORd data infrastructure: ORdTraceData, corrupt_states, ORdTraceStorage."""

import torch
import pytest
import tempfile
import os
import sys
from pathlib import Path


class TestORdTraceData:
    """Tests for the 101-column ORd trace format."""

    def test_column_count(self):
        from surrogate.data.ord_trace_data import ORdTraceData
        assert ORdTraceData.N_COLUMNS == 101

    def test_column_layout(self):
        """Verify column arithmetic: 3 + 40 + 2 + 28 + 28 = 101."""
        from surrogate.data.ord_trace_data import ORdTraceData
        assert ORdTraceData.STATES_START == 3
        assert ORdTraceData.STATES_END == 43        # 3 + 40
        assert ORdTraceData.I_ION == 43
        assert ORdTraceData.CLAMP_MASK == 44
        assert ORdTraceData.GATE_INF_START == 45
        assert ORdTraceData.GATE_INF_END == 73      # 45 + 28
        assert ORdTraceData.GATE_TAU_START == 73
        assert ORdTraceData.GATE_TAU_END == 101     # 73 + 28

    def test_state_count(self):
        from surrogate.data.ord_trace_data import ORdTraceData
        assert ORdTraceData.N_STATES == 40
        assert ORdTraceData.N_RL_GATES == 28
        assert len(ORdTraceData.RL_GATE_INDICES) == 28

    def test_rl_gate_indices_exclude_nca(self):
        """nca (index 29) should NOT be in RL_GATE_INDICES."""
        from surrogate.data.ord_trace_data import ORdTraceData
        assert 29 not in ORdTraceData.RL_GATE_INDICES

    def test_conc_indices(self):
        """Concentration indices: nai=0, ki=1, cai=2, cass=6."""
        from surrogate.data.ord_trace_data import ORdTraceData
        assert ORdTraceData.CONC_INDICES == [0, 1, 2, 6]

    def test_create_valid(self):
        from surrogate.data.ord_trace_data import ORdTraceData
        data = torch.randn(100, 101, dtype=torch.float64)
        trace = ORdTraceData(data=data, metadata={'test': True})
        assert trace.data.shape == (100, 101)

    def test_create_wrong_columns(self):
        from surrogate.data.ord_trace_data import ORdTraceData
        data = torch.randn(100, 47)  # TTP06 column count
        with pytest.raises(ValueError, match="101 columns"):
            ORdTraceData(data=data)


class TestCorruptStatesORd:
    """Tests for model_type-aware corrupt_states."""

    def test_ttp06_backward_compat(self):
        """Default model_type='ttp06' works as before."""
        from surrogate.data.augmentation import corrupt_states
        states = torch.zeros(18)
        states[5:17] = 0.5  # gates
        result = corrupt_states(states, 'random_gates', severity=1.0, model_type='ttp06')
        # Gates should be perturbed
        assert not torch.allclose(result[5:17], states[5:17])
        # Concentrations unchanged
        assert torch.allclose(result[:5], states[:5])

    def test_ord_random_gates(self):
        """ORd corrupt_states perturbs gates 8-36, not concentrations 0-7."""
        from surrogate.data.augmentation import corrupt_states
        states = torch.zeros(40)
        states[8:37] = 0.5  # ORd gates
        result = corrupt_states(states, 'random_gates', severity=1.0, model_type='ord')
        # Gates should be perturbed
        assert not torch.allclose(result[8:37], states[8:37])
        # Concentrations unchanged
        assert torch.allclose(result[:8], states[:8])
        # CaMKt, SR release unchanged
        assert torch.allclose(result[37:], states[37:])

    def test_ord_extreme_ca(self):
        from surrogate.data.augmentation import corrupt_states
        states = torch.ones(40) * 0.001
        result = corrupt_states(states, 'extreme_ca', severity=1.0, model_type='ord')
        # cai (index 2) should be amplified
        assert result[2] > states[2]
        # Other states unchanged
        assert torch.allclose(result[0], states[0])

    def test_unknown_model_type(self):
        from surrogate.data.augmentation import corrupt_states
        with pytest.raises(ValueError, match="Unknown model_type"):
            corrupt_states(torch.zeros(18), 'random_gates', model_type='unknown')


class TestORdStorage:
    """Tests for ORdTraceStorage save/load roundtrip."""

    def test_save_load_roundtrip(self):
        from surrogate.data.ord_trace_data import ORdTraceData
        from surrogate.data.ord_storage import ORdTraceStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ORdTraceStorage(base_dir=tmpdir)
            data = torch.randn(50, 101, dtype=torch.float64)
            trace = ORdTraceData(data=data, metadata={'cell_type': 'EPI', 'bcl': 1000})
            storage.save_trace(trace, tier=1, protocol_name='test_proto')

            loaded = storage.load_trace(tier=1, protocol_name='test_proto')
            assert isinstance(loaded, ORdTraceData)
            assert loaded.data.shape == (50, 101)
            assert torch.allclose(loaded.data, data)
            assert loaded.metadata['cell_type'] == 'EPI'

    def test_returns_ord_trace_not_ttp06(self):
        from surrogate.data.ord_trace_data import ORdTraceData
        from surrogate.data.ord_storage import ORdTraceStorage
        from surrogate.data.single_cell_generator import TraceData

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ORdTraceStorage(base_dir=tmpdir)
            data = torch.randn(10, 101, dtype=torch.float64)
            trace = ORdTraceData(data=data)
            storage.save_trace(trace, tier=1, protocol_name='test')
            loaded = storage.load_trace(tier=1, protocol_name='test')
            assert isinstance(loaded, ORdTraceData)
            assert not isinstance(loaded, TraceData)

    def test_list_protocols(self):
        from surrogate.data.ord_trace_data import ORdTraceData
        from surrogate.data.ord_storage import ORdTraceStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ORdTraceStorage(base_dir=tmpdir)
            for name in ['proto_a', 'proto_b', 'proto_c']:
                data = torch.randn(10, 101, dtype=torch.float64)
                storage.save_trace(ORdTraceData(data=data), tier=1, protocol_name=name)
            protos = storage.list_protocols(tier=1)
            assert sorted(protos) == ['proto_a', 'proto_b', 'proto_c']


class TestORdShardProcessor:
    """Tests for ORd shard processing."""

    def test_shard_uses_101_columns(self):
        from surrogate.data.ord_storage import ORdShardProcessor
        proc = ORdShardProcessor(
            raw_dir='/tmp/test_raw', shard_dir='/tmp/test_shards',
            segment_length=1000, shard_size_mb=200.0
        )
        # 101 cols * 1000 steps * 4 bytes = 404,000 bytes per segment
        expected_segs = int(200e6 / 404_000)
        assert proc._segments_per_shard == expected_segs


# Add engine path for ORd model imports
_ENGINE_PATH = str(Path(__file__).resolve().parents[1] / '..' / 'Bidomain' / 'Engine_V1')
if _ENGINE_PATH not in sys.path:
    sys.path.insert(0, _ENGINE_PATH)


class TestORdSingleCellGenerator:
    """Tests for ORd single-cell data generator."""

    def test_creates_trace(self):
        """Run 10ms, verify shape and initial Vm."""
        from surrogate.data.ord_single_cell_generator import ORdSingleCellGenerator
        from surrogate.data.protocols import SteadyStatePacing

        gen = ORdSingleCellGenerator(cell_type='EPI', device='cpu', warmup_beats=0)
        proto = SteadyStatePacing(bcl=1000, n_beats=1, dt_default=0.01)
        proto.duration_ms = 10.0
        trace = gen.run_protocol(proto)
        assert trace.data.shape[1] == 101
        assert trace.data.shape[0] > 0
        # Initial Vm near resting
        assert abs(trace.data[0, 0].item() - (-87.5)) < 2.0

    def test_produces_ap(self):
        """1-beat pacing produces AP (Vm > 0)."""
        from surrogate.data.ord_single_cell_generator import ORdSingleCellGenerator
        from surrogate.data.protocols import SteadyStatePacing

        gen = ORdSingleCellGenerator(cell_type='EPI', device='cpu', warmup_beats=0)
        proto = SteadyStatePacing(bcl=1000, n_beats=1, dt_default=0.01)
        trace = gen.run_protocol(proto)
        Vm_max = trace.data[:, 0].max().item()
        assert Vm_max > 0, f"No AP: Vm_max = {Vm_max}"

    def test_gate_inf_tau_shape(self):
        """gate_inf has 28 cols, gate_tau has 28 cols."""
        from surrogate.data.ord_single_cell_generator import ORdSingleCellGenerator
        from surrogate.data.ord_trace_data import ORdTraceData
        from surrogate.data.protocols import SteadyStatePacing

        gen = ORdSingleCellGenerator(cell_type='EPI', device='cpu', warmup_beats=0)
        proto = SteadyStatePacing(bcl=1000, n_beats=1, dt_default=0.01)
        proto.duration_ms = 5.0
        trace = gen.run_protocol(proto)
        gate_inf = trace.data[:, ORdTraceData.GATE_INF_START:ORdTraceData.GATE_INF_END]
        gate_tau = trace.data[:, ORdTraceData.GATE_TAU_START:ORdTraceData.GATE_TAU_END]
        assert gate_inf.shape[1] == 28
        assert gate_tau.shape[1] == 28

    def test_gate_inf_physiological(self):
        """gate_inf values should be in [0, 1]."""
        from surrogate.data.ord_single_cell_generator import ORdSingleCellGenerator
        from surrogate.data.ord_trace_data import ORdTraceData
        from surrogate.data.protocols import SteadyStatePacing

        gen = ORdSingleCellGenerator(cell_type='EPI', device='cpu', warmup_beats=0)
        proto = SteadyStatePacing(bcl=1000, n_beats=1, dt_default=0.01)
        proto.duration_ms = 5.0
        trace = gen.run_protocol(proto)
        gate_inf = trace.data[:, ORdTraceData.GATE_INF_START:ORdTraceData.GATE_INF_END]
        assert gate_inf.min() >= 0.0, f"gate_inf min = {gate_inf.min()}"
        assert gate_inf.max() <= 1.0, f"gate_inf max = {gate_inf.max()}"
