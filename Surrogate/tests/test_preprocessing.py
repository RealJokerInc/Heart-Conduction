"""Tests for V3 preprocessor: TTP06 47-col → v3 training format."""

import torch
import pytest


class TestV3Preprocessor:

    def _make_fake_47col(self, T=100):
        """Create fake 47-col data with physiological-ish values."""
        data = torch.zeros(T, 47, dtype=torch.float64)
        # Vm: -85 to +30
        data[:, 0] = torch.linspace(-85, 30, T)
        # I_stim
        data[:, 1] = 0.0
        # dt
        data[:, 2] = 0.01
        # States (cols 3-20): 18 state variables
        # Concentrations: Ki~138, Nai~10, Cai~0.0001, CaSR~1.5, CaSS~0.0002
        data[:, 3] = 138.0   # Ki (index 0)
        data[:, 4] = 10.0    # Nai (index 1)
        data[:, 5] = 0.0001  # Cai (index 2)
        data[:, 6] = 1.5     # CaSR (index 3)
        data[:, 7] = 0.0002  # CaSS (index 4)
        # Gates: all ~0.5 for simplicity
        for i in range(5, 18):
            data[:, 3 + i] = 0.5
        # Specific gate values for conductance product test
        data[:, 3 + 5] = 0.8   # m
        data[:, 3 + 6] = 0.7   # h
        data[:, 3 + 7] = 0.6   # j
        data[:, 3 + 8] = 0.3   # r
        data[:, 3 + 9] = 0.9   # s
        data[:, 3 + 10] = 0.4  # d
        data[:, 3 + 11] = 0.95 # f
        data[:, 3 + 12] = 0.85 # f2
        data[:, 3 + 13] = 0.99 # fCass
        data[:, 3 + 14] = 0.2  # Xr1
        data[:, 3 + 15] = 0.6  # Xr2
        data[:, 3 + 16] = 0.1  # Xs
        data[:, 3 + 17] = 0.01 # RR
        # I_ion
        data[:, 21] = -5.0
        # clamp_mask
        data[:, 22] = 0.0
        # gate_inf (12 cols)
        data[:, 23:35] = 0.5
        # gate_tau (12 cols)
        data[:, 35:47] = 10.0
        return data

    def test_column_mapping(self):
        """Concentrations reordered from [Ki,Nai,Cai,CaSR,CaSS] to [Na_i,K_i,Ca_i,Ca_ss]."""
        from surrogate.data.preprocessor import V3Preprocessor

        proc = V3Preprocessor()
        data = self._make_fake_47col()
        result = proc.process_segment(data)

        conc = result['concentrations']
        assert conc.shape == (100, 4)
        # Na_i should be 10.0 (was at state index 1 = col 4)
        assert torch.allclose(conc[:, 0], torch.full((100,), 10.0, dtype=torch.float64))
        # K_i should be 138.0 (was at state index 0 = col 3)
        assert torch.allclose(conc[:, 1], torch.full((100,), 138.0, dtype=torch.float64))
        # Ca_i should be 0.0001
        assert torch.allclose(conc[:, 2], torch.full((100,), 0.0001, dtype=torch.float64))
        # Ca_ss should be 0.0002
        assert torch.allclose(conc[:, 3], torch.full((100,), 0.0002, dtype=torch.float64))

    def test_gate_count(self):
        """12 HH gates extracted (m through Xs, NOT RR)."""
        from surrogate.data.preprocessor import V3Preprocessor

        proc = V3Preprocessor()
        data = self._make_fake_47col()
        result = proc.process_segment(data)

        assert result['gates'].shape == (100, 12)
        # First gate should be m = 0.8
        assert torch.allclose(result['gates'][:, 0], torch.full((100,), 0.8, dtype=torch.float64))
        # Last gate should be Xs = 0.1
        assert torch.allclose(result['gates'][:, 11], torch.full((100,), 0.1, dtype=torch.float64))

    def test_ionic_states(self):
        """14 ionic state targets: 13 gates (m-RR) + CaSR."""
        from surrogate.data.preprocessor import V3Preprocessor

        proc = V3Preprocessor()
        data = self._make_fake_47col()
        result = proc.process_segment(data)

        assert result['ionic_states'].shape == (100, 14)
        # First 13 are gates (m through RR)
        assert torch.allclose(result['ionic_states'][:, 0], torch.full((100,), 0.8, dtype=torch.float64))  # m
        assert torch.allclose(result['ionic_states'][:, 12], torch.full((100,), 0.01, dtype=torch.float64))  # RR
        # Last is CaSR
        assert torch.allclose(result['ionic_states'][:, 13], torch.full((100,), 1.5, dtype=torch.float64))

    def test_nernst(self):
        """Nernst output matches NernstComputer module."""
        from surrogate.data.preprocessor import V3Preprocessor
        from surrogate.model.nernst import NernstComputer

        proc = V3Preprocessor()
        nernst = NernstComputer()

        data = self._make_fake_47col(T=10)
        result = proc.process_segment(data)

        # Compare preprocessor E values vs NernstComputer
        Na_i = result['concentrations'][:, 0].float()
        K_i = result['concentrations'][:, 1].float()
        Ca_i = result['concentrations'][:, 2].float()

        E_Na_ref, E_K_ref, E_Ca_ref, E_Ks_ref = nernst(Na_i, K_i, Ca_i)

        assert torch.allclose(result['E'][:, 0].float(), E_Na_ref, atol=1e-4)
        assert torch.allclose(result['E'][:, 1].float(), E_K_ref, atol=1e-4)
        assert torch.allclose(result['E'][:, 2].float(), E_Ca_ref, atol=1e-4)
        assert torch.allclose(result['E'][:, 3].float(), E_Ks_ref, atol=1e-4)

    def test_conductance_products(self):
        """5 effective gate conductance products computed correctly."""
        from surrogate.data.preprocessor import V3Preprocessor

        proc = V3Preprocessor()
        data = self._make_fake_47col(T=1)
        result = proc.process_segment(data)

        cp = result['conductance_products']
        assert cp.shape == (1, 5)

        # G_Na = m³·h·j = 0.8³ × 0.7 × 0.6
        expected_gna = 0.8**3 * 0.7 * 0.6
        assert abs(cp[0, 0].item() - expected_gna) < 1e-10

        # G_CaL = d·f·f2·fCass = 0.4 × 0.95 × 0.85 × 0.99
        expected_gcal = 0.4 * 0.95 * 0.85 * 0.99
        assert abs(cp[0, 1].item() - expected_gcal) < 1e-10

        # G_to = r·s = 0.3 × 0.9
        assert abs(cp[0, 2].item() - 0.3 * 0.9) < 1e-10

        # G_Kr = Xr1·Xr2 = 0.2 × 0.6
        assert abs(cp[0, 3].item() - 0.2 * 0.6) < 1e-10

        # G_Ks = Xs² = 0.1²
        assert abs(cp[0, 4].item() - 0.01) < 1e-10

    def test_no_nan(self):
        """Physiological inputs produce no NaN in any output."""
        from surrogate.data.preprocessor import V3Preprocessor

        proc = V3Preprocessor()
        data = self._make_fake_47col()
        result = proc.process_segment(data)

        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), f"NaN/Inf in {key}"

    def test_all_shapes(self):
        """Verify all output shapes."""
        from surrogate.data.preprocessor import V3Preprocessor

        proc = V3Preprocessor()
        T = 50
        data = self._make_fake_47col(T=T)
        result = proc.process_segment(data)

        assert result['Vm'].shape == (T,)
        assert result['dt'].shape == (T,)
        assert result['I_stim'].shape == (T,)
        assert result['I_ion'].shape == (T,)
        assert result['clamp_mask'].shape == (T,)
        assert result['concentrations'].shape == (T, 4)
        assert result['gates'].shape == (T, 12)
        assert result['ionic_states'].shape == (T, 14)
        assert result['conductance_products'].shape == (T, 5)
        assert result['E'].shape == (T, 4)
        assert result['gate_inf'].shape == (T, 12)
        assert result['gate_tau'].shape == (T, 12)
