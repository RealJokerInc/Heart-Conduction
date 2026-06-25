"""Tests for cardiac_core.io — result save/load."""

import torch
import pytest

from cardiac_core.io import save_result, load_result


class TestResultIO:
    def test_round_trip_monodomain(self, tmp_path):
        times = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        V = torch.randn(3, 21, 11, dtype=torch.float64)

        path = str(tmp_path / "result.npz")
        save_result(path, times, V)
        t2, V2, phi_e2, meta = load_result(path)

        torch.testing.assert_close(t2, times)
        torch.testing.assert_close(V2, V)
        assert phi_e2 is None
        assert meta == {}

    def test_round_trip_bidomain(self, tmp_path):
        times = torch.tensor([1.0, 2.0])
        V = torch.randn(2, 11, 11, dtype=torch.float64)
        phi_e = torch.randn(2, 11, 11, dtype=torch.float64)

        path = str(tmp_path / "bidomain.npz")
        save_result(path, times, V, phi_e)
        t2, V2, pe2, meta = load_result(path)

        torch.testing.assert_close(pe2, phi_e)

    def test_metadata(self, tmp_path):
        times = torch.tensor([1.0])
        V = torch.randn(1, 5, 5, dtype=torch.float64)

        path = str(tmp_path / "meta.npz")
        save_result(path, times, V, dx=0.025, engine='monodomain', ionic_model='ttp06')
        _, _, _, meta = load_result(path)

        assert meta['dx'] == pytest.approx(0.025)
        assert meta['engine'] == 'monodomain'
        assert meta['ionic_model'] == 'ttp06'

    def test_load_to_device(self, tmp_path):
        times = torch.tensor([1.0])
        V = torch.randn(1, 5, 5, dtype=torch.float64)

        path = str(tmp_path / "dev.npz")
        save_result(path, times, V)
        t2, V2, _, _ = load_result(path, device='cpu')
        assert V2.device.type == 'cpu'

    def test_vm_keyword(self, tmp_path):
        """The canonical Vm= keyword saves identically to the positional voltage."""
        times = torch.tensor([1.0, 2.0])
        Vm = torch.randn(2, 6, 4, dtype=torch.float64)

        path = str(tmp_path / "vm.npz")
        save_result(path, times, Vm=Vm)
        _, V2, _, _ = load_result(path)
        torch.testing.assert_close(V2, Vm)

    def test_legacy_v_keyword_warns(self, tmp_path):
        """Legacy V= keyword still works but emits a DeprecationWarning."""
        import pytest
        times = torch.tensor([1.0])
        V = torch.randn(1, 5, 5, dtype=torch.float64)

        path = str(tmp_path / "legacy.npz")
        with pytest.warns(DeprecationWarning):
            save_result(path, times, V=V)
        _, V2, _, _ = load_result(path)
        torch.testing.assert_close(V2, V)

    def test_positional_phi_e_back_compat(self, tmp_path):
        """AUDIT HIGH: save_result(path, times, V, phi_e) positional call must still work."""
        times = torch.tensor([1.0, 2.0])
        V = torch.randn(2, 5, 5, dtype=torch.float64)
        phi_e = torch.randn(2, 5, 5, dtype=torch.float64)

        path = str(tmp_path / "pos.npz")
        save_result(path, times, V, phi_e)  # 4th arg positional — must not raise
        _, V2, pe2, _ = load_result(path)
        torch.testing.assert_close(V2, V)
        torch.testing.assert_close(pe2, phi_e)
