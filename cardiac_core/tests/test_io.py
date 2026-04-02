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
