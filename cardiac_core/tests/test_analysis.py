"""Tests for cardiac_core.analysis — pure tensor analysis functions."""

import torch
import pytest

from cardiac_core.analysis import (
    activation_time,
    conduction_velocity,
    apd_at,
    apd_map,
    dominant_frequency,
    wavefront_mask,
    phase_map,
    phase_singularities,
    restitution_curve,
)


def _make_traveling_wave(Nx=41, Ny=11, n_saves=100, cv_idx_per_save=2):
    """Create synthetic voltage history: a planar wave traveling in +x.

    Wave moves cv_idx_per_save grid points per save interval.
    V=-85 at rest, V=+20 when activated (instant upstroke, slow repol).
    """
    times = torch.arange(n_saves, dtype=torch.float64)  # ms, save_every=1ms
    V = torch.full((n_saves, Nx, Ny), -85.0, dtype=torch.float64)

    for t_idx in range(n_saves):
        # Wavefront position at this time
        front_x = t_idx * cv_idx_per_save
        # Everything behind the front is activated
        if front_x > 0:
            activated_end = min(front_x, Nx)
            # Simple AP shape: peak then linear decay
            for ix in range(activated_end):
                time_since_activation = t_idx - ix / cv_idx_per_save
                if time_since_activation < 0:
                    continue
                elif time_since_activation < 30:
                    # Plateau phase
                    V[t_idx, ix, :] = 20.0
                elif time_since_activation < 60:
                    # Repolarization: linear from 20 to -85
                    frac = (time_since_activation - 30) / 30
                    V[t_idx, ix, :] = 20.0 - frac * 105.0
                else:
                    # Returned to rest
                    V[t_idx, ix, :] = -85.0

    return times, V


class TestActivationTime:
    def test_traveling_wave(self):
        times, V = _make_traveling_wave(Nx=41, Ny=11, cv_idx_per_save=2)
        lat = activation_time(V, times, threshold=-20.0)

        assert lat.shape == (41, 11)
        # Node (0, 0) activates first
        # Node (10, 0) activates at t=10/2=5ms
        assert lat[0, 0].item() == pytest.approx(0.0, abs=1.0)
        assert lat[10, 0].item() == pytest.approx(5.0, abs=1.0)

        # Later nodes activate later
        assert lat[20, 0] > lat[10, 0]

    def test_unactivated_is_nan(self):
        times, V = _make_traveling_wave(Nx=41, Ny=11, n_saves=5, cv_idx_per_save=2)
        lat = activation_time(V, times, threshold=-20.0)

        # Far-right nodes never activated in 5ms with cv=2idx/save
        assert torch.isnan(lat[30, 0])

    def test_uniform_along_y(self):
        """Planar wave in x → all y nodes at same x activate simultaneously."""
        times, V = _make_traveling_wave(Nx=41, Ny=11, cv_idx_per_save=2)
        lat = activation_time(V, times)

        # At x=10, all y should have same activation time
        row = lat[10, :]
        valid = row[~torch.isnan(row)]
        assert valid.std() < 1e-10


class TestConductionVelocity:
    def test_known_cv(self):
        times, V = _make_traveling_wave(Nx=41, Ny=11, cv_idx_per_save=2)
        dx = 0.025  # cm
        # CV = 2 indices/ms * 0.025 cm/index = 0.05 cm/ms = 50 cm/s
        cv = conduction_velocity(V, times, dx, x1=5, x2=15, y=5)
        assert cv == pytest.approx(50.0, rel=0.2)

    def test_unactivated_returns_nan(self):
        times, V = _make_traveling_wave(Nx=41, Ny=11, n_saves=3, cv_idx_per_save=2)
        cv = conduction_velocity(V, times, 0.025, x1=5, x2=35, y=5)
        assert cv != cv  # NaN


class TestAPD:
    def test_apd_at_known(self):
        times, V = _make_traveling_wave(Nx=41, Ny=11, n_saves=100, cv_idx_per_save=2)
        # AP at x=0: activates at t~0, plateau 30ms, repol 30ms = APD ~60ms
        apd = apd_at(V, times, ix=0, iy=5, repol=0.9)
        # Should be between 50 and 70 ms (approximate due to discrete sampling)
        assert 40.0 < apd < 70.0

    def test_unactivated_returns_nan(self):
        times, V = _make_traveling_wave(Nx=41, Ny=11, n_saves=5, cv_idx_per_save=2)
        apd = apd_at(V, times, ix=30, iy=5)
        assert apd != apd  # NaN

    def test_apd_map_shape(self):
        times, V = _make_traveling_wave(Nx=21, Ny=11, n_saves=100, cv_idx_per_save=2)
        result = apd_map(V, times, repol=0.9)
        assert result.shape == (21, 11)


class TestDominantFrequency:
    def test_sinusoidal(self):
        """Pure 5 Hz sine wave → dominant frequency ~5 Hz."""
        n = 1000
        dt_ms = 1.0  # 1ms spacing = 1000 Hz sampling
        times = torch.arange(n, dtype=torch.float64) * dt_ms
        freq_hz = 5.0
        trace = torch.sin(2 * torch.pi * freq_hz * times / 1000.0)

        V = trace.unsqueeze(1).unsqueeze(2).expand(n, 3, 3)
        df = dominant_frequency(V, times, 1, 1)
        assert df == pytest.approx(5.0, abs=1.0)


class TestWavefrontMask:
    def test_wavefront_detection(self):
        Nx, Ny = 21, 11
        V_snap = torch.full((Nx, Ny), -85.0)
        V_snap[:10, :] = 20.0  # left half activated

        front = wavefront_mask(V_snap, threshold=-20.0)
        assert front.shape == (Nx, Ny)

        # Wavefront should be at x=9 (last activated row, has resting neighbor at x=10)
        assert front[9, 5].item() == True
        # Interior of activated region should NOT be wavefront
        assert front[5, 5].item() == False
        # Resting region should NOT be wavefront
        assert front[15, 5].item() == False


class TestPhaseMap:
    def test_shape(self):
        """Phase map should have correct shape."""
        times, V = _make_traveling_wave(Nx=21, Ny=11, n_saves=100)
        phase = phase_map(V, times, t_idx=50)
        assert phase.shape == (21, 11)

    def test_range(self):
        """Phase should be in [-pi, pi]."""
        times, V = _make_traveling_wave(Nx=21, Ny=11, n_saves=100)
        phase = phase_map(V, times, t_idx=50)
        assert phase.min() >= -torch.pi - 0.01
        assert phase.max() <= torch.pi + 0.01


class TestPhaseSingularities:
    def test_no_singularity_planar_wave(self):
        """Planar wave should have no singularities."""
        times, V = _make_traveling_wave(Nx=21, Ny=11, n_saves=100)
        phase = phase_map(V, times, t_idx=50)
        charge = phase_singularities(phase)
        # No charge should be near ±1
        assert charge.abs().max() < 0.5

    def test_output_shape(self):
        phase = torch.randn(21, 11)
        charge = phase_singularities(phase)
        assert charge.shape == (20, 10)


class TestRestitutionCurve:
    def _make_multi_beat(self):
        """Create synthetic multi-beat voltage trace."""
        n_saves = 500
        times = torch.arange(n_saves, dtype=torch.float64)
        V = torch.full((n_saves, 5, 5), -85.0, dtype=torch.float64)

        # 3 beats at BCL ~150ms
        for beat_start in [10, 160, 310]:
            for dt in range(50):  # 50ms AP
                t = beat_start + dt
                if t >= n_saves:
                    break
                if dt < 5:
                    V[t, :, :] = 20.0  # upstroke
                elif dt < 30:
                    V[t, :, :] = 10.0  # plateau
                else:
                    frac = (dt - 30) / 20
                    V[t, :, :] = 10.0 - frac * 95.0  # repol

        return times, V

    def test_returns_tensors(self):
        times, V = self._make_multi_beat()
        DI, APD = restitution_curve(V, times, ix=2, iy=2)
        assert isinstance(DI, torch.Tensor)
        assert isinstance(APD, torch.Tensor)

    def test_multiple_beats(self):
        times, V = self._make_multi_beat()
        DI, APD = restitution_curve(V, times, ix=2, iy=2)
        # Should detect at least 1 DI-APD pair from 3 beats
        assert len(DI) >= 1
        assert len(APD) >= 1
        assert len(DI) == len(APD)
