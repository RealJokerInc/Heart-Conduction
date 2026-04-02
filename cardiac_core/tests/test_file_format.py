"""Tests for cardiac_core.file_format — save/load round-trip and create_cardiac_mesh."""

import tempfile
import os
import numpy as np
import pytest

from cardiac_core.file_format import (
    CardiacMeshData,
    save_cardiac_mesh,
    load_cardiac_mesh,
    create_cardiac_mesh,
)


class TestRoundTrip:
    """Save → load round-trip preserves all fields."""

    def _make_data(self, with_bidomain=False):
        Nx, Ny = 21, 11
        mask = np.ones((Nx, Ny), dtype=bool)
        mask[0, :] = False  # knock out left column

        stim_mask = np.zeros((Nx, Ny), dtype=bool)
        stim_mask[1:5, :] = True

        data = CardiacMeshData(
            dx=0.025,
            dy=0.025,
            mask=mask,
            D_xx=np.full((Nx, Ny), 0.001),
            D_yy=np.full((Nx, Ny), 0.0008),
            D_xy=np.full((Nx, Ny), 0.0001),
            chi=1400.0,
            Cm=1.0,
            ionic_model='ttp06',
            dt=0.02,
            group_labels=['myocardium', 'scar'],
            group_cell_types=['ENDO', 'EPI'],
            stimuli=[{
                'mask': stim_mask,
                'label': 'S1',
                'amplitude': -80.0,
                'duration': 2.0,
                'start_time': 1.0,
                'bcl': 1000.0,
                'num_pulses': 3,
            }],
            boundary='insulated',
        )

        if with_bidomain:
            data.sigma_i = (
                np.full((Nx, Ny), 1.74),
                np.full((Nx, Ny), 0.174),
                np.zeros((Nx, Ny)),
            )
            data.sigma_e = (
                np.full((Nx, Ny), 6.25),
                np.full((Nx, Ny), 2.36),
                np.zeros((Nx, Ny)),
            )

        return data

    def test_basic_round_trip(self, tmp_path):
        data = self._make_data()
        path = str(tmp_path / "test.npz")
        save_cardiac_mesh(path, data)
        loaded = load_cardiac_mesh(path)

        assert loaded.dx == data.dx
        assert loaded.dy == data.dy
        assert loaded.chi == data.chi
        assert loaded.Cm == data.Cm
        assert loaded.ionic_model == data.ionic_model
        assert loaded.dt == data.dt
        assert loaded.boundary == data.boundary
        np.testing.assert_array_equal(loaded.mask, data.mask)
        np.testing.assert_array_almost_equal(loaded.D_xx, data.D_xx)
        np.testing.assert_array_almost_equal(loaded.D_yy, data.D_yy)
        np.testing.assert_array_almost_equal(loaded.D_xy, data.D_xy)

    def test_group_metadata(self, tmp_path):
        data = self._make_data()
        path = str(tmp_path / "test.npz")
        save_cardiac_mesh(path, data)
        loaded = load_cardiac_mesh(path)

        assert loaded.group_labels == data.group_labels
        assert loaded.group_cell_types == data.group_cell_types

    def test_stimulus_round_trip(self, tmp_path):
        data = self._make_data()
        path = str(tmp_path / "test.npz")
        save_cardiac_mesh(path, data)
        loaded = load_cardiac_mesh(path)

        assert len(loaded.stimuli) == 1
        s = loaded.stimuli[0]
        np.testing.assert_array_equal(s['mask'], data.stimuli[0]['mask'])
        assert s['label'] == 'S1'
        assert s['amplitude'] == -80.0
        assert s['duration'] == 2.0
        assert s['start_time'] == 1.0
        assert s['bcl'] == 1000.0
        assert s['num_pulses'] == 3

    def test_bidomain_fields(self, tmp_path):
        data = self._make_data(with_bidomain=True)
        path = str(tmp_path / "test.npz")
        save_cardiac_mesh(path, data)
        loaded = load_cardiac_mesh(path)

        assert loaded.sigma_i is not None
        assert loaded.sigma_e is not None
        np.testing.assert_array_almost_equal(loaded.sigma_i[0], data.sigma_i[0])
        np.testing.assert_array_almost_equal(loaded.sigma_e[1], data.sigma_e[1])

    def test_no_bidomain_fields(self, tmp_path):
        data = self._make_data(with_bidomain=False)
        path = str(tmp_path / "test.npz")
        save_cardiac_mesh(path, data)
        loaded = load_cardiac_mesh(path)

        assert loaded.sigma_i is None
        assert loaded.sigma_e is None


class TestCreateCardiacMesh:
    """create_cardiac_mesh() produces valid data with correct defaults."""

    def test_default_rectangle(self):
        mesh = create_cardiac_mesh(Lx=2.0, Ly=0.5, dx=0.025)
        Nx = round(2.0 / 0.025) + 1   # 81
        Ny = round(0.5 / 0.025) + 1    # 21

        assert mesh.mask.shape == (Nx, Ny)
        assert mesh.mask.all()
        assert mesh.dx == 0.025
        assert mesh.dy == 0.025
        assert mesh.D_xx.shape == (Nx, Ny)
        np.testing.assert_allclose(mesh.D_xx, 0.001)
        np.testing.assert_allclose(mesh.D_yy, 0.001)
        np.testing.assert_allclose(mesh.D_xy, 0.0)
        assert mesh.ionic_model == 'ttp06'
        assert mesh.dt == 0.02

    def test_stimulus_defaults(self):
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.025)
        assert len(mesh.stimuli) == 1
        stim = mesh.stimuli[0]
        assert stim['amplitude'] == -80.0
        assert stim['duration'] == 2.0
        assert stim['start_time'] == 1.0

        # Left-edge stimulus: x < 0.1 cm
        Nx = round(1.0 / 0.025) + 1
        x_coords = np.arange(Nx) * 0.025
        expected_active = (x_coords < 0.1).sum()
        assert stim['mask'][:, 0].sum() == expected_active

    def test_custom_mask(self):
        mask = np.ones((41, 21), dtype=bool)
        mask[:10, :] = False
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.025, mask=mask)
        np.testing.assert_array_equal(mesh.mask, mask)

        # Stimulus should be intersected with mask
        stim = mesh.stimuli[0]
        assert not stim['mask'][:10, :].any()

    def test_custom_physics(self):
        mesh = create_cardiac_mesh(
            Lx=1.0, Ly=1.0, dx=0.05,
            D=0.002, ionic_model='ord', dt=0.01,
            chi=1200.0, Cm=0.8,
        )
        np.testing.assert_allclose(mesh.D_xx, 0.002)
        assert mesh.ionic_model == 'ord'
        assert mesh.dt == 0.01
        assert mesh.chi == 1200.0
        assert mesh.Cm == 0.8

    def test_round_trip_after_create(self, tmp_path):
        """create → save → load preserves everything."""
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05)
        path = str(tmp_path / "created.npz")
        save_cardiac_mesh(path, mesh)
        loaded = load_cardiac_mesh(path)

        assert loaded.dx == mesh.dx
        assert loaded.ionic_model == mesh.ionic_model
        np.testing.assert_array_equal(loaded.mask, mesh.mask)
        np.testing.assert_array_equal(loaded.stimuli[0]['mask'], mesh.stimuli[0]['mask'])
