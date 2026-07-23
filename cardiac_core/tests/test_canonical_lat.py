"""SimulationResult analysis-context + canonical interpolated LAT.

Covers the result carrying mask/boundary_mode/Cm/chi/conductivity/model identity on BOTH
build paths, and the torch interpolated activation time (LAT) plus max_dvdt_time. The
default-method flip and CV-family routing are additionally exercised by the golden
comparison in test_analysis.py.
"""

import dataclasses
import math

import numpy as np
import pytest
import torch

from cardiac_core import (
    monodomain, bidomain, Grid, ConductivityConfig, create_cardiac_mesh, simulate, Stim,
)
from cardiac_core.analysis import activation_time, max_dvdt_time
from cardiac_core.ionic.registry import build_ionic_model


def _stim(g):
    return Stim.from_region(g, (lambda x, y: x < 0.06),
                            start_time=1.0, duration=2.0, amplitude=-52.0)


def _cond():
    return ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)


# ======================================================================
# result carries analysis context
# ======================================================================

class TestResultContext:
    def test_result_carries_context(self):
        g = Grid(16, 12, 0.05)
        r = monodomain(g, 'ttp06', _cond(), _stim(g)).run(2.0, save_every=1.0)
        assert r.domain_mask is None            # full rectangle -> no mask
        assert r.boundary_mode == 'face_mirror'
        assert r.Cm == 1.0

        mask = np.ones((16, 12), dtype=bool)
        mask[6:9, 5:8] = False                  # an interior hole
        gm = Grid(16, 12, 0.05, mask=mask)
        rm = monodomain(gm, 'ttp06', _cond(), _stim(gm)).run(2.0, save_every=1.0)
        assert rm.domain_mask is not None
        assert tuple(rm.domain_mask.shape) == (16, 12)
        assert rm.domain_mask.dtype == torch.bool
        assert rm.domain_mask.device == rm.Vm.device

    def test_declarative_chi_is_one(self):
        # R2-critical guard: declarative monodomain sets data.chi=1 (Form-A), so
        # D_eff = D_raw/(chi*Cm) — NOT off by a hard-coded 1400.
        g = Grid(16, 12, 0.05)
        r = monodomain(g, 'ttp06', _cond(), _stim(g)).run(2.0, save_every=1.0)
        assert r.chi == 1.0
        c = r.conductivity
        assert c is not None and c.D_eff is not None and c.D_raw is not None
        expect = c.D_raw / (r.chi * r.Cm)
        assert torch.allclose(c.D_eff, expect, atol=1e-12)
        # physiological ~1e-3 cm^2/ms, NOT ~1e-3/1400 (the hard-coded-1400 failure mode)
        assert 3e-4 < float(c.D_eff.mean()) < 5e-3

    def test_both_builders_thread(self):
        # .run() path (api._result_from) AND simulate() path (run._collect) both populate ctx.
        g = Grid(16, 12, 0.05)
        r_run = monodomain(g, 'ttp06', _cond(), _stim(g)).run(2.0, save_every=1.0)
        mesh = create_cardiac_mesh(Lx=0.8, Ly=0.6, dx=0.05)   # ttp06 default
        r_sim = simulate(mesh, t_end=2.0, save_every=1.0, engine='monodomain')
        for r in (r_run, r_sim):
            assert r.boundary_mode == 'face_mirror'
            assert r.Cm is not None and r.chi is not None
            assert r.conductivity is not None
            assert r.ionic_model == 'ttp06'
            assert r.cell_type == 'ENDO'

    def test_bidomain_result_has_sigma(self):
        g = Grid(16, 12, 0.05)
        r = bidomain(g, 'ttp06', _cond(), _stim(g)).run(2.0, save_every=1.0)
        c = r.conductivity
        assert c is not None and c.is_bidomain
        assert c.sigma_i is not None and c.sigma_e is not None
        assert c.D_eff is None
        assert r.phi_e is not None

    def test_result_carries_model_identity(self):
        # EPI mesh -> cell_type EPI; the mesh's resolved ionic name is carried and rebuildable.
        mesh = create_cardiac_mesh(Lx=0.8, Ly=0.6, dx=0.05, ionic_model='ttp06')
        mesh_epi = dataclasses.replace(mesh, group_cell_types=['EPI'])
        r = monodomain(mesh_epi).run(2.0, save_every=1.0)
        assert r.cell_type == 'EPI'
        assert r.ionic_model == 'ttp06'
        model = build_ionic_model(r.ionic_model, r.cell_type, device=r.Vm.device)
        assert model is not None

    def test_ionic_name_is_resolved_not_stale(self):
        # An explicit ionic_model= override on a legacy mesh -> the RESOLVED name is carried,
        # not the stale data.ionic_model.
        mesh = create_cardiac_mesh(Lx=0.6, Ly=0.4, dx=0.05, ionic_model='ttp06')
        r = monodomain(mesh, ionic_model='phas13').run(0.2, save_every=0.1)
        assert r.ionic_model == 'phas13'

    def test_bidomain_cell_type_is_endo_not_mesh_group(self):
        # The bidomain/LBM factories force ENDO (no cell_type is passed to build_ionic_model),
        # so r.cell_type must be the ACTUALLY-RUN ENDO, not the mesh's group_cell_types[0].
        mesh = create_cardiac_mesh(Lx=0.8, Ly=0.6, dx=0.05)
        mesh_epi = dataclasses.replace(mesh, group_cell_types=['EPI'])
        r = bidomain(mesh_epi).run(2.0, save_every=1.0)
        assert r.cell_type == 'ENDO'      # what the engine ran, NOT 'EPI'


# ======================================================================
# canonical interpolated LAT + max_dvdt_time
# ======================================================================

class TestCanonicalLAT:
    def _synthetic(self, device='cpu'):
        times = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, device=device)
        # crosses -40 between t=1 (-60) and t=2 (0): frac=(-40+60)/(0+60)=1/3 -> lat=1.3333
        V = torch.tensor([-80.0, -60.0, 0.0], dtype=torch.float64, device=device).reshape(3, 1, 1)
        return V, times

    def test_interp_subframe(self):
        V, times = self._synthetic()
        lat = activation_time(V, times, threshold=-40.0, method='interp')
        assert abs(lat[0, 0].item() - (1.0 + 1.0 / 3.0)) < 1e-9

    def test_nearest_override_reproduces_frame_quantized(self):
        V, times = self._synthetic()
        lat_near = activation_time(V, times, threshold=-40.0, method='nearest')
        assert lat_near[0, 0].item() == 2.0        # first frame >= -40 is t=2

    def test_default_is_canonical_interp(self):
        # The DEFAULT is interp/-40 (crossing at 1.3333), NOT nearest/-20.
        V, times = self._synthetic()
        assert abs(activation_time(V, times)[0, 0].item() - (1.0 + 1.0 / 3.0)) < 1e-9

    def test_nonactivating_is_nan(self):
        times = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        V = torch.tensor([-80.0, -75.0, -70.0], dtype=torch.float64).reshape(3, 1, 1)
        lat = activation_time(V, times, threshold=-40.0, method='interp')
        assert math.isnan(lat[0, 0].item())

    def test_activated_at_t0(self):
        times = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        V = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float64).reshape(3, 1, 1)  # above at t0
        lat = activation_time(V, times, threshold=-40.0, method='interp')
        assert lat[0, 0].item() == 0.0

    def test_interp_matches_numpy_reference(self):
        # torch interp must agree with the numpy activation_time_interp on a random smooth field.
        from cardiac_core.analysis import activation_time_interp
        torch.manual_seed(0)
        T, Nx, Ny = 12, 8, 6
        ramp = torch.linspace(-85, 30, T, dtype=torch.float64).reshape(T, 1, 1)
        V = ramp + 3.0 * torch.rand(T, Nx, Ny, dtype=torch.float64)
        times = torch.linspace(0, 22, T, dtype=torch.float64)
        lat_t = activation_time(V, times, threshold=-40.0, method='interp')
        lat_n = torch.as_tensor(activation_time_interp(V.numpy(), times.numpy(), -40.0))
        both = torch.isfinite(lat_t) & torch.isfinite(lat_n)
        assert both.all()          # everything activates on this ramp
        assert torch.allclose(lat_t[both], lat_n[both], atol=1e-9)

    def test_invalid_method_raises(self):
        V, times = self._synthetic()
        with pytest.raises(ValueError):
            activation_time(V, times, method='bogus')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
    def test_on_cuda(self):
        V, times = self._synthetic(device='cuda')
        lat = activation_time(V, times, threshold=-40.0, method='interp')
        assert lat.device.type == 'cuda'
        assert abs(lat[0, 0].item() - (1.0 + 1.0 / 3.0)) < 1e-9

    def test_max_dvdt_parabolic(self):
        times = torch.linspace(0.0, 10.0, 11, dtype=torch.float64)
        mid = 5.3
        V = (100.0 / (1.0 + torch.exp(-(times - mid)))).reshape(11, 1, 1) - 80.0
        tp = max_dvdt_time(V, times)
        assert abs(tp[0, 0].item() - mid) < 0.5     # sub-frame vertex near the logistic midpoint

    def test_max_dvdt_too_few_frames_nan(self):
        times = torch.tensor([0.0, 1.0], dtype=torch.float64)
        V = torch.tensor([-80.0, 0.0], dtype=torch.float64).reshape(2, 1, 1)
        assert math.isnan(max_dvdt_time(V, times)[0, 0].item())


class TestCVFamilyCanonical:
    """The scalar CV family DEFAULTS to interp/-40, and the method='nearest', threshold=-20
    OVERRIDE reproduces the historical frame-quantized value. Uses a SMOOTH two-node trace
    with DIFFERENT upstroke slopes, so interp/-40 and nearest/-20 give DIFFERENT CV — a
    step-wave synthetic cannot distinguish the two and would leave the behavior untested."""

    def _two_node(self):
        from cardiac_core.analysis import conduction_velocity  # noqa: F401 (import check)
        times = torch.arange(9, dtype=torch.float64)          # 0..8
        # Node A rises 25 mV/frame; Node B 12.5 mV/frame (both from -80).
        vA = -80.0 + 25.0 * times                              # crosses -40@1.6, -20@2.4 (nearest 3)
        vB = -80.0 + 12.5 * times                              # crosses -40@3.2, -20@4.8 (nearest 5)
        V = torch.stack([vA, vB], dim=1).reshape(9, 2, 1)      # (T, Nx=2, Ny=1)
        return V, times

    def test_conduction_velocity_default_is_interp40(self):
        from cardiac_core.analysis import conduction_velocity
        V, times = self._two_node()
        # interp/-40: dt = 3.2-1.6 = 1.6 -> 0.1/1.6*1000 = 62.5 ; nearest/-20: dt = 5-3 = 2 -> 50
        assert conduction_velocity(V, times, 0.1, 0, 1, 0) == pytest.approx(62.5, rel=1e-6)
        assert conduction_velocity(V, times, 0.1, 0, 1, 0,
                                   threshold=-20.0, method='nearest') == pytest.approx(50.0, rel=1e-6)

    def test_cv_between_default_is_interp40(self):
        from cardiac_core.analysis import cv_between
        V, times = self._two_node()
        assert cv_between(V, times, (0, 0), (1, 0), 0.1) == pytest.approx(62.5, rel=1e-6)
        assert cv_between(V, times, (0, 0), (1, 0), 0.1,
                          threshold=-20.0, method='nearest') == pytest.approx(50.0, rel=1e-6)

    def test_radial_cv_default_is_interp40(self):
        from cardiac_core.analysis import radial_cv
        V, times = self._two_node()
        rcv_i = radial_cv(V, times, (0, 0), 0.1)
        assert rcv_i[1, 0].item() == pytest.approx(62.5, rel=1e-6)
        rcv_n = radial_cv(V, times, (0, 0), 0.1, threshold=-20.0, method='nearest')
        assert rcv_n[1, 0].item() == pytest.approx(50.0, rel=1e-6)
