"""Tests for the public Stim object."""

import numpy as np
import pytest
import torch

from cardiac_core import Grid, Stim
from cardiac_core import geometry


def _grid(Nx=20, Ny=12, dx=0.05):
    return Grid(Nx, Ny, dx)


# ======================================================================
# the Stim class + presets + lowering
# ======================================================================

class TestStimConstruction:
    def test_boundary_constructor(self):
        g = _grid()
        s = Stim.boundary(g, "left", bcl=1000, num_pulses=3)
        # grid-only width default (~2*dx=0.1) → the two leftmost x-columns, all y; cross-check geometry
        assert s.mask.dtype == bool and s.mask.shape == (20, 12)
        assert np.array_equal(s.mask, geometry.left_edge_mask(20, 12, 0.05, 2 * 0.05))
        assert s.n_nodes() > 0
        d = s.to_dict()
        assert d['mask'].dtype == bool and d['amplitude'] == -52.0
        assert s.times() == [0.0, 1000.0, 2000.0]
        # the other sides hit the right columns/rows
        assert Stim.boundary(g, "right").mask[-1, :].all()
        assert Stim.boundary(g, "top").mask[:, -1].all()
        assert Stim.boundary(g, "bottom").mask[:, 0].all()

    def test_point_center_region(self):
        g = _grid()
        pt = Stim.point(g, (0.5, 0.3))          # a blob at (0.5, 0.3) cm
        assert pt.n_nodes() > 0 and pt.mask[10, 6]  # node nearest (0.5,0.3) = (10,6)
        ctr = Stim.center(g)
        assert ctr.n_nodes() > 0
        # explicit mask passthrough via the base constructor
        m = np.zeros((20, 12), dtype=bool); m[3:5, 3:5] = True
        assert Stim(m).n_nodes() == 4
        # from_region (callable) == boundary("left") for the same rule
        fr = Stim.from_region(g, lambda x, y: x < 2 * 0.05)
        assert np.array_equal(fr.mask, Stim.boundary(g, "left").mask)

    def test_to_from_dict(self):
        g = _grid()
        s = Stim.boundary(g, "left", amplitude=-80, start_time=1.0, duration=2.0)
        s2 = Stim.from_dict(s.to_dict())
        assert np.array_equal(s2.mask, s.mask) and s2.amplitude == -80 and s2.start_time == 1.0
        assert s2.to_dict()['mask'].dtype == bool
        # to_dict on a CLAMP stim raises (current-mode only)
        with pytest.raises(ValueError, match="current-mode"):
            Stim.boundary(g, "left", clamp=-85).to_dict()

    def test_validation(self):
        g = _grid()
        with pytest.raises(ValueError):
            Stim.boundary(g, "north")                                   # bad side
        with pytest.raises(ValueError):
            Stim(np.zeros((5, 5, 5), dtype=bool))                      # wrong-shape mask (3-D)
        with pytest.raises(ValueError, match="not both"):
            Stim(np.ones((4, 4), dtype=bool), amplitude=-80, clamp=-85)  # ambiguous mode
        with pytest.raises(ValueError, match="periodic"):
            Stim.boundary(g, "left", clamp=-20, bcl=1000, num_pulses=5)   # periodic clamp unsupported

    def test_mode_inference(self):
        g = _grid()
        assert Stim.boundary(g, "left").mode == "current"
        assert Stim.boundary(g, "left", clamp=-85).mode == "clamp"
        assert Stim.boundary(g, "left", clamp=-85).clamp == -85.0

    def test_is_one_type(self):
        g = _grid()
        assert type(Stim.boundary(g, "left")) is type(Stim.point(g, (0.1, 0.1))) is Stim

    def test_torch_mask_accepted(self):
        m = torch.zeros(20, 12, dtype=torch.bool); m[0, :] = True
        s = Stim(m)
        assert isinstance(s.mask, np.ndarray) and s.mask.dtype == bool and s.mask[0, :].all()


# ======================================================================
# non-breaking coexistence at the _normalize_stimulus seam
# ======================================================================

from cardiac_core import monodomain, bidomain, lbm, ConductivityConfig, create_cardiac_mesh
from cardiac_core.api import _normalize_stimulus, _partition_stimulus

_COND = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)


def _region_V(r, mask):
    """Last-frame membrane voltage at the masked nodes (numpy)."""
    return r.Vm[-1].detach().cpu().numpy()[mask]


class TestCoexistence:
    def test_current_stim_lowers_like_dict(self):
        """A current Stim through a factory yields the SAME data.stimuli entry as the equal dict."""
        g = Grid(20, 20, 0.05)
        stim = Stim.boundary(g, "left", amplitude=-52.0, start_time=1.0, duration=2.0)
        d = {'region': (lambda x, y: x < 2 * 0.05), 'start_time': 1.0, 'duration': 2.0,
             'amplitude': -52.0}
        s_obj = monodomain(g, 'ttp06', _COND, stim)
        s_dict = monodomain(g, 'ttp06', _COND, d)
        eo, ed = s_obj._data.stimuli[0], s_dict._data.stimuli[0]
        assert np.array_equal(eo['mask'], ed['mask'])
        for k in ('amplitude', 'start_time', 'duration', 'bcl', 'num_pulses'):
            assert eo[k] == ed[k], k

    def test_normalize_accepts_stim_and_list(self):
        g = Grid(20, 20, 0.05)
        coords = g.coordinates
        one = _normalize_stimulus(Stim.boundary(g, "left"), coords)
        assert len(one) == 1 and one[0]['mask'].dtype == bool
        mixed = _normalize_stimulus(
            [Stim.boundary(g, "left"), {'region': (lambda x, y: x > 0.9), 'amplitude': -60.0}],
            coords)
        assert len(mixed) == 2 and mixed[1]['amplitude'] == -60.0

    def test_normalize_raises_on_clamp_stim(self):
        g = Grid(20, 20, 0.05)
        with pytest.raises(ValueError, match="clamp"):
            _normalize_stimulus(Stim.boundary(g, "left", clamp=-85.0), g.coordinates)

    def test_stimulate_accepts_current_stim(self):
        g = Grid(20, 20, 0.05)
        sim = monodomain(g, 'ttp06', _COND, stimulus=None)
        n0 = len(sim._data.stimuli)
        sim.stimulate(Stim.boundary(g, "left", start_time=0.0, duration=2.0, amplitude=-52.0))
        assert len(sim._data.stimuli) == n0 + 1
        assert sim._data.stimuli[-1]['mask'].any()


# ======================================================================
# voltage-clamp mode: factory routing + native LBM clamp
# ======================================================================

REST = -85.23   # TTP06 ENDO V_rest


class TestPartition:
    def test_partition_splits_clamp_from_current(self):
        g = Grid(20, 20, 0.05)
        cur, clamp = _partition_stimulus(
            [Stim.boundary(g, "left"), {'region': lambda x, y: x > 0.9},
             Stim.point(g, (0.5, 0.5), clamp=20)])
        assert len(cur) == 2 and len(clamp) == 1 and clamp[0].mode == "clamp"
        assert _partition_stimulus(None) == (None, [])
        # a lone clamp Stim → current is None (nothing reaches data.stimuli)
        cur2, clamp2 = _partition_stimulus(Stim.boundary(g, "left", clamp=-85))
        assert cur2 is None and len(clamp2) == 1


class TestClampRouting:
    def test_clamp_stim_not_in_data_stimuli(self):
        """The BLOCKER guard: a clamp Stim goes to the clamp mechanism, NEVER data.stimuli."""
        g = Grid(20, 20, 0.05)
        sim = monodomain(g, 'ttp06', _COND, Stim.boundary(g, "left", clamp=-85))
        assert sim._data.stimuli == []           # not lowered as a current
        assert sim._clamp_mask is not None        # applied via clamp_voltage

    def test_mixed_current_and_clamp(self):
        g = Grid(20, 20, 0.05)
        cpt = (0.5, 0.5)
        sim = monodomain(g, 'ttp06', _COND,
                         [Stim.boundary(g, "left", start_time=0.0, duration=2.0),
                          Stim.point(g, cpt, clamp=20.0, start_time=0.0, duration=50.0)])
        assert len(sim._data.stimuli) == 1        # the current stim
        assert sim._clamp_mask is not None        # the clamp point
        r = sim.run(t_end=4.0, save_every=1.0)
        pmask = Stim.point(g, cpt).mask
        assert np.allclose(_region_V(r, pmask), 20.0, atol=1e-3)   # clamp point held

    def test_legacy_mesh_stim_dropped(self):
        """stimulus= on the mesh= path is dropped for BOTH current and clamp (no asymmetry).

        The mesh keeps its OWN baked-in stimuli; the ``stimulus=`` clamp Stim is neither lowered
        into ``data.stimuli`` (as a current) NOR applied via the clamp mechanism.
        """
        mesh = create_cardiac_mesh(1.0, 0.5, 0.05, D=1.4, chi=1400.0)
        g = Grid(21, 11, 0.05)
        sim = monodomain(mesh, stimulus=Stim.boundary(g, "left", clamp=-85))
        assert len(sim._data.stimuli) == len(mesh.stimuli)   # unchanged — the Stim was dropped
        assert sim._clamp_mask is None                       # clamp NOT applied on the legacy path

    def test_clamp_periodic_rejected(self):
        g = Grid(20, 20, 0.05)
        with pytest.raises(ValueError, match="periodic"):
            Stim.boundary(g, "left", clamp=-20, bcl=1000, num_pulses=5)


class TestClampHolds:
    @pytest.mark.parametrize("engine", [monodomain, bidomain, lbm])
    def test_clamp_holds_voltage(self, engine):
        """A clamp Stim holds V near its value on ALL THREE engines."""
        g = Grid(20, 12, 0.05)
        sim = engine(g, 'ttp06', _COND,
                     Stim.boundary(g, "left", clamp=-20.0, start_time=0.0, duration=50.0))
        r = sim.run(t_end=5.0, save_every=1.0)
        left = Stim.boundary(g, "left").mask
        assert np.allclose(_region_V(r, left), -20.0, atol=1e-2)

    def test_clamp_vs_current(self):
        """Current (default amp) drives an AP; a rest-level clamp holds rest (no AP)."""
        g = Grid(30, 6, 0.05)
        cur = monodomain(g, 'ttp06', _COND,
                         Stim.boundary(g, "left", start_time=0.0, duration=2.0))
        clamp = monodomain(g, 'ttp06', _COND,
                           Stim.boundary(g, "left", clamp=-85.0, start_time=0.0, duration=50.0))
        rc = cur.run(t_end=12.0, save_every=1.0)
        rk = clamp.run(t_end=12.0, save_every=1.0)
        assert rc.Vm.max().item() > 0.0          # an AP fired
        assert rk.Vm.max().item() < -50.0        # held near rest, nothing propagated

    def test_lbm_clamp_survives_reset(self):
        """A native LBM clamp still holds V after stimulate() (which calls reset())."""
        g = Grid(30, 4, 0.05)
        sim = lbm(g, 'ttp06', _COND,
                  Stim.boundary(g, "left", clamp=-20.0, start_time=0.0, duration=50.0))
        sim.stimulate(Stim.point(g, (1.0, 0.1), start_time=0.0, duration=2.0, amplitude=-52.0))
        r = sim.run(t_end=4.0, save_every=1.0)
        left = Stim.boundary(g, "left").mask
        assert np.allclose(_region_V(r, left), -20.0, atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    def test_lbm_clamp_cuda(self):
        """LBM clamp on cuda holds V (guards numpy-mask indexing a CUDA f tensor)."""
        g = Grid(20, 8, 0.05)
        sim = lbm(g, 'ttp06', _COND,
                  Stim.boundary(g, "left", clamp=-20.0, start_time=0.0, duration=50.0),
                  device='cuda')
        r = sim.run(t_end=4.0, save_every=1.0)
        left = Stim.boundary(g, "left").mask
        assert np.allclose(_region_V(r, left), -20.0, atol=1e-2)


class TestDeprecation:
    """The dict path is soft-deprecated (warns) but still works; Stim never warns."""

    def test_dict_warns(self):
        g = Grid(20, 12, 0.05)
        with pytest.warns(DeprecationWarning, match="Stim"):
            monodomain(g, 'ttp06', _COND, {'region': lambda x, y: x < 0.1})
        # a Stim (any mode) never warns; nor does stimulate() with a Stim OR a callable/mask
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", DeprecationWarning)
            monodomain(g, 'ttp06', _COND, Stim.boundary(g, "left"))
            sim = monodomain(g, 'ttp06', _COND, stimulus=None)
            sim.stimulate(Stim.boundary(g, "left", start_time=0.0, duration=2.0))
            sim.stimulate(lambda x, y: x < 0.1)          # builds a Stim internally → no warn

    def test_dict_path_unchanged(self):
        """Back-compat guard: the legacy dict still lowers to the same stimulus entry as a Stim."""
        g = Grid(20, 12, 0.05)
        with pytest.warns(DeprecationWarning):
            s_dict = monodomain(g, 'ttp06', _COND,
                                {'region': lambda x, y: x < 0.1, 'amplitude': -52.0,
                                 'start_time': 1.0, 'duration': 2.0})
        s_stim = monodomain(g, 'ttp06', _COND,
                            Stim.from_region(g, lambda x, y: x < 0.1, amplitude=-52.0,
                                             start_time=1.0, duration=2.0))
        ed, es = s_dict._data.stimuli[0], s_stim._data.stimuli[0]
        assert np.array_equal(ed['mask'], es['mask'])
        for k in ('amplitude', 'start_time', 'duration'):
            assert ed[k] == es[k], k


class TestLBMClampPhysics:
    def test_lbm_clamp_matches_mono(self):
        """Arbiter: the additive LBM clamp holds V AND conducts inward, like the mono hard-write
        clamp (ground truth). Different CV is fine — both must hold the value AND spread."""
        cval = 20.0
        left_slug, inner_col = "left", 6
        results = {}
        for engine in (monodomain, lbm):
            g = Grid(30, 4, 0.05)
            sim = engine(g, 'ttp06', _COND,
                         Stim.boundary(g, left_slug, clamp=cval, start_time=0.0, duration=50.0))
            r = sim.run(t_end=10.0, save_every=1.0)
            left = Stim.boundary(g, left_slug).mask
            Vgrid = r.Vm[-1].detach().cpu().numpy()
            results[engine] = (_region_V(r, left), Vgrid[inner_col, :].mean())
        for engine, (Vleft, Vinner) in results.items():
            assert np.allclose(Vleft, cval, atol=1e-2), f"{engine.__name__}: clamp not held"
            assert Vinner > -40.0, f"{engine.__name__}: no inward spread (Vinner={Vinner:.1f})"

    def test_lbm_clamp_preserves_nonequilibrium(self):
        """Mechanistic arbiter (retires the pure-reset scheme): the additive correction drives
        Σf→value EXACTLY while leaving the non-equilibrium (flux-carrying) part f^neq UNCHANGED;
        a pure reset f=w·value would zero f^neq instead."""
        g = Grid(10, 6, 0.05)
        eng = lbm(g, 'ttp06', _COND)._engine
        w = eng.w                                      # (Q,), Σw = 1
        assert torch.isclose(w.sum(), torch.tensor(1.0, dtype=w.dtype), atol=1e-12)
        V0 = -30.0
        neq = torch.zeros_like(w); neq[1] = 0.7; neq[2] = -0.7   # Σneq = 0, flux-carrying
        f_pre = w * V0 + neq
        assert torch.isclose(f_pre.sum(), torch.tensor(V0, dtype=w.dtype), atol=1e-12)
        value = 15.0
        # additive (the implemented scheme): f += w·(value − Σf)
        f_add = f_pre + w * (value - f_pre.sum())
        assert torch.isclose(f_add.sum(), torch.tensor(value, dtype=w.dtype), atol=1e-12)  # Σf→value
        assert torch.allclose(f_add - w * value, neq, atol=1e-12)                          # f^neq preserved
        # pure reset (the rejected alternative): f = w·value → f^neq zeroed, so the schemes differ
        assert torch.allclose(w * value - w * value, torch.zeros_like(w))
        assert not torch.allclose(neq, torch.zeros_like(neq))
