"""
Tests for PHAS13 hiPSC-CM Ionic Model

Paci-Hyttinen-Aalto-Setala-Severi 2013.

Organized by implementation phase:
- Phase 1 (P1): parameters.py
- Phase 2 (P2): gating.py
- Phase 3 (P3): currents.py
- Phase 4 (P4): calcium.py
- Phase 5 (P5): model.py
- Phase 6 (P6): validation
"""

import pytest
import torch

# ============================================================================
# Phase 1: Parameters
# ============================================================================

class TestPHAS13Parameters:
    """P1: StateIndex, PHAS13Parameters, get_initial_state()."""

    def test_n_states(self):
        """P1-V1: N_STATES == 17."""
        from cardiac_sim.ionic.phas13.parameters import StateIndex
        assert StateIndex.N_STATES == 17

    def test_parameters_instantiate(self):
        """P1-V2: PHAS13Parameters dataclass instantiates."""
        from cardiac_sim.ionic.phas13.parameters import PHAS13Parameters
        p = PHAS13Parameters()
        assert p.Cm == pytest.approx(9.87109e-11)
        assert p.Ki == 150.0
        assert p.RTONF == pytest.approx(26.713, rel=1e-3)

    def test_initial_state_shape(self):
        """P1-V3: get_initial_state() shape == (17,)."""
        from cardiac_sim.ionic.phas13.parameters import StateIndex, get_initial_state
        state = get_initial_state(device=torch.device('cpu'))
        assert state.shape == (StateIndex.N_STATES,)
        assert state.shape == (17,)

    def test_v_rest(self):
        """P1-V4: V_REST ~ -74.3 mV."""
        from cardiac_sim.ionic.phas13.parameters import V_REST
        assert V_REST == pytest.approx(-74.334, abs=0.01)

    def test_ki_fixed(self):
        """P1-V5: Ki is fixed at 150 mM, not in state vector."""
        from cardiac_sim.ionic.phas13.parameters import (
            StateIndex, PHAS13Parameters, STATE_NAMES
        )
        p = PHAS13Parameters()
        assert p.Ki == 150.0
        # Ki should not appear in state names
        assert 'Ki' not in STATE_NAMES
        # Verify no Ki index in StateIndex (only N_STATES and actual states)
        state_indices = [s for s in StateIndex if s != StateIndex.N_STATES]
        assert len(state_indices) == 17


# ============================================================================
# Phase 2: Gating
# ============================================================================

class TestPHAS13Gating:
    """P2: Gating steady-states and time constants."""

    @pytest.fixture
    def V_range(self):
        """Voltage range for sweep tests."""
        return torch.linspace(-100.0, 50.0, 151, dtype=torch.float64)

    def test_inf_range(self, V_range):
        """P2-V1: All _inf() return [0,1] for V in [-100, +50]."""
        from cardiac_sim.ionic.phas13.gating import (
            INa_m_inf, INa_h_inf, INa_j_inf,
            ICaL_d_inf, ICaL_f1_inf, ICaL_f2_inf,
            IKr_Xr1_inf, IKr_Xr2_inf, IKs_Xs_inf,
            Ito_q_inf, Ito_r_inf, If_Xf_inf,
        )
        for fn in [INa_m_inf, INa_h_inf, INa_j_inf,
                    ICaL_d_inf, ICaL_f1_inf, ICaL_f2_inf,
                    IKr_Xr1_inf, IKr_Xr2_inf, IKs_Xs_inf,
                    Ito_q_inf, Ito_r_inf, If_Xf_inf]:
            vals = fn(V_range)
            assert (vals >= -1e-10).all(), f"{fn.__name__} below 0"
            assert (vals <= 1.0 + 1e-10).all(), f"{fn.__name__} above 1"

    def test_tau_positive(self, V_range):
        """P2-V2: All _tau() return positive for V in [-100, +50]."""
        from cardiac_sim.ionic.phas13.gating import (
            INa_m_tau, INa_h_tau, INa_j_tau,
            ICaL_d_tau, ICaL_f1_tau, ICaL_f2_tau,
            IKr_Xr1_tau, IKr_Xr2_tau, IKs_Xs_tau,
            Ito_q_tau, Ito_r_tau, If_Xf_tau,
        )
        for fn in [INa_m_tau, INa_h_tau, INa_j_tau,
                    ICaL_d_tau, ICaL_f1_tau, ICaL_f2_tau,
                    IKr_Xr1_tau, IKr_Xr2_tau, IKs_Xs_tau,
                    Ito_q_tau, Ito_r_tau, If_Xf_tau]:
            vals = fn(V_range)
            assert (vals > 0).all(), f"{fn.__name__} not positive"

    def test_m_inf_at_rest(self):
        """P2-V3: INa_m_inf(-74.3) ~ 0.103."""
        from cardiac_sim.ionic.phas13.gating import INa_m_inf
        V = torch.tensor(-74.334, dtype=torch.float64)
        assert INa_m_inf(V).item() == pytest.approx(0.103, abs=0.005)

    def test_h_inf_at_rest(self):
        """P2-V4: INa_h_inf(-74.3) ~ 0.772 (not at steady state for beating model)."""
        from cardiac_sim.ionic.phas13.gating import INa_h_inf
        V = torch.tensor(-74.334, dtype=torch.float64)
        # h_inf = 1/sqrt(1+exp((-74.334+72.1)/5.7)) = 0.7725
        assert INa_h_inf(V).item() == pytest.approx(0.7725, abs=0.005)

    def test_xf_inf_at_rest(self):
        """P2-V5: If_Xf_inf(-74.3) ~ 0.331 (If gate is far from steady state)."""
        from cardiac_sim.ionic.phas13.gating import If_Xf_inf
        V = torch.tensor(-74.334, dtype=torch.float64)
        # Xf_inf = 1/(1+exp((-74.334+77.85)/5)) = 0.331
        assert If_Xf_inf(V).item() == pytest.approx(0.331, abs=0.01)

    def test_h_j_biphasic_smooth(self, V_range):
        """P2-V6: h/j tau smooth at V=-40 (no NaN/Inf)."""
        from cardiac_sim.ionic.phas13.gating import INa_h_tau, INa_j_tau
        # Include V=-40 exactly
        V = torch.tensor([-40.001, -40.0, -39.999], dtype=torch.float64)
        h_tau = INa_h_tau(V)
        j_tau = INa_j_tau(V)
        assert torch.isfinite(h_tau).all()
        assert torch.isfinite(j_tau).all()
        assert (h_tau > 0).all()
        assert (j_tau > 0).all()

    def test_batched(self):
        """P2-V7: Batched (100,) tensor works."""
        from cardiac_sim.ionic.phas13.gating import INa_m_inf, INa_m_tau
        V = torch.randn(100, dtype=torch.float64) * 30 - 40
        m_inf = INa_m_inf(V)
        m_tau = INa_m_tau(V)
        assert m_inf.shape == (100,)
        assert m_tau.shape == (100,)


# ============================================================================
# Phase 3: Currents
# ============================================================================

class TestPHAS13Currents:
    """P3: Ion current functions."""

    @pytest.fixture
    def rest_state(self):
        """Return (V, state_dict) at initial conditions."""
        from cardiac_sim.ionic.phas13.parameters import get_initial_state, StateIndex, V_REST
        state = get_initial_state(device=torch.device('cpu'))
        V = torch.tensor(V_REST, dtype=torch.float64)
        return V, {
            'Nai': state[StateIndex.Nai],
            'Cai': state[StateIndex.Cai],
            'CaSR': state[StateIndex.CaSR],
            'm': state[StateIndex.m],
            'h': state[StateIndex.h],
            'j': state[StateIndex.j],
            'd': state[StateIndex.d],
            'f1': state[StateIndex.f1],
            'f2': state[StateIndex.f2],
            'fCa': state[StateIndex.fCa],
            'Xr1': state[StateIndex.Xr1],
            'Xr2': state[StateIndex.Xr2],
            'Xs': state[StateIndex.Xs],
            'q': state[StateIndex.q],
            'r_gate': state[StateIndex.r_gate],
            'Xf': state[StateIndex.Xf],
        }

    def test_INa_inward_at_depolarized(self):
        """P3-V1: I_Na large inward at V=-20."""
        from cardiac_sim.ionic.phas13.currents import I_Na
        V = torch.tensor(-20.0, dtype=torch.float64)
        Nai = torch.tensor(10.925, dtype=torch.float64)
        # At V=-20, m_inf~0.95, assuming m~0.9, h~0.1, j~0.1
        m = torch.tensor(0.9, dtype=torch.float64)
        h = torch.tensor(0.1, dtype=torch.float64)
        j = torch.tensor(0.1, dtype=torch.float64)
        INa = I_Na(V, m, h, j, Nai)
        assert INa.item() < -1.0  # Inward current (negative)

    def test_ICaL_no_nan_at_v0(self):
        """P3-V2: I_CaL no NaN at V=0."""
        from cardiac_sim.ionic.phas13.currents import I_CaL
        V = torch.tensor(0.0, dtype=torch.float64)
        d = torch.tensor(0.5, dtype=torch.float64)
        f1 = torch.tensor(0.5, dtype=torch.float64)
        f2 = torch.tensor(0.5, dtype=torch.float64)
        fCa = torch.tensor(0.5, dtype=torch.float64)
        Cai = torch.tensor(1e-5, dtype=torch.float64)
        result = I_CaL(V, d, f1, f2, fCa, Cai)
        assert torch.isfinite(result)

    def test_If_direction(self):
        """P3-V3: I_f inward at V=-80, outward at V=0."""
        from cardiac_sim.ionic.phas13.currents import I_f
        Xf = torch.tensor(0.5, dtype=torch.float64)
        # At V=-80: V - E_f = -80 - (-17) = -63, so inward
        If_neg = I_f(torch.tensor(-80.0, dtype=torch.float64), Xf)
        assert If_neg.item() < 0
        # At V=0: V - E_f = 0 - (-17) = 17, so outward
        If_pos = I_f(torch.tensor(0.0, dtype=torch.float64), Xf)
        assert If_pos.item() > 0

    def test_IK1_rectification(self):
        """P3-V4: IK1 rectification — larger at negative V."""
        from cardiac_sim.ionic.phas13.currents import I_K1
        # IK1 should be larger (more negative) at hyperpolarized potentials
        IK1_neg = I_K1(torch.tensor(-100.0, dtype=torch.float64))
        IK1_pos = I_K1(torch.tensor(0.0, dtype=torch.float64))
        # Inward rectifier: large inward at negative, small outward at positive
        assert IK1_neg.item() < IK1_pos.item()

    def test_sum_at_rest(self, rest_state):
        """P3-V5: Sum of all currents at initial state ~ 0."""
        from cardiac_sim.ionic.phas13.currents import (
            I_Na, I_CaL, I_Kr, I_Ks, I_K1, I_to, I_f,
            I_NaCa, I_NaK, I_pCa, I_bNa, I_bCa
        )
        V, s = rest_state
        INa = I_Na(V, s['m'], s['h'], s['j'], s['Nai'])
        ICaL = I_CaL(V, s['d'], s['f1'], s['f2'], s['fCa'], s['Cai'])
        IKr = I_Kr(V, s['Xr1'], s['Xr2'])
        IKs = I_Ks(V, s['Xs'], s['Nai'], s['Cai'])
        IK1 = I_K1(V)
        Ito = I_to(V, s['q'], s['r_gate'])
        If = I_f(V, s['Xf'])
        INaCa = I_NaCa(V, s['Nai'], s['Cai'])
        INaK = I_NaK(V, s['Nai'])
        IpCa = I_pCa(s['Cai'])
        IbNa = I_bNa(V, s['Nai'])
        IbCa = I_bCa(V, s['Cai'])

        I_total = (INa + ICaL + IKr + IKs + IK1 + Ito + If +
                   INaCa + INaK + IpCa + IbNa + IbCa)

        # Not exactly 0 (spontaneous beating), but should be small
        assert abs(I_total.item()) < 5.0, f"I_total = {I_total.item()}"

    def test_batched_currents(self):
        """P3-V6: All currents work on batched tensors."""
        from cardiac_sim.ionic.phas13.currents import (
            I_Na, I_CaL, I_Kr, I_Ks, I_K1, I_to, I_f,
            I_NaCa, I_NaK, I_pCa, I_bNa, I_bCa
        )
        N = 100
        V = torch.randn(N, dtype=torch.float64) * 30 - 50
        gates = torch.rand(N, dtype=torch.float64)
        Nai = torch.full((N,), 10.0, dtype=torch.float64)
        Cai = torch.full((N,), 1e-5, dtype=torch.float64)

        assert I_Na(V, gates, gates, gates, Nai).shape == (N,)
        assert I_CaL(V, gates, gates, gates, gates, Cai).shape == (N,)
        assert I_Kr(V, gates, gates).shape == (N,)
        assert I_Ks(V, gates, Nai, Cai).shape == (N,)
        assert I_K1(V).shape == (N,)
        assert I_to(V, gates, gates).shape == (N,)
        assert I_f(V, gates).shape == (N,)
        assert I_NaCa(V, Nai, Cai).shape == (N,)
        assert I_NaK(V, Nai).shape == (N,)
        assert I_pCa(Cai).shape == (N,)
        assert I_bNa(V, Nai).shape == (N,)
        assert I_bCa(V, Cai).shape == (N,)

    def test_If_reversal(self):
        """P3-V7: I_f reversal at -17 mV."""
        from cardiac_sim.ionic.phas13.currents import I_f
        Xf = torch.tensor(0.5, dtype=torch.float64)
        V_rev = torch.tensor(-17.0, dtype=torch.float64)
        assert abs(I_f(V_rev, Xf).item()) < 1e-10


# ============================================================================
# Phase 4: Calcium Handling
# ============================================================================

class TestPHAS13Calcium:
    """P4: Calcium handling functions."""

    def test_iup_increases_with_cai(self):
        """P4-V1: i_up increases with Cai."""
        from cardiac_sim.ionic.phas13.calcium import i_up
        Cai_low = torch.tensor(1e-5, dtype=torch.float64)
        Cai_high = torch.tensor(1e-3, dtype=torch.float64)
        assert i_up(Cai_high) > i_up(Cai_low)

    def test_irel_triggered_by_d(self):
        """P4-V2: i_rel triggered by d > 0."""
        from cardiac_sim.ionic.phas13.calcium import i_rel
        CaSR = torch.tensor(0.3, dtype=torch.float64)
        g_rel = torch.tensor(1.0, dtype=torch.float64)
        # d=0: no release
        rel_d0 = i_rel(CaSR, torch.tensor(0.0, dtype=torch.float64), g_rel)
        assert rel_d0.item() == pytest.approx(0.0, abs=1e-12)
        # d=0.5: release
        rel_d05 = i_rel(CaSR, torch.tensor(0.5, dtype=torch.float64), g_rel)
        assert rel_d05.item() > 0

    def test_grel_frozen_at_depolarized(self):
        """P4-V3: g_rel frozen at V > -60 when recovering."""
        from cardiac_sim.ionic.phas13.calcium import update_g_rel
        # V > -60, g_inf > g (recovery condition) -> should freeze
        V = torch.tensor(0.0, dtype=torch.float64)
        g_rel = torch.tensor(0.5, dtype=torch.float64)
        Cai = torch.tensor(1e-5, dtype=torch.float64)  # g_inf ~ 1.0 > 0.5
        g_new = update_g_rel(V, g_rel, Cai, dt=1.0)
        assert g_new.item() == pytest.approx(0.5, abs=1e-10)  # No change

        # V < -60: should update
        V_neg = torch.tensor(-70.0, dtype=torch.float64)
        g_new2 = update_g_rel(V_neg, g_rel, Cai, dt=1.0)
        assert g_new2.item() != pytest.approx(0.5, abs=1e-3)  # Changed

    def test_buffering_in_01(self):
        """P4-V4: Buffering factors in (0,1)."""
        from cardiac_sim.ionic.phas13.calcium import buffering_factor_cyt, buffering_factor_sr
        Cai = torch.tensor(1e-5, dtype=torch.float64)
        CaSR = torch.tensor(0.3, dtype=torch.float64)
        bc = buffering_factor_cyt(Cai)
        bs = buffering_factor_sr(CaSR)
        assert 0.0 < bc.item() < 1.0
        assert 0.0 < bs.item() < 1.0

    def test_steady_state_cai(self):
        """P4-V5: Cai near initial value after small step at rest."""
        from cardiac_sim.ionic.phas13.calcium import update_concentrations
        from cardiac_sim.ionic.phas13.parameters import get_initial_state, StateIndex, V_REST
        state = get_initial_state(device=torch.device('cpu'))
        V = torch.tensor(V_REST, dtype=torch.float64)
        dt = 0.01
        zero = torch.tensor(0.0, dtype=torch.float64)

        # Approximate zero currents at rest
        Nai_new, Cai_new, CaSR_new, g_new = update_concentrations(
            V, state[StateIndex.Nai], state[StateIndex.Cai],
            state[StateIndex.CaSR], state[StateIndex.g_rel],
            state[StateIndex.d],
            zero, zero, zero, zero, zero, zero, zero,
            dt
        )
        # Should stay near initial values with zero currents
        assert Cai_new.item() == pytest.approx(state[StateIndex.Cai].item(), rel=0.1)


# ============================================================================
# Phase 5: Model
# ============================================================================

class TestPHAS13Model:
    """P5: PHAS13Model(IonicModel) integration."""

    def test_instantiation(self):
        """P5-V1: Instantiation, n_states == 17."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        model = PHAS13Model(device='cpu')
        assert model.n_states == 17
        assert model.name == "PHAS13"

    def test_step_runs(self):
        """P5-V2: step() runs 1 timestep without error."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        model = PHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)
        V_new, states_new = model.step(V, states, dt=0.01)
        assert torch.isfinite(V_new)
        assert torch.isfinite(states_new).all()

    def test_compute_Iion_at_rest(self):
        """P5-V3: compute_Iion() at rest small (not exactly 0)."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        model = PHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)
        Iion = model.compute_Iion(V, states)
        assert abs(Iion.item()) < 5.0

    def test_gate_steady_states_shape(self):
        """P5-V4: gate_steady_states shape correct."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        model = PHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)
        ss = model.compute_gate_steady_states(V, states)
        assert ss.shape[-1] == len(model.gate_indices)

    def test_gate_time_constants_positive(self):
        """P5-V5: gate_time_constants all positive."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        model = PHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)
        tau = model.compute_gate_time_constants(V, states)
        assert (tau > 0).all()

    def test_batch_and_single(self):
        """P5-V6: Single-cell and batch (100,) both work."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        model = PHAS13Model(device='cpu')

        # Single cell
        V1 = torch.tensor(model.V_rest, dtype=torch.float64)
        s1 = model.get_initial_state(n_cells=1)
        V1_new, s1_new = model.step(V1, s1, dt=0.01)
        assert V1_new.dim() == 0
        assert s1_new.shape == (17,)

        # Batch
        N = 100
        V_batch = torch.full((N,), model.V_rest, dtype=torch.float64)
        s_batch = model.get_initial_state(n_cells=N)
        V_b_new, s_b_new = model.step(V_batch, s_batch, dt=0.01)
        assert V_b_new.shape == (N,)
        assert s_b_new.shape == (N, 17)

    def test_stimulus_produces_spike(self):
        """P5-V7: Stimulus produces depolarization."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        model = PHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)

        # Apply strong stimulus for several ms
        dt = 0.01
        for _ in range(500):  # 5 ms
            I_stim = torch.tensor(-5.0, dtype=torch.float64)
            V, states = model.step(V, states, dt, I_stim)

        # Continue without stimulus
        V_peak = V.item()
        for _ in range(2000):  # 20 ms
            V, states = model.step(V, states, dt)
            V_peak = max(V_peak, V.item())

        # Should have depolarized significantly
        assert V_peak > -20.0, f"V_peak = {V_peak}"


# ============================================================================
# Phase 6: Validation
# ============================================================================

class TestPHAS13Validation:
    """P6: Validation against published behavior and Myokit reference."""

    @pytest.fixture(scope='class')
    def long_run(self):
        """Run model for 5s (no stimulus) and cache result."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        model = PHAS13Model(device='cpu')
        t, V = model.run(t_end=5000.0, dt=0.01, stim_times=[],
                         save_interval=1.0)
        return t.numpy(), V.numpy()

    def _find_peaks(self, V, t):
        peaks = []
        for i in range(1, len(V) - 1):
            if V[i] > V[i-1] and V[i] > V[i+1] and V[i] > 0:
                peaks.append((t[i], V[i]))
        return peaks

    @pytest.mark.slow
    def test_spontaneous_beating(self, long_run):
        """P6-V1: Model beats spontaneously without stimulus."""
        t, V = long_run
        peaks = self._find_peaks(V, t)
        assert len(peaks) >= 2, f"Only {len(peaks)} peaks found"

    @pytest.mark.slow
    def test_cycle_length(self, long_run):
        """P6-V2: Spontaneous CL ~ 1.0-2.0 s."""
        t, V = long_run
        peaks = self._find_peaks(V, t)
        assert len(peaks) >= 2
        cls = [peaks[j+1][0] - peaks[j][0] for j in range(len(peaks)-1)]
        mean_cl = sum(cls) / len(cls)
        assert 1000 < mean_cl < 2000, f"CL = {mean_cl:.0f} ms"

    @pytest.mark.slow
    def test_v_peak(self, long_run):
        """P6-V3: V_peak > +20 mV."""
        t, V = long_run
        assert V.max() > 20.0, f"V_peak = {V.max():.1f} mV"

    @pytest.mark.slow
    def test_apd90(self, long_run):
        """P6-V4: APD90 = 300-600 ms."""
        t, V = long_run
        peaks = self._find_peaks(V, t)
        assert len(peaks) >= 1
        t_peak, v_peak = peaks[-1]
        v_min = V.min()
        v90 = v_peak - 0.9 * (v_peak - v_min)
        idx_peak = int(t_peak)
        apd90 = None
        for k in range(idx_peak + 1, min(idx_peak + 2000, len(V))):
            if V[k] < v90:
                apd90 = t[k] - t_peak
                break
        assert apd90 is not None, "Could not find APD90"
        assert 300 < apd90 < 600, f"APD90 = {apd90:.0f} ms"

    @pytest.mark.slow
    def test_dvdt_max(self, long_run):
        """P6-V5: dV/dt_max > 10 V/s."""
        import numpy as np
        t, V = long_run
        dVdt = np.diff(V) / np.diff(t)  # mV/ms = V/s
        assert dVdt.max() > 10.0, f"dVdt_max = {dVdt.max():.1f} V/s"

    def test_current_match_myokit(self):
        """P6-V6: All 12 currents match Myokit at t=0 (ratio=1.0)."""
        from cardiac_sim.ionic.phas13.model import PHAS13Model
        from cardiac_sim.ionic.phas13.parameters import V_REST, StateIndex, get_initial_state
        from cardiac_sim.ionic.phas13.currents import (
            I_Na, I_CaL, I_Kr, I_Ks, I_K1, I_to, I_f,
            I_NaCa, I_NaK, I_pCa, I_bNa, I_bCa
        )
        model = PHAS13Model(device='cpu')
        V = torch.tensor(V_REST, dtype=torch.float64)
        states = get_initial_state(device=torch.device('cpu'))
        p = model.params

        # Expected values from Myokit reference (computed above)
        expected = {
            'INa':   -1.15676591e-01,
            'ICaL':  -4.96351584e-03,
            'IKr':   +1.45396248e-03,
            'IKs':   -4.29850668e-06,
            'IK1':   +4.05511734e-01,
            'Ito':   +2.08185147e-03,
            'If':    -1.73654868e-01,
            'INaCa': -4.62877396e-02,
            'INaK':  +1.92119172e-01,
            'IpCa':  +1.43934603e-02,
            'IbNa':  -1.30041674e-01,
            'IbCa':  -1.57958951e-01,
        }

        computed = {
            'INa':   I_Na(V, states[3], states[4], states[5], states[0],
                          p.g_Na, p.Nao).item(),
            'ICaL':  I_CaL(V, states[6], states[7], states[8], states[9],
                           states[1], p.g_CaL, p.Cao).item(),
            'IKr':   I_Kr(V, states[10], states[11], p.g_Kr, p.Ki, p.Ko).item(),
            'IKs':   I_Ks(V, states[12], states[0], states[1],
                          p.g_Ks, p.Ki, p.Ko, p.Nao, p.PkNa).item(),
            'IK1':   I_K1(V, p.g_K1, p.Ki, p.Ko).item(),
            'Ito':   I_to(V, states[13], states[14], p.g_to, p.Ki, p.Ko).item(),
            'If':    I_f(V, states[15], p.g_f, p.E_f).item(),
            'INaCa': I_NaCa(V, states[0], states[1], p.kNaCa, p.Cao, p.Nao,
                            p.KmNai, p.KmCa, p.Ksat, p.alpha_ncx,
                            p.gamma_ncx).item(),
            'INaK':  I_NaK(V, states[0], p.PNaK, p.Ki, p.Ko,
                           p.Km_K, p.Km_Na).item(),
            'IpCa':  I_pCa(states[1], p.g_pCa, p.KpCa).item(),
            'IbNa':  I_bNa(V, states[0], p.g_bNa, p.Nao).item(),
            'IbCa':  I_bCa(V, states[1], p.g_bCa, p.Cao).item(),
        }

        for key in expected:
            assert computed[key] == pytest.approx(expected[key], rel=1e-6), \
                f"{key}: {computed[key]} != {expected[key]}"

    @pytest.mark.slow
    def test_gpu_matches_cpu(self):
        """P6-V8: GPU results match CPU (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from cardiac_sim.ionic.phas13.model import PHAS13Model

        # Run 100 steps on CPU
        model_cpu = PHAS13Model(device='cpu')
        V_cpu = torch.tensor(model_cpu.V_rest, dtype=torch.float64)
        s_cpu = model_cpu.get_initial_state()
        for _ in range(100):
            V_cpu, s_cpu = model_cpu.step(V_cpu, s_cpu, 0.01)

        # Run 100 steps on GPU
        model_gpu = PHAS13Model(device='cuda')
        V_gpu = torch.tensor(model_gpu.V_rest, dtype=torch.float64,
                             device='cuda')
        s_gpu = model_gpu.get_initial_state()
        for _ in range(100):
            V_gpu, s_gpu = model_gpu.step(V_gpu, s_gpu, 0.01)

        # Compare
        assert V_cpu.item() == pytest.approx(V_gpu.cpu().item(), rel=1e-10)
        assert torch.allclose(s_cpu, s_gpu.cpu(), rtol=1e-10)


# ============================================================================
# Backward Compatibility
# ============================================================================

class TestBackwardCompat:
    """Verify old import paths still work."""

    def test_paci_import(self):
        """Old import from ionic.paci still works."""
        from cardiac_sim.ionic.paci import PaciModel, PaciParameters, StateIndex
        model = PaciModel(device='cpu')
        assert model.name == "PHAS13"
        assert model.n_states == 17
        p = PaciParameters()
        assert p.g_Na == pytest.approx(3.6712302)

    def test_parent_import(self):
        """Old import from ionic still works."""
        from cardiac_sim.ionic import PaciModel
        model = PaciModel(device='cpu')
        assert model.name == "PHAS13"
