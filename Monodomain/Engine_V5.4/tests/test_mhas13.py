"""
Tests for MHAS13 (Matured hiPSC-CM) Ionic Model

Validates that maturation modifications (TTP06 IK1 + g_f=0) produce
a quiescent cell that fires only when paced.

Reference values from Verkerk et al. (2019), Biophys J 117:2303-15,
Table 2 (TTP06 IK1 formulation at GK1_critical, 1 Hz pacing).
"""

import pytest
import torch
import numpy as np


# ============================================================================
# Model Basics
# ============================================================================

class TestMHAS13Basics:
    """Model instantiation and properties."""

    def test_instantiation(self):
        """Model instantiates with correct name and n_states."""
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        model = MHAS13Model(device='cpu')
        assert model.name == "MHAS13"
        assert model.n_states == 17

    def test_parameters(self):
        """MHAS13Parameters has maturation overrides."""
        from cardiac_sim.ionic.mhas13 import MHAS13Parameters
        p = MHAS13Parameters()
        assert p.g_f == 0.0                    # If suppressed
        assert p.GK1_ttp06 == pytest.approx(3.170)  # TTP06 IK1 conductance
        # Inherits PHAS13 params
        assert p.g_Na == pytest.approx(3.6712302)
        assert p.Ki == 150.0

    def test_v_rest(self):
        """V_rest near Verkerk Table 2 value."""
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        model = MHAS13Model(device='cpu')
        assert model.V_rest == pytest.approx(-83.7, abs=1.0)

    def test_step_runs(self):
        """step() runs without error."""
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        model = MHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)
        V_new, states_new = model.step(V, states, dt=0.01)
        assert torch.isfinite(V_new)
        assert torch.isfinite(states_new).all()

    def test_batch_mode(self):
        """Batch (N,) mode works."""
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        model = MHAS13Model(device='cpu')
        N = 50
        V = torch.full((N,), model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=N)
        V_new, states_new = model.step(V, states, dt=0.01)
        assert V_new.shape == (N,)
        assert states_new.shape == (N, 17)

    def test_parent_import(self):
        """Can import from ionic package."""
        from cardiac_sim.ionic import MHAS13Model
        model = MHAS13Model(device='cpu')
        assert model.name == "MHAS13"


# ============================================================================
# Quiescence
# ============================================================================

class TestMHAS13Quiescence:
    """Verify the model does NOT beat spontaneously."""

    def test_no_spontaneous_ap_5s(self):
        """V1: No spontaneous AP in 5s unstimulated run."""
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        model = MHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)

        dt = 0.05
        V_max_seen = model.V_rest
        n_steps = int(5000.0 / dt)  # 5 seconds

        for _ in range(n_steps):
            V, states = model.step(V, states, dt)
            V_max_seen = max(V_max_seen, V.item())

        # No depolarization above -40 mV (well below AP threshold)
        assert V_max_seen < -40.0, f"V reached {V_max_seen:.1f} mV — spontaneous AP!"

    def test_stable_resting_potential(self):
        """V2: V settles near V_rest after 2s."""
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        model = MHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)

        dt = 0.05
        for _ in range(int(2000.0 / dt)):
            V, states = model.step(V, states, dt)

        # Should be near V_rest (-83.7 mV)
        assert V.item() < -75.0, f"V = {V.item():.1f} mV (expected < -75)"
        assert V.item() > -90.0, f"V = {V.item():.1f} mV (expected > -90)"


# ============================================================================
# Paced AP
# ============================================================================

class TestMHAS13PacedAP:
    """Verify the model fires correctly when paced."""

    @pytest.fixture(scope='class')
    def paced_run(self):
        """Pace at 1 Hz for 5 beats, return last beat trace."""
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        model = MHAS13Model(device='cpu')
        V = torch.tensor(model.V_rest, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)

        dt = 0.05
        cl = 1000.0  # 1 Hz
        n_beats = 5
        stim_duration = 2.0  # ms
        # Matured IK1 requires stronger stimulus than immature PHAS13
        stim_amplitude = -40.0  # A/F (threshold ~-15 A/F)

        # Record last 2 beats
        save_start = cl * (n_beats - 2)
        t_list = []
        v_list = []
        t_current = 0.0

        for _ in range(int(cl * n_beats / dt)):
            I_stim = None
            if (t_current % cl) < stim_duration:
                I_stim = torch.tensor(stim_amplitude, dtype=torch.float64)
            V, states = model.step(V, states, dt, I_stim)
            t_current += dt
            if t_current >= save_start:
                t_list.append(t_current)
                v_list.append(V.item())

        return np.array(t_list), np.array(v_list)

    def test_fires_when_paced(self, paced_run):
        """V3: Stimulus produces AP (V > 0 mV)."""
        t, V = paced_run
        assert V.max() > 0.0, f"V_peak = {V.max():.1f} mV (expected > 0)"

    def test_dvdt_max(self, paced_run):
        """V4: dV/dt_max > 50 V/s (Verkerk: 108 V/s)."""
        t, V = paced_run
        dVdt = np.diff(V) / np.diff(t)  # mV/ms = V/s
        assert dVdt.max() > 50.0, f"dVdt_max = {dVdt.max():.1f} V/s"

    def test_apd90_range(self, paced_run):
        """V5: APD90 in range 300-800 ms at 1 Hz (Verkerk: 546 ms)."""
        t, V = paced_run
        # Find last peak
        peaks = []
        for i in range(1, len(V) - 1):
            if V[i] > V[i-1] and V[i] > V[i+1] and V[i] > 0:
                peaks.append(i)

        assert len(peaks) >= 1, "No AP peaks found"
        peak_idx = peaks[-1]
        v_peak = V[peak_idx]
        v_min = V.min()
        v90 = v_peak - 0.9 * (v_peak - v_min)

        apd90 = None
        for k in range(peak_idx + 1, len(V)):
            if V[k] < v90:
                apd90 = t[k] - t[peak_idx]
                break

        assert apd90 is not None, "Could not measure APD90"
        assert 300 < apd90 < 800, f"APD90 = {apd90:.0f} ms"

    def test_v_peak(self, paced_run):
        """V_peak > 20 mV (Verkerk: ~30 mV)."""
        _, V = paced_run
        assert V.max() > 20.0, f"V_peak = {V.max():.1f} mV"


# ============================================================================
# Current Conservation
# ============================================================================

class TestMHAS13Currents:
    """Verify non-IK1 currents match PHAS13 exactly."""

    def test_shared_currents_match_phas13(self):
        """V6: All 10 shared currents identical to PHAS13 at same state."""
        from cardiac_sim.ionic.mhas13 import MHAS13Model
        from cardiac_sim.ionic.phas13 import PHAS13Model
        from cardiac_sim.ionic.phas13.parameters import StateIndex, V_REST, get_initial_state
        from cardiac_sim.ionic.phas13.currents import (
            I_Na, I_CaL, I_Kr, I_Ks, I_to, I_f,
            I_NaCa, I_NaK, I_pCa, I_bNa, I_bCa
        )

        V = torch.tensor(V_REST, dtype=torch.float64)
        states = get_initial_state(device=torch.device('cpu'))

        phas13 = PHAS13Model(device='cpu')
        mhas13 = MHAS13Model(device='cpu')
        pp = phas13.params
        mp = mhas13.params

        # These 10 currents should be IDENTICAL (same formulation, same params)
        shared = {
            'INa':   I_Na(V, states[3], states[4], states[5], states[0], pp.g_Na, pp.Nao),
            'ICaL':  I_CaL(V, states[6], states[7], states[8], states[9], states[1], pp.g_CaL, pp.Cao),
            'IKr':   I_Kr(V, states[10], states[11], pp.g_Kr, pp.Ki, pp.Ko),
            'IKs':   I_Ks(V, states[12], states[0], states[1], pp.g_Ks, pp.Ki, pp.Ko, pp.Nao, pp.PkNa),
            'Ito':   I_to(V, states[13], states[14], pp.g_to, pp.Ki, pp.Ko),
            'INaCa': I_NaCa(V, states[0], states[1], pp.kNaCa, pp.Cao, pp.Nao,
                            pp.KmNai, pp.KmCa, pp.Ksat, pp.alpha_ncx, pp.gamma_ncx),
            'INaK':  I_NaK(V, states[0], pp.PNaK, pp.Ki, pp.Ko, pp.Km_K, pp.Km_Na),
            'IpCa':  I_pCa(states[1], pp.g_pCa, pp.KpCa),
            'IbNa':  I_bNa(V, states[0], pp.g_bNa, pp.Nao),
            'IbCa':  I_bCa(V, states[1], pp.g_bCa, pp.Cao),
        }

        # MHAS13 should give identical values for these
        shared_m = {
            'INa':   I_Na(V, states[3], states[4], states[5], states[0], mp.g_Na, mp.Nao),
            'ICaL':  I_CaL(V, states[6], states[7], states[8], states[9], states[1], mp.g_CaL, mp.Cao),
            'IKr':   I_Kr(V, states[10], states[11], mp.g_Kr, mp.Ki, mp.Ko),
            'IKs':   I_Ks(V, states[12], states[0], states[1], mp.g_Ks, mp.Ki, mp.Ko, mp.Nao, mp.PkNa),
            'Ito':   I_to(V, states[13], states[14], mp.g_to, mp.Ki, mp.Ko),
            'INaCa': I_NaCa(V, states[0], states[1], mp.kNaCa, mp.Cao, mp.Nao,
                            mp.KmNai, mp.KmCa, mp.Ksat, mp.alpha_ncx, mp.gamma_ncx),
            'INaK':  I_NaK(V, states[0], mp.PNaK, mp.Ki, mp.Ko, mp.Km_K, mp.Km_Na),
            'IpCa':  I_pCa(states[1], mp.g_pCa, mp.KpCa),
            'IbNa':  I_bNa(V, states[0], mp.g_bNa, mp.Nao),
            'IbCa':  I_bCa(V, states[1], mp.g_bCa, mp.Cao),
        }

        for key in shared:
            assert shared[key].item() == pytest.approx(shared_m[key].item(), rel=1e-12), \
                f"{key}: PHAS13={shared[key].item()}, MHAS13={shared_m[key].item()}"

    def test_ik1_differs_from_phas13(self):
        """IK1 uses TTP06 formulation, NOT Paci formulation."""
        from cardiac_sim.ionic.phas13.currents import I_K1 as I_K1_paci
        from cardiac_sim.ionic.mhas13.currents import I_K1_ttp06

        V = torch.tensor(-80.0, dtype=torch.float64)
        ik1_paci = I_K1_paci(V, g_K1=0.0281492).item()
        ik1_ttp06 = I_K1_ttp06(V, GK1=3.170).item()

        # TTP06 IK1 should be larger (different formulation + higher conductance)
        assert abs(ik1_ttp06) > abs(ik1_paci) * 2, \
            f"TTP06 IK1={ik1_ttp06:.4f}, Paci IK1={ik1_paci:.4f}"

    def test_if_is_zero(self):
        """If current is zero with g_f=0."""
        from cardiac_sim.ionic.phas13.currents import I_f
        V = torch.tensor(-80.0, dtype=torch.float64)
        Xf = torch.tensor(0.5, dtype=torch.float64)
        i_f = I_f(V, Xf, g_f=0.0)
        assert i_f.item() == 0.0
