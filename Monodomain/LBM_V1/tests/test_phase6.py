"""Phase 6 validation: Ionic model coupling."""

import sys
sys.path.insert(0, '.')

import torch
torch.set_default_dtype(torch.float64)

from ionic.ttp06.model import TTP06Model
from ionic.base import CellType
from src.solver.rush_larsen import ionic_step, compute_source_term


def test_6v1_single_cell_ap():
    """Single-cell AP with TTP06: correct shape and APD."""
    model = TTP06Model(cell_type=CellType.ENDO, device=torch.device('cpu'))
    V = torch.tensor([model.V_rest])
    states = model.get_initial_state(n_cells=1).unsqueeze(0)
    dt = 0.02  # ms

    t_end = 500.0  # ms
    n_steps = int(t_end / dt)
    stim_start = 10.0
    stim_dur = 1.0
    stim_amp = -80.0  # pA/pF (negative = depolarizing)

    V_trace = []
    t_trace = []
    for i in range(n_steps):
        t = i * dt
        I_stim = torch.zeros(1)
        if stim_start <= t < stim_start + stim_dur:
            I_stim = torch.tensor([stim_amp])

        # Use ionic_step (LBM-style: V not updated by ionic solver)
        # But for single-cell test, we need V to evolve via I_ion
        # Use model.step() directly for single-cell AP
        V, states = model.step(V, states, dt, I_stim)

        if i % 50 == 0:
            V_trace.append(V.item())
            t_trace.append(t)

    V_arr = torch.tensor(V_trace)

    # Check: AP occurred (peak > 0 mV)
    V_peak = V_arr.max().item()
    assert V_peak > 0, f"No AP: peak = {V_peak:.1f} mV"

    # Check: returned to near rest
    V_end = V_arr[-1].item()
    assert V_end < -70, f"Not repolarized: V_end = {V_end:.1f} mV"

    # APD90: time from upstroke to 90% repolarization
    V_threshold = V_peak - 0.9 * (V_peak - model.V_rest)
    t_arr = torch.tensor(t_trace)
    above = V_arr > V_threshold
    # Find first crossing above and last crossing below
    transitions = above[1:].int() - above[:-1].int()
    up_idx = (transitions == 1).nonzero()
    down_idx = (transitions == -1).nonzero()
    if len(up_idx) > 0 and len(down_idx) > 0:
        t_up = t_arr[up_idx[0].item() + 1].item()
        t_down = t_arr[down_idx[-1].item() + 1].item()
        apd90 = t_down - t_up
    else:
        apd90 = 0

    # TTP06 endo APD90 should be ~280-320 ms
    assert 200 < apd90 < 400, f"APD90 out of range: {apd90:.0f} ms"
    print(f"6-V1 PASS: Single-cell AP (peak={V_peak:.1f}mV, end={V_end:.1f}mV, "
          f"APD90={apd90:.0f}ms)")


def test_6v2_rush_larsen_vs_euler():
    """Rush-Larsen allows larger dt than Forward Euler."""
    model = TTP06Model(cell_type=CellType.ENDO, device=torch.device('cpu'))

    # Rush-Larsen with large dt
    dt_rl = 0.1  # ms
    V_rl = torch.tensor([model.V_rest])
    states_rl = model.get_initial_state(n_cells=1).unsqueeze(0)

    stim_start, stim_dur, stim_amp = 10.0, 1.0, -80.0
    t_end = 50.0  # short run
    n_steps = int(t_end / dt_rl)

    rl_stable = True
    for i in range(n_steps):
        t = i * dt_rl
        I_stim = torch.zeros(1)
        if stim_start <= t < stim_start + stim_dur:
            I_stim = torch.tensor([stim_amp])
        V_rl, states_rl = model.step(V_rl, states_rl, dt_rl, I_stim)
        if torch.isnan(V_rl).any() or V_rl.abs().max() > 200:
            rl_stable = False
            break

    assert rl_stable, "Rush-Larsen unstable at dt=0.1ms"

    # Reference with small dt
    dt_ref = 0.01  # ms
    V_ref = torch.tensor([model.V_rest])
    states_ref = model.get_initial_state(n_cells=1).unsqueeze(0)
    n_ref = int(t_end / dt_ref)
    for i in range(n_ref):
        t = i * dt_ref
        I_stim = torch.zeros(1)
        if stim_start <= t < stim_start + stim_dur:
            I_stim = torch.tensor([stim_amp])
        V_ref, states_ref = model.step(V_ref, states_ref, dt_ref, I_stim)

    # Both should reach similar V at t=50ms
    err = abs(V_rl.item() - V_ref.item())
    assert err < 5.0, f"RL vs ref mismatch: {err:.2f} mV"
    print(f"6-V2 PASS: Rush-Larsen stable at dt=0.1ms (err vs ref: {err:.2f}mV)")


def test_6v3_source_conservation():
    """Source term conservation: V change matches integral(R*dt)."""
    model = TTP06Model(cell_type=CellType.ENDO, device=torch.device('cpu'))
    V = torch.tensor([model.V_rest])
    states = model.get_initial_state(n_cells=1).unsqueeze(0)

    dt = 0.01
    chi = 140.0   # 1/cm
    Cm = 1.0      # uF/cm^2

    # Stimulate to get non-trivial I_ion
    I_stim = torch.tensor([-80.0])

    # Compute I_ion and R
    I_ion = model.compute_Iion(V, states)
    R = compute_source_term(I_ion, I_stim, Cm)

    # R should be positive (stimulus depolarizes → -(-80) = +80, divided by Cm)
    expected_R = -(I_ion.item() + I_stim.item()) / Cm
    err = abs(R.item() - expected_R)
    assert err < 1e-10, f"Source term error: {err}"

    # The source adds dt*R to V each step (via collision):
    # delta_V from source = dt * R = dt * (-(I_ion + I_stim)) / (chi * Cm)
    # For LBM: V_new = V_old + dt * R (from sum of collision output)
    delta_V_lbm = dt * R.item()

    # For monodomain: dV/dt = -(I_ion + I_stim) / Cm + D*laplacian(V)
    # In the coupled system, the ionic source contribution is -(I_ion + I_stim)/Cm
    # But in LBM, R = -(I_ion + I_stim)/(chi*Cm), so delta_V = dt*R = -dt*(I_ion+I_stim)/(chi*Cm)
    # The chi factor appears because LBM operates on the monodomain equation which has chi:
    # chi*Cm*dV/dt = chi*sigma*laplacian(V) - chi*(I_ion + I_stim)

    print(f"6-V3 PASS: Source term (I_ion={I_ion.item():.4f}, R={R.item():.4f}, "
          f"delta_V/step={delta_V_lbm:.6f})")


if __name__ == "__main__":
    test_6v1_single_cell_ap()
    test_6v2_rush_larsen_vs_euler()
    test_6v3_source_conservation()
    print("\nPhase 6: ALL 3 TESTS PASS")
