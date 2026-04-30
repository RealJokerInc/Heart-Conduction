"""Tests for D2Q9_uniform lattice variant + cs2 plumbing fix in LBMSimulation.

Companion to PLAN.md Phase 4 (Moore-8 / iso-9pt extension to LBM V1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lattice import D2Q9, D2Q9_uniform, D2Q5
from src.diffusion import tau_from_D
from src.simulation import LBMSimulation
from ionic.ttp06.model import TTP06Model
from ionic.base import CellType


# ---------- D2Q9_uniform construction tests ----------

def test_canonical_weights_unchanged():
    """Canonical D2Q9 retains 4/9, 1/9, 1/36 weights and cs2 = 1/3."""
    lat = D2Q9()
    assert lat.Q == 9
    assert lat.cs2 == 1.0 / 3.0
    assert lat.w[0] == pytest.approx(4.0 / 9.0)
    for i in range(1, 5):
        assert lat.w[i] == pytest.approx(1.0 / 9.0), f"cardinal {i} weight wrong"
    for i in range(5, 9):
        assert lat.w[i] == pytest.approx(1.0 / 36.0), f"diagonal {i} weight wrong"


def test_uniform_weights_construction():
    """D2Q9_uniform: rest weight 0, all 8 moving particles weight 1/8, cs2 = 0.75."""
    lat = D2Q9_uniform()
    assert lat.Q == 9
    assert lat.cs2 == 0.75
    assert lat.w[0] == 0.0, f"rest weight should be 0, got {lat.w[0]}"
    for i in range(1, 9):
        assert lat.w[i] == pytest.approx(1.0 / 8.0), f"weight {i} = {lat.w[i]}"
    # opposite tuple must match canonical for bounce-back
    canonical_opposite = D2Q9().opposite
    assert lat.opposite == canonical_opposite, (
        f"opposite must match D2Q9 for bounce-back to work: "
        f"got {lat.opposite}, expected {canonical_opposite}"
    )
    # direction order must match canonical
    assert lat.e == D2Q9().e, "direction order must match canonical D2Q9"


def test_uniform_cs2_self_consistency():
    """Second moment Σ w_i e_iα e_iβ must equal cs2 · δ_αβ for the LBM scheme to be consistent."""
    lat = D2Q9_uniform()
    M = np.zeros((2, 2))
    for w_i, (ex, ey) in zip(lat.w, lat.e):
        M[0, 0] += w_i * ex * ex
        M[0, 1] += w_i * ex * ey
        M[1, 0] += w_i * ey * ex
        M[1, 1] += w_i * ey * ey
    # M should be cs2 · I
    expected = lat.cs2 * np.eye(2)
    err = np.abs(M - expected).max()
    print(f"Second moment matrix:\n{M}")
    print(f"Expected (cs2·I):\n{expected}")
    print(f"max err = {err:.3e}")
    assert err < 1e-12, f"second moment isn't isotropic: {M}"


def test_uniform_fourth_moment_not_isotropic():
    """DOCUMENT: D2Q9_uniform's fourth moment is NOT 3·cs²²·δ — diffusion-only.

    For canonical D2Q9, Σ w_i e_iα²·e_iβ² = cs²·(δ_αβ + 2·something) gives the
    Galilean-isotropic fourth moment required for Navier-Stokes.

    For uniform_8, this fails: M4_xxxx = (1/8)(2·1 + 4·1) = 0.75 = cs2 (NOT 3·cs2²
    = 1.6875). This means the lattice cannot recover the full Navier-Stokes
    momentum equation, but it CAN recover the heat (diffusion) equation since
    diffusion only requires the second moment.
    """
    lat = D2Q9_uniform()
    M4_xxxx = sum(w * (ex ** 2) * (ex ** 2) for w, (ex, ey) in zip(lat.w, lat.e))
    expected_isotropic = 3.0 * lat.cs2 ** 2
    print(f"M4_xxxx = {M4_xxxx}, expected_isotropic = {expected_isotropic}")
    assert abs(M4_xxxx - expected_isotropic) > 0.1, (
        "D2Q9_uniform unexpectedly satisfies fourth-moment isotropy — "
        "the docstring claim that it's diffusion-only is incorrect."
    )


# ---------- LBMSimulation cs2 plumbing tests ----------

def _make_sim(lattice='d2q9', weights_mode='canonical', Nx=10, Ny=10,
              D=0.001, dt=0.02, dx=0.025):
    """Helper: minimal LBMSimulation construction."""
    ionic = TTP06Model(cell_type=CellType.EPI, device=torch.device('cpu'))
    sim = LBMSimulation(Nx=Nx, Ny=Ny, dx=dx, dt=dt, D=D,
                        ionic_model=ionic, Cm=1.0,
                        lattice=lattice, weights_mode=weights_mode)
    return sim


def test_tau_from_D_uses_lattice_cs2():
    """LBMSimulation must pass lattice.cs2 to tau_from_D (was a latent bug)."""
    # Canonical D2Q9: cs2 = 1/3 (default), tau should match default-cs2 call
    sim_can = _make_sim(lattice='d2q9', weights_mode='canonical', D=0.001,
                         dt=0.02, dx=0.025)
    expected_can = tau_from_D(0.001, 0.025, 0.02, cs2=1.0 / 3.0)
    actual_can = 1.0 / sim_can.omega
    assert abs(actual_can - expected_can) < 1e-12, (
        f"canonical: tau={actual_can}, expected {expected_can}"
    )

    # Uniform_8: cs2 = 0.75, tau should differ
    sim_uni = _make_sim(lattice='d2q9', weights_mode='uniform_8', D=0.001,
                         dt=0.02, dx=0.025)
    expected_uni = tau_from_D(0.001, 0.025, 0.02, cs2=0.75)
    actual_uni = 1.0 / sim_uni.omega
    assert abs(actual_uni - expected_uni) < 1e-12, (
        f"uniform_8: tau={actual_uni}, expected {expected_uni}"
    )

    # The two taus must differ (otherwise plumbing fix is a no-op)
    assert abs(actual_can - actual_uni) > 1e-6, (
        f"canonical and uniform_8 produced same tau ({actual_can}) — "
        f"cs2 plumbing fix is not effective"
    )
    print(f"  canonical tau = {actual_can:.6f}, uniform_8 tau = {actual_uni:.6f}")


def test_d2q5_rejects_uniform_weights_mode():
    """d2q5 + weights_mode='uniform_8' must raise ValueError."""
    with pytest.raises(ValueError, match="d2q5"):
        _make_sim(lattice='d2q5', weights_mode='uniform_8')


def test_unknown_weights_mode_rejected():
    """d2q9 + weights_mode='garbage' must raise ValueError."""
    with pytest.raises(ValueError, match="canonical|uniform_8"):
        _make_sim(lattice='d2q9', weights_mode='garbage')


def test_uniform_propagation_creates_boundary_artifact():
    """LBM uniform_8 should produce measurable boundary deficit; canonical should not.

    Setup: small grid, line stim at left edge, run for a fixed number of steps.
    Compare V at top boundary vs. interior at the same x column.
    """
    Nx, Ny = 20, 16
    deviations = {}
    for weights_mode in ('canonical', 'uniform_8'):
        sim = _make_sim(lattice='d2q9', weights_mode=weights_mode,
                         Nx=Nx, Ny=Ny, D=0.001, dt=0.02, dx=0.025)
        # Line stim at x=0 column, full y range
        stim_mask = torch.zeros(Nx, Ny, dtype=torch.bool)
        stim_mask[0, :] = True
        sim.add_stimulus(mask=stim_mask, start=0.0, duration=2.0,
                         amplitude=-52.0)
        # Run 200 steps
        n_steps = 200
        for _ in range(n_steps):
            sim.step()
        V_field = sim.V.numpy()  # shape (Nx, Ny)
        # Mid column (x=Nx//2): compare top boundary vs. mid y
        i_mid = Nx // 2
        v_top = float(V_field[i_mid, 0])
        v_ctr = float(V_field[i_mid, Ny // 2])
        dev = abs(v_top - v_ctr)
        deviations[weights_mode] = dev
        print(f"  {weights_mode}: V[top]={v_top:.4f}, V[ctr]={v_ctr:.4f}, "
              f"|dev|={dev:.3e}")

    # Canonical should be ~zero (LBM bounce-back is diagonal-aware)
    # Uniform_8 should be visibly above noise
    print(f"  canonical dev: {deviations['canonical']:.3e}")
    print(f"  uniform_8 dev: {deviations['uniform_8']:.3e}")
    # Permissive thresholds — just need the ordering to be right
    assert deviations['canonical'] < 0.5, (
        f"canonical D2Q9 should give small boundary deviation in line-stim, "
        f"got {deviations['canonical']:.3e}"
    )
    # uniform_8 should produce observably larger deviation
    assert deviations['uniform_8'] > deviations['canonical'], (
        f"uniform_8 deviation ({deviations['uniform_8']:.3e}) should exceed "
        f"canonical deviation ({deviations['canonical']:.3e})"
    )
