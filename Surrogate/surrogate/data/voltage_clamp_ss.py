"""Voltage-clamp steady-state integrator for TTP06 (Session 27, Step 2.1).

Clamps V and selected concentrations at rest, integrates the 14-dim
scaffold-observable subset (12 HH gates + RR + CaSR) to steady state using
V5.4's TTP06 primitives (no RHS reimplementation). Produces
``z_ss_grid.pt`` as a diagnostic artifact and correctness check for the
rest-attractor contract.

MVP scope: the V grid is ``[-85.23]`` only. Full-grid L_vclamp is deferred
(ARCHITECTURE_v4 §10.2 / PLAN Phase 2.5 follow-up) because the latent target
at non-rest voltages requires inverting a training-time-varying decoder.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

# V5.4's cardiac_sim is not a distributed package — add it to sys.path before
# importing. This mirrors the sys.path append pattern used by Surrogate's
# data_cache.py when reading raw simulator outputs.
_V54 = Path(__file__).resolve().parents[3] / "Monodomain" / "Engine_V5.4"
if _V54.is_dir() and str(_V54) not in sys.path:
    sys.path.insert(0, str(_V54))

from cardiac_sim.ionic.ttp06.model import TTP06Model  # noqa: E402
from cardiac_sim.ionic.ttp06.parameters import (  # noqa: E402
    StateIndex,
    get_initial_state as _v54_get_initial_state,
)

from surrogate.model.stage1 import TTP06_REST_IONIC_STATE  # noqa: E402

# Resting concentrations match V5.4's CellML reference epicardial steady state
# (``cardiac_sim.ionic.ttp06.parameters.get_initial_state``). Using V5.4's own
# rest values keeps the clamp self-consistent with its RHS; the mismatched
# Surrogate INIT_CONC values cause slow CaSR drift under a clamp.
_V54_REST = _v54_get_initial_state(device=torch.device("cpu"), dtype=torch.float64)

# Surrogate 14-dim observable index -> V5.4 StateIndex. Matches
# preprocessor.py:149-157 (gates_plus_rr = states[:, 5:18]; plus CaSR at 3).
_SURR_TO_V54 = (
    StateIndex.m, StateIndex.h, StateIndex.j,
    StateIndex.r, StateIndex.s,
    StateIndex.d, StateIndex.f, StateIndex.f2, StateIndex.fCass,
    StateIndex.Xr1, StateIndex.Xr2,
    StateIndex.Xs,
    StateIndex.RR,
    StateIndex.CaSR,
)

# V5.4 compute_concentration_rates returns 6 dims in order
# [dKi, dNai, dCai, dCaSR, dCaSS, dRR].
_DCASR_IDX = 3
_DRR_IDX = 5

# Output artifact lives with other diagnostic outputs.
_DEFAULT_OUT = Path(__file__).resolve().parents[2] / "diagnostics" / "artifacts" / "z_ss_grid.pt"


def _build_full_state(z14: torch.Tensor) -> torch.Tensor:
    """Embed the 14-dim observable vector into V5.4's (1, 18) state tensor.

    Concentration slots not included in the observable (Ki, Nai, Cai, CaSS)
    are held at V5.4's canonical rest values; CaSR and RR come from ``z14``.
    """
    state = torch.zeros(1, int(StateIndex.N_STATES), dtype=torch.float64)
    state[0, StateIndex.Ki] = _V54_REST[StateIndex.Ki]
    state[0, StateIndex.Nai] = _V54_REST[StateIndex.Nai]
    state[0, StateIndex.Cai] = _V54_REST[StateIndex.Cai]
    state[0, StateIndex.CaSS] = _V54_REST[StateIndex.CaSS]
    for surr_idx, v54_idx in enumerate(_SURR_TO_V54):
        state[0, v54_idx] = z14[surr_idx]
    return state


def _primitives(
    model: TTP06Model, V_scalar: torch.Tensor, z14: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose V5.4 primitives for one voltage-clamped snapshot.

    Returns:
        g_inf: (12,) steady-state gate values.
        g_tau: (12,) gate time constants.
        dRR:   scalar RR rate (from concentration_rates index 5).
        dCaSR: scalar CaSR rate (from concentration_rates index 3).
    """
    full = _build_full_state(z14).to(device=V_scalar.device)
    V_batched = V_scalar.view(1)
    g_inf = model.compute_gate_steady_states(V_batched, full)[0]
    g_tau = model.compute_gate_time_constants(V_batched, full)[0]
    conc_rates = model.compute_concentration_rates(V_batched, full)[0]
    return g_inf, g_tau, conc_rates[_DRR_IDX], conc_rates[_DCASR_IDX]


def _rate_14(model: TTP06Model, V_scalar: torch.Tensor, z14: torch.Tensor) -> torch.Tensor:
    """14-dim rate vector via V5.4 primitives — thin composition, not an RHS rewrite.

    - HH gates: ``dg_i/dt = (g_inf_i - g_i) / tau_i``.
    - RR and CaSR: extracted from ``compute_concentration_rates``.

    Used for the convergence criterion (``|dg|/(1+|g|)``). Integration itself
    uses Rush-Larsen for the gates — see :func:`_step`.
    """
    g_inf, g_tau, dRR, dCaSR = _primitives(model, V_scalar, z14)
    gates = z14[:12]
    dgate = (g_inf - gates) / g_tau
    rate = torch.empty(14, dtype=torch.float64, device=V_scalar.device)
    rate[:12] = dgate
    rate[12] = dRR
    rate[13] = dCaSR
    return rate


def _step(
    model: TTP06Model, V_scalar: torch.Tensor, z14: torch.Tensor, dt_ms: float
) -> torch.Tensor:
    """One time-step using Rush-Larsen for gates and Euler for CaSR + RR.

    Rush-Larsen is exact for linear gate dynamics and unconditionally stable
    across the stiff m-gate near rest (where ``tau_m`` is ~0.1 ms). CaSR and
    RR are slow (~10-1000 ms scale) so Euler is fine.
    """
    g_inf, g_tau, dRR, dCaSR = _primitives(model, V_scalar, z14)
    gates = z14[:12]
    decay = torch.exp(-dt_ms / g_tau)
    gates_new = g_inf + (gates - g_inf) * decay
    z_new = z14.clone()
    z_new[:12] = gates_new
    z_new[12] = z14[12] + dt_ms * dRR
    z_new[13] = z14[13] + dt_ms * dCaSR
    return z_new


def compute_z_ss_grid(
    V_grid: Iterable[float] = (-85.23,),
    *,
    initial_state: Optional[torch.Tensor] = None,
    sim_dt_ms: float = 0.01,
    max_t_ms: float = 2000.0,
    rel_tol: float = 1e-4,
    device: Optional[torch.device] = None,
) -> dict:
    """Integrate the 14-observable subset to steady state under V clamp.

    The "steady state" is the held-V fixed point given V5.4's CellML rest
    concentrations clamped in slots Ki/Nai/Cai/CaSS. It is NOT identical to
    the paced-BCL steady state that ``TTP06_REST_IONIC_STATE`` was sampled
    from — CaSR in particular drifts a bit because paced Cai oscillations
    are absent under the clamp. Purpose of the artifact is twofold: (1) a
    V5.4 import / RHS-composition correctness check, and (2) a reference
    fixed point for future L_vclamp extensions (Phase 2.5).

    Args:
        V_grid: iterable of clamped membrane voltages (mV).
        initial_state: optional (14,) float64 starting vector. Defaults to
            ``TTP06_REST_IONIC_STATE.clone()`` when None.
        sim_dt_ms: time-step (ms). Rush-Larsen for gates, Euler for CaSR/RR.
        max_t_ms: integration horizon before giving up on convergence (ms).
            Raised from 500 to 2000 so the slow CaSR drift relaxes within
            the budget at ``rel_tol=1e-4``.
        rel_tol: per-dim relative tolerance — convergence when
            ``max_i |dg_i| / (1 + |g_i|) < rel_tol``. Loosened to 1e-4 for
            MVP; the clamped CaSR rate floor is ~7e-5 at ``t=500ms``.
        device: torch device; defaults to CPU.

    Returns:
        dict with ``V_grid`` (n,), ``z_ss_grid`` (n, 14), and ``converged``
        (n,) bool tensor flagging per-voltage convergence.
    """
    dev = device if device is not None else torch.device("cpu")
    start = (
        initial_state.clone() if initial_state is not None else TTP06_REST_IONIC_STATE.clone()
    ).to(dtype=torch.float64, device=dev)
    assert start.shape == (14,), f"initial_state must be (14,), got {tuple(start.shape)}"

    model = TTP06Model(device=dev, dtype=torch.float64)
    n_steps = int(max_t_ms / sim_dt_ms)

    results: list[torch.Tensor] = []
    converged_flags: list[bool] = []
    for V_value in V_grid:
        g = start.clone()
        V_scalar = torch.tensor(V_value, dtype=torch.float64, device=dev)
        converged = False
        for _ in range(n_steps):
            dg = _rate_14(model, V_scalar, g)
            if (dg.abs() / (1.0 + g.abs())).max().item() < rel_tol:
                converged = True
                break
            g = _step(model, V_scalar, g, sim_dt_ms)
        results.append(g.detach().cpu())
        converged_flags.append(converged)

    return {
        "V_grid": torch.tensor(list(V_grid), dtype=torch.float64),
        "z_ss_grid": torch.stack(results, dim=0),
        "converged": torch.tensor(converged_flags),
    }


def main() -> None:
    out = compute_z_ss_grid([-85.23])
    _DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, _DEFAULT_OUT)
    print(
        f"Saved {_DEFAULT_OUT}: V_grid={out['V_grid'].tolist()} "
        f"z_ss shape={tuple(out['z_ss_grid'].shape)} "
        f"converged={out['converged'].tolist()}"
    )


if __name__ == "__main__":
    main()
