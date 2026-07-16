"""
Optimizer V1 — cardiac_core-backed single-cell AP runner (P-1 backend unification).

WHY THIS EXISTS (JOINT_TUNING_ARCHITECTURE §5 "FOUNDATIONAL BLOCKER"):
The V1 pipeline measured dV/dt/APD on the `cardiac_sim` (V5.4) model (`cell_runner`/
`batch_ionic`) but tissue CV on `cardiac_core` (`cc_runner`). Two different code paths
for the "same" model → a cardiac_core Na-kinetics axis (P1.5) would move CV but leave
the V5.4-measured dV/dt unchanged → the axis meant to decouple them is invisible to the
objective that identifies it. This runner measures the AP on the SAME cardiac_core model,
via the SAME hook path the tissue solver uses.

HOOK PARITY (load-bearing): cardiac_core's monodomain `RushLarsenSolver` drives the model
through `compute_Iion` / `compute_gate_steady_states` / `compute_gate_time_constants` — it
NEVER calls `MHAS13Model.step()` (which applies Cai-dependent ICaL `constf1`/`constfCa`
modifiers the hooks lack). So the cell AP is driven as a 0-D `run_monodomain` on a small
uniform strip (all cells stimulated → spatially flat field → diffusion is inert), which
goes through the identical hook-based Rush-Larsen path. A P1.5 kinetics edit placed in the
hooks is then visible to BOTH observables.

PACING PARITY (load-bearing): the V5.4 baseline (`cell_runner`) paces `n_beats` at
`pacing_cl` and measures the *last* AP. `create_cardiac_mesh` hard-codes a single
stimulus (`num_pulses=1, bcl=0`), which differs from the Nth paced beat by ≫1%. So we
patch the mesh stimulus to `num_pulses=n_beats, bcl=pacing_cl` (honored by
`cardiac_core.api` `_build_stimulus_protocol*`, api.py:1013) and measure the last beat.
"""

import numpy as np
import torch

from cardiac_core import create_cardiac_mesh, run_monodomain

from .config import TuningConfig
from .cell_result import CellResult
from .cc_runner import _build_model
from .metrics import (
    measure_apd, measure_dvdt_max, measure_v_rest, measure_peak,
)

# Uniform-strip geometry. A few cells (NOT 1×1) avoids degenerate CN/PCG operators;
# every cell is stimulated so the field stays flat and diffusion contributes nothing,
# reproducing a 0-D cell through the tissue solver's hook path.
_STRIP_NCELLS = 8          # cells along the strip (x)
_STRIP_NROWS = 3           # rows across the strip (y)
# Effective diffusivity is irrelevant under a flat field, but a physiological value
# keeps the CN operator well-conditioned. chi=1.0 → D is already-effective (cm²/ms).
_STRIP_D = 1.0e-3


def run_single_cell_cc(theta_ionic, config: TuningConfig,
                       *, n_beats: int = None, cl: float = None,
                       return_trace: bool = False, model=None) -> CellResult:
    """Single-cell AP biomarkers (APD90, dV/dt_max, V_rest, V_peak) on cardiac_core.

    Drives a small uniform strip via `run_monodomain` (hook path), paced to steady
    state, and extracts the last AP at a single node.

    Parameters
    ----------
    theta_ionic : tensor | dict | None
        Conductance scaling (same convention as `cc_runner._build_model`).
    config : TuningConfig
        Uses `dt_cell` (cell dt), `n_beats`, `pacing_cl`, stim_* and `ionic_model`.
    n_beats, cl : optional overrides for beat count / cycle length.
    model : optional pre-built ionic model (e.g. a kinetic-scaled instance from
        decision_space.apply). If given, `theta_ionic` is ignored — the SAME model is
        used for the cell AP as for tissue CV, so a P1.5 kinetics axis is identifiable
        from both observables.
    """
    if n_beats is None:
        n_beats = config.n_beats
    if cl is None:
        cl = config.pacing_cl

    dt = config.dt_cell
    if model is None:
        model = _build_model(theta_ionic, config)

    # --- uniform strip mesh; stimulate the WHOLE strip (stim_width ≥ Lx) ---
    dx = config.dx_cm
    Lx = (_STRIP_NCELLS - 1) * dx
    Ly = (_STRIP_NROWS - 1) * dx
    mesh = create_cardiac_mesh(
        Lx=Lx, Ly=Ly, dx=dx, D=_STRIP_D, chi=1.0, Cm=1.0,
        ionic_model=config.ionic_model, dt=dt,
        stim_width=Lx + dx,                       # cover every column → flat field
        stim_amplitude=config.stim_amplitude,
        stim_duration=config.stim_duration,
        stim_start=config.stim_start,
    )
    # Patch the single stimulus into a multi-pulse pacing train (parity with V5.4).
    mesh.stimuli[0]['num_pulses'] = int(n_beats)
    mesh.stimuli[0]['bcl'] = float(cl)

    # Cover the full last cycle so the final AP fully repolarizes before t_end.
    t_end = config.stim_start + n_beats * cl

    times, V = run_monodomain(
        mesh, t_end=t_end, save_every=dt,         # fine save resolves the upstroke (dV/dt)
        ionic_model=model, dt=dt, device=config.device,
    )

    if not torch.is_tensor(V):
        V = torch.as_tensor(V)
    if not torch.isfinite(V).all():
        return CellResult(converged=False)

    Nx, Ny = V.shape[1], V.shape[2]
    Vn = V[:, Nx // 2, Ny // 2].detach().cpu().numpy()      # flat field → any node
    t = (times.detach().cpu().numpy() if torch.is_tensor(times)
         else np.asarray(times))

    if len(Vn) == 0 or not np.isfinite(Vn).all():
        return CellResult(converged=False)

    result = CellResult(
        apd90=measure_apd(Vn, t),                 # measure_apd(V, t)  — voltage first
        dvdt_max=measure_dvdt_max(Vn, t),         # measure_dvdt_max(V, t)
        v_rest=measure_v_rest(Vn, t),
        v_peak=measure_peak(Vn),                  # measure_peak(V)    — one arg
        converged=True,
    )
    if return_trace:
        result.V_trace = Vn
        result.t_trace = t
    return result


def run_cell_batch_cc(theta_batch, config: TuningConfig, targets=None):
    """Serial batch AP eval on cardiac_core → List[CellResult].

    cardiac_core ionic models carry SCALAR `self.params` (no per-node conductances),
    so distinct-candidate batches are serialized. This is acceptable: the joint
    architecture RETIRES the 200-iter BO cell loop, so serial cell eval only ever
    runs the feasibility map + emulator training (~O(100) evals), not thousands.

    `targets` is accepted for signature-parity with `extract_biomarkers_batch`
    (unused here — constraint checking stays in the caller).
    """
    from .config import theta_to_dict  # local: keep module import cheap

    if torch.is_tensor(theta_batch) and theta_batch.dim() == 1:
        theta_batch = theta_batch.unsqueeze(0)

    results = []
    for i in range(len(theta_batch)):
        theta_i = theta_batch[i]
        results.append(run_single_cell_cc(theta_i, config))
    return results
