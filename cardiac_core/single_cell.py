"""0-D single-cell driver — the reaction term of the master equation alone.

``Cm dV/dt = −(I_ion + I_stim)`` with diffusion removed, integrated by the SAME per-node ionic
``step`` the tissue reaction substep uses (Rush–Larsen gates + forward-Euler V/concentrations). Two
payoffs (DESIGN § 9): a 1×1 tissue node and ``single_cell`` agree to tolerance (the ORd
concentration-ordering bug is sidestepped by sharing the step), and it produces the threshold-charge
curve the safety factor needs. Does NOT re-implement the ODE — it drives ``IonicModel.step``.
"""

from dataclasses import dataclass
from typing import Optional

import torch

# Per-model default dt (ms): stiffer models (ORd) need a finer step.
_DEFAULT_DT = {'ttp06': 0.02, 'ord': 0.005, 'phas13': 0.02, 'paci': 0.02, 'mhas13': 0.02}


@dataclass
class SingleCellResult:
    """0-D single-cell output: the V trace + the final ionic state (for a tissue init / SF calib)."""
    times: torch.Tensor
    V: torch.Tensor
    final_state: torch.Tensor
    model: object
    dt: float
    Cm: float

    def apd(self, repol: float = 0.9, threshold: float = -20.0) -> float:
        """APD (ms) of the LAST beat in the trace (via :func:`cardiac_core.analysis.apd_at`)."""
        from . import analysis
        V3 = self.V.reshape(-1, 1, 1)
        return analysis.apd_at(V3, self.times, 0, 0, repol=repol, threshold=threshold)

    @property
    def v_peak(self) -> float:
        return float(self.V.max())

    @property
    def v_rest(self) -> float:
        return float(self.V.min())


def _pace(model, V, states, dt, *, bcl, n_beats, t0, stim_amplitude, stim_duration, Cm,
          record: bool, save_every: int):
    """Drive ``n_beats`` at cycle length ``bcl`` via ``model.step``. Returns (times, V, final_state);
    times/V are empty when ``record`` is False (pre-pace). The external ``Cm`` rescales the per-step
    VOLTAGE change (the reaction-``/Cm`` property) while gates evolve normally."""
    device, dtype = model.device, model.dtype
    total = t0 + (n_beats - 1) * bcl + bcl
    n_steps = int(round(total / dt))
    stim_windows = [(t0 + k * bcl, t0 + k * bcl + stim_duration) for k in range(n_beats)]
    t_list, V_list = [], []
    for i in range(n_steps):
        t = i * dt
        Istim = 0.0
        for a, b in stim_windows:
            if a <= t < b:
                Istim = stim_amplitude
                break
        Is = torch.tensor(Istim, dtype=dtype, device=device)
        V_new, states = model.step(V, states, dt, Is)
        V = V + (V_new - V) / Cm if Cm != 1.0 else V_new     # reaction-/Cm on the voltage update
        if record and (i % save_every == 0):
            t_list.append(t)
            V_list.append(float(V))
    times = torch.tensor(t_list, dtype=torch.float64, device=device)
    V_tr = torch.tensor(V_list, dtype=torch.float64, device=device)
    return times, V_tr, V, states


def single_cell(model: str = "ttp06", *, celltype: str = "ENDO", dt: Optional[float] = None,
                bcl: float = 1000.0, n_beats: int = 1, pre_pace: int = 0,
                stim_amplitude: float = -52.0, stim_duration: float = 2.0, t0: float = 10.0,
                Cm: float = 1.0, save_every: Optional[float] = None,
                device: str = "cpu") -> SingleCellResult:
    """Run a 0-D single-cell action potential and return the trace + final state.

    Parameters
    ----------
    model : ionic model name ('ttp06'/'ord'/'phas13'/'paci'/'mhas13') or a pre-built instance.
    celltype : 'ENDO'/'EPI'/'MID' (applied to TTP06/ORd only).
    dt : step (ms); default per-model.
    bcl, n_beats : pacing cycle length + number of RECORDED beats.
    pre_pace : beats to run first and DISCARD (drive toward steady state before recording).
    stim_amplitude, stim_duration, t0 : the depolarizing stimulus (current uA/uF; ``<0`` = inward).
    Cm : membrane capacitance scaling of the reaction (``Cm=2`` halves the per-step reaction dV).
    """
    from .ionic.registry import build_ionic_model
    m = build_ionic_model(model, celltype, device=device) if isinstance(model, str) else model
    name = model if isinstance(model, str) else getattr(m, 'name', 'ttp06')
    dt = dt if dt is not None else _DEFAULT_DT.get(str(name).lower(), 0.02)
    save_every = max(1, int(round((save_every if save_every is not None else 1.0) / dt)))

    V = torch.tensor(m.V_rest, dtype=m.dtype, device=m.device)
    states = m.get_initial_state(n_cells=1)

    if pre_pace > 0:
        _, _, V, states = _pace(m, V, states, dt, bcl=bcl, n_beats=pre_pace, t0=t0,
                                stim_amplitude=stim_amplitude, stim_duration=stim_duration,
                                Cm=Cm, record=False, save_every=save_every)

    times, V_tr, V, states = _pace(m, V, states, dt, bcl=bcl, n_beats=n_beats, t0=t0,
                                   stim_amplitude=stim_amplitude, stim_duration=stim_duration,
                                   Cm=Cm, record=True, save_every=save_every)
    return SingleCellResult(times=times, V=V_tr, final_state=states, model=m, dt=dt, Cm=Cm)


def threshold_charge(model: str = "ttp06", celltype: str = "ENDO", *, duration: float = 2.0,
                     dt: Optional[float] = None, amp_strong: float = -200.0, amp_weak: float = -4.0,
                     tol: float = 2.0, device: str = "cpu") -> float:
    """Minimum stimulus charge ``|amp_thr|·duration`` (uA/uF·ms) that triggers an AP — the ``Q_thr``
    the safety factor normalizes by. Bisects the stimulus amplitude (``amp_strong`` fires,
    ``amp_weak`` does not). ``nan`` if even the strong pulse fails to capture."""
    def fires(amp):
        r = single_cell(model, celltype=celltype, dt=dt, n_beats=1, bcl=duration + 60.0, t0=5.0,
                        stim_amplitude=amp, stim_duration=duration, save_every=1.0, device=device)
        return r.v_peak > 0.0                       # overshoot ⇒ a real AP fired
    if not fires(amp_strong):
        return float('nan')
    if fires(amp_weak):
        return abs(amp_weak) * duration
    lo, hi = amp_strong, amp_weak                   # lo fires, hi does not
    while abs(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if fires(mid):
            lo = mid
        else:
            hi = mid
    return abs(lo) * duration


def safety_factor(r, *, q_thr: Optional[float] = None, window=(2.0, 6.0)) -> torch.Tensor:
    """Boyle–Vigmond conduction safety factor per node: ``SF = ∫_A source_sink dt / Q_thr``.

    The numerator is the total membrane charge over each node's activation window
    ``A = [LAT − window[0], LAT + window[1]]`` ms — i.e. the ``source_sink`` field ``∇·(D∇V)``
    integrated over A. Because ``∇·(D∇V) = Cm·dV/dt + I_ion`` (the PDE), this IS the Boyle–Vigmond
    ``Cm·ΔV + Q_ion`` numerator, computed from the (solver-matched) ``source_sink`` field alone — no
    separately-recorded ionic states needed. ``Q_thr`` is the single-cell threshold charge
    (auto-calibrated via :func:`threshold_charge` if ``None``). ``SF < 1`` ⇒ conduction block; NaN at
    nodes that never activate. Monodomain + isotropic (inherits the ``source_sink`` guards).

    Returns an ``(Nx, Ny)`` map.
    """
    ss = r.fields.source_sink                       # (T, Nx, Ny) — raises on bidomain/anisotropic
    lat = r.lat()                                   # (Nx, Ny), NaN where unactivated
    times = r.times
    if times.numel() < 2:
        return torch.full_like(lat, float('nan'))
    dt_save = float(times[1] - times[0])
    if q_thr is None:
        q_thr = threshold_charge(getattr(r, 'ionic_model', None) or 'ttp06',
                                 getattr(r, 'cell_type', None) or 'ENDO',
                                 device=str(r.Vm.device))

    back, fwd = window
    t = times.reshape(-1, 1, 1)
    lo = (lat - back).reshape(1, *lat.shape)
    hi = (lat + fwd).reshape(1, *lat.shape)
    in_window = (t >= lo) & (t <= hi) & torch.isfinite(lo)
    # Integrate the INWARD charge only (source_sink > 0 = the node being charged by its neighbours,
    # the [t_1%, t_Im0] charging phase); the later sourcing phase (source_sink < 0) would otherwise
    # cancel it. This is the Boyle–Vigmond numerator (charge delivered to reach threshold).
    inward = torch.clamp(ss, min=0.0)
    charge = torch.nansum(torch.where(in_window, inward, torch.zeros_like(ss)), dim=0) * dt_save
    sf = charge / q_thr
    return torch.where(torch.isfinite(lat), sf, torch.full_like(sf, float('nan')))
