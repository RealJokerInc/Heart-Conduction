"""
Optimizer V1 — cardiac_core-backed runner (Phase 1).

Replaces the per-engine tissue runners with a single runner over cardiac_core's
FUNCTIONAL API (`run_monodomain`/`run_lbm`/`run_bidomain` + `analysis`). Uses the
Phase 0 tuning seam: a tuner-scaled IonicModel instance (via `apply_scaling`) +
per-axis mesh D. The stubbed OO `CardiacSimulation` methods are NOT used.

CV convention: on a 1-D cable, CV depends only on the diffusion ALONG the cable,
so `run_1d_cable(theta, D, config)` measures CV for diffusion `D` — the caller
passes D_long for CV_L and D_trans for CV_T (mirrors the legacy run_cv_measurement
signature, so it shims cleanly).

IMPORTANT mesh convention (Phase 0): `create_cardiac_mesh` treats `D` as the
already-effective diffusion (cm²/ms) but the FDM operator divides by chi, so we
build meshes with chi=1.0 (else effective D is rescaled ~1400x and the wave dies).
"""
import numpy as np
import torch

from cardiac_core import create_cardiac_mesh, run_monodomain, run_lbm, run_bidomain
from cardiac_core import analysis
from cardiac_core.ionic import MHAS13Model, PHAS13Model, TTP06Model

from .config import TuningConfig, apply_scaling, theta_to_dict

_MODELS = {
    "mhas13": MHAS13Model,
    "phas13": PHAS13Model,
    "paci": PHAS13Model,
    "ttp06": TTP06Model,
}

_ACT_THRESHOLD = -30.0   # mV — matches the legacy run_cv_measurement


def _build_model(theta_ionic, config: TuningConfig):
    """Construct the ionic model and apply tuner-scaling on the tuner side."""
    cls = _MODELS.get(config.ionic_model, MHAS13Model)
    model = cls(device=config.device)
    if theta_ionic is not None:
        theta_dict = (theta_to_dict(theta_ionic, config.tier)
                      if torch.is_tensor(theta_ionic) else dict(theta_ionic))
        apply_scaling(model.params, theta_dict)
    return model


def _default_t_end(config: TuningConfig) -> float:
    """Generous t_end so even a slow (~3 cm/s) front crosses the probe span."""
    min_cv_cm_per_ms = 0.003          # 3 cm/s lower bound
    span_cm = 0.5 * config.cable_length_cm
    return config.stim_start + span_cm / min_cv_cm_per_ms + 20.0


def run_1d_cable(theta_ionic, D: float, config: TuningConfig,
                 *, t_end: float = None, save_every: float = 1.0,
                 return_vmax: bool = False):
    """Measure conduction velocity (cm/s) for a cable with diffusion `D`.

    Dispatches by `config.engine` over the cardiac_core functional API.
    Returns NaN if the wave fails to cross both probes. With ``return_vmax=True``
    returns ``(cv, vmax)`` — the peak |V| over the whole field, which distinguishes
    a high-D over-depolarization blow-up (Vmax non-physical) from a low-D source-sink
    block (Vmax physiological but the wave dies) when cv is NaN (used by the Step 1.3
    hiPSC-window diagnostic).
    """
    model = _build_model(theta_ionic, config)
    dx = config.dx_cm
    L = config.cable_length_cm
    Ly = max(4 * dx, 0.04)            # thin strip across the cable
    if t_end is None:
        t_end = _default_t_end(config)

    mesh = create_cardiac_mesh(
        Lx=L, Ly=Ly, dx=dx, D=D, chi=1.0, Cm=1.0,
        ionic_model=config.ionic_model, dt=config.dt,
        stim_width=max(2 * dx, 0.02), stim_amplitude=config.stim_amplitude,
        stim_duration=config.stim_duration, stim_start=config.stim_start,
    )

    engine = config.engine
    if engine == "lbm":
        dt = getattr(config, "dt_lbm", None) or config.dt
        times, V = run_lbm(mesh, t_end=t_end, save_every=save_every,
                           ionic_model=model, dt=dt, device=config.device)
    elif engine == "bidomain":
        # bidomain() auto-derives D_i/D_e from the effective mesh D + sigma_ratio
        # (no sigma_i/sigma_e needed) — reuse the same chi=1.0 effective-D mesh.
        times, V, _phi = run_bidomain(mesh, t_end=t_end, save_every=save_every,
                                      ionic_model=model, dt=config.dt,
                                      sigma_ratio=config.De_Di_ratio,
                                      device=config.device)
    else:  # monodomain
        times, V = run_monodomain(mesh, t_end=t_end, save_every=save_every,
                                  ionic_model=model, dt=config.dt, device=config.device)

    if not torch.is_tensor(times):
        times = torch.as_tensor(times, dtype=torch.float64)
    Nx, Ny = mesh.mask.shape
    cv = analysis.conduction_velocity(
        V, times, dx=dx, x1=Nx // 4, x2=3 * Nx // 4, y=Ny // 2,
        threshold=_ACT_THRESHOLD,
    )
    if return_vmax:
        vmax = float(V.max()) if torch.is_tensor(V) else float(np.max(V))
        return float(cv), vmax
    return float(cv)


def run_2d_tissue(theta_ionic, D_long: float, D_trans: float, config: TuningConfig,
                  *, t_end: float = None, save_every: float = 1.0) -> dict:
    """Measure CV_long, CV_trans (cm/s) on anisotropic tissue.

    On a cable, CV along an axis depends only on that axis's diffusion, so both
    axes are measured by `run_1d_cable` with the respective D. (A future 2-D
    point-source variant can refine tissue-APD; CV is well-captured per-axis.)
    """
    cv_long = run_1d_cable(theta_ionic, D_long, config, t_end=t_end, save_every=save_every)
    cv_trans = run_1d_cable(theta_ionic, D_trans, config, t_end=t_end, save_every=save_every)
    return {"cv_long": cv_long, "cv_trans": cv_trans}


def _bracket_down(cv_fn, D_start, D_lo):
    """Halve D from D_start toward D_lo until a propagating point is found.

    Chip regime: the default warm-start D0≈1e-3 sits ABOVE the propagating window
    (high-D over-depolarization → NaN), and the window is BELOW it. Bumping D *up*
    on failure (the old ×4 fallback) walks into the NaN zone and returns a fake D
    (Known Failure); bracketing DOWN steps into the window. Returns (D, cv) or None.
    """
    import math
    D = D_start
    while D >= D_lo:
        c = cv_fn(D)
        if math.isfinite(c) and c > 0:
            return D, c
        D *= 0.5
    return None


def fit_D_for_cv(theta_ionic, target_cv, config, *, D0=0.001, n=8, tol=0.02,
                 D_lo=1e-6, D_hi=1e-2, t_end=None):
    """Secant on diffusion D to hit `target_cv` (cm/s) via run_1d_cable.

    Warm-started by CV ∝ √D, then two-point secant (NOT Newton — Known Failure).
    On a non-propagating start, brackets DOWN into the propagating window (the chip
    window is BELOW D0; the old ×4-up-bump was a Known Failure). Returns
    (D, cv_achieved), or **(NaN, NaN)** if no propagating D is found down to D_lo —
    honest infeasibility, never a fake fallback D.
    Shared by run_chip_fit (Phase 3) and cross_engine.recalibrate_lbm (Phase 4).
    """
    import math

    def cv(D):
        return run_1d_cable(theta_ionic, D, config, t_end=t_end)

    def ok(x):
        return math.isfinite(x) and x > 0

    cv0 = cv(D0)
    if not ok(cv0):
        found = _bracket_down(cv, D0 * 0.5, D_lo)
        if found is None:
            return float("nan"), float("nan")     # no propagating D — do NOT fake one
        D0, cv0 = found

    D1 = min(D_hi, max(D_lo, D0 * (target_cv / cv0) ** 2))
    cv1 = cv(D1)
    if not ok(cv1):
        # warm-start jumped out of the window — bracket down toward the known point.
        found = _bracket_down(cv, min(D1, D0) * 0.5, D_lo)
        if found is None:
            return D0, cv0                         # keep the known propagating point
        D1, cv1 = found
    it = 2
    while it < n and abs(cv1 - target_cv) / target_cv > tol:
        if cv1 == cv0:
            break
        D2 = min(D_hi, max(D_lo, D1 + (target_cv - cv1) * (D1 - D0) / (cv1 - cv0)))
        cv2 = cv(D2)
        if not ok(cv2):
            break
        D0, cv0, D1, cv1 = D1, cv1, D2, cv2
        it += 1
    return D1, cv1
