"""Stimulation protocols that RUN simulations — the layer above trace analysis.

``erp`` (effective refractory period) is a *protocol*, not a trace read: it paces an S1 train,
delivers an S2 extrastimulus at a coupling interval, detects whether S2 produces a PROPAGATED
response, and bisects the coupling interval. Because it calls ``simulate``/``monodomain`` it must
live OUTSIDE ``analysis.py`` (which ``run``/``api`` import — a ``simulate`` call there is a
circular-import hazard); the engine factories are imported lazily at call time.
"""

import warnings

_THRESHOLD = -40.0


def _count_upcrossings(trace, threshold=_THRESHOLD) -> int:
    """Number of below→above threshold crossings in a 1-D voltage trace (activation count)."""
    above = trace >= threshold
    return int(((~above[:-1]) & above[1:]).sum().item())


def _captures(geometry, ionic_model, conductivity, stim_region, *, bcl, n_s1, ci,
              amplitude, duration, t0, probe_ix, probe_iy, dt, device, **sim_kwargs) -> bool:
    """Run one S1(×n_s1)+S2 episode and report whether S2 launched a propagated response.

    Capture ⇔ the downstream probe activates ONCE MORE than the S1 train alone would drive it."""
    from .api import monodomain
    from .stimulus.stim import Stim

    t_s2 = t0 + (n_s1 - 1) * bcl + ci
    stims = [Stim.from_region(geometry, stim_region, start_time=t0 + k * bcl,
                              duration=duration, amplitude=amplitude) for k in range(n_s1)]
    stims.append(Stim.from_region(geometry, stim_region, start_time=t_s2,
                                  duration=duration, amplitude=amplitude))
    sim = monodomain(geometry, ionic_model, conductivity, stims, dt=dt, device=device, **sim_kwargs)
    t_end = t_s2 + max(bcl, 400.0)
    r = sim.run(t_end, save_every=1.0)
    probe = r.Vm[:, probe_ix, probe_iy]
    # An S1-only run drives the probe n_s1 times; a captured S2 adds a (n_s1+1)-th activation.
    return _count_upcrossings(probe) > n_s1


def erp(geometry=None, ionic_model: str = 'ttp06', conductivity=None, *,
        stim_region=None, bcl: float = 1000.0, n_s1: int = 4, amplitude: float = -52.0,
        duration: float = 2.0, t0: float = 5.0, ci_min: float = 20.0, ci_max: float = 400.0,
        tol: float = 2.0, probe: str = 'far', dt=None, device: str = 'cpu', **sim_kwargs) -> float:
    """Effective refractory period (ms) via an S1S2 protocol + capture-detection bisection.

    Paces ``n_s1`` S1 beats at cycle length ``bcl``, then an S2 extrastimulus at a coupling interval
    CI; the ERP is the LONGEST CI that fails to elicit a propagated response, bisected to ``tol`` ms.
    Runs ~``log2((ci_max-ci_min)/tol)`` simulations. ``geometry``/``conductivity`` are the usual
    declarative ``monodomain`` inputs (default: a short 1-D cable). Returns ``nan`` if S2 captures
    even at ``ci_min`` (no refractoriness resolved) or blocks even at ``ci_max``.

    See :func:`erp_proxy` (APD90 shortcut) and :func:`post_repol_refractoriness` (ERP − APD90).
    """
    from .conductivity import ConductivityConfig
    from .grid import Grid

    if geometry is None:
        geometry = Grid(60, 3, 0.02)
    if conductivity is None:
        conductivity = ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    if stim_region is None:
        stim_region = lambda x, y: x < 0.04         # left-edge line stimulus
    probe_ix = (geometry.Nx * 3) // 4 if probe == 'far' else geometry.Nx // 2
    probe_iy = geometry.Ny // 2

    def cap(ci):
        return _captures(geometry, ionic_model, conductivity, stim_region, bcl=bcl, n_s1=n_s1,
                         ci=ci, amplitude=amplitude, duration=duration, t0=t0,
                         probe_ix=probe_ix, probe_iy=probe_iy, dt=dt, device=device, **sim_kwargs)

    if cap(ci_min):
        warnings.warn(f"erp: S2 captures even at ci_min={ci_min} ms — ERP is below the search "
                      f"window; returning nan (lower ci_min).", stacklevel=2)
        return float('nan')
    if not cap(ci_max):
        warnings.warn(f"erp: S2 blocks even at ci_max={ci_max} ms — ERP is above the search window; "
                      f"returning nan (raise ci_max / check capture).", stacklevel=2)
        return float('nan')

    lo, hi = ci_min, ci_max                          # lo blocks, hi captures
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if cap(mid):
            hi = mid
        else:
            lo = mid
    return lo                                        # longest CI that still fails to capture


def erp_proxy(apd90: float) -> float:
    """The APD90 shortcut for ERP — valid only when there's no post-repolarization refractoriness."""
    return apd90


def post_repol_refractoriness(erp_ms: float, apd90: float) -> float:
    """Post-repolarization refractoriness = ERP − APD90 (ms). Positive ⇒ the cell stays refractory
    after it has repolarized (reduced excitability); ``≈0`` in healthy tissue."""
    return erp_ms - apd90
