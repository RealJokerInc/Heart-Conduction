"""Phase-7 (analysis.fields): the Boyle–Vigmond safety factor (Step 7.2).

SF = ∫_A source_sink dt / Q_thr. Qualitative validation: SF is finite/positive in a healthy
propagating wake and DROPS at a source–sink mismatch (an expansion); it raises on a bidomain result
(inherits the source_sink guard).
"""

import numpy as np
import pytest
import torch

from cardiac_core import monodomain, bidomain, Grid, ConductivityConfig, Stim
from cardiac_core.single_cell import safety_factor


def _cond():
    return ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)


def _stim(g):
    return Stim.from_region(g, (lambda x, y: x < 0.04), start_time=1.0, duration=2.0,
                            amplitude=-52.0)


# A fixed Q_thr keeps the tests fast (the auto-calibration bisects many single_cell sims).
_QTHR = 40.0


def test_sf_healthy_cable_positive():
    g = Grid(70, 6, 0.02)
    r = monodomain(g, 'ttp06', _cond(), _stim(g)).run(22.0, save_every=0.5)
    sf = safety_factor(r, q_thr=_QTHR)
    assert sf.shape == (70, 6)
    wake = sf[10:55, 3]
    wake = wake[torch.isfinite(wake)]
    assert wake.numel() > 0
    assert (wake > 0).all()                       # charge is delivered along the propagating wake


def test_sf_drops_at_source_sink_mismatch():
    # A narrow isthmus (source) opening into a wide region (large sink) — the classic source–sink
    # mismatch. SF at the expansion mouth must be LOWER than in the uniform isthmus upstream.
    Nx, Ny = 80, 40
    mask = np.zeros((Nx, Ny), dtype=bool)
    mask[:40, 18:22] = True                       # left half: a thin 4-wide isthmus
    mask[40:, :] = True                           # right half: full-width chamber
    g = Grid(Nx, Ny, 0.02, mask=mask)
    r = monodomain(g, 'ttp06', _cond(), _stim(g)).run(30.0, save_every=0.5)
    sf = safety_factor(r, q_thr=_QTHR)
    isthmus = sf[20:35, 20]                        # well inside the uniform isthmus
    mouth = sf[41:45, 20]                          # just past the expansion
    isthmus = isthmus[torch.isfinite(isthmus)]
    mouth = mouth[torch.isfinite(mouth)]
    assert isthmus.numel() > 0 and mouth.numel() > 0
    assert mouth.median().item() < isthmus.median().item()   # the mismatch lowers SF


def test_sf_bidomain_raises():
    g = Grid(16, 12, 0.05)
    r = bidomain(g, 'ttp06', _cond(), _stim(g)).run(2.0, save_every=1.0)
    with pytest.raises(ValueError, match="monodomain"):
        safety_factor(r, q_thr=_QTHR)
