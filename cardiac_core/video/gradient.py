"""``Gradient`` — the reusable colour object.

In this project the colour *range* is a scientific choice, not decoration: the wall artifact
`render_audit_video.py` exists to show spans 5.8% of the default -90..40 mV colormap but 90.4% of
the zoom window — a 15.7x visibility gain. Making the mapping a first-class object is also what
lets multi-panel comparisons share one colorbar (and therefore be comparable at all).

The five presets encode the five colour intents found across the project's ~20 render scripts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, PowerNorm

__all__ = ["Gradient"]

_LEGAL_RANGES = ("physiological", "rest", "zoom", "auto", "auto99")
_LEGAL_INTERP = ("nearest", "bilinear")

# Bounded, deterministic subsample budget per frame for the auto99 percentiles.
_SAMPLE_PER_FRAME = 20_000


class _Stats:
    """Single-pass accumulator: min/max, a deterministic subsample, and frame 0's finite values.

    The iterator handed to :meth:`Gradient.resolve` may be a generator, so it is consumed EXACTLY
    once — ``first_frame_vals`` is captured during that same pass because ``infer_v_rest`` cannot
    re-read it.
    """

    def __init__(self):
        self.count = 0
        self.min = np.inf
        self.max = -np.inf
        self._sample = []
        self.first_frame_vals = np.empty(0, dtype=np.float64)
        self._seen_first = False

    def update(self, frame: np.ndarray) -> None:
        vals = frame[np.isfinite(frame)]
        if not self._seen_first:
            self.first_frame_vals = vals.astype(np.float64, copy=True)
            self._seen_first = True
        if vals.size == 0:
            return
        self.count += int(vals.size)
        self.min = min(self.min, float(vals.min()))
        self.max = max(self.max, float(vals.max()))
        # Deterministic stride (no RNG): identical data must give identical vmin/vmax.
        step = max(1, vals.size // _SAMPLE_PER_FRAME)
        self._sample.append(vals[::step])

    @property
    def sample(self) -> np.ndarray:
        if not self._sample:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(self._sample)


@dataclass(frozen=True, eq=False)
class Gradient:
    """How values map to colour.

    ``eq=False`` because a list-valued ``cmap`` (or a ``Colormap``) breaks the generated
    ``__eq__``/``__hash__``; use :meth:`key` to compare two gradients.
    """

    cmap: Union[str, list, Colormap] = "viridis"
    value_range: Union[str, tuple] = "physiological"
    gamma: float = 1.0
    levels: Optional[int] = None
    bad: str = "0.55"
    interpolation: str = "nearest"
    v_rest: Optional[float] = None
    rest_vmax: float = 40.0
    zoom_span: float = 8.0
    zoom_below: float = 0.3

    def __post_init__(self):
        vr = self.value_range
        ok = (isinstance(vr, str) and vr in _LEGAL_RANGES) or (
            isinstance(vr, (tuple, list)) and len(vr) == 2)
        if not ok:
            raise ValueError(
                f"value_range must be one of {_LEGAL_RANGES} or an explicit (lo, hi), got {vr!r}")
        if self.interpolation not in _LEGAL_INTERP:
            raise ValueError(
                f"interpolation must be one of {_LEGAL_INTERP}, got {self.interpolation!r}. "
                f"(It drives BOTH imshow and the PIL resampler, so other matplotlib names "
                f"cannot be honoured consistently.)")

    # ---- presets: the five colour intents observed in the render corpus ----
    @classmethod
    def physiological(cls, **kw) -> "Gradient":
        """-90..40 mV, viridis — the bare/PURE-DATA convention. The project default."""
        return cls(cmap="viridis", value_range="physiological", **kw)

    @classmethod
    def rest_anchored(cls, vmax: float = 40.0, **kw) -> "Gradient":
        """V_rest..vmax, inferno + grey masked tissue — the masked-obstacle convention."""
        return cls(cmap="inferno", value_range="rest", rest_vmax=vmax, bad="0.55", **kw)

    @classmethod
    def zoom(cls, span: float = 8.0, below: float = 0.3, **kw) -> "Gradient":
        """A narrow window just above rest — makes a few-mV artifact visible."""
        return cls(cmap="magma", value_range="zoom", zoom_span=span, zoom_below=below,
                   bad="0.6", **kw)

    @classmethod
    def diverging(cls, **kw) -> "Gradient":
        """-90..50 mV, RdBu_r — the comparison-panel convention."""
        return cls(cmap="RdBu_r", value_range=(-90.0, 50.0), **kw)

    @classmethod
    def autoscale(cls, **kw) -> "Gradient":
        """Full finite range of the data."""
        return cls(cmap="viridis", value_range="auto", **kw)

    def key(self) -> tuple:
        """Comparable identity for "do these panels share a gradient?".

        Compared with ``==``, never hashed — a list-valued ``cmap`` would make the tuple
        unhashable.
        """
        cmap_id = self.cmap if isinstance(self.cmap, str) else (
            tuple(self.cmap) if isinstance(self.cmap, list) else getattr(self.cmap, "name", id(self.cmap)))
        vr = tuple(self.value_range) if isinstance(self.value_range, (tuple, list)) else self.value_range
        return (cmap_id, vr, self.gamma, self.levels, self.bad, self.interpolation,
                self.v_rest, self.rest_vmax, self.zoom_span, self.zoom_below)

    # ---- resolution ----
    def _colormap(self) -> Colormap:
        c = self.cmap
        if isinstance(c, str):
            # matplotlib.colormaps (NOT plt.get_cmap) so gradient.py never imports pyplot,
            # which would make the Agg-backend guarantee depend on import order.
            base = matplotlib.colormaps[c]
        elif isinstance(c, Colormap):
            base = c
        else:
            base = LinearSegmentedColormap.from_list("custom", list(c))
        # ALWAYS copy: set_bad mutates in place. Registered colormaps are already safe (the
        # registry hands out copies), but a CALLER-SUPPLIED Colormap instance would be mutated.
        out = base.copy()
        if self.levels:
            out = out.resampled(int(self.levels))
        out.set_bad(self.bad)
        return out

    def _infer_v_rest(self, stats: _Stats, field) -> float:
        if isinstance(field, str) and field == "phi_e":
            raise ValueError(
                "value_range='rest'/'zoom' needs an explicit v_rest for phi_e "
                "(phi_e has no resting potential to anchor to)")
        vals = stats.first_frame_vals
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            warnings.warn("cannot infer v_rest (frame 0 has no finite values); using -85.0 mV",
                          UserWarning, stacklevel=3)
            return -85.0
        spread = float(np.percentile(vals, 95) - np.percentile(vals, 5))
        if spread > 5.0:
            warnings.warn(
                f"frame 0 is not at rest (5-95th percentile spread {spread:.1f} mV); "
                f"using the global finite minimum as v_rest", UserWarning, stacklevel=3)
            return float(stats.min)
        return float(np.median(vals))

    def resolve(self, masked_values, *, field: Any = "Vm"):
        """``(Colormap, Normalize, lo, hi)`` from an iterable of MASKED per-frame arrays.

        ``masked_values`` is consumed exactly once.
        """
        stats = _Stats()
        for frame in masked_values:
            stats.update(np.asarray(frame))

        vr = self.value_range
        if isinstance(vr, (tuple, list)):
            lo, hi = float(vr[0]), float(vr[1])       # an EXPLICIT range always wins
        elif stats.count == 0:
            warnings.warn("no finite unmasked data; falling back to (-90, 40) mV",
                          UserWarning, stacklevel=2)
            lo, hi = -90.0, 40.0
        elif vr == "physiological":
            lo, hi = -90.0, 40.0
        elif vr == "auto":
            lo, hi = stats.min, stats.max
        elif vr == "auto99":
            s = stats.sample
            lo, hi = float(np.percentile(s, 0.5)), float(np.percentile(s, 99.5))
        elif vr in ("rest", "zoom"):
            v = self.v_rest if self.v_rest is not None else self._infer_v_rest(stats, field)
            lo, hi = ((v - self.zoom_below, v + self.zoom_span) if vr == "zoom"
                      else (v, self.rest_vmax))
        else:                                         # pragma: no cover - __post_init__ guards it
            raise ValueError(f"unknown value_range {vr!r}")

        if not np.isfinite(lo) or not np.isfinite(hi):
            warnings.warn("resolved a non-finite colour range; falling back to (-90, 40) mV",
                          UserWarning, stacklevel=2)
            lo, hi = -90.0, 40.0
        if hi <= lo:
            warnings.warn(f"degenerate colour range ({lo}, {hi}); widening by +/-0.5",
                          UserWarning, stacklevel=2)
            lo, hi = lo - 0.5, lo + 0.5

        cm = self._colormap()
        # PowerNorm handles a NEGATIVE vmin (it normalises to [0,1] before applying the power):
        # PowerNorm(2.0, -90, 40)(-25.0) == 0.25 exactly.
        norm = PowerNorm(self.gamma, lo, hi) if self.gamma != 1.0 else Normalize(lo, hi)
        return cm, norm, float(lo), float(hi)
