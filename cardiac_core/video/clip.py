"""``Video`` — the spec object describing ONE panel.

Separating "what to render" from "when/where to write" is the point of the design: the spec is
inspectable and reusable, a multi-panel comparison is just a list of these, and ``preview()`` lets
you check colours before paying for a long encode.

This module is also the single seam where torch->numpy conversion and masking happen, so both are
done once and correctly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

import numpy as np

from .gradient import Gradient

__all__ = ["Video"]

_LEGAL_STYLE = ("bare", "annotated")
_LEGAL_ASPECT = ("equal", "auto")
_LEGAL_UNITS = ("auto", "cm", "nodes")


def _to_numpy(x):
    """torch (possibly CUDA) / array-like -> numpy on the CPU.

    Mandatory: ``np.asarray`` on a CUDA tensor raises ``TypeError``.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass(eq=False)     # data/field/mask may be ndarrays -> a generated __eq__ would raise
class Video:
    """A renderable description of one field of one run.

    ``style="bare"`` (the default) is the full-frame, unlabelled convention; ``label``, ``front``,
    ``isochrones``, ``aspect`` and ``units`` are FIGURE-PRODUCER ONLY.
    """

    data: Any
    field: Union[str, np.ndarray] = "Vm"
    gradient: Gradient = Gradient.physiological()      # safe as a plain default: Gradient is frozen
    label: Optional[str] = None                        # panel title (FIGURE ONLY)
    front: Optional[float] = None                      # mV isoline, per frame (FIGURE ONLY)
    isochrones: bool = False                           # static LAT contours (FIGURE ONLY)
    mask: Any = None                                   # None=auto | array | False=explicitly none
    style: str = "bare"
    aspect: str = "equal"                              # FIGURE ONLY
    units: str = "auto"                                # FIGURE ONLY

    def __post_init__(self):
        if self.style not in _LEGAL_STYLE:
            raise ValueError(f"style must be one of {_LEGAL_STYLE}, got {self.style!r}")
        if self.aspect not in _LEGAL_ASPECT:
            raise ValueError(f"aspect must be one of {_LEGAL_ASPECT}, got {self.aspect!r}")
        if self.units not in _LEGAL_UNITS:
            raise ValueError(f"units must be one of {_LEGAL_UNITS}, got {self.units!r}")

        frames, times, dx, dy, value_label, result = self._resolve_data()
        self.result = result                # KEEP the torch result: activation_time is torch-only
        self.value_label = value_label
        self.dx, self.dy = dx, dy

        frames = np.asarray(frames, dtype=np.float64)     # float64 contract
        if frames.ndim == 2:
            frames = frames[None, ...]                    # a single frame -> a 1-frame movie
        if frames.ndim != 3:
            raise ValueError(
                f"expected frames shaped (T, Nx, Ny); got {frames.shape}. Pass a SimulationResult, "
                f"a (times, V) pair, a (T, Nx, Ny) array, or a .npz path.")
        if frames.shape[0] == 0:
            raise ValueError(
                "nothing to render: this result has 0 saved frames (t_end < save_every?). "
                "Increase t_end or decrease save_every.")
        self.frames = frames

        if times is None or len(times) != frames.shape[0]:
            if times is not None and len(times) != frames.shape[0]:
                warnings.warn(
                    "time axis length does not match the frame count; using frame indices "
                    "(a time stamp will read 't = 7.0 ms' for frame 7)", UserWarning, stacklevel=2)
            elif times is None:
                warnings.warn(
                    "no time axis supplied; frame indices are being shown as milliseconds",
                    UserWarning, stacklevel=2)
            times = np.arange(frames.shape[0], dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)

        # mask: False = explicitly none; None = auto (the result's domain_mask). True = ACTIVE.
        if self.mask is False:
            active = None
        elif self.mask is not None:
            active = _to_numpy(self.mask).astype(bool)
        else:
            dm = getattr(result, "domain_mask", None) if result is not None else None
            active = _to_numpy(dm).astype(bool) if dm is not None else None
        if active is not None and active.shape != self.frames.shape[1:]:
            raise ValueError(
                f"mask shape {active.shape} does not match the grid {self.frames.shape[1:]}")
        self.active_mask = active

    # ---- input normalisation ----
    def _resolve_data(self):
        data, field = self.data, self.field
        result = None
        times = dx = dy = None
        label = None

        if isinstance(data, str):                          # a saved .npz
            from .. import io as _io
            times, V, phi_e, _meta = _io.load_result(data)
            warnings.warn(
                "a .npz result carries no dx/dy or domain_mask — axes fall back to node indices "
                "and no automatic masking is applied", UserWarning, stacklevel=3)
            frames = _to_numpy(phi_e if (isinstance(field, str) and field == "phi_e") else V)
            label = "phi_e (mV)" if (isinstance(field, str) and field == "phi_e") else "Vm (mV)"
            return frames, _to_numpy(times), None, None, label, None

        if isinstance(data, (tuple, list)) and len(data) == 2:
            return (_to_numpy(data[1]), _to_numpy(data[0]), None, None, "Vm (mV)", None)

        if hasattr(data, "Vm") or hasattr(data, "V"):
            result = data
            if isinstance(field, str):
                if field in ("Vm", "V"):
                    frames = _to_numpy(getattr(result, "Vm", None) if hasattr(result, "Vm")
                                       else result.V)
                    label = "Vm (mV)"
                elif field == "phi_e":
                    phi = getattr(result, "phi_e", None)
                    if phi is None:
                        raise ValueError(
                            "field='phi_e' but this result has no extracellular potential "
                            "(phi_e is None — only the bidomain engine produces it)")
                    frames, label = _to_numpy(phi), "phi_e (mV)"
                else:
                    attr = getattr(result, field, None)
                    if attr is None:
                        raise ValueError(
                            f"field={field!r} is not available on this result. Use 'Vm', 'phi_e', "
                            f"or pass an explicit (T, Nx, Ny) array as field=.")
                    frames, label = _to_numpy(attr), field
            else:
                frames, label = _to_numpy(field), "value"
            t = getattr(result, "times", None)
            return (frames, _to_numpy(t) if t is not None else None,
                    getattr(result, "dx", None), getattr(result, "dy", None), label, result)

        return _to_numpy(data), None, None, None, "Vm (mV)", None

    # ---- presets ----
    @classmethod
    def bare(cls, data, **kw) -> "Video":
        """Full-frame, no labels — the default convention."""
        kw["style"] = "bare"
        return cls(data, **kw)

    @classmethod
    def annotated(cls, data, **kw) -> "Video":
        """Axes, colorbar and labels — the figure producer."""
        kw["style"] = "annotated"
        return cls(data, **kw)

    # ---- the single masking seam ----
    def display_values(self, t: int) -> np.ndarray:
        """Frame ``t`` with inactive tissue set to NaN. UNtransposed ``(Nx, Ny)``.

        Masking routes through ``domain_mask`` (True = ACTIVE) rather than ``isfinite``, because
        LBM leaves masked nodes FINITE — an isfinite-only rule would paint obstacles as real
        voltage.
        """
        a = self.frames[t]
        if self.active_mask is None:
            return a
        return np.where(self.active_mask, a, np.nan)

    def masked_iter(self, idx) -> Iterator[np.ndarray]:
        """Yield ONE masked ``(Nx, Ny)`` array per frame in ``idx`` — what ``Gradient.resolve`` eats."""
        for t in idx:
            yield self.display_values(t)

    def requires_figure(self) -> bool:
        """True if this clip needs the matplotlib producer.

        Style-driven as well as overlay-driven: an annotated clip with NO overlays must still
        route to the figure producer, or the legacy delegation silently renders bare.
        """
        return (self.style == "annotated" or self.label is not None
                or self.front is not None or bool(self.isochrones))

    def preview(self, t_ms: Optional[float] = None, *, frame: Optional[int] = None,
                slug: str = "preview", question: Optional[str] = None,
                bulk: Optional[bool] = None, **kw):
        """Render ONE frame through this clip's OWN producer (PNG unless ``path=`` says otherwise).

        Displays inline; writes a file only when a destination is named (``path=`` or the
        ``media/`` convention keywords).
        """
        from .render import preview_frame          # local import: clip.py must not import render
        return preview_frame(self, t_ms=t_ms, frame=frame, slug=slug,
                             question=question, bulk=bulk, **kw)

    def __repr__(self) -> str:
        T, Nx, Ny = self.frames.shape
        try:
            _, _, lo, hi = self.gradient.resolve(self.masked_iter(range(T)), field=self.field)
            rng = f"({lo:.1f}, {hi:.1f}) provisional"
        except Exception:
            rng = "unresolved"
        overlays = [n for n, on in (("front", self.front is not None),
                                    ("isochrones", bool(self.isochrones)),
                                    ("label", self.label is not None)) if on]
        fld = self.field if isinstance(self.field, str) else "<array>"
        return (f"Video(field={fld!r}, grid=({Nx}, {Ny}), frames={T}, "
                f"range~{rng}, style={self.style!r}, overlays={overlays})")
