"""``Image`` — the spec object describing ONE spatial map.

Separating "what to draw" from "where it goes" is the point: the spec is inspectable and reusable,
a multi-panel comparison is just a list of them, and the same object serves the headline
``r.image()`` and the full ``draw(Image(...), ...)`` form.

An ``Image`` composes a one-frame :class:`cardiac_core.video.Video` internally rather than
re-implementing the map producer, so torch->numpy ingest, the ``domain_mask`` masking seam, the
cm-vs-node extents and the orientation convention are shared with the video layer instead of
drifting from it. The construction order in ``__post_init__`` is load-bearing: ``Video`` resolves
its mask, time axis and gradient in its own ``__post_init__``, so those must go through the
constructor, while ``value_label``/``dx``/``dy``/``result`` are read at draw time and are assigned
after.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from ..video.clip import Video, _to_numpy
from ..video.gradient import Gradient

__all__ = ["Image"]

_LEGAL_STYLE = ("bare", "annotated")
_LEGAL_ASPECT = ("equal", "auto")
_LEGAL_UNITS = ("auto", "cm", "nodes")

# The four named intents. Everything else in the registry is a `fields.*` property name.
_NAMED_INTENTS = ("snapshot", "activation", "apd", "frequency")
# Trace's namespace — named here only so an `Image` can point the caller at the right spec.
_TRACE_INTENTS = ("trace", "restitution", "apd_per_beat")
# `Fields` members that are not renderable maps.
_FIELDS_NOT_DRAWABLE = ("derivatives", "integrals", "mask")

_DEFAULT_CONTOUR_LEVELS = 12


def _fields_names() -> tuple:
    """Renderable ``fields.*`` property names, by introspection (never a hard-coded list)."""
    from ..fields import Fields
    return tuple(sorted(
        n for n, v in vars(Fields).items()
        if isinstance(v, property) and not n.startswith("_") and n not in _FIELDS_NOT_DRAWABLE
    ))


@dataclass(eq=False)     # data/mask may be ndarrays -> a generated __eq__ would raise
class Image:
    """A renderable description of one spatial map.

    ``style="annotated"`` is the default — unlike :class:`~cardiac_core.video.Video`, whose default
    is bare. A video carries its information through motion; a still carries it through labels, and
    the audience for this layer cannot add labels afterwards.

    ``at`` is always a TIME in milliseconds. (On :class:`Trace` the same keyword means a NODE.)
    """

    data: Any
    what: str = "snapshot"
    at: Optional[float] = None
    field: Optional[str] = None
    what_kwargs: Optional[dict] = None
    gradient: Optional[Gradient] = None
    label: Optional[str] = None
    front: Optional[float] = None
    isochrones: Optional[bool] = None
    filled: bool = False
    contour_levels: int = _DEFAULT_CONTOUR_LEVELS
    mask: Any = None
    style: str = "annotated"
    aspect: str = "equal"
    units: str = "auto"
    value_label: Optional[str] = None

    def __post_init__(self):
        # --- 1. shape/enum validation -------------------------------------------------
        if self.style not in _LEGAL_STYLE:
            raise ValueError(f"style must be one of {_LEGAL_STYLE}, got {self.style!r}")
        if self.aspect not in _LEGAL_ASPECT:
            raise ValueError(f"aspect must be one of {_LEGAL_ASPECT}, got {self.aspect!r}")
        if self.units not in _LEGAL_UNITS:
            raise ValueError(f"units must be one of {_LEGAL_UNITS}, got {self.units!r}")

        # --- 1b. reject the wrong input types BEFORE touching `data` -------------------
        result = self._resolve_result()

        # --- 1c. validate `what` BEFORE resolving it ----------------------------------
        self._validate_what()

        # --- 2. resolve the selector; the gradient must become REAL before the clip ----
        explicit = self.gradient is not None
        arr2d, value_label, t_ms, default_gradient = self._resolve_selector(result)
        if not explicit:
            self.gradient = default_gradient
        # `Gradient.rest`/`zoom` anchor to a resting potential; on a non-voltage map
        # `_infer_v_rest` would take percentiles of durations and call the result millivolts.
        # It cannot self-guard: every Image clip reports field='Vm'.
        if self.gradient.value_range in ("rest", "zoom") and not self._is_voltage():
            raise ValueError(
                f"value_range={self.gradient.value_range!r} anchors to a resting potential, which "
                f"{self._selector_name()!r} does not have. Use value_range='auto'."
            )

        # --- 3. resolve `isochrones` (explicit always wins) ---------------------------
        if self.isochrones is None:
            self.isochrones = (self.what == "activation") and not self.filled

        # --- 3b. figure-only fields are illegal on a bare spec ------------------------
        if self.style == "bare":
            offenders = [n for n, on in (
                ("isochrones", self.isochrones),
                ("filled", self.filled),
                ("value_label", self.value_label is not None),
                ("contour_levels", self.contour_levels != _DEFAULT_CONTOUR_LEVELS),
                ("units", self.units != "auto"),
            ) if on]
            if offenders:
                raise ValueError(
                    f"{'/'.join(offenders)} need a figure and are not available on a bare spec. "
                    f'Use style="annotated".'
                )

        # --- 4. cache the LAT when the selector already computed it -------------------
        self._lat = arr2d if self.what == "activation" else None

        # --- 5. build the clip (mask/times/gradient MUST go through the constructor) ---
        if self.mask is False:
            m = False
        elif self.mask is not None:
            m = self.mask
        else:
            dm = getattr(result, "domain_mask", None) if result is not None else None
            m = dm if dm is not None else False
        self._clip = Video(
            (np.asarray([t_ms], dtype=np.float64), arr2d[None, ...]),
            gradient=self.gradient, mask=m, style=self.style, aspect=self.aspect,
            units=self.units, label=self.label, front=self.front,
            isochrones=False,        # the overlay is driven by `lat=`, not by the clip
        )
        # Read at DRAW time, so post-hoc assignment is correct here and only here.
        self._clip.value_label = self.value_label if self.value_label is not None else value_label
        self.value_label = self._clip.value_label
        self._clip.dx = getattr(result, "dx", None)
        self._clip.dy = getattr(result, "dy", None)
        self._clip.result = result

        # --- 5b. the overlay LAT must be masked exactly as the display array is --------
        if self.isochrones and self._lat is not None and self._clip.active_mask is not None:
            self._lat = np.where(self._clip.active_mask, self._lat, np.nan)

    # ---- input handling ----
    def _resolve_result(self):
        """Return the SimulationResult, or None for a bare 2-D array. Rejects everything else."""
        data = self.data
        if isinstance(data, str):
            raise ValueError(
                "Image does not accept a .npz path — load it and pass a 2-D array, or use "
                "Video(...).preview() for a saved run."
            )
        if isinstance(data, (tuple, list)):
            raise ValueError(
                "Image does not accept a (times, V) pair — pass a 2-D array, or use "
                "Video(...).preview()."
            )
        if hasattr(data, "Vm") and getattr(data, "Vm", None) is not None and hasattr(data, "times"):
            vm = data.Vm
            if getattr(vm, "ndim", 0) == 1:      # SingleCellResult: 1-D V, no grid
                raise ValueError(
                    "Image draws a spatial map; a single-cell result has no grid. Use .trace()."
                )
            if len(vm) == 0:
                raise ValueError(
                    "nothing to draw: this result has 0 saved frames (t_end < save_every?). "
                    "Increase t_end or decrease save_every."
                )
            return data
        if hasattr(data, "V") and not hasattr(data, "Vm"):       # SingleCellResult
            raise ValueError(
                "Image draws a spatial map; a single-cell result has no grid. Use .trace()."
            )
        arr = _to_numpy(data)
        if arr.ndim != 2:
            raise ValueError(
                f"a bare array must be 2-D (Nx, Ny); got shape {arr.shape}. A (T, Nx, Ny) history "
                f"is not accepted — pass one frame, or use Video(...).preview()."
            )
        return None

    def _validate_what(self):
        if self.what == "mask":
            raise ValueError(
                "'mask' is the domain gate, not a renderable field — use Image(mask=…)"
            )
        if self.what in _TRACE_INTENTS:
            raise ValueError(
                f"what={self.what!r} is a Trace intent, not a map — use Trace(...) / r.trace()."
            )
        valid = _NAMED_INTENTS + _fields_names()
        if self.what not in valid:
            raise ValueError(
                f"unknown what={self.what!r}; valid: {sorted(valid)}. "
                f"(electric_field/current_flux are bidomain-only.)"
            )

    def _selector_name(self) -> str:
        return self.field if self.field is not None else self.what

    def _is_voltage(self) -> bool:
        return self.field in ("Vm", "phi_e") or (self.field is None and self.what == "snapshot")

    # ---- the registry ----
    def _resolve_selector(self, result):
        """-> (arr2d float64 numpy, value_label, t_ms, default_gradient)."""
        from .. import analysis

        kw = self.what_kwargs or {}

        # A bare array: only the default `what` is meaningful.
        if result is None:
            if self.field is not None or self.what != "snapshot":
                raise ValueError(
                    "a bare array carries no run to analyse: only the default what='snapshot' is "
                    "legal, and field= needs a SimulationResult."
                )
            if kw:
                raise ValueError("what_kwargs= has no effect for a bare array")
            if self.at is not None:
                raise ValueError("at= is a time in ms and needs a SimulationResult")
            arr = np.asarray(_to_numpy(self.data), dtype=np.float64)
            return arr, "value", math.nan, Gradient(cmap="viridis", value_range="auto")

        if self.field is not None:
            if self.what != "snapshot":
                raise ValueError(
                    f"field={self.field!r} selects the raw voltage and is legal only with "
                    f"what='snapshot'; got what={self.what!r}."
                )
            if kw:
                raise ValueError("what_kwargs= has no effect for field='Vm'/'phi_e'")
            if self.field == "phi_e":
                phi = getattr(result, "phi_e", None)
                if phi is None:
                    raise ValueError(
                        "field='phi_e' but this result has no extracellular potential "
                        "(phi_e is None — only the bidomain engine produces it)"
                    )
                k, t_ms = self._frame_index(result)
                return (np.asarray(_to_numpy(phi[k]), dtype=np.float64),
                        "phi_e (mV)", t_ms, Gradient.physiological())
            if self.field not in ("Vm", "V"):
                raise ValueError(f"field must be 'Vm' or 'phi_e', got {self.field!r}")

        if self.what == "snapshot":
            if kw:
                raise ValueError("what_kwargs= has no effect for what='snapshot'")
            k, t_ms = self._frame_index(result)
            return (np.asarray(_to_numpy(result.Vm[k]), dtype=np.float64),
                    "Vm (mV)", t_ms, Gradient.physiological())

        # Every remaining selector is static: `at` has nothing to select.
        if self.at is not None:
            raise ValueError(
                f"at= is a time in ms and has no meaning for what={self.what!r}; use "
                f"what='snapshot' or a time-varying fields.* member."
            )

        if self.what == "activation":
            arr = _to_numpy(analysis.activation_time(result.Vm, result.times, **kw))
            return (np.asarray(arr, dtype=np.float64), "activation time (ms)", math.nan,
                    Gradient(cmap="plasma", value_range="auto"))
        if self.what == "apd":
            arr = _to_numpy(analysis.apd_map(result.Vm, result.times, **kw))
            return (np.asarray(arr, dtype=np.float64), "APD90 (ms)", math.nan,
                    Gradient(cmap="viridis", value_range="auto"))
        if self.what == "frequency":
            if kw:
                raise ValueError("what_kwargs= has no effect for what='frequency'")
            arr = _to_numpy(analysis.dominant_frequency_map(result.Vm, result.times))
            return (np.asarray(arr, dtype=np.float64), "dominant frequency (Hz)", math.nan,
                    Gradient(cmap="turbo", value_range="auto"))

        # A fields.* member. Unwrap a VectorField FIRST, then branch on rank — never on the name.
        if kw:
            raise ValueError(f"what_kwargs= has no effect for what={self.what!r}")
        val = getattr(result.fields, self.what)
        raw = val.magnitude if hasattr(val, "magnitude") else val
        arr = np.asarray(_to_numpy(raw), dtype=np.float64)
        default = Gradient(cmap="RdBu_r", value_range="auto99")
        if arr.ndim == 2:
            return arr, self.what, math.nan, default
        if arr.ndim == 3:
            k, t_ms = self._frame_index(result, n=arr.shape[0])
            return arr[k], self.what, t_ms, default
        raise ValueError(                                        # pragma: no cover - defensive
            f"fields.{self.what} has unexpected rank {arr.ndim}; expected 2 or 3 after unwrapping.")

    def _frame_index(self, result, n: Optional[int] = None):
        """(index, time_ms) for `at`, defaulting to the middle frame."""
        times = _to_numpy(result.times).astype(np.float64)
        n = len(times) if n is None else n
        if self.at is None:
            k = n // 2
        else:
            k = int(np.argmin(np.abs(times[:n] - float(self.at))))
        return k, float(times[k]) if k < len(times) else math.nan

    # ---- draw-time helpers ----
    def display_values(self) -> np.ndarray:
        """The single frame with inactive tissue set to NaN. UNtransposed ``(Nx, Ny)``."""
        return self._clip.display_values(0)

    def requires_figure(self) -> bool:
        return self._clip.requires_figure() or self.filled or self.isochrones

    def __repr__(self) -> str:
        Nx, Ny = self._clip.frames.shape[1:]
        overlays = [n for n, on in (("front", self.front is not None),
                                    ("isochrones", bool(self.isochrones)),
                                    ("filled", bool(self.filled))) if on]
        return (f"Image(what={self._selector_name()!r}, grid=({Nx}, {Ny}), "
                f"label={self.value_label!r}, style={self.style!r}, overlays={overlays})")
