"""Public ``Stim`` object — a masked, timed stimulus (current injection OR voltage clamp).

Eager classmethod factories (``Stim.boundary``/``point``/``center``/``from_region``) build a concrete
``(Nx, Ny)`` bool mask from a ``Grid`` via the shared ``geometry.py`` builders (ONE mask system — no
inline edge rule). Mode is inferred: ``clamp=<mV>`` ⇒ voltage clamp, else current injection
(``amplitude`` µA/µF). A CURRENT Stim lowers to the normalized stimulus dict (``to_dict``) that the
engines/``.npz`` already consume; a CLAMP Stim is routed to ``clamp_voltage`` at the factory (it is
NOT serialized to ``data.stimuli``). NOT subclasses — one ``Stim`` type.

Imports only ``geometry`` (numpy) + numpy, so ``api`` can import ``Stim`` locally without a cycle.
"""

import numpy as np

from .. import geometry


def _as_bool_mask(mask) -> np.ndarray:
    """torch|numpy array → an ``(Nx, Ny)`` numpy bool mask (shape-validated to 2-D)."""
    if hasattr(mask, 'detach'):          # torch tensor
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"stimulus mask must be 2-D (Nx, Ny); got shape {m.shape}")
    return m


def _resolve_where(grid, where, width=None, radius=None) -> np.ndarray:
    """Build the ``(Nx, Ny)`` bool mask for a location spec, delegating to the ``geometry.py`` builders.

    ``where`` ∈ ``"left"/"right"/"top"/"bottom"`` (a boundary strip, thickness ``width``) | ``"center"``
    (a blob at the domain centre) | an ``(x, y)`` cm point | an explicit ``(Nx, Ny)`` mask | a
    ``callable(x, y) -> mask``. ``width``/``radius`` default to a thin strip (``~2·dx``).
    """
    Nx, Ny, dx, dy = grid.Nx, grid.Ny, grid.dx, grid.dy
    w = 2.0 * dx if width is None else width
    r = 2.0 * dx if radius is None else radius
    if isinstance(where, str):
        s = where.lower()
        if s == 'left':   return geometry.left_edge_mask(Nx, Ny, dx, w)
        if s == 'right':  return geometry.right_edge_mask(Nx, Ny, dx, w)
        if s == 'top':    return geometry.top_edge_mask(Nx, Ny, dx, w, dy)
        if s == 'bottom': return geometry.bottom_edge_mask(Nx, Ny, dx, w, dy)
        if s == 'center': return geometry.circle_mask(Nx, Ny, dx, (grid.Lx / 2.0, grid.Ly / 2.0), r, dy)
        raise ValueError(
            f"unknown stimulus location {where!r}; expected 'left'/'right'/'top'/'bottom'/'center', "
            "an (x, y) point, an (Nx, Ny) mask, or a callable(x, y)")
    if callable(where):
        return _as_bool_mask(where(*grid.coordinates))
    if isinstance(where, tuple) and len(where) == 2 and all(isinstance(v, (int, float)) for v in where):
        return geometry.circle_mask(Nx, Ny, dx, where, r, dy)     # an (x, y) cm point → a small blob
    return _as_bool_mask(where)                                    # an explicit (Nx, Ny) mask


class Stim:
    """A masked, timed stimulus. Mode inferred from ``clamp``: ``clamp=<mV>`` ⇒ voltage clamp, else
    current injection (``amplitude`` µA/µF, default −52; negative = depolarizing)."""

    def __init__(self, mask, *, amplitude=None, clamp=None, start_time=0.0, duration=2.0,
                 bcl=0.0, num_pulses=1, label="stim"):
        if clamp is not None and amplitude is not None:
            raise ValueError("pass amplitude (current) OR clamp (voltage), not both — the mode is inferred")
        if clamp is not None and (bcl > 0 or num_pulses > 1):
            raise ValueError("periodic pacing (bcl/num_pulses) is not supported for a clamp Stim — the clamp "
                             "mechanism holds ONE window; for repeated clamps pass one clamp Stim per window")
        self.mask = _as_bool_mask(mask)
        self.mode = "clamp" if clamp is not None else "current"
        self.amplitude = -52.0 if amplitude is None else float(amplitude)
        self.clamp = None if clamp is None else float(clamp)
        self.start_time = float(start_time)
        self.duration = float(duration)
        self.bcl = float(bcl)
        self.num_pulses = int(num_pulses)
        self.label = label

    # --- eager classmethod factories (grid + a location; each a FULL constructor via **kw) ---
    @classmethod
    def boundary(cls, grid, side, *, width=None, **kw) -> "Stim":
        """A boundary strip: ``side`` ∈ ``"left"/"right"/"top"/"bottom"`` (the sole edge API)."""
        return cls(_resolve_where(grid, side, width=width), **kw)

    @classmethod
    def point(cls, grid, center, *, radius=None, **kw) -> "Stim":
        """A blob at an ``(x, y)`` cm point (``radius`` default ~2·dx)."""
        return cls(_resolve_where(grid, tuple(center), radius=radius), **kw)

    @classmethod
    def center(cls, grid, *, radius=None, **kw) -> "Stim":
        """A blob at the domain centre."""
        return cls(_resolve_where(grid, "center", radius=radius), **kw)

    @classmethod
    def from_region(cls, grid, region, **kw) -> "Stim":
        """From any ``callable(x, y) -> bool mask`` (or an explicit mask), resolved against the grid."""
        return cls(_resolve_where(grid, region), **kw)

    # --- lowering / helpers ---
    def to_dict(self) -> dict:
        """CURRENT-mode lowering → the 7-key normalized stimulus dict (raises on a clamp Stim)."""
        if self.mode == "clamp":
            raise ValueError("to_dict() is for current-mode Stims; a clamp Stim routes to clamp_voltage")
        return {'mask': self.mask.astype(bool), 'label': self.label, 'amplitude': self.amplitude,
                'duration': self.duration, 'start_time': self.start_time,
                'bcl': self.bcl, 'num_pulses': self.num_pulses}

    @classmethod
    def from_dict(cls, d) -> "Stim":
        """Inverse of :meth:`to_dict` (current-mode)."""
        return cls(d['mask'], amplitude=d.get('amplitude', -52.0), start_time=d.get('start_time', 0.0),
                   duration=d.get('duration', 2.0), bcl=d.get('bcl', 0.0),
                   num_pulses=d.get('num_pulses', 1), label=d.get('label', 'stim'))

    def times(self) -> list:
        """Pulse times: ``[start_time + k·bcl]`` for a train, or ``[start_time]``."""
        if self.bcl > 0 and self.num_pulses > 1:
            return [self.start_time + k * self.bcl for k in range(self.num_pulses)]
        return [self.start_time]

    def n_nodes(self) -> int:
        return int(self.mask.sum())

    def __repr__(self) -> str:
        lvl = f"clamp={self.clamp}mV" if self.mode == "clamp" else f"amp={self.amplitude}"
        return f"Stim(mode={self.mode}, {lvl}, n_nodes={self.n_nodes()}, t0={self.start_time})"
