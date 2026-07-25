"""``cardiac_core.image`` — spec-first still figures.

A spec object holds the description and a verb turns it into bytes. **Drawing displays; naming a
destination saves** — the same contract as :mod:`cardiac_core.video`, and the same one matplotlib
uses for a figure::

    r = sim.run(t_end=200.0, save_every=1.0)
    r.image()                                 # displays inline in Jupyter/Colab, writes no file
    r.image(path="wave.png")                  # writes ./wave.png
    r.image("wave", bulk=True)                # media/lab/_sim_outputs/images/{date}/…

Two spec types, because a map and a series do not share a description:

* :class:`Image` — a spatial map (a snapshot, an activation/APD/frequency map, any ``fields.*``).
* :class:`Trace` — a series (an action potential, a restitution curve, per-beat APD).

Colour reuses :class:`cardiac_core.video.Gradient` unchanged: the value *range* is a scientific
choice, and sharing one ``Gradient`` across panels is what makes a comparison comparable.

The submodule holding :func:`draw` is ``_draw`` rather than ``draw`` on purpose: a submodule whose
name matches a public export shadows it under PEP 562, which would make ``cardiac_core.draw`` a
module instead of the function.
"""

import importlib

__all__ = ["Image", "Trace", "draw", "ImageInfo"]

# Public name -> submodule that defines it. Resolved lazily so that importing this package does not
# drag in matplotlib and the vendored solver stack before anything is actually drawn.
_LAZY = {
    "ImageInfo": "info",
    "Image": "panel",
    "Trace": "panel",
    "draw": "_draw",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    obj = getattr(importlib.import_module(f".{mod}", __name__), name)
    # Cache into globals() so repeated access is cheap AND stable — see the module docstring.
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))
