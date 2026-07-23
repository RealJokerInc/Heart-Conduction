"""
cardiac_core.video — spec-first video rendering.

A spec object holds the description, a render function turns it into frames, and the output lands
at a convention-compliant ``media/`` path::

    r = sim.run(t_end=200.0, save_every=1.0)
    r.video("spiral-wave")                    # -> media/lab/_sim_outputs/videos/{date}/spiral-wave_01.mp4

The zero-argument default is the video you usually want: the raw voltage field, full-frame, no
labels anywhere, in the standard preset, at 1080p. Everything else is opt-in::

    from cardiac_core import Video, Gradient, render

    render(Video.annotated(r, gradient=Gradient.zoom(span=8.0), isochrones=True),
           "wall-artifact", question="my_study", bulk=False)

Colour is a reusable object (:class:`Gradient`) because the value range is a scientific choice,
not decoration: a few-mV feature is invisible on the -90..40 mV scale and obvious on a zoom
window. Sharing one ``Gradient`` across panels is what makes a comparison comparable.
"""

from .encoders import VideoInfo
from .gradient import Gradient
from .clip import Video
from .render import render, render_video, preview_frame

__all__ = ["Video", "Gradient", "render", "render_video", "VideoInfo", "preview_frame"]
