"""
cardiac_core.video — spec-first video rendering.

A spec object holds the description and a render function turns it into frames. **Rendering
displays; naming a destination saves** — the same contract matplotlib uses for figures::

    r = sim.run(t_end=200.0, save_every=1.0)
    r.video()                                 # plays inline in Jupyter/Colab, writes no file
    r.video(path="spiral-wave.mp4")           # writes ./spiral-wave.mp4
    r.video("spiral-wave", bulk=True)         # media/lab/_sim_outputs/videos/{date}/…

Inline playback embeds the encoded bytes in the notebook as a data URI, so it needs no file
server and survives an ephemeral runtime such as Colab.

The zero-argument default is the video you usually want: the raw voltage field, full-frame, no
labels anywhere, in the standard preset, at 1080p. Everything else is opt-in::

    from cardiac_core import Video, Gradient, render

    render(Video.annotated(r, gradient=Gradient.zoom(span=8.0), isochrones=True),
           "wall-artifact", question="my_study", bulk=False)

Colour is a reusable object (:class:`Gradient`) because the value range is a scientific choice,
not decoration: a few-mV feature is invisible on the -90..40 mV scale and obvious on a zoom
window. Sharing one ``Gradient`` across panels is what makes a comparison comparable.
"""

from .encoders import ImagePath, VideoInfo
from .gradient import Gradient
from .clip import Video
from .render import render, render_video, preview_frame

__all__ = ["Video", "Gradient", "render", "render_video", "VideoInfo", "ImagePath",
           "preview_frame"]
