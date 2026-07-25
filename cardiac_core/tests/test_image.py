"""Tests for cardiac_core.image — spec-first still figures.

Two module-scoped fixtures, priced deliberately: `wave` is 2.3 s and serves everything except APD,
which is 100 % NaN on a run shorter than one action potential (~230 ms for TTP06 here). Anything
asserting on APD needs `long_wave`, which costs ~44 s — so only APD uses it.
"""

import os
import sys
import types
import warnings

import numpy as np
import pytest
import torch

import cardiac_core as cc
from cardiac_core import Image, ImageInfo, draw
from cardiac_core.image._draw import _UNSET


@pytest.fixture(scope="module")
def wave():
    """A tiny propagating run (0.725 x 0.175 cm, 20 ms). LAT is finite; APD is NOT."""
    g = cc.Grid(30, 8, 0.025)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    sim = cc.monodomain(g, "ttp06", cond,
                        cc.Stim.boundary(g, "left", amplitude=-80.0,
                                         start_time=1.0, duration=2.0))
    return sim.run(t_end=20.0, save_every=1.0)


@pytest.fixture(scope="module")
def long_wave():
    """Long enough for a full action potential, so apd_map is finite. ~44 s — APD only."""
    g = cc.Grid(30, 8, 0.025)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    sim = cc.monodomain(g, "ttp06", cond,
                        cc.Stim.boundary(g, "left", amplitude=-80.0,
                                         start_time=1.0, duration=2.0))
    return sim.run(t_end=400.0, save_every=5.0)


@pytest.fixture(scope="module")
def masked_wave(wave):
    """A synthetic obstacle that stays FINITE in Vm — the LBM case."""
    obst = torch.zeros(30, 8, dtype=torch.bool)
    obst[12:18, 2:6] = True
    V = wave.Vm.clone()
    V[:, obst] = 12.3
    r = cc.SimulationResult(times=wave.times, Vm=V, dx=wave.dx, dy=wave.dy,
                            domain_mask=~obst)
    return r, obst.cpu().numpy()


# --------------------------------------------------------------------------- the contract

def test_default_displays_and_writes_nothing(wave):
    info = wave.image()
    assert info.path is None and info.saved is False
    assert info.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert info.width and info.height
    assert "data:image/png;base64" in info._repr_html_()


def test_default_is_snapshot(wave):
    assert Image(wave).what == "snapshot"
    assert Image(wave)._clip.value_label == "Vm (mV)"


def test_media_keywords_save(wave):
    info = wave.image("test-image", bulk=True)
    assert info.saved and os.path.getsize(info.path) > 0
    assert "_sim_outputs" in info.path and info.path.endswith(".png")


def test_path_is_obeyed_literally(wave, tmp_path):
    p = tmp_path / "exactly-here.png"
    info = draw(Image(wave), path=str(p))
    assert info.path == str(p) and p.exists()


def test_jpeg_extension_is_not_rewritten(wave, tmp_path):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        info = draw(Image(wave), path=str(tmp_path / "a.jpeg"))
    assert info.path.endswith(".jpeg")
    assert not any("backend downgrade" in str(w.message) for w in rec)


@pytest.mark.parametrize("fmt,vector", [("png", False), ("jpg", False), ("webp", False),
                                        ("svg", True), ("pdf", True)])
def test_formats(wave, tmp_path, fmt, vector):
    info = draw(Image(wave), path=str(tmp_path / f"f.{fmt}"))
    assert os.path.getsize(info.path) > 0
    assert (info.width is None) is vector


def test_fspath_raises_when_unsaved(wave):
    with pytest.raises(TypeError, match="path="):
        os.fspath(wave.image())


def test_save_after_the_fact(wave, tmp_path):
    info = wave.image()
    p = info.save(str(tmp_path / "kept.png"))
    assert open(p, "rb").read() == info.read()


def test_read_falls_back_to_the_file(wave, tmp_path):
    info = draw(Image(wave), path=str(tmp_path / "f.png"))
    assert info.data is None and info.read()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- the registry

@pytest.mark.parametrize("what", ["snapshot", "activation", "frequency",
                                  "source_sink", "speed", "velocity", "curvature"])
def test_every_map_selector_renders_at_the_right_rank(wave, what):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # dominant_frequency warns on a short run
        spec = Image(wave, what=what)
        assert spec.display_values().shape == (30, 8)
        assert draw(spec).data


def test_derived_maps_carry_their_own_units_and_range(long_wave):
    apd = Image(long_wave, what="apd")
    assert apd._clip.value_label == "APD90 (ms)"
    assert apd.gradient.value_range == "auto"          # NOT the -90..40 mV scale
    info = draw(apd)
    assert info.vmax > 100.0, "an APD map in ms must not be squeezed onto a mV range"


def test_apd_is_all_nan_on_a_short_run(wave):
    """Guards the fixture split: this is why APD assertions use `long_wave`."""
    from cardiac_core import analysis
    apd = analysis.apd_map(wave.Vm, wave.times)
    assert not torch.isfinite(apd).any()


def test_drawn_colorbar_shows_the_derived_label(long_wave):
    spec = Image(long_wave, what="apd")
    seen = {}
    mod = sys.modules["cardiac_core.image._draw"]
    orig = mod._build_figure

    def spy(clip, cmap, norm, **kw):
        st = orig(clip, cmap, norm, **kw)
        seen["label"] = st.fig.axes[-1].get_ylabel()
        return st

    mod._build_figure = spy
    try:
        draw(spec)
    finally:
        mod._build_figure = orig
    assert seen["label"] == "APD90 (ms)"


def test_explicit_value_label_wins(wave):
    assert Image(wave, value_label="delta CV (cm/s)")._clip.value_label == "delta CV (cm/s)"


# --------------------------------------------------------------------------- the overlay

def _figure_artists(spec):
    """(lat_was_passed, n_images, n_collections) as the real draw() call produced them."""
    seen = {}
    mod = sys.modules["cardiac_core.image._draw"]
    orig = mod._build_figure

    def spy(clip, cmap, norm, **kw):
        st = orig(clip, cmap, norm, **kw)
        seen["v"] = (kw.get("lat") is not None, len(st.ax.images), len(st.ax.collections))
        return st

    mod._build_figure = spy
    try:
        draw(spec)
    finally:
        mod._build_figure = orig
    return seen["v"]


def test_activation_draws_contours(wave):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = _figure_artists(Image(wave, what="activation"))
        msgs = [str(w.message) for w in rec]
    assert got == (True, 1, 1), f"expected imshow + ONE contour set, got {got}"
    assert not any("isochrones need >= 2 frames" in m for m in msgs)


def test_filled_is_contourf_only_and_never_double_draws(wave):
    got = _figure_artists(Image(wave, what="activation", filled=True, contour_levels=15))
    assert got == (False, 0, 1), f"filled must be contourf ONLY (2 == double-draw), got {got}"


def test_filled_suppresses_the_auto_isochrones(wave):
    assert Image(wave, what="activation").isochrones is True
    assert Image(wave, what="activation", filled=True).isochrones is False
    assert Image(wave, what="activation", filled=True, isochrones=True).isochrones is True


def test_isochrones_on_a_snapshot(wave):
    got = _figure_artists(Image(wave, isochrones=True))
    assert got[2] >= 1, "isochrones=True on a snapshot must draw contours"


def test_apd_draws_no_isochrones(long_wave):
    assert _figure_artists(Image(long_wave, what="apd")) == (False, 1, 0)


# --------------------------------------------------------------------------- masking

def test_finite_obstacle_is_masked(masked_wave):
    r, obst = masked_wave
    vals = Image(r).display_values()
    mid = r.Vm[len(r.Vm) // 2].cpu().numpy()
    assert np.isfinite(mid[obst]).all(), "the source obstacle must be FINITE (the LBM case)"
    assert np.isnan(vals[obst]).all(), "masked nodes must be NaN once drawn"
    assert np.isfinite(vals[~obst]).any()


def test_mask_false_disables_masking(masked_wave):
    r, obst = masked_wave
    assert np.isfinite(Image(r, mask=False).display_values()[obst]).all()


def test_overlay_lat_is_masked_too(masked_wave):
    r, obst = masked_wave
    spec = Image(r, what="activation")
    assert np.isnan(spec._lat[obst]).all(), "the isochrone LAT must be masked like the map"


# --------------------------------------------------------------------------- bare

def test_bare_still_is_upscaled_without_padding(wave):
    info = draw(Image(wave, style="bare"))
    assert max(info.width, info.height) >= 512
    assert abs(info.width / info.height - 30 / 8) < 0.05, "aspect must be preserved, no letterbox"


def test_bare_still_carries_a_burned_stamp(wave, tmp_path):
    from PIL import Image as PILImage
    a = np.asarray(PILImage.open(draw(Image(wave, style="bare"),
                                      path=str(tmp_path / "a.png")).path).convert("L"))
    b = np.asarray(PILImage.open(draw(Image(wave, style="bare"), show_time=False,
                                      path=str(tmp_path / "b.png")).path).convert("L"))
    assert not np.array_equal(a[:40, :240], b[:40, :240]), "the time stamp must be burned in"


def test_bare_rejects_svg(wave, tmp_path):
    with pytest.raises(ValueError, match="annotated"):
        draw(Image(wave, style="bare"), path=str(tmp_path / "x.svg"))


# --------------------------------------------------------------------------- guards

@pytest.mark.parametrize("kw,needle", [
    ({"what": "nope"}, "source_sink"),
    ({"what": "mask"}, "domain gate"),
    ({"what": "trace"}, "Trace"),
    ({"what": "apd", "at": 5.0}, "snapshot"),
    ({"what": "snapshot", "what_kwargs": {"x": 1}}, "what_kwargs"),
    ({"what": "apd", "field": "phi_e"}, "snapshot"),
    ({"style": "bare", "filled": True}, "annotated"),
    ({"style": "bare", "units": "cm"}, "annotated"),
])
def test_image_guards_raise(wave, kw, needle):
    with pytest.raises(ValueError, match=needle):
        Image(wave, **kw)


def test_rest_zoom_on_a_derived_map_raises(long_wave):
    with pytest.raises(ValueError, match="auto"):
        Image(long_wave, what="apd", gradient=cc.Gradient.zoom())


def test_rank3_array_and_single_cell_are_rejected(wave):
    with pytest.raises(ValueError, match="preview"):
        Image(np.zeros((3, 4, 5)))
    with pytest.raises(ValueError, match="trace"):
        Image(cc.single_cell("ttp06", n_beats=1, bcl=50.0))


@pytest.mark.parametrize("kw,needle", [
    ({"slug": "x", "bulk": True, "format": "pdf"}, "path="),
    ({"frame": 1}, "at="),
    ({"labels": ["a"]}, "multi-panel"),
    ({"rows": 2}, "multi-panel"),
    ({"resolution": "auto"}, "figsize"),
])
def test_draw_guards_raise(wave, kw, needle):
    with pytest.raises(ValueError, match=needle):
        draw(Image(wave), **kw)


def test_format_path_disagreement_raises(wave, tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        draw(Image(wave), path=str(tmp_path / "a.pdf"), format="png")


@pytest.mark.parametrize("kw", [{"tight": True}, {"transparent": True}])
def test_bare_rejects_figure_only_draw_knobs(wave, kw):
    with pytest.raises(ValueError, match="bare"):
        draw(Image(wave, style="bare"), **kw)


@pytest.mark.parametrize("ikw", [{"label": "x"}, {"front": -40.0}])
def test_bare_capability_errors_speak_image_vocabulary(wave, ikw):
    """The audience never typed `Video`; the message must not send them there."""
    with pytest.raises(ValueError) as exc:
        draw(Image(wave, style="bare", **ikw))
    assert 'style="annotated"' in str(exc.value)
    assert "Video" not in str(exc.value)


def test_multi_panel_is_deferred(wave):
    with pytest.raises(NotImplementedError, match="Phase 3"):
        draw([Image(wave), Image(wave)])


# --------------------------------------------------------------------------- wiring

def test_image_keys_covers_every_spec_field():
    """A new Image field must not silently become a TypeError from r.image()."""
    from cardiac_core.run import _IMAGE_KEYS
    fields = {f for f in Image.__dataclass_fields__ if f != "data"}
    assert not (fields - set(_IMAGE_KEYS)), sorted(fields - set(_IMAGE_KEYS))


def test_public_exports_are_stable_and_callable():
    for name in ("Image", "draw", "ImageInfo"):
        first = getattr(cc, name)
        assert not isinstance(first, types.ModuleType), f"cc.{name} resolved to a module"
        assert getattr(cc, name) is first, f"cc.{name} changed identity on repeated access"
    assert callable(cc.draw)


def test_import_chain_stays_free_of_encoder_backends():
    assert "imageio" not in sys.modules and "cv2" not in sys.modules


def test_video_behaviour_is_unchanged_by_the_new_params(wave):
    """The three additive render.py params must default to the historical behaviour."""
    from cardiac_core.video.render import _build_figure
    from cardiac_core import Video
    clip = Video.annotated((wave.times.cpu().numpy(), wave.Vm.cpu().numpy()))
    cmap, norm, _, _ = clip.gradient.resolve(clip.masked_iter([0]), field=clip.field)
    st = _build_figure(clip, cmap, norm, colorbar_on=True, title=None, figsize=None,
                       dpi=100, units=None, idx=[0])
    try:
        assert len(st.ax.images) == 1 and len(st.ax.collections) == 0
        assert hasattr(st.im, "set_data")
    finally:
        import matplotlib.pyplot as plt
        plt.close(st.fig)


# =========================================================================== Phase 2: Trace

@pytest.fixture(scope="module")
def multibeat():
    """A SYNTHETIC 4-beat result. `long_wave` fires one stimulus, so restitution is empty on it;
    a real paced run would cost ~98 s, and restitution needs only `times` + `Vm`."""
    V_REST, V_PEAK, BCL, DT = -85.0, 20.0, 400.0, 1.0
    apds = (225.0, 245.0, 215.0, 235.0)     # VARY per beat, or the curve is identical points
    n = int(len(apds) * BCL / DT)
    trace = torch.full((n,), V_REST, dtype=torch.float64)
    for k, apd in enumerate(apds):
        s, d = int(k * BCL / DT), int(apd / DT)
        trace[s:s + 2] = V_PEAK
        trace[s + 2:s + d] = torch.linspace(V_PEAK, V_REST, d - 2)
    return cc.SimulationResult(times=torch.arange(n, dtype=torch.float64) * DT,
                               Vm=trace.view(-1, 1, 1).expand(n, 30, 8).contiguous())


def test_trace_named_series_and_legend(wave):
    info = wave.trace(at={"edge": (0, 4), "centre": (20, 4)}, hline=(-40.0, "threshold"))
    assert info.path is None and info.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert info.vmin is None and info.vmax is None, "a trace has no colour range"
    spec = cc.Trace(wave, at={"edge": (0, 4), "centre": (20, 4)})
    assert [lab for lab, _, _ in spec.series] == ["edge", "centre"]
    assert spec.legend is True


def test_trace_defaults_to_the_grid_centre(wave):
    spec = cc.Trace(wave)
    assert len(spec.series) == 1 and spec.legend is False
    assert spec.xlabel == "time (ms)" and spec.ylabel == "Vm (mV)"


def test_single_cell_trace(wave):
    info = cc.single_cell("ttp06", n_beats=1, bcl=200.0).trace()
    assert info.data and not info.saved


def test_single_cell_rejects_at():
    with pytest.raises(ValueError, match="no grid"):
        cc.Trace(cc.single_cell("ttp06", n_beats=1, bcl=100.0), at=(0, 0))


def test_restitution_is_marker_only_and_non_degenerate(multibeat):
    from cardiac_core import analysis
    DI, APD = analysis.restitution_curve(multibeat.Vm, multibeat.times, 20, 4)
    assert DI.numel() >= 2 and APD.unique().numel() >= 2, "fixture must not be degenerate"
    spec = cc.Trace(multibeat, what="restitution", at=(20, 4))
    assert spec.marker == "o" and spec.linestyle == "none"
    assert multibeat.trace(what="restitution", at=(20, 4)).data


def test_restitution_warns_on_a_single_beat_run(wave):
    with pytest.warns(UserWarning, match="multi-beat"):
        wave.trace(what="restitution", at=(20, 4))


def test_apd_per_beat(multibeat):
    assert multibeat.trace(what="apd_per_beat", at=(20, 4)).data


def test_reference_lines(wave):
    spec = cc.Trace(wave, hline=(-40.0, "threshold"), vline=[1.0, 3.0])
    assert spec.hlines == [(-40.0, "threshold")]
    assert spec.vlines == [(1.0, None), (3.0, None)]


def test_raw_xy_and_dict_data(wave):
    xy = cc.Trace((np.arange(5.0), np.arange(5.0) ** 2))
    assert len(xy.series) == 1 and xy.series[0][0] is None
    d = cc.Trace({"a": (np.arange(3.0), np.zeros(3)), "b": (np.arange(3.0), np.ones(3))})
    assert [lab for lab, _, _ in d.series] == ["a", "b"]
    assert draw(d).data


def test_trace_rejects_map_knobs(wave):
    with pytest.raises(ValueError, match="r.image"):
        wave.trace(gradient=cc.Gradient.zoom())
    with pytest.raises(ValueError, match="Trace has no image"):
        draw(cc.Trace(wave), colorbar=True)


def test_out_of_range_node_raises_with_the_valid_range(wave):
    with pytest.raises(ValueError, match=r"0\.\.29"):
        wave.trace(at=(999, 0))


def test_trace_keys_covers_every_spec_field():
    from cardiac_core.run import _TRACE_KEYS
    fields = {f for f in cc.Trace.__dataclass_fields__ if f != "data"}
    assert not (fields - set(_TRACE_KEYS)), sorted(fields - set(_TRACE_KEYS))


def test_trace_export_is_stable():
    first = cc.Trace
    assert not isinstance(first, types.ModuleType) and cc.Trace is first
