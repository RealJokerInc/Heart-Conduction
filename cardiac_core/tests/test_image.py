"""Tests for cardiac_core.image — spec-first still figures.

Two module-scoped fixtures, priced deliberately: `wave` is 2.3 s and serves everything except APD,
which is 100 % NaN on a run shorter than one action potential (~230 ms for TTP06 here). Anything
asserting on APD needs `long_wave`, which costs ~44 s — so only APD uses it.
"""

import os
import subprocess
import tempfile
import sys
import types
import warnings

import numpy as np
import pytest
import torch

import cardiac_core as cc
from cardiac_core import Image, ImageInfo, Trace, draw
from cardiac_core.image._draw import _UNSET


@pytest.fixture(autouse=True)
def _isolate_show_cache(tmp_path, monkeypatch):
    """Point .show()'s materialise-cache at a per-test tmp dir, so terminal-branch show tests never
    write into the developer's real ~/.cache/cardiac_core/show."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "_show_cache"))


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

def _axes_at_savefig(draw_call) -> int:
    """How many axes the figure carried when it was written.

    N panels sharing ONE colorbar gives N+1; a per-panel-colorbar regression gives 2N. Counting
    axes is robust where a pixel heuristic is not.
    """
    import matplotlib.figure as mfig
    seen = {}
    real = mfig.Figure.savefig

    def spy(self, *a, **k):
        seen["n"] = len(self.axes)
        return real(self, *a, **k)

    mfig.Figure.savefig = spy
    try:
        draw_call()
    finally:
        mfig.Figure.savefig = real
    return seen["n"]


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


def test_empty_panel_list_is_rejected(wave):
    with pytest.raises(ValueError, match="at least one"):
        draw([])


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
    """Drawing a still must not drag in a VIDEO encoder backend.

    Checked in a SUBPROCESS: asserting on this process's ``sys.modules`` only held because
    ``test_image.py`` happens to sort before ``test_video.py``, which imports imageio via
    ``_Writer``. Any ``-k`` subset, ``-p xdist`` split or random-order plugin flipped it.
    """
    import subprocess
    probe = (
        "import sys; import cardiac_core as cc;"
        "g = cc.Grid(8, 4, 0.025);"
        "cond = cc.ConductivityConfig.isotropic(1.0);"
        "sim = cc.monodomain(g, 'ttp06', cond, cc.Stim.boundary(g, 'left'));"
        "r = sim.run(t_end=1.0, save_every=0.5);"
        "cc.draw(cc.Image(r));"
        "leaked = [m for m in ('imageio', 'cv2') if m in sys.modules];"
        "print('LEAKED:' + ','.join(leaked))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "LEAKED:\n" in out.stdout or out.stdout.strip().endswith("LEAKED:"), \
        f"a still-image draw imported a video encoder backend: {out.stdout.strip()}"


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


# =========================================================================== Phase 3: layout

def _panel_artists(specs, **kw):
    """Per-panel (n_images, n_collections), read AFTER draw() — artists survive plt.close()."""
    axes_seen = []
    mod = sys.modules["cardiac_core.image._draw"]
    orig = mod._setup_panel

    def spy(clip, ax, cmap, norm, **kwargs):
        axes_seen.append(ax)
        return orig(clip, ax, cmap, norm, **kwargs)

    mod._setup_panel = spy
    try:
        info = draw(specs, **kw)
    finally:
        mod._setup_panel = orig
    return info, [(len(a.images), len(a.collections)) for a in axes_seen]


def test_two_panels_share_one_colorbar(wave):
    info, _ = _panel_artists([Image(wave, label="control"), Image(wave, label="drug")])
    assert info.n_panels == 2 and info.data
    # The NAME of this test is the claim, so assert it: 2 panels + ONE shared colorbar = 3 axes.
    # Previously this only checked n_panels, which a per-panel-colorbar regression also satisfies.
    n_axes = _axes_at_savefig(
        lambda: draw([Image(wave, label="control"), Image(wave, label="drug")]))
    assert n_axes == 3, f"expected 2 panels + 1 shared colorbar = 3 axes, got {n_axes}"


def test_both_activation_panels_draw_contours(wave):
    info, counts = _panel_artists([Image(wave, what="activation"),
                                   Image(wave, what="activation")])
    assert counts == [(1, 1), (1, 1)], f"expected contours on BOTH panels, got {counts}"


def test_front_survives_the_layout_path(wave):
    _, counts = _panel_artists([Image(wave, front=-40.0), Image(wave)])
    assert counts[0][1] > counts[1][1], f"front= was dropped on the layout path: {counts}"


def test_mixed_map_and_trace_lays_out(wave):
    info = draw([Image(wave), cc.Trace(wave)])
    assert info.n_panels == 2 and info.data


def test_different_quantities_are_not_pooled(wave, long_wave):
    """An APD map and a voltage map both report field='Vm'; only value_label separates them."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")     # defeat the per-location __warningregistry__
        draw([Image(wave), Image(long_wave, what="apd")])
    assert any("NOT directly comparable" in str(w.message) for w in rec), [
        str(w.message) for w in rec]


def test_labels_do_not_mutate_the_caller(wave):
    a = Image(wave, label="original")
    draw([a, Image(wave)], labels=["override", "b"])
    assert a.label == "original"


def test_video_in_a_list_is_rejected(wave):
    from cardiac_core import Video
    with pytest.raises(ValueError, match="render"):
        draw([Video(wave), Video(wave)])


def test_map_panels_must_share_a_grid(wave):
    g = cc.Grid(20, 6, 0.025)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    other = cc.monodomain(g, "ttp06", cond,
                          cc.Stim.boundary(g, "left", amplitude=-80.0,
                                           start_time=1.0, duration=2.0)).run(t_end=5.0)
    with pytest.raises(ValueError, match="share a grid"):
        draw([Image(wave), Image(other)])


# --------------------------------------------------------------------------- delegations

def test_preview_is_pixel_identical_on_both_paths(wave, tmp_path):
    """`preview_frame` now routes through draw(); its output must not move a pixel."""
    from PIL import Image as PILImage
    from cardiac_core import Video
    bare = Video.bare(wave).preview(path=str(tmp_path / "b.png"))
    ann = Video.annotated(wave).preview(path=str(tmp_path / "a.png"))
    assert PILImage.open(bare).size == (30, 8), "a bare preview is the raw grid, unscaled"
    w, h = PILImage.open(ann).size
    assert w > 400 and h > 200, (w, h)          # the historical dpi-100 figure, not dpi-150


def test_annotated_preview_does_not_raise(wave):
    """R3 C-1: the delegation passes resolution=None on an ANNOTATED clip."""
    from cardiac_core import Video
    assert Video.annotated(wave).preview()


def test_preview_still_returns_an_image_path(wave, tmp_path):
    from cardiac_core import Video
    p = Video(wave).preview(t_ms=5.0, path=str(tmp_path / "f.png"))
    assert isinstance(p, str) and p.endswith(".png") and os.path.exists(p)


def test_viz_stills_keep_their_titles_and_shape(wave):
    """The delegated stills preserve composition; size may move <=15% (suptitle+tight_layout)."""
    from PIL import Image as PILImage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # a 20 ms run has no APD -> empty-range warning
        p = cc.apd_map_figure(wave, "t-apd", bulk=True)
    w, h = PILImage.open(p).size
    assert 500 < w < 900 and 250 < h < 450, (w, h)
    q = cc.activation_isochrones(wave, "t-iso", bulk=True)
    assert os.path.getsize(q) > 0 and q.endswith(".png")


def test_isochrones_delegation_is_filled_without_line_overlay(wave):
    """viz.activation_isochrones draws contourf ONLY — line contours on top would double-draw."""
    spec = Image(wave, what="activation", filled=True, isochrones=False)
    assert spec.isochrones is False and spec._lat is not None
    assert _figure_artists(spec) == (False, 0, 1)


# ---------------- destination contract (shared with the video layer, previously untested here)

def _media_files():
    root = os.environ.get("CARDIAC_MEDIA_ROOT")
    assert root and os.path.isdir(root), \
        "conftest's media-root fixture is not active; this check would be vacuous"
    return frozenset(os.path.join(dp, f)
                     for dp, _d, fs in os.walk(root) for f in fs)


def test_unsaved_draw_writes_nothing_anywhere(wave, tmp_path, monkeypatch):
    """The video-side version of this test was vacuous until round 1: conftest points
    CARDIAC_MEDIA_ROOT elsewhere, so asserting only on cwd would miss a media_path regression."""
    monkeypatch.chdir(tmp_path)
    before_media, before_tmp = _media_files(), set(os.listdir(tempfile.gettempdir()))
    info = draw(Image(wave))
    assert info.path is None and info.saved is False and info.data
    assert list(tmp_path.iterdir()) == []
    assert _media_files() == before_media
    assert not (set(os.listdir(tempfile.gettempdir())) - before_tmp), "leaked a temp file"


def test_draw_failure_never_deletes_a_preexisting_file(wave, tmp_path):
    """CRITICAL regression, image side: the cleanup guard used to delete out_path unconditionally,
    which for a path= draw is the CALLER'S file.

    The failure must be injected INSIDE the guard. The original version used the bare+SVG
    rejection, but that was hoisted above `_resolve_destination`, so it stopped reaching
    discard_partial and passed for the wrong reason.
    """
    victim = tmp_path / "thesis_figure.png"
    victim.write_bytes(b"\x89PNG\r\n\x1a\n" + b"five years of work" * 10)
    original = victim.read_bytes()

    import sys
    dm = sys.modules["cardiac_core.image._draw"]
    real = dm._produce_figure
    try:
        dm._produce_figure = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.warns(UserWarning, match="untouched"):
            with pytest.raises(RuntimeError):
                draw(Image(wave), path=str(victim))
    finally:
        dm._produce_figure = real

    assert victim.exists(), "a failed draw DELETED the caller's file"
    assert victim.read_bytes() == original, "a failed draw MODIFIED the caller's file"


def test_bare_svg_is_rejected_before_a_destination_is_acquired(wave, tmp_path):
    """The hoist that made the test above misleading is itself worth pinning."""
    victim = tmp_path / "keep.svg"
    victim.write_text("<svg>original</svg>")
    with pytest.raises(ValueError, match="cannot produce SVG"):
        draw(Image(wave, style="bare"), path=str(victim))
    assert victim.read_text() == "<svg>original</svg>"


def test_draw_path_is_a_directory_raises(wave, tmp_path):
    with pytest.raises(IsADirectoryError):
        draw(Image(wave), path=str(tmp_path))


def test_draw_path_plus_convention_warns(wave, tmp_path):
    with pytest.warns(UserWarning, match="path= wins"):
        draw(Image(wave), "x", path=str(tmp_path / "a.png"), bulk=True)


def test_imageinfo_save_marks_itself_saved(wave, tmp_path):
    """Round-1 fixed this on VideoInfo but not here; sim-media points agents at r.image()."""
    info = draw(Image(wave))
    assert info.saved is False
    dest = info.save(tmp_path / "kept.png")
    assert info.saved is True and info.path == str(dest) and os.fspath(info) == str(dest)
    assert str(info) == str(dest), "str() must give the path, as VideoInfo does"
    assert info.data is None


def test_imageinfo_repr_does_not_dump_the_payload(wave):
    info = draw(Image(wave))
    assert len(repr(info)) < 300 and "data=" not in repr(info)


# ---------------- audit round 3: gaps the video layer covered but this one did not

def test_draw_failure_warns_about_the_untouched_file(wave, tmp_path):
    """Surviving is not enough — a silent no-op would also 'survive'. The caller must be told."""
    victim = tmp_path / "existing.png"
    victim.write_bytes(b"\x89PNG\r\n\x1a\n-original")
    original = victim.read_bytes()
    import sys
    dm = sys.modules["cardiac_core.image._draw"]
    real = dm._produce_figure

    def boom(*a, **k):
        raise RuntimeError("simulated failure")

    try:
        dm._produce_figure = boom
        with pytest.warns(UserWarning, match="already existed"):
            with pytest.raises(RuntimeError):
                draw(Image(wave), path=str(victim))
    finally:
        dm._produce_figure = real
    assert victim.read_bytes() == original


def test_trace_failure_cleans_up_only_what_it_created(wave, tmp_path):
    """The _draw_trace guard must remove a genuinely PARTIAL file it created.

    The failure has to be in the WRITE. An earlier version injected it into the post-write
    measurement instead, which meant the render had succeeded — so the test asserted that a
    complete, valid figure got deleted, enshrining data loss rather than guarding against it.
    """
    dest = tmp_path / "ours.png"
    import matplotlib.figure as mfig
    real = mfig.Figure.savefig

    def boom(self, fname, *a, **k):
        with open(fname, "wb") as fh:          # a genuine half-written output
            fh.write(b"\x89PNG\r\n\x1a\n-truncated")
        raise RuntimeError("simulated write failure")

    try:
        mfig.Figure.savefig = boom
        with pytest.raises(RuntimeError):
            draw(Trace(wave, at=(5, 2)), path=str(dest))
    finally:
        mfig.Figure.savefig = real
    assert not dest.exists(), "a partial file we created must be removed"


def test_a_probe_failure_never_destroys_a_good_figure(wave, tmp_path):
    """The measurement runs inside the cleanup guard, so an unreadable-probe error used to send a
    SUCCESSFULLY written figure to discard_partial and delete it."""
    dest = tmp_path / "good.png"
    from PIL import Image as PILImage
    real = PILImage.open

    def boom(*a, **k):
        raise OSError("cannot identify image file")

    try:
        PILImage.open = boom
        info = draw(Image(wave), path=str(dest))
    finally:
        PILImage.open = real

    assert dest.exists() and dest.stat().st_size > 0, "a good figure was deleted by a probe failure"
    assert info.saved and info.width is None, "dimensions are informational; the bytes are not"


def test_panels_with_a_path_writes_exactly_there(wave, tmp_path):
    """Only the unsaved multi-panel form was covered."""
    dest = tmp_path / "panels.png"
    info = draw([Image(wave, label="a"), Image(wave, label="b")], path=str(dest))
    assert info.path == str(dest) and info.saved and info.n_panels == 2
    assert os.path.exists(dest) and os.path.getsize(dest) > 0


def test_imageinfo_size_cap_reported_without_reading(wave, monkeypatch):
    """Mirror of the video-side cap test, including that the payload is NOT read first."""
    from cardiac_core.image import info as info_mod
    got = draw(Image(wave))
    monkeypatch.setattr(info_mod, "_MAX_INLINE_BYTES", 8)

    def explode():
        raise AssertionError("_repr_html_ read the payload before checking the size cap")

    monkeypatch.setattr(got, "read", explode)
    assert "too large to display inline" in got._repr_html_()


def test_imageinfo_str_unsaved_branch(wave):
    got = draw(Image(wave))
    assert "not saved" in str(got) and got.path is None


def test_imageinfo_repr_html_degrades_when_the_file_is_gone(wave, tmp_path):
    got = draw(Image(wave), path=str(tmp_path / "gone.png"))
    os.remove(got.path)
    assert "unavailable" in got._repr_html_()


def test_inline_caps_agree():
    """One policy, two modules. They are separate constants on purpose (importing the video
    package into image/info.py would force matplotlib's Agg backend process-wide), so pin them."""
    from cardiac_core.image import info as info_mod
    from cardiac_core.video import encoders as enc_mod
    assert info_mod._MAX_INLINE_BYTES == enc_mod.INLINE_MAX_BYTES


def test_importing_imageinfo_does_not_pull_in_matplotlib():
    """cardiac_core/image/__init__.py resolves lazily so a type-only import stays light."""
    import subprocess
    probe = (
        "import sys; import cardiac_core as cc; cc.ImageInfo;"
        "print('MPL:' + str('matplotlib.pyplot' in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "MPL:False" in out.stdout, "touching cc.ImageInfo imported matplotlib.pyplot"


def test_panels_failure_cleans_up_only_what_it_created(wave, tmp_path):
    """_draw_panels' cleanup guard had no test entering its body — the same gap that preceded
    two data-loss defects on the single-panel paths."""
    dest = tmp_path / "panels.png"
    import matplotlib.figure as mfig
    real = mfig.Figure.savefig

    def boom(self, fname, *a, **k):
        with open(fname, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n-truncated")
        raise RuntimeError("simulated write failure")

    try:
        mfig.Figure.savefig = boom
        with pytest.raises(RuntimeError):
            draw([Image(wave, label="a"), Image(wave, label="b")], path=str(dest))
    finally:
        mfig.Figure.savefig = real
    assert not dest.exists(), "the panel guard left a partial file behind"


def test_panels_failure_never_deletes_a_preexisting_file(wave, tmp_path):
    victim = tmp_path / "keep.png"
    victim.write_bytes(b"\x89PNG\r\n\x1a\n-original")
    original = victim.read_bytes()
    import matplotlib.figure as mfig
    real = mfig.Figure.savefig

    def boom(self, fname, *a, **k):
        raise RuntimeError("simulated failure inside savefig")

    try:
        mfig.Figure.savefig = boom
        # `opened` is set before savefig deliberately: real matplotlib truncates the file on
        # open, so once savefig is entered the conservative report is "we opened it". The
        # guarantee under test is that the guard does not DELETE it.
        with pytest.warns(UserWarning, match="after opening"):
            with pytest.raises(RuntimeError):
                draw([Image(wave, label="a"), Image(wave, label="b")], path=str(victim))
    finally:
        mfig.Figure.savefig = real
    assert victim.exists(), "a failed panel draw DELETED the caller's file"
    assert victim.read_bytes() == original


def test_probe_failure_under_warnings_as_errors_keeps_the_figure(wave, tmp_path):
    """_measure runs INSIDE the cleanup guard and warns on a probe failure. Under `-W error`
    that warn raises, reaching discard_partial(owned=True) — which deleted the good figure."""
    dest = tmp_path / "good.png"
    from PIL import Image as PILImage
    real = PILImage.open
    try:
        PILImage.open = lambda *a, **k: (_ for _ in ()).throw(OSError("bad probe"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # the condition that made it destructive
            info = draw(Image(wave), path=str(dest))
    finally:
        PILImage.open = real
    assert dest.exists() and dest.stat().st_size > 0, "a good figure was deleted"
    assert info.saved and info.width is None


def test_imageinfo_saving_onto_itself_does_not_destroy_the_file(wave, tmp_path):
    """The class r.image()/r.trace() actually return — pinned like the other two."""
    info = draw(Image(wave), path=str(tmp_path / "fig.png"))
    original = open(info.path, "rb").read()
    assert len(original) > 0 and info.data is None
    info.save(info.path)
    assert open(info.path, "rb").read() == original, "save-onto-self destroyed the figure"


# ------------------------------------- show(): the matplotlib contract

def test_imageinfo_show_displays_inline(wave, monkeypatch):
    info = draw(Image(wave), "fig")
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: True)
    shown = []
    import IPython.display as ipd
    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj))
    assert info.show() is None, "show() must return None — returning self would double-embed"
    assert len(shown) == 1, "display must be called exactly once"
    assert shown[0] is info


def test_imageinfo_show_opens_a_viewer(wave, monkeypatch):
    info = draw(Image(wave), "fig")            # unsaved → materialised for the viewer
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    opened = {}
    monkeypatch.setattr(disp, "open_externally", lambda p: (opened.setdefault("p", p), True)[1])
    info.show()
    assert opened["p"].endswith(".png") and os.path.exists(opened["p"])


def test_imageinfo_show_reports_the_path_when_nothing_opens(wave, monkeypatch, capsys):
    info = draw(Image(wave), "fig")
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    monkeypatch.setattr(disp, "open_externally", lambda p: False)
    info.show()
    assert "No image viewer" in capsys.readouterr().out


def test_imageinfo_show_after_the_file_was_deleted(wave, tmp_path, monkeypatch, capsys):
    info = draw(Image(wave), "fig", path=str(tmp_path / "gone.png"))
    os.remove(info.path)
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    monkeypatch.setattr(disp, "open_externally", lambda p: True)
    info.show()                                # must not raise
    assert "figure unavailable" in capsys.readouterr().out


def test_calling_imageinfo_show_does_not_import_matplotlib(tmp_path):
    """The durable guard the import-time test cannot give: CALLING .show() must not pull in
    matplotlib. ImageInfo is hand-constructed (NOT via draw(), which imports matplotlib and would
    make this vacuous)."""
    code = (
        "import sys\n"
        "from cardiac_core import ImageInfo\n"
        "from cardiac_core import _display\n"
        "_display.in_notebook = lambda: False\n"
        "_display.open_externally = lambda p: True\n"
        "info = ImageInfo(path=None, data=b'\\x89PNG not-real', format='png', width=4, height=4,"
        " n_panels=1, vmin=None, vmax=None, size_bytes=12)\n"
        "info.show()\n"
        "print('matplotlib' in sys.modules)\n"
    )
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", f"matplotlib was imported by .show(): {out.stdout!r}\n{out.stderr}"
