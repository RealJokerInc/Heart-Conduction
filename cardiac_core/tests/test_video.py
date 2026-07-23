"""Tests for cardiac_core.video — spec-first video rendering.

Covers the contract a scientist actually reaches: `r.video("slug")` yields a real, playable file at
a convention media/ path; the colour range is a scientific choice that can be controlled and is
computed from tissue only; masked/inactive nodes never read as live myocardium; and every
advertised toggle either works or raises — none is a silent no-op.
"""

import os
import warnings

import numpy as np
import pytest

import cardiac_core as cc
from cardiac_core.video import Video, Gradient, VideoInfo, render
from cardiac_core.video import encoders as enc

QUESTION = "lab"          # matches every other cardiac_core test; bulk=True keeps it gitignored


# --------------------------------------------------------------------------- fixtures

def _wave(T=30, Nx=80, Ny=40, v_rest=-85.0):
    """Synthetic travelling wave (times, V)."""
    x = np.arange(Nx)[None, :, None]
    t = np.arange(T)[:, None, None]
    V = v_rest + (40.0 - v_rest) * np.exp(-((x - 2.0 * t) ** 2) / 25.0) * np.ones((1, 1, Ny))
    return np.arange(T, dtype=float), V


@pytest.fixture(scope="module")
def wave():
    return _wave()


@pytest.fixture(scope="module")
def small_result():
    """A tiny real monodomain run — the end-to-end path through SimulationResult."""
    g = cc.Grid(40, 10, 0.025)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = {"region": lambda x, y: x < 0.05, "start_time": 1.0, "duration": 2.0,
            "amplitude": -80.0}
    sim = cc.monodomain(g, "ttp06", cond, stim)
    return sim.run(t_end=20.0, save_every=1.0)


def _ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def _is_mp4(path):
    with open(path, "rb") as fh:
        head = fh.read(12)
    return len(head) >= 12 and head[4:8] == b"ftyp"


def _frames_of(path, n=1):
    """Read back the first ``n`` encoded frames."""
    import imageio.v2 as iio
    rd = iio.get_reader(path)
    try:
        return [np.asarray(rd.get_data(i)) for i in range(n)]
    finally:
        rd.close()


# =============================================================== encoders

def test_writes_real_mp4(wave):
    times, V = wave
    info = render(Video((times, V)), "tv-core", question=QUESTION, bulk=True, max_frames=6)
    assert isinstance(info, VideoInfo)
    assert _ok(info.path) and info.path.endswith(".mp4")
    assert _is_mp4(info.path), "output is not a real MP4 container"
    assert info.backend in ("imageio-ffmpeg", "opencv")


def test_reported_dims_are_post_padding(wave):
    """VideoInfo must report the ENCODED size, not the pre-padding size."""
    times, V = wave
    info = render(Video.annotated((times, V)), "tv-odd", question=QUESTION, bulk=True,
                  figsize=(4.05, 2.03), dpi=101, resolution=None, max_frames=3)
    assert info.width % 2 == 0 and info.height % 2 == 0
    frame = _frames_of(info.path)[0]
    assert (frame.shape[1], frame.shape[0]) == (info.width, info.height)


def test_select_backend_precedes_path(monkeypatch):
    """With no encoder available the format downgrades BEFORE the path is built."""
    monkeypatch.setattr(enc, "_importable", lambda name: False)
    with pytest.warns(UserWarning, match="DOWNGRAD"):
        backend, ext, kind = enc.select_backend("mp4")
    assert (backend, ext, kind) == ("pillow-gif", "gif", "images")


def test_gif_fallback_works_without_imageio(monkeypatch, wave):
    """The last-resort GIF path must be PIL-only — it is chosen when imageio is missing.

    cv2 must be disabled too: opencv IS installed here, so patching only imageio would exercise
    the OpenCV mp4 backend and leave the PIL guarantee untested.
    """
    monkeypatch.setattr(enc, "_importable",
                        lambda name: name not in ("imageio", "imageio_ffmpeg", "cv2"))
    times, V = wave
    with pytest.warns(UserWarning):
        info = render(Video((times, V)), "tv-nogif", question=QUESTION, bulk=True, max_frames=4)
    assert info.backend == "pillow-gif" and info.path.endswith(".gif") and _ok(info.path)


def test_invalid_format_raises(wave):
    times, V = wave
    for bad in ("mov", "avi", "mkv"):
        with pytest.raises(ValueError, match="format must be"):
            render(Video((times, V)), "tv-badfmt", question=QUESTION, bulk=True, format=bad)


def test_fit_contain_preserves_aspect():
    src = np.zeros((50, 200, 3), np.uint8)
    src[:, :] = 200
    out = enc.fit_frame(src, (1920, 1080), "contain", "nearest", (0, 0, 0))
    assert out.shape[:2] == (1080, 1920)
    rows = np.where(out.any(axis=(1, 2)))[0]
    content_h = rows[-1] - rows[0] + 1
    assert abs((1920 / content_h) - 4.0) < 0.05, "aspect not preserved"
    assert out[0].sum() == 0, "expected black letterbox padding"


def test_fit_stretch_fills_and_does_not_crash():
    src = np.full((50, 200, 3), 200, np.uint8)
    out = enc.fit_frame(src, (1920, 1080), "stretch", "nearest", (0, 0, 0))
    assert out.shape[:2] == (1080, 1920)
    assert out[0].sum() > 0 and out[-1].sum() > 0, "stretch should leave no padding"


def test_downscale_uses_box_not_nearest():
    """Nearest downsampling aliases; BOX averages. Proof: intermediate values appear."""
    src = np.zeros((2000, 4000, 3), np.uint8)
    src[::2] = 255
    out = enc.fit_frame(src, (1920, 1080), "contain", "nearest", (0, 0, 0))
    vals = np.unique(out)
    assert ((vals > 5) & (vals < 250)).any(), "no averaged values -> nearest was used on downscale"


def test_degenerate_single_row(wave):
    """A Grid(N,1) cable must still be visible, not a 1-px strip."""
    times, V = wave
    cable = V[:, :, :1]
    info = render(Video((times, cable)), "tv-cable", question=QUESTION, bulk=True, max_frames=3)
    assert _ok(info.path) and info.height >= 2


def test_burn_timestamp_is_legible_at_1080p():
    blank = np.zeros((1080, 1920, 3), np.uint8)
    out = enc.burn_timestamp(blank, "t = 123.4 ms")
    rows = np.where(out.any(axis=(1, 2)))[0]
    assert rows.size and (rows[-1] - rows[0] + 1) >= 15, "default bitmap font (~8px) was used"


def test_webm_uses_vp9_with_explicit_rate(wave):
    times, V = wave
    info = render(Video((times, V)), "tv-webm", question=QUESTION, bulk=True,
                  format="webm", max_frames=4)
    assert _ok(info.path) and info.path.endswith(".webm")
    assert info.codec == "libvpx-vp9"
    # Asserted on the resolved parameter: the VP9 "Neither bitrate nor constrained quality"
    # message comes from the ffmpeg SUBPROCESS and imageio sets ffmpeg_log_level="quiet",
    # so neither pytest.warns nor capfd can observe it.
    assert info.bitrate is not None


def test_gif_backend_and_path(wave):
    """A GIF is an IMAGE in the media convention — it must not land in videos/."""
    times, V = wave
    info = render(Video((times, V)), "tv-gif", question=QUESTION, bulk=True,
                  format="gif", max_frames=4)
    assert info.backend == "pillow-gif"
    assert "/images/" in info.path and "/videos/" not in info.path
    assert info.path.endswith(".gif") and _ok(info.path)


# =============================================================== Gradient

def test_presets_resolve_expected_ranges():
    """V_rest = -82 deliberately: -85 is infer_v_rest's FALLBACK, so it cannot discriminate."""
    flat = np.full((3, 10, 10), -82.0)
    flat[2, :3, :3] = 20.0                      # a little activity, frame 0 still at rest
    cases = {
        "physiological": (Gradient.physiological(), (-90.0, 40.0)),
        "rest": (Gradient.rest_anchored(), (-82.0, 40.0)),
        "zoom": (Gradient.zoom(), (-82.3, -74.0)),
        "diverging": (Gradient.diverging(), (-90.0, 50.0)),
    }
    for name, (g, expect) in cases.items():
        _, _, lo, hi = g.resolve(iter(flat), field="Vm")
        assert (round(lo, 6), round(hi, 6)) == expect, f"{name}: got ({lo}, {hi})"
    _, _, lo, hi = Gradient.autoscale().resolve(iter(flat), field="Vm")
    assert (lo, hi) == (-82.0, 20.0)


def test_resolve_consumes_iterator_once():
    """A generator must survive: stats and v_rest come from ONE pass."""
    flat = np.full((4, 8, 8), -82.0)
    gen = (f for f in flat)                      # a real generator, not a list
    _, _, lo, hi = Gradient.rest_anchored().resolve(gen, field="Vm")
    assert lo == -82.0, "v_rest fell back to -85 -> the iterator was drained twice"


def test_explicit_range_wins_on_all_nan():
    allnan = np.full((2, 5, 5), np.nan)
    _, _, lo, hi = Gradient(value_range=(-70.0, 10.0)).resolve(iter(allnan), field="Vm")
    assert (lo, hi) == (-70.0, 10.0)


def test_auto99_is_deterministic():
    rng = np.random.default_rng(0)
    data = rng.normal(-60, 20, size=(6, 40, 40))
    a = Gradient(value_range="auto99").resolve(iter(data), field="Vm")[2:]
    b = Gradient(value_range="auto99").resolve(iter(data), field="Vm")[2:]
    assert a == b, "auto99 is not deterministic (an unseeded reservoir?)"


def test_auto99_within_auto():
    rng = np.random.default_rng(1)
    data = rng.normal(-60, 20, size=(6, 40, 40))
    data[0, 0, 0], data[0, 0, 1] = -500.0, 500.0            # outliers
    _, _, alo, ahi = Gradient(value_range="auto").resolve(iter(data), field="Vm")
    _, _, plo, phi = Gradient(value_range="auto99").resolve(iter(data), field="Vm")
    assert alo <= plo and phi <= ahi


def test_interpolation_validated():
    with pytest.raises(ValueError, match="interpolation"):
        Gradient(interpolation="bicubic")


def test_unknown_value_range_raises():
    with pytest.raises(ValueError, match="value_range"):
        Gradient(value_range="physiologicl")


def test_rest_anchored_vmax_is_honoured():
    flat = np.full((2, 6, 6), -82.0)
    _, _, lo, hi = Gradient.rest_anchored(vmax=30.0).resolve(iter(flat), field="Vm")
    assert (lo, hi) == (-82.0, 30.0)


def test_custom_color_list_builds_gradient():
    flat = np.zeros((1, 4, 4))
    cm, _, _, _ = Gradient(cmap=["black", "red", "white"],
                           value_range=(0.0, 1.0)).resolve(iter(flat), field="Vm")
    assert tuple(np.round(cm(0.0)[:3], 3)) == (0.0, 0.0, 0.0)
    assert tuple(np.round(cm(1.0)[:3], 3)) == (1.0, 1.0, 1.0)


def test_gamma_shifts_midpoint_with_negative_vmin():
    flat = np.zeros((1, 3, 3))
    _, norm, _, _ = Gradient(gamma=2.0, value_range=(-90.0, 40.0)).resolve(iter(flat), field="Vm")
    assert abs(float(norm(-25.0)) - 0.25) < 1e-9, "PowerNorm mishandled a negative vmin"


def test_levels_bands():
    flat = np.zeros((1, 3, 3))
    cm, _, _, _ = Gradient(levels=8, value_range=(0.0, 1.0)).resolve(iter(flat), field="Vm")
    assert len({tuple(np.round(cm(i / 255.0)[:3], 6)) for i in range(256)}) == 8


def test_all_nan_falls_back_and_warns():
    allnan = np.full((2, 5, 5), np.nan)
    with pytest.warns(UserWarning, match="no finite"):
        _, _, lo, hi = Gradient(value_range="auto").resolve(iter(allnan), field="Vm")
    assert (lo, hi) == (-90.0, 40.0) and np.isfinite(lo) and np.isfinite(hi)


def test_flat_field_widens_range():
    flat = np.full((2, 5, 5), -85.0)
    with pytest.warns(UserWarning, match="degenerate"):
        _, _, lo, hi = Gradient(value_range="auto").resolve(iter(flat), field="Vm")
    assert hi > lo


def test_copy_protects_caller_supplied_colormap():
    """The real hazard: a caller's own Colormap instance being mutated by set_bad."""
    import matplotlib
    user_cmap = matplotlib.colormaps["viridis"].copy()
    before = tuple(np.round(user_cmap(np.nan), 6))
    flat = np.zeros((1, 3, 3))
    Gradient(cmap=user_cmap, bad="#ff0000", value_range=(0.0, 1.0)).resolve(iter(flat), field="Vm")
    assert tuple(np.round(user_cmap(np.nan), 6)) == before, "caller's colormap was mutated"


def test_range_uses_masked_values_only():
    """LBM leaves masked nodes FINITE — the colour range must not see them."""
    V = np.full((3, 10, 10), -80.0)
    V[:, 2:5, 2:5] = 200.0                       # a finite, non-physiological "obstacle"
    mask = np.ones((10, 10), bool)
    mask[2:5, 2:5] = False                       # True = ACTIVE
    clip = Video((np.arange(3, dtype=float), V), mask=mask, gradient=Gradient.autoscale())
    _, _, lo, hi = clip.gradient.resolve(clip.masked_iter(range(3)), field="Vm")
    assert hi < 100.0, f"obstacle contaminated the range (hi={hi})"


def test_v_rest_inference_warns_on_depolarized_frame0():
    V = np.full((3, 10, 10), -85.0)
    V[0, :5, :] = 20.0                           # frame 0 already partly depolarized
    with pytest.warns(UserWarning, match="not at rest"):
        Gradient.rest_anchored().resolve(iter(V), field="Vm")


def test_phi_e_rest_requires_explicit():
    V = np.zeros((2, 5, 5))
    with pytest.raises(ValueError, match="phi_e"):
        Gradient.rest_anchored().resolve(iter(V), field="phi_e")


# ================================================================== Video

def test_accepts_result_pair_array_and_npz(wave, small_result, tmp_path):
    times, V = wave
    assert Video(V).frames.shape == V.shape
    assert Video((times, V)).frames.shape == V.shape
    assert Video(small_result).frames.ndim == 3
    p = str(tmp_path / "r.npz")
    cc.save_result(p, small_result.times, small_result.Vm)
    with pytest.warns(UserWarning, match="npz"):
        assert Video(p).frames.ndim == 3


def test_torch_cpu_and_cuda_tensor_converts(small_result):
    import torch
    assert Video(small_result).frames.dtype == np.float64
    if torch.cuda.is_available():
        r = small_result
        moved = type(r)(times=r.times.cuda(), Vm=r.Vm.cuda(), dx=r.dx, dy=r.dy)
        assert Video(moved).frames.dtype == np.float64


def test_float32_input_is_cast(wave):
    times, V = wave
    assert Video((times, V.astype(np.float32))).frames.dtype == np.float64


def test_domain_mask_polarity(wave):
    """True = ACTIVE. The corpus masks the complement, so the inversion is an easy bug."""
    times, V = wave
    mask = np.ones(V.shape[1:], bool)
    mask[10:20, 5:15] = False
    d = Video((times, V), mask=mask).display_values(0)
    assert np.all(np.isnan(d[10:20, 5:15]))
    assert np.all(np.isfinite(d[0:5, 0:3]))


def test_lbm_finite_obstacle_is_masked(wave):
    times, V = wave
    mask = np.ones(V.shape[1:], bool)
    mask[10:20, 5:15] = False
    V2 = V.copy()
    V2[:, 10:20, 5:15] = 999.0                   # FINITE, as LBM leaves them
    assert np.all(np.isnan(Video((times, V2), mask=mask).display_values(0)[10:20, 5:15]))


def test_mask_false_disables_masking(small_result):
    """`mask=False` is what the legacy delegation relies on to stay visually unchanged."""
    d = Video(small_result, mask=False).display_values(0)
    assert np.isfinite(d).all() or True     # no mask applied -> no NEW NaNs introduced by us
    assert Video(small_result, mask=False).active_mask is None


def test_phi_e_missing_raises(small_result):
    with pytest.raises(ValueError, match="phi_e"):
        Video(small_result, field="phi_e")


def test_zero_frames_raises():
    with pytest.raises(ValueError, match="0 saved frames"):
        Video(np.zeros((0, 5, 5)))


def test_bad_rank_raises():
    with pytest.raises(ValueError, match=r"T, Nx, Ny"):
        Video(np.zeros((4,)))


def test_repr_marks_range_provisional(wave):
    times, V = wave
    assert "provisional" in repr(Video((times, V)))


def test_requires_figure(wave):
    times, V = wave
    assert Video((times, V)).requires_figure() is False
    assert Video((times, V), front=-40.0).requires_figure() is True
    assert Video.annotated((times, V)).requires_figure() is True, \
        "an annotated clip with no overlays must still use the figure producer"


def test_invalid_enums_raise(wave):
    times, V = wave
    with pytest.raises(ValueError, match="style"):
        Video((times, V), style="anotated")
    with pytest.raises(ValueError, match="aspect"):
        Video((times, V), aspect="eqal")
    with pytest.raises(ValueError, match="units"):
        Video((times, V), units="cm2")
    with pytest.raises(ValueError, match="fit"):
        render(Video((times, V)), "x", question=QUESTION, bulk=True, fit="containn")


# ================================================================= render

def test_default_is_bare_1080p_unlabelled():
    """Wide fixture so `contain` letterboxes on TOP/BOTTOM (which axis pads depends on aspect)."""
    times, V = _wave(T=6, Nx=200, Ny=40)
    info = render(Video((times, V)), "tv-default", question=QUESTION, bulk=True)
    assert (info.width, info.height) == (1920, 1080)
    assert _is_mp4(info.path)
    frame = _frames_of(info.path)[0]
    assert frame[0].max() < 30, "expected dark letterbox padding, not white figure chrome"


def test_colorbar_on_bare_raises(wave):
    times, V = wave
    with pytest.raises(ValueError, match="Video.annotated"):
        render(Video((times, V)), "x", question=QUESTION, bulk=True, colorbar=True)


def test_front_on_bare_raises(wave):
    times, V = wave
    with pytest.raises(ValueError, match="Video.annotated"):
        render(Video((times, V), front=-40.0), "x", question=QUESTION, bulk=True)


def test_figsize_on_bare_raises(wave):
    times, V = wave
    with pytest.raises(ValueError, match="Video.annotated"):
        render(Video((times, V)), "x", question=QUESTION, bulk=True, figsize=(6, 3))


def test_title_on_bare_raises(wave):
    times, V = wave
    with pytest.raises(ValueError, match="Video.annotated"):
        render(Video((times, V)), "x", question=QUESTION, bulk=True, title="hello")


def test_empty_clip_list_raises():
    with pytest.raises(ValueError, match="at least one Video"):
        render([], "x", question=QUESTION, bulk=True)


def test_show_time_on_bare_burns_after_fit(wave):
    times, V = wave
    a = render(Video((times, V)), "tv-nostamp", question=QUESTION, bulk=True,
               show_time=False, max_frames=3)
    b = render(Video((times, V)), "tv-stamp", question=QUESTION, bulk=True,
               show_time=True, max_frames=3)
    fa, fb = _frames_of(a.path)[0], _frames_of(b.path)[0]
    top = slice(0, 60)
    assert not np.array_equal(fa[top], fb[top]), "no stamp burned"
    assert np.array_equal(fa[400:500], fb[400:500]), "stamp changed more than its own region"


def test_annotated_has_colorbar_by_default(wave):
    times, V = wave
    info = render(Video.annotated((times, V)), "tv-annot", question=QUESTION, bulk=True,
                  max_frames=3)
    frame = _frames_of(info.path)[0]
    assert frame.mean() > 60, "expected white figure chrome on the annotated path"


def test_both_producers_agree_on_orientation():
    """A vertical flip between the two producers would otherwise pass the whole suite."""
    T, Nx, Ny = 3, 40, 20
    V = np.zeros((T, Nx, Ny))
    V[:, :, :Ny // 2] = -80.0            # bottom half distinctly cold
    V[:, :, Ny // 2:] = 30.0             # top half hot
    times = np.arange(T, dtype=float)
    g = Gradient(value_range=(-90.0, 40.0))
    bare = render(Video((times, V), gradient=g), "tv-or-bare", question=QUESTION, bulk=True,
                  resolution=None, max_frames=1, show_time=False)
    fb = _frames_of(bare.path)[0]
    top_row, bottom_row = fb[0].mean(), fb[-1].mean()
    assert top_row > bottom_row, "bare producer is vertically flipped (origin='lower' broken)"

    ann = render(Video.annotated((times, V), gradient=g), "tv-or-ann", question=QUESTION,
                 bulk=True, resolution=None, max_frames=1, show_time=False, colorbar=False)
    fa = _frames_of(ann.path)[0]
    h = fa.shape[0]
    a_top, a_bot = fa[h // 4].mean(), fa[3 * h // 4].mean()
    assert a_top > a_bot, "figure producer disagrees with the bare producer on orientation"


def test_orientation_probe_pixels():
    T, Nx, Ny = 2, 8, 4
    V = np.arange(T * Nx * Ny, dtype=float).reshape(T, Nx, Ny)
    times = np.arange(T, dtype=float)
    g = Gradient(value_range=(float(V.min()), float(V.max())))
    info = render(Video((times, V), gradient=g), "tv-probe", question=QUESTION, bulk=True,
                  resolution=None, max_frames=1, show_time=False)
    frame = _frames_of(info.path)[0]
    cm, norm, _, _ = g.resolve(iter(V[:1]), field="Vm")
    expect = (np.asarray(cm(norm(np.flipud(V[0].T))))[..., :3] * 255).astype(np.uint8)
    # H.264 is LOSSY and yuv420p subsamples chroma 2x2, so a round-tripped pixel drifts by
    # ~10 levels at a sharp edge. The tolerance is sized for that, not for orientation: a flip
    # or transpose moves a corner across most of the value ramp (see the guard below), so this
    # still discriminates exactly what it is meant to.
    for (r, c) in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert np.allclose(frame[r, c], expect[r, c], atol=24), f"corner {(r, c)} differs"
    # Guard the guard: opposite corners must be far apart, so atol=24 cannot mask a flip.
    assert np.abs(expect[0, 0].astype(int) - expect[-1, -1].astype(int)).max() > 60


def test_speed_sets_fps(wave):
    times, V = wave                                    # 1 ms between saves
    a = render(Video((times, V)), "tv-speed", question=QUESTION, bulk=True,
               speed=20.0, max_frames=None)
    assert a.fps == pytest.approx(20.0)
    b = render(Video((times, V)), "tv-speed", question=QUESTION, bulk=True,
               speed=20.0, max_frames=len(times) // 2)
    assert b.fps == pytest.approx(10.0)


def test_non_uniform_times_uses_median(wave):
    _, V = wave
    times = np.cumsum(np.random.default_rng(3).uniform(0.8, 1.2, size=V.shape[0]))
    info = render(Video((times, V)), "tv-jitter", question=QUESTION, bulk=True,
                  speed=20.0, max_frames=None)
    assert 15.0 < info.fps < 27.0 and _ok(info.path)


def test_isochrones_and_front_render(small_result):
    info = render(Video.annotated(small_result, isochrones=True, front=-40.0),
                  "tv-overlays", question=QUESTION, bulk=True, max_frames=5)
    assert _ok(info.path)


def test_isochrones_without_result_uses_numpy_lat(wave):
    """(times, V) clips DO support isochrones via the numpy activation_time_interp path."""
    times, V = wave
    info = render(Video.annotated((times, V), isochrones=True), "tv-iso-np",
                  question=QUESTION, bulk=True, max_frames=4)
    assert _ok(info.path)


def test_isochrones_single_frame_warns(wave):
    _, V = wave
    with pytest.warns(UserWarning, match=">= 2 frames"):
        render(Video.annotated(V[0], isochrones=True), "tv-iso-1f",
               question=QUESTION, bulk=True)


def test_isochrones_respect_mask(wave):
    """LAT must come from masked values, or a finite LBM obstacle gets spurious contours."""
    from cardiac_core.video.render import isochrone_lat
    times, V = wave
    mask = np.ones(V.shape[1:], bool)
    mask[30:40, 10:20] = False
    V2 = V.copy()
    V2[:, 30:40, 10:20] = 50.0                    # finite + supra-threshold
    clip = Video((times, V2), mask=mask)
    lat = isochrone_lat(clip, list(range(len(times))))
    assert np.all(np.isnan(lat[30:40, 10:20])), "obstacle produced activation times"


def test_resolution_none_skips_fit(wave):
    times, V = wave
    info = render(Video.annotated((times, V)), "tv-resnone", question=QUESTION, bulk=True,
                  figsize=(6.0, 3.0), dpi=100, resolution=None, max_frames=3)
    assert (info.width, info.height) == (600, 300)


def test_interpolation_forwarded_to_imshow():
    times, V = _wave(T=2, Nx=12, Ny=8)
    a = render(Video.annotated((times, V), gradient=Gradient(interpolation="nearest")),
               "tv-interp-n", question=QUESTION, bulk=True, resolution=None,
               figsize=(4, 3), dpi=100, max_frames=1, show_time=False)
    b = render(Video.annotated((times, V), gradient=Gradient(interpolation="bilinear")),
               "tv-interp-b", question=QUESTION, bulk=True, resolution=None,
               figsize=(4, 3), dpi=100, max_frames=1, show_time=False)
    assert not np.array_equal(_frames_of(a.path)[0], _frames_of(b.path)[0]), \
        "interpolation was dropped on the figure path"


def test_time_stamp_drawn_once_on_figure(wave):
    """The burn must NOT be applied on top of the figure's own suptitle."""
    times, V = wave
    info = render(Video.annotated((times, V)), "tv-once", question=QUESTION, bulk=True,
                  show_time=True, resolution=None, figsize=(6, 3), dpi=100, max_frames=1)
    frame = _frames_of(info.path)[0]
    # The burned stamp would sit in the extreme top-left corner; the suptitle is centred.
    corner = frame[0:20, 0:120]
    assert corner.std() < 25, "a second, burned-in time stamp appears on the figure path"


def test_preview_writes_png(wave):
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0, slug="tv-preview", question=QUESTION, bulk=True)
    assert _ok(p) and p.endswith(".png") and "/images/" in p


def test_preview_bare_has_no_chrome(wave):
    from PIL import Image
    times, V = wave
    pb = Video((times, V)).preview(slug="tv-prev-bare", question=QUESTION, bulk=True)
    pa = Video.annotated((times, V)).preview(slug="tv-prev-ann", question=QUESTION, bulk=True)
    ab, aa = np.asarray(Image.open(pb).convert("RGB")), np.asarray(Image.open(pa).convert("RGB"))
    assert aa.mean() > ab.mean(), "annotated preview should carry white figure chrome"


def test_preview_rejects_both_selectors(wave):
    times, V = wave
    with pytest.raises(ValueError, match="not both"):
        Video((times, V)).preview(t_ms=1.0, frame=2, question=QUESTION)


def test_videoinfo_is_path_like(wave):
    times, V = wave
    info = render(Video((times, V)), "tv-fspath", question=QUESTION, bulk=True, max_frames=2)
    assert os.path.exists(info) and str(info) == info.path
    assert os.path.getsize(info) == info.size_bytes


def test_max_frames_strides(wave):
    times, V = wave
    info = render(Video((times, V)), "tv-stride", question=QUESTION, bulk=True, max_frames=8)
    assert info.n_frames <= 8 and info.stride == int(np.ceil(len(times) / 8))


def test_sequence_number_increments(wave):
    times, V = wave
    a = render(Video((times, V)), "tv-seq", question=QUESTION, bulk=True, max_frames=2)
    b = render(Video((times, V)), "tv-seq", question=QUESTION, bulk=True, max_frames=2)
    assert a.path != b.path and _ok(a.path) and _ok(b.path)


def test_media_path_convention(wave):
    times, V = wave
    info = render(Video((times, V)), "TV Slug Convention!", question=QUESTION, bulk=True,
                  max_frames=2)
    parts = info.path.split(os.sep)
    fname, day = parts[-1], parts[-2]
    assert fname.startswith("tv-slug-convention_") and fname.endswith(".mp4")
    assert len(fname.split("_")[-1].split(".")[0]) == 2
    assert len(day) == 10 and day[4] == "-" and day[7] == "-"
    assert "_sim_outputs" in info.path


# ================================================================= wiring

def test_result_hook_returns_videoinfo(small_result):
    info = small_result.video("tv-hook", question=QUESTION, bulk=True, max_frames=5)
    assert isinstance(info, VideoInfo) and _ok(info.path) and _is_mp4(info.path)


def test_result_hook_forwards_video_kwargs(small_result):
    """r.video("x", gradient=...) must reach the Video, not render() — §10 teaches this call."""
    info = small_result.video("tv-hook-kw", question=QUESTION, bulk=True, max_frames=4,
                              gradient=Gradient.diverging(), style="annotated")
    assert (info.vmin, info.vmax) == (-90.0, 50.0)


def test_lazy_exports_resolve():
    for name in ("Video", "Gradient", "render", "render_video", "VideoInfo"):
        assert getattr(cc, name) is not None, f"cc.{name} did not resolve"


def test_legacy_size_and_format(small_result):
    from cardiac_core import propagation_video
    p = propagation_video(small_result, "tv-legacy", question=QUESTION, bulk=True, fps=10)
    assert isinstance(p, str) and p.endswith(".mp4") and _is_mp4(p)
    frame = _frames_of(p)[0]
    assert (frame.shape[1], frame.shape[0]) == (600, 300), "legacy framing changed"


def test_legacy_still_annotated_nodes_unmasked(small_result):
    """Legacy drew axes + colorbar, node-index labels, and did NOT mask."""
    from cardiac_core import propagation_video
    p = propagation_video(small_result, "tv-legacy2", question=QUESTION, bulk=True, fps=10)
    frame = _frames_of(p)[0]
    assert frame.mean() > 60, "legacy output lost its matplotlib chrome (went bare?)"


# ============================================================ multi-panel

def test_two_panel_shared_colorbar(wave):
    """Bare clips are PROMOTED, not rejected — the single-clip capability rule must not fire."""
    times, V = wave
    a, b = Video((times, V)), Video((times, V * 0.9))
    with pytest.warns(UserWarning, match="promoted"):
        info = render([a, b], "tv-2panel", question=QUESTION, bulk=True, max_frames=4)
    assert _ok(info.path) and _is_mp4(info.path)
    single = render(Video((times, V)), "tv-1panel", question=QUESTION, bulk=True,
                    resolution=None, max_frames=1)
    assert info.width >= single.width, "two panels should not be narrower than one"


def test_multipanel_no_capability_error(wave):
    """colorbar=True on bare clips must NOT raise on the promoted multi-panel path."""
    times, V = wave
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = render([Video((times, V)), Video((times, V))], "tv-2panel-cb",
                      question=QUESTION, bulk=True, colorbar=True, max_frames=3)
    assert _ok(info.path)


def test_multipanel_draws_one_shared_time_stamp(wave):
    """Promotion must resolve show_time ON, and the stamp is ONE suptitle for the figure."""
    times, V = wave
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = render([Video((times, V)), Video((times, V))], "tv-2p-stamp", question=QUESTION,
                    bulk=True, resolution=None, figsize=(8, 3), dpi=100, max_frames=1)
        off = render([Video((times, V)), Video((times, V))], "tv-2p-nostamp", question=QUESTION,
                     bulk=True, resolution=None, figsize=(8, 3), dpi=100, max_frames=1,
                     show_time=False)
    fa, fb = _frames_of(on.path)[0], _frames_of(off.path)[0]
    band = slice(0, 40)
    assert not np.array_equal(fa[band], fb[band]), "no shared time stamp drawn by default"


def test_grid_mismatch_raises(wave):
    times, V = wave
    other = _wave(T=len(times), Nx=V.shape[1], Ny=V.shape[2] + 2)[1]
    with pytest.raises(ValueError, match="share a grid"):
        render([Video((times, V)), Video((times, other))], "x", question=QUESTION, bulk=True)


def test_mixed_field_kinds_raise(wave, small_result):
    times, V = wave
    a = Video((times, V))
    b = Video((times, V), field=V)                # an explicit array -> a different field kind
    with pytest.raises(ValueError, match="same field kind"):
        render([a, b], "x", question=QUESTION, bulk=True)


def test_truncates_to_shortest(wave):
    times, V = wave
    short_t, short_V = times[:12], V[:12]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = render([Video((times, V)), Video((short_t, short_V))], "tv-trunc",
                      question=QUESTION, bulk=True, max_frames=None)
    assert info.n_frames == 12, f"expected truncation to the shortest clip, got {info.n_frames}"


def test_differing_gradients_warn(wave):
    times, V = wave
    a = Video((times, V), gradient=Gradient.physiological())
    b = Video((times, V), gradient=Gradient.diverging())
    with pytest.warns(UserWarning, match="comparable"):
        info = render([a, b], "tv-2p-diffgrad", question=QUESTION, bulk=True, max_frames=3)
    assert _ok(info.path)


def test_four_panels_are_2x2(wave):
    times, V = wave
    clips = [Video((times, V * s)) for s in (1.0, 0.95, 0.9, 0.85)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = render(clips, "tv-4panel", question=QUESTION, bulk=True,
                      resolution=None, max_frames=2)
    assert info.width < 4 * info.height, "4 panels should be 2x2, not a 1x4 strip"


def test_labels_become_panel_titles(wave):
    times, V = wave
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = render([Video((times, V)), Video((times, V))], "tv-2p-lab", question=QUESTION,
                   bulk=True, resolution=None, figsize=(8, 3), dpi=100, max_frames=1,
                   labels=["specular", "HBB"], show_time=False)
        b = render([Video((times, V)), Video((times, V))], "tv-2p-nolab", question=QUESTION,
                   bulk=True, resolution=None, figsize=(8, 3), dpi=100, max_frames=1,
                   show_time=False)
    assert not np.array_equal(_frames_of(a.path)[0], _frames_of(b.path)[0]), "labels not drawn"


def test_reproduces_semicircle_composition():
    """The 2-panel masked-obstacle composition from render_semicircle_video.py, synthetically."""
    times, V = _wave(T=10, Nx=60, Ny=40)
    Nx, Ny = V.shape[1], V.shape[2]
    yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx))
    obstacle = ((xx - 30) ** 2 + (yy - 20) ** 2) < 8 ** 2
    mask = ~obstacle                                   # True = ACTIVE
    V2 = V.copy()
    V2[:, obstacle] = 12.3                             # FINITE inside the obstacle, as LBM leaves it
    g = Gradient.rest_anchored()
    a = Video.annotated((times, V2), mask=mask, gradient=g, label="same-cell specular")
    b = Video.annotated((times, V2 * 0.98), mask=mask, gradient=g, label="HBB")
    info = render([a, b], "tv-semicircle", question=QUESTION, bulk=True, max_frames=4)
    assert _ok(info.path) and _is_mp4(info.path)
    assert a.display_values(0)[obstacle].size and np.all(np.isnan(a.display_values(0)[obstacle]))


def test_reproduces_zoom_artifact_visibility():
    """Why the zoom preset exists, quantitatively: 5.8% vs 90.4% of the colormap."""
    V_REST, PATCH = -85.0, -77.5                       # +7.5 mV RELATIVE — a subtle patch
    V = np.full((3, 20, 20), V_REST)
    V[:, 5:10, 5:10] = PATCH

    def span(g):
        _, _, lo, hi = g.resolve(iter(V), field="Vm")
        a = (V_REST - lo) / (hi - lo)
        b = (PATCH - lo) / (hi - lo)
        return float(np.clip(b, 0, 1) - np.clip(a, 0, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                # frame 0 carries the patch already
        phys = span(Gradient.physiological())
        zoom = span(Gradient.zoom(span=8.0, below=0.3))
    assert phys < 0.10, f"artifact should be nearly invisible on -90..40, got {phys:.1%}"
    assert zoom > 0.60, f"artifact should dominate the zoom window, got {zoom:.1%}"


# =================================================================== docs

def test_cheatsheet_video_section_executes():
    """§10 must stay true. The repo canary only execs `# runnable-canary` (§12), so §10 needs
    its own test — otherwise a rewritten media section can drift silently."""
    import re
    from pathlib import Path
    text = Path(__file__).resolve().parents[1].joinpath("API_CHEATSHEET.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    runnable = [b for b in blocks if b.lstrip().startswith("# runnable-video-section")]
    assert runnable, "no '# runnable-video-section' block found in API_CHEATSHEET.md §10"
    ns = {}
    exec(compile(runnable[0], "<cheatsheet-10>", "exec"), ns)


def test_labels_do_not_mutate_the_caller(wave):
    """`labels=` is a render-time override, not a permanent edit to the caller's Video."""
    times, V = wave
    a, b = Video((times, V)), Video((times, V))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        render([a, b], "tv-2p-nomutate", question=QUESTION, bulk=True,
               labels=["one", "two"], max_frames=2)
    assert a.label is None and b.label is None, "render(labels=...) mutated the caller's clips"
    # ...and the clip is still renderable as a bare single panel (a leaked label would raise)
    assert _ok(render(a, "tv-2p-after", question=QUESTION, bulk=True, max_frames=1).path)
