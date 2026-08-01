"""Tests for cardiac_core.video — spec-first video rendering.

Covers the contract a scientist actually reaches: `r.video("slug")` yields a real, playable file at
a convention media/ path; the colour range is a scientific choice that can be controlled and is
computed from tissue only; masked/inactive nodes never read as live myocardium; and every
advertised toggle either works or raises — none is a silent no-op.
"""

import os
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cardiac_core as cc
from cardiac_core.video import Video, Gradient, VideoInfo, render
from cardiac_core.video import encoders as enc

QUESTION = "lab"          # matches every other cardiac_core test; bulk=True keeps it gitignored


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def _isolate_show_cache(tmp_path, monkeypatch):
    """Point .show()'s materialise-cache at a per-test tmp dir, so terminal-branch show tests never
    write into the developer's real ~/.cache/cardiac_core/show."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "_show_cache"))


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


def _media_tree():
    """Every file under the session media root (conftest points it at a tmpdir).

    The 'writes nothing' tests need this: asserting only on cwd would still pass if a regression
    started writing to media_path(), because that root is elsewhere.
    """
    root = os.environ.get("CARDIAC_MEDIA_ROOT")
    assert root and os.path.isdir(root), \
        "conftest's media-root fixture is not active; this check would be vacuous"
    return frozenset(
        os.path.join(dirpath, f)
        for dirpath, _dirs, files in os.walk(root) for f in files
    )


def _render_module():
    """The render MODULE. `import cardiac_core.video.render` yields the re-exported FUNCTION."""
    import sys
    return sys.modules["cardiac_core.video.render"]


def _throw_on_frame(n, exc):
    """A _produce_bare stand-in that renders normally until frame ``n``, then raises.

    Failing on frame 1 leaves the buffered backend with nothing to flush, which hides any bug in
    the recovery path — several of these tests are only meaningful past the first frame.
    """
    real = _render_module()._produce_bare
    seen = {"n": 0}

    def _inner(*a, **kw):
        seen["n"] += 1
        if seen["n"] >= n:
            raise exc("simulated failure")
        return real(*a, **kw)
    return _inner


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
    raw = small_result.Vm[0].detach().cpu().numpy()
    # `or True` made the old assertion unfalsifiable. The real claim is that mask=False introduces
    # no NaN the raw data did not already have.
    assert np.array_equal(np.isnan(d), np.isnan(raw)), "mask=False introduced NaNs of its own"
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
    text = Path(cc.__file__).parent.joinpath("API_CHEATSHEET.md").read_text()
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


# ------------------------------------------------- destination: display vs save
# Rendering displays; naming a destination saves — the matplotlib contract. These guard the
# rule itself, the three ways to name a destination, and that nothing leaks when none is named.

def test_bare_render_writes_no_file(wave, tmp_path, monkeypatch):
    """No destination named -> bytes in memory, nothing on disk, nothing left in cwd."""
    times, V = wave
    monkeypatch.chdir(tmp_path)
    before_tmp = set(os.listdir(tempfile.gettempdir()))
    before_media = _media_tree()

    info = render(Video((times, V)), max_frames=3)

    assert info.path is None and info.saved is False
    assert info.data and info.data[4:8] == b"ftyp", "encoded bytes should be a real MP4"
    assert info.size_bytes > 0
    assert list(tmp_path.iterdir()) == [], "a bare render must not write into the working dir"
    # The cwd check alone is decorative: conftest points CARDIAC_MEDIA_ROOT at a temp dir, so a
    # regression to media_path() would write THERE and still leave tmp_path empty.
    assert _media_tree() == before_media, "a bare render must not write under the media root"
    leaked = set(os.listdir(tempfile.gettempdir())) - before_tmp
    assert not leaked, f"temp file was not cleaned up: {sorted(leaked)}"


def test_path_writes_exactly_there(wave, tmp_path):
    """`path=` is obeyed literally — no media/ tree, no date folder, no _NN suffix."""
    times, V = wave
    dest = tmp_path / "nested" / "my-wave.mp4"

    info = render(Video((times, V)), path=str(dest), max_frames=3)

    assert info.path == str(dest) and info.saved is True
    assert _ok(info.path) and _is_mp4(info.path)
    assert info.data is None, "a saved render keeps its bytes on disk, not in memory"
    assert os.fspath(info) == str(dest)


def test_convention_keywords_still_save(wave):
    """Regression guard for the live consumers (Lab/, cardiac_mcp, run-template): each passes
    bulk=True and no path=, and must keep getting a real file at the media/ convention path."""
    times, V = wave
    info = render(Video((times, V)), "tv-conv", question=QUESTION, bulk=True, max_frames=3)

    assert info.saved and _ok(info.path)
    assert "/media/" in info.path and "/_sim_outputs/" in info.path
    assert info.path.endswith(".mp4")


def test_bulk_alone_names_a_destination(wave):
    """`bulk=` with no question= still saves — the exact shape every consumer call site uses."""
    times, V = wave
    info = render(Video((times, V)), "tv-bulkonly", bulk=True, max_frames=3)
    assert info.saved and _ok(info.path) and "/media/lab/" in info.path


def test_format_inferred_from_path_extension(wave, tmp_path):
    times, V = wave
    info = render(Video((times, V)), path=str(tmp_path / "anim.gif"), max_frames=3)
    assert info.backend == "pillow-gif" and info.codec == "gif" and _ok(info.path)


def test_unknown_path_extension_raises(wave, tmp_path):
    times, V = wave
    with pytest.raises(ValueError, match="cannot infer a video format"):
        render(Video((times, V)), path=str(tmp_path / "anim.mkv"), max_frames=3)


def test_save_after_the_fact(wave, tmp_path):
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    dest = info.save(tmp_path / "later.mp4")
    assert _ok(dest) and _is_mp4(dest)
    assert info.read() == open(dest, "rb").read()


def test_fspath_raises_with_guidance_when_unsaved(wave):
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    with pytest.raises(TypeError, match=r"path=|\.save\("):
        os.fspath(info)
    assert "not saved" in str(info)


# ------------------------------------------------------------- notebook display

def test_repr_html_embeds_a_playable_video(wave):
    times, V = wave
    html = render(Video((times, V)), max_frames=3)._repr_html_()
    assert "<video" in html and "controls" in html
    assert "data:video/mp4;base64," in html


def test_repr_html_gif_uses_img_tag(wave, tmp_path):
    times, V = wave
    html = render(Video((times, V)), path=str(tmp_path / "a.gif"), max_frames=3)._repr_html_()
    assert "<img" in html and "data:image/gif;base64," in html


def test_repr_html_works_for_a_saved_render(wave, tmp_path):
    """Display must not depend on retaining bytes — a saved render reads them back."""
    times, V = wave
    html = render(Video((times, V)), path=str(tmp_path / "b.mp4"), max_frames=3)._repr_html_()
    assert "data:video/mp4;base64," in html


def test_repr_html_reports_an_unplayable_codec():
    """OpenCV's mp4v writes a valid .mp4 no browser can decode — say so instead of embedding."""
    info = VideoInfo(path=None, n_frames=1, fps=20.0, backend="opencv", codec="mp4v",
                     width=64, height=64, duration_s=0.05, vmin=-90.0, vmax=40.0,
                     stride=1, size_bytes=10, data=b"x" * 10)
    html = info._repr_html_()
    assert "cannot play" in html and "imageio-ffmpeg" in html
    assert "base64" not in html


def test_repr_html_size_cap(wave, monkeypatch):
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    monkeypatch.setattr(enc, "INLINE_MAX_BYTES", 8)
    html = info._repr_html_()
    assert "too large to embed" in html and "base64" not in html


def test_preview_unsaved_displays_but_writes_nothing(wave, tmp_path, monkeypatch):
    times, V = wave
    monkeypatch.chdir(tmp_path)
    before_media = _media_tree()
    p = Video((times, V)).preview(t_ms=5.0)
    assert p.saved is False and p.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert "<img" in p._repr_html_() and "data:image/png;base64," in p._repr_html_()
    assert list(tmp_path.iterdir()) == []
    assert _media_tree() == before_media, "an unsaved preview must not write under the media root"


def test_preview_saved_is_still_a_plain_path_string(wave, tmp_path):
    """ImagePath subclasses str, so every existing caller keeps working."""
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0, path=str(tmp_path / "frame.png"))
    assert isinstance(p, str) and p.endswith(".png") and _ok(p)
    assert os.path.basename(p) == "frame.png"


def test_path_extension_follows_the_encoder(wave, tmp_path, monkeypatch):
    """A backend downgrade must not leave the caller's extension describing the wrong bytes.

    media_path self-corrects because it derives ext from the backend; `path=` is taken verbatim,
    so without this the writer either mislabels the file or (for PIL) dies on an extension it
    does not know.
    """
    times, V = wave
    real = enc._importable
    monkeypatch.setattr(enc, "_importable",
                        lambda n: False if n == "imageio_ffmpeg" else real(n))

    with pytest.warns(UserWarning, match="describes its own contents"):
        info = render(Video((times, V)), path=str(tmp_path / "out.webm"), max_frames=3)

    assert info.path.endswith(".gif"), "extension must follow the encoder, not the request"
    assert not (tmp_path / "out.webm").exists()
    with open(info.path, "rb") as fh:
        assert fh.read(6) in (b"GIF87a", b"GIF89a")


def test_path_without_extension_gets_one(wave, tmp_path):
    times, V = wave
    info = render(Video((times, V)), path=str(tmp_path / "noext"), max_frames=3)
    assert info.path.endswith(".mp4") and _is_mp4(info.path)


def test_close_failure_removes_the_partial_file(wave, tmp_path, monkeypatch):
    """The pillow-gif backend writes the WHOLE file in close(), so close must be inside the
    cleanup guard — otherwise a failed finalize leaves a partial file and burns the NN slot."""
    times, V = wave
    dest = tmp_path / "boom.mp4"
    real_close = enc._Writer.close
    calls = {"n": 0}

    def boom(self):
        calls["n"] += 1
        if calls["n"] == 1:
            self._closed = True          # real close() sets this before writing, too
            raise OSError("disk full during finalize")
        return real_close(self)

    monkeypatch.setattr(enc._Writer, "close", boom)
    with pytest.raises(OSError, match="disk full"):
        render(Video((times, V)), path=str(dest), max_frames=3)
    assert not dest.exists(), "a failed finalize must not leave a partial file behind"


# ------------------------------------- audit round 1: gaps the first pass missed

def test_repr_does_not_dump_the_payload(wave):
    """`data` must be repr=False — otherwise a REPL echo, a log line or a failing assertion
    prints the entire encoded video as an escaped bytes literal."""
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    text = repr(info)
    assert len(text) < 500, f"repr is {len(text)} chars — the payload is leaking into it"
    assert "data=" not in text and b"ftyp".decode() not in text


def test_save_makes_the_result_saved(wave, tmp_path):
    """`.save()` must update the object, not just write bytes: `.saved`, `os.fspath()` and the
    over-cap display all read `path`, and a Lab record gated on `.saved` would report no file."""
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    assert info.saved is False

    dest = info.save(tmp_path / "kept.mp4")

    assert info.saved is True and info.path == str(dest)
    assert os.fspath(info) == str(dest) and str(info) == str(dest)
    assert _ok(dest) and _is_mp4(dest)


def test_multipanel_unsaved_returns_bytes_and_writes_nothing(wave, tmp_path, monkeypatch):
    """The panel path has its own destination/finalize code — cover it, not just the single clip."""
    times, V = wave
    monkeypatch.chdir(tmp_path)
    before_media = _media_tree()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # bare clips are promoted for a shared colorbar
        info = render([Video((times, V)), Video((times, V))], max_frames=2)
    assert info.path is None and info.saved is False
    assert info.data and info.data[4:8] == b"ftyp"
    assert list(tmp_path.iterdir()) == [] and _media_tree() == before_media


def test_bulk_defaults_to_true_with_only_question(wave):
    """Documented rule: any convention keyword implies the gitignored bulk subtree unless the
    caller says otherwise. The docstring claimed the opposite before the audit."""
    times, V = wave
    a = render(Video((times, V)), "tv-bulkdefault", question=QUESTION, max_frames=2)
    assert "/_sim_outputs/" in a.path
    b = render(Video((times, V)), "tv-curated", question=QUESTION, bulk=False, max_frames=2)
    assert "/_sim_outputs/" not in b.path and "/media/lab/" in b.path


def test_path_plus_convention_keywords_warns(wave, tmp_path):
    """Both name a destination; path= wins. Silently dropping bulk=True would break the Lab
    record for an agent that followed sim-media's 'ALWAYS pass bulk=True'."""
    times, V = wave
    with pytest.warns(UserWarning, match="path= wins"):
        info = render(Video((times, V)), "x", path=str(tmp_path / "p.mp4"),
                      question=QUESTION, bulk=True, max_frames=2)
    assert info.path == str(tmp_path / "p.mp4")


def test_path_that_is_a_directory_raises(wave, tmp_path):
    times, V = wave
    with pytest.raises(IsADirectoryError, match="pass a FILE path"):
        render(Video((times, V)), path=str(tmp_path), max_frames=2)


def test_failure_before_the_writer_leaves_no_temp_file(wave, tmp_path, monkeypatch):
    """_resolve_destination has side effects (mkstemp / NN slot). Everything that can raise must
    raise BEFORE it, or a bad resolution= orphans a temp file on every call."""
    times, V = wave
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tempfile.gettempdir()))
    with pytest.raises(ValueError):
        render(Video((times, V)), resolution="9000p", max_frames=2)
    assert not (set(os.listdir(tempfile.gettempdir())) - before), "orphaned a temp file"


def test_repr_html_degrades_when_the_saved_file_is_gone(wave, tmp_path):
    """Re-running a cell after a media/ clean must not raise out of IPython's formatter."""
    times, V = wave
    info = render(Video((times, V)), path=str(tmp_path / "gone.mp4"), max_frames=2)
    os.remove(info.path)
    html = info._repr_html_()
    assert "unavailable" in html and "<video" not in html


def test_imagepath_survives_pickling(wave):
    """str.__getnewargs__ would rebuild an UNSAVED ImagePath from its human summary, silently
    turning it into a 'saved' object whose path is a sentence."""
    import pickle
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0)
    assert p.saved is False
    back = pickle.loads(pickle.dumps(p))
    assert back.saved is False and back.data == p.data and str(back) == str(p)
    assert back.format == p.format, "__reduce__ dropped `format`"


def test_imagepath_save_writes_and_returns_the_path(wave, tmp_path):
    """All three payload objects agree: save() writes and returns the path str."""
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0)
    dest = p.save(tmp_path / "frame.png")
    assert isinstance(dest, str) and _ok(dest) and dest.endswith("frame.png")
    assert p.saved is False, "a str cannot change its own value; the original stays unsaved"


# ------------------------------------- audit round 2: data loss + layer consistency

def test_failed_render_never_deletes_a_preexisting_file(wave, tmp_path, monkeypatch):
    """CRITICAL regression. The cleanup guard used to `os.remove(out_path)` unconditionally — but
    for a path= render that is the CALLER'S file, and a render can raise before writing a byte."""
    times, V = wave
    victim = tmp_path / "irreplaceable.gif"
    victim.write_bytes(b"GIF89a" + b"-five-years-of-work" * 20)
    original = victim.read_bytes()

    # Fail on the SECOND frame, not the first. pillow-gif buffers every frame and writes the
    # whole file in close(); with nothing buffered the recovery path writes nothing and this
    # test would pass even with the bug present.
    monkeypatch.setattr(_render_module(), "_produce_bare", _throw_on_frame(2, KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render(Video((times, V)), path=str(victim), format="gif", max_frames=5)

    assert victim.exists(), "a failed render DELETED the caller's pre-existing file"
    assert victim.read_bytes() == original, "a failed render OVERWROTE the caller's file"


def test_failed_render_warns_rather_than_silently_leaving_a_clobbered_file(wave, tmp_path,
                                                                          monkeypatch):
    times, V = wave
    victim = tmp_path / "existing.gif"
    victim.write_bytes(b"GIF89a-old")
    original = victim.read_bytes()
    monkeypatch.setattr(_render_module(), "_produce_bare", _throw_on_frame(2, RuntimeError))
    with pytest.warns(UserWarning, match="already existed"):
        with pytest.raises(RuntimeError):
            render(Video((times, V)), path=str(victim), format="gif", max_frames=5)
    assert victim.read_bytes() == original, "warned, but the bytes were clobbered anyway"


def test_failed_render_still_cleans_up_a_file_it_created(wave, tmp_path, monkeypatch):
    """The guard must still remove OUR partial output — only the caller's pre-existing file is spared."""
    times, V = wave
    dest = tmp_path / "ours.mp4"        # mp4 streams to disk, so a real partial file exists
    monkeypatch.setattr(_render_module(), "_produce_bare", _throw_on_frame(3, RuntimeError))
    with pytest.raises(RuntimeError):
        render(Video((times, V)), path=str(dest), max_frames=5)
    assert not dest.exists(), "a partial file WE created must be removed"


def test_multipanel_failure_before_the_writer_leaves_no_temp_file(wave, tmp_path, monkeypatch):
    """Round 1 hoisted the fallible work in render() but missed _render_panels()."""
    times, V = wave
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tempfile.gettempdir()))
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render([Video((times, V)), Video((times, V))], resolution="9000p", max_frames=2)
    assert not (set(os.listdir(tempfile.gettempdir())) - before), "panel path orphaned a temp file"


def test_save_releases_the_in_memory_payload(wave, tmp_path):
    """The documented invariant is that bytes are held ONLY while unsaved — otherwise read()
    keeps returning a stale copy of a file that may since have changed."""
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    assert info.data is not None
    info.save(tmp_path / "kept.mp4")
    assert info.data is None and info.saved
    assert info.read() == open(info.path, "rb").read()


def test_preview_mime_follows_the_actual_format(wave, tmp_path):
    """preview() follows path='s extension now, so the inline <img> must not hardcode PNG."""
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0, path=str(tmp_path / "f.jpg"))
    assert p.format in ("jpg", "jpeg") and "data:image/jpeg;base64," in p._repr_html_()


def test_size_cap_does_not_read_the_payload(wave, tmp_path, monkeypatch):
    """The cap must be decided from size_bytes. Reading first would pull a huge file into RAM
    only to decline to embed it — which is what the cap exists to avoid."""
    times, V = wave
    info = render(Video((times, V)), path=str(tmp_path / "big.mp4"), max_frames=3)
    monkeypatch.setattr(enc, "INLINE_MAX_BYTES", 8)

    def explode():
        raise AssertionError("_repr_html_ read the payload before checking the size cap")

    monkeypatch.setattr(info, "read", explode)
    assert "too large to embed" in info._repr_html_()


def test_writer_releases_its_handle_when_close_fails():
    """A failing close() cannot be retried (_closed is already set), so it must drop the encoder
    handle rather than leak the ffmpeg subprocess for the life of the process."""
    w = enc._Writer.__new__(enc._Writer)
    w.path, w.fps, w.backend, w.bitrate = "/nonexistent/x.gif", 20.0, "pillow-gif", None
    w.width = w.height = 0
    w._closed = False
    w.codec = "gif"
    w._impl = None

    class _Boom:
        def save(self, *a, **k):
            raise OSError("no such directory")

    w._frames = [_Boom()]
    with pytest.raises(OSError):
        w.close()
    assert w._frames is None and w._impl is None, "handles were not released"


def test_imagepath_size_cap_without_reading(wave, monkeypatch):
    """ImagePath had no inline cap at all — the anti-pattern the other two objects fixed."""
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0)
    monkeypatch.setattr(enc, "INLINE_MAX_BYTES", 8)

    def explode():
        raise AssertionError("read the payload before checking the size cap")

    monkeypatch.setattr(p, "read", explode)
    assert "too large to display inline" in p._repr_html_()


# ------------------------------------- audit round 4

def test_build_figure_closes_its_figure_when_it_raises():
    """_build_figure creates the figure, then does a lot of fallible work. The caller only closes
    the _FigState it RECEIVES, so a raise mid-build would leak the figure into pyplot's Gcf."""
    import matplotlib.axes
    times, V = _wave()
    clip = Video.annotated((times, V))
    cmap, norm, _, _ = clip.gradient.resolve(clip.masked_iter([0, 1]), field=clip.field)
    before = len(plt.get_fignums())

    # imshow runs AFTER plt.subplots, i.e. exactly the window where the figure exists but the
    # caller has not received it yet.
    real_imshow = matplotlib.axes.Axes.imshow

    def boom(self, *a, **k):
        raise RuntimeError("simulated failure inside the figure build")

    matplotlib.axes.Axes.imshow = boom
    try:
        with pytest.raises(RuntimeError):
            _render_module()._build_figure(
                clip, cmap, norm, colorbar_on=True, title=None, figsize=None, dpi=100,
                units=None, idx=[0, 1])
    finally:
        matplotlib.axes.Axes.imshow = real_imshow

    assert len(plt.get_fignums()) == before, "a failed _build_figure leaked a matplotlib figure"


def test_abort_drops_buffered_frames_without_writing(tmp_path):
    """abort() is the round-3 fix and was only reached transitively. Pin it directly."""
    dest = tmp_path / "never.gif"
    w = enc.open_writer(str(dest), 20.0, "pillow-gif", "gif")
    w.append(np.zeros((8, 8, 3), dtype=np.uint8))
    w.abort()
    assert not dest.exists(), "abort() wrote the file it was supposed to discard"
    assert w._frames is None and w._impl is None
    w.abort()                      # idempotent, and must not raise


def test_abort_never_raises_even_on_a_broken_backend():
    w = enc._Writer.__new__(enc._Writer)
    w.path, w.fps, w.backend, w.bitrate, w.codec = "/x.mp4", 20.0, "imageio-ffmpeg", None, "libx264"
    w.width = w.height = 0
    w._closed = False
    w._frames = None

    class _Boom:
        def close(self):
            raise OSError("subprocess already gone")

    w._impl = _Boom()
    w.abort()                      # must swallow: it runs inside an except block
    assert w._impl is None, "the handle was not released"


def test_isochrones_on_a_single_drawn_frame_warns_instead_of_lying(wave):
    """A 1-index draw of a multi-frame clip used to pass the `< 2 frames` guard and produce a
    constant LAT — an invisible, wrong overlay rather than an absent one."""
    times, V = wave
    clip = Video.annotated((times, V), isochrones=True)
    with pytest.warns(UserWarning, match="isochrones need >= 2 drawn frames"):
        lat = _render_module().isochrone_lat(clip, [3])
    assert np.isnan(lat).all()


# ------------------------------------- audit round 5

def test_isochrones_survive_a_single_drawn_frame_on_a_result(small_result):
    """The result-backed LAT comes from the FULL torch history, so it does NOT depend on how
    many frames are drawn. Round 4's guard was hoisted above this branch and silently dropped a
    computable overlay while warning that it was uncomputable."""
    clip = Video.annotated(small_result, isochrones=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any "needs >= 2 frames" warning fails the test
        lat = _render_module().isochrone_lat(clip, [3])
    assert np.isfinite(lat).any(), "a computable isochrone overlay was dropped"


def test_preview_with_isochrones_keeps_the_overlay(small_result, tmp_path):
    """The end-to-end shape of the regression: the documented 'check before a long encode' call."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = Video.annotated(small_result, isochrones=True).preview(
            t_ms=5.0, path=str(tmp_path / "prev.png"))
    assert _ok(p)


def test_multipanel_failure_after_the_writer_cleans_up_what_it_created(wave, tmp_path,
                                                                      monkeypatch):
    """The multi-panel copy of the cleanup guard had NO test entering its body — and this exact
    logic produced two separate data-loss defects in its single-clip sibling."""
    times, V = wave
    dest = tmp_path / "panels.mp4"
    # _setup_panel runs AFTER open_writer, so the guard body is genuinely entered.
    monkeypatch.setattr(_render_module(), "_setup_panel",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render([Video((times, V)), Video((times, V))], path=str(dest), max_frames=3)
    assert not dest.exists(), "the multi-panel guard left a partial file behind"


def test_multipanel_failure_never_deletes_a_preexisting_file(wave, tmp_path, monkeypatch):
    times, V = wave
    victim = tmp_path / "keep.mp4"
    victim.write_bytes(b"\x00\x00\x00\x18ftypmp42-original")
    original = victim.read_bytes()
    monkeypatch.setattr(_render_module(), "_setup_panel",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render([Video((times, V)), Video((times, V))], path=str(victim), max_frames=3)
    assert victim.exists() and victim.read_bytes() == original


@pytest.mark.parametrize("kw", [{"date": "2020-01-02"}, {"root": None}])
def test_date_and_root_are_save_triggers(wave, tmp_path, kw):
    """Both are documented destination keywords but neither had a single test."""
    times, V = wave
    if "root" in kw:
        kw = {"root": str(tmp_path)}
    info = render(Video((times, V)), "tv-trigger", max_frames=2, **kw)
    assert info.saved and _ok(info.path)
    if "date" in kw:
        assert "/2020-01-02/" in info.path
    else:
        assert info.path.startswith(str(tmp_path))


# ------------------------------------- audit round 6

def test_untouched_and_clobbered_warnings_are_distinct(wave, tmp_path, monkeypatch):
    """`opened` must reflect whether bytes actually reached the path — not merely that a writer
    object exists. pillow-gif writes only in close(), which abort() prevents, so a failed GIF
    render to an existing file must report it UNTOUCHED."""
    times, V = wave
    victim = tmp_path / "existing.gif"
    victim.write_bytes(b"GIF89a-original")
    monkeypatch.setattr(_render_module(), "_produce_bare", _throw_on_frame(2, RuntimeError))
    with pytest.warns(UserWarning, match="untouched"):
        with pytest.raises(RuntimeError):
            render(Video((times, V)), path=str(victim), format="gif", max_frames=5)
    assert victim.read_bytes() == b"GIF89a-original"


def test_streaming_failure_reports_the_file_as_opened(wave, tmp_path, monkeypatch):
    """The other side of the same predicate: mp4 streams, so bytes really do land mid-render."""
    times, V = wave
    victim = tmp_path / "existing.mp4"
    victim.write_bytes(b"\x00\x00\x00\x18ftypmp42-original")
    monkeypatch.setattr(_render_module(), "_produce_bare", _throw_on_frame(3, RuntimeError))
    with pytest.warns(UserWarning, match="after opening"):
        with pytest.raises(RuntimeError):
            render(Video((times, V)), path=str(victim), max_frames=5)


# ------------------------------------- audit round 7

def test_saving_onto_the_same_path_does_not_destroy_the_file(wave, tmp_path):
    """CRITICAL regression. open(path,"wb") truncates on open, so a saved result whose bytes live
    on disk would read back b"" and zero its own file — while still reporting saved=True."""
    times, V = wave
    info = render(Video((times, V)), path=str(tmp_path / "wave.mp4"), max_frames=4)
    original = open(info.path, "rb").read()
    assert len(original) > 0 and info.data is None, "precondition: bytes live on disk"

    info.save(info.path)

    assert open(info.path, "rb").read() == original, "save-onto-self destroyed the video"


def test_imagepath_saving_onto_itself_does_not_destroy_the_file(wave, tmp_path):
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0, path=str(tmp_path / "frame.png"))
    original = open(str(p), "rb").read()
    assert len(original) > 0
    p.save(str(p))
    assert open(str(p), "rb").read() == original, "save-onto-self destroyed the still"


# ------------------------------------- show(): the matplotlib contract

def test_show_displays_inline_in_a_notebook(wave, monkeypatch):
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: True)
    shown = []
    import IPython.display as ipd
    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj))
    assert info.show() is None, "show() must return None — returning self would double-embed"
    # display() called EXACTLY ONCE is the real double-embed guard: the harm is display() PLUS a
    # trailing-expression _repr_html_ when show() returns self.
    assert len(shown) == 1, "display must be called exactly once"
    assert "<video" in shown[0]._repr_html_()


def test_show_hands_a_file_to_the_os_player_in_a_terminal(wave, tmp_path, monkeypatch):
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    opened = {}
    monkeypatch.setattr(disp, "open_externally", lambda p: (opened.setdefault("p", p), True)[1])
    info.show()
    assert opened["p"].endswith(".mp4") and os.path.getsize(opened["p"]) > 0, \
        "an unsaved render must be materialised before a player can open it"


def test_show_reports_the_path_when_nothing_can_be_opened(wave, monkeypatch, capsys):
    """Headless / SSH: degrade to telling the user where the file is, never crash."""
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    monkeypatch.setattr(disp, "open_externally", lambda p: False)
    info.show()
    out = capsys.readouterr().out
    assert "No video player" in out and ".mp4" in out


def test_show_reuses_an_already_saved_file(wave, tmp_path, monkeypatch):
    times, V = wave
    info = render(Video((times, V)), path=str(tmp_path / "kept.mp4"), max_frames=3)
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    opened = {}
    monkeypatch.setattr(disp, "open_externally", lambda p: (opened.setdefault("p", p), True)[1])
    info.show()
    assert opened["p"] == info.path, "a saved render must not be copied to a temp file"


def test_show_does_not_mark_the_result_saved(wave, monkeypatch):
    """A scratch file materialised for the player is not a destination — .show() must not flip
    saved/path, or a later Lab record would think this render was written where the user asked."""
    times, V = wave
    info = render(Video((times, V)), max_frames=3)
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    monkeypatch.setattr(disp, "open_externally", lambda p: True)
    info.show()
    assert info.saved is False and info.path is None


def test_show_after_the_saved_file_was_deleted(wave, tmp_path, monkeypatch, capsys):
    """The new code re-reads (unlike the draft, which passed a stale path straight to the opener),
    so a saved-then-deleted file must degrade to a message, never raise."""
    times, V = wave
    info = render(Video((times, V)), path=str(tmp_path / "gone.mp4"), max_frames=3)
    os.remove(info.path)
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    monkeypatch.setattr(disp, "open_externally", lambda p: True)
    info.show()          # must not raise
    assert "video unavailable" in capsys.readouterr().out


def test_show_suffix_follows_the_codec(wave, monkeypatch):
    """The materialised temp file's extension must follow the codec, not hardcode .mp4 — else a
    GIF or VP9 payload opens in the wrong app."""
    times, V = wave
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    for fmt in ("gif", "webm"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # webm may DOWNGRADE to gif without ffmpeg
            info = render(Video((times, V)), format=fmt, max_frames=3)
        opened = {}
        monkeypatch.setattr(disp, "open_externally",
                            lambda p: (opened.setdefault("p", p), True)[1])
        info.show()
        expected = "." + enc._EXT_FOR_CODEC[info.codec]
        assert opened["p"].endswith(expected), \
            f"format={fmt} codec={info.codec}: opener got {opened['p']}, expected {expected}"


def test_imagepath_show_never_passes_the_unsaved_sentinel(wave, monkeypatch):
    """An unsaved ImagePath's string value is a human sentence, not a path — .show() must
    materialise from bytes and hand the opener a real file, never UNSAVED_TEXT."""
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0)          # unsaved → str(p) is UNSAVED_TEXT
    assert not p.saved
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    opened = {}
    monkeypatch.setattr(disp, "open_externally", lambda q: (opened.setdefault("p", q), True)[1])
    p.show()
    assert opened["p"] != enc.ImagePath.UNSAVED_TEXT
    assert opened["p"].endswith(".png") and os.path.exists(opened["p"])


def test_imagepath_show_uses_the_saved_file(wave, tmp_path, monkeypatch):
    times, V = wave
    p = Video((times, V)).preview(t_ms=5.0, path=str(tmp_path / "frame.png"))
    assert p.saved
    import cardiac_core._display as disp
    monkeypatch.setattr(disp, "in_notebook", lambda: False)
    opened = {}
    monkeypatch.setattr(disp, "open_externally", lambda q: (opened.setdefault("p", q), True)[1])
    assert p.show() is None
    assert opened["p"] == str(p)
