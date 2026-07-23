# PLAN: cardiac_core `Video` object — spec-first video rendering, full gradient control, multi-panel

Created: 2026-07-22
Engine(s): cardiac_core (media layer; **no solver changes at all**)
Research question: [engine_consolidation](../Research/Active/engine_consolidation/README.md)
Source: [engine_consolidation IDEALOG](../Research/Active/engine_consolidation/IDEALOG.md) — the
"Video-as-object (DESIGN LOCKED 2026-07-22)" bullet + a full read of the `boundary_conduction_speedup` /
`fig4c_sourcesink` render corpus (~20 scripts).

> **Design LOCKED (user, 2026-07-22).** A **spec object holds the description**, a **render function** turns it
> into frames, output lands at a `media_path` convention path. Locked: **full gradient control** (a reusable
> `Gradient`); **built-in streaming render**; **multi-panel**; overlays = **live front contour** + **static LAT
> isochrones** IN, **time stamp/colorbar optional**, **geometry outline DEFERRED**. **The zero-argument default**
> is a bare, unlabelled, 1080p render in the standard preset.
> **This is a PLAN — implementation is a separate, explicit user go (hard gate).**
>
> **AUDIT R1 → R5 (Opus adversarial, code-verified, each round a fresh cold-context agent).**
> R1 3C/11H/11M/8L · R2 3C/10H/14M/14L · R3 1C/6H/14M/12L · R4 0C/4H/19M/9L · R5 0C/3H/11M/17L ·
> **R6 0C/0H/11M/19L → CONVERGED.** All addressed — per-round disposition in the Mutation Log.
> R6's verdict: *"The document has CONVERGED and is ready for implementation."* R4/R5/R6 each independently
> re-verified full blueprint-structure compliance and ~25-40 source claims against the actual tree.
> **The R6 mediums/lows were then folded in as a final non-adversarial editing pass (R6's own recommendation).**
> **Implementation remains gated on an explicit user go.**
> R1's decisive catch: the plan's motivating defect was **factually false** (measured in a shell without conda's
> PATH). R2's decisive catches: **every `Verify` command was unexecutable** (`conda activate` does not work
> non-interactively), **no torch→numpy conversion existed anywhere** (crash on any CUDA result), Phase 2's shared
> colorbar was **unreachable** under Phase 1's own capability rule, and the range/`v_rest` computation still read
> **unmasked** frames. Full per-finding disposition in the Mutation Log.

## Objective
Give cardiac_core one obvious, built-in way to turn a run into a video, so research scripts stop hand-rolling
matplotlib animation. The unit of description is a `Video` spec object; `render()` streams it to an H.264 mp4 at
`media/{question}/[_sim_outputs/]videos/{date}/{slug}_NN.mp4`. Color is a **first-class reusable object**
(`Gradient`), because the range is a scientific choice: `render_audit_video.py` exists to show a **7.48 mV**
artifact that the default −90…40 mV scale renders essentially invisible (measured: 5.8% of the colormap vs 90.4%
under the zoom preset — a **15.7× visibility gain**). Multi-panel is native, because most polished prior art *is*
a comparison sharing one colorbar.

## THE DEFAULT — the common case (user, 2026-07-22)
> `r.video("slug")` with no other arguments must produce **the video we want 90% of the time**: the raw voltage
> field, **full-frame, no labels anywhere**, in the **standard color preset**, at **1080p**.

| Default | Value | Why |
|---|---|---|
| `style` | `"bare"` | The corpus's PURE-DATA convention: `add_axes([0,0,1,1])`, `set_axis_off()` (14 scripts). |
| labels | all off (`colorbar=None`, `show_time=None`, `title=None`, `label=None` → resolve OFF on bare) | User: "no label everywhere". |
| `gradient` | `Gradient.physiological()` — **viridis, −90…40 mV** | ⚠ open nit below. |
| `resolution` | `"1080p"` → 1920×1080 | User. (Current `viz` output is **600×300**.) |
| `fit` | `"contain"` — aspect preserved, letterbox-padded black | Distorting a wavefront misrepresents curvature. |
| `interpolation` | `"nearest"` | Honest about grid resolution. |
| `max_frames` | `300` | Keeps a 10k-save run watchable. |
| `question` | `"lab"` | Matches `viz`'s existing default and every cardiac_core test. |
| `bulk` | `True` | **Writes to gitignored `media/lab/_sim_outputs/…`** — regenerable by default; pass `bulk=False` to curate into git. |
| `fps` | `20.0` | Corpus median (12–20). |
| `format` | `"mp4"` | H.264/yuv420p. |

**ONE COSMETIC PREFERENCE, NOT A BLOCKER — viridis or inferno?** The bare/PURE-DATA family uses **viridis**
(pairing with the bare layout defaulted here); `cardiac_core.viz`'s current default is **inferno**. **This plan is
fully specified and executable as written: `Gradient.physiological()` = viridis.** If the user prefers inferno it
is a one-line change to that preset plus the expected-value in `::test_presets_resolve_expected_ranges` — no
structural consequence. Legacy `propagation_video` pins **inferno** regardless, so no existing output changes
either way. Flagged for the user; **does not gate implementation.**

**`fit=` resolves the full-bleed/letterbox tension.** The bare corpus scripts genuinely *stretch*
(`render_bc_videos.py` renders 41×21 into `figsize=(8,4)`, `aspect="auto"`). Default is **`fit="contain"`**;
`fit="stretch"` reproduces the corpus exactly and is offered, not hidden; `fit="cover"` fills by cropping.

## Motivating defect — CORRECTED after audit R1 (read carefully)
`viz.propagation_video` wraps `anim.save(path, writer="ffmpeg", fps=fps)` in a **bare `except Exception`** that
rewrites the output to `media_path(question, "images", f"{slug}-propagation", ext="gif")`. **The defect is a
silent, PATH-dependent FORMAT DOWNGRADE**: a caller asking for `.mp4` can receive a `.gif` at a different path and
extension, and codec/disk/permission failures are swallowed identically.

**⚑ AN EARLIER REVISION CLAIMED THIS FUNCTION ALWAYS PRODUCED GIFs. THAT WAS FALSE — DO NOT RE-INTRODUCE IT.**
Measured 2026-07-22:

| Invocation | `which ffmpeg` | `animation.writers.list()` |
|---|---|---|
| `/opt/miniforge3/bin/conda run -n heart-conduction python` | `…/envs/heart-conduction/bin/ffmpeg` | `['ffmpeg','ffmpeg_file','html','pillow']` |
| direct `…/envs/heart-conduction/bin/python` | `None` | `['html','pillow']` |

Same interpreter, different PATH. The false claim came from the second row. `propagation_video` **does** write real
H.264 mp4 in the user's environment (R1 ffprobe'd one written by `test_viz.py`).
**Design consequence (unchanged, smaller reason):** use the bundled `imageio_ffmpeg` binary so rendering is
PATH-independent, and make any fallback **loud**.

## Success Criteria
- [ ] **Zero-argument default** — `r.video("slug")` → full-frame 1920×1080 H.264 mp4, no axes/colorbar/title/time
      stamp, `Gradient.physiological()`, at a convention `media/` path.
- [ ] `Gradient` with `cmap` (name | color list | `Colormap`), `value_range`, `gamma`, `levels`, `bad`,
      `interpolation`, `v_rest`; five presets; `resolve()` → `(Colormap, Normalize, lo, hi)`; **never mutates a
      CALLER-SUPPLIED `Colormap` instance** (always `.copy()` — registered colormaps are already safe, see
      gotchas); **resolves over MASKED display values**, not raw frames.
- [ ] Degenerate guards: all-NaN → `(-90, 40)` + warn; `lo == hi` → widen + warn; never NaN in `VideoInfo`.
- [ ] `Video` spec (+ `Video.bare` default / `Video.annotated`); `repr` shows a **provisional** range.
- [ ] `video.preview(t_ms=…)` renders ONE frame to PNG.
- [ ] `render(...) -> VideoInfo` streams to disk; reports `.backend`; **no silent downgrade**.
- [ ] **Capability rule:** a **bare single-clip** render supports a burned-in time stamp and both interpolations but
      **cannot** draw a colorbar/contour/label — requesting those raises, naming `Video.annotated`.
      **Multi-panel always renders through the figure producer** (bare clips are promoted, warned once).
- [ ] `resolution` × `fit` with explicit upscale AND downscale paths; aspect never silently distorted by default.
- [ ] Masking routes through **`domain_mask` (True = ACTIVE)** ∪ `isfinite` — correct on LBM, where masked nodes
      stay **finite**.
- [ ] Multi-panel (Phase 2): shared gradient → ONE colorbar + ONE suptitle; identical `(Nx,Ny)` required; one
      stride for all panels; truncate to shortest.
- [ ] `SimulationResult.video("slug")`; `viz.propagation_video` delegates preserving framing/format/labels.
- [ ] **torch→numpy at ingest** (`.detach().cpu().numpy()`); float64 in, uint8 RGB out.
- [ ] The **cardiac_core suite** (`cardiac_core/tests/`, not the whole repo) shows **no NEW failures vs a
      baseline captured BEFORE the phase**; integrity goldens bit-identical.

## The five color intents (from the corpus — these ARE the presets)
| Preset | Range | Cmap | Prior art |
|---|---|---|---|
| `Gradient.physiological()` | −90 … 40 | `viridis` | the 10 bare PURE-DATA scripts |
| `Gradient.rest_anchored(vmax=40)` | V_rest … vmax | `inferno`, `bad="0.55"` | semicircle, propagation, oblique |
| `Gradient.zoom(span=8.0, below=0.3)` | V_rest−below … V_rest+span | `magma`, `bad="0.6"` | `render_audit_video.py` |
| `Gradient.diverging()` | −90 … 50 | `RdBu_r` | boundary-modes, stencil combos |
| `Gradient.autoscale()` | finite min … max | `viridis` | **Closest prior art, not an exact match:** the `diag_*` family (`diag_hourglass.py:177,181`) autoscales only the LOW end (`vmin=float(V.min())`, `vmax=40` fixed) and uses **inferno**+`set_bad("0.55")`; `video_horizontal_longrun.py` is viridis but fully FIXED. This preset is the generalisation of both, not a reproduction of either. |

`gamma` (PowerNorm) and `levels` (banding) appear in NO existing script — the new gradient-shaping knobs.

## Architecture Changes
- NEW `cardiac_core/video/` package (mirrors `fields/`): `encoders.py`, `gradient.py`, `clip.py`, `render.py`,
  `__init__.py` (exports `Video`, `Gradient`, `render`, `VideoInfo`, **and the alias `render_video = render`**).
- MOD `cardiac_core/run.py` — `SimulationResult.video(slug, **kw)`.
- MOD `cardiac_core/viz.py` — `propagation_video` delegates (signature + `str` return preserved).
- MOD `cardiac_core/__init__.py` — `_LAZY` += `Video`, `Gradient`, `render`, `render_video`, `VideoInfo`
  (**all five**; `render` AND `render_video` both, or the cheatsheet import fails). **⚠ CONFLICT POINT.**
- MOD `cardiac_core/API_CHEATSHEET.md` §10. **⚠ CONFLICT POINT.**
- NEW `cardiac_core/tests/test_video.py`.
- MOD `cardiac_core/tests/test_self_contained.py` — add `"video"` to
  `test_subpackage_importable`'s parametrize list. **Verified current list:**
  `["ionic", "mesh", "stimulus", "_monodomain", "_bidomain", "_lbm"]` — note `fields` is **absent**, a
  pre-existing gap from the analysis-fields ship. Add `"video"`, and add `"fields"` while there (one-line
  drive-by fix; if it fails, that is a real pre-existing bug worth surfacing, not something to silently skip).
- MOD `Research/Active/engine_consolidation/API_REFERENCE.md` — add `Video`/`Gradient`/`render`/`VideoInfo`.
  **Verified:** the file exists (~20 KB) and the parallel STIM plan updates it (Step 2.1: "update
  `API_REFERENCE.md` if it exists"; 31 `Stim` mentions already present). **⚠ It is ALSO contended** — it was
  modified minutes ago by that session; treat it like `API_CHEATSHEET.md` (surgical, LAST, re-read first).
- **Plan location note:** this file lives at `cardiac_core/VIDEO_OBJECT_PLAN.md` rather than
  `Research/Active/{question}/PLAN.md`, matching its sibling `STIM_OBJECT_PLAN.md`. The Final Cleanup archive
  step already uses the actual path.
- **Runtime deps:** this adds `imageio` + `imageio-ffmpeg` (and optionally `cv2`) to cardiac_core's effective
  runtime requirements. `pyproject.toml` declares only `mcp>=1.2.0` — torch/numpy/scipy/matplotlib are ALREADY
  undeclared, so this is consistent with existing practice; note it, do not fix it here (packaging itself is
  fine: `include = ["cardiac_core*", "cardiac_mcp*"]` picks up `cardiac_core.video` automatically).
- OUT OF SCOPE: geometry-outline overlay; any solver/engine/analysis change.

## Optional reference: a reverted prototype
A single-module prototype was written earlier and **reverted out of the shared working tree**; it is NOT in the
repo, NOT in git, and its scratchpad copy may be gone. **Every step below is self-contained and must NOT depend on
it.** Its load-bearing empirical results (all re-verified under `conda run`):
- imageio + bundled `imageio_ffmpeg` (**package 0.6.0**, bundled binary **ffmpeg-linux-x86_64-v7.0.2**) writes a
  real `ftyp`/H.264/yuv420p file; `get_writer(format="FFMPEG", fps=, codec="libx264", quality=8,
  pixelformat="yuv420p", macro_block_size=1)` accepts every one of those kwargs with **no warnings**.
- libx264 rejects odd dimensions — pad. Verified 409×205 → 410×206.
- Build-figure-once + `set_data` = **7.9 ms/frame**; rebuilding = 12.7; direct colormap (no figure) = **0.10**.
- (Historical: imageio's pillow GIF plugin deprecates `fps=` in favour of `duration=1000/fps` ms. **Moot now** —
  after the R3 fix the GIF path is written with PIL directly, not imageio; the `duration` unit is the same.)

## Known Failures (from IDEALOG) / gotchas
- **Verify env-dependent claims with `/opt/miniforge3/bin/conda run -n heart-conduction`.** Bare `conda` is not on
  the non-interactive PATH; `conda activate` does not work there at all (so `X && conda activate Y && cmd`
  short-circuits and never runs `cmd`); and the **direct env python hides ffmpeg**. This produced R1's false
  CRITICAL. All `Verify` blocks below use the full-path `conda run` form.
- **Never let a fallback be silent** (the actual defect). Warn, and report `backend` on `VideoInfo`.
- **A GIF is an IMAGE in the media convention.** `media_path` validates ext against kind and RAISES on mismatch —
  so the **format/backend decision must happen BEFORE the path is built** (see Step 1.1).
- **`media_path`'s `NN` contract is get-path-then-save-immediately.** Never pre-compute a batch of paths; an `.mp4`
  at `_01` does not reserve the `.gif` slot.
- **Stream; never accumulate RGB.** 600 frames of 1920×1080 RGB ≈ 3.6 GB. (Range resolution needs a data pass —
  do it streaming, see Step 1.2 — a full strided stack is ~460 MB/panel and is NOT acceptable.)
  **Scope of the claim:** `Video.__post_init__` DOES materialise the `(T,Nx,Ny)` float64 history once (a real
  copy for float32/CUDA inputs). The streaming rule is about not making a SECOND copy per render and not
  accumulating the far larger RGB side — it is not an end-to-end constant-memory guarantee.
- **`domain_mask` polarity is True = ACTIVE** (`mesh/structured.py`). The corpus masks the COMPLEMENT
  (`np.ma.array(V.T, mask=obstacle.T)`), so inversion is an easy bug.
- **LBM masked nodes stay FINITE.** B8's NaN-fill only patched `flat_to_grid` (mono + bidomain);
  `test_api_hardening.py::test_lbm_masked_hole_nonconducting` asserts LBM obstacles remain finite. Masking by
  `isfinite` alone paints an LBM obstacle as real voltage — **and the same contamination hits `value_range="auto"`
  and `v_rest` inference**, so the color path must use masked values too.
- **torch→numpy is mandatory.** `result.Vm`/`times`/`domain_mask` are `torch.Tensor`, possibly on CUDA;
  `np.asarray` on a CUDA tensor raises `TypeError`. Use `.detach().cpu().numpy()` (as `viz._vm_numpy` does).
  **Conversely, `analysis.activation_time` is torch-only** (`V.gather`, `torch.where`) — feed it the ORIGINAL
  tensors and convert its output, never the numpy frames.
- **Orientation — VERIFIED, not assumed.** Display is `V[t].T` with `origin="lower"`; the no-matplotlib producer
  emits `np.flipud(V[t].T)`. Confirmed pixel-identical at both corners against a real
  `imshow(V.T, origin="lower", interpolation="nearest")` render. For contours,
  `meshgrid(x, y, indexing="ij")` pairs with the **untransposed** `V` (`render_oblique_videos.py:58`).
- **Always `.copy()` before `set_bad/over/under` — but for the RIGHT reason (corrected in R3).** `set_bad` mutates
  in place. **It does NOT contaminate matplotlib's globals**: `plt.get_cmap(name)` returns a *fresh copy* per call
  (verified: `plt.get_cmap("magma") is plt.get_cmap("magma")` → `False`; mutating the result leaves both
  `plt.get_cmap("magma")` and `matplotlib.colormaps["magma"]` unchanged). **The real hazard is a caller-supplied
  `Colormap` instance** — `Gradient(cmap=my_cmap)` without `.copy()` mutates the *caller's own object* (verified:
  it does). So `.copy()` stays mandatory, and the test must target the caller-supplied-instance case, not a
  name-based one. (An earlier revision asserted global contamination — FALSE, inferred from "no exception raised".)
- **`io.load_result` returns a 4-tuple** `(times, V, phi_e, metadata)` — not a result object; no `dx/dy/domain_mask`;
  a stored float32 `V` round-trips as float32 → cast to float64, fall back to node axes.
- **Downscaling must not use nearest** — it aliases wavefronts. Use `PIL.Image.BOX` (area-average) for scale < 1.
  Verified present in PIL 12.1.0 alongside `NEAREST`/`BILINEAR`/`LANCZOS`. (PIL has **no `AREA`**; that is cv2's name.)
- **A 1080p burned-in time stamp needs a scalable font.** PIL's default bitmap font is ~11 px and invisible on a
  1920×1080 frame. Use matplotlib's bundled `mpl-data/fonts/ttf/DejaVuSans.ttf` via `ImageFont.truetype` (no new
  dependency), sized ~`canvas_height/40`, and **burn it AFTER the canvas fit**, never before (otherwise it is
  drawn at grid scale and blown up 48×).
- **`Grid(N, 1)` cables are supported** (B5) — a 1-node axis must be given a minimum displayed thickness.
- **`test_cheatsheet_examples_execute` only execs blocks whose FIRST LINE is `# runnable-canary`**
  (`tests/test_usability_fixes.py`) — that is §12, NOT §10. §10 needs its own test.
- **⚠ PARALLEL-SESSION STATUS (re-checked at R4 — the surface has largely closed).** The Stim work has LANDED:
  `'Stim': 'stimulus.stim'` is in `_LAZY`, **Stim Phase 2 is done** (the dict stimulus path already emits a
  `DeprecationWarning`, `api.py:1301`), and `cardiac_core/` reports clean — including `API_CHEATSHEET.md`,
  `api.py`, `protocols.py`, tests, and `API_REFERENCE.md`. So the files this plan touches are no longer dirty;
  still **re-read each immediately before editing** (that session may resume). **Write new fixtures with `Stim`,
  not the dict form** — the dict form now warns, and a `filterwarnings("error")` run would fail on it.
  **A THIRD in-flight plan is GATED ON THIS ONE:** `cardiac_core/tutorials/PLAN.md` states no notebook is written
  until the `Video`/`Gradient` pipeline lands — so this plan is on that critical path.
- **float64 in, uint8 out.** No float32 between ingest and the final RGB buffer.

---

## Phase 1: `Gradient` + `Video` + single-panel `render` + encoders

**Goal**: the whole user-facing feature for ONE clip — spec object, full color control, the zero-argument 1080p
bare default, streaming render, overlays on the annotated style. Independently deliverable.
**Tier**: large
**Estimated scope**: 1 new package (5 modules) + result hook + viz delegation + ~50 tests.

### Phase Context
`viz.propagation_video(result, slug, *, question="lab", fps=20, vmin=-90.0, vmax=40.0, cmap="inferno",
bulk=False) -> str` builds a `FuncAnimation` with axes + colorbar + title at `figsize=(6,3)`, `dpi=100` (600×300),
**node-index axes** (`"x (nodes)"`/`"y (nodes)"`), **no masking**, and saves via the bare-except path above.
`media_path(question, kind, slug, ext="png", *, date=None, bulk=False, root=None)` owns the path convention.
`SimulationResult` carries `times, Vm, phi_e, dx, dy, ionic_states, domain_mask, boundary_mode, Cm, chi,
conductivity, ionic_model, cell_type` — **all torch**, no `.video()` yet.
`analysis.activation_time(Vm, times, **kw) -> (Nx,Ny)` is **torch-only**.
`__init__.py` resolves public names through a `_LAZY` `{name: submodule}` map via PEP-562 `__getattr__` — the
exported name **must exist in the submodule** or `getattr` raises `AttributeError`.
**Phase 1 is SINGLE-CLIP ONLY**: passing a list raises `NotImplementedError("multi-panel lands in Phase 2")`.
**No solver interaction anywhere.**

---

### Step 1.1: `video/encoders.py` — backend selection, canvas fit, `VideoInfo`
**Model**: opus

#### Read First
- `cardiac_core/media.py` — `media_path` signature, ext/kind validation, the `NN` contract.
- `cardiac_core/fields/__init__.py` — package layout precedent.

#### Why
Everything depends on writing a real file at the right path with the right dimensions. **Backend selection must
precede path construction**: the fallback to GIF changes both the extension and the `kind` directory, and
`media_path` raises on an ext/kind mismatch — so a writer that discovers its own fallback *after* the path exists
cannot fix it (R2 CRITICAL-adjacent HIGH).

#### Implementation Spec
**Files to create:** `cardiac_core/video/__init__.py` (**a STUB at this step** — an empty file or a docstring
only; the full five-name `__all__` is written in Step 1.4, since exporting names whose modules do not exist yet
would break `import cardiac_core.video`), `cardiac_core/video/encoders.py`,
and `cardiac_core/tests/test_video.py` (created here; Steps 1.2-1.5, 2.1 and 3.1-3.2 all append to it).
**Interfaces:**
```python
RESOLUTIONS = {"720p": (1280,720), "1080p": (1920,1080), "1440p": (2560,1440), "4k": (3840,2160)}

@dataclass
class VideoInfo:                       # str-like: __fspath__ / __str__ -> .path
    path: str; n_frames: int; fps: float; backend: str
    codec: str            # "libx264" | "libvpx-vp9" | "mp4v" (opencv) | "gif" (pillow) — never None
    bitrate: Optional[str] = None   # the RESOLVED rate actually handed to the encoder ("2M" for webm/VP9).
                                    # Surfaced because it is otherwise untestable: the VP9
                                    # "Neither bitrate nor constrained quality" message is written by the
                                    # ffmpeg SUBPROCESS, and imageio defaults ffmpeg_log_level="quiet"
                                    # (plugins/ffmpeg.py:552), so neither pytest.warns NOR capfd can see it.
    width: int; height: int; duration_s: float; vmin: float; vmax: float
    stride: int; size_bytes: int

def select_backend(fmt: str) -> tuple[str, str, str]   # -> (backend, ext, kind) BEFORE any path is built
def resolve_canvas(resolution) -> Optional[tuple[int,int]]   # name | (w,h) | None -> None means "no canvas"
def fit_frame(rgb, canvas, fit="contain", interpolation="nearest", pad=(0,0,0)) -> np.ndarray
    # NOTE: `Gradient.interpolation` does double duty — a matplotlib name on imshow (data space) AND
    # the PIL resample selector here (RGB space, post-colormap). Only "nearest"/"bilinear" are mapped;
    # VALIDATE in Gradient.__post_init__ and reject other matplotlib names (bicubic/spline16/hanning/…)
    # rather than silently collapsing them to BILINEAR in the fit.
def burn_timestamp(rgb, text) -> np.ndarray            # DejaVuSans truetype, size ~H/40, AFTER fit
class _Writer:                       # uniform interface over all three backends
    path: str; backend: str; codec: str
    def append(self, rgb: np.ndarray) -> None   # pads to even dims, records the PADDED dims
    def close(self) -> None                     # idempotent
def open_writer(path, fps, backend, fmt, *, quality=8, bitrate=None) -> _Writer
    # `fmt` is REQUIRED: codec + pixelformat are looked up from CODECS[fmt]. Without it the
    # CODECS map is dead code and webm silently gets libx264+yuv420p (an invalid combination).
```
**Codec per format (verified against the bundled ffmpeg's `-encoders`):** `mp4` → `libx264` + `pixelformat=
"yuv420p"`; `webm` → `libvpx-vp9` (present; a webm writes fine with or without `yuv420p`). **VP9 caveat:** it warns
`"Neither bitrate nor constrained quality specified, using default CRF of 32"` — imageio's `quality=` does NOT map
to a VP9 rate control, so pass an explicit `bitrate` (or `output_params=["-crf","30"]`) on the webm path.
**Font (verified):** `os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")` exists; at
`size = H//40` it renders a **20 px**-tall stamp on a 1080p frame, versus **8 px** for PIL's default bitmap font.
**Guard:** clamp to `size = max(8, H//40)` — on a bare clip with `resolution=None` the frame is grid-sized
(H can be ~21), and `ImageFont.truetype(size=0)` is not a valid call.

#### Pseudocode
```
CODECS = {"mp4": ("libx264", "yuv420p"), "webm": ("libvpx-vp9", None)}   # per-container; verified available
#   NOTE: the webm `None` is documentation, not suppression — imageio's writer DEFAULTS to
#   pixelformat="yuv420p" (plugins/ffmpeg.py:548), so omitting the kwarg re-supplies it. Harmless
#   (webm was verified to write correctly either way); do not expect `None` to disable it.

select_backend(fmt):                       # probe availability FIRST, so the path is built once, correctly
    if fmt not in ("mp4", "webm", "gif"):
        raise ValueError(f"format must be 'mp4', 'webm' or 'gif', got {fmt!r}")
        # NOTE: media_path's _VIDEO_EXT also allows 'mov'/'avi', but we do NOT expose them —
        # otherwise they reach open_writer and get libx264+yuv420p in a container that rejects it.
    if fmt == "gif":   return ("pillow-gif", "gif", "images")
    if importable("imageio") and importable("imageio_ffmpeg"):
                       return ("imageio-ffmpeg", fmt, "videos")     # codec/pixfmt from CODECS[fmt]
    if fmt == "webm":  warn(LOUD, "no ffmpeg backend; webm cannot be produced by OpenCV — DOWNGRADING to GIF")
                       return ("pillow-gif", "gif", "images")
    if importable("cv2"):  warn("imageio-ffmpeg unavailable; falling back to OpenCV mp4v"); return ("opencv","mp4","videos")
    warn(LOUD, "no video encoder available; DOWNGRADING to animated GIF"); return ("pillow-gif","gif","images")
    # caller then does: media_path(question, kind, slug, ext=ext, bulk=bulk)

resolve_canvas(resolution):
    None -> None                            # legacy/figure path: canvas comes from figsize x dpi; skip fit entirely
    str  -> RESOLUTIONS[name]; tuple -> as given;  force both dims EVEN

fit_frame(rgb(h,w,3), (W,H), fit, interp):
    if fit == "stretch":  sx, sy = W/w, H/h
    else:                 s = max(W/w, H/h) if fit=="cover" else min(W/w, H/h); sx = sy = s
    if round(h*sy) < 2: sy = 2.0/h          # Grid(N,1): 1-node y axis
    if round(w*sx) < 2: sx = 2.0/w          # and the transposed case: a 1-node x axis
    smin = min(sx, sy)                      # NOTE: defined for ALL branches incl. stretch (R2 MED: NameError)
    resample = Image.NEAREST if (interp=="nearest" and smin >= 1) else \
               Image.BOX     if smin < 1 else \
               Image.BILINEAR               # BOX = area-average; PIL has no AREA
    out = Image.fromarray(rgb).resize((round(w*sx), round(h*sy)), resample)
    if fit == "cover": center-crop to (W,H)
    else:              crop 1 px if round(w*sx) or round(h*sy) overshot (rounding can exceed W/H by 1),
                       then pad symmetrically with `pad` to exactly (W,H)
    return asarray(out)

open_writer(path, fps, backend, fmt, *, quality=8, bitrate=None):
    codec, pixfmt = CODECS.get(fmt, (None, None))
    "imageio-ffmpeg" -> iio.get_writer(path, format="FFMPEG", mode="I", fps=fps, codec=codec,
                                       quality=quality, macro_block_size=1,
                                       **({'pixelformat': pixfmt} if pixfmt else {}),
                                       **({'bitrate': bitrate} if bitrate else {}))
                        # webm/VP9: pixfmt is None and an explicit bitrate MUST be supplied,
                        # else VP9 warns 'Neither bitrate nor constrained quality specified'
    "opencv"         -> cv2.VideoWriter(path, fourcc("mp4v"), fps, (w,h)) lazily on first append; BGR
    "pillow-gif"     -> **PIL ONLY, never imageio.** This backend is the last resort selected precisely WHEN
                        imageio is unimportable, so an imageio-based implementation would crash in the only
                        environment that chooses it. Buffer PIL Images; on close:
                          imgs[0].save(path, save_all=True, append_images=imgs[1:],
                                       duration=round(1000/fps), loop=0, optimize=False)
                        MEMORY EXCEPTION (documented): GIF must hold all frames for the palette, so this path
                        alone accumulates. The cap is applied in `render` when computing `stride`
                        (None-safe: `eff_max = 200 if max_frames is None else min(max_frames, 200)`),
                        NOT by dropping frames inside the writer — a post-hoc
                        drop would double playback speed and desync VideoInfo.n_frames/duration_s.
    append(rgb): pad to EVEN dims; record the PADDED dims as reported width/height
```

#### Test Spec
- `::test_writes_real_mp4` — 12 synthetic frames. Expected: bytes[4:8] == `b"ftyp"`, size > 0.
- `::test_reported_dims_are_post_padding` — force odd pre-pad dims. Expected: `info.width/height` even AND equal
  to the shape of frame 0 read back via `imageio.get_reader`.
- `::test_select_backend_precedes_path` — monkeypatch imageio_ffmpeg + cv2 unavailable. Expected:
  `select_backend("mp4") == ("pillow-gif","gif","images")` and a `UserWarning` matching "DOWNGRAD".
- `::test_gif_fallback_works_without_imageio` — monkeypatch **both `imageio` AND `cv2`** unimportable, then
  render. Expected: a real, non-empty GIF. **`cv2` matters:** opencv 4.13 IS installed in this env, so patching
  only `imageio` yields the OpenCV mp4 backend and the PIL path never runs — the R3 "PIL-only last resort"
  guarantee would go untested.
- `::test_invalid_format_raises` — `format="mov"` / `"avi"` / `"mkv"`. Expected: `ValueError` naming the three
  supported formats (they must not reach `open_writer` and get an incompatible codec).
- `::test_fit_contain_preserves_aspect` — 200×50 → (1920,1080). Expected: content 1920×480, black pad to 1080,
  aspect within 1e-6 of 4.0.
- `::test_fit_stretch_fills_and_does_not_crash` — same input, `fit="stretch"`. Expected: exactly 1920×1080, no
  pad rows (guards the `smin` NameError).
- `::test_downscale_uses_box_not_nearest` — a 4000×2000 two-tone checkerboard → 1080p. Expected: output contains
  values strictly between the two source values (proof of averaging).
- `::test_degenerate_single_row` — an (N,1) grid. Expected: rendered height ≥ 2, no exception.
- `::test_burn_timestamp_is_legible_at_1080p` — burn on a 1920×1080 frame. Expected: the drawn glyph bbox spans
  ≥ 15 px in height. **Measured:** DejaVuSans at `size=H//40` gives **20 px**; PIL's default bitmap font gives
  **8 px** — so this test genuinely discriminates truetype from the default.
- `::test_webm_uses_vp9_with_explicit_rate` — `format="webm"`. Expected: a non-empty file, `info.codec ==
  "libvpx-vp9"`, and **`info.bitrate is not None`**. **Assertion mechanism matters:** the VP9 "Neither bitrate nor
  constrained quality" message comes from the **ffmpeg subprocess**, and imageio sets `ffmpeg_log_level="quiet"`
  by default — so `pytest.warns` cannot see it and neither can `capfd`. Assert on `VideoInfo.bitrate`, which
  exists precisely to make this checkable.

#### Checklist
- [ ] `video/` package dir + `encoders.py`
- [ ] `VideoInfo` with `__fspath__`/`__str__`
- [ ] `select_backend` returning `(backend, ext, kind)` BEFORE any path exists
- [ ] `resolve_canvas` incl. the `None` case
- [ ] `fit_frame` with contain/stretch/cover, `smin` defined on all branches, BOX downscale, min-thickness
- [ ] `burn_timestamp` with DejaVuSans truetype
- [ ] `open_writer` per backend; report PADDED dims

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_video.py \
  -k "mp4 or fit or backend or dims or degenerate or downscale or timestamp or format or gif or webm" -v
```

#### Exit Criteria
- [ ] A real, playable H.264 mp4 is produced without `ffmpeg` on `PATH`
- [ ] Reported dimensions match the encoded file exactly
- [ ] Every fallback warns AND is chosen before the path is built

#### Risk
`imageio.get_reader` on a just-written file can lag on some filesystems — mitigate by closing the writer before
reading back in the dims test.

---

### Step 1.2: `video/gradient.py` — the color object
**Model**: opus

#### Read First
- `Research/Active/boundary_conduction_speedup/render_audit_video.py:53-54` — the zoom window + `set_bad("0.6")`.
- `Research/Active/boundary_conduction_speedup/render_semicircle_video.py:45-46` (`.copy()` + `set_bad("0.55")`) and `:52-53` (the rest-anchored `vmin=V_rest, vmax=40`).
- `Research/Active/boundary_conduction_speedup/render_bc_videos.py:37-39` — physiological viridis −90/40.

#### Why
The range is a scientific choice (5.8% vs 90.4% of the colormap for the same artifact). Making `Gradient` reusable
is the precondition for panels sharing one mapping. **It must resolve over MASKED values**: on LBM the obstacle
nodes are finite, so an `auto` range or a `v_rest` median computed on raw frames is contaminated by non-tissue
values (R2 HIGH).

#### Implementation Spec
**Files to create:** `cardiac_core/video/gradient.py`
**Interfaces:**
```python
@dataclass(frozen=True, eq=False)     # eq=False: a list/Colormap cmap breaks dataclass __eq__/__hash__
class Gradient:
    cmap: Union[str, list, Colormap] = "viridis"
    value_range: Union[str, tuple] = "physiological"   # renamed from `range` (shadowed the builtin)
        # legal: "physiological" | "rest" | "zoom" | "auto" | "auto99" | (lo, hi)
    gamma: float = 1.0
    levels: Optional[int] = None
    bad: str = "0.55"
    interpolation: str = "nearest"
    v_rest: Optional[float] = None
    rest_vmax: float = 40.0        # carries Gradient.rest_anchored(vmax=…); resolve must NOT hard-code 40
    zoom_span: float = 8.0
    zoom_below: float = 0.3
    # __post_init__ VALIDATES: value_range in the legal enum (or a 2-tuple), and
    #   interpolation in {"nearest","bilinear"} — both raise ValueError rather than failing later.
    # `Video.__post_init__` likewise validates style in {bare,annotated}, aspect in {equal,auto} and
    #   units in {auto,cm,nodes}; `render()` validates fit in {contain,stretch,cover}. Unvalidated,
    #   `style="anotated"` silently renders BARE (losing axes+colorbar) and `fit="containn"` silently
    #   behaves as contain — the same silent-no-op class this plan exists to remove.
    def key(self) -> tuple   # COMPARABLE identity for "do these panels share a gradient?" (compared with
                             # ==, never hashed — a list-valued `cmap` would make the tuple unhashable)
    def resolve(self, masked_values, *, field="Vm") -> tuple[Colormap, Normalize, float, float]
    # presets: physiological() rest_anchored(vmax=40) zoom(span=8.0, below=0.3) diverging() autoscale()
```
`Gradient.zoom(...)` sets `value_range="zoom"`. The `("rest", span)` tuple form is REMOVED (ambiguous with `(lo,hi)`).
`resolve` receives an **iterator/stack of already-masked display values** — never raw frames.
`field` is compared with `isinstance(field, str) and field == "phi_e"` (it may legally be an ndarray).

#### Pseudocode
```
resolve(masked_values, field):
    # ONE pass. The iterator is consumed EXACTLY ONCE — it may be a generator, so a second
    # traversal would silently yield nothing (this broke rest/zoom in an earlier revision).
    stats = streaming_stats(masked_values)
    #   accumulates: count, min, max, and a DETERMINISTIC per-frame subsample for percentiles.
    #   One-pass-safe rule (the total element count is NOT knowable up front from an iterator):
    #     per frame, take a fixed stride m = max(1, frame.size // 20_000) -> at most 20k values
    #     per frame, appended in order. Deterministic (no RNG), bounded, and computable from the
    #     frame alone. Two renders of identical data therefore give identical vmin/vmax.
    #   AND stats.first_frame_vals = the finite values of the FIRST frame, captured during the
    #   same pass, because infer_v_rest needs them and cannot re-read the iterator.
    if value_range is a 2-tuple:        lo, hi = value_range      # an EXPLICIT range always wins,
                                                                  # even on all-NaN data
    elif stats.count == 0:
        warn("no finite unmasked data; falling back to (-90, 40)"); lo, hi = -90.0, 40.0
    elif value_range == "physiological":lo, hi = -90.0, 40.0
    elif value_range == "auto":         lo, hi = stats.min, stats.max
    elif value_range == "auto99":       lo, hi = pct(stats.sample, 0.5), pct(stats.sample, 99.5)
    elif value_range in ("rest","zoom"):
        vr = v_rest if v_rest is not None else infer_v_rest(stats, field)   # stats only — NOT the iterator
        lo, hi = (vr - zoom_below, vr + zoom_span) if value_range == "zoom" else (vr, rest_vmax)
    else: raise ValueError(f"unknown value_range {value_range!r}")   # never fall through with lo/hi unbound
    if not isfinite(lo) or not isfinite(hi): warn; lo, hi = -90.0, 40.0
    if hi <= lo:  warn("degenerate range"); lo, hi = lo - 0.5, lo + 0.5
    cm = get_cmap_from(cmap).copy()              # ALWAYS copy before set_bad (see the gotcha: the real
                                             # hazard is a caller-supplied Colormap instance)
    if levels: cm = cm.resampled(levels)         # verified: resampled(8) -> exactly 8 unique colors
    cm.set_bad(bad)
    norm = PowerNorm(gamma, lo, hi) if gamma != 1.0 else Normalize(lo, hi)
        # VERIFIED: PowerNorm handles NEGATIVE vmin (it normalizes to [0,1] before the power).
        # PowerNorm(2.0,-90,40)(-25.0) == 0.25 exactly. No custom Normalize subclass needed.
    return cm, norm, lo, hi

infer_v_rest(stats, field):                    # reads ONLY the single-pass stats
    if isinstance(field,str) and field == "phi_e":
        raise ValueError("value_range='rest'/'zoom' needs an explicit v_rest for phi_e")
    vals = stats.first_frame_vals              # captured during the same pass
    if vals.size == 0: warn; return -85.0
    if (pct(vals,95) - pct(vals,5)) > 5.0:        # frame 0 already depolarized / mid-run window
        warn("frame 0 is not at rest; using the global finite minimum"); return stats.min
    return median(vals)

get_cmap_from(c): str -> matplotlib.colormaps[c]  (NOT plt.get_cmap — gradient.py must not import pyplot,
                        which would make the Agg-backend guarantee depend on import order)
                | list -> LinearSegmentedColormap.from_list("custom", c) | Colormap -> c
```

#### Test Spec
- `::test_presets_resolve_expected_ranges` — **V_rest = −82.0** synthetic (deliberately NOT −85: −85.0 is
  `infer_v_rest`'s *fallback constant*, so an −85 fixture cannot distinguish a working inference from a silently
  failed one — the exact blind spot that hid the exhausted-iterator bug). Expected: physiological (−90,40);
  rest_anchored (−82,40); zoom (−82.3,−74.0); diverging (−90,50); autoscale = masked min/max. Tol 1e-6.
- `::test_resolve_consumes_iterator_once` — pass a **generator** (not a list) to `resolve` with
  `Gradient.rest_anchored()`. Expected: the resolved `lo` equals the fixture V_rest, NOT the −85.0 fallback,
  proving `first_frame_vals` was captured in the single pass.
- `::test_explicit_range_wins_on_all_nan` — all-NaN input with `value_range=(-70, 10)`. Expected: exactly
  (−70, 10), no fallback override.
- `::test_auto99_is_deterministic` — resolve the same data twice. Expected: byte-identical `(lo, hi)` (no RNG).
- `::test_auto99_within_auto` — data with outliers. Expected: `auto.lo <= auto99.lo` and `auto99.hi <= auto.hi`
  (the percentile window is strictly inside the full range).
- `::test_interpolation_validated` — `Gradient(interpolation="bicubic")`. Expected: `ValueError` — only
  `"nearest"`/`"bilinear"` are supported (they must map to BOTH imshow and the PIL resampler).
- `::test_unknown_value_range_raises` — `Gradient(value_range="physiologicl")`. Expected: `ValueError` from
  `__post_init__`, not an unbound-local NameError deep inside `resolve`.
- `::test_rest_anchored_vmax_is_honoured` — `Gradient.rest_anchored(vmax=30)`. Expected: `hi == 30.0`.
- `::test_custom_color_list_builds_gradient` — `cmap=["black","red","white"]`. Expected: endpoints (0,0,0)/(1,1,1).
- `::test_gamma_shifts_midpoint_with_negative_vmin` — `gamma=2.0`, range (−90,40). Expected: `norm(-25.0) == 0.25`.
- `::test_levels_bands` — `levels=8`. Expected: exactly 8 unique colors.
- `::test_all_nan_falls_back_and_warns` — Expected: `pytest.warns(UserWarning)`, (−90,40), no NaN.
- `::test_flat_field_widens_range` — constant field. Expected: `hi > lo`, warns.
- `::test_copy_protects_caller_supplied_colormap` — pass a **`Colormap` instance**, resolve, then assert the
  caller's own object still returns its original `get_bad()`. (Guards the real hazard; `plt.get_cmap(name)`
  already returns a fresh object per call, so a name-based test guards nothing.)
- `::test_range_uses_masked_values_only` — an LBM-style array whose **finite** obstacle nodes are +200 mV, masked
  out by `domain_mask`. Expected: `autoscale()` max reflects tissue only, NOT 200 (the R2 HIGH).
- `::test_v_rest_inference_warns_on_depolarized_frame0` — frame 0 spanning −85…+20. Expected: warns; uses min.
- `::test_phi_e_rest_requires_explicit` — Expected: `ValueError`.

#### Checklist
- [ ] Dataclass (`value_range`, `eq=False`) + `key()` + five presets
- [ ] `resolve` over masked values via **streaming stats**, all branches + both degenerate guards
- [ ] `.copy()` before `set_bad`; `resampled` for levels; `PowerNorm` direct
- [ ] `infer_v_rest` guards (depolarized frame 0, phi_e raise, ndarray-safe field test)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_video.py \
  -k "preset or gradient or gamma or levels or nan or rest or masked_values or colormap or flat or phi_e \
      or iterator or deterministic or explicit_range or auto99 or validated or unknown_value" -v
```

#### Exit Criteria
- [ ] All five presets reproduce the corpus ranges
- [ ] No degenerate input yields a NaN range or an exception
- [ ] The range is provably computed from masked tissue only

#### Risk
Low — the three previously-flagged API risks (PowerNorm/negative vmin, `resampled`, `set_bad` mutation) were
**empirically verified** under `conda run` (matplotlib 3.10.8). Residual risk is the streaming-percentile
approximation for `auto99`. Mitigate with the DETERMINISTIC per-frame stride above (NOT a reservoir — an unseeded
reservoir made vmin/vmax vary between identical renders, which is why R3 replaced it), and assert the bracketing
`auto.lo <= auto99.lo` and `auto99.hi <= auto.hi` in `::test_auto99_within_auto` (defined in the Test Spec below).

---

### Step 1.3: `video/clip.py` — the `Video` spec object
**Model**: opus

#### Read First
- `cardiac_core/run.py:20-131` — `SimulationResult` fields (all torch) + hook style.
- `cardiac_core/viz.py:21-23` — `_vm_numpy`'s `.detach().cpu().numpy()` (the conversion this plan must not drop).
- `cardiac_core/io.py` — `load_result`'s ACTUAL 4-tuple return.

#### Why
Separating "what to render" from "when/where to write" is the locked architecture. This is also the single seam
where torch→numpy conversion and masking happen, so both are done once and correctly.

#### Implementation Spec
**Files to create:** `cardiac_core/video/clip.py`
**Interfaces:**
```python
@dataclass(eq=False)     # eq=False: data/field/mask may be ndarrays -> dataclass __eq__ would raise (as Gradient)
class Video:
    data: Any                       # SimulationResult | (times,V) | (T,Nx,Ny) array | .npz path
    field: Union[str, np.ndarray] = "Vm"     # "Vm" | "phi_e" | explicit (T,Nx,Ny) array
    gradient: Gradient = Gradient.physiological()   # SAFE as a plain default: Gradient is frozen (immutable)
    label: Optional[str] = None              # panel title (FIGURE PRODUCER ONLY)
    front: Optional[float] = None            # mV isoline, per frame (FIGURE ONLY)
    isochrones: bool = False                 # static LAT contours (FIGURE ONLY)
    mask: Any = None                         # None=auto (use domain_mask) | array | False=explicitly none
    style: str = "bare"          # VALIDATED in __post_init__ (see below)
    aspect: str = "equal"                    # FIGURE ONLY (the bare producer has no axes)
    units: str = "auto"                      # FIGURE ONLY — "auto" (cm if dx known) | "cm" | "nodes"
    # resolved: .frames float64 numpy, .times, .dx, .dy, .active_mask, .value_label, .result (torch, kept)
    @classmethod
    def bare(cls, data, **kw) -> "Video"        # style="bare"
    @classmethod
    def annotated(cls, data, **kw) -> "Video"   # style="annotated"
    def display_values(self, t) -> np.ndarray   # masked, float64, UNtransposed (Nx,Ny)
    def masked_iter(self, idx) -> Iterator
        # CONTRACT: yields ONE (Nx, Ny) float64 array PER FRAME in idx order, already masked
        # (inactive nodes = NaN). Per-frame, NOT flattened and NOT a stack — `first_frame_vals`
        # is only definable if the first yield is a whole frame.
    def preview(self, t_ms=None, *, frame=None, slug="preview", question="lab", bulk=True, **kw) -> str
        # DECLARED here, IMPLEMENTED in Step 1.4: the body is a one-line local import
        #   `from .render import preview_frame; return preview_frame(self, ...)`
        # so clip.py never imports render.py at module level (mirrors SimulationResult.video).
        # The single-frame producers do not exist until Step 1.4 — do NOT try to implement it here.
    def requires_figure(self) -> bool
        # TRUE if style == "annotated" OR any figure-only feature is set
        #   (label / front / isochrones). NOT overlay-driven alone — an annotated clip with no
        #   overlays MUST still route to the figure producer, or the legacy delegation (which is
        #   annotated-with-no-overlays) silently renders bare and loses axes/colorbar/figsize.
        return self.style == "annotated" or self.label is not None \
               or self.front is not None or self.isochrones
```

#### Pseudocode
```
_to_numpy(x): x.detach().cpu().numpy() if hasattr(x,"detach") else np.asarray(x)   # MANDATORY for torch/CUDA

__post_init__:
    frames, times, dx, dy, value_label, result = _resolve_data(data, field)   # each via _to_numpy
    self.result = result                       # KEEP the torch result: analysis.activation_time is torch-only
    frames = np.asarray(frames, dtype=np.float64)      # float64 contract (load_result may hand back float32)
    if frames.ndim == 2: frames = frames[None]
    if frames.ndim != 3: raise ValueError("expected (T, Nx, Ny)")
    if frames.shape[0] == 0: raise ValueError("0 saved frames (t_end < save_every?)")
    if mask is False:            active = None                      # legacy/opt-out: no masking at all
    elif mask is not None:       active = _to_numpy(mask).astype(bool)
    else:                        active = _to_numpy(result.domain_mask) if result has one else None
    validate active.shape == frames.shape[1:]
    if times is None or len(times) != T: times = arange(T)

display_values(t):                    # the ONE masking seam. True = ACTIVE.
    a = frames[t]
    return a if active is None else np.where(active, a, np.nan)

.npz path: times, V, phi_e, meta = io.load_result(path)    # 4-tuple; NO dx/dy/domain_mask
           -> units fall back to "nodes"; warn once

__repr__: "Video(field=…, grid=(Nx,Ny), frames=T, range≈(lo,hi) provisional, style=…, overlays=[…])"
          # provisional: the FINAL range is resolved in render() AFTER striding

preview(...): declared here, implemented in Step 1.4 (see `preview_frame`). Body is the local-import delegation.

if times were not supplied (bare array / length mismatch):
          warn("no time axis supplied; frame indices are being shown as milliseconds")
          # otherwise a burned stamp reads "t = 7.0 ms" for frame 7 with no indication it is an index
```

#### Test Spec
- `::test_accepts_result_pair_array_and_npz` — all four forms. Expected: `.frames.shape == (T,Nx,Ny)`, float64.
- `::test_torch_cuda_tensor_converts` — build a result with CUDA tensors **if `torch.cuda.is_available()`, else
  skip**; also a CPU-tensor case that always runs. Expected: no `TypeError`, `.frames` is numpy float64.
- `::test_float32_input_is_cast` — Expected: `.frames.dtype == float64`.
- `::test_domain_mask_polarity` — `domain_mask` False block. Expected: NaN exactly where False (guards inversion).
- `::test_lbm_finite_obstacle_is_masked` — finite values inside a False region. Expected: NaN in display.
- `::test_mask_false_disables_masking` — `mask=False` with a `domain_mask` present. Expected: no NaN introduced
  (this is what the legacy delegation relies on).
- `::test_phi_e_missing_raises` / `::test_zero_frames_raises` / `::test_repr_marks_range_provisional`
- `::test_invalid_enums_raise` — `style="anotated"`, `aspect="eqal"`, `units="cm2"` (and `fit="containn"` on
  `render`). Expected: `ValueError` naming the legal values for each — never a silent fallback.
- `::test_requires_figure` — parametrized: `front=-40.0` on a bare clip → `True`; `Video.annotated(r)` with NO
  overlays → **`True`** (the case that would otherwise route the legacy path to the bare producer);
  a plain `Video(r)` → `False`. **Note the intentional division of labour:**
  `requires_figure()` is a *router* ("would this need a figure?") and answers `True`; `enforce_capabilities` is a
  *gate* and REJECTS that same combination for a single-clip render. They are not in conflict — the router exists
  so Phase 2 can promote bare clips, where the combination is legal.

#### Checklist
- [ ] `_to_numpy` used on frames, times AND mask
- [ ] Keep `.result` (torch) for analysis calls
- [ ] `display_values` (the single masking seam) AND `masked_iter` (what `Gradient.resolve` consumes, once)
- [ ] `mask=False` sentinel; True=ACTIVE polarity; shape validation
- [ ] `bare`/`annotated` classmethods; `units`; `repr`; `requires_figure`; `preview` **stub only** (local-import
      delegation to Step 1.4's `preview_frame` — its producers do not exist yet)
- [ ] Warn when no time axis was supplied (indices shown as ms)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_video.py \
  -k "clip or mask or repr or phi_e or float32 or torch or npz or zero_frames or requires_figure" -v
```

#### Exit Criteria
- [ ] A torch (and CUDA, when available) result renders without conversion errors
- [ ] Masking correct on a finite LBM-style obstacle; `mask=False` opts out
- [ ] `repr` never claims a final range

#### Risk
**⚠ NAME SHADOWING — this bites at import time.** The attribute is named `field`, so inside the class body the
name `field` is bound to the string `"Vm"`. Calling `dataclasses.field(default_factory=…)` anywhere below that
line raises `TypeError: 'str' object is not callable` **when the module is imported**, breaking
`import cardiac_core` entirely. Two consequences: (1) `gradient` uses a **plain default**
`Gradient.physiological()`, which is safe precisely because `Gradient` is `frozen=True` (immutable — no shared
mutable state); (2) if a `default_factory` is ever genuinely needed here, import it aliased —
`from dataclasses import field as dc_field` — never bare `field`.
Also `eq=False`: `data`/`field`/`mask` may hold ndarrays, and a generated `__eq__` would raise
"truth value of an array is ambiguous" (the same reason `Gradient` has it).

---

### Step 1.4: `video/render.py` — streaming render, both producers, overlays
**Model**: opus

#### Read First
- `Research/Active/boundary_conduction_speedup/render_oblique_videos.py:58` (meshgrid `indexing="ij"`) and
  `:71-92` (per-frame contour + the `c.remove()` / `c.collections` fallback).
- `cardiac_core/analysis.py` — `activation_time` (torch-only) for the isochrone overlay.
- `cardiac_core/viz.py:12-13` — `matplotlib.use("Agg")` must be set here too.

#### Why
Two producers because the common case must be unlabelled AND fast (0.10 vs 7.9 ms/frame). The capability split has
to be explicit or `colorbar=True` on a bare clip is a silent no-op — the exact class of defect this plan exists to
remove.

#### Implementation Spec
**Files to create:** `cardiac_core/video/render.py`; complete `video/__init__.py`
(`__all__ = ["Video","Gradient","render","render_video","VideoInfo"]`, `render_video = render`).
**Interfaces:**
```python
def preview_frame(video, t_ms=None, *, frame=None, slug="preview", question="lab", bulk=True, **kw) -> str
    # ONE frame through the clip's OWN producer (bare -> bare, annotated -> figure), so a preview of the
    # locked bare default shows no chrome the video will not have. Saves PNG via media_path(..., "images").
    # Routes on requires_figure(), exactly as render() does — the two must never disagree.
    # It ALSO calls enforce_capabilities() with the same rules, otherwise
    # Video(r, front=-40).preview() would succeed while render() on the same clip raises.
    # t_ms -> frame: idx = argmin(|clip.times - t_ms|); `frame=` takes an index directly;
    # passing both raises; passing neither uses the middle frame.

def enforce_capabilities(clip, *, colorbar, show_time, figsize, dpi, title) -> None
    # Raises ValueError naming `Video.annotated` when a BARE clip is asked for a figure-only feature:
    # colorbar=True, title=..., figsize=..., dpi=..., or a clip carrying label/front/isochrones.
    # show_time=True is ALLOWED on bare (burned in). Not called on the multi-panel path (Phase 2),
    # where bare clips are promoted to the figure producer instead of rejected.

def render(video, slug, *, question="lab", bulk=True, resolution="1080p", fit="contain",
           fps=20.0, speed=None, max_frames=300, format="mp4", bitrate=None,
                           # `format` shadows the builtin inside render()'s scope; kept as the idiomatic
                           # name (cf. savefig) and bound locally to `fmt` on entry.
                           # `bitrate` defaults to "2M" for webm/VP9 (no `quality` mapping); None elsewhere.
           show_time=None, colorbar=None, title=None,
           figsize=None, dpi=None, units=None, progress=False,
           date=None, root=None) -> VideoInfo
render_video = render
```
`show_time`/`colorbar` default `None` = **follow the style** (bare → off, annotated → on); an explicit bool wins.
**`speed` is in SIMULATION MILLISECONDS PER REAL SECOND** (`speed=50` plays 50 ms of simulated time per second of
video); it overrides `fps`. `progress=True` prints a frame counter every 50 frames.
**Capability rule (single clip):** `colorbar=True`, or a clip with `front`/`isochrones`/`label`, on a **bare** clip
→ `ValueError` naming `Video.annotated`. `show_time=True` on bare IS supported (burned in, post-fit).
**Phase 1 accepts ONE `Video`**; a list raises `NotImplementedError("multi-panel lands in Phase 2")`.
`figsize`/`dpi` given explicitly WIN over `resolution`; `resolution=None` skips `fit_frame` entirely.

#### Pseudocode
```
matplotlib.use("Agg") at module import (mirrors viz.py). NOTE gradient.py must NOT rely on this —
it uses matplotlib.colormaps[...] rather than plt.get_cmap, so no pyplot import-order dependency.

# ---- ORDERED SEQUENCE (this order is load-bearing; do not reorder) ----
# enforce -> select_backend -> stride(+GIF cap) -> resolve -> path -> writer -> loop
# Rationale: the GIF cap depends on the BACKEND, so backend selection must precede stride;
# resolve must follow stride (range is computed on the frames actually rendered); and the
# path must follow backend (ext/kind come from it, and media_path's NN slot is consumed on call).

def render(video, slug, *, ..., bitrate=None, ...):      # `bitrate` IS a declared parameter
    fmt = format                                          # bind locally; `format` shadows the builtin

    # 1. validate + capability gate ------------------------------------------------
    if isinstance(video, (list, tuple)): raise NotImplementedError("multi-panel lands in Phase 2")
    clip = video
    enforce_capabilities(clip, colorbar=colorbar, show_time=show_time,
                         figsize=figsize, dpi=dpi, title=title)     # raises BEFORE any work
    show_time_resolved = show_time if show_time is not None else (clip.style == "annotated")
    colorbar_resolved  = colorbar  if colorbar  is not None else (clip.style == "annotated")

    # 2. backend FIRST (the GIF cap and the path both depend on it) ----------------
    backend, ext, kind = select_backend(fmt)               # raises on anything outside {mp4,webm,gif}

    # 3. stride, with the backend-dependent GIF cap -------------------------------
    T = len(clip.frames)
    eff_max = max_frames
    if backend == "pillow-gif":
        eff_max = 200 if eff_max is None else min(eff_max, 200)     # None-safe: the legacy
        #   delegation passes max_frames=None and can still hit the GIF fallback path.
        #   Capping HERE (not inside the writer) keeps fps/n_frames/duration_s self-consistent.
    stride = ceil(T / eff_max) if (eff_max and T > eff_max) else 1
    idx    = list(range(0, T, stride))

    # 4. colour range, over MASKED values, AFTER striding -------------------------
    cmap, norm, lo, hi = clip.gradient.resolve(clip.masked_iter(idx), field=clip.field)

    # 5. playback rate -------------------------------------------------------------
    if speed is not None:
        d  = diff(clip.times[idx]); dt = median(d) if d.size else 1.0   # median: non-uniform saves
        raw = speed / max(dt, 1e-12); fps = clamp(raw, 1.0, 240.0)
        if fps != raw: warn(f"requested speed implies {raw:.1f} fps; clamped to {fps} — playback rate
                             will not match `speed`")   # silent clamping would misreport the rate

    # 6. path (consumes the NN slot -> save immediately) + writer -------------------
    path   = media_path(question, kind, slug, ext=ext, bulk=bulk, date=date, root=root)
    canvas = resolve_canvas(resolution) if (figsize is None and dpi is None) else None
    if fmt == "webm" and bitrate is None:
        bitrate = "2M"      # VP9 has no `quality` mapping; without an explicit rate ffmpeg logs
                            # "Neither bitrate nor constrained quality specified" and picks CRF 32
    use_figure = clip.requires_figure()

    # 7. stream — writer AND build_figure go INSIDE the guarded region ---------------
    #    build_figure is the most failure-prone call here (isochrone LAT, contour, colorbar,
    #    extent/units): a raise there must not leak the writer or strand a file on the NN slot.
    n = 0; writer = None; fig = None
    try:
        writer = open_writer(path, fps, backend, fmt, quality=8, bitrate=bitrate)
        fig    = build_figure(clip, ...) if use_figure else None    # built ONCE, outside the loop
        for k, t in enumerate(idx):
            rgb = produce_figure(fig, clip, t) if use_figure else produce_bare(clip, t)
            if canvas is not None:
                rgb = fit_frame(rgb, canvas, fit, clip.gradient.interpolation)
            # TIME STAMP IS DRAWN EXACTLY ONCE, BY WHICHEVER PRODUCER OWNS IT:
            #   figure path -> fig.suptitle inside produce_figure (vector text)
            #   bare path   -> burned here, AFTER the fit, so it is legible at 1080p (20 px, not 8)
            if show_time_resolved and not use_figure:
                rgb = burn_timestamp(rgb, f"t = {clip.times[t]:.1f} ms")
            writer.append(rgb); n += 1
            if progress and k % 50 == 0: print(f"  ... {k}/{len(idx)}")
    except BaseException:
        if writer is not None: writer.close()
        if os.path.exists(path): os.remove(path)   # a truncated file would otherwise KEEP the
                                                   # media_path NN slot it consumed -- closing alone
                                                   # does not release it
        raise
    finally:
        if fig is not None: plt.close(fig)         # else the suite leaks a figure per render()
    writer.close()
    return VideoInfo(path=writer.path, n_frames=n, fps=fps, backend=writer.backend,
                     codec=writer.codec, vmin=lo, vmax=hi, stride=stride, ...)

produce_bare(clip, t):                             # no matplotlib figure
    a = np.flipud(clip.display_values(t).T)        # VERIFIED == imshow(V.T, origin="lower")
    return (cmap(norm(np.ma.masked_invalid(a)))[..., :3] * 255).astype(uint8)   # NaN -> "bad" colour

build_figure(clip, ...) -> _FigState:              # ONCE
    # _FigState carries EVERYTHING produce_figure mutates per frame — it must be a real object,
    # not free names: fig, ax, im, Xc, Yc, contour (the live front-contour handle), suptitle.
    # Phase 2 holds a LIST of these (one per panel) plus the single shared fig/suptitle.
    units_resolved = units or clip.units           # "auto" -> "cm" if dx/dy else "nodes"
    if units_resolved == "cm": extent = [0,(Nx-1)*dx, 0,(Ny-1)*dy]; xlab,ylab = "x (cm)","y (cm)"
    else:                      extent = [0, Nx-1,     0, Ny-1];     xlab,ylab = "x (nodes)","y (nodes)"
    # Contour coordinates MUST be built from the SAME extent, or a cm-space contour lands on a
    # node-index axis (the .npz/array clips default to "nodes"):
    x  = linspace(extent[0], extent[1], Nx);  y = linspace(extent[2], extent[3], Ny)
    Xc, Yc = meshgrid(x, y, indexing="ij")         # pairs with the UNtransposed array
    im = ax.imshow(np.ma.masked_invalid(clip.display_values(0).T),
                   origin="lower",            # MUST be explicit — the bare producer's flipud(.T) is pinned
                                              # by a gotcha and a test; omitting it here would let the two
                                              # producers disagree vertically and the suite would not notice
                   extent=extent, aspect=clip.aspect, cmap=cmap, norm=norm,
                   interpolation=clip.gradient.interpolation)   # forward it, or bilinear is a no-op
    if colorbar_resolved: fig.colorbar(im, ax=ax, label=clip.value_label)   # ONCE
    if clip.label: ax.set_title(clip.label)
    if clip.isochrones:                            # ONE definition of this logic (see note below)
        lat = isochrone_lat(clip)
        if isfinite(lat).any():                    # viz.activation_isochrones guards this way too
            ax.contour(Xc, Yc, ma.masked_invalid(lat), levels=12, colors="white",
                       linewidths=0.6, alpha=0.55)
    return _FigState(fig=fig, ax=ax, im=im, Xc=Xc, Yc=Yc, contour=None, suptitle=sup)

isochrone_lat(clip):                               # the SINGLE isochrone definition
    if len(clip.frames) < 2:
        warn("isochrones need >= 2 frames"); return full(nan)
        # VERIFIED: activation_time_interp on T=1 does NOT raise — it returns an all-NaN map. The guard
        # is for a clear WARNING rather than a silently blank overlay (the isfinite().any() check below
        # would skip the contour anyway).
    if clip.result is not None and (isinstance(clip.field, str) and clip.field in ("Vm", "V")):
        # torch path is only valid for Vm — result.Vm is NOT the displayed field for
        # field="phi_e" or an explicit array, which would silently draw Vm isochrones over another field
        lat = _to_numpy(analysis.activation_time(clip.result.Vm, clip.result.times))
        lat = where(clip.active_mask, lat, nan) if clip.active_mask is not None else lat
    else:
        masked = stack(clip.display_values(t) for t in idx)   # built ONLY here, and STRIDED (idx, not
        #   all T) — an unconditional full-history stack would violate the plan's own memory rule.
        #   MASKED so LBM's finite obstacles do not get spurious contours.
        lat = analysis.activation_time_interp(masked, clip.times[idx], threshold=-40.0)   # NUMPY path
        #   analysis.py:593 — so (times,V)/array/.npz clips DO support isochrones. Do NOT raise.
        #   VERIFIED: it accepts a NaN-masked array cleanly — the masked region comes back all-NaN and
        #   tissue stays finite, with no warnings. That is exactly the desired obstacle behaviour.
    return lat

produce_figure(st: _FigState, clip, t):            # only data swaps here (st = the carrier above)
    st.im.set_data(np.ma.masked_invalid(clip.display_values(t).T))
    if clip.front is not None:
        remove previous contour (try c.remove() except AttributeError: for coll in c.collections: coll.remove())
        c = ax.contour(Xc, Yc, clip.display_values(t), levels=[clip.front],
                       colors="white", linewidths=1.4)
    if show_time_resolved:                         # RE-COMPOSE with `title` each frame, else the
        fig.suptitle(f"{title} — t = {clip.times[t]:.1f} ms" if title            # once-composed
                     else f"t = {clip.times[t]:.1f} ms")                          # title is lost
    elif title: fig.suptitle(title)
    fig.canvas.draw(); return asarray(fig.canvas.buffer_rgba())[..., :3]

# figsize/dpi: explicit args win over `resolution` (and imply the FIGURE producer -> a bare clip
# with figsize/dpi is rejected by enforce_capabilities). Otherwise derive figsize x dpi from
# `resolution` so the drawn canvas matches the target and `fit_frame` is a near no-op.
```

#### Test Spec
- `::test_default_is_bare_1080p_unlabelled` — `render(Video(r), "x")` on a **wide fixture** (grid aspect > 16:9,
  e.g. 200×40) so `fit="contain"` letterboxes on the TOP/BOTTOM. Expected: 1920×1080; real mp4; the outermost
  ROWS are black pad and contain no white matplotlib chrome. (Which axis is padded depends on the grid aspect —
  pin the fixture rather than assuming.)
- `::test_colorbar_on_bare_raises` / `::test_front_on_bare_raises` — Expected: `ValueError` naming `Video.annotated`.
- `::test_multipanel_raises_in_phase1` — `render([c1,c2], "x")`. Expected: `NotImplementedError` matching "Phase 2".
- `::test_show_time_on_bare_burns_after_fit` — Expected: frame differs from `show_time=False` only in the stamp
  region, and the glyph height ≥ 15 px (proves post-fit burn).
- `::test_annotated_has_colorbar_by_default` — Expected: a non-background column exists at the right margin.
- `::test_both_producers_agree_on_orientation` — render the SAME asymmetric ramp bare and annotated; compare
  the data region's corner probe pixels. Expected: the same corners hold the same values. Without this, a
  vertical flip between the two producers passes the whole suite — a silent scientific error.
- `::test_orientation_probe_pixels` — an asymmetric ramp. Expected: the **four corner probe pixels** of the bare
  frame match the corresponding corners of a matplotlib `imshow(V.T, origin="lower", interpolation="nearest")`
  render within `atol=6`. (Probe pixels, not full-array equality — Agg resampling makes exact equality fragile.)
- `::test_speed_sets_fps` — 1 ms saves, `speed=20` → `fps == 20`; with stride 2 → `fps == 10`.
- `::test_non_uniform_times_uses_median` — jittered times. Expected: no exception; fps within 10% of nominal.
- `::test_isochrones_and_front_render` — annotated, built from a `SimulationResult`. Expected: non-empty mp4
  (exercises the torch `activation_time` branch of `isochrone_lat`).
- `::test_preview_writes_png` — Expected: non-empty `.png` under `/images/`.
- `::test_preview_bare_has_no_chrome` — `Video(r).preview()` vs `Video.annotated(r).preview()`. Expected: the
  bare preview's border pixels are data/pad, the annotated one's are white figure chrome (proves preview routes
  on the clip's own producer, not always the figure one).
- `::test_resolution_none_skips_fit` — **an ANNOTATED clip** with `figsize=(6,3), dpi=100, resolution=None`.
  Expected: 600×300 exactly. (`figsize`/`dpi` only mean anything on the figure producer; a bare clip ignores them
  and emits grid-sized frames — so passing `figsize`/`dpi` with a bare clip must **raise**, see below.)
- `::test_figsize_on_bare_raises` — bare clip + `figsize=(6,3)`. Expected: `ValueError` explaining that
  `figsize`/`dpi` apply to `Video.annotated` only (another would-be silent no-op).
- `::test_title_on_bare_raises` — bare clip + `title="x"`. Expected: `ValueError` naming `Video.annotated`
  (`title` is figure-only; without this it is a silent no-op on the default path).
- `::test_interpolation_forwarded_to_imshow` — annotated clip, `Gradient(interpolation="bilinear")` vs
  `"nearest"` on a coarse grid. Expected: the two rendered frames differ (proves it is not dropped).
- `::test_isochrones_without_result_uses_numpy_lat` — `Video((times, V), isochrones=True)` on an ANNOTATED clip.
  Expected: renders successfully via `analysis.activation_time_interp` (NOT a `ValueError` — the numpy LAT path
  serves result-less inputs).
- `::test_isochrones_single_frame_warns` — a 1-frame clip with `isochrones=True`. Expected: warns and renders
  (no negative-index wrap in `activation_time_interp`).
- `::test_isochrones_respect_mask` — an LBM-style clip whose masked obstacle nodes are FINITE. Expected: no
  contour lines inside the obstacle (LAT computed from masked display values, not raw frames).
- `::test_time_stamp_drawn_once_on_figure` — annotated + `show_time=True`. Expected: exactly one time string in
  the frame — i.e. the burned-in overlay is NOT applied on top of the suptitle (guards the double-stamp bug).

#### Checklist
- [ ] `matplotlib.use("Agg")` at import
- [ ] `enforce_capabilities` before any work; list → `NotImplementedError`
- [ ] Resolve gradient from `masked_iter` AFTER striding
- [ ] Bare producer `flipud(.T)`; timestamp burned POST-fit
- [ ] Figure producer built once; contour remove/re-add fallback; isochrones via the torch result
- [ ] `figsize`/`dpi` precedence; `resolution=None` skips fit; `render_video` alias exported
- [ ] `preview_frame` implemented here + `Video.preview`'s local-import body wired to it
- [ ] **The ordered sequence is load-bearing**: enforce → select_backend → stride(+GIF cap) → resolve →
      path → writer → loop. Backend BEFORE stride (the GIF cap needs it); path AFTER backend (ext/kind).
- [ ] webm resolves `bitrate="2M"` by default and surfaces it on `VideoInfo`

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_video.py -v
```

#### Exit Criteria
- [ ] `r.video("x")` yields a 1920×1080 unlabelled real mp4
- [ ] Every advertised toggle works or raises — none is a silent no-op
- [ ] Orientation matches the `imshow` convention at all four corners

#### Risk
The bare path colormaps outside matplotlib, so the NaN→"bad" mapping must match `imshow`. Mitigate with
`test_orientation_probe_pixels` including a masked cell among the probes.

---

### Step 1.5: wiring — result hook, viz delegation, exports
**Model**: opus (edits `__init__.py`, contended with the parallel session — not the trivial tier)

#### Read First
- `cardiac_core/run.py:106-131` — hook style to mirror.
- `cardiac_core/viz.py:26-53` — the CURRENT `propagation_video`: axes + colorbar + title, **node-index labels**,
  **no masking**, `figsize=(6,3)`, `dpi=100`.
- `cardiac_core/__init__.py` — the `_LAZY` map (re-read immediately before editing).

#### Why
`r.video("slug")` is the ergonomic point. The delegation must preserve the legacy *look*, which R2 showed is more
than size: legacy drew **node axes** and did **no masking**, so a naive delegation would silently add cm axes and
grey obstacles.

#### Implementation Spec
**Files to modify:**
- `cardiac_core/run.py` — `SimulationResult.video(self, slug, **kw)` → `render(Video(self), slug, **kw)`.
- `cardiac_core/viz.py` — `propagation_video` body → delegation. **Exact mapping:**
  ```python
  from .video import render, Video, Gradient
  info = render(
      Video(result,
            gradient=Gradient(cmap=cmap, value_range=(vmin, vmax), bad="0.55"),
            style="annotated",
            aspect="auto",       # a Video field, NOT a render() kwarg
            units="nodes",       # legacy drew node indices
            mask=False),         # legacy did NOT mask; keep it that way
      slug, question=question, bulk=bulk, fps=fps,
      figsize=(6.0, 3.0), dpi=100, resolution=None,   # explicit figsize wins -> 600x300 preserved
      max_frames=None, colorbar=True, show_time=True)
  return info.path              # str, as before
  ```
  **Deliberate behaviour changes, documented rather than hidden — (a)** the per-frame time text moves from
  `ax.set_title` (`viz.py:41`) to `fig.suptitle`; visually near-identical, but it is a change.
  **(b)** if the encoder falls back to GIF, the legacy
  function used the slug `f"{slug}-propagation"` (`viz.py:50`); the new path uses plain `slug`. The fallback is now
  loud (a warning names the backend), so the filename no longer has to carry that signal. Note it in the docstring.
- `cardiac_core/__init__.py` — `_LAZY` += `{'Video':'video', 'Gradient':'video', 'render':'video',
  'render_video':'video', 'VideoInfo':'video'}`. **All five** — the cheatsheet imports `render`, and `_LAZY`
  raises `AttributeError` for unmapped names.

#### Pseudocode
```
run.py:  def video(self, slug, **kw):
             from .video import render, Video
             # SPLIT the kwargs: forwarding everything to render() makes every Video-level knob
             # (gradient=, style=, front=, isochrones=, label=, mask=, field=, aspect=, units=) a
             # TypeError from the headline API — including `gradient=`, which §10 teaches to exactly
             # the users who call r.video().
             VIDEO_KEYS = {"field","gradient","label","front","isochrones","mask","style","aspect","units"}
             vkw = {k: kw.pop(k) for k in list(kw) if k in VIDEO_KEYS}
             return render(Video(self, **vkw), slug, **kw)
```

#### Test Spec
- `::test_result_hook_returns_videoinfo` — Expected: a `VideoInfo`; real mp4.
- `::test_result_hook_forwards_video_kwargs` — `r.video("x", gradient=Gradient.zoom(), style="annotated",
  isochrones=True)`. Expected: no `TypeError`; the resolved `info.vmin/vmax` match the zoom preset (proves
  Video-level kwargs reach the `Video`, not `render`).
- `::test_lazy_exports_resolve` — `cc.Video, cc.Gradient, cc.render, cc.render_video, cc.VideoInfo` all import.
  Expected: no `AttributeError` (the exact `render`/`render_video` trap R2 found re-introduced).
- `::test_legacy_size_and_format` — Expected: `str` ending `.mp4`, real MP4, **600×300**.
- `::test_legacy_still_annotated_nodes_unmasked` — a result WITH a `domain_mask`. Expected: matplotlib chrome
  present (white margin column) AND no grey `bad` pixels introduced (proves `mask=False` + `units="nodes"`).

#### Checklist
- [ ] `SimulationResult.video`
- [ ] viz delegation with the full mapping (incl. `mask=False`, `units="nodes"`, `aspect` on the Video)
- [ ] All five `_LAZY` entries, added LAST after re-reading
- [ ] Add `"video"` (and `"fields"`) to `test_self_contained.py::test_subpackage_importable`
      — Architecture Changes lists this MOD; it is owned HERE so no step drops it

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction python -m pytest \
  cardiac_core/tests/test_video.py cardiac_core/tests/test_viz.py \
  cardiac_core/tests/test_self_contained.py -v      # the last one is EDITED by this step
/opt/miniforge3/bin/conda run -n heart-conduction python -c \
  "import cardiac_core as cc; print(cc.Video, cc.Gradient, cc.render, cc.render_video, cc.VideoInfo)"
```

#### Exit Criteria
- [ ] `test_viz.py` passes unchanged
- [ ] Legacy output: annotated, node axes, unmasked, 600×300, `.mp4`

#### Risk
Editing `__init__.py` races the parallel session — mitigate by doing it last and keeping the edit to five entries.

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
SCRATCH="${TMPDIR:-/tmp}"    # no CLAUDE_SCRATCHPAD var exists in this project; TMPDIR or /tmp
# BASELINE — capture BEFORE writing any Phase-1 code (a post-hoc "before" run is worthless).
# Record the FAILED/ERROR test IDs, not the summary line: absolute counts are not attributable while a
# parallel session is dirty in this tree, but the SET of failing tests is.
# VERIFIED empirically (2026-07-23), both details matter:
#   * `-rf` alone reports ZERO pytest collection ERRORs; `-rfE` reports them. A collection/import error
#     is the LIKELIEST failure when adding a subpackage and editing __init__.py, and under `-rf` the
#     diff comes back empty => a suite that cannot even be collected would PASS this gate.
#   * `conda run` prints its OWN `ERROR conda.cli.main_run:execute(...)` line on any non-zero exit,
#     which `^(FAILED|ERROR)` would otherwise capture as a bogus "test id" in every failing baseline.
#     Hence the `grep -v "conda.cli"`.
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction \
  python -m pytest cardiac_core/tests/ -q -rfE 2>&1 | grep -E "^(FAILED|ERROR)" | grep -v "conda.cli" | awk '{print $1" "$2}' | sort > "$SCRATCH/vp_base_p1.txt"
# ... implement Phase 1 ...
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction \
  python -m pytest cardiac_core/tests/ -q -rfE 2>&1 | grep -E "^(FAILED|ERROR)" | grep -v "conda.cli" | awk '{print $1" "$2}' | sort > "$SCRATCH/vp_after_p1.txt"
comm -13 "$SCRATCH/vp_base_p1.txt" "$SCRATCH/vp_after_p1.txt"   # MUST be empty = no NEW failures
# GUARD against a VACUOUS pass: if the "after" run died before producing a short summary (usage error,
# INTERNALERROR, conftest/plugin crash, conda failure) the grep yields nothing and comm is empty too.
# Require evidence the suite actually ran:
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction \
  python -m pytest cardiac_core/tests/ -q --collect-only 2>&1 | tail -1   # must report a sane test count
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction \
  python -m pytest cardiac_core/tests/test_integrity.py -q
```
(`--no-capture-output` matters: without it `conda run` buffers pytest's output until exit and a long suite
looks hung.)

### Phase 1 Exit Criteria
- [ ] All new tests pass
- [ ] No NEW failures vs the saved pre-phase baseline
- [ ] Integrity goldens bit-identical (atol=0)
- [ ] `r.video("slug")` produces the locked default

### Phase 1 Cleanup
- float64 consistency — frames float64 end-to-end; only the final RGB buffer is uint8; no float32 from `load_result`
- V5.3 not modified — `Monodomain/Engine_V5.3/` untouched
- No cross-engine duplication — no ionic/mesh code copied; analysis reused via `cardiac_core.analysis`
- EXPERIMENT.md backlinks — N/A (no engine experiment created)
- Remove debug prints; delete stray media written with `bulk=False`

**-> Commit point: git commit after Phase 1 passes**

---

## Phase 2: multi-panel comparison

**Goal**: `render([a, b, …], slug)` — N panels, ONE shared colorbar, ONE shared time stamp.
**Tier**: medium
**Estimated scope**: extend `render()` + layout + compatibility rules + ~8 tests.

### Phase Context
Most polished prior art is a comparison: `render_semicircle_video.py` (2 panels side-by-side,
`fig.colorbar(imA, ax=(axA,axB))`, `fig.suptitle` time, `nf = min(len(spec_V), len(hbb_V))`),
`render_oblique_videos.py` (3 stacked + per-frame front), `video_boundary_modes.py` (**`subplots(1,4)`
side-by-side with per-panel colorbars**), `render_combined_axis.py` (5 stacked). Phase 1 raises
`NotImplementedError` for a list; this phase implements it.

### Step 2.1: panel layout, shared color, compatibility rules
**Model**: opus

#### Read First
- `Research/Active/boundary_conduction_speedup/render_semicircle_video.py:51-71` — 2-panel + shared colorbar.
- `Research/Active/boundary_conduction_speedup/video_boundary_modes.py:107-120` — 4-panel side-by-side.

#### Why
Panels only mean something if they share a color mapping — that is why `Gradient` is reusable and why a mismatch
must warn rather than quietly mislead.

#### Implementation Spec
**Files to modify:** `cardiac_core/video/render.py`.
**Interfaces:** `render(clips: list[Video], ..., labels=None, rows=None, cols=None)` — `labels`, `rows`, `cols`
are **introduced here** (they were deliberately absent from the Phase-1 signature).
**Capability resolution (this is what unblocks Phase 2 vs Phase 1's rule):** a multi-panel render **always uses
the figure producer**; any clip with `style="bare"` is **promoted to annotated for layout purposes**, with a
single `UserWarning` naming the promotion. **`colorbar` AND `show_time` both therefore default ON for multi-panel** — promoting a bare clip to the figure
producer without also resolving `show_time` would leave the Phase-2 'ONE shared time stamp' undrawn (bare's
burn path is skipped for promoted clips, and `show_time=None` would still resolve to bare→off).
Layout: 2 panels → `subplots(1,2)`; 3 → `(3,1)`; 4 → `(2,2)`; 5+ → `(N,1)`; `rows`/`cols` override.
`constrained_layout=True`.
**Layout justification, corrected:** the 4-panel prior art (`video_boundary_modes.py:107`) is actually
`subplots(1, 4)` with **per-panel** colorbars. We deliberately default 4 → **2×2** instead, because four panels
side-by-side inside a 1920-wide canvas gives each ~480 px and the wavefront becomes unreadable. This is a
considered deviation from the corpus, not a reproduction of it; `cols=4` restores the original arrangement.

#### Pseudocode
```
validate: all clips same (Nx,Ny)                -> else ValueError listing every shape
          median save interval equal within 1e-9 -> else warn (frames pair by INDEX, not time)
T = min(len(c.frames) for c in clips); ONE stride; ONE idx list, applied to ALL panels
enforce_capabilities is NOT called on this path — bare clips are PROMOTED, not rejected
         (calling it would raise on colorbar=True before promotion could run)
promote: for c in clips if c.style == "bare": use the figure producer (warn once)
require: all clips share the same FIELD KIND (all "Vm", or all "phi_e") -> else ValueError.
         Mixing Vm and phi_e in one shared-colorbar figure is meaningless, and `infer_v_rest`
         raises for phi_e, so a mixed pool has undefined behaviour.
shared = all(c.gradient.key() == clips[0].gradient.key() for c in clips)
if shared: resolve ONCE over chain(all clips' masked_iter(idx)), forwarding field=clips[0].field;
           ONE fig.colorbar(im0, ax=all_axes)
else:      resolve per clip; per-panel colorbars; warn("panels use different gradients; not directly comparable")
suptitle time uses clips[0].times[idx[k]]        # panel 0 drives the clock (documented)
```

#### Test Spec
- `::test_two_panel_shared_colorbar` — two clips, same `Gradient` (both left at the bare default). Expected: one
  mp4; a promotion `UserWarning`; frame wider than single-panel; no `ValueError` from the Phase-1 capability rule.
- `::test_grid_mismatch_raises` — `(40,10)` vs `(40,12)`. Expected: `ValueError` naming both shapes.
- `::test_truncates_to_shortest` — lengths 40 and 25. Expected: `n_frames <= 25`; panels in step.
- `::test_differing_gradients_warn` — different presets. Expected: `pytest.warns(UserWarning, match="comparable")`.
- `::test_four_panels_are_2x2` — 4 clips. Expected: renders; aspect closer to square than a 4×1 stack would give.
- `::test_multipanel_draws_one_shared_time_stamp` — two clips left at the BARE default, `show_time` unset.
  Expected: a time stamp IS drawn (promotion resolves `show_time` on) and appears exactly ONCE for the figure,
  not once per panel. Guards the R4-H4 fix, which was prose-only until now.
- `::test_multipanel_no_capability_error` — two bare clips + `colorbar=True`. Expected: no `ValueError` (the
  single-clip capability rule must not fire on the promoted multi-panel path).
- `::test_labels_become_panel_titles` — `labels=["a","b"]`. Expected: renders; no exception.

#### Checklist
- [ ] Shape + save-interval validation
- [ ] Bare→figure promotion with a single warning
- [ ] Shared vs per-panel colorbar branch + warning
- [ ] Layout defaults (2→1×2, 4→2×2) + `rows`/`cols` + `labels`

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_video.py \
  -k "panel or mismatch or truncat or gradients or labels" -v
```

#### Exit Criteria
- [ ] `render([a,b], slug)` → one mp4, one colorbar, one time stamp
- [ ] Incompatible inputs raise or warn — never silently mislead

#### Risk
`fig.colorbar(im, ax=[...])` with `constrained_layout` can shrink axes unpredictably at 5 panels — assert
structural properties only, never exact pixel geometry.

### Phase 2 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
SCRATCH="${TMPDIR:-/tmp}"    # no CLAUDE_SCRATCHPAD var exists in this project; TMPDIR or /tmp
# capture the Phase-2 baseline BEFORE implementing (record FAILED test IDs, not just the count):
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction \
  python -m pytest cardiac_core/tests/ -q -rfE 2>&1 | grep -E "^(FAILED|ERROR)" | grep -v "conda.cli" | awk '{print $1" "$2}' | sort > "$SCRATCH/vp_base_p2.txt"
# ... implement Phase 2 ... then:
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction \
  python -m pytest cardiac_core/tests/ -q -rfE 2>&1 | grep -E "^(FAILED|ERROR)" | grep -v "conda.cli" | awk '{print $1" "$2}' | sort > "$SCRATCH/vp_after_p2.txt"
comm -13 "$SCRATCH/vp_base_p2.txt" "$SCRATCH/vp_after_p2.txt"   # MUST be empty = no NEW failures
```

### Phase 2 Exit Criteria
- [ ] All new tests pass; no NEW failures vs the saved Phase-2 baseline
- [ ] Multi-panel matches the semicircle composition structurally

### Phase 2 Cleanup
- float64 consistency; V5.3 untouched; no cross-engine duplication; no EXPERIMENT.md needed; no stray media

**-> Commit point: git commit after Phase 2 passes**

---

## Phase 3: docs + acceptance

**Goal**: make the feature discoverable (cheatsheet + skill) and prove the abstraction covers the real prior-art
compositions.
**Tier**: small
**Estimated scope**: 1 cheatsheet section + 2 synthetic acceptance tests + 1 skill note.

### Phase Context
`API_CHEATSHEET.md` §10 currently documents the three `viz` one-liners. `test_cheatsheet_examples_execute`
(`tests/test_usability_fixes.py`) execs ONLY blocks whose first line is `# runnable-canary` (that is §12), so §10
is untested today. `.claude/skills/sim-media/SKILL.md` calls only `cardiac_core.viz`, forbids hand-rolled
matplotlib, and points readers at "§7" for the media functions — a stale cross-reference to fix while there.

### Step 3.1: `API_CHEATSHEET.md` §10 rewrite
**Model**: opus (edits a file actively contended by the parallel Stim session)

#### Read First
- `cardiac_core/API_CHEATSHEET.md` §10 — re-read immediately before editing (contended file).
- `cardiac_core/tests/test_usability_fixes.py::test_cheatsheet_examples_execute` — confirm the
  `# runnable-canary` first-line rule.

#### Why
§10 is how a scientist discovers this feature, and R2 confirmed a rewritten §10 would otherwise be untested and
free to drift.

#### Implementation Spec
**Files to modify:** `cardiac_core/API_CHEATSHEET.md` §10 — document `Gradient` presets, `Video`, `render`,
`r.video`, keeping `apd_map_figure`/`activation_isochrones`. Use
`from cardiac_core import Video, Gradient, render` — **NOT** `cc.video.…`. **Use `cc.Stim` for the stimulus,
not the dict form** — the dict path now emits a `DeprecationWarning`, which a `filterwarnings("error")` run
would turn into a failure inside the cheatsheet exec test. (the `_LAZY` `__getattr__` does not
resolve bare submodule names).
**Files to create:** `::test_cheatsheet_video_section_executes` in `test_video.py`.

#### Pseudocode
```
The §10 block MUST be made SELF-CONTAINED (the current one references an undefined `result`/`slug` —
API_CHEATSHEET.md:236-247 — so an exec test would NameError). Mirror the §12 canary shape: a tiny
grid + short run inline, then the media calls. Keep it small (`cc.Grid(40, 10, 0.025)` — **dx is a required
positional arg**, `grid.py:40`; `Grid(40,10)` would TypeError — and `t_end=20`) so the test is fast.

extract §10: read API_CHEATSHEET.md; find the "## 10." heading; take the first ```python fence after it
exec it in a fresh namespace; assert no exception and that any returned path exists
```

#### Test Spec
- `::test_cheatsheet_video_section_executes` — Expected: the §10 block runs to completion; every symbol it imports
  resolves from the top-level `cardiac_core` namespace.

#### Checklist
- [ ] Rewrite §10 · [ ] Add the §10 exec test · [ ] Surgical edit, re-read first
- [ ] Add `Video`/`Gradient`/`render`/`VideoInfo` to
      `Research/Active/engine_consolidation/API_REFERENCE.md` (contended — re-read first)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_video.py -k cheatsheet -v
```

#### Exit Criteria
- [ ] §10 documents the object API and is covered by an executing test

#### Risk
Concurrent cheatsheet edits by the Stim session — surgical single-section edit, re-read immediately before.

### Step 3.2: acceptance — reproduce the corpus compositions on SYNTHETIC data
**Model**: opus

#### Read First
- `Research/Active/boundary_conduction_speedup/render_semicircle_video.py` — the 2-panel composition.
- `Research/Active/boundary_conduction_speedup/render_audit_video.py:48-67` — the zoom composition.

#### Why
Honest proof the abstraction covers real prior art. **These must run on SYNTHETIC data**: `render_audit_video.py`
imports `audit_specular_every_surface` + `src.collision.bgk` / `src.streaming.d2q9` / `src.state`, and **no `src/`
exists on either path it inserts** — the reference does not run as-is, so there is nothing to reproduce against.
The semicircle script needs multi-GB tracked HDF5. `cardiac_core/tests/` importing research scripts or research
data would also be a layering violation.

#### Implementation Spec
**Files to modify:** `cardiac_core/tests/test_video.py` — two composition tests from synthetic arrays. Assert
**structural/quantitative** properties, never pixel-identity. Use `question="lab"`, `bulk=True` (matching every
other cardiac_core test).

#### Pseudocode
```
semicircle-like: build (T,Nx,Ny) travelling wave; domain_mask=False inside a half-disc;
                 two clips sharing Gradient.rest_anchored(); render([a,b], …); assert one file,
                 obstacle pixels == the resolved `bad` colour, promotion warning raised.
zoom-like:       uniform field at V_REST=-85 with a local patch at V_REST+7.5 (RELATIVE, i.e. -77.5 mV);
                 measure the fraction of the colormap the patch spans under each preset.
```

#### Test Spec
- `::test_reproduces_semicircle_composition` — Expected: one mp4; obstacle pixels equal the `bad` colour; one
  shared colorbar; no exception.
- `::test_reproduces_zoom_artifact_visibility` — a uniform **−85 mV** field with a patch at **−77.5 mV**
  (i.e. `V_rest + 7.5`, the `render_audit_video.py` artifact scale — **relative, not absolute**).
  **Computed and verified expected values:** `Gradient.physiological()` (−90,40) → the patch spans **5.8%** of the
  colormap; `Gradient.zoom(span=8.0, below=0.3)` → (−85.3,−77.0) → **90.4%**; a **15.7× visibility gain**.
  Assert `< 10%` and `> 60%` (margins chosen so the test is not brittle).

#### Checklist
- [ ] Two synthetic composition tests · [ ] Zero research-data / research-script imports

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_video.py -k reproduces -v
```

#### Exit Criteria
- [ ] Both compositions render from synthetic data with no research-tree dependency

#### Risk
If the measured spans disagree with 5.8%/90.4%, first check the patch is **relative** (−77.5 mV, not +7.5 mV
absolute) — that ambiguity, not the preset, is the likely cause.

### Step 3.3: `/sim-media` skill note
**Model**: sonnet

#### Read First
- `.claude/skills/sim-media/SKILL.md` — it calls ONLY `cardiac_core.viz`; note its stale "§7" cross-reference.

#### Why
The skill inherits the delegation automatically; it should also learn the object API exists, and its cheatsheet
pointer is wrong.

#### Implementation Spec
**Files to modify:** `.claude/skills/sim-media/SKILL.md` — add `Video`/`Gradient`/`render` (gradient presets,
multi-panel) as the richer option; keep the three one-liners as the default path; **fix the "§7" reference to §10**.

#### Pseudocode
```
insert a "Richer option" subsection after the existing one-liner list; leave the
"no bespoke plotting / only cardiac_core" rule intact; s/§7/§10/ in the cheatsheet pointer
```

#### Test Spec
- No automated test (skill markdown). Manual check via the Verify grep below.

#### Checklist
- [ ] Object-API section · [ ] Keep the no-hand-rolled-matplotlib rule · [ ] Fix §7 → §10

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
grep -n "Video\|Gradient\|render(\|§10\|§7" .claude/skills/sim-media/SKILL.md
```

#### Exit Criteria
- [ ] The skill documents both the one-liners and the object API, with a correct §10 pointer

#### Risk
None material (documentation only).

### Phase 3 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
SCRATCH="${TMPDIR:-/tmp}"
# capture the Phase-3 baseline BEFORE implementing (Phase 3 previously asserted "no NEW failures"
# without ever capturing one):
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction \
  python -m pytest cardiac_core/tests/ -q -rfE 2>&1 | grep -E "^(FAILED|ERROR)" | grep -v "conda.cli" | awk '{print $1" "$2}' | sort > "$SCRATCH/vp_base_p3.txt"
# ... implement Phase 3 ... then:
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction \
  python -m pytest cardiac_core/tests/ -q -rfE 2>&1 | grep -E "^(FAILED|ERROR)" | grep -v "conda.cli" | awk '{print $1" "$2}' | sort > "$SCRATCH/vp_after_p3.txt"
comm -13 "$SCRATCH/vp_base_p3.txt" "$SCRATCH/vp_after_p3.txt"   # MUST be empty
```

### Phase 3 Exit Criteria
- [ ] Docs updated; acceptance tests green; `comm -13` on the Phase-3 baseline is empty

### Phase 3 Cleanup
- float64 consistency; V5.3 untouched; no cross-engine duplication; no committed test media

**-> Commit point: git commit after Phase 3 passes**

---

## Final Cleanup
1. **Archive the completed plan:**
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp cardiac_core/VIDEO_OBJECT_PLAN.md \
   "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_video-object-spec-first-rendering.md"
```
2. **Revert the bottom tmux pane from PLAN.md back to WHITEBOARD.md:**
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/engine_consolidation/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/engine_consolidation/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```
3. float64 consistency — no float32 leaks into the media layer (`load_result` cast verified)
4. V5.3 not modified — `Monodomain/Engine_V5.3/` read-only throughout
5. No cross-engine duplication — no ionic/mesh/analysis code copied into `video/`
6. EXPERIMENT.md backlinks — N/A (no engine experiment created)
7. Delete regenerable test media under `media/lab/_sim_outputs/`
8. Log to IDEALOG + KNOWLEDGE — **including the CORRECTED defect statement** (a PATH-dependent silent format
   downgrade, NOT "every mp4 was a GIF") and the `conda run` verification lesson

## Deferred (explicitly OUT)
- **Geometry-outline overlay** (user: "differ for now") — needs caller-supplied analytic geometry.
- Per-frame derived-statistic overlays (`render_audit_video.py`'s live "loose max +X mV" suptitle).
- Migrating the ~20 existing research render scripts onto the new API.
- Non-cardiac renderers (`simulation/` tank videos, MonthlyReport cartoons).

## Mutation Log
- 2026-07-22 — created. Design locked: full gradient control, built-in render, multi-panel, front + static
  isochrone overlays, optional time stamp/colorbar, geometry outline deferred.
- 2026-07-22 — defaults locked (user): zero-argument case is bare, unlabelled, standard preset, 1080p.
- 2026-07-22 — **REVISED after audit R1 (3C/11H/11M/8L, all addressed).** Draft-on-disk reference removed
  (reverted); the motivating defect rewritten after it proved FALSE; bare-style capability rule added; `_LAZY`
  alias; legacy delegation pinned to `annotated`; LBM finite-mask fix; fit/upscale math; orientation;
  `Gradient` degenerate guards; `("rest", span)` ambiguity removed; `v_rest` guards; all steps given the 9
  mandated sections; phase closings; canary claim corrected; acceptance moved to synthetic data.
- 2026-07-22 — **REVISED after audit R2 (3C/10H/14M/14L, all addressed).**
  **C1** `cc.render` unmapped while §10 imports it → `_LAZY` now exports **all five** names (`Video`, `Gradient`,
  `render`, `render_video`, `VideoInfo`). **C2** legacy delegation passed `aspect`/`resolution=None` to `render()`
  where `aspect` is a `Video` field → mapping corrected; `resolution=None` made explicitly legal (skips fit).
  **C3** Phase 2's shared colorbar was unreachable under Phase 1's capability rule → multi-panel now **always**
  uses the figure producer and **promotes** bare clips (warned once); the capability rule is scoped to single-clip.
  **H** every `Verify` used `conda activate` (a no-op non-interactively, silently short-circuiting the whole
  command) → all now use `/opt/miniforge3/bin/conda run -n heart-conduction`, and the trap is documented in
  gotchas + saved to memory; **no torch→numpy conversion existed** → `_to_numpy` mandated at ingest, and the torch
  result is retained because `analysis.activation_time` is torch-only; the PIL timestamp was burned **before** the
  48× upscale → moved post-fit with a DejaVuSans truetype at ~H/40; `open_writer` could not build its own GIF
  fallback path → backend/ext/kind now selected **before** `media_path`; the "baseline" ran twice post-hoc →
  captured to a file **before** implementation; `Gradient.resolve`/`infer_v_rest` still read **unmasked** frames
  (LBM contamination) → both now consume `masked_iter`; `labels`/`rows`/`cols` were in the Phase-1 signature with
  no implementation → moved to Phase 2, and a list now raises `NotImplementedError` in Phase 1; the
  orientation test was pixel-exact-fragile → four corner **probe pixels** with `atol=6`; "unchanged appearance"
  was false (legacy is unmasked + node axes) → `mask=False` sentinel + `units="nodes"` in the delegation, with a
  test asserting both. **M/L** `smin` NameError on the stretch branch; `Video.bare/annotated` signatures added;
  the colormap-mutation test retargeted to a caller-supplied `Colormap` instance; ndarray-safe `field` comparison;
  `"zoom"` added to the legal `value_range` enum; constant-memory claim honoured via **streaming stats** instead of
  a strided stack; `speed` units documented; `bulk`/`fps`/`question`/`format` added to THE DEFAULT table;
  `resolution=None` figure path specified; Pseudocode added to Steps 3.1–3.3; Phase 3 Goal/Context and per-phase
  Estimated scope added; tmux-revert restored as Final Cleanup item 2; `-k` filters widened to cover every test;
  `Gradient.autoscale()` corrected to **viridis**; 4-panel layout corrected to **2×2** side-by-side per
  `video_boundary_modes.py`; meshgrid citation corrected to `:58`; `AREA`→`Image.BOX`; `progress` specified;
  `range`→`value_range` (builtin shadowing); bilinear's RGB-vs-data-space difference documented; `VideoInfo`
  exported; the GIF-path test made behavioural; sim-media's stale "§7"→§10 fix folded into Step 3.3;
  `matplotlib.use("Agg")` mandated in `render.py`; header `Source:` now links IDEALOG.
- 2026-07-22 — **empirically verified under `conda run`** (so R3 need not re-litigate): `PowerNorm` handles a
  negative `vmin` (`PowerNorm(2.0,-90,40)(-25.0) == 0.25`) → the custom-`Normalize` workaround was removed as
  unnecessary; `Colormap.resampled(8)` → exactly 8 colors; mutating a registered colormap IS permitted (so
  `.copy()` is mandatory); `flipud(V.T)` is pixel-identical to `imshow(V.T, origin="lower")` at both corners;
  `Image.{NEAREST,BOX,BILINEAR,LANCZOS}` all exist in PIL 12.1.0; imageio accepts every listed FFMPEG kwarg with
  no warnings; the zoom-preset assertion arithmetic (5.8% vs 90.4%, 15.7×) holds.
- 2026-07-22 — **REVISED after audit R3 (1C/6H/14M/12L, all addressed).** R3 confirmed every R2 headline fix
  landed and that the `value_range` rename + `_LAZY` five-name export + all sampled line-number citations check
  out against source — but found the R1→R2 pattern repeating: **three of six HIGHs were follow-ons to R2's own
  fixes.**
  **C1** `Video`'s attribute is named `field`, so the class body's `field(default_factory=…)` for `gradient`
  would raise `TypeError: 'str' object is not callable` **at import**, breaking `import cardiac_core` entirely →
  plain default `Gradient.physiological()` (safe: `Gradient` is frozen) + a Risk note mandating
  `from dataclasses import field as dc_field` if a factory is ever needed; `Video` also given `eq=False`
  (ndarray fields, same reason as `Gradient`).
  **H1** the post-fit burn (R2's own fix) applied to BOTH producers while `produce_figure` also set a suptitle →
  **double time stamp** on every annotated/legacy render → the burn is now bare-path-only, with
  `::test_time_stamp_drawn_once_on_figure`. **H2** `resolve` drained the `masked_iter` generator in
  `streaming_stats` and then `infer_v_rest` re-read it → empty → silent `-85.0`, breaking `rest_anchored`/`zoom`
  for every input, and the preset test **could not catch it** because its fixture V_rest was exactly −85 → single
  pass now captures `first_frame_vals`, `infer_v_rest` takes stats only, fixture moved to **−82**, plus
  `::test_resolve_consumes_iterator_once`. **H3** the last-resort `pillow-gif` backend is selected precisely when
  imageio is unimportable, yet was specified as `imageio.mimwrite` → reimplemented on **PIL `save(save_all=True)`**
  with `::test_gif_fallback_works_without_imageio`. **H4** `select_backend` did no format validation, so
  `webm`/`mov`/`avi` reached `open_writer` and got hard-coded `libx264`+`yuv420p` → `CODECS` per container +
  explicit `ValueError` for anything outside `{mp4, webm, gif}`. **H5** `Gradient.interpolation` was never
  forwarded to `imshow` (silent no-op on the annotated path, so the semicircle prior art was unreproducible) →
  forwarded, tested, and the bare path's RGB-space-vs-data-space difference documented. **H6** the gotcha
  "mutating a registered colormap contaminates the global" was a **second false 'verified' claim** — re-measured:
  `plt.get_cmap` returns a fresh copy per call and neither it nor `matplotlib.colormaps` is contaminated; the real
  hazard is a **caller-supplied `Colormap` instance** (which IS mutated) → gotcha corrected, `.copy()` retained
  for the right reason, lesson added to memory.
  **M/L** explicit `(lo,hi)` now wins over the all-NaN fallback; `auto99` uses a **deterministic strided
  subsample** (an unseeded reservoir made `vmin/vmax` vary between identical renders); `isochrones` without a
  `SimulationResult` raises; `title` specified as figure-only and composed with the time stamp; `preview` honours
  the clip's OWN style (a figure preview of a bare clip showed chrome the video lacks); `figsize`/`dpi` on a bare
  clip now raise; missing time axis warns instead of silently printing indices as ms; GIF accumulation documented
  as an explicit, capped exception to the streaming rule; multi-panel requires a single field kind and forwards
  `field=clips[0].field`; the 4-panel `2×2` default relabelled a **considered deviation** from the corpus's
  `subplots(1,4)`; Phase 2/3 baselines now capture **FAILED test IDs** before implementation and compare with
  `comm -13` (the ellipsis placeholder is gone), all Verify blocks use `--no-capture-output` and the session
  scratchpad instead of `/tmp`; §10 must be made self-contained or its exec test NameErrors; `requires_figure`
  vs `enforce_capabilities` division of labour spelled out; Steps 1.5/3.1 raised to opus (they edit contended
  files, not trivial-tier work); `test_self_contained.py` + `API_REFERENCE.md` updates added; the legacy GIF-slug
  change documented; `autoscale`/semicircle citations corrected; `-k` filter fixed; test-count estimate corrected.
- 2026-07-22 — **REVISED after audit R4 (0C/4H/19M/9L, all addressed). R4 confirmed 0 CRITICAL and verified the
  blueprint structure is fully compliant** (all 9 sections on every step, all 8 per-phase elements, Final Cleanup
  archive→tmux order) **and re-verified ~25 source claims as correct** (viz/media/run/io/analysis signatures, the
  `_LAZY` shape, `domain_mask` polarity, every preset's prior-art range, the 5.8%/90.4%/15.7× arithmetic).
  All four HIGHs were again follow-ons to R3's own fixes:
  **H1** `use_figure = clip.requires_figure()` while `requires_figure()` was specified as overlay-driven only →
  an **annotated clip with no overlays routed to the BARE producer**, which is exactly the legacy delegation →
  `requires_figure()` now returns True for `style=="annotated"` OR any figure-only feature, with a parametrized
  test covering the no-overlay annotated case. **H2** `preview()` was specified in Step 1.3 but needs producers
  built in Step 1.4 (and would have inverted the clip→render module dependency) → `preview_frame` is implemented
  in Step 1.4; `Video.preview` is a declared stub whose body is a local import (the `SimulationResult.video`
  pattern); its test moved with it. **H3** the R3 `CODECS` map was dead code — the loop called
  `open_writer(path, fps, backend)` with no format, and `open_writer` still hard-coded `pixelformat="yuv420p"`,
  so the webm test could not pass → `open_writer(..., fmt, bitrate=…)` now looks codec+pixfmt up from `CODECS`.
  **H4** Phase 2's "ONE shared time stamp" was unreachable: promotion set only `colorbar`, leaving `show_time`
  to resolve bare→off while the burn path is skipped for promoted clips → multi-panel now defaults **both** on.
  **M/L** `enforce_capabilities` now also receives `figsize`/`dpi`/`title` (so `title` on a bare clip is not a new
  silent no-op); the figure suptitle **re-composes** `"{title} — t = …"` each frame instead of overwriting it;
  `masked_iter`'s yield contract pinned to per-frame `(Nx,Ny)` arrays; the `auto99` subsample replaced with a
  **one-pass-implementable per-frame stride** (the previous "every k-th of the total" needed a count no iterator
  can supply) and the stale reservoir wording removed from Risk; Success Criteria's "never mutates a registered
  colormap" corrected to the caller-supplied-instance framing; `try/finally` + `plt.close(fig)` mandated (figure
  leak / truncated-file-holding-an-NN-slot); the GIF frame cap moved into `stride` computation (a post-hoc drop
  doubled playback speed and desynced `n_frames`); **isochrones no longer hard-raise for non-result inputs** —
  `analysis.activation_time_interp` is an existing NUMPY path (analysis.py:593) that serves them; `fit_frame`
  guards a 1-node **x** axis as well as y; the two orphaned MODs (`test_self_contained.py`, `API_REFERENCE.md`)
  are now owned by Steps 1.5/3.1 checklists; `CLAUDE_SCRATCHPAD` (which does not exist) replaced with `TMPDIR`;
  baselines key on the **test ID** via `awk` (the `-rf` exception text varies run to run and produced spurious
  "new" failures); the parallel-session note refreshed (**Stim Phase 2 has LANDED — the dict path already warns,
  so new fixtures must use `Stim`**; and `cardiac_core/tutorials/PLAN.md` is **gated on this plan**);
  `VideoInfo.codec` defined for every backend; interpolation's double duty (imshow data-space vs PIL RGB-space)
  now validated rather than silently collapsed; font size clamped to ≥8 (grid-sized frames would compute 0);
  1-px rounding overshoot cropped before padding; runtime-dependency note added; the viridis/inferno item
  downgraded from a blocker to a flagged cosmetic preference (the plan is executable as written).
- 2026-07-23 — **REVISED after audit R5 (0C/3H/11M/17L, all addressed).** R5 confirmed 0 CRITICAL again,
  re-verified the R4-era source claims (`activation_time_interp` at analysis.py:593 is genuinely numpy with a
  positionally-compatible signature; Stim Phase 2 landed with the dict `DeprecationWarning` at api.py:1301;
  `tutorials/PLAN.md` is genuinely gated on THIS plan; `fields` genuinely absent from
  `test_subpackage_importable`; `pyproject` deps are only `mcp>=1.2.0`), and confirmed the `TMPDIR`/`awk`
  baseline shell is correct POSIX. All three HIGHs were once more follow-ons to R4's own fixes, and R5
  localised them: **Step 1.4's `render()` pseudocode had become the least internally consistent block in the
  plan.** Acting on its recommendation, that block was **rewritten as one explicitly ordered sequence**
  (enforce → select_backend → stride+GIF-cap → resolve → path → writer → loop), which closed all three at once:
  **H1** the GIF cap was specified at a point where the backend was not yet known, and `min(None, 200)` would
  `TypeError` on the legacy `max_frames=None` path → backend now precedes stride and the cap is None-safe;
  **H2** `bitrate` was passed to `open_writer` but declared nowhere → it is now a real `render()` parameter
  defaulting to `"2M"` for webm/VP9, and the webm test's assertion mechanism corrected (the VP9 message is
  **ffmpeg subprocess stderr**, not a Python warning — `pytest.warns` would never see it); **H3** the
  de-hard-raised isochrone path still carried its old `ValueError` test plus a duplicated, contradictory
  branch → one `isochrone_lat(clip)` helper is now the single definition, its test flipped to assert the numpy
  fallback WORKS, plus new tests for the 1-frame wrap and for mask-respect (LAT is now computed from MASKED
  display values, so LBM's finite obstacles no longer get spurious contours — the R2 masking fix propagated to
  the overlay path).
  **M/L** `rest_anchored(vmax=…)` was silently dropped (`resolve` hard-coded 40) → a `rest_vmax` field;
  `resolve`'s if/elif could fall through with `lo`/`hi` unbound → explicit `else: raise` + `__post_init__`
  enum/interpolation validation (the interpolation rule previously belonged to no step and had no test);
  `::test_auto99_within_auto` was named in Risk but never defined, with a malformed `auto <= auto99 <= auto`
  assertion → defined properly as lo/hi bracketing; the §10 canary used `Grid(40,10)` but `dx` is a required
  positional (`grid.py:40`) → `Grid(40, 10, 0.025)`; the baseline gate grepped only `^FAILED` under `-rf`, so a
  collection **ERROR** — the likeliest failure when adding a subpackage — produced an empty, passing diff →
  `-rfE` + `^(FAILED|ERROR)`; Step 1.5 edited `test_self_contained.py` without running it → added to its Verify;
  `title` on a bare clip had no raise-test; Phase 2's shared-time-stamp fix was prose-only → two tests added,
  and the pseudocode now states `enforce_capabilities` is skipped on the promoted path; contour coordinates are
  now derived from the SAME `extent` as the image (a cm-space contour on a node-index axis was possible for
  every `.npz`/array clip); `try/finally` now `os.remove`s the truncated file (closing alone does not release
  the consumed `media_path` NN slot); an all-NaN LAT guard before contouring (mirroring `viz.py:82`);
  `_Writer` and `enforce_capabilities` given declared interfaces; `gradient.py` switched to
  `matplotlib.colormaps[...]` so it never imports pyplot (removing an Agg-backend import-order dependency);
  Step 1.1 now states `video/__init__.py` is a STUB at that point and owns the creation of `test_video.py`;
  the header audit banner updated to R1→R5; the legacy `ax.set_title`→`fig.suptitle` move documented as a
  second deliberate change; the streaming claim scoped honestly (`__post_init__` does materialise one float64
  copy); `pyproject` include quote corrected.
- 2026-07-23 — **AUDIT R6: 0 CRITICAL / 0 HIGH → CONVERGED.** R6 verified every R5 fix landed correctly
  (the ordered-sequence rewrite, `rest_vmax`, the `else: raise`, `__post_init__` validation, `Grid(40,10,0.025)`,
  `-rfE`, the contour/extent coupling, `os.remove`, the all-NaN LAT guard, `matplotlib.colormaps`), re-verified
  ~40 source claims, and confirmed blueprint compliance (9/9 sections on all 8 steps; all 8 per-phase elements;
  Final Cleanup archive→tmux). Its 11 MEDIUM + 19 LOW were then folded in as a single non-adversarial editing
  pass, per its own recommendation:
  **M** `build_figure` + `open_writer` moved INSIDE the try (a raise in the most failure-prone call was leaking
  the writer and stranding a file on the consumed NN slot — the very failure `os.remove` was added for);
  `::test_gif_fallback_works_without_imageio` must disable **cv2 as well as imageio** (opencv 4.13 is installed,
  so patching only imageio silently exercised the OpenCV mp4 path and left the PIL-only guarantee untested);
  `VideoInfo` gained a `bitrate` field because the webm assertion was otherwise impossible (imageio sets
  `ffmpeg_log_level="quiet"`, so neither `pytest.warns` nor `capfd` can observe the VP9 message);
  `ax.imshow` now pins `origin="lower"`/`cmap`/`norm` and a cross-producer orientation test was added (a vertical
  flip between the bare and figure producers would otherwise pass the entire suite — a silent scientific error);
  `build_figure` returns a declared `_FigState` carrier instead of leaving `im`/`ax`/`Xc`/`Yc`/contour as free
  names (Phase 2 needs N of them); `SimulationResult.video` now SPLITS Video-level kwargs from render-level ones
  (previously `r.video("x", gradient=…)` — the exact call §10 teaches — was a `TypeError`); `isochrone_lat` builds
  its masked stack only in the numpy branch and only over `idx` (it was unconditional and unstrided, violating the
  plan's own memory rule) and the torch branch is now gated on `field == "Vm"` (it would otherwise draw Vm
  isochrones over a phi_e clip); the no-new-failures gate gained a collect-only guard (it passed VACUOUSLY when
  the after-run produced no short summary at all); Step 1.2's `-k` filter widened to its two new tests; and
  `style`/`aspect`/`units`/`fit` are now validated like the other enums (`style="anotated"` silently rendered
  bare, losing axes and colorbar).
  **L** the None-unsafe cap prose, `open_writer`'s pseudocode header, the webm `pixelformat=None` no-op
  (imageio re-supplies yuv420p), `preview_frame` now shares `enforce_capabilities` and defines `t_ms` resolution,
  the `autoscale` prior-art cell corrected (the `diag_*` family is inferno with a FIXED vmax — this preset
  generalises rather than reproduces it), the test-file append list, Phase 2's scope count, Step 1.4's checklist
  (ordering + bitrate), the stale isochrone-test parenthetical, a warning when `speed` is clamped, `Gradient.key`
  reworded as comparable-not-hashable, and §10 mandated to use `cc.Stim` (the dict path now warns).
  **Also fixed during this pass, from my own empirical checks rather than the audit:** the baseline gate's
  `grep -E "^(FAILED|ERROR)"` was capturing `conda run`'s OWN `ERROR conda.cli.main_run:…` line and turning it
  into a bogus test id in every failing baseline → `grep -v "conda.cli"` added in all 6 places; and I confirmed
  by measurement that `-rf` reports **0** pytest collection errors where `-rfE` reports them, that
  `activation_time_interp` accepts the NaN-masked array cleanly (obstacle → all-NaN, tissue → finite, no
  warnings), and that a 1-frame clip returns all-NaN rather than crashing (so that guard is for a clear warning,
  not to prevent an exception — an earlier overstatement corrected).

- 2026-07-22 — **further empirical verification of the R2-revision's OWN new claims** (pre-empting R3):
  `matplotlib.get_data_path()/fonts/ttf/DejaVuSans.ttf` **exists**, and at `size=H//40` renders a **20 px** stamp
  at 1080p vs **8 px** for PIL's default bitmap font (so the ≥15 px test discriminates correctly); the bundled
  ffmpeg **does** ship `libvpx-vp9` (and `libvpx`, `libx264`), and a `webm` writes successfully with or without
  `pixelformat="yuv420p"`. **NEW finding folded in:** VP9 warns `"Neither bitrate nor constrained quality
  specified, using default CRF of 32"` because imageio's `quality=` does not map to a VP9 rate control → the webm
  path must pass an explicit bitrate/CRF, and `open_writer` gained a `bitrate` parameter plus a per-format codec
  rule. `select_backend` also now handles `format="webm"` with no ffmpeg backend (OpenCV cannot produce webm).
