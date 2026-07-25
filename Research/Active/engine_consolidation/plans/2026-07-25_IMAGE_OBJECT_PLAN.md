# PLAN: cardiac_core `Image` layer — spec-first still figures, traces, multi-panel

Created: 2026-07-25
Engine(s): cardiac_core (media layer; **no solver changes at all**)
Research question: [engine_consolidation](../Research/Active/engine_consolidation/README.md)
Source: user directive 2026-07-25 — *"notice what we built over video. do the same equivalent but for
`.image`; because we assume people don't know matplotlib"* — plus a census of the repo's figure corpus
and the shipped `cardiac_core/video/` layer this mirrors.

> **✅ EXECUTED 2026-07-25 — all 4 phases shipped** (`fba6ed3` → `c3ace72` → `e4079d4` → `7ba147f` on
> `video-portable-output`). **580 passed / 7 failed / 2 xfailed**; the 7 are the identical CUDA-OOM set
> present in the pre-implementation baseline (another user holds 32 GB of the shared GPU). **No NEW failures
> at any of the four phase gates; integrity goldens bit-identical throughout** — no solver code was touched.
> **+87 tests.** Both preview paths are pixel-identical to their pre-phase capture; the two viz stills land at
> +5.6 %/−13.4 % and +10.6 %/−13.4 %, exactly the deviations R8 measured when it proved equality unachievable.
> All four corpus acceptance compositions reproduce in ONE `draw()` call. This file is now a historical record.
>
> **Implementation-time finding worth keeping:** the plan's R9 M-2 rule earned itself back immediately — the
> first cut of `draw()` used the *condemned* "differs from the default" comparison, so `resolution="auto"` on
> an annotated spec silently did nothing. The Verify block the plan mandates caught it in seconds. Two test
> defects of the same family also surfaced during the build: a needle asserting `"not directly comparable"`
> against a message reading `"NOT directly comparable"`, and a `pytest.warns` defeated by Python's
> per-location `__warningregistry__` de-duplication.

> **Design premise.** The `Video` layer proved the shape: a **spec object holds the description**, a **verb
> turns it into bytes**, **rendering displays and naming a destination saves**, colour is a **reusable
> `Gradient`**, and **multi-panel is native**. This plan applies that shape to STILLS — and extends it to the
> figure kind the video layer has no analogue for and which dominates the corpus: the **line plot**.
>
> **Audit R1** (5C/8H/15M/11L) → all folded in. Decisive catch: the load-bearing premise *"an `Image` is a
> one-frame `Video`, so reuse the producers"* had never been executed against the real `clip.py`. Four of the
> five things `Video` derives from a result (masking, `dx`/`dy`, `value_label`, `times`) are computed in
> `__post_init__` and are **not** recoverable by a post-hoc copy.
> **Audit R2** (1C/5H/13M/12L) → all folded in. **Five of its six C/H findings were follow-ons to R1's own
> fixes** — the documented failure mode of this cycle. Decisive catches: R1's `gradient=None` default **crashes
> the zero-argument headline call**; R1's `Video` dispatch had no parameter to dispatch on; R1's new bare
> pipeline silently **deletes** the burned time stamp it promised to preserve; R1's `filled` mode meets a
> `_produce_figure` nobody re-read; R1's `lat=` seam stops one function short of the multi-panel path.
> **Audit R3** (3C/7H/13M/9L) → all folded in. **Eight of its ten C/H findings were follow-ons to R1/R2 fixes.**
> R3 re-verified and confirmed the census, the baseline, the working-tree inventory, all ~30 `file:line`
> citations, the 15-name import list, the rank rule, the format matrix, and the R2 gradient/`filled`/upscale
> fixes as *necessary and sufficient*. What it still broke: **R2's own H-3 and M-14 fixes contradict each
> other** and would turn a green test red; the delegated annotated preview **silently resizes 1.5×** (dpi
> 100→150); **`lat=` alone draws nothing** because the overlay gate is `if clip.isochrones:`, which the recipe
> hard-codes `False`; `fig.colorbar(None, …)` **does not raise** — it fabricates a meaningless 0–1 bar; and
> **every APD assertion in the plan was vacuous** because `apd_map` is 100 % NaN on the fixtures the plan used.
> **Audit R4** (3C/4H/7M/9L) → all folded in. R4 re-verified the census, baseline, tree inventory, every line
> citation, the rank rule, the format matrix, the fixture timings and R3's colorbar fix as **correct**, then
> found: R3's `dpi` fix would **raise on every bare preview** (4 green tests); the viz pixel-equality criterion
> is **unsatisfiable** (the residual height comes from `suptitle`+`tight_layout`, not `figsize`); the
> `activation_isochrones` delegation **re-creates the double-draw R2 removed**; the overlay bypasses masking;
> and **no fixture in the plan can test restitution**. Its most useful finding was structural, not a bug:
> *three rounds had each added a rule to the `isochrones`/`lat`/`filled` triangle and none had written the
> three down together* — hence the **RESOLVED SEMANTICS** section below.
> **Audit R5** (1C/5H/8M/10L) → all folded in. R5 re-verified ~40 line citations, every measured number, the
> tree inventory and the export lists, and confirmed R4's `dpi` and ±15 % fixes as correct — *"the factual
> layer of this plan is in excellent shape"*. Its finding was that R4 **declared the sweep but did not execute
> it**: the resolved rules lived in prose while the *contradicting* forms stayed in the implementation steps,
> i.e. the wrong half sat where an implementer copies from. Three of its six C/H findings were that single
> omission; a fourth was that the Verify meant to catch it computed its own answer and asserted on that.
> **The sweep is now done** — every mention of `_lat`/`isochrones`/`filled`/`show_time`/`resolution` defers to
> or restates RESOLVED SEMANTICS, superseded forms are deleted rather than annotated, and the Verify drives
> `draw()` instead of poking `_build_figure`. Per-finding disposition in the Mutation Log.

## Objective
Give cardiac_core one obvious, built-in way to turn a run into a **figure**, so neither research scripts nor
lab members hand-roll matplotlib. The unit of description is a spec object — `Image` for a spatial map,
`Trace` for a series — and `draw()` turns it into bytes that **display inline and are written only when a
destination is named**. Colour reuses the shipped `Gradient`. Multi-panel is native.

## THE DEFAULT — the common case
> `r.image()` with no arguments must produce **the figure we want most of the time**: a readable, labelled
> picture of the run that a lab member can drop into a slide without touching matplotlib.

| Default | Value | Why |
|---|---|---|
| content (`what`) | `"snapshot"` — Vm at the **middle saved frame** | Symmetric with `r.video()`; matches `preview_frame`'s existing `len(frames)//2` (`render.py:421`). ⚠ see the COSMETIC PREFERENCE block. |
| `style` | **`"annotated"`** — axes, colorbar, units, time stamp | **Deliberate DIVERGENCE from video's bare default.** A video carries information through motion; a still carries it through labels, and this audience cannot add labels themselves. `style="bare"` is one keyword away. |
| `gradient` | `Gradient.physiological()` for voltage; **per-`what` for derived maps** | A derived map is not in mV — Step 1.2 item 2. |
| `units` | `"auto"` → cm when `dx`/`dy` are known, else node indices | `"y (cm)"` is the corpus's most common map y-label. |
| `format` | `"png"` | Universal; `svg`/`pdf` opt-in for publication (both verified). |
| `size` — annotated | `figsize=None` (so `_build_figure`'s aspect-aware sizing survives), `dpi=150` | L-38: a fixed figsize letterboxes long strips badly. **`resolution=`/`fit=` do NOT apply here** (M-14). |
| `size` — bare | `resolution="auto"` → integer nearest upscale to a long edge ≥ 512 px, **no padding** | M-15: an unfitted bare still is a 30×8 postage stamp, but a padded 1080p canvas is ~53% black — and a bare still is PURE DATA, so every pixel should be data. |
| `tight` | `None` → **True** on the annotated branch (`bbox_inches="tight"`); **any NON-`None` value** raises on the bare branch (R11 M-2 — `None` itself must not raise, or every default bare draw would) | What `viz.py` does today. ⚠ It **changes the output pixel size**. |
| `question` | `"lab"` | Matches `viz`/`video`. |
| `bulk` | `True` **when a media keyword is used** | Identical to video: regenerable by default. |
| destination | **none** → displays inline, writes nothing | The matplotlib contract, and the Colab lesson from `video-portable-output`. |

**ONE COSMETIC PREFERENCE, NOT A BLOCKER — `"snapshot"` or `"activation"` as the default `what`?**
A snapshot is *symmetric* with `r.video()` and never fails; an **activation map** summarises the whole run and
cannot be blank because the wave already passed. **This plan is executable as written: the default is
`"snapshot"`.** Flipping it is a one-line change to the `what: str = "snapshot"` dataclass default in Step 1.2,
plus `::test_default_is_snapshot` (R7 L-1: there is no `_DEFAULT_WHAT` constant — do not invent one).
**Does not gate implementation.**

## Motivating gaps (verified in source, 2026-07-25)
1. **No trace/plot route exists at ALL.** cardiac_core can produce **two standardized figures**
   (`apd_map_figure`, `activation_isochrones`) plus `Video.preview()` — and every one is a spatial map
   (L-25/L-30; `propagation_video` is a video, not a figure). The dominant corpus figure kind is the line plot.

   Census — **command and numbers reproduce** (M-7 replaced an earlier command whose `-h -o` flags stripped the
   path, making its `grep -v` exclusion a silent no-op):
   ```bash
   grep -rnoE "\b(ax[0-9a-z_]*|plt)\.(plot|imshow|axhline|axvline|contour)\(" --include=*.py . \
     | grep -v "_archive/\|code_examples/" | sed -E 's/.*\.([a-z]+\()$/\1/' | sort | uniq -c | sort -rn
   ```
   Excluding `Monodomain/_archive/` and vendored `Research/code_examples/`:
   `plot(` **163** · `axhline(` **74** · `imshow(` **70** · `axvline(` **32** · `contour(` **29**, over **87**
   files containing `savefig`. Repo-wide (no exclusions): `383 · 112 · 115 · 54 · 36` over 167 files.
   **Line plots dominate either way** (the 2nd/3rd places swap — excluded has `axhline > imshow`, repo-wide has
   `imshow > axhline` — so do NOT claim the ranking is identical, R3 L-1), and `axhline`+`axvline` = **106**
   excluded / **166** repo-wide make **reference lines a first-class requirement**.
2. **`viz`'s two stills always write a file** and return a path string — they cannot display inline, so they
   are useless in Colab exactly as `r.video()` was before `video-portable-output`.
3. **The two stills are frozen compositions** — hard-coded `figsize=(6,3)`, `aspect="auto"`, node axes,
   `dpi=120`, own colorbar label, no masking.
4. **Masking is absent from both.** `apd_map_figure` does `np.where(isfinite, apd, nan)` only, and **LBM leaves
   masked obstacle nodes FINITE** — measured on a real LBM run: a masked node reads **+44.6 mV, finite**. Both
   stills paint an obstacle as fully depolarized tissue today.

## Success Criteria
- [ ] **Zero-argument default** — `r.image()` returns a displayable `ImageInfo`, writes **no file**, and shows
      an annotated Vm snapshot with cm axes + colorbar + time stamp. **It must not raise** (R2 C-1).
- [ ] **Drawing displays; naming a destination saves.**
- [ ] `Image` spec: correctly **masked** (`domain_mask` ∪ finiteness), correctly **labelled** (the drawn
      colorbar shows the quantity's own units), with **working isochrone contours** in both single- and
      multi-panel layouts.
- [ ] `Trace` spec: N named series, `hline`/`vline` reference lines, legend, log axes, markers, `linestyle`
      (so a marker-only scatter is expressible), `xlim`/`ylim`.
- [ ] `what=` registry covering the corpus intents, **rank-correct** for every `fields.*` member.
- [ ] `draw(spec | [specs] | Video, ...) -> ImageInfo`; map panels sharing a `Gradient` **and** a `value_label`
      share ONE colorbar.
- [ ] Formats `png`/`svg`/`pdf`/`jpg`/`jpeg`/`webp`; `pdf`/`webp` rejected on the `media/` path clearly;
      **an extension alias (`.jpeg`) is never silently rewritten**.
- [ ] `SimulationResult.image(...)` / `.trace(...)`; `SingleCellResult.trace(...)`.
- [ ] `viz.apd_map_figure` / `viz.activation_isochrones` delegate preserving signature, **title**, **`figsize`**,
      composition, always-save and `str` return. `Video.preview()` delegates **pixel-identically on BOTH the
      bare and the annotated path** (R3 C-2), does **not raise** (R3 C-1), and returns `ImagePath`.
- [ ] **torch→numpy at ingest**; the frame is indexed **before** conversion.
- [ ] No NEW failures vs a baseline captured BEFORE the phase; integrity goldens bit-identical (atol=0).
- **OUT OF SCOPE, stated (M-18 + R3 M-13):** in-axes **`text`/`annotate` (48 corpus calls — larger than
  `contour`'s 29)**, `fill_between` (4), `errorbar` (0), `bar` (4), `scatter` (6), `pcolormesh` (3), twin axes
  (1), and the `analysis` maps with no `fields.*` equivalent (`phase_map`, `phase_singularities`,
  `wavefront_mask`, `radial_cv`). **This is a decision, not an oversight**: each is a new field on a spec whose
  surface is already the largest risk in this plan, and the escape hatch — `Trace(series=…)` /
  `Image(field=<array>)` plus raw matplotlib — is genuinely available. Revisit after v1; `annotations=` is the
  most likely v2 addition.

## The figure intents (the `what=` registry)
| `what` | Kind | Content |
|---|---|---|
| `"snapshot"` *(default)* | map | Vm at `at=` ms (default: middle frame) |
| `"activation"` | map | canonical interp/−40 LAT + isochrone contours |
| `"apd"` | map | APD90 map |
| `"frequency"` | map | dominant-frequency map (Hz) |
| **static scalar** `fields.*` | map | `speed`, `curvature`, `divergence`, `vorticity`, `quality` — `(Nx,Ny)` |
| **static vector** `fields.*` | map | `velocity`, `direction` — `(Nx,Ny,2)` → `.magnitude`, **no frame selection** |
| **time-varying** `fields.*` | map | `source_sink` `(T,Nx,Ny)`; `voltage_gradient`, `voltage_flux`, `electric_field`, `current_flux` `(T,Nx,Ny,2)` → `.magnitude` → frame at `at` |
| `"trace"` | trace | Vm(t) at one or more nodes |
| `"restitution"` | trace | APD vs DI (marker-only) |
| `"apd_per_beat"` | trace | APD per beat (alternans staircase) |

## Architecture Changes
- NEW `cardiac_core/image/`: `info.py` (`ImageInfo`), `panel.py` (`Image`, `Trace`), **`_draw.py`**
  (`draw()`, layout, registry — **the leading underscore is MANDATORY**, see below), `__init__.py` — whose `__all__` is `["Image", "Trace", "draw", "ImageInfo"]` and which
  `_LAZY` resolves every one of those through (R9 L-14a). **At Step 1.1 only `info.py` exists**, so
  `__init__.py` must import lazily or export only what is present, or Step 1.1's `import cardiac_core.image`
  Verify fails.
  **⚑⚑ R10 H-2 — the module MUST be `_draw.py`, not `draw.py`.** A submodule named `draw` alongside a public
  `draw` export reproduces the **`single_cell` shadowing bug this plan already forbids** ("Never name a `_LAZY`
  export the same as its submodule"): `importlib.import_module('.draw', …)` binds
  `cardiac_core.image.draw = <module>`, PEP 562 `__getattr__` then never fires again, and `cc.draw` — **the
  headline verb of this entire layer** — is a non-callable module on every access after the first. Reproduced
  against this project's own non-caching `__getattr__` (`cardiac_core/__init__.py:68-73`):
  `1st: function · 2nd: module · "'module' object is not callable"`. **The guard's collision check cannot catch it** — `test_single_cell.py:26` compares `n == mod`, and the entry
  is `'draw': 'image'`, not `'draw': 'draw'`. Its **third** assertion (`getattr(cc, name) is first`) DOES fail
  (verified), so the bug surfaces — but only as an opaque identity error with no pointer to the cause
  (R11 L-1 corrects an earlier overstatement that the guard could not catch it at all). `_draw.py` matches the `_single_cell.py` precedent the rule was written from. (Caching into
  `globals()` inside `image/__init__.py.__getattr__` also works — verified — but the rename is the established
  convention and needs no new machinery.)
- MOD `cardiac_core/video/render.py` — **all default-preserving**:
  - `_build_figure` **and `_setup_panel`** (H-6) gain `lat=None, contour_levels=12, filled=False`.
  - `_produce_figure` guards its data swap with `if hasattr(st.im, "set_data"):` (H-4 + R3 H-2) — in `filled`
    mode `st.im` holds a `QuadContourSet`, which is the colorbar mappable but has no `set_data`.
  - `_render_panels`' per-frame loop (`render.py:613`) gets the same `hasattr` guard. Defensive only:
    `_render_panels` never passes `filled=`, so its `st.im` is always an `AxesImage`. **Owned by Step 1.3's
    edit list** (R8 M-9 / R9 L-8 — it previously sat here in Architecture, which owns no steps, and would have
    been silently skipped).
  - `preview_frame` delegates via a **function-local** import (Phase 3).
- MOD `cardiac_core/run.py` — `SimulationResult.image(...)`, `.trace(...)`.
- MOD `cardiac_core/_single_cell.py` — `SingleCellResult.trace(...)`.
- MOD `cardiac_core/viz.py` — both stills delegate.
- MOD `cardiac_core/__init__.py` — `_LAZY` += `Image`, `draw`, `ImageInfo` (Phase 1), `Trace` (Phase 2).
  **Add each name only in the phase that implements it** — `test_public_exports_not_shadowed_by_submodules`
  `getattr`s every `_LAZY` name.
- MOD `cardiac_core/tests/test_self_contained.py` — add `"image"`. **Verified current list:**
  `["ionic","mesh","stimulus","fields","video","_monodomain","_bidomain","_lbm"]`.
- MOD `API_CHEATSHEET.md` §10; MOD `API_OBJECTS.md`; MOD `.claude/skills/sim-media/SKILL.md`.
- NEW `cardiac_core/tests/test_image.py`.
- **Import direction**: `image/` imports `video/` at module scope. Phase 3 reverses one edge — `preview_frame`
  uses a **function-local** `from ..image._draw import draw`, as `SimulationResult.video` and `Video.preview`
  (`clip.py:205`) already do. **Never add a module-scope `image` import to `video/`.**
- **The import FORM is load-bearing** (M-16 R1): `video/__init__.py` does `from .render import render`, so
  `cardiac_core.video.render` **as an attribute is the `render` FUNCTION, not the module** — and so is
  `import cardiac_core.video.render as R` (verified: `AttributeError: 'function' object has no attribute
  '_FigState'`, R4 L-5). The module is reachable **only** via `from ..video.render import <name>` (or
  `sys.modules`). Use, verified importable (L-21 — the last four live in `encoders.py`, not `render.py`):
  ```python
  from ..video.render import (_build_figure, _produce_figure, _produce_bare, _extent_and_labels,
                              _named_destination, _resolve_destination, _finalize,
                              enforce_capabilities, _default_layout, _setup_panel, _PAD_BLACK,
                              _LEGAL_FIT)          # R9 L-10: needed by the fit validation
  from ..video.encoders import resolve_canvas, fit_frame, burn_timestamp, ImagePath
  ```
- OUT OF SCOPE: any solver/engine/analysis change; a new colour object; animation; widening `media._IMAGE_EXT`.

## ⚑ RESOLVED SEMANTICS: the `isochrones` / `lat` / `filled` triangle
**R4's design finding: three separate rounds each added a rule to this triangle and none wrote the three down
together — which is exactly why R4 found a double-draw (C-3), an unmasked overlay (H-1) and two contradictory
compute rules (H-3) living in it.** This paragraph is the single source of truth; every other mention defers
to it.

1. **`isochrones` (resolved, item 3) is the ONLY switch.** **An explicit value always wins**; only when the
   caller leaves it `None` does it derive to `True` iff `what == "activation"` **and not `filled`**:
   ```python
   resolved = self.isochrones if self.isochrones is not None else (what == "activation" and not filled)
   ```
   ⚑ **R5 H-3**: an earlier phrasing — *"`True` iff the caller passed `isochrones=True`, OR `what ==
   "activation"` and not `filled`"* — is boolean-wrong. Read as written, `Image(what="activation",
   isochrones=False)` gives `False or (True and True)` → **True**, silently discarding the user's explicit
   "no isochrones". The tri-state is the whole reason this rule exists; do not collapse it to a disjunction.
   Nothing else turns the overlay on.
2. **`_lat` is a CACHE, not a switch.** It is populated for `what == "activation"` because the selector already
   computed that array. Whether it reaches `draw()` is decided **solely** by rule 1 — and when the switch is on
   but the cache is empty, rule 4 fills it (R5 M-6: the snippet must be COMPLETE, because it is the line an
   implementer transcribes; a `lat = <cache or None>` one-liner silently draws nothing for
   `Image(what="snapshot", isochrones=True)`):
   ```python
   lat = None
   if isinstance(spec, Image) and spec.isochrones:          # rule 1, the ONLY switch
       lat = spec._lat if spec._lat is not None else _lat_from_result(spec)   # rule 4 (returns NUMPY)
       # rule 5: mask it exactly as the display array is masked
       if lat is not None and spec._clip.active_mask is not None:
           lat = np.where(spec._clip.active_mask, lat, np.nan)
   ```
   ⚑ Without the `spec.isochrones` conjunct, `activation_isochrones` (`what="activation", filled=True,
   isochrones=False`) populates `_lat`, the gate fires on `lat is not None`, and white lines are drawn over the
   `contourf` bands — **re-creating the exact double-draw R2's H-5 fix removed**, and shipping it in the
   delegation (R4 C-3).
3. **The overlay gate is `if lat is not None or clip.isochrones:`** in both `_build_figure` and `_setup_panel`.
   The `clip.isochrones` half is vestigial for `Image` (item 6 hard-codes it `False`) and live for `Video`.
4. **Computing `lat` when `_lat` is empty** — for `isochrones=True` on a non-activation selector (e.g. a
   snapshot with isochrones, the only way to get isochrones on a voltage map):
   `analysis.activation_time(result.Vm, result.times)` with **NO `what_kwargs`** (R3 H-3 — they belong to the
   selector's own function; `repol=` is a `TypeError`, `threshold=` silently retargets the LAT). If
   `result is None`, warn *"isochrones need a SimulationResult to compute activation times; skipping the
   overlay"* and leave `lat = None`.
   ⚑ **Never compute a LAT for an `Image` whose resolved `isochrones` is False** — an earlier revision's
   "otherwise compute it" rule drew isochrones on APD maps that never asked for them (R4 H-3).
5. **`lat` MUST be masked exactly as the display array is** (R4 H-1):
   `lat = np.where(self._clip.active_mask, lat, np.nan)` when `active_mask is not None`. `isochrone_lat` does
   this itself (`render.py:170-171`), and bypassing it bypasses the masking — `arr2d` from
   `analysis.activation_time` is **raw and unmasked**, so contours would run straight through a grey obstacle
   while the imshow beneath shows it masked. Apply after clip construction (item 5), never before.
6. **`filled` is a rendering mode, not an overlay.** `filled=True` ⇒ `contourf` bands *are* the isochrones;
   `imshow` is not drawn; the `QuadContourSet` is the colorbar mappable and goes on `_FigState.im`.

**Test the triangle by counting artists, never by `len(data) > 1000`** — measured in mpl 3.10.8: a line contour
adds exactly 1 collection and `contourf` alone is also 1, so `line ⇒ images==1, collections==1` vs
`filled ⇒ images==0, collections==1` discriminates, and a double-draw shows up as `collections==2`.

## Reuse decision — what transfers and what does NOT
A spatial-map still **is** a one-frame video panel, and the video layer already solved torch→numpy ingest, the
masking seam, cm-vs-node extents, the orientation equivalence, `Gradient` resolution over masked values, and
both producers. **Decision: `Image` composes a `Video` internally** (`self._clip`) and `draw()` calls the same
producers. R1 proved this only works if the clip is constructed correctly up front:

| `Video` attribute | Set where | Post-hoc? | So `Image` must… |
|---|---|---|---|
| `active_mask` | `__post_init__` (`clip.py:96-106`) | **NO** | pass `mask=` **into the constructor** |
| `times` | `__post_init__` (`clip.py:83-93`) | **NO** — and there is **no `times=` kwarg** (verified `TypeError`) | use the **`(times, V)` 2-tuple** `data` form |
| `gradient` | dataclass default (`clip.py:48`) — **an explicit `None` OVERWRITES it** | n/a | **resolve to a real `Gradient` before constructing** (R2 C-1) |
| `value_label` | `_resolve_data` — hard-coded `"Vm (mV)"` for tuple input (`clip.py:126`) | **YES** — read at draw time (`render.py:213`) | assign after construction |
| `dx`, `dy` | `_resolve_data` → `None` for tuple input | **YES** — read by `_extent_and_labels` | assign after construction |
| `result` | `_resolve_data` → `None` for tuple input | **YES** — read by `isochrone_lat` | assign after construction |
| `field` | `_resolve_data` → **stays `"Vm"`** for tuple input (M-8) | — | **do not rely on it.** Every `Image` clip reports `field='Vm'`, including an APD map. It reaches `Gradient.resolve(field=)`, `isochrone_lat`'s `is_vm` branch, `_render_panels`' kind check and `__repr__`. Never use it to identify an `Image`'s quantity. |

- **Rejected: re-implement the map producer** — it would duplicate the bugs the video layer already paid for.
- **Rejected (for now): extract the producers to a neutral `_draw.py`** — tidier, but rewrites a module
  carrying 105 green tests for no user-visible gain. Logged as a follow-on.
- **Consequence:** `image/` depends on `video/` at the module level of `_draw.py`/`panel.py` (though the lazy
`image/__init__` means a bare `import cardiac_core.image` does not pull it in — R12 L-2). `image/` must NOT
  import `imageio`/`cv2` at module scope; verified `encoders.py` keeps both **inside functions**
  (`encoders.py:365,375`), and Step 1.1 **tests** it.

## Known Failures / gotchas (verified 2026-07-25)
- **⚠ WORKING-TREE STATUS.** On branch `video-portable-output`, uncommitted **right now** (exactly 5 files):
  `__init__.py` (the `single_cell`→`_single_cell` fix), `tests/test_single_cell.py`,
  `tests/test_safety_factor.py`, **`video/render.py`** and **`tests/test_video.py`** (in-flight: an
  extension-rewrite warning in `_resolve_destination`, `writer.close()` moved inside the guard, +48 test lines).
  This plan MODs `__init__.py` and `render.py`. **Re-read each immediately before editing**; never revert a hunk
  you did not write. The Phase-1 baseline **was captured with this tree state**, which is the correct "before".
  **All `render.py` line numbers below are working-tree numbers and will shift when that hunk is committed**
  (L-20) — locate by symbol, not by line.
- **Verify env-dependent claims with `/opt/miniforge3/bin/conda run -n heart-conduction`.** Bare `conda` is not
  on the non-interactive PATH; `conda activate` does not work there at all.
- **⚑⚑ `conda run` SILENTLY DISCARDS STDIN — a heredoc script NEVER RUNS, and it exits 0** (R7 C-1).
  Reproduced:
  ```
  $ /opt/miniforge3/bin/conda run -n heart-conduction python - <<'PY'
  open('/tmp/_sidefx.txt','w').write('ran'); print("stdout line")
  PY
  exit=0
  cat: /tmp/_sidefx.txt: No such file or directory        # the script never ran
  ```
  So `conda run … python - <<'PY' … PY > out.txt` writes a **0-byte** `out.txt` and reports success — a
  silent-wrong-result with a green exit code. **Use `python -c "…"` (what every other block here does), or
  `conda run --no-capture-output` (verified working), or the env python directly
  (`/home/norepinephrine/.conda/envs/heart-conduction/bin/python`).** This applies to implementation work too,
  not just to this plan's Verify blocks.
- **The GPU is shared and currently full.** Baseline: **493 passed / 7 failed / 2 xfailed**, and **all 7 are
  `CUDA error: out of memory`** — 32000/32623 MiB held by PID 924308 (another user), including
  `test_video.py::test_torch_cpu_and_cuda_tensor_converts`. **Never gate a phase on an absolute pass count.**
- **`media_path` REJECTS pdf and webp.** `media.py:55` `_IMAGE_EXT = {"png","jpg","jpeg","svg","gif"}`.
  `_IMAGE_EXT` is private to `media.py` (refs: `media.py:55,103`); `media_path` itself is used by **54 files**,
  so widening the set changes a shared convention — out of scope.
- **`_resolve_destination` silently REWRITES a mismatched extension** and warns with a *video-specific*
  explanation ("the encoder produced … backend downgrade"). **This fires on legitimate aliases too**: a
  `.jpeg` path against `ext="jpg"` is rewritten to `.jpg` (M-13). `draw()` must therefore (a) RAISE when an
  explicit `format=` disagrees with `path=`'s extension, and (b) **pass `path`'s own extension through
  unchanged** when it is legal — never normalise `jpeg→jpg`.
- **`bbox_inches="tight"` CHANGES the output size**, by an amount that depends on the figure's contents.
  **Never assert an exact `(w,h)` in a test that uses `tight=True`** — that is the whole operative lesson.
  Do not quote a literal ratio here: R3 and R4 measured different numbers for the "same" figure (R4 L-1), which
  is precisely the point. Capture a reference in the test if a size must be pinned.
- **⚑ APD NEEDS A LONG RUN — every APD assertion is otherwise vacuous (R3 H-5).** Measured on
  `Grid(30, 8, 0.025)`, TTP06:

  | fixture | wall | frames | `apd_map` finite | `activation_time` finite |
  |---|---|---|---|---|
  | `t_end=20.0, save_every=1.0` | **2.3 s** | 20 | **0 %** (all-NaN) | 100 % |
  | `t_end=400.0, save_every=5.0` | **44.6 s** | 80 | **100 %** (max 230.0 ms) | 100 % |

  A TTP06 APD90 is ~230 ms here, so a 12 ms or 20 ms run produces a **fully NaN** APD map — and an all-NaN map
  through `Gradient(value_range="auto")` falls back to **(−90, 40)** with a *"no finite unmasked data"* warning,
  which means the intended test *"a derived map does not use the −90…40 range"* would **FAIL** on the very map
  it targets. **Runtime is set by the solver `dt`, not the save cadence** (44.0 s at `save_every=2.0` vs 44.6 s
  at 5.0), so the long fixture is priced per-run, not per-frame. **Use THREE module-scoped fixtures**:
  | fixture | what it is | used by |
  |---|---|---|
  | `wave` | `t_end=20.0, save_every=1.0` — **2.3 s** | everything: snapshot, activation/LAT, `fields.*`, formats, guards, masking, bare sizing |
  | `long_wave` | `t_end=400.0, save_every=5.0` — **~44 s** (43.4–44.6 measured across runs; shared-GPU noise) | **APD maps and `apd_per_beat` only** |
  | `multibeat` | a **SYNTHETIC** `SimulationResult` — hand-built `times` + multi-beat `Vm`, no solver | **restitution only** |
  **⚑ R4 H-2 — `long_wave` cannot test restitution.** It fires a single `Stim.boundary`, and
  `restitution_curve` returns `(n_beats-1,)` (`analysis.py:516-519`). Measured on the exact fixture:
  `DI.shape = (0,)`, `APD.shape = (0,)`, `apd_per_beat = [225.0]`. So it exercises only the *warns-single-beat*
  half, and "works multi-beat" would have had **no fixture anywhere in the plan** — the same
  assertion-that-cannot-fail class as R3 H-5. A real 3-beat paced run would cost ~98 s (runtime scales with
  `t_end`, so 900 ms ⇒ ~2.2× the 400 ms run), which is **not** worth it: `restitution_curve` consumes only `V`
  and `times`, so a synthetic multi-beat array is exact, deterministic and free.
  **⚑ R5 M-8 — specify it concretely and pin it, or it silently reproduces the vacuity it fixes.**
  (`test_canonical_lat.py::_synthetic` is NOT the model to copy: it returns a 3-sample `(V, times)` **pair**,
  not a `SimulationResult`, and not a beat train.) A hand-built result works — only `times` and `Vm` are
  required — but a non-empty curve is not automatic: `restitution_curve` returns `(n_beats-1,)` on **detected**
  beats, and a 3-beat construction was measured to still yield `DI.shape == (1,)`. So:
  ```python
  # R7 M-4: BUILD IT HERE. `restitution_curve` needs a rising -20 mV crossing per beat, then a peak, then a
  # sample at or below V_peak - 0.9*(V_peak - V_rest); a naive 3-beat build was measured to give DI.shape=(1,).
  V_REST, V_PEAK, BCL, DT = -85.0, 20.0, 400.0, 1.0
  APDS = (225.0, 245.0, 215.0, 235.0)          # VARY per beat, or the curve is two identical points (R6 L-4)
  n = int(len(APDS) * BCL / DT)                # 1600 samples
  trace = torch.full((n,), V_REST, dtype=torch.float64)
  for k, apd in enumerate(APDS):
      s = int(k * BCL / DT)
      d = int(apd / DT)
      trace[s:s + 2] = V_PEAK                                        # upstroke, crosses -20 rising
      trace[s + 2:s + d] = torch.linspace(V_PEAK, V_REST, d - 2)     # linear repolarization
  V = trace.view(-1, 1, 1).expand(n, 30, 8).contiguous()             # (1600, 30, 8), matches `wave`'s grid
  r = SimulationResult(times=torch.arange(n, dtype=torch.float64) * DT, Vm=V)
  DI, APD = analysis.restitution_curve(r.Vm, r.times, 20, 4)         # node (20,4) = the exit criterion's
  assert DI.numel() >= 2 and APD.unique().numel() >= 2   # anti-vacuity AND non-degenerate
  ```
  Every other `SimulationResult` field defaults, so `times` + `Vm` is genuinely sufficient (verified).
  **Pin the resulting `DI`/`APD` shapes in the test** once measured, so a later change cannot quietly
  re-empty them.
- **`_to_numpy` lives in `clip.py:28`** — a THIRD module beyond `render.py`/`encoders.py` (R3 M-4). The full
  import surface is `from ..video.clip import Video, _to_numpy` plus the two lists above (**18** names, 3 modules — 12 from `render.py` incl. `_LEGAL_FIT`, 4 from `encoders.py`, 2 from `clip.py`).
- **`media_path`'s `NN` contract is get-path-then-save-immediately.**
- **`domain_mask` polarity is True = ACTIVE**; the corpus masks the COMPLEMENT.
- **LBM masked nodes stay FINITE** — mask via `domain_mask` ∪ finiteness, never `isfinite` alone.
- **torch→numpy is mandatory**; **index the frame BEFORE converting** (`_to_numpy(result.Vm[k])`) or a
  one-frame still materialises the whole history (L-37). **`analysis.*` is torch-only.**
- **`analysis.apd_map` defaults `threshold=-20.0`, `activation_time` `-40.0`** — deliberate; do not harmonise.
- **`restitution_curve(V, times, ix, iy, repol=0.9, threshold=-20.0) -> (DI, APD)`**, in that order.
- **`SingleCellResult`** is `times, V, final_state, model, dt, Cm` + `.apd()`, `.v_peak`, `.v_rest`; its `V` is
  **1-D `(T,)`**. It **has `.V`**, so `Video._resolve_data` (`clip.py:128`) accepts it and dies with
  "expected frames shaped (T, Nx, Ny); got (1010,)" — `Image` needs an explicit guard naming `.trace()` (L-27).
- **`Gradient._infer_v_rest` only guards `field == "phi_e"`** (`gradient.py:157-174`) — and every `Image` clip
  reports `field='Vm'`, so that guard never fires for a derived map. The `rest`/`zoom` protection **must** be a
  hard raise in `Image.__post_init__` (M-24/M-8).
- **`Gradient.levels` is colormap QUANTIZATION** (`gradient.py:78`, `out.resampled(...)`), documented in the
  shipped cheatsheet. `Image`'s isoline count is therefore named **`contour_levels`** (M-11) — never `levels`.
- **`_build_figure` hard-codes `levels=12`** while `viz.activation_isochrones` defaults to **15** — hence the
  `contour_levels` seam. `_setup_panel` hard-codes 12 as well.
- **`viz` sets panel titles** — `"APD map"` (`viz.py:72`), `"Activation isochrones"` (`viz.py:92`) — and
  `_build_figure` titles only from `clip.label`; delegation must pass `label=`.
- **`viz.py:89` guards `if np.isfinite(lat).any():`** — an all-NaN LAT yields a titled, contour-free,
  **colorbar-free** figure. The delegated path must reproduce that (H-5).
- **Never name a `_LAZY` export the same as its submodule** (the `single_cell` bug).
- **`tests/conftest.py` redirects `$CARDIAC_MEDIA_ROOT` to a tmpdir session-wide**; tests passing `path=` must
  use `tmp_path`.
- **Write new fixtures with `Stim`**, and **use a module-scoped fixture** (L-26) — the suite already takes
  336 s and ~75 new tests each building a `SimulationResult` would dominate it (`test_video.py`'s `wave`
  fixture is the pattern).
- **`matplotlib.use("Agg")` before any `pyplot` import**; **`plt.close(fig)` in a `finally`** — on EVERY path
  that creates a figure: the single-panel annotated branch, the Phase-3 layout figure, **and the Phase-2
  `Trace` figure** (R5 L-5; matplotlib warns past 20 open figures and ~20 new `Trace` tests would each leak one).
- **`enforce_capabilities`' inherited messages name the WRONG API for this audience — DECIDED, not deferred**
  (R5 L-4, R6 closing note). `r.image(style="bare", label="x")` raises *"…Use `Video.annotated(...)`"*
  (`render.py:139,148,150`) at a user who has never typed `Video`. **`draw()` catches the `ValueError` from
  `enforce_capabilities` and re-raises it with the `Image` vocabulary** — replace `Video.annotated(...)` with
  `style="annotated"` in the message text, preserving the rest verbatim. The whole premise of this layer is
  that the audience does not know the library underneath; pointing them at it in an error message is the one
  place that premise leaks. `Video`'s own callers keep the original message (the catch is in `image/_draw.py`).
- **The integrity-golden gate is near-vacuous by construction** (R5 L-3): `test_integrity.py` compares solver
  `times`/`Vm` bit-identically, and this plan changes **no solver code**. Keep it (it is the cheap proof that
  the claim "no solver code touched" is true) but do not read a pass as evidence the image layer is correct.
- **`PIL.Image` collides with the new `Image` class** — in `_draw.py` use `from PIL import Image as PILImage`.
- **PIL writes PDF and WebP fine; only SVG fails** (`ValueError: unknown file extension: .svg`) — restrict SVG
  only on the bare path, and state the real reason.
- **`at` is overloaded**: on `Image` it is a TIME in ms; on `Trace` it is a NODE / list / `{label: node}` dict.
  Both docstrings must say so.
- **Environment:** matplotlib 3.10.8, numpy 2.4.1, Pillow 12.1.0 with webp.

## Verification helper (used by EVERY phase)
⚑ **R3 M-1**: an earlier revision defined this inside the Phase-1 fenced block, so `run_suite img_p2_before`
in Phases 2/3/4 was `command not found`. It lives here now; **re-paste it in each phase's shell**.
The vacuity guard is on the **AFTER** file — R1 put it on the BEFORE run, which certifies the
pre-implementation suite, precisely the run whose success is uninteresting.
```bash
run_suite () {  # $1 = output stem
  /opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q -rfE 2>&1 \
    | tee "/tmp/$1_raw.txt" | grep -E "^(FAILED|ERROR)" | grep -v "conda.cli" | sort > "/tmp/$1.txt"
  grep -qE "[0-9]+ (passed|failed)" "/tmp/$1_raw.txt" || { echo "VACUOUS — the suite did not run"; return 1; }
}
```
Every phase: `run_suite img_pN_before` **before any edit** → implement → then **assert emptiness in the shell,
not by eye** (R4 L-8 — "must be EMPTY" is otherwise a gate that passes by inattention):
```bash
# R6 H-3: rm first and fail HARD on a vacuous run — otherwise `run_suite` returns 1, the `&&`
# short-circuits, `comm` never writes the file, and the `if` reads a MISSING/STALE one and prints
# "NO NEW FAILURES" for a suite that never ran (reproduced).
rm -f /tmp/img_pN_new.txt
run_suite img_pN_after || { echo "GATE FAILED — suite did not run"; exit 1; }
# R7 H-1: comm's failure goes to stderr and its exit status is discarded, so a MISSING baseline
# still creates an empty _new.txt and prints "NO NEW FAILURES". /tmp is reboot-volatile and the
# baseline is captured in a SEPARATE shell, so this is the likely failure, not a theoretical one.
[ -s /tmp/img_pN_before.txt ] || { echo "NO BASELINE at /tmp/img_pN_before.txt — capture it first"; exit 1; }
comm -13 /tmp/img_pN_before.txt /tmp/img_pN_after.txt > /tmp/img_pN_new.txt
if [ -s /tmp/img_pN_new.txt ]; then cat /tmp/img_pN_new.txt; echo "REGRESSION"; exit 1; else echo "NO NEW FAILURES"; fi
```
Phase 1 uses `/tmp/img_baseline.txt` as its `before`.

---

## Phase 1: `ImageInfo` + `Image` (map) + `draw()` single-panel + `r.image()`

**Goal**: the whole user-facing feature for ONE spatial-map still. **Tier**: large
**Scope**: 1 new package (4 modules) + 3 additive params on two `render.py` helpers + a `_produce_figure`
guard + result hook + ~35 tests.

### Phase Context
`video/` provides `Gradient` (frozen; `.resolve(masked_iter, field=) -> (Colormap, Normalize, lo, hi)`; 5
presets; `.key()`), `Video` (`.display_values(t)` is the masking seam; `.requires_figure()`; `.frames`
`(T,Nx,Ny)` float64 numpy; `.times`, `.dx`, `.dy`, `.value_label`, `.result`, `.active_mask`, `.field`), and in
`render.py`/`encoders.py` the eighteen helpers named above. `SimulationResult` carries `times, Vm, phi_e, dx,
dy, ionic_states, domain_mask, boundary_mode, Cm, chi, conductivity, ionic_model, cell_type` (`run.py:55-68`) — **all torch** (R6 L-2). **No solver interaction.**

**Phase 1 is MAPS ONLY**: `Trace` lands in Phase 2; a list raises `NotImplementedError("multi-panel lands in
Phase 3")`. **`labels=`/`rows=`/`cols=` must also raise in Phase 1** when passed with a single spec, with the wording
pinned as `"labels= applies to multi-panel rendering; pass a list of specs"` (R10 L-5, so Step 1.3's
`'multi-panel'` needle matches conforming code) —
they are multi-panel-only, and leaving them accepted-and-ignored is exactly the silent no-op M-14 forbids
everywhere else in this plan.

---

### Step 1.1: `image/info.py` — the delivery contract
**Model**: opus

```python
@dataclass
class ImageInfo:
    path: Optional[str]
    data: Optional[bytes]        # the sole copy when nothing was written
    format: str                  # "png" | "svg" | "pdf" | "jpg" | "jpeg" | "webp"
    width: Optional[int]         # None for vector formats — do NOT fabricate
    height: Optional[int]
    n_panels: int
    vmin: Optional[float]
    vmax: Optional[float]
    size_bytes: int
```
`saved` property (`path is not None`), `read()`, `save(path)` (makedirs, returns path), `__fspath__` (raises
`TypeError` naming `path=` / `.save('fig.png')` when unsaved), `_repr_html_`, `__repr__`.

- `_repr_html_` embeds a base64 data URI (**including SVG** — `data:image/svg+xml;base64`, in both the saved
  and unsaved cases; raw-SVG injection is sanitized by some notebook frontends). MIME: `png→image/png`,
  `jpg`/`jpeg`→`image/jpeg`, `webp→image/webp`, `svg→image/svg+xml`.
- **PDF has no inline representation** — return an honest one-line summary, never a broken `<img>`.
- Cap the inline payload at **16 MB** (the `VideoInfo` precedent); above it report the size instead.

**Verify**
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -c "
import os
from cardiac_core.image.info import ImageInfo
i = ImageInfo(path=None, data=b'\x89PNG\r\n\x1a\n', format='png', width=10, height=5,
              n_panels=1, vmin=-90.0, vmax=40.0, size_bytes=8)
assert i.saved is False and i.read() == b'\x89PNG\r\n\x1a\n'
assert 'data:image/png;base64' in i._repr_html_()
try:
    os.fspath(i); raise SystemExit('should have raised')
except TypeError as e:
    assert 'path=' in str(e), e
pdf = ImageInfo(path=None, data=b'%PDF-1.4', format='pdf', width=None, height=None,
                n_panels=1, vmin=None, vmax=None, size_bytes=8)
assert '<img' not in pdf._repr_html_(), 'PDF must not claim an inline image'
print('OK', repr(i))
"
/opt/miniforge3/bin/conda run -n heart-conduction python -c "
import sys, cardiac_core.image
assert 'imageio' not in sys.modules and 'cv2' not in sys.modules
print('import chain clean')
"
```

---

### Step 1.2: `image/panel.py` — the `Image` spec object
**Model**: opus

```python
@dataclass(eq=False)     # data/mask may be ndarrays -> a generated __eq__ would raise
class Image:
    data: Any                                   # SimulationResult | a bare (Nx,Ny) array  (R5 M-3)
    what: str = "snapshot"
    at: Optional[float] = None                  # ALWAYS a TIME in ms — never an index (R4 M-6).
                                                # NOTE: on Trace, `at` is a NODE.
    field: Optional[str] = None                 # "Vm" | "phi_e" only — an ARRAY goes in `data` (R3 M-7)
    what_kwargs: Optional[dict] = None          # forwarded to the analysis fn
    gradient: Optional[Gradient] = None         # None -> per-`what` default (resolved in item 2)
    label: Optional[str] = None                 # panel title
    front: Optional[float] = None               # mV isoline
    isochrones: Optional[bool] = None           # None -> True iff what=="activation" and not filled
    filled: bool = False                        # contourf instead of imshow + line contours
    contour_levels: int = 12                    # isoline count. NOT Gradient.levels (quantization).
    mask: Any = None                            # None=auto | array | False=explicitly none
    style: str = "annotated"                    # NOTE: annotated, unlike Video's "bare"
    aspect: str = "equal"
    units: str = "auto"
    value_label: Optional[str] = None           # colorbar label; None -> derived from `what`
```

**INPUT RULE (M-12, narrowed by R4 M-6).** `Image.data` is a **`SimulationResult`**, or a bare **`(Nx,Ny)`**
array with `what` left at its default (`dx`/`dy`/mask are unavailable, axes fall back to node indices, and the
colorbar reads `"value"` unless `value_label=` is given — **a promise kept only because item 6 gives an explicit
`value_label` precedence, R3 H-4**).
**⚑ A `(T,Nx,Ny)` array is NOT accepted** — raise, naming `Video(...).preview()`. R4 M-6: allowing it forced
`at` to mean a frame **INDEX** for array input while meaning a **TIME in ms** for a `SimulationResult`, so
`at=5` meant "5 ms" or "frame 5" with no error path and no way for the user to tell which they got. For an
audience that does not read docstrings that was the worst seam in the API. **`at` is now always a time**;
one meaning per name.
**A `.npz` path and a `(times, V)` pair are NOT accepted either** — load them and pass a 2-D array, or use
`Video(...).preview()`. **`field=` is legal ONLY with `what="snapshot"`** (R9 M-3 — `Image(r, what="apd", field="phi_e")` matched
two selector rows and the natural implementation would silently discard the `what`); any other `what` with an
explicit `field=` raises, naming both. **`field=` is a STRING only**
(`"Vm"`/`"phi_e"`, `SimulationResult` only); an array belongs in `data` (R3 M-7 — the earlier
`Union[str, np.ndarray]` annotation contradicted both this rule and the item-2 table, which has no array row).
A `SingleCellResult` raises, naming `.trace()` (L-27). A `SimulationResult` with **0 saved frames** re-raises
`Video`'s own message ("nothing to render: this result has 0 saved frames (t_end < save_every?)") rather than
an `IndexError` (L-29).

**RESOLVED-AT-CONSTRUCTION (R3 M-8).** An `Image` resolves its array, gradient, label and clip in
`__post_init__`, so **mutating a field afterwards has no effect** — `im.gradient = Gradient.zoom()` is silently
ignored. This is a deliberate divergence from `Video`, whose fields ARE live (verified: mutating `v.gradient`
after construction does change `render()`'s output). State it in the class docstring; the alternative
(rebuilding `_clip` lazily inside `draw()`) is rejected as more moving parts for a spec object that is cheap to
reconstruct.

`__post_init__`, **in this order** (`result = self.data if isinstance(self.data, SimulationResult) else None`
— bound once, used throughout items 2/5/5b; R9 L-14b):

1. Validate `style` ∈ `("bare","annotated")`, `aspect` ∈ `("equal","auto")`, `units` ∈ `("auto","cm","nodes")`.
1b. **Reject the wrong input types BEFORE touching `data`** (R4 L-6): a `SingleCellResult` → raise naming
   `.trace()`; a `(T,Nx,Ny)` array / `.npz` path / `(times, V)` pair → raise naming `Video(...).preview()`;
   a `SimulationResult` with 0 saved frames → re-raise `Video`'s own message. Placed **after** item 2 these
   become an `AttributeError` on `result.Vm` instead of the promised message.
1c. **Validate `what` BEFORE resolving it → `ValueError` listing the valid keys.** ⚑ **R9 M-7 / R10 L-1 —
   this runs HERE, before item 2, because item 2 resolves the selector: an unknown `what` would raise a bare
   `KeyError`/`AttributeError` there long before this pinned message could fire, and the Step-1.2 Verify greps
   this message.** Contents: **the four named intents `('snapshot','activation','apd','frequency')` PLUS** the
   introspected `Fields` properties minus `{"derivatives","integrals","mask"}`, annotating that
   `electric_field`/`current_flux` are bidomain-only. **`what="mask"` raises the PINNED message
   `"'mask' is the domain gate, not a renderable field — use Image(mask=…)"` HERE** (R11 H-1 — 1c subtracts
   `mask` from the valid set and runs *before* item 2, so item 2's identical message became dead code and the
   Verify's `'domain gate'` needle failed on conforming code; item 2's clause is now rationale, not a second
   raise site). **When `what` ∈ `('trace','restitution','apd_per_beat')`,
   name `Trace` / `.trace()` instead** — those are the other spec's namespace (R3 M-6: a message built only
   from `Fields` would list `curvature, divergence, …` and never mention `snapshot`, leaving a user who typed
   `what="trace"` with no pointer at all).
2. **Capture `explicit = self.gradient is not None` FIRST**, then resolve the selector to
   `(arr2d, value_label, t_ms, default_gradient)`, then **immediately assign
   `self.gradient = self.gradient if explicit else default_gradient`.**
   **`explicit` is a LOCAL, not stored state** (R3 M-2): its only consumer is the `rest`/`zoom` hard-raise
   message below, which reads *"you passed `Gradient.zoom()`, whose range is anchored to a resting potential…"*
   for an explicit gradient and never fires for a default one (no per-`what` default uses `rest`/`zoom`). An
   earlier revision stored it as `self._gradient_was_explicit` with **no consumer anywhere** — dead state that
   invites deletion, and deleting it invites deleting the ordering rationale with it.
   **⚑ R2 C-1 — this assignment is mandatory and must happen HERE, not in item 6.** `Video.gradient` has a
   *dataclass default*, not a `None` guard: passing `None` explicitly overwrites it, and every annotated draw
   then dies on `clip.gradient.interpolation` (`render.py:208`) / `gradient.resolve(...)`. The R1 text left
   `self.gradient is None` until after clip construction, which made the **zero-argument headline call raise
   `AttributeError`** — while Step 1.2's own Verify passed, because `Video.__post_init__` accepts `None` and
   `display_values()` never touches the gradient. `explicit`, **not** `is None`, is what answers
   "did the caller pass one?".

   | selector | array | `value_label` | default gradient |
   |---|---|---|---|
   | `field="Vm"` / `what="snapshot"` | `result.Vm[k]`, frame nearest `at` (default `T//2`) | `"Vm (mV)"` | `Gradient.physiological()` |
   | `field="phi_e"` | `result.phi_e[k]`; **raise `Video`'s clear message when it is None** (`clip.py:136-141`) | `"phi_e (mV)"` | `Gradient.physiological()` |
   | `what="activation"` | `analysis.activation_time(Vm, times, **what_kwargs)` | `"activation time (ms)"` | `Gradient(cmap="plasma", value_range="auto")` |
   | `what="apd"` | `analysis.apd_map(Vm, times, **what_kwargs)` | `"APD90 (ms)"` | `Gradient(cmap="viridis", value_range="auto")` |
   | `what="frequency"` | `analysis.dominant_frequency_map(Vm, times)` | `"dominant frequency (Hz)"` | `Gradient(cmap="turbo", value_range="auto")` |
   | any `fields.*` | see the RANK RULE below | the field name | `Gradient(cmap="RdBu_r", value_range="auto99")` |
   | **a bare `(Nx,Ny)` array as `data`** (R5 M-4) | the array itself | `"value"` | `Gradient(cmap="viridis", value_range="auto")` |

   **⚑ R5 M-4 — the array row is not optional.** Without it the INPUT RULE promises array input works while the
   table defines nothing for it, so an implementer falls through to `Gradient.physiological()` and renders an
   arbitrary non-voltage array on a **−90…40 mV** scale — the exact defect the per-selector defaults exist to
   prevent. Its `t_ms` is `nan`, and because `what` is `"snapshot"` the naive `show_time` rule would resolve
   **True** and caption the figure `"t = nan ms"`; hence the time-based `show_time` rule in Step 1.3.

   **RANK RULE (M-10 supersedes R1's name-list split).** **Unwrap `VectorField → .magnitude` FIRST, then branch
   on `arr.ndim`: 2 → use directly; 3 → select the frame nearest `at` (default `T//2`) and note the time in the
   panel. Never branch on the field NAME.** Measured ranks: `velocity`/`direction` are **static** VectorFields
   `(Nx,Ny,2)` → `.magnitude` `(Nx,Ny)`; `voltage_gradient`/`voltage_flux` are `(T,Nx,Ny,2)`; `source_sink` is
   `(T,Nx,Ny)`; `speed`/`curvature`/`divergence`/`vorticity`/`quality` are `(Nx,Ny)`. A name-based split
   misfiles `velocity`/`direction` and would slice a rank-2 array.
   `electric_field`/`current_flux` **raise on a monodomain result** — let the `ValueError` propagate unwrapped.
   **`mask` is NOT renderable** — it is the domain gate (`torch.bool`; `auto99` over an all-True mask gives
   `hi <= lo` → a degenerate-range warning every draw) and its name collides with `Image.mask=`. Reject it with
   `"'mask' is the domain gate, not a renderable field — use Image(mask=…)"`.
   **⚑ A derived map is NOT in mV.** `Gradient.physiological()` (−90…40) renders a ~300 ms APD map as uniform
   top-of-scale — hence the per-selector defaults.
   **Hard raise** when the resolved `gradient.value_range` ∈ `("rest","zoom")` and the selector is not a voltage
   map, naming `value_range="auto"`. This cannot be delegated to `Gradient`: `_infer_v_rest` only guards
   `field == "phi_e"`, and every `Image` clip reports `field='Vm'` (M-8).
   **`t_ms` for a STATIC map** (`apd`/`activation`/`frequency`/static `fields.*`) is `float("nan")` (R3 M-5) —
   there is no meaningful time — and because `show_time` keys on a **non-finite time** (Step 1.3, the single
   formula), `nan` is exactly what suppresses the stamp. Say `nan` explicitly rather
   than relying on `np.asarray([None], dtype=np.float64)`'s coercion. (**The `show_time=True`-on-a-static-map
   raise lives in Step 1.3, NOT here** — R5 M-2: `show_time` is a `draw()` parameter, not an `Image` field, so
   a rule placed in `__post_init__` can never fire. This is verbatim the R3 M-10 failure mode: *split by WHERE
   they live, or the promised message never appears*.)
   **`what_kwargs` on a selector with no analysis function** (`what="snapshot"`, `field="Vm"/"phi_e"`) must
   **raise** rather than be silently discarded (R4 M-5) — that one IS an `Image` field and belongs here.
   **`at` on a selector whose `t_ms` is `nan`** (`apd`/`activation`/`frequency`/static `fields.*`/bare array)
   must **also raise**, naming `what="snapshot"` or a time-varying field (R8 M-4). It was the last surviving
   silent no-op: this plan raises for `frame=` on an `Image`, `what_kwargs` on a no-analysis selector,
   `resolution=` on annotated, `tight`/`transparent` on bare, map knobs on a `Trace`, and
   `labels`/`rows`/`cols` in Phase 1 — each citing *"never a silent no-op"* — while `Image(r, what="apd",
   at=5.0)` quietly discarded `at`.
3. **Resolve `isochrones`**: `None` → `True` iff `what == "activation"` **and not `self.filled`** (H-5:
   filled contours ARE the isochrones; overlaying lines on them double-draws). An explicit `isochrones=True`
   with `filled=True` is honoured, since the caller asked for both.
3b. **⚑ Guard the figure-only fields against `style="bare"` (R3 H-1).** Raise when `style == "bare"` and any of
   (resolved `isochrones`, `self.filled`, `self.value_label is not None`, `self.contour_levels != 12`), naming
   `style="annotated"`. **`Image` defeats BOTH of `Video`'s own protections** and must replace them itself:
   `Video.requires_figure()` (`clip.py:194-195`) promotes an isochrone clip to the figure producer, and
   `enforce_capabilities` (`render.py:151-152`) rejects one on a bare clip — but the item-5 recipe hard-codes
   `isochrones=False` on the clip, so neither fires. Verified: `Image(r, what="activation", style="bare")`
   currently renders a plain colormapped array **with no error and no warning**, and `filled`/`contour_levels`
   have no gate at all.
4. **Store `self._lat`** (M-9): for `what == "activation"` it **is** `arr2d` — no recompute. Otherwise `None`.
5. **Build the clip — construction order is load-bearing:**
   ```python
   m = False if self.mask is False else (
       self.mask if self.mask is not None else
       (result.domain_mask if result is not None else False))
   self._clip = Video(
       (np.asarray([t_ms], dtype=np.float64), arr2d[None, ...]),   # the ONLY way to set `times`
       gradient=self.gradient,          # never None by now — item 2
       mask=m, style=self.style, aspect=self.aspect, units=self.units,
       label=self.label, front=self.front,
       isochrones=False,                # the overlay is drawn from `lat=`, see Step 1.3
   )
   # An EXPLICIT Image.value_label always wins over the derived one (R3 H-4), and the RESOLVED label is
   # written back so `spec.value_label` and `spec._clip.value_label` can never disagree (R4 M-3).
   self._clip.value_label = self.value_label if self.value_label is not None else value_label
   self.value_label = self._clip.value_label
   self._clip.dx, self._clip.dy = getattr(result, "dx", None), getattr(result, "dy", None)
   self._clip.result = result
   ```
   - **`mask` and `gradient` and `times` go through the CONSTRUCTOR**; `value_label`/`dx`/`dy`/`result` are
     assigned after, because all three are read at DRAW time. See the Reuse table for why each is where it is.
   - `isochrones=False` on the clip is deliberate: the one-frame guard in `isochrone_lat` (`render.py:159`)
     makes the clip's own overlay path unusable. **`clip.result` IS read** — RESOLVED SEMANTICS rule 4 uses it
     to compute the LAT for `Image(what="snapshot", isochrones=True)`. **R6 M-5** corrects an earlier note
     (written before rule 4 existed) which claimed it was unread on the `Image` path — do not "clean it up".
**`_clip`, `_lat` and `_gradient`-resolution state are plain `__post_init__` attributes, NOT dataclass
   fields** (R10 L-4) — declaring them as fields would break the `_IMAGE_KEYS` guard, which subtracts
   `_IMAGE_KEYS` from `Image.__dataclass_fields__`.
5b. **Re-mask the LAT cache (R5 H-5 — RESOLVED SEMANTICS rule 5 was stated in prose but absent from this
   ordered recipe, which is the normative implementation spec).** `analysis.activation_time` returns an
   **unmasked** map, so without this the overlay's contours run straight through a grey obstacle while the
   imshow beneath shows it masked:
   ```python
   if self.isochrones and self._lat is not None and self._clip.active_mask is not None:
       self._lat = np.where(self._clip.active_mask, self._lat, np.nan)
   ```
   It must come **after** item 5 (the clip owns `active_mask`) and before any use of `_lat`.
6. Expose `display_values()` (no index), `requires_figure()`, and the resolved `gradient`/`isochrones`/
   `value_label`/`_lat` as attributes.

**Verify**
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -c "
import warnings, numpy as np, cardiac_core as cc
from cardiac_core.image.panel import Image          # submodule path: _LAZY lands in Step 1.4 (R12 M-1)
from cardiac_core import Gradient
g = cc.Grid(30, 8, 0.025)
cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
sim = cc.monodomain(g, 'ttp06', cond, cc.Stim.boundary(g, 'left', amplitude=-80.0,
                                                       start_time=1.0, duration=2.0))
r = sim.run(t_end=12.0, save_every=1.0)
for w in ('snapshot','activation','apd','source_sink','speed','velocity'):
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # clip construction must be warning-free
        im = Image(r, what=w)
    a = im.display_values()
    assert a.shape == (30, 8), (w, a.shape)
    assert im._clip.dx == 0.025, w
    # R2 C-1: the gradient must be a real object on the CLIP, not None.
    assert im._clip.gradient is not None and im.gradient is not None, w
    print(f'{w:12s} label={im._clip.value_label!r} range={im.gradient.value_range!r}')
assert Image(r)._clip.gradient.value_range == 'physiological'
assert Image(r)._clip.value_label == 'Vm (mV)'
assert Image(r, what='apd')._clip.value_label == 'APD90 (ms)'
assert Image(r, what='activation').isochrones is True
assert Image(r, what='activation', filled=True).isochrones is False   # H-5
assert Image(r, what='activation')._lat is not None                   # M-9
with warnings.catch_warnings():
    warnings.simplefilter('ignore')             # frequency-resolution warning expected on 12 samples
    assert Image(r, what='frequency').display_values().shape == (30, 8)
for bad, needle in (('nope', 'source_sink'), ('mask', 'domain gate')):
    try:
        Image(r, what=bad); raise SystemExit(f'should have raised: {bad}')
    except ValueError as e:
        assert needle in str(e), (bad, e)
# R9 M-5 — the Image-level raise rules, each added because the silent version was a defect.
import numpy as _np
for kw, needle in ((dict(what='apd', at=5.0), 'snapshot'),                 # R8 M-4
                   (dict(what_kwargs={'x': 1}), 'what_kwargs'),           # R4 M-5 (no-analysis selector)
                   (dict(what='apd', field='phi_e'), 'snapshot'),         # R9 M-3
                   (dict(style='bare', filled=True), 'annotated')):       # R3 H-1 (item 3b's list)
    try:
        Image(r, **kw); raise SystemExit(f'should have raised: {kw}')
    except ValueError as e:
        assert needle in str(e), (kw, e)
try:                                                                       # R4 M-6
    Image(_np.zeros((3, 4, 5))); raise SystemExit('should have raised')
except (ValueError, TypeError) as e:
    assert 'preview' in str(e), e
try:
    Image(r, what='apd', gradient=Gradient.zoom()); raise SystemExit('should have raised')
except ValueError as e:
    assert 'auto' in str(e), e
try:    # n_beats=1,bcl=50 keeps this to ~50 ms of 0-D pacing; the default is a 1010-frame run (R3 L-7)
    Image(cc.single_cell('ttp06', n_beats=1, bcl=50.0)); raise SystemExit('should have raised')
except (ValueError, TypeError) as e:
    assert 'trace' in str(e), e
print('panel OK')
"
```

---

### Step 1.3: `image/_draw.py` — the verb, single panel
**Model**: opus

**Ordered sequence (load-bearing):**
```
dispatch on spec type -> resolve `frame_resolved` -> validate format vs path -> media-convention guard
  -> enforce_capabilities (RAW figsize/dpi, re-wrapped for Image) -> apply spec-type defaults
  -> resolve gradient over MASKED values at [frame_resolved] -> resolve destination
  -> produce + save -> read back (w,h) + getsize -> _finalize -> ImageInfo
```
**`frame_resolved` is the ONE name for "which frame" and it is bound HERE, first (R7 M-6** — an earlier
revision used a bare `i` in the producer calls and `frame_resolved` in the `show_time` formula, two names for
one concept, neither bound on the `Image` path**):** `0` for an `Image` (its clip always holds exactly one
frame), `frame` if given for a `Video`, else `len(clip.frames) // 2`. Every later `idx=[…]`,
`_produce_bare(clip, …)`, `_produce_figure(st, clip, …)` and `clip.times[…]` uses it.

```python
def draw(spec, slug="figure", *, path=None, question=None, bulk=None, date=None, root=None,
         format=None, frame=None, figsize=None, dpi=None, tight=None, title=None,
         colorbar=None, show_time=None, units=None, transparent=False,
         resolution=_UNSET, fit=_UNSET,
         labels=None, rows=None, cols=None) -> ImageInfo
# `tight` is Optional[bool] (R4 M-1), NOT `True`: it resolves to True on the annotated branch and RAISES on
# any non-None value with style="bare". Defaulting it to True would let `tight=True` — the ACTUAL silent
# no-op — sail through on every bare draw. (It does also raise on `tight=False`; R5 L-9 notes the prose
# overstated the benefit, but catching the real no-op is the point.)
#
# ⚑ R5 H-4 — `resolution`/`fit` MUST use a module-level `_UNSET` sentinel, not a literal default. There is
# one signature but TWO semantic defaults (Image: "auto"/"contain"; Video: None/"contain"), and R4 H-4
# requires "the default never raises, an explicit non-default does". With `resolution="auto"` written as the
# literal there is no way to tell "caller passed nothing" from "caller passed the literal", and BOTH readings
# are broken: treating "auto" as passed makes `draw(Video.annotated(v))` RAISE (direct draw() on a Video
# becomes unusable), while treating it as unpassed makes `draw(..., resolution="auto")` a pure silent no-op —
# exactly what H-4 forbids. Resolve AFTER dispatch: Image -> "auto"/"contain", Video -> None/"contain",
# Trace -> n/a.
#
# ⚑⚑ R8 H-3 / R9 M-2 — THE RAISE IS SCOPED, and this is the sentence an implementer transcribes, so state it
# as a comparison against the SENTINEL, never against "the default":
#   * annotated `Image` or `Trace` -> raise whenever resolution/fit is not `_UNSET`;
#   * ANNOTATED `Video` spec     -> raise only when the value differs from None/"contain", so the
#                                   delegation's explicit `resolution=None` passes;
#   * BARE `Video` spec          -> never raises (same as a bare Image; R10 L-2 scopes this row, which
#                                   previously read "Video spec" unqualified);
#   * bare `Image`               -> never raises; all three resolution states ("auto" / None / "1080p"|(w,h))
#                                   and ALL of `_LEGAL_FIT` are legal.
# Phrasing it as "neither `_UNSET` nor that spec type's resolved default" re-created R5 H-4's forbidden silent
# no-op: an annotated Image's resolved default reads as "auto", so `draw(Image(r), resolution="auto")` would
# neither raise nor do anything. That is the bare producer's default leaking onto a path that cannot use it.
# An unqualified "raise when != the default" deletes the documented `resolution="1080p"` and `fit="cover"`
# paths, and no Verify would catch it (the existing one tests `resolution='1080p'` on an ANNOTATED spec,
# which raises under either reading).
```

- **`spec` accepts an `Image`, a `Trace` (Phase 2), or an existing `Video`.** Dispatch on `isinstance`; a
  `Video` is used as `_clip` directly. **Its default frame is `len(clip.frames) // 2`** (R6 M-1), matching
  `preview_frame` (`render.py:421`), and **the gradient is resolved over `clip.masked_iter([frame_resolved])`**
  (R6 M-2) — `preview_frame` resolves over exactly that single frame (`render.py:425`), so resolving over `[0]`
  or over all frames (as `render()` does at `render.py:330`) would change the colour scale and silently break
  the *"`Video.preview()` is pixel-identical"* exit criterion. **`frame=`** (H-2 — R1's dispatch had no
  parameter to dispatch on)
  selects which frame of a multi-frame clip to draw: **`Video` specs only**. It **raises** on an `Image`
  (*"frame= selects a frame of a Video; an Image selects with at= (a TIME in ms)"*) and on a `Trace` — R4 M-2:
  "ignored for `Image`" would silently discard the caller's frame selection, the same silent-no-op class M-14
  forbids. This is the seam `preview_frame` delegates through.
  **The `Video` dispatch supplies no spec-level knobs** (M-19): use
  `lat=<the RESOLVED SEMANTICS rule-2 block, verbatim — gated on spec.isochrones, filled from rule 4, masked
  per rule 5>`,
  `contour_levels=getattr(spec, "contour_levels", 12)`, `filled=getattr(spec, "filled", False)` — **never
  `self.<attr>`; `draw()` is a module-level function**, and a `Video` has none of those fields.
- **Format** — explicit `format=` wins, else `path=`'s extension, else `"png"`. Unknown extension raises,
  naming the legal set `png|svg|pdf|jpg|jpeg|webp`. **If both are given and disagree, RAISE** with exactly:
  `"format='png' disagrees with path='fig.pdf' — pass one or the other."` (L-22 pins the wording the Verify
  greps for.) **When `path=` is given, hand `_resolve_destination` `path`'s OWN extension unchanged** — do not
  normalise `jpeg→jpg` (M-13), or the in-flight rewrite warns about a nonexistent "backend downgrade" and
  silently renames the file.
- **Media-convention guard** — if the resolved format is `pdf`/`webp` **and `path is None`** and
  `_named_destination(None, question, bulk, date, root)` (L-35), raise:
  `"format='pdf' cannot be written to a media/ path — media_path accepts png/jpg/jpeg/svg/gif. Pass path='fig.pdf' instead."`
- **Re-wrap `enforce_capabilities`' message for an `Image` spec (R7 H-4 — the decision existed in the gotchas
  but had no implementation site, so it would simply not have been built):**
  ```python
  try:
      enforce_capabilities(clip, colorbar=colorbar, show_time=show_time,
                           figsize=figsize, dpi=dpi, title=title)
  except ValueError as e:
      if isinstance(spec, Image):        # Video's own callers keep the original wording
          msg = (str(e).replace("Use Video.annotated(...) — the bare producer",
                                'Use style="annotated" — the bare producer')
                        .replace("a bare clip", 'style="bare"'))    # R8 L-13: "clip" is Video's noun
          raise ValueError(msg) from None
      raise
  ```
  The `isinstance(spec, Image)` guard is what keeps a **direct** `draw(Video.bare(v), title="x")` on the
  original wording (R7 L-5). This is the path `Image(r, style="bare", label="x")` and `front=` actually take — **neither is in item 3b's
  list**, so they reach `render.py:147-150` and would otherwise tell a user who has never typed `Video` to
  "Use `Video.annotated(...)`".
- **`enforce_capabilities` is called with the caller's RAW `figsize`/`dpi`, BEFORE defaults** — `render.py:144`
  raises whenever either is non-`None` on a bare clip, so defaulting first makes **every** bare render a
  `ValueError`. Defaults (`dpi=150`, `figsize=None`) apply on the annotated branch only.
- **⚑ SPEC-TYPE-DEPENDENT DEFAULTS (R3 C-1 + C-2).** R2's H-3 fix (*"`resolution=None` is mandatory"* on the
  `preview_frame` delegation) and its M-14 fix (*"a non-default `resolution=` on an annotated spec raises"*)
  **contradicted each other**: `preview_frame` is called on annotated clips today — `test_video.py:605`
  `Video.annotated((times, V)).preview(...)` is currently green — so the delegation would have raised and
  turned that test RED. And the delegation's `dpi=dpi` (i.e. `None`) would have picked up `draw()`'s new **150**
  default where `preview_frame` uses `dpi or 100` in two places (`render.py:431,434`), silently resizing a
  shipped API by ~1.5× (**measured 719×299 → ~1079×448**; capture the exact pair on the `wave` fixture
  immediately before Phase 3 rather than trusting this literal — R9 L-13). Resolve both with one rule:

  | knob | `Image` spec | `Video` spec (the delegation path) |
  |---|---|---|
  | `dpi` default | `150` | **`dpi or 100`** — `preview_frame`/`render`'s historical value |
  | `figsize` default | `None` (so `_build_figure`'s aspect-aware sizing survives) | passed through unchanged |
  | `resolution` default | `"auto"` (bare) | `None` (no resize) |
  | `resolution`/`fit` **not `_UNSET`** on an **annotated** spec (never "non-default" — R11 M-1) | **raises**, naming `figsize=`/`dpi=` | **the DEFAULT `None` never raises; an EXPLICITLY non-default value raises the same message** (R4 H-4) |

  **⚑ R4 C-1 — `enforce_capabilities` MUST see exactly what the caller passed.** These defaults are resolved
  **after** the capability gate, on the annotated branch only. `render.py:144-146` raises whenever
  `figsize is not None or dpi is not None` on a **bare** clip, and `preview_frame` today passes the user's raw
  `None` (`render.py:413-414`) and therefore passes the gate. If the delegation or the table substituted a
  concrete `dpi=100` before the gate, **every bare preview would raise** — reddening four currently-green tests
  (`test_video.py:595, :601, :963, :972` — the bare `Video((times, V))` calls are at `:597, :604, :966, :975`) and breaking
  `Video.bare(...).preview()` for every user. **No rule anywhere may substitute a non-`None` `figsize`/`dpi`
  before `enforce_capabilities`.** The delegation passes `dpi=dpi` **raw**; the `or 100` lives inside `draw()`.
  **R4 H-4** narrows the `Video` exemption to the *default value*: exempting the whole column would make
  `draw(Video.annotated(...), resolution="1080p")` a pure silent no-op, which is precisely what M-14's "never a
  silent no-op" principle forbids. The green test at `render.py`'s `test_video.py:605` passes `resolution`
  nowhere, so only the default needs exempting.

  Neither regression is catchable by the existing suite: `test_preview_bare_has_no_chrome` asserts only
  `annotated.mean() > bare.mean()`, which survives a resize *and* black padding. Phase 3 therefore pins both
  preview paths explicitly, including `Video.bare(...).preview()` returning a raw-grid PNG (measured `(30, 8)`
  on the `wave` fixture).
- **`resolution=`/`fit=` apply to the BARE producer only** (M-14). (`render()` differs — it applies the canvas
  to both producers. The divergence is deliberate and documented.) **On a BARE spec every state below is
  legal, and so is every member of `_LEGAL_FIT`** (R8 H-3 — the raise is annotated-only). Validate `fit`
  against `_LEGAL_FIT` (`render.py:41`) **AFTER the `_UNSET` → per-spec-type resolution, never before**
  (R8 M-7: `_LEGAL_FIT` does not contain `_UNSET`, so validating first raises on **every default call** —
  structurally the same trap as R4 C-1's "no rule may substitute a value before the gate"). Three states:
  | `resolution` | behaviour |
  |---|---|
  | `"auto"` *(default)* | integer nearest upscale, **no padding**: `k = max(1, ceil(512 / max(Nx, Ny)))`, canvas `(Nx*k, Ny*k)` through `fit_frame(..., fit="contain")` — exact multiple ⇒ zero padding. 30×8 → 18× → 540×144. |
  | `None` | **no resizing at all** — raw grid pixels. What the `preview_frame` delegation passes, to stay pixel-identical. |
  | `"1080p"` / `(w,h)` | fixed canvas via `resolve_canvas` + `fit_frame`, padded with `_PAD_BLACK` per `fit`. |
  M-15's rationale: a padded 1080p bare still is ~53% black (measured: a 30×8 grid contains into a 1920×512
  data strip inside 1080), and black sits at the bottom of viridis/inferno. A bare still is PURE DATA — every
  pixel should be data.
- **Destination**: reuse `_resolve_destination` / `_finalize` by import, `kind="images"`.
- **Producers**:
  - **bare** → `_produce_bare(clip, frame_resolved, cmap, norm)` → resize per the table → **`burn_timestamp` if
    `show_time_resolved`** → `PILImage.fromarray(rgb).save(out_path)`.
    **⚑ H-3 — the burn is mandatory and must come AFTER the resize** (`encoders.py:317`: *"Call AFTER the
    canvas fit, or it is drawn at grid scale"*). R1's bare pipeline omitted `burn_timestamp` entirely while
    MEDIUM-26 promised `preview_frame`'s stamp would survive — a silent regression that
    `test_preview_bare_has_no_chrome` provably does not catch (it asserts only
    `annotated.mean() > bare.mean()`, which stays true).
    **`colorbar=True|False` and a NON-DEFAULT `units=` raise on a bare `Image`** (R10 L-6, scoped per
    R11 L-5), with the wording pinned as
    `"units= draws axis labels and needs a figure. Use style=\"annotated\"."` and the `colorbar` twin.
    ⚑ **`units` is in `_IMAGE_KEYS`, so `r.image(style="bare", units="cm")` routes it to the Image
    CONSTRUCTOR and never reaches `draw()`'s parameter — the same user intent would raise via
    `draw(Image(r, style="bare"), units="cm")` and not via the headline API (R12 L-4). Put the
    non-default-`units` check in `Image.__post_init__` item 3b as well, so both routes behave alike.**
    ⚑ **Three qualifications R11 L-5 caught, all load-bearing:** (a) `colorbar=True` is *already* rejected
    by `enforce_capabilities` (`render.py:146`) — only `colorbar=False` is new, and it must compare
    against `None`, not against `False`; (b) **`aspect` is NOT included** — it has a non-`None` default
    (`"equal"`) and is always forwarded to the clip, so "raises on a bare spec" would be unimplementable
    without a non-default qualifier that cannot be expressed; (c) the rule applies to the **`Image`**
    spec only — `draw()`'s `units=` reaches a bare `Video` too, and `Video.bare(v).preview(units="cm")`
    is a **shipped call that works today** (Phase 3's delegation forwards `units=units`), so raising for a
    `Video` spec would redden it. Add BOTH new tuples (`colorbar=False`, non-default `units=`) to Step 1.3's raise loop — `aspect` is
    excluded by (b), so there are two, not three (R12 L-3).
    SVG is not writable by PIL → raise naming `style="annotated"`. (PDF/WebP **do** work through PIL —
    so do not claim "raster only".) **`tight=` and `transparent=` are matplotlib-figure concepts and have no
    effect on this path** — per M-14's own "never a silent no-op" principle, raise if either is passed
    non-default with `style="bare"` (R3 L-5).
  - **annotated** → `_build_figure(clip, cmap, norm, colorbar_on=…, title=…, figsize=…, dpi=…, units=…,
    idx=[frame_resolved], lat=…, contour_levels=…, filled=…)` → `_produce_figure(st, clip, frame_resolved, show_time=…, title=…)` →
    `st.fig.savefig(out_path, dpi=…, bbox_inches=("tight" if tight_resolved else None),
    transparent=transparent)` inside `try/finally: plt.close(st.fig)`, where
    **`tight_resolved = True if tight is None else tight`** on the annotated branch.
    **⚑ R6 H-1 — the resolved name is load-bearing.** `tight` is `Optional[bool] = None`, so a literal
    transcription of `("tight" if tight else None)` yields `bbox_inches=None` on **every default annotated
    draw** — the opposite of THE DEFAULT table and of `viz.py:76,96`. And the ±15 % Phase-3 gate **cannot
    catch it**: measured on the exact viz composition, legacy 627×380 → delegated-with-tight 692×330
    (+10.4 %/−13.2 %) → delegated-**without**-tight **720×360 (+14.8 %/−5.3 %)** — inside the gate on both
    axes. So the composition would silently diverge while the only pixel check passes, and every `r.image()`
    in the wild would lose the tight crop.
- **`render.py` edits — all default-preserving:**
  - `_build_figure(..., lat=None, contour_levels=12, filled=False)`:
    **⚑ The overlay GATE must change too (R3 C-3).** Today it is `if clip.isochrones:` (`render.py:223`,
    and `render.py:483` in `_setup_panel`) — and the item-5 recipe hard-codes `isochrones=False` on every
    `Image` clip, so an implementer who reads `lat=` as merely *"a cache that avoids `isochrone_lat`"* ships
    `what="activation"` **with zero contours and no warning**. The gate becomes:
    ```python
    if lat is not None or clip.isochrones:
        lat_arr = lat if lat is not None else isochrone_lat(clip, idx)
    ```
    When `lat is not None`, `isochrone_lat` is **not called**, bypassing its `len(frames) < 2` guard.
    `contour_levels` replaces the hard-coded `levels=12`.
    `filled=True` — draw **no `imshow`**, draw `cs = ax.contourf(Xc, Yc, arr, levels=…, cmap=…, norm=…)`, and
    **`ax.set_aspect(clip.aspect)`** since `aspect` otherwise only reaches matplotlib via `imshow` (L-28).
    When the array is **all-NaN**, draw no contour set and no colorbar, but still produce a titled, labelled
    figure (mirroring `viz.py:89`).
    **⚑ Store the `QuadContourSet` on `_FigState.im` (R3 H-2)** — it IS the colorbar mappable, just not an
    `AxesImage`. Do **not** leave `im=None`: `fig.colorbar(None, ax=…)` **does not raise**; matplotlib 3.10
    fabricates a `_ScalarMappable` with `Normalize(0, 1)` and draws a plausible-looking **numerically
    meaningless** colorbar (measured: norm 0.0–1.0 against a true LAT range of 1.344–11.526). That is a pure
    silent-wrong-result with nothing to except on, and it lands directly on the multi-panel shared colorbar and
    on the `activation_isochrones` delegation shape.
  - **`_render_panels`' per-frame loop (`render.py:613`) gets the SAME `hasattr` guard** (R9 L-8 — this edit
    is owned here, not by the Architecture list). Defensive only: `_render_panels` never passes `filled=`.
  - **`_produce_figure` guards its data swap with `if hasattr(st.im, "set_data"):`** — ⚑ **H-4**:
    `render.py:243` calls `st.im.set_data(...)` unconditionally, so a contour set (which has no `set_data`)
    raises `AttributeError`. `hasattr`, **not `is not None`**, because H-2 now puts a `QuadContourSet` there.
    Video always sets a real `AxesImage`, so the guard changes nothing there. Verified: `front=` still draws
    correctly and the suptitle path is unaffected when `st.im` is not an image artist.
  - **`_setup_panel` gains the SAME three parameters** with identical semantics — **including the filled-mode
    `ax.set_aspect(clip.aspect)`** (R7 L-3; "identical semantics" alone left it implicit) — ⚑ **H-6**: the multi-panel
    path draws through `_setup_panel` (`render.py:461-489`), a **separate** function with its own `imshow`, its
    own hard-coded `levels=12` and its own `isochrone_lat` call. Every `Image._clip` has `isochrones=False`, so
    without this a `draw([Image(a, what="activation"), ...])` yields contour-free maps **with no warning** —
    the one-frame guard is never even reached. Stopping the `lat=` seam at `_build_figure` is a silent wrong
    result in Phase 3.
  - A test must assert all three params **default to today's exact video behaviour**.
- **The LAT for the overlay — see RESOLVED SEMANTICS rules 2/4/5; that block is the implementation.**
  ⚑ **R5 H-2**: an earlier phrasing here read *"use `spec._lat` when set … **Otherwise** call
  `activation_time(...)`"* — **ungated on `isochrones`**, so `Image(r, what="apd")` (whose `_lat` is `None`)
  computed a LAT, the gate fired on `lat is not None`, and an APD map shipped with activation contours drawn
  on it, silently. **Compute a LAT ONLY when the resolved `isochrones` is True.**
  `_lat_from_result(spec)` is
  `np.asarray(_to_numpy(analysis.activation_time(result.Vm, result.times)), dtype=np.float64)` **with NO
  kwargs** (R3 H-3). ⚑ **The `_to_numpy` is mandatory (R8 M-5)** — `analysis.*` returns **torch**, rule 2 then
  feeds the result to `np.where(...)`, which silently works on a CPU tensor and raises
  `TypeError: can't convert cuda:0 device type tensor to numpy` on a GPU result. The shipped function this
  replaces does it correctly (`render.py:168-169`); `_to_numpy` is already in the 18-name import list.
  `what_kwargs` belong to the *selector's own* analysis function, so `what_kwargs={"repol": 0.8}` would be a
  **`TypeError`** (measured) and `{"threshold": -20.0}` would **silently** draw a −20 mV LAT over a −40 mV map.
  When `spec._clip.result is None`, warn *"isochrones need a SimulationResult to compute activation times;
  skipping the overlay"* and leave `lat = None`.
- **`colorbar` resolution** (R3 M-11): `colorbar_resolved = colorbar if colorbar is not None else
  (clip.style == "annotated")` — the rule `render()` (`render.py:316`) and `preview_frame` (`render.py:430`)
  both use. **The `filled`-mode colorbar must sit behind the same flag**, or `colorbar=False` is ignored there.
- **`show_time` — ONE formula, stated once (R6 H-2).** Let `t0 = clip.times[frame_resolved]` (R6 L-6: a
  `Video` spec has no `_clip`; the `Video` **is** the clip):
  ```python
  show_time_resolved = show_time if show_time is not None else bool(np.isfinite(t0))
  if show_time is True and not np.isfinite(t0):
      raise ValueError("show_time=True needs a real time; this map has none (use what='snapshot')")
  ```
  ⚑ **The earlier text contradicted itself and the RANK RULE.** It said *"resolves True only for
  `what="snapshot"` … thread the panel's `what` through"* and, in the next sentence, *"make the rule depend on
  the TIME, not the selector"*. For `Image(r, what="source_sink")` — a time-varying `fields.*` where the RANK
  RULE selects a real frame and says to *note the time in the panel* — sentence 1 gives **no stamp** and
  sentence 2 gives a **stamp**, with no tiebreak. The time-based rule is the correct one: it keeps an APD map
  (`t_ms = nan`) and a bare-array panel unstamped, stamps a real frame of a time-varying field, and needs no
  `what` threading at all. **Delete the "only for `what='snapshot'`" phrasing wherever it survives.**
  A `Video` spec always has finite times ⇒ resolves True, matching `preview_frame`'s unconditional stamp;
  the delegation passes `show_time=True` explicitly anyway.
- **Width/height**: read back from the written file with PIL for raster formats, `None` for vector. **Both the
  PIL read-back and `os.path.getsize` must happen BEFORE `_finalize`**, which deletes the temp file
  (`render()` itself does `getsize` first).
- On ANY exception after the path was resolved, delete a partially written file before re-raising.

**Verify**
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -c "
import os, tempfile, cardiac_core as cc
from cardiac_core.image.panel import Image          # submodule paths: _LAZY lands in Step 1.4 (R12 M-1)
from cardiac_core.image._draw import draw
g = cc.Grid(30, 8, 0.025); cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
sim = cc.monodomain(g, 'ttp06', cond, cc.Stim.boundary(g, 'left', amplitude=-80.0,
                                                       start_time=1.0, duration=2.0))
r = sim.run(t_end=12.0, save_every=1.0)
i = draw(Image(r))                                    # the ZERO-ARGUMENT headline call (R2 C-1)
assert i.path is None and i.saved is False and i.data[:8] == b'\x89PNG\r\n\x1a\n'
assert i.width and i.height, (i.width, i.height)
d = tempfile.mkdtemp()
assert draw(Image(r), path=os.path.join(d, 'wave.png')).saved
for fmt in ('svg', 'pdf'):
    v = draw(Image(r), path=os.path.join(d, f'wave.{fmt}'))
    assert v.width is None and os.path.getsize(v.path) > 0, fmt
# M-13: a .jpeg path stays .jpeg, with no warning.
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    j = draw(Image(r), path=os.path.join(d, 'a.jpeg'))
assert j.path.endswith('.jpeg') and not any('backend downgrade' in str(x.message) for x in w), (j.path, w)
# ⚑ R9 M-5 — EVERY raise rule this plan added was added because the SILENT version was a defect. Each one
# therefore needs an executable check, or an implementer who skips it ships the original defect with all
# gates green. This loop pins the `draw()`-level ones; Step 1.2's Verify pins the `Image`-level ones.
for kw, needle in ((dict(slug='x', bulk=True, format='pdf'), 'path='),
                   (dict(path=os.path.join(d,'a.pdf'), format='png'), 'disagrees'),
                   (dict(frame=1), 'at='),                      # R4 M-2
                   (dict(labels=['a']), 'multi-panel'),         # R6 M-7
                   (dict(rows=2), 'multi-panel'),               # R6 M-7
                   (dict(resolution='auto'), 'figsize')):       # R9 M-2 — annotated + non-_UNSET
    try:
        draw(Image(r), **kw); raise SystemExit(f'should have raised: {kw}')
    except ValueError as e:
        assert needle in str(e), (kw, e)
for kw, needle in ((dict(tight=True), 'bare'), (dict(transparent=True), 'bare')):   # R3 L-5
    try:
        draw(Image(r, style='bare'), **kw); raise SystemExit(f'should have raised: {kw}')
    except ValueError as e:
        assert needle in str(e), (kw, e)
# R10 H-1: `label`/`front` are NOT in item 3b's constructor list — they reach enforce_capabilities
# inside draw(), which is also where the R7 H-4 re-wrap runs. Assert them HERE, not on Image(...).
for ikw in (dict(style='bare', label='x'), dict(style='bare', front=-40.0)):
    try:
        draw(Image(r, **ikw)); raise SystemExit(f'should have raised: {ikw}')
    except ValueError as e:
        assert 'style="annotated"' in str(e) and 'Video' not in str(e), (ikw, e)
# M-15 / H-9: a bare still is upscaled with NO padding and keeps its aspect.
b = draw(Image(r, style='bare'))
assert max(b.width, b.height) >= 512 and abs(b.width / b.height - 30 / 8) < 0.05, (b.width, b.height)
# M-14: resolution= on an annotated spec must RAISE, not silently no-op.
try:
    draw(Image(r), resolution='1080p'); raise SystemExit('should have raised')
except ValueError as e:
    assert 'figsize' in str(e), e
print('OK', i)
"
# H-4 / C-3: filled mode must not crash, and activation must POSITIVELY draw contours.
# R3 C-3: the earlier version of this block asserted only the ABSENCE of a warning plus len(data) > 1000 —
# both hold whether or not a single contour is drawn, so it certified nothing. Count the artists instead.
/opt/miniforge3/bin/conda run -n heart-conduction python -c "
import warnings, cardiac_core as cc
from cardiac_core.image.panel import Image
from cardiac_core.image._draw import draw
from cardiac_core import Gradient
from cardiac_core.video.render import _build_figure
import matplotlib.pyplot as plt
g = cc.Grid(30, 8, 0.025); cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
sim = cc.monodomain(g, 'ttp06', cond, cc.Stim.boundary(g, 'left', amplitude=-80.0,
                                                       start_time=1.0, duration=2.0))
r = sim.run(t_end=20.0, save_every=1.0)          # LAT is 100% finite at 20 ms; APD is NOT (use the long fixture)
spec = Image(r, what='activation')
cmap, norm, lo, hi = spec.gradient.resolve(spec._clip.masked_iter([0]), field=spec._clip.field)
st = _build_figure(spec._clip, cmap, norm, colorbar_on=True, title=None, figsize=None, dpi=100,
                   units=None, idx=[0], lat=spec._lat, contour_levels=12, filled=False)
n_contours = len(st.ax.collections)
plt.close(st.fig)
assert n_contours > 0, f'activation drew NO contours (gate never fired): {n_contours}'
assert len(st.ax.images) == 1 and n_contours == 1, (len(st.ax.images), n_contours)   # imshow + ONE line set
# R4 M-7: `a.data != f.data` certified nothing — it holds under a double-draw, a missing set_aspect and a
# wrong colorbar alike (R3 C-3's own criticism, one block lower). Count artists and check the colorbar norm.
fst = _build_figure(spec._clip, cmap, norm, colorbar_on=True, title=None, figsize=None, dpi=100,
                    units=None, idx=[0], lat=None, contour_levels=15, filled=True)
n_img, n_col = len(fst.ax.images), len(fst.ax.collections)
cb_norm = fst.fig.axes[-1]._colorbar.norm if hasattr(fst.fig.axes[-1], '_colorbar') else None
plt.close(fst.fig)
assert n_img == 0 and n_col == 1, f'filled must be contourf-only, got images={n_img} collections={n_col}'
assert cb_norm is not None and abs(cb_norm.vmin - lo) < 1e-9, (cb_norm, lo)   # NOT a fabricated 0-1 bar
# ⚑ R5 H-1 — this MUST go through the public verb. An earlier version computed `dlat` in the test script
# using the correct rule and asserted on it, exercising NO production code.
# ⚑⚑ R6 C-1 — and the FIRST attempt at fixing that was itself inert: patching
# `sys.modules['cardiac_core.video.render']._build_figure` CANNOT reach `_draw.py`, because the Architecture
# section mandates a module-scope `from ..video.render import _build_figure`, which binds the original
# function object into `cardiac_core.image._draw`'s namespace at import time. Verified against a faithful
# mock of the prescribed structure: `spy fired? False`, then `KeyError: 'lat_is_none'`.
# Patch the CONSUMER'S binding.
import os, tempfile, sys          # R6 H-4: this is a separate `python -c`; block 1's names do not exist here
d = tempfile.mkdtemp()
draw(Image(r), path=os.path.join(d, 'warmup.png'))     # force cardiac_core.image.draw to be imported
_D = sys.modules['cardiac_core.image._draw']
seen = {}
_orig = _D._build_figure
def _spy(clip, cmap, norm, **kw):
    seen['lat_is_none'] = kw.get('lat') is None
    seen['filled'] = kw.get('filled')
    st = _orig(clip, cmap, norm, **kw)
    seen['images'], seen['collections'] = len(st.ax.images), len(st.ax.collections)
    return st
_D._build_figure = _spy
try:
    draw(Image(r, what='activation', filled=True, isochrones=False, contour_levels=15),
         path=os.path.join(d, 'iso.png'))
finally:
    _D._build_figure = _orig
assert seen, 'the spy never fired — patch the CONSUMER binding, not the render module (R6 C-1)'
assert seen['lat_is_none'], 'filled+isochrones=False must pass lat=None (R4 C-3 / R5 C-1 double-draw)'
assert seen['images'] == 0 and seen['collections'] == 1, seen   # contourf ONLY; 2 == the double-draw
print(f'line: images=1 collections={n_contours} | delegation shape: {seen} | cb vmin={cb_norm.vmin:.3f}')
"
```

---

### Step 1.4: wiring — `r.image()`, exports, tests
**Model**: sonnet

1. `run.py` → `SimulationResult.image(self, slug="figure", **kw)`, splitting panel keys from draw keys as
   `.video()` does (`run.py:155-157`):
   **Define it as a MODULE-LEVEL constant `run._IMAGE_KEYS`** (R8 H-2 — **not** a method-local `image_keys`:
   a set inside the method cannot be imported, so the guard test below would be unwritable and the agent would
   fall back to the can't-fail form this cycle has killed four times):
   `_IMAGE_KEYS = {"what","at","field","what_kwargs","gradient","label","front","isochrones","filled","contour_levels","mask","style","aspect","units","value_label"}`
   (15 names = `Image`'s 16 fields minus `data`).
   **Guard test**: `from cardiac_core.run import _IMAGE_KEYS; assert not ({f for f in Image.__dataclass_fields__
   if f != "data"} - set(_IMAGE_KEYS))` — so a new field cannot silently become a `TypeError` from the
   headline API. Step 2.3 does the same for `_TRACE_KEYS`.
2. `__init__.py` → `_LAZY` += `'Image': 'image', 'draw': 'image', 'ImageInfo': 'image'` (NOT `Trace`).
   **Re-read the file first** — it has an uncommitted `single_cell` → `_single_cell` hunk.
3. `tests/test_self_contained.py` → add `"image"`.
4. `tests/test_image.py` — **THREE module-scoped fixtures** (L-26 + R3 H-5 + R4 H-2), specified in the APD
   gotcha above: `wave` (2.3 s) for everything, `long_wave` (44.6 s) for **APD maps and `apd_per_beat` only**,
   and a **synthetic `multibeat` result** for restitution (`long_wave` has one stimulus and yields
   `DI.shape == (0,)` — measured). Cover: the zero-argument default writes nothing,
   displays, **and does not raise** (C-1); each media keyword saves; `path=` obeyed literally; `.jpeg` preserved
   without warning (M-13); every `what` renders at the right rank incl. `velocity`/`direction` (M-10); a derived
   map does not use the −90…40 range (**on `long_wave` — on `wave` the APD map is all-NaN and the range IS
   (−90, 40) by fallback, so this assertion is FALSE there**, R3 H-5); **the DRAWN colorbar text** equals the
   derived `value_label` (not the attribute); `what="activation"` draws **`>0` contour artists**, counted, not
   inferred from a missing warning (R3 C-3); `filled=True` renders and resolves `isochrones` False (H-5);
   the pdf-on-media, format/path-disagreement, `resolution=`-on-annotated, `mask`-as-field and
   `rest`/`zoom`-on-derived guards all raise; **`Image(r, style="bare", label="x")` raises with
   `style="annotated"` in the message and NOT `Video.annotated`** (R7 H-4); **masking on a synthetic finite
   obstacle** (the pattern in
   `test_video.py:779::test_reproduces_semicircle_composition`, which sets `V2[:, obstacle] = 12.3` — R3 M-3
   corrects an earlier citation of `test_two_panel_masked_obstacle`, which **does not exist**) is NaN in
   `display_values()` while finite in the source; a bare still is ≥512 px with preserved aspect and **carries a
   burned stamp** (pixel-diff vs `show_time=False` in the top-left 20×120 block — H-3); an explicit
   `value_label=` wins over the derived one, asserted on the **drawn** colorbar (R3 H-4); every figure-only
   field raises under `style="bare"` (R3 H-1); `__fspath__` raises when unsaved; `.save()` bytes == `.read()`
   **and `.read()` falls back to the FILE when `data is None`** (R3 L-4 — `_finalize` returns `data=None` for a
   saved render, so an unsaved-only Verify never exercises this); **`_build_figure`/`_setup_panel`/
   `_produce_figure` are unchanged for video** when the new params are omitted.

### Phase 1 Verification
**Uses the shared `run_suite` helper — see "Verification helper" above; re-paste it in each phase's shell.**
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# Re-paste the `run_suite` helper (Verification helper section) FIRST — it is a shell function and does
# not survive across Bash invocations; without it this exits 127 and captures NOTHING (R8 L-14).
# BASELINE already captured 2026-07-25 at /tmp/img_baseline.txt (WITH the dirty tree — the correct "before"):
# 493 passed / 7 failed / 2 xfailed, all 7 CUDA OOM. /tmp is reboot-volatile: if the file is missing,
# recapture with `run_suite img_baseline` BEFORE making any edit.
# ... implement Phase 1 ...
# R6 H-3: rm first and fail HARD on a vacuous run — otherwise `run_suite` returns 1, the `&&`
# short-circuits, `comm` never writes the file, and the `if` reads a MISSING/STALE one and prints
# "NO NEW FAILURES" for a suite that never ran (reproduced).
rm -f /tmp/img_p1_new.txt
run_suite img_p1_after || { echo "GATE FAILED — suite did not run"; exit 1; }
# R7 H-1: comm's failure goes to stderr and its exit status is discarded, so a MISSING baseline
# still creates an empty _new.txt and prints "NO NEW FAILURES". /tmp is reboot-volatile and the
# baseline is captured in a SEPARATE shell, so this is the likely failure, not a theoretical one.
[ -s /tmp/img_baseline.txt ] || { echo "NO BASELINE at /tmp/img_baseline.txt — capture it first"; exit 1; }
comm -13 /tmp/img_baseline.txt /tmp/img_p1_after.txt > /tmp/img_p1_new.txt
if [ -s /tmp/img_p1_new.txt ]; then cat /tmp/img_p1_new.txt; echo "REGRESSION"; exit 1; else echo "NO NEW FAILURES"; fi
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_integrity.py -q
```

### Phase 1 Exit Criteria
- [ ] `r.image()` **does not raise**, displays, and writes nothing; `r.image("x", bulk=True)` writes under
      `media/lab/_sim_outputs/images/{date}/`.
- [ ] Every map `what` renders at the correct rank; derived maps are labelled with their own units **on the
      drawn colorbar** and do not use the mV range.
- [ ] `what="activation"` draws contours; `filled=True` renders without an `AttributeError`.
- [ ] png/svg/pdf/jpg/jpeg/webp write; `.jpeg` is not rewritten; all five guards raise.
- [ ] Masking correct on a finite obstacle; a bare still is ≥512 px, aspect-preserved, stamped.
- [ ] `comm -13` empty; integrity goldens bit-identical.
- [ ] Commit: `feat(cardiac_core): image — spec-first still figures (Image + draw + ImageInfo)`.

---

## Phase 2: `Trace` — the series panel

**Goal**: the figure kind the corpus is made of and cardiac_core has never had. **Tier**: medium
**Scope**: 1 spec class + registry + 2 hooks + ~20 tests.

### Step 2.1: the `Trace` spec + registry
**Model**: opus

```python
@dataclass(eq=False)
class Trace:
    data: Any                                  # SimulationResult | SingleCellResult | (x, y) | {label: (x, y)}
                                               # (x, y): ONE unlabelled series. dict: one series per key,
                                               # each value an (x, y) pair. Both bypass `what` ENTIRELY
                                               # (R9 M-4): `at=` raises, and xlabel/ylabel default to None
                                               # rather than inheriting `what`'s "time (ms)"/"Vm (mV)"
                                               # (R10 L-7) — pass them explicitly.
    what: str = "trace"                        # "trace" | "restitution" | "apd_per_beat"
    at: Any = None                             # a NODE (ix,iy), a LIST, or a {label: node} dict.
                                               # NOTE: on Image, `at` is a TIME in ms.
    series: Optional[Sequence] = None          # explicit [(label, x, y), ...] override
    label: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    hline: Any = None                          # float | list | [(y, label), ...]
    vline: Any = None
    legend: Optional[bool] = None              # None -> on when >1 series
    marker: Optional[str] = None               # None -> "o" for restitution/apd_per_beat
    linestyle: Optional[str] = None            # None -> "-"; "none" gives a MARKER-ONLY scatter (M-18)
    xlim: Optional[tuple] = None
    ylim: Optional[tuple] = None
    logx: bool = False
    logy: bool = False
    colors: Optional[Sequence[str]] = None
    what_kwargs: Optional[dict] = None         # R3 M-9: restitution's repol=/threshold= are otherwise
                                               # unreachable (restitution_curve(..., repol=0.9, threshold=-20.0))
```

| `what` | x | y | xlabel / ylabel | marker / linestyle default |
|---|---|---|---|---|
| `"trace"` | `times` | `Vm[:, ix, iy]` per node in `at` | `"time (ms)"` / `"Vm (mV)"` | none / `"-"` |
| `"restitution"` | `DI` | `APD` | `"DI (ms)"` / `"APD90 (ms)"` | `"o"` / `"none"` |
| `"apd_per_beat"` | beat index (1-based) | `APD` | `"beat"` / `"APD90 (ms)"` | `"o"` / `"none"` |

- **`at` accepts one node, a list, or a `{label: node}` dict** — the dict form is the corpus's dominant idiom
  (`diag_column_boundary_vs_center.py` labels every series). Default: the grid centre.
- A `SingleCellResult` has **no grid** — reject `at` clearly; label the single series with the model name; its
  `V` is 1-D `(T,)`.
- **Bounds-check node indices** and raise with this wording pinned (R8 M-8 — Step 2.4 greps for `range`, and
  "raise naming the valid range" alone lets a conforming implementation fail the Verify):
  `"node (999, 0) is out of range for a 30x8 grid (ix must be 0..29, iy 0..7)"`. Never let an out-of-range
  index become a torch negative-index wrap.
- torch→numpy at ingest; float64.
- `hline`/`vline` accept a scalar, a list, or `(value, label)` pairs; a labelled reference line joins the
  legend. This covers the corpus's 106 reference-line calls.
- An empty/all-NaN series (restitution on a single-beat run) **warns and still draws an empty axes** — never
  raises. **The test must wrap it in `pytest.warns`** (L-39).

### Step 2.2: `draw()` renders a `Trace`
**Model**: opus

**⚑ Split it in two (R7 H-2) — Phase 3 needs an axes-level seam, and the plan had none:**
```python
def _draw_trace_on(spec, ax) -> None:      # NO figure ownership — Phase 3 calls this per Trace panel
    ...one ax.plot(x, y, label=…, marker=…, linestyle=…, color=…) per series
       → axhline/axvline → xlabel/ylabel/title → ax.legend() when resolved on
       → set_xlim/ylim, set_xscale/yscale
```
and a thin single-panel wrapper that owns `fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)`, calls
`_draw_trace_on`, then runs the same save/finalize path as Phase 1 **inside `try/finally: plt.close(fig)`**
(R5 L-5). Without this split, Phase 3's *"a mixed `[Image, Trace]` list lays out"* exit criterion has no
implementable path: the only `Trace` renderer would open its **own** figure, leaking it past the layout's
`try/finally` and leaving the layout's Trace axes **blank** — a silent wrong result.
**Defaults, stated so a cold-start agent need not guess (R6 M-6):** `figsize=(6.4, 3.6)`, `dpi=150`,
`tight_resolved = True if tight is None else tight` — the same rule as `Image`'s annotated branch, so a
caller-passed `tight=False` is honoured on a `Trace` too (R7 L-4); `colors=None` → matplotlib's property cycle.
On a trace-only figure `ImageInfo.vmin`/`vmax` are **`None`** (no colour range was resolved) and `n_panels=1`.
**Map-only knobs (R3 M-10 — split by WHERE they live, or the promised message never appears):**
- `draw()` **parameters** — `colorbar=`, `frame=`, `resolution=`, `fit=`, `show_time=`, `units=` — are checked
  inside `draw()` and raise a named `ValueError` on a `Trace`.
- `front=`, `isochrones=`, `filled=`, `gradient=`, `contour_levels=`, `value_label=`, `mask=`, `aspect=` are
  **`Image` constructor fields, not `draw()` parameters**. `r.trace(gradient=…)` would fall through
  `trace_keys` into `draw()` and surface as a raw `TypeError: draw() got an unexpected keyword argument` —
  **not** the promised message. So `SimulationResult.trace`'s key split must catch these names explicitly and
  raise *"`gradient=` is a map knob — use `r.image(...)`"*.

### Step 2.3: hooks
**Model**: sonnet

- `SimulationResult.trace(self, slug="trace", **kw)` with **its own key set** (M-17 — `image_keys` does not
  cover `Trace`'s ten extra fields, so `r.trace(hline=-40)` would be a `TypeError`):
  **Define it as a MODULE-LEVEL constant `run._TRACE_KEYS`** (not a literal inside the method) so the
  introspection guard can import and assert against the real set — R7 H-3:
  `_TRACE_KEYS = {"what","at","series","label","xlabel","ylabel","hline","vline","legend","marker","linestyle","xlim","ylim","logx","logy","colors","what_kwargs"}`.
  Do the same for `_IMAGE_KEYS` in Step 1.4. The guard asserts every `Trace` field except `data` is in it.
- `SingleCellResult.trace(...)` — **the single most requested wet-lab figure, with no route today.**
- `__init__.py` `_LAZY` += `'Trace': 'image'`. Docs deferred to Phase 4.

### Step 2.4: Verify (R6 M-4 — Phase 2 had no executable check at all)
**Model**: sonnet
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
/opt/miniforge3/bin/conda run -n heart-conduction python -c "
import warnings, cardiac_core as cc
from cardiac_core import Trace, draw
g = cc.Grid(30, 8, 0.025); cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
sim = cc.monodomain(g, 'ttp06', cond, cc.Stim.boundary(g, 'left', amplitude=-80.0,
                                                       start_time=1.0, duration=2.0))
r = sim.run(t_end=20.0, save_every=1.0)
i = r.trace(at={'edge': (0, 4), 'centre': (20, 4)}, hline=(-40.0, 'threshold'))
assert i.path is None and i.data[:8] == b'\x89PNG\r\n\x1a\n' and i.vmin is None, i   # M-6: no colour range
sc = cc.single_cell('ttp06', n_beats=1, bcl=200.0).trace()
assert sc.data and not sc.saved
for kw, needle in ((dict(gradient=cc.Gradient.zoom()), 'r.image'), (dict(frame=1), 'frame'),
                   (dict(at=(999, 0)), 'range')):
    try:
        r.trace(**kw); raise SystemExit(f'should have raised: {kw}')
    except (ValueError, TypeError) as e:
        assert needle in str(e), (kw, e)
# R7 H-3: the previous form subtracted a set of CHARACTERS from a set of field NAMES, removed nothing,
# and never asserted — dead code that printed 'trace OK' regardless (the same can't-fail class R3 C-3,
# R4 M-7 and R5 H-1 each killed). Assert against the REAL key set, which Step 2.3 must export.
from cardiac_core.image.panel import Trace as T
from cardiac_core.run import _TRACE_KEYS
missing = {f for f in T.__dataclass_fields__ if f != 'data'} - set(_TRACE_KEYS)
assert not missing, f'Trace fields unreachable from r.trace(): {sorted(missing)}'
# ⚑ R9 L-11 — the `multibeat` fixture exists BECAUSE restitution had none (R4 H-2); check the thing it was
# built for, or its only gate is an unchecked checkbox.
import torch
from cardiac_core.run import SimulationResult
from cardiac_core import analysis
V_REST, V_PEAK, BCL, DT = -85.0, 20.0, 400.0, 1.0
APDS = (225.0, 245.0, 215.0, 235.0)
n = int(len(APDS) * BCL / DT)
tr = torch.full((n,), V_REST, dtype=torch.float64)
for k, apd in enumerate(APDS):
    st_, dur = int(k * BCL / DT), int(apd / DT)
    tr[st_:st_ + 2] = V_PEAK
    tr[st_ + 2:st_ + dur] = torch.linspace(V_PEAK, V_REST, dur - 2)
mb = SimulationResult(times=torch.arange(n, dtype=torch.float64) * DT,
                      Vm=tr.view(-1, 1, 1).expand(n, 30, 8).contiguous())
DI, APD = analysis.restitution_curve(mb.Vm, mb.times, 20, 4)
assert DI.numel() >= 2 and APD.unique().numel() >= 2, (DI, APD)
res = mb.trace(what='restitution', at=(20, 4))
assert res.data and res.vmin is None, res
print('trace OK; restitution DI=%s APD=%s' % (DI.tolist(), APD.tolist()))
"
```

### Phase 2 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# Re-paste the `run_suite` helper (Verification helper section) FIRST — it is a shell function and
# does not survive across Bash invocations; without it this line exits 127 and captures NOTHING (R7 H-1).
run_suite img_p2_before          # BEFORE any Phase-2 edit
```
⚑ **R9 L-12: STOP HERE.** The capture above and the gate below are deliberately in SEPARATE fenced
blocks — pasting one block containing both would run them against the same tree and print
"NO NEW FAILURES" for zero work. Implement the phase, THEN run the next block.
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# ... implement Phase 2 ...
# R6 H-3: rm first and fail HARD on a vacuous run — otherwise `run_suite` returns 1, the `&&`
# short-circuits, `comm` never writes the file, and the `if` reads a MISSING/STALE one and prints
# "NO NEW FAILURES" for a suite that never ran (reproduced).
rm -f /tmp/img_p2_new.txt
run_suite img_p2_after || { echo "GATE FAILED — suite did not run"; exit 1; }
# R7 H-1: comm's failure goes to stderr and its exit status is discarded, so a MISSING baseline
# still creates an empty _new.txt and prints "NO NEW FAILURES". /tmp is reboot-volatile and the
# baseline is captured in a SEPARATE shell, so this is the likely failure, not a theoretical one.
[ -s /tmp/img_p2_before.txt ] || { echo "NO BASELINE at /tmp/img_p2_before.txt — capture it first"; exit 1; }
comm -13 /tmp/img_p2_before.txt /tmp/img_p2_after.txt > /tmp/img_p2_new.txt
if [ -s /tmp/img_p2_new.txt ]; then cat /tmp/img_p2_new.txt; echo "REGRESSION"; exit 1; else echo "NO NEW FAILURES"; fi
```

### Phase 2 Exit Criteria
- [ ] `r.trace(at={"edge": (0,4), "centre": (20,4)})` draws 2 labelled series with a legend.
- [ ] `cc.single_cell('ttp06', pre_pace=2).trace()` draws the 0-D AP and displays inline.
- [ ] `r.trace(what="restitution", at=(20,4))` is marker-only; **works multi-beat on the synthetic `multibeat`
      fixture** and warns single-beat on `long_wave` (R4 H-2 — `long_wave` alone cannot test the multi-beat half).
- [ ] `hline=(-40, "threshold")` draws a labelled reference line; `r.trace(hline=-40)` is not a `TypeError`.
- [ ] Map-only knobs raise on a `Trace`; out-of-range `at` raises naming the valid range.
- [ ] `comm -13` empty. Commit: `feat(cardiac_core): image — Trace panels (series, reference lines, legend)`.

---

## Phase 3: multi-panel + delegation of the legacy stills

**Tier**: **large** (R3 raised this from medium) · **Scope**: layout + shared colorbar + 3 delegations + ~25 tests.
**⚑ This is the ONLY phase that changes shipped behaviour, and it is where R3's C-1, C-2, H-6 and H-7 all
lived.** Treat every delegation as a regression risk with a pre-phase pixel capture, not as a refactor.

### Step 3.0: capture the pre-phase pixel references
**Model**: sonnet
`run_suite img_p3_before`, then **write the reference sizes this phase is gated on** (R6 M-3 — two exit
criteria and Step 3.2 demand a "pre-phase capture" that no step created; the same defect class R3 M-1 fixed for
`run_suite`):
```bash
# R8 L-11: conftest.py's media redirect is pytest-only, so a bare invocation writes into the CHECKOUT.
export CARDIAC_MEDIA_ROOT=$(mktemp -d)
# --no-capture-output is MANDATORY: plain `conda run` discards stdin, so this heredoc would never execute
# and would write a 0-byte refs file while exiting 0 (R7 C-1, reproduced).
/opt/miniforge3/bin/conda run --no-capture-output -n heart-conduction python - <<'PY' > /tmp/img_p3_refs.txt
from PIL import Image as P
import cardiac_core as cc
from cardiac_core import apd_map_figure, activation_isochrones, Video
g = cc.Grid(40, 10, 0.025); cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
sim = cc.monodomain(g, 'ttp06', cond, cc.Stim.boundary(g, 'left', amplitude=-80.0,
                                                       start_time=1.0, duration=2.0))
r = sim.run(t_end=20.0, save_every=1.0)
print('apd', P.open(apd_map_figure(r, 'ref', bulk=True)).size)
print('iso', P.open(activation_isochrones(r, 'ref', bulk=True)).size)
print('prev_bare', P.open(Video.bare(r).preview(slug='ref-b', bulk=True)).size)
print('prev_ann',  P.open(Video.annotated(r).preview(slug='ref-a', bulk=True)).size)
PY
[ -s /tmp/img_p3_refs.txt ] || { echo "STEP 3.0 PRODUCED NOTHING — see the conda-run stdin gotcha"; exit 1; }
cat /tmp/img_p3_refs.txt
```
Phase-3 assertions compare against this file: **±15 %** for the two viz stills (R4 C-2), **exact** for both
previews (R4 C-1/C-2). Executed on the real fixture it yields
`apd (656, 380) · iso (616, 380) · prev_bare (40, 10) · prev_ann (719, 299)`, and Step 3.3's
`dict(l.split(None, 1))` + `eval(refs[k])` parse chain works against exactly that format (verified).
**The A/B table in Step 3.2 used seeded synthetic data (627×380 / 632×380) — those are NOT these numbers**
(R7 L-7); the gate compares against this capture, never against the table.

### Step 3.1: `draw([...])` — grid layout
**Model**: opus

- Accept a list of `Image` and/or `Trace` panels; `_default_layout(n)` reused (1→1×1, 2→1×2, 4→2×2, else n×1);
  `rows=`/`cols=` override with the same "one given, derive the other" rule.
- **Map panels are drawn with `_setup_panel(clip, ax, cmap, norm, units=…, idx=[frame_resolved], label=…,
  lat=<RESOLVED SEMANTICS rule 2, gated on spec.isochrones>, contour_levels=spec.contour_levels,
  filled=spec.filled)`** — the H-6 seam, with the SAME
  `if lat is not None or clip.isochrones:` gate as `_build_figure` (R3 C-3). In `filled` mode `_setup_panel`
  stores the **`QuadContourSet` on `_FigState.im`** (R3 H-2 — it is the colorbar mappable; leaving `im=None`
  makes `fig.colorbar(None, …)` silently draw a meaningless 0–1 bar), and the per-frame `set_data` is guarded
  by `hasattr` (H-4).
- **Shared colorbar — share the mappable of the first map panel with `st.im is not None`; if there is none,
  draw no colorbar** (R9 M-6, single predicate per R10 M-2). ⚑ Do **not** phrase the second clause as "if every
  panel is all-NaN": those are different predicates. `st.im is None` arises **only** in the `filled` + all-NaN
  branch — the default non-filled path gets an `AxesImage` from `ax.imshow` even for a 100 %-NaN array, and a
  data-based test would then suppress the colorbar on a non-filled all-NaN multi-panel, diverging from the
  single-panel path (which draws one under `Gradient`'s documented `(−90, 40)` fallback). The all-NaN `filled` branch leaves `im=None`, which is
  harmless single-panel (no colorbar is drawn) but on the layout path would hand `states[0].im = None` to
  `fig.colorbar`, which **does not raise** — it fabricates a `Normalize(0,1)` bar. Reachable with this plan's
  own `wave` fixture, whose APD map is 0 % finite. Then: only when every MAP panel matches on **BOTH
  `gradient.key()` AND `spec._clip.value_label`**
  (M-23; compare the CLIP's label explicitly — R4 M-3: reading `spec.value_label` before the item-6 write-back
  gives `None` for two panels with *different* derived labels, they compare equal, and an APD map is pooled
  with a voltage map under one colorbar). `render.py:505-509`'s field-kind check is useless here: **every `Image` clip reports
  `field='Vm'`** (M-8 — the recipe passes data as a 2-tuple, not `field=`), so an APD map and a voltage map
  both self-report `"Vm"` and would be pooled under one colorbar. Mismatch → warn "not directly comparable"
  and draw per-panel colorbars. **Trace panels never get a colorbar** and are excluded from the shared axes list.
- **Do NOT mutate the caller's panels** — `labels=` is a render-time override; build a local list, as
  `_render_panels` had to be fixed to do (that bug was found by self-review, not by tests).
- **`try/finally: plt.close(fig)`** around the layout figure (R4 M-4) — Step 1.3 is explicit about this for the
  single-panel case and `render()` closes at `render.py:644-646`, but Phase 3 said nothing, and a leaked figure
  per multi-panel `draw()` across ~25 new tests is a real suite cost.
- **⚑ `front=` and the time stamp must be drawn explicitly on this path (R5 M-5) — but NOT via
  `_produce_figure` (R6 C-2).** `_setup_panel` draws imshow, labels, title and isochrones — **not** `front` and
  **not** a stamp. But calling `_produce_figure` here **raises on every multi-panel draw**: `_setup_panel`
  returns `_FigState(fig=None, …, suptitle=None)` (`render.py:489`) and `_produce_figure` does
  `st.suptitle.set_text(...)` unguarded (`render.py:258,260`; 256 is the stamp f-string — R7 M-1) → verified
  `AttributeError: 'NoneType' object has no attribute 'set_text'`, firing on Phase-3 Exit Criterion #1. Calling
  it per panel would also set the SHARED suptitle N times (last panel wins — a silently wrong stamp when panels
  differ in time) and do a full `fig.canvas.draw()` + RGBA copy per panel whose result is discarded.
  **Mirror the shipped pattern instead** (`render.py:607-609` — R7 M-1 corrects an earlier cite of 602-604, which is the *colorbar* block), in this order:
  1. `sup = fig.suptitle(title or "")`, then assign `st.suptitle = sup` to **every** state;
  2. inline the `front` contour block per panel (`render.py:245-254`) — the one thing `_setup_panel` omits;
  3. set the stamp **once**, from the **first MAP panel** — not "panel 0" (R7 M-3: a mixed
     `draw([Trace(...), Image(...)])` is legal per this step, and a `Trace` panel has no clip and no times).
     **No map panels ⇒ no stamp.** `show_time_resolved` for the layout comes from that same first map panel's
     `clip.times[frame_resolved]`, using the single Step-1.3 formula.
  **Do not call `_produce_figure` on the layout path at all.**
- **Figure construction, stated so a cold-start agent need not guess (R7 M-2, mirroring `render.py:574-591`):**
  `nrows, ncols` from `_default_layout` / `rows=` / `cols=`; `figsize=(min(6.5*ncols, 19.0), min(3.6*nrows, 10.0))`
  when not given; `dpi=150`; `plt.subplots(..., constrained_layout=True, squeeze=False)`;
  `ax.set_axis_off()` on the unused axes beyond `len(specs)`; `tight_resolved` applies to `savefig` as in Step 1.3.
- **`n_panels` on `ImageInfo`** is `len(specs)` here, and **`1`** on every single-spec path — `Image`, `Trace`
  and the `Video` dispatch alike (R7 L-2).
- **⚑ `Trace` panels are drawn with `_draw_trace_on(spec, ax)` (Step 2.2) — NEVER the single-panel wrapper**
  (R8 H-1). The wrapper owns its own `plt.subplots()`, so calling it here leaks a figure past this path's
  `try/finally` **and** leaves the layout's Trace axes blank — the exact silent wrong result the Step-2.2 split
  exists to prevent. Legend placement stays per-axes (`ax.legend()`); the layout figure owns `plt.close`.
  (R7 created the seam and never wired it up: `_draw_trace_on` appeared only in Step 2.2, while Phase 3's
  *"a mixed `[Image, Trace]` list lays out"* exit criterion lives here.)
- **`frame_resolved` is `0` on the layout path** (R8 M-6) — every `Image._clip` holds exactly one frame.
  `frame=` still raises.
- **A BARE `Image` in a list is promoted to the figure producer with a warning**, matching
  `_render_panels` (`render.py:517-523`), and **`enforce_capabilities` is NOT called on the list path** —
  a shared colorbar needs axes (R8 M-10).
- **A `Video` in a list raises**, naming `render([...])` for video multi-panel (R8 L-15).
- Map panels must share `(Nx, Ny)`; raise otherwise. Traces are exempt. Mixed layouts are legal; the shared
  time stamp is map-only.

### Step 3.2: delegate the legacy stills
**Model**: opus

Both delegations are written out in full — "the same treatment" hid four divergences in R1 (H-5).

- **`viz.apd_map_figure(result, slug, *, question="lab", cmap="viridis", bulk=False, root=None, **apd_kw)`**
  ```python
  info = draw(Image(result, what="apd", what_kwargs=apd_kw,
                    gradient=Gradient(cmap=cmap, value_range="auto"),
                    aspect="auto", units="nodes", mask=False, label="APD map"),
              slug + "-apd", question=question, bulk=bulk, root=root,
              figsize=(6.0, 3.0), dpi=120)          # figsize is MANDATORY — R3 H-7
  return info.path
  ```
  **Preserves**: the `-apd` suffix, `aspect="auto"`, node-index labelling, no masking, **`figsize=(6,3)`**,
  dpi 120, the `str` return, always-save, **and the `"APD map"` title** (`viz.py:72`; `_build_figure` titles
  only from `clip.label`, so `label=` is mandatory).
  **⚑ R3 H-7 — `figsize` is not optional**, but it only fixes HALF the divergence. `viz.py:69`/`viz.py:88` use
  `plt.subplots(figsize=(6,3))`; without passing it, `_build_figure`'s aspect-aware default `(6.0+1.6, h+1.2)`
  takes over. Controlled A/B on identical seeded data (`Nx,Ny = 40,10`):

  | | legacy | delegated, `figsize=(6,3)` | delegated, `figsize=None` |
  |---|---|---|---|
  | `apd` | 627×380 | **691×329** | 855×329 |
  | `isochrones` | 632×380 | **694×329** | 860×329 |

  **⚑ R4 C-2 — the residual −13 % height is NOT fixable by any `figsize`/`dpi` value.** It comes from
  `_build_figure`'s `fig.suptitle(title or "")` + `fig.tight_layout()` (`render.py:229-230`), which legacy
  `viz` never calls: adding exactly those two lines to the legacy composition reproduces the delegated size to
  within 1 px (measured 692×330 vs 691×329). **Decision — accept the change and state it, rather than add a
  `tight_layout=`/`suptitle=` parameter pair to `_build_figure`** (which would also need a `st.suptitle is None`
  guard in `_produce_figure`, i.e. more surface on the shipped video path for a diagnostic PNG's pixel count).
  These two functions produce regenerable diagnostic figures; the composition, title, colour and data are
  preserved exactly and only the canvas size moves ~10 %. **The exit criterion is therefore "within ±15 % of
  the pre-phase capture", not equality** — an equality gate would be unsatisfiable and would block the phase.
  The alternative (parameterise `_build_figure`) is recorded here in case exact preservation is ever required.
  Both existing tests assert only `os.path.exists(p) and getsize(p) > 0` (`test_viz.py:22,33,38`), so nothing
  in the suite would have noticed either half.
  **Accepted cosmetic difference (M-28):** `viz.py:70` calls `imshow` with **no `extent`**, so matplotlib uses
  `(-0.5, Nx-0.5, …)` where `_extent_and_labels` returns `[0, Nx-1, …]` — a half-node shift. Documented.
  **Accepted behavioural difference (R3 H-5):** on an all-NaN APD map (any run shorter than ~1 APD — see the
  APD gotcha) the delegated path emits `Gradient.resolve`'s *"no finite unmasked data; falling back to
  (-90, 40) mV"* warning where legacy `viz` was silent. **Keep the warning** — a silently blank APD map is
  exactly the kind of wrong-looking-right result this layer exists to surface — but state it here, and have
  the delegation test assert it with `pytest.warns` rather than being surprised by it.
- **`viz.activation_isochrones(result, slug, *, question="lab", levels=15, cmap="plasma", bulk=False,
  root=None, **lat_kw)`**
  ```python
  info = draw(Image(result, what="activation", what_kwargs=lat_kw,
                    gradient=Gradient(cmap=cmap, value_range="auto"),
                    aspect="auto", units="nodes", mask=False,
                    filled=True, contour_levels=levels, isochrones=False,
                    label="Activation isochrones"),
              slug + "-isochrones", question=question, bulk=bulk, root=root,
              figsize=(6.0, 3.0), dpi=120)          # figsize is MANDATORY — R3 H-7
  return info.path
  ```
  **H-5 — four divergences R1's "same treatment" would have shipped silently:** (1) `filled=True` alone still
  left `isochrones` resolving True, so the figure would have carried filled bands **plus** white lines on top —
  fixed in Step 1.2 item 3 (`filled` suppresses the auto-isochrones) and pinned by `isochrones=False` here;
  (2) `**lat_kw` was never mapped (`viz.py:85` forwards it to `activation_time`) → silently changed threshold,
  now `what_kwargs=lat_kw`; (3) `mask=False`/`units="nodes"`/`aspect="auto"`/`dpi=120` were only implied;
  (4) `viz.py:89`'s all-NaN guard — handled by the `filled` all-NaN rule in Step 1.3 (no contour set, no
  colorbar, still titled).
- **`video.render.preview_frame(...)`** → a **function-local** `from ..image._draw import draw`, passing the
  existing `Video` straight through. **The real code (R3 H-6 — an earlier revision showed
  `return ImagePath(*_finalize(...))` with the `draw(...)` call commented out, which is not implementable:
  `draw()` already runs `_finalize` internally, and `_finalize` DELETES the temp file (`render.py:118-119`),
  so a second call raises `FileNotFoundError`):**
  ```python
  info = draw(video, slug, frame=t, show_time=True, resolution=None, format="png", path=path,
              question=question, bulk=bulk, date=date, root=root, units=units,
              title=title, figsize=figsize, dpi=dpi)      # RAW dpi — see R4 C-1
  return ImagePath(info.path, info.data)        # ImagePath(path, data) — encoders.py:37
  ```
  **`format="png"` is explicit (R5 M-7).** `preview_frame` is documented "Render ONE frame to PNG", and today
  `_resolve_destination(slug, "images", "png", …)` forces the extension. `draw()` instead derives `format`
  from `path`, so without this `Video(r).preview(path="frame.jpg")` writes a **JPEG** while the returned
  `ImagePath._repr_html_` hard-codes `data:image/png;base64` (`encoders.py:61-64`) — a JPEG payload served
  under a PNG MIME type. With `format="png"` explicit, a `.jpg` path now hits the format/path disagreement
  raise, which is the honest outcome. Also preserve `preview_frame`'s
  `IndexError(f"frame {t} out of range for {len} frames")` (`render.py:422-423`, R5 L-8).
  **`resolution=None`** (H-3): without it the new `"auto"` upscale changes a bare preview from the raw grid
  (measured 30×8) to 540×144. **`dpi=dpi`, passed RAW (R4 C-1)** — *not* `(dpi or 100)`, which would make
  `enforce_capabilities` raise on every bare preview and redden four green tests. `draw()` applies
  `dpi or 100` for a `Video` spec **after** the gate, on the annotated branch, so the annotated preview keeps
  its historical size (`preview_frame` passes `dpi` raw to `_build_figure` at `render.py:431`, which applies
  `dpi or 100` at `render.py:201`, and again at `render.py:434` — R4 L-3 corrects the earlier citation).
  Without that, the annotated preview silently resizes ~1.5× (capture the exact before/after on the `wave`
  fixture immediately before Phase 3 rather than trusting a literal — R4 L-2). **`show_time=True`** preserves
  the unconditional stamp.
  An annotated `Video` spec is exempt from the `resolution`-raises rule **only when `resolution` is its
  default `None`** — which is exactly what the delegation passes (R3 C-1, scoped per R12 M-3; the whole-column
  form is what R4 H-4 rejected as a silent no-op) — `test_video.py:605`
  previews an annotated clip and is currently green.
  `test_video.py` is **not** a sufficient gate: `test_preview_bare_has_no_chrome` asserts only
  `annotated.mean() > bare.mean()`, which survives a resize AND black padding. Phase 3 therefore adds **three**
  explicit tests: bare-preview pixel dims equal the raw grid; **annotated-preview dims equal a pre-phase
  capture**; and a burned-stamp pixel-diff.
- **Pin both viz compositions too (R3 M-12, bounded by R4 C-2).** Before Phase 3, capture
  `PIL.Image.open(p).size` for `apd_map_figure` and `activation_isochrones` on the `test_viz.py` fixture;
  assert the delegated outputs are **within ±15 %** on both axes, and record the exact before/after pair in the
  Mutation Log. **Measured against the Step-3.0 references (R8 L-12): apd (656,380) → (693,329) = +5.6 %/
  −13.4 %; iso (616,380) → (681,329) = +10.6 %/−13.4 %.** Both pass, but the height sits **1.6 pp** from the
  gate on both — record these so a future `tight_layout` metric change reads as diagnosable, not mysterious. Equality is unsatisfiable (R4 C-2), but an unbounded gate would have missed H-7's +31 % width.

### Step 3.3: Verify (R6 M-4 — Phase 3 is the only phase that changes shipped behaviour and had no executable check)
**Model**: sonnet
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
export CARDIAC_MEDIA_ROOT=$(mktemp -d)   # R8 L-11: keep bulk=True output out of the checkout
/opt/miniforge3/bin/conda run -n heart-conduction python -c "
from PIL import Image as P
import cardiac_core as cc
from cardiac_core import Image, draw, Video, apd_map_figure, activation_isochrones
g = cc.Grid(40, 10, 0.025); cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
sim = cc.monodomain(g, 'ttp06', cond, cc.Stim.boundary(g, 'left', amplitude=-80.0,
                                                       start_time=1.0, duration=2.0))
r = sim.run(t_end=20.0, save_every=1.0)
refs = dict(l.split(None, 1) for l in open('/tmp/img_p3_refs.txt'))
def size(p): return P.open(p).size
# R6 C-2: a two-panel draw must NOT raise (the _produce_figure/suptitle=None crash).
two = draw([Image(r, label='control'), Image(r, label='drug')], path='/tmp/_cmp.png')
assert two.saved and two.n_panels == 2, two
# ⚑ R9 H-1 — `assert act.saved` / `assert fr.saved` CANNOT FAIL for the reason they claim: a two-panel
# activation figure with ZERO contours still saves, and a layout that silently drops `front` still saves.
# This is the only executable gate in the only phase that changes shipped behaviour, so COUNT ARTISTS via
# the consumer-binding spy already proven in Step 1.3 (patch _setup_panel, the layout's drawing seam).
# Keep a REFERENCE to each panel's Axes: matplotlib artists stay inspectable after plt.close(), so the
# counts can be read once draw() has returned — which is the only point at which `front` has been drawn.
import sys
_D = sys.modules['cardiac_core.image._draw']
axes_seen = []
_orig_sp = _D._setup_panel
def _spy_sp(clip, ax, cmap, norm, **kw):
    axes_seen.append(ax)
    return _orig_sp(clip, ax, cmap, norm, **kw)
_D._setup_panel = _spy_sp
try:
    axes_seen.clear()
    act = draw([Image(r, what='activation'), Image(r, what='activation')], path='/tmp/_act.png')
    act_counts = [len(a.collections) for a in axes_seen]
    axes_seen.clear()
    fr = draw([Image(r, front=-40.0), Image(r)], path='/tmp/_front.png')
    front_counts = [len(a.collections) for a in axes_seen]
finally:
    _D._setup_panel = _orig_sp
# H-6: contours on BOTH activation panels, exactly one set each (2 would be the filled+lines double-draw).
assert act_counts == [1, 1], f'activation panels drew {act_counts} contour sets, expected [1, 1]'
# R5 M-5: `front` is drawn on the layout path — the front panel must carry MORE artists than the plain one.
assert front_counts[0] > front_counts[1], f'front= was dropped on the layout path: {front_counts}'
assert act.saved and fr.saved
# previews EXACT vs the pre-phase capture; viz stills within +/-15%.
for k, got in (('prev_bare', size(Video.bare(r).preview(slug='c-b', bulk=True))),
               ('prev_ann',  size(Video.annotated(r).preview(slug='c-a', bulk=True)))):
    assert str(got) == refs[k].strip(), (k, got, refs[k])
for k, got in (('apd', size(apd_map_figure(r, 'c', bulk=True))),
               ('iso', size(activation_isochrones(r, 'c', bulk=True)))):
    ref = eval(refs[k]); assert all(abs(a-b) <= 0.15*b for a, b in zip(got, ref)), (k, got, ref)
print('phase 3 OK')
"
/opt/miniforge3/bin/conda run -n heart-conduction python -c "import cardiac_core.video; print('video imports standalone')"
```

### Phase 3 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# Re-paste the `run_suite` helper (Verification helper section) FIRST — it is a shell function and
# does not survive across Bash invocations; without it this line exits 127 and captures NOTHING (R7 H-1).
run_suite img_p3_before          # BEFORE any Phase-3 edit
```
⚑ **R9 L-12: STOP HERE.** The capture above and the gate below are deliberately in SEPARATE fenced
blocks — pasting one block containing both would run them against the same tree and print
"NO NEW FAILURES" for zero work. Implement the phase, THEN run the next block.
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# An absolute-pass gate on test_video.py is unsatisfiable today (CUDA OOM from another user's process),
# hence the baseline diff below rather than a pass count.
# ... implement Phase 3 ...
# R6 H-3: rm first and fail HARD on a vacuous run — otherwise `run_suite` returns 1, the `&&`
# short-circuits, `comm` never writes the file, and the `if` reads a MISSING/STALE one and prints
# "NO NEW FAILURES" for a suite that never ran (reproduced).
rm -f /tmp/img_p3_new.txt
run_suite img_p3_after || { echo "GATE FAILED — suite did not run"; exit 1; }
# R7 H-1: comm's failure goes to stderr and its exit status is discarded, so a MISSING baseline
# still creates an empty _new.txt and prints "NO NEW FAILURES". /tmp is reboot-volatile and the
# baseline is captured in a SEPARATE shell, so this is the likely failure, not a theoretical one.
[ -s /tmp/img_p3_before.txt ] || { echo "NO BASELINE at /tmp/img_p3_before.txt — capture it first"; exit 1; }
comm -13 /tmp/img_p3_before.txt /tmp/img_p3_after.txt > /tmp/img_p3_new.txt
if [ -s /tmp/img_p3_new.txt ]; then cat /tmp/img_p3_new.txt; echo "REGRESSION"; exit 1; else echo "NO NEW FAILURES"; fi
/opt/miniforge3/bin/conda run -n heart-conduction python -c "import cardiac_core.video; print('video imports standalone')"
```

### Phase 3 Exit Criteria
- [ ] `draw([Image(a, label="control"), Image(b, label="drug")], "compare")` → 2 panels, ONE shared colorbar.
- [ ] `draw([Image(a, what="activation"), Image(b, what="activation")])` → **contours on BOTH panels** (H-6),
      counted as artists.
- [ ] A filled multi-panel's shared colorbar `norm.vmin/vmax` equal the **pooled data range**, not 0–1 (R3 H-2).
- [ ] Panels with different `value_label`s get per-panel colorbars + a warning.
- [ ] A mixed `[Image, Trace]` list lays out; only the map gets a colorbar.
- [ ] `Video.preview()` is **pixel-identical** to today on **both** the bare and the annotated path, still
      stamped, and `Video.annotated(...).preview()` **does not raise** (R3 C-1/C-2).
- [ ] Both viz delegations are within **±15 %** of their pre-phase pixel dimensions, with the exact pair
      recorded in the Mutation Log (R3 H-7 / R4 C-2 — equality is unsatisfiable).
- [ ] `Video.bare(...).preview()` still returns a raw-grid PNG and does not raise (R4 C-1).
- [ ] `import cardiac_core.video` alone succeeds.
- [ ] `comm -13` empty. Commit: `refactor(cardiac_core): image — multi-panel + legacy stills delegate`.

---

## Phase 4: docs + acceptance

**Tier**: small

### Step 4.0: capture the Phase-4 baseline
(R3 M-1 — the Exit Criteria referenced `/tmp/img_p4_before.txt`, which no step created; R10 L-3 gives it the
same fenced block + STOP split as Phases 1-3.)
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# Re-paste the `run_suite` helper (Verification helper section) FIRST — it is a shell function and does
# not survive across Bash invocations; without it this exits 127 and captures NOTHING.
run_suite img_p4_before          # BEFORE touching any doc
```
⚑ **STOP HERE.** Implement Phase 4, THEN run the gate block below.
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
rm -f /tmp/img_p4_new.txt
run_suite img_p4_after || { echo "GATE FAILED — suite did not run"; exit 1; }
[ -s /tmp/img_p4_before.txt ] || { echo "NO BASELINE at /tmp/img_p4_before.txt — capture it first"; exit 1; }
comm -13 /tmp/img_p4_before.txt /tmp/img_p4_after.txt > /tmp/img_p4_new.txt
if [ -s /tmp/img_p4_new.txt ]; then cat /tmp/img_p4_new.txt; echo "REGRESSION"; exit 1; else echo "NO NEW FAILURES"; fi
/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_integrity.py -q
```

### Step 4.1: `API_CHEATSHEET.md` §10
**⚑ §10 is exec'd by `cardiac_core/tests/test_video.py::test_cheatsheet_video_section_executes`, which takes
the FIRST block whose first line is exactly `# runnable-video-section`.** **Extend that existing block — do not
add a second marked block** (L-24: only `runnable[0]` is exec'd, so a second one would never run and would
rot). Keep `r.video()` / `Gradient` / `render([...])` executable, add the figure API alongside, state the
display-vs-save rule once for both media kinds, and demonstrate `path=`/`save()` in comments only so the canary
does not litter the working directory.

### Step 4.2: `API_OBJECTS.md`
Add `Image`, `Trace`, `ImageInfo`, **verified by MRO introspection**, using the established
`| Access | Meaning |` / `| Call | Does |` tables. **Placement (L-33):** the file runs `§9 Video/Gradient/
VideoInfo/ImagePath`, `§10 CardiacMeshData`, `§11 Distribution` — insert the image objects as a **new §10** and
renumber the two following sections. **Also add a row to the summary object map at `API_OBJECTS.md:9-24`**,
which already lists `Video`/`Gradient`/`VideoInfo` (R3 L-3). **Verified structure (R4 L-9):** `## 8.
SingleCellResult` (line 336) · `## 9. Video, Gradient, VideoInfo` (365 — note the heading omits `ImagePath`
even though the section documents it) · `## 10. CardiacMeshData` (481) · `## 11. Distribution and
SimulationSnapshot` (511) · then **un-numbered** `## Free functions` (544) and `## Units` (560), which the
renumbering does not touch. The file is currently **untracked in git**; `git add` it with this commit.

### Step 4.3: `/sim-media` skill
Lead with `r.image()`/`r.trace()`, state the display-vs-save rule, and keep "report the exact saved paths;
never claim a figure you didn't write to disk" — which now has teeth, since a figure is often deliberately NOT
written.

### Step 4.4: acceptance — reproduce the corpus on synthetic data
A scratch script (written to the scratchpad, then deleted) reproducing four real corpus compositions **without
touching the research scripts**:
1. the 3-labelled-series boundary-vs-centre trace (`diag_column_boundary_vs_center.py`),
2. the 2-panel curve figure with a zero reference line (`render_proof_summary_still.py`),
3. a bare full-frame Vm still (`render_proof_sourcesink.py`) — **assert pixel dimensions and aspect**, not just
   that it renders,
4. a 2-panel map comparison with one shared colorbar.
Each must be **one `draw()` call**. Anything inexpressible is a gap to fix or a documented limitation — record
which, in the Mutation Log.

### Phase 4 Exit Criteria
- [ ] §10 covers figures; `test_cheatsheet_video_section_executes` still passes; no stray files.
- [ ] All four acceptance compositions render (item 3 dimension-checked), or are documented as limitations.
- [ ] The Phase-4 `run_suite` + `[ -s … ]` regression gate passes; integrity goldens bit-identical.
- [ ] Commit: `docs(cardiac_core): image layer — cheatsheet, object atlas, sim-media`.

---

## Final Cleanup
- Archive this plan to `Research/Active/engine_consolidation/plans/{date}_IMAGE_OBJECT_PLAN.md`.
- Update the question's `KNOWLEDGE.md` (a SHIPPED callout) and `IDEALOG.md` (Session Log + Current Direction).
- **Sync note:** `cardiac_core` is also a standalone public repo (`github.com/RealJokerInc/cardiac-core`). A new
  `cardiac_core/image/` subpackage is picked up automatically by `include = ["cardiac_core*"]`, but the
  monorepo→standalone sync is a known-open item and this layer does not reach GitHub until `extract.sh` re-runs.

## Mutation Log
| Date | Round | Change |
|---|---|---|
| 2026-07-25 | created | Blueprint from the user directive + corpus census + a verified environment probe. |
| 2026-07-25 | **R1** (5C/8H/15M/11L) | **C1** mask through the constructor. **C2** `value_label` assigned onto the clip. **C3** `_build_figure` gains `lat=`. **C4** `fields.*` rank split. **C5** the `(times,V)` 2-tuple (no `times=` kwarg). **H6** `enforce_capabilities` before defaults. **H7** function-local import for the Phase-3 edge. **H8** `draw()` dispatches a `Video`. **H9** the bare path needs a canvas fit. **H10** Phase-3 baseline discipline. **H11** input rule narrowed. **H12** legacy titles + `filled`/`levels` decided. **H13** working-tree section. **M14–M28 / L29–L39** as listed in the R1 entry of the previous revision (all retained below where still applicable). |
| 2026-07-25 | **R2** (1C/5H/13M/12L) — *five of six C/H were follow-ons to R1's own fixes* | **C-1** R1's `gradient=None` default **crashes the zero-argument headline call** (`Video.gradient` is a dataclass default, not a `None` guard; `_build_figure` reads `clip.gradient.interpolation`) — resolve the gradient in item 2 and capture `_gradient_was_explicit` first; R1's Verify passed vacuously over it. **H-2** R1's `Video` dispatch had **no parameter to dispatch on** → `draw(frame=…)`. **H-3** R1's new bare pipeline **silently deletes `burn_timestamp`**, and its `resolution` default would change `preview_frame`'s pixel size; the named gate provably catches neither → burn after the resize, delegate with `resolution=None`, add two explicit tests. **H-4** `_produce_figure` calls `st.im.set_data` unconditionally → `filled` mode (`im=None`) raises; add the guard. **H-5** the `activation_isochrones` delegation would draw filled bands **plus** lines, drop `**lat_kw`, and lose the all-NaN guard → both delegations written out in full; `filled` suppresses auto-isochrones. **H-6** the `lat=` seam stopped at `_build_figure`; the multi-panel path draws through **`_setup_panel`**, so activation panels would come out contour-free with no warning → same three params on both. **M-7** the census command's `-h -o` stripped the path, making its exclusion a no-op; command and numbers restated and reproduced. **M-8** `_clip.field` is `'Vm'`, not an ndarray — corrected reason, `field` added to the Reuse table. **M-9** `what_kwargs` threaded into the LAT; `_lat` cached. **M-10** unwrap `VectorField` then branch on `ndim`; `velocity`/`direction` are static; `mask` rejected. **M-11** `levels` → `contour_levels` (collides with `Gradient.levels`). **M-12** input rule narrowed to what actually works. **M-13** `.jpeg` alias must not be rewritten. **M-14** `resolution=`/`fit=` raise on an annotated spec. **M-15** bare default is an integer no-padding upscale, not a 53%-black 1080p canvas. **M-16** one `run_suite` helper; vacuity guard on the **after** file; every phase gated. **M-17** `trace_keys`. **M-18** `linestyle`/`xlim`/`ylim` added; out-of-scope verbs stated. **M-19** `self.levels` in a module-level `draw()` → `getattr` on the spec. **L-20** line numbers are working-tree and will shift. **L-21** import list extended to 15 names across two modules. **L-22** the mismatch message pinned. **L-23** dead Verify line removed. **L-24** extend the existing `# runnable-video-section` block. **L-25** `propagation_video` is a video, not a figure. **L-26** module-scoped fixture. **L-27** `SingleCellResult` guard. **L-28** `ax.set_aspect` in filled mode. **L-29** 0-frame message. **L-30** `clip.result` is currently unread on the `Image` path. **L-31** synthetic finite obstacle for the mask test. |
| 2026-07-25 | **R3** (3C/7H/13M/9L) — *eight of ten C/H were follow-ons to R1/R2 fixes*; R3 independently re-verified the census, baseline, working-tree inventory, ~30 file:line citations, the import list, the rank rule, the format matrix, and R2's gradient/`filled`/upscale fixes as correct | **C-1** R2's H-3 (`resolution=None` mandatory on the delegation) and M-14 (`resolution` raises on annotated) **contradict**; `test_video.py:605` previews an annotated clip and is green → a spec-type-dependent rule table, `Video` specs exempt. **C-2** the delegation's `dpi=None` picks up the new 150 default → **silent 1.5× resize** of the annotated preview (720×299 → 1079×448); pass `dpi=(dpi or 100)` and pin it with a test. **C-3** `lat=` alone draws nothing — the gate is `if clip.isochrones:` and the recipe hard-codes it `False`; gate becomes `if lat is not None or clip.isochrones:`, and the Verify now COUNTS contour artists instead of asserting a missing warning. **H-1** `style="bare"` silently discards `isochrones`/`filled` because the recipe defeats both of `Video`'s own guards → explicit raise. **H-2** `fig.colorbar(None, …)` **does not raise** — matplotlib fabricates a `Normalize(0,1)` bar (measured 0–1 vs a true LAT range 1.344–11.526) → store the `QuadContourSet` on `_FigState.im`, guard with `hasattr(…, "set_data")`. **H-3** `what_kwargs` forwarded to `activation_time` for a non-activation `what` → `TypeError` on `repol=` or a silently retargeted LAT on `threshold=` → forward only for `what="activation"`. **H-4** `Image.value_label` was documented but unconditionally overwritten → explicit precedence. **H-5** **every APD assertion was vacuous**: `apd_map` is 0 % finite at `t_end` 12 **and** 20 ms (measured), and an all-NaN map falls back to exactly the (−90,40) range the test was meant to exclude → two module-scoped fixtures (2.3 s / 44.6 s, APD max 230 ms), plus the delegation's new degenerate-range warning documented and asserted. **H-6** the `preview_frame` snippet was unimplementable (double `_finalize` → `FileNotFoundError`) → real code. **H-7** both viz delegations dropped `figsize=(6,3)` → **+31 % width** (656×380 → 858×329, measured), invisible to tests asserting only `getsize > 0`. **M-1** `run_suite` was defined inside the Phase-1 block → hoisted; Phase 4 gained step 4.0. **M-2** `_gradient_was_explicit` had no consumer → a local with a stated use. **M-3** `test_two_panel_masked_obstacle` **does not exist** → `test_reproduces_semicircle_composition`. **M-4** `_to_numpy` lives in `clip.py`, a third module. **M-5** `t_ms = nan` for static maps. **M-6** the unknown-`what` message now lists the four named intents and points at `Trace`. **M-7** `field=` narrowed to `Optional[str]`. **M-8** `Image` is resolved-at-construction, unlike live `Video` — documented divergence. **M-9** `Trace.what_kwargs`. **M-10** map-only knobs split by where they live (`draw()` params vs `Image` fields). **M-11** `colorbar` resolution stated; the filled colorbar put behind it. **M-12** pre-phase pixel capture for both viz delegations. **M-13** `text`/`annotate` (48 calls — more than `contour`) added to OUT OF SCOPE as a decision. **L-1** the ranking is NOT identical across census scopes. **L-2** the `tight` measurement generalised wrongly (it can grow a figure: 600×300 → 516×316). **L-3** the `API_OBJECTS.md` summary map needs a row. **L-4** `read()` must fall back to the file. **L-5** `tight`/`transparent` raise on the bare path. **L-6** `at` is an INDEX for array input. **L-7** the `.trace()` guard Verify used a 1010-frame 0-D run. **L-9** validate `fit` against `_LEGAL_FIT`. |
| 2026-07-25 | **R4** (3C/4H/7M/9L) — R4 re-verified the census, baseline, tree inventory, every line citation, `ImagePath.__new__`, the rank rule, the format matrix, the fixture timings and R3's H-2 colorbar fix as correct | **⚑ Design fix first:** R4's key observation — *three rounds each added a rule to the `isochrones`/`lat`/`filled` triangle and none wrote the three down together* — is addressed by a new **"RESOLVED SEMANTICS"** section that states all six rules in one place; C-3/H-1/H-3 all lived in that gap. **C-1** R3's `dpi=(dpi or 100)` would make `enforce_capabilities` **raise on every bare preview** (`render.py:144`), reddening 4 green tests (`test_video.py:595,601,963,972`) and breaking `Video.bare(...).preview()` → the delegation passes `dpi` **raw**; no rule may substitute a non-`None` figsize/dpi before the gate. **C-2** the viz pixel-equality criterion is **unsatisfiable**: `figsize` fixes only the width half; the residual −13 % height is `_build_figure`'s `suptitle("")`+`tight_layout()`, which legacy `viz` never calls (adding exactly those two lines to legacy reproduces the delegated size to 1 px) → accept the change, document the measured A/B, relax the gate to ±15 %. **C-3** `_lat` was populated for `what="activation"` and passed ungated, so the `activation_isochrones` delegation (`filled=True, isochrones=False`) drew lines over the bands — **re-creating the exact double-draw R2's H-5 removed** → `lat` is passed iff the RESOLVED `isochrones` is true. **H-1** `_lat` bypasses `isochrone_lat`'s masking (`render.py:170-171`), so contours ran through masked obstacles → mask `lat` exactly as the display array. **H-2** **no fixture supports restitution**: `long_wave` fires one stimulus and yields `DI.shape == (0,)` (measured) → a third, SYNTHETIC `multibeat` result (a real 3-beat run would cost ~98 s). **H-3** two contradictory `lat`-compute rules — one drew isochrones on APD maps that never asked, the other drew nothing for `Image(isochrones=True)` on a snapshot → one rule, in RESOLVED SEMANTICS. **H-4** the `Video` `resolution` exemption was written for the whole column, making `draw(Video.annotated(…), resolution="1080p")` a silent no-op → exempt the DEFAULT value only. **M-1** `tight` defaulted to `True`, so the guard raised on `tight=False` and stayed silent on the actual no-op → `Optional[bool] = None`. **M-2** `frame=` "ignored for `Image`" → raises. **M-3** `value_label` write-back so `spec.value_label` and `spec._clip.value_label` cannot disagree (the shared-colorbar comparison reads the clip). **M-4** `_render_panels`' `set_data` needs the same `hasattr` guard; multi-panel needs `plt.close` in a `finally`. **M-5** `what_kwargs` on a no-analysis selector and an explicit `show_time=True` on a NaN-time map both raise (the stamp would render `"t = nan ms"`). **M-6** **`at` had three meanings** (ms / frame index / node) — R4 called it the worst seam in the API for this audience → `(T,Nx,Ny)` array input rejected; `at` is always a time. **M-7** the filled Verify certified nothing (R3 C-3's own criticism one block lower) → artist counts + colorbar-norm check + a double-draw assertion. **L-1/L-2** two "measured" literals did not reproduce → removed rather than re-quoted. **L-3** `dpi or 100` is at `render.py:201`, not `:431`. **L-5** `import … as R` also yields the function. **L-6** the input-type guards move to item 1b. **L-8** the `comm -13` gate now fails the shell, not the eye. **L-9** `API_OBJECTS.md`'s real section structure recorded. |
| 2026-07-25 | **R12 — ✅ CONVERGED (0C/0H/3M/5L).** Verdict: *"READY TO IMPLEMENT: YES. Nothing blocks it."* R12 executed all three R11 fixes (item 1c now yields the pinned `domain gate` message for `what="mask"` AND the valid-keys message for `what="nope"`; all **10** spec×style×value `resolution`/`tight` combinations agree; the numbering is gapless; the scoped bare raise leaves the shipped `Video.bare(v).preview(units="cm")` working — and measured that it is byte-identical to the default today, i.e. a genuine pre-existing no-op), re-ran the census, the out-of-scope counts, Step 3.0, the `_draw.py` shadowing probe, the artist counts, the `multibeat` gate, the 7/7 re-wrap, the fixture timings and **~30 citations** — all exact — and `bash -n`-checked all 16 fenced blocks | **M-1** Steps 1.2/1.3's Verify blocks imported `Image`/`draw` from `cardiac_core` before Step **1.4** registers them in `_LAZY` (`ImportError`, verified) → submodule paths, matching Step 1.1. **M-2** R11's renumber left **eight** stale `item 6`/`6b` cross-references pointing at what is now item 5/5b → swept. **M-3** Step 3.2 restated the `Video` resolution exemption unqualified — the whole-column form R4 H-4 rejected — in the step the delegation is copied from → scoped to the default `None`. **L-1** `render.py:146`→`140-141`. **L-2** the lazy `image/__init__` means a bare `import cardiac_core.image` does NOT pull in video/matplotlib, so Step 1.1's imageio/cv2 guard cannot fail (kept; rationale corrected). **L-3** "all three tuples" → two. **L-4** `units` is in `_IMAGE_KEYS`, so the headline `r.image(style="bare", units="cm")` bypassed `draw()`'s check → the guard goes in item 3b too. **L-5** the 18 helpers span three modules, not two. |
| 2026-07-25 | **R11** (0C/1H/2M/5L) — R11 **built the three candidate package layouts and executed them**, confirming the `_draw.py` rename fixes the shadowing (`identity_stable=True`, `cc.draw` callable twice) where `draw.py` gives `'module' object is not callable`; ran the re-wrap against **all 7** raise paths (`style="annotated"` present, `Video` absent, incl. `label=` and `front=`); `bash -n`-checked **all 16** fenced blocks; and re-verified the census, the artist counts, the multibeat gate, the clip recipe, the working tree and ~25 citations | **H-1** R10's own L-1 relocation broke item 2: **1c subtracts `mask` from the valid set and runs BEFORE item 2**, so item 2's `"'mask' is the domain gate"` message became dead code and Step 1.2's Verify needle failed on conforming code (`needle 'domain gate' present? False`) — the same can't-pass-on-correct-code class as R10 H-1, re-created by R10's fix → `mask` special-cased in 1c with the pinned message. **M-1** the spec-type table restated the `resolution` rule in the exact *"non-default"* phrasing the signature block condemns by name (it re-creates R5 H-4's silent no-op) → *"not `_UNSET`"*. **M-2** THE DEFAULT table read as if `tight=None` raises on the bare branch, which would blow up Step 1.3's own bare Verify → *any NON-`None` value* raises. **L-1** the "guard CANNOT catch it" claim was overstated — the collision check can't, but the identity assertion does (verified). **L-2** three surviving `draw.py` references. **L-3** stale "fifteen helpers"/"17-name" counts. **L-4** the `__post_init__` recipe skipped ordinal 5 → renumbered gapless (1, 1b, 1c, 2, 3, 3b, 4, 5, 5b, 6). **L-5** R10 L-6's new raise was unimplementable for `aspect` (non-`None` default), ambiguous for `units` (a `draw()` param AND an `Image` field), and would have **reddened the shipped `Video.bare(v).preview(units="cm")`** → scoped to the `Image` spec, `aspect` dropped, wording pinned. |
| 2026-07-25 | **R10** (0C/2H/2M/7L) — R10 re-executed Step 3.0 (references exact), the `multibeat` block, the census, the artist-count mechanics (**confirming matplotlib Axes stay inspectable after `plt.close()`**, so R9's spy is sound and cannot pass vacuously), the 18-name import surface, the 7/7 re-wrap, and ~20 citations — all correct | **H-1** Step 1.2's new raise loop asserted `Image(r, style='bare', label='x')` raises, but **`label` is not in item 3b's list** and the re-wrap lives in `draw()` — the plan says so twice — so the constructor returns normally and the Verify **dies on conforming code**, with the only way to green it being to falsify Step 1.3's own rationale → the tuple (plus a `front=` twin) moved into the `draw()` loop. **H-2** ⚑ **`image/draw.py` + a `draw` export reproduces the `single_cell` shadowing bug this plan itself forbids**: importing the submodule binds `cardiac_core.image.draw = <module>`, PEP 562 never fires again, and `cc.draw` — the headline verb — is a non-callable module after first access (reproduced against the project's own non-caching `__getattr__`). The top-level guard **cannot** catch it (`n == mod` vs `'draw': 'image'`) → renamed **`_draw.py`**, matching the `_single_cell.py` precedent. **M-1** Phase 3's gate block had a corrupted `cd` line (`cd: too many arguments`) → restored. **M-2** R9 M-6's two clauses tested different predicates (`st.im is None` arises only in the filled+all-NaN branch; a data-based test would suppress the colorbar on a non-filled all-NaN panel) → one predicate. **L-1** the unknown-`what` check is now genuinely renumbered `1c` and MOVED before item 2, not annotated in place. **L-2** the bare-`Video` resolution row scoped. **L-3** Phase 4 gained a real fenced gate + STOP split. **L-4** `_clip`/`_lat` are not dataclass fields (the `_IMAGE_KEYS` guard subtracts from `__dataclass_fields__`). **L-5** the `labels=`/`multi-panel` needle pinned. **L-6** `units=`/`aspect=`/`colorbar=False` on a bare spec were the last silent no-ops → raise. **L-7** a raw `(x, y)`/dict `Trace` gets `xlabel`/`ylabel` `None`, not `what`'s defaults. |
| 2026-07-25 | **R9** (0C/1H/6M/7L) — ⚑ **R9 answered the decisive question YES: *"a cold-start agent can execute Phase 1 through Phase 4 from this document. Nothing blocks it."*** It executed Step 3.0 verbatim (reproducing the four references exactly), the `multibeat` snippet, the census, the out-of-scope counts, the fixture timings and every field rank — and **emulated the `preview_frame` delegation under the plan's stated rules, getting BYTE-IDENTICAL output to the shipped function on both the bare and annotated paths** (`e39072d1…`/`86f22ca2…`), plus both viz delegations reproducing R8 L-12's numbers to the pixel | **H-1** Step 3.3's `assert act.saved` / `assert fr.saved` **cannot fail for the reason they claim** — a contour-free activation figure and a layout that drops `front` both still save; it was the only executable gate in the only phase that changes shipped behaviour, and the fifth instance of the can't-fail class → the spy now counts artists per panel (`[1,1]`, and `front_counts[0] > front_counts[1]`). **M-2** R8's own H-3 re-scoping re-created **R5 H-4's forbidden silent no-op**: "neither `_UNSET` nor that spec type's resolved default" makes `draw(Image(r), resolution="auto")` on the annotated branch neither raise nor act → the rule now compares against the **sentinel**, per spec type. **M-3** `field=` with a non-default `what=` matched two selector rows and was undefined → legal only with `what="snapshot"`. **M-4** `Trace.data`'s `(x, y)` and `dict` forms were declared and never defined → defined. **M-5** **six raise rules — each added because the silent version was a defect — had no executable check**, incl. R8's own M-4 → both Verify blocks gained raise loops. **M-6** all-NaN + `filled` + multi-panel hands `states[0].im = None` to `fig.colorbar`, re-creating R3 H-2's fabricated 0–1 bar, and it is reachable with this plan's own `wave` fixture → pick the first non-`None` mappable, else no colorbar. **M-7** the unknown-`what` raise sat *after* the selector resolution that cannot run for an unknown `what` → folded into item 2's dispatch. **L-8** the `_render_panels` guard's declared owner did not exist → moved into Step 1.3's edit list. **L-9** `run.py:54-67` → `55-68`; the bare-preview call lines noted alongside the test lines. **L-10** the import surface is **18** names (`_LEGAL_FIT` was missing). **L-11** Step 2.4 never exercised `multibeat` — the fixture's only gate was an unchecked checkbox → the restitution build + `mb.trace(what="restitution")` are now in the Verify. **L-12** Phases 2/3 had `before` and `after` in ONE pasteable fence (paste it and zero work certifies green) → split, with an explicit STOP. **L-13/L-14** the 719-vs-720 literal, `image/__init__`'s exports (only `info.py` exists at Step 1.1), and `result`'s binding. |
| 2026-07-25 | **R8** (0C/3H/7M/5L) — **first round with no criticals.** R8 EXECUTED every R7 fix: Step 3.0 runs and yields `apd (656,380) · iso (616,380) · prev_bare (40,10) · prev_ann (719,299)` with Step 3.3's parse chain working; all four baseline-guard paths correct; the `enforce_capabilities` re-wrap matches **7/7** raise paths with no `Video` surviving; the `multibeat` snippet executes and passes its own gate (`DI=[180,207]`, `APD=[193,211]`) | **H-1** R7 created `_draw_trace_on` and **never wired it up** — `grep` found it only in Step 2.2, while the *"mixed `[Image, Trace]` lays out"* exit criterion lives in Step 3.1, which said nothing about drawing a trace → the two-layer problem again, now fixed in Step 3.1. **H-2** Step 1.4 was not swept: it still specified a **method-local `image_keys`** while its own guard test and Step 2.3 require an importable `run._IMAGE_KEYS` — and Phase 1 is committed before Step 2.3 is ever read → module-level constant + a real assert. **H-3** the `resolution`/`fit` raise rule was stated **two incompatible ways**, and the unqualified version (inside the signature block an implementer transcribes) deletes the documented bare `resolution="1080p"`/`fit="cover"` paths; no Verify distinguished them → the raise is now explicitly annotated-only. **M-4** `at=` on a static map was the **last surviving silent no-op** → raises. **M-5** `_lat_from_result` omitted the mandatory `_to_numpy`, so the overlay would raise `can't convert cuda:0 device type tensor to numpy` on any GPU result. **M-6** `frame_resolved` unbound on the layout path. **M-7** validating `fit` against `_LEGAL_FIT` *before* `_UNSET` resolution raises on every default call. **M-8** the out-of-range wording is now pinned so Step 2.4's needle matches conforming code. **M-9** the `_render_panels` guard had no owning step. **M-10** a bare `Image` in a list is promoted with a warning, `enforce_capabilities` skipped. **L-11** Steps 3.0/3.3 wrote PNGs into the checkout (conftest's redirect is pytest-only) → `CARDIAC_MEDIA_ROOT=$(mktemp -d)`. **L-12** the ±15 % gate's measured pairs recorded (height sits 1.6 pp from the limit). **L-13** the re-wrap now also replaces "a bare clip". **L-14** Phase 1 gained the `run_suite` re-paste reminder. **L-15** a `Video` in a list raises, naming `render([...])`. |
| 2026-07-25 | **R7** (1C/4H/7M/8L) — R7 **executed** R6's fixes and verified the consumer-binding spy, the multi-panel mirror, the `show_time` sweep, the `tight_resolved` sweep, 3 of 4 shell-gate paths and all 11 RESOLVED-SEMANTICS restatements as **correct** | **C-1** ⚑ **`conda run` silently discards STDIN**: Step 3.0's heredoc **never executed**, wrote a 0-byte refs file and **exited 0**, taking out two Phase-3 exit criteria and killing Step 3.3 with `KeyError`. Reproduced and confirmed independently; `--no-capture-output` fixes it. Promoted to a top-level gotcha because it applies to **implementation work**, not just this plan. **H-1** the hardened gate still passed vacuously on a **missing baseline** — `comm`'s error goes to stderr, its status is discarded, `>` still creates an empty file (and `/tmp` is reboot-volatile, and the baseline is captured in a *separate shell*) → `[ -s before ]` guard + re-paste reminders for the `run_suite` shell function in Phases 2/3/4. **H-2** there was **no seam to draw a `Trace` into a layout** although "mixed `[Image, Trace]` lays out" is an exit criterion — the only Trace renderer owned its own figure → split into `_draw_trace_on(spec, ax)` + a thin wrapper. **H-3** Step 2.4's introspection guard subtracted a set of **characters** from a set of field names, removed nothing and never asserted → assert against a module-level `run._TRACE_KEYS`. **H-4** the "DECIDED" `enforce_capabilities` re-wrap had **no implementation site** and no test → concrete `try/except` in the ordered sequence + a test. **M-1** both new R6 citations were wrong (`607-609` is the suptitle mirror, not `602-604`; `258,260` not `256,258`). **M-2** Step 3.1 had no figure-construction defaults. **M-3** "stamp from panel 0" is undefined when panel 0 is a `Trace` → first **map** panel. **M-4** the `multibeat` fixture still had no construction for `V` → the trapezoidal 4-beat build is now written out. **M-5** Step 2.4 grepped a message the plan never pinned for a `Trace`. **M-6** `i` was unbound and competed with `frame_resolved` → one name, bound first in the ordered sequence. **M-7** THE DEFAULT table said `tight=True` where the signature is `Optional[bool]`. **L-1** `_DEFAULT_WHAT` is a phantom. **L-2** `n_panels` defined on every path. **L-3** `set_aspect` stated for `_setup_panel`. **L-4** `Trace` honours a caller-passed `tight`. **L-5** the `isinstance` guard is what keeps direct `draw(Video…)` on the original wording. **L-7** the Step-3.2 A/B literals are seeded-data, not the fixture's. |
| 2026-07-25 | **R6** (2C/4H/7M/7L) — R6 re-verified every line citation, the census, the out-of-scope counts, the 17-name import surface, the artist counts, the `multibeat` gate and the viz A/B as **correct**; *"the factual layer remains excellent"* | **C-1** ⚑ **the R5 spy Verify is INERT**: the Architecture mandates a module-scope `from ..video.render import _build_figure`, which binds the original at import time, so patching `sys.modules['cardiac_core.video.render']` cannot reach it — reproduced (`spy fired? False`, then `KeyError`). → patch the **consumer's** binding (`cardiac_core.image._draw._build_figure`) + assert `seen` is non-empty. **C-2** the R5 M-5 multi-panel `_produce_figure` call **raises on every list draw**: `_setup_panel` returns `suptitle=None` and `_produce_figure` calls `st.suptitle.set_text` unguarded (verified `AttributeError`), the stated order is wrong, and per-panel calls would overwrite the shared stamp N times → mirror `render.py:607-609`, inline only the `front` block, set the stamp once, never call `_produce_figure` on the layout path. **H-1** the `tight` snippet used the RAW parameter (`Optional[bool] = None`) → `bbox_inches=None` on every default annotated draw, and the no-tight output (**720×360**) sits **inside** the plan's own ±15 % gate, so the check passes while the composition diverges → `tight_resolved`. **H-2** the `show_time` bullet **contradicted itself in consecutive sentences** and disagreed with the RANK RULE for time-varying `fields.*` → one formula keyed on a non-finite time, `what`-threading deleted. **H-3** the R5 gate prints "NO NEW FAILURES" when the suite never ran (`run_suite` returns 1 → `&&` short-circuits → `comm` never writes → the `if` reads a stale/missing file) → `rm -f` + fail hard. **H-4** Verify block 2 is a separate process and used undefined `os`/`d`. **M-1** `draw(<Video>)` default frame = `len(frames)//2`. **M-2** the `Video` gradient index must be `masked_iter([frame_resolved])` or pixel-identity breaks. **M-3** Phase 3 gained **Step 3.0**, which actually writes the pixel references two exit criteria depend on. **M-4** Phases 2 and 3 gained executable Verify blocks (Steps 2.4 / 3.3). **M-5** the stale "`clip.result` is unread" note. **M-6** `Trace` render defaults + `vmin/vmax=None` for a trace-only figure. **M-7** `labels`/`rows`/`cols` raise in Phase 1. **L-1** `_gradient_was_explicit` residue. **L-2** `ionic_states`/`boundary_mode` added to the Phase Context. **L-4** the `multibeat` fixture now VARIES APD per beat (a constant train gives two identical points). **L-6** `t0` comes from the clip, which a `Video` spec IS. **Audience wart DECIDED**: `draw()` catches `enforce_capabilities`' `ValueError` and re-raises it naming `style="annotated"` instead of `Video.annotated(...)`. |
| 2026-07-25 | **R5** (1C/5H/8M/10L) — R5 re-verified ~40 line citations, every measured number, the tree inventory, the `_LAZY`/self-contained lists, and confirmed R4's C-1 (`dpi` raw through the gate) and C-2 (±15 %) as **correct and reproduced**; *"the factual layer of this plan is in excellent shape"* | **⚑ Meta-fix: the TWO-LAYER PROBLEM.** R5's central finding is that R4 wrote RESOLVED SEMANTICS but **never swept the implementation steps**, so the normative prose and the executable text disagreed and the *wrong half sat in the position an implementer copies*. Three findings were that one omission. The sweep is now done: every `_lat`/`isochrones`/`filled`/`show_time`/`resolution`/`(T,Nx,Ny)` occurrence defers to or restates the section, and superseded forms are **deleted, not annotated**. **C-1** the ungated `lat=spec._lat` survived verbatim at Step 1.3 and Step 3.1 → both replaced. **H-1** the Verify meant to catch it computed `dlat` **in the test script** and asserted on it, exercising no production code (the same assertion-that-cannot-fail class R3 C-3 and R4 M-7 killed) → it now drives `draw()` and spies on what `_build_figure` actually receives. **H-2** Step 1.3 still carried the "**otherwise** compute a LAT" rule that rule 4 forbids → an APD map would ship with activation contours. **H-3** rule 1 was **boolean-wrong**: `(passed is True) or (what=="activation" and not filled)` makes an explicit `isochrones=False` resolve **True** — the section that declares itself the source of truth was wrong on the tri-state it exists to pin. **H-4** `resolution="auto"`/`fit="contain"` literals cannot express "the default never raises, an explicit non-default does" with two spec-type defaults — both readings broken (either `draw(Video.annotated(v))` raises, or `resolution="auto"` is the silent no-op H-4 forbade) → `_UNSET` sentinels. **H-5** rule 5's masking was prose-only and absent from the load-bearing `__post_init__` recipe → new item 5b + a test. **M-1** the `[ -s … ] && {…}` gate returns exit 1 on **both** branches → `if/then/else`. **M-2** the `show_time=True` raise was placed in `Image.__post_init__`, which cannot see a `draw()` parameter (R3 M-10's own failure mode) → moved to Step 1.3, and the rule now keys on a **non-finite time**, not the selector. **M-3** the stale `(T,Nx,Ny)` annotation. **M-4** the bare-array input had **no selector row** → undefined gradient (a non-voltage array on a −90…40 mV scale) and a `"t = nan ms"` caption. **M-5** `front=` and the stamp are silently dropped on the multi-panel path (`_setup_panel` draws neither). **M-7** the delegation lost its PNG guarantee → a JPEG served as `data:image/png`. **M-8** the `multibeat` pointer named a helper that is neither a `SimulationResult` nor multi-beat, with no anti-vacuity target → concrete spec + `numel() >= 2` gate. **L-1** 17 names, not 16. **L-3** the golden gate is near-vacuous by construction — kept, but not read as evidence. **L-4** inherited `enforce_capabilities` messages name `Video` to an audience that never typed it. **L-5** `plt.close` on the `Trace` path. **L-7** `show_time` for a `Video` spec. **L-8** preserve `preview_frame`'s `IndexError`. **L-9/L-10** prose corrections + the real-fixture ±15 % numbers. |
