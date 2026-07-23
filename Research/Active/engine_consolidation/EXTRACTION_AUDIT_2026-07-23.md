# Extraction audit — `cardiac_core` as a standalone public library (2026-07-23)

> Pre-publication audit run before extracting `cardiac_core/` from the Heart-Conduction monorepo into
> its own public GitHub repository. Four independent adversarial lanes (monorepo coupling · packaging ·
> test portability · public-exposure) plus direct verification of every load-bearing finding.
>
> **VERDICT: NOT READY TO PUSH.** Nothing here is hard, but three items must be settled by a human —
> one of them concerns a third party who cannot be asked by proxy.

## Git state (checked first — the user's premise was correct)

`HEAD`, `stim-object` and `origin/main` are all at `9ef97f3`. Both parallel agents' work is merged and
pushed: video pipeline `2a4b0e3`, Stim work behind it. Nothing ahead, nothing behind. The only
uncommitted paths under `cardiac_core/` are the tutorial lesson (this session) and another session's
untracked `IONIC_PRESET_PLAN.md`. **The library code itself is fully committed on `main`.**

## What is already good

- **The package genuinely stands alone at the module level.** Built the wheel, installed it into a
  fresh venv, imported from `/` so the monorepo could not be on `sys.path`: monodomain runs,
  `single_cell` returns a physiological 217 ms APD90, LBM and bidomain construct. 450 KB, pure
  Python, `py3-none-any`.
- **No `sys.path` hacks, no imports outside the package** in library code. There is even an active
  guard against regressions: `tests/test_self_contained.py:17`. Keep it.
- **No third-party code is vendored.** `_monodomain`/`_bidomain`/`_lbm` are all the author's own code
  copied from sibling monorepo folders. Ionic models are original implementations from published
  equations with primary citations. **No licensing entanglement.**
- **No secrets.** No keys, tokens, credentials, `.env`, emails, IPs, or URLs anywhere in the package.
  The `cardiac_mcp` signing gate has no key material to leak (`sha256(json)[:16]`, no secret).
- **The suite is green today**: **482 passed, 2 xfailed, 0 failed, 0 errors** in 8m26s (the 2 xfails are
  the intentional permanent-deferred oblique cells in the contract matrix). No test needs a GPU — four
  are `skipif(not cuda)` and skip cleanly. That is the baseline any extraction must preserve.
- **Nothing needs to be carried over from the monorepo test config.** Root `conftest.py` only prepends
  `Surrogate/` to `sys.path` and no cardiac_core test imports it — proven empirically by running tests
  from `/tmp` with a different rootdir (18 passed). There is **no** `[tool.pytest.ini_options]`,
  `pytest.ini`, `setup.cfg` or `tox.ini` anywhere: the suite runs on stock pytest defaults.
- **The integrity goldens travel.** `tests/_integrity/golden_*.pt` + `engine_src_sha.json` are all
  git-tracked, so a git-based extraction carries them.
- **History is extractable.** `git subtree split -P cardiac_core` yields 61 commits. **But see L1 —
  the split's default layout is wrong and must be corrected.**
- **`cardiac-core` is available on PyPI** (also `cardiac_core`, `cardiaccore`).
- **No tracked junk** — 226 tracked files, zero `__pycache__`/`.pyc`.

## BLOCKERS — human decision required

### H1. A collaborator is named in shipped solver code ⚑ NOT THE AUTHOR'S CALL ALONE
`_monodomain/simulation/classical/discretization_scheme/fdm.py:322,329,433` — "the failure we hit on
**John's tanks**", "this is the **John-equivalent**", "Faithful **John-equivalent**". Verified in
source. This names a real collaborator without context and identifies his **unpublished** discrete-
reduction tank model as the validation reference for a shipped numerical stencil.
**Fix**: rewrite as "the discrete-neighbour-count reference model" (3 lines). Must happen before the
repo exists publicly — a rename after the fact does not un-publish it.

### H2. Unpublished research is productized in public docstrings
`_lbm/boundary/wall_modes.py:1-19` opens "boundary_conduction_speedup research → productized" and
states the findings outright: "ZERO bias", "INVERSE crescent", "the β-controlled curvature knob".
Same exposure at `analysis.py:582-586` (eikonal source-sink metrics → `source_sink_mismatch_investigation`).
**This is the author's own work — but publishing precedes the papers.** Decide deliberately.

### H3. License choice
There is **no LICENSE file anywhere in the repo** (verified). The existing public monorepo is therefore
already published under default all-rights-reserved, so nobody may legally use any of it. Needs an
explicit choice (MIT / BSD-3 / Apache-2.0 are the academic-simulation norms), and Cornell IP policy
likely governs.

## BLOCKERS — mechanical

### M1. `media.py` writes into `site-packages` when installed
`media.py:24` `_REPO_ROOT = dirname(dirname(abspath(__file__)))` → `media.py:77` `root = root or _REPO_ROOT`.
**Reproduced against the installed wheel**: resolves to `…/site-packages`, so a default
`r.video("x")` creates `site-packages/media/lab/_sim_outputs/videos/…` — wrong location, and a
`PermissionError` on any read-only install. Inherited by `run.py:135` (`SimulationResult.video()`),
`video/render.py:253,325,466`, and `viz.py:74,94` — **and the three `viz.py` functions expose no
`root=` escape hatch at all**.

**Fix — backward-compatible, because 50 monorepo files depend on today's behavior** (Monodomain 23,
Research 22, Optimizer 2, Lab 2, cardiac_mcp 1). A plain "default to cwd" would silently relocate all
of their output. Use instead:
```
root arg  →  $CARDIAC_MEDIA_ROOT  →  walk up from cwd for a .git dir  →  cwd
```
The `.git` walk reproduces the current repo-root behavior exactly for every in-repo caller while
behaving sanely when pip-installed. Add `root=` passthrough to the three `viz.py` functions.

### M2. Declared dependencies are 100% wrong
`pyproject.toml:9` declares exactly `mcp>=1.2.0` — a package `cardiac_core` **never imports** (it
belongs to the sibling `cardiac_mcp`) — and omits every real one. Verified by AST scan + import trace.

Core (all verified as bare imports with **no** try/except fallback):
| dep | why core |
|---|---|
| `torch` | 132 sites, 131 module-level |
| `numpy` | 45 sites |
| `torch-dct` | `_bidomain/.../linear_solver/spectral.py:24` module-level; `bidomain()` **auto-selects** the spectral solver on the default isotropic-Neumann rectangle → **`bidomain()` fails out of the box without it** |
| `scipy` | `geometry.py:189` bare `from scipy.ndimage import …` inside the exported `boundary_distance()` |
| `scikit-image` | `fields/integrals.py:122` bare `from skimage.measure import …` inside `isochrone()` / `wavefront_length()` |

Optional extra `viz`: `matplotlib`, `pillow`, `imageio`, `imageio-ffmpeg`. Do **not** declare `cv2`
(60 MB wheel for a path that already degrades gracefully). Extra `test`: `pytest`.

⚠ **Supply-chain note**: `torch-dct` is single-maintainer, last released 2020. The package already
hand-rolls DST-I via FFT in the same file; vendoring the ~60-line DCT would remove the dependency.

### L1. ⚑ THE EXTRACTION LAYOUT IS LOAD-BEARING — do not flatten
`tests/test_integrity.py:21` does `from cardiac_core.tests._integrity.make_goldens import …`, and
`tests/__init__.py` exists, so **`tests/` must remain a subpackage of `cardiac_core`**.
- **Correct**: repo root `cardiac-core/` containing `cardiac_core/` (the package) + `pyproject.toml`.
- **Broken**: flattening the package contents to the repo root — which is exactly what
  `git subtree split -P cardiac_core` produces by default. That kills the import above AND makes the
  test-suite media writes (L3) escape the checkout entirely into its parent directory.
So the split must be followed by a re-nest commit **before** anything else is evaluated.

### M3. The test suite fails on a fresh clone
`tests/test_integrity.py::test_originals_untouched` (with `_integrity/make_goldens.py:30-34`) hashes
`Monodomain/Engine_V5.5/cardiac_sim`, `Bidomain/Engine_V1/cardiac_sim`, `LBM/Engine_V1/src` — monorepo
siblings absent from a standalone repo. It does **not** skip: `tree_hash` of a missing tree returns the
empty-input digest (`e3b0c442…`) and the assertion fails.
**Consequence**: the first thing anyone cloning the library sees is a red suite.
**Better fix than skipping** (revised after the test lane): **delete** `test_originals_untouched` plus
`ENGINE_SRC`/`tree_hash`/`save_hashes`/`engine_src_sha.json`. It is a monorepo-only "don't edit the
originals" guard that has no meaning once there are no originals. The other three integrity tests stay
green — they need only the `.pt` goldens.

### L2. `test_live_cv_gate` — 128 s, and points outside the repo
`tests/test_conductivity.py:21-23` + `tests/_live_cv_gate_driver.py:17-19,30` import
`test_phase10_cm_scaling` from `Monodomain/Engine_V5.5` — the only import in the package of a module
that isn't installable. It degrades to `skip` after extraction (so not a red suite), but becomes dead
code resolving *above* the repo root, and it is **25% of total suite runtime**.
**Fix**: delete it and `_live_cv_gate_driver.py`, and drop that name from `_EXCLUDE` at
`test_self_contained.py:26`. (Stronger option: vendor the 323-byte `bidomain_cm_ref.json` and re-derive
CV through the vendored `cardiac_core._monodomain`, which *is* V5.5 — buys back ~2 min of runtime.)

### L3. The tests write ~60 media files outside the package
Same `media.py:24` root as M1. Today those land in repo-root `media/lab/_sim_outputs/` (**22 MB already
accumulated**). Writers: `test_viz.py:26,33,38`, ~60 `render()`/`.video()`/`.preview()` calls in
`test_video.py`, and `test_video.py:820` which `exec`s the cheatsheet's `# runnable-video-section`.
**No test passes `root=`**, and the three `viz.py` entry points don't accept one.
**Fix**: M1's `CARDIAC_MEDIA_ROOT` support plus a new `cardiac_core/tests/conftest.py` with an autouse
session fixture pointing it at `tmp_path_factory`. Without this the standalone suite pollutes the
checkout (or its parent — see L1).

### L4. Two CI hazards to decide before wiring public CI
- **Golden bit-identity is hardware/BLAS/torch-version sensitive.** `test_integrity.py` asserts
  `torch.equal` (atol=0). On public CI hardware these three are the most likely spurious reds. Pin torch
  in CI, document regeneration via `make_goldens.py`, or loosen to a tight `allclose`.
- **`test_video.py` has no `importorskip` guards** — absent `imageio`/`imageio_ffmpeg`/`PIL`/
  `matplotlib` produces *collection errors*, not skips. Needs guards if `viz` is an optional extra.

### M4. Packaging metadata is absent, and the wheel drops files it needs
Wheel METADATA carries only Name/Version/Summary/Requires-Python. Missing: license, authors, readme,
classifiers, URLs, keywords, `py.typed` (84% of 1121 non-test functions are annotated — type-checkers
see none of it), `__version__`.
Non-`.py` files are silently dropped, including ones read at runtime:
`API_CHEATSHEET.md` (read by `cardiac_mcp/core.py:32,551` and two tests) and
`tests/_integrity/golden_*.pt` + `engine_src_sha.json`. Needs `[tool.setuptools.package-data]`.
Also `namespaces` defaults to **true** under `packages.find`, which is why the current wheel silently
ships `tutorials/_build/build_01_build_a_simulation.py`; set `namespaces = false` + an explicit exclude.

### M5. `cardiac_mcp` must be cut out of the distribution
`pyproject.toml:12-13,21` bundles `cardiac_mcp*` and installs a `cardiac-mcp` console script. Extracting
only `cardiac_core` leaves **a broken executable on every user's PATH**. Delete both. Then
`cardiac_mcp/core.py:32` (`REPO_ROOT/cardiac_core/API_CHEATSHEET.md`) must switch to
`importlib.resources.files("cardiac_core")`, which depends on M4 shipping the cheatsheet.

## CLEANUP — should do, not strictly blocking

- **~360 KB of internal planning docs ship as top-level package files** — more bytes than the library.
  Verdicts: `VIDEO_OBJECT_PLAN.md` (12 home paths, 20 machine conda paths), `STIM_OBJECT_PLAN.md`,
  `ANALYSIS_FIELDS_PLAN.md`, `IONIC_PRESET_PLAN.md` (plans unshipped work), `ANALYSIS_FIELDS_DESIGN.md`
  (contains a **stale** "silent, undocumented discrepancy" claim about `r.cv()` that was since fixed) →
  **MOVE OUT** to `Research/Active/engine_consolidation/plans/`.
  `ANALYSIS_METHODS_PRIOR_ART.md` and `ANALYSIS_FIELDS_DATA_MODEL.md` → **REDACT → SHIP** as `docs/`;
  they are the two genuinely publishable documents. `API_CHEATSHEET.md` → **SHIP** (strip the `/sim-*`
  skill and `Lab/_validate/smoke.py` references). `engines_SOURCE.md` → rewrite as a short provenance note.
- `tests/test_api_contract.py:24` — hardcoded `/home/norepinephrine/.conda/envs/…` in a docstring.
- `tutorials/README.md:18,26` instructs readers to `conda activate heart-conduction` — an env named
  after a private project; should be generic.
- ~60 dangling doc references in shipped `.py` (`improvement.md` ×42, `Research/0X_*`, `CLAUDE.md`,
  `IDEALOG.md`) make the package read as a monorepo excerpt.
- `_bidomain/utils/backend.py:8` + `_monodomain/utils/backend.py:8` show `from utils import …` in a
  docstring "Usage:" block — a V5.3 leftover that will confuse a standalone reader.
- `question="lab"` default in `viz.py:25,62,80` and `video/render.py:187,305` — a monorepo `Lab/` notion.
- `.gitignore` covers only `__pycache__/` and `*.pyc`; a standalone repo needs `build/`, `dist/`,
  `*.egg-info/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `.venv/`, `.DS_Store`.
- `torch.load(..., weights_only=False)` on the shipped `.pt` goldens (`tests/test_integrity.py:31`) —
  pickle-loading in a public repo invites a security question; one-line hardening.
- CellML/PMR provenance for `ionic/paci/parameters.py:167`, `ionic/phas13/parameters.py:167`,
  `ionic/ttp06/parameters.py:227-228` deserves an explicit attribution note (those exposures are
  commonly CC-BY). `ionic/base.py:5` "Based on openCARP's LIMPET pattern" → phrase as "inspired by".

## Deliberately KEEP

`_bidomain/.../explicit_rkc.py:25-40` — the 16-line `KNOWN LIMITATION` block documenting a ~0.8%
steady-state error that does not shrink under `dt` refinement. It is precise, explains the correct fix,
and notes the scheme is opt-in and unreachable from the public API. **Reads as rigor, not embarrassment.**

## Recommended order

1. **H1** de-name the collaborator (blocking, and cannot be undone after publication).
2. **H2/H3** author decisions: research disclosure, license.
3. `git subtree split` → **re-nest under `cardiac_core/` (L1 — non-negotiable)** → new repo, still local.
4. **M1 + L3** media root (`root=` → `$CARDIAC_MEDIA_ROOT` → `.git` walk → cwd) + a tests conftest fixture.
5. **M3** delete the originals-untouched guard; **L2** delete the live-CV gate.
6. **M2/M4/M5** pyproject (real deps, extras, package-data, `namespaces=false`, drop `cardiac_mcp` +
   the console script) + LICENSE + README + `py.typed` + `.gitignore`.
7. Cleanup pass (docs move-out, paths, provenance, `question="lab"` default).
8. Verify: fresh venv, wheel install, **full suite green from a neutral cwd** (baseline: 482 passed /
   2 xfailed), tutorial notebook runs.
9. **L4** decide the CI hazards, then create the remote and push.

## Baseline to preserve
`482 passed, 2 xfailed` in 8m26s. Any extraction that does not reproduce this has lost something.
