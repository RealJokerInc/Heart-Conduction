# Engine Consolidation — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
**IMAGE LAYER — SHIPPED 2026-07-25** (4 commits `fba6ed3`→`7ba147f` on `video-portable-output`).
`cardiac_core/image/` gives stills the shape the video layer proved: `Image`/`Trace` specs, `draw()`,
`ImageInfo`, `Gradient` reused. `r.image()` / `r.trace()` are the one-liners; drawing displays, naming a
destination saves. **580 passed / 7 failed** (pre-existing CUDA OOM), **+87 tests**, integrity goldens
bit-identical at all four phase gates. Plan audited to convergence over **12 Opus rounds** first:
[plans/2026-07-25_IMAGE_OBJECT_PLAN.md](./plans/2026-07-25_IMAGE_OBJECT_PLAN.md).

The layer exists because the corpus census says `plot(` 163 vs `imshow(` 70 — the line plot is what this
project's figures actually are, and there was no route to one. The 0-D action potential now has one.

## Next Step
**★ PRIORITY (2026-07-23, user) — `single_cell(conductances={...})`, the 0-D drug knob.** Full spec in
KNOWLEDGE.md § "SPEC — `single_cell(conductances=...)`". One-line summary: tissue can apply a drug
(`scale_conductance`), 0-D cannot — so the CHEAPEST drug question has no public route, and the only
workaround (mutate a model instance) **bypasses name validation**, making a mis-cased conductance a
SILENT no-op in a drug experiment. Fix is small: `api.py:43::_scale_ionic_conductances` already takes a
dict and validates names; move it to a light home (`cardiac_core/ionic/scaling.py` — `single_cell.py`
must stay free of the heavy `api` import) and add one keyword, applied BEFORE `pre_pace`. ~5 tests,
no solver code, goldens unaffected. NOT built.

**Backlog logged (2026-07-22, post-ship):**
- **✗ `Grid` dx unit convention cm → mm — CONSIDERED AND REJECTED (user, 2026-07-23). `cm` STAYS CANONICAL.**
  Proposal was: `Grid(Nx, Ny)` primary, `dx` demoted to an optional tissue-size knob in **mm** (default 0.1 mm,
  chosen over the initially-floated 1 mm because the upstroke is only ~0.5–1 mm wide, so dx=1 mm puts it on ~1 node →
  grid-dominated WRONG CV — the ionic-tuner's phantom "conduction block" failure). Blueprinted + adversarially
  audited (19 findings: 4 critical / 5 high), then **scrapped: it is not the "small fix" it looked like.**
  **Why rejected — the real blast radius is 102 executable `Grid(` sites across FIVE subsystems**, not the ~80
  cardiac_core sites first estimated: 18 cardiac_core test files + `protocols.py` + the tutorial notebook & its
  `_build` script; **`cardiac_mcp/core.py` ×2 — including the public `simulate(dx=…)` tool parameter AND the
  generated-`run.py` template string**, so every newly committed Lab experiment would ship a script that raises;
  `.claude/skills/sim-experiment/reference/run-template.py`; 3 `Lab/` scripts + `Lab/presets/*.yaml` + `_SCHEMA.md`;
  plus the cheatsheet/API_REFERENCE/skill docs. Renaming the unit into every boundary variable (`DX`→`DX_MM`, the
  MCP tool param, preset keys) was required to stop a cm value being pasted into `dx_mm=` (which raises nothing and
  silently builds a 10×-too-fine grid). Not worth it for a unit preference.
  **⚑ Two durable findings worth keeping (independent of this decision):**
  1. **The integrity goldens are structurally blind to the `Grid` path.** `tests/_integrity/make_goldens.py` builds
     all three goldens via `create_cardiac_mesh(Lx, Ly, dx)` — **no golden ever constructs a `Grid`**. So the
     declarative `Grid` construction path has NO numerics drift-guard at all. Separately actionable.
  2. **Census-grep trap** (it hid 5 real call sites): use `\bGrid\(` — `[^a-zA-Z_.]Grid\(` silently excludes every
     `cc.Grid(` dotted call, and `grep -v "A|B|C"` is a BRE so the `|` is literal and filters nothing (use `-vE`).
  Current convention stands: cardiac_core length is **cm** everywhere (`Grid(Nx, Ny, dx_cm)`, `Lx = dx*(Nx-1)`),
  time ms, Vm mV, D cm²/ms, σ mS/cm, χ cm⁻¹, CV cm/s; the Optimizer alone speaks mm at its edge and divides by 10.
- **`IonicPreset` — a savable ionic-model config object. PLAN WRITTEN + GATED (2026-07-23, user: "worry about it
  later").** Spec: `cardiac_core/IONIC_PRESET_PLAN.md` (1 phase / 3 steps, tier large). A first-class object: base
  model + a `{param: factor}` scaling map, accepted anywhere `ionic_model=` is, with `.save()`/`.load()` JSON — closes
  the "a tuned conductance set has no home" gap (`scale_conductance` mutates in-memory; `.npz` stores only the model
  NAME string; `set_parameter` is stubbed). Locked design (user, 2026-07-23): scalings canonical + resolved `.values`
  (**BOTH**); any named param — conductance/concentration/kinetic — validate-exists + warn-denylist (**BREADTH**);
  **CORE OBJECT ONLY** — `.npz` scalings-persistence and the tuner bridge (`g_Na`↔`GNa`, `from_tuning_result`)
  DEFERRED. Resolves at the SINGLE `ionic/registry.py::build_ionic_model` seam (all 3 engines; already passes scaled
  instances through). Design NOT audited (offered `/audit`, user shelved). Related: the ★ `single_cell(conductances=)`
  priority above (both are the ionic drug/conductance knob — an `IonicPreset` could be the savable form of that knob)
  + the gated [[project_ionic_tuner_redesign]].
- **Tutorial notebook series — DESIGN CONVERGED 2026-07-22, AUTHORING GATED.** 11 lessons on a **lab-experiment
  spine**, two tiers (Core 01–06 = one cell → drug → CV → video → pacing → scar+block, an afternoon; Advanced
  07–11 = fibers → clamp → `fields` → two engines → bidomain capstone). Caveats **minimal/operational only**
  (user call). Spec: `cardiac_core/tutorials/PLAN.md` (rewritten — the 2026-07-21 8-lesson version is
  superseded: its `single_cell` prep step is already shipped and its dict-stimulus spine is now deprecated).
  **Gate: author nothing until the video pipeline lands** (L04) **and Stim merges to `main`**. See the
  2026-07-22 (3rd parallel agent) Thread entry.
- **API_REFERENCE.md ("Object Atlas") generator + drift canary.** An introspection-driven reference (full
  `inspect.signature` + first-docstring-line over the `_LAZY` export map, + the object-atlas shape tables from a
  tiny fixture run) → a NEW `cardiac_core/gen_api_reference.py` emitting `cardiac_core/API_REFERENCE.md`, kept honest
  by a `test_api_reference.py` drift canary (regenerate-into-string == committed doc; mirrors
  `test_cheatsheet_examples_execute`). Pairs with the recipe-oriented `API_CHEATSHEET.md` (verbs) as the
  reference-oriented atlas (nouns). GATED — approved as a task, not yet built.
- **Stim-as-object — SHIPPED Phases 1-2 (2026-07-22); Phase 3 DEFERRED (user).** Branch `stim-object`, commits
  `c087b8c` (P1: the `Stim` object + presets + current/clamp modes + native additive flux-preserving LBM clamp +
  dict coexistence; 24 tests, integrity atol=0) + `743e6d4` (P2: Stim canonical — internals build Stims, dict path
  soft-deprecated with a `DeprecationWarning`, cheatsheet + 11 test files migrated; full suite 395 pass/0 fail).
  Phase 3 (migrate live consumers Surrogate×5/Optimizer×2/mcp/Lab×3 off dicts) DEFERRED — optional, non-blocking,
  one-PR-per-consumer, gate on each consumer's suite; the dict path keeps working. Branch NOT yet merged to main.
- **Stim-as-object — BLUEPRINTED + AUDITED-TO-CONVERGENCE (2026-07-22).** `cardiac_core/STIM_OBJECT_PLAN.md` (3 phases:
  Stim object+presets+coexistence+clamp → steer cardiac_core → per-consumer migration). **`/audit` 3 Opus rounds →
  CONVERGED** (R1 1blk/6maj → R2 1blk/3maj → R3 0blk/0maj, all code-verified). Real catches (all fixed): R1 — a clamp
  Stim would be silently injected as a −52 current (the single-`_normalize_stimulus`-seam thesis fails for clamp →
  needs a factory-level `_partition_stimulus` split + post-build `clamp_voltage`); R2 — that fix's tail: periodic clamp
  isn't expressible (`add_clamp_protocol` sig mismatch → reject bcl/num_pulses on a clamp Stim), LBM clamp mask
  numpy-vs-torch CUDA crash, LBM clamp dropped on `reset()` (store on the wrapper, re-push), `self.V` re-sync, timing
  after `self.t+=dt`; R3 — 5 minors (`_resolve_where` must delegate to geometry.py not a 3rd mask system; overlap-sum
  is mono/bidomain-only, LBM OVERWRITES). Implementation GATED (hard gate). Design summary below.
- **Stim-as-object (DESIGN LOCKED 2026-07-22).** Promote stimulus to a public mask-first `Stim` object.
  **Locked design decisions (user):**
  - **Mask primary + callable convenience** — `Stim(mask=(Nx,Ny)bool, amplitude, start_time, duration, bcl,
    num_pulses, label)`; a `region=lambda` is accepted and resolved to the mask at build (stored as the mask).
  - **One fixed mask per Stim** — moving/multi-site = a list of `Stim`s (overlaps sum, as `StimulusProtocol` already does).
  - **COEXIST, non-breaking → Stim is the final canonical form.** Factories accept `Stim | dict | list[either]`;
    the dict path KEEPS WORKING (soft-deprecated), the blueprint steers cardiac_core + consumers toward `Stim` as
    the documented product. NOT a big-bang rip (the dict form reaches ~19 cardiac_core tests + Surrogate datagen ×5
    + Optimizer tuner ×2 + cardiac_mcp + Lab ×3 + the `.npz`/`CardiacMeshData.stimuli` serialization — a same-PR
    removal would break live cross-project consumers, the pattern consolidation always defers). Depth = front-door:
    `Stim ⇄ dict` lowering keeps the `.npz` format stable.
  - **Location — EAGER-ONLY classmethod factory constructors (FINAL 2026-07-22).** Primary API =
    `Stim.boundary(grid, side, bcl=…, num_pulses=…, amplitude=…, width=…)` (a FULL constructor — grid + side + any
    timing params); plus `Stim.point(grid, (x,y), radius=…, **kw)`, `Stim.center(grid, **kw)`,
    `Stim.from_region(grid, callable, **kw)`; base `Stim(mask, **kw)` for an explicit mask. `side ∈
    {left,right,top,bottom}` is the sole edge API (no `*_edge`). **NOT subclasses** — one `Stim` type (the
    datetime.fromtimestamp pattern); a resolved boundary vs point differ only in their mask, so a type hierarchy adds
    nothing (user asked; confirmed no subclass). **Deferred/grid-free path SCRAPPED** — its only benefit was not
    re-typing the grid; eager is self-contained, inspectable, serializable, validated-early. Each classmethod builds
    the concrete mask via `_resolve_where(grid, where, width, radius)` (side rule `x≤x.min()+w` etc.; distance for
    point/center; `width/radius=None`→thin strip ~2·dx) → returns `cls(mask, **kw)`. A Stim always has a mask, so
    `_normalize_stimulus` just `to_dict()`s it (no coords/`.on`/`.resolve`); serialization/engines unchanged. Plan:
    cardiac_core/STIM_OBJECT_PLAN.md. **Design churn note:** iterated grid-classmethods → deferred → eager → both →
    **eager-only classmethod factories** (settled).
  - **TWO MODES on Stim (user, 2026-07-22): current injection + voltage clamp.** `clamp=<mV>` ⇒ voltage clamp, else
    current injection (`amplitude` µA/µF). Current-mode lowers to the stimulus protocol (`data.stimuli`/`Istim`);
    clamp-mode routes to the clamp mechanism. **Clamp now on ALL THREE engines**: mono/bidomain via the existing
    `_clamp_mask`/`_stepping_run` (hard-write `v[mask]=value`, api.py:720 — NOT Istim, user confirmed the intuition);
    **LBM via a NEW native clamp — ADDITIVE, non-equilibrium-preserving.** Since `V=Σf_i` and `f_i=w_i·V + f^neq`
    (the flux lives in `f^neq`), the clamp is per-step `V=Σf; f[:,mask]+=w·(value−V)` (forces Σf=value EXACTLY while
    PRESERVING `f^neq`). **NOT** `f=w·value` (a pure equilibrium reset that ZEROS the local flux — crude/low-order),
    **NOT** multiplicative rescale (V is signed, blows up / sign-flips at V≈0). Guo-style non-eq-preserving Dirichlet;
    reaction runs BEFORE the clamp (gates integrate). **User caught this** ("V=Σf_i; normalize the injection w.r.t. the
    current distribution configuration") — a real improvement over the initial reset idea. Opt-in → no-clamp LBM run
    byte-identical → integrity goldens atol=0. Step 1.4 + cross-engine tests (`test_lbm_clamp_matches_mono`,
    `..._preserves_nonequilibrium`).
    **Reasoning conclusion (why B not A):** a voltage clamp pins VALUE not FLUX — current flows THROUGH the node
    (in≠out, electrode supplies the residual). The flux lives in `f^neq`; B preserves it (conducts through, on-node,
    O(h²)), A zeros it (isotropic reservoir, insulating flat point, O(h) value slip — the user's "funky" intuition).
    **Arbiter = the mono/bidomain hard-write clamp** (ground truth: pins V on-node, Laplacian gives in≠out flux → it
    ALSO conducts through), so B is the one consistent with mono; A deviates. Decision: SHIP B; the
    `test_lbm_clamp_matches_mono` test computes BOTH A and B vs mono to conclusively retire A (a one-time comparison,
    NOT a shipped toggle). Anchored on the mono cross-check, not pure LBM-BC theory (Zou-He equilibrium vs Guo
    non-equilibrium extrapolation) — reasoning + cross-check agree.
  - Wins: serializable/inspectable/composable/visualizable stim (vs opaque lambdas) — matters for save/load + the MCP
    accountability path. The internal `Stimulus`/`StimulusProtocol` (stimulus/protocol.py) already supports a
    mask-region + amplitude-summing overlaps → `Stim` lowers onto it.
- **Video-as-object — ✅ SHIPPED 2026-07-23** (design locked 2026-07-22 · plan audited to convergence · all 3
  phases implemented, `cardiac_core/video/`). `Video` + `Gradient` + `render` + `VideoInfo`, exported top-level;
  `r.video("slug")` is the one-liner; `viz.propagation_video` delegates with its 600×300 annotated framing intact.
  85 video tests; full suite green vs a pre-implementation baseline of 395 passed / 2 xfailed / **0 failures**.
  Implementation-time findings worth keeping: (1) **`labels=` was mutating the caller's `Video` objects** — a
  render-time override that persisted, and would then make a bare single-clip render of the same object RAISE
  (label is figure-only); found by self-review, not by the tests, now fixed + regression-tested. (2) The
  orientation probe test failed at `atol=8` purely from **yuv420p chroma subsampling** (H.264 is lossy) — the
  tolerance is now sized for the codec with a companion assertion that opposite corners differ by >60 levels, so
  it still catches a flip. (3) A `D=0` scar correctly renders as *non-conducting tissue at rest*, not masked grey
  — grey is reserved for `domain_mask` holes; verified by rendering a real scar sim and looking at the frame.
  **Below is the plan/audit record.**
  **`/audit` cycle: 6 Opus rounds → CONVERGED** (R1 3C/11H/11M/8L → R2 3C/10H/14M/14L → R3 1C/6H/14M/12L →
  R4 0C/4H/19M/9L → R5 0C/3H/11M/17L → **R6 0C/0H** — verdict: *"CONVERGED and ready for implementation"*; R6's
  mediums/lows folded in as a final non-adversarial pass). **Pattern, every round: the majority of findings were
  follow-ons to the PREVIOUS round's own fixes**, never new bug classes — R5 localised the whole tail to one
  block (`render()`'s pseudocode) and rewriting it as an explicitly ordered sequence closed all three of its
  HIGHs at once. **⚑ THE CYCLE CAUGHT TWO FALSE "VERIFIED" CLAIMS OF MINE** (see the corrected defect note below
  and [[feedback-verify-env-with-conda-run]]): the ffmpeg-absence claim, and "mutating a registered colormap
  contaminates the global" (it does not — `plt.get_cmap` returns a fresh copy; the real hazard is a
  caller-supplied `Colormap` instance). Both came from inferring a conclusion from a weaker observation.
  Load-bearing catches beyond those: **LBM masked nodes stay FINITE**, so `isfinite`-only masking would have
  painted every obstacle as live tissue (masking now routes through `domain_mask`, True=ACTIVE) — and the same
  contamination had to be propagated to the colour-range and isochrone paths; **no torch→numpy conversion
  existed anywhere** (crash on any CUDA result); the `Video.field` attribute shadows `dataclasses.field`, which
  would have `TypeError`ed at import; and every `Verify` block was unexecutable (`conda activate` is a silent
  no-op non-interactively). Implementation remains **GATED on an explicit user go**. NOTE: `cardiac_core/tutorials/PLAN.md`
  is explicitly **blocked on this plan** (its lesson 04 names the file), so this sits on that plan's critical path. A built-in video
  renderer, designed in parallel with (and deliberately mirroring) the `Stim` object shape: a **spec object holds the
  description**, a **render function** turns it into frames, output lands at a `media_path` convention path.
  **Locked (user):** (1) **full gradient control** — a reusable `Gradient` object (cmap by name / **a list of colors** →
  custom LinearSegmentedColormap / a Colormap; `range` presets; `gamma` PowerNorm; `levels` banding; `bad`;
  `interpolation`), with the FIVE color intents found in the render corpus shipped as classmethod presets
  (`physiological` −90…40 viridis · `rest_anchored` V_rest…40 inferno · `zoom` V_rest−0.3…+8 magma · `diverging`
  RdBu_r · `autoscale`). Color range is a SCIENTIFIC choice here, not cosmetic — `render_audit_video.py` exists to show
  a **7.48 mV** artifact that the default −90…40 scale renders invisible. (2) **built-in streaming render** (writes as
  it goes — 600 frames ≈ 570 MB if accumulated, so never accumulate) + a cheap single-frame `preview()`.
  (3) **multi-panel native** — a panel IS a `Video`, so `render([a,b], slug)` shares ONE colorbar + ONE suptitle time
  stamp (this is what most polished prior art already is: specular-vs-HBB, 3-BC oblique, 4 boundary modes, 5-panel
  axis sweep). Overlays: **live −40 mV front contour** + **static LAT isochrones** IN, **time stamp/colorbar
  optional**, **geometry outline DEFERRED** (needs caller-supplied analytic geometry). Returns a path-like `VideoInfo`
  reporting the encoder, so a fallback is never silent.
  **⚑ DEFECT (CORRECTED by the R1 audit — read this, an earlier version of this entry was WRONG):**
  `viz.propagation_video` calls `anim.save(writer="ffmpeg")` inside a bare `except Exception` that silently rewrites
  the output to a **GIF at a different path+extension** (`images/{slug}-propagation.gif`). The real defect is a
  **PATH-dependent SILENT FORMAT DOWNGRADE**: a caller who asks for `.mp4` can receive `.gif` with no warning, and the
  bare except also swallows codec/disk/permission errors identically.
  **⚑ MY EARLIER CLAIM — "there is no ffmpeg on PATH, so every mp4 this API ever produced was silently a GIF" — WAS
  FALSE.** It was an artifact of the Bash tool's **non-activated** shell. Under the documented
  `conda activate heart-conduction` workflow, `which ffmpeg` →
  `/home/norepinephrine/.conda/envs/heart-conduction/bin/ffmpeg` and `animation.writers.list()` →
  `['ffmpeg','ffmpeg_file','html','pillow']`, i.e. **`propagation_video` really does write H.264 mp4 in the user's
  actual environment** (the auditor ffprobe'd a genuine h264/yuv420p file written by `test_viz.py`). MEMORY.md already
  warned that conda is not on the non-interactive shell PATH — I measured in the wrong shell and generalized.
  **Lesson: verify environment-dependent claims under `conda activate`, not in the raw tool shell.**
  Fix direction is unchanged but for a smaller reason: make the fallback LOUD, and prefer the bundled
  `imageio_ffmpeg` binary so rendering is PATH-independent (~20 research scripts already do this via
  `rcParams["animation.ffmpeg_path"]`; 6 hardcode the conda path — machine-specific, do NOT carry forward).
  **Measured this session:** build-figure-once + `set_data` = **7.9 ms/frame** (vs 12.7 rebuilding; bare cmap path
  0.10) → a 1000-frame video ≈ 8 s. libx264 needs **even dims** → pad (409×205 → 410×206 verified). A prototype
  single-module `video.py` + 26 tests (25 green) was written then REVERTED out of the shared working tree (the Stim
  session works in the same tree); draft preserved in the session scratchpad, to be folded into the `video/` package
  per the plan. **Implementation is GATED on an explicit user go.**

Analysis-fields is DONE. Open threads (unchanged by this session): **consolidation Phases 2–5** (mesh/stimulus/
ConductivityConfig unify → engine rewire+delete; blocker: Surrogate/Optimizer consumers); the two **deferred solver
decisions** #13 GPU sync-free PCG + #14 mono-ionic V5.3 alignment (both change a default path → need a regolden — now
on main via the ff, so a decision is more pressing); **MCP follow-ups**; and the code-audit fix backlog (2026-07-02,
P1 bidomain M4 etc.). Analysis-fields FUTURE polish (documented in the PLAN, not blocking): Shaw–Rudy SF, `front_metrics`
torch migration, a co-area identity explicit test. See the open-threads block below.

## Thread

### 2026-07-23 — EXTRACTION AUDIT: cardiac_core as a standalone public repo → **DONE, PUSHED** (see Session Log)
**⚑ RESOLVED 2026-07-23**: the audit below said "NOT READY TO PUSH"; the blockers were then fixed and the repo
was extracted, verified (480/2 from a clean-venv install), and **pushed to `github.com/RealJokerInc/cardiac-core`
(public, MIT)**. Colab install verified from the live URL. Full execution in the 2026-07-23 Session Log entry.
The audit findings below stand as the record of what had to be fixed first.
**User intent**: push `cardiac_core` as its own public GitHub repo (so Colab can `pip install` it) and use
that instead of the monorepo. Asked for a comprehensive readiness audit first.
**Method**: 4 independent adversarial lanes (monorepo coupling · packaging · test portability ·
public-exposure) + direct verification by me of every load-bearing finding. Full report:
[EXTRACTION_AUDIT_2026-07-23.md](./EXTRACTION_AUDIT_2026-07-23.md).
**Git premise CONFIRMED**: HEAD == `stim-object` == `origin/main` == `9ef97f3`; video (`2a4b0e3`) + Stim
merged and pushed; library code fully committed.
**GOOD**: package stands alone at module level (wheel → fresh venv → imported from `/` → all 3 engines
run, `single_cell` APD90 217 ms); suite **482 passed / 2 xfailed** in 8m26s; NO third-party code vendored
(engines are all ours; ionic models original from published equations w/ citations) → no licensing
entanglement; no secrets; 61 commits extractable via subtree split; `cardiac-core` free on PyPI; nothing
to carry from monorepo pytest config (root conftest only adds `Surrogate/`, proven unused empirically).
**BLOCKERS needing a HUMAN decision:**
- **⚑ H1 — a collaborator is NAMED in shipped solver code.** `_monodomain/.../fdm.py:322,329,433`:
  "the failure we hit on **John's tanks**", "the **John-equivalent**" ×2. Verified in source. Names a real
  person without context AND identifies his **unpublished** discrete-reduction tank model as the validation
  reference for a shipped stencil. 3-line rewrite, but **cannot be undone after publication**, and is not
  ours to consent to on his behalf.
- **H2 — unpublished research productized in public docstrings**: `_lbm/boundary/wall_modes.py:1-19`
  ("boundary_conduction_speedup research → productized", ZERO bias / INVERSE crescent / β-curvature knob)
  + `analysis.py:582-586`. Our own work — but it publishes ahead of the papers.
- **H3 — no LICENSE anywhere.** The EXISTING public monorepo is therefore already all-rights-reserved.
**MECHANICAL blockers**: M1 `media.py:24` `_REPO_ROOT` → `site-packages` when installed (REPRODUCED;
fix must be backward-compatible — **50 monorepo files** depend on today's root: Monodomain 23, Research 22,
Optimizer 2, Lab 2, mcp 1 → use `root=` → `$CARDIAC_MEDIA_ROOT` → walk-up-for-`.git` → cwd);
M2 deps 100% wrong (declares `mcp`, which the package never imports; omits torch/numpy/**scipy**/
**scikit-image**/**torch-dct** — the last three verified as bare imports with NO fallback, and torch_dct is
CORE because `bidomain()` auto-selects the spectral solver → **bidomain fails out of the box without it**;
torch-dct is single-maintainer, last released 2020 → consider vendoring the ~60-line DCT);
M3 `test_originals_untouched` HARD-FAILS on a fresh clone (hashes 3 monorepo trees, empty-digest → assert
false, not skip) → **delete it**, it is a monorepo-only guard; M4 no license/authors/readme/classifiers/
`py.typed` (84% of 1121 fns annotated!) + wheel drops `API_CHEATSHEET.md` which `cardiac_mcp` reads at
runtime + `namespaces` defaults true so the wheel silently ships `tutorials/_build/`; M5 `cardiac_mcp` +
its console script must be cut or every user gets a broken `cardiac-mcp` on PATH.
**⚑ L1 — LAYOUT IS LOAD-BEARING**: `test_integrity.py:21` imports `cardiac_core.tests._integrity.make_goldens`
and `tests/__init__.py` exists → `tests/` MUST stay a subpackage. `git subtree split` flattens contents to
the repo root by default, which breaks that import AND makes the suite's ~60 media writes escape the
checkout into its PARENT. Re-nest commit is mandatory.
**Also**: L2 `test_live_cv_gate` = 25% of runtime and points outside the repo (delete); L3 tests write
~60 media files to repo-root `media/lab/_sim_outputs/` (22 MB accumulated) with no `root=` available →
needs a tests conftest fixture; L4 CI hazards — golden bit-identity is BLAS/hardware sensitive (atol=0),
and `test_video.py` has no `importorskip` (missing imageio/PIL → collection ERRORS, not skips).
**KEEP AS-IS**: the RKC `KNOWN LIMITATION` block (`explicit_rkc.py:25-40`) — honest, precise, opt-in and
unreachable from the public API. Reads as rigor.
**~360 KB of internal planning docs** ship as top-level package files (more bytes than the library);
`ANALYSIS_METHODS_PRIOR_ART.md` + `ANALYSIS_FIELDS_DATA_MODEL.md` are the two publishable ones.
**Status**: staging NOT started — gated on H1/H2/H3.

### 2026-07-23 (3rd parallel agent, cont.) — tutorial LESSON 01 SHIPPED (grid → stim → 3 engines → CV)
**User re-scoped the session** ("something simpler — a simple little interval that guides a person to
build a simulation": grid, stim, using the simulation, and a monodomain + bidomain + LBM run). Built it.
**Shipped**: `cardiac_core/tutorials/01_build_a_simulation.ipynb` (34 cells / 12 code, ~90 s),
`cardiac_core/tutorials/README.md` (index), and `cardiac_core/tutorials/_build/build_01_build_a_simulation.py`
(the reviewable source of truth — a `.py` diff instead of an `.ipynb` diff; `--script PATH` emits a flat
concatenation of the code cells for headless regression until the nbconvert gate exists).
**Shipped config, all MEASURED not guessed**: `Grid(201,51,0.01)` (2×0.5 cm), `ConductivityConfig.bidomain(1.74,6.25)`,
`Stim.boundary(g,"left",start_time=1,duration=2,amplitude=-52)` (102 nodes = 2 cols), `t_end=40, save_every=0.5`;
**mono dt=0.05 → 58.8 cm/s (20.6 s)**, **bidomain dt=0.05 → 59.6 cm/s (31.8 s)**, **lbm dt=0.01 → 64.6 cm/s (11.5 s)**;
total run wall 63.9 s. Tuning basis: dt=0.05 costs only ~1.2% CV vs the dt=0.02 default (59.49) for ~2× the
speed; lbm dt=0.01 costs 0.2% vs dt=0.005 for ~2×. Baselines at default dt: 59.49 / 60.16 / 64.75.
**Verified, not assumed**: ran the exact reader-facing code end-to-end under `conda run` (exit 0), and
**rendered and LOOKED AT the figures** — the wavefront snapshots put the front at 0.17→1.62 cm between
t=5 and 30 ms ⇒ ~58 cm/s, independently reproducing `r.cv()`; the `phi_e` panel confirmed positive AHEAD /
negative BEHIND the front before the prose asserted it. Two defects caught this way and fixed: (1) the
stim-mask figure was a near-empty box (a 0.02 cm electrode across 2 cm is a hairline) → now a two-panel
full+zoom, and the emptiness became the teaching point; (2) the `phi_e` prose said "biphasic deflection",
which is a TIME-domain statement about a SPATIAL map → rewritten to describe the sign flip they can see,
then explain why a fixed electrode therefore records positive-then-negative.
**⚠ Consequence for the 11-lesson arc**: lesson 01 front-loads the mechanics and now overlaps designed
L03 (cell→monolayer CV), L10 (two engines) and part of L11 (`phi_e`). **The Core tier must be re-cut
before authoring continues** — do not author 03/10 as specced. Noted at the top of the tutorials PLAN.
**Tooling finding**: `nbformat`/`nbconvert` are **NOT installed** in `heart-conduction` (ipykernel,
jupyter_client, jupyter_core, matplotlib are) — the old plan's P0.1 flagged this risk and it was real.
Worked around by emitting the `.ipynb` as plain JSON (validated structurally: nbformat 4.5, unique cell
ids, correct code-cell fields). Consequence: the notebook ships with EMPTY outputs, and the § 8
execute-all anti-rot gate cannot be built until they're installed. Deliberately did NOT `pip install`
— two other agents were running in the shared env and a mid-flight dependency upgrade wasn't worth it.
**Observation for whoever owns the cheatsheet**: it states LBM CV runs **~30–47% higher** than FDM
monodomain; measured here it is **+9.9%** (64.6 vs 58.8, same σ/grid/stim). Not investigated — may be
dx- or config-dependent — but the documented range did not reproduce on the default declarative path.

### 2026-07-22 Session (3rd parallel agent) — tutorial notebook series: DESIGN CONVERGED, authoring GATED
**Context**: three agents working `cardiac_core` concurrently — (1) the video pipeline, (2) the Stim pipeline,
(3) this one, on the Jupyter tutorial series. No code touched here; the deliverable is the lesson design.
**Found — the 2026-07-21 `cardiac_core/tutorials/PLAN.md` had gone stale in 4 ways** (all from work that
shipped between it being written and now):
- **P0.2 is DONE, not a prerequisite.** The plan's blocking prep step ("implement `cc.single_cell()` before
  writing any lesson") shipped in `63f6982`. Real signature `cc.single_cell('ttp06', celltype='EPI',
  pre_pace=5)` → `sc.V`/`sc.apd(0.9)`/`sc.final_state` — NOT the `stim=/t_end=/dt=/cell_type=` the plan guessed.
- **The plan's pedagogical SPINE was voided by the Stim work.** It hung lesson continuity on "L1 teaches the
  `{start_time,duration,amplitude}` dict keys, they carry into L3+" — that dict now raises `DeprecationWarning`
  (Phase 2, `743e6d4`), so the notebooks would ship warnings. Replaced by a better spine (below).
- **`r.fields.*` + `safety_factor` + `wavelength`/`di`/`erp` didn't exist** when the plan was written; its one
  "EP toolkit" lesson is far too small a container.
- **Voltage clamp shipped as a first-class `Stim` mode** (`clamp=`, all 3 engines) — the plan lists it as v2 bonus.
**Decisions (user, this session):**
- **D1 — spine = LAB-EXPERIMENT ladder** (each lesson is a bench experiment, simulated; API taught as a side
  effect). Rejected: the API-concept ladder (reads as software docs) and a physics-first ladder.
- **D2 — two tiers: Core 01–06 + Advanced 07–11.** Core = one cell → drug → CV on a monolayer → video →
  pacing/restitution → scar+block, completable in an afternoon, ends by handing the reader `/sim-experiment`.
  Advanced = fibers → voltage clamp → `fields` → two engines → bidomain-infarct capstone.
- **D3 — caveats MINIMAL / operational only** (user overrode my "inline where it bites" recommendation). No
  numerics disclaimers in lessons; they live in `API_CHEATSHEET.md`. Two structural exceptions kept: L10 IS the
  engine-comparison lesson (the differing numbers ARE its content), and masked-node `NaN` in L06 (the reader
  sees holes in their own plot). Rule of thumb recorded in the plan: anything else that seems to need a warning
  is a signal to pick better lesson PARAMETERS instead of explaining bad ones.
- **New stimulus through-line** (replaces the dead dict bridge): `Stim` as *a named place*. L01–02 have no Stim
  at all (0-D, `pre_pace=`); L03 introduces `cc.Stim.boundary(g,"left")` as "where you put the electrode" — it
  reads as one English sentence where `lambda x,y: x<0.05` never did; L05 = the same Stim + `bcl/num_pulses`
  (a pacing train is the electrode firing repeatedly); L07 `Stim.center`; L08 the same object with `clamp=`
  instead of `amplitude=`. One object, learned once, re-used four ways.
- **`tutorial_helpers.py` — default to NOT writing one.** With `cc.viz` + the incoming `Video`/`Gradient`, a
  helper would teach a vocabulary that exists only inside the tutorials. Bar = "the library genuinely can't do
  this"; likely residue is one `plot_ap` for L01. Decide after the video branch lands.
**Gate**: authoring is BLOCKED until the video pipeline lands (L04 is the long pole — do NOT author it against
`cc.propagation_video` and then rewrite) and Stim merges to `main`. L01–03 could go first if video stretches.
**Also kept**: the execute-all anti-rot gate (nbconvert `--execute` every notebook, wired to `/verify`) — the
plan this replaces rotted exactly this way.
**Written to**: `cardiac_core/tutorials/PLAN.md` (full rewrite; §1 decision table, §3 the 11 lessons, §4 the
stim through-line, §5 caveat policy, §6 branch dependencies, §8 anti-rot gate, §10 open questions).
**API-SURFACE GAP FOUND while verifying lesson signatures (2026-07-23) — per-edge bath coupling is
unreachable publicly.** `cc.bidomain(..., boundary=)` validates against exactly `('bath','insulated')`
(`api.py:1751`) and maps `'bath'` → `BoundarySpec.bath_coupled()` = **ALL FOUR EDGES** (`api.py:1771`).
`BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])` is real + test-covered
(`tests/test_solver_fixes.py:148`) but reachable ONLY by importing `cardiac_core.mesh.boundary` and
hand-assembling a `StructuredGrid`. So a MIXED-BC bidomain — the exact configuration the Kléber
boundary-loading and `bidomain_parabolic_parabolic` questions care about — has no declarative route.
**Candidate fix**: let `boundary=` accept a list/tuple of edges (`boundary=["top","bottom"]`) alongside
the two strings. Consequence today: the tutorial capstone was re-scoped to `boundary="bath"` (all edges)
and mixed-BC dropped from v1.
**Second gap — `single_cell()` has NO conductance knob.** Verified sig: `single_cell(model, *,
celltype='ENDO', dt, bcl, n_beats, pre_pace, stim_amplitude, stim_duration, t0, Cm, save_every,
device)`. `scale_conductance` is a `CardiacSimulation` (tissue) method, so "apply a drug to ONE cell"
— the single most natural 0-D experiment, and the classic hERG/APD story — has no public route; the
only path is building an ionic-model instance, mutating a conductance attribute, and passing it as
`model=`. **Candidate fix (small, additive): `single_cell(..., conductances={'GKr': 0.5})`**, lowering
to the same name validation `scale_conductance` already does. Blocks tutorial lesson 02 (fallback:
move the drug lesson after tissue). Note also `celltype` is ENDO/EPI/**MID** (not 'M'), and 0-D pacing
already exists (`bcl`+`n_beats`+`pre_pace`).
**Third (cosmetic, consistency)**: the 0-D stim kwargs are `stim_amplitude`/`stim_duration`/`t0` while
`Stim` uses `amplitude`/`duration`/`start_time` — three renamed keywords for one idea, met by anyone
graduating from a 0-D script to tissue.
**Also corrected a second stale signature**: it's `ConductivityConfig.anisotropic(sigma_l, sigma_t,
fiber_angle, chi, Cm)` — raw σ in mS/cm and the angle in RADIANS, ONE global angle (no per-node fiber
field via the factories); `sigma_eff`/`D_eff` return a 3-tuple (xx,yy,xy) when anisotropic. The old
tutorial plan said `anisotropic(D_l, D_t, …)`, which would have taught a χ·Cm units error.
**Observed (worth knowing for 3-agent work)**: the Stim session committed `743e6d4` INTO THE SHARED WORKING TREE
mid-session — 16 `cardiac_core` files went from modified to committed between two of my reads. Re-check `git
status`/`log` before reasoning about tree state; don't cache it.

### 2026-07-22 Session (cont.) — analysis-fields IMPLEMENTATION: all 7 phases shipped → committed → pushed to main
**Worked on:** executing `ANALYSIS_FIELDS_PLAN.md` end-to-end (user: "ready for plan.md implementation" → "run audit.
then keep working through all phases, stop only when u need my intervention" → "commit on main" → "push it").
**Accomplished:**
- **Phase-1 `/audit` (Opus, before building on the foundation): 0 blockers / 0 majors / 3 med / 4 low.** Fixed the two
  real ones live — MED-1 `r.cell_type` now forces ENDO for bidomain/LBM (those factories don't thread cell_type → a
  Phase-7 I_ion re-eval would build the wrong model); MED-3 added the missing CV-family canonicalization guard (step-
  wave synthetics can't distinguish interp/−40 from nearest/−20, so a smooth two-slope trace was needed). MED-2
  (legacy-bidomain conductivity) routed into Phase 3 → guard keys on `r.phi_e is not None`, not `conductivity.is_bidomain`.
- **All 7 phases implemented, each test-gated + a full-suite gate at phase boundaries.** P2 operators (staggered
  laplacian matches the engine's OWN FDM-5pt operator rel<1e-6; divergence theorem exact 1e-10). P3 source_sink via a
  conservative harmonic-face-D `diffusion_term` → matches `apply_diffusion` on uniform D AND a masked scar. P4 LAT
  fields (Bayly/SG conv, divergence gating, the shared `winding_loop_sum`; `phase_singularities` refactored onto it).
  P5 integrals (divergence/Stokes/Gauss-Bonnet cross-checks; isochrones via skimage marching squares). P6 scalar EP
  (`wavelength`/`di`, the 3-function apd baseline fix, `erp` protocol). P7 `single_cell` (0d-vs-tissue match — the ORd-
  bug guard) + Boyle–Vigmond `safety_factor`.
- **Key implementation findings (also in the PLAN Mutation Log):** the apd baseline fix is an **upstroke-foot walk**,
  not min-over-interval (the latter wrongly catches the previous beat's lower repolarization undershoot on a drifting
  baseline); `safety_factor` needs NO recorded ionic states because `∇·(D∇V)=Cm·dV/dt+I_ion` makes the source_sink
  integral the whole Boyle–Vigmond numerator (integrate only the INWARD/positive part = the charging phase, or the
  sourcing phase cancels it); collocated `grad`/`div`/`curl` use a WHOLE-sample mirror boundary (normal derivative = 0
  at a no-flux edge) while the staggered `laplacian` is a separate core.
- **Final gate: 369 passed / 2 xfailed / 0 failures** (8m36s). Integrity goldens atol=0 (no solver touched).
- **Shipped:** selective commit `63f6982` (26 files, +5,529 lines: code + 8 test files + the 4 spec docs + cheatsheet
  + archived plan) on `solver-hardening`, then `git branch -f main` (clean ff, main was a 34-commit linear ancestor)
  → `git push origin main` (`d1f43f6..63f6982`). Unrelated working-tree changes (MASTER.md, Optimizer, other Research
  questions, fig4c scripts) left untouched. **NOTE: the ff also landed solver-hardening's shipped work on main** — the
  deferred #13/#14 solver decisions are now on main too.
**Next:** consolidation Phases 2–5, or resolve the deferred #13/#14 (now on main), or MCP follow-ups.

### 2026-07-22 Session — analysis-fields: prior-art → math/calc docs → blueprint → audit-to-convergence → data model
**Worked on:** turning the 2026-07-21 `analysis.fields` design specs into a research-grounded, audited implementation
blueprint (user arc: "research the standard math for every feature" → "document the math" → "blueprint everything, run
audit cycle until converge" → API-shape Q&A → "save session").
**Accomplished:**
- **Prior-art survey (10 web agents across the session)** → `ANALYSIS_METHODS_PRIOR_ART.md`. Verdict: most of our
  design is the field standard. Five design changes adopted (canonical LAT interp/−40 + max-dV/dt reference; staggered
  `div=−grad*`; ghost-mirror boundary; one winding primitive; `source_sink`=SF numerator). §7 velocity-field deep dive:
  LAT-gradient ≡ optical-flow (both = |∂V/∂t|/|∇V| → LAT is *incidental*; optical flow is the no-LAT reentry fallback);
  GP/GPMI for CV uncertainty; Vigmond 2024/25 accuracy reality (CV trustworthy only ~≤2 cm from pacing, ≤1 mm). §8
  structured-grid: **optimap is our direct code analog**; the Bayly fit on a uniform grid = a fixed Savitzky-Golay
  conv kernel (one fit → velocity+curvature+residual); resolution law `δc/c≈c·δt/dx` → dx-ladder acceptance test.
- **Math + cited-calculation layers** written into `ANALYSIS_FIELDS_DESIGN.md`: per-field derivations (def → equivalent
  routes → why equal → trap → canonical) for every field + every existing `analysis.py` op; the operator toolkit; and
  a cited uniform-grid stencil/quadrature reference (A1–A8, B1–B7; LeVeque/Fornberg/Osher-Sethian/Savitzky-Golay/DLMF/
  Lorensen-Cline, 2 verifier agents).
- **Blueprint → `ANALYSIS_FIELDS_PLAN.md`** (7 phases, ~19 steps, test-gated, no solver changes) via `/blueprint`.
- **`/audit` cycle: 5 Opus rounds → CONVERGED** (R1 1C/6H/10M/4L → R2 1C/1H → R3 0C/1H → R4 0C/1H → R5 0C/0H). Real
  catches (all code-verified + fixed): result carried no conductivity/Cm (source_sink uncomputable); my R1 fix's
  **non-existent `sim._chi`+1400 fallback → 1400× wrong D_eff on the declarative chi=1 path** (→ `D_eff=data.D_xx/
  (data.chi·data.Cm)`); collocated `div(grad)`=wide checkerboard (→ staggered); flipping `activation_time`'s default
  left the scalar-CV family on their own −20 threshold (→ flip each to −40); result needs the RESOLVED ionic-model
  name + `group_cell_types[0]` cell_type + device-aware model rebuild for I_ion re-eval. **Pattern: each round's lone
  finding was a follow-on to that round's own fix — convergence tail, not new bug classes.**
- **`ANALYSIS_FIELDS_DATA_MODEL.md`** created from an API-shape Q&A: object hierarchy, property-vs-method-vs-operator
  and Fields-vs-VectorField terminology, and every torch shape (rule: `T` present ⇔ per-frame Vm/φ_e, absent ⇔
  LAT-based; trailing `2` ⇔ vector). Also nailed for the user: spatial ops = FD conv stencils not FFT; curl = cross
  pattern `Dx⊛Fy−Dy⊛Fx` vs div straight; central-diff `/(2dx)` = two-cell span (Taylor O(h²)).
**Next:** implement PLAN Phase 1 (canonical LAT) on the user's go. Docs uncommitted — 3 modified (DESIGN, KNOWLEDGE,
IDEALOG) + 3 new (PLAN, PRIOR_ART, DATA_MODEL). No solver code touched.

**SOLVER HARDENING — SHIPPED (2026-07-21, branch `solver-hardening`, 4 commits, NOT merged).** The audit findings
below drove a fix campaign ("work through all"): Step 1 (non-convergence signal + Chebyshev M1 fix), Step 2 (mid-run
voltage clamp + state injection), opt-in solver fixes (pcg_spectral mixed-BC, IMEX-SBDF2, RKC doc-defer). All
test-gated, integrity goldens bit-identical, full suite 260/2. **DEFERRED awaiting user decision:** #13 GPU sync-free
PCG (regolden or GPU-only) + #14 mono-ionic V5.3 alignment (regolden default — user flagged doubt it's a bug). See
the 2026-07-21 Session-Log entry + the "Solver hardening — SHIPPED" callout in KNOWLEDGE. **NEW SIDEQUEST:** an
"Intro to Cardiac Core" 8-lesson Jupyter tutorial series — plan at `cardiac_core/tutorials/PLAN.md` (prep-first;
L1 single-cell → L8 bidomain infarct + mixed BC). Also assessed library packaging: importable (cardiac-core 0.1.0)
but deps under-declared (only mcp; torch/numpy/scipy/torch_dct missing) + no README/LICENSE/__version__.

**Audit provenance (2026-07-16). No solver code changed at audit time (audit + measure only).**
6-lane adversarial audit + empirical GPU benchmark; every HIGH/MED finding independently
reproduced. Full ranked table + GPU characterization are in **KNOWLEDGE.md → "Solver +
GPU audit — 2026-07-16"** (the reference; scan there). Scratchpad artifacts: `gpu_bench.py`/`gpu_bench_results.json`,
`FINDINGS_task6_gpu.md`, `AUDIT_collation.md`, `cheby_repro.py` + agent repros.

**Verdicts (30-sec scan):**
- **device='cuda' IS using the GPU** — full residency cuda:0/float64 across all 3 engines; result hooks on cuda.
- **The "crossover weirdness" = per-iteration GPU→CPU host syncs in the iterative solvers, NOT a CPU-compute
  fallback** (the user's mental model was close but the mechanism is a pipeline stall, not offload). Syncs/step:
  explicit 0, mono CN+pcg 24, CN+dct 1; bidomain default (pcg+pcg_spectral) syncs the heaviest; LBM = 0. GPU
  per-step is launch-latency bound (~6-10 ms flat) → wins over CPU only above ~10k dof. float64 on a 1:64-FP64 card.
- **2 HIGH silent-wrong bugs** — (1) mono Chebyshev-Jacobi tunes to raw A not D⁻¹A → 46% err at high diffusion-number
  (opt-in, machine-precision at default dt); (2) bidomain pcg_spectral singular Neumann precond on anisotropic
  mixed-BC → stalls, wrong phi_e. **Systemic MED:** ALL iterative solvers silently return unconverged as converged.
  IMEX-SBDF2 silently 1st-order; RKC refinement-immune ~0.8% err; mono ionic conc-currents use post-RL gates
  (diverges from V5.3, inherited from V5.4 — bidomain copy is correct). **The DEFAULT mono (pcg+CN) and bidomain
  (pcg auto) paths are correctness-solid; risk is opt-in solvers failing silently.** LBM audit = clean.
- **Highest-value single fix (future):** a shared non-convergence signal (warn/raise + surface residual) — closes
  the systemic finding across 4 lanes and makes the HIGH bugs fail LOUD instead of silent.

**Next-direction plan recorded (task #9, detail in KNOWLEDGE):** advanced features (masked per-step voltage clamp +
mid-run state injection, both via one `_stepping_run` per-step hook in CardiacSimulation) → GPU opt follow-ups (PCG
sync-free convergence, auto-dct-on-GPU, isotropic-bidomain→Tier-1 spectral, COO→CSR, torch.compile LBM) →
consolidation Phase 2-5 (mesh/stimulus/ConductivityConfig unify → engine rewire+delete [blocker: Surrogate/Optimizer
consumers] → clean namespace). Advanced features are independent of the dedup phases and can land first.

**What NOT to retry / gotchas learned this session:**
- A quick end-to-end pcg-vs-chebyshev compare at the DEFAULT config shows them AGREEING (both ~machine precision) —
  do NOT conclude Chebyshev is fine from that. The Jacobi-bounds bug only bites at high diffusion-number (off/diag
  ≳0.24); you must sweep dt/dx to see it. Goldens don't catch it (they froze the safe regime).
- cardiac_core faithfully copied V5.4; the mono-ionic conc-ordering deviation from V5.3 is a V5.4-lineage defect, NOT
  a copy bug. Self-goldens can't catch V5.3 divergence (they're self-referential).
- Bidomain declarative path ALWAYS builds D_i_field (even isotropic) → is_isotropic=False → never auto-selects the
  fast Tier-1 direct 'spectral' solver.

---
**PLAN.md EXECUTED — P0/P1/P2 usability fixes SHIPPED (2026-07-16, branch `usability-fixes-p0-p1`, NOT yet merged to main).**
All 5 PLAN phases done + a 5-lane adversarial audit of the whole branch (see the 2026-07-16 final-audit Thread
entry — 4 more real bugs found + fixed), each test-gated + per-engine integrity goldens bit-identical (atol=0).
Commits P1 `a37d325` → P2 `d78a86d` → P3 `d94aa6d` → P4 `d6a3237` → P5 `99e1fa3` → audit-remediation `c0306d2` → round-2 remediation `9f387ef`.
Suite **260 passed / 2 xfailed** (218 baseline + 42 tests in `cardiac_core/tests/test_usability_fixes.py`).
- **P1 six P0 bugs:** B1 GPU device-mismatch (`_result_from` builds `times` on Vm's device); B8 NaN-fill masked
  nodes (`StructuredGrid.flat_to_grid`, mono+bidomain; LBM has no flat_to_grid, untouched); B3/B4 `apd_at`
  beat-bounded peak + dome-aware LAST-crossing (spike-and-dome safe); B5 `Grid(N,1)` degenerate-axis guard;
  B6 `forward_euler` CFL warn (FDM retains `_D_max`); B7 `record=` key validation.
- **P2 (B2):** DCT/FFT wired through `_build_linear_solver(spatial,dt,scheme)` + `_spectral_kwargs` — fast path
  restored (was TypeError→silent PCG fallback = the runtime wall). DCT CV matches PCG; default pcg untouched;
  full-rectangle only (masked → pcg).
- **P3:** de-trapped ~18 `NotImplementedError` stubs (informative errors, removed misleading `>>>` examples);
  IMPLEMENTED `scale_conductance` / `set_conductivity` / `scale_conductivity` (rebuild-from-t=0). Adversarial
  audit of the diff caught 2 CROSS-ENGINE bugs (my mono-only tests missed them): (i) declarative bidomain uses
  `sigma_i/sigma_e` fields not `D_xx` → scar was a SILENT no-op → fix applies the mask op to sigma too, and a
  nonzero absolute D on sigma-bidomain now RAISES; (ii) `scale_conductance` re-derived the model from
  name+mesh-cell_type but bidomain/LBM build ENDO by default → CELL-TYPE FLIP (Gto 0.073→0.294) → fix
  deep-copies the LIVE engine model (`_live_ionic_model()`), preserving cell type + prior scalings.
- **P4:** `API_CHEATSHEET.md` rewrite — Solver&dt section, drug/conductance map (**PCa = ICaL, NOT "GCaL"**;
  ORd adds GNaL), **ORd LBM-only** (raises on mono/bidomain) + paci/phas13/mhas13 on mono, `record=`/
  save_result arg-order/df/two-step phase_map, fiber_angle radians, bath cost, LBM CV ~+30–47%; +
  `test_cheatsheet_examples_execute` canary (execs a tagged runnable block).
- **P5 (optional):** analysis aggregates — `dominant_frequency_map`/`df_map`, `cv_between`, `radial_cv`,
  `apd_per_beat`, `restitution_slope` (+ result hooks + top-level exports) + DF-resolution warning +
  zero-node-stimulus warning.
Key facts verified empirically this session: `monodomain('ord')` RAISES (SR-release/CaMKII concentration path
unwired for classical splitting) but `lbm('ord')` runs; `paci`/`phas13`/`mhas13` run on monodomain; the FDM
`_harm` guards 0/0 (`s>0 else 0.0`) so a D=0 scar BLOCK is NaN-safe.

**API usability audit ROUND 2 (full-solve-and-run, +30 tasks) — 2026-07-16 → same report, "ROUND 2" section.**
10 agents, 30 new tasks (25–54) + full-scale re-run of the prior 24; agents had to actually SOLVE+RUN each to
completion. Running fully LOWERED the grade — it surfaced a class of defects a smoke test hides. **13 concrete
bugs (B1–B13)**, the load-bearing ones: **B1** GPU `device="cuda"` crashes ALL analysis/viz (`_result_from`
puts `times` on CPU, `Vm` on CUDA — one-line fix, mine); **B2** `linear_solver='fft'/'dct'` broken via factory
(`FFTSolver.__init__` missing args) → everything stuck on slow PCG = root of the runtime wall; **B3/B4**
`apd_at` peak-over-remaining + notch bugs → silently wrong multi-beat/low-repol APD; **B5** `Grid(N,1)` crash;
**B6** `forward_euler` silent blowup past dt-stability; **B7** `record=` silently ignores unknown keys; **B8**
masked nodes returned at 0.0 mV → 23% silent CV error on every scar/fibrosis study; **B9** dead `stim_amplitudes_e`
(no defibrillation). PERF: fixed per-step wall (~1.5–3 ms/step CPU, ~13 ms/step GPU, grid-independent) → long
protocols 5–11 min; escape = `forward_euler`+`none`+`dt`≈0.04 (undocumented). Verdict flips: T3 automaticity
No→Yes (undoc `paci`), T6 non-hole scar No→Possible (per-node D=0), T8 isthmus block Yes→No-via-geometry,
T15 5/5→broken-on-GPU. Reentry ACHIEVED (anchored CL=296 ms, figure-8 CL≈344 ms, ring min-circ≈λ=2.82 cm) with
the solver workarounds; blockers = runtime wall + no rotor-seeding/mid-run-state API (all `set_*`/`get_state`/
`scale_conductance`/`clamp_voltage`/`add_pacing` are NotImplementedError stubs). Report ends with a **MERGED
P0–P4 fix list** = the blueprint target. Contention caveat: box oversubscribed, absolute wall-times inflated ~2–4×.

**Task-based API USABILITY audit — 2026-07-15 → [API_USABILITY_AUDIT_2026-07-15.md](./API_USABILITY_AUDIT_2026-07-15.md).**
Agentic walkthrough: 24 realistic scientist tasks across 7 categories, 6 parallel agents, each WRITING +
RUNNING the minimal cardiac_core script and rating Possible?/Ease(1–5) empirically. **Verdict: "possible
but painful," mean ease ≈2.7/5**; strong at *expressing* a sim, weak at *parameterizing* + *measuring* it.
**2 tasks IMPOSSIBLE** via the public API (transmural cell-type gradient; clean single-cell automaticity);
the motivating category (ionic tuning/pharmacology) is the worst-served (1.5/5). Ranked themes: **(CRIT)**
the parameter/heterogeneity layer (`scale_conductance`/`set_parameter`/`set_conductivity`/`scale_conductivity`/
`clamp_voltage`) is `NotImplementedError` stubs WITH inviting worked-example docstrings → #1 hallucination
trap, blocks 7 tasks; **(HIGH)** no documented conductance knob (only working route = inject a model
instance, undocumented + inconsistent TTP06-vs-ORd); **(HIGH)** cheatsheet ERRORS (lists `ord` as a
monodomain model but ORd runs LBM-only; omits paci/hiPSC; save/load + dominant_frequency + phase_map absent;
`phase_singularities` mis-primed); **(HIGH)** analysis is single-point/x-axis-only (no cv-map/radial/
restitution, no DF map, no tip tracking); **(MED)** silent failures (nan CV on block, masked nodes returned
at 0.0 mV → counted "activated", zero-node stim = silent no-op); **(MED)** no 0-D single-cell mode; no
sweep/fitting helper; uniform-only conductivity + global-only cell type (MID crashes). Bright spots: viz
(5/5), the LBM d2q9 boundary errors (validates the same-day F2 change), stimulus expression, masks. Fix
priority: kill the stub trap → cheatsheet correctness pass → NaN-fill masked Vm → analysis aggregates →
0-D + sweep → per-node fields. NOTE: connects to the CODE_AUDIT #7/#12 "planned-not-shipped methods" note —
usability audit shows that's an active runtime trap, not a benign doc gap.

**API failure-mode sweep + F1/F2 hardening — 2026-07-15.** Ran a full public-API failure-mode check
(all ~40 `_LAZY` exports; both construction paths — declarative factory + file-format mesh; all 3
engines; analysis/io/geometry/viz; + degenerate-input and expected-raise probes; 103 checks). **Verdict:
the whole public surface is complete and working** — every documented call completes or raises its
documented error; all 4 contract guards and 6/7 degenerate inputs already degrade gracefully. Two real
gaps found and FIXED (commit `2938cf9`, main; 218 passed / 2 xfailed): **F1** — an empty run
(`t_end < save_every`) crashed the analysis hooks (`.apd()`/`.lat()`/`.cv()`) because `_collect`/
`_result_from` returned a rank-1 `(0,)` Vm; now they return rank-3 `(0,Nx,Ny)` and `activation_time`/
`apd_map` guard the zero-length time axis → NaN maps / NaN, no crash. **F2** — `hbb` reclassified as
**D2Q9-only** (joins ncs/scs/combined in `wall_modes.D2Q9_ONLY`; `hbb`+d2q5 now raises instead of
silently acting as a neumann no-op), and the LBM boundary **default is now lattice-aware**: `neumann`
on d2q5 (UNCHANGED — tuner calls `run_lbm` with no lattice/boundary, and goldens pin d2q5, so both are
untouched), `hbb` on d2q9 (label-only; neumann≡hbb numerically on d2q9). Cheatsheet §4 documents the
d2q9 requirement (the prior doc gap). **User decisions (2026-07-15):** keep the global d2q5/neumann
default (do NOT flip to d2q9); merge only the 12 committed tuner commits to main. Two cosmetic nits
ALSO FIXED (commit `e707fe1`): **F3** `point_distance` now takes `center=(x,y)` matching
`circle_mask`/`annulus_mask` (was scalar `x0,y0`; only 2 internal test callers); **F4** cheatsheet §2
now notes `ConductivityConfig.sigma_eff`/`D_eff` return a scalar (iso/bi) but a 3-tuple `(xx,yy,xy)`
(aniso). Also landed the `engine-tuner-v2-joint` branch (12 commits) onto main (merge
`9d82f56`; one MASTER_KNOWLEDGE_INDEX.md conflict resolved keeping both the β-dt-guide bullet and the
SCS-gate-decontamination correction).

**Deep code audit — math integrity + API — 2026-07-02.** After pushing the API-consistency work to
`main`, ran a 6-lane agentic walkthrough of ALL of cardiac_core with per-lane NUMERICAL verification →
[CODE_AUDIT_2026-07-02.md](./CODE_AUDIT_2026-07-02.md). **0 blockers / 4 majors / ~22 minors.** Default
paths + all Phase-0–5 hardening + the 4 ionic models verified SOUND (cross-derivative fix confirmed;
time-stepper orders exact CN=2.00/RK4=4.01; LBM Chapman–Enskog + `mrt_wall` + masked-bounce correct; CV
matches ref 54.14/54.35; D_eff=9.72e-4 + Cm-trap correct). The 4 majors: **mono Chebyshev** (Gershgorin
bounds on raw A not preconditioned D⁻¹A → 94 mV wrong at a stiff config, opt-in solver; bidomain already
has this fix), **mono FFT** (continuum −k² vs discrete 5-pt eigenvalue, opt-in), **bidomain `step()`**
AttributeError (pre-existing, untested public method), and — most serious — **bidomain spatially-varying
anisotropy** breaks elliptic-operator symmetry → CG-family solvers silently return **~13% wrong phi_e**
(per-node fiber fields; the uniform-angle public API is SAFE). M1+M3 orchestrator-reproduced; M2/M4 on the
auditors' numerical evidence. **Audit only — NO source changed**; findings triaged for later (fix priority
P1–P4 in the doc). Commit `619460c`. Recurring weak spot = the **FDM cross-derivative** (M4 is the same
family as the deferred C7 / boundary anisotropy).

**API-consistency hardening + contract-matrix harness SHIPPED (2026-07-01).** The two post-ship boundary
gaps turned out to be a *class* of API-surface fragility. Ran a 4-lens adversarial audit
([API_CONSISTENCY_AUDIT.md](./API_CONSISTENCY_AUDIT.md): 7 HIGH/8 MED/6 LOW), **audited the fix PLAN to
convergence over 4 rounds** (R1 5blk/10maj → R2 1blk/5maj → R3 1blk/1maj → R4 1blk-mechanical →
CONVERGED — each round narrower: code-bugs → coverage → mechanism text → a namedtuple field order), then
executed 6 phases on `engine-tuner-cardiac-core` (`1a65d3d`→`9702bb7`). Keystone =
`tests/test_api_contract.py`, the contract matrix **written FIRST** with `xfail(strict=True)` forcing each
fix's in-phase flip (the post-mortem's process cure, mechanized). **217 passed / 2 xfailed** (C2 oblique
capability + C7, documented-deferred; the audit caught that oblique LBM is real numerics/Audit #46, not
wiring — would've shipped silent-wrong). Goldens bit-identical every phase. See KNOWLEDGE "API-consistency
hardening". **This also RESOLVES** the prior Next-Step item "mono `boundary_mode`/`stencil` not surfaced"
(C6, now exposed) and the ionic-registry parity #6/paci-alias #12 (C3/C8). **Remaining:** Form-A→B
monodomain convergence; oblique-LBM moment-space rotation (Audit #46) if ever wanted.

**Foundation cleanup + boundary modes SHIPPED (2026-07-01).** A cardiac_core+cardiac_mcp adversarial audit (46 findings → [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md); 8 lanes, find→refute→completeness) drove a 3-phase, audit-to-convergence-hardened cleanup ([PLAN.md](./PLAN.md); R1 5blk/6maj → R2 2/2 → R3 SOUND): **P1** fixed a real mono FDM anisotropic cross-derivative BUG (wrong sign + half magnitude; `V=x·y` now `+2·Dxy`) + unified the chi/D convention (`D_xx` RAW everywhere, effective = `D/(χ·Cm)` in every engine, default `D=1.4`, blocked-default fixed) + ionic-override replay + MCP path-traversal; **P2** removed FEM/TriangularMesh + dead code + API footguns; **P3** productized the LBM flat-wall boundary modes as `cardiac_core.lbm(boundary=, alpha=)` — hbb / `specular_nextcell` (NCS) / `specular_samecell` (SCS) / `combined`-α (the β curvature knob), 'ncs'/'scs' aliases (default `neumann` bit-identical → goldens safe). Suite 148/1 → **196 passed / 0 failed**; 5 commits pushed to `origin/engine-tuner-cardiac-core`. **RESOLVES** the boundary-mode API→engine gap (for LBM), the `create_cardiac_mesh` chi firewall-bypass (cluster #1), and FEM removal.

**cardiac_mcp MCP server — STANDARDIZED + SHIPPED to `main` (2026-06-28→30).** The `cardiac_mcp/` server (built 2026-06-26) was audited against the OFFICIAL MCP spec **2025-11-25** (4 spec-research agents, verified vs live spec + installed SDK source) → a 4-tier PLAN → hardened through **3 adversarial audit rounds (12→5→0, CONVERGENCE CLEAR)** → executed **Tiers 1–3** (T1 honest annotations + `serverInfo.version` + MIME + two path-traversal guards; T2 typed `outputSchema`/`structuredContent` + 2 prompts + README + Option-B installable `cardiac-mcp` console script; T3 provenance + CPU/FSIZE-limited `run_experiment` + localhost HTTP transport + `REMOTE_DEPLOY.md`). **16 cardiac_mcp + 140 cardiac_core tests green; merged --no-ff → `main`, pushed (`41d17f4`).** Phase 4 (registry publish) SKIPPED. Detail: KNOWLEDGE "Goal-2 MCP server — standardization audit" + the 2026-06-28 Session Log + PLAN.md Mutation Log.

**Goal-2 MCP server — `cardiac-core` — SHIPPED local (2026-06-26).** Built `cardiac_mcp/` — an MCP (Model Context Protocol) server exposing `cardiac_core` to ANY MCP host (Claude Desktop/Code/IDE), breaking the skills' Claude-Code-terminal-only ceiling — the real reach step for the wet-lab audience. **Two-track tool surface** (user decision): a DIRECT `simulate()` (ephemeral CV, no record, coarse-dx fast ~8s) + the GATED `build_manifest`→`commit_experiment`→`run_experiment` chain that ports the `/sim-experiment` accountability gate STRUCTURALLY — `build_manifest` returns a self-signed `experiment_token` embedding the exact manifest+params; `commit_experiment` refuses unless that token verifies AND `confirmed=True`, so the committed `Lab/{date}_{slug}/` script is provably what the scientist reviewed. **Local stdio now, designed for remote-HTTP later** (user decision): all logic in transport-agnostic `cardiac_mcp/core.py`, `server.py` only wires FastMCP, `__main__` picks transport (HTTP = one-line swap). Registered via `.mcp.json`; `mcp` SDK (1.28.0) installed in the env. Validated: 10 core tests + server boot (5 tools/2 resources) + real stdio client↔server roundtrip. See KNOWLEDGE "Goal-2 MCP server". **Next:** activate in Claude Code (approve the project server), then optionally add resources/prompts (presets, glossary), media tool, and the HTTP transport for remote scientists.

**Goal 2 — the LLM layer — SHIPPED (2026-06-25).** The script-generating skill suite for wet-lab scientists is built + committed (`/sim-experiment` keystone + `/sim-preset` + `/sim-media` + `/sim-notebook` + `cardiac_core/API_CHEATSHEET.md` + `cardiac_core/viz.py`); 140 tests green; validated end-to-end (control/knockdown CV series). See KNOWLEDGE "Goal-2 LLM layer — SHIPPED". **Both north-star goals now delivered.** Remaining options: Layer-A `SimulationSpec`; programmatic claude-api; Form-A→B convergence; FEM removal. Keystone `/sim-experiment` to be `/audit`'d (the double-check gate).

**Consolidation (A2 vendoring) SHIPPED (2026-06-25).** `cardiac_core` is one self-contained package — 3 engines vendored `_monodomain`/`_bidomain`/`_lbm` + shared `ionic`/`mesh`/`stimulus`, `_prepare_engine()` hack deleted, bit-identical goldens, originals frozen. Phases 0–5 `935160b`→`37dc381`. → KNOWLEDGE "cardiac_core unified ground-up package". *(Predecessors condensed — full detail in Thread: V5.5 Cm-correct fork + consolidation Phase-1 copy-only, 2026-05-30/31; the Goal-2 design reframe to wet-lab code-gen, 2026-06-25; the "ditch FEM → structured-grid only" pending constraint — RESOLVED by the 2026-07-01 FEM/TriangularMesh removal. The deferred code-dedup [engines import from cardiac_core + delete copies] stays per-consumer — big-bang breaks Surrogate/Optimizer.)*

### 2026-07-16 (superseded next-step snapshot — kept for the narrative)
*The merge named below happened: `usability-fixes-p0-p1` has been on `main` since 2026-07-22.*
**▶ MERGE `usability-fixes-p0-p1` → `main`** (commits `a37d325`→`c0306d2`: 5 phases + `562a7a0`/`257b2c1` docs + `c0306d2`/`9f387ef` audit-remediation; PLAN EXECUTED + 5-lane final audit clean, see Current Direction). Suite 260/2xfail,
goldens bit-identical. The branch is review-ready; nothing else in the plan is outstanding. After the merge, the OPEN threads below are unchanged (none were touched this session). Two
usability items that were NOT in this PLAN and remain P3/future work: **B9** dead `stim_amplitudes_e` (no
defibrillation) and the rotor-seeding / mid-run-state API (`set_voltage`/`get_state`/`clamp_voltage`/`add_pacing`
are now HONEST stubs, not implemented) + a 0-D single-cell mode + `cc.sweep`/`fit_conductivity` (all documented in
PLAN.md "Future work").

The A2 unification, Goal-2 skill suite, cardiac_mcp server, AND the 2026-07-01 foundation cleanup + LBM boundary modes are all SHIPPED (see Current Direction). RESOLVED this session: `create_cardiac_mesh` chi firewall-bypass (P1 cluster #1 — D_xx RAW convention + default D=1.4 + band guard); boundary-mode API→engine gap for LBM (P3 — `lbm(boundary=, alpha=)`); FEM/TriangularMesh removal (P2). Remaining open threads:
- **Code-audit fix backlog (2026-07-02, [CODE_AUDIT_2026-07-02.md](./CODE_AUDIT_2026-07-02.md), NOT yet actioned):** P1 = bidomain M4 (symmetrize the FDM cross-term + guard CG on non-SPD — silent ~13% phi_e error on per-node fiber fields); P2 = mono Chebyshev M1 (port bidomain's preconditioned-Gershgorin + fix `set_eigenvalue_bounds`) + mono FFT M2 (discrete 5-pt eigenvalue) + DCT/FFT precondition guard; P3 = bidomain `step()` M3, degenerate-input NaN guards (`activation_time`/`dominant_frequency`/`Grid` 1-D/empty-result shape), BGK stability gate, `dt or`→`is not None`, declarative ionic-instance-leak; P4 = docstrings + `add_stimulus` amplitude + hole-cell zeroing + LUT kink + conductivity guards + PCG-threshold unify + LBM `save_every` cadence.
- **Form-A→B convergence** (convert monodomain diffusion in `_monodomain`, delete `ConductivityConfig.for_monodomain()`) — confirmed-but-deferred.
- **Deferred audit backlog** (in PLAN.md "Findings Coverage"): Research/ doc reconciliation (#9/#10/#11/#24/#25); wiring the orphaned LBM `dirichlet`/`absorbing` as selectable modes (#37, needs a bc-value); ionic-registry parity (#6) + the `paci→PHAS13` alias check (#12); the FDM boundary-Dxy Neumann (#14) + bidomain bath≠insulated assert (#39); the 12 completeness gaps (analysis.py, io round-trip, LBM masked grids, device/dtype, second mesh loader).
- ~~**Monodomain/Bidomain boundary-mode API exposure**~~ **DONE (2026-07-01, Phase-4/C6):** mono `stencil`/`boundary_mode` + bidomain `stencil` now surfaced by the factories. (A fully-unified cross-engine boundary *concept* using bidomain `BoundarySpec` as template is still a future nicety, not blocking.)
- **Surrogate/Optimizer ionic migration** off engine-local `cardiac_sim.ionic` (per-consumer, test-gated; never delete out from under a live consumer).
- **MCP follow-ups** (optional): media tool wrapping `cardiac_core.viz`; more resources/prompts; reentry/restitution recipes; the remote-HTTP auth stack (`REMOTE_DEPLOY.md`) + Phase-4 registry publish when wanted.
- Deferred Goal-1 Layer-A `SimulationSpec`; programmatic claude-api; `API_REFERENCE.md` `[design]`→`[now]` tags.

--- prior (consolidation track, still valid) ---
**cardiac_core drift RECONCILED (2026-05-30):** the post-Phase-0 additions (`run.py`/`analysis.py`/`geometry.py`/`io.py`) are a benign wrapper-level convenience layer (77 tests now, not 34); no shared-code packages yet, so Phase 1 is unblocked. `Engines/` symlink index fixed (cardiac_core un-broken; lbm_v1 → real `LBM/Engine_V1`; monodomain_v5.5 added). See KNOWLEDGE "cardiac_core drift reconciled".

**Phase 1 (copy) DONE (2026-05-31):** `cardiac_core/ionic/` is the canonical superset copy (from V5.5; latent LUT keyword `cell_type_is_endo`→`celltype_is_endo` fixed); `cardiac_core/__init__` made lazy (PEP 562 — `import cardiac_core.ionic` is engine-free, no `_prepare_engine`); `pyproject.toml` + `pip install -e .` make cardiac_core a real importable package (cwd-independent, scoped to `cardiac_core*` — does NOT expose Builder/cardiac_ml/engines). 77 cardiac_core tests green; V5.5 golden still exact (engines untouched).

**Scope pivot (post-audit, 2026-05-30):** the engine rewire+delete was DROPPED to copy-only after the audit found big-bang deletion breaks engine tests/examples AND active cross-project consumers (`Surrogate/surrogate/data/*_generator.py`, `Optimizer/V1/tuner/tissue_runner_bidomain.py` import `cardiac_sim.ionic` via the Bidomain path). User: "don't delete the originals — just copy them over."

**Next Step:** the DEFERRED migration (PLAN.md "Deferred" section) — when resumed, migrate consumers REPO-WIDE (engines' tests/examples + Surrogate datagen + Optimizer + `cv_shared` bare `from ionic`) to `cardiac_core.ionic`, per-consumer with test gates, never deleting out from under a live consumer; exclude V5.3/V5.4/_archive/torchcor from any survivor check. cardiac_core is now editable-installed (engines/consumers gain `import cardiac_core` for free once rewired).

<!-- (continues the Thread section above) -->

### 2026-07-16 (audit round 2): audit-the-fix + completeness critic → 2 more real bugs + 2 masked-data gaps
Ran a SECOND round (4 lanes: gate-correctness, analysis-fixes, P3-validation, and a whole-branch completeness
critic) — the discipline that a fix round which changed code must itself be audited. Fixed (commit `9f387ef`):
- **hiPSC regression I introduced in round 1.** The round-1 conductance allow-list keyed on UPPERCASE `G*/P*`,
  but paci/phas13/mhas13 name conductances LOWERCASE `g_*` → scale_conductance rejected ALL their conductances
  (and regressed `g_Na` scaling that worked pre-fix). Now case-insensitive first-letter + explicit denylist for the
  two dimensionless params that merely start with g/p (`gamma_ncx` NCX-partition, `PkNa` IKs Nernst ratio). Verified
  by enumerating g/G/p/P params across ALL 5 models.
- **Bidomain masked default crash** (completeness critic — the cross-engine symmetry both per-file lanes missed):
  `elliptic_solver='auto'` picked spectral on a hole → cryptic `shape [15,15] invalid for size 216`. Monodomain got
  the DCT gate in round 1; bidomain's parallel path didn't. Auto now falls back to `pcg` on a masked domain (golden-
  safe — full-rect unaffected).
- **P5 silent-wrong on masked/NaN data** (all P5 tests had used clean synthetic tensors): `dominant_frequency_map`
  returned a phantom low freq at NaN holes → now NaN; `radial_cv` silently all-NaN on a dead center → now warns.
- Lesser: `restitution_slope` → LAST descending crossing (alternans boundary on noisy curves); `_rebuild_with_
  conductivity` transactional; cheatsheet count/LBM-record/masked notes.
- **Round-2 VERIFIED SOUND:** the DCT gate is complete (no silent-wrong slips through; allowed set is an exact
  subset of the match-set, residual 7e-15); both round-1 analysis fixes; flat_to_grid guard (all dtype branches);
  LBM guard; the two Phase-3 cross-engine fixes. Accepted/left: B1 CPU-coverage (GPU-only bug, GPU test covers it);
  apd_per_beat/apd_at shared `V_rest=trace[0]`; bidomain scar φ_e pcg_spectral-vs-pcg ~533 mV (PRE-EXISTING elliptic
  accuracy = the deferred 2026-07-02 code-audit M4; Vm/cv/apd unaffected, only φ_e/ECG).
**Convergence signal:** round 2's findings were 1 self-inflicted regression + cross-cutting completeness gaps (not
a new bug class in the core logic); the completeness critic called the per-lane work "genuinely thorough." Suite
**260 passed / 2 xfailed**; goldens bit-identical.

### 2026-07-16 (final audit): 5-lane adversarial audit of the whole branch → 4 more real bugs fixed
After executing all 5 phases, ran a **5-lane parallel adversarial audit** (general-purpose subagents; A=P1,
B=P2, C=P3, D=P5, E=tests+docs) over the whole `main..HEAD` diff. Verdict: default paths sound, but **reachable
non-default configs were silently wrong**. Fixed (commit `c0306d2`):
- **B2 was the worst — my own regression.** Wiring dct/fft removed the TypeError but let users SELECT them in
  configs where they're silently wrong: DCT/FFT ignore the assembled matrix and invert an idealized scalar-D
  Neumann eigen-operator. Measured: anisotropic D → up to **68% CV error**; scar D=0 → **invisible**; bdf2 →
  **CV=nan** (BDF2's BDF1-bootstrap step gets the BDF2 denom → DC/3); fft on any (Neumann) mono mesh → **CV=nan**.
  Fix: `_check_spectral_preconditions` gates dct/fft to iso-uniform + full-rect + face_mirror + cardinal4 + CN/BDF1,
  rejects fft entirely; FDM exposes `_is_iso_uniform`. GOTCHA: the mono factory ALWAYS builds `from_mask` (full-rect
  gets an all-True mask), so the "masked" check must be `not domain_mask.all()`, not `is not None` (first gate cut
  falsely rejected the valid uniform dct path). Default `pcg` untouched; dct on the valid path is exact (Vm≈CN 1e-3).
- **restitution_slope.DI_star** returned the steep short-DI end, not the slope=1 crossing → now interpolates the
  descending crossing. **apd_per_beat** emitted 0.0 for a beat in progress at t=0 → now only measures clean-upstroke
  beats. **_scale_ionic_conductances** accepted any attr (F/T/Cm/concentrations/`*_scale`) → restricted to G*/P*
  non-`*_scale`. **flat_to_grid** NaN-fill guarded for int/bool flats. **LBM** regional set/scale_conductivity now
  raises a clear error (was a misleading "oblique fibers"). **Cheatsheet** `load_result` fixed to the real 4-tuple.
- **Verified SOUND by the audit:** B1 device fix (empty-branch consistent, batch path covered), B3/B4 apd_at exact
  reduction to old behavior on monotonic APs, B5 all four degenerate cases, B6 CFL formula/attrs/non-FDM-skip, B7
  eager both entry points; the two Phase-3 cross-engine fixes (sigma no-op, cell-type flip); every cheatsheet claim
  except load_result (ORd-LBM-only, PCa=ICaL, paci-on-mono, save_result arg order all TRUE).
- **Accepted/left (minor):** B1 CPU test is tautological (real coverage = the GPU-gated test, runs on this box; the
  bug is GPU-only anyway); df_map flat-node→0 Hz (consistent w/ per-node fn); radial_cv all-NaN on a bad center;
  transposed square-mask silent-accept. Documented, not fixed.
Suite after remediation: **254 passed / 2 xfailed**; goldens bit-identical.

### 2026-07-16 (exec): PLAN.md usability fixes EXECUTED — 5 phases, audit-hardened
Cold-started from PLAN.md and ran all 5 phases on branch `usability-fixes-p0-p1`, each implement → targeted test
→ goldens (bit-identical, atol=0) → commit. Details in Current Direction. **Execution catches beyond the plan:**
- **Bidomain masked runs can't use the default spectral elliptic solver** (`SpectralSolver.solve` reshapes to the
  full `nx*ny`; a hole makes `n_dof < nx*ny` → RuntimeError). The B8 bidomain masked test uses
  `elliptic_solver='pcg'`. Documented in the cheatsheet.
- **The Phase-3 adversarial audit was the load-bearing step.** My scale_conductance/set_conductivity tests were
  monodomain-only and passed, but the audit (a general-purpose subagent, since the Opus `/audit` path was rate-
  limited) found 2 REAL cross-engine bugs on the paths I hadn't tested: the declarative-bidomain sigma no-op and
  the cell-type flip. Both fixed + regression-tested (declarative bidomain scar, cell-type-preservation, sigma
  scaling). Lesson (again): a green mono test says nothing about the bidomain/LBM surface — test at the level a
  USER reaches the feature (this is the SAME class as the 2026-07-01 Phase-3 boundary-gap post-mortem).
- **B2 DCT is exact, not approximate:** the DCT solver's denom (`chi*Cm - 0.5*dt*D*λ`) matches the FDM CN operator
  when `D=_D_max` (raw) — so CV matches PCG to tolerance, not just "close". Confirmed the FDM's raw-D convention
  flows correctly into the spectral solve.
- **apd_at B3/B4 fix is regression-safe:** existing `test_analysis.py` synthetic APs are monotonic/dome-free, so
  beat-windowing + dome-aware last-crossing reduce to the old first-crossing there (verified: values unchanged).
**Next:** merge to main (see Next Step). Optional: an independent Opus `/audit` of the whole branch once the rate
limit clears (this session's Phase-3 audit used a general-purpose subagent).

### 2026-07-02 Session — deep code audit (math integrity + API)
**Worked on:** after pushing the API-consistency work to `main` (fast-forward, `abc54db`→`94a2689`), the
user asked for further rounds of audit over ALL of cardiac_core — math integrity per engine, then the API
— as an agentic walkthrough.
**Accomplished:** 6 parallel deep auditors (mono math · bidomain math · LBM math · ionic+conductivity ·
API factories · run/io/analysis/mesh), each verifying NUMERICALLY (not trusting the suite). Verdict **0
blockers / 4 majors / ~22 minors**, written up in [CODE_AUDIT_2026-07-02.md](./CODE_AUDIT_2026-07-02.md)
(committed `619460c`). Orchestrator independently reproduced M1 (Chebyshev 94 mV wrong at stiff effD=0.5/
dt=0.05, benign at physiological) and M3 (bidomain `step()` AttributeError). Verified SOUND: default-path
math on all 3 engines, the prior cross-derivative fix (2·Dxy exact), time-stepper orders, reaction Cm,
LBM Chapman–Enskog + the new `mrt_wall`/masked-bounce, all 4 ionic models (stable rest + physiological AP,
correct APD ordering), the conductivity firewall, analysis math, AND every Phase-0–5 change. 4 majors:
mono Chebyshev (opt-in, wrong Gershgorin operator), mono FFT (opt-in, continuum vs discrete eigenvalue),
bidomain `step()` (pre-existing untested), bidomain per-node-anisotropy non-symmetric elliptic (silent
~13% phi_e). **No source modified** — findings triaged only.
**Next:** if the user wants, action the P1–P4 fix backlog (Next Step above), starting with M4 (symmetrize
the bidomain FDM cross-term). Otherwise the audit stands as a recorded triage. Could also do further audit
rounds (e.g. adversarial re-verification of M2/M4, or a 3D/thickness-path sweep).

### 2026-07-01 (exec): API-consistency hardening EXECUTED — 6 phases, contract-first harness, 217/2
Executed the audit-converged PLAN phase-by-phase, each test+golden-gated and committed. **Contract-first
worked exactly as designed:** Phase 0 wrote all 22 cells as `xfail(strict=True)` (green — expected-fails);
each phase's fix then made its cells XPASS → strict-fail → *forced* me to flip `to_fix→landed` in that same
commit. No cell could be silently left behind. Ended 20 landed + 2 deferred (C2-capability, C7).
**Execution catches (beyond the plan):**
- The round-2 ENDO-vs-EPI warning paid off: the shared ionic builder MUST default `cell_type='ENDO'`
  (verified ORd/TTP06 ctors default ENDO; mono derives from the mesh) — an EPI default would've flipped the
  bidomain+LBM goldens. Shipped ENDO; goldens held every phase.
- `mrt_collide_d2q9`'s `w` is 5th (right after `dt`); `lbm_step_d2q9_mrt_wall` mirrors it exactly (the
  round-2 signature fix was right).
- `_lbm_bounce_masks` had to UNION hole rim ∪ outer rect edges because `precompute_bounce_masks` uses
  periodic `torch.roll` (the B2 finding) — wired to BOTH LBM construction branches.
- Minor self-inflicted: passed a stencil value (`moore8_iso`) as a `boundary_mode` in one named test —
  distinct vocabularies; caught on first run, fixed.
Commits `1a65d3d`(P0)→`40cd2ca`(P1)→`1dda8f6`(P2)→`35327f5`(P3)→`9702bb7`(P4); PLAN archived to
`plans/2026-07-01_api-consistency-hardening-stress-harness.md`; audit docs marked RESOLVED.

### 2026-07-01 (Phase-3 boundary gaps found post-ship): two real gaps + a testing-failure post-mortem → [PHASE3_BOUNDARY_GAPS.md](./PHASE3_BOUNDARY_GAPS.md)
User caught two gaps in the shipped LBM boundary feature (commit `736296d`) that my 17 green tests missed:
- **Gap A — `run_lbm()` doesn't forward `boundary`/`alpha`** (still HBB-only). The params reached only the declarative `lbm()` factory + `simulate(**kwargs)`; the explicit one-shot `run.py::run_lbm` forwards only ionic/dt/lattice/device. Exactly the one-shot-vs-factory asymmetry the audit flagged as #18/#19 (deferred) and re-opened.
- **Gap B — wall modes are BGK-only; anisotropy is blocked.** `lbm(aniso, boundary='ncs')` raises, because anisotropic D forces MRT and a guard requires `collision='bgk'`. But the overlay is POST-STREAM → collision-agnostic; the guard was over-conservative and `test_lbm_rejects_oblique_Dxy` *codified* the over-restriction (conflating oblique `D_xy≠0` (real limit) with per-axis anisotropy (should work)).
**Root cause (post-mortem):** tested/audited the feature at the level it was BUILT (`lbm()` factory, isotropic BGK), not at the level a USER reaches it (`run_lbm`/`simulate`; anisotropic fibers). The contract (entry points × physics) was never written down, so nothing checked against it — and the 3-round plan-audit couldn't catch NEW-feature under-specification (it checks the plan vs the 46 existing findings, not surface-coverage of a not-yet-written feature). **Fix approach (settled, in the gaps doc):** (A) add `boundary`/`alpha` to `run_lbm`; (B) add `lbm_step_d2q9_mrt_wall`, remove the bgk-only guard, dispatch mrt+special→mrt_wall, keep the oblique-`D_xy≠0` rejection. Tests must span `{lbm, run_lbm, simulate} × {isotropic-BGK, per-axis-aniso-MRT} × modes`. → blueprint the fix.

### 2026-06-30 (system audit + cleanup decisions): adversarial audit of cardiac_core + cardiac_mcp + docs → [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md)
Ran an 8-lane adversarial multi-agent audit (find → refute → completeness; 64 agents, ~3.1M tokens) over the shipped surface, to tidy the foundation BEFORE building the boundary-mode work (which the PI has now adopted as REAL, β/discreteness-driven, unreported physics — stop treating the α-blend/specular speedup as artifact). **55 raised → 46 confirmed** (4 HIGH, 11 MED, 27 LOW, 4 INFO), 9 refuted, 12 completeness gaps. Full report: `CARDIAC_CORE_AUDIT.md`.
**Standout:** a real numerics bug (#4, HIGH) — the monodomain FDM anisotropic cross-derivative has the WRONG SIGN and HALF MAGNITUDE vs the validated bidomain builder (`cxy=1/(4dxdy)` should be `1/(2dxdy)`; diagonal signs flipped) → oblique-fiber runs quantitatively wrong, untested. Directly on the anisotropic-boundary path; my own boundary audit missed it.
**Systemic cluster:** the `CardiacMeshData` chi/D ambiguity (#2/#8/#21) — same mesh runs at ~1400× different D across engines; default `create_cardiac_mesh(D=0.001, chi=1400)` is conduction-blocked. The firewall has a hole at the file-format layer.
**DECISION (user, 2026-06-30): `CardiacMeshData.D_xx` = RAW conductivity-like; `chi·Cm` divides in EVERY engine → physical effective D = `D_xx/(χ·Cm)` uniformly.** Monodomain FDM already does this (χ·Cm in mass term = the "correct" one); **the LBM factory (api.py:1375-1395) currently passes `D_xx` straight through as effective and MUST be changed to divide by χ·Cm first**; bidomain already divides. **Corollary:** the `create_cardiac_mesh` DEFAULTS must also change (raw-σ-scale default D, or default `chi=1.0`, or a guard) — "raw everywhere" makes the convention consistent but leaves the blocked default. **DECISION (user): blueprint the whole cleanup first** (blockers → tidy → boundary build), no code until the PLAN is reviewed. Sequence: fix #4 + cluster-#1 + #1 (ionic-override lost on reset) + #3 (MCP path traversal) BEFORE the boundary-mode build; fold #29/#37/#38/#14/#39/#40 INTO the boundary build (they are its surface); batch the dead-code/docs tidy; the 12 completeness gaps are the next audit round.

### 2026-06-30 (API-debt finding): `create_cardiac_mesh` BYPASSES the Formulation-A/B firewall → silent ~1400× D mis-scale
Surfaced while implementing the `ionic_model_optimization` ↔ chip-fit cross-plan (tuning cardiac_core to Kit Parker chip EP). **Symptom:** building a mesh with `create_cardiac_mesh(D=<effective diffusivity ~1e-3>)` at the **default `chi=1400`** gave NO propagation on every engine config — the stimulus pooled the source nodes to a non-physical **Vmax ≈ 80–123 mV** with zero downstream activation (CV=NaN). Isotropic *and* anisotropic; not the anisotropy/MRT work — a mesh-assembly issue. `chi=1.0` → clean propagation, CV=59 cm/s.
**Mechanism (verified in `_monodomain/.../fdm.py:37,159`):** the FDM operator solves `χ·Cm·∂V/∂t = ∇·(D·∇V)` — the stiffness Laplacian is built from `D` alone, `χ·Cm` sits only in the mass/time term, so the **membrane-effective diffusivity is `D/(χ·Cm)`**. An effective `D≈1e-3` with `chi=1400` → effective ≈7e-7 (1400× too low) → CV ∝ √D drops ~37× → below the discrete source–sink launch threshold at chip `dx=0.01` (the space constant shrinks below `dx`) → conduction block. This is *faithful physics of the wrong number*, not a solver bug: `(D=1e-3, χ=1400)` is exactly degenerate with `(D=7.14e-7, χ=1)`.
**The API debt (this is the report):** we built the **Formulation-A/B firewall** (`conductivity.py`, ConductivityConfig) precisely to prevent σ-vs-D_eff confusion (see 2026-03-16 / 2026-06-24 entries) — but `create_cardiac_mesh` is a **second, unguarded entry point** that re-exposes raw `D` + `chi`, and its default (`chi=1400`) contradicts its own docstring ("D : diffusion coefficient cm²/ms"). So the trap the firewall closed is reopened by the convenience builder.
**Recommended fix (pick one):** (a) docstring warning — DONE (`file_format.py` + `ionic_model_optimization/KNOWLEDGE.md`); (b) default `chi=1.0` + treat `D` as a true diffusivity — *breaks* the ConductivityConfig/declarative callers that pass conductivity-style with `chi=1400`; (c) add a non-breaking `mode={'diffusivity'|'conductivity'}` flag that validates the `D`/`chi` pairing; (d) route `create_cardiac_mesh` through `ConductivityConfig` so there is ONE firewall. **Recommend (c) or (d).** Mitigated in the chip pipeline already (`cc_runner`/`chip.chip_mesh` pin `chi=1.0`). LBM unaffected (maps D→τ directly, no χ division).
**Empirical (2026-06-30, TTP06, dx in cm):** (1) **Degeneracy CONFIRMED** — `(D=1e-3, χ=1400)` and `(D=7.14e-7, χ=1)` are bit-identical (Vmax 81.6, CV=None, both blocked at dx=0.01). χ is a pure rescaling in monodomain. (2) **The block is DISCRETIZATION, not "merely slow"** — the same effD=7e-7 at 4× finer dx=0.0025 *does* propagate (CV 0.72 cm/s); at dx=0.01 the ~37× thinner front is unresolved → block. (3) **The REAL chip regime is FINE** — effD=2.5e-5 (NRVM target) propagates cleanly at dx=0.01, CV≈6.06 cm/s (~36× above the block threshold); no artificial block. The tuner's secant fit dials D to the exact target (9.33). Minor caveat: at chip-slow CV the stim-site upstroke runs a bit hot (Vmax~81 vs ~50) at dx=0.01 — finer dx / the points-per-λ guard cleans it up if needed.

### 2026-06-28 (MCP audit + blueprint): standardized vs spec 2025-11-25 → 4-tier PLAN
After shipping `cardiac_mcp` (2026-06-26), the user flagged that a working server isn't a *standardized* one — they intuited "a list of supporting documents" an MCP server needs and wanted an audit against proper guidelines (not my from-memory build). Ran **4 parallel spec-research agents** against the OFFICIAL spec (modelcontextprotocol.io + schema.ts), confirmed current revision **2025-11-25**. Key learnings: (1) **Tool annotations** are the headline gap — unset → spec defaults make ALL our tools advertise `destructiveHint=true/openWorldHint=true`, dishonest for the read-only ones and under-flagging the code-runner; set `readOnlyHint`/`destructiveHint=false`/`openWorldHint=false` per tool. (2) `serverInfo.version` falls back to the SDK version (1.28.0) unless set. (3) **outputSchema/structuredContent** SHOULD back structured dict returns (type the returns). (4) Errors: recoverable → `isError` tool-result (FastMCP already does this for raised exceptions); reserve JSON-RPC errors for protocol faults. (5) **Two path-traversal bugs** (MUST validate inputs): `run_experiment` `(REPO_ROOT/experiment_dir).resolve()` escapes on absolute/`..`; `commit_experiment` uses unsanitized `date` in the folder name. (6) Distribution layer = `server.json` (reverse-DNS name, immutable semver), README + ownership marker, `pyproject` console-script, LICENSE/Dockerfile — REQUIRED only to publish to the registry. (7) **stdio→HTTP is a big delta**: OAuth 2.1+PKCE(S256), RFC 9728 PRM, RFC 8707 resource indicators, `Origin`→403, secure session IDs (never auth-via-session), SSRF defenses, no token passthrough; AND the spec SHOULD-sandboxes a code-executing tool (our `run_experiment` remote-without-sandbox = RCE-as-a-service). All audited features (annotations/outputSchema/mime) are available in the installed SDK 1.28.0 (they predate 2025-11-25). Synthesized into a 4-tier remediation order (T1 now → T4 optional) → `/blueprint` → PLAN.md.

### 2026-06-25 (Goal 2 design): skill suite for wet-lab scientists — settled, ready to blueprint
**Audience REFRAMED** (corrected the README "non-coder conversational builder" wording twice): target = WET-LAB scientists (cell culture, tissue-on-chip / lab-chip) WITHOUT computational-simulation exposure. Goal 2 = a SKILL SUITE that **lowers coding complexity by GENERATING runnable `cardiac_core` scripts** (code-gen, not an interactive wizard; no auto-teaching). Drives the shipped `cardiac_core` API directly — Layer-A `SimulationSpec`/`create_simulation` DEFERRED; programmatic claude-api comes later.

**The suite (user's order):**
1. `/sim-experiment` — free-form description → runnable `cardiac_core` script. (KEYSTONE)
2. `/sim-preset` — save / store / reuse named parameter sets.
3. `/sim-media` — standardized figures & videos (canonical `media/`).
4. `/sim-notebook` — lab-notebook organization (master log + per-experiment folders).

**`/sim-experiment` protocol (settled with user):**
- RECEIVE free-form input (any shape — a sentence, a paragraph, a chip protocol).
- INTERPRET → build `cardiac_core` params (infer engine + map to API); ask ONLY for genuine gaps.
- MANIFEST → present a plain-**TEXT** summary of ALL params: goal, engine (+why), ionic model, geometry (Nx/Ny/dx), tissue (σ/χ/Cm), **delivery/stimulus method**, **sim length** (t_end/dt/save_every), measure, outputs, script path.
- ⛔ **DOUBLE-CHECK GATE** — scientist confirms or corrects; the skill **NEVER runs without it**. THE accountability principle (user: "no crazy vibe coding runoff").
- ON "GO" → create a **dedicated experiment folder**, write `MANIFEST.md` (the confirmed text, verbatim = the record) + `run.py`, append a one-line entry to the **master log**.
- RUN (offer) → verify results are sane, save standardized media, write results back to manifest/log.

**Folder structure:** `Lab/` (new top-level) — `Lab/NOTEBOOK.md` master log + per-experiment `Lab/{date}_{slug}/` (`MANIFEST.md`, `run.py`, `outputs/`). Skills 1 & 4 share this home (the notebook organizes itself as each run drops a folder + a log line).

**Manifest fields:** the listed ones are core; extra fields (scientist/initials, hypothesis, expected runtime) are **OPTIONAL** (user: "make it optional") — include when relevant, never required.

**Key asset = a MAINTAINED `api-cheatsheet`** (current, correct `cardiac_core` calls) in the skill bundle. This is what prevents the #1 LLM-sim-code failure mode (hallucinated API). Refresh it now that `cardiac_core` just shipped (137 tests). The existing `API_REFERENCE.md` predates the consolidation and is design-oriented — distill a CURRENT cheatsheet from the shipped API (`cc.Grid`, `cc.ConductivityConfig`, `cc.monodomain/bidomain/lbm`, `cc.simulate`, `result.run().cv()/.apd()`, `cc.media`).

**Skill format:** `.claude/skills/{name}/SKILL.md` + a `reference/` folder — the FIRST *bundled* skill in this repo (all existing skills are single `SKILL.md`). Output = `.py` script (notebook option left open). Build ON existing conventions: `media/` + `cardiac_core.media.media_path`, the experiments pattern, the `Research/` doc architecture.

**Next:** `/blueprint` the build — phased: api-cheatsheet → `/sim-experiment` (keystone) → `/sim-preset` → `/sim-media` → `/sim-notebook`. (User: "think we should blueprint.")

### 2026-06-25 (exec): consolidation EXECUTED — cardiac_core self-contained (Phases 0–5, 137 tests)
Ran the vendoring plan phase-by-phase, each test+integrity-gated and committed. Backup first (tag `pre-consolidation-vendoring` + 739M bundle). Result: `cardiac_core` owns all 3 engines (`_monodomain`/`_bidomain`/`_lbm`) + shared `ionic`/`mesh`/`stimulus`; hack deleted; 137 green.

**Two real bugs hit + fixed during execution (worth remembering):**
1. **Cross-ref rewrite regex corrupted internal imports.** First pass used `from \.+(ionic|tissue_builder)` WITHOUT `\b` → matched the `ionic` prefix of `ionic_time_stepping` → rewrote 4 solver-INTERNAL imports to `cardiac_core.ionic_time_stepping` (broken). Caught immediately (source untouched), re-copied clean, re-ran with `\b`-anchored regex → exactly 8 cross-refs, internals intact. Lesson: ALWAYS `\b`-anchor `ionic`/`tissue_builder` in the rewrite.
2. **Solver package name shadowed the factory.** Named it `cardiac_core/monodomain/` → `from cardiac_core import monodomain` returned the PACKAGE (real submodule beats the lazy `__getattr__`), so `monodomain(mesh)` → `'module' object is not callable` (ordering-dependent: only failed once the package was imported). Fix: underscore-prefix the solver packages (`_monodomain` etc.) so they don't collide with the public factory names. This is exactly the "don't import _* directly" design I'd floated; now enforced.

**Other notes:** the self-containment guard initially false-positived on PROSE (comments/docstrings saying "no `_prepare_engine()`") — refined to match the call form `_prepare_engine(` with inline-comments stripped. `stimulus/protocol.py` reconciliation = bidomain's `+=` (canonical; the V5.5 `=` overwrite differs only for overlapping stims — goldens single-stim so bit-identical). The `conda run python - <<heredoc` inline form silently no-op'd the rewrite (stdin issue) → wrote the rewrite script to a file and ran it (per CLAUDE.md temp-file guidance).

### 2026-06-25: DECISION — consolidation = unified ground-up package (Approach A2), source-verified
After the Goal-1 API shipped, did a "final alignment" with the user on what cardiac_core actually is. Confirmed it's a WRAPPER that references the 3 engines from `Monodomain/Engine_V5.5/`, `Bidomain/Engine_V1/`, `LBM/Engine_V1/` via `_prepare_engine()` (sys.modules flush + sys.path swap) — NOT a single unified codebase. The `_prepare_engine` hack exists because V5.5 and Bidomain BOTH name their top package `cardiac_sim` (collision); LBM uses `src`.

**Key source finding (reshaped the whole plan):** the 3 engine trees are **100% relative-import internally with ZERO gotchas** — `grep` over all of them: 0 absolute `cardiac_sim`/`src` imports, 0 `sys.path`, 0 `importlib`, 0 `__file__`, 0 name-as-string-literal. (V5.5 85 .py / 186 relative-import lines; Bidomain 78 / 165; LBM 25 / 17.) So the engines are hermetic, relocatable bricks. My earlier estimate that removing the hack meant "~70+ absolute-import rewrites" was WRONG — relative imports are name/location agnostic, so renaming/moving a tree changes nothing inside it. The solver code reaches OUT to shared code only ~7×/engine (`from .....ionic`, `from ....tissue_builder` in V5.5's `simulation/`); everything else is solver-internal (`from ..base`, `from ....state`).

**Decisions (user, 2026-06-25):**
- **Approach A2 — unified, flat, ground-up.** NOT Approach B (relocate-but-keep-hack, leaves nested `cardiac_core/engines/X/cardiac_sim/...` blobs + packaging-exclusion). NOT rename-only-A1 (silos with triplicated shared code). User: "I wanted a unified simple ground up end product not nested reference and import."
- **`cardiac_sim` DISSOLVES** (user OK'd renaming mono+bidomain's package). No `cardiac_sim`, no `engines/` nesting. Shared parts → top-level siblings; solver parts → `cardiac_core/{monodomain,bidomain,lbm}/`.
- **Unify ionic + mesh + stimulus** this pass (the "do the first" option). `ionic/` already extracted (canonical copy from Phase-1). `mesh/` needs a SUPERSET (bidomain `StructuredGrid` adds `boundary_spec`; mono's doesn't). `stimulus/` already aligned (all accumulate `+=`). `tissue`/conductivity *internals* stay per-engine (solver-specific, already fronted by `ConductivityConfig`).
- **Copy, don't delete.** The 3 engine folders (`Monodomain/Engine_V5.5/` etc.) stay untouched as-is (their own dev/tests). cardiac_core gets the unified copy. Drift is the accepted cost; note a re-vendor strategy.
- **Delete `_prepare_engine()`** — once each piece has a unique `cardiac_core.*` dotted name there's no collision, so the hack and the packaging-exclusion both go away. Both engines become importable simultaneously (normal Python).

**Target layout** (audit-ready): `cardiac_core/{ionic,mesh,stimulus}` (shared) + `cardiac_core/{monodomain,bidomain,lbm}` (slim solvers) + the existing api/run/conductivity/grid/simulation/analysis/geometry/io/file_format/media. Internal imports: solver-internal stay RELATIVE (untouched as the subtree moves intact); the ~10–20 solver→shared cross-refs rewrite to absolute `from cardiac_core.{ionic,mesh,stimulus}...`. `api.py` swaps the `_prepare_engine + from cardiac_sim...` blocks for direct `from cardiac_core.{monodomain,bidomain,lbm}... import ...`.

**Risk (bounded):** step that merges the slightly-divergent shared modules — only `mesh` (StructuredGrid superset incl. `boundary_spec`) is non-trivial; ionic done, stimulus aligned. FEM/TriangularMesh: keep as-is during the move (FEM-removal is a SEPARATE confirmed-but-deferred cleanup; don't entangle). **Verification each phase: all 121 cardiac_core tests green + a guard that no `cardiac_core/**` file references `Monodomain/`,`Bidomain/`,`LBM/` paths or the `_prepare_engine` hack.**

**DECISION (user, 2026-06-25): `cardiac_core` is the CENTRALIZED home.** Future engine improvements happen in `cardiac_core/{monodomain,bidomain,lbm}` — the original engine folders (`Engine_V5.5`, `Engine_V1`×2) become frozen/legacy. This resolves the only real objection to copy-vendoring (drift): there is no drift if the vendored copy is the single living source going forward. The vendoring consolidation is greenlit. Also confirmed: public API is engine-as-parameter (`cardiac_core.simulate(engine='lbm')` / the `monodomain()`/`lbm()` factories) — the `cardiac_core.monodomain` etc. subpackages are INTERNAL plumbing the user never imports (mark them private). Optional thin add: a single `build(engine=…)` live-sim entry.

**Next:** `/blueprint` [DONE — PLAN.md 2026-06-25] → `/audit` → execute per-phase (test-gated) → audit the final folder structure.

### 2026-06-24 (impl): PLAN.md executed — Goal-1 construction API shipped (Phases 0–5, 121 tests)
After a 3rd `/audit` pass on PLAN.md (5 findings folded in: the HIGH `save_result` positional-`phi_e` break + 2 MED missing-section + 2 LOW), executed all 6 API-track phases end-to-end. Each phase: implement → targeted test → full-suite gate. No engine source touched (V5.3/V5.4/V5.5/Bidomain/LBM unchanged); cardiac_core grew `conductivity.py`, `grid.py`, `simulation.py` + refactored `api.py`/`run.py`/`io.py`. Result: **121 cardiac_core tests pass** (80→121), incl. the live-CV firewall gate.

Key implementation decisions / gotchas (beyond the design):
- **`ConductivityConfig.sigma_eff` is the PUBLIC property** (per API_REFERENCE), so the isotropic stored conductivity lives in a `sigma_iso` field (the plan's sketch had a `sigma_eff` field + `sigma_eff_value` property — would clash). Arithmetic mirrors the probe exactly.
- **Live-CV gate runs in a SUBPROCESS** (`tests/_live_cv_gate_driver.py`) — running test_phase10's V5.5 cable inside the cardiac_core pytest session would collide on the shared `cardiac_sim` namespace (flushed by `_prepare_engine`). Subprocess isolates it; ~2 min, skips cleanly if the V5.5 dir / ref JSON is absent. (First run hit a `numpy.bool_` not-JSON-serializable bug → cast `bool()/float()`.)
- **Bidomain σ-tuples must be `(Nx,Ny)` FIELDS, not scalars** — the bidomain FDM indexes `dxx[i,j]`; passing scalar σ gave a 0-d-array IndexError. `_build_mesh_data` now emits `np.full((Nx,Ny), σ)` tuples.
- **`stimulate()`/`reset()`/`with_()` unified across both construction paths** by routing ALL stimuli through `data.stimuli` and replaying the factory with `mesh=self._data` + stored `_build_kwargs`. This made the audit-MEDIUM "must work for both paths" trivially true and sidestepped the LBM `start`/`start_time` audit-LOW (the existing LBM factory loop already reads `data.stimuli` positionally).
- **`run()` eager flip needed a 34-site migration** `*.run(`→`*.snapshots(` across 6 test files + the production `run.py::_collect` (engine-direct `sim_direct.run` left alone). Done with a word-boundary regex script.
- **CV smoke needed t_end=40ms** (front ~50 cm/s = 0.05 cm/ms; x2=1.0cm activates ~21ms) — at t_end=20 the far probe never activated → CV=nan.

OPEN (handed back to user): git — all work is uncommitted on `main` alongside pre-existing unrelated changes (source_sink, MASTER.md, the design docs). Per harness rule I did NOT auto-commit to the default branch. The plan's per-phase commit points are ready to apply once a branch/commit strategy is chosen.

### 2026-06-24: Unified API drafted — `Simulation` Protocol + 4 idioms + `SimulationSpec`
Picked up from the glossary's 3 open items and produced **`API_DESIGN.md`** (Goal-1 interface in the resolved vocabulary). Settled the open items + drafted the interface.

**Open glossary items resolved:**
- **#9 default stim amplitude → −52** (user). Rationale: ionic model is byte-identical across engines and the stimulus enters the same `R=-(Iion+Istim)/Cm` term in the same units → an amplitude that depolarizes in M/B depolarizes identically in L. LBM's −80 was author drift; retire it. The glossary's "verify under L" is automatically satisfied.
- **#5 internal live-State → unify + defer.** One internal `State(t, Vm, ionic_states, …, Cm, coords)` with optional `phi_e` (bidomain) / `f` (LBM); LBM adopts it. Zero public-contract impact (only `SimulationResult` is user-facing) → land in a code phase, not now.
- **#12 ConductivityConfig interface — drafted, then source-verified + CORRECTED.** The class stores **physics** (`sigma_i/sigma_e/sigma_eff, chi, Cm, fiber_angle`) and emits per-engine inputs. Construction via classmethods `.isotropic/.bidomain/.anisotropic`. **#13 (chi only in ConductivityConfig) makes #12 work.**

### 2026-06-24 (cont.): VERIFIED ConductivityConfig vs source — caught a Cm≠1 bug in my own draft
Read the actual operators instead of trusting the KNOWLEDGE summary. Confirmed:
- `fdm.py:195–238` (V5.5): implicit solve `(χ·Cm·I − ½dt·L)Vⁿ⁺¹ = (χ·Cm·I + ½dt·L)Vⁿ`, `L` built from input `D` (NOT χ·Cm). ⟹ **physical diffusivity = `D_input/(χ·Cm)`** (Form A confirmed). Reaction divides by `state.Cm` (V5.5 fix, test_phase10 @3.55e-15).
- `BidomainConductivity` (`conductivity.py`): `D_i,D_e` PRE-scaled `=σ/(χ·Cm)`; has `get_effective_monodomain_D()=D_i·D_e/(D_i+D_e)` (the harmonic i/e collapse — the `D_eff` reduction). LBM `sigma_to_D`: `D=σ/(χ·Cm)` pre-scaled. (Form B confirmed.)
- **BUG in my first §4 draft:** "feed `D_eff` with `chi=1, Cm=1` no-op" is correct ONLY at Cm=1. At Cm≠1, pinning the engine's `Cm=1` makes the **reaction** divide by 1 instead of the real Cm → silently wrong. **Same Cm-trap family as the false time-dilation invariant** (invisible at pinned Cm=1, bites otherwise). The real Cm must reach EVERY engine.
- **Corrected mechanic:** only the *diffusion input's* Cm-scaling differs by formulation. Form-A monodomain scales diffusion by Cm internally (mass term) → feed it Cm-**un**scaled `D = D_eff·Cm = sigma_eff/chi`, with engine `chi=1` (chi folded in) and the **real Cm** (drives mass term + reaction). Form-B (bidomain/LBM) → feed fully-scaled `D = σ/(χ·Cm)` + real Cm. ConductivityConfig now exposes per-engine emitters `for_monodomain()/for_bidomain()/for_lbm()` so this arithmetic lives in ONE place. At Cm=1 all collapse to `D_eff = sigma_eff/chi` (= what the 2026-05-30 cross-engine test used: `D=D_EFF, chi=1`).
- **Also fixed a units trap in the §7 smoke test:** `ConductivityConfig.isotropic(sigma=...)` takes raw CONDUCTIVITY (mS/cm), not pre-divided `D`. `0.00097` is the *D_eff*, not σ. Standard tissue σ_i=1.74, σ_e=6.25 → D_eff=0.000972.
- **Build-time gate (still open):** confirm `for_monodomain()` reproduces test_phase10 CVs at Cm∈{1,2} once coded.

### 2026-06-24 (cont.): DECISION — canonical formulation = B (converge in Phase 4)
User asked: now that both A and B are physically correct, which is "good"? Reasoned it through: **the V5.5 reaction fix made the physics a TIE, so this is now a pure software-engineering decision** — and B wins on every axis: (1) consolidation alignment — B confines all χ/Cm scaling to `ConductivityConfig` (decision #13); A's engine is a *second* scaling authority (its χ·Cm mass term); (2) non-fragile — A scatters χ·Cm across FDM mass + FEM M + FVM Vol + DCT/FFT denominators + reaction (that scattering caused the V5.4 bug); B has one σ→D line; (3) majority — 2/3 engines already B; (4) clean operator `(I−θ·dt·L)` vs A carrying χ·Cm=1400 into the linear algebra. A's only edge (operator reads like the PDE) is a *docs* value, neutralized at the API (user passes σ/χ/Cm to ConductivityConfig either way).
**DECISION (user): Form B target; converge in Phase 4.** Two-phase: keep both now (firewall `for_monodomain()` absorbs the asymmetry); convert monodomain Form-A diffusion→B *as part of* the Phase-4 rewire into cardiac_core (no new fork) → then DELETE `for_monodomain()`, ConductivityConfig collapses to one emitter (physical D + Cm). Recorded in KNOWLEDGE Key Decisions ("Canonical formulation = B") + migration-plan Phase 4 + API_DESIGN §4 + glossary.

### 2026-06-24 (cont.): GATE CLOSED + FEM ditch CONFIRMED + API reference doc
- **ConductivityConfig firewall gate — CLOSED (numerically).** Wrote `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py`: raw `sigma_i=1.74, sigma_e=6.25, chi=1400` → `for_monodomain()` → live V5.5 cable (reuses `test_phase10.run_cable_v55`). Result: arithmetic `D=0.0009721973895941` = reference `D_EFF` to **1.1e-19** (Cm-independent ✓); CV(Cm=1)=**54.35** (0.00% vs bidomain ref), CV(Cm=2)=**28.09** vs 27.77 (**1.15%** < 5%). The Cm≠1 firewall path is correct in the live engine, not just on paper. (Probe is a keep-or-toss artifact; the permanent test lands in `cardiac_core/tests` when ConductivityConfig is built in Phase 3.)
- **FEM ditch — CONFIRMED (user).** Structured grid is now the ONLY standard (P2→P2′). Drops the unstructured/flat-`(n_dof,)` geometry path, `TriangularMesh`, monodomain's `FEMDiscretization`. **FDM primary; FVM survives** (structured-grid-native); collapsing FVM→FDM is a SEPARATE later question. Composes with the Form-A→B convergence in the Phase-4 rewire. API_DESIGN §9 marked CONFIRMED.
- **Deliverable: `API_REFERENCE.md`** — library-style reference (every class + function, signatures, params, returns, examples). Built from API_DESIGN.

**New decisions (user, 2026-06-24):**
- **CHANGE idiom = functional `sim.with_(**overrides)` → new Simulation** (immutable, sweep-safe; no mutable setters in the public API — clean for Optimizer).
- **Construction = factories + spec, layered.** Per-model `monodomain()/bidomain()/lbm()` factories (programmer surface) AND `create_simulation(SimulationSpec)` on top (LLM-intake surface). Spec→factory; building the spec now keeps the questionnaire from drifting from engine needs ("spec schema = the intake questionnaire" — the cross-goal leverage point).

**`API_DESIGN.md` structure:** §0 four idioms · §1 `Simulation` Protocol · §2 factories · §3 stimulus · §4 ConductivityConfig (the chi/Form-A/B firewall) · §5 `SimulationResult` (only public output; eager + `batch=`) · §6 `SimulationSpec`/`create_simulation` (3 tiers required/defaulted/derived; Goal-2 bridge) · §7 minimal-spec smoke test · §8 open/deferred · §9 the FEM-ditch pending note.

**FEM-ditch flagged (user, discuss later):** all sims on structured grid → P2 strengthens to "structured ONLY." Drops the unstructured/flat path, `TriangularMesh`, FEM knob; simplifies State #5; touches engine implementation (Phase-4). Recorded as pending in `API_DESIGN.md` §9, not yet committed.

**Next:** discuss the FEM-ditch (its knock-on for engine implementation) → then either (a) finalize `ConductivityConfig` class shape against source (verify the chi=1 no-op), or (b) start turning `API_DESIGN.md` into the actual `cardiac_core` `Simulation`/Protocol + factory code.

### 2026-05-31: Session vision — unified API + LLM wrapper → conversational simulation builder
North star (now the question's main goal, see README). A non-coder converses with Claude to build cardiac sims and learn how conduction works. Two goals:
1. **Unified construction API (Goal 1)** — one standardized, engine-agnostic, easy-to-construct way to declare + run: a declarative, validated, serializable **SimulationSpec** → run → **SimulationResult** → analysis. Consolidates today's split config (`CardiacMeshData` fields + `simulate()` call-args) into ONE object. Three field tiers: **required** (LLM asks) / **defaulted** (silent good values) / **derived** (computed).
2. **Self-contained LLM wrapper (Goal 2)** — Claude skills + reference docs driving Goal 1 under a strict protocol (gather → validate → construct → run → verify → present).

Key design insights (settled-ish; revisit when building Goal 2):
- **Spec schema = the intake questionnaire.** Make spec fields self-describing (`{required?, prompt, options, default}`); the LLM "gather" step = ask the prompt of each unfilled required field. The questionnaire can't drift from what engines need (same schema). THE cross-goal leverage point.
- **Pacing abstraction.** High-level protocol (`single` / `s1s2` / `regular(bcl, n_beats)`) that EXPANDS into the low-level stimulus list engines consume. Non-coder speaks in beats, not timestamps.
- **Outputs drive the run.** What the user wants to MEASURE (CV / APD / LAT / reentry) feeds back into numerics/run (`save_every`, `t_end`), not just post-hoc analysis.
- **Engine = explicit in spec, but LLM-inferred from the scientific question** + records rationale (auditable, overridable). e.g. bath/boundary effects → bidomain; fast/simple → monodomain.
- **Defaults philosophy.** A minimal spec ("pace this sheet, measure CV") must RUN via physiological defaults (TTP06/EPI, dt=0.02, strang, CN/pcg, chi=1400, Cm=1) — "one obvious way".

Deferred: user geometry input (Fiji drawing → Builder image→mesh; a designated drawings inbox; the Fiji-export→mask format contract). Assume geometry is provided for now.

**Decided focus (2026-05-31):** build the foundational **API** FIRST — everything (the spec questionnaire, the LLM wrapper) is contingent on a clean, standardized construct + run + results surface in `cardiac_core`.

### 2026-05-30: Phase 1 scoped — ionic is ~unified across engines; full direct migration decided
Verified the plan's "ionic identical across engines" assumption before executing. Findings:
- **Classical engines (V5.5 ↔ Bidomain V1): shared model files byte-identical** — `base.py`, `lut.py`, `ttp06`, `ord`, `mhas13`, `phas13`. V5.5 only adds `paci` + `__init__` exports.
- **LBM V1 ionic is NOT a fork** (initial diff misled): same `IonicModel` ABC (base.py byte-identical), same `ttp06/` structure (calcium/celltypes/currents/gating/model/parameters), `ord/model.py` byte-identical. Differs only by: (1) one keyword rename `cell_type_is_endo` (V5.5) vs `celltype_is_endo` (LBM) in a lut call; (2) top-level `ionic/` namespace vs `cardiac_sim.ionic`; (3) model subset (ttp06+ord only); (4) a dead stray `LBM/Engine_V1/ionic/ionic/` (imported by nothing).
- **Rewire surface:** ~23 consumer sites (14 V5.5, 9 Bidomain V1) + LBM's, all relative imports of varying depth (`from ...ionic`, `from .....ionic.base`).

DECISIONS (2026-05-30):
- **Scope: all three engines** (divergence is trivial, not a fork). cardiac_core/ionic/ = canonical SUPERSET (include paci/mhas13/phas13; reconcile the one keyword rename to a single canonical name).
- **Strategy: Option B — direct rewire, EXACT migration. NOT shims, NOT a sys.path trick.** User directive: "exact migration of all engines, not as path." Engine-local ionic copies DELETED; all consumers import `cardiac_core.ionic.*` absolutely; cardiac_core made a properly importable package (editable install / real package), not per-engine `sys.path.insert`.
- **Sequencing: 1a** build canonical `cardiac_core/ionic/` superset → **1b** rewire+delete classical engines (V5.5 + Bidomain V1), run their suites → **1c** rewire+delete LBM V1 (handle top-level namespace, keyword, drop stray ionic/ionic), run its suite. Verify cardiac_core 77 + every engine suite green.
- Rejected: re-export shim (A) and A-then-B staging — user wants the clean end-state directly.

Baseline before Phase 1: cardiac_core 77/77 pass; classical/LBM suites green (this session).

### 2026-03-16: The core tension is engine-centric vs. research-centric layout
Engines serve multiple research questions, but the directory structure forces navigation by engine. A single experiment may touch Bidomain V1 and Monodomain V5.4, yet there is no natural place for it. Proposed restructuring: `Engines/` top-level with `cardiac_core/` as the shared package, `Pipelines/` for optimizer/surrogate/builder, `Research/` for writing only.

### 2026-03-16: Three proposed structures, iterated to final form
First proposal grouped everything under `Engines/` and `Pipelines/`. User pushed back on groupings. Second revision separated concerns more cleanly. Third iteration established the principle: Research = writing (no .py files), Engines = code, cross-linked via MASTER.md and EXPERIMENT.md backlinks.

### 2026-03-16: Identified the experiment gap
Traced the research cycle (Hypothesis -> Script -> Run -> Outputs -> Analysis -> Finding -> Knowledge) and found each step lived in a different location with no links between them. Scripts lived in engine test directories, outputs were ephemeral, and analysis was manual. Solved by adding `experiments/` directories inside engines with EXPERIMENT.md backlinks to research questions.

### 2026-03-16: Chi/Cm audit revealed two valid formulations
Audited chi/Cm handling across all three engines. Found Formulation A (V5.4: chi*Cm in mass term, ionic solver does NOT divide by Cm) and Formulation B (Bidomain V1, LBM V1: D pre-scaled, ionic solver divides by Cm). Both produce identical results when Cm=1.0. Decision: keep both, unify at the API level. Converting V5.4 would risk 77 tests for no practical benefit since Cm is always 1.0 (ionic models output pA/pF).

### 2026-03-16: Diffusion tensor encoding differs by method but is reconcilable
Mapped what each discretization method receives (FDM 5pt gets Dxx/Dyy, FDM 9pt gets full tensor, LBM D2Q5 gets scalar, D2Q9 MRT gets full tensor). ConductivityConfig can be the single entry point: user provides sigma, it converts to D with chi/Cm in one place, then each engine extracts what it needs.

### 2026-03-17: Phase 0 completed — API wrapper with 34 tests
Built `cardiac_core/` as an API wrapper: `monodomain()`, `bidomain()`, `lbm()` functions return `CardiacSimulation` with `.run()` generator. File format: `CardiacMeshData` dataclass with `.npz` save/load. Verified wrapper output matches direct engine construction exactly (atol=1e-10). The `_prepare_engine()` hack flushes `sys.modules` because both engines use `cardiac_sim` namespace. This is temporary; goes away in Phase 1.

### 2026-03-17: Code duplication inventory — 15+ files across 3 engines
IonicModel ABC + lut.py (3 copies, identical), TTP06 5 files (3 copies, identical), ORd 6 files (3 copies, identical), PCG solver (2 copies, minor divergence), splitting strategies (2 copies, identical logic), stimulus protocol (2 copies, semantic difference: += vs =), StructuredGrid (2 copies, bidomain adds boundary_spec).

### 2026-03-17: Decided LBM V1 is canonical over V5.4's LBM
LBM V1 has more features: MRT collision, D2Q9 lattice, 3 boundary condition types, torch.compile kernel fusion. V5.4's LBM is simpler but less capable. Decision: LBM V1 is the canonical implementation.

### 2026-03-17: Full project document/folder map established
Mapped every document and folder in the project, clarifying what belongs where. Established conventions: Research/Active for open questions, Research/Complete for answered questions (read-only KNOWLEDGE.md), Research/Knowledge for promoted findings, experiments inside engine directories.

### 2026-05-29: V5.5 detour decided — fix the Formulation-A reaction Cm bug in an independent fork
Revisited the chi/Cm audit. Derived *why* chi is safe but Cm is not: normalizing the parent PDE by chi·Cm cancels chi out of the reaction term entirely (chi·I_ion / (chi·Cm) = I_ion/Cm), so chi lives only in D (one half → can't break splitting). Cm appears in BOTH halves (D = sigma/(chi·Cm) AND the reaction /Cm), so Formulation A — which handles Cm in the diffusion mass term but drops it in the reaction — is silently wrong for any Cm != 1.0. NOT a "change-Cm-midway" hazard; it's wrong at t=0 for any Cm != 1. Safe today only because the project pins Cm=1.0 (ionic models output pA/pF).

Decision: create **Monodomain Engine_V5.5** as a full independent copy of V5.4 (rationale: backup — V5.4 stays the frozen validated baseline; don't risk its 77 tests). The ONLY change is making the reaction Cm-correct. Diffusion is NOT touched (its chi·Cm mass term already handles arbitrary Cm — verified: dividing the implicit theta-solve by chi·Cm yields effective D = sigma/(chi·Cm) for any Cm). This is a reaction-only minimal fix, NOT a full Formulation-B structural conversion.

Code facts located (V5.4):
- Bug sites (operator-split ionic steppers, both miss /Cm):
  - `cardiac_sim/simulation/classical/solver/ionic_time_stepping/rush_larsen.py:83` → `state.V = V + dt * (-(Iion + Istim))`
  - `cardiac_sim/simulation/classical/solver/ionic_time_stepping/forward_euler.py:64` → same
  - Fix: `... / Cm`, dividing by the TISSUE Cm.
- Plumbing gap: `SimulationState` (`state.py`) has no Cm field; `_build_ionic_solver(name, ionic_model)` (`monodomain.py:91`) doesn't pass Cm. Need to wire tissue Cm → ionic step. Preferred: add `Cm` to SimulationState and read it in the stepper, mirroring Bidomain V1's `Cm = getattr(state, 'Cm', 1.0)`.
- TWO-Cm hazard: tissue/cable Cm (`tissue/isotropic.py:23`, the `chi·Cm` in `fft.py` denominators) is the one to divide by. The ionic models' internal `Cm` (`ttp06/model.py:548 p.Cm=0.185`, paci/ord calcium `inv_VcF`) is a fixed per-cell constant for Ca/Na concentration flux — DO NOT touch or conflate.
- Aside: ORd standalone `model.py:802` already does `/p.Cm`, and TTP06 `model.py:297 dV=-I_ion` does not — but neither standalone path is used by the classical operator-split solver, so the steppers are the real fix sites.

Test protocol (settled intent, blueprint to formalize):
- **Regression (Cm=1):** V5.5 reproduces V5.4 bit-identically across the existing 77-test suite (the copy must not change Cm=1 behavior). atol ~1e-12.
- **Cm-scaling invariant (Cm=k):** scaling tissue Cm by k is equivalent to time-dilation by k — solution V(x, t; Cm=k) == V(x, t/k; Cm=1); observable as CV → CV/k and APD → k·APD, identical spatial structure. Requires BOTH halves to scale with Cm, so it fails on V5.4 (broken reaction) and passes on V5.5. This is the discriminating test.
- **0D single-cell version:** no diffusion; verify dV/dt = -(I_ion)/Cm directly (trajectory at Cm=k is the Cm=1 trajectory slowed by k).

**Bidomain V1 as independent oracle (added 2026-05-29):** Bidomain V1 `rush_larsen.py:81-84` is ALREADY the exact target form — `Cm=getattr(state,'Cm',1.0); state.V = V + dt*(-(Iion+Istim)/Cm)` (Formulation B). Use it two ways: (1) code-parity anchor for V5.5's fixed line; (2) cross-engine dilation oracle — run a matched cable in bidomain (reuse `cv_shared.py` `run_bidomain`, `measure_cv_from_history`) at Cm=1 and Cm=k, confirm CV ratio→1/k, assert V5.5's ratio matches. Bidomain shares NO solver code with monodomain → strong independent check. Process-isolated (bidomain also uses `cardiac_sim`): generate a `bidomain_cm_ref.json` in a separate process, load it in the V5.5 test.

**V5.4 internal LBM is dead code — DROP it in V5.5 (decided 2026-05-29):** Verified across the repo: `cardiac_sim/simulation/lbm/` has ZERO importers outside itself, ZERO standing tests (PROGRESS "Phase 5 LBM DONE" was historical; no `test_phase5.py` ships; the live suite test_phase7/8 + test_boundary_modes + tissue tests never touch it), ZERO experiments. `simulation/__init__.py` only names it in a docstring. `step_with_V` (ionic base) is the LBM path's only hook and is called solely from `lbm/monodomain.py:237`. The boundary-conduction research — the only active LBM work — runs on the SEPARATE `LBM/Engine_V1` engine (`diag_lbm_specular.py` → `sys.path.insert(LBM/Engine_V1)`, `from src.simulation import LBMSimulation`); all new BCs (same-cell specular, HBB, bounce-back, 27-rule enumeration) live in LBM V1's `src/`. So instead of guarding V5.5's LBM at Cm!=1, just DELETE the `lbm/` package + dead `step_with_V`. V5.5 becomes clean classical-only + Cm-correct; the chi/Cm source-term entanglement (the reason a guard was considered) vanishes with it. V5.4 keeps its LBM as the faithful backup. (Earlier plan had a fail-loud guard phase; superseded.) NOTE: `test_boundary_modes.py` is an FDM boundary-mode test, NOT an LBM test — don't use it to "verify LBM".

**Formulation A vs B D-input asymmetry (CRITICAL for the comparison):** Monodomain V5.4/V5.5 (Form. A) takes input `D` = sigma; engine forms physical diffusivity = D/(chi·Cm) internally. So "scale Cm" = hold D fixed, change Cm (diffusion dilates automatically). Bidomain V1 (Form. B) takes input D_i/D_e = already-scaled sigma/(chi·Cm); so the SAME experiment requires rescaling D_i,D_e→/k when Cm→k·Cm. Compare dimensionless ratios (CV→1/k), not absolute CV, so engines needn't match in absolute units.

Open: exact plumbing (state.Cm field vs constructor arg) — recommend state.Cm. Whether V5.5 keeps the `cardiac_sim` package name (collides with V5.4 if both imported, like the existing Bidomain/V5.4 collision) — for standalone test runs it's fine; note for the eventual cardiac_core consolidation.

### 2026-05-30: Phase 2 physics correction — the Cm time-dilation invariant is FALSE
While executing Phase 2, the 0D test empirically failed: APD90(Cm=2)/APD90(Cm=1) = 1.34, not the predicted 2.0. Root cause: the tissue Cm divides ONLY the voltage update (`dV = -(Iion+Istim)/Cm`). The gate kinetics (`tau` from `compute_gate_time_constants(V,S)`) and the concentration rates carry NO Cm — they are intrinsic membrane kinetics. So scaling Cm→k·Cm does NOT rescale the whole system in time: V slows but gates keep their kinetics, so the AP MORPHOLOGY changes; it is not a `t→t/k` stretch. Substitution proof: for `W(t)=V(t/k)` to satisfy the Cm=k system you'd need `k·tau = tau` ⇒ only k=1. The invariant (and `CV→CV/k`, `APD→k·APD`) is wrong. Both `/reason` and BOTH audit passes missed this (the audit even "verified the physics sound"); the empirical run caught it. The FIX itself is correct — `dV=-(Iion+Istim)/Cm` is the right cable equation; only the *validation strategy* was flawed.

Corrected, rigorous validation (passing): (1) **exact one-step scaling** — from an identical state, `dV·Cm` is invariant across Cm∈{0.5,1,2,4} to 3.55e-15 (machine precision); this directly proves the reaction divides by Cm exactly, independent of morphology. (2) **direction** — larger Cm slows the upstroke (peak dV/dt 368→211 mV/ms) and changes APD (218→292 ms, NOT 2×). Together with the Cm=1 golden (max|dV|=0) and the full existing suite, the fix is rigorously validated. Test file: `Engine_V5.5/test_phase10_cm_scaling.py`.

Step 2.3 RESOLVED (user chose the proper cross-engine check): implemented absolute CV agreement vs Bidomain V1 (independent Formulation-B engine; isotropic + insulated BC reduces to monodomain with D_eff in the bulk). Reference generated by `Engine_V5.5/_regression/bidomain_cm_ref.py` (runs in the Bidomain engine, separate process — both use `cardiac_sim`). Matched physical diffusivity: bidomain D_i,D_e -> /Cm; V5.5 holds input D=D_EFF fixed with chi=1 (so D_phys=D_EFF/Cm) — both give D_eff=D_EFF/Cm. RESULT: Cm=1 V5.5 54.35 vs bidomain 54.35 cm/s (0.0%, exact threshold-grid match; also reproduces the historical 54.3 benchmark); Cm=2 V5.5 28.09 vs bidomain 27.77 cm/s (1.1%). Both << 5% tol. Phase 2 PASSES.

Refinement on CV vs APD scaling: empirically CV(Cm=2)/CV(Cm=1) ≈ 0.51 in BOTH engines — i.e. CV ~ 1/Cm. This is eikonal scaling, NOT dilation: CV ∝ sqrt(D_phys · upstroke_rate), and both D_phys ∝ 1/Cm and the upstroke rate (dV/dt = -Iion/Cm) ∝ 1/Cm, so CV ∝ 1/Cm. APD does NOT scale (set by repolarization gate kinetics, no Cm — measured 218→292 ms, not 2x). So the original plan's "CV→CV/k" was approximately right for the wrong reason; "APD→k·APD" was simply wrong. The cross-engine test does not depend on either — it compares two correct engines' absolute CVs.

NOTE: `cv_shared.run_monodomain_fdm` is NOT Cm-aware (line 303 has no /Cm, takes no Cm arg) — it cannot serve as a Cm!=1 reference. Only Bidomain V1 (run_bidomain) is a confirmed Cm-correct independent engine. cv_shared SIGMA_I=1.74, SIGMA_E=6.25, chi=1400 -> D_EFF=0.000972 (the test reads D_EFF_input from the ref JSON to avoid hardcoding drift).

## Failed Approaches
- **Flat engine-centric structure** (2026-03-16) — failed because: engines serve multiple research questions, making it impossible to find all work related to a single question. No natural place for cross-engine experiments.
- **First proposed restructure** (2026-03-16) — failed because: user wanted different groupings; initial Pipelines/Research separation didn't match actual workflow.
- **Converting V5.4 to Formulation B IN PLACE** (2026-03-16) — rejected: would risk V5.4's 77 passing tests. RESOLUTION (2026-05-30): instead FORKED V5.5 with the Formulation-B reaction; V5.4 stays frozen. (So Formulation B was the right target — just not destructively on V5.4.)
- **Cm time-dilation invariant for validation** (2026-05-30) — FALSE. Assumed `V(x,t;Cm=k)==V(x,t/k;Cm=1)` (⇒ CV→CV/k, APD→k·APD). Tissue Cm divides only the voltage update; gate kinetics/concentration rates carry no Cm, so Cm changes AP morphology, not timescale (APD 218→292 ms at k=2, not 2×). Asserted by the plan AND both audit passes; caught empirically (0D APD ratio 1.34). Replaced with exact 1/Cm one-step scaling (machine precision) + Bidomain V1 absolute-CV cross-check. The fix was always correct; only this validation premise was wrong.
- **`cv_shared.run_monodomain_fdm` as a Cm≠1 reference** (2026-05-30) — won't work: it has no `/Cm` and takes no Cm arg (hardcoded Cm=1). Used Bidomain V1 (`run_bidomain`) instead.
- **Merging solver internals into cardiac_core** (2026-03-16) — rejected because: solvers are engine-specific (decoupled GS for bidomain, CN/BDF for monodomain, BGK/MRT for LBM). Only shared code (ionic, mesh, stimulus) should be unified.
- **sys.modules hack as permanent solution** (2026-03-17) — recognized as temporary: `_prepare_engine()` flushes modules because both engines use `cardiac_sim` namespace. Acceptable for Phase 0 wrapper but must be eliminated when shared code moves into `cardiac_core/`.

## Session Log

### 2026-07-25 Session: image layer — 12 audit rounds, then 4 phases shipped
**Worked on**: "notice what we built over video. do the same equivalent but for `.image`; because we assume
people don't know matplotlib." Blueprint → `/audit`-revise to convergence → implement all 4 phases.
**Accomplished**:
- **Corpus census first**, because the design question was *which* figure kinds. `plot(` **163** ·
  `axhline(` 74 · `imshow(` 70 · `axvline(` 32 · `contour(` 29 over 87 `savefig` scripts. Line plots
  dominate → `Trace` is a first-class spec, not an afterthought. ⚑ The first census command used
  `grep -h -o`, which strips the path so its own `grep -v` exclusion was a **silent no-op** — a
  "reproducible" command that reproduced nothing.
- **12 adversarial rounds** (5C/8H → 1C/5H → 3C/7H → 3C/4H → 1C/5H → 2C/4H → 1C/4H → 0C/3H → 0C/1H →
  0C/2H → 0C/1H → **0C/0H CONVERGED**). **The dominant pattern every round: most C/H findings were
  follow-ons to the PREVIOUS round's own fixes.** Two structural lessons worth carrying forward:
  1. **The two-layer problem (R5).** Rules in a normative prose section contradicted the implementation
     steps, because a fix updated only the prose — and the *contradicting* half sat where an implementer
     copies from. Fix: sweep every restatement, delete superseded forms rather than annotate them.
  2. **Write an interaction down in ONE place (R4).** Three rounds each added a rule to the
     `isochrones`/`lat`/`filled` triangle and none wrote the three together; that gap produced a
     double-draw, an unmasked overlay and two contradictory compute rules.
- **Five assertions-that-cannot-fail** were caught across the cycle (`assert info.saved` for "contours were
  drawn", a spy patching the wrong binding, a set-of-characters subtraction, …) and **three Verify blocks
  that would have failed on conforming code**. Both classes recurred *after* being named.
- **Implementation confirmed the audit's value immediately**: the first cut of `draw()` used the exact
  `resolution` comparison R9 M-2 condemned, and `resolution="auto"` silently did nothing until the mandated
  Verify caught it.
**Durable findings**:
1. **`conda run` discards stdin** — a heredoc never runs and **exits 0**. Repo-wide hazard.
2. **`apd_map` is 100 % NaN below one AP** (0 % finite at 12 ms *and* 20 ms; TTP06 APD90 ≈ 230 ms). Any APD
   assertion on a short fixture is vacuous, and an all-NaN map falls back to exactly the (−90, 40) range such
   a test usually tries to exclude.
3. **`fig.colorbar(None, …)` does not raise** — it fabricates `Normalize(0, 1)` and draws a plausible,
   meaningless bar.
4. **Never name a `_LAZY` submodule after a public export** — now proven twice (`single_cell`, `draw`). The
   existing guard's collision check cannot see the second form.
**Next**: nothing blocking. Candidates: `annotations=` on `Image`/`Trace` (48 corpus `text`/`annotate` calls,
deliberately out of scope for v1), and the auto-generated Object Atlas + drift canary.


### 2026-07-23 Session (3rd parallel agent): cardiac_core EXTRACTED + PUSHED as a standalone public library
**Worked on**: took `cardiac_core` from monorepo package → standalone public GitHub repo, so Google Colab can
`pip install` it and a lab member can run the tutorial with no local setup. Four-lane readiness audit →
systematic source cleanup → extraction → verification → push → Colab wiring.
**Accomplished**:
- **Four-lane extraction audit** (monorepo-coupling · packaging · test-portability · public-exposure) →
  `EXTRACTION_AUDIT_2026-07-23.md`. Every load-bearing finding re-verified by hand in source. Headline
  blockers all real: `media.py` wrote into `site-packages` when installed; declared deps were 100% wrong
  (declared `mcp`, which the package never imports; omitted torch/numpy/**scipy**/**scikit-image**/
  **torch-dct**, the last three bare imports with no fallback); a collaborator ("John") named in shipped
  `fdm.py`; the suite hard-FAILED on a fresh clone (`test_originals_untouched` hashed absent monorepo trees).
- **Systematic source de-narrativization** (user framing: "Claude's artifacts" — session logs that leaked into
  `.py` files). 4 parallel cleanup agents, one per subtree (`_monodomain`/`_bidomain`/`_lbm`+top-level/`tests`).
  Removed: the collaborator name + storage-tank/pipe metaphor, unpublished research findings (LBM wall modes,
  eikonal source-sink), ~25 `Ref: improvement.md:Lnnn` pointers, audit tags, dated fix stamps, engine version
  lineage. KEPT: all numerics, published citations, the RKC `KNOWN LIMITATION` block (de-framed). **Verified
  comments-only** by comparing docstring-stripped ASTs vs HEAD — 122/127 byte-identical; the other 5 were the
  deliberate code edits (media root, viz `root=`, api message, 2 backend banners).
- **Code fixes shipped**: `media.py` default root `root=`→`$CARDIAC_MEDIA_ROOT`→nearest-`.git`→cwd (the `.git`
  walk keeps all ~50 monorepo callers writing where they did); `viz.py` gained `root=` (had NO escape hatch);
  removed 2 monorepo-only tests; added a tests `conftest.py` redirecting media to a tmpdir (suite wrote ~60
  files/run into the tree → now **0**, verified). Coupled-edit trap caught: a test matched `oblique|Audit #46`
  case-sensitively against "**O**blique" → it was passing ONLY on the audit tag; stripping naively would have
  turned it red. Fixed message + matcher together.
- **Moved 8 internal planning docs OUT of the package** (~360 KB, > the library itself) → `plans/`. Only
  `API_CHEATSHEET.md` ships. Added `py.typed`, MIT `LICENSE` (Copyright 2026 Li Chang), standalone
  `pyproject.toml` (real deps + viz/test extras + package-data + `namespaces=false`), README, `.gitignore`.
- **Extraction is layout-critical**: `git subtree split` flattens contents to the repo root, but `tests/` MUST
  stay a subpackage (`test_integrity` imports `cardiac_core.tests._integrity.make_goldens`) → mandatory re-nest
  commit. Scripted in `extract.sh`. Result: 64 commits of preserved history, 225 files.
- **Verified NOT assumed**: full suite **480 passed / 2 xfailed** (was 482/2; −2 monorepo-only) both in-repo
  AND from a **clean venv running in `/tmp` against the installed wheel**. Then `pip install git+https://…`
  from the real pushed GitHub URL into a fresh venv → all three engines reproduce 58.8/59.6/64.6 cm/s exactly.
- **PUSHED**: user created empty `RealJokerInc/cardiac-core`; `git push -u origin main` succeeded (SSH key
  authenticates as RealJokerInc; no `gh`/token on the box, so repo creation needed the user). Public, MIT.
- **Colab**: lesson 01 got a setup cell (cell 0) that `pip install`s from GitHub only if `cardiac_core` is
  absent — no-op locally, works on a fresh Colab runtime. Regenerated notebook (36 cells), re-verified. Share
  link + brief in the [[reference-drive-shared]] Drive folder.
**Next**: (1) merge branch `cardiac-core-extraction` (`9ba0442`) to monorepo `main` when the video/stim agents
are quiet. (2) DECIDE monorepo↔standalone sync: keep the monorepo copy or switch it to `pip install -e
~/cardiac-core` — two drifting copies is the failure that bit twice this week. (3) `cardiac_mcp/core.py:32`
still resolves the cheatsheet by monorepo path → switch to `importlib.resources.files("cardiac_core")` (works
now, the wheel ships the file). (4) open: does Cornell need to be on the copyright; the monorepo itself is
public with NO license (all-rights-reserved today).

### 2026-07-21 Session (cont.): audit-to-convergence + analysis `fields` design + LAT big issue
**Worked on**: /save-session → adversarial audit of the solver-hardening branch to convergence; then a
long design conversation on next features (single_cell, an analysis `fields` branch, adjacent EP metrics).
**Accomplished**:
- **Adversarial audit CONVERGED (4 rounds)** on `solver-hardening`; remediation commits `4003ab5`→`1b66939`
  + stale-stub-test fix `134626f`. R1: HIGH (warning FLOODED the default bidomain path — declarative
  isotropic stores conductivity as a field → is_isotropic=False → pcg_spectral breaks down ~1e-4, warns
  every step / 437 in the suite → fixed via warn-ONCE-per-instance) + MED (clamp `_stepping_run` copied
  mono save-cadence → extra bidomain frame at save_every==dt) + LOW (reason mislabel). R2: MED (warn-once
  was per-LIFETIME not per-run → `_reset_solver_diagnostics` re-arms each run) + 2 LOW; completeness sweep
  found NO CRIT/HIGH/MED (GPU clamp/injection, batch/callback, degenerate inputs all clean). R3: MED (my
  R2 "chebyshev check once per run" was UNSOUND — residual is b-dependent + A not fixed across a run for
  bdf2/IMEX → reverted to check-every-solve, warn-once-per-run). R4 CONVERGED (1 LOW: two vendored
  `SolverConvergenceWarning` classes, pre-existing). Severity HIGH→MED→MED→∅; integrity bit-identical
  throughout; **full suite 283/2**. Branch `solver-hardening` = 16 commits (4 hardening + 6 audit + docs).
- **Design decisions (spec only, NOTHING built) → `cardiac_core/ANALYSIS_FIELDS_DESIGN.md` + tutorial PLAN:**
  - **`cc.single_cell()`** — a dedicated 0-D feature that taps the ionic model directly (get_initial_state +
    monolithic model.step loop, the way Surrogate/Optimizer always did), NOT the small-uniform-grid trick.
    Consistent stim API (same {start_time,duration,amplitude}, no region). Closes the audit "no 0-D mode"
    gap; sidesteps the #14 mono-ionic bug (monolithic V5.3 step); may unlock ORd single-cell. In tutorial PLAN P0.2.
  - **`analysis.fields` branch** — parent namespace; everything operates on a field. Three layers: user-facing
    NAMED cached fields (`r.fields.voltage_flux`/`voltage_gradient`/`source_sink`/`current_flux`/`electric_field`
    /`velocity`/`direction`/`speed`/`curvature`/`vorticity`) over two toolkits `fields.derivatives`
    (grad/div/curl/laplacian) + `fields.integrals` (line/region, = Stokes/divergence-theorem partners →
    built-in consistency test). RIGOR: operators typed by input field; `curl(∇V)≡0` guarded (the meaningful
    curl is on the velocity field = vorticity = rotor). Vector fields stored `(...,2)` last-axis, wrapped
    `VectorField.x/.y/.magnitude`. Boundary = SAME as the tissue edge (`boundary_mode`, default face_mirror)
    → result must carry boundary_mode + domain_mask. Ergonomics: mesh/mask AS the region (`over=mask`, flux
    derives ∂mask + outward normals). front_metrics = the LAT subset, migrate-later. Probe deferred; r.grid/
    r.coord small+separate.
  - **Adjacent EP-metrics wishlist (separate track):** `analysis.wavelength` (λ=CV·ERP, the reentry master
    var; CV·APD proxy), consolidated `analysis.apd`, erp/di/safety-factor.
- **⚑ BIG ISSUE found + documented (not fixed): LAT is defined THREE inconsistent ways.** `r.lat()`/
  `activation_time` = first frame V≥−20mV NEAREST-frame; `r.cv()`/`conduction_velocity` = its own −20 nearest
  crossing; `apd_map` uses activation_time; BUT `activation_time_interp` = INTERPOLATED −40mV (numpy) — used
  by front_metrics/eikonal + the **source_sink_mismatch** research + fig4c_sourcesink. → default hooks (−20
  nearest, frame-quantized) vs research path (−40 interpolated, sub-frame) give DIFFERENT CV/curvature on the
  same run, silently. Neither uses max-dV/dt; first-crossing (ill-defined for reentry → phase). Fix (open):
  unify to ONE canonical LAT. Recorded: KNOWLEDGE ⚑ callout + ANALYSIS_FIELDS_DESIGN § LAT gate + source_sink
  IDEALOG cross-ref. **This is the natural prerequisite before any LAT-based `fields` gets built.**
**Next**: user's call on — (a) #13/#14 deferred default-path fixes; (b) LAT unification (prereq for LAT fields);
(c) build `single_cell()` / the `fields` branch / `wavelength`; (d) tutorial Phase P0; (e) merge `solver-hardening`.
Rejected this session: rotor seeding, dynamic pacing. Deferred: probe, r.grid ergonomics.

### 2026-07-21 Session (solver hardening SHIPPED "work through all" + tutorial-plan sidequest)
**Worked on**: Executed the audit-driven fix roadmap on branch `solver-hardening`; discussed cardiac_core as a
proper importable+documented library; started a Jupyter tutorial-series sidequest (plan only).
**Accomplished**:
- **Step 1 — make failure loud + Chebyshev M1** (`60acfbe`): shared `SolverConvergenceWarning` (warn by default;
  `filterwarnings('error', ...)` to escalate) at every non-convergence exit across mono pcg, bidomain pcg,
  pcg_spectral, and both Chebyshev solvers. Ported `_gershgorin_bounds_preconditioned` (CH-1) to the mono Chebyshev
  → the 07-02 **M1** bug fixed (regime sweep: 46% err → 3e-15). Immediately surfaced a real under-solved elliptic
  solve in scar-bidomain (M4-family), previously silent. Integrity bit-identical.
- **Step 2 — advanced features** (`91ac993`): `clamp_voltage`/`add_clamp_protocol`/`release_clamp` (per-step
  wrapper-driven `_stepping_run`; scalar/field/callable value; verified holds a strip at exactly 10 mV every frame
  while the rest evolves), `set_voltage`/`set_state`/`get_state`/`state_names` wired to live state. Gave the bidomain
  engine a `step()` (was missing → wrapper `step()` had been broken for bidomain). LBM raises (V is a lattice moment).
  17 tests. Integrity bit-identical.
- **Opt-in solver fixes** (`68b5847`): pcg_spectral mixed-BC → falls back to plain PCG (1.8e-2 stall → 4.2e-8);
  IMEX-SBDF2 2nd-order coupling extrapolation; RKC documented-and-deferred. 6 tests.
- **HONEST correction (verified on the REAL solver, not the toy repro):** the IMEX "fix to 2nd order" only HALVES
  the error — the decoupled parabolic→elliptic *staggering* imposes its own O(dt) floor that extrapolation can't
  lift (self-convergence order stayed ~1.0 before AND after; error 6.5e-3 → 2.2e-3). Documented, not overclaimed.
- **Full suite 260 passed / 2 xfailed** after each step; branch `solver-hardening` (4 code commits) NOT merged.
- **Library-as-product assessment**: `cardiac-core` 0.1.0 IS pip-installed + importable with a clean lazy `__init__`;
  BLOCKER = deps under-declared (only `mcp`; torch/numpy/scipy/torch_dct missing → clean install can't import);
  gaps = no README/LICENSE/`__version__`, API_REFERENCE stranded in Research/ not shipped. ~65% to "importable+documented".
- **Tutorial sidequest**: `cardiac_core/tutorials/PLAN.md` — 8 standalone lessons (L1 import+single-cell → L8 bidomain
  infarct+mixed BC), prep-first (P0.1-P0.6). Recon: single-cell = small uniform grid + whole-domain stim (`Grid(1,1)`
  fails, `Grid(2,2)`+ works); `ipykernel` present, must confirm `nbformat`/`nbconvert`; execute-all gate in Phase W.
- **ADVERSARIAL AUDIT of the solver-hardening branch → CONVERGED (4 rounds).** R1 (3 lanes) found HIGH (the new
  non-convergence warning FLOODED the default bidomain path — declarative-isotropic stores conductivity as a field →
  is_isotropic=False → pcg_spectral breaks down at ~1e-4, warns every step, 437 in the suite; fixed via warn-ONCE-
  per-instance), MED (clamp `_stepping_run` hardcoded mono save-cadence → extra bidomain frame when save_every==dt),
  LOW (reason mislabel). R2 (audit-the-fix + completeness) found MED (warn-once was per-LIFETIME not per-run → reused
  sim silent on run 2+ → added `_reset_solver_diagnostics` per-run re-arm) + 2 LOW (chebyshev check perf, clamp
  docstring overclaim); completeness sweep found NO CRIT/HIGH/MED (GPU clamp/injection, batch/callback run-modes,
  degenerate inputs all verified clean). R3 (audit-the-fix) found MED (my R2 "chebyshev check once per run" was
  UNSOUND — residual is b-dependent + A not fixed across a run for bdf2/IMEX → reverted to check-every-solve, warn-
  once-per-run). R4 CONVERGED — 1 LOW only (two vendored `SolverConvergenceWarning` classes, pre-existing/dedup).
  Severity decayed HIGH→MED→MED→∅. Remediation commits `4003ab5`→`1b66939` + stale-stub-test fix; integrity
  bit-identical throughout; full suite green. Total solver-hardening branch = 10 commits (4 hardening + 6 audit).
**Next**: (1) get the user's call on #13 (GPU sync-free — GPU-only vs regolden vs skip) and #14 (mono-ionic V5.3
align + regolden vs document-only); (2) OR start tutorial Phase P0; (3) merge `solver-hardening` → main (user's call);
(4) optional library packaging pass (declare deps, README/LICENSE/__version__, ship docs).

### 2026-07-16→17 Session (solver + GPU audit → roadmap deliverable; NO code changed)
**Worked on**: New user direction — audit EVERY solver + a dedicated GPU-implementation audit, empirically test
whether `device='cuda'` uses the GPU and is optimized (user's "explicit→GPU, implicit→CPU crossover weirdness"),
then set the forward path (advanced features + Phase 2-5). Audit-only per user ("measure first, don't change
solver code").
**Accomplished**:
- **GPU benchmark** (`scratchpad/gpu_bench.py`): device='cuda' IS on GPU (21/21 mono + bidomain + LBM tensors
  cuda:0/float64; results on cuda). The "crossover" = **per-iteration host syncs** (mono CN+pcg 24/step, explicit
  0, dct 1; bidomain heaviest; LBM 0), NOT a CPU-compute fallback. GPU per-step latency-bound (~6-10 ms flat), CPU
  scales with dof, crossover ~10k dof. float64 on a 1:64-FP64 card.
- **6-lane adversarial solver audit** (mono diffusion/linear, mono ionic/splitting, bidomain diffusion/splitting,
  bidomain elliptic, LBM, GPU-impl). **Every HIGH/MED finding independently reproduced by me** (ran the agents'
  repros + my own regime sweeps). 2 HIGH silent-wrong (mono Chebyshev-Jacobi 46% err at high diffusion-number;
  bidomain pcg_spectral singular precond on anisotropic mixed-BC), a systemic silent-non-convergence across ALL
  iterative solvers, IMEX-SBDF2 silently 1st-order, RKC refinement-immune ~0.8% err, mono ionic conc-currents use
  post-RL gates (diverges from V5.3, V5.4-lineage; bidomain copy is correct). **LBM clean. Default mono/bidomain
  paths solid.** Full table in KNOWLEDGE "Solver + GPU audit — 2026-07-16".
- **Cross-ref win:** finding #1 (mono Chebyshev) = the 07-02 CODE_AUDIT **M1** — KNOWN + UNFIXED 2 weeks; #9 FFT
  overlaps **M2**; bidomain silent-phi_e relates to **M4**. New this session: systemic framing, IMEX/RKC/mono-ionic,
  all GPU. Argues for landing the shared non-convergence signal, not re-auditing.
- **Decisions:** dedup (Phase 2-5) DEPRIORITIZED to backlog with the user's universal-vs-engine-specific framing +
  the internal-vs-repo-wide split. RK: mono has correct `rk2`/`rk4` (diffusion sub-step only, the GPU-clean 0-sync
  path); bidomain RK-family = the buggy `explicit_rkc`; **LBM has no RK4 and it's a category mismatch** (collide-
  stream, not an ODE march; its upgrade axis is BGK→MRT). Commits 91b52a7 (audit results) + the dedup-backlog note.
**Next**: build the roadmap deliverable (audit state + advanced-features + build path). Then, when the user greenlights
CODE changes: land the shared non-convergence signal (closes the systemic finding + M1 port) as the cheap first fix,
then the advanced features (masked per-step voltage clamp + mid-run state injection via one `_stepping_run` hook).
Still unmerged: `usability-fixes-p0-p1` branch → main (user's call).

### 2026-06-28 Session (cardiac_mcp standardization — audited → blueprinted → executed Tiers 1–3 → merged)
**Worked on**: Took the just-built `cardiac_mcp` server from "working" to "standardized" against the official MCP spec — the user flagged that a working server ≠ a standardized one and wanted its supporting materials audited against proper guidelines, then iterated audit↔revise to convergence, then executed.
**Accomplished**:
- **Audited the supporting materials** via 4 parallel spec-research agents (primitives/metadata · lifecycle/transport/errors · security/authorization · packaging/distribution), verified against the live spec (revision **2025-11-25**) + the installed `mcp` 1.28.0 source — not memory. Surfaced two real path-traversal input-validation bugs (`run_experiment` ran any `run.py`; `commit_experiment` used the unsanitized `date`) plus the annotations / `serverInfo.version` / `outputSchema` / packaging gaps.
- **Blueprinted** a 4-tier PLAN (`/blueprint`), then **audited it to convergence** — 3 adversarial Opus rounds, **12 → 5 low → 0** findings (CONVERGENCE CLEAR), all FastMCP/SDK claims source-verified, every step 9/9 sections. Folded findings via `/blueprint-revise` with mutation-log tracing. Key revisions: **Option B** packaging (extend the root pyproject, not a 2nd editable); **drop `RLIMIT_AS`** (caps virtual AS → aborts torch); a mandatory subprocess-limits test.
- **Executed Tiers 1–3** phase-by-phase, test-gated and committed on branch `mcp-standardization` (6 commits): T1 honest `ToolAnnotations` + `serverInfo.version=0.1.0` + markdown MIME + the two path-traversal guards; T2 typed `TypedDict` returns → `outputSchema`/`structuredContent` + 2 prompts + README + installable `cardiac-mcp` console script; T3 provenance-marker + CPU/FSIZE-limited `run_experiment` + `CARDIAC_MCP_TRANSPORT=http` localhost transport + `REMOTE_DEPLOY.md`. **Phase 4 (registry publishing) SKIPPED** by user.
- **16 cardiac_mcp + 140 cardiac_core tests green; HTTP mode live-verified (406, uvicorn 127.0.0.1).** One execution deviation: `RLIMIT_CPU = timeout_s*ncpu` (multi-threaded torch sums CPU-time → the plan's `≈timeout_s` would false-kill a real run). **Merged `--no-ff` → `main` and pushed to `origin` (`41d17f4`).**
**Next**: optional MCP follow-ups (media tool wrapping `cardiac_core.viz`; more resources/prompts; reentry/restitution recipes; the remote-HTTP auth stack per `REMOTE_DEPLOY.md` when a real deploy target exists; Phase-4 registry publish if public discoverability is wanted — needs a GitHub handle + license). Other engine_consolidation threads in "Next Step" (the `create_cardiac_mesh` chi firewall-bypass, Form-A→B convergence, FEM removal, Surrogate/Optimizer ionic migration).

### 2026-06-26 Session (cardiac-core MCP server — Goal-2 portability layer)
**Worked on**: Explained MCP to the user, then built `cardiac_mcp/` — an MCP server exposing `cardiac_core` to any MCP host (the reach step beyond the Claude-Code-only `/sim-*` skills).
**Accomplished**:
- **`cardiac_mcp/` shipped + validated** (uncommitted on `main`): `core.py` (transport-agnostic logic), `server.py` (FastMCP wrapper), `__main__.py` (stdio), `tests/test_core.py` (10 tests), `.mcp.json` (Claude Code registration). `mcp` SDK 1.28.0 installed in env.
- **Two-track tool surface** (user decision "both, separate tools"): DIRECT `simulate()` (ephemeral CV, coarse-dx fast) + GATED `build_manifest`→`commit_experiment`→`run_experiment` (the `/sim-experiment` gate made STRUCTURAL via a self-signed `experiment_token`; commit refuses w/o `confirmed=True` + intact token). Resources: `cardiac://cheatsheet`, `cardiac://notebook`.
- **Local stdio now, remote-HTTP later** (user decision): all logic in `core.py`; transport swap is one line in `__main__`.
- **Validated**: 10 tests green; server boots (5 tools/2 resources); real stdio client↔server roundtrip (build_manifest + cheatsheet read over the wire). Timing probe sized the defaults (coarse dx=0.02 ~8s, fine dx=0.01 ~38s). See KNOWLEDGE "Goal-2 MCP server".
**Next**: user approves the project server in Claude Code (then tools go live); optionally add a media tool (wraps `cardiac_core.viz`), more resources/prompts, reentry/restitution recipes, and the streamable-HTTP transport + hosting for remote scientists. Commit decision handed back to user (on `main`, uncommitted).

### 2026-06-25 Session (consolidation executed + Goal-2 skill suite shipped — BOTH north-star goals done)
**Worked on**: Made `cardiac_core` a single self-contained package (A2 vendoring), then built the Goal-2 LLM layer — a script-generating skill suite for **wet-lab scientists** (audience reframed from "non-coder conversational builder").
**Accomplished**:
- **Consolidation SHIPPED** (Phases 0–5, commits `935160b`→`37dc381` on `main`): vendored the 3 engines into `cardiac_core/_monodomain/_bidomain/_lbm` + shared `ionic/mesh/stimulus`; **deleted the `_prepare_engine()` sys.modules hack**; no cross-folder imports. Per-engine **bit-identical integrity goldens** (atol=0, captured pre-vendor) + source-hash baseline; 3 engine originals byte-untouched. Backup before vendoring: tag `pre-consolidation-vendoring` + bundle. **137 tests green.** Two real bugs hit+fixed: (1) cross-ref rewrite regex missing `\b` corrupted internal `ionic_time_stepping` imports; (2) naming the package `cardiac_core/monodomain/` SHADOWED the `monodomain()` factory → underscore-prefix the solver packages. See KNOWLEDGE "cardiac_core unified ground-up package — SHIPPED".
- **Goal-2 skill suite SHIPPED** (Phases 1–5, commits `126ff25`→`7635404`): `/sim-experiment` (keystone, manifest + double-check accountability gate) · `/sim-preset` · `/sim-media` · `/sim-notebook`, backed by `cardiac_core/API_CHEATSHEET.md` (anti-hallucination asset, canary-tested) + `cardiac_core/viz.py` (tested). Validated end-to-end with a control/knockdown CV series (59.3 → 41.0 cm/s, eikonal √D). `/audit` of the keystone: 0 critical (gate holds); all 11 findings folded in (notably the slug-overwrite guard + `failed`-status recording). README north-star Goal-2 wording corrected. **140 tests green.**
- Both shipped, committed, **pushed** to `origin/main`. Memory `project-goal2-skill-suite` records the audience reframe.
**Next**: Layer-A `SimulationSpec`/`create_simulation` declarative bridge, OR the programmatic claude-api wrapper, OR the deferred consolidation cleanups (Form-A→B convergence + delete `for_monodomain()`; FEM/`TriangularMesh` removal) — all now easy since the code is in one place.

### 2026-06-02 Session (glossary draft)
**Worked on**: Built the **unified glossary** off the 2026-06-01 capability census, source-verified.
**Accomplished**:
- **`GLOSSARY.md` created** (new artifact for this question). Four parallel read-only census agents (one per surface: Monodomain V5.5, Bidomain V1, LBM V1, `cardiac_core`) harvested every public identifier with `file:line`. Synthesized into a 3-tier doc: Tier 1 universal concepts (one enforced name), Tier 2 engine-specific (canonical where applicable), Tier 3 internal-module rename targets. Each row tagged ✅ aligned / 🟡 minor / 🔴 decision / ⚙️ engine-specific.
- **Census confirmed in source**: `IonicModel` ABC byte-identical across all 3 (the unification proof); `dt`/`Cm`/`(Nx,Ny)` ij/stimulus-accumulate(`+=`)/pacing-helpers(M+B) already aligned. Divergence concentrated in: voltage name, State container, run/result contract + output shape, conductivity input, χ handling, LBM-as-outlier (no grid obj, raw-mask stim, state on free attributes).
- **New evidence surfaced**: `cardiac_core` already shipped **`V`** end-to-end (snapshot/result/analysis + 77 tests) — i.e. the existing wrapper had already (silently) decided the #1 contested name against the IDEALOG `Vm` lean.
- **DECISION (user): voltage = `Vm`** (revokes the CC `V`). Glossary §2 RESOLVED. Migration: rename `V`→`Vm` in M+L, revert CC's `V` + tests, keep read-only `.V` alias then deprecate. Ionic-ABC positional `V` param left as-is (wide-blast rename = separate follow-up).
- **DECISION (user): two naming principles** added to glossary (govern all rows):
  - **P1 — mixed/subscripted notation** for any intra/extra/membrane quantity; bare symbol only when the concept is *identical across engines*. `Vm` qualifies (membrane potential is the same thing in mono+bidomain → bare `V` rejected). Bare `D` does NOT — mono/LBM carry the **effective** diffusivity, bidomain the **components**.
  - **P2 — structured grid is the primary standard** (grid-shaped `(Nx,Ny)`, LBM-simple notation), since almost all our sims are structured; the unstructured/complex-mesh path (FEM, `TriangularMesh`, flat `(n_dof,)`) becomes the explicit *secondary* standard, not the default.
- **Cascaded resolutions from P1/P2**:
  - **#7 output shape RESOLVED → grid `(Nx,Ny)` torch f64** (structured primary; flat reserved for FEM path).
  - **#12 conductivity NAMING RESOLVED → `D_eff` (mono/LBM), `D_i`/`D_e` (bidomain); inputs `sigma_*`** (bare `D` banned; user: "`D_i` is not true for monodomain, `D_eff` is more correct"). Interface (the `ConductivityConfig` class shape) still 🟡.
  - **#13 chi RESOLVED → lives only inside `ConductivityConfig`**, never a free solver knob.
  - **#6 partially resolved**: voltage field name/shape/type now fixed (`Vm`, grid, f64); only delivery style (eager `Result` vs generator vs both) still open.
- **DECISION (user): #6 run/result contract RESOLVED** — `run()` is **eager by default** (returns one `SimulationResult`); streaming folded into the SAME method via **`batch=k`** (yields `Iterator[SimulationResult]` in chunks of ≤k save-points; k=1 = frame-by-frame). User: "rename stream as run(,,batch=x)". Consequences: **no separate `stream()` method, no `Snapshot` type** — a streamed chunk is just a `SimulationResult` with T≤k, so the ONLY public output type is `SimulationResult`. Added `record=("Vm",)` knob (phi_e auto for bidomain, ionic_states opt-in for Surrogate) + kept `callback` for eager early-stop. Accepted wart: return type varies with `batch`.
- **#5 cascaded**: the batch model collapsed the old 3-object view (live State / Snapshot / Result) to **2** — public = `SimulationResult` ONLY (✅ RESOLVED; live mutable state never exposed, killing the yield-the-mutable-object footgun); internal live `State` = recommend one unified dataclass (LBM adopts, drops free attrs) but it's an internal refactor, **deferrable**.
**Still open (🔴)**: #5 internal live-State unification (deferrable), #9 default stim amplitude (−52 vs −80; likely author drift), #12 ConductivityConfig interface shape.
**Toolchain note**: a parallel research session's in-progress hook (`enforce-media-path.py`, project-root-relative path) deadlocked Write/Edit/Bash whenever cwd drifted out of repo root (python exit-2 = block). Recovered by user `cd` back to root. Not this question's bug.
**Next (resume cold)**: settle #6 delivery style + #5 (coupled — the State the generator yields); then #9, #12 interface; then start Goal 1's `Simulation` interface/Protocol + idioms in this vocabulary. Also noted: most divergences trace to 4 root causes — (A) M+B shared lineage vs LBM independent, (B) physics dimensionality (bidomain's extra potential/conductivity), (C) Formulation A/B chi bookkeeping, (D) plain naming drift.

### 2026-06-01 Session (handoff)
**Worked on**: Finished the V5.5 detour + consolidation Phase 1, then pivoted to the north-star (conversational simulation builder) and began designing the unified vocabulary/API — including a full 3-engine capability census.
**Accomplished**:
- **V5.5 Cm-correct fork** — Phases 0–2 done + committed (`ac30af55`→`5171bbce`); exact 1/Cm scaling (3.55e-15), Bidomain cross-check (0.0%/1.1%), Cm=1 golden exact. (Earlier this session.)
- **cardiac_core drift reconciled** + committed (`8f032687`); Engines/ symlink index fixed.
- **Consolidation Phase 1 — COPY-ONLY** + committed (`1f6c72e`): canonical `cardiac_core/ionic/` superset (keyword fix `cell_type_is_endo`→`celltype_is_endo`), lazy `__init__` (engine-free `import cardiac_core.ionic`), `pyproject.toml` + `pip install -e .`. Engine rewire + downstream (Surrogate/Optimizer) migration DEFERRED — audit found big-bang deletion breaks repo-wide consumers. README Phase-1 marked PARTIAL.
- **North-star set** (now the question's main goal in README): Goal 1 unified construction API + Goal 2 self-contained LLM wrapper (skills+docs, strict protocol) → conversational builder for non-coders. Key insight: **spec schema = the intake questionnaire**. Build order REFRAMED: **vocabulary first** (a ubiquitous language across the 3 engines), **then** the unified API (interface/Protocol + idioms).
- **3-engine capability census** run (read-only Explore agents) + synthesized into the cross-engine comparison (see KNOWLEDGE "Cross-engine capability census"). Found: ionic ABC + physical conventions + stimulus `+=` already aligned; divergence concentrated in construction, voltage naming (V/Vm), state, and the run/result contract; LBM is the consistent outlier.
**Next (resume cold)**: build the **glossary** off the census — settle the highest-leverage divergences first: (1) voltage `V` vs `Vm` [lean `Vm`], (2) the `State` concept (dataclass vs LBM on-object), (3) the `run()`/result contract (generator vs `(times,V_history)` + flat-vs-grid output). Then the rest of the universal-tier vocabulary, then Goal 1's interface/idioms. Geometry input (Fiji→Builder) and the Optimizer downstream migration both remain DEFERRED.

### 2026-05-30 Session
**Worked on**: Reasoned through the chi/Cm audit (why chi is safe but Cm is the troublemaker — Cm couples to both operator-split halves, chi to only diffusion); decided the V5.5 detour; blueprinted it; ran two adversarial audit passes (11 + 4 findings, all applied); executed Phases 0–2.
**Accomplished**:
- **Engine_V5.5** forked from V5.4 (Phase 0): faithful clone, dead internal LBM path removed (zero importers; boundary work uses LBM/Engine_V1), Cm=1 regression golden captured (`_regression/`, max|dV|=0).
- **Cm fix** (Phase 1): `SimulationState.Cm` plumbed from `spatial.Cm` (fail-loud, no getattr fallback); reaction divides by Cm in rush_larsen + forward_euler; FEM `_Cm`/`_chi` storage added (audit-CRITICAL — FEM only baked them into `self.M`). Cm=1 stays bit-identical; FDM/FEM/FVM all expose `.Cm`.
- **Validation** (Phase 2): `test_phase10_cm_scaling.py` 3/3 — exact 1/Cm reaction scaling to 3.55e-15; Cm-direction; Bidomain V1 cross-check (CV 54.35 vs 54.35 cm/s @Cm=1, 28.09 vs 27.77 @Cm=2).
- **Physics correction**: the Cm time-dilation invariant (assumed by the plan AND both audits) is FALSE — gate kinetics/concentrations carry no Cm, so Cm changes AP morphology, not timescale (APD 218→292 ms, not 2×). CV~1/Cm holds by eikonal scaling, not dilation. Caught empirically by the 0D test. The fix was correct throughout; only the validation strategy was wrong.
- 4 commits on `main` (`ac30af55`→`5171bbce`) + plan archived; README/KNOWLEDGE updated.
**Next**: Consolidation Phase 1 — move ionic models into `cardiac_core/ionic/` (build against V5.5). First reconcile the live `cardiac_core/` drift (added geometry/io/analysis/run; `Engines/lbm_v1` symlink deleted).

### 2026-07-01 Session — cardiac_core+mcp audit → 3-phase cleanup → LBM boundary modes SHIPPED
**Worked on**: Tidy the cardiac_core+cardiac_mcp foundation (user: "no bug or loophole… everything tidied up") BEFORE productizing the LBM boundary modes. Ran an 8-lane adversarial multi-agent audit, blueprinted a 3-phase plan, hardened it audit-to-convergence, executed all 3 phases, committed + pushed.
**Accomplished**:
- **Audit** (64 agents, ~3.1M tok): 46 confirmed (4 HIGH), 9 refuted, 12 gaps → [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md). Plus a cross-engine boundary-handling audit → [BC_IMPLEMENTATION_AUDIT.md](../boundary_conduction_speedup/BC_IMPLEMENTATION_AUDIT.md). Plan audited to convergence (R1 5blk/6maj → R2 2/2 → R3 SOUND).
- **P1** (`a3915d1`): mono FDM anisotropic cross-derivative BUG fixed (sign+½-magnitude; the audit's standout — my own boundary audit missed it) in cardiac_core+V5.5, pinned by `test_fdm_anisotropy`; chi/D=RAW unified — the depth: the DECLARATIVE `_build_mesh_data` LBM branch stores already-effective D, so the fix stores raw there too (no double-divide, Cm-safe); default D=1.4 (old default was conduction-BLOCKED); ionic-override replay; MCP path-traversal. Goldens regenerated (mono golden had captured the blocked sim; bidomain/lbm unchanged since they already ran at 1e-3).
- **P2** (`945350f`): FEM/TriangularMesh + orphaned fft.py/d2q5-mrt deleted; pcg_gmg warn; zero-save guard; positional-mesh clash; NOTEBOOK escape; stale docstrings.
- **P3** (`736296d` + terminology fix): `cardiac_core.lbm(boundary=, alpha=)` — hbb / specular_nextcell (NCS) / specular_samecell (SCS) / combined-α; kernels lifted from `diag_lbm_specular.py`; default neumann bit-identical → goldens safe; 17 boundary tests. User-corrected terminology: next-cell/same-cell specular, 'ncs'/'scs' aliases.
- Suite **148/1 → 196 passed / 0 failed**. 5 commits, pushed to `origin/engine-tuner-cardiac-core`.
- 2 own-test bugs surfaced real LBM subtleties (precompute_bounce_masks → all-False on a full periodic domain; modes only differ off-equilibrium).
**Next**: deferred backlog only (see Next Step). Nothing blocking. Candidate next real work: Form-A→B convergence, or surface mono `boundary_mode`/`stencil` through `cardiac_core.monodomain()` to match the LBM productization.

### 2026-07-15→16 Session — API failure-mode + two-round usability audit + fix blueprint
**Worked on**: (1) an API failure-mode check → F1/F2 fixes; (2) merged the `engine-tuner-v2-joint` branch (12 commits) to `main`; (3) F3/F4 cosmetic nits; (4) a **task-based usability audit** — round 1 (24 tasks, light) then round 2 (30 new tasks + full-scale re-run of the 24, **actually solved & run to completion** via 10 parallel agents); (5) a machine-targeted **PLAN.md** for the fixes, audit-revised to convergence.
**Accomplished**:
- **F1** empty-run analysis hooks no longer crash (rank-3 `(0,Nx,Ny)` + T=0 guards); **F2** `hbb`→D2Q9-only + lattice-aware LBM boundary default (d2q5/neumann kept — tuner/goldens safe); **F3** `point_distance(center=)`; **F4** cheatsheet scalar/3-tuple note. Commits `2938cf9`/`e707fe1`/`2d241af`. A clean adversarial correctness audit (0 crit/high/med) + inline hardening of 2 LOWs.
- **Tuner→main**: merge `9d82f56` (resolved 1 MASTER_KNOWLEDGE_INDEX conflict); working tree left untouched.
- **Usability audit** (report [API_USABILITY_AUDIT_2026-07-15.md](./API_USABILITY_AUDIT_2026-07-15.md), commits `09ee644` R1, `04611ed` R2): verdict "possible but painful," mean ease ≈2.7/5; 2 tasks impossible; **13 concrete bugs (B1–B13)** — B1 GPU crash-all-analysis, B2 broken fft/dct fast path, B3/B4 apd_at, B8 masked-node 23% CV error, + a fixed per-step runtime wall. Full running FLIPPED 2 verdicts up (paci automaticity; per-node-D scar) and several down.
- **PLAN.md** (`9e6a0e7`) audit-converged (inline — see caveat): 5 phases, test-gated, golden-guarded; P3/P4 as future work.
**Next**: **EXECUTE PLAN.md Phase 1** (see the ▶ HANDOVER at the top of Next Step). Optional: independent-subagent audit of PLAN.md after the session rate limit resets (~4:50am ET).

### 2026-07-22 Session (cont.) — Stim-as-object: Phases 1-2 IMPLEMENTED → committed; Phase 3 deferred
**Worked on**: Executed `cardiac_core/STIM_OBJECT_PLAN.md` (audit-converged) phase by phase on branch `stim-object`.
**Accomplished**:
- **Phase 1** (`c087b8c`): the public `Stim` object. `geometry.py` top/bottom edge masks; `stimulus/stim.py` — eager
  classmethod factories `.boundary/.point/.center/.from_region` (NOT subclasses; one Stim type), mode inferred from
  `clamp=` (voltage) vs `amplitude=` (current, default −52), `to_dict`/`from_dict` current-mode lowering (byte-identical
  to the dict → `data.stimuli`/`.npz`). Clamp routing: `_partition_stimulus` splits clamp Stims out of the current
  stimulus (declarative path only; mesh= drops stimulus=) → applied post-build via `clamp_voltage`; `_normalize_stimulus`
  raises on a clamp Stim. `clamp_voltage` gains an LBM branch → a NEW native additive, flux-preserving clamp in
  `LBMSimulation.step()` — `f[:,mask]+=w·(value−Σf)` drives Σf→value while preserving f^neq (conducts through, matches
  the mono/bidomain hard-write clamp = arbiter). Stored on the wrapper (`_lbm_clamp`), re-pushed on `reset()`; opt-in →
  integrity goldens **atol=0**. 24 new `test_stim.py` tests (holds[mono|bidomain|lbm], nonequilibrium-preservation,
  survives-reset, cuda, partition, legacy-drop). Updated the now-stale `test_advanced_features::test_lbm_clamp_and_
  injection_raise` (LBM clamp is supported now; set_voltage still raises).
- **Phase 2** (`743e6d4`): Stim canonical. `stimulate()`/`add_stimulus`/`protocols.py` build Stims internally (no raw
  dict through the warning path); `_normalize_stimulus` emits a `DeprecationWarning` (stacklevel=4) for a raw dict ONLY;
  cheatsheet §3/§11/§12 + the 11 dict-using cardiac_core test files migrated to `Stim` (a general-purpose subagent did
  the mechanical grid-threaded migration, gated on `-W error::DeprecationWarning` → 141 passed). Guards
  `test_dict_warns`/`test_dict_path_unchanged`. Full `cardiac_core/tests/` **395 passed / 0 failed** (2 xfailed).
- Kept both commits strictly scoped to Stim files — the working tree also carries a PARALLEL cloud session's video
  pipeline (`run.py`/`viz.py`/`video.py`/`test_video.py`, `r.video()`), left entirely untouched.
**Next**: Phase 3 (per-consumer dict→Stim migration) DEFERRED by the user — optional, non-blocking, one PR per consumer,
gate on each consumer's own suite. Branch `stim-object` not yet merged to main / not pushed.

### 2026-07-23 Session — Stim→main; IonicPreset plan (gated); Grid dx cm→mm REJECTED; Object Atlas + PDFs; single_cell bug fixed
**Worked on**: finished/merged the Stim object, then a run of API-surface + docs work interleaved with two parallel
cloud agents (video pipeline, tutorial notebooks) sharing the working tree.
**Accomplished**:
- **Stim object → `main`.** Verified full `cardiac_core/tests/` green (395 pass / 0 fail, only the parallel session's
  `test_video.py` failing), then fast-forwarded `main` to the 3 Stim commits (`c087b8c`/`743e6d4`/`b7fc061`) via
  `git branch -f` (NO checkout — the shared tree had two agents' uncommitted work; 197 dirty entries preserved
  byte-for-byte) and pushed. `docs` commit `b7fc061` added the `Stim` class to `API_REFERENCE.md` + an IDEALOG snapshot.
  Phase 3 (consumer dict→Stim migration) still DEFERRED.
- **`IonicPreset` plan — WRITTEN + GATED.** `cardiac_core/IONIC_PRESET_PLAN.md` (1 phase / 3 steps). Savable
  ionic-model config = base model + `{param: factor}` scaling map, accepted anywhere `ionic_model=` is, JSON save/load.
  Locked design (user): scalings-canonical + resolved `.values` (BOTH); any named param — conductance/concentration/
  kinetic (BREADTH); CORE OBJECT ONLY — `.npz` scalings-persistence + the tuner bridge DEFERRED. Resolves at the single
  `ionic/registry.py::build_ionic_model` seam (all 3 engines). User shelved implementation ("worry about it later").
- **⚑ Grid dx cm→mm — PROPOSED → BLUEPRINTED → AUDITED → REJECTED (user).** Proposal: `Grid(Nx, Ny)` primary, `dx`
  demoted to optional mm knob (default 0.1 mm, not 1 mm — 1 mm under-resolves the ~0.5–1 mm upstroke → grid-dominated
  wrong CV, the ionic-tuner phantom-block failure). Blueprinted `GRID_DX_MM_PLAN.md`, one Opus adversarial audit → **19
  findings (4 critical)**. Scrapped because the audit proved it is NOT a small fix: **102 executable `Grid(` sites
  across 5 subsystems** — incl. `cardiac_mcp/core.py` ×2 (the public `simulate(dx=…)` tool param AND the generated
  `run.py` template), the `/sim-experiment` template, 3 Lab scripts + presets, tutorials, docs. **cm stays canonical.**
  Plan file deleted; IDEALOG backlog + memory corrected to REJECTED. Two durable findings kept (below).
- **Object Atlas `cardiac_core/API_OBJECTS.md`** (the nouns, companion to the cheatsheet's verbs) — every public
  object with its full attribute + method surface, **verified by introspection** over each class MRO (regular/class/
  static methods, properties, cached_properties), not from memory. Consistent tables throughout: `| Access | Meaning |`
  for attrs, `| Call | Does |` for callables. Reflection caught real gaps my draft had wrong (`Video.bare/annotated`,
  5 `Gradient` preset classmethods, TTP06's real `compute_*` methods, `SingleCellResult` fields).
- **Markdown→PDF renderer `cardiac_core/_build/md_to_pdf.py`** (python-markdown + pygments → styled HTML → Playwright/
  Chromium PDF; same approach as the textbook build). Rendered `API_OBJECTS.pdf` (8 pp) + `API_CHEATSHEET.pdf` (6 pp),
  visually inspected via pdftoppm.
- **BUG FIXED — `single_cell` export was shadowed by its own submodule.** `from cardiac_core import single_cell` gave
  the MODULE (non-callable), and `cc.single_cell` was the function on 1st access, the module thereafter (PEP 562
  `__getattr__` only fires when normal lookup FAILS; importing submodule `single_cell` binds it as a package attr).
  Fix: renamed `single_cell.py`→`_single_cell.py` (matching `_monodomain`/`_bidomain`/`_lbm`), `_LAZY` updated, the 2
  test files switched to the public `from cardiac_core import single_cell` form. Added
  `test_public_exports_not_shadowed_by_submodules` guarding the WHOLE export map. 19 tests pass.
**Durable findings (survive the Grid rejection, worth acting on independently)**:
1. **The integrity goldens are structurally blind to the `Grid` construction path** — `tests/_integrity/make_goldens.py`
   builds all 3 goldens via `create_cardiac_mesh(Lx, Ly, dx)`, NEVER a `Grid`. So the declarative `Grid`→factory path
   (what the cheatsheet, tutorials, MCP, every Lab script use) has NO numerics drift-guard. Separately actionable.
2. **Census-grep trap**: use `\bGrid\(`. `[^a-zA-Z_.]Grid\(` silently excludes every dotted `cc.Grid(`; `grep -v
   "A|B|C"` is a BRE (the `|` is literal → filters nothing; use `-vE`). This combo hid 5 real call sites twice.
3. **Never name a `_LAZY` submodule the same as a public export** — it gets shadowed (the single_cell bug). Guarded now.
**Next**: nothing blocking. Candidate follow-ons: (a) close the goldens-blind-to-Grid gap (a `Grid`-path integrity
golden); (b) build the IonicPreset object when un-gated; (c) the GPU test failures this session were CUDA OOM from an
UNRELATED process holding 29.5 GB — not a cardiac_core regression (477 non-GPU tests pass).
