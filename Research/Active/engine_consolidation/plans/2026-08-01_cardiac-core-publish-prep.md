# PLAN: cardiac_core publish-prep — package isolation + de-narrativization + README

Created: 2026-08-01
Engine(s): cardiac_core (packaging/tests/docs only — NO solver, engine, or numerics code changes)
Research question: [engine_consolidation](README.md)
Branch: `cardiac-core-publish-prep` (off `main`)
Source: publish-prep session (2026-08-01) — user wants `cardiac_core/` to BE the shareable package (private, Cornell BME): only library code + support docs + README; tests + tutorials + plan docs live OUTSIDE it.

## Objective
Prepare `cardiac_core/` to be shared as a clean, private library for Cornell BME. Three changes: (1) MOVE `tests/` and `tutorials/` OUT of the package to the `engine_consolidation` research folder, so the package is pure library code + docs; (2) strip the last **3** in-package AI/dev-process narration comments (an agentic audit found the rest of the package already clean); (3) add a `README.md`. Tutorial CONTENT is unchanged (private audience — the Li Chang references stay; only the location changes).

## Success Criteria
- [ ] `cardiac_core/` contains ONLY: library code (`.py` + `py.typed`), support docs (`API_CHEATSHEET.md/.pdf`, `API_OBJECTS.md/.pdf`), `_build/md_to_pdf.py` (builds those PDFs), and a new `README.md`. **No `tests/`, no `tutorials/`, no plan docs.**
- [ ] The moved test suite runs **green from its new location** (the 1 import + 4 path-anchor re-anchors all land; goldens load via `__file__`-anchored paths; per-engine integrity goldens bit-identical, atol=0).
- [ ] The 3 production narration findings removed/rewritten (`mesh/structured.py` ×2, `reentry_study.py` ×1).
- [ ] `cardiac_core/README.md` present, accurate to the current API, framed **private / Cornell BME** (not public MIT).
- [ ] **No monorepo breakage** — `import cardiac_core` still works for Surrogate / Optimizer / cardiac_mcp (the package dir is unchanged in place; only `tests/`+`tutorials/` leave it).

## Architecture Changes
- **MOVE** `cardiac_core/tests/` → `Research/Active/engine_consolidation/cardiac_core_tests/` (whole subtree incl. `_integrity/` + goldens + `conftest.py`).
- **MOVE** `cardiac_core/tutorials/` → `Research/Active/engine_consolidation/cardiac_core_tutorials/` (whole subtree).
- **MOD** `cardiac_core_tests/test_integrity.py:19` — fix the cross-import (`from cardiac_core.tests._integrity...` → `from cardiac_core_tests._integrity...`).
- **MOD** `cardiac_core_tests/_integrity/make_goldens.py:21` — fix the `REPO` `..`-depth (3→5; belt-and-suspenders sys.path insert; only matters for standalone regeneration).
- **MOD** `cardiac_core_tests/test_usability_fixes.py:400` — re-anchor the `API_CHEATSHEET.md` lookup from `Path(__file__).resolve().parents[1]` (today = package root) to `Path(cardiac_core.__file__).parent` (the cheatsheet STAYS in the package). **CRITICAL: not skipped → FileNotFoundError → RED without this.**
- **MOD** `cardiac_core_tests/test_video.py:874` — same `parents[1]` → `cc.__file__` re-anchor (module binds `import cardiac_core as cc`) for `API_CHEATSHEET.md` (also not skipped → RED).
- **MOD** `cardiac_core_tests/test_self_contained.py:14` — re-anchor `PKG_ROOT` from `os.path.dirname(os.path.dirname(__file__))` (today = package root) to `os.path.dirname(cardiac_core.__file__)`; otherwise the self-containment guard silently `os.walk`s the research folder and passes as a **no-op** (green, undetectable by the verify run).
- **MOD** `cardiac_core_tests/test_tutorials.py:21` — re-anchor `_TUTORIALS` from `../tutorials` to `../cardiac_core_tutorials` (test + tutorials both move to `engine_consolidation/` as siblings, so a `__file__`-relative anchor works); otherwise the anti-rot gate globs 0 notebooks (silent no-op when the gate is enabled).
- **MOD** `cardiac_core/mesh/structured.py` — strip the `B5`/`B8` audit bug-ID tags from two comments.
- **MOD** `cardiac_core/ionic/ttp06/celltypes/custom/reentry_study.py` — drop the "Generated for Heart Conduction project" author line.
- **NEW** `cardiac_core/README.md`.

## Known constraints / do-NOT (violating any of these is a defect)
- **Do NOT nest the package** (`cardiac_core/cardiac_core/`) — that changes what `import cardiac_core` resolves to from the monorepo root and breaks Surrogate/Optimizer/mcp. The package dir stays exactly where it is; only `tests/`+`tutorials/` move out.
- **Do NOT create `__init__.py` in `Research/Active/engine_consolidation/`** — pytest (prepend import mode) inserts the first parent WITHOUT `__init__.py` onto `sys.path`; that parent must be `engine_consolidation/` so `cardiac_core_tests` is importable as a top-level package. An `__init__.py` there would push the sys.path anchor higher and break test collection.
- **Do NOT change any tutorial CONTENT** (markdown cells / prose). The Li Chang references are intentional and stay — this is a private artifact and the tutorials only change LOCATION. (The agentic audit flagged them; they are explicitly out of scope here.)
- **Do NOT touch solver/engine/numerics code** — integrity goldens must stay bit-identical (atol=0). The only code edits are 3 comment strips.
- **Do NOT clean the test/tutorial comment-narration** the agentic audit found (33+6 findings) — those files are leaving the package for the internal research folder, where dev narration is fine. Only the 3 in-package production comments are in scope.

---

## Phase 1: Move `tests/` + `tutorials/` out; make the moved suite runnable

**Goal**: `cardiac_core/` no longer contains `tests/` or `tutorials/`; the relocated test suite runs green against the in-place `cardiac_core` package. Independently deliverable; own commit.
**Tier**: large (has real pytest-import fragility → warrants the /audit the user asked for)
**Estimated scope**: 2 `git mv` subtree moves + 2 small edits + a verification run.

### Phase Context
- **Env**: `/opt/miniforge3/bin/conda run -n heart-conduction --no-capture-output python`. `conda run python - <<EOF` discards stdin and exits 0 — use `python file.py` or `-c`.
- `cardiac_core` is **editable-installed**, so `import cardiac_core` works from any cwd regardless of where the tests live. That is why the move is feasible.
- **Cross-references to fix — SIX edits, not one.** The move breaks every test that anchored to its own location to reach the package. `cardiac_core/` and its `API_CHEATSHEET.md` do NOT move; the tests do — so any `Path(__file__).parents[1]` / `dirname(dirname(__file__))` that used to land on the package root now lands on `engine_consolidation/`. The robust re-anchor is the INSTALLED package: `Path(cardiac_core.__file__).parent`. The five path/import edits (verified against the real files by the round-1 audit):
  - `test_integrity.py:19` — `from cardiac_core.tests._integrity...` → `from cardiac_core_tests._integrity...`.
  - `test_usability_fixes.py:400` + `test_video.py:874` — `API_CHEATSHEET.md` via `parents[1]` → `Path(cardiac_core.__file__).parent / "API_CHEATSHEET.md"`. **Both are NOT skipped → FileNotFoundError → RED without this** (the round-1 CRITICAL).
  - `test_self_contained.py:14` — `PKG_ROOT = dirname(dirname(__file__))` → `os.path.dirname(cardiac_core.__file__)`; else the guard silently passes on the wrong tree.
  - `test_tutorials.py:21` — `../tutorials` → `../cardiac_core_tutorials`; else 0 notebooks globbed.
  Each edit needs `import cardiac_core` (or `os` already present) in that test module — add the import if absent. `make_goldens.HERE` (golden `.pt` loads) is ALREADY `__file__`-anchored and moves correctly.
- **Goldens load correctly by construction**: `make_goldens.HERE = dirname(abspath(__file__))` — `__file__`-anchored, so it points at the moved `_integrity/` dir and `torch.load(HERE/golden_*.pt)` still finds the 3 `.pt` files (they move with it). No golden regeneration needed.
- `tests/conftest.py` is self-contained (a session media-root fixture) and moves with the suite.
- No root `pytest.ini`/`testpaths`; the suite is run by path. The repo-root `conftest.py` (adds `Surrogate/` to `sys.path`) is harmless here and stays.

### Step 1.1: Move the two subtrees
**Model**: opus
#### Read First
- `cardiac_core/tests/test_integrity.py:19` — the cross-import.
- `cardiac_core/tests/_integrity/make_goldens.py:20-25` — `HERE`/`REPO`/sys.path insert.
- Confirm `Research/Active/engine_consolidation/` has NO `__init__.py` (`ls`).
#### Why
Moving the whole `tests/` subtree (with `_integrity/` + goldens + `conftest.py`) as a unit keeps every intra-suite relationship intact except the one absolute import. Moving `tutorials/` is dependency-free (nothing in the package imports it; its `_build` scripts import `cardiac_core`, which still resolves).
#### Implementation Spec
```bash
git mv cardiac_core/tests Research/Active/engine_consolidation/cardiac_core_tests
git mv cardiac_core/tutorials Research/Active/engine_consolidation/cardiac_core_tutorials
```
Keep `cardiac_core_tests/__init__.py` and `cardiac_core_tests/_integrity/__init__.py` (they make it an importable package). Do NOT add an `__init__.py` to `engine_consolidation/`.
#### Pseudocode
N/A — two directory moves.
#### Test Spec
Covered by Step 1.3 (run the moved suite).
#### Checklist
- [ ] `cardiac_core/tests` and `cardiac_core/tutorials` no longer exist.
- [ ] `Research/Active/engine_consolidation/cardiac_core_tests/` and `.../cardiac_core_tutorials/` exist with their subtrees.
- [ ] `git status` shows the moves as renames (R).
#### Verify
```bash
ls cardiac_core/tests cardiac_core/tutorials 2>&1 | grep -q "No such" && echo "removed from package"
ls Research/Active/engine_consolidation/cardiac_core_tests/_integrity/golden_monodomain.pt && echo "goldens moved"
```
#### Exit Criteria
- [ ] Both subtrees relocated; goldens present at the new path.
#### Risk
`git mv` on a large subtree is atomic in git; low risk. — mitigation: the two verify checks.

### Step 1.2: Fix the cross-references (1 import + 4 path anchors) + the `REPO` depth
**Model**: opus
#### Read First
- `cardiac_core_tests/test_integrity.py` (whole — `HERE` + `canonical_sim` usage; check if `import cardiac_core` is present).
- `cardiac_core_tests/test_usability_fixes.py:396-406`, `test_video.py:869-876`, `test_self_contained.py:1-50`, `test_tutorials.py:1-30` — the anchor lines + whether each already imports `cardiac_core`.
- `cardiac_core_tests/_integrity/make_goldens.py:20-25`.
#### Why
The move breaks every test anchor that assumed the file sits inside the package. `cardiac_core/` (and `API_CHEATSHEET.md`) do not move; the tests do. `Path(__file__).parents[1]` / `dirname(dirname(__file__))` used to hit the package root and now hits `engine_consolidation/`. Two of these (`API_CHEATSHEET.md` readers) are non-skipped and go RED; two (`test_self_contained`, `test_tutorials`) silently pass on the wrong target — worse, because the verify run cannot catch a silent green. Re-anchor to the INSTALLED package (`cardiac_core.__file__`), which is location-independent.
#### Implementation Spec (SIX edits)
- `test_integrity.py:19`: `from cardiac_core.tests._integrity.make_goldens import canonical_sim, HERE`
  → `from cardiac_core_tests._integrity.make_goldens import canonical_sim, HERE`.
  (Fallback if pytest does NOT collect it as `cardiac_core_tests.*`: relative `from ._integrity.make_goldens import ...`; last resort inline `canonical_sim` + `HERE`. Pick whichever Step 1.3 proves green; log it.)
- `test_usability_fixes.py:400`: `cheatsheet = Path(__file__).resolve().parents[1] / "API_CHEATSHEET.md"`
  → `import cardiac_core` (if absent) then `cheatsheet = Path(cardiac_core.__file__).parent / "API_CHEATSHEET.md"`.
- `test_video.py:874`: `Path(__file__).resolve().parents[1].joinpath("API_CHEATSHEET.md").read_text()`
  → `Path(cc.__file__).parent.joinpath("API_CHEATSHEET.md").read_text()`. **Use `cc`** — this module binds
  `import cardiac_core as cc` (line 19), so the bare name `cardiac_core` is UNBOUND (a literal
  `cardiac_core.__file__` here → NameError). (`test_usability_fixes.py` + `test_self_contained.py` bind
  neither → add a bare `import cardiac_core` there.)
- `test_self_contained.py:14`: `PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
  → `import cardiac_core` (if absent) then `PKG_ROOT = os.path.dirname(cardiac_core.__file__)`.
- `test_tutorials.py:21`: `_TUTORIALS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tutorials"))`
  → `... "..", "cardiac_core_tutorials"))` (relative to the moved test, whose sibling is the moved tutorials).
- `make_goldens.py:21`: `REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))` → add two more `".."` (5 levels: `_integrity`→`cardiac_core_tests`→`engine_consolidation`→`Active`→`Research`→repo root). Standalone-regeneration only; `# repo root` stays accurate.
#### Pseudocode
N/A — six line edits.
#### Test Spec
`test_matches_golden[*]` atol=0; `test_cheatsheet_examples_execute`, `test_cheatsheet_video_section_executes` pass (find the cheatsheet); `test_no_cross_folder_imports`/`test_no_prepare_engine_hack` still walk the ACTUAL package (spot-check by pointing PKG_ROOT print).
#### Checklist
- [ ] all 6 edits applied; each affected test module imports `cardiac_core` where now needed.
- [ ] `grep -rn "cardiac_core\.tests\|parents\[1\]" Research/Active/engine_consolidation/cardiac_core_tests/` → no stale package-root anchors remain.
#### Verify — see Step 1.3 (runs the suite).
#### Exit Criteria
- [ ] No `cardiac_core.tests` / package-root `parents[1]` anchor remains; the two cheatsheet tests and the two self-containment guards target the real package.
#### Risk
A missed anchor stays silently green (H2/H3 class). — mitigation: after Step 1.3, spot-verify `test_self_contained` actually inspects the package (e.g. temporarily assert PKG_ROOT endswith `/cardiac_core`) and `test_tutorials` collects >0 notebooks under the gate.

### Step 1.3: Verify the relocated suite runs green
**Model**: opus
#### Read First — none (execution step).
#### Why
The move is only correct if the full suite still passes from the new path, especially the integrity goldens (the one check that proves no numerics regression) and the media/video tests (which depend on the moved `conftest.py` media-root fixture).
#### Implementation Spec — run pytest against the new path.
#### Pseudocode — N/A.
#### Test Spec
Full moved suite green; integrity goldens bit-identical; count matches the pre-move baseline (678 passed / 2 xfailed on a free GPU, modulo CUDA-OOM allowlist).
#### Checklist
- [ ] integrity subset green first, then the full moved suite.
#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
CR="/opt/miniforge3/bin/conda run -n heart-conduction --no-capture-output python"
$CR -m pytest Research/Active/engine_consolidation/cardiac_core_tests/test_integrity.py -q
$CR -m pytest Research/Active/engine_consolidation/cardiac_core_tests/ -q
```
#### Exit Criteria
- [ ] integrity goldens pass (atol=0); full moved suite green vs baseline (CUDA-OOM allowlist excepted).
#### Risk
A test hard-codes `cardiac_core/tests/...` or a repo-relative path. — mitigation: the full run surfaces it; fix any such path to be `__file__`-anchored.

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
CR="/opt/miniforge3/bin/conda run -n heart-conduction --no-capture-output python"
$CR -c "import cardiac_core, cardiac_core.api, cardiac_core.ionic.scaling; print('monorepo import OK')"   # package still importable in place
$CR -m pytest Research/Active/engine_consolidation/cardiac_core_tests/ -q
```
### Phase 1 Exit Criteria
- [ ] `cardiac_core/` has no `tests/` or `tutorials/`.
- [ ] moved suite green; integrity atol=0.
- [ ] `import cardiac_core` unaffected.
### Phase 1 Cleanup
- [ ] no `__init__.py` was added to `engine_consolidation/`.
- [ ] no stray `cardiac_core.tests` references anywhere (`grep -rn "cardiac_core\.tests" --include=*.py .`).
- [ ] tutorial content byte-unchanged (`git diff --stat` for the tutorial move shows pure renames, no content deltas).
**→ Commit point: `refactor(cardiac_core): move tests + tutorials out of the package to engine_consolidation`**

---

## Phase 2: De-narrativize the 3 in-package comments
**Goal**: remove the last AI/dev-process narration inside the shipped package. **Tier**: small.
### Phase Context
The agentic audit (8 read-only agents over all 223 non-tutorial package files) found the engine internals, `fields/image/video`, and all 16 top-level modules **clean**. Only these 3 remain in package code (the rest were in `tests/`/`tutorials/`, now moved out). Each edit is comment-only → zero behavior change → goldens unaffected.
### Step 2.1: Strip the tags / author line
**Model**: opus
#### Read First
- `cardiac_core/mesh/structured.py` around lines 65 and 191.
- `cardiac_core/ionic/ttp06/celltypes/custom/reentry_study.py` around line 19.
#### Why
`B5`/`B8` are internal bug-tracker IDs from a past usability audit; "Generated for Heart Conduction project" is an internal-project author tag. Neither belongs in a shared library; both are pure narration with a real technical comment to preserve (mesh) or nothing to preserve (author line).
#### Implementation Spec (exact edits)
- `mesh/structured.py:65`: `# B5: guard the single-row/single-column degenerate cases (Nx==1 or Ny==1)` → `# Guard the single-row/single-column degenerate cases (Nx==1 or Ny==1)`
- `mesh/structured.py:191`: `... with ``fill_value`` (NaN by default, B8). NaN makes downstream analysis ...` → strip `, B8` → `... with ``fill_value`` (NaN by default). NaN makes downstream analysis ...`
- `reentry_study.py:19`: delete the `Author: Generated for Heart Conduction project` line (and, if it leaves a dangling empty docstring/comment block, tidy the surrounding line only).
#### Pseudocode — N/A (3 text edits).
#### Test Spec — none new; behavior unchanged (comments only).
#### Checklist
- [ ] 3 edits applied; `grep -rniE "\bB5\b|\bB8\b|Generated for Heart Conduction" cardiac_core/mesh/structured.py cardiac_core/ionic/ttp06/celltypes/custom/reentry_study.py` → clean.
#### Verify
```bash
CR="/opt/miniforge3/bin/conda run -n heart-conduction --no-capture-output python"
$CR -c "import cardiac_core.mesh.structured, cardiac_core.ionic.ttp06.celltypes.custom.reentry_study; print('imports OK')"
```
#### Exit Criteria — [ ] the 3 comments cleaned; modules still import.
#### Risk — deleting the author line could break a module docstring if it's the only line. — mitigation: read the surrounding block first; keep valid syntax.
**→ folded into the Phase 3 commit (or its own trivial commit).**

---

## Phase 3: `cardiac_core/README.md` (private, Cornell BME)
**Goal**: a publication-ready README so the folder reads as a shareable library. **Tier**: medium.
### Phase Context
Model on `~/cardiac-core/README.md` (the extracted repo's README — title, one-liner, a `Grid → ConductivityConfig → Stim → monodomain → run → cv` quick example, engines list, PyTorch/float64 note), then REFRESH for the current API and reframe for a private audience.
### Step 3.1: Write the README
**Model**: opus
#### Read First
- `~/cardiac-core/README.md` (the model; ~stale but structurally good).
- `cardiac_core/API_CHEATSHEET.md` §1–§8 (current construction API, `single_cell(conductances=)`, media, analysis) — the README must not contradict it.
- `cardiac_core/__init__.py` (the actual public exports).
#### Why
The folder is being shared as a library; a README is the entry point. It must be accurate (a wrong quick-start is worse than none) and correctly framed (private, not an open MIT release).
#### Implementation Spec — sections:
1. **Title + one-liner** — unified cardiac EP simulation API; 3 engines (monodomain FDM / bidomain / LBM); PyTorch, float64, CPU or CUDA.
2. **Quick example** — the cheatsheet's canonical `Grid → ConductivityConfig.bidomain → Stim.boundary → monodomain → run → r.cv()` (~59 cm/s), copied from a RUNNING cheatsheet snippet so it's correct.
3. **What's inside** — construction API; `single_cell()` incl. the `conductances={...}` 0-D drug knob; media layer (`r.video()`/`r.image()`/`r.trace()` + `.show()`); `analysis` + `analysis.fields`; `Stim` object.
4. **Docs** — point to `API_CHEATSHEET.md` (verbs) and `API_OBJECTS.md` (nouns). Note tutorials are maintained separately (they now live outside the package).
5. **Status / access** — **private; shared with Cornell BME**; not a public release. No MIT/PyPI claims (the public mirror is a separate, older repo — do not conflate).
6. **Install/run** — `import cardiac_core as cc`; deps (torch, numpy, scipy, scikit-image, torch-dct); editable-install note.
#### Pseudocode — N/A (prose).
#### Test Spec — the quick-example code block must execute (paste-run it once under conda before finalizing).
#### Checklist
- [ ] README covers the 6 sections; quick example runs; no MIT/public-release language; tutorials referenced as separate.
#### Verify
```bash
# paste the README quick-start into a scratch .py and run it:
$CR /tmp/.../readme_quickstart.py   # prints a CV ~59 cm/s
```
#### Exit Criteria — [ ] README accurate + runs; private framing.
#### Risk — overclaiming (MIT/public) or a stale example. — mitigation: copy the example from the canary'd cheatsheet; frame private explicitly.

### Phase 3 Verification
```bash
CR="/opt/miniforge3/bin/conda run -n heart-conduction --no-capture-output python"
$CR -m pytest Research/Active/engine_consolidation/cardiac_core_tests/ -q -k "integrity or golden"   # still atol=0
git -C . status --short cardiac_core/   # only README added + the 3 comment files modified + tests/tutorials removed
```
### Phase 3 Exit Criteria
- [ ] README present + accurate; goldens bit-identical; package dir clean of non-library content.
### Phase 3 Cleanup
- [ ] float64/V5.3-untouched N/A (no numerics touched).
- [ ] `cardiac_core/` top-level = library `.py` + `API_*` docs + `_build/` + `README.md` only.
**→ Commit point: `docs(cardiac_core): README + strip last in-package narration (publish-prep)`**

---

## Final Cleanup
1. Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_cardiac-core-publish-prep.md"
```
2. Update KNOWLEDGE.md + IDEALOG.md: record the package isolation (tests/tutorials → `cardiac_core_tests/`+`cardiac_core_tutorials/`) + the README; and fix the stale layout refs the round-1 audit found — `Research/Active/engine_consolidation/README.md:69,80` (say `cardiac_core/tests/`) and the `cardiac_core/tutorials/PLAN.md` mentions in `MASTER_KNOWLEDGE_INDEX.md:17` + `KNOWLEDGE.md:879`.
3. `CLAUDE.md` "Running Tests" does NOT reference `cardiac_core/tests` (it documents Bidomain/Monodomain/LBM only) — no CLAUDE.md edit needed for this move. The new test-invocation path is `pytest Research/Active/engine_consolidation/cardiac_core_tests/`.
- [ ] no cross-engine duplication introduced; V5.3 untouched.

## Mutation Log

**MUTATED 2026-08-01 (audit R1)**: Architecture + Phase-1 Context + Step 1.2 EXPANDED — CRITICAL/HIGH: the
plan's "only one cross-import to fix" was FALSE. Four more tests anchor to the package root via
`Path(__file__).parents[1]` / `dirname(dirname(__file__))` to reach files that DON'T move
(`API_CHEATSHEET.md` stays in the package): `test_usability_fixes.py:400` + `test_video.py:874` (non-skipped
→ FileNotFoundError → RED — the CRITICAL), `test_self_contained.py:14` (silent no-op walking the research
folder — HIGH), `test_tutorials.py:21` (`../tutorials` renamed → 0 notebooks — HIGH). MOD list 2→6 files;
each re-anchored to `cardiac_core.__file__`. Added a spot-verify to Step 1.2 Risk so the two SILENT defeats
can't pass unnoticed.
**MUTATED 2026-08-01 (audit R1)**: Final Cleanup #2/#3 CORRECTED — LOW: named the stale layout refs to fix
(engine_consolidation/README.md:69,80; MASTER:17; KNOWLEDGE:879) and dropped the moot CLAUDE.md item
(CLAUDE.md "Running Tests" never referenced `cardiac_core/tests`).
**MUTATED 2026-08-01 (impl)**: Step 1.2 test_integrity import — the ABSOLUTE
`from cardiac_core_tests._integrity.make_goldens import ...` worked first try (pytest prepend-mode anchored
sys.path at `engine_consolidation/`); no fallback needed. Confirmed by the 3 golden tests passing atol=0 and
by `test_tutorials` collecting 12 notebooks + `PKG_ROOT` resolving to `.../cardiac_core`.
**MUTATED 2026-08-01 (impl)**: Phase 2 scope EXPANDED 3→16. After the agentic audit, a manual
`grep -E "[BPF][0-9]"` sweep of the package found **13 MORE** internal audit/bug/phase-ID tags the agentic
auditors MISSED (they only flagged mesh `B5`/`B8`): `B1` (api.py:1427); `B3`/`B4`/`B10`/`B13` + `F1`×4 +
`P2` (analysis.py); `P2` (run.py:123); `P1.5`×2 (ionic/mhas13/model.py). Same class as `B5`/`B8`; all
stripped (technical comment kept, internal ID removed). Lesson: the fan-out auditors under-reported on the
top-level package files — a targeted `[BPF][0-9]` grep was the exhaustive check.

## Convergence note (2026-08-01)
Audit→revise: **round 1** = 1 critical / 2 high / 0 med / 1 low (all genuine breakage the plan missed — the
4 package-root anchors); **round 2** = 0 crit / 0 high / 0 med / 1 low (an executor-trap phrasing fix:
`test_video.py` binds `cardiac_core as cc`). An independent grep confirmed the 4 anchors are exhaustive (no
5th). Round 2 re-verified the fixes match the real lines, resolve correctly, and the 6-edit plan is
internally consistent; REPO=5 and the integrity import are intact. **CONVERGED — the plan correctly does the
simple thing.** Proceeding to implementation.
