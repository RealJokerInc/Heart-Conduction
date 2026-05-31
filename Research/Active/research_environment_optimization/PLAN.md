# PLAN: Repository Structure Cleanup

Created: 2026-05-30
Engine(s): All (cross-cutting workspace hygiene — no engine logic changes)
Research question: [research_environment_optimization](README.md)
Source: This session's 5-area folder audit (2026-05-30). Findings live in the session conversation; key results are summarized in each phase's Context. IDEALOG entry to be written by `/save-session`.

## Objective
The repo root has accumulated ~12 undocumented dirs, 286 git-tracked `.pyc` files, a stale duplicate `Images/` tree, deprecated `Q1–Q8/` research folders, and several dead duplicates. This plan backs up the exact current state to GitHub, then removes dead/duplicate content and documents the live-but-undocumented structure — in independently-committable phases, lowest-risk first.

## Execution model (READ FIRST — makes the safety gates real)
Every fenced `bash` block that contains an `exit 1` gate (Steps 2.1, 3.2, and the index-clean check in 1.2) MUST be executed as a SINGLE script, not pasted line-by-line — otherwise `exit 1` ends only one command and the gate fails open. Per CLAUDE.md Permission Handling: write the block to a temp file and run it, e.g. `cat > /tmp/step.sh <<'EOF' … EOF; bash /tmp/step.sh` (and STOP if it exits non-zero). The blocks assume `set -o pipefail` (included inline) so a broken pipe in a gate is not silently read as "no match."

## Success Criteria
- [ ] Immutable backup branch of current state pushed to `origin` (recoverable restore point)
- [ ] Zero `.pyc` / `__pycache__` files tracked in git (`git ls-files | grep -c '\.pyc$'` → 0)
- [ ] No redundant `Images/`/`Videos/`/`Media/` trees; authored diagrams preserved in a documented home
- [ ] Deprecated `Q1–Q8/`, root `Research/papers/`, dead `harness_v1/`, dead `LBM/Engine_V1/ionic/ionic/` removed
- [ ] CLAUDE.md documents every live root dir; `Engines/cardiac_core` symlink resolves
- [ ] All engine test suites still pass (no import breakage from deletions)

## Architecture Changes
- DEL: 286 tracked `.pyc` + `__pycache__/` dirs (untrack only — files stay on disk, already gitignored)
- DEL: `Images/` (after relocating 41 unique authored files), `Videos/`, `Media/` (empty stubs)
- DEL: `Research/Q1_*`…`Q8_*` (superseded by `Research/Active/`)
- DEL: `Research/papers/` (root, deprecated — after migrating the 4 referenced LBM PDFs)
- DEL: `LBM/Engine_V1/ionic/ionic/` (dead nested duplicate, zero imports)
- DEL: `harness_v1/` (byte-identical dead duplicate of `cardiac_ml/`+`conf/`+`scripts/`)
- NEW: `Surrogate/docs/diagrams/` — relocated IonicSurrogateV3 diagram sources from `Images/`
- MOD: `.gitignore` — add figure/output + checkpoint conventions
- MOD: `CLAUDE.md` — document `cardiac_core/`, `cardiac_ml/`+`conf/`+`scripts/`, `simulation/`, `Engines/`/`Pipelines/` symlink note
- FIX: `Engines/cardiac_core` symlink `../../cardiac_core` → `../cardiac_core`

## Known Failures / Corrections (from this session's verification)
- **"`Images/` is 99% duplicate → just delete it"** — WRONG. 41/140 files are unique authored content (`_diagram_archive/`, `ionic_surrogate_v3.*`, `generate_v3*.py`). Must relocate before deleting. (Subagent over-claimed; corrected by direct `find`.)
- **"`simulation/outputs/` = 322 MB committed"** — WRONG. 0 files tracked (`git ls-files simulation/outputs | wc -l` → 0). Local-only. Do NOT attempt to untrack it.
- **"12 `WHITEBOARD.md` committed"** — WRONG. 0 tracked; `**/WHITEBOARD.md` gitignore works. Leave alone.
- **"`MonthlyReport/` 64 MB bloating git"** — WRONG. 0 tracked. Local-only; out of scope for git cleanup.
- **Mixing `git rm` modes** — use `git rm -r --cached` to untrack live files (keep on disk); `git add -u` / `git rm` to stage real deletions. Don't conflate.

---

## Phase 0: Backup current state to GitHub (DO THIS FIRST)

**Goal**: Capture the EXACT current working tree (including all 173 uncommitted changes) as an immutable, pushed restore point before any destructive operation.
**Tier**: small
**Estimated scope**: One backup branch, committed and pushed; return to `main` untouched.

### Phase Context
- Remote: `origin = git@github.com:RealJokerInc/Heart-Conduction.git`. Current branch `main`, ~5 commits ahead of `origin/main`, working tree dirty (173 changes incl. in-progress deletions of `lbm_cardiac`, `Engines/lbm_v1`, modified `.pyc`/binaries).
- `.git` is 4.2 GB; the push sends the ~5 unpushed commits + one new snapshot commit. May take minutes — let it finish.
- The backup branch is the safety net for ALL later phases. Rollback any file with `git checkout backup/pre-cleanup-2026-05-30 -- <path>`.
- Pushing is outward-facing and was explicitly requested by the user — it is authorized.

### Step 0.1: Create and push timestamped backup branch
**Model**: sonnet

#### Read First
- (none — pure git)

#### Why
A dirty working tree can't be restored from `main` alone. Committing the full current state (`git add -A`) onto a dedicated branch freezes it; pushing makes it survive local disk loss and gives one-command rollback for every subsequent phase.

#### Implementation Spec
**Files to modify:** none (git refs only)
**Branch name:** `backup/pre-cleanup-2026-05-30`

#### Pseudocode / exact commands
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
git rev-parse --abbrev-ref HEAD            # confirm: main
BR=backup/pre-cleanup-2026-05-30
git switch -c "$BR"
git add -A                                  # stage ALL 173 changes incl. deletions
git commit -m "backup: full working-tree snapshot before repo structure cleanup (2026-05-30)"
git push -u origin "$BR"                     # immutable restore point on GitHub
git switch main                              # return to main
git log --oneline -1 "$BR"                   # record backup SHA in Mutation Log
```

#### Test Spec
- `git ls-remote origin "$BR"` returns a SHA → backup exists on GitHub.

#### Checklist
- [ ] On `main` before starting
- [ ] Backup branch created; `git add -A` staged everything
- [ ] Commit created; SHA recorded in Mutation Log
- [ ] `git push` succeeded (retry on transient network/auth failure)
- [ ] `git ls-remote origin backup/pre-cleanup-2026-05-30` shows the SHA
- [ ] Switched back to `main`

#### Verify
```bash
git ls-remote origin backup/pre-cleanup-2026-05-30
git branch --show-current   # → main
```

#### Exit Criteria
- [ ] Remote backup branch confirmed via `git ls-remote`
- [ ] Local back on `main`

#### Risk
Push fails (auth / 4.2 GB transfer). Two-tier hard gate:
- **Phase 1** (untrack `.pyc`, commit deletions — all non-destructive / on-disk-preserving) may proceed once the LOCAL backup commit exists.
- **Phases 2 & 3** (irreversible `git rm` of files / dirs) MUST NOT start until `git ls-remote origin backup/pre-cleanup-2026-05-30` returns a SHA — i.e. the push SUCCEEDED. Rationale: the stated failure mode is local-disk loss; a local-only commit gives zero protection against it. If the push is blocked, fix connectivity/auth and retry before any deletion; do not work around the gate.

### Phase 0 Verification
```bash
git ls-remote origin backup/pre-cleanup-2026-05-30 && echo "REMOTE BACKUP OK"
git switch main && git branch --show-current
```
> ⚠️ CORRECTION (2026-05-30, learned the hard way): the `git switch -c backup → git add -A → commit → git switch main` sequence is WRONG. Committing on the backup branch then switching back REVERTS `main`'s working tree to its last commit — the dirty changes move to the backup branch and DISAPPEAR from `main`. DO NOT use the branch-switch method. Correct, non-disruptive backup: build a snapshot WITHOUT switching — `git stash create` or `T=$(git write-tree after add -A)` / orphan `git commit-tree`, push that — leaving the working tree untouched. (Phase 0 was already executed this session; see Mutation Log for what actually happened and the orphan-snapshot push.)

### Phase 0 Exit Criteria
- [ ] Backup branch on GitHub (or local backup commit exists and the push blocker is logged)
- [ ] Back on `main`

**-> Commit point: the backup commit itself IS this phase's commit (on the backup branch).**

---

## Phase 1: Git hygiene

**Goal**: Stop tracking build artifacts; resolve the pending working-tree deletions on `main`.
**Tier**: small
**Estimated scope**: Two commits on `main` — untrack bytecode, then settle pending deletions.

### Phase Context
- `.gitignore` ALREADY contains `__pycache__/` and `*.pyc`, but 286 `.pyc` remain tracked (committed before the rule). gitignore never retroactively untracks. Fix: `git rm -r --cached` (removes from index, keeps on disk).
- `main`'s tree has 173 changes; many are legitimate prior cleanup already done on disk: `Research/Complete/lbm_cardiac/` deleted, `Engines/lbm_v1` deleted, ~10 `Surrogate/run_*.py` + docs deleted, plus modified `.pyc`/binaries. These should be committed.
- After untracking `.pyc`, the bytecode churn disappears from future diffs; binaries are handled in Phase 2.

### Step 1.1: Untrack all `.pyc` / `__pycache__`
**Model**: sonnet

#### Read First
- `.gitignore` — confirm `__pycache__/` and `*.pyc` present (they are).

#### Why
286 tracked bytecode files pollute every diff and re-appear as "modified" constantly. Untrack once + existing gitignore = permanently clean.

#### Implementation Spec
**Files to modify:** git index only (no disk deletion).

#### Pseudocode / exact commands
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
git ls-files '*.pyc' | xargs -r git rm --cached --quiet
git ls-files | grep '__pycache__/' | grep -v '\.pyc$' | xargs -r git rm --cached --quiet
git commit -m "chore: untrack committed __pycache__/*.pyc (already gitignored)"
git ls-files | grep -c '\.pyc$'                            # expect 0
```

#### Test Spec
- `git ls-files | grep -c '\.pyc$'` → `0`.
- `.pyc` still on disk (e.g. `ls Bidomain/Engine_V1/cardiac_sim/__pycache__/*.pyc` non-empty).

#### Checklist
- [ ] `.pyc` untracked via `git rm --cached`
- [ ] Stray non-`.pyc` `__pycache__` index entries untracked
- [ ] Committed
- [ ] 0 tracked `.pyc`
- [ ] Disk `.pyc` untouched (Python imports fine)

#### Verify
```bash
test "$(git ls-files | grep -c '\.pyc$')" -eq 0 && echo "PYC UNTRACKED OK"
```

#### Exit Criteria
- [ ] 0 tracked `.pyc`; commit made on `main`

#### Risk
Empty `git ls-files '*.pyc'` → guarded by `xargs -r` (no-op if empty). Untracking never deletes source. NOTE (intentional): some tracked `.pyc` currently show as MODIFIED in the working tree; `git rm --cached` discards that pending index modification — this is fine and intended, because the `.pyc` are regenerated on next import and will never be tracked again.

### Step 1.2: Commit pending working-tree deletions
**Model**: sonnet

#### Read First
- `git status --short` — review what remains after Step 1.1 (bytecode noise now gone).

#### Why
On-disk deletions (`lbm_cardiac`, `Engines/lbm_v1`, `Surrogate/run_*.py` + docs) are correct prior cleanup never committed. Settling them gives Phases 2–4 a clean baseline.

#### Implementation Spec
**Files to modify:** stage on-disk deletions + already-intended tracked modifications. Do NOT stage new untracked dirs or anything Phase 2/3 handles.

#### Pseudocode / exact commands
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
git diff --cached --quiet || { echo "ABORT: index not empty before staging — run 'git reset' and re-review"; exit 1; }
git status --short | grep -E '^ ?D' | head -50      # review deletions
# Stage ONLY deleted tracked files — NOT modifications. This deliberately avoids
# sweeping in-progress modified binaries (e.g. boundary_conduction_speedup
# figures/*.pdf, *.mp4, *.png) into this "pending deletions" commit.
git ls-files --deleted | sed 's/^/   will rm: /'    # preview exactly what gets staged
git ls-files --deleted -z | xargs -0 -r git rm --quiet --
git status --short                                   # confirm: only D (deletions) staged; M left unstaged
git commit -m "chore: commit pending deletions (lbm_cardiac promotion, Engines/lbm_v1, legacy Surrogate scripts/docs)"
```

#### Test Spec
- After commit, `git status --short` shows NO staged deletions; modified tracked files (figures/videos) remain UNSTAGED (untouched by this step).

#### Checklist
- [ ] Reviewed deletion list — all intentional (cross-check `Research/Knowledge/lbm_cardiac.md` exists → promotion done)
- [ ] Staged ONLY deletions via `git ls-files --deleted | git rm` (NOT `git add -u`)
- [ ] Confirmed modified binaries (boundary figures/videos) are NOT in the staged set
- [ ] Committed
- [ ] No accidental staging of Phase 2/3 targets or unrelated modifications

#### Verify
```bash
test -f Research/Knowledge/lbm_cardiac.md && echo "lbm_cardiac promotion intact"
git diff --cached --name-status | grep -v '^D' | head   # expect EMPTY (only deletions committed)
```

#### Exit Criteria
- [ ] Pending deletions committed; ONLY deletions were staged (verified no `M`/`A` in cached diff)

#### Risk
Blanket `git add -u` would stage every tracked modification — including in-progress research binaries — folding them into a mislabeled commit. Mitigation: this step stages ONLY `git ls-files --deleted` (removals), never modifications. Modified tracked files are intentionally left for the user to handle in their own commits.

### Phase 1 Verification
```bash
conda run -n heart-conduction bash -c "cd Bidomain/Engine_V1 && python -c 'import cardiac_sim'" && echo "import OK"
test "$(git ls-files | grep -c '\.pyc$')" -eq 0 && echo "no tracked pyc"
```

### Phase 1 Exit Criteria
- [ ] 0 tracked `.pyc`
- [ ] Pending deletions committed
- [ ] Python imports unaffected

### Phase 1 Cleanup
- No float64/V5.3 concerns (no source edits). Confirm only index/disk-deletion changes were made.

**-> Commit point: 2 commits made above.**

---

## Phase 2: Consolidate images & videos into media/

**Goal**: Move ALL repo-wide images/videos into one canonical tree `media/{question}/{images|videos}/{YYYY-MM-DD}/`. This SUPERSEDES the earlier "delete Images/" design — instead of scattering figures next to scripts, everything lives under `media/`, grouped by research question, then images/videos, then dated session.
**Tier**: large
**Estimated scope**: ~516 media files (469 images + 47 videos) across 7 areas; question-mapping; a 322 MB bulk-output gitignore decision; then a separate renaming pass.

### Why split into dedicated plans
This is too large and judgment-heavy to inline. It is specified in two dedicated, self-contained plans in this folder:
- **`PLAN_media_consolidate.md`** — move/placement into the `media/` tree (keeps original filenames).
- **`PLAN_media_rename.md`** — the additional plan: rename each file to `{slug}_{NN}` via a reviewable manifest.

### Ordering & gate
Both run AFTER Phase 1 and BEFORE Phase 3, behind the Phase 0 **pushed-backup** hard gate (irreversible moves). Phase 3's "delete dead dups" and Phase 4 docs follow unchanged.

### Carryover from the old Phase 2 (still done, now inside consolidation)
- The authored IonicSurrogateV3 diagrams (`Images/_diagram_archive/`, `ionic_surrogate_v3.*`) are preserved — they map to `media/surrogate_pipeline/images/...` instead of `Surrogate/docs/diagrams/`.
- `.gitignore` figures/outputs convention is added by `PLAN_media_consolidate.md` (bulk `simulation` session folders under `media/` are gitignored).
- The byte-identical-duplicate insight (md5, not basename) is reused: when the same image exists in multiple places, only one copy lands in `media/`; the others are dropped (recoverable via Phase 0 backup).

**-> Commit point: consolidation + rename commits per the two dedicated plans.**

---

## Phase 3: Finish half-done migrations

**Goal**: Remove deprecated/dead duplicate trees left by incomplete migrations.
**Tier**: medium
**Estimated scope**: Migrate 4 PDFs, four deletions, verify no import/link breakage, one commit.

### Phase Context
- `Research/Q1_*…Q8_*` (8 dirs): old flat numbering, fully superseded by `Research/Active/` semantic folders. Content already migrated.
- `Research/papers/` (root, 5 PDFs): deprecated per CLAUDE.md. 4 are LBM refs (`rapaka_2012`, `belmiloudi_2019`, `campos_2016`, `lbm_review_macro_flows`) cited by `lbm_ep`; 1 orphan never summarized (`12859_2023_article_5513.pdf`, 6.8 MB).
- `LBM/Engine_V1/ionic/ionic/`: dead nested duplicate, zero imports (real code is `LBM/Engine_V1/ionic/`).
- `harness_v1/`: byte-identical dead copy of `cardiac_ml/`+`conf/`+`scripts/` (diff shows only missing `__pycache__`); zero references.
- **PRECONDITION (hard gate):** irreversible `git rm`. Do NOT start until `git ls-remote origin backup/pre-cleanup-2026-05-30` returns a SHA (push confirmed).

### Step 3.1: Migrate referenced PDFs, then delete root `Research/papers/`
**Model**: opus

#### Read First
- `ls Research/Active/lbm_ep/papers/ 2>/dev/null` — see what's already there.
- `grep -rn "Research/papers/\|papers/12859\|papers/rapaka\|papers/campos\|papers/belmiloudi\|papers/lbm_review" --include=*.md Research/ | head` — references to fix.

#### Why
The 4 LBM PDFs are still cited by `lbm_ep`; land them in that question's `papers/` before removing the root dir. The orphan was never integrated → drop (recoverable via backup).

#### Implementation Spec
**Files to move:** 4 LBM PDFs → `Research/Active/lbm_ep/papers/` (skip if already present).
**Files to delete:** `Research/papers/` including the orphan.

#### Pseudocode / exact commands
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
mkdir -p Research/Active/lbm_ep/papers
for p in rapaka_2012_lbm_ep belmiloudi_2019_coupled_lbm_fv campos_2016_lbm_gpu lbm_review_macro_flows; do
  if [ ! -e "Research/Active/lbm_ep/papers/$p.pdf" ]; then
    git mv "Research/papers/$p.pdf" "Research/Active/lbm_ep/papers/$p.pdf"
  else
    git rm "Research/papers/$p.pdf"
  fi
done
# Orphan PDF — confirm uncited before dropping (recoverable via backup regardless):
if grep -rln "12859_2023" --include=*.md Research/ | grep -v 'Q[1-8]_'; then
  echo "NOTE: orphan PDF is referenced in a surviving doc above — review before dropping"
fi
[ -e "Research/papers/12859_2023_article_5513.pdf" ] && git rm "Research/papers/12859_2023_article_5513.pdf"
# Remove now-empty root papers/ (idempotent):
[ -e "Research/papers" ] && { git rm -r --quiet Research/papers 2>/dev/null; rm -rf Research/papers; }
# List markdown refs to the old path for the agent to FIX (update to Active/lbm_ep/papers/):
grep -rln "Research/papers/" --include=*.md Research/ | while read -r m; do echo "FIX REF in: $m"; done
```

#### Test Spec
- `test ! -d Research/papers` and the 4 PDFs exist under `Research/Active/lbm_ep/papers/`.

#### Checklist
- [ ] 4 LBM PDFs in `lbm_ep/papers/`
- [ ] Orphan removed
- [ ] Root `Research/papers/` gone
- [ ] Markdown refs updated (INDEX.md, lbm_ep KNOWLEDGE — note Q4 refs vanish with Q-folders in 3.2)

#### Verify
```bash
test ! -d Research/papers && echo "root papers/ gone"
ls Research/Active/lbm_ep/papers/*.pdf | wc -l
```

#### Exit Criteria
- [ ] Root `papers/` removed; referenced PDFs preserved; live refs fixed

#### Risk
A PDF referenced only by a soon-deleted Q-folder. Mitigation: only `Active/` + `INDEX.md` refs matter; backup recovers any wrongly-dropped PDF.

### Step 3.2: Delete deprecated `Q1–Q8/`, dead `LBM/ionic/ionic/`, dead `harness_v1/`
**Model**: opus

#### Read First
- `grep -rln "Q[1-8]_" --include=*.py --include=*.md . | grep -v '\.git' | head` — confirm no live dep on Q-paths.
- `grep -rn "ionic.ionic\|ionic/ionic" LBM/Engine_V1 --include=*.py | head` — confirm nested dir unused.
- `grep -rln "harness_v1" . | grep -v '\.git' | head` — confirm zero references.

#### Why
Confirmed dead/superseded. Removing them is the bulk of the perceived "mess." Grep-gating first prevents deleting something still wired in.

#### Implementation Spec
**Files to delete:** `Research/Q1_*`…`Q8_*`, `LBM/Engine_V1/ionic/ionic/`, `harness_v1/`.

#### Pseudocode / exact commands
```bash
cd /home/norepinephrine/Documents/Heart-Conduction

set -o pipefail   # gates must observe pipe failures (else a broken pipe reads as "no match")
# --- REAL BLOCKING GATES (abort on live CODE dependency; not just echoes) ---
# A real dependency is a Python IMPORT, not a prose mention. Scope grep to *.py and
# exclude the doomed dir's own files. (Markdown mentions — incl. THIS PLAN.md — are docs,
# not dependencies, so they must NOT trigger an abort.)
if grep -rn "ionic\.ionic\|ionic/ionic" LBM/Engine_V1 --include=*.py 2>/dev/null \
     | grep -v 'LBM/Engine_V1/ionic/ionic/'; then
  echo "ABORT: live import of LBM ionic/ionic above — investigate before deleting"; exit 1
fi
if grep -rn "harness_v1" . --include=*.py 2>/dev/null | grep -vE '/\.git/|^\./harness_v1/|harness_v1/'; then
  echo "ABORT: live .py reference to harness_v1 above — investigate before deleting"; exit 1
fi
echo "GATES PASSED — proceeding with deletions"

# --- Per-path guarded removals (each independent; an unmatched glob skips, never aborts the rest) ---
for q in Research/Q1_* Research/Q2_* Research/Q3_* Research/Q4_* \
         Research/Q5_* Research/Q6_* Research/Q7_* Research/Q8_*; do
  [ -e "$q" ] && git rm -r --quiet "$q" && echo "removed $q"
done
[ -e LBM/Engine_V1/ionic/ionic ] && git rm -r --quiet LBM/Engine_V1/ionic/ionic && echo "removed nested ionic"
[ -e harness_v1 ] && git rm -r --quiet harness_v1 && echo "removed harness_v1"
```

#### Test Spec
- LBM tests pass (Phase 3 verification) — proves `ionic/ionic/` was dead.
- `git ls-files | grep -c '^Research/Q[1-8]_'` → 0.

#### Checklist
- [ ] Grep gates reviewed — no live deps
- [ ] Q1–Q8 removed
- [ ] `LBM/Engine_V1/ionic/ionic/` removed
- [ ] `harness_v1/` removed

#### Verify
```bash
test "$(git ls-files | grep -c '^Research/Q[1-8]_')" -eq 0 && echo "Q-folders gone"
test ! -d LBM/Engine_V1/ionic/ionic && test ! -d harness_v1 && echo "dead dups gone"
```

#### Exit Criteria
- [ ] All four targets removed; grep gates were clean

#### Risk
LBM imports `ionic.ionic` indirectly. Mitigation: Phase 3 Verification runs the LBM suite; if it breaks, `git checkout backup/pre-cleanup-2026-05-30 -- LBM/Engine_V1/ionic/ionic` and investigate.

### Phase 3 Verification
```bash
conda run -n heart-conduction bash -c "cd LBM/Engine_V1 && python -m pytest tests/ -q" 2>&1 | tail -15
conda run -n heart-conduction bash -c "cd Bidomain/Engine_V1 && pytest tests/ -q" 2>&1 | tail -15
```

### Phase 3 Exit Criteria
- [ ] LBM test suite passes (confirms `ionic/ionic/` dead)
- [ ] Bidomain suite passes (no collateral)
- [ ] Q1–Q8, root papers/, nested ionic, harness_v1 all removed

### Phase 3 Cleanup
- V5.3 untouched. No float64 concerns. Confirm `Research/INDEX.md` / `MASTER_KNOWLEDGE_INDEX.md` no longer reference deleted Q-folders.

**-> Commit point: `git commit -m "cleanup: remove deprecated Q1-Q8, root papers/, dead LBM ionic/ionic and harness_v1"`**

---

## Phase 4: Documentation + symlink fix

**Goal**: Make CLAUDE.md describe the real structure so the root stops *reading* as chaos; fix the broken navigation symlink.
**Tier**: small
**Estimated scope**: CLAUDE.md edits + one symlink fix, one commit.

### Phase Context
- Live-but-undocumented root dirs: `cardiac_core/` (consolidation API, 34 tests), `cardiac_ml/`+`conf/`+`scripts/`+`mlruns/`+`outputs/` (ML harness; `conf/`+`scripts/` at root by Hydra convention), `simulation/` (Zimmerman storage-tank harness for boundary research). `Engines/` and `Pipelines/` are symlink navigation indices.
- `Engines/cardiac_core` → `../../cardiac_core` is BROKEN (from `Engines/`, `../../` escapes the repo). Correct: `../cardiac_core`.

### Step 4.1: Fix `Engines/cardiac_core` symlink
**Model**: sonnet

#### Why
A broken symlink in the navigation index is a silent papercut; one-character fix.

#### Pseudocode / exact commands
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
rm Engines/cardiac_core
ln -s ../cardiac_core Engines/cardiac_core
[ -e Engines/cardiac_core ] && echo "RESOLVES OK" || echo "STILL BROKEN"
git add Engines/cardiac_core
```

#### Verify
```bash
test -e Engines/cardiac_core && echo "symlink OK"
```

#### Exit Criteria
- [ ] `Engines/cardiac_core` resolves

#### Risk
None material.

### Step 4.2: Document live root dirs in CLAUDE.md
**Model**: opus

#### Read First
- `CLAUDE.md` — "Active Engines" and "Supporting Components" tables, "Project Architecture" section.

#### Why
~80% of perceived mess is undocumented-but-healthy dirs. Documenting them is the cheapest leverage on cleanliness and prevents future confusion.

#### Implementation Spec
**Files to modify:** `CLAUDE.md`
- "Active Engines" table: add `cardiac_core` — `cardiac_core/` — Phase 0 done (34 tests) — unified cross-engine API (`monodomain()`/`bidomain()`/`lbm()`/mesh/analysis).
- "Supporting Components": add `cardiac_ml` harness (note `conf/`, `scripts/`, `mlruns/`, `outputs/` live at project root by Hydra convention); `simulation/` (Zimmerman storage-tank harness for boundary_conduction_speedup).
- New short "Navigation indices" note: `Engines/` and `Pipelines/` are symlink-only convenience indices → edit files in their real homes.

#### Pseudocode
Insert rows/notes matching existing table formatting. One line per entry; link paths.

#### Test Spec
- `grep -c 'cardiac_core\|cardiac_ml\|simulation/\|Pipelines/' CLAUDE.md` increases; manual read confirms accuracy.

#### Checklist
- [ ] cardiac_core row added
- [ ] cardiac_ml + conf/scripts/mlruns/outputs note added
- [ ] simulation/ documented
- [ ] Engines/Pipelines symlink note added
- [ ] Formatting matches existing tables

#### Verify
```bash
grep -n "cardiac_core\|cardiac_ml\|simulation/\|Navigation" CLAUDE.md | head
```

#### Exit Criteria
- [ ] Every live root dir documented in CLAUDE.md

#### Risk
Doc drift. Mitigation: descriptions from verified audit; keep terse.

### Phase 4 Verification
```bash
test -e Engines/cardiac_core && echo "symlink OK"
grep -q "cardiac_core" CLAUDE.md && grep -q "simulation/" CLAUDE.md && echo "docs updated"
```

### Phase 4 Exit Criteria
- [ ] Symlink resolves; CLAUDE.md documents all live root dirs

### Phase 4 Cleanup
- Re-read edited CLAUDE.md sections for accuracy. No code/float64/V5.3 concerns.

**-> Commit point: `git commit -m "docs: document cardiac_core/cardiac_ml/simulation, note Engines|Pipelines symlinks, fix cardiac_core symlink"`**

---

## Final Cleanup (cross-phase)

1. Confirm `main` is clean and all cleanup commits present:
```bash
git log --oneline -6
git status --short | wc -l
```
2. Verify nothing references deleted paths in committed docs:
```bash
# Exclude this question's own docs (PLAN.md / IDEALOG / KNOWLEDGE mention these paths by design):
grep -rn "Images/\|Research/papers/\|harness_v1\|Q[1-8]_" --include=*.md . \
  | grep -vE '\.git/|Surrogate/docs|plans/|Research/Active/research_environment_optimization/' | head
```
3. Re-run fast smoke of active suites (Bidomain + LBM) to confirm no regressions.
4. (Optional, USER DECISION) Push `main` to `origin` once satisfied. The backup branch remains regardless.
5. Domain checklist: float64 N/A · V5.3 untouched · no new experiments (no EXPERIMENT.md needed) · no cross-engine duplication introduced.
6. Archive this plan:
```bash
mkdir -p Research/Active/research_environment_optimization/plans
cp Research/Active/research_environment_optimization/PLAN.md \
   "Research/Active/research_environment_optimization/plans/2026-05-30_repository-structure-cleanup.md"
```

## Mutation Log
_(populated during execution — record backup SHA in Step 0.1, and any SKIPPED/SPLIT/INSERTED steps)_

**EXECUTED 2026-05-30 Step 0.1 (with incident + recovery)**:
- Local full-state backup commit `d125df30` (branch `backup/pre-cleanup-2026-05-30`, full history) created from main HEAD `5171bbce`.
- Full-history push to GitHub FAILED — 2 GiB pack cap exceeded (history contains a committed `venv/` of CUDA libs; `.git` ≈ 4.7 GB). DEVIATION: pushed a PARENTLESS orphan snapshot of the current tree (`a2af23c4`, ~1.2 GB) to remote `backup/pre-cleanup-2026-05-30`. Remote gate satisfied. ⚠️ GitHub backup has NO history (current tree only); full history is LOCAL-only at `d125df30`. Gitignored content (incl. 322 MB `simulation/outputs`) is in NEITHER backup — regenerable, accepted.
- INCIDENT: the branch-switch in Step 0.1 reverted `main`'s working tree (dirty changes + this session's plan files moved onto the backup branch). RECOVERED by `git merge --ff-only` the snapshot then `git reset 5171bbce` (mixed), restoring HEAD=`5171bbce` (V5.5 commits intact), the 176 original dirty changes, and the plan files. Verified. Step 0.1's branch-switch method corrected in the Phase Context note above — do NOT re-run Step 0.1.

**MUTATED 2026-05-30**: Step 1.1 MODIFIED — audit CRITICAL: documented that `git rm --cached` intentionally discards the pending modified-`.pyc` index state (regenerated on import).
**MUTATED 2026-05-30**: Step 1.2 MODIFIED — audit CRITICAL: replaced blanket `git add -u` (which would sweep in-progress boundary_conduction_speedup figure/video modifications into a mislabeled commit) with `git ls-files --deleted | git rm` so ONLY deletions are staged; added cached-diff verification.
**MUTATED 2026-05-30**: Step 2.1 MODIFIED — audit HIGH: base unique-file classification on `find Images` (on-disk) instead of `git ls-files` so the 4 untracked files are covered; relocation now ACTUALLY moves files (tracked+untracked) preserving subpaths and is idempotent; added a real HARD-GATE re-scan that `exit 1`s if any unique remains, blocking Step 2.2.
**MUTATED 2026-05-30**: Step 2.2 MODIFIED — audit HIGH/MEDIUM: split atomic `git rm -r Images Videos Media` into per-path guarded removals handling tracked + untracked remainder; idempotent.
**MUTATED 2026-05-30**: Step 2.3 MODIFIED — audit MEDIUM: added a concrete grep-based determination method for the diag_boundary golden-reference decision (load+assert ⇒ keep; write-only/none ⇒ untrackable).
**MUTATED 2026-05-30**: Step 3.1 MODIFIED — audit MEDIUM: orphan-PDF removal made idempotent + a citation check before drop; papers/ dir removal idempotent.
**MUTATED 2026-05-30**: Step 3.2 MODIFIED — audit HIGH/MEDIUM: converted advisory grep echoes into real blocking `if grep … ; then echo ABORT; exit 1` gates (ionic/ionic, harness_v1); split Q1–Q8 + combined `git rm` into per-path guarded, glob-safe, idempotent removals.
**MUTATED 2026-05-30**: Phase 0 / Phase 2 / Phase 3 MODIFIED — audit MEDIUM: push SUCCESS (`git ls-remote` confirms remote backup) is now a HARD GATE before any irreversible `git rm` in Phases 2–3 (Phase 1 non-destructive steps still gated only on local backup commit).

--- second-pass audit (2026-05-30) ---
**MUTATED 2026-05-30**: Step 2.1 MODIFIED — 2nd-pass CRITICAL: replaced name-only `is_unique()` with md5 CONTENT-based `is_redundant()`. Files are preserved unless a byte-identical external copy exists, eliminating the silent-deletion path where a same-basename-but-different-bytes authored file was misclassified as duplicate. Added `set -o pipefail`; `mv` failures now warned and caught by the gate.
**MUTATED 2026-05-30**: Step 3.2 MODIFIED — 2nd-pass HIGH: the `harness_v1` gate scanned `*.md` over `.` and FALSE-ABORTED on this PLAN.md's own text. Rescoped both gates to `*.py` (real imports), excluded the doomed dirs + `.git`, added `set -o pipefail`.
**MUTATED 2026-05-30**: Step 1.2 MODIFIED — 2nd-pass HIGH: added `git diff --cached --quiet` precheck so the deletions-only verify can't false-pass on a pre-dirtied index.
**MUTATED 2026-05-30**: Header MODIFIED — 2nd-pass HIGH (exec model): added "Execution model" note requiring `exit 1` gate blocks to run as a single script (temp-file + `bash`), so the hard gates actually halt rather than ending one pasted command.
**MUTATED 2026-05-30**: Step 2.3 / Step 2.1 comments / Final Cleanup MODIFIED — 2nd-pass MEDIUM/LOW: dropped stale ".gitignore 9 lines" assertion; clarified `git add -A "$DEST"` stages additions only (source-path deletions land in Step 2.2); Final Cleanup grep now excludes this question's own docs to avoid false-positive self-matches.
**OPEN (accepted, backup-mitigated)**: Step 3.1 orphan-PDF citation check remains advisory (echo, not `exit 1`) — orphan verified uncited and Phase 0 backup recovers any error; diag_boundary determination heuristic is line-coupled (fuzzy) but gates only a USER decision with the rule left commented (no destructive action).
