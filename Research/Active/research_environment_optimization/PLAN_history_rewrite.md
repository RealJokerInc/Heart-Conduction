# PLAN: Strip committed venv/ from git history

Created: 2026-05-30
Engine(s): All (repo-wide history rewrite — no source logic changes)
Research question: [research_environment_optimization](README.md)
Relationship: PREREQUISITE for the full-history GitHub backup that PLAN.md Phase 0 wanted. Run this BEFORE the structure cleanup (PLAN.md Phases 1+) and the media consolidation.

## Execution model (READ FIRST)
This plan REWRITES ALL GIT HISTORY (every commit SHA changes) and FORCE-PUSHES. It is the highest-risk operation in this engagement. Each fenced block with an `exit 1` gate runs as ONE script (temp file + `bash`, STOP on non-zero). Do not improvise around a failed gate.

## Objective
A `venv/` of CUDA/PyTorch libraries was committed into history: **7,625 MB of blobs** across history (top: `libtorch_cuda.so` 649 MB, `libcublasLt` 482 MB, …). It bloats `.git` to 4.7 GB, blocks any full-history push (GitHub 2 GiB pack cap), and slows every clone/fetch. `venv/` is NOT in the current tree (0 tracked files) — it only haunts history. Strip it from ALL commits to take `.git` 4.7 GB → ~1 GB and make a full-history backup/push possible.

## Success Criteria
- [ ] `venv/` absent from ALL history: `git log --all --oneline -- venv | head` empty AND `git rev-list --objects --all | grep -c '^.* venv/'` == 0
- [ ] `.git` size ≤ ~1.3 GB after gc
- [ ] Current working tree byte-identical to pre-rewrite for all NON-venv paths (no source/data lost)
- [ ] Full-history `git push` to origin/main succeeds (pack < 2 GiB)
- [ ] A verified pre-rewrite bundle backup exists OUTSIDE the repo
- [ ] `origin` remote re-added (filter-repo drops it) and main pushed

## What is NOT changed
- Source code, research files, current tracked content (only `venv/` history is removed)
- The on-disk untracked `venv/` (your live virtualenv keeps working — it's not in git)
- Engine behavior / tests

## Known consequences (ACCEPT before running)
- **Every commit SHA changes.** Any reference to an old SHA breaks. Known references to update afterward: MEMORY.md commit pins (`8f191f77` model-tree drift check; `b9d9c718`, `eb057232`, `57b7efac`, `77114cb4`, `2d90fdaf`, `b20fabf7`, `67d3e6a8` cardiac_ml phase commits). Post-rewrite, the drift check `git diff --quiet 8f191f77 -- Surrogate/...` will fail — re-pin to the rewritten SHA.
- **Force-push rewrites origin/main.** Acceptable for a solo repo; any other clone of this repo becomes incompatible and must re-clone.
- **GitHub disk** won't shrink until stale remote branches pinning old objects (`snapshot-2026-04-09`, the orphan `backup/pre-cleanup-2026-05-30`, old `origin/main`) are deleted and GitHub GCs (may lag / need support for true reclaim). Local shrink is immediate.
- `git-filter-repo` strips the `origin` remote as a safety measure — must be re-added.

## Tooling
- Install `git-filter-repo` into the conda env (preferred over BFG; precise, fast): `conda run -n heart-conduction pip install git-filter-repo`. (BFG via the present Java 21 is the fallback.)
- Disk free: 3.6 TB — bundle (~4 GB) is fine.

---

## Phase 0: Pre-rewrite safety backup (HARD GATE)

**Goal**: A complete, verified, OUTSIDE-the-repo backup of ALL history+refs that survives the rewrite. This is the only true rollback.
**Tier**: small

### Phase Context
- The existing local branch `d125df30` and the remote orphan `a2af23c4` are NOT sufficient rollbacks: `d125df30` gets rewritten by filter-repo, and the orphan has no history. A `git bundle --all` is the canonical full-repo backup in one restorable file.
- Audit-confirmed: `--all` captures local heads/tags AND remote-tracking refs (so `origin/main`, `origin/snapshot-2026-04-09`, the orphan backup are all in the bundle); both named local branches exist and are bundled. Caveat: the **reflog is NOT in the bundle** (filter-repo expires it) — rollback relies on the bundled refs, not reflog, so this is acceptable.

### Step 0.1: Bundle all refs + record current SHAs
**Model**: sonnet

#### Pseudocode / exact commands
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
BUNDLE="$HOME/heart-conduction-PREWRITE-2026-05-30.bundle"
git bundle create "$BUNDLE" --all
git bundle verify "$BUNDLE" | tail -3
git for-each-ref --format='%(objectname) %(refname)' > "$HOME/heart-conduction-PREWRITE-refs.txt"
ls -lh "$BUNDLE"
echo "BUNDLE OK: $BUNDLE"
```
> Optional belt-and-suspenders (disk is ample): `tar czf $HOME/heart-conduction-PREWRITE.tar.gz --exclude=venv .` for a full working-tree copy too.

#### Verify
```bash
git bundle verify "$HOME/heart-conduction-PREWRITE-2026-05-30.bundle" && echo "RESTORABLE"
```

#### Exit Criteria
- [ ] Bundle created and `git bundle verify` passes
- [ ] Ref-SHA list saved

#### Risk
Corrupt/partial bundle. Mitigation: `git bundle verify` is mandatory; do NOT proceed to any rewrite step until it passes.

**-> HARD GATE: no rewrite until the bundle verifies.**

---

## Phase 1: Freeze working state into history

**Goal**: Commit the current 176 working changes + this session's plan files so NOTHING uncommitted is at risk during the rewrite (filter-repo operates on committed history; it will `reset --hard` the tree).
**Tier**: small

### Step 1.1: Commit current state
**Model**: opus

#### Why
filter-repo rewrites committed history and hard-resets the worktree to the rewritten HEAD. Any uncommitted change would be destroyed. Committing first makes the rewrite safe and keeps the in-progress work (it's reorganized later by the cleanup/media plans).

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
# GUARD (audit HIGH): gitignore venv BEFORE `git add -A`, and abort if a live venv is
# tracked — prevents re-committing 7.6 GB of libs right before we strip them.
grep -qxF 'venv/' .gitignore || echo 'venv/' >> .gitignore
git check-ignore venv >/dev/null 2>&1 || { echo "ABORT: venv still not ignored"; exit 1; }
[ "$(git ls-files venv | wc -l)" -eq 0 ] || { echo "ABORT: venv has TRACKED files — investigate before add -A"; exit 1; }
git add -A
git status --short | head
git commit -m "checkpoint: freeze working state (+ gitignore venv) before history rewrite (2026-05-30)"
git rev-parse HEAD
```
> This is a deliberate "kitchen-sink" commit (includes the mess + plans). It is FINE: the structure-cleanup and media plans reorganize it afterward. Do NOT spend effort curating it here.

#### Verify
```bash
git status --short | wc -l   # expect 0 (clean tree)
```

#### Exit Criteria
- [ ] Working tree clean; everything committed

#### Risk
Committing junk. Mitigation: acceptable — later phases clean it; bundle has the exact prior state too.

**-> Commit point: the checkpoint commit.**

---

## Phase 2: Prune extra refs so only main is rewritten

**Goal**: Delete local branches that would otherwise be rewritten/kept by filter-repo, leaving a single clean line of history. All are preserved in the Phase 0 bundle.
**Tier**: small

### Step 2.1: Drop extra local branches
**Model**: sonnet

#### Why
filter-repo rewrites ALL refs it sees. Fewer refs = a clean single-history result and no stray big objects retained. `backup/pre-cleanup-2026-05-30` (d125df30) and `snapshot-2026-04-09` are fully captured in the bundle + remote.

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
git branch -D backup/pre-cleanup-2026-05-30 2>/dev/null || true
git branch -D snapshot-2026-04-09 2>/dev/null || true
git branch        # expect: only main (and you are on it)
git tag           # if any tags reference history, note them (filter-repo rewrites tags too)
```

#### Verify
```bash
test "$(git branch | wc -l)" -eq 1 && echo "only main remains"
```

#### Exit Criteria
- [ ] Only `main` remains locally; bundle still holds the deleted branches

#### Risk
Deleting a branch you needed. Mitigation: bundle restore (`git fetch <bundle> <ref>`).

---

## Phase 3: Rewrite history (strip venv/)

**Goal**: Remove `venv/` from every commit. (Audit-verified: there are NO tracked nested `.git` dirs under `code_examples/` — that optional strip was a no-op and is dropped.)
**Tier**: large

### Step 3.1: Install tool + dry-run analysis
**Model**: opus

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction pip install git-filter-repo
conda run -n heart-conduction git filter-repo --analyze
# inspect the report it writes under .git/filter-repo/analysis/ — confirm venv dominates
head -40 .git/filter-repo/analysis/path-all-sizes.txt 2>/dev/null
```

#### Exit Criteria
- [ ] filter-repo installed; analysis confirms `venv/` is the dominant path

### Step 3.2: Execute the strip
**Model**: opus

#### Why
`--path venv --invert-paths` removes exactly the `venv/` tree from all commits, leaving everything else byte-identical.

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
# Primary: strip venv from all history. --force needed (repo has a remote / not fresh clone).
conda run -n heart-conduction git filter-repo --path venv --invert-paths --force
# NOTE (audit): the optional code_examples/*/.git strip is a NO-OP (not tracked) — omitted.
# filter-repo ALREADY expires reflogs + runs gc as its final step, so NO manual gc is
# needed. (`git gc --prune=now` would be redundant; `--aggressive` just wastes CPU.)
du -sh .git
```

#### Test Spec
- `git log --all --oneline -- venv` → EMPTY.
- `git rev-list --objects --all | grep -c ' venv/'` → 0.
- `.git` ≤ ~1.3 GB.
- Current tree unchanged for non-venv paths (spot-check a few files + `git status` clean).

#### Checklist
- [ ] Ran as one script; filter-repo completed without error
- [ ] reflog expired + aggressive gc run
- [ ] venv absent from history (both checks)
- [ ] `.git` shrank to ~1 GB
- [ ] A known source file (e.g. `Bidomain/Engine_V1/cardiac_sim/__init__.py`) still present & unchanged

#### Verify
```bash
echo "venv in history: $(git rev-list --objects --all | grep -c ' venv/')  (want 0)"
du -sh .git
conda run -n heart-conduction bash -c "cd Bidomain/Engine_V1 && python -c 'import cardiac_sim' && echo import-OK"
```

#### Exit Criteria
- [ ] venv gone from history, .git ~1 GB, current content intact, imports OK

#### Risk
filter-repo removes more than intended / corrupts. Mitigation: Phase 0 bundle is the rollback — `git clone bundle restored && diff`. If anything looks wrong, STOP and restore from bundle.

---

## Phase 4: Re-add remote and force-push

**Goal**: Restore `origin` (filter-repo drops it) and push the rewritten, slim history.
**Tier**: medium

### Step 4.1: Re-add origin, force-push main
**Model**: opus

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
git remote -v | grep origin || git remote add origin git@github.com:RealJokerInc/Heart-Conduction.git
git push --force origin main 2>&1 | tail -15   # rewritten history; pack now < 2 GiB
```

#### Test Spec
- Push succeeds (no "pack exceeds 2.00 GiB").
- `git ls-remote origin main` matches local `git rev-parse main`.

#### Checklist
- [ ] origin re-added
- [ ] `git push --force origin main` succeeded
- [ ] remote main == local main

#### Verify
```bash
test "$(git ls-remote origin main | cut -f1)" = "$(git rev-parse main)" && echo "remote main in sync"
```

#### Exit Criteria
- [ ] Full-history main pushed; remote == local

#### Risk
Push still too big (other large blobs remain). Mitigation: re-run Step 3 analysis; strip the next-biggest offender; re-push.

### Step 4.2: Clean stale remote branches (optional, enables GitHub-side shrink)
**Model**: sonnet

#### Why
Old venv objects stay on GitHub while any remote ref reaches them. Deleting stale branches lets GitHub GC reclaim space.

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
# Only after confirming the rewritten main is good and you no longer need these:
git push origin --delete snapshot-2026-04-09 2>/dev/null || true
git push origin --delete backup/pre-cleanup-2026-05-30 2>/dev/null || true   # the orphan snapshot
```
> Keep the LOCAL bundle backup regardless. Delete the orphan remote backup only once you trust the rewritten main.

#### Exit Criteria
- [ ] Stale remote branches removed (or consciously kept)

---

## Phase 5: Verify by fresh clone

**Goal**: Prove the rewritten remote is clean, small, and complete.
**Tier**: small

### Step 5.1: Clone to /tmp and compare
**Model**: opus

#### Pseudocode
```bash
set -o pipefail
rm -rf /tmp/hc-verify
git clone git@github.com:RealJokerInc/Heart-Conduction.git /tmp/hc-verify
du -sh /tmp/hc-verify/.git
test -d /tmp/hc-verify/venv && echo "FAIL: venv present" || echo "OK: no venv in clone history"
# content parity vs working tree. Exclude ALL gitignored/untracked-on-disk paths so the
# diff shows only REAL discrepancies (audit MEDIUM: incomplete excludes + head can mask loss).
diff -rq --exclude=.git --exclude=venv --exclude=outputs --exclude=mlruns \
         --exclude=__pycache__ --exclude=.pytest_cache --exclude=archive \
         --exclude=WHITEBOARD.md --exclude='*.pyc' --exclude='_manifest.csv' \
  /tmp/hc-verify /home/norepinephrine/Documents/Heart-Conduction > /tmp/hc-diff.txt
echo "REAL discrepancies (must be 0 — inspect EVERY line, do NOT truncate): $(wc -l < /tmp/hc-diff.txt)"
cat /tmp/hc-diff.txt
```

#### Exit Criteria
- [ ] Fresh clone is ~1 GB, contains no venv history, content matches working tree

#### Risk
Clone diff shows unexpected loss. Mitigation: investigate; bundle restore available.

---

## Final Cleanup
1. Update MEMORY.md SHA references to the rewritten commits (esp. the `8f191f77` drift-check pin) — or note they're retired.
2. Keep `$HOME/heart-conduction-PREWRITE-2026-05-30.bundle` until fully satisfied (then delete to reclaim ~4 GB).
3. `venv/` was added to `.gitignore` in Phase 1 (audit fix) — verify it's still present and committed (`git check-ignore venv`).
4. Hand back to PLAN.md Phase 1 (structure cleanup) — Phase 0 there is now SATISFIED (full-history backup achievable; remote main is the clean baseline).
5. Archive this plan:
```bash
cp Research/Active/research_environment_optimization/PLAN_history_rewrite.md \
   "Research/Active/research_environment_optimization/plans/2026-05-30_history-rewrite.md"
```

## Mutation Log
_(populated during execution)_

**REVISED 2026-05-30 (adversarial audit, 0 crit / 1 high / 1 med / 4 low — verdict: safe with fixes):**
- HIGH FIX — Phase 1 Step 1.1: gitignore `venv/` + guards (`git check-ignore`, abort if venv tracked) NOW run BEFORE `git add -A`, so a live on-disk venv can't be re-committed pre-strip.
- MEDIUM FIX — Phase 5 Step 5.1: verify-diff now excludes all gitignored/on-disk paths (`__pycache__`, `.pytest_cache`, `archive`, `WHITEBOARD.md`, `*.pyc`, `_manifest.csv`) and writes full output (no `head` truncation) so real loss can't be masked.
- LOW FIX — Phase 3 Step 3.2: removed redundant manual `reflog expire`/`gc --aggressive` (filter-repo gc's internally); corrected the `code_examples/*/.git` optional strip to a documented no-op (verified untracked).
- LOW NOTE — Phase 0: documented that `--all` bundles remote-tracking refs but NOT the reflog (rollback uses bundled refs).
- CONFIRMED by audit: `--path venv --invert-paths` is correct (no false "venv" matches in history); shrink estimate validated (7625 MB venv vs 1270 MB keep → ~1–1.3 GB packed, push < 2 GiB); bundle is a genuine restorable backup capturing all 3 branches.
