# PLAN: Consolidate all images & videos into media/

Created: 2026-05-30
Engine(s): All (repo-wide asset reorg — no engine logic changes)
Research question: [research_environment_optimization](README.md)
Parent: [PLAN.md](PLAN.md) Phase 2 (this supersedes the old "delete Images/" Phase 2)
Companion: [PLAN_media_rename.md](PLAN_media_rename.md) (runs AFTER this)

## Execution model (READ FIRST)
Every fenced `bash` block with an `exit 1` gate MUST run as a SINGLE script (write to a temp file, `bash` it, STOP on non-zero) — otherwise `exit 1` ends only one command. Blocks assume `set -o pipefail` (included inline).

## Objective
Move ALL repo-wide images/videos (~469 images + 47 videos across 7 areas) into one canonical tree:
```
media/{question}/{images|videos}/{YYYY-MM-DD}/<original-filename>
```
Grouped first by research question, then images vs videos, then a dated session subfolder. Filenames are kept AS-IS in this plan; `PLAN_media_rename.md` renames them afterward. Decided design (user, 2026-05-30): scope = everything repo-wide; move all (don't edit generator scripts); session = dated subfolder; bulk `simulation/outputs` is moved-but-gitignored.

## Success Criteria
- [ ] Every in-scope image/video lives under `media/{question}/{images|videos}/{date}/` or `media/_unmapped/`
- [ ] No image/video remains in the old scattered locations (except regenerable outputs scripts will recreate)
- [ ] Byte-identical duplicates collapsed to ONE copy (md5)
- [ ] Bulk `simulation` session folders under `media/` are gitignored (not committed)
- [ ] File count conserved: `moved + deduped + unmapped == original inventory`
- [ ] A dry-run manifest was reviewed before any move

## Architecture Changes
- NEW: `media/` tree at repo root (absorbs the old empty `Media/` stub)
- MOVE: images/videos from `Images/`, `Research/**`, `Bidomain/**`, `Monodomain/**`, `LBM/**`, `Builder/**`, `Surrogate/**`, `MonthlyReport/*/figures`, `simulation/outputs` → `media/`
- MOD: `.gitignore` — gitignore bulk `media/**/simulation*/` session dirs; remove now-stale `Media/` intent
- OUT OF SCOPE: `.pptx`/`.pdf` decks (not images/videos — stay in MonthlyReport); `.pt`/`.npz`/`.npy` (not images/videos)

## Known Failures / Corrections (inherited + new)
- Duplicate detection is by **md5 content**, never basename (a same-name/different-bytes file is NOT a duplicate).
- `simulation/outputs/` is 322 MB and currently UNTRACKED — committing it would bloat the 4.2 GB `.git`. It is moved for organization but its `media/` session dirs are gitignored.
- Mapping asset→question is heuristic; anything not confidently mapped goes to `media/_unmapped/` for human triage, NEVER silently guessed into the wrong question.
- Generator scripts are NOT edited (user choice) — some will recreate their old output paths on next run; accepted.

## Precondition (HARD GATE)
This plan performs irreversible moves/deletes. Do NOT start until the Phase 0 backup is pushed:
```bash
git ls-remote origin backup/pre-cleanup-2026-05-30 | grep . || { echo "ABORT: no remote backup"; exit 1; }
```
Also requires PLAN.md Phase 1 (git hygiene) committed, so the working tree is clean before a 500-file move.

---

## Phase A: Build & review the move manifest (NO moves yet)

**Goal**: Produce a reviewable `old_path,question,type,session_date,dest_path` manifest for every in-scope file. Nothing moves until the user reads it.
**Tier**: medium

### Phase Context
- In-scope extensions: images `png jpg jpeg svg gif`; videos `mp4 webm mov avi`. NOT `pptx pdf pt npz npy`.
- Question list (mapping targets): `boundary_conduction_speedup surrogate_pipeline ionic_model_optimization lbm_ep bidomain_parabolic_parabolic geometry_induced_pacemaking mature_hipsc_cm_models mesh_builder monthly_report_pipeline cardiac_ml_harness engine_consolidation research_environment_optimization`.
- Session date per file: tracked → `git log -1 --format=%ad --date=short -- <path>`; untracked → file mtime `date -r <path> +%F`.

### Step A.1: Generate the manifest
**Model**: opus

#### Read First
- `Research/Active/` folder names — the canonical question slugs.
- This plan's mapping table below.

#### Why
A 500-file move with question-mapping is too risky to do blind. The manifest makes every destination explicit and reviewable; `_unmapped` makes ambiguity visible instead of silently wrong.

#### Implementation Spec
**Source→question mapping (longest-prefix wins; first match):**
| Source path prefix / pattern | question |
|---|---|
| `Research/Active/{Q}/` (any media under a question folder) | `{Q}` (use that folder's slug) |
| `Bidomain/Engine_V1/tests/diag_boundary/` | `boundary_conduction_speedup` |
| `simulation/outputs/` | `boundary_conduction_speedup` (BULK → gitignored, see Phase C) |
| `LBM/` | `lbm_ep` |
| `Builder/` | `mesh_builder` |
| `MonthlyReport/*/figures/` (png/gif/mp4 only) | `monthly_report_pipeline` |
| `Images/_diagram_archive/`, `Images/ionic_surrogate_v3*`, `Images/generate_v3*` | `surrogate_pipeline` |
| `Surrogate/` | `surrogate_pipeline` |
| `Monodomain/_archive/` | `_unmapped/legacy` (triage — consider leaving in place) |
| `Images/` (anything else) | `_unmapped` (legacy snapshot; triage by content) |
| anything not matched above | `_unmapped` |

**Output:** `media/_manifest.csv` with header `old_path,question,type,session_date,dest_path`.

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
QUESTIONS="boundary_conduction_speedup surrogate_pipeline ionic_model_optimization lbm_ep bidomain_parabolic_parabolic geometry_induced_pacemaking mature_hipsc_cm_models mesh_builder monthly_report_pipeline cardiac_ml_harness engine_consolidation research_environment_optimization"
mkdir -p media
echo "old_path,question,type,session_date,dest_path" > media/_manifest.csv

classify_q() {  # echo question slug for a path
  local p="$1"
  case "$p" in
    Research/Active/*) echo "$p" | sed -E 's#Research/Active/([^/]+)/.*#\1#'; return;;
    Bidomain/Engine_V1/tests/diag_boundary/*) echo boundary_conduction_speedup; return;;
    simulation/outputs/*) echo boundary_conduction_speedup; return;;
    LBM/*) echo lbm_ep; return;;
    Builder/*) echo mesh_builder; return;;
    MonthlyReport/*/figures/*) echo monthly_report_pipeline; return;;
    Images/_diagram_archive/*|Images/ionic_surrogate_v3*|Images/generate_v3*) echo surrogate_pipeline; return;;
    Surrogate/*) echo surrogate_pipeline; return;;
    Monodomain/_archive/*) echo _unmapped/legacy; return;;
    *) echo _unmapped; return;;
  esac
}
sess_date() {  # YYYY-MM-DD from git, else mtime
  local d; d=$(git log -1 --format=%ad --date=short -- "$1" 2>/dev/null)
  [ -n "$d" ] || d=$(date -r "$1" +%F 2>/dev/null)
  echo "${d:-undated}"
}
# enumerate in-scope media, skip media/ itself, .git, pycache
find . -path ./.git -prune -o -path ./media -prune -o -type f \
   \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.svg' -o -iname '*.gif' \
      -o -iname '*.mp4' -o -iname '*.webm' -o -iname '*.mov' -o -iname '*.avi' \) -print 2>/dev/null \
 | grep -v '__pycache__' | sed 's#^\./##' | while IFS= read -r f; do
    q=$(classify_q "$f")
    case "$f" in *.mp4|*.MP4|*.webm|*.mov|*.avi) t=videos;; *) t=images;; esac
    d=$(sess_date "$f")
    b=$(basename "$f")
    echo "\"$f\",$q,$t,$d,\"media/$q/$t/$d/$b\"" >> media/_manifest.csv
 done
echo "manifest rows: $(($(wc -l < media/_manifest.csv) - 1))"
echo "--- per-question counts ---"; tail -n +2 media/_manifest.csv | cut -d, -f2 | sort | uniq -c | sort -rn
echo "--- UNMAPPED (review these) ---"; grep -c '_unmapped' media/_manifest.csv
```

#### Test Spec
- `media/_manifest.csv` row count == repo-wide in-scope media count (469 img + 47 video ≈ 516, minus anything already under `media/`).
- Per-question counts printed; `_unmapped` count surfaced.

#### Checklist
- [ ] Manifest generated with all 5 columns
- [ ] Row count matches independent `find` total
- [ ] Per-question + `_unmapped` tallies printed
- [ ] **USER REVIEW**: open `media/_manifest.csv`, sanity-check mappings & `_unmapped` before Phase B

#### Verify
```bash
test -s media/_manifest.csv && echo "manifest present: $(($(wc -l < media/_manifest.csv)-1)) rows"
```

#### Exit Criteria
- [ ] Manifest reviewed by user; mappings acceptable (or table adjusted and regenerated)

#### Risk
Mis-mapping. Mitigation: dry-run manifest + `_unmapped` bucket + Phase 0 backup. No file moves in Phase A.

### Phase A Exit Criteria
- [ ] Reviewed manifest exists; user approves the destination mapping.

**-> No commit (manifest is a scratch artifact; gitignored in Phase C).**

---

## Phase B: Move tracked media per manifest

**Goal**: `git mv` every TRACKED file to its manifest destination, collapsing md5-duplicates.
**Tier**: large

### Phase Context
- Tracked files only here (untracked bulk handled in Phase C). `git mv` preserves history.
- Duplicate rule: if two manifest entries resolve to the same dest dir with identical md5, keep the first, `git rm` the rest (recoverable via backup).

### Step B.1: Execute tracked moves + dedup
**Model**: opus

#### Read First
- `media/_manifest.csv` (the approved manifest).

#### Why
Manifest-driven moves are deterministic and auditable; dedup prevents the old "many copies of one figure" problem from re-entering `media/`.

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
declare -A SEEN  # dest -> md5 of first file placed there
tail -n +2 media/_manifest.csv | while IFS=, read -r oldq qq tt dd destq; do
  old=$(echo "$oldq" | sed 's/^"//;s/"$//')
  dest=$(echo "$destq" | sed 's/^"//;s/"$//')
  git ls-files --error-unmatch "$old" >/dev/null 2>&1 || continue   # tracked only
  [ -f "$old" ] || continue
  mkdir -p "$(dirname "$dest")"
  if [ -e "$dest" ]; then
    # collision in dest: if identical content, drop the source as a dup; else suffix
    if [ "$(md5sum "$old"|cut -d' ' -f1)" = "$(md5sum "$dest"|cut -d' ' -f1)" ]; then
      git rm -q "$old"; echo "DUP-DROP $old (== $dest)"; continue
    else
      dest="${dest%.*}__$(md5sum "$old"|cut -c1-6).${dest##*.}"   # disambiguate
    fi
  fi
  git mv "$old" "$dest" && echo "MV $old -> $dest"
done
```

#### Test Spec
- After run, `git ls-files` shows tracked media only under `media/` (no tracked png/jpg/mp4 left in old locations except any intentionally-excluded).

#### Checklist
- [ ] Ran as one script (gate semantics)
- [ ] All tracked manifest rows processed (moved or dup-dropped)
- [ ] Same-name/different-content collisions disambiguated with md5 suffix (not overwritten)
- [ ] No tracked image/video remains outside `media/` (verify below)

#### Verify
```bash
git ls-files '*.png' '*.jpg' '*.jpeg' '*.svg' '*.gif' '*.mp4' '*.webm' '*.mov' \
  | grep -v '^media/' | grep -v '__pycache__' | head -20
echo "^ should be EMPTY (all tracked media now under media/)"
```

#### Exit Criteria
- [ ] Tracked media fully under `media/`; collisions handled; dups dropped

#### Risk
Path with commas/quotes breaks CSV parse. Mitigation: filenames here are simple; the manifest quotes paths; spot-check the `MV`/`DUP-DROP` log. Backup recovers errors.

### Phase B Exit Criteria
- [ ] `git ls-files` media check is empty outside `media/`

**-> Commit point: `git commit -m "media: consolidate tracked images/videos into media/{question}/{type}/{date}/"`**

---

## Phase C: Move untracked bulk + gitignore + create skeleton

**Goal**: Move untracked media (chiefly `simulation/outputs`, 322 MB) into `media/` for organization, but gitignore those session dirs so they aren't committed.
**Tier**: medium

### Step C.1: Move untracked media (not committed)
**Model**: opus

#### Why
User wants everything organized under `media/`, but 322 MB of regenerable simulation output must not enter git. Move on disk, ignore in git.

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
tail -n +2 media/_manifest.csv | while IFS=, read -r oldq qq tt dd destq; do
  old=$(echo "$oldq" | sed 's/^"//;s/"$//'); dest=$(echo "$destq" | sed 's/^"//;s/"$//')
  git ls-files --error-unmatch "$old" >/dev/null 2>&1 && continue   # skip tracked (done in B)
  [ -f "$old" ] || continue
  mkdir -p "$(dirname "$dest")"
  [ -e "$dest" ] || mv "$old" "$dest" && echo "MV(untracked) $old -> $dest"
done
```

#### Checklist
- [ ] Untracked media moved into `media/`
- [ ] Originating dirs (e.g. `simulation/outputs/`) emptied of media (scripts may recreate later — accepted)

#### Verify
```bash
find simulation/outputs -type f \( -iname '*.png' -o -iname '*.mp4' \) 2>/dev/null | head; echo "^ should be empty"
```

#### Exit Criteria
- [ ] Untracked media relocated

#### Risk
Bulk size. Mitigation: gitignored in C.2 so no git impact.

### Step C.2: gitignore bulk + manifest; absorb old Media/ stub
**Model**: sonnet

#### Pseudocode (append to `.gitignore`)
```
# --- media/ consolidation (2026-05-30) ---
media/_manifest.csv
# Bulk regenerable simulation outputs: organized under media/ but NOT committed
media/boundary_conduction_speedup/**/simulation*/
media/**/*/__pycache__/
```
> Also remove the legacy empty `Media/` stub: `git rm -r Media 2>/dev/null; rm -rf Media`.
> Decision (surface in report): the gitignore glob for bulk simulation outputs assumes they sort under `boundary_conduction_speedup`. If the manifest placed any bulk elsewhere, widen the glob.

#### Verify
```bash
git check-ignore media/_manifest.csv && echo "manifest ignored"
```

#### Exit Criteria
- [ ] `.gitignore` updated; `Media/` stub gone

### Phase C Exit Criteria
- [ ] Untracked media organized under `media/`; bulk gitignored; counts conserved
- [ ] `media/_unmapped/` reviewed (move its contents into a question or accept as triage backlog)

### Phase C Cleanup
- Conserve check: `(files now under media/) + (dup-drops) >= original inventory`. Log the numbers.
- Confirm no broken `Images/`/`figures/` links in committed `*.md`: `grep -rn "Images/\|/figures/" --include=*.md . | grep -vE '\.git/|research_environment_optimization/' | head` — fix or note.

**-> Commit point: `git commit -m "media: organize untracked bulk, gitignore simulation outputs, drop Media/ stub"`**

---

## Final Cleanup
1. `git log --oneline -4` — consolidation commits present.
2. Verify counts conserved (log the inventory math).
3. Hand off to `PLAN_media_rename.md` for the rename pass.
4. Archive this plan:
```bash
mkdir -p Research/Active/research_environment_optimization/plans
cp Research/Active/research_environment_optimization/PLAN_media_consolidate.md \
   "Research/Active/research_environment_optimization/plans/2026-05-30_media-consolidate.md"
```

## Mutation Log
_(populated during execution)_
