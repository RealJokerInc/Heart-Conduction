# PLAN: Rename media/ files to a fixed format

Created: 2026-05-30
Engine(s): All (asset renaming — no logic changes)
Research question: [research_environment_optimization](README.md)
Depends on: [PLAN_media_consolidate.md](PLAN_media_consolidate.md) (must complete first)

## Execution model (READ FIRST)
Fenced `bash` blocks with `exit 1` gates run as a SINGLE script (temp file + `bash`, STOP on non-zero). Blocks assume `set -o pipefail`.

## Objective
Rename every file already consolidated under `media/{question}/{images|videos}/{date}/` to the fixed format:
```
media/{question}/{images|videos}/{date}/{slug}_{NN}.{ext}
```
- `{slug}` = sanitized current basename (sans extension): lowercase, non-alphanumeric → `-`, collapse repeats, trim.
- `{NN}` = 2-digit sequence (`01`, `02`, …) that orders/deduplicates files sharing a slug WITHIN the same session folder.
- Date + question are carried by the folder path (chosen format: `slug_NN`).

Example: `media/boundary_conduction_speedup/images/2026-05-28/Vm_heatmap_t10.png` → `.../inverse-crescent-bc_01.png` (slug curated) or, by default, `.../vm-heatmap-t10_01.png` (auto-derived).

## Success Criteria
- [ ] Every file under `media/` matches `{slug}_{NN}.{ext}` within its session folder
- [ ] No two files in the same session folder share a name (NN disambiguates)
- [ ] A rename manifest (old→new) was reviewed before applying
- [ ] File count conserved (renames only, zero deletions)
- [ ] In-repo references to renamed files updated (or flagged)

## Architecture Changes
- RENAME (in place): all files under `media/**` to `{slug}_{NN}.{ext}`
- NEW (scratch, gitignored): `media/_rename_manifest.csv`

## Known Failures / Corrections
- Default slug is auto-derived from the existing filename — NOT a hand-written description. The user may curate slugs in the manifest BEFORE applying; this plan does not invent semantic names.
- Rename only — never deletes. A name collision is resolved by incrementing `NN`, never by overwriting.
- Applies to BOTH tracked (`git mv`) and untracked/gitignored-bulk (`mv`) files so the on-disk tree is uniform; gitignored files just won't show in git history.

## Precondition (HARD GATE)
```bash
test -d media || { echo "ABORT: media/ not built — run PLAN_media_consolidate.md first"; exit 1; }
git ls-remote origin backup/pre-cleanup-2026-05-30 | grep . || { echo "ABORT: no remote backup"; exit 1; }
```

---

## Phase A: Build & review the rename manifest (NO renames)

**Goal**: Produce `media/_rename_manifest.csv` (`old_path,new_path`) with auto-derived slugs + NN, for review/curation before applying.
**Tier**: medium

### Step A.1: Generate manifest
**Model**: opus

#### Read First
- The slug/NN rules above.

#### Why
Renaming hundreds of files is irreversible-in-spirit (history churn); a reviewable manifest lets the user curate slugs and catch bad derivations before anything moves.

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
echo "old_path,new_path" > media/_rename_manifest.csv
# Process per session folder so NN is scoped correctly.
find media -type d | while IFS= read -r dir; do
  # only leaf session dirs that directly contain files
  ls -1p "$dir" 2>/dev/null | grep -qv '/$' || continue
  declare -A CNT       # slug -> running count within this dir
  find "$dir" -maxdepth 1 -type f ! -name '_*' | sort | while IFS= read -r f; do
    b=$(basename "$f"); ext="${b##*.}"; stem="${b%.*}"
    slug=$(echo "$stem" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')
    [ -n "$slug" ] || slug=file
    # NN: count existing manifest rows already targeting this slug in this dir
    n=$(grep -c ",\"$dir/$slug"'_[0-9]' media/_rename_manifest.csv 2>/dev/null)
    nn=$(printf "%02d" $((n+1)))
    echo "\"$f\",\"$dir/${slug}_${nn}.${ext,,}\"" >> media/_rename_manifest.csv
  done
done
echo "rename rows: $(($(wc -l < media/_rename_manifest.csv)-1))"
echo "--- collisions check (same new_path twice = BUG) ---"
tail -n +2 media/_rename_manifest.csv | cut -d, -f2 | sort | uniq -d | head
```

#### Test Spec
- Row count == number of files under `media/` (excluding `_*` scratch).
- The collisions check prints NOTHING (every `new_path` unique).

#### Checklist
- [ ] Manifest has `old_path,new_path` for every media file
- [ ] No duplicate `new_path` (collision check empty)
- [ ] **USER REVIEW / optional curation**: edit slugs in the manifest for any files you want named meaningfully (keep `_NN` + ext)

#### Verify
```bash
test -s media/_rename_manifest.csv && echo "rows: $(($(wc -l < media/_rename_manifest.csv)-1))"
tail -n +2 media/_rename_manifest.csv | cut -d, -f2 | sort | uniq -d | grep . && echo "COLLISION!" || echo "no collisions"
```

#### Exit Criteria
- [ ] Reviewed (and optionally curated) manifest with zero collisions

#### Risk
Auto-slug ugliness. Mitigation: manifest is editable before apply; NN guarantees uniqueness regardless.

**-> No commit (manifest is scratch; add `media/_rename_manifest.csv` to gitignore).**

---

## Phase B: Apply renames

**Goal**: Apply the manifest — `git mv` tracked, `mv` untracked — collision-safe.
**Tier**: medium

### Step B.1: Execute renames
**Model**: opus

#### Read First
- `media/_rename_manifest.csv` (curated).

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
tail -n +2 media/_rename_manifest.csv | while IFS=, read -r oq nq; do
  old=$(echo "$oq" | sed 's/^"//;s/"$//'); new=$(echo "$nq" | sed 's/^"//;s/"$//')
  [ "$old" = "$new" ] && continue
  [ -f "$old" ] || { echo "skip (missing): $old"; continue; }
  [ -e "$new" ] && { echo "ABORT: target exists, manifest collision: $new"; exit 1; }
  if git ls-files --error-unmatch "$old" >/dev/null 2>&1; then git mv "$old" "$new"; else mv "$old" "$new"; fi
  echo "REN $old -> $new"
done
```

#### Test Spec
- Every file under `media/` matches `*_[0-9][0-9].*` within its session dir; counts unchanged from pre-rename.

#### Checklist
- [ ] Ran as one script
- [ ] All rows applied (or skipped-missing logged)
- [ ] No `ABORT: target exists`
- [ ] File count under `media/` unchanged

#### Verify
```bash
find media -type f ! -name '_*' | grep -vE '_[0-9]{2}\.[A-Za-z0-9]+$' | head
echo "^ should be EMPTY (all files match slug_NN.ext)"
```

#### Exit Criteria
- [ ] All media files conform to `{slug}_{NN}.{ext}`

#### Risk
Mid-run abort leaves a partial rename. Mitigation: idempotent (already-renamed rows skip via `old==new`/missing); re-run after fixing the colliding manifest row. Backup recovers.

### Step B.2: Update in-repo references
**Model**: opus

#### Why
Docs/scripts may reference old media basenames; stale links should be fixed or flagged.

#### Pseudocode
```bash
set -o pipefail
cd /home/norepinephrine/Documents/Heart-Conduction
# Flag committed docs that reference old image/figure basenames now renamed:
grep -rn "\.png\|\.mp4\|\.gif\|\.jpg\|/figures/\|Images/" --include=*.md . \
  | grep -vE '\.git/|research_environment_optimization/|media/_' | head -40
echo "^ review: update any of these links to the new media/ paths"
```

#### Checklist
- [ ] Reviewed flagged references; updated the ones pointing at moved/renamed assets
- [ ] Remaining hits are intentional (e.g. this question's own plan docs)

#### Verify
```bash
echo "manual review step — confirm no broken asset links remain in committed docs"
```

#### Exit Criteria
- [ ] Doc references reconciled or explicitly accepted

#### Risk
Missed reference → broken link in a doc. Mitigation: grep sweep + low blast radius (docs, not code).

### Phase B Exit Criteria
- [ ] All `media/` files conform to the format; references reconciled

### Phase B Cleanup
- Add `media/_rename_manifest.csv` to `.gitignore` (scratch). Confirm count conservation vs pre-rename.

**-> Commit point: `git commit -m "media: rename consolidated assets to {slug}_{NN} fixed format"`**

---

## Final Cleanup
1. `git log --oneline -3` — rename commit present.
2. Confirm `find media -type f ! -name '_*' | grep -vE '_[0-9]{2}\.'` is empty.
3. Archive this plan:
```bash
cp Research/Active/research_environment_optimization/PLAN_media_rename.md \
   "Research/Active/research_environment_optimization/plans/2026-05-30_media-rename.md"
```

## Mutation Log
_(populated during execution)_
