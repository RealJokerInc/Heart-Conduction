# PLAN: Merge NOTEBOOK.md into IDEALOG.md

Created: 2026-03-18
Engine(s): None (cross-cutting workflow)
Research question: [research_environment_optimization](README.md)
Source: [IDEALOG.md](IDEALOG.md) — "2026-03-18: NOTEBOOK.md merged back into IDEALOG.md"

## Objective
Remove NOTEBOOK.md as a separate document. IDEALOG.md becomes the single log file for both strategic insights and raw technical detail. Eliminates routing friction that caused NOTEBOOK.md to go unused during `/reason` sessions.

## Success Criteria
- [ ] No skill file references NOTEBOOK.md as a separate document
- [ ] CLAUDE.md architecture section shows IDEALOG.md only (no NOTEBOOK.md)
- [ ] `.gitignore` no longer lists NOTEBOOK.md
- [ ] Existing NOTEBOOK.md content merged into IDEALOG.md
- [ ] `/save-session` Job 6 (NOTEBOOK graduation) removed or repurposed
- [ ] `/reason-end` no longer wipes NOTEBOOK.md separately

## Architecture Changes
- DEL: `Research/Active/research_environment_optimization/NOTEBOOK.md` — content merged into IDEALOG.md
- MOD: `CLAUDE.md:124-137` — remove NOTEBOOK.md from architecture table and routing rules
- MOD: `.claude/skills/reason/SKILL.md` — remove NOTEBOOK.md routing, write everything to IDEALOG.md
- MOD: `.claude/skills/quicksave/SKILL.md` — remove NOTEBOOK.md references, write to IDEALOG.md only
- MOD: `.claude/skills/blueprint/SKILL.md` — read IDEALOG.md for technical detail (was NOTEBOOK.md)
- MOD: `.claude/skills/save-session/SKILL.md` — remove Job 6 (NOTEBOOK graduation), update job count 6→5
- MOD: `.claude/skills/reason-end/SKILL.md` — remove NOTEBOOK.md wipe step
- MOD: `.claude/skills/strategic-compact/SKILL.md` — remove NOTEBOOK.md from survival table
- MOD: `.gitignore` — remove `**/NOTEBOOK.md` line
- MOD: `Research/Active/research_environment_optimization/KNOWLEDGE.md` — update NOTEBOOK.md design section

## Known Failures (from IDEALOG)
- NOTEBOOK.md routing friction — the rule "strategic → IDEALOG, technical → NOTEBOOK" required constant judgment calls that faded from context in long conversations, leading to everything defaulting to IDEALOG anyway.

---

## Phase 1: Merge Content + Update Skills

**Goal**: Move NOTEBOOK.md content into IDEALOG.md, update all 6 skill files that reference NOTEBOOK.md, update CLAUDE.md and .gitignore. All changes ship together to avoid inconsistent state.
**Tier**: medium
**Estimated scope**: 10 files, ~69 occurrences of "NOTEBOOK" to address

### Phase Context
Every change is a text edit to a markdown file — no code, no tests, no engine work. The risk is missing a reference, not breaking functionality. All NOTEBOOK.md references must become IDEALOG.md references or be removed entirely. The existing NOTEBOOK.md for research_environment_optimization has valuable technical content (tmux/zellij commands, glow findings, pane ratios) that must be merged into IDEALOG.md before deletion.

### Step 1.1: Merge existing NOTEBOOK.md content into IDEALOG.md
**Model**: sonnet

#### Read First
- `Research/Active/research_environment_optimization/NOTEBOOK.md` — full content to merge
- `Research/Active/research_environment_optimization/IDEALOG.md` — target file

#### Why
The only existing NOTEBOOK.md has technical findings (tmux commands, glow rendering, pane ratios) that would be lost if we just delete it. Must merge into IDEALOG.md first.

#### Implementation Spec
**Files to modify:** `Research/Active/research_environment_optimization/IDEALOG.md` — append NOTEBOOK.md content as a new thread entry
**Files to delete:** `Research/Active/research_environment_optimization/NOTEBOOK.md` — after merge

#### Pseudocode
```
1. Read NOTEBOOK.md content
2. Append to IDEALOG.md Thread section as:
   ### 2026-03-17: Technical findings (merged from NOTEBOOK.md)
   {full NOTEBOOK.md content}
3. Delete NOTEBOOK.md
```

#### Test Spec
- Manual: verify IDEALOG.md contains all NOTEBOOK.md content (tmux commands, glow findings, comparison table, pane ratios)
- Manual: verify NOTEBOOK.md is deleted

#### Checklist
- [ ] Read NOTEBOOK.md
- [ ] Append content to IDEALOG.md Thread
- [ ] Delete NOTEBOOK.md

#### Verify
```bash
test ! -f Research/Active/research_environment_optimization/NOTEBOOK.md && echo "DELETED" || echo "STILL EXISTS"
grep "tmux Pane Ratio" Research/Active/research_environment_optimization/IDEALOG.md && echo "MERGED" || echo "MISSING"
```

#### Exit Criteria
- [ ] NOTEBOOK.md deleted
- [ ] All technical content present in IDEALOG.md

#### Risk
None — straightforward content merge.

---

### Step 1.2: Update `/reason` skill
**Model**: sonnet

#### Read First
- `.claude/skills/reason/SKILL.md` — 10 NOTEBOOK references

#### Why
`/reason` currently routes technical findings to NOTEBOOK.md. Must route everything to IDEALOG.md instead.

#### Implementation Spec
**Files to modify:** `.claude/skills/reason/SKILL.md`

Changes:
1. Remove NOTEBOOK.md location section (lines ~34-36)
2. Update routing table (Step 5): remove NOTEBOOK.md row, all content goes to IDEALOG.md
3. Remove "NOTEBOOK.md is scratch paper..." paragraph
4. Update Step 8 (checkpoint): `/quicksave` dumps to IDEALOG only
5. Update Rules: remove "NOTEBOOK.md is owned by /reason" rule, update allowed tools list
6. Keep WHITEBOARD.md references — those are unchanged

#### Checklist
- [ ] Remove NOTEBOOK.md location section
- [ ] Update routing table — all to IDEALOG.md
- [ ] Remove NOTEBOOK.md scratch paper paragraph
- [ ] Update checkpoint description
- [ ] Update Rules section (allowed tools, ownership rule)
- [ ] Verify no NOTEBOOK references remain

#### Verify
```bash
grep -c "NOTEBOOK" .claude/skills/reason/SKILL.md  # should be 0
```

#### Exit Criteria
- [ ] Zero NOTEBOOK references in reason/SKILL.md

#### Risk
Must not accidentally remove WHITEBOARD.md references — those stay.

---

### Step 1.3: Update `/quicksave` skill
**Model**: sonnet

#### Read First
- `.claude/skills/quicksave/SKILL.md` — 7 NOTEBOOK references

#### Why
`/quicksave` currently dumps to both IDEALOG.md and NOTEBOOK.md. Now dumps to IDEALOG.md only.

#### Implementation Spec
**Files to modify:** `.claude/skills/quicksave/SKILL.md`

Changes:
1. Update description in frontmatter: remove "+ NOTEBOOK.md"
2. Remove Step 2 (Dump Detail → NOTEBOOK.md) entirely
3. Update Step 3 report format: remove NOTEBOOK.md line
4. Update Rules: remove "Create NOTEBOOK.md if it doesn't exist"

#### Checklist
- [ ] Update frontmatter description
- [ ] Remove Step 2 (NOTEBOOK dump)
- [ ] Update report format
- [ ] Update Rules
- [ ] Verify no NOTEBOOK references remain

#### Verify
```bash
grep -c "NOTEBOOK" .claude/skills/quicksave/SKILL.md  # should be 0
```

#### Exit Criteria
- [ ] Zero NOTEBOOK references in quicksave/SKILL.md

#### Risk
None.

---

### Step 1.4: Update `/blueprint` skill
**Model**: sonnet

#### Read First
- `.claude/skills/blueprint/SKILL.md` — 3 NOTEBOOK references

#### Why
`/blueprint` currently reads NOTEBOOK.md as primary implementation detail source. Now reads IDEALOG.md for both strategic decisions and technical detail.

#### Implementation Spec
**Files to modify:** `.claude/skills/blueprint/SKILL.md`

Changes:
1. Remove Step 1b (NOTEBOOK.md input section) entirely
2. Update Step 1a (IDEALOG.md): note it now contains both strategic decisions AND raw technical findings
3. Renumber 1c→1b, 1d→1c, 1e→1d
4. Update precondition: remove NOTEBOOK.md fallback check

#### Checklist
- [ ] Remove Step 1b (NOTEBOOK.md)
- [ ] Update Step 1a to note IDEALOG has technical detail too
- [ ] Renumber remaining steps
- [ ] Update precondition
- [ ] Verify no NOTEBOOK references remain

#### Verify
```bash
grep -c "NOTEBOOK" .claude/skills/blueprint/SKILL.md  # should be 0
```

#### Exit Criteria
- [ ] Zero NOTEBOOK references in blueprint/SKILL.md

#### Risk
None.

---

### Step 1.5: Update `/save-session` skill
**Model**: sonnet

#### Read First
- `.claude/skills/save-session/SKILL.md` — 8 NOTEBOOK references

#### Why
`/save-session` has Job 6 (graduate NOTEBOOK findings to KNOWLEDGE). This job is no longer needed — IDEALOG.md content is graduated via Job 3 (cross-reference). Job count drops from 6 to 5.

#### Implementation Spec
**Files to modify:** `.claude/skills/save-session/SKILL.md`

Changes:
1. Update frontmatter description: "6 jobs" → "5 jobs", remove NOTEBOOK mention
2. Update body text: "six editorial jobs" → "five editorial jobs"
3. Remove NOTEBOOK.md from Project Structure Reference
4. Remove Job 6 entirely
5. Update Step Final report: remove NOTEBOOK.md line
6. Update Rules: remove "NOTEBOOK.md: graduate, never delete" rule

#### Checklist
- [ ] Update frontmatter (6→5, remove NOTEBOOK)
- [ ] Update body text job count
- [ ] Remove NOTEBOOK from structure reference
- [ ] Remove Job 6
- [ ] Update report format
- [ ] Update Rules
- [ ] Verify no NOTEBOOK references remain

#### Verify
```bash
grep -c "NOTEBOOK" .claude/skills/save-session/SKILL.md  # should be 0
```

#### Exit Criteria
- [ ] Zero NOTEBOOK references in save-session/SKILL.md
- [ ] Job count is 5

#### Risk
Must verify Job numbering is still sequential (Jobs 1-5).

---

### Step 1.6: Update `/reason-end` skill
**Model**: sonnet

#### Read First
- `.claude/skills/reason-end/SKILL.md` — 15 NOTEBOOK references

#### Why
`/reason-end` currently wipes NOTEBOOK.md as a separate step. That step is removed entirely. The skill now only wipes WHITEBOARD.md and kills panes.

#### Implementation Spec
**Files to modify:** `.claude/skills/reason-end/SKILL.md`

Changes:
1. Update frontmatter description: remove NOTEBOOK mentions
2. Remove Step 3 (Wipe NOTEBOOK.md) entirely
3. Renumber: old Step 4→3, old Step 5→4, old Step 6→5
4. Update Step 2 (/save-session): remove mention of NOTEBOOK graduation
5. Update report format: remove NOTEBOOK.md line
6. Update Rules: remove "Only wipe the active question's NOTEBOOK.md"

#### Checklist
- [ ] Update frontmatter
- [ ] Remove NOTEBOOK wipe step
- [ ] Renumber remaining steps
- [ ] Update /save-session description
- [ ] Update report format
- [ ] Update Rules
- [ ] Verify no NOTEBOOK references remain

#### Verify
```bash
grep -c "NOTEBOOK" .claude/skills/reason-end/SKILL.md  # should be 0
```

#### Exit Criteria
- [ ] Zero NOTEBOOK references in reason-end/SKILL.md

#### Risk
None.

---

### Step 1.7: Update `/strategic-compact` skill
**Model**: sonnet

#### Read First
- `.claude/skills/strategic-compact/SKILL.md` — 2 NOTEBOOK references

#### Why
NOTEBOOK.md is listed in the "what survives compaction" table and per-question save checklist. Remove both entries.

#### Implementation Spec
**Files to modify:** `.claude/skills/strategic-compact/SKILL.md`

Changes:
1. Remove NOTEBOOK.md row from "Survives Compaction" table
2. Remove NOTEBOOK.md row from "Per-question documents" table

#### Checklist
- [ ] Remove from survival table
- [ ] Remove from per-question table
- [ ] Verify no NOTEBOOK references remain

#### Verify
```bash
grep -c "NOTEBOOK" .claude/skills/strategic-compact/SKILL.md  # should be 0
```

#### Exit Criteria
- [ ] Zero NOTEBOOK references

#### Risk
None.

---

### Step 1.8: Update CLAUDE.md
**Model**: sonnet

#### Read First
- `CLAUDE.md:124-137` — Three-Document Architecture section

#### Why
CLAUDE.md is always loaded. It currently lists NOTEBOOK.md in the architecture table and routing rules. Must be removed so future sessions don't create NOTEBOOK.md files.

#### Implementation Spec
**Files to modify:** `CLAUDE.md`

Changes:
1. Remove NOTEBOOK.md row from the document table (line ~129)
2. Remove "Raw technical detail during reasoning → NOTEBOOK.md" routing rule (line ~135)
3. Update `/reason` description in skill table if it mentions NOTEBOOK
4. Update `/save-session` description: "6 jobs" → "5 jobs", remove NOTEBOOK mention
5. Update `/reason-end` description: remove NOTEBOOK mention

#### Checklist
- [ ] Remove NOTEBOOK from architecture table
- [ ] Remove NOTEBOOK routing rule
- [ ] Update skill table descriptions
- [ ] Verify no NOTEBOOK references remain

#### Verify
```bash
grep -c "NOTEBOOK" CLAUDE.md  # should be 0
```

#### Exit Criteria
- [ ] Zero NOTEBOOK references in CLAUDE.md

#### Risk
None.

---

### Step 1.9: Update .gitignore
**Model**: sonnet

#### Implementation Spec
**Files to modify:** `.gitignore`

Remove the `**/NOTEBOOK.md` line. Keep `WHITEBOARD.md`.

#### Checklist
- [ ] Remove `**/NOTEBOOK.md` line

#### Verify
```bash
grep "NOTEBOOK" .gitignore  # should return nothing
```

#### Exit Criteria
- [ ] .gitignore has no NOTEBOOK reference

#### Risk
None.

---

### Step 1.10: Update KNOWLEDGE.md (research_environment_optimization)
**Model**: sonnet

#### Read First
- `Research/Active/research_environment_optimization/KNOWLEDGE.md` — 8 NOTEBOOK references

#### Why
KNOWLEDGE.md has the NOTEBOOK.md design section and references throughout. Update to reflect the merge — NOTEBOOK.md design becomes historical note, active references point to IDEALOG.md.

#### Implementation Spec
**Files to modify:** `Research/Active/research_environment_optimization/KNOWLEDGE.md`

Changes:
1. Update NOTEBOOK.md Design section: add note that NOTEBOOK was merged into IDEALOG (historical)
2. Update document interaction diagram: remove NOTEBOOK.md box
3. Update folder structure diagram: remove NOTEBOOK.md
4. Update any routing rules that mention NOTEBOOK

#### Checklist
- [ ] Mark NOTEBOOK design section as historical
- [ ] Update diagrams
- [ ] Update routing references
- [ ] Verify references are historical-only (not active instructions)

#### Exit Criteria
- [ ] No active instructions reference NOTEBOOK.md as a current document

#### Risk
KNOWLEDGE.md is large (1000+ lines). Use targeted edits, not full rewrite.

---

### Phase 1 Verification
```bash
# Zero NOTEBOOK references in all skill files
for f in .claude/skills/*/SKILL.md; do echo "$f: $(grep -c NOTEBOOK "$f")"; done

# Zero in CLAUDE.md
grep -c NOTEBOOK CLAUDE.md

# Zero in .gitignore
grep NOTEBOOK .gitignore

# NOTEBOOK.md file deleted
test ! -f Research/Active/research_environment_optimization/NOTEBOOK.md && echo "DELETED"

# Content preserved in IDEALOG.md
grep "tmux Pane Ratio" Research/Active/research_environment_optimization/IDEALOG.md && echo "CONTENT PRESERVED"
```

### Phase 1 Exit Criteria
- [ ] All 10 files updated
- [ ] Zero active NOTEBOOK.md references across all skill files + CLAUDE.md
- [ ] Existing NOTEBOOK.md content merged into IDEALOG.md
- [ ] .gitignore cleaned
- [ ] KNOWLEDGE.md references are historical only

### Phase 1 Cleanup
- Verify no stale NOTEBOOK.md files elsewhere: `find . -name "NOTEBOOK.md"`
- Check git diff for any missed references

**→ Commit point: "Merge NOTEBOOK.md into IDEALOG.md — single log file, remove NOTEBOOK from architecture"**

---

## Final Cleanup

- Verify all 18 skills load without errors
- Run `/research-status` to confirm no broken references

## Mutation Log
{To be filled during execution}
