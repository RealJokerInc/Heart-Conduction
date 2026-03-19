# PLAN: Move WHITEBOARD.md to per-question folder

Created: 2026-03-18
Engine(s): None (cross-cutting workflow)
Research question: [research_environment_optimization](README.md)
Source: [IDEALOG.md](IDEALOG.md) — user discovered conflict when multitasking across research questions

## Objective
Move WHITEBOARD.md from project root (shared, conflicts between concurrent sessions) to per-question folder (`Research/Active/{question}/WHITEBOARD.md`). Each `/reason` session gets its own whiteboard scoped to its research question.

## Success Criteria
- [ ] WHITEBOARD.md path is per-question in all skill files and CLAUDE.md
- [ ] `.gitignore` uses `**/WHITEBOARD.md` (catches all question folders)
- [ ] `/reason` tmux pane points to question-scoped WHITEBOARD.md
- [ ] `/reason-end` wipes question-scoped WHITEBOARD.md
- [ ] No project-root WHITEBOARD.md references remain in active instructions

## Architecture Changes
- MOD: `.claude/skills/reason/SKILL.md` — WHITEBOARD path in Step 1b (tmux) and Step 10 (visualization)
- MOD: `.claude/skills/reason-end/SKILL.md` — WHITEBOARD wipe path
- MOD: `.claude/skills/strategic-compact/SKILL.md` — WHITEBOARD location in tables
- MOD: `CLAUDE.md` — WHITEBOARD description in Document Architecture table
- MOD: `.gitignore` — `WHITEBOARD.md` → `**/WHITEBOARD.md`
- MOD: `Research/Active/research_environment_optimization/KNOWLEDGE.md` — Workspace Integration section
- MOD: `.claude/skills/research-new/SKILL.md` — add WHITEBOARD.md to folder creation template
- DEL: `WHITEBOARD.md` at project root (if exists)

## Known Failures (from IDEALOG)
- None specific to this change

---

## Phase 1: Update all references and paths

**Goal**: Change every WHITEBOARD.md reference from project root to `Research/Active/{question}/WHITEBOARD.md`. Ship all changes together.
**Tier**: small

### Phase Context
6 files contain WHITEBOARD.md references. All are markdown text edits. The key change is replacing project-root paths with question-scoped paths. In skill files, `{question}` is a placeholder that gets replaced with the active question name at runtime.

### Step 1.1: Update `/reason` skill
**Model**: sonnet

#### Read First
- `.claude/skills/reason/SKILL.md:59,228-254` — tmux setup and whiteboard rules

#### Why
`/reason` writes to WHITEBOARD.md and sets up the tmux pane to watch it. Both paths must point to the question folder.

#### Implementation Spec
**Files to modify:** `.claude/skills/reason/SKILL.md`

1. Line 59 (tmux pane 2 setup): change `WHITEBOARD.md` to `Research/Active/{question}/WHITEBOARD.md`
2. Line 230 (Step 10): change "write them to `WHITEBOARD.md` in the project root" to "write them to `Research/Active/{question}/WHITEBOARD.md`"
3. Lines 244, 254: update any rules referencing WHITEBOARD.md location

#### Checklist
- [ ] Update tmux send-keys path (Step 1b)
- [ ] Update Step 10 description
- [ ] Update Rules section references
- [ ] Verify no project-root WHITEBOARD references remain

#### Verify
```bash
grep "WHITEBOARD" .claude/skills/reason/SKILL.md | grep -v "Research/Active/{question}" | grep -v "WHITEBOARD.md writes"
```

#### Exit Criteria
- [ ] All WHITEBOARD paths are question-scoped

#### Risk
The tmux `send-keys` command is already long. Adding the full path makes it longer but still functional.

---

### Step 1.2: Update `/reason-end` skill
**Model**: sonnet

#### Read First
- `.claude/skills/reason-end/SKILL.md:24-28` — wipe step

#### Why
`/reason-end` must wipe the correct WHITEBOARD.md — the one in the active question's folder, not project root.

#### Implementation Spec
**Files to modify:** `.claude/skills/reason-end/SKILL.md`

1. Line 3 (description): already generic, OK
2. Line 27: change `rm -f WHITEBOARD.md` to `rm -f "Research/Active/{question}/WHITEBOARD.md"` — the `{question}` placeholder works the same way as in `/reason`: Claude reads the skill instruction and substitutes the active question name when composing the bash command. `/reason-end` already determines the active question for `/save-session` (Step 1) — use the same question for the WHITEBOARD wipe.

#### Checklist
- [ ] Update wipe command path
- [ ] Ensure question detection works for the wipe

#### Verify
```bash
grep "rm.*WHITEBOARD" .claude/skills/reason-end/SKILL.md
```

#### Exit Criteria
- [ ] Wipe targets question-scoped path

#### Risk
None.

---

### Step 1.3: Update `/strategic-compact`, CLAUDE.md, .gitignore, KNOWLEDGE.md
**Model**: sonnet

#### Read First
- `.claude/skills/strategic-compact/SKILL.md:53,65` — WHITEBOARD in tables
- `CLAUDE.md:129,135` — Document Architecture table
- `.gitignore:1` — current entry
- `Research/Active/research_environment_optimization/KNOWLEDGE.md:279-289` — Workspace Design section

#### Why
All remaining references to WHITEBOARD.md location must be updated for consistency.

#### Implementation Spec
**Files to modify:**
1. `.claude/skills/strategic-compact/SKILL.md` — change "Project root" to "`Research/Active/{question}/`" in the per-question documents table
2. `.claude/skills/research-new/SKILL.md` — add `WHITEBOARD.md` to the folder structure template in Step 4 (so new questions get it automatically)
3. `CLAUDE.md` — update WHITEBOARD description: "Per-question, in `Research/Active/{question}/`"
4. `.gitignore` — change `WHITEBOARD.md` to `**/WHITEBOARD.md`
5. `Research/Active/research_environment_optimization/KNOWLEDGE.md` — update Workspace Design section

#### Checklist
- [ ] Update strategic-compact tables
- [ ] Update CLAUDE.md Document Architecture table
- [ ] Update .gitignore
- [ ] Update KNOWLEDGE.md Workspace Integration section
- [ ] Delete project-root WHITEBOARD.md if it exists

#### Verify
```bash
grep -r "WHITEBOARD" CLAUDE.md .claude/skills/strategic-compact/SKILL.md .gitignore | grep -v "Research/Active"
```

#### Exit Criteria
- [ ] No project-root WHITEBOARD references in any active file
- [ ] .gitignore catches all question folders

#### Risk
None.

---

### Step 1.4: Create empty WHITEBOARD.md in all 6 active question folders
**Model**: sonnet

#### Read First
- `Research/Active/` — list all active question folders

#### Why
The tmux glow watcher will error if WHITEBOARD.md doesn't exist when `/reason` starts. Each question needs an empty WHITEBOARD.md ready to be written to.

#### Implementation Spec
**Files to create:**
- `Research/Active/boundary_conduction_speedup/WHITEBOARD.md`
- `Research/Active/ionic_model_optimization/WHITEBOARD.md`
- `Research/Active/engine_consolidation/WHITEBOARD.md`
- `Research/Active/geometry_induced_pacemaking/WHITEBOARD.md`
- `Research/Active/mature_hipsc_cm_models/WHITEBOARD.md`
- `Research/Active/research_environment_optimization/WHITEBOARD.md`
- `Research/Active/surrogate_pipeline/WHITEBOARD.md`

Each file contains just: `# Whiteboard\n` (minimal content so glow renders without error)

Also delete project-root `WHITEBOARD.md` if it exists.

#### Checklist
- [ ] Create WHITEBOARD.md in all 7 active question folders
- [ ] Delete project-root WHITEBOARD.md
- [ ] Verify all 7 exist

#### Verify
```bash
ls Research/Active/*/WHITEBOARD.md | wc -l  # should be 7
test ! -f WHITEBOARD.md && echo "ROOT CLEAN" || echo "ROOT EXISTS"
```

#### Exit Criteria
- [ ] 6 WHITEBOARD.md files exist in question folders
- [ ] No project-root WHITEBOARD.md

#### Risk
None.

---

### Phase 1 Verification
```bash
echo "=== Skills ===" && grep -rn "WHITEBOARD" .claude/skills/*/SKILL.md | grep -v "Research/Active/{question}" | grep -v "WHITEBOARD.md writes\|WHITEBOARD.md)" && echo "=== CLAUDE.md ===" && grep "WHITEBOARD" CLAUDE.md && echo "=== .gitignore ===" && cat .gitignore | grep WHITEBOARD && echo "=== Project root ===" && test -f WHITEBOARD.md && echo "EXISTS (should be deleted)" || echo "CLEAN"
```

### Phase 1 Exit Criteria
- [ ] All WHITEBOARD paths are per-question in skills and CLAUDE.md
- [ ] .gitignore uses `**/WHITEBOARD.md`
- [ ] No project-root WHITEBOARD.md exists

### Phase 1 Cleanup
None needed — text edits only.

**→ Commit point: "Move WHITEBOARD.md to per-question folder — no conflicts between concurrent sessions"**

---

## Mutation Log
{To be filled during execution}
