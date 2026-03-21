# PLAN: Workflow Refinements — tmux window name, /reason speedup, skill catalog, maintenance section, background writes

Created: 2026-03-19
Engine(s): None (cross-cutting workflow)
Research question: [research_environment_optimization](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-03-19 entries (5 items)

## Objective
Five workflow improvements: (1) tmux window name as per-session question identifier, (2) parallelize /reason startup, (3) add full skill catalog to KNOWLEDGE.md with Maintenance section, (4) per-question maintenance instructions in KNOWLEDGE.md, (5) background agent writes for IDEALOG/WHITEBOARD during /reason.

## Success Criteria
- [ ] `/research-resume` sets tmux window name on question selection
- [ ] `/research-new` sets tmux window name on question creation
- [ ] `/reason` reads tmux window name for auto-resume (Step 0)
- [ ] `/reason` startup fires parallel tool calls (reads + tmux in one turn)
- [ ] KNOWLEDGE.md has full skill catalog (19 skills, purposes, line counts, categories)
- [ ] KNOWLEDGE.md has Maintenance section with question-specific verification rules
- [ ] `/reason` IDEALOG/WHITEBOARD writes use background agents

## Architecture Changes
- MOD: `.claude/skills/research-resume/SKILL.md` — add `tmux rename-window` after question selection
- MOD: `.claude/skills/research-new/SKILL.md` — add `tmux rename-window` after question creation
- MOD: `.claude/skills/reason/SKILL.md` — read tmux window name in Step 0, explicit parallel calls in Step 1, background agent writes in Step 5
- MOD: `.claude/skills/reason-end/SKILL.md` — reset tmux window name to "claude"
- MOD: `Research/Active/research_environment_optimization/KNOWLEDGE.md` — add skill catalog + Maintenance section

## Known Failures (from IDEALOG)
- Can't access Claude Code `/rename` session name programmatically — use tmux window name instead
- Skill instructions fade in long conversations above ~150 lines — keep changes minimal

---

## Phase 1: tmux Window Name Integration

**Goal**: `/research-resume` and `/research-new` set the tmux window name. `/reason` reads it. `/reason-end` resets it.
**Tier**: small

### Phase Context
tmux window name is per-window (no conflicts between concurrent sessions), survives compaction (external to Claude context). Format: capitalized readable title (e.g., "Research Environment Optimization"). Maps to folder via lowercase + spaces→underscores. Tested and confirmed working.

### Step 1.1: Update `/research-resume` — set tmux window name
**Model**: opus

#### Read First
- `.claude/skills/research-resume/SKILL.md:45-50` — after question selection, before Step 2

#### Why
When a question is selected, the tmux window should reflect it. This enables `/reason` to auto-detect the active question without conversation history.

#### Implementation Spec
**Files to modify:** `.claude/skills/research-resume/SKILL.md`

`/research-resume` has TWO code paths for question selection — both need the rename:
1. **Argument provided** (goes directly to Step 2): add rename immediately after resolving the folder
2. **No argument** (user picks from list, then proceeds to Step 2): add rename after user selection, before Step 2

Add to BOTH paths:
```markdown
If running in tmux, rename the window to the question's readable title:
```bash
tmux rename-window "{Capitalized Question Title}" 2>/dev/null
```
```

#### Checklist
- [ ] Add tmux rename-window in the argument-provided path (before Step 2)
- [ ] Add tmux rename-window in the user-selection path (after user picks, before Step 2)
- [ ] Use capitalized readable title format

#### Verify
```bash
grep "rename-window" .claude/skills/research-resume/SKILL.md
```

#### Exit Criteria
- [ ] `/research-resume` sets tmux window name

#### Risk
None — `2>/dev/null` silently fails if not in tmux.

---

### Step 1.2: Update `/research-new` — set tmux window name
**Model**: opus

#### Read First
- `.claude/skills/research-new/SKILL.md:55-70` — after folder creation

#### Why
When creating a new question, the tmux window should immediately reflect it.

#### Implementation Spec
**Files to modify:** `.claude/skills/research-new/SKILL.md`

After Step 4 (folder creation), add same tmux rename-window instruction.

#### Checklist
- [ ] Add tmux rename-window after folder creation

#### Verify
```bash
grep "rename-window" .claude/skills/research-new/SKILL.md
```

#### Exit Criteria
- [ ] `/research-new` sets tmux window name

#### Risk
None.

---

### Step 1.3: Update `/reason` Step 0 — read tmux window name
**Model**: opus

#### Read First
- `.claude/skills/reason/SKILL.md:15-22` — current Step 0

#### Why
The tmux window name is the PRIMARY fallback for question detection — it survives compaction (unlike conversation history) and is per-window (unlike a file). It should be checked BEFORE falling back to asking the user, and it works regardless of whether `/research-resume` was recent.

#### Implementation Spec
**Files to modify:** `.claude/skills/reason/SKILL.md`

Rewrite Step 0 with tmux window name as the second check (after explicit argument):
```markdown
**If argument provided**: Use it as the reasoning topic.

**If no argument**: Try these in order:
1. Check conversation history for a recent `/research-resume`. If found, use that question.
2. Check tmux window name: `tmux display-message -p '#{window_name}' 2>/dev/null`. If the result is not "claude" or "bash" (default names), convert to folder name (lowercase, spaces→underscores) and check if `Research/Active/{folder}/IDEALOG.md` exists. If yes, auto-resume that question.
3. If neither works, ask the user which question to reason about.
```

This makes tmux window name the durable fallback that works after compaction.

#### Checklist
- [ ] Add tmux window name detection to Step 0
- [ ] Add folder name conversion logic
- [ ] Add fallback to asking user

#### Verify
```bash
grep "window_name" .claude/skills/reason/SKILL.md
```

#### Exit Criteria
- [ ] `/reason` reads tmux window name for auto-resume

#### Risk
Window name might be something unexpected (e.g., "vim", "htop"). Mitigated by checking if the converted name maps to an actual question folder.

---

### Step 1.4: Update `/reason-end` — reset tmux window name
**Model**: opus

#### Read First
- `.claude/skills/reason-end/SKILL.md:32-43` — pane kill step

#### Why
After teardown, the window name should reset so `/reason` doesn't auto-resume a closed session. Should restore the original name instead of hardcoding "claude" (user may have custom window names).

#### Implementation Spec
**Files to modify:** `.claude/skills/reason-end/SKILL.md`

Add to the `/reason-end` flow:
1. At the START (before /save-session), the original window name should have been saved by `/reason` Step 1b (add a note there too)
2. At the END (after pane kill), restore:
```bash
tmux rename-window "${ORIGINAL_WINDOW_NAME:-claude}" 2>/dev/null
```

Also update `/reason` Step 1b to save the original name before renaming:
```bash
# Save original window name for /reason-end to restore
ORIG_WIN=$(tmux display-message -p '#{window_name}' 2>/dev/null)
```
Note: since `/reason-end` runs in a different context, the original name can't be passed via variable. Instead, `/reason` should note the original name in the conversation context, and `/reason-end` reads it. Alternatively, just reset to "claude" — simpler, and the user can re-rename if needed.

**Decision**: Reset to "claude" for simplicity. If the user had a custom name, they can re-set it.

```bash
tmux rename-window "claude" 2>/dev/null
```

#### Checklist
- [ ] Add tmux window name reset to "claude"

#### Verify
```bash
grep "rename-window" .claude/skills/reason-end/SKILL.md
```

#### Exit Criteria
- [ ] `/reason-end` resets window name to "claude"

#### Risk
None.

---

### Phase 1 Verification
```bash
grep -l "rename-window" .claude/skills/research-resume/SKILL.md .claude/skills/research-new/SKILL.md .claude/skills/reason-end/SKILL.md
grep "window_name" .claude/skills/reason/SKILL.md
```

### Phase 1 Exit Criteria
- [ ] All 4 skills updated
- [ ] tmux window name set/read/reset cycle works

### Phase 1 Cleanup
None.

**→ Commit point: "tmux window name as per-session question identifier"**

---

## Phase 2: Parallelize /reason Startup

**Goal**: `/reason` Step 1 fires all reads + tmux setup as parallel tool calls in a single turn.
**Tier**: trivial

### Step 2.1: Make parallel execution explicit in `/reason` Step 1
**Model**: opus

#### Read First
- `.claude/skills/reason/SKILL.md:25-64` — current Step 1

#### Why
Step 1a says "read in parallel" but doesn't instruct Claude to make parallel tool calls in a single message. Step 1b (tmux) is independent and can run simultaneously. Making this explicit speeds up initialization.

**Note (from audit M2)**: Step 0 (question detection) must complete FIRST — it determines the question folder name needed for the file paths. Parallelization is Step 0 (sequential) → Step 1 (parallel).

#### Implementation Spec
**Files to modify:** `.claude/skills/reason/SKILL.md`

Replace the current Step 1 intro with:
```markdown
## Step 1: Load Context + Set Up Workspace

**After Step 0 determines the question**, execute ALL of the following in a SINGLE message as parallel tool calls:

1. `Read` — Research/Active/{question}/IDEALOG.md
2. `Read` — Research/Active/{question}/KNOWLEDGE.md
3. `Read` — Research/Active/{question}/README.md
4. `Bash` — tmux pane setup + window rename (if in tmux and panes don't exist)

Do NOT read files sequentially. All 4 calls go out in one turn. Step 0 must complete first (it determines {question}).
```

#### Checklist
- [ ] Add explicit "SINGLE message as parallel tool calls" instruction
- [ ] List the 4 parallel calls

#### Verify
```bash
grep -c "parallel tool calls\|SINGLE message" .claude/skills/reason/SKILL.md
```

#### Exit Criteria
- [ ] Parallelism is explicitly instructed

#### Risk
None — Claude already supports parallel tool calls.

---

### Phase 2 Verification
```bash
grep "SINGLE message" .claude/skills/reason/SKILL.md
```

### Phase 2 Exit Criteria
- [ ] Parallel execution explicitly instructed

### Phase 2 Cleanup
None.

**→ Commit point: "Parallelize /reason startup — 4 tool calls in single turn"**

---

## Phase 3: Skill Catalog + Maintenance Section in KNOWLEDGE.md

**Goal**: Add a comprehensive skill reference table and per-question maintenance instructions to KNOWLEDGE.md.
**Tier**: small

### Step 3.1: Add full skill catalog and Maintenance section to KNOWLEDGE.md
**Model**: opus

#### Read First
- `Research/Active/research_environment_optimization/KNOWLEDGE.md:135-162` — current Skill Pipeline section
- `.claude/skills/*/SKILL.md` — get current line counts for all 19 skills

#### Why
The current skill table lists names by category but not purposes or line counts. A full reference means anyone reading KNOWLEDGE.md can understand what each skill does without opening 19 files. The Maintenance section tells `/save-session` Job 2 what to verify specifically for this question.

#### Implementation Spec
**Files to modify:** `Research/Active/research_environment_optimization/KNOWLEDGE.md`

1. Replace the brief skill category table in Skill Pipeline > Design with a full table:

```markdown
| Skill | Purpose | Lines | Category |
|-------|---------|-------|----------|
| `/research-new` | Scaffold new question (README + KNOWLEDGE + IDEALOG + WHITEBOARD) | {N} | Research |
| `/research-resume` | Resume question, load context, present briefing | {N} | Research |
| ... | ... | ... | ... |
```

Get actual line counts by reading each skill file.

2. Add a `## Maintenance` section at the end (before Connections):

```markdown
## Maintenance

When `/save-session` Job 2 edits this file, also verify:
- Skill catalog matches actual `.claude/skills/` directory (count, names, line counts)
- Skill line counts are current (re-check with `wc -l`)
- Future Work items in README.md are reflected if relevant
- Per-topic subsections follow Findings/Design/Reference/Decisions order
```

#### Checklist
- [ ] Read all 19 skill files for line counts
- [ ] Write full skill catalog table
- [ ] Add Maintenance section
- [ ] Verify table has all 19 skills

#### Verify
```bash
grep -c "^|" Research/Active/research_environment_optimization/KNOWLEDGE.md | head -1
grep "## Maintenance" Research/Active/research_environment_optimization/KNOWLEDGE.md
```

#### Exit Criteria
- [ ] Full 19-skill catalog in KNOWLEDGE.md
- [ ] Maintenance section present

#### Risk
KNOWLEDGE.md is 319 lines. Adding ~25 lines for the full table + ~10 for Maintenance = ~354 lines. Acceptable.

---

### Phase 3 Verification
```bash
grep "## Maintenance" Research/Active/research_environment_optimization/KNOWLEDGE.md && echo "MAINTENANCE SECTION EXISTS"
grep -c "/reason\|/blueprint\|/save-session\|/verify\|/audit" Research/Active/research_environment_optimization/KNOWLEDGE.md
```

### Phase 3 Exit Criteria
- [ ] Full skill catalog with purposes and line counts
- [ ] Maintenance section with verification rules

### Phase 3 Cleanup
None.

**→ Commit point: "Add skill catalog + Maintenance section to KNOWLEDGE.md"**

---

## Phase 4: Background Agent Writes in /reason

**Goal**: IDEALOG and WHITEBOARD writes during `/reason` use background agents instead of blocking the main thread.
**Tier**: medium

### Phase Context
Currently, `/reason` pauses the conversation to write Edit calls to IDEALOG.md and WHITEBOARD.md. Background agents can do these writes without interrupting flow. IDEALOG is append-only during `/reason` (no conflict), WHITEBOARD is overwrite-only (no conflict). The Agent tool supports `run_in_background: true`.

### Step 4.1: Update `/reason` Step 5 and Step 10 for background writes
**Model**: opus

#### Read First
- `.claude/skills/reason/SKILL.md:93-117` — Step 5 (IDEALOG writes)
- `.claude/skills/reason/SKILL.md:228-240` — Step 10 (WHITEBOARD writes)

#### Why
Blocking the conversation for file writes disrupts the flow of interactive reasoning. Background agents eliminate this interruption.

#### Implementation Spec
**Files to modify:** `.claude/skills/reason/SKILL.md`

In Step 5, add:
```markdown
**Use background agents for writes.** When writing to IDEALOG.md, spawn a background Agent with the content to append. Do NOT block the conversation waiting for the write to complete. Continue the discussion immediately.

Example:
- Agent prompt: "Append this to Research/Active/{question}/IDEALOG.md under the Thread section: ### {date}: {title}\n{content}"
- run_in_background: true
```

In Step 10, add similar instruction for WHITEBOARD.md writes.

Also update `/quicksave` to run as a background agent when invoked from Step 8.

#### Checklist
- [ ] Add background agent instruction to Step 5 (IDEALOG writes)
- [ ] Add background agent instruction to Step 10 (WHITEBOARD writes)
- [ ] Update Step 8 to note /quicksave can run in background
- [ ] Verify instructions are clear about run_in_background: true

#### Verify
```bash
grep -c "background\|run_in_background" .claude/skills/reason/SKILL.md
```

#### Exit Criteria
- [ ] IDEALOG writes use background agents
- [ ] WHITEBOARD writes use background agents
- [ ] /quicksave can run in background

#### Risk
Race condition if two background agents write to IDEALOG.md simultaneously. Mitigated by: writes are append-only (no conflicts), and transition-based triggers mean at most one write per transition (never concurrent).

---

### Phase 4 Verification
```bash
grep "background" .claude/skills/reason/SKILL.md | head -5
```

### Phase 4 Exit Criteria
- [ ] Background agent write instructions in `/reason`

### Phase 4 Cleanup
None.

**→ Commit point: "Background agent writes for IDEALOG/WHITEBOARD during /reason"**

---

## Final Cleanup
- Verify all skill files updated correctly
- Verify KNOWLEDGE.md Maintenance section works with `/save-session`

## Mutation Log

**MUTATED 2026-03-19**: Step 1.1 MODIFIED — audit H3: `/research-resume` has two code paths for question selection, tmux rename must go in both
**MUTATED 2026-03-19**: Step 1.3 MODIFIED — audit H4: tmux window name should be primary fallback (not just branch 3), works regardless of conversation history
**MUTATED 2026-03-19**: Step 1.4 MODIFIED — audit M5: considered save/restore original name, decided reset to "claude" for simplicity
**MUTATED 2026-03-19**: Step 2.1 MODIFIED — audit M2: clarified Step 0 must complete before parallel Step 1 (question name needed for file paths)
