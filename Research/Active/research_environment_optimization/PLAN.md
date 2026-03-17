# PLAN: Research Environment Optimization — Implementation

Created: 2026-03-17
Engine(s): All (cross-cutting)
Research question: [research_environment_optimization](README.md)
Source: [IDEALOG.md](IDEALOG.md) — full session designing three-document architecture + skill pipeline

## Objective
Implement the three-document architecture (KNOWLEDGE + IDEALOG + PLAN), extend 4 existing research skills, create 7 new skills, and add 1 hook. This restructures our entire research workflow from a single-document system (KNOWLEDGE.md does everything) to a separated system where KNOWLEDGE is high-res reference, IDEALOG is low-res thinking trail, and PLAN is machine-targeted execution steps.

## Success Criteria
- [ ] All 6 active research questions have IDEALOG.md files populated from chat log backtrace
- [ ] All 6 KNOWLEDGE.md files contain only reference material (no process/narrative content)
- [ ] CLAUDE.md reflects the three-document architecture and lists all new skills
- [ ] All 4 extended research skills work with the new document structure
- [ ] All 7 new skills are functional
- [ ] PreCompact hook fires and logs correctly
- [ ] MASTER_KNOWLEDGE_INDEX.md exists with cross-question summaries and research statement
- [ ] No regressions — existing `/research-*` workflows still work

## Architecture Changes
- NEW: `MASTER_KNOWLEDGE_INDEX.md` — cross-cutting knowledge hub (research statement + per-question summaries + cross-references)
- NEW: `Research/Active/{question}/IDEALOG.md` — 6 files (one per active question)
- MOD: `Research/Active/{question}/KNOWLEDGE.md` — 6 files (migrate process content out)
- MOD: `CLAUDE.md` — rewrite architecture section, add skill table
- MOD: `.claude/skills/research-new/SKILL.md` — add IDEALOG.md creation
- MOD: `.claude/skills/research-resume/SKILL.md` — read IDEALOG, updated briefing
- MOD: `.claude/skills/research-update/SKILL.md` — add type routing
- MOD: `.claude/skills/research-complete/SKILL.md` — archive IDEALOG
- NEW: `.claude/skills/reason/SKILL.md` — interactive reasoning buddy
- NEW: `.claude/skills/blueprint/SKILL.md` — PLAN.md generator
- NEW: `.claude/skills/save-session/SKILL.md` — cleanup agent
- NEW: `.claude/skills/audit/SKILL.md` — adversarial review
- NEW: `.claude/skills/verify/SKILL.md` — test runner
- NEW: `.claude/skills/build-fix/SKILL.md` — error resolution
- NEW: `.claude/skills/strategic-compact/SKILL.md` — compaction guidance
- NEW: `scripts/extract_idealog.py` — chat log extraction utility
- NEW: `.claude/hooks/pre-compact.sh` — PreCompact hook script
- MOD: `.claude/settings.json` — add hook configuration (create if doesn't exist)

## Known Failures (from IDEALOG)
- Tried creating 6 skill files directly without analyzing existing skills first — reverted. Must understand current architecture before writing code.
- `/aside` skill investigated — BTW disappearing-from-transcript behavior is hardcoded in Claude Code UI, not extensible. Don't attempt to replicate.
- WORKLOG.md as separate document — dropped. Failed approaches and session log belong in IDEALOG. Three documents sufficient, not four.

---

## Phase 0: Foundation

**Goal**: Create the document infrastructure that all skills depend on. After this phase, every active research question has IDEALOG.md, every KNOWLEDGE.md is clean reference material, and CLAUDE.md reflects the new architecture.
**Tier**: large
**Estimated scope**: 6 active questions to process, ~16 chat logs to backtrace, CLAUDE.md rewrite

### Phase Context
This is the riskiest phase. We're modifying 6 existing KNOWLEDGE.md files and creating 6 new IDEALOG.md files. The chat logs are in `~/.claude/projects/-home-norepinephrine-Documents-Heart-Conduction/*.jsonl` (16 files, JSONL format with `type`, `role`, `timestamp` fields). Each line is a JSON object — user messages, assistant responses, file operations.

Key convention: KNOWLEDGE.md keeps ALL high-resolution reference material (analysis, comparisons, designs, templates, decisions with rationale). Only the narrative "thinking trail" moves to IDEALOG.md. When in doubt, keep it in KNOWLEDGE.md.

The 6 active questions and their KNOWLEDGE.md sizes:
- `boundary_conduction_speedup` — 4.7KB
- `ionic_model_optimization` — 7.6KB
- `engine_consolidation` — 9.0KB
- `geometry_induced_pacemaking` — 15.4KB
- `mature_hipsc_cm_models` — 4.1KB
- `research_environment_optimization` — 63.3KB (this question — largest, most mixed)

### Step 0.1: Create IDEALOG.md for each question (primary: from existing docs, secondary: chat log extraction)

**Model**: opus

#### Read First
- `Research/Active/{question}/README.md` — for each of 6 questions (status, completion criteria, next steps)
- `Research/Active/{question}/KNOWLEDGE.md` — for each (identify narrative/process content that belongs in IDEALOG)
- `MASTER.md` — current status of each question

#### Why
IDEALOG.md should capture the thinking trail — decisions, failed approaches, branching insights. The **primary source** is existing documents (README.md status, KNOWLEDGE.md process content, MASTER.md next steps) because these are already structured and readable. Chat log backtrace is **secondary** — the 16 JSONL files total 53.7MB with nested structures, too large to read directly. A Python script handles extraction offline.

#### Implementation Spec

**Primary approach (per question):**

1. Read README.md → extract Current Direction (from status + next step) and completion criteria state
2. Read KNOWLEDGE.md → identify narrative/process content (decisions that evolved, approaches discussed, "we then tried X"). These become Thread entries.
3. Read MASTER.md → extract the current one-liner status
4. Write IDEALOG.md with what we have from existing docs. For older questions with little process content, note "Pre-IDEALOG history — thinking trail started {date}."

**Secondary approach (best-effort chat log extraction):**

Write a Python script (`scripts/extract_idealog.py`) that:
1. Reads each `.jsonl` file one line at a time (no full-file loading)
2. Greps for the question folder name in each line
3. For matching lines, extracts the `message.content` text
4. Filters for decision/failure/insight patterns
5. Outputs a summary `.md` file per question to `/tmp/idealog_extracts/`
6. The implementing agent then reads these small summary files and merges relevant entries into the IDEALOGs

```python
#!/usr/bin/env python3
"""Extract thinking trail from Claude Code session logs."""
import json, sys, os, re
from pathlib import Path

SESSION_DIR = Path.home() / ".claude/projects/-home-norepinephrine-Documents-Heart-Conduction"
QUESTIONS = [
    "boundary_conduction_speedup", "ionic_model_optimization",
    "engine_consolidation", "geometry_induced_pacemaking",
    "mature_hipsc_cm_models", "research_environment_optimization"
]
OUTPUT_DIR = Path("/tmp/idealog_extracts")
OUTPUT_DIR.mkdir(exist_ok=True)

for question in QUESTIONS:
    entries = []
    for jsonl_file in sorted(SESSION_DIR.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if question not in line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") not in ("user", "assistant"):
                        continue
                    content = str(msg.get("message", {}).get("content", ""))
                    # Extract decisions, failures, insights
                    if any(kw in content.lower() for kw in
                           ["decided", "let's do", "go with", "failed because",
                            "didn't work", "tried", "realized", "oh wait",
                            "this means", "spawned"]):
                        timestamp = msg.get("timestamp", "unknown")
                        snippet = content[:500]  # truncate long messages
                        entries.append(f"### {timestamp}\n{snippet}\n")
                except json.JSONDecodeError:
                    continue

    output = OUTPUT_DIR / f"{question}.md"
    output.write_text(f"# Chat Log Extracts: {question}\n\n" +
                      (("\n".join(entries)) if entries else "No relevant entries found.\n"))
    print(f"{question}: {len(entries)} entries → {output}")
```

Run this script first, then read the small output files to supplement the IDEALOGs.

**Files to create:**
- `Research/Active/boundary_conduction_speedup/IDEALOG.md`
- `Research/Active/ionic_model_optimization/IDEALOG.md`
- `Research/Active/engine_consolidation/IDEALOG.md`
- `Research/Active/geometry_induced_pacemaking/IDEALOG.md`
- `Research/Active/mature_hipsc_cm_models/IDEALOG.md`
- `Research/Active/research_environment_optimization/IDEALOG.md`
- `scripts/extract_idealog.py` (utility script for chat log extraction)

**Template:**
```markdown
# {Question Title} — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
{What we're currently pursuing and why}

## Next Step
{Precise enough to resume with zero context}

## Thread

### {date}: {insight or idea title}
{What the idea is, why it matters, what it implies}

## Failed Approaches
- **{approach}** ({date}) — failed because: {exact error or reasoning}

## Session Log

### {date} Session
**Worked on**: {what}
**Accomplished**: {what, with evidence}
**Next**: {exact next step}
```

#### Test Spec
- Manual verification: each IDEALOG has Current Direction + Next Step populated
- Check: narrative content from KNOWLEDGE.md appears in IDEALOG Thread
- Check: Current Direction matches README.md status

#### Checklist
- [ ] Write `scripts/extract_idealog.py`
- [ ] Run the extraction script
- [ ] Read README.md + KNOWLEDGE.md for boundary_conduction_speedup → write IDEALOG
- [ ] Same for ionic_model_optimization
- [ ] Same for engine_consolidation
- [ ] Same for geometry_induced_pacemaking
- [ ] Same for mature_hipsc_cm_models
- [ ] Same for research_environment_optimization
- [ ] Merge chat log extracts (from /tmp/idealog_extracts/) into each IDEALOG
- [ ] Verify each IDEALOG has meaningful Current Direction + Next Step

#### Risk
Older questions may have sparse process content in KNOWLEDGE.md and sparse chat logs. Mitigation: for these, create IDEALOG with just Current Direction + Next Step from README.md and note "Pre-IDEALOG history — thinking trail started 2026-03-17." Future sessions will populate the trail going forward.

**→ Intermediate commit point: "Create 6 IDEALOG.md files + extraction script"**
(This gives a clean rollback point before Step 0.2 modifies KNOWLEDGE.md files)

---

### Step 0.2a: Audit and split 5 smaller KNOWLEDGE.md files

**Model**: sonnet

#### Read First
- `Research/Active/{question}/KNOWLEDGE.md` — for the 5 smaller questions (4-15KB each)
- `Research/Active/{question}/IDEALOG.md` — the newly created files from Step 0.1

#### Why
KNOWLEDGE.md files currently contain mixed content — reference material AND process/narrative content. After this step, KNOWLEDGE.md contains only high-res reference material. Narrative "how we got here" content moves to IDEALOG.md.

#### Implementation Spec

For each of the 5 smaller questions (boundary_conduction_speedup 4.7KB, ionic_model_optimization 7.6KB, engine_consolidation 9KB, geometry_induced_pacemaking 15.4KB, mature_hipsc_cm_models 4.1KB):

1. Read the current KNOWLEDGE.md
2. Classify each section/paragraph as:
   - **Reference** → stays (facts, analysis, designs, comparisons, decisions with rationale)
   - **Narrative** → moves to IDEALOG.md (process descriptions, "we then tried", "this led us to")
3. Move narrative content to the appropriate IDEALOG.md section
4. Reorganize remaining KNOWLEDGE.md for clean reference lookup

#### Checklist
- [ ] Audit boundary_conduction_speedup KNOWLEDGE.md — classify and split
- [ ] Audit ionic_model_optimization KNOWLEDGE.md
- [ ] Audit engine_consolidation KNOWLEDGE.md
- [ ] Audit geometry_induced_pacemaking KNOWLEDGE.md
- [ ] Audit mature_hipsc_cm_models KNOWLEDGE.md
- [ ] Verify no reference material was accidentally moved out
- [ ] Verify IDEALOGs got the narrative content that was missing from backtrace

---

### Step 0.2b: Audit and split research_environment_optimization KNOWLEDGE.md

**Model**: opus

#### Read First
- `Research/Active/research_environment_optimization/KNOWLEDGE.md` — 63KB, must read in chunks (offset/limit)
- `Research/Active/research_environment_optimization/IDEALOG.md` — from Step 0.1

#### Why
This KNOWLEDGE.md is 63KB and heavily mixed — the largest and most complex. It contains ECC structural comparisons, template designs, decision tables (reference material, stays), AND design conflict evolution narratives, "Revised Skills Plan" progression, abandoned approaches (narrative, moves to IDEALOG). Needs dedicated attention with chunked reading strategy.

#### Implementation Spec

1. Read KNOWLEDGE.md in 200-line chunks to build full understanding
2. Classify sections:
   - **Stays**: ECC analysis, structural comparisons, template designs, document type tables, final decision tables, skill designs (/reason, /blueprint, /save-session, /audit), architecture diagrams
   - **Moves to IDEALOG**: "Design Conflict Analysis" narrative (the story of conflicts discovered and resolved), "Revised Skills Plan" evolution history (the story of how the plan changed from 6 skills → extend 3 + add 3 → three-doc → four-doc → back to three-doc), abandoned approaches (WORKLOG, /aside)
   - **Judgment calls**: Decision tables contain both the final decision (reference) AND the evolution (narrative). Keep the final decision rows, move the strikethrough/evolution history to IDEALOG Thread.
3. Reorganize remaining KNOWLEDGE.md — group by topic (document architecture, skill designs, ECC comparison), not by chronological discussion order

#### Checklist
- [ ] Read full KNOWLEDGE.md in chunks (0-200, 200-400, 400-600, 600-800, 800-end)
- [ ] Classify all sections
- [ ] Move narrative content to IDEALOG.md Thread
- [ ] Reorganize remaining KNOWLEDGE.md by topic
- [ ] Verify file size (should shrink — narrative removed)
- [ ] Verify no reference material lost (spot-check: template designs, comparison tables, final decisions all still present)

#### Risk
63KB file — can't read in one pass. Must use chunked reads. Risk of losing information during reorganization. Mitigation: record file size before and after, do targeted edits, keep a mental inventory of key sections (templates, tables, diagrams) and verify they survive.

---

### Step 0.3: Create MASTER_KNOWLEDGE_INDEX.md

**Model**: opus

#### Read First
- `Research/Active/*/KNOWLEDGE.md` — all 6 question knowledge files (after migration in Step 0.2)
- `Research/Knowledge/*.md` — 3 promoted knowledge files from completed questions
- `ResearchStatement/` — researcher's personal statement materials

#### Why
Research questions are currently knowledge silos. MASTER_KNOWLEDGE_INDEX.md is the index book — it tells you where knowledge lives and how questions connect, without duplicating findings. Without it, `/reason` and `/save-session` can't see cross-question connections.

#### Implementation Spec

**Files to create:**
- `MASTER_KNOWLEDGE_INDEX.md` (project root)

**Structure:**
1. Research Statement section (researcher's goals, thesis direction — lives here only, not duplicated elsewhere)
2. Knowledge Index table (one-liner per question + status + link to KNOWLEDGE.md)
3. Cross-References section (how questions connect to each other, with links to both sides)
4. Cover all 6 active + 3 completed questions
5. NO duplicated findings — just pointers and connections

#### Checklist
- [ ] Read all 9 question KNOWLEDGE.md files (6 active + 3 completed)
- [ ] Read research statement materials
- [ ] Write Research Statement section
- [ ] Write Cross-References section (how questions connect, with links to both sides)
- [ ] Write Knowledge Index table (one-liner per question + status + link — no duplicated findings)
- [ ] Verify all links point to correct KNOWLEDGE.md files

#### Risk
May miss cross-question connections on first pass. Mitigation: `/save-session` Job 5 will update this incrementally over time. The initial version doesn't need to be perfect.

---

### Step 0.4: Update CLAUDE.md

**Model**: sonnet

#### Read First
- `CLAUDE.md` — current 209-line file
- `Research/Active/research_environment_optimization/KNOWLEDGE.md` — three-document architecture section for reference

#### Why
CLAUDE.md loads every session. It must reflect the new architecture so Claude knows about IDEALOG.md, the three-document structure, and all new skills. Without this update, Claude will continue treating KNOWLEDGE.md as the sole knowledge document.

#### Implementation Spec

**Sections to add/modify in CLAUDE.md:**

1. Add to `## Project Architecture` or create new section `## Research Document Architecture`:
   - Three-document structure: KNOWLEDGE (reference) + IDEALOG (thinking trail) + PLAN (agent steps)
   - When to write to which document
   - IDEALOG.md is NOT promoted on completion
   - MASTER_KNOWLEDGE_INDEX.md as cross-cutting index book

2. Update the skill table in `## Research & Textbook Workflows`:
   - **Fix header**: currently says "Four custom slash commands...defined in `.claude/commands/`" — wrong. Skills are in `.claude/skills/`, not `commands/`. Fix the header and count.
   - Add: `/reason`, `/blueprint`, `/save-session`, `/audit`, `/verify`, `/build-fix`, `/strategic-compact`
   - Update descriptions for `/research-new`, `/research-resume`, `/research-update`, `/research-complete`

3. Add to `## V5.4 Implementation Workflow` or create `## Planning Workflow`:
   - `/reason` → `/blueprint` → `/audit` → execute pipeline
   - Reference to PLAN.md format

#### Checklist
- [ ] Read current CLAUDE.md
- [ ] Add three-document architecture section
- [ ] Update skill table with all new/modified skills
- [ ] Add planning workflow section
- [ ] Verify total length stays reasonable (<300 lines — currently 209)
- [ ] No existing engine workflow rules removed

#### Risk
CLAUDE.md is loaded every session. If it gets too long (>400 lines), it wastes context. Mitigation: keep additions concise, use tables not prose, reference KNOWLEDGE.md for details rather than inlining.

---

### Phase 0 Verification
```bash
# All 6 IDEALOGs exist
ls Research/Active/*/IDEALOG.md

# MASTER_KNOWLEDGE_INDEX.md exists
ls MASTER_KNOWLEDGE_INDEX.md

# No KNOWLEDGE.md has grown (they should have shrunk or stayed same)
wc -c Research/Active/*/KNOWLEDGE.md

# CLAUDE.md is under 300 lines
wc -l CLAUDE.md

# Spot-check: research_environment_optimization IDEALOG has content
wc -l Research/Active/research_environment_optimization/IDEALOG.md
```

### Phase 0 Exit Criteria
- [ ] 6 IDEALOG.md files exist with content from chat log backtrace
- [ ] 6 KNOWLEDGE.md files contain only reference material
- [ ] MASTER_KNOWLEDGE_INDEX.md exists with research statement + per-question summaries + cross-references
- [ ] CLAUDE.md reflects three-document architecture and lists all new skills
- [ ] No information lost — every fact still findable in KNOWLEDGE, every narrative in IDEALOG

### Phase 0 Cleanup
- Remove any temporary files created during backtrace
- Verify git status is clean (no untracked temp files)

**→ Commit point: "Foundation: three-document architecture, IDEALOG backtrace, CLAUDE.md update"**

---

## Phase 1: Rewire Existing Skills

**Goal**: Extend the 4 research lifecycle skills to work with the three-document structure. All 4 must ship together — inconsistent state breaks the workflow.
**Tier**: medium
**Estimated scope**: 4 skill files, ~50-80 lines of changes each

### Phase Context
Skill files are in `.claude/skills/{name}/SKILL.md`. They're markdown files with YAML frontmatter. The skill body is the prompt that Claude follows when the skill is invoked. Changes must preserve existing functionality while adding IDEALOG.md awareness.

Current skill sizes: research-new (193 lines), research-resume (135 lines), research-update (127 lines), research-complete (130 lines).

### Step 1.1: Extend `/research-new`

**Model**: sonnet

#### Read First
- `.claude/skills/research-new/SKILL.md` — current 193 lines

#### Why
`/research-new` creates the folder structure for a new research question. It must now also create IDEALOG.md with the template.

#### Implementation Spec

**Files to modify:**
- `.claude/skills/research-new/SKILL.md`

**Changes:**
1. In Step 4 (Create Folder Structure), add IDEALOG.md to the created files list
2. Add Step 5b: Write IDEALOG.md with the template (Current Direction, Next Step, Thread, Failed Approaches, Session Log — all empty initially)
3. Update the folder structure diagram to include IDEALOG.md

#### Checklist
- [ ] Read current research-new SKILL.md
- [ ] Add IDEALOG.md to folder structure creation
- [ ] Add IDEALOG.md template writing step
- [ ] Update folder structure diagram
- [ ] Verify YAML frontmatter is valid

---

### Step 1.2: Extend `/research-resume`

**Model**: sonnet

#### Read First
- `.claude/skills/research-resume/SKILL.md` — current 135 lines

#### Why
`/research-resume` loads context and presents a briefing when resuming work on a question. It must now read IDEALOG.md and include "What NOT to retry", "Current Direction", and "Next Step" in the briefing.

#### Implementation Spec

**Files to modify:**
- `.claude/skills/research-resume/SKILL.md`

**Changes:**
1. In Step 2 (Read Context), add IDEALOG.md to the parallel reads
2. In Step 6 (Present Session Brief), add three new sections:
   - "Current Direction" — from IDEALOG.md
   - "Next Step" — from IDEALOG.md
   - "What NOT to retry" — from IDEALOG.md Failed Approaches
3. Add fallback for two cases:
   - If IDEALOG.md doesn't exist: skip these briefing sections gracefully
   - If IDEALOG.md exists but sections contain only template placeholder text (curly-brace patterns like `{What we're currently pursuing}`): treat as empty and skip, don't display raw template text

#### Checklist
- [ ] Read current research-resume SKILL.md
- [ ] Add IDEALOG.md to context reads
- [ ] Add three new briefing sections
- [ ] Add fallback for missing IDEALOG.md
- [ ] Verify briefing format is clean

---

### Step 1.3: Extend `/research-update`

**Model**: sonnet

#### Read First
- `.claude/skills/research-update/SKILL.md` — current 127 lines

#### Why
`/research-update` records findings and status changes. It must now route updates to the correct document based on type: reference material → KNOWLEDGE.md, thinking trail → IDEALOG.md.

#### Implementation Spec

**Files to modify:**
- `.claude/skills/research-update/SKILL.md`

**Changes:**
1. In Step 3 (Classify the Update), add new types: `idea`, `failure`, `issue`, `next-step`
2. In Step 4 (Apply the Update), add routing logic:
   - `finding`, `correction`, `criterion met`, `status change` → KNOWLEDGE.md (existing behavior)
   - `idea` → IDEALOG.md Thread section
   - `failure` → IDEALOG.md Failed Approaches section
   - `issue` → IDEALOG.md Thread section (as a discovered issue)
   - `next-step` → IDEALOG.md Next Step field
3. Update the classification table with new types

#### Checklist
- [ ] Read current research-update SKILL.md
- [ ] Add new update types to classification table
- [ ] Add routing logic for IDEALOG.md writes
- [ ] Preserve all existing KNOWLEDGE.md routing
- [ ] Verify the skill can handle both documents

---

### Step 1.4: Extend `/research-complete`

**Model**: sonnet

#### Read First
- `.claude/skills/research-complete/SKILL.md` — current 130 lines

#### Why
`/research-complete` moves a question from Active/ to Complete/ and promotes KNOWLEDGE.md. It must now also handle IDEALOG.md — archive it with the question (don't promote, don't delete).

#### Implementation Spec

**Files to modify:**
- `.claude/skills/research-complete/SKILL.md`

**Changes:**
1. In Step 5 (Move to Complete), note that IDEALOG.md moves with the folder (it's inside the question folder, so it moves automatically)
2. In Step 6 (Promote Knowledge File), explicitly note: "IDEALOG.md is NOT promoted to Research/Knowledge/. It stays in Complete/{question}/ as a historical archive."
3. Add a step to verify IDEALOG.md exists before completing (warn if missing)

#### Checklist
- [ ] Read current research-complete SKILL.md
- [ ] Add IDEALOG.md archival note
- [ ] Add explicit "not promoted" statement
- [ ] Add existence check/warning

---

### Phase 1 Verification
```bash
# Test /research-new creates IDEALOG.md (dry run on a test question)
# Test /research-resume reads IDEALOG.md (resume an existing question)
# Test /research-update routes correctly (update with type=failure)
# Test /research-complete handles IDEALOG (manual verification of skill text)
```

### Phase 1 Exit Criteria
- [ ] All 4 skill files updated
- [ ] `/research-new` creates IDEALOG.md with template
- [ ] `/research-resume` briefing includes Direction, Next Step, What NOT to retry
- [ ] `/research-update` routes idea/failure/issue/next-step to IDEALOG.md
- [ ] `/research-complete` archives IDEALOG.md (not promoted)
- [ ] Backward compatible — works even if IDEALOG.md doesn't exist yet

### Phase 1 Cleanup
- Verify no syntax errors in YAML frontmatter
- Read each skill file once more to check for consistency

**→ Commit point: "Extend research skills: IDEALOG.md support in new/resume/update/complete"**

---

## Phase 2: Core New Skills

**Goal**: Create the planning pipeline (/reason → /blueprint → /audit) and the cleanup agent (/save-session).
**Tier**: large
**Estimated scope**: 4 new skill files, each 100-300 lines

### Phase Context
New skills go in `.claude/skills/{name}/SKILL.md`. Each skill is a markdown file with YAML frontmatter (name, description, argument-hint) and a body that instructs Claude on what to do when the skill is invoked. The skill body references the three-document architecture and the templates defined in this question's KNOWLEDGE.md.

### Step 2.1: Create `/reason`

**Model**: opus

#### Read First
- `Research/Active/research_environment_optimization/KNOWLEDGE.md` — `/reason` design sections (write behavior, presentation model, ECC comparison, borrowed patterns)
- `.claude/skills/research-resume/SKILL.md` — similar interactive skill for reference
- Existing active question IDEALOGs — to understand what `/reason` will read and write

#### Why
`/reason` is the interactive reasoning buddy — the most complex new skill. It must handle: big→middle→small zoom presentation, organic flow following, transition-based writes to IDEALOG.md, trade-off analysis tables, domain-specific red flags, and seamless handoff to `/blueprint`.

#### Implementation Spec

**Files to create:**
- `.claude/skills/reason/SKILL.md` (~150-200 lines — keep concise, reference KNOWLEDGE.md for detailed templates and examples rather than inlining)

**Frontmatter:**
```yaml
---
name: reason
description: Interactive reasoning buddy for planning and exploring ideas. Presents big-picture map, drills into details on demand, follows organic thinking flow. Writes settled decisions and failed approaches to IDEALOG.md on natural transitions. Can invoke /blueprint when ready to implement.
argument-hint: "[objective or topic to reason about]"
---
```

**Body structure:**
1. Context loading (read IDEALOG, KNOWLEDGE, README, engine files)
2. Big-picture presentation format (ASCII map with phases, risks, connections)
3. Middle-level presentation format (phase detail with steps, dependencies, key questions)
4. Small-level presentation format (step detail with files, changes, risks)
5. Write behavior rules (when to write, what to write, where to write)
6. Trade-off analysis format (visual comparison tables)
7. Red flags checklist (domain-specific anti-patterns to flag proactively)
8. Handoff to `/blueprint` (when user says "let's build this")
9. Organic flow rules (follow the user, don't force hierarchy)

#### Pseudocode
```
# /reason skill flow
1. Read IDEALOG.md (Current Direction, Failed Approaches)
2. Read KNOWLEDGE.md (relevant findings)
3. Read README.md (completion criteria)
4. If engine work: read PROGRESS.md, IMPLEMENTATION.md
5. Present BIG PICTURE map (ASCII diagram, scannable in 5s)
6. WAIT for user input
7. LOOP:
   a. If user drills down → present next zoom level
   b. If user jumps to new topic → follow, present at appropriate level
   c. If user settles a decision → WRITE to IDEALOG.md Thread
   d. If user rejects approach → WRITE to IDEALOG.md Failed Approaches
   e. If user shifts topic → WRITE summary of previous topic conclusions
   f. If user says "write that down" → WRITE to appropriate IDEALOG section
   g. If user says "let's build this" → trigger /blueprint, exit loop
   h. If user says "done" or changes subject → exit loop
```

#### Test Spec
- Manual test: invoke `/reason "add anisotropic diffusion"` and verify:
  - Big picture map appears with phases, risks, connections
  - Can drill into phases and steps
  - Can jump organically without being forced back
  - Decisions get written to IDEALOG.md (check file after session)
  - "Let's build this" invokes /blueprint

#### Checklist
- [ ] Create `.claude/skills/reason/SKILL.md`
- [ ] Write YAML frontmatter
- [ ] Write context loading section
- [ ] Write big-picture presentation format with ASCII diagram template
- [ ] Write middle-level presentation format
- [ ] Write small-level presentation format
- [ ] Write transition-based write behavior rules
- [ ] Write trade-off analysis format
- [ ] Write red flags checklist (domain-specific)
- [ ] Write /blueprint handoff logic
- [ ] Write organic flow rules
- [ ] Verify skill loads correctly

#### Risk
Skill files consume context for the entire session. Target 200 lines max. Mitigation: the skill contains rules and triggers only. ASCII diagram templates, trade-off table formats, and detailed examples reference KNOWLEDGE.md rather than being inlined. The implementing agent should test context impact by checking response quality after loading the skill.

---

### Step 2.2: Create `/blueprint`

**Model**: opus

#### Read First
- `Research/Active/research_environment_optimization/KNOWLEDGE.md` — `/blueprint` design section (PLAN.md format, complexity tiers, ECC patterns)
- This file (`PLAN.md`) — as a reference for the format `/blueprint` should generate

#### Why
`/blueprint` is the autonomous pipeline that converts settled ideas in IDEALOG.md into a machine-targeted PLAN.md. It reads the codebase, generates phased steps with full scaffolding (context briefs, implementation specs, pseudocode, test specs, checklists), and asks about adversarial audit.

#### Implementation Spec

**Files to create:**
- `.claude/skills/blueprint/SKILL.md` (~200-250 lines)

**Body structure:**
1. Input: reads IDEALOG.md (settled approach, known failures) + codebase. **Precondition**: if IDEALOG.md Thread section is empty or has no settled decisions, advise the user to use `/reason` first and exit.
2. Analysis: identify affected files, existing tests, current IMPLEMENTATION.md sections
3. Decomposition: break into phases (independently deliverable) → steps within phases
4. For each step: generate Context Brief, Implementation Spec, Pseudocode, Test Spec, Checklist
5. Assign complexity tiers and model routing per step
6. Add Known Failures section from IDEALOG.md
7. Add Cleanup Pass (de-sloppify) per phase + final
8. Write PLAN.md to the research question folder (`Research/Active/{question}/PLAN.md`)
9. Ask: "Want adversarial audit? (/audit)"

#### Checklist
- [ ] Create `.claude/skills/blueprint/SKILL.md`
- [ ] Write YAML frontmatter
- [ ] Write IDEALOG + codebase reading logic
- [ ] Write phase decomposition rules
- [ ] Write step scaffolding generation (all 6 sections per step)
- [ ] Write PLAN.md output format (matching the template in KNOWLEDGE.md)
- [ ] Write complexity tier assignment rules
- [ ] Write /audit prompt
- [ ] Verify skill generates valid PLAN.md structure

---

### Step 2.3: Create `/save-session`

**Model**: opus

#### Read First
- `Research/Active/research_environment_optimization/KNOWLEDGE.md` — `/save-session` detailed design (5 jobs)

#### Why
`/save-session` is the comprehensive cleanup agent. Five jobs: (1) session snapshot → IDEALOG, (2) full editorial pass on KNOWLEDGE.md, (3) cross-reference IDEALOG ↔ KNOWLEDGE, (4) condense verbose IDEALOG entries, (5) update MASTER_KNOWLEDGE_INDEX.md with cross-question findings. Can take as long as needed.

#### Implementation Spec

**Files to create:**
- `.claude/skills/save-session/SKILL.md` (~180-230 lines)

**Body structure:**
1. Determine which research question is active (from conversation context or ask). If no research question is active (e.g., engine-only work without `/research-resume`), skip Jobs 1-4 and offer to run Job 5 only (MASTER_KNOWLEDGE_INDEX.md update), or ask the user which question to associate the session with.
2. Job 1: Append session snapshot to IDEALOG.md Session Log
3. Job 2: Full editorial pass on KNOWLEDGE.md (restructure, rewrite, merge, polish)
4. Job 3: Cross-reference IDEALOG ↔ KNOWLEDGE (graduate validated ideas, flag inconsistencies)
5. Job 4: Condense verbose IDEALOG thread entries (preserve narrative + decisions)
6. Job 5: Update MASTER_KNOWLEDGE_INDEX.md — check for cross-question relevance, update per-question summary, add/update cross-references between questions
7. Report what was changed

#### Checklist
- [ ] Create `.claude/skills/save-session/SKILL.md`
- [ ] Write YAML frontmatter
- [ ] Write question detection logic
- [ ] Write Job 1: session snapshot format and append logic
- [ ] Write Job 2: KNOWLEDGE.md editorial pass rules
- [ ] Write Job 3: cross-reference check table
- [ ] Write Job 4: IDEALOG condensation rules
- [ ] Write Job 5: MASTER_KNOWLEDGE_INDEX.md update logic (one-liner + cross-references)
- [ ] Write report format
- [ ] Verify skill handles large KNOWLEDGE.md files

---

### Step 2.4: Create `/audit`

**Model**: sonnet

#### Read First
- `Research/Active/research_environment_optimization/KNOWLEDGE.md` — `/audit` design (adversarial review, author-bias elimination)

#### Why
`/audit` spawns an Opus subagent with read-only tools to adversarially review any document. It's opt-in, never auto-triggered.

#### Implementation Spec

**Files to create:**
- `.claude/skills/audit/SKILL.md` (~80-100 lines)

**Body structure:**
1. Accept argument: path to document (defaults to PLAN.md in current research question)
2. Instruct Claude to use an Agent subagent (model: opus) for the review. The skill prompts the Agent to use only Read/Grep/Glob tools (instruction-enforced, not system-enforced — Claude Code Agent tool supports `model` parameter but tool restriction is honor-system in the agent prompt).
3. Agent prompt: review for completeness, dependency errors, missed edge cases, domain-specific anti-patterns (float64, V5.3 protection, cardiac_core dedup, backlinks). Use ONLY Read, Grep, Glob tools — do NOT edit any files.
4. Return severity-sorted issue list
5. For PLAN.md reviews: check each step has all required sections

**Limitation**: Tool restriction is prompt-based (honor-system), not system-enforced. The Agent tool supports `model` selection but not tool whitelisting. The audit agent COULD edit files if it ignores the prompt. Acceptable risk — the prompt is explicit and the review is read-only by nature.

#### Checklist
- [ ] Create `.claude/skills/audit/SKILL.md`
- [ ] Write YAML frontmatter
- [ ] Write Agent subagent invocation with model: opus
- [ ] Write review prompt with explicit "DO NOT edit files" instruction
- [ ] Write domain-specific review checklist
- [ ] Write output format (severity-sorted issues)

---

### Phase 2 Verification
```bash
# All 4 new skill directories exist
ls .claude/skills/reason/SKILL.md
ls .claude/skills/blueprint/SKILL.md
ls .claude/skills/save-session/SKILL.md
ls .claude/skills/audit/SKILL.md

# Each has valid YAML frontmatter
head -5 .claude/skills/reason/SKILL.md
head -5 .claude/skills/blueprint/SKILL.md
head -5 .claude/skills/save-session/SKILL.md
head -5 .claude/skills/audit/SKILL.md
```

### Phase 2 Exit Criteria
- [ ] `/reason` skill file created with full big→middle→small presentation + write behavior
- [ ] `/blueprint` skill file created with PLAN.md generation logic
- [ ] `/save-session` skill file created with 5-job cleanup agent
- [ ] `/audit` skill file created with subagent spawn + domain-specific review
- [ ] All skills have valid YAML frontmatter and load without errors

### Phase 2 Cleanup
- Verify skill file sizes are reasonable (<300 lines each)
- Check for consistency across skills (same document names, same conventions)

**→ Commit point: "New skills: reason, blueprint, save-session, audit"**

---

## Phase 2b: Independent Skills (no ordering dependency with Phase 2)

**Goal**: Create the 3 utility skills that don't depend on the document restructuring.
**Tier**: small
**Estimated scope**: 3 simple skill files, each 50-120 lines

### Phase Context
These skills are standalone — they read engine structure and run commands. No dependency on IDEALOG.md, PLAN.md, or the three-document architecture.

**Note on "parallel"**: This means Phase 2b has no ordering dependency on Phase 2 — it can be done before, after, or interleaved. It does NOT mean "execute simultaneously in separate agents." If done in the same session as Phase 2, just do them sequentially. Commits can be separate or combined.

### Step 2b.1: Create `/verify`

**Model**: sonnet

#### Implementation Spec

**Files to create:**
- `.claude/skills/verify/SKILL.md` (~80-100 lines)

Auto-detect engine from working directory or recent file changes. Run the right test suite. Produce pass/fail report.

| Engine | Test Command |
|--------|-------------|
| Bidomain V1 | `cd Bidomain/Engine_V1 && conda run -n heart-conduction pytest tests/ -v` |
| Monodomain V5.4 | `cd Monodomain/Engine_V5.4 && conda run -n heart-conduction python test_phase7.py && python test_phase8.py` |
| LBM V1 | `cd Monodomain/LBM_V1 && conda run -n heart-conduction python -m pytest tests/ -v` |

Modes: `quick` (tests only), `full` (tests + artifact check + diff), `pre-commit`.

#### Checklist
- [ ] Create `.claude/skills/verify/SKILL.md`
- [ ] Write engine detection logic
- [ ] Write test command table
- [ ] Write report format
- [ ] Write modes (quick/full/pre-commit)

---

### Step 2b.2: Create `/build-fix`

**Model**: sonnet

#### Implementation Spec

**Files to create:**
- `.claude/skills/build-fix/SKILL.md` (~100-120 lines)

Parse test failures, group by type (import → type → assertion → numerical), fix one at a time, re-run after each fix, stop if fix introduces more errors or same error persists after 2 attempts. Never modify V5.3.

#### Checklist
- [ ] Create `.claude/skills/build-fix/SKILL.md`
- [ ] Write error parsing and prioritization logic
- [ ] Write fix-one-at-a-time loop
- [ ] Write guardrails (stop conditions)
- [ ] Write V5.3 protection rule

---

### Step 2b.3: Create `/strategic-compact`

**Model**: sonnet

#### Implementation Spec

**Files to create:**
- `.claude/skills/strategic-compact/SKILL.md` (~60-80 lines)

Decision table for phase transitions. Pre-compact checklist (save to IDEALOG, commit, note next step). What survives vs what's lost.

#### Checklist
- [ ] Create `.claude/skills/strategic-compact/SKILL.md`
- [ ] Write decision table (when to compact: yes/no/maybe per transition type)
- [ ] Write pre-compact checklist
- [ ] Write "what survives" reference
- [ ] Write integration with `/save-session` (suggest running it first)

---

### Phase 2b Exit Criteria
- [ ] All 3 skill files created and loadable
- [ ] `/verify` correctly identifies engine from test paths

**→ Commit point: "New skills: verify, build-fix, strategic-compact"**

---

## Phase 3: Hook + Final Polish

**Goal**: Add the PreCompact hook and do a final CLAUDE.md pass to register everything.
**Tier**: small
**Estimated scope**: 1 hook script, 1 settings.json update, 1 CLAUDE.md final edit

### Step 3.1: Create PreCompact hook

**Model**: sonnet

#### Implementation Spec

**Files to create:**
- `.claude/hooks/pre-compact.sh` — shell script

**Files to create or modify:**
- `.claude/settings.json` — add hook configuration (create file if doesn't exist, merge with existing settings.local.json permissions)

Hook behavior:
1. Log compaction event with timestamp to `~/.claude/sessions/compaction-log.txt`
2. Output reminder: "After compaction, re-read PROGRESS.md and IDEALOG.md for the active question"
3. If `/reason` session is active (detected via temp file marker): emergency dump of unsaved insights to IDEALOG.md

#### Checklist
- [ ] Verify PreCompact is a supported hook event type (confirmed in Claude Code docs — valid events include PreCompact, PostCompact, PreToolUse, PostToolUse, Stop, SessionStart, SessionEnd)
- [ ] Create `.claude/hooks/pre-compact.sh`
- [ ] Make executable (`chmod +x`)
- [ ] Create or update `.claude/settings.json` with PreCompact hook config (merge with existing settings.local.json — don't overwrite permissions)
- [ ] Test: pipe empty JSON and verify exit code 0
- [ ] Verify settings.json is valid JSON after merge

---

### Step 3.2: Final CLAUDE.md verification

**Model**: sonnet

#### Implementation Spec

Read CLAUDE.md one final time. Verify:
- All 7 new skills listed in skill table
- All 4 extended skills have updated descriptions
- Three-document architecture section is accurate
- Planning workflow section references `/reason` → `/blueprint` → `/audit`
- No stale references to old architecture

#### Checklist
- [ ] Read CLAUDE.md
- [ ] Verify all skills listed
- [ ] Verify architecture section
- [ ] Verify no stale references
- [ ] Final line count check (<300 lines)

---

### Phase 3 Exit Criteria
- [ ] PreCompact hook fires on compaction events
- [ ] CLAUDE.md is complete and accurate
- [ ] All settings.json entries are valid

### Phase 3 Cleanup
- Run `/verify` on all engines to confirm no regressions
- Run `/research-status` to verify all questions show correct state (note: `/research-status` doesn't check IDEALOG.md yet — it only validates README-level status. Consider extending it in a future pass.)

**→ Commit point: "PreCompact hook, final CLAUDE.md update — research environment optimization complete"**

---

## Final Cleanup (after all phases)

- Update this question's README.md — check off all completion criteria
- Update MASTER.md — status to "Implementation complete"
- Run `/save-session` on this question — organize KNOWLEDGE.md, cross-reference with IDEALOG
- Consider `/research-complete` if all criteria are met

## Mutation Log
{To be filled during execution}
