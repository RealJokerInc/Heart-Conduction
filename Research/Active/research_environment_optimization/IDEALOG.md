# Research Environment Optimization — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Architecture is live (19 skills, 4-doc system, tmux workspace). Real-world usage has surfaced four systemic pain points, with #4 as the root amplifier: (1) agentic tasks don't self-improve, (2) skill guidelines decay in long conversations, (3) no dedicated plan execution agent, **(4) compaction destroys half the context budget because too much state lives in conversation instead of persistent files**. Core solution direction: continuous state externalization during sessions (not just at session end), L0/L1 progressive loading for post-compaction recovery, and structured memory categories for agent behavioral learnings. OpenViking's architecture analyzed as design reference.

## Next Step
Design solutions for the three identified system pain points: (1) agentic task self-improvement via structured memory feedback loops, (2) skill guideline enforcement that survives context decay, (3) dedicated plan execution agent/skill with human-in-the-loop gates.

## Thread

### 2026-05-30/31: Repository structure cleanup + venv history rewrite (executed)
Big workspace-hygiene session. Audited all major folders (5 parallel agents), found the "mess" was ~80% undocumented-but-live dirs, ~15% half-done migrations (two of everything), ~5% real git debt. Planned in PLAN.md + two media plans, each adversarially audited (caught real bugs: name-only dedup → md5; advisory grep "gates" that didn't actually block → real `exit 1` guards; a backup step whose branch-switch silently reverted the working tree).

**Root issue discovered mid-execution:** `.git` was 4.7 GB because a `venv/` of CUDA libs (7.6 GB across history) had been committed once (2026-04-09) then deleted-from-tree but stranded in history → blocked all pushes (GitHub 2 GiB pack cap). Split out PLAN_history_rewrite.md, audited it, executed: `git bundle --all` backup → `git filter-repo --path venv --invert-paths` → force-push. **`.git` 4.7 GB → 737 MB**, venv gone, full-history push works again. All commit SHAs changed; re-pinned MEMORY.md SHA references (drift-check `8f191f77`→`ebea5b5c`).

**Media model (user decision):** abandoned the old "Images/ centralized copies" (stale + duplicated) for a single `media/{question}/{images|videos}/{YYYY-MM-DD}/{slug}_NN.ext` tree. Moved 296 files repo-wide, dropped 50 byte-dups, renamed 187 to fixed format. Left vendored `code_examples/`, Builder inputs, and `Monodomain/_archive/` figures in place. 322 MB simulation bulk → `media/.../_sim_outputs/` (gitignored). Diagram *sources* (.tex/.py) → `Surrogate/docs/diagrams/`.

**Also:** untracked 286 `.pyc`; deleted deprecated Q1–Q8, dead `harness_v1/`, dead nested `LBM/Engine_V1/ionic/ionic/`; migrated 4 LBM PDFs to `lbm_ep/papers/`; fixed broken `Engines/cardiac_core` symlink; documented `cardiac_core`/`cardiac_ml`/`simulation`/`media` + the Engines|Pipelines symlink convention in CLAUDE.md. Backups: `~/heart-conduction-PREWRITE-2026-05-30.bundle` (4.7 GB, verified) + `origin/main @ 9f7e6a7` (pre-cleanup state).

**Incident/recovery (learned):** the original PLAN.md Phase 0 backup used `git switch -c backup → add -A → commit → switch main`, which REVERTS main's working tree (changes move to the backup branch). Recovered via `merge --ff-only` + `git reset`. Corrected method: snapshot without switching branches (`commit-tree`/stash). Logged in PLAN.md.

**Open / next:** triage the 72 files in `media/_unmapped/` into questions; run engine test suites (Bidomain/LBM/V5.4) to confirm no script relied on a moved image path (none expected — moves were figures/outputs, golden `.npy` untouched); delete the prewrite bundle once confident; concurrent user commit `1f6c72e` (cardiac_core canonical ionic) coexists cleanly on main.

### 2026-05-11: Two-system architecture + Obsidian vault as sibling repo, coarse-first

After community research on how Obsidian is used in AI-assisted academic workflows, three settled decisions:

**1. Two parallel systems, not one.** A user notebook (for human reading/navigating/thinking) and an AI memory KB (for Claude recall + behavioral learnings) are separate concerns with separate substrates. The current KNOWLEDGE.md / IDEALOG.md serve as pseudo AI-memory, but the actual fixes for pain points #1, #2, #4 belong in a structured AI-side memory system. The user notebook (Obsidian) is quality-of-life; the AI memory KB is where workflow leverage lives. Both designed separately, neither overloaded.

**2. Obsidian vault is a sibling repo (Option C), mobile-carryable.** Lives at e.g. `~/Documents/Heart-Conduction-Vault/`, separate git history, independent backup, mobile sync via Obsidian Sync or self-hosted alternative. Keeps the cardiac simulation repo lean — no `.obsidian/workspace.json` churn polluting cardiac git history. Vault travels with the user across machines without dragging the engine code along.

**3. One-way translation, start coarse, emergent atomization.** Research/ remains source of truth. Vault is a rendered downstream view. Translation skills run one-way only (Research → Vault); vault never writes back to Research/. Three-tier vault structure:

- `Vault/Maps/{question}.md` — generated, mechanical translation of README + KNOWLEDGE.md (add frontmatter, convert section links to wikilinks, no rewriting)
- `Vault/Journal/{date}_{question}.md` — generated, one dated note per IDEALOG Thread entry, verbatim prose
- `Vault/Concepts/{slug}.md` — hand-authored only, NOT generated, NEVER auto-written
- `Vault/Index.md` — generated root MOC linking all Maps

Translation skill (`/obsidian-sync`) writes Maps + Journal + Index mechanically (lossless, no LLM rewriting → no AI ghosts in generated content). For Concepts/, the skill emits a *suggestions report* listing concepts that appear across ≥2 questions ("Kleber effect mentioned in 4 KNOWLEDGE files — promote to atomic concept?") but writes nothing. User authors Concepts/ by hand as concepts are re-encountered. This is the emergent-atomization pattern documented by Andy Matuschak (evergreen notes graduate, not start), Eric Ma (atomization emerges from rewriting), and Emile van Krieken (tags + typed links, minimal atomization).

**Promotion criterion (empirically derived from cross-question grep):** a concept is atomic-worthy iff it appears in ≥2 active questions. Examples: Kleber speedup (4 questions), chi=1.0/Cm=1.0 convention (all engines), TTP06 ionic model (4+ questions), Rush-Larsen integration (all engines). Counter-examples that stay in their question: decoupled GS splitting, DST-I via odd-extension FFT, three-tier elliptic solver (all Bidomain-specific).

**Rejected**: (a) mass atomization on day one — community consensus is this is the #1 documented failure mode; (b) two-way translation — invites drift, merge headaches, AI-ghost risk for IDEALOG voice; (c) AI plugins inside Obsidian (Copilot for Obsidian, Smart Connections write capabilities) — redundant with Claude Code and creates dual authoring surfaces; (d) Zotero integration for v1 — papers/ folders too sparse to justify the plumbing yet; (e) vault inside cardiac repo — `.obsidian/workspace.json` git noise + couples two unrelated lifecycles.

**Anti-patterns to watch for**: AI ghosts in Concepts/ (Caspar Addyman field guide), "memory" marketing on plugins that are just file viewers (limitededitionjonathan's critique — markdown is text not a database), wikilink dialect drift in agent prompts (Eric Ma rule: markdown links by default, wikilinks only in human-curated MOCs), plugin sprawl (community-documented driver of Obsidian migration regret).

**References worth following up**: Eric Ma's PKM-with-AI blog (ericmjl.github.io, closest stack match), Karpathy LLM-Wiki gist + AgriciDaniel/claude-obsidian (slash-command pattern mirrors our existing skills), Andy Matuschak evergreen notes (notes.andymatuschak.org), Emile van Krieken academic Obsidian (counter-voice: don't let AI write notes). Smart Connections + Dataview as the two plugins worth piloting; everything else deferred.

**Next**: design `/obsidian-init`, `/obsidian-sync`, `/obsidian-status` skill API + worked example of what `Vault/Maps/{question}.md` actually contains. Then `/blueprint` for execution.

### 2026-03-30: Three systemic pain points identified from real-world usage

After several weeks of daily use, three major issues have surfaced that the current architecture does not address:

**Pain Point 1: Agentic tasks lack self-improvement / memory feedback loops**

Agentic tasks (subagents, background agents, skill executions) keep making the same errors across sessions. The auto-memory system captures user preferences and project facts, but does not capture *agent behavioral corrections*. Example: an agent repeatedly uses the wrong API pattern, gets corrected, but the next session's agent makes the same mistake. The issue is structural — corrections during agentic work are ephemeral (live only in conversation context) and never flow back into persistent memory. OpenViking addresses this with "cases" (problem→solution pairs) and "patterns" (reusable workflows), both stored as agent-side memories with merge operations. Our `feedback` memory type captures user-stated preferences but not agent-discovered corrections from execution errors.

**Pain Point 2: Skill guidelines and thresholds bypass in long conversations**

Skills define rules (e.g., "/reason NEVER implements", "/blueprint STOPS after PLAN.md", format requirements, logging requirements). These rules live in SKILL.md files and are only loaded when the skill is invoked. In long conversations, the skill instructions fade from active context, and the agent reverts to default behavior — skipping format requirements, ignoring thresholds, not logging to the correct document. This was already noted (2026-03-18 thread entry on skill guardrail persistence) but no solution was implemented. The root cause: **skill instructions are ephemeral context, not persistent behavioral rules**. CLAUDE.md and MEMORY.md are always loaded, but they can't contain the full text of 19 skills. Need a mechanism to keep critical behavioral rules in always-loaded context even when the originating skill has faded.

**Pain Point 3: No dedicated plan execution agent — ad-hoc plan following causes errors**

After `/blueprint` generates PLAN.md and the user approves, there is no dedicated agent/skill for executing the plan. The main conversation thread just reads PLAN.md and works through steps ad-hoc. This causes multiple failure modes:
- **Format violations**: Steps specify output formats, documentation requirements, or checklist items that get skipped.
- **Missed logging**: Steps require updating IDEALOG.md or KNOWLEDGE.md at checkpoints, but this gets forgotten mid-implementation.
- **Poor human-in-the-loop communication**: Steps may require user decisions (e.g., choosing between implementation alternatives, confirming a destructive action, approving a design trade-off). These decision points are not surfaced clearly — the agent either makes the decision silently or buries the question in output noise.
- **No step-level verification**: Steps have exit criteria and verification commands, but these are often skipped or partially executed.
- **Context drift**: In long implementations, earlier plan context (rationale, constraints, dependencies between steps) fades, leading to decisions that contradict the plan.

The missing piece is a `/execute` or `/implement` skill that: reads PLAN.md step by step, enforces each step's checklist before proceeding, surfaces human-decision-required gates explicitly, logs progress to IDEALOG.md at checkpoints, and runs verification commands before marking steps complete.

**Pain Point 4: Compaction destroys half the context budget**

Long sessions accumulate massive conversation context. When compaction fires, the summary itself consumes ~50% of the context window, leaving far less room for actual work in the continued session. This is the **root amplifier** for all other pain points:
- Pain Point 1 worsens: agent corrections made pre-compaction are compressed into lossy summaries, so the agent "forgets" behavioral learnings even within the same session.
- Pain Point 2 worsens: skill guidelines loaded pre-compaction are lost entirely — post-compaction, the agent has no memory that a skill was even invoked, let alone its rules.
- Pain Point 3 worsens: mid-plan execution context (which step we're on, decisions made, partial results) gets compressed, causing the agent to re-read files it already processed or skip steps it thinks it completed.

The root cause is **too much state lives in conversation context instead of persistent files**. The current approach: accumulate everything in context → eventually /save-session or /quicksave → hope compaction doesn't hit before then. The fix: **continuously externalize state to persistent files during the session**, so the conversation context stays lean and compaction summaries are small. This is exactly what OpenViking's session compression does — extract memories *during* the session (on commit), not just at the end.

Concrete symptoms observed:
- Post-compaction, re-reading KNOWLEDGE.md + IDEALOG.md + PLAN.md to recover context already consumes significant budget
- The compaction summary tries to preserve everything (tool outputs, code snippets, error traces, decisions) instead of just pointers to where that info is persisted
- `/strategic-compact` checklist helps but is reactive (triggered when context is already large) — needs proactive continuous externalization

**Connection to OpenViking**: Their session compression architecture directly addresses this. Phase 1 (sync): archive messages immediately. Phase 2 (async): generate L0/L1 summaries + extract memories. The key insight is **commit early and often** — don't wait for session end. Their `active_count` tracking means the system knows which memories are hot (recently used) vs cold (safe to drop from active context). Their L0/L1 layering means post-compaction recovery loads ~100 tokens (L0) per topic instead of full documents — exactly the progressive loading that would reduce our post-compaction budget consumption.

### 2026-03-30: OpenViking architecture analyzed for integration patterns

Cloned and analyzed `volcengine/OpenViking` (15K+ stars). Key architectural concepts:

**L0/L1/L2 progressive loading**: Every directory has `.abstract.md` (~100 tokens), `.overview.md` (~2k tokens), and full files. Generated bottom-up by LLM. This is the most transferable pattern — our KNOWLEDGE.md files are always L2 (full detail) with no lighter alternatives. Adding L0/L1 frontmatter to KNOWLEDGE.md would let `/research-resume` load progressively.

**6 memory categories**: profile (user identity), preferences (per-topic likes/dislikes), entities (Zettelkasten-style linked cards), events (immutable records), cases (problem→solution, agent-side), patterns (reusable workflows, agent-side). Each defined via YAML schema with field-level merge operations (patch=search/replace, sum=numeric, immutable=no-edit).

**ReAct extraction loop**: Session commit → LLM with tools (read/search) → extract candidate memories → vector pre-filter → LLM dedup decision (skip/create/merge/delete) → write to filesystem. Fully automated, no human intervention.

**Hotness scoring**: `sigmoid(log1p(access_count)) * exp_decay(recency, half_life=7d)`. Cold memories sink in search rankings. Simple but effective for preventing stale context pollution.

**What to adopt vs skip**: L0/L1 headers (high value, low effort), structured memory categories (medium value), hotness ordering (medium value). Skip: vector DB (overkill for ~10 files), YAML schemas (over-engineered for our scale), server mode (single user), Viking URI scheme (we have file paths).

### 2026-03-17: everything-claude-code analysis reveals engineering skill gap
Analyzed the everything-claude-code repository (50K+ stars, 25 agents, 57 commands, 108 skills, 22 MCP configs, 16 hooks). Core philosophy: "context is the scarcest resource." Audited our own 9 existing skills against their patterns. Key finding: our skills cover the research writing lifecycle comprehensively (6 skills from `/research-new` through `/research-complete`) but have zero coverage for engineering workflow — no planning, session persistence, verification, debugging, or compaction management. Recommended adopting: PreCompact hook, strategic compaction skill, cost tracking hook, model routing for subagents. Deferred: modular rules system, formalized agent library.

### 2026-03-17: ECC pattern conflicts discovered — two design collisions
Deeper comparison of ECC patterns against our existing architecture revealed two significant conflicts: (1) `/blueprint` contradicts our 3-file planning architecture (PROGRESS.md + IMPLEMENTATION.md + improvement.md) — two solutions to the same compaction recovery problem. (2) `/save-session` + `/resume-session` overlap heavily with `/research-resume` + `/research-update`. The real gap was negative knowledge — our entire research lifecycle captured positive knowledge only. Nothing recorded "tried X, failed because Y — don't retry." This was the single most valuable thing ECC's session system does. Resolution: three-document architecture (see next entry).

### 2026-03-17: Three-document architecture crystallized
After exploring several document structures, settled on three per-question documents: KNOWLEDGE.md (high-resolution reference — facts, analysis, designs, comparisons; promoted on completion), IDEALOG.md (low-resolution thinking trail — insights, failures, session snapshots; archived, not promoted), and PLAN.md (cold-start agent execution steps, generated by `/blueprint`). This resolves the overlap between ECC's `/blueprint` and our PROGRESS.md, and between ECC's `/save-session` and our `/research-update`. README.md remains the umbrella (status, criteria, literature, experiments).

### 2026-03-17: WORKLOG.md dropped — three documents is sufficient
Briefly explored adding WORKLOG.md as a separate "lab notebook" for tactical session details. Dropped because failed approaches and session snapshots are part of the thinking trail and belong in IDEALOG.md. Adding a fourth document would fragment context without adding value.

### 2026-03-17: /aside dropped — /btw is hardcoded, not extensible
Investigated implementing a multi-turn sidebar skill (`/aside`) with auto-cleanup of the conversation transcript. Discovery: the disappearing-from-transcript behavior is hardcoded into Claude Code's `/btw` UI overlay and not exposed to skills, hooks, or any extensibility system. Recommendation: use built-in `/btw` instead.

### 2026-03-17: /reason write behavior — transition-based, not per-exchange
Key design decision for the `/reason` interactive planning skill: it does NOT write to IDEALOG.md after every chat exchange. Instead, it accumulates ideas in context and writes batch updates on natural transitions — user settles a decision, rejects an approach, shifts topics, says "write that down", or PreCompact fires. A typical 30-minute `/reason` session produces 3-4 writes, not 15-20. This minimizes workflow disruption while ensuring nothing is lost.

### 2026-03-17: Renamed /plan to /reason
The original name `/plan` conflicted with the concept of PLAN.md (the cold-start agent execution document). Renamed to `/reason` to reflect its actual purpose: interactive thinking and discussion that eventually feeds into `/blueprint` for plan generation.

### 2026-03-17: PLAN.md location — per-question-folder, not project root
Audit found a contradiction: KNOWLEDGE.md said plans live in `./plans/` at project root (following ECC convention), but the actual PLAN.md was written inside the research question folder. Resolved: per-question-folder is correct for our domain-specific research structure. Generic project-root convention makes sense for ECC but not for us.

### 2026-03-17: Skills plan evolved through three iterations
Originally planned 6 new skills directly borrowed from ECC. After discovering the design conflicts with our existing architecture, revised to: extend 3 existing skills + add 3 new ones. After the three-document architecture crystallized, the plan settled on its final form: extend 4 existing research skills (new/resume/update/complete) + 7 new skills (/reason, /blueprint, /save-session, /audit, /verify, /build-fix, /strategic-compact) + 1 PreCompact hook. The key insight was that extending existing skills to handle IDEALOG routing was more natural than creating parallel skill trees.

### 2026-03-17: tmux integration works — lazy setup is viable
Tested tmux control from Claude Code's Bash tool. Despite earlier research saying it was impossible, `tmux split-window`, `tmux send-keys`, `tmux capture-pane`, and `tmux kill-pane` all work from within a Claude session. This enables Option B (lazy setup on `/research-resume`) — panes created when we know which question is active, not at SessionStart. Layout: Option A (2-column, Claude left, KNOWLEDGE + WHITEBOARD right). Renderer: glow (snap) with custom style at `.glow-style.json` (removes `#` prefix from headers). Auto-refresh via `watch --color`. Files must be in project directory, not `/tmp` (snap sandboxing). Style file must also be in project dir (snap can't read `~/.config/`). Tested rich-cli as alternative — inferior table rendering, no markdown tables at all. Glow is the winner.

### 2026-03-17: Rollout dependency analysis
Identified the core implementation tension: skills reference document templates, documents reference skills, CLAUDE.md references both. Cannot implement one at a time without intermediate broken states. Requires careful batching — retroactive IDEALOGs first (Phase 0), then skill extensions, then new skills, then hook, then CLAUDE.md rewrite.

### 2026-03-17: NOTEBOOK.md as scratch pad between /reason and /blueprint
During tmux workspace testing, discovered a resolution gap: `/reason` generates high-res technical detail (exact commands, configurations, error messages, tested paths) that `/blueprint` needs for good PLAN.md steps. But IDEALOG is too low-res (narrative) and KNOWLEDGE is too permanent (reference). NOTEBOOK.md is scratch paper — `/reason` dumps freely without formatting pressure, `/blueprint` reads it for detail, wiped clean after implementation. Key difference from dropped WORKLOG: NOTEBOOK is explicitly scratch (wiped, not archived). Also keeps the KNOWLEDGE.md tmux viewer pane clean — raw dumps don't pollute the reference you're looking at mid-session.

### 2026-03-17: Implementation guardrails added to /reason and /blueprint
Caught that `/reason` had no explicit rule preventing implementation. Added hard gates: /reason NEVER implements (no file creation except IDEALOG/WHITEBOARD/NOTEBOOK writes), /blueprint STOPS after writing PLAN.md (waits for user "go"). Also renamed /end-session to /reason-end (counterpart to workspace setup, not generic).

### 2026-03-17: Zellij workspace tested — different commands, same result
Zellij uses different pane commands than tmux: `zellij action new-pane --direction right`, `zellij action write-chars` + `zellij action write 10` (Enter), `zellij action move-focus left/right/up/down`, `zellij action close-pane`. Key differences from tmux: (1) `watch --color` does NOT pass through glow colors in zellij — must use shell loop `while true; do clear; glow ...; sleep N; done` instead. (2) Navigation is trickier — `move-focus right` lands on the nearest right pane, not necessarily the top one. Need `move-focus right` then `move-focus up` to reach top-right. (3) Must `cd` to project dir in each new pane — they don't inherit Claude's cwd. (4) md5sum change detection prevents flicker: only re-render when file content actually changes. (5) `close-pane` closes whichever pane has focus — dangerous if focus is on Claude's pane. Must be careful in `/reason-end`.

### 2026-03-17: tmux workspace design finalized
Layout: Option A (Claude left, KNOWLEDGE top-right, WHITEBOARD bottom-right). Glow with custom `.glow-style.json` as renderer (tested rich-cli — inferior tables). Setup triggered lazily by `/research-resume`. WHITEBOARD.md is ephemeral (gitignored), lives in project root. `/reason` writes to WHITEBOARD.md for visualizations (no separate `/draw` skill). `/save-session` does NOT kill panes. New `/end-session` skill tears down workspace (kill panes, delete WHITEBOARD.md). Separate from `/save-session` because save is a checkpoint, end is a teardown — and sometimes you save mid-design without ending.

### 2026-03-18: Future Work section in README.md for deferred items
"Patterns We're Evaluating" (tmux layout hook, cost tracking, /draw skill, etc.) don't fit KNOWLEDGE.md (not validated findings), IDEALOG.md (gets condensed per session), or Open Questions (they're proposals, not unknowns). Solution: README.md gets a "Future Work" section — persistent, per-question, never purged. Dependency chain: `/reason` (no topic) should read Future Work as possible topics, `/research-new` should include it in template, `/research-resume` should show it in briefing, `/research-complete` should warn about unfinished items.

### 2026-03-18: /blueprint should be iterative, not one-off
Currently `/blueprint` always creates a fresh PLAN.md. Should be recursive: if PLAN.md exists, read it and update (incorporate new IDEALOG decisions, audit findings, Phase 1 learnings) instead of regenerating from scratch. Preserve completed steps (`[x]`), modify open steps, record changes in Mutation Log. This enables: /blueprint v1 → /audit → /reason → /blueprint v2 → implement Phase 1 → /reason → /blueprint v3 (adjust remaining phases). Key behaviors: (1) detect existing PLAN.md, (2) diff IDEALOG.md against what the plan already reflects, (3) only modify open/future steps, (4) log mutations.

### 2026-03-18: README.md gets "Future Work" section for deferred items
Deferred patterns (tmux layout hook, cost tracking, /draw skill) don't fit KNOWLEDGE (not validated), IDEALOG (gets condensed), or Open Questions (they're proposals, not unknowns). README.md is per-question, persistent, never purged — natural home. Dependency chain: `/reason` reads Future Work when no topic given, `/research-new` includes empty section in template, `/research-resume` shows in briefing, `/research-complete` warns about unfinished items.

### 2026-03-19: Quick implement — /blueprint auto-switches bottom pane to PLAN.md
`/blueprint` Step 3b switches bottom tmux pane to show PLAN.md after writing it. `/blueprint-revise` Step 6b does the same after revision. The generated PLAN.md's Final Cleanup section includes a revert step that switches back to WHITEBOARD.md after implementation is complete. The plan carries its own revert instruction — any executing agent will hit it.

### 2026-03-19: Per-question maintenance instructions in KNOWLEDGE.md
`/save-session` Job 2 is generic — it can't have project-specific checks hardcoded (checking skill catalog only matters for research_environment_optimization, not boundary_conduction_speedup). Solution: each KNOWLEDGE.md has a `## Maintenance` section with question-specific verification rules that Job 2 reads naturally during its editorial pass. Self-referential — the rules about maintaining KNOWLEDGE.md live inside KNOWLEDGE.md. No new files, no skill modification needed, `/save-session` Job 2 already reads the full file. Examples: "verify skill catalog matches .claude/skills/", "verify CV values match experiment outputs", "verify parameter tables match optimizer results."

### 2026-03-19: KNOWLEDGE.md needs explicit skill catalog
The skill table in KNOWLEDGE.md (Skill Pipeline > Design) just lists skill names by category. Should be a full reference with: skill name, purpose (one-liner), line count, and workflow category. This way `/save-session` Job 2 can verify the catalog is current, and anyone reading KNOWLEDGE.md can understand what each skill does without opening 19 skill files. Also: `/save-session` Job 2 currently has no specific improvement instructions — just generic "restructure, rewrite, deduplicate." Could accept targeted guidance but adding that complexity is deferred.

### 2026-03-19: Full agentic speedup catalog
Beyond basic parallelism, identified multiple agent-based speedups across the workflow:

**During /reason**: (1) Background research agent — searches codebase for relevant files when a topic comes up, results appear async. (2) Literature scout — greps Research/Knowledge/ and literature/ for prior findings you might have forgotten.

**During /blueprint**: (3) Parallel phase generation — each phase generated by a separate agent. (4) Codebase impact analysis agent — scans affected files while main thread writes plan header.

**During execution**: (5) Background test runner — `/verify` runs in background after each step, notifies on failure. (6) Cleanup agent — de-sloppify runs in background (float64, stray prints, V5.3 check).

**During /save-session**: (7) Parallel jobs — Jobs 1,2,5 are independent, could each be a separate agent.

**New patterns**: (8) Watchdog agent — continuous background monitor for float64 violations and V5.3 modifications after every Edit/Write. (9) Pre-fetch agent — when /reason detects user drilling into a topic, preemptively reads relevant engine files.

Biggest immediate wins: background /save-session (#7), background test runner (#5), pre-fetch during /reason (#9).

### 2026-03-19: Agentic speedup opportunities
Beyond parallel tool calls, background Agent subagents could speed up heavy tasks: (1) `/save-session` as background agent — 5-job editorial pass blocks conversation, could run in background while user keeps working. (2) `/blueprint` codebase analysis — agent reads codebase while main thread reads IDEALOG. (3) `/reason` context loading — parallel tool calls (Read × 3 + Bash × 1) in single message, no agent needed. Biggest win: background `/save-session` — it's the slowest skill and blocks the most.

### 2026-03-19: All document writes across skills could be backgrounded
Audited all skills for main-thread document writes. `/reason` already has background agent instructions (Step 5, 8, 10). Two more candidates: (1) `/save-session` — the heaviest writer (5 jobs touching IDEALOG + KNOWLEDGE + MASTER_KNOWLEDGE_INDEX), could run entirely as a background agent when called from `/reason-end` instead of blocking teardown. (2) `/quicksave` — the whole skill is a single write to IDEALOG, could be spawned as background agent from `/reason` Step 8. Pattern: instead of calling the skill and waiting, spawn it as a background agent and continue. `/reason-end` would spawn `/save-session` in background → immediately proceed to pane teardown → user gets their terminal back faster.

### 2026-03-19: IDEALOG + WHITEBOARD writes could be background agents
During `/reason`, the main thread pauses to write Edit calls to IDEALOG.md and WHITEBOARD.md — interrupts conversational flow. Could spawn background agents for these writes instead: main thread detects transition, spawns agent with content, continues conversation immediately. IDEALOG is append-only during /reason (no conflict), WHITEBOARD is overwrite-only (no conflict). Same pattern for /quicksave — could be a background agent that dumps to IDEALOG while user keeps talking.

### 2026-03-19: /reason initialization is too slow — parallelize startup
Current `/reason` startup does things sequentially: determine topic → read 3 files → set up tmux → present big picture. The file reads and tmux setup are independent and should fire in one turn: Read IDEALOG + Read KNOWLEDGE + Read README + Bash(tmux setup) all as parallel tool calls. The skill says "read in parallel" but needs to be more explicit — instruct Claude to make 4 parallel tool calls in a single turn, not sequential. Also: tmux window rename (`tmux rename-window`) should be part of the same Bash call as pane setup.

### 2026-03-19: tmux window name as per-session question identifier
Can't access Claude Code's `/rename` session name programmatically (no env var, no hook payload, no CLI). But tmux window name works: `/research-resume` renames the window (`tmux rename-window "Research Environment Optimization"`), `/reason` reads it (`tmux display-message -p '#{window_name}'`). Use readable capitalized title — maps back to folder via lowercase + spaces→underscores. Per-window (no conflicts between concurrent sessions), survives compaction (tmux state is external to Claude context). Tested — rename and read both work from Bash tool. `/research-new` should also set the window name when creating a new question. Also discovered: WHITEBOARD.md needed to be per-question (was project root, caused conflicts between concurrent sessions) — fixed in separate commit.

### 2026-03-18: Skill file length correlates with instruction compliance
Long skill files (200+ lines) cause instructions to fade from active context, especially in long conversations. Observed: NOTEBOOK.md routing rules ignored, implementation guardrails violated, per-topic structure guidance not followed by /save-session. The /blueprint skill was already 210 lines and would have hit 250+ with iterative mode — split into /blueprint + /blueprint-revise instead. Rule of thumb: keep skills under 150 lines. If a skill needs more, split it or reference KNOWLEDGE.md for detail instead of inlining.

### 2026-03-18: Comprehensive session — architecture refinements + implementation planning
Key decisions and technical detail from this session:

**KNOWLEDGE.md structural rework**: Diagnosed the 1109-line file as fundamentally broken — ~400 lines of embedded templates (IDEALOG, PLAN.md, MASTER_KNOWLEDGE_INDEX templates pasted as reference), 232-line ECC analysis that could be condensed, 568-line skill designs section mixing active reference with historical evolution. Solution: per-topic structure where each topic has optional-but-ordered subsections: Findings → Design → Reference → Decisions. Topics are self-contained (look up "workspace integration" → everything in one place). No templates (those live in skill files), no evolution history (IDEALOG's job).

**`/save-session` Job 2 needs structural guidance**: Previous runs did surface polish (fix counts, update stale refs) but didn't reorganize because Job 2 has no target structure defined. Will add per-topic skeleton to Job 2 instructions so it maintains the structure on every editorial pass. Key rule: new findings slot into existing topics; if none fits, create a new topic.

**README.md "Future Work" section**: Deferred ideas (tmux layout hook, cost tracking, /draw skill, MASTER_KNOWLEDGE.md for textbook) need a persistent home. KNOWLEDGE.md = validated findings only. IDEALOG.md = gets condensed per session. Open Questions = unknowns, not proposals. README.md is per-question, persistent, never purged. Dependency chain: `/reason` reads Future Work when no topic, `/research-new` adds to template, `/research-resume` shows in briefing, `/research-complete` warns about unfinished items.

**`/blueprint` iterative mode**: Currently always creates fresh PLAN.md. Should detect existing PLAN.md and switch to update mode — preserve completed steps (`[x]`), incorporate new IDEALOG decisions, log mutations. Enables: blueprint v1 → audit → reason → blueprint v2 → implement → reason → blueprint v3 cycle.

**Other skills needing updates**: `/save-session` Job 2 (add structure guidance), `/reason` Step 0 (read README Future Work), `/research-new` (add Future Work to template), `/research-resume` (show Future Work in briefing), `/research-complete` (warn about unfinished Future Work), `/blueprint` (add iterative mode). Plus the KNOWLEDGE.md restructure itself across all 6 active questions.

**Skill guardrail limitation discovered**: Skill instructions fade from active context in long conversations. The routing rules for NOTEBOOK.md (now merged) and the "never implement during /reason" rule both got violated because they only exist in skill files, not in always-loaded CLAUDE.md or MEMORY.md. Open design question: how to make skill behavioral rules persist globally.

**tmux workspace refinements**: Dynamic glow width (`-w $(tput cols)` instead of hardcoded `-w 70`), 50-25-25 pane ratio, shell loop with md5sum universal fix for color rendering in both tmux and zellij, scrolling works via tmux copy mode.

### 2026-03-18: KNOWLEDGE.md per-topic structure decided
Every topic in KNOWLEDGE.md follows optional-but-ordered subsections: Findings → Design → Reference → Decisions. Topics are self-contained — you look up "workspace integration" and find everything in one place. This replaces the flat dump structure that mixed templates, history, and reference material at the same level. `/save-session` Job 2 needs to maintain this structure on editorial passes.

### 2026-03-18: Three issues identified during live testing
1. **KNOWLEDGE.md is disorganized** — ~1000 lines grew organically, no clear hierarchy, mixed historical/current/stale sections. Needs `/save-session` Job 2 (full editorial pass) but we keep skipping it for quick saves.
2. **IDEALOG.md format too low-res** — thread entries are 2-3 line summaries, lacking the technical depth that was supposed to go to NOTEBOOK.md (now merged). Need richer entries without losing scannability. Consider: short summary line + detail underneath.
3. **WHITEBOARD.md needs persistent + variable sections** — top section always shows: Current Focus + To-Do list (persistent across session). Below a divider: Scratch area (diagrams, maps, trade-offs) that `/reason` overwrites freely. Persistent sections never overwritten during session.

### 2026-03-18: Quick implement — dynamic glow width for tmux panes
Changed `-w 70` to `-w $(tput cols)` in `/reason` skill tmux pane setup. Panes are 108 cols wide but glow was rendering at 70, wasting 38 columns. Now captures pane width dynamically at loop start.

### 2026-03-18: NOTEBOOK.md merged back into IDEALOG.md — single log file
The IDEALOG/NOTEBOOK split caused routing friction. The rule "strategic insight → IDEALOG, technical detail → NOTEBOOK" required constant judgment calls that faded from context in long conversations, leading to everything defaulting to IDEALOG anyway. Solution: merge into one file. No routing decision needed — just write everything to IDEALOG.md. `/reason` writes here, `/quicksave` writes here, `/save-session` reads and organizes from here. NOTEBOOK.md dropped as a separate document. IDEALOG.md now holds both the thinking trail AND raw technical findings.

### 2026-03-17: Memory.md role undefined in our architecture
Discovered that MEMORY.md (always loaded every session) is not integrated into our document workflow. Our entire architecture (KNOWLEDGE, IDEALOG, NOTEBOOK, PLAN, WHITEBOARD, MASTER_KNOWLEDGE_INDEX) never reads or writes to it. But it's free always-loaded context — currently holds stale project facts and one feedback entry. Gap: KNOWLEDGE.md is per-question and only loaded on demand. Some cross-cutting facts need to be in every session (API conventions, environment setup, key gotchas). Memory.md could serve as the "always-loaded cheat sheet" but we haven't designed that role. Also raised: skill guardrails (like "never implement during /reason") only apply when the skill is actively invoked — they don't persist as global rules. Memory.md or CLAUDE.md could hold global behavioral rules. Needs further design.

### 2026-03-17: Workspace ownership clarified — /reason owns panes, not /research-resume
`/research-resume` is about documents (loads KNOWLEDGE, IDEALOG, README, presents briefing). `/reason` is the workspace owner — creates tmux panes with glow viewers when active reasoning starts, because that's when you need the side panels. `/reason-end` tears them down. Also: `/reason` with no argument should auto-resume from IDEALOG.md of the most recent `/research-resume` question, not ask for a topic. Decided: tmux only, drop zellij support.

### 2026-03-17: Technical findings (merged from NOTEBOOK.md)

**Zellij pane commands**: `new-pane --direction right`, `write-chars` + `write 10` (Enter), `write 3` (Ctrl-C), `move-focus {direction}`, `close-pane` (focused only — dangerous). Gotcha: `move-focus right` lands on nearest right pane, need `right` then `up` for top-right.

**Glow rendering**: `watch --color` strips glow ANSI codes in both tmux AND zellij. Fix: shell loop with md5sum change detection (`H=""; while true; do N=$(md5sum FILE | cut -d" " -f1); [ "$N" != "$H" ] && clear && glow -s .glow-style.json -w 70 FILE && H=$N; sleep 2; done`). Works because glow sees a real TTY in a shell loop but not in `watch` subprocess.

**tmux pane ratio**: 50-25-25 chosen. At 216x72: Claude=107x72, KNOWLEDGE=108x35, WHITEBOARD=108x36. Commands: `tmux split-window -h -l ${HALF_W} -d` then `tmux split-window -v -t 1 -l ${HALF_H} -d`.

**tmux vs zellij**: tmux wins — pane targeting by index (safe), `capture-pane` for reading, `kill-pane -t N` (safe close). Zellij is focus-based (risky), no capture, needs zjctl for pane ID targeting. Decided: tmux only.

**Multiplexer detection**: `if [ -n "$TMUX" ]; then echo "tmux"; elif [ -n "$ZELLIJ" ]; then echo "zellij"; fi`

**Zellij pane cwd**: New panes do NOT inherit Claude's cwd. Must `cd` explicitly. Snap glow can't access `/tmp` or `~/.config/` — style file must be in project dir.

## Failed Approaches
- **WORKLOG.md as separate document** (2026-03-17) — failed because: tactical session details (failures, snapshots) are part of the thinking trail and belong in IDEALOG.md. A fourth document fragments context without adding value. Three documents is the right number.
- **/aside skill for sidebar conversations** (2026-03-17) — failed because: the disappearing-from-transcript behavior needed for a true sidebar is hardcoded into Claude Code's `/btw` UI overlay and not exposed to the extensibility system. Cannot replicate without Claude Code source changes.
- **Scaffolding research folders early** (2026-03-17) — failed because: premature folder/file creation before the document architecture was settled led to a partial revert. Design first, then scaffold.
- **Implementing during /reason without approval** (2026-03-17) — failed because: jumped from `/reason` discussion straight to writing code (tmux workspace, /end-session skill, glow install) without calling `/blueprint` or getting user approval. The workflow is: `/reason` (discuss) → user says "let's build this" → `/blueprint` (generates PLAN.md) → user approves → implement. Never skip the approval gate.

## Session Log

Pre-IDEALOG history — thinking trail started 2026-03-17.

### 2026-03-17 Session
**Worked on**: Full implementation of the research environment optimization — from design through rollout of all skills, documents, and hooks.
**Accomplished**:
- Designed three-document architecture (KNOWLEDGE + IDEALOG + PLAN) through iterative discussion
- Designed 7 new skills (/reason, /blueprint, /save-session, /audit, /verify, /build-fix, /strategic-compact) with detailed specs
- Designed PreCompact hook
- Ran two audit passes (Opus adversarial review) on PLAN.md — fixed 2 Critical, 5 High, 7 Medium issues
- Executed full PLAN.md rollout: Phase 0 (6 IDEALOGs created via chat log backtrace, 6 KNOWLEDGE.md files audited, MASTER_KNOWLEDGE_INDEX.md created, CLAUDE.md updated), Phase 1 (4 skills extended), Phase 2+2b (7 skills created), Phase 3 (hook + settings.json)
- 6 commits total covering the implementation
- 16 skills operational (9 existing + 7 new), all under 210 lines
**Next**: Test the new skills in real research sessions — try `/reason` on an actual research question, then `/blueprint` to generate a PLAN.md, then `/save-session` at end of day. Update README.md completion criteria to reflect implementation done.

### 2026-03-17 Session (continued)
**Worked on**: Post-implementation testing — tmux/zellij workspace integration, audit fixes, NOTEBOOK.md design, skill guardrails.
**Accomplished**:
- Tested `/save-session` successfully (first real use)
- Tested `/research-resume` successfully (question selection + briefing)
- Tested `/reason` invocation (missing topic handling, with-topic handling)
- Tested tmux pane control from Claude's Bash tool — all commands work (split, send-keys, capture-pane, kill-pane)
- Tested zellij pane control — works but with limitations (focus-based, no pane ID targeting, no capture-pane)
- Installed glow (snap) for markdown rendering in viewer panes
- Created custom `.glow-style.json` (removes # prefix from headers)
- Discovered: `watch --color` works in tmux but NOT in zellij — need shell loop with md5sum change detection
- Discovered: snap glow can't access `/tmp` or `~/.config/` — style file must be in project dir
- Created `/reason-end` skill (renamed from `/end-session`)
- Added implementation guardrails to `/reason` (NEVER implement) and `/blueprint` (STOP after PLAN.md)
- Designed NOTEBOOK.md as scratch pad between `/reason` and `/blueprint` — owned by `/reason`, graduated by `/save-session` Job 6, wiped by `/reason-end`
- Ran full `/audit` — found and fixed 2 Critical + 5 High + 7 Medium + 2 Low issues
- Tested zellij workspace with glow color rendering — shell loop with md5sum works, `watch` doesn't
- Researched zellij vs tmux for pane management — tmux has better native support, zellij needs zjctl for pane ID targeting
**Next**: Decide tmux vs zellij for workspace. Update `/research-resume` and `/reason-end` to support the chosen multiplexer. Consider zjctl if staying with zellij.

### 2026-03-18 Session
**Worked on**: Finalized architecture decisions — tmux only, NOTEBOOK merge, workspace ownership, `/quick-implement` skill, `/reason` auto-resume.
**Accomplished**:
- Decided tmux only (dropped zellij support)
- Clarified workspace ownership: `/reason` creates panes, `/reason-end` kills them (not `/research-resume`)
- Created `/quick-implement` skill (bypasses full planning pipeline for small changes)
- Added `/reason` auto-resume from active question when no argument provided
- Discovered and documented: skill guardrails fade from context in long conversations (CLAUDE.md routing rules don't prevent violations)
- Discovered: NOTEBOOK.md routing friction — the IDEALOG/NOTEBOOK split caused everything to default to IDEALOG anyway
- Merged NOTEBOOK.md into IDEALOG.md — single log file, no routing decisions needed
- Blueprint + audit cycle on the NOTEBOOK merge (0 critical, 3 high, 6 medium, 4 low — all addressed)
- Updated all 10 files (6 skills + CLAUDE.md + .gitignore + KNOWLEDGE.md + IDEALOG.md) to remove 69 NOTEBOOK references
- 18 skills total, document architecture finalized: KNOWLEDGE + IDEALOG + PLAN + WHITEBOARD
**Next**: Test full workflow on a real research question (boundary_conduction_speedup anisotropic study).

### 2026-03-18 Session (continued)
**Worked on**: KNOWLEDGE.md restructure, Future Work system, /blueprint-revise, skill dependency chain, skill length audit.
**Accomplished**:
- Restructured KNOWLEDGE.md from 1109 → 304 lines (73% reduction) with per-topic sections (Findings/Design/Reference/Decisions)
- Added Future Work section to all 6 active README.md files — persistent home for deferred items
- Created `/blueprint-revise` skill (107 lines) — update existing PLAN.md without overwriting
- Updated 5 skills for Future Work dependency chain (save-session, reason, research-new, research-resume, research-complete)
- `/save-session` Job 2 now has per-topic structure guidance
- Documented skill length risk: instructions fade above ~150 lines. 4 skills over 200 lines flagged for future splitting
- Dynamic glow width fix (`-w $(tput cols)` instead of hardcoded `-w 70`)
- Ran 2 blueprint + audit cycles (NOTEBOOK merge plan, architecture refinements plan)
- 19 skills total
**Next**: Test full workflow on real research question. Then tackle Future Work (split oversized skills, WHITEBOARD persistent sections, Memory.md role).

### 2026-03-19 Session (continued)
**Worked on**: plans/ archive system, /blueprint pane switch reliability, background agent writes audit, WHITEBOARD per-question migration.
**Accomplished**:
- Created plans/ archive system: /blueprint Final Cleanup auto-archives completed PLAN.md to plans/{date}_{slug}.md before pane revert. /research-new creates plans/ for new questions. Bootstrapped plans/ in all 8 active question folders.
- Fixed /blueprint pane switch: removed separate Step 3b (got forgotten in long conversations), merged into Step 3 with immediate + background agent safety net.
- Audited all skills for document writes — identified /save-session and /quicksave as candidates for background agent execution (documented in IDEALOG).
- WHITEBOARD.md moved to per-question folder (Research/Active/{question}/WHITEBOARD.md) — no conflicts between concurrent tmux sessions.
- tmux window name integration: /research-resume and /research-new set window name, /reason reads it for auto-resume, /reason-end resets to "claude".
- Parallelized /reason startup: explicit instruction for 4 parallel tool calls in single message.
- Added full 19-skill catalog + Maintenance section to KNOWLEDGE.md.
- Added background agent write instructions to /reason (Step 5, 8, 10).
- Created /blueprint-revise skill for iterative plan updates.
- Multiple blueprint → audit → revise → implement cycles tested successfully.
**Next**: Test full pipeline on real research question. Tackle Future Work items.
