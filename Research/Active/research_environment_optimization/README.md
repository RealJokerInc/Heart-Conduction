# Research Environment Optimization

## Question
What Claude Code workflow patterns, hooks, agent configurations, and automation techniques from the open-source ecosystem (notably `everything-claude-code`) can we adopt to improve session efficiency, reduce compaction pain, and accelerate research-to-code cycles in this cardiac simulation project?

## Status: Active

## Why It Matters
Our multi-engine cardiac EP project has long-running research sessions that frequently hit context limits. We already have a CLAUDE.md, memory system, and custom skills, but lack hooks for state preservation, strategic compaction, session persistence, and cost tracking. Borrowing proven patterns could significantly reduce wasted context and manual recovery overhead.

## Engines
- All engines (this is a cross-cutting workflow concern, not engine-specific)

## Completion Criteria
- [x] Audit of everything-claude-code patterns against our current setup
- [x] Audit of our own 9 existing skills — identified zero engineering workflow coverage
- [x] Document adopted patterns and rationale in KNOWLEDGE.md
- [x] Design conflict analysis — identified overlaps, then revised based on user feedback
- [x] Three-document architecture finalized: KNOWLEDGE (high-res reference) + IDEALOG (low-res thinking trail) + PLAN (agent steps)
- [x] `/reason` write behavior designed: transition-based writes to IDEALOG, not per-exchange
- [x] Full skill pipeline designed: /reason → /blueprint → /audit → execute
- [x] Extend `/research-new`: create IDEALOG.md with template
- [x] Extend `/research-resume`: read IDEALOG.md, add "What NOT to retry" + "Current Direction" + "Next Step"
- [x] Extend `/research-update`: route by type — finding/analysis→KNOWLEDGE, idea/failure/issue/next-step→IDEALOG
- [x] Extend `/research-complete`: archive IDEALOG.md with question (not promoted)
- [x] Add `/reason` skill: interactive reasoning buddy (189 lines)
- [x] Add `/blueprint` skill: autonomous PLAN.md generator (209 lines)
- [x] Add `/save-session` skill: 5-job cleanup agent (154 lines)
- [x] Add `/audit` skill: adversarial Opus subagent (97 lines)
- [x] Add `/verify` skill: auto-detect engine, run tests (77 lines)
- [x] Add `/build-fix` skill: systematic error resolution (114 lines)
- [x] Add `/strategic-compact` skill: compaction decision guide (74 lines)
- [x] Add PreCompact hook + settings.json
- [x] Update CLAUDE.md with three-document architecture, 16-skill table (247 lines)

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| — | — | — |

## Key Findings So Far

### Source: everything-claude-code (github.com/affaan-m/everything-claude-code)

**Repository overview**: A production plugin ecosystem for Claude Code (50K+ GitHub stars) by Affaan Mustafa. Provides 25 agents, 57 commands, 108 skills, 22 MCP server configs, 16 hooks, and modular rules. Targets general software development teams, not scientific computing.

---

### 1. Overall Philosophy

| Principle | Their Approach | Our Current Approach | Gap |
|-----------|---------------|---------------------|-----|
| **Context as scarce resource** | Aggressive token budgeting: replace MCPs with CLI wrappers, keep files <400 lines, strategic compaction at phase boundaries | We have CLAUDE.md + memory + skills; no token-aware practices | Medium |
| **Agent-first delegation** | 25 purpose-built agents with scoped tools and model selection (Opus for planning, Sonnet for code, Haiku for search) | We use ad-hoc subagents; no formalized agent library | Medium |
| **Verification-driven** | Research -> Plan -> TDD -> Code Review -> Security Review -> Verify -> Commit pipeline | We have incremental testing feedback but no formalized pipeline | Low |
| **Continuous learning** | Hooks capture every tool call, extract "instincts" (atomic behaviors with confidence scores) that evolve into skills | We have memory system; no automated learning from sessions | Low priority |
| **Parallelization** | Git worktrees + tmux for parallel Claude instances; "Cascade Method" for task management | We use single-instance sessions | Low priority |

---

### 2. High-Impact Patterns to Adopt

#### A. PreCompact Hook for State Preservation
**What**: Before context compaction, a hook automatically saves current state (phase, files being worked on, next steps) to a session file.
**Why it matters for us**: Our #1 pain point is compaction during multi-phase engine work. We already have a manual "Compaction Recovery" protocol in CLAUDE.md. A hook could automate the save step.
**Effort**: Low (single hook script)
**Decision**: ADOPT

#### B. Strategic Compaction Skill
**What**: Instead of auto-compaction, compact at logical phase boundaries. Decision guide: "Compact after exploration, before execution" and "Don't compact mid-implementation."
**Why it matters for us**: Matches our workflow perfectly — we do research phases then implementation phases.
**Effort**: Low (skill + guidance in CLAUDE.md)
**Decision**: ADOPT

#### C. Session Persistence Architecture
**What**: `session-start.js` loads previous session context (files modified, phase, last action); `session-end.js` extracts summaries from transcripts. Stored in `.claude/sessions/`.
**Why it matters for us**: When resuming after a break or new conversation, we currently rely on PROGRESS.md + memory. Automated session capture would reduce manual overhead.
**Effort**: Medium (hook scripts + session directory)
**Decision**: EVALUATE — our PROGRESS.md + memory system may already cover this adequately

#### D. Dynamic Context Injection via CLI Aliases
**What**: `alias claude-research='claude --system-prompt "$(cat ~/.claude/contexts/research.md)"'` loads mode-specific behavior without bloating CLAUDE.md.
**Why it matters for us**: We have distinct modes (research literature review, engine implementation, experiment running, optimizer tuning). Each could benefit from different system prompts.
**Effort**: Low (shell aliases + context files)
**Decision**: EVALUATE — only useful if our CLAUDE.md is hitting size limits

#### E. Model Routing for Subagents
**What**: Explicitly choose model per agent: Haiku for exploration/search, Sonnet for coding, Opus for architecture decisions.
**Why it matters for us**: Our Explore agents could use Haiku to save cost; planning agents could use Opus for better reasoning.
**Effort**: Zero (already supported via `model` parameter in Agent tool)
**Decision**: ADOPT — already available, just need to use it deliberately

#### F. Post-Edit Hooks (Format + Type Check)
**What**: After every file edit, automatically run formatter and type checker.
**Why it matters for us**: We don't have linting/formatting automation. For Python, `ruff` format + `mypy` check after edits could catch issues early.
**Effort**: Low (hook scripts)
**Decision**: EVALUATE — may slow down rapid iteration

#### G. Cost Tracking Hook
**What**: After every response, log token usage to `~/.claude/metrics/costs.jsonl` with model-specific pricing.
**Why it matters for us**: Research sessions can be expensive. Knowing cost per session/question helps budget.
**Effort**: Low (single hook script)
**Decision**: ADOPT

---

### 3. Medium-Impact Patterns

#### H. Modular Rules System
**What**: Instead of one large CLAUDE.md, use `~/.claude/rules/{topic}.md` files that are loaded selectively.
**Why it matters for us**: Our CLAUDE.md is already ~200 lines and growing. Splitting into `rules/cardiac-conventions.md`, `rules/engine-workflow.md`, `rules/research-workflow.md` could help.
**Effort**: Medium (restructure + test that rules load correctly)
**Decision**: DEFER — CLAUDE.md isn't at pain point yet

#### I. Doc File Warning Hook
**What**: Block creation of non-standard documentation files outside the established structure.
**Why it matters for us**: We have a strict `Research/` = writing, `Engines/` = code convention. A hook could enforce this.
**Effort**: Low
**Decision**: EVALUATE

#### J. Formalized Agent Library
**What**: Pre-define agents with scoped tools and descriptions. The description field drives automatic delegation.
**Why it matters for us**: We could define `bidomain-expert`, `literature-reviewer`, `experiment-runner` agents with appropriate tool scoping.
**Effort**: Medium
**Decision**: DEFER — our skill system already handles most delegation

---

### 4. Low-Priority / Not Applicable

| Pattern | Why Not Now |
|---------|-------------|
| Continuous learning / instincts | Over-engineered for solo research. Our memory system is sufficient. |
| Multi-instance parallelization | Single-researcher workflow; not worth the complexity. |
| 57 language-specific commands | We only use Python + CUDA. Our existing skills cover this. |
| MCP server marketplace | We already have PubMed, Gmail. No need for 22 MCPs. |
| Supply chain / business skills | Not applicable to cardiac simulation research. |
| Security review agent | Not a production app; simulation code has low attack surface. |
| PR workflow automation | Solo researcher; git workflow is simple. |

---

### 5. Architecture Comparison

```
THEIR STRUCTURE                          OUR STRUCTURE
~/.claude/                               ~/.claude/
  CLAUDE.md (lean, routing)                CLAUDE.md (in project root)
  rules/common/*.md                        projects/{project}/memory/
  rules/{lang}/*.md                        skills/ (custom skills)
  contexts/{dev,review,research}.md
  agents/*.md
  skills/*.md                            CLAUDE.md (project, 200+ lines)
  commands/*.md                          Research/Active/
  hooks/hooks.json                       Engines/
  scripts/hooks/*.js                     Pipelines/
  mcp-configs/mcp-servers.json
  sessions/
  metrics/
```

**Key architectural difference**: They build a general-purpose plugin ecosystem. We have a domain-specific research workflow. Their modularity is valuable; their generality is not.

## Future Work
- SessionStart tmux layout hook (auto-setup on session start)
- `/draw` skill for WHITEBOARD.md visualization
- Cost tracking hook (token usage per session)
- Post-edit ruff formatting hook
- MASTER_KNOWLEDGE.md (textbook-oriented consolidated knowledge)
- Memory.md role in document architecture (always-loaded cheat sheet?)
- Skill guardrail persistence (rules that survive beyond skill invocation)
- IDEALOG.md format improvement (richer entries without losing scannability)
- WHITEBOARD.md persistent sections (Current Focus + To-Do above divider, variable Scratch below)
- Split oversized skills: `/research` (287), `/reason` (254), `/research-new` (221), `/blueprint` (210) — all over 150-line threshold

## Literature
| Paper | Summary | Key Insight |
|-------|---------|-------------|
| — | — | — |
