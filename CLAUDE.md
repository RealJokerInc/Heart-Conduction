# Project Instructions for Claude Code

## Virtual Environment

Use the project venv at `./venv/` for Python execution:
```bash
./venv/bin/python script.py
```

## Permission Handling

When running Bash commands that require user approval:
- **DO NOT** request permissions for complete inline commands with embedded code
- Commands should use pattern-based permissions ending with `:*`
- Example: Use `Bash(python3:*)` not `Bash(python3 -c "import sys...")`

If a command requires specific inline code, prefer:
1. Writing the code to a temporary file first
2. Then running that file with a pattern-friendly command

## Plan Mode Usage

When using plan mode:
- Plan files should contain **structured implementation steps only**
- Never write conversation transcripts, tool outputs, or raw session data to plan files
- Plans should be concise markdown with clear action items

---

## Textbook Writing (Bidomain Textbook)

When editing or creating content for `Research/Bidomain/bidomain_textbook.html`:

1. **Read `Research/Bidomain/STYLE_GUIDE.md` first** — it defines the "Feynman style" writing approach
2. **Read `Research/Bidomain/CHANGELOG.md`** — tracks all major edits with dates, prevents re-doing work
3. **Read `Research/Bidomain/INDEX.md`** — document structure with line numbers, page counts, status per chapter
4. **Pipeline**: HTML + local MathJax + Playwright → PDF via `html_to_pdf_v3.py` (in session root)
5. **After edits**: Always regenerate PDF, update CHANGELOG.md and INDEX.md
6. **Equation numbering**: Chapter N uses equations (N.x). Check for conflicts after adding equations.
7. **Splice scripts**: For large content replacements (>50 lines), write content to a temp file and use a Python splice script rather than Edit tool.

---

## V5.4 Implementation Workflow

This project is a multi-phase engine rewrite (7 phases, 70+ validation tests). Sessions will frequently hit compaction. These rules ensure continuity.

### Session Startup — Orientation Protocol

**Every session (new or resumed after compaction), before doing any work:**

1. Read `Monodomain/Engine_V5.4/PROGRESS.md` — tells you current phase, what's done, what's in-progress, what's next
2. Read the relevant IMPLEMENTATION.md section for the current phase (don't re-read the whole file)
3. If implementing a specific file, read its ABC from `improvement.md` at the line number listed in PROGRESS.md
4. Check MEMORY.md for any notes about gotchas or prior failures

**Do NOT re-read entire documents speculatively.** Use the line numbers in PROGRESS.md and MEMORY.md to jump to exactly what you need.

### Compaction Recovery

When a conversation is compacted (you see "continued from a previous conversation"):

1. **Stop.** Do not continue blindly — the summary may be lossy.
2. Run the orientation protocol above (read PROGRESS.md first).
3. If you were mid-file when compaction hit, re-read that file to verify your in-progress work.
4. If the user says "continue", pick up from the next incomplete task in PROGRESS.md.
5. Do NOT re-do work that PROGRESS.md marks as done.

### Task Management

- **Use TaskCreate** at the start of each phase to track that phase's action items
- **Mark tasks in_progress** before starting work on them
- **Mark tasks completed** only after validation passes
- After completing a task, **update PROGRESS.md** immediately — this is your checkpoint
- If a session ends mid-work, the user should be able to start a new session and you'll know exactly where to resume from PROGRESS.md

### Progress Tracking (PROGRESS.md)

`Monodomain/Engine_V5.4/PROGRESS.md` is the single source of truth for implementation state. Update it:

- When a phase starts (mark `IN PROGRESS`)
- When each file is completed (add to done list with date)
- When validation tests pass/fail (record results)
- When a phase completes (mark `DONE`, move to next)

### Implementation Rules

1. **Always check the ABC first.** Before implementing any file, read its abstract base class from `improvement.md` (line numbers in PROGRESS.md). The ABC defines the interface contract — do not deviate.

2. **Always check research references.** Each file has a primary research doc listed in `IMPLEMENTATION.md § Summary of Key Research References`. Read the relevant lines before implementing. Don't guess algorithms.

3. **V5.3 is ground truth for migrated code.** When migrating code from V5.3 (FEM, PCG, CN, BDF, ionic models), the V5.4 version MUST produce bitwise-identical output for the same inputs. Run V5.3 tests through the new interface as the first validation step.

4. **New code validates against IMPLEMENTATION.md criteria.** Each phase has a validation table (e.g., `2-V3: FDM Laplacian convergence`). Follow those criteria exactly.

5. **One file at a time.** Implement → validate → commit → update PROGRESS.md → next file. Don't batch multiple files without intermediate validation.

6. **Explicit before implicit, simple before complex.** Within a phase, implement the simplest variant first (e.g., ForwardEuler before CrankNicolson, BGK before MRT, isotropic before anisotropic). Each simpler variant serves as a reference for the next.

### Cross-Referencing (Don't Memorize — Look Up)

The project has extensive documentation. Use it instead of relying on context:

| What you need | Where to find it |
|---------------|-----------------|
| Current progress | `Engine_V5.4/PROGRESS.md` |
| Phase plan, validation criteria | `Engine_V5.4/IMPLEMENTATION.md` |
| ABC interfaces, design decisions | `Engine_V5.4/improvement.md` (use line numbers from PROGRESS.md) |
| High-level architecture | `Engine_V5.4/README.md` |
| Algorithm details, code snippets | `Research/openCARP_FDM_FVM/01-04_*.md` |
| Reference implementations | `Research/code_examples/` |
| V5.3 validated code | `Engine_V5.3/` (source) and `Engine_V5.3/IMPLEMENTATION.md` (docs) |

### What NOT To Do

- **Don't re-read improvement.md in full** — it's 1750 lines. Use line-number jumps.
- **Don't implement without reading the ABC** — you'll drift from the spec.
- **Don't skip validation** — every file gets tested before moving on.
- **Don't modify V5.3** — it's the validated baseline. All work happens in V5.4.
- **Don't write code from memory after compaction** — re-read the source first.
- **Don't create new architectural patterns** — the architecture is fully specified in improvement.md. If something seems missing, ask the user before inventing.
