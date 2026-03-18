---
name: research-resume
description: Resume working on a research question. Reads all context (README, KNOWLEDGE, experiments, literature) and picks up where you left off.
argument-hint: "[research question name, or leave blank to choose]"
---

# Resume Research Session

Pick up where you left off on a research question.

Input: $ARGUMENTS

### Project Structure Reference
```
MASTER.md                              — project dashboard
Research/Active/{question}/            — README.md, KNOWLEDGE.md, literature/, papers/, figures/
Research/Knowledge/                     — promoted knowledge files
{Engine}/experiments/{experiment}/      — EXPERIMENT.md (backlinks to question + MASTER.md), run.py, outputs/
Engines/cross_engine/{question}/        — cross-engine experiments
```
Research/ = writing only. Engines/ = code + outputs.

---

## Step 1: Select the Research Question

**If argument provided**: Find the matching folder in `Research/Active/`. Go to Step 2.

**If no argument provided**: You MUST do the following steps in order. Do NOT skip ahead.

1. Use the Glob tool to list all folders in `Research/Active/`
2. For EACH folder found, read the FIRST 10 LINES of its `README.md` to extract the question title and the current status/next step. Read these in parallel.
3. Present the list to the user with the ACTUAL status from the files (do NOT use hardcoded or memorized statuses):

```
Active research questions:

  {folder_name}    — {status extracted from README.md}
  {folder_name}    — {status extracted from README.md}
  ...

Which question do you want to work on?
```

4. **STOP HERE. Ask the user and WAIT for their response.** Do NOT read KNOWLEDGE.md, experiments, literature, or any other files until the user has chosen a question. Reading everything upfront wastes context window on questions the user isn't working on.

5. Once the user responds with their choice, proceed to Step 2.

---

## Step 1b: Set Up tmux Workspace (if in tmux)

Check if running inside tmux (`$TMUX` environment variable). If yes AND only 1 pane exists (no workspace yet), create the research workspace:

```bash
# Only set up if in tmux and no viewer panes exist yet
if [ -n "$TMUX" ] && [ "$(tmux list-panes | wc -l)" -eq 1 ]; then
  # Create right column (35% width)
  tmux split-window -h -l 75 -d
  # Split right column: top for KNOWLEDGE, bottom for WHITEBOARD
  tmux split-window -v -t 1 -l 20 -d
  # Start glow watchers with custom style
  tmux send-keys -t 1 "watch -n 2 -t --color glow -s $(pwd)/.glow-style.json -w 70 Research/Active/{question}/KNOWLEDGE.md" Enter
  tmux send-keys -t 2 "watch -n 1 -t --color glow -s $(pwd)/.glow-style.json -w 70 $(pwd)/WHITEBOARD.md" Enter
fi
```

Replace `{question}` with the actual question folder name selected in Step 1.

If not in tmux, skip this step silently. If panes already exist (workspace already set up), skip to avoid duplicating.

---

## Step 2: Read Context for the Chosen Question

Read these files (use parallel reads):

1. **`Research/Active/{question}/README.md`** — question, status, completion criteria, sub-questions, experiment table, **Engine References**
2. **`Research/Active/{question}/KNOWLEDGE.md`** — current understanding, key decisions, open questions
3. **`Research/Active/{question}/IDEALOG.md`** — current direction, next step, failed approaches

Do NOT read MASTER.md yet — only read what's needed for this question.

## Step 3: Read Engine References

The README.md contains an **Engine References** table listing specific engine files relevant to this question (source code, PROGRESS.md, tests, related knowledge files). Read the key ones — especially:
- Any PROGRESS.md files (to check if engine status has changed)
- Any KNOWLEDGE.md files from connected research questions
- Specific source files only if the user's intended work requires understanding the implementation

Do NOT read all listed files upfront. Read PROGRESS.md and connected KNOWLEDGE.md files now; read source code files on-demand when the user starts working on something specific.

## Step 4: Scan Experiments

Check the experiment table in README.md. For each listed experiment, read its `EXPERIMENT.md` to get the current status (Created / Complete / Inconclusive).

Only read EXPERIMENT.md files that are listed in the README's experiment table.

## Step 5: Scan Literature

Count files in `Research/Active/{question}/literature/`. Check whether KNOWLEDGE.md references their findings — papers filed but not synthesized into KNOWLEDGE.md is a gap to flag.

## Step 6: Present Session Brief

```
=== RESUMING: {Question Title} ===

Status: {from README}

Current understanding:
  {2-3 sentence summary from KNOWLEDGE.md}

Current direction:
  {from IDEALOG.md Current Direction section}

Next step:
  {from IDEALOG.md Next Step section}

What NOT to retry:
  {from IDEALOG.md Failed Approaches section — list each failed approach}

Completion criteria:
  [x] done items
  [ ] remaining items    ← these are what we can work on

Open questions:
  - {from KNOWLEDGE.md}

Experiments:
  {name} — {status} — {engine}

Literature: {N} papers, {M} not yet synthesized into KNOWLEDGE.md

What would you like to work on?
```

If IDEALOG.md doesn't exist (pre-migration question) or contains only template placeholder text (curly-brace patterns like `{What we're currently pursuing}`), skip the Current Direction, Next Step, and What NOT to retry sections gracefully — don't display raw template text.

---

## Step 7: Work

Based on the user's response:

- **Run an experiment**: Create experiment folder in the relevant engine's `experiments/`, write `EXPERIMENT.md` with backlinks (research question README + MASTER.md), write `run.py`, execute, capture results. Add row to research question README experiment table.
- **Find papers**: Use the `/research` skill (PubMed search → screen → acquire → summarize → file into this question's `literature/` and `papers/`)
- **Analyze results**: Read experiment outputs, update KNOWLEDGE.md
- **Write up findings**: Update KNOWLEDGE.md with new synthesis
- **Work on a sub-question**: Read its README.md, narrow focus
- **Start a new sub-question**: Create sub-question folder with README.md under the parent

## Step 8: Session Wrap-Up

Before ending the session (or when the user switches topics), update:

1. **KNOWLEDGE.md** — add any new findings or decisions from this session
2. **README.md** — update completion criteria checkboxes if any were met
3. **MASTER.md** — update the question's row if status changed
4. **Experiment table** — add any new experiments that were created or completed

---

## Rules

- **Read KNOWLEDGE.md before starting work.** Don't re-derive what's already known.
- **Update KNOWLEDGE.md after significant findings.** This is the running synthesis — it should reflect the latest state at session end.
- **Don't read everything upfront.** Only read the selected question's files. Don't read engine codebases, other questions, or MASTER.md unless needed.
- **Experiments live in engine folders, not Research/.** Research/ is writing only.
- **Every experiment needs backlinks.** EXPERIMENT.md must link to its research question and MASTER.md.
- **Don't present the brief until you've read the context.** The brief should reflect actual file contents, not guesses.
