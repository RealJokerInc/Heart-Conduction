---
name: research
description: Full research pipeline — discover, screen, acquire, summarize, and file papers into the Research knowledge base. Use when the user wants to find papers, summarize a PDF, or build literature on a topic.
argument-hint: "[topic, PMID, DOI, citation, or PDF path]"
---

# Research Pipeline

Full 5-stage protocol for ingesting research papers into the knowledge base.

Input: $ARGUMENTS

### Project Structure Reference
```
Research/Active/{question}/literature/  — paper summaries (.md) filed here
Research/Active/{question}/papers/      — PDFs filed here
Research/Active/{question}/KNOWLEDGE.md — update after every paper
MASTER.md                              — project dashboard
```
Papers and summaries go in the question's own folders, NOT in a flat Research/papers/ directory.

---

## Stage 1: DISCOVER

Determine entry point from what the user provided:

**A) Topic or research question:**
```
→ mcp__claude_ai_PubMed__search_articles(query=..., max_results=10, sort="relevance")
→ Collect PMIDs
```

**B) DOI (e.g., "10.1038/..."):**
```
→ mcp__claude_ai_PubMed__convert_article_ids(ids=[...], id_type="doi")
→ Get PMID
```

**C) Partial citation (e.g., "Kleber 1987 Circulation"):**
```
→ mcp__claude_ai_PubMed__lookup_article_by_citation(citations=[{author, journal, year}])
→ Get PMID
```

**D) PMID (e.g., "35486828"):**
```
→ Skip to Stage 2
```

**E) Local PDF path (e.g., "~/Downloads/paper.pdf"):**
```
→ Skip to Stage 4
```

**Optional expansion:** After finding a seed paper, offer to run:
```
mcp__claude_ai_PubMed__find_related_articles(pmids=[...], link_type="pubmed_pubmed", max_results=10)
```
to snowball and find related work.

---

## Stage 2: SCREEN

Batch-fetch metadata for all candidate PMIDs:
```
mcp__claude_ai_PubMed__get_article_metadata(pmids=[...])
```

Present results as a table:

| # | Citation Key | Title | Journal (Year) | Relevance |
|---|-------------|-------|----------------|-----------|
| 1 | smith_2023_... | ... | Nature (2023) | High — directly addresses Q2 solver convergence |
| 2 | ... | ... | ... | Medium — tangential |

For each, note:
- Which Q-folder(s) it maps to
- Whether it's likely to change our understanding or just confirm existing knowledge

Ask the user which papers to proceed with, or recommend the top candidates.

---

## Stage 3: ACQUIRE

For each selected paper:

### 3a. Check full-text availability
```
mcp__claude_ai_PubMed__convert_article_ids(ids=[PMID], id_type="pmid")
→ Check if PMCID exists in response
```

### 3b. If PMCID exists (open access full text):
```
mcp__claude_ai_PubMed__get_full_text_article(pmc_ids=[PMCID])
→ Structured full text — use this for summarization (better than PDF for Claude)
```

### 3c. If no PMCID:
```
mcp__claude_ai_PubMed__get_copyright_status(pmids=[PMID])
→ Check if open access
```
- If open access: tell user the DOI and suggest downloading from publisher
- If paywalled: tell user the DOI for institutional download
- In both cases: ask user to provide the PDF path once downloaded

### 3d. Rename the PDF
When a PDF is available (user-provided or downloaded), rename it to the paper's actual title:

```python
import re, shutil

# Sanitize title for filename: lowercase, replace spaces with underscores, remove special chars
def title_to_filename(title):
    clean = re.sub(r'[^\w\s-]', '', title.lower())
    clean = re.sub(r'\s+', '_', clean.strip())
    return clean[:80]  # cap length

# Example: "A Novel Method for Cardiac EP" → "a_novel_method_for_cardiac_ep.pdf"
new_name = f"{title_to_filename(title)}.pdf"
shutil.copy(original_path, f"Research/Active/{question}/papers/{new_name}")
```

Also create a citation-key symlink or note in the summary for lookup:
- File stored as: `Research/Active/{question}/papers/{sanitized_title}.pdf`
- Citation key: `{firstauthor}_{year}_{slug}` (used in MASTER.md and summary frontmatter)

---

## Stage 4: SUMMARIZE

Read the content:
- **PMC full text** (from Stage 3b): already in memory, use directly
- **Local PDF**: use Read tool with `pages` parameter (pages 1-5 first for abstract/methods, then results/discussion)

### 4a. Generate citation key
Format: `{first_author_surname}_{year}_{2-3_word_slug}`
Examples: `rapaka_2012_lbm_ep`, `plank_2007_bidomain_solvers`

### 4b. Write the summary

Create a markdown file with this template:

```markdown
---
paper: {citation_key}
title: "{full title}"
authors: "{first 3 authors + et al.}"
year: {year}
journal: "{journal name}"
doi: "{DOI}"
pmid: "{PMID}"
pdf: ../papers/{sanitized_title}.pdf
questions: [{Q1, Q4, etc.}]
---

## Key Findings
- {3-5 bullet points — what did they discover/demonstrate?}

## Method
- {Numerical method, model, domain, mesh, parameters}
- {Key experimental or computational conditions}

## Key Equations / Results
- {Equations we'd reference during implementation — LaTeX notation}
- {Numerical results: convergence rates, CV values, timing benchmarks}

## Connections to Our Models

### Relevant Engine Components
- {Which engine(s) this connects to: Bidomain V1, Monodomain V5.4, LBM V1}
- {Which specific files/modules implement what the paper describes}
  - e.g., "Their AMG preconditioner approach is implemented in `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/linear_solver/pcg.py`"
  - e.g., "Their LBM collision operator matches our `Monodomain/LBM_V1/src/collision/bgk.py`"

### Agreements
- {Where our implementation aligns with the paper's approach}
- {Validation: does their reported CV/convergence/timing match ours?}

### Disagreements or Gaps
- {Where our implementation differs and why}
- {Things the paper does that we don't (and whether we should)}
- {Things we do differently and the justification}

### Actionable Insights
- {Concrete things we could change or add based on this paper}
- {Parameter values to try, algorithmic improvements, validation targets}
- {Priority: high/medium/low for each insight}

## Limitations / Caveats
- {What the paper doesn't cover or gets wrong}
- {Assumptions that don't hold for our use case}
```

---

## Stage 5: FILE & INDEX

### 5a. Identify the target research question

**Default to the current working question if one is active in this conversation.** If a `/research-resume` session was started earlier in this conversation, file the paper into that question without asking.

Otherwise, determine the target by:
1. If the paper's topic clearly maps to one Active question, use it
2. If it maps to a sub-question, file in the parent's `literature/`
3. If ambiguous or no match, ask the user — they may want to `/research-new` first

Research questions are organized by status under `Research/`:
```
Research/Active/{question}/       # Currently being investigated
Research/Complete/{question}/     # Fully answered
Research/Backlog/{question}/      # Literature gathered, no active work
```

### 5b. File the summary
Place the summary .md in the question's `literature/` folder:
```
Research/Active/{question}/literature/{citation_key}.md
```

### 5c. File the PDF
Place the PDF in the question's `papers/` folder:
```
Research/Active/{question}/papers/{sanitized_title}.pdf
```

If the user provided a path outside the repo, copy it in:
```bash
cp "{original_path}" "Research/Active/{question}/papers/{sanitized_title}.pdf"
```

### 5d. Update MASTER.md (if significant)
If this paper is for a question not yet listed in MASTER.md, or if the paper changes the question's status or next step, update the relevant row in `MASTER.md`.

### 5e. Update question README.md
Add a row to the Literature table in the question's `README.md`:
```markdown
| [{citation_key}](literature/{citation_key}.md) | [PDF](papers/{sanitized_title}.pdf) | {key insight} |
```

If the paper materially changes the answer to the question:
- Update the "Key Findings So Far" section
- Note in the summary's "Actionable Insights" what changed

### 5f. Update KNOWLEDGE.md
**This is critical.** After filing a paper, check the question's `KNOWLEDGE.md`:
- Does the paper confirm, contradict, or extend the current understanding?
- If yes, update the relevant section of KNOWLEDGE.md immediately
- If the paper introduces a new sub-topic, suggest creating a sub-question via `/research-new`

### 5g. Report to user
Output:
- Citation key and title
- Which question folder it was filed in
- The "Actionable Insights" section (most valuable part)
- Whether KNOWLEDGE.md was updated (and what changed)
- Whether any existing code should be revisited based on findings

---

## Batch Mode

When processing multiple papers (e.g., from a topic search):
1. Run Stages 1-2 once to get the candidate list
2. Run Stages 3-5 for each selected paper
3. After all papers are processed, do a single pass to:
   - Check for cross-references between the new papers
   - Update question README.md with any synthesized insights
   - Note any contradictions between papers
   - **Update KNOWLEDGE.md** with the combined findings from all new papers

---

## Rules

- **Always cite PubMed and include DOIs** when using PubMed tools (required by the API)
- **PubMed only covers biomedical literature.** For papers in pure CS/math/physics journals, the user must provide the PDF directly (enter at Stage 4)
- **Rename every PDF** to its sanitized title — never leave publisher filenames like `12859_2023_Article_5513.pdf`
- **Always fill the "Connections to Our Models" section** — this is the most valuable part. A summary without connections to our codebase is just a book report.
- **Check if the paper is already in the question's literature/ folder** before summarizing — don't create duplicates
- **Read paper summaries before PDFs** when cross-referencing — structured summaries are faster to scan
- **File into question folders, not flat directories.** Papers go in `{question}/papers/`, summaries in `{question}/literature/`. The flat `Research/papers/` directory is deprecated.
- **Always update KNOWLEDGE.md after filing.** A paper filed but not synthesized into knowledge is only half-processed.
