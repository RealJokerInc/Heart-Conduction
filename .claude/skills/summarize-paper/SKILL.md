---
name: summarize-paper
description: Quick-summarize a single paper from a local PDF. For full discovery+screening+acquisition pipeline, use /research instead.
argument-hint: "[pdf-path]"
---

# Summarize Paper (Quick)

Summarize a single local PDF into the research knowledge base. This is Stages 4-5 of the full `/research` pipeline — use `/research` instead if you need to search PubMed, screen candidates, or acquire papers.

Input PDF: $ARGUMENTS

### Project Structure Reference
```
Research/Active/{question}/literature/  — summary filed here
Research/Active/{question}/papers/      — PDF copied here
Research/Active/{question}/KNOWLEDGE.md — update after summarizing
```

## Steps

### 1. Read the PDF
Use the Read tool with `pages` parameter:
- Pages 1-5 first (abstract, intro, methods)
- Then results/discussion pages as needed
- Max 20 pages per Read call

### 2. Extract metadata
- Title, authors, year, journal, DOI
- Generate citation key: `{first_author_surname}_{year}_{2-3_word_slug}`

### 3. Identify target research question
**Default to the current working question** if a `/research-resume` session is active in this conversation. Otherwise, ask the user which active research question this paper belongs to. List the active questions if unclear.

### 4. Rename and file the PDF
Rename to sanitized title and copy to the question's `papers/` folder:
```python
import re, shutil
clean = re.sub(r'[^\w\s-]', '', title.lower())
clean = re.sub(r'\s+', '_', clean.strip())[:80]
shutil.copy(original_path, f"Research/Active/{question}/papers/{clean}.pdf")
```

### 5. Write the summary
Create `Research/Active/{question}/literature/{citation_key}.md` using the full template from `/research` Stage 4b — including the **Connections to Our Models** section (agreements, disagreements, actionable insights).

### 6. Update question files
- Add row to question's `README.md` Literature table
- **Update KNOWLEDGE.md** if the paper changes the current understanding

### 7. Report
- Citation key, question folder, and the Actionable Insights section
