---
name: textbook-edit
description: Write or revise content in the cardiac computational modeling textbook, following the Feynman-style guide
argument-hint: "[chapter number or section description]"
---

# Textbook Edit

Write or revise content in the cardiac computational modeling textbook.

The user wants to edit: $ARGUMENTS

## Source files

**Canonical source = the split website chapters** (since 2026-07-02). Edit the per-chapter file, NOT any single-file HTML.

- **HTML source**: `Research/Active/textbook/website/chapters/chN.html` — one file per chapter (`ch1.html`…`ch20.html`), plus `appendix-a.html`…`appendix-d.html` and `references.html`. Each is small; open the whole chapter file.
- **Rendered whole-book**: `Research/Active/textbook/Cardiac_Textbook_Website.html` (generated snapshot — do NOT hand-edit).
- **Style guide**: `Research/Active/textbook/STYLE_GUIDE.md`
- **Chapter index**: `Research/Active/textbook/INDEX.md` (section map, equation registry)
- **Edit history**: `Research/Active/textbook/CHANGELOG.md`
- **Audits**: `Research/Active/textbook/audits/` (chapter-by-chapter quality reviews)

> ⚠️ The former single-file source `Bidomain_Textbook.html` is **ARCHIVED** at `Research/Active/textbook/_archive/monolithic_pre-fork_2026-07-02/` — it is stale (old Part III with the deleted Schur/FGMRES chapters, only 2 appendices). Never edit it.

## Before writing anything

1. **Read STYLE_GUIDE.md** — defines the Feynman-style writing approach:
   - 5-layer complexity: ELI5 → Feynman analogy → 3Blue1Brown visual → Worked example → Implementation
   - Every section opens with a motivating question, not equations
   - Math is introduced with plain-English explanation first, then notation
   - Active voice, concise, honest about tradeoffs

2. **Read CHANGELOG.md** — check what's already been done. Don't redo work.

3. **Read INDEX.md** — locate the chapter and its section/equation registry, then open the single chapter file `website/chapters/chN.html` (small — read it whole; no line-offset juggling).

4. **Check relevant audits** — `audits/MONODOMAIN_CHAPTER_AUDIT.md` (Ch 7–11), `audits/LBM_CHAPTER_AUDIT.md` (Ch 18–20), `audits/READER_B_AUDIT.md` (accessibility). `audits/BIDOMAIN_CHAPTER_AUDIT.md` (Ch 12–15) is ⚠️ PRE-13b-rewrite and largely stale — confirm against the current chapter text before acting on it.

## Writing rules

### Content structure
- Each chapter follows: motivation → intuition → math → worked example → implementation notes
- Sections within a chapter: opening question → ELI5 → formal treatment → summary box
- Comparison tables for any "A vs B" discussion (e.g., FEM vs FDM, Strang vs Godunov)

### Equations
- **Numbering**: Chapter N uses equations (N.1), (N.2), etc. Check INDEX.md equation registry for the last used number.
- **Never orphan an equation** — every equation must have a plain-English sentence before it explaining what it means and a sentence after explaining what to notice
- **Use underbrace** for annotating terms in key equations
- **Check for conflicts** after adding new equations — renumber if needed

### Cross-references
- Reference other chapters by number: "as we saw in Chapter 7..."
- Reference equations by number: "from equation (12.3)..."
- When citing research, use the citation key from `Research/INDEX.md` and note the paper title inline

### Editing large sections
Chapter files are small (one chapter each), so most edits use the Edit tool directly. For a full-section rewrite:
1. Write the new content to a temp file (e.g., `/tmp/ch12_sec3.html`)
2. Splice it into the chapter file:
```python
# Read chapter, splice in new content, write back
path = 'Research/Active/textbook/website/chapters/ch12.html'
lines = open(path).readlines()
new_content = open('/tmp/ch12_sec3.html').readlines()
lines[start:end] = new_content
open(path, 'w').writelines(lines)
```
3. This avoids Edit-tool failures on large non-unique blocks

## After editing

1. **Update CHANGELOG.md** — add a dated entry describing what changed, design decisions made, and any issues resolved
2. **Update INDEX.md** — if line numbers shifted, update the chapter line ranges. If equations were added/renumbered, update the equation registry.
3. **Rebuild deliverables** — use `/textbook-compile`. The multi-page site (`website/index.html` + `app.js`) loads `chapters/` live, so edits appear there immediately; the bundled `Cardiac_Textbook_Website.html` and the PDF are generated snapshots that must be rebuilt.
