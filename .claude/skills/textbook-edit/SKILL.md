---
name: textbook-edit
description: Write or revise content in the cardiac computational modeling textbook, following the Feynman-style guide
argument-hint: "[chapter number or section description]"
---

# Textbook Edit

Write or revise content in the cardiac computational modeling textbook.

The user wants to edit: $ARGUMENTS

## Source files
- **HTML source**: `Research/Active/textbook/Bidomain_Textbook.html` (~12,300 lines, single file)
- **Style guide**: `Research/Active/textbook/STYLE_GUIDE.md`
- **Chapter index**: `Research/Active/textbook/INDEX.md` (line numbers, page counts, equation registry)
- **Edit history**: `Research/Active/textbook/CHANGELOG.md`
- **Audits**: `Research/Active/textbook/audits/` (chapter-by-chapter quality reviews)

## Before writing anything

1. **Read STYLE_GUIDE.md** — defines the Feynman-style writing approach:
   - 5-layer complexity: ELI5 → Feynman analogy → 3Blue1Brown visual → Worked example → Implementation
   - Every section opens with a motivating question, not equations
   - Math is introduced with plain-English explanation first, then notation
   - Active voice, concise, honest about tradeoffs

2. **Read CHANGELOG.md** — check what's already been done. Don't redo work.

3. **Read INDEX.md** — find exact line numbers for the chapter you're editing. Jump directly to the relevant section using line offsets — do NOT read the entire 12,300-line HTML file.

4. **Check relevant audits** — if editing chapters 12-15, read `audits/bidomain_chapter_audit.md` for known issues. Chapters 7-11: `monodomain_chapter_audit.md`. LBM chapters: `lbm_chapter_audit.md`.

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
For replacements larger than 50 lines:
1. Write the new content to a temporary file (e.g., `/tmp/chapter_12_section_3.html`)
2. Use a Python splice script to replace the line range:
```python
# Read original, splice in new content, write back
lines = open('Bidomain_Textbook.html').readlines()
new_content = open('/tmp/chapter_12_section_3.html').readlines()
lines[start:end] = new_content
open('Bidomain_Textbook.html', 'w').writelines(lines)
```
3. This avoids Edit tool failures on large non-unique blocks

## After editing

1. **Update CHANGELOG.md** — add a dated entry describing what changed, design decisions made, and any issues resolved
2. **Update INDEX.md** — if line numbers shifted, update the chapter line ranges. If equations were added/renumbered, update the equation registry.
3. **Compile the PDF** — use `/textbook-compile` to build and verify
