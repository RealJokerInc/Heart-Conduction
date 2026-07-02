---
name: textbook-compile
description: Build the cardiac computational modeling textbook PDF from the website source via Playwright
---

# Textbook Compile

Build the cardiac computational modeling textbook deliverables from the **website source**.

> ⚠️ **Canonical source changed 2026-07-02.** The book is now authored as split per-chapter files
> under `Research/Active/textbook/website/chapters/*.html`. The former single-file source
> `Bidomain_Textbook.html` is **archived** (`_archive/monolithic_pre-fork_2026-07-02/`) and must not
> be built — it is stale (deleted Part III Schur chapters, only 2 appendices).

> 🚧 **The PDF build is currently BLOCKED.** The build script `html_to_pdf_v3.py` that this skill
> used to invoke is **missing from the repo** (it built the now-archived monolithic file). Until it
> is rebuilt against the website source, there is **no working PDF pipeline**. See "Rebuilding the
> PDF pipeline" below. The multi-page website itself works today with no build step.

## Artifacts

| Artifact | How it's produced | Status |
|----------|-------------------|--------|
| Multi-page site — `website/index.html` + `app.js` + `toc.json` | Loads `chapters/*.html` live in the browser; **no build step** | ✅ Works; reflects edits immediately |
| Bundled whole-book HTML — `Cardiac_Textbook_Website.html` | Generated snapshot: all chapters inlined into `window.__CHAPTERS__` | ⚠️ Regenerate after edits (generator not in repo — snapshot only) |
| PDF — `Cardiac_Computational_Modeling.pdf` | Playwright print of the assembled HTML | 🚧 Blocked — build script missing |

## Preview the site (works now)

```bash
# The interactive site loads chapters/ live; just open it in a browser:
xdg-open Research/Active/textbook/website/index.html
```

## Rebuilding the PDF pipeline (the blocker)

The intended pipeline, adapted to the split source:

```
website/chapters/*.html   →  assemble in toc.json order  →  single print HTML
   →  MathJax (local)  →  Playwright (headless Chrome, wait for JS)  →  PDF
```

To restore it, write a new `html_to_pdf.py` that:
1. Reads `Research/Active/textbook/website/toc.json` for chapter order.
2. Concatenates the `website/chapters/*.html` fragments into one HTML doc with the print
   `<style>`/MathJax `<head>` (crib the head/CSS from `Cardiac_Textbook_Website.html` or
   `website/standalone.html`).
3. Renders with Playwright headless Chrome, waiting for MathJax typeset to finish, then `page.pdf(...)`.
4. Writes `Research/Active/textbook/Cardiac_Computational_Modeling.pdf`.

Alternatively, render the already-bundled `Cardiac_Textbook_Website.html` directly with Playwright
(it renders chapters via JS) — but confirm its print CSS and that JS has settled before printing.

## Verify the output (once a PDF exists)

```bash
conda activate heart-conduction
python -c "
import subprocess
r = subprocess.run(['pdfinfo', 'Research/Active/textbook/Cardiac_Computational_Modeling.pdf'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if 'Pages' in line: print(line)
"
```

Post-build checks:
- **Equation rendering** — no raw LaTeX in the PDF (MathJax typeset completed before print).
- **Page breaks** — chapters start on new pages (`page-break-before` CSS present).
- **Content currency** — spot-check Part III shows Ch 12–15 (NOT the old Schur Ch 16/17) and the
  appendices show A/B/C/D. If the PDF shows the old structure, it was built from the archived file.

## After a successful compile

1. Update `Research/Active/textbook/INDEX.md` page counts if they changed.
2. Add a compile note to `Research/Active/textbook/CHANGELOG.md`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `html_to_pdf_v3.py` not found | Expected — it's gone. Build a new `html_to_pdf.py` (see above). |
| PDF shows "Schur Complement" Ch 16/17 | You built the archived monolithic file. Build from `website/` instead. |
| MathJax not rendering | Ensure the local MathJax `<script>` path resolves in the assembled HTML. |
| Playwright timeout | MathJax on the full book can take 30s+; increase the wait. |
| Missing fonts | `apt install fonts-liberation` or check the Playwright browser install. |
