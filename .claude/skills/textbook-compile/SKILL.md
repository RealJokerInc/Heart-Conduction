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

> ✅ **PDF build restored (2026-07-02).** The build script is now `website/build/html_to_pdf.py` — it
> assembles `website/chapters/*` in `toc.json` order and renders via Playwright headless Chromium after
> MathJax typesets. Requires the `playwright` package + Chromium (already installed in the
> `heart-conduction` env). The old `html_to_pdf_v3.py` (built the archived monolithic file) is gone; do
> not use it.

## Artifacts

| Artifact | How it's produced | Status |
|----------|-------------------|--------|
| Multi-page site — `website/index.html` + `app.js` + `toc.json` | Loads `chapters/*.html` live in the browser; **no build step** | ✅ Works; reflects edits immediately |
| Bundled whole-book HTML — `Cardiac_Textbook_Website.html` | Generated snapshot: all chapters inlined into `window.__CHAPTERS__` | ⚠️ Regenerate after edits (generator not in repo — snapshot only) |
| PDF — `Cardiac_Computational_Modeling.pdf` | `website/build/html_to_pdf.py` → Playwright print | ✅ Working (195 pp, A4) |

## Preview the site (works now)

```bash
# The interactive site loads chapters/ live; just open it in a browser:
xdg-open Research/Active/textbook/website/index.html
```

## Building the PDF

`conda` is not on the non-interactive PATH — use the env python directly:

```bash
PY=/home/norepinephrine/.conda/envs/heart-conduction/bin/python
$PY Research/Active/textbook/website/build/html_to_pdf.py \
   -o Research/Active/textbook/Cardiac_Computational_Modeling.pdf
```

The script reads `website/toc.json` for order, stitches `website/chapters/*.html` (parts +
chapters + appendices) into one print HTML with the MathJax `tex-svg` head + a print stylesheet
(neutralizes the sidebar/topbar chrome, adds page breaks), then renders via Playwright, waiting on a
`window.__mathjax_done__` flag before `page.pdf()`.

One-time dependencies (already installed in the env):

```bash
$PY -m pip install playwright && $PY -m playwright install chromium
```

To just assemble the combined HTML and skip Playwright (verify the source stitches, or produce a
standalone whole-book file), add `--html-only`:

```bash
$PY Research/Active/textbook/website/build/html_to_pdf.py --html-only -o /tmp/book.html
```

Note: the build fetches MathJax from `cdn.jsdelivr.net`, so it needs network access at render time.

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
| `html_to_pdf_v3.py` not found | Expected — replaced by `website/build/html_to_pdf.py`. Use that. |
| `ModuleNotFoundError: playwright` | `$PY -m pip install playwright && $PY -m playwright install chromium` |
| PDF shows "Schur Complement" Ch 16/17 | You built the archived monolithic file. Build from `website/` instead. |
| MathJax not rendering | Ensure the local MathJax `<script>` path resolves in the assembled HTML. |
| Playwright timeout | MathJax on the full book can take 30s+; increase the wait. |
| Missing fonts | `apt install fonts-liberation` or check the Playwright browser install. |
