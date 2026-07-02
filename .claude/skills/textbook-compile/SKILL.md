---
name: textbook-compile
description: Build the cardiac computational modeling textbook PDF from HTML source via Playwright
---

# Textbook Compile

Build the cardiac computational modeling textbook from HTML source to PDF.

## Pipeline

```
Bidomain_Textbook.html  →  MathJax (local)  →  Playwright (headless Chrome)  →  PDF
```

The build script is `html_to_pdf_v3.py` located in the project root or Research/Active/textbook/.

## Steps

### 1. Locate the build script
```bash
find /home/norepinephrine/Documents/Heart-Conduction -name "html_to_pdf_v3.py" -maxdepth 3
```

### 2. Activate the environment
```bash
conda activate heart-conduction
```

### 3. Run the build
```bash
python html_to_pdf_v3.py Research/Active/textbook/Bidomain_Textbook.html
```

If the script takes an output path argument:
```bash
python html_to_pdf_v3.py Research/Active/textbook/Bidomain_Textbook.html -o Research/Active/textbook/Cardiac_Computational_Modeling.pdf
```

### 4. Verify the output
- Check that the PDF was created and has a reasonable file size
- If possible, report the page count:
```bash
python -c "
import subprocess
result = subprocess.run(['pdfinfo', 'Research/Active/textbook/Cardiac_Computational_Modeling.pdf'], capture_output=True, text=True)
for line in result.stdout.splitlines():
    if 'Pages' in line:
        print(line)
"
```
- If `pdfinfo` is not available, use Python:
```bash
python -c "
from PyPDF2 import PdfReader
r = PdfReader('Research/Active/textbook/Cardiac_Computational_Modeling.pdf')
print(f'Pages: {len(r.pages)}')
"
```

### 5. Post-compile checks
- **Equation rendering**: Spot-check that MathJax rendered correctly (no raw LaTeX in PDF)
- **Page breaks**: Verify chapters start on new pages
- **Table of contents**: Check that page numbers are correct if TOC is auto-generated

### 6. Update metadata
After a successful compile:
1. Update `Research/Active/textbook/INDEX.md` with new page counts if they changed
2. Add a compile note to `Research/Active/textbook/CHANGELOG.md` if this follows content edits:
   ```
   ### {date} — Compile
   - Rebuilt PDF after {description of recent edits}
   - Pages: {N} (was {M})
   ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| MathJax not rendering | Check that local MathJax path in HTML `<script>` tag is correct |
| Playwright timeout | Increase timeout in html_to_pdf_v3.py; MathJax on large docs can take 30s+ |
| Missing fonts | Install via `apt install fonts-liberation` or check Playwright browser install |
| Script not found | Check both project root and Research/Active/textbook/ for html_to_pdf_v3.py |
| Blank pages | Check CSS `page-break-before` rules in the HTML `<style>` block |
