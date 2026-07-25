#!/usr/bin/env python
"""Render a cardiac_core markdown doc into a print-quality PDF.

    python cardiac_core/_build/md_to_pdf.py cardiac_core/API_CHEATSHEET.md [more.md ...]

Markdown -> styled HTML (python-markdown + pygments) -> PDF (Playwright/Chromium, with a
headless google-chrome fallback). Output lands next to the source as `<name>.pdf`.

Same rendering approach as the textbook build (Research/Active/textbook/website/build/html_to_pdf.py).
"""
import re
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

CSS = """
@page {
  size: A4;
  margin: 16mm 14mm 18mm 14mm;
}
:root {
  --ink:      #1a1c1f;
  --muted:    #5b6470;
  --rule:     #dfe3e8;
  --accent:   #a4262c;      /* cardiac red */
  --code-bg:  #f7f8fa;
  --note-bg:  #fbf7ec;
  --note-bar: #d9a441;
}
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Charter", "Bitstream Charter", "Source Serif 4", Georgia, "Times New Roman", serif;
  font-size: 9.6pt;
  line-height: 1.5;
  color: var(--ink);
  margin: 0;
}

/* ---- headings ---------------------------------------------------------- */
h1, h2, h3, h4 {
  font-family: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-weight: 650;
  line-height: 1.22;
  color: #101215;
  break-after: avoid-page;
}
h1 {
  font-size: 20pt; margin: 0 0 2mm;
  padding-bottom: 2.5mm; border-bottom: 2.2pt solid var(--accent);
  letter-spacing: -0.2pt;
}
h2 {
  font-size: 13pt; margin: 7.5mm 0 2.5mm;
  padding-bottom: 1.2mm; border-bottom: 0.6pt solid var(--rule);
}
h3 { font-size: 10.8pt; margin: 5mm 0 1.8mm; color: #22262b; }
h4 { font-size: 9.8pt; margin: 4mm 0 1.5mm; color: var(--muted); text-transform: uppercase;
     letter-spacing: 0.4pt; }
h1 + p, h2 + p { margin-top: 0; }

p { margin: 0 0 2.6mm; orphans: 3; widows: 3; }

/* ---- lists ------------------------------------------------------------- */
ul, ol { margin: 0 0 2.8mm; padding-left: 5.2mm; }
li { margin: 0 0 1.1mm; }
li > ul, li > ol { margin: 1.1mm 0 0; }

/* ---- code -------------------------------------------------------------- */
code, kbd, samp {
  font-family: "JetBrains Mono", "SF Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace;
  font-size: 0.86em;
}
p code, li code, td code, h2 code, h3 code {
  background: var(--code-bg);
  border: 0.4pt solid var(--rule);
  border-radius: 2.5pt;
  padding: 0.3mm 1.1mm;
  color: #8a1f26;
  white-space: nowrap;
}
pre {
  background: var(--code-bg);
  border: 0.5pt solid var(--rule);
  border-left: 2.2pt solid var(--accent);
  border-radius: 3pt;
  padding: 2.6mm 3.2mm;
  margin: 0 0 3.2mm;
  overflow-x: auto;
  break-inside: avoid-page;
  line-height: 1.42;
}
pre code {
  background: none; border: 0; padding: 0; color: inherit;
  white-space: pre-wrap; word-wrap: break-word; font-size: 8.3pt;
}

/* ---- tables ------------------------------------------------------------ */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 3.4mm;
  font-size: 8.7pt;
  break-inside: avoid-page;
}
thead { display: table-header-group; }
th {
  background: #eef1f4;
  text-align: left;
  font-family: "Inter", Helvetica, Arial, sans-serif;
  font-weight: 620;
  font-size: 8.4pt;
  padding: 1.5mm 2mm;
  border-bottom: 1pt solid #c9ced6;
}
td { padding: 1.4mm 2mm; border-bottom: 0.4pt solid var(--rule); vertical-align: top; }
tbody tr:nth-child(even) { background: #fafbfc; }
td code { white-space: normal; }

/* ---- callouts ---------------------------------------------------------- */
blockquote {
  margin: 0 0 3.2mm;
  padding: 2.4mm 3.2mm;
  background: var(--note-bg);
  border-left: 2.4pt solid var(--note-bar);
  border-radius: 0 3pt 3pt 0;
  color: #4a4127;
  break-inside: avoid-page;
}
blockquote p:last-child { margin-bottom: 0; }
blockquote code { background: #fff4d9; }

hr { border: 0; border-top: 0.6pt solid var(--rule); margin: 6mm 0; }
a { color: #275ea8; text-decoration: none; }
strong { font-weight: 680; color: #0d0f12; }

/* keep a heading from being stranded at the foot of a page */
h2, h3 { page-break-after: avoid; }
"""


def render_html(md_path: Path) -> tuple[str, str]:
    text = md_path.read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.+)$", text, re.M)
    title = m.group(1).strip() if m else md_path.stem

    html_body = markdown.markdown(
        text,
        extensions=["fenced_code", "codehilite", "tables", "attr_list", "sane_lists", "admonition"],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": False}},
    )
    pyg = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    return title, (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{CSS}\n{pyg}</style></head>"
        f"<body>{html_body}</body></html>"
    )


def to_pdf(html: str, out: Path, title: str) -> bool:
    """Playwright first; fall back to headless google-chrome."""
    header = (
        "<div style=\"font-family:Inter,Helvetica,Arial,sans-serif;font-size:6.5pt;color:#8a929c;"
        "width:100%;padding:0 14mm;display:flex;justify-content:space-between;\">"
        f"<span>{title}</span><span>cardiac_core</span></div>"
    )
    footer = (
        "<div style=\"font-family:Inter,Helvetica,Arial,sans-serif;font-size:6.5pt;color:#8a929c;"
        "width:100%;padding:0 14mm;text-align:center;\">"
        "<span class='pageNumber'></span> / <span class='totalPages'></span></div>"
    )
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            page.pdf(path=str(out), format="A4", print_background=True,
                     display_header_footer=True,
                     header_template=header, footer_template=footer,
                     margin={"top": "16mm", "bottom": "18mm", "left": "14mm", "right": "14mm"})
            browser.close()
        return True
    except Exception as e:                                    # noqa: BLE001
        print(f"    playwright unavailable ({type(e).__name__}); trying headless chrome", flush=True)

    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise RuntimeError("no PDF backend: playwright failed and no chrome/chromium on PATH")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "doc.html"
        src.write_text(html, encoding="utf-8")
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", f"--print-to-pdf={out}", src.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
    return True


def main(argv):
    if not argv:
        argv = ["cardiac_core/API_CHEATSHEET.md"]
    for a in argv:
        src = Path(a)
        if not src.exists():
            print(f"!! {src} not found"); continue
        out = src.with_suffix(".pdf")
        title, html = render_html(src)
        print(f"--> {src}  ->  {out}")
        to_pdf(html, out, title)
        kb = out.stat().st_size / 1024
        print(f"    done: {out}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1:])
