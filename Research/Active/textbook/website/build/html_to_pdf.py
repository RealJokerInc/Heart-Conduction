#!/usr/bin/env python3
"""Build the *Cardiac Computational Modeling* PDF from the split website source.

Replaces the missing `html_to_pdf_v3.py`. Assembles `website/chapters/*.html`
in `website/toc.json` order into one print-optimized HTML document, then renders
it to PDF via Playwright (headless Chromium) after MathJax finishes typesetting.

Canonical source is `website/chapters/` — NOT the archived, stale
`_archive/monolithic_pre-fork_2026-07-02/Bidomain_Textbook.html`.

Usage
-----
    # assemble + render PDF (needs Playwright):
    python website/build/html_to_pdf.py -o Cardiac_Computational_Modeling.pdf

    # just assemble the combined HTML (no Playwright needed) — useful to verify
    # the source stitches together and to produce a standalone whole-book file:
    python website/build/html_to_pdf.py --html-only -o Cardiac_Computational_Modeling.html

Requires (for PDF mode): `pip install playwright && playwright install chromium`.
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]   # .../Research/Active/textbook
CH = ROOT / "website" / "chapters"
TOC = ROOT / "website" / "toc.json"
CSS = ROOT / "website" / "style.css"
DEFAULT_OUT = ROOT / "Cardiac_Computational_Modeling.pdf"

# part id -> part-divider fragment
PART_DIVIDER = {
    "part-i": "part-i.html",
    "part-ii": "part-ii.html",
    "part-iii": "part-iii.html",
    "part-iv": "part-iv.html",
    "appendices": "appendices.html",
}

# MathJax config matches the website (tex-svg; $...$ inline, $$...$$ display).
# The startup hook flips a flag once typesetting is done so Playwright can wait.
MATHJAX = """
<script>
window.MathJax = {
  tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']] },
  svg: { fontCache: 'global' },
  startup: {
    ready() {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(function () { window.__mathjax_done__ = true; });
    }
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
"""

# Neutralize the interactive app chrome (sidebar/topbar) for a single-column
# print flow, and add page-break rules.
PRINT_CSS = """
<style>
.topbar, .sidebar, .bottom-nav, .right-margin, #sidebar, #topbar,
nav.sidebar, .sidebar-toggle { display: none !important; }
body { background: #fff; }
.content, main.content {
  margin: 0 auto !important;
  max-width: 46em !important;
  padding: 0 1.5em !important;
}
mjx-container[jax="SVG"] { max-width: 100% !important; }
.figure svg { max-width: 100%; height: auto; }
@media print {
  .part-page  { page-break-before: always; page-break-after: always; }
  .chapter    { page-break-before: always; }
  .title-page { page-break-after: always; }
  h3, h4      { page-break-after: avoid; }
  .figure, .equation-block, table, .insight { page-break-inside: avoid; }
}
</style>
"""


def _read(name: str) -> str:
    return (CH / name).read_text(encoding="utf-8")


def assemble_html() -> str:
    toc = json.loads(TOC.read_text(encoding="utf-8"))
    css = CSS.read_text(encoding="utf-8")

    body = ['<main class="content" id="main-content">']
    if (CH / "title.html").exists():
        body.append(_read("title.html"))

    for part in toc:
        div = PART_DIVIDER.get(part.get("id"))
        if div and (CH / div).exists():
            body.append(_read(div))
        for ch in part.get("chapters", []):
            frag = f"{ch.get('id')}.html"
            if (CH / frag).exists():
                body.append(_read(frag))
            else:
                print(f"  warn: missing fragment {frag}", file=sys.stderr)
    body.append("</main>")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en" data-theme="light">\n<head>\n'
        '<meta charset="UTF-8">\n'
        "<title>Cardiac Computational Modeling</title>\n"
        f"<style>{css}</style>\n{PRINT_CSS}\n{MATHJAX}\n"
        "</head>\n<body>\n" + "".join(body) + "\n</body>\n</html>\n"
    )


def build_pdf(out: pathlib.Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError:
        sys.exit(
            "Playwright is not installed. Run:\n"
            "  pip install playwright && playwright install chromium\n"
            "or use --html-only to just assemble the combined HTML."
        )

    html = assemble_html()
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(html)
        tmp = tf.name
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto("file://" + tmp, wait_until="networkidle", timeout=180000)
            page.wait_for_function("window.__mathjax_done__ === true", timeout=180000)
            page.pdf(
                path=str(out),
                format="A4",
                print_background=True,
                margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
            )
            browser.close()
    finally:
        os.unlink(tmp)
    print(f"  wrote {out} ({os.path.getsize(out) // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--html-only",
        action="store_true",
        help="write the assembled combined HTML and skip PDF rendering",
    )
    args = ap.parse_args()

    if args.html_only:
        out = pathlib.Path(args.output).with_suffix(".html")
        out.write_text(assemble_html(), encoding="utf-8")
        print(f"  wrote {out} ({out.stat().st_size // 1024} KB)")
    else:
        build_pdf(pathlib.Path(args.output))


if __name__ == "__main__":
    main()
