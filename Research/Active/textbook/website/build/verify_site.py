#!/usr/bin/env python3
"""Visual + print-safety verification harness for the textbook website.

The website has no pytest suite — verification is visual (both themes) plus two
print-safety assertions that protect the Playwright PDF pipeline.

What it does
------------
  1. Serves `website/` on a local port and, for each requested chapter × theme,
     loads `index.html#<chapter>`, forces the theme, waits for the chapter to
     render, screenshots it full-page, and records any console/pageerror events.
  2. Print-safety (guards the PDF build — see PLAN.md Step 1.0 / MED-2):
       (a) `html_to_pdf.assemble_html()` must contain NO reference to the widget
           loader (`figures.js` / `mountFigures`) — the loader must never be
           injected into the print path.
       (b) The assembled print HTML, rendered headless (as the PDF build does),
           must contain ZERO live `canvas.fig-widget` elements.
     (a) is vacuously green until the widget framework lands (Phase 2); both
     guard against a future regression that wires interactivity into print.

Exit non-zero on any console error or either print-safety failure.

Usage
-----
    python website/build/verify_site.py --chapters ch2,ch5,ch10 --themes light,dark --out /tmp/verify
"""
import argparse
import functools
import http.server
import json
import pathlib
import socket
import sys
import tempfile
import threading

HERE = pathlib.Path(__file__).resolve().parent          # website/build
WEBSITE = HERE.parent                                    # website
sys.path.insert(0, str(HERE))
import html_to_pdf  # noqa: E402  (sibling module; __main__-guarded, safe to import)

# console messages matching these substrings are ignored (benign, not site bugs)
BENIGN = ("favicon", "mathjax")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass


def _serve(directory: str, port: int) -> http.server.HTTPServer:
    handler = functools.partial(_QuietHandler, directory=directory)
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _print_safety() -> list:
    """Return a list of failure strings (empty = pass)."""
    fails = []
    src = html_to_pdf.assemble_html()
    for needle in ("figures.js", "mountFigures"):
        if needle in src:
            fails.append(f"print path references widget loader '{needle}' (PDF would mount widgets)")
    # render the assembled print HTML headless; assert no live widget canvas
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError:
        fails.append("playwright not installed — cannot run print-safety render")
        return fails
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tf:
        tf.write(src)
        tmp = tf.name
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + tmp, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_timeout(500)
        n = pg.evaluate("document.querySelectorAll('canvas.fig-widget').length")
        b.close()
    pathlib.Path(tmp).unlink(missing_ok=True)
    if n:
        fails.append(f"assembled print HTML mounted {n} canvas.fig-widget (widget leaked into print)")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chapters", default="ch1,ch2,ch5,ch10",
                    help="comma-separated ids (e.g. ch2,ch5,title,part-i)")
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--out", default="/tmp/verify_site")
    ap.add_argument("--skip-print-check", action="store_true")
    args = ap.parse_args()

    chapters = [c.strip() for c in args.chapters.split(",") if c.strip()]
    themes = [t.strip() for t in args.themes.split(",") if t.strip()]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    port = _free_port()
    httpd = _serve(str(WEBSITE), port)
    base = f"http://127.0.0.1:{port}/index.html"
    summary, any_err = [], False

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for ch in chapters:
                for th in themes:
                    errors = []
                    page = browser.new_page(viewport={"width": 1440, "height": 900},
                                            device_scale_factor=2)
                    page.on("console",
                            lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error"
                            and not any(b in m.text.lower() for b in BENIGN) else None)
                    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
                    page.goto(f"{base}#{ch}", wait_until="networkidle", timeout=60000)
                    page.evaluate(f"() => document.documentElement.setAttribute('data-theme','{th}')")
                    try:
                        page.wait_for_selector(
                            "#chapter-content .chapter, #chapter-content .part-page, "
                            "#chapter-content .title-page", timeout=15000)
                    except Exception:
                        errors.append(f"chapter '{ch}' did not render (no .chapter/.part-page/.title-page)")
                    page.wait_for_timeout(1500)  # settle MathJax + fonts
                    shot = out / f"{ch}_{th}.png"
                    page.screenshot(path=str(shot), full_page=True)
                    page.close()
                    if errors:
                        any_err = True
                    summary.append({"chapter": ch, "theme": th,
                                    "screenshot": str(shot), "errors": errors})
                    tag = "ERR" if errors else "ok "
                    print(f"  [{tag}] {ch:12s} {th:5s} -> {shot.name}"
                          + ("  " + " | ".join(errors) if errors else ""))
            browser.close()
    finally:
        httpd.shutdown()

    print_fails = [] if args.skip_print_check else _print_safety()
    for f in print_fails:
        print(f"  [PRINT-SAFETY FAIL] {f}")
    if not print_fails and not args.skip_print_check:
        print("  [ok ] print-safety: loader not in print path; 0 widget canvases in assembled HTML")

    (out / "summary.json").write_text(json.dumps(
        {"chapters": chapters, "themes": themes, "results": summary,
         "print_safety_failures": print_fails}, indent=2))
    print(f"\n  summary -> {out/'summary.json'}")

    if any_err or print_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
