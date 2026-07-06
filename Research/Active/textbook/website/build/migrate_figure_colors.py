#!/usr/bin/env python3
"""Migrate hardcoded figure colors in chapter SVGs → themeable `--fig-*` tokens.

The figures hardcode ~60 hex values + named colors inside `<svg>`, which forced
the destructive blanket dark-mode override (`svg path{stroke:var(--text-secondary)}`).
This tool moves every figure color into an inline `style="…:var(--fig-*)"` so a
single theme swap recolors every figure correctly — then the blanket override can
be deleted (done separately in style.css). See PLAN.md Step 1.2.

Two modes (run --census first, review the map, then --apply):

  --census : scan all chapters/*.html + appendix-*.html, collect every distinct
             fill/stroke color VALUE (hex AND named colors like `white`) inside
             <svg> spans, propose a token by hue/luminance, and write
             build/figure_color_map.json (authoritative, human-reviewable).

  --apply  : load figure_color_map.json and rewrite each color attribute inside
             <svg> spans to inline style="…:var(--fig-TOKEN)". Idempotent.
             Values mapped to null are left untouched and re-reported.

`var()` is emitted ONLY inside a `style="…"` attribute — never as a presentation
attribute (`fill="var(--x)"` does NOT resolve). `fill/stroke="none"` are ignored.
Operates only within <svg>…</svg> spans (skips any mjx-container just in case).
"""
import argparse
import colorsys
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]      # .../Research/Active/textbook
CH = ROOT / "website" / "chapters"
MAP_PATH = pathlib.Path(__file__).resolve().parent / "figure_color_map.json"

SVG_SPAN = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)
TAG = re.compile(r"<[^>]+>")
COLOR_ATTR = re.compile(r'\s(fill|stroke)="([^"]+)"')
STYLE_ATTR = re.compile(r'\sstyle="([^"]*)"')

SKIP_VALUES = {"none", "transparent", "currentcolor", "inherit"}
NAMED = {  # the few CSS named colors that appear in the figures
    "white": "#ffffff", "black": "#000000", "red": "#ff0000",
    "gray": "#808080", "grey": "#808080",
}


def expand_hex(v: str):
    """#abc → #aabbcc; return normalized #rrggbb or None if not a hex/named color."""
    v = v.strip().lower()
    if v in NAMED:
        v = NAMED[v]
    if not v.startswith("#"):
        return None
    h = v[1:]
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6 or any(c not in "0123456789abcdef" for c in h):
        return None
    return "#" + h


def propose_token(value: str):
    """Classify a color VALUE (hex or named) → a --fig-* token name (without '--')."""
    hx = expand_hex(value)
    if hx is None:
        return None
    r = int(hx[1:3], 16) / 255
    g = int(hx[3:5], 16) / 255
    b = int(hx[5:7], 16) / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)     # h in [0,1], s,v in [0,1]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    hue = h * 360
    # achromatic → neutral scale by luminance
    if s < 0.10:
        if lum > 0.93:
            return "fig-stage"
        if lum > 0.78:
            return "fig-grid"
        if lum > 0.52:
            return "fig-faint"
        if lum > 0.30:
            return "fig-muted"
        return "fig-ink"
    # very light + chromatic → panel tint
    if lum > 0.86 and s < 0.30:
        fam = _hue_family(hue)
        tint = {"crimson": "red", "orange": "red", "amber": "amber", "green": "green",
                "teal": "teal", "blue": "blue", "purple": "purple"}[fam]
        return f"fig-tint-{tint}"
    # otherwise a categorical hue
    return f"fig-{_hue_family(hue)}"


def _hue_family(hue: float) -> str:
    if hue < 12 or hue >= 345:
        return "crimson"
    if hue < 33:
        return "orange"
    if hue < 68:
        return "amber"
    if hue < 165:
        return "green"
    if hue < 200:
        return "teal"
    if hue < 258:
        return "blue"
    if hue < 310:
        return "purple"
    return "crimson"     # 310–345 magenta/pink → crimson family


def _files():
    return sorted(CH.glob("ch*.html")) + sorted(CH.glob("appendix-*.html"))


def census():
    counts, byfile = {}, {}
    for f in _files():
        html = f.read_text(encoding="utf-8")
        for svg in SVG_SPAN.findall(html):
            if "mjx-container" in svg:
                continue
            for _attr, val in COLOR_ATTR.findall(svg):
                v = val.strip().lower()
                if v in SKIP_VALUES or v.startswith(("url(", "var(")):
                    continue
                if expand_hex(v) is None:      # not a color literal we handle
                    continue
                counts[v] = counts.get(v, 0) + 1
                byfile.setdefault(v, set()).add(f.name)
    out = {}
    for v, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        out[v] = {"count": n, "token": propose_token(v),
                  "files": sorted(byfile[v]),
                  "review": v in NAMED or v == "white"}
    MAP_PATH.write_text(json.dumps(out, indent=2))
    print(f"census: {len(out)} distinct color values across {len(_files())} files "
          f"-> {MAP_PATH}")
    nulls = [v for v, m in out.items() if not m["token"]]
    if nulls:
        print(f"  {len(nulls)} UNMAPPED (review): {nulls}")
    print("  review figure_color_map.json, then run --apply")


def _rewrite_tag(tag: str, cmap: dict, unmapped: dict, fname: str) -> str:
    extra = []
    for attr in ("fill", "stroke"):
        m = re.search(rf'\s{attr}="([^"]+)"', tag)
        if not m:
            continue
        raw = m.group(1)
        v = raw.strip().lower()
        if v in SKIP_VALUES or v.startswith(("url(", "var(")) or expand_hex(v) is None:
            continue
        entry = cmap.get(v)
        tok = entry.get("token") if entry else None
        if not tok:
            unmapped.setdefault(v, set()).add(fname)
            continue
        extra.append(f"{attr}:var(--{tok})")
        tag = re.sub(rf'\s{attr}="{re.escape(raw)}"', "", tag, count=1)
    if not extra:
        return tag
    sm = STYLE_ATTR.search(tag)
    if sm:
        merged = (sm.group(1).strip().rstrip(";") + ";" + ";".join(extra)).lstrip(";")
        tag = tag[:sm.start()] + f' style="{merged}"' + tag[sm.end():]
    elif tag.endswith("/>"):
        tag = tag[:-2] + f' style="{";".join(extra)}"/>'
    else:
        tag = tag[:-1] + f' style="{";".join(extra)}">'
    return tag


def apply():
    if not MAP_PATH.exists():
        raise SystemExit("figure_color_map.json missing — run --census first.")
    cmap = json.loads(MAP_PATH.read_text())
    unmapped, changed = {}, 0
    for f in _files():
        html = f.read_text(encoding="utf-8")

        def sub_svg(msvg):
            return TAG.sub(lambda mt: _rewrite_tag(mt.group(0), cmap, unmapped, f.name),
                           msvg.group(0)) if "mjx-container" not in msvg.group(0) else msvg.group(0)

        new = SVG_SPAN.sub(sub_svg, html)
        if new != html:
            f.write_text(new, encoding="utf-8")
            changed += 1
    print(f"apply: rewrote {changed} files")
    if unmapped:
        print("  UNMAPPED (left untouched — resolve in figure_color_map.json):")
        for v, fs in sorted(unmapped.items()):
            print(f"    {v}  ({', '.join(sorted(fs))})")
    else:
        print("  no unmapped color values remain")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.census:
        census()
    elif args.apply:
        apply()
    else:
        ap.error("choose --census or --apply")


if __name__ == "__main__":
    main()
