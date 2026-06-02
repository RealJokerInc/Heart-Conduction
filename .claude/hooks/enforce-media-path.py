#!/usr/bin/env python3
"""PreToolUse hook — enforce the media/ convention for new images & videos.

Blocks Write/Edit/NotebookEdit (and obvious literal Bash savefig/imwrite/cp/mv)
that would create an image or video OUTSIDE the allowed locations. See CLAUDE.md
"Saving images & videos".

Mechanism: exit code 2 + stderr = block the tool call and tell Claude why.
Fails OPEN (exit 0) on any parse error so it can never wedge a session.

Limitation: a path computed at runtime inside a Python script (e.g. via
`media_path(...)` or a variable) is invisible here — the hook only sees the tool
call, not the script's internals. It catches direct file writes and literal paths.
"""
import json
import re
import sys

IMAGE_EXT = ("png", "jpg", "jpeg", "gif", "svg")
VIDEO_EXT = ("mp4", "webm", "mov", "avi")
MEDIA_EXT = IMAGE_EXT + VIDEO_EXT

# locations where image/video files may legitimately live (repo-relative, lowercased)
ALLOWED_PREFIXES = (
    "media/",                      # the canonical home
    "research/code_examples/",     # vendored external repos
    "builder/",                    # input / mesh assets read by scripts
    "monodomain/_archive/",        # frozen legacy figures
    "surrogate/docs/diagrams/",    # diagram sources + renders
    ".git/",
)
ALLOWED_SUBSTR = ("/tests/", "/test_", "/__pycache__/")  # regenerable test outputs


def is_blocked_path(path: str) -> bool:
    """True if `path` is an image/video at a non-allowed location."""
    if not path:
        return False
    p = path.strip().strip('"').strip("'").replace("\\", "/")
    ext = p.rsplit(".", 1)[-1].lower() if "." in p else ""
    if ext not in MEDIA_EXT:
        return False
    low = p.lower()
    if low.startswith("/tmp/") or low.startswith("/var/tmp/"):
        return False  # scratch
    if "/heart-conduction/" in low:                 # strip absolute repo prefix
        low = low.split("/heart-conduction/", 1)[1]
    low = low.lstrip("./")
    if any(low.startswith(a) for a in ALLOWED_PREFIXES):
        return False
    if any(s in ("/" + low) for s in ALLOWED_SUBSTR):
        return False
    return True


def candidate_paths(tool_name: str, ti: dict):
    if tool_name in ("Write", "Edit", "NotebookEdit"):
        fp = ti.get("file_path") or ti.get("notebook_path")
        if fp:
            yield fp
    elif tool_name == "Bash":
        cmd = ti.get("command", "") or ""
        ext = "|".join(MEDIA_EXT)
        # 1) literal output paths in common figure/video-writing calls
        for m in re.finditer(
            r"""(?:savefig|imsave|imwrite|VideoWriter|write_videofile|write_png)\s*\(\s*["']([^"']+\.(?:%s))["']""" % ext,
            cmd, re.I):
            yield m.group(1)
        # 2) cp/mv DESTINATION (last token of each simple command) with a media ext
        for seg in re.split(r"[;\n|&]+", cmd):
            toks = seg.split()
            if len(toks) >= 2 and toks[0] in ("cp", "mv"):
                dest = toks[-1].strip("\"'")
                if "." in dest and dest.rsplit(".", 1)[-1].lower() in MEDIA_EXT:
                    yield dest


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail open
    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {}) or {}
    bad = sorted({p for p in candidate_paths(tool, ti) if is_blocked_path(p)})
    if bad:
        sys.stderr.write(
            "BLOCKED by media-path hook: images/videos must be saved under media/ "
            "(CLAUDE.md “Saving images & videos”).\n"
            "Offending path(s): " + ", ".join(bad) + "\n"
            "Use media/{question}/{images|videos}/{YYYY-MM-DD}/{slug}_NN.ext — "
            "helper: `from cardiac_core.media import media_path`.\n"
            "Allowed elsewhere only: media/, Research/code_examples/, Builder/, "
            "Monodomain/_archive/, Surrogate/docs/diagrams/, test dirs, /tmp.\n"
        )
        sys.exit(2)  # block the tool call
    sys.exit(0)


if __name__ == "__main__":
    main()
