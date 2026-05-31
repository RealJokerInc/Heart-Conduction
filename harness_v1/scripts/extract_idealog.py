#!/usr/bin/env python3
"""Extract thinking trail from Claude Code session logs for IDEALOG.md creation.

Reads .jsonl session files line-by-line (no full-file loading), greps for
research question folder names, and extracts decision/failure/insight patterns.
Outputs a summary .md file per question to /tmp/idealog_extracts/.
"""
import json
import os
import re
from pathlib import Path
from datetime import datetime

SESSION_DIR = Path.home() / ".claude/projects/-home-norepinephrine-Documents-Heart-Conduction"
QUESTIONS = [
    "boundary_conduction_speedup",
    "ionic_model_optimization",
    "engine_consolidation",
    "geometry_induced_pacemaking",
    "mature_hipsc_cm_models",
    "research_environment_optimization",
]

# Keywords that suggest decisions, failures, or insights
DECISION_KEYWORDS = [
    "decided", "let's do", "let's go with", "go with", "chose", "picking",
    "we'll use", "settled on", "confirmed", "validated", "adopt",
]
FAILURE_KEYWORDS = [
    "failed because", "didn't work", "doesn't work", "tried but",
    "reverted", "abandoned", "rejected", "error", "bug", "broke",
    "wrong", "incorrect", "diverge",
]
INSIGHT_KEYWORDS = [
    "realized", "oh wait", "this means", "key insight", "important",
    "discovered", "noticed", "turns out", "actually", "spawned",
    "implies", "breakthrough",
]

OUTPUT_DIR = Path("/tmp/idealog_extracts")
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_text(msg: dict) -> str:
    """Extract readable text from a message object."""
    content = msg.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        # Capture file paths from tool inputs
                        for key in ("file_path", "command", "pattern", "path"):
                            if key in inp:
                                parts.append(f"[{key}: {inp[key]}]")
        return " ".join(parts)
    return str(content)


def classify_entry(text: str) -> list[str]:
    """Classify text as decision, failure, insight, or none."""
    text_lower = text.lower()
    categories = []
    if any(kw in text_lower for kw in DECISION_KEYWORDS):
        categories.append("decision")
    if any(kw in text_lower for kw in FAILURE_KEYWORDS):
        categories.append("failure")
    if any(kw in text_lower for kw in INSIGHT_KEYWORDS):
        categories.append("insight")
    return categories


def main():
    jsonl_files = sorted(SESSION_DIR.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} session files in {SESSION_DIR}")
    print(f"Total size: {sum(f.stat().st_size for f in jsonl_files) / 1024 / 1024:.1f} MB")
    print()

    for question in QUESTIONS:
        entries = {"decision": [], "failure": [], "insight": []}
        sessions_seen = set()

        for jsonl_file in jsonl_files:
            file_has_match = False

            with open(jsonl_file) as f:
                for line_num, line in enumerate(f):
                    # Quick string check before JSON parsing
                    if question not in line:
                        continue

                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type", "")
                    if msg_type not in ("user", "assistant"):
                        continue

                    text = extract_text(msg)
                    if question not in text:
                        continue

                    timestamp = msg.get("timestamp", "")
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            date_str = dt.strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError):
                            date_str = timestamp[:16]
                    else:
                        date_str = "unknown"

                    categories = classify_entry(text)
                    if not categories:
                        continue

                    # Truncate to keep output manageable
                    snippet = text[:600].replace("\n", " ").strip()
                    if len(text) > 600:
                        snippet += "..."

                    for cat in categories:
                        entries[cat].append(f"**[{date_str}]** ({msg_type}): {snippet}")

                    if not file_has_match:
                        file_has_match = True
                        sessions_seen.add(jsonl_file.stem[:8])

        # Write output
        output = OUTPUT_DIR / f"{question}.md"
        total = sum(len(v) for v in entries.values())

        lines = [f"# Chat Log Extracts: {question}\n"]
        lines.append(f"Sessions with matches: {len(sessions_seen)}")
        lines.append(f"Total entries: {total}\n")

        if entries["decision"]:
            lines.append(f"## Decisions ({len(entries['decision'])})\n")
            for e in entries["decision"]:
                lines.append(f"- {e}\n")

        if entries["failure"]:
            lines.append(f"\n## Failures ({len(entries['failure'])})\n")
            for e in entries["failure"]:
                lines.append(f"- {e}\n")

        if entries["insight"]:
            lines.append(f"\n## Insights ({len(entries['insight'])})\n")
            for e in entries["insight"]:
                lines.append(f"- {e}\n")

        if total == 0:
            lines.append("No relevant entries found.\n")

        output.write_text("\n".join(lines))
        print(f"{question}: {total} entries ({len(entries['decision'])}d/{len(entries['failure'])}f/{len(entries['insight'])}i) → {output}")


if __name__ == "__main__":
    main()
