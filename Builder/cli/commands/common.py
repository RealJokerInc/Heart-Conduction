"""
Shared CLI utilities for prompts and formatting.
"""

from typing import List, Optional, Tuple, Any


def prompt_string(prompt: str, default: Optional[str] = None) -> str:
    """Prompt for a string value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    value = input(prompt).strip()
    return value if value else (default or "")


def prompt_float(prompt: str, default: Optional[float] = None) -> float:
    """Prompt for a float value."""
    while True:
        if default is not None:
            prompt_str = f"{prompt} [{default}]: "
        else:
            prompt_str = f"{prompt}: "

        value = input(prompt_str).strip()
        if not value and default is not None:
            return default

        try:
            return float(value)
        except ValueError:
            print("  Invalid number. Please try again.")


def prompt_int(prompt: str, default: Optional[int] = None) -> int:
    """Prompt for an integer value."""
    while True:
        if default is not None:
            prompt_str = f"{prompt} [{default}]: "
        else:
            prompt_str = f"{prompt}: "

        value = input(prompt_str).strip()
        if not value and default is not None:
            return default

        try:
            return int(value)
        except ValueError:
            print("  Invalid integer. Please try again.")


def prompt_choice(prompt: str, choices: List[str], default: Optional[int] = None) -> int:
    """Prompt user to select from a list of choices. Returns index."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = "*" if default == i else " "
        print(f"  {marker}{i}. {choice}")

    while True:
        if default is not None:
            value = input(f"Select [1-{len(choices)}] (default {default}): ").strip()
        else:
            value = input(f"Select [1-{len(choices)}]: ").strip()

        if not value and default is not None:
            return default - 1

        try:
            idx = int(value)
            if 1 <= idx <= len(choices):
                return idx - 1
        except ValueError:
            pass

        print(f"  Please enter a number between 1 and {len(choices)}")


def prompt_confirm(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    value = input(f"{prompt} {suffix}: ").strip().lower()

    if not value:
        return default
    return value in ('y', 'yes')


def print_table(headers: List[str], rows: List[List[Any]], col_widths: Optional[List[int]] = None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(str(header))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)

    # Header
    header_str = ""
    for header, width in zip(headers, col_widths):
        header_str += str(header).ljust(width)
    print(header_str)
    print("-" * len(header_str))

    # Rows
    for row in rows:
        row_str = ""
        for cell, width in zip(row, col_widths):
            row_str += str(cell).ljust(width)
        print(row_str)


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"[OK] {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"[ERROR] {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"[WARN] {message}")
