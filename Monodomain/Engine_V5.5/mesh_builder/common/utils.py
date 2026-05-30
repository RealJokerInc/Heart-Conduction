"""
Color utilities shared between builders.
"""

from typing import Tuple

BACKGROUND_COLOR_RGB = (255, 255, 255)


def is_background_color(color: Tuple[int, ...]) -> bool:
    """Check if a color should be treated as background."""
    # Transparent (alpha = 0)
    if len(color) == 4 and color[3] == 0:
        return True
    # White
    if color[:3] == BACKGROUND_COLOR_RGB[:3]:
        return True
    return False


def color_to_hex(color: Tuple[int, ...]) -> str:
    """Convert RGB(A) tuple to hex string for display."""
    if len(color) == 3:
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    elif len(color) == 4:
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}{color[3]:02x}"
    return str(color)


def hex_to_color(hex_str: str) -> Tuple[int, ...]:
    """Convert hex string to RGB(A) tuple."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 6:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    elif len(hex_str) == 8:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4, 6))
    raise ValueError(f"Invalid hex color: {hex_str}")
