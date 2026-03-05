"""
Stimulus Region Definitions

Common spatial region functions for stimulus application.

Migrated from V5.3 tissue/stimulus.py (region helper functions).
"""


def rectangular_region(x0: float, y0: float, x1: float, y1: float):
    """Create rectangular stimulus region."""
    def region(x, y):
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return region


def circular_region(cx: float, cy: float, radius: float):
    """Create circular stimulus region."""
    def region(x, y):
        return (x - cx)**2 + (y - cy)**2 <= radius**2
    return region


def left_edge_region(width: float = 0.1):
    """Create left edge stimulus region."""
    def region(x, y):
        return x <= width
    return region


def point_stimulus(cx: float, cy: float, radius: float = 0.05):
    """Create point stimulus (small circle)."""
    return circular_region(cx, cy, radius)
