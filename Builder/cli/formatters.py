"""
Output formatting utilities for CLI.
"""

from typing import List, Dict, Any, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.RESET = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls.RED = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.BLUE = ''
        cls.MAGENTA = ''
        cls.CYAN = ''
        cls.WHITE = ''


def colorize(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def bold(text: str) -> str:
    """Make text bold."""
    return f"{Colors.BOLD}{text}{Colors.RESET}"


def dim(text: str) -> str:
    """Make text dim."""
    return f"{Colors.DIM}{text}{Colors.RESET}"


def success(text: str) -> str:
    """Format as success (green)."""
    return colorize(text, Colors.GREEN)


def error(text: str) -> str:
    """Format as error (red)."""
    return colorize(text, Colors.RED)


def warning(text: str) -> str:
    """Format as warning (yellow)."""
    return colorize(text, Colors.YELLOW)


def info(text: str) -> str:
    """Format as info (cyan)."""
    return colorize(text, Colors.CYAN)


def format_color_swatch(rgb: Tuple[int, ...], width: int = 2) -> str:
    """
    Format a color as a visual swatch using ANSI 24-bit color.

    Args:
        rgb: RGB tuple (r, g, b) or RGBA tuple
        width: Width of the swatch in characters

    Returns:
        Colored block characters
    """
    r, g, b = rgb[:3]
    # Use 24-bit color escape sequence
    return f"\033[48;2;{r};{g};{b}m{' ' * width}\033[0m"


def format_hex_color(rgb: Tuple[int, ...]) -> str:
    """Format RGB as hex string."""
    return '#{:02X}{:02X}{:02X}'.format(*rgb[:3])


def format_percentage(value: float, total: float) -> str:
    """Format value as percentage of total."""
    if total == 0:
        return "0.0%"
    pct = (value / total) * 100
    return f"{pct:.1f}%"


def format_dimensions(width: float, height: float, unit: str = "cm") -> str:
    """Format dimensions string."""
    return f"{width} x {height} {unit}"


def format_resolution(nx: int, ny: int) -> str:
    """Format mesh resolution."""
    return f"{nx} x {ny} nodes"


class TableFormatter:
    """Formatted table output."""

    def __init__(self, headers: List[str], col_widths: Optional[List[int]] = None):
        self.headers = headers
        self.rows: List[List[Any]] = []
        self.col_widths = col_widths

    def add_row(self, *cells):
        """Add a row to the table."""
        self.rows.append(list(cells))

    def _calculate_widths(self) -> List[int]:
        """Calculate column widths."""
        if self.col_widths:
            return self.col_widths

        widths = [len(str(h)) for h in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        return [w + 2 for w in widths]

    def render(self) -> str:
        """Render the table as a string."""
        widths = self._calculate_widths()
        lines = []

        # Header
        header_line = ""
        for header, width in zip(self.headers, widths):
            header_line += bold(str(header).ljust(width))
        lines.append(header_line)
        lines.append("-" * sum(widths))

        # Rows
        for row in self.rows:
            row_line = ""
            for cell, width in zip(row, widths):
                row_line += str(cell).ljust(width)
            lines.append(row_line)

        return "\n".join(lines)

    def print(self):
        """Print the table."""
        print(self.render())


class ProgressBar:
    """Simple progress bar for CLI."""

    def __init__(self, total: int, width: int = 40, prefix: str = ""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0

    def update(self, current: int):
        """Update progress."""
        self.current = current
        self._render()

    def increment(self, amount: int = 1):
        """Increment progress."""
        self.current += amount
        self._render()

    def _render(self):
        """Render the progress bar."""
        if self.total == 0:
            pct = 100
        else:
            pct = int((self.current / self.total) * 100)

        filled = int(self.width * self.current / self.total) if self.total > 0 else self.width
        bar = "█" * filled + "░" * (self.width - filled)

        print(f"\r{self.prefix}[{bar}] {pct}% ({self.current}/{self.total})", end="", flush=True)

    def finish(self):
        """Complete the progress bar."""
        self.current = self.total
        self._render()
        print()
