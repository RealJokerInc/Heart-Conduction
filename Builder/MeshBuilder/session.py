"""
MeshBuilder session management.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ..common.image import (
    load_image,
    detect_colors,
    threshold_transparency,
    filter_small_groups,
)
from ..common.utils import color_to_hex
from .models import CellGroup


@dataclass
class MeshBuilderSession:
    """
    Session for building a mesh from an image.

    Workflow:
    1. Load image
    2. Detect distinct colors
    3. Label each color group
    4. Assign properties (cell type, conductivity tensor)
    5. Generate mesh output
    """
    image_path: Optional[Path] = None
    image_array: Optional[np.ndarray] = None
    image_size: Tuple[int, int] = (0, 0)

    color_groups: Dict[Tuple[int, ...], CellGroup] = field(default_factory=dict)

    tissue_dimensions: Tuple[float, float] = (1.0, 1.0)
    dx: float = 0.01

    # --- Image Loading ---

    def load_image(self, path: str) -> None:
        """Load an image file."""
        self.image_path = Path(path)
        self.image_array, self.image_size = load_image(path)

    # --- Color Detection & Cleaning ---

    def detect_colors(self, auto_detect_background: bool = True) -> List[CellGroup]:
        """Detect all distinct colors in the loaded image."""
        if self.image_array is None:
            raise ValueError("No image loaded.")

        color_info = detect_colors(self.image_array, auto_detect_background)
        self._update_groups_from_info(color_info)
        return self.get_color_groups()

    def threshold_transparency(self, alpha_threshold: int = 128) -> List[CellGroup]:
        """Binarize alpha to remove anti-aliasing artifacts."""
        if self.image_array is None:
            raise ValueError("No image loaded.")

        self.image_array = threshold_transparency(self.image_array, alpha_threshold)
        return self.detect_colors()

    def filter_small_groups(self, min_percent: float = 0.1) -> List[CellGroup]:
        """Filter small color groups using adaptive mode filtering."""
        if self.image_array is None:
            raise ValueError("No image loaded.")

        color_info = {
            color: {'pixel_count': g.pixel_count, 'is_background': g.is_background}
            for color, g in self.color_groups.items()
        }

        self.image_array, new_info = filter_small_groups(
            self.image_array, color_info, min_percent
        )
        self._update_groups_from_info(new_info)
        return self.get_color_groups()

    def _update_groups_from_info(self, color_info: Dict) -> None:
        """Update color_groups from detect_colors output."""
        self.color_groups.clear()
        for color, info in color_info.items():
            group = CellGroup(
                color=color,
                pixel_count=info['pixel_count'],
                is_background=info['is_background'],
            )
            if info['is_background']:
                group.label = "background"
            self.color_groups[color] = group

    # --- Group Management ---

    def get_color_groups(self) -> List[CellGroup]:
        """Get all color groups, sorted by pixel count."""
        return sorted(
            self.color_groups.values(),
            key=lambda g: g.pixel_count,
            reverse=True
        )

    def mark_as_background(self, color: Tuple[int, ...]) -> None:
        """Mark a color group as background."""
        if color not in self.color_groups:
            raise KeyError(f"Color {color} not found.")
        self.color_groups[color].is_background = True
        self.color_groups[color].label = "background"

    def mark_as_tissue(self, color: Tuple[int, ...]) -> None:
        """Mark a color group as tissue."""
        if color not in self.color_groups:
            raise KeyError(f"Color {color} not found.")
        self.color_groups[color].is_background = False
        if self.color_groups[color].label == "background":
            self.color_groups[color].label = None

    def label_group(self, color: Tuple[int, ...], label: str) -> None:
        """Assign a label to a color group."""
        if color not in self.color_groups:
            raise KeyError(f"Color {color} not found.")
        self.color_groups[color].label = label

    def configure_group(
        self,
        color: Tuple[int, ...],
        label: str,
        cell_type: str,
        D_xx: float,
        D_yy: float,
        D_xy: float = 0.0
    ) -> None:
        """Fully configure a color group with all properties."""
        if color not in self.color_groups:
            raise KeyError(f"Color {color} not found.")

        group = self.color_groups[color]
        group.label = label
        group.cell_type = cell_type
        group.set_conductivity(D_xx, D_yy, D_xy)

    # --- Properties ---

    @property
    def tissue_groups(self) -> List[CellGroup]:
        """Get all non-background color groups."""
        return [g for g in self.color_groups.values() if not g.is_background]

    @property
    def background_groups(self) -> List[CellGroup]:
        """Get all background color groups."""
        return [g for g in self.color_groups.values() if g.is_background]

    @property
    def all_groups_configured(self) -> bool:
        """Check if all tissue groups have been configured."""
        return all(g.is_configured for g in self.tissue_groups)

    @property
    def unconfigured_groups(self) -> List[CellGroup]:
        """Get tissue groups that haven't been configured."""
        return [g for g in self.tissue_groups if not g.is_configured]

    # --- Mesh Settings ---

    def set_dimensions(
        self,
        tissue_width: float,
        tissue_height: float,
        dx: float
    ) -> None:
        """Set physical dimensions and resolution."""
        self.tissue_dimensions = (tissue_width, tissue_height)
        self.dx = dx

    def get_mesh_resolution(self) -> Tuple[int, int]:
        """Calculate mesh resolution based on dimensions and dx."""
        nx = int(self.tissue_dimensions[0] / self.dx) + 1
        ny = int(self.tissue_dimensions[1] / self.dx) + 1
        return (nx, ny)

    # --- Output ---

    def summary(self) -> str:
        """Get a summary of the current session state."""
        lines = ["MeshBuilder Session Summary", "=" * 40]

        if self.image_path:
            lines.append(f"Image: {self.image_path.name}")
            lines.append(f"Image size: {self.image_size[0]} x {self.image_size[1]} px")
        else:
            lines.append("No image loaded")

        lines.append(f"\nTissue: {self.tissue_dimensions[0]} x {self.tissue_dimensions[1]} cm")
        lines.append(f"dx: {self.dx} cm")
        lines.append(f"Mesh resolution: {self.get_mesh_resolution()}")

        tissue = self.tissue_groups
        lines.append(f"\nTissue groups ({len(tissue)}):")
        for i, g in enumerate(sorted(tissue, key=lambda x: x.pixel_count, reverse=True), 1):
            status = "configured" if g.is_configured else "not configured"
            label = g.label or "(unlabeled)"
            hex_color = color_to_hex(g.color)
            lines.append(f"  {i}. [{hex_color}] {label} - {g.pixel_count} px [{status}]")

        bg = self.background_groups
        if bg:
            lines.append(f"\nBackground ({len(bg)}):")
            for g in bg:
                hex_color = color_to_hex(g.color)
                lines.append(f"  - [{hex_color}] {g.pixel_count} px")

        return "\n".join(lines)
