"""
StimBuilder session management.
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
from .models import StimRegion, StimProtocol, StimType, StimTarget


@dataclass
class StimBuilderSession:
    """
    Session for building stimulation maps from an image.

    Workflow:
    1. Load image
    2. Detect distinct colors
    3. Configure each region (label, type, amplitude, protocol)
    4. Export masks for simulation
    """
    image_path: Optional[Path] = None
    image_array: Optional[np.ndarray] = None
    image_size: Tuple[int, int] = (0, 0)

    stim_regions: Dict[Tuple[int, ...], StimRegion] = field(default_factory=dict)

    # --- Image Loading ---

    def load_image(self, path: str) -> None:
        """Load an image file."""
        self.image_path = Path(path)
        self.image_array, self.image_size = load_image(path)

    # --- Color Detection & Cleaning ---

    def detect_colors(self, auto_detect_background: bool = True) -> List[StimRegion]:
        """Detect all distinct colors in the loaded image."""
        if self.image_array is None:
            raise ValueError("No image loaded.")

        color_info = detect_colors(self.image_array, auto_detect_background)
        self._update_regions_from_info(color_info)
        return self.get_stim_regions()

    def threshold_transparency(self, alpha_threshold: int = 128) -> List[StimRegion]:
        """Binarize alpha to remove anti-aliasing artifacts."""
        if self.image_array is None:
            raise ValueError("No image loaded.")

        self.image_array = threshold_transparency(self.image_array, alpha_threshold)
        return self.detect_colors()

    def filter_small_groups(self, min_percent: float = 0.1) -> List[StimRegion]:
        """Filter small color groups using adaptive mode filtering."""
        if self.image_array is None:
            raise ValueError("No image loaded.")

        color_info = {
            color: {'pixel_count': r.pixel_count, 'is_background': r.is_background}
            for color, r in self.stim_regions.items()
        }

        self.image_array, new_info = filter_small_groups(
            self.image_array, color_info, min_percent
        )
        self._update_regions_from_info(new_info)
        return self.get_stim_regions()

    def _update_regions_from_info(self, color_info: Dict) -> None:
        """Update stim_regions from detect_colors output."""
        self.stim_regions.clear()
        for color, info in color_info.items():
            region = StimRegion(
                color=color,
                pixel_count=info['pixel_count'],
                is_background=info['is_background'],
            )
            if info['is_background']:
                region.label = "background"
            self.stim_regions[color] = region

    # --- Region Management ---

    def get_stim_regions(self) -> List[StimRegion]:
        """Get all stim regions, sorted by pixel count."""
        return sorted(
            self.stim_regions.values(),
            key=lambda r: r.pixel_count,
            reverse=True
        )

    def mark_as_background(self, color: Tuple[int, ...]) -> None:
        """Mark a color as background."""
        if color not in self.stim_regions:
            raise KeyError(f"Color {color} not found.")
        self.stim_regions[color].is_background = True
        self.stim_regions[color].label = "background"

    def mark_as_stim_region(self, color: Tuple[int, ...]) -> None:
        """Mark a color as a stim region."""
        if color not in self.stim_regions:
            raise KeyError(f"Color {color} not found.")
        self.stim_regions[color].is_background = False
        if self.stim_regions[color].label == "background":
            self.stim_regions[color].label = None

    # --- Region Configuration ---

    def configure_region(
        self,
        color: Tuple[int, ...],
        label: str,
        stim_type: StimType,
        amplitude: float,
        duration: float = 1.0,
        start_time: float = 0.0,
        bcl: float = 1000.0,
        num_pulses: Optional[int] = None,
        target: StimTarget = StimTarget.INTRACELLULAR
    ) -> None:
        """
        Fully configure a stimulation region.

        Args:
            color: RGB(A) tuple identifying the region
            label: Region name (e.g., "S1_pacing")
            stim_type: CURRENT_INJECTION or VOLTAGE_CLAMP
            amplitude: Stimulus strength (μA/cm² or mV)
            duration: Pulse duration (ms)
            start_time: First pulse start time (ms)
            bcl: Basic cycle length (ms)
            num_pulses: Number of pulses (None = continuous)
            target: INTRACELLULAR or EXTRACELLULAR
        """
        if color not in self.stim_regions:
            raise KeyError(f"Color {color} not found.")

        self.stim_regions[color].configure(
            label=label,
            stim_type=stim_type,
            amplitude=amplitude,
            duration=duration,
            start_time=start_time,
            bcl=bcl,
            num_pulses=num_pulses,
            target=target
        )

    def configure_current_injection(
        self,
        color: Tuple[int, ...],
        label: str,
        amplitude: float,
        duration: float = 1.0,
        start_time: float = 0.0,
        bcl: float = 1000.0,
        num_pulses: Optional[int] = None
    ) -> None:
        """Convenience: configure current injection stimulus."""
        self.configure_region(
            color=color,
            label=label,
            stim_type=StimType.CURRENT_INJECTION,
            amplitude=amplitude,
            duration=duration,
            start_time=start_time,
            bcl=bcl,
            num_pulses=num_pulses
        )

    def configure_voltage_clamp(
        self,
        color: Tuple[int, ...],
        label: str,
        voltage: float,
        duration: float = 1.0,
        start_time: float = 0.0,
        bcl: float = 1000.0,
        num_pulses: Optional[int] = None
    ) -> None:
        """Convenience: configure voltage clamp stimulus."""
        self.configure_region(
            color=color,
            label=label,
            stim_type=StimType.VOLTAGE_CLAMP,
            amplitude=voltage,
            duration=duration,
            start_time=start_time,
            bcl=bcl,
            num_pulses=num_pulses
        )

    # --- Properties ---

    @property
    def active_regions(self) -> List[StimRegion]:
        """Get all non-background stim regions."""
        return [r for r in self.stim_regions.values() if not r.is_background]

    @property
    def background_regions(self) -> List[StimRegion]:
        """Get all background regions."""
        return [r for r in self.stim_regions.values() if r.is_background]

    @property
    def all_regions_configured(self) -> bool:
        """Check if all active regions have been configured."""
        return all(r.is_configured for r in self.active_regions)

    @property
    def unconfigured_regions(self) -> List[StimRegion]:
        """Get active regions that haven't been configured."""
        return [r for r in self.active_regions if not r.is_configured]

    # --- Mask Generation ---

    def get_region_mask(self, color: Tuple[int, ...]) -> np.ndarray:
        """Get a boolean mask for a specific color region."""
        if self.image_array is None:
            raise ValueError("No image loaded.")
        if color not in self.stim_regions:
            raise KeyError(f"Color {color} not found.")

        color_array = np.array(color)
        return np.all(self.image_array == color_array, axis=-1)

    def get_all_masks(self) -> Dict[str, np.ndarray]:
        """Get boolean masks for all configured regions."""
        masks = {}
        for color, region in self.stim_regions.items():
            if region.is_configured and region.label and not region.is_background:
                masks[region.label] = self.get_region_mask(color)
        return masks

    # --- Export for Simulation ---

    def get_stim_config(self) -> List[dict]:
        """
        Get stimulus configuration for simulation.

        Returns:
            List of dicts with full stimulus info for each region
        """
        configs = []
        for region in self.active_regions:
            if region.is_configured:
                configs.append(region.summary_dict())
        return configs

    def get_region_by_label(self, label: str) -> Optional[StimRegion]:
        """Get a region by its label."""
        for region in self.stim_regions.values():
            if region.label == label:
                return region
        return None

    # --- Output ---

    def summary(self) -> str:
        """Get a summary of the current session state."""
        lines = ["StimBuilder Session Summary", "=" * 50]

        if self.image_path:
            lines.append(f"Image: {self.image_path.name}")
            lines.append(f"Size: {self.image_size[0]} x {self.image_size[1]} px")
        else:
            lines.append("No image loaded")

        active = self.active_regions
        lines.append(f"\nStim regions ({len(active)}):")

        for i, r in enumerate(sorted(active, key=lambda x: x.pixel_count, reverse=True), 1):
            hex_color = color_to_hex(r.color)
            label = r.label or "(unlabeled)"

            if r.is_configured:
                type_str = r.stim_type.value if r.stim_type else ""
                proto = r.protocol
                lines.append(f"  {i}. [{hex_color}] {label}")
                lines.append(f"      Type: {type_str}, Amplitude: {r.amplitude} {r.amplitude_unit}")
                lines.append(f"      Duration: {proto.duration} ms, BCL: {proto.bcl:.1f} ms")
                lines.append(f"      Start: {proto.start_time} ms, Pulses: {proto.num_pulses or 'continuous'}")
            else:
                lines.append(f"  {i}. [{hex_color}] {label} - {r.pixel_count} px [not configured]")

        bg = self.background_regions
        if bg:
            lines.append(f"\nBackground ({len(bg)}):")
            for r in bg:
                hex_color = color_to_hex(r.color)
                lines.append(f"  - [{hex_color}] {r.pixel_count} px")

        return "\n".join(lines)
