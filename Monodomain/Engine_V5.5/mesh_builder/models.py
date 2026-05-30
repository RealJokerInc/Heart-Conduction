"""
MeshBuilder data models.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CellGroup:
    """Represents a group of cells identified by a unique color."""
    color: Tuple[int, ...]
    pixel_count: int
    label: Optional[str] = None
    cell_type: Optional[str] = None
    is_background: bool = False

    # Full 2x2 conductivity tensor: [[D_xx, D_xy], [D_xy, D_yy]]
    conductivity_tensor: Optional[np.ndarray] = None

    def set_conductivity(self, D_xx: float, D_yy: float, D_xy: float = 0.0):
        """Set the conductivity tensor values."""
        self.conductivity_tensor = np.array([
            [D_xx, D_xy],
            [D_xy, D_yy]
        ])

    @property
    def is_configured(self) -> bool:
        """Check if this group has been fully configured."""
        if self.is_background:
            return True
        return (
            self.label is not None and
            self.cell_type is not None and
            self.conductivity_tensor is not None
        )
