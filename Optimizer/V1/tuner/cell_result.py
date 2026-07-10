"""
Optimizer V1 — CellResult dataclass (backend-neutral).

Extracted from cell_runner.py so it can be shared by both the V5.4-backed
(`cell_runner`) and cardiac_core-backed (`cell_runner_cc`) AP evaluators WITHOUT
pulling the `cardiac_sim` (V5.4) import that `cell_runner` performs at module load.

This is the single canonical definition. `cell_runner` re-exports it for
backward compatibility, so `from .cell_runner import CellResult` still works.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CellResult:
    """Results from a single-cell simulation."""
    apd90: Optional[float] = None
    dvdt_max: Optional[float] = None
    v_rest: float = 0.0
    v_peak: float = 0.0
    cl: Optional[float] = None
    V_trace: Optional[np.ndarray] = None
    t_trace: Optional[np.ndarray] = None
    restitution: Optional[List[Tuple[float, float]]] = None
    converged: bool = True
