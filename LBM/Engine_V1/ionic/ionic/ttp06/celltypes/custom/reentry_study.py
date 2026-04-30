"""
Re-entry Circuit Study Cell Types

Custom cell types for dual-pathway re-entry circuit simulations.

Defines cell types with modified APD/ERP characteristics:
- FAST_PATH: Reduced GKs for longer APD/ERP (~400ms)
- SLOW_PATH: Enhanced GKs for shorter APD/ERP (~220ms)
- INLET_OUTLET: Standard EPI parameters

The fast pathway has longer refractory period, so premature stimuli
block there while conducting through the slow pathway, enabling
circus movement re-entry.

Reference physiology:
- AV nodal dual pathway: fast pathway has longer ERP
- See: PMC3893335 - Dual AV Nodal Pathways Physiology

Author: Generated for Heart Conduction project
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum

from ionic.ttp06.celltypes.standard import CellTypeConfig


class ReentryCellTypes(str, Enum):
    """Cell types for re-entry circuit study."""
    FAST_PATH = "FAST_PATH"
    SLOW_PATH = "SLOW_PATH"
    INLET_OUTLET = "INLET_OUTLET"


# Re-entry study cell type configurations
# Base: EPI cell parameters (Gto=0.294, GKs=0.392)

REENTRY_CONFIGS: Dict[str, CellTypeConfig] = {
    ReentryCellTypes.FAST_PATH: CellTypeConfig(
        name="FAST_PATH",
        description="Fast conduction, very long APD/ERP - for upper ring pathway",
        # Greatly reduced IKs → much longer APD → much longer ERP
        # GKs = 0.05 (~0.13× of EPI) gives APD ~500-550ms
        Gto=0.294,      # Same as EPI
        GKs=0.05,       # 0.13× of standard - very long APD
        use_epi_ito_kinetics=True,
    ),

    ReentryCellTypes.SLOW_PATH: CellTypeConfig(
        name="SLOW_PATH",
        description="Slow conduction, short APD/ERP - for lower ring pathway",
        # Enhanced IKs → shorter APD → shorter ERP
        # GKs = 0.588 (1.5× of EPI) gives APD ~220ms
        Gto=0.294,      # Same as EPI
        GKs=0.588,      # 1.5× of standard - shorter APD
        use_epi_ito_kinetics=True,
    ),

    ReentryCellTypes.INLET_OUTLET: CellTypeConfig(
        name="INLET_OUTLET",
        description="Standard EPI parameters for inlet/outlet regions",
        Gto=0.294,      # Standard EPI
        GKs=0.392,      # Standard EPI
        use_epi_ito_kinetics=True,
    ),
}


def get_reentry_parameters(celltype: ReentryCellTypes) -> CellTypeConfig:
    """
    Get parameter configuration for a re-entry study cell type.

    Parameters
    ----------
    celltype : ReentryCellTypes
        One of FAST_PATH, SLOW_PATH, or INLET_OUTLET

    Returns
    -------
    CellTypeConfig
        Configuration with parameter overrides
    """
    if celltype not in REENTRY_CONFIGS:
        raise ValueError(f"Unknown re-entry cell type: {celltype}. "
                        f"Valid types: {list(REENTRY_CONFIGS.keys())}")
    return REENTRY_CONFIGS[celltype]


# Expected APD values (approximate, at 1 Hz pacing):
# - FAST_PATH (GKs=0.05): APD ~500-550 ms (very long)
# - INLET_OUTLET (GKs=0.392): APD ~280-300 ms
# - SLOW_PATH (GKs=0.588): APD ~200-240 ms
#
# ERP is typically ~90% of APD, so:
# - FAST_PATH ERP: ~450-500 ms (very long)
# - INLET_OUTLET ERP: ~250-270 ms
# - SLOW_PATH ERP: ~180-220 ms
#
# For re-entry induction, the S1-S2 interval should be:
# - Longer than SLOW_PATH ERP (wave conducts through slow path)
# - Shorter than FAST_PATH ERP (wave blocks in fast path)
# - Optimal window: ~220-450 ms after previous beat (wider window now)
