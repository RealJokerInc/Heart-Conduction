"""
Standard TTP06 Cell Types

Defines the three standard ventricular cell types from the original
ten Tusscher-Panfilov 2006 publication:
- ENDO: Endocardial cells
- EPI: Epicardial cells
- M_CELL: Mid-myocardial cells

Reference:
ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a
human ventricular tissue model." Am J Physiol Heart Circ Physiol.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from enum import Enum


class StandardCellTypes(str, Enum):
    """Standard TTP06 cell types."""
    ENDO = "ENDO"
    EPI = "EPI"
    M_CELL = "M_CELL"


@dataclass
class CellTypeConfig:
    """
    Configuration for a cell type.

    Contains parameter overrides relative to the base TTP06Parameters.
    Only parameters that differ from base need to be specified.
    """
    name: str
    description: str = ""

    # Conductance parameters (nS/pF)
    GNa: float = None      # Fast sodium
    GK1: float = None      # Inward rectifier K+
    Gto: float = None      # Transient outward K+
    GKr: float = None      # Rapid delayed rectifier K+
    GKs: float = None      # Slow delayed rectifier K+
    GCaL: float = None     # L-type Ca2+ (via PCa scaling)
    GpCa: float = None     # Sarcolemmal Ca2+ pump
    GpK: float = None      # Plateau K+
    GbNa: float = None     # Background Na+
    GbCa: float = None     # Background Ca2+

    # Permeability
    PCa: float = None      # L-type Ca2+ permeability

    # Exchanger/pump
    KNaCa: float = None    # Na+/Ca2+ exchanger
    PNaK: float = None     # Na+/K+ pump

    # Kinetic modifiers
    use_epi_ito_kinetics: bool = False  # Use epicardial Ito inactivation

    def get_overrides(self) -> Dict[str, Any]:
        """Return dict of non-None parameter overrides."""
        overrides = {}
        for param in ['GNa', 'GK1', 'Gto', 'GKr', 'GKs', 'GCaL',
                      'GpCa', 'GpK', 'GbNa', 'GbCa', 'PCa',
                      'KNaCa', 'PNaK']:
            value = getattr(self, param)
            if value is not None:
                overrides[param] = value
        if self.use_epi_ito_kinetics:
            overrides['use_epi_ito_kinetics'] = True
        return overrides


# Standard cell type definitions
STANDARD_CONFIGS: Dict[str, CellTypeConfig] = {
    StandardCellTypes.ENDO: CellTypeConfig(
        name="ENDO",
        description="Endocardial cell - smaller Ito, standard IKs",
        Gto=0.073,      # Much smaller than EPI
        GKs=0.392,      # Standard
        use_epi_ito_kinetics=False,
    ),

    StandardCellTypes.EPI: CellTypeConfig(
        name="EPI",
        description="Epicardial cell - larger Ito with different kinetics",
        Gto=0.294,      # 4x larger than ENDO
        GKs=0.392,      # Standard
        use_epi_ito_kinetics=True,
    ),

    StandardCellTypes.M_CELL: CellTypeConfig(
        name="M_CELL",
        description="Mid-myocardial cell - reduced IKs for longer APD",
        Gto=0.294,      # Same as EPI
        GKs=0.098,      # 0.25x - gives longer APD (~400ms)
        use_epi_ito_kinetics=True,
    ),
}


def get_standard_parameters(celltype: StandardCellTypes) -> CellTypeConfig:
    """
    Get parameter configuration for a standard cell type.

    Parameters
    ----------
    celltype : StandardCellTypes
        One of ENDO, EPI, or M_CELL

    Returns
    -------
    CellTypeConfig
        Configuration with parameter overrides
    """
    if celltype not in STANDARD_CONFIGS:
        raise ValueError(f"Unknown cell type: {celltype}. "
                        f"Valid types: {list(STANDARD_CONFIGS.keys())}")
    return STANDARD_CONFIGS[celltype]
