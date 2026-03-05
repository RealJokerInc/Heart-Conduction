"""
Custom TTP06 Cell Type Definitions

User-defined cell types for specific studies and simulations.
Each study can define its own cell type configurations.
"""

# Import custom cell types as they are created
from .reentry_study import (
    ReentryCellTypes,
    REENTRY_CONFIGS,
    get_reentry_parameters,
)

__all__ = [
    'ReentryCellTypes',
    'REENTRY_CONFIGS',
    'get_reentry_parameters',
]
