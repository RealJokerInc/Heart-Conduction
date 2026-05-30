"""
TTP06 Cell Type Definitions

Provides standard and custom cell type parameter configurations.

Standard cell types:
- ENDO: Endocardial
- EPI: Epicardial
- M_CELL: Mid-myocardial

Custom cell types can be defined in the custom/ subdirectory.
"""

from .standard import (
    StandardCellTypes,
    get_standard_parameters,
)

__all__ = [
    'StandardCellTypes',
    'get_standard_parameters',
]
