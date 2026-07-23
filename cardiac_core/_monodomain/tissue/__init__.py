"""
Tissue Subpackage

Provides tissue material property definitions:
- IsotropicTissue: Uniform scalar conductivity
- AnisotropicTissue: Fiber-oriented conductivity (planned)
- HeterogeneousTissue: Scar tissue (D=0 regions) (planned)
"""

from .isotropic import IsotropicTissue

__all__ = [
    'IsotropicTissue',
]
