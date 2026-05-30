"""
Tissue Subpackage

Provides tissue material property definitions:
- IsotropicTissue: Uniform scalar conductivity
- AnisotropicTissue: Fiber-oriented conductivity [Phase 7]
- HeterogeneousTissue: Scar tissue (D=0 regions) [Phase 7]
"""

from .isotropic import IsotropicTissue

__all__ = [
    'IsotropicTissue',
]
