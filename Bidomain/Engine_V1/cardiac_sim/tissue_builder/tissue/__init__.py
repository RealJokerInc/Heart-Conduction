"""Tissue properties — conductivity definitions."""

from .isotropic import IsotropicTissue
from .conductivity import BidomainConductivity

__all__ = ['IsotropicTissue', 'BidomainConductivity']
