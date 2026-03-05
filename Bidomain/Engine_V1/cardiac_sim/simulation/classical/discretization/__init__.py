"""Spatial discretization for bidomain equations."""

from .base import BidomainSpatialDiscretization
from .fdm import BidomainFDMDiscretization

__all__ = ['BidomainSpatialDiscretization', 'BidomainFDMDiscretization']
