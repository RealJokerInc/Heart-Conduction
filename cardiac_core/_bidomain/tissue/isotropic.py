"""
Isotropic Tissue Properties

Provides uniform (scalar) conductivity tissue definition.

Extracted from V5.3 tissue/simulation.py SimulationConfig.
"""

from dataclasses import dataclass


@dataclass
class IsotropicTissue:
    """
    Isotropic tissue with uniform scalar conductivity.

    Attributes
    ----------
    D : float
        Diffusion coefficient (cm^2/ms). Isotropic — same in all directions.
    chi : float
        Surface-to-volume ratio (cm^-1).
    Cm : float
        Membrane capacitance (uF/cm^2).
    """
    D: float = 0.001          # cm^2/ms
    chi: float = 1400.0       # cm^-1
    Cm: float = 1.0           # uF/cm^2
