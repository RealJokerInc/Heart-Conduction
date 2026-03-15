"""
Engine_V2 - Optimized Cardiac Simulation Framework
==================================================

Improvements over Engine_V1:
- Numba JIT acceleration (50-100x speedup)
- Fixed diffusion method using proper np.pad
- Corrected boundary conditions
- Eliminated ring artifacts
- Natural wave speed physics around infarcts

Author: Generated with Claude Code
Date: 2025-12-09
"""

__version__ = "2.0.0"
