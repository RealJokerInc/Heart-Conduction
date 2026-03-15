"""
Shared fixtures for Optimizer V1 tests.
"""

import sys
import os

# Add paths so tests can import tuner and cardiac_sim
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'Monodomain', 'Engine_V5.4'))
