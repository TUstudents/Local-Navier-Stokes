"""
Core infrastructure for LNS Solver.

This module contains the fundamental building blocks of the LNS solver:
- Grid management and computational geometry
- State vector management and variable conversions  
- Physics models and constitutive relations
- Numerical methods and algorithms

All core classes follow professional software engineering practices with:
- Complete type hints
- Comprehensive docstrings
- Extensive unit testing
- Performance optimization
"""

from lns_solver.core.grid import LNSGrid
from lns_solver.core.state_enhanced import EnhancedLNSState, StateConfiguration, LNSVariables
from lns_solver.core.physics import LNSPhysics
from lns_solver.core.numerics_optimized import OptimizedLNSNumerics

__all__ = [
    "LNSGrid",
    "EnhancedLNSState",
    "StateConfiguration",
    "LNSVariables", 
    "LNSPhysics",
    "OptimizedLNSNumerics",
]

# Backward compatibility alias
LNSState = EnhancedLNSState