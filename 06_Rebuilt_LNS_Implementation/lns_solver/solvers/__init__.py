"""
LNS Solvers: Complete solver implementations for different dimensions.

This package contains the main solver classes that integrate all core
infrastructure components into working LNS simulation engines.
"""

from lns_solver.solvers.solver_1d_final import FinalIntegratedLNSSolver1D

__all__ = [
    'FinalIntegratedLNSSolver1D',
]

# Provide backward compatibility alias
LNSSolver1D = FinalIntegratedLNSSolver1D