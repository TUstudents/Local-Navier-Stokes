"""
LNS Validation Framework: Analytical solutions and classical method comparisons.

This package provides comprehensive validation tools for the LNS solver against:
- Analytical solutions (Riemann problems, heat conduction, acoustic waves)
- Classical Navier-Stokes solutions
- NSF limit recovery studies
- Performance benchmarks
"""

from lns_solver.validation.analytical_solutions import (
    RiemannExactSolver,
    HeatConductionExact,
    AcousticWaveExact
)
from lns_solver.validation.classical_solvers import (
    EulerSolver1D,
    NavierStokesSolver1D
)
from lns_solver.validation.validation_framework import (
    ValidationSuite,
    ComparisonMetrics
)

__all__ = [
    'RiemannExactSolver',
    'HeatConductionExact', 
    'AcousticWaveExact',
    'EulerSolver1D',
    'NavierStokesSolver1D',
    'ValidationSuite',
    'ComparisonMetrics'
]