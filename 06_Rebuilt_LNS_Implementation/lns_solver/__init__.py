"""
LNS Solver: Research Implementation of Local Navier-Stokes Equations

A prototype implementation of Local Navier-Stokes equations for research
and educational purposes, focusing on 1D finite volume methods.

Features:
- 1D finite volume solver with HLL flux computation
- Simplified Maxwell-Cattaneo-Vernotte and Upper Convected Maxwell models
- Basic validation against analytical Riemann solutions
- Standard numerical methods (SSP-RK2, ghost cells)

Limitations:
- 1D simplification of 3D tensor physics
- Research prototype - not validated for production use
- Simplified constitutive models

Example:
    Basic 1D shock tube simulation:

    >>> from lns_solver import LNSGrid, LNSState, LNSPhysics
    >>> grid = LNSGrid.create_uniform_1d(100, 0.0, 1.0)
    >>> state = LNSState(grid, n_variables=5)
    >>> physics = LNSPhysics()
"""

__version__ = "0.1.0"
__author__ = "LNS Development Team"
__email__ = "noreply@example.com"
__license__ = "MIT"

# Core infrastructure imports (available immediately)
from lns_solver.core.grid import LNSGrid
from lns_solver.core.state_enhanced import EnhancedLNSState, StateConfiguration, LNSVariables
from lns_solver.core.physics import LNSPhysics
from lns_solver.core.numerics_optimized import OptimizedLNSNumerics

# Solver imports
from lns_solver.solvers.solver_1d_final import FinalIntegratedLNSSolver1D

# Backward compatibility aliases
LNSSolver1D = FinalIntegratedLNSSolver1D
LNSState = EnhancedLNSState  # Backward compatibility for legacy code

# Utility imports
from lns_solver.utils.constants import PhysicalConstants
from lns_solver.utils.io import LNSDataWriter, LNSDataReader

# Version and metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core classes
    "LNSGrid",
    "EnhancedLNSState",
    "StateConfiguration", 
    "LNSVariables",
    "LNSState",  # Backward compatibility alias
    "LNSPhysics",
    "OptimizedLNSNumerics",
    # Solvers
    "FinalIntegratedLNSSolver1D",
    "LNSSolver1D",  # Backward compatibility alias
    # Utilities
    "PhysicalConstants",
    "LNSDataWriter",
    "LNSDataReader",
]

# Package-level configuration
import logging

# Set up package logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Package validation on import
def _validate_environment():
    """Validate that required dependencies are available."""
    try:
        import numpy as np
        import scipy
        import matplotlib
        
        # Check minimum versions
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        if numpy_version < (1, 21):
            logger.warning(f"NumPy version {np.__version__} < 1.21.0 may cause issues")
            
    except ImportError as e:
        logger.error(f"Required dependency missing: {e}")
        raise

# Validate environment on import
_validate_environment()

# Package information
def get_version_info():
    """Get detailed version information."""
    import sys
    import numpy as np
    import scipy
    import matplotlib
    
    return {
        'lns_solver': __version__,
        'python': sys.version,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'matplotlib': matplotlib.__version__,
    }

def print_version_info():
    """Print formatted version information."""
    info = get_version_info()
    print("LNS Solver Version Information:")
    print("=" * 35)
    for package, version in info.items():
        if package == 'python':
            version = version.split()[0]  # Remove extra info
        print(f"{package:12s}: {version}")

# Configuration management
class LNSConfig:
    """Global configuration for LNS Solver."""
    
    # Numerical tolerances
    RTOL = 1e-8
    ATOL = 1e-12
    
    # Performance settings
    USE_NUMBA = True
    PARALLEL_THRESHOLD = 1000
    
    # Validation settings
    VALIDATION_TOLERANCE = 1e-3
    CONSERVATION_TOLERANCE = 1e-12
    
    # I/O settings
    DEFAULT_OUTPUT_FORMAT = 'hdf5'
    COMPRESSION_LEVEL = 6
    
    @classmethod
    def set_tolerance(cls, rtol=None, atol=None):
        """Set global numerical tolerances."""
        if rtol is not None:
            cls.RTOL = rtol
        if atol is not None:
            cls.ATOL = atol
            
    @classmethod
    def enable_performance_mode(cls):
        """Enable high-performance settings."""
        cls.USE_NUMBA = True
        cls.PARALLEL_THRESHOLD = 100
        
    @classmethod
    def enable_debug_mode(cls):
        """Enable debug-friendly settings."""
        cls.USE_NUMBA = False
        cls.RTOL = 1e-6
        cls.ATOL = 1e-10

# Make config available at package level
config = LNSConfig()