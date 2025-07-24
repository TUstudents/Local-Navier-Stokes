"""
LNS Solver: Professional Local Navier-Stokes Implementation

A research-grade implementation of Local Navier-Stokes equations with 
rigorous physics validation and proven accuracy.

Key Features:
- Correct physics implementation with proper deviatoric stress formula
- Complete objective derivatives for 2D flows
- Rigorous validation against analytical solutions
- High-performance O(N²) algorithms
- Professional software engineering practices

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
from lns_solver.core.state import LNSState
from lns_solver.core.physics import LNSPhysics
from lns_solver.core.numerics import LNSNumerics

# Solver imports
from lns_solver.solvers.solver_1d import LNSSolver1D

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
    "LNSState", 
    "LNSPhysics",
    "LNSNumerics",
    # Solvers
    "LNSSolver1D",
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