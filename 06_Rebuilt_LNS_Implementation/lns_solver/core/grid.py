"""
LNS Grid: Computational grid management with boundary conditions.

This module provides the LNSGrid class for managing computational grids,
boundary conditions, and geometric operations. It supports 1D, 2D, and 3D
grids with multiple boundary condition types.

Example:
    Create a 1D uniform grid:
    
    >>> grid = LNSGrid.create_uniform_1d(nx=100, x_min=0.0, x_max=1.0)
    >>> print(f"Grid spacing: {grid.dx:.4f}")
    
    Create a 2D grid around a cylinder:
    
    >>> grid = LNSGrid.create_cylinder_grid(radius=0.5, nx=100, ny=100)
    >>> # Use solver.set_boundary_condition() for boundary conditions
"""

from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Type aliases for clarity
GridType = Literal['uniform', 'stretched', 'unstructured']
CoordinateArray = np.ndarray


class LNSGrid:
    """
    Computational grid for LNS solver - handles geometry and grid metrics only.
    
    This class handles the computational geometry and grid metrics for finite 
    volume discretization. It supports uniform and non-uniform grids in 1D, 2D, and 3D.
    
    ARCHITECTURAL CHANGE: Boundary conditions are now handled exclusively by
    GhostCellBoundaryHandler from boundary_conditions.py to avoid dual systems.
    
    Attributes:
        ndim: Number of spatial dimensions
        nx, ny, nz: Number of cells in each direction
        x, y, z: Cell center coordinates
        dx, dy, dz: Grid spacing (uniform) or spacing arrays
        
    Example:
        >>> grid = LNSGrid.create_uniform_1d(100, 0.0, 1.0)
        >>> # For boundary conditions, use GhostCellBoundaryHandler instead
    """
    
    def __init__(
        self,
        ndim: int,
        coordinates: Dict[str, CoordinateArray],
        spacing: Dict[str, Union[float, CoordinateArray]],
        grid_type: GridType = 'uniform'
    ):
        """
        Initialize LNS grid.
        
        Args:
            ndim: Number of spatial dimensions (1, 2, or 3)
            coordinates: Dictionary with 'x', 'y', 'z' coordinate arrays
            spacing: Dictionary with 'dx', 'dy', 'dz' spacing values/arrays
            grid_type: Type of grid ('uniform', 'stretched', 'unstructured')
            
        Raises:
            ValueError: If ndim not in [1, 2, 3] or inconsistent inputs
            TypeError: If coordinates/spacing have wrong types
        """
        if ndim not in [1, 2, 3]:
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
            
        self.ndim = ndim
        self.grid_type = grid_type
        
        # Store coordinates and validate
        self.coordinates = coordinates
        self.spacing = spacing
        self._validate_grid_data()
        
        # Extract dimensions
        if 'x' in coordinates:
            self.nx = len(coordinates['x'])
            self.x = coordinates['x']
            self.dx = spacing['dx']
        else:
            self.nx = 0
            self.x = np.array([])
            self.dx = 0.0
            
        if 'y' in coordinates and ndim >= 2:
            self.ny = len(coordinates['y'])
            self.y = coordinates['y']
            self.dy = spacing['dy']
        else:
            self.ny = 0
            self.y = np.array([])
            self.dy = 0.0
            
        if 'z' in coordinates and ndim >= 3:
            self.nz = len(coordinates['z'])
            self.z = coordinates['z']
            self.dz = spacing['dz']
        else:
            self.nz = 0
            self.z = np.array([])
            self.dz = 0.0
            
        # Compute grid metrics
        self._compute_grid_metrics()
        
        logger.info(f"Created {ndim}D {grid_type} grid: {self._format_dimensions()}")
    
    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.nx * (self.ny if self.ny > 0 else 1) * (self.nz if self.nz > 0 else 1)
    
    @property
    def x_bounds(self) -> Tuple[float, float]:
        """X-direction domain bounds."""
        if self.nx > 0:
            dx_half = self.dx / 2 if isinstance(self.dx, (int, float)) else self.dx[0] / 2
            return (float(self.x[0] - dx_half), float(self.x[-1] + dx_half))
        return (0.0, 0.0)
        
    def _validate_grid_data(self) -> None:
        """Validate grid coordinates and spacing data."""
        required_coords = ['x']
        if self.ndim >= 2:
            required_coords.append('y')
        if self.ndim >= 3:
            required_coords.append('z')
            
        for coord in required_coords:
            if coord not in self.coordinates:
                raise ValueError(f"Missing coordinate '{coord}' for {self.ndim}D grid")
            if not isinstance(self.coordinates[coord], np.ndarray):
                raise TypeError(f"Coordinate '{coord}' must be numpy array")
                
        # Validate spacing
        for coord in required_coords:
            spacing_key = f'd{coord}'
            if spacing_key not in self.spacing:
                raise ValueError(f"Missing spacing '{spacing_key}' for {self.ndim}D grid")
                
    def _compute_grid_metrics(self) -> None:
        """Compute grid metrics like cell volumes and face areas."""
        if self.ndim == 1:
            if isinstance(self.dx, np.ndarray):
                self.cell_volumes = self.dx
            else:
                self.cell_volumes = np.full(self.nx, self.dx)
        elif self.ndim == 2:
            dx_array = self.dx if isinstance(self.dx, np.ndarray) else np.full(self.nx, self.dx)
            dy_array = self.dy if isinstance(self.dy, np.ndarray) else np.full(self.ny, self.dy)
            self.cell_volumes = np.outer(dx_array, dy_array).flatten()
        elif self.ndim == 3:
            dx_array = self.dx if isinstance(self.dx, np.ndarray) else np.full(self.nx, self.dx)
            dy_array = self.dy if isinstance(self.dy, np.ndarray) else np.full(self.ny, self.dy)
            dz_array = self.dz if isinstance(self.dz, np.ndarray) else np.full(self.nz, self.dz)
            # 3D volume computation
            dx_3d, dy_3d, dz_3d = np.meshgrid(dx_array, dy_array, dz_array, indexing='ij')
            self.cell_volumes = (dx_3d * dy_3d * dz_3d).flatten()
            
    def _format_dimensions(self) -> str:
        """Format grid dimensions for logging."""
        if self.ndim == 1:
            return f"{self.nx} cells"
        elif self.ndim == 2:
            return f"{self.nx}×{self.ny} cells"
        else:
            return f"{self.nx}×{self.ny}×{self.nz} cells"
    
    @classmethod
    def create_uniform_1d(
        cls,
        nx: int,
        x_min: float,
        x_max: float
    ) -> 'LNSGrid':
        """
        Create uniform 1D grid.
        
        Args:
            nx: Number of cells
            x_min: Left boundary coordinate
            x_max: Right boundary coordinate
            
        Returns:
            LNSGrid: Initialized 1D grid
            
        Example:
            >>> grid = LNSGrid.create_uniform_1d(100, 0.0, 1.0)
            >>> print(f"Grid spacing: {grid.dx:.4f}")
            Grid spacing: 0.0100
        """
        if nx <= 0:
            raise ValueError(f"nx must be positive, got {nx}")
        if x_max <= x_min:
            raise ValueError(f"x_max ({x_max}) must be > x_min ({x_min})")
            
        dx = (x_max - x_min) / nx
        x = np.linspace(x_min + dx/2, x_max - dx/2, nx)  # Cell centers
        
        coordinates = {'x': x}
        spacing = {'dx': dx}
        
        return cls(ndim=1, coordinates=coordinates, spacing=spacing, grid_type='uniform')
    
    @classmethod
    def create_uniform_2d(
        cls,
        nx: int,
        ny: int, 
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float]
    ) -> 'LNSGrid':
        """
        Create uniform 2D grid.
        
        Args:
            nx: Number of cells in x-direction
            ny: Number of cells in y-direction
            x_bounds: (x_min, x_max) tuple
            y_bounds: (y_min, y_max) tuple
            
        Returns:
            LNSGrid: Initialized 2D grid
            
        Example:
            >>> grid = LNSGrid.create_uniform_2d(50, 50, (0.0, 1.0), (0.0, 1.0))
            >>> print(f"Grid: {grid.nx}×{grid.ny}")
            Grid: 50×50
        """
        if nx <= 0 or ny <= 0:
            raise ValueError(f"nx and ny must be positive, got {nx}, {ny}")
            
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid bounds: max must be > min")
            
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        
        x = np.linspace(x_min + dx/2, x_max - dx/2, nx)
        y = np.linspace(y_min + dy/2, y_max - dy/2, ny)
        
        coordinates = {'x': x, 'y': y}
        spacing = {'dx': dx, 'dy': dy}
        
        return cls(ndim=2, coordinates=coordinates, spacing=spacing, grid_type='uniform')
    
    @classmethod
    def create_cylinder_grid(
        cls,
        radius: float,
        nx: int,
        ny: int,
        domain_size: float = 4.0
    ) -> 'LNSGrid':
        """
        Create 2D grid suitable for flow around cylinder.
        
        Args:
            radius: Cylinder radius
            nx: Number of cells in x-direction
            ny: Number of cells in y-direction  
            domain_size: Domain size as multiple of radius
            
        Returns:
            LNSGrid: 2D grid with cylinder boundary
            
        Example:
            >>> grid = LNSGrid.create_cylinder_grid(0.5, 100, 100)
            >>> # Use solver.set_boundary_condition() for boundary conditions
        """
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
            
        # Create rectangular domain around cylinder
        x_min, x_max = -domain_size * radius, domain_size * radius
        y_min, y_max = -domain_size * radius, domain_size * radius
        
        grid = cls.create_uniform_2d(nx, ny, (x_min, x_max), (y_min, y_max))
        
        # Mark cylinder boundary (this is simplified - real implementation would
        # use immersed boundary or body-fitted coordinates)
        grid.cylinder_radius = radius
        
        return grid
    
    # ARCHITECTURAL CHANGE: Boundary condition methods removed
    # 
    # The conflicting boundary condition system has been removed from LNSGrid.
    # All boundary condition handling is now done exclusively through
    # GhostCellBoundaryHandler from boundary_conditions.py.
    #
    # This eliminates the dual, incompatible systems that caused confusion
    # and potential bugs. Use solver.set_boundary_condition() which delegates
    # to the proper GhostCellBoundaryHandler system.
    
    # REMOVED: apply_boundary_conditions method
    # 
    # This method was DANGEROUS because it directly modified physical cells,
    # violating finite volume conservation principles. 
    # 
    # USE INSTEAD: GhostCellBoundaryHandler from boundary_conditions.py
    # which correctly applies BCs to ghost cells only, preserving conservation.
    
    def get_neighbor_indices(self, i: int, j: int = 0, k: int = 0) -> Dict[str, Tuple[int, ...]]:
        """
        Get neighbor cell indices for finite difference operations.
        
        Args:
            i, j, k: Cell indices
            
        Returns:
            Dictionary with neighbor indices for each direction
        """
        neighbors = {}
        
        if self.ndim >= 1:
            neighbors['left'] = (max(0, i-1), j, k)
            neighbors['right'] = (min(self.nx-1, i+1), j, k)
            
        if self.ndim >= 2:
            neighbors['bottom'] = (i, max(0, j-1), k)
            neighbors['top'] = (i, min(self.ny-1, j+1), k)
            
        if self.ndim >= 3:
            neighbors['back'] = (i, j, max(0, k-1))
            neighbors['front'] = (i, j, min(self.nz-1, k+1))
            
        return neighbors
    
    def compute_cell_volumes(self) -> np.ndarray:
        """
        Compute cell volumes for finite volume discretization.
        
        Returns:
            Array of cell volumes
        """
        return self.cell_volumes
    
    def is_boundary_cell(self, i: int, j: int = 0, k: int = 0) -> bool:
        """Check if cell is on domain boundary."""
        if self.ndim == 1:
            return i == 0 or i == self.nx - 1
        elif self.ndim == 2:
            return (i == 0 or i == self.nx - 1 or 
                   j == 0 or j == self.ny - 1)
        else:
            return (i == 0 or i == self.nx - 1 or
                   j == 0 or j == self.ny - 1 or
                   k == 0 or k == self.nz - 1)
    
    def get_grid_info(self) -> Dict[str, Union[int, float, str]]:
        """Get comprehensive grid information."""
        info = {
            'ndim': self.ndim,
            'grid_type': self.grid_type,
            'total_cells': self.nx * (self.ny if self.ny > 0 else 1) * (self.nz if self.nz > 0 else 1),
        }
        
        if self.ndim >= 1:
            info.update({
                'nx': self.nx,
                'x_min': float(np.min(self.x)),
                'x_max': float(np.max(self.x)),
                'dx': float(self.dx) if isinstance(self.dx, (int, float)) else 'variable',
            })
            
        if self.ndim >= 2:
            info.update({
                'ny': self.ny,
                'y_min': float(np.min(self.y)),
                'y_max': float(np.max(self.y)),
                'dy': float(self.dy) if isinstance(self.dy, (int, float)) else 'variable',
            })
            
        if self.ndim >= 3:
            info.update({
                'nz': self.nz,
                'z_min': float(np.min(self.z)),
                'z_max': float(np.max(self.z)),
                'dz': float(self.dz) if isinstance(self.dz, (int, float)) else 'variable',
            })
            
        return info
    
    def __repr__(self) -> str:
        """String representation of grid."""
        info = self.get_grid_info()
        return f"LNSGrid({self.ndim}D, {info['total_cells']} cells, {self.grid_type})"
    
    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [f"LNS Grid ({self.ndim}D {self.grid_type})"]
        lines.append("=" * 30)
        
        info = self.get_grid_info()
        for key, value in info.items():
            if key not in ['ndim', 'grid_type']:
                lines.append(f"{key:12s}: {value}")
                
        lines.append("\nNote: Boundary conditions handled by GhostCellBoundaryHandler")
                
        return "\n".join(lines)