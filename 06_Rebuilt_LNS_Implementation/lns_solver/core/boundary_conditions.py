"""
Ghost Cell-Based Boundary Conditions for Finite Volume Methods.

This module implements proper FVM boundary conditions using ghost cells
that preserve conservation and enable correct flux computation at boundaries.

The key principle: boundary conditions are applied to ghost cells OUTSIDE
the computational domain, not to physical cells within the domain.
"""

import numpy as np
from typing import Dict, Optional, Union, List
from enum import Enum
from dataclasses import dataclass

from lns_solver.utils.constants import PhysicalConstants


class BCType(Enum):
    """Boundary condition types."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann" 
    OUTFLOW = "outflow"
    PERIODIC = "periodic"
    WALL = "wall"
    ISOTHERMAL_WALL = "isothermal_wall"  # Fixed temperature wall
    ADIABATIC_WALL = "adiabatic_wall"    # Zero heat flux wall
    MOVING_WALL = "moving_wall"          # Wall with specified velocity
    INFLOW = "inflow"


@dataclass
class BoundaryCondition:
    """
    ENHANCED boundary condition specification with complete wall support.
    
    For ghost cell-based FVM, the BC is applied to ghost cells
    to ensure proper flux computation at boundaries.
    
    Supports:
    - Isothermal walls: Fixed temperature T_wall
    - Adiabatic walls: Zero heat flux ∂T/∂n = 0
    - Moving walls: Specified wall velocity u_wall
    - Mixed thermal/velocity conditions
    """
    bc_type: BCType
    values: Optional[Union[float, np.ndarray, Dict[str, float]]] = None
    gradient: Optional[float] = None  # For Neumann BCs
    wall_temperature: Optional[float] = None  # For isothermal walls
    wall_velocity: Optional[float] = None     # For moving walls
    thermal_condition: Optional[str] = None   # 'isothermal' or 'adiabatic'
    
    def __post_init__(self):
        """Validate BC specification."""
        if self.bc_type in [BCType.DIRICHLET, BCType.INFLOW] and self.values is None:
            raise ValueError(f"{self.bc_type} BC requires values")
        if self.bc_type == BCType.NEUMANN and self.gradient is None:
            raise ValueError("Neumann BC requires gradient")
        if self.bc_type == BCType.ISOTHERMAL_WALL and self.wall_temperature is None:
            raise ValueError("Isothermal wall BC requires wall_temperature")
        if self.bc_type == BCType.MOVING_WALL and self.wall_velocity is None:
            raise ValueError("Moving wall BC requires wall_velocity")
        
        # Set defaults for wall conditions
        if self.bc_type in [BCType.WALL, BCType.ISOTHERMAL_WALL, BCType.ADIABATIC_WALL, BCType.MOVING_WALL]:
            if self.wall_velocity is None:
                self.wall_velocity = 0.0  # Stationary wall by default
            if self.thermal_condition is None:
                if self.bc_type == BCType.ISOTHERMAL_WALL:
                    self.thermal_condition = 'isothermal'
                elif self.bc_type == BCType.ADIABATIC_WALL:
                    self.thermal_condition = 'adiabatic'
                else:
                    self.thermal_condition = 'adiabatic'  # Default for generic WALL


class GhostCellBoundaryHandler:
    """
    Handles ghost cell population for finite volume boundary conditions.
    
    This class properly implements FVM boundary conditions by:
    1. Managing ghost cell layers outside the physical domain
    2. Populating ghost cells based on BC type and physical cell data
    3. Preserving conservation properties at boundaries
    
    The ghost cell approach ensures that:
    - Physical cells are never overwritten by BCs
    - Boundary fluxes are computed correctly
    - Conservation is maintained exactly
    """
    
    def __init__(self, n_ghost: int = 2):
        """
        Initialize ghost cell handler.
        
        Args:
            n_ghost: Number of ghost cell layers (typically 2 for 2nd order)
        """
        self.n_ghost = n_ghost
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
    
    def set_boundary_condition(self, location: str, bc: BoundaryCondition):
        """
        Set boundary condition for a domain boundary.
        
        Args:
            location: Boundary location ('left', 'right', 'bottom', 'top', etc.)
            bc: Boundary condition specification
        """
        self.boundary_conditions[location] = bc
    
    def create_ghost_state(self, Q_physical: np.ndarray, grid_shape: tuple) -> np.ndarray:
        """
        Create state array with ghost cells from physical state.
        
        Args:
            Q_physical: Physical state array [nx, n_vars] or [ny, nx, n_vars]
            grid_shape: Grid dimensions (nx,) or (ny, nx)
            
        Returns:
            Ghost state array with padding for ghost cells
        """
        if len(grid_shape) == 1:
            # 1D case
            nx = grid_shape[0]
            n_vars = Q_physical.shape[1]
            Q_ghost = np.zeros((nx + 2*self.n_ghost, n_vars))
            
            # Copy physical cells to interior of ghost array
            Q_ghost[self.n_ghost:-self.n_ghost, :] = Q_physical
            
        elif len(grid_shape) == 2:
            # 2D case  
            ny, nx = grid_shape
            n_vars = Q_physical.shape[2]
            Q_ghost = np.zeros((ny + 2*self.n_ghost, nx + 2*self.n_ghost, n_vars))
            
            # Copy physical cells to interior
            Q_ghost[self.n_ghost:-self.n_ghost, self.n_ghost:-self.n_ghost, :] = Q_physical
            
        else:
            raise ValueError(f"Unsupported grid dimension: {len(grid_shape)}")
        
        return Q_ghost
    
    def apply_boundary_conditions_1d(
        self, 
        Q_ghost: np.ndarray,
        dx: float,
        primitive_vars: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Apply boundary conditions to 1D ghost cells.
        
        Args:
            Q_ghost: State with ghost cells [nx_ghost, n_vars]
            dx: Grid spacing
            primitive_vars: Pre-computed primitive variables for efficiency
        """
        # CRITICAL FIX: Handle periodic boundaries first (must be applied to both ends together)
        left_bc = self.boundary_conditions.get('left')
        right_bc = self.boundary_conditions.get('right')
        
        if (left_bc and left_bc.bc_type == BCType.PERIODIC and
            right_bc and right_bc.bc_type == BCType.PERIODIC):
            # Apply periodic BC to both ends together
            self.apply_periodic_bc_1d(Q_ghost)
        else:
            # Apply individual boundary conditions
            if 'left' in self.boundary_conditions:
                bc = self.boundary_conditions['left']
                self._apply_left_bc_1d(Q_ghost, bc, dx, primitive_vars)
            
            if 'right' in self.boundary_conditions:
                bc = self.boundary_conditions['right']
                self._apply_right_bc_1d(Q_ghost, bc, dx, primitive_vars)
    
    def _apply_left_bc_1d(
        self,
        Q_ghost: np.ndarray, 
        bc: BoundaryCondition,
        dx: float,
        primitive_vars: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """Apply left boundary condition to ghost cells."""
        
        # Physical cell indices (after ghost padding)
        phys_start = self.n_ghost
        
        if bc.bc_type == BCType.OUTFLOW:
            # Extrapolate from first physical cell
            for g in range(self.n_ghost):
                Q_ghost[g, :] = Q_ghost[phys_start, :]
                
        elif bc.bc_type == BCType.DIRICHLET:
            # Set ghost cells to maintain desired boundary value
            # For temperature BC, need to compute corresponding conservative state
            if isinstance(bc.values, (int, float)):
                # Temperature boundary condition
                T_bc = bc.values
                for g in range(self.n_ghost):
                    # Mirror approach: set ghost cell to enforce BC at interface
                    rho_phys = Q_ghost[phys_start, 0] 
                    u_phys = Q_ghost[phys_start, 1] / rho_phys if rho_phys > 0 else 0.0
                    
                    # Compute total energy for BC temperature
                    cv = PhysicalConstants.AIR_GAS_CONSTANT / (PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO - 1)
                    e_internal = rho_phys * cv * T_bc
                    kinetic = 0.5 * rho_phys * u_phys**2
                    
                    Q_ghost[g, 0] = rho_phys  # Same density
                    Q_ghost[g, 1] = rho_phys * u_phys  # Same velocity  
                    Q_ghost[g, 2] = e_internal + kinetic  # BC temperature
                    
                    # Heat flux and stress: extrapolate
                    if Q_ghost.shape[1] > 3:
                        Q_ghost[g, 3:] = Q_ghost[phys_start, 3:]
                        
        elif bc.bc_type in [BCType.WALL, BCType.ISOTHERMAL_WALL, BCType.ADIABATIC_WALL, BCType.MOVING_WALL]:
            # COMPLETE wall boundary condition implementation
            self._apply_wall_bc_left(Q_ghost, bc, phys_start)
                    
        elif bc.bc_type == BCType.PERIODIC:
            # Will be handled in pair with right boundary
            pass
    
    def _apply_wall_bc_left(
        self, 
        Q_ghost: np.ndarray, 
        bc: BoundaryCondition, 
        phys_start: int
    ) -> None:
        """
        COMPLETE wall boundary condition implementation for left boundary.
        
        Supports:
        - Isothermal walls: T_ghost = 2*T_wall - T_phys (linear interpolation)
        - Adiabatic walls: T_ghost = T_phys (zero gradient)
        - Moving walls: u_ghost = 2*u_wall - u_phys (enforce wall velocity)
        - Stationary walls: u_ghost = -u_phys (no-slip, u=0 at interface)
        
        Args:
            Q_ghost: Ghost state array
            bc: Boundary condition specification
            phys_start: Index of first physical cell
        """
        for g in range(self.n_ghost):
            # Start with physical cell data
            Q_ghost[g, :] = Q_ghost[phys_start, :]
            
            # Extract physical state
            rho_phys = Q_ghost[phys_start, 0]
            if rho_phys <= 0:
                continue  # Skip invalid cells
                
            u_phys = Q_ghost[phys_start, 1] / rho_phys
            E_phys = Q_ghost[phys_start, 2]
            
            # === VELOCITY BOUNDARY CONDITION ===
            wall_velocity = bc.wall_velocity if bc.wall_velocity is not None else 0.0
            
            if wall_velocity == 0.0:
                # Stationary wall: enforce u = 0 at interface
                # Ghost cell velocity: u_ghost = -u_phys (reflection)
                u_ghost = -u_phys
            else:
                # Moving wall: enforce u = u_wall at interface
                # Ghost cell velocity: u_ghost = 2*u_wall - u_phys
                u_ghost = 2.0 * wall_velocity - u_phys
            
            # Update momentum in ghost cell
            Q_ghost[g, 1] = rho_phys * u_ghost
            
            # === THERMAL BOUNDARY CONDITION ===
            thermal_condition = getattr(bc, 'thermal_condition', 'adiabatic')
            
            if thermal_condition == 'isothermal' and bc.wall_temperature is not None:
                # Isothermal wall: enforce T = T_wall at interface
                # Ghost temperature: T_ghost = 2*T_wall - T_phys
                
                # Compute physical temperature
                gamma = PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO
                R = PhysicalConstants.AIR_GAS_CONSTANT
                kinetic_phys = 0.5 * rho_phys * u_phys**2
                internal_phys = E_phys - kinetic_phys
                p_phys = (gamma - 1) * internal_phys
                T_phys = p_phys / (rho_phys * R)
                
                # Ghost temperature for isothermal BC
                T_ghost = 2.0 * bc.wall_temperature - T_phys
                T_ghost = max(T_ghost, 0.1 * bc.wall_temperature)  # Prevent negative T
                
                # Recompute ghost energy with isothermal temperature
                p_ghost = rho_phys * R * T_ghost
                internal_ghost = p_ghost / (gamma - 1)
                kinetic_ghost = 0.5 * rho_phys * u_ghost**2
                E_ghost = internal_ghost + kinetic_ghost
                
                Q_ghost[g, 2] = E_ghost
                
            else:
                # Adiabatic wall: zero temperature gradient (∂T/∂n = 0)
                # Ghost temperature: T_ghost = T_phys (simple extrapolation)
                # Energy needs to account for different velocity
                kinetic_phys = 0.5 * rho_phys * u_phys**2
                kinetic_ghost = 0.5 * rho_phys * u_ghost**2
                internal_energy = E_phys - kinetic_phys  # Extract internal energy
                E_ghost = internal_energy + kinetic_ghost  # Recompute total energy
                
                Q_ghost[g, 2] = E_ghost
            
            # === LNS VARIABLES (if present) ===
            if Q_ghost.shape[1] > 3:
                # Heat flux and stress: depends on thermal condition
                if thermal_condition == 'isothermal':
                    # For isothermal wall, heat flux may be non-zero
                    # Extrapolate from physical cell for now
                    Q_ghost[g, 3:] = Q_ghost[phys_start, 3:]
                else:
                    # For adiabatic wall, heat flux should be zero at wall
                    # Set ghost heat flux to enforce zero flux at interface
                    if Q_ghost.shape[1] > 3:
                        Q_ghost[g, 3] = -Q_ghost[phys_start, 3]  # Reflect heat flux
                    if Q_ghost.shape[1] > 4:
                        # Stress: extrapolate (stress is non-zero at walls)
                        Q_ghost[g, 4:] = Q_ghost[phys_start, 4:]
    
    def _apply_right_bc_1d(
        self,
        Q_ghost: np.ndarray,
        bc: BoundaryCondition, 
        dx: float,
        primitive_vars: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """Apply right boundary condition to ghost cells."""
        
        # Physical cell indices
        phys_end = Q_ghost.shape[0] - self.n_ghost - 1
        
        if bc.bc_type == BCType.OUTFLOW:
            # Extrapolate from last physical cell
            for g in range(self.n_ghost):
                ghost_idx = Q_ghost.shape[0] - 1 - g
                Q_ghost[ghost_idx, :] = Q_ghost[phys_end, :]
                
        elif bc.bc_type == BCType.DIRICHLET:
            # Temperature BC similar to left boundary
            if isinstance(bc.values, (int, float)):
                T_bc = bc.values
                for g in range(self.n_ghost):
                    ghost_idx = Q_ghost.shape[0] - 1 - g
                    
                    rho_phys = Q_ghost[phys_end, 0]
                    u_phys = Q_ghost[phys_end, 1] / rho_phys if rho_phys > 0 else 0.0
                    
                    cv = PhysicalConstants.AIR_GAS_CONSTANT / (PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO - 1)
                    e_internal = rho_phys * cv * T_bc
                    kinetic = 0.5 * rho_phys * u_phys**2
                    
                    Q_ghost[ghost_idx, 0] = rho_phys
                    Q_ghost[ghost_idx, 1] = rho_phys * u_phys
                    Q_ghost[ghost_idx, 2] = e_internal + kinetic
                    
                    if Q_ghost.shape[1] > 3:
                        Q_ghost[ghost_idx, 3:] = Q_ghost[phys_end, 3:]
                        
        elif bc.bc_type in [BCType.WALL, BCType.ISOTHERMAL_WALL, BCType.ADIABATIC_WALL, BCType.MOVING_WALL]:
            # COMPLETE wall boundary condition implementation
            self._apply_wall_bc_right(Q_ghost, bc, phys_end)
    
    def _apply_wall_bc_right(
        self, 
        Q_ghost: np.ndarray, 
        bc: BoundaryCondition, 
        phys_end: int
    ) -> None:
        """
        COMPLETE wall boundary condition implementation for right boundary.
        
        Mirrors the left boundary implementation with appropriate indexing.
        
        Args:
            Q_ghost: Ghost state array
            bc: Boundary condition specification
            phys_end: Index of last physical cell
        """
        for g in range(self.n_ghost):
            ghost_idx = Q_ghost.shape[0] - 1 - g
            
            # Start with physical cell data
            Q_ghost[ghost_idx, :] = Q_ghost[phys_end, :]
            
            # Extract physical state
            rho_phys = Q_ghost[phys_end, 0]
            if rho_phys <= 0:
                continue
                
            u_phys = Q_ghost[phys_end, 1] / rho_phys
            E_phys = Q_ghost[phys_end, 2]
            
            # === VELOCITY BOUNDARY CONDITION ===
            wall_velocity = bc.wall_velocity if bc.wall_velocity is not None else 0.0
            
            if wall_velocity == 0.0:
                # Stationary wall: enforce u = 0 at interface
                u_ghost = -u_phys
            else:
                # Moving wall: enforce u = u_wall at interface
                u_ghost = 2.0 * wall_velocity - u_phys
            
            # Update momentum in ghost cell
            Q_ghost[ghost_idx, 1] = rho_phys * u_ghost
            
            # === THERMAL BOUNDARY CONDITION ===
            thermal_condition = getattr(bc, 'thermal_condition', 'adiabatic')
            
            if thermal_condition == 'isothermal' and bc.wall_temperature is not None:
                # Isothermal wall: enforce T = T_wall at interface
                
                # Compute physical temperature
                gamma = PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO
                R = PhysicalConstants.AIR_GAS_CONSTANT
                kinetic_phys = 0.5 * rho_phys * u_phys**2
                internal_phys = E_phys - kinetic_phys
                p_phys = (gamma - 1) * internal_phys
                T_phys = p_phys / (rho_phys * R)
                
                # Ghost temperature for isothermal BC
                T_ghost = 2.0 * bc.wall_temperature - T_phys
                T_ghost = max(T_ghost, 0.1 * bc.wall_temperature)  # Prevent negative T
                
                # Recompute ghost energy with isothermal temperature
                p_ghost = rho_phys * R * T_ghost
                internal_ghost = p_ghost / (gamma - 1)
                kinetic_ghost = 0.5 * rho_phys * u_ghost**2
                E_ghost = internal_ghost + kinetic_ghost
                
                Q_ghost[ghost_idx, 2] = E_ghost
                
            else:
                # Adiabatic wall: zero temperature gradient
                kinetic_phys = 0.5 * rho_phys * u_phys**2
                kinetic_ghost = 0.5 * rho_phys * u_ghost**2
                internal_energy = E_phys - kinetic_phys
                E_ghost = internal_energy + kinetic_ghost
                
                Q_ghost[ghost_idx, 2] = E_ghost
            
            # === LNS VARIABLES (if present) ===
            if Q_ghost.shape[1] > 3:
                # Heat flux and stress: depends on thermal condition
                if thermal_condition == 'isothermal':
                    # For isothermal wall, heat flux may be non-zero
                    Q_ghost[ghost_idx, 3:] = Q_ghost[phys_end, 3:]
                else:
                    # For adiabatic wall, heat flux should be zero at wall
                    if Q_ghost.shape[1] > 3:
                        Q_ghost[ghost_idx, 3] = -Q_ghost[phys_end, 3]  # Reflect heat flux
                    if Q_ghost.shape[1] > 4:
                        # Stress: extrapolate (stress is non-zero at walls)
                        Q_ghost[ghost_idx, 4:] = Q_ghost[phys_end, 4:]
    
    def apply_periodic_bc_1d(self, Q_ghost: np.ndarray) -> None:
        """Apply periodic boundary conditions to both ends."""
        
        # Check if both boundaries are periodic
        left_bc = self.boundary_conditions.get('left')
        right_bc = self.boundary_conditions.get('right') 
        
        if (left_bc and left_bc.bc_type == BCType.PERIODIC and
            right_bc and right_bc.bc_type == BCType.PERIODIC):
            
            # Left ghost cells get data from right physical cells
            phys_start = self.n_ghost
            phys_end = Q_ghost.shape[0] - self.n_ghost
            
            # CORRECTED periodic BC indexing
            # Left ghost cells get data from RIGHT physical cells (wraparound)
            # Right ghost cells get data from LEFT physical cells (wraparound)
            
            # Copy last n_ghost physical cells to left ghost cells
            Q_ghost[0:self.n_ghost, :] = Q_ghost[phys_end-self.n_ghost:phys_end, :]
            
            # Copy first n_ghost physical cells to right ghost cells  
            Q_ghost[phys_end:phys_end+self.n_ghost, :] = Q_ghost[phys_start:phys_start+self.n_ghost, :]
    
    def extract_physical_state(self, Q_ghost: np.ndarray, grid_shape: tuple) -> np.ndarray:
        """
        Extract physical state from ghost state array.
        
        Args:
            Q_ghost: State with ghost cells
            grid_shape: Original grid dimensions
            
        Returns:
            Physical state array without ghost cells
        """
        if len(grid_shape) == 1:
            return Q_ghost[self.n_ghost:-self.n_ghost, :]
        elif len(grid_shape) == 2:
            return Q_ghost[self.n_ghost:-self.n_ghost, self.n_ghost:-self.n_ghost, :]
        else:
            raise ValueError(f"Unsupported grid dimension: {len(grid_shape)}")


def create_outflow_bc() -> BoundaryCondition:
    """Create outflow boundary condition."""
    return BoundaryCondition(BCType.OUTFLOW)


def create_wall_bc() -> BoundaryCondition:
    """Create basic adiabatic, stationary wall boundary condition.""" 
    return BoundaryCondition(
        BCType.ADIABATIC_WALL,
        wall_velocity=0.0,
        thermal_condition='adiabatic'
    )


def create_isothermal_wall_bc(wall_temperature: float, wall_velocity: float = 0.0) -> BoundaryCondition:
    """
    Create isothermal wall boundary condition.
    
    Args:
        wall_temperature: Fixed temperature at the wall [K]
        wall_velocity: Wall velocity (0.0 for stationary wall) [m/s]
        
    Returns:
        Isothermal wall boundary condition
    """
    return BoundaryCondition(
        BCType.ISOTHERMAL_WALL,
        wall_temperature=wall_temperature,
        wall_velocity=wall_velocity,
        thermal_condition='isothermal'
    )


def create_adiabatic_wall_bc(wall_velocity: float = 0.0) -> BoundaryCondition:
    """
    Create adiabatic wall boundary condition.
    
    Args:
        wall_velocity: Wall velocity (0.0 for stationary wall) [m/s]
        
    Returns:
        Adiabatic wall boundary condition with zero heat flux
    """
    return BoundaryCondition(
        BCType.ADIABATIC_WALL,
        wall_velocity=wall_velocity,
        thermal_condition='adiabatic'
    )


def create_moving_wall_bc(
    wall_velocity: float, 
    thermal_condition: str = 'adiabatic',
    wall_temperature: Optional[float] = None
) -> BoundaryCondition:
    """
    Create moving wall boundary condition.
    
    Args:
        wall_velocity: Wall velocity [m/s]
        thermal_condition: 'adiabatic' or 'isothermal'
        wall_temperature: Fixed temperature if isothermal [K]
        
    Returns:
        Moving wall boundary condition
    """
    bc_type = BCType.ISOTHERMAL_WALL if thermal_condition == 'isothermal' else BCType.MOVING_WALL
    
    if thermal_condition == 'isothermal' and wall_temperature is None:
        raise ValueError("Isothermal moving wall requires wall_temperature")
    
    return BoundaryCondition(
        bc_type,
        wall_velocity=wall_velocity,
        wall_temperature=wall_temperature,
        thermal_condition=thermal_condition
    )


def create_temperature_bc(temperature: float) -> BoundaryCondition:
    """Create temperature (Dirichlet) boundary condition."""
    return BoundaryCondition(BCType.DIRICHLET, values=temperature)


def create_periodic_bc() -> BoundaryCondition:
    """Create periodic boundary condition."""
    return BoundaryCondition(BCType.PERIODIC)


# Example usage and testing
if __name__ == "__main__":
    # Test 1D ghost cell handling
    print("Testing 1D Ghost Cell Boundary Conditions")
    
    # Create handler
    bc_handler = GhostCellBoundaryHandler(n_ghost=2)
    
    # Set boundary conditions
    bc_handler.set_boundary_condition('left', create_outflow_bc())
    bc_handler.set_boundary_condition('right', create_temperature_bc(350.0))
    
    # Create test physical state
    nx = 10
    n_vars = 5
    Q_physical = np.random.rand(nx, n_vars)
    Q_physical[:, 0] = 1.0  # Set reasonable density
    
    # Create and populate ghost state
    Q_ghost = bc_handler.create_ghost_state(Q_physical, (nx,))
    bc_handler.apply_boundary_conditions_1d(Q_ghost, dx=0.1)
    
    print(f"Physical shape: {Q_physical.shape}")
    print(f"Ghost shape: {Q_ghost.shape}")
    print(f"Left ghost cells (outflow): \n{Q_ghost[:2, :3]}")
    print(f"Right ghost cells (T=350K): \n{Q_ghost[-2:, :3]}")
    
    # Extract physical state
    Q_extracted = bc_handler.extract_physical_state(Q_ghost, (nx,))
    print(f"Extraction preserves data: {np.allclose(Q_physical, Q_extracted)}")