"""
LNS State: State vector management with efficient conversions.

This module provides the LNSState class for managing state vectors in the LNS solver,
including conversions between conservative and primitive variables, validation,
and memory-efficient storage.

Example:
    Create and manage 1D state vector:
    
    >>> grid = LNSGrid.create_uniform_1d(100, 0.0, 1.0)
    >>> state = LNSState(grid, n_variables=5)  # [ρ, m, E, q, σ'xx]
    >>> state.set_from_primitive(density=1.0, velocity=0.1, temperature=300.0)
    
    Convert between variable types:
    
    >>> primitives = state.get_primitive_variables()
    >>> print(f"Density: {primitives['density'][0]:.3f}")
"""

from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
from dataclasses import dataclass
import logging

from lns_solver.core.grid import LNSGrid

logger = logging.getLogger(__name__)

# Type aliases
StateArray = np.ndarray
VariableName = Literal[
    'density', 'momentum_x', 'momentum_y', 'momentum_z', 'energy_total',
    'heat_flux_x', 'heat_flux_y', 'heat_flux_z', 
    'stress_xx', 'stress_yy', 'stress_zz', 'stress_xy', 'stress_xz', 'stress_yz'
]


@dataclass
class MaterialProperties:
    """Material properties for state conversions."""
    gas_constant: float = 287.0      # J/(kg·K) for air
    specific_heat_ratio: float = 1.4  # γ for air
    reference_temperature: float = 300.0  # K
    reference_pressure: float = 101325.0   # Pa
    

class LNSState:
    """
    State vector management for LNS solver with efficient conversions.
    
    This class manages the conservative state vector Q and provides efficient
    conversions to primitive variables. It handles 1D, 2D, and 3D systems
    with proper variable organization and memory management.
    
    Conservative Variables (1D):
        Q = [ρ, ρu, E_total, q_x, σ'_xx]
        
    Conservative Variables (2D):
        Q = [ρ, ρu_x, ρu_y, E_total, q_x, q_y, σ'_xx, σ'_yy, σ'_xy]
        
    Conservative Variables (3D):
        Q = [ρ, ρu_x, ρu_y, ρu_z, E_total, q_x, q_y, q_z, 
             σ'_xx, σ'_yy, σ'_zz, σ'_xy, σ'_xz, σ'_yz]
    
    Attributes:
        grid: Associated computational grid
        Q: Conservative state vector
        n_variables: Number of variables per cell
        material_props: Material properties for conversions
        
    Example:
        >>> grid = LNSGrid.create_uniform_1d(100, 0.0, 1.0)
        >>> state = LNSState(grid, n_variables=5)
        >>> state.initialize_uniform(density=1.0, pressure=101325.0)
    """
    
    def __init__(
        self,
        grid: LNSGrid,
        n_variables: int,
        material_props: Optional[MaterialProperties] = None
    ):
        """
        Initialize LNS state vector.
        
        Args:
            grid: Computational grid
            n_variables: Number of variables per cell
            material_props: Material properties for conversions
            
        Raises:
            ValueError: If n_variables doesn't match expected for grid dimension
        """
        self.grid = grid
        self.n_variables = n_variables
        self.material_props = material_props or MaterialProperties()
        
        # Validate variable count for dimension
        expected_vars = self._get_expected_variable_count()
        if n_variables != expected_vars:
            logger.warning(
                f"Expected {expected_vars} variables for {grid.ndim}D, got {n_variables}"
            )
        
        # Initialize state vector
        total_cells = self._get_total_cells()
        self.Q = np.zeros((total_cells, n_variables))
        
        # Variable indexing for easy access
        self._setup_variable_indices()
        
        logger.info(f"Initialized {grid.ndim}D state with {n_variables} variables")
        
    def _get_expected_variable_count(self) -> int:
        """Get expected number of variables for grid dimension."""
        if self.grid.ndim == 1:
            return 5  # [ρ, ρu, E, q_x, σ'_xx]
        elif self.grid.ndim == 2:
            return 9  # [ρ, ρu_x, ρu_y, E, q_x, q_y, σ'_xx, σ'_yy, σ'_xy]
        else:  # 3D
            return 13  # [ρ, ρu_x, ρu_y, ρu_z, E, q_x, q_y, q_z, σ'_xx, σ'_yy, σ'_zz, σ'_xy, σ'_xz, σ'_yz]
    
    def _get_total_cells(self) -> int:
        """Get total number of cells in grid."""
        return self.grid.nx * max(1, self.grid.ny) * max(1, self.grid.nz)
    
    def _setup_variable_indices(self) -> None:
        """Setup indices for easy variable access."""
        self.var_indices = {}
        
        # Universal variables
        self.var_indices['density'] = 0
        
        if self.grid.ndim == 1:
            self.var_indices.update({
                'momentum_x': 1,
                'energy_total': 2,
                'heat_flux_x': 3,
                'stress_xx': 4,
            })
        elif self.grid.ndim == 2:
            self.var_indices.update({
                'momentum_x': 1,
                'momentum_y': 2,
                'energy_total': 3,
                'heat_flux_x': 4,
                'heat_flux_y': 5,
                'stress_xx': 6,
                'stress_yy': 7,
                'stress_xy': 8,
            })
        else:  # 3D
            self.var_indices.update({
                'momentum_x': 1,
                'momentum_y': 2,
                'momentum_z': 3,
                'energy_total': 4,
                'heat_flux_x': 5,
                'heat_flux_y': 6,
                'heat_flux_z': 7,
                'stress_xx': 8,
                'stress_yy': 9,
                'stress_zz': 10,
                'stress_xy': 11,
                'stress_xz': 12,
                'stress_yz': 13,
            })
    
    def get_variable(self, name: VariableName) -> StateArray:
        """
        Get a specific variable from the state vector.
        
        Args:
            name: Variable name
            
        Returns:
            Array of variable values for all cells
            
        Example:
            >>> density = state.get_variable('density')
            >>> print(f"Max density: {np.max(density):.3f}")
        """
        if name not in self.var_indices:
            raise ValueError(f"Variable '{name}' not available for {self.grid.ndim}D")
        
        idx = self.var_indices[name]
        return self.Q[:, idx].copy()
    
    def set_variable(self, name: VariableName, values: Union[float, StateArray]) -> None:
        """
        Set a specific variable in the state vector.
        
        Args:
            name: Variable name
            values: Scalar value or array of values
            
        Example:
            >>> state.set_variable('density', 1.2)  # Uniform density
            >>> state.set_variable('heat_flux_x', flux_array)  # From array
        """
        if name not in self.var_indices:
            raise ValueError(f"Variable '{name}' not available for {self.grid.ndim}D")
        
        idx = self.var_indices[name]
        
        if isinstance(values, (int, float)):
            self.Q[:, idx] = values
        else:
            if len(values) != len(self.Q):
                raise ValueError(f"Array size {len(values)} doesn't match cells {len(self.Q)}")
            self.Q[:, idx] = values
    
    def Q_to_P_1d(self, Q_conservative: StateArray) -> Dict[str, float]:
        """
        Convert 1D conservative to primitive variables.
        
        Args:
            Q_conservative: Conservative state [ρ, ρu, E, q_x, σ'_xx]
            
        Returns:
            Dictionary with primitive variables
            
        Example:
            >>> Q = np.array([1.0, 0.1, 253125.0, 0.0, 0.0])
            >>> P = state.Q_to_P_1d(Q)
            >>> print(f"Temperature: {P['temperature']:.1f} K")
        """
        rho = Q_conservative[0]
        rho_u = Q_conservative[1]
        E_total = Q_conservative[2]
        q_x = Q_conservative[3]
        sigma_xx = Q_conservative[4]
        
        # Validate physical realizability
        if rho <= 0:
            raise ValueError(f"Non-physical density: {rho}")
        
        # Primitive variables
        u_x = rho_u / rho
        
        # Internal energy
        kinetic_energy = 0.5 * rho * u_x**2
        e_internal = E_total - kinetic_energy
        
        if e_internal <= 0:
            raise ValueError(f"Non-physical internal energy: {e_internal}")
        
        # Temperature from internal energy (ideal gas)
        cv = self.material_props.gas_constant / (self.material_props.specific_heat_ratio - 1)
        temperature = e_internal / (rho * cv)
        
        # Pressure from ideal gas law
        pressure = rho * self.material_props.gas_constant * temperature
        
        return {
            'density': rho,
            'velocity': u_x,
            'pressure': pressure,
            'temperature': temperature,
            'heat_flux_x': q_x,
            'stress_xx': sigma_xx,
        }
    
    def P_to_Q_1d(self, primitives: Dict[str, float]) -> StateArray:
        """
        Convert 1D primitive to conservative variables.
        
        Args:
            primitives: Dictionary with primitive variables
            
        Returns:
            Conservative state vector
            
        Example:
            >>> P = {'density': 1.0, 'velocity': 0.1, 'temperature': 300.0}
            >>> Q = state.P_to_Q_1d(P)
        """
        rho = primitives['density']
        u_x = primitives['velocity']
        T = primitives['temperature']
        q_x = primitives.get('heat_flux_x', 0.0)
        sigma_xx = primitives.get('stress_xx', 0.0)
        
        # Conservative variables
        rho_u = rho * u_x
        
        # Internal energy
        cv = self.material_props.gas_constant / (self.material_props.specific_heat_ratio - 1)
        e_internal = rho * cv * T
        
        # Total energy
        kinetic_energy = 0.5 * rho * u_x**2
        E_total = e_internal + kinetic_energy
        
        return np.array([rho, rho_u, E_total, q_x, sigma_xx])
    
    def Q_to_P_2d(self, Q_conservative: StateArray) -> Dict[str, float]:
        """
        Convert 2D conservative to primitive variables.
        
        Args:
            Q_conservative: Conservative state [ρ, ρu_x, ρu_y, E, q_x, q_y, σ'_xx, σ'_yy, σ'_xy]
            
        Returns:
            Dictionary with primitive variables
        """
        rho = Q_conservative[0]
        rho_u_x = Q_conservative[1]
        rho_u_y = Q_conservative[2]
        E_total = Q_conservative[3]
        q_x = Q_conservative[4]
        q_y = Q_conservative[5]
        sigma_xx = Q_conservative[6]
        sigma_yy = Q_conservative[7]
        sigma_xy = Q_conservative[8]
        
        # Validate physical realizability
        if rho <= 0:
            raise ValueError(f"Non-physical density: {rho}")
        
        # Primitive variables
        u_x = rho_u_x / rho
        u_y = rho_u_y / rho
        
        # Internal energy
        kinetic_energy = 0.5 * rho * (u_x**2 + u_y**2)
        e_internal = E_total - kinetic_energy
        
        if e_internal <= 0:
            raise ValueError(f"Non-physical internal energy: {e_internal}")
        
        # Temperature and pressure
        cv = self.material_props.gas_constant / (self.material_props.specific_heat_ratio - 1)
        temperature = e_internal / (rho * cv)
        pressure = rho * self.material_props.gas_constant * temperature
        
        return {
            'density': rho,
            'velocity_x': u_x,
            'velocity_y': u_y,
            'pressure': pressure,
            'temperature': temperature,
            'heat_flux_x': q_x,
            'heat_flux_y': q_y,
            'stress_xx': sigma_xx,
            'stress_yy': sigma_yy,
            'stress_xy': sigma_xy,
        }
    
    def P_to_Q_2d(self, primitives: Dict[str, float]) -> StateArray:
        """
        Convert 2D primitive to conservative variables.
        
        Args:
            primitives: Dictionary with primitive variables
            
        Returns:
            Conservative state vector
        """
        rho = primitives['density']
        u_x = primitives['velocity_x']
        u_y = primitives['velocity_y']
        T = primitives['temperature']
        q_x = primitives.get('heat_flux_x', 0.0)
        q_y = primitives.get('heat_flux_y', 0.0)
        sigma_xx = primitives.get('stress_xx', 0.0)
        sigma_yy = primitives.get('stress_yy', 0.0)
        sigma_xy = primitives.get('stress_xy', 0.0)
        
        # Conservative variables
        rho_u_x = rho * u_x
        rho_u_y = rho * u_y
        
        # Internal energy
        cv = self.material_props.gas_constant / (self.material_props.specific_heat_ratio - 1)
        e_internal = rho * cv * T
        
        # Total energy
        kinetic_energy = 0.5 * rho * (u_x**2 + u_y**2)
        E_total = e_internal + kinetic_energy
        
        return np.array([rho, rho_u_x, rho_u_y, E_total, q_x, q_y, 
                        sigma_xx, sigma_yy, sigma_xy])
    
    def get_primitive_variables(self) -> Dict[str, StateArray]:
        """
        Get primitive variables for all cells.
        
        Returns:
            Dictionary with arrays of primitive variables
            
        Example:
            >>> primitives = state.get_primitive_variables()
            >>> avg_temp = np.mean(primitives['temperature'])
        """
        primitives = {}
        
        if self.grid.ndim == 1:
            # Vectorized conversion for all cells
            rho = self.Q[:, 0]
            rho_u = self.Q[:, 1]
            E_total = self.Q[:, 2]
            
            # Validate
            if np.any(rho <= 0):
                raise ValueError("Non-physical density detected")
            
            u_x = rho_u / rho
            kinetic_energy = 0.5 * rho * u_x**2
            e_internal = E_total - kinetic_energy
            
            if np.any(e_internal <= -1e-3):  # Allow small negative values for numerical stability
                raise ValueError("Non-physical internal energy detected")
            
            cv = self.material_props.gas_constant / (self.material_props.specific_heat_ratio - 1)
            temperature = e_internal / (rho * cv)
            pressure = rho * self.material_props.gas_constant * temperature
            
            primitives = {
                'density': rho,
                'velocity': u_x,
                'pressure': pressure,
                'temperature': temperature,
                'heat_flux_x': self.Q[:, 3],
                'stress_xx': self.Q[:, 4],
            }
            
        elif self.grid.ndim == 2:
            # 2D vectorized conversion
            rho = self.Q[:, 0]
            rho_u_x = self.Q[:, 1]
            rho_u_y = self.Q[:, 2]
            E_total = self.Q[:, 3]
            
            if np.any(rho <= 0):
                raise ValueError("Non-physical density detected")
            
            u_x = rho_u_x / rho
            u_y = rho_u_y / rho
            kinetic_energy = 0.5 * rho * (u_x**2 + u_y**2)
            e_internal = E_total - kinetic_energy
            
            if np.any(e_internal <= -1e-3):  # Allow small negative values for numerical stability
                raise ValueError("Non-physical internal energy detected")
            
            cv = self.material_props.gas_constant / (self.material_props.specific_heat_ratio - 1)
            temperature = e_internal / (rho * cv)
            pressure = rho * self.material_props.gas_constant * temperature
            
            primitives = {
                'density': rho,
                'velocity_x': u_x,
                'velocity_y': u_y,
                'pressure': pressure,
                'temperature': temperature,
                'heat_flux_x': self.Q[:, 4],
                'heat_flux_y': self.Q[:, 5],
                'stress_xx': self.Q[:, 6],
                'stress_yy': self.Q[:, 7],
                'stress_xy': self.Q[:, 8],
            }
        
        return primitives
    
    def initialize_uniform(
        self,
        density: float,
        pressure: float,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0
    ) -> None:
        """
        Initialize with uniform primitive variables.
        
        Args:
            density: Uniform density
            pressure: Uniform pressure
            velocity_x, velocity_y, velocity_z: Uniform velocity components
            
        Example:
            >>> state.initialize_uniform(density=1.0, pressure=101325.0)
        """
        # Temperature from ideal gas law
        temperature = pressure / (density * self.material_props.gas_constant)
        
        if self.grid.ndim == 1:
            primitives = {
                'density': density,
                'velocity': velocity_x,
                'temperature': temperature,
                'heat_flux_x': 0.0,
                'stress_xx': 0.0,
            }
            Q_uniform = self.P_to_Q_1d(primitives)
            
        elif self.grid.ndim == 2:
            primitives = {
                'density': density,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y,
                'temperature': temperature,
                'heat_flux_x': 0.0,
                'heat_flux_y': 0.0,
                'stress_xx': 0.0,
                'stress_yy': 0.0,
                'stress_xy': 0.0,
            }
            Q_uniform = self.P_to_Q_2d(primitives)
        
        # Set uniform state for all cells
        for i in range(len(self.Q)):
            self.Q[i, :] = Q_uniform
            
        logger.info(f"Initialized uniform state: ρ={density:.3f}, p={pressure:.0f}, T={temperature:.1f}K")
    
    def initialize_sod_shock_tube(self) -> None:
        """
        Initialize classic Sod shock tube problem.
        
        Left state: ρ=1.0, p=1.0, u=0.0
        Right state: ρ=0.125, p=0.1, u=0.0
        Interface at x=0.5
        """
        if self.grid.ndim != 1:
            raise ValueError("Sod shock tube only defined for 1D")
        
        # Left and right states (using physical units for proper temperatures)
        rho_L, p_L, u_L = 1.0, 101325.0, 0.0  # Standard pressure on left
        rho_R, p_R, u_R = 0.125, 10132.5, 0.0  # 1/10 pressure on right
        
        # Convert to temperatures
        T_L = p_L / (rho_L * self.material_props.gas_constant)
        T_R = p_R / (rho_R * self.material_props.gas_constant)
        
        # Set left and right states
        for i, x in enumerate(self.grid.x):
            if x < 0.5:  # Left state
                primitives = {
                    'density': rho_L,
                    'velocity': u_L,
                    'temperature': T_L,
                    'heat_flux_x': 0.0,
                    'stress_xx': 0.0,
                }
            else:  # Right state
                primitives = {
                    'density': rho_R,
                    'velocity': u_R,
                    'temperature': T_R,
                    'heat_flux_x': 0.0,
                    'stress_xx': 0.0,
                }
            
            self.Q[i, :] = self.P_to_Q_1d(primitives)
            
        logger.info("Initialized Sod shock tube problem")
    
    def validate_state(self, tolerance: float = 1e-10) -> bool:
        """
        Validate physical realizability of state.
        
        Args:
            tolerance: Tolerance for validation checks
            
        Returns:
            True if state is physically realizable
            
        Example:
            >>> is_valid = state.validate_state()
            >>> if not is_valid:
            ...     print("Warning: Non-physical state detected")
        """
        try:
            # Check density
            density = self.Q[:, 0]
            if np.any(density <= tolerance):
                logger.error(f"Non-physical density: min={np.min(density):.2e}")
                return False
            
            # Check internal energy
            primitives = self.get_primitive_variables()
            
            if self.grid.ndim == 1:
                kinetic = 0.5 * density * primitives['velocity']**2
            elif self.grid.ndim == 2:
                kinetic = 0.5 * density * (primitives['velocity_x']**2 + primitives['velocity_y']**2)
            else:
                # 3D case would go here
                kinetic = 0.0
            
            E_total = self.Q[:, self.var_indices['energy_total']]
            e_internal = E_total - kinetic
            
            if np.any(e_internal <= tolerance):
                logger.error(f"Non-physical internal energy: min={np.min(e_internal):.2e}")
                return False
            
            logger.debug("State validation passed")
            return True
            
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            return False
    
    def compute_conserved_quantities(self) -> Dict[str, float]:
        """
        Compute globally conserved quantities.
        
        Returns:
            Dictionary with total mass, momentum, energy
            
        Example:
            >>> conserved = state.compute_conserved_quantities()
            >>> print(f"Total mass: {conserved['mass']:.6f}")
        """
        cell_volumes = self.grid.compute_cell_volumes()
        
        # Mass
        total_mass = np.sum(self.Q[:, 0] * cell_volumes)
        
        # Momentum
        if self.grid.ndim == 1:
            total_momentum_x = np.sum(self.Q[:, 1] * cell_volumes)
            total_momentum_y = 0.0
            total_momentum_z = 0.0
        elif self.grid.ndim == 2:
            total_momentum_x = np.sum(self.Q[:, 1] * cell_volumes)
            total_momentum_y = np.sum(self.Q[:, 2] * cell_volumes)
            total_momentum_z = 0.0
        else:
            total_momentum_x = np.sum(self.Q[:, 1] * cell_volumes)
            total_momentum_y = np.sum(self.Q[:, 2] * cell_volumes)
            total_momentum_z = np.sum(self.Q[:, 3] * cell_volumes)
        
        # Energy
        energy_idx = self.var_indices['energy_total']
        total_energy = np.sum(self.Q[:, energy_idx] * cell_volumes)
        
        return {
            'mass': total_mass,
            'momentum_x': total_momentum_x,
            'momentum_y': total_momentum_y,
            'momentum_z': total_momentum_z,
            'energy': total_energy,
        }
    
    def copy(self) -> 'LNSState':
        """Create a deep copy of the state."""
        new_state = LNSState(self.grid, self.n_variables, self.material_props)
        new_state.Q = self.Q.copy()
        return new_state
    
    def __repr__(self) -> str:
        """String representation of state."""
        return f"LNSState({self.grid.ndim}D, {self.n_variables} vars, {len(self.Q)} cells)"
    
    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [f"LNS State ({self.grid.ndim}D)"]
        lines.append("=" * 20)
        lines.append(f"Variables: {self.n_variables}")
        lines.append(f"Cells: {len(self.Q)}")
        
        # Add range information for key variables
        try:
            primitives = self.get_primitive_variables()
            lines.append("\nVariable Ranges:")
            for var_name in ['density', 'pressure', 'temperature']:
                if var_name in primitives:
                    values = primitives[var_name]
                    lines.append(f"  {var_name:12s}: [{np.min(values):.3e}, {np.max(values):.3e}]")
        except:
            lines.append("\nVariable ranges: Unable to compute")
        
        return "\n".join(lines)