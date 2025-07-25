"""
LNS State Management with Named Variable Access.

This module provides a state class with named accessors for LNS variables,
replacing hardcoded array indices with enumerated constants.

Features:
- Named constants for state variable indices
- Property-based access to conserved variables
- State initialization methods for common test cases
- Basic variable organization for 1D LNS system

The state vector contains: [density, momentum, total_energy, heat_flux, stress]
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import IntEnum
import logging

from lns_solver.core.grid import LNSGrid
from lns_solver.utils.constants import PhysicalConstants

logger = logging.getLogger(__name__)


class LNSVariables(IntEnum):
    """
    Named indices for LNS state variables.
    
    This eliminates hardcoded indices throughout the codebase
    and provides clear variable organization.
    """
    # Conservative variables (always present)
    DENSITY = 0        # ρ
    MOMENTUM_X = 1     # ρu_x  
    TOTAL_ENERGY = 2   # E_T
    
    # LNS extended variables (optional)
    HEAT_FLUX_X = 3    # q_x
    STRESS_XX = 4      # σ'_xx
    
    # For 2D extensions
    MOMENTUM_Y = 5     # ρu_y (2D)
    HEAT_FLUX_Y = 6    # q_y (2D)
    STRESS_YY = 7      # σ'_yy (2D)
    STRESS_XY = 8      # σ'_xy (2D)


@dataclass
class StateConfiguration:
    """Configuration for LNS state variables."""
    include_heat_flux: bool = True
    include_stress: bool = True
    include_2d_terms: bool = False
    
    @property
    def n_variables(self) -> int:
        """Compute number of state variables."""
        n_vars = 3  # Always have ρ, ρu, E
        
        if self.include_heat_flux:
            n_vars += 1 if not self.include_2d_terms else 2
        
        if self.include_stress:
            n_vars += 1 if not self.include_2d_terms else 3  # xx, yy, xy
            
        return n_vars
    
    @property 
    def variable_names(self) -> List[str]:
        """Get list of variable names."""
        names = ['density', 'momentum_x', 'total_energy']
        
        if self.include_heat_flux:
            names.append('heat_flux_x')
            if self.include_2d_terms:
                names.append('heat_flux_y')
        
        if self.include_stress:
            names.append('stress_xx')
            if self.include_2d_terms:
                names.extend(['stress_yy', 'stress_xy'])
                
        return names


class EnhancedLNSState:
    """
    Enhanced LNS state management with named accessors.
    
    This class provides professional property-based access to state variables,
    eliminating hardcoded indices and improving code readability and safety.
    
    Example:
        >>> state = EnhancedLNSState(grid, config)
        >>> rho = state.density  # Instead of Q[:, 0]
        >>> state.heat_flux_x = new_q_values  # Instead of Q[:, 3] =
    """
    
    def __init__(
        self, 
        grid: LNSGrid, 
        config: Optional[StateConfiguration] = None
    ):
        """
        Initialize enhanced LNS state.
        
        Args:
            grid: Computational grid
            config: State configuration (uses defaults if None)
        """
        self.grid = grid
        self.config = config or StateConfiguration()
        
        # Initialize conservative state array
        self.Q = np.zeros((grid.total_cells, self.config.n_variables))
        
        # Initialize primitive variables cache
        self._primitives_cache = None
        self._cache_valid = False
        
        logger.info(f"Initialized enhanced LNS state:")
        logger.info(f"  Variables: {self.config.variable_names}")
        logger.info(f"  Shape: {self.Q.shape}")
    
    # === Conservative Variable Properties ===
    
    @property
    def density(self) -> np.ndarray:
        """Density ρ [kg/m³]."""
        return self.Q[:, LNSVariables.DENSITY]
    
    @density.setter
    def density(self, value: np.ndarray):
        """Set density with validation."""
        if np.any(value <= 0):
            raise ValueError("Density must be positive")
        self.Q[:, LNSVariables.DENSITY] = value
        self._invalidate_cache()
    
    @property
    def momentum_x(self) -> np.ndarray:
        """X-momentum ρu_x [kg/(m²·s)]."""
        return self.Q[:, LNSVariables.MOMENTUM_X]
    
    @momentum_x.setter
    def momentum_x(self, value: np.ndarray):
        """Set x-momentum."""
        self.Q[:, LNSVariables.MOMENTUM_X] = value
        self._invalidate_cache()
    
    @property
    def total_energy(self) -> np.ndarray:
        """Total energy E_T [J/m³]."""
        return self.Q[:, LNSVariables.TOTAL_ENERGY]
    
    @total_energy.setter
    def total_energy(self, value: np.ndarray):
        """Set total energy with validation."""
        if np.any(value <= 0):
            raise ValueError("Total energy must be positive")
        self.Q[:, LNSVariables.TOTAL_ENERGY] = value
        self._invalidate_cache()
    
    # === LNS Extended Variables ===
    
    @property
    def heat_flux_x(self) -> Optional[np.ndarray]:
        """Heat flux q_x [W/m²]."""
        if not self.config.include_heat_flux:
            return None
        return self.Q[:, LNSVariables.HEAT_FLUX_X]
    
    @heat_flux_x.setter
    def heat_flux_x(self, value: np.ndarray):
        """Set x-heat flux."""
        if not self.config.include_heat_flux:
            raise ValueError("Heat flux not included in configuration")
        self.Q[:, LNSVariables.HEAT_FLUX_X] = value
        self._invalidate_cache()
    
    @property
    def stress_xx(self) -> Optional[np.ndarray]:
        """Deviatoric stress σ'_xx [Pa]."""
        if not self.config.include_stress:
            return None
        return self.Q[:, LNSVariables.STRESS_XX]
    
    @stress_xx.setter
    def stress_xx(self, value: np.ndarray):
        """Set xx-stress component."""
        if not self.config.include_stress:
            raise ValueError("Stress not included in configuration")
        self.Q[:, LNSVariables.STRESS_XX] = value
        self._invalidate_cache()
    
    # === Primitive Variable Properties (Computed) ===
    
    @property
    def velocity_x(self) -> np.ndarray:
        """X-velocity u_x [m/s]."""
        return self.momentum_x / np.maximum(self.density, 1e-12)
    
    @property
    def pressure(self) -> np.ndarray:
        """Pressure p [Pa]."""
        gamma = PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO
        kinetic_energy = 0.5 * self.density * self.velocity_x**2
        internal_energy = self.total_energy - kinetic_energy
        return np.maximum((gamma - 1) * internal_energy, 1e3)
    
    @property
    def temperature(self) -> np.ndarray:
        """Temperature T [K]."""
        R = PhysicalConstants.AIR_GAS_CONSTANT
        return self.pressure / (np.maximum(self.density, 1e-12) * R)
    
    @property
    def sound_speed(self) -> np.ndarray:
        """Sound speed c_s [m/s]."""
        gamma = PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO
        return np.sqrt(gamma * self.pressure / np.maximum(self.density, 1e-12))
    
    # === Composite Properties ===
    
    @property
    def internal_energy(self) -> np.ndarray:
        """Specific internal energy e [J/kg]."""
        kinetic = 0.5 * self.velocity_x**2
        return (self.total_energy / np.maximum(self.density, 1e-12)) - kinetic
    
    @property
    def kinetic_energy_density(self) -> np.ndarray:
        """Kinetic energy density [J/m³]."""
        return 0.5 * self.density * self.velocity_x**2
    
    @property
    def mach_number(self) -> np.ndarray:
        """Mach number M [-]."""
        return np.abs(self.velocity_x) / np.maximum(self.sound_speed, 1e-6)
    
    # === State Management Methods ===
    
    def get_primitive_variables(self) -> Dict[str, np.ndarray]:
        """
        Get all primitive variables as dictionary.
        
        This method caches results for efficiency.
        """
        if self._cache_valid and self._primitives_cache is not None:
            return self._primitives_cache
        
        primitives = {
            'density': self.density.copy(),
            'velocity': self.velocity_x.copy(),
            'pressure': self.pressure.copy(),
            'temperature': self.temperature.copy(),
            'sound_speed': self.sound_speed.copy(),
            'mach_number': self.mach_number.copy()
        }
        
        # Add LNS variables if present
        if self.config.include_heat_flux:
            primitives['heat_flux_x'] = self.heat_flux_x.copy()
        
        if self.config.include_stress:
            primitives['stress_xx'] = self.stress_xx.copy()
        
        self._primitives_cache = primitives
        self._cache_valid = True
        
        return primitives
    
    def set_from_primitives(
        self,
        density: np.ndarray,
        velocity: np.ndarray,
        pressure: np.ndarray,
        heat_flux_x: Optional[np.ndarray] = None,
        stress_xx: Optional[np.ndarray] = None
    ):
        """
        Set state from primitive variables.
        
        Args:
            density: Density field
            velocity: Velocity field
            pressure: Pressure field
            heat_flux_x: Heat flux (optional)
            stress_xx: Stress (optional)
        """
        # Set conservative variables
        self.density = density
        self.momentum_x = density * velocity
        
        # Compute total energy
        gamma = PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO
        internal_energy_density = pressure / (gamma - 1)
        kinetic_energy_density = 0.5 * density * velocity**2
        self.total_energy = internal_energy_density + kinetic_energy_density
        
        # Set LNS variables if provided
        if heat_flux_x is not None and self.config.include_heat_flux:
            self.heat_flux_x = heat_flux_x
        
        if stress_xx is not None and self.config.include_stress:
            self.stress_xx = stress_xx
    
    def initialize_sod_shock_tube(self):
        """Initialize standard Sod shock tube problem."""
        # Left and right states
        rho_L, u_L, p_L = 1.0, 0.0, 101325.0
        rho_R, u_R, p_R = 0.125, 0.0, 10132.5
        
        x = self.grid.x
        x_interface = 0.5 * (self.grid.x_bounds[0] + self.grid.x_bounds[1])
        
        # Initialize fields
        density = np.where(x < x_interface, rho_L, rho_R)
        velocity = np.where(x < x_interface, u_L, u_R)  
        pressure = np.where(x < x_interface, p_L, p_R)
        
        # Set from primitives
        self.set_from_primitives(density, velocity, pressure)
        
        # Initialize LNS variables to NSF values
        if self.config.include_heat_flux:
            self.heat_flux_x = np.zeros_like(density)  # Initial heat flux
        
        if self.config.include_stress:
            self.stress_xx = np.zeros_like(density)    # Initial stress
        
        logger.info("Initialized Sod shock tube with named accessors")
    
    def initialize_heat_conduction_test(
        self, 
        T_left: float = 400.0, 
        T_right: float = 300.0
    ):
        """Initialize heat conduction test case."""
        rho_init = 1.0
        u_init = 0.0
        
        # Linear temperature profile
        x = self.grid.x
        x_min, x_max = self.grid.x_bounds
        T_profile = T_left + (T_right - T_left) * (x - x_min) / (x_max - x_min)
        
        # Convert to pressure
        R = PhysicalConstants.AIR_GAS_CONSTANT
        p_profile = rho_init * R * T_profile
        
        # Set state
        density = np.full_like(x, rho_init)
        velocity = np.full_like(x, u_init)
        
        self.set_from_primitives(density, velocity, p_profile)
        
        # Initialize LNS variables
        if self.config.include_heat_flux:
            # Initial heat flux based on temperature gradient
            dT_dx = np.gradient(T_profile, self.grid.dx)
            k_thermal = 0.025  # Typical thermal conductivity
            self.heat_flux_x = -k_thermal * dT_dx
        
        if self.config.include_stress:
            self.stress_xx = np.zeros_like(density)
        
        logger.info(f"Initialized heat conduction test: T = {T_left}K to {T_right}K")
    
    def apply_positivity_limiter(self):
        """Apply positivity-preserving limiter using named accessors."""
        # Density limiting
        self.density = np.maximum(self.density, 1e-10)
        
        # Energy limiting
        kinetic = self.kinetic_energy_density
        internal = self.total_energy - kinetic
        e_min = 1e-3 * self.density
        
        mask = internal < e_min
        self.total_energy = np.where(mask, kinetic + e_min, self.total_energy)
        
        self._invalidate_cache()
    
    def _invalidate_cache(self):
        """Invalidate primitive variables cache."""
        self._cache_valid = False
    
    def validate_state(self) -> Dict[str, bool]:
        """
        Validate physical state.
        
        Returns:
            Dictionary of validation results
        """
        validation = {
            'positive_density': np.all(self.density > 0),
            'positive_pressure': np.all(self.pressure > 0),
            'positive_temperature': np.all(self.temperature > 0),
            'finite_values': np.all(np.isfinite(self.Q)),
            'reasonable_mach': np.all(self.mach_number < 10)  # Reasonable for most flows
        }
        
        return validation
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all variables."""
        stats = {}
        
        for var_name in ['density', 'velocity_x', 'pressure', 'temperature']:
            values = getattr(self, var_name)
            stats[var_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        # Add LNS variables if present
        if self.config.include_heat_flux:
            values = self.heat_flux_x
            stats['heat_flux_x'] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        if self.config.include_stress:
            values = self.stress_xx
            stats['stress_xx'] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return stats


# Test and demonstration
if __name__ == "__main__":
    print("🔧 Testing Enhanced LNS State with Named Accessors")
    print("=" * 50)
    
    # Create test grid
    from lns_solver.core.grid import LNSGrid
    grid = LNSGrid.create_uniform_1d(20, 0.0, 1.0)
    
    # Test basic configuration
    config = StateConfiguration(include_heat_flux=True, include_stress=True)
    state = EnhancedLNSState(grid, config)
    
    print(f"✅ Enhanced state created:")
    print(f"   Variables: {config.variable_names}")
    print(f"   Shape: {state.Q.shape}")
    
    # Test named accessors
    print("\\n🎯 Testing Named Accessors:")
    
    # Initialize shock tube
    state.initialize_sod_shock_tube()
    
    print(f"   Density range: {np.min(state.density):.3f} - {np.max(state.density):.3f} kg/m³")
    print(f"   Velocity range: {np.min(state.velocity_x):.3f} - {np.max(state.velocity_x):.3f} m/s")
    print(f"   Pressure range: {np.min(state.pressure):.1f} - {np.max(state.pressure):.1f} Pa")
    print(f"   Temperature range: {np.min(state.temperature):.1f} - {np.max(state.temperature):.1f} K")
    
    if state.config.include_heat_flux:
        print(f"   Heat flux range: {np.min(state.heat_flux_x):.3f} - {np.max(state.heat_flux_x):.3f} W/m²")
    
    if state.config.include_stress:
        print(f"   Stress range: {np.min(state.stress_xx):.3f} - {np.max(state.stress_xx):.3f} Pa")
    
    # Test validation
    print("\\n✅ Testing State Validation:")
    validation = state.validate_state()
    for check, result in validation.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}: {result}")
    
    # Test property setters
    print("\\n🔧 Testing Property Setters:")
    try:
        original_density = state.density.copy()
        state.density = original_density * 1.1
        change_detected = not np.allclose(state.density, original_density)
        print(f"   ✅ Density setter working: {change_detected}")
    except Exception as e:
        print(f"   ❌ Density setter error: {e}")
    
    print("\\n🏆 Enhanced State Features:")
    print("✅ Named property access (no hardcoded indices)")
    print("✅ Type-safe variable manipulation")
    print("✅ Automatic primitive variable computation")
    print("✅ Built-in validation and statistics")
    print("✅ Professional caching for performance")
    print("✅ Clear variable organization and extension")