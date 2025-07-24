"""
Physical constants and material properties for LNS solver.

This module provides commonly used physical constants and material properties
for fluid dynamics simulations.
"""

import numpy as np
from typing import Dict, Any


class PhysicalConstants:
    """Physical constants for fluid dynamics simulations."""
    
    # Universal constants
    BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
    AVOGADRO_NUMBER = 6.02214076e23    # mol⁻¹
    UNIVERSAL_GAS_CONSTANT = 8.314462618  # J/(mol·K)
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
    
    # Standard conditions
    STANDARD_PRESSURE = 101325.0       # Pa
    STANDARD_TEMPERATURE = 273.15      # K
    STANDARD_DENSITY_AIR = 1.225       # kg/m³
    
    # Air properties (at STP)
    AIR_GAS_CONSTANT = 287.0           # J/(kg·K)
    AIR_SPECIFIC_HEAT_RATIO = 1.4      # γ
    AIR_PRANDTL_NUMBER = 0.71          # Pr
    AIR_DYNAMIC_VISCOSITY = 1.81e-5    # Pa·s
    AIR_THERMAL_CONDUCTIVITY = 0.0257  # W/(m·K)
    
    # Water properties (at 20°C)
    WATER_DENSITY = 998.2              # kg/m³
    WATER_DYNAMIC_VISCOSITY = 1.002e-3 # Pa·s
    WATER_KINEMATIC_VISCOSITY = 1.004e-6  # m²/s
    WATER_THERMAL_CONDUCTIVITY = 0.598 # W/(m·K)
    WATER_SPECIFIC_HEAT = 4182.0       # J/(kg·K)
    WATER_PRANDTL_NUMBER = 7.01        # Pr
    
    @classmethod
    def get_air_properties(cls, temperature: float = 300.0) -> Dict[str, float]:
        """
        Get temperature-dependent air properties.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary with air properties at given temperature
        """
        # Temperature dependence (Sutherland's law for viscosity)
        T_ref = 273.15
        mu_ref = cls.AIR_DYNAMIC_VISCOSITY
        S = 110.4  # Sutherland constant for air
        
        mu = mu_ref * (T_ref + S) / (temperature + S) * (temperature / T_ref)**1.5
        
        # Thermal conductivity (proportional to viscosity for gases)
        k = mu * cls.AIR_GAS_CONSTANT * cls.AIR_SPECIFIC_HEAT_RATIO / cls.AIR_PRANDTL_NUMBER
        
        # Density from ideal gas law
        rho = cls.STANDARD_PRESSURE / (cls.AIR_GAS_CONSTANT * temperature)
        
        return {
            'density': rho,
            'dynamic_viscosity': mu,
            'thermal_conductivity': k,
            'gas_constant': cls.AIR_GAS_CONSTANT,
            'specific_heat_ratio': cls.AIR_SPECIFIC_HEAT_RATIO,
            'prandtl_number': cls.AIR_PRANDTL_NUMBER,
        }
    
    @classmethod
    def get_water_properties(cls, temperature: float = 293.15) -> Dict[str, float]:
        """
        Get temperature-dependent water properties.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary with water properties at given temperature
        """
        # Simplified temperature dependence for water
        T_ref = 293.15  # 20°C
        
        # Viscosity decreases with temperature
        mu = cls.WATER_DYNAMIC_VISCOSITY * np.exp(1500 * (1/temperature - 1/T_ref))
        
        # Density variation (small for water)
        rho = cls.WATER_DENSITY * (1 - 2.1e-4 * (temperature - T_ref))
        
        # Thermal conductivity (weak temperature dependence)
        k = cls.WATER_THERMAL_CONDUCTIVITY * (1 + 1.5e-3 * (temperature - T_ref))
        
        return {
            'density': rho,
            'dynamic_viscosity': mu,
            'kinematic_viscosity': mu / rho,
            'thermal_conductivity': k,
            'specific_heat': cls.WATER_SPECIFIC_HEAT,
            'prandtl_number': cls.WATER_PRANDTL_NUMBER,
        }
    
    @classmethod
    def compute_reynolds_number(
        cls,
        velocity: float,
        length_scale: float,
        kinematic_viscosity: float
    ) -> float:
        """Compute Reynolds number."""
        return velocity * length_scale / kinematic_viscosity
    
    @classmethod
    def compute_prandtl_number(
        cls,
        kinematic_viscosity: float,
        thermal_diffusivity: float
    ) -> float:
        """Compute Prandtl number."""
        return kinematic_viscosity / thermal_diffusivity
    
    @classmethod  
    def compute_mach_number(
        cls,
        velocity: float,
        sound_speed: float
    ) -> float:
        """Compute Mach number."""
        return velocity / sound_speed