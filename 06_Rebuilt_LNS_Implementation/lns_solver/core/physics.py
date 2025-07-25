"""
LNS Physics: Correct Local Navier-Stokes physics implementation.

This module provides the LNSPhysics class with mathematically correct formulations
of the Local Navier-Stokes equations, including proper deviatoric stress targets
and complete objective derivatives.

Example:
    Compute corrected 1D NSF targets:
    
    >>> physics = LNSPhysics()
    >>> du_dx = 0.1  # velocity gradient
    >>> dT_dx = -10.0  # temperature gradient
    >>> material_props = {'mu_viscous': 1e-3, 'k_thermal': 0.025}
    >>> q_nsf, sigma_nsf = physics.compute_1d_nsf_targets(du_dx, dT_dx, material_props)
    >>> print(f"Correct deviatoric stress: {sigma_nsf:.6f}")
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging

from lns_solver.core.grid import LNSGrid
from lns_solver.core.state_enhanced import EnhancedLNSState

logger = logging.getLogger(__name__)

# Type aliases
GradientTuple = Tuple[np.ndarray, np.ndarray]
ObjectiveDerivativeResult = Dict[str, np.ndarray]


@dataclass
class LNSPhysicsParameters:
    """Physical parameters for LNS equations."""
    # Material properties
    mu_viscous: float = 1e-3        # Dynamic viscosity (Pa·s)
    k_thermal: float = 0.025        # Thermal conductivity (W/(m·K))
    rho_reference: float = 1.0      # Reference density (kg/m³)
    
    # Relaxation times
    tau_q: float = 1e-6            # Thermal relaxation time (s)
    tau_sigma: float = 1e-6        # Stress relaxation time (s)
    
    # Gas properties
    gas_constant: float = 287.0     # Specific gas constant (J/(kg·K))
    specific_heat_ratio: float = 1.4  # Heat capacity ratio
    prandtl_number: float = 0.71    # Prandtl number
    
    def __post_init__(self):
        """Validate physical parameters."""
        if self.mu_viscous <= 0:
            raise ValueError(f"mu_viscous must be positive, got {self.mu_viscous}")
        if self.k_thermal <= 0:
            raise ValueError(f"k_thermal must be positive, got {self.k_thermal}")
        if self.tau_q <= 0:
            raise ValueError(f"tau_q must be positive, got {self.tau_q}")
        if self.tau_sigma <= 0:
            raise ValueError(f"tau_sigma must be positive, got {self.tau_sigma}")


class LNSPhysics:
    """
    Correct Local Navier-Stokes physics implementation.
    
    This class implements mathematically correct LNS physics including:
    - Proper 1D deviatoric stress formula with 4/3 factor
    - Complete 2D objective derivatives with all transport terms
    - Maxwell-Cattaneo-Vernotte heat conduction
    - Upper Convected Maxwell stress evolution
    
    Key Corrections from Previous Implementation:
    - Fixed 1D deviatoric stress: σ'_xx = (4/3)μ(∂u/∂x)
    - Complete 2D convective transport: u·∇q, u·∇σ
    - Proper velocity gradient coupling in UCM model
    
    Example:
        >>> physics = LNSPhysics(params)
        >>> q_nsf, sigma_nsf = physics.compute_1d_nsf_targets(du_dx, dT_dx, material_props)
    """
    
    def __init__(self, params: Optional[LNSPhysicsParameters] = None):
        """
        Initialize LNS physics.
        
        Args:
            params: Physical parameters (uses defaults if None)
        """
        self.params = params or LNSPhysicsParameters()
        
        # Derived quantities
        self.thermal_diffusivity = (
            self.params.k_thermal / 
            (self.params.rho_reference * self._compute_specific_heat_cp())
        )
        
        # Thermal wave speed
        self.thermal_wave_speed = np.sqrt(
            self.params.k_thermal / 
            (self.params.rho_reference * self._compute_specific_heat_cp() * self.params.tau_q)
        )
        
        logger.info(f"Initialized LNS physics: c_th = {self.thermal_wave_speed:.2e} m/s")
    
    def _compute_specific_heat_cp(self) -> float:
        """Compute specific heat at constant pressure."""
        cv = self.params.gas_constant / (self.params.specific_heat_ratio - 1)
        cp = self.params.specific_heat_ratio * cv
        return cp
    
    @staticmethod
    def compute_1d_nsf_targets(
        du_dx: Union[float, np.ndarray],
        dT_dx: Union[float, np.ndarray],
        material_props: Dict[str, float]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Compute CORRECT 1D NSF targets with proper deviatoric stress formula.
        
        This is the CORRECTED version that fixes the critical physics error
        in the previous implementation.
        
        Args:
            du_dx: Velocity gradient ∂u/∂x
            dT_dx: Temperature gradient ∂T/∂x
            material_props: Dictionary with 'k_thermal' and 'mu_viscous'
            
        Returns:
            Tuple of (q_nsf, sigma_xx_nsf)
            q_nsf: Heat flux NSF target = -k * ∂T/∂x
            sigma_xx_nsf: CORRECT deviatoric stress = (4/3) * μ * ∂u/∂x
            
        Example:
            >>> du_dx = 0.1
            >>> dT_dx = -10.0
            >>> props = {'k_thermal': 0.025, 'mu_viscous': 1e-3}
            >>> q_nsf, sigma_nsf = LNSPhysics.compute_1d_nsf_targets(du_dx, dT_dx, props)
            >>> print(f"Correct stress: {sigma_nsf:.6f}")  # Should include 4/3 factor
        """
        # Maxwell-Cattaneo-Vernotte heat flux target
        q_nsf = -material_props['k_thermal'] * dT_dx
        
        # CORRECTED 1D compressible deviatoric stress target
        # This is the CRITICAL FIX: (4/3) factor for compressible flow
        sigma_xx_nsf = (4.0/3.0) * material_props['mu_viscous'] * du_dx
        
        logger.debug(f"1D NSF targets: q_nsf={np.mean(q_nsf) if hasattr(q_nsf, '__iter__') else q_nsf:.3e}, "
                    f"σ'_xx={np.mean(sigma_xx_nsf) if hasattr(sigma_xx_nsf, '__iter__') else sigma_xx_nsf:.3e}")
        
        return q_nsf, sigma_xx_nsf
    
    @staticmethod
    def compute_2d_nsf_targets(
        velocity_gradients: Dict[str, np.ndarray],
        temperature_gradients: GradientTuple,
        material_props: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D NSF targets for heat flux and stress tensor.
        
        Args:
            velocity_gradients: Dict with 'dux_dx', 'dux_dy', 'duy_dx', 'duy_dy'
            temperature_gradients: Tuple (dT_dx, dT_dy)
            material_props: Material properties
            
        Returns:
            Tuple of (q_nsf, sigma_nsf)
            q_nsf: Heat flux vector [q_x, q_y]
            sigma_nsf: Deviatoric stress tensor [σ'_xx, σ'_yy, σ'_xy]
        """
        dT_dx, dT_dy = temperature_gradients
        dux_dx = velocity_gradients['dux_dx']
        dux_dy = velocity_gradients['dux_dy']
        duy_dx = velocity_gradients['duy_dx']
        duy_dy = velocity_gradients['duy_dy']
        
        # Heat flux NSF targets
        q_x_nsf = -material_props['k_thermal'] * dT_dx
        q_y_nsf = -material_props['k_thermal'] * dT_dy
        q_nsf = np.stack([q_x_nsf, q_y_nsf], axis=-1)
        
        # Strain rate tensor components
        S_xx = dux_dx
        S_yy = duy_dy
        S_xy = 0.5 * (dux_dy + duy_dx)
        
        # Deviatoric stress NSF targets (Newton's law of viscosity)
        # For compressible flow: σ'_ij = 2μ(S_ij - (1/3)δ_ij*∇·u)
        div_u = dux_dx + duy_dy
        mu = material_props['mu_viscous']
        
        sigma_xx_nsf = 2.0 * mu * (S_xx - div_u/3.0)
        sigma_yy_nsf = 2.0 * mu * (S_yy - div_u/3.0)
        sigma_xy_nsf = 2.0 * mu * S_xy
        
        sigma_nsf = np.stack([sigma_xx_nsf, sigma_yy_nsf, sigma_xy_nsf], axis=-1)
        
        return q_nsf, sigma_nsf
    
    @staticmethod
    def compute_2d_objective_derivatives_complete(
        state_field: np.ndarray,
        velocity_field: np.ndarray,
        dx: float,
        dy: float
    ) -> ObjectiveDerivativeResult:
        """
        COMPLETE implementation of 2D objective derivatives with ALL transport terms.
        
        This function contains the COMPLETE UCM and MCV physics that was missing
        in the previous implementation. All spatial gradients are properly computed
        and all transport terms are included.
        
        Args:
            state_field: State field [N_x, N_y, 5] with [q_x, q_y, σ'_xx, σ'_yy, σ'_xy]
            velocity_field: Velocity field [N_x, N_y, 2] with [u_x, u_y]
            dx, dy: Grid spacing
            
        Returns:
            Dictionary with complete objective derivatives:
            - 'heat_flux': MCV objective derivatives [D_qx/Dt, D_qy/Dt]
            - 'stress': UCM objective derivatives [D_σxx/Dt, D_σyy/Dt, D_σxy/Dt]
            
        Physics Implemented:
            Maxwell-Cattaneo-Vernotte (MCV):
            D_q/Dt = ∂q/∂t + u·∇q + (∇·u)q
            
            Upper Convected Maxwell (UCM):
            D_σ/Dt = ∂σ/∂t + u·∇σ - σ·∇u - (∇u)ᵀ·σ
        """
        # Extract field components
        q_x = state_field[:, :, 0]
        q_y = state_field[:, :, 1]
        sigma_xx = state_field[:, :, 2]
        sigma_yy = state_field[:, :, 3]
        sigma_xy = state_field[:, :, 4]
        
        u_x = velocity_field[:, :, 0]
        u_y = velocity_field[:, :, 1]
        
        # Compute ALL spatial gradients using efficient NumPy operations
        # This is the CRITICAL fix - no more zero placeholders!
        
        # Heat flux gradients
        dqx_dx = np.gradient(q_x, dx, axis=0)
        dqx_dy = np.gradient(q_x, dy, axis=1)
        dqy_dx = np.gradient(q_y, dx, axis=0)
        dqy_dy = np.gradient(q_y, dy, axis=1)
        
        # Stress tensor gradients
        dsxx_dx = np.gradient(sigma_xx, dx, axis=0)
        dsxx_dy = np.gradient(sigma_xx, dy, axis=1)
        dsyy_dx = np.gradient(sigma_yy, dx, axis=0)
        dsyy_dy = np.gradient(sigma_yy, dy, axis=1)
        dsxy_dx = np.gradient(sigma_xy, dx, axis=0)
        dsxy_dy = np.gradient(sigma_xy, dy, axis=1)
        
        # Velocity gradients
        dux_dx = np.gradient(u_x, dx, axis=0)
        dux_dy = np.gradient(u_x, dy, axis=1)
        duy_dx = np.gradient(u_y, dx, axis=0)
        duy_dy = np.gradient(u_y, dy, axis=1)
        
        # Velocity gradient tensor components
        L_xx = dux_dx
        L_xy = dux_dy
        L_yx = duy_dx
        L_yy = duy_dy
        div_u = L_xx + L_yy
        
        # === MAXWELL-CATTANEO-VERNOTTE OBJECTIVE DERIVATIVES ===
        # D_q/Dt = ∂q/∂t + u·∇q + (∇·u)q
        
        # Convective transport: u·∇q
        D_qx_Dt_conv = u_x * dqx_dx + u_y * dqx_dy + div_u * q_x
        D_qy_Dt_conv = u_x * dqy_dx + u_y * dqy_dy + div_u * q_y
        
        # === UPPER CONVECTED MAXWELL OBJECTIVE DERIVATIVES ===
        # D_σ/Dt = ∂σ/∂t + u·∇σ - σ·∇u - (∇u)ᵀ·σ
        
        # Convective transport: u·∇σ
        conv_sxx = u_x * dsxx_dx + u_y * dsxx_dy
        conv_syy = u_x * dsyy_dx + u_y * dsyy_dy
        conv_sxy = u_x * dsxy_dx + u_y * dsxy_dy
        
        # Velocity gradient coupling: -σ·∇u - (∇u)ᵀ·σ
        # This is the complete tensor algebra that was missing before
        
        # For σ_xx component:
        # -σ·∇u term: -σ_xx*L_xx - σ_xy*L_yx
        # -(∇u)ᵀ·σ term: -σ_xx*L_xx - σ_yx*L_xy = -σ_xx*L_xx - σ_xy*L_xy
        stretch_sxx = -2.0 * sigma_xx * L_xx - sigma_xy * (L_yx + L_xy)
        
        # For σ_yy component:
        # -σ·∇u term: -σ_yx*L_xy - σ_yy*L_yy = -σ_xy*L_xy - σ_yy*L_yy
        # -(∇u)ᵀ·σ term: -σ_xy*L_yx - σ_yy*L_yy
        stretch_syy = -sigma_xy * (L_xy + L_yx) - 2.0 * sigma_yy * L_yy
        
        # For σ_xy component:
        # -σ·∇u term: -σ_xx*L_xy - σ_xy*L_yy
        # -(∇u)ᵀ·σ term: -σ_yx*L_xx - σ_yy*L_yx = -σ_xy*L_xx - σ_yy*L_yx
        stretch_sxy = -sigma_xx * L_xy - sigma_xy * L_yy - sigma_xy * L_xx - sigma_yy * L_yx
        
        # Complete objective derivatives (convection + stretching)
        D_sxx_Dt_conv = conv_sxx + stretch_sxx
        D_syy_Dt_conv = conv_syy + stretch_syy
        D_sxy_Dt_conv = conv_sxy + stretch_sxy
        
        # Return complete results
        result = {
            'heat_flux': np.stack([D_qx_Dt_conv, D_qy_Dt_conv], axis=-1),
            'stress': np.stack([D_sxx_Dt_conv, D_syy_Dt_conv, D_sxy_Dt_conv], axis=-1)
        }
        
        logger.debug("Computed complete 2D objective derivatives with all transport terms")
        
        return result
    
    def compute_equation_of_state(
        self,
        density: np.ndarray,
        temperature: np.ndarray
    ) -> np.ndarray:
        """
        Compute pressure using ideal gas equation of state.
        
        Args:
            density: Density field
            temperature: Temperature field
            
        Returns:
            Pressure field
        """
        pressure = density * self.params.gas_constant * temperature
        return pressure
    
    def compute_sound_speed(
        self,
        density: np.ndarray,
        pressure: np.ndarray
    ) -> np.ndarray:
        """
        Compute acoustic sound speed.
        
        Args:
            density: Density field
            pressure: Pressure field
            
        Returns:
            Sound speed field
        """
        sound_speed = np.sqrt(self.params.specific_heat_ratio * pressure / density)
        return sound_speed
    
    def compute_thermal_wave_speed(self) -> float:
        """
        Compute thermal wave speed for LNS system.
        
        Returns:
            Thermal wave speed c_th = √(k/(ρ*c_p*τ_q))
        """
        return self.thermal_wave_speed
    
    def compute_viscous_wave_speed(self) -> float:
        """
        Compute viscous wave speed for LNS system.
        
        Returns:
            Viscous wave speed c_v = √(μ/(ρ*τ_σ))
        """
        viscous_wave_speed = np.sqrt(
            self.params.mu_viscous / 
            (self.params.rho_reference * self.params.tau_sigma)
        )
        return viscous_wave_speed
    
    def compute_cfl_condition(
        self,
        dx: float,
        dy: float = None,
        max_velocity: float = 0.0
    ) -> float:
        """
        Compute maximum stable time step based on CFL condition.
        
        Args:
            dx: Grid spacing in x-direction
            dy: Grid spacing in y-direction (for 2D)
            max_velocity: Maximum fluid velocity
            
        Returns:
            Maximum stable time step
        """
        # Acoustic CFL
        c_sound = np.sqrt(self.params.specific_heat_ratio * 
                         self.params.gas_constant * self.params.rho_reference)
        cfl_acoustic = dx / (c_sound + max_velocity)
        
        # Thermal wave CFL
        cfl_thermal = dx / self.thermal_wave_speed
        
        # Viscous wave CFL
        cfl_viscous = dx / self.compute_viscous_wave_speed()
        
        # Most restrictive condition
        dt_max = 0.8 * min(cfl_acoustic, cfl_thermal, cfl_viscous)
        
        if dy is not None:
            # 2D case - check y-direction as well
            cfl_acoustic_y = dy / (c_sound + max_velocity)
            cfl_thermal_y = dy / self.thermal_wave_speed
            cfl_viscous_y = dy / self.compute_viscous_wave_speed()
            
            dt_max = min(dt_max, 0.8 * min(cfl_acoustic_y, cfl_thermal_y, cfl_viscous_y))
        
        logger.debug(f"CFL-limited time step: {dt_max:.2e} s")
        
        return dt_max
    
    def compute_relaxation_source_terms(
        self,
        q_current: np.ndarray,
        sigma_current: np.ndarray,
        q_nsf: np.ndarray,
        sigma_nsf: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute relaxation source terms for semi-implicit integration.
        
        Args:
            q_current: Current heat flux
            sigma_current: Current stress
            q_nsf: NSF target heat flux
            sigma_nsf: NSF target stress
            
        Returns:
            Tuple of (q_source, sigma_source)
        """
        # Relaxation source terms: -(flux - NSF_target) / relaxation_time
        q_source = -(q_current - q_nsf) / self.params.tau_q
        sigma_source = -(sigma_current - sigma_nsf) / self.params.tau_sigma
        
        return q_source, sigma_source
    
    def get_physics_dict(self) -> Dict[str, float]:
        """
        Get physics parameters as dictionary for numerical methods.
        
        Returns:
            Dictionary of physics parameters
        """
        return {
            'specific_heat_ratio': self.params.specific_heat_ratio,
            'gas_constant': self.params.gas_constant,
            'mu_viscous': self.params.mu_viscous,
            'k_thermal': self.params.k_thermal,
            'tau_q': self.params.tau_q,
            'tau_sigma': self.params.tau_sigma
        }
    
    def get_physics_info(self) -> Dict[str, Union[float, str]]:
        """Get comprehensive physics information."""
        return {
            'mu_viscous': self.params.mu_viscous,
            'k_thermal': self.params.k_thermal,
            'tau_q': self.params.tau_q,
            'tau_sigma': self.params.tau_sigma,
            'thermal_diffusivity': self.thermal_diffusivity,
            'thermal_wave_speed': self.thermal_wave_speed,
            'viscous_wave_speed': self.compute_viscous_wave_speed(),
            'gas_constant': self.params.gas_constant,
            'specific_heat_ratio': self.params.specific_heat_ratio,
            'prandtl_number': self.params.prandtl_number,
        }
    
    def __repr__(self) -> str:
        """String representation of physics."""
        return f"LNSPhysics(τ_q={self.params.tau_q:.1e}, τ_σ={self.params.tau_sigma:.1e})"
    
    def __str__(self) -> str:
        """Detailed string representation."""
        lines = ["LNS Physics Parameters"]
        lines.append("=" * 25)
        
        info = self.get_physics_info()
        for key, value in info.items():
            if isinstance(value, float):
                if abs(value) >= 1e3 or abs(value) <= 1e-3:
                    lines.append(f"{key:20s}: {value:.2e}")
                else:
                    lines.append(f"{key:20s}: {value:.6f}")
            else:
                lines.append(f"{key:20s}: {value}")
        
        return "\n".join(lines)