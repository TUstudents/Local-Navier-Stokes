"""
LNS Numerics: Efficient numerical methods with correct algorithms.

This module provides the LNSNumerics class with mathematically correct and 
computationally efficient numerical methods for the LNS solver, including:
- O(N²) gradient computations using vectorized NumPy
- Corrected hyperbolic updates with proper signs
- HLL Riemann solver implementation
- Stability-preserving time stepping

Example:
    Compute gradients efficiently:
    
    >>> numerics = LNSNumerics()
    >>> fields = [temperature_field, velocity_field]
    >>> gradients = numerics.compute_gradients_efficient(fields, dx, dy)
    >>> print(f"Gradient computation: O(N²) complexity achieved")
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, Literal
import numpy as np
import logging

# Optional Numba import
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda x: x  # No-op decorator
    prange = range

from lns_solver.core.grid import LNSGrid
from lns_solver.core.state import LNSState

logger = logging.getLogger(__name__)

# Type aliases
FluxFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
GradientDict = Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
RiemannSolution = Dict[str, Union[np.ndarray, float]]


class LNSNumerics:
    """
    Efficient numerical methods for LNS solver with correct algorithms.
    
    This class implements computationally efficient and mathematically correct
    numerical methods, addressing all bugs and performance issues from the
    previous implementation:
    
    Key Corrections:
    - O(N²) gradient computations (was O(N⁴))
    - Corrected hyperbolic update signs
    - Proper finite volume flux computation
    - Vectorized operations throughout
    
    Key Features:
    - Efficient gradient computation using NumPy vectorization
    - HLL Riemann solver for robust flux computation
    - Semi-implicit time stepping for stiff source terms
    - Adaptive time step control
    
    Example:
        >>> numerics = LNSNumerics()
        >>> rhs = numerics.compute_hyperbolic_rhs_2d(state, flux_func, dx, dy)
    """
    
    def __init__(self, use_numba: bool = True):
        """
        Initialize LNS numerics.
        
        Args:
            use_numba: Whether to use Numba JIT compilation for performance
        """
        self.use_numba = use_numba
        
        # Numerical parameters
        self.gradient_limiter = True
        self.flux_limiter = True
        self.entropy_fix = True
        
        logger.info(f"Initialized LNS numerics (Numba: {'enabled' if use_numba else 'disabled'})")
    
    @staticmethod
    def compute_gradients_efficient(
        fields: List[np.ndarray],
        dx: float,
        dy: Optional[float] = None,
        field_names: Optional[List[str]] = None
    ) -> GradientDict:
        """
        Compute gradients with O(N²) complexity using NumPy vectorization.
        
        This is the CORRECTED version that achieves O(N²) complexity instead
        of the O(N⁴) disaster in the previous implementation.
        
        Args:
            fields: List of field arrays to compute gradients for
            dx: Grid spacing in x-direction
            dy: Grid spacing in y-direction (for 2D/3D)
            field_names: Optional names for fields (for dictionary keys)
            
        Returns:
            Dictionary of gradient arrays for each field
            
        Complexity: O(N²) for 2D grids (vs O(N⁴) in previous implementation)
        
        Example:
            >>> fields = [temperature, velocity_x, velocity_y]
            >>> gradients = LNSNumerics.compute_gradients_efficient(fields, 0.01, 0.01)
            >>> dT_dx, dT_dy = gradients['field_0']
        """
        gradients = {}
        
        # Generate field names if not provided
        if field_names is None:
            field_names = [f'field_{i}' for i in range(len(fields))]
        
        if len(field_names) != len(fields):
            raise ValueError(f"Number of field names ({len(field_names)}) must match fields ({len(fields)})")
        
        for field, name in zip(fields, field_names):
            if dy is None:  # 1D case
                # Simple 1D gradient
                grad_x = np.gradient(field, dx)
                gradients[name] = grad_x
                
            else:  # 2D case
                # Efficient 2D gradients using NumPy vectorization
                # This is O(N²) per field, not O(N⁴) like the old implementation
                grad_x = np.gradient(field, dx, axis=0)
                grad_y = np.gradient(field, dy, axis=1)
                gradients[name] = (grad_x, grad_y)
        
        logger.debug(f"Computed gradients for {len(fields)} fields with O(N²) complexity")
        
        return gradients
    
    @staticmethod
    def compute_hyperbolic_rhs_1d(
        state_field: np.ndarray,
        flux_function: FluxFunction,
        dx: float
    ) -> np.ndarray:
        """
        Compute 1D hyperbolic RHS with correct finite volume method.
        
        Args:
            state_field: State array [N_cells, N_vars]
            flux_function: Function to compute numerical flux
            dx: Grid spacing
            
        Returns:
            RHS array for hyperbolic update
            
        Formula: RHS[i] = -(F_{i+1/2} - F_{i-1/2}) / dx
        """
        N_cells, N_vars = state_field.shape
        RHS = np.zeros_like(state_field)
        
        # Compute fluxes at all interfaces
        for i in range(N_cells - 1):
            # Interface i+1/2 between cells i and i+1
            Q_left = state_field[i, :]
            Q_right = state_field[i + 1, :]
            
            # Numerical flux (HLL, Lax-Friedrichs, etc.)
            flux = flux_function(Q_left, Q_right)
            
            # Apply to adjacent cells with CORRECT signs
            RHS[i, :] -= flux / dx      # Left cell: -F_{i+1/2}/dx
            RHS[i + 1, :] += flux / dx  # Right cell: +F_{i+1/2}/dx
        
        return RHS
    
    @staticmethod
    def compute_hyperbolic_rhs_2d(
        state_field: np.ndarray,
        flux_function: FluxFunction,
        dx: float,
        dy: float
    ) -> np.ndarray:
        """
        Compute 2D hyperbolic RHS with CORRECTED signs.
        
        This fixes the critical sign error in the previous implementation
        that caused incorrect wave propagation.
        
        Args:
            state_field: State array [N_x, N_y, N_vars]
            flux_function: Function to compute numerical flux
            dx, dy: Grid spacing
            
        Returns:
            RHS array for hyperbolic update
            
        Formula: 
            RHS[i,j] = -(F_{i+1/2,j} - F_{i-1/2,j})/dx - (G_{i,j+1/2} - G_{i,j-1/2})/dy
        """
        N_x, N_y, N_vars = state_field.shape
        RHS = np.zeros_like(state_field)
        
        # X-direction fluxes with CORRECTED signs
        for i in range(N_x - 1):
            for j in range(N_y):
                # Interface i+1/2, j
                Q_left = state_field[i, j, :]
                Q_right = state_field[i + 1, j, :]
                flux_x = flux_function(Q_left, Q_right, direction='x')
                
                # Apply with CORRECT signs (fixes critical bug)
                RHS[i, j, :] -= flux_x / dx      # -F_{i+1/2} for left cell
                RHS[i + 1, j, :] += flux_x / dx  # +F_{i+1/2} for right cell
        
        # Y-direction fluxes with CORRECTED signs
        for i in range(N_x):
            for j in range(N_y - 1):
                # Interface i, j+1/2
                Q_bottom = state_field[i, j, :]
                Q_top = state_field[i, j + 1, :]
                flux_y = flux_function(Q_bottom, Q_top, direction='y')
                
                # Apply with CORRECT signs (fixes critical bug)
                RHS[i, j, :] -= flux_y / dy      # -G_{j+1/2} for bottom cell
                RHS[i, j + 1, :] += flux_y / dy  # +G_{j+1/2} for top cell
        
        logger.debug("Computed 2D hyperbolic RHS with corrected signs")
        
        return RHS
    
    @staticmethod
    def hll_flux_1d(
        Q_left: np.ndarray,
        Q_right: np.ndarray,
        physics_params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """
        HLL (Harten-Lax-van Leer) Riemann solver for 1D LNS system.
        
        This provides a robust, entropy-satisfying numerical flux that
        properly handles the characteristic structure of the LNS equations.
        
        Args:
            Q_left: Left state [ρ, ρu, E, q_x, σ'_xx]
            Q_right: Right state [ρ, ρu, E, q_x, σ'_xx]
            physics_params: Physical parameters for wave speed estimation
            
        Returns:
            Numerical flux at interface
            
        Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
        """
        # Extract conserved variables
        rho_L, rho_u_L, E_L, q_L, sigma_L = Q_left
        rho_R, rho_u_R, E_R, q_R, sigma_R = Q_right
        
        # Primitive variables
        if rho_L <= 0 or rho_R <= 0:
            logger.warning("Non-physical density in HLL solver")
            return np.zeros_like(Q_left)
        
        u_L = rho_u_L / rho_L
        u_R = rho_u_R / rho_R
        
        # Pressure estimation (simplified ideal gas)
        gamma = physics_params.get('specific_heat_ratio', 1.4)
        R = physics_params.get('gas_constant', 287.0)
        
        # Internal energy approximation
        e_L = E_L - 0.5 * rho_L * u_L**2
        e_R = E_R - 0.5 * rho_R * u_R**2
        
        if e_L <= 0 or e_R <= 0:
            logger.warning("Non-physical internal energy in HLL solver")
            return np.zeros_like(Q_left)
        
        # Temperature and pressure (ideal gas approximation)
        cv = R / (gamma - 1)
        T_L = e_L / (rho_L * cv)
        T_R = e_R / (rho_R * cv)
        p_L = rho_L * R * T_L
        p_R = rho_R * R * T_R
        
        # Sound speeds
        c_L = np.sqrt(gamma * p_L / rho_L)
        c_R = np.sqrt(gamma * p_R / rho_R)
        
        # Estimate wave speeds (Einfeldt approximation)
        # For LNS, we also need to consider thermal and viscous wave speeds
        tau_q = physics_params.get('tau_q', 1e-6)
        tau_sigma = physics_params.get('tau_sigma', 1e-6)
        k_thermal = physics_params.get('k_thermal', 0.025)
        mu_viscous = physics_params.get('mu_viscous', 1e-3)
        
        # Additional LNS wave speeds
        c_th_L = np.sqrt(k_thermal / (rho_L * cv * tau_q)) if tau_q > 0 else 0
        c_th_R = np.sqrt(k_thermal / (rho_R * cv * tau_q)) if tau_q > 0 else 0
        c_v_L = np.sqrt(mu_viscous / (rho_L * tau_sigma)) if tau_sigma > 0 else 0
        c_v_R = np.sqrt(mu_viscous / (rho_R * tau_sigma)) if tau_sigma > 0 else 0
        
        # Maximum wave speeds
        c_max_L = max(c_L, c_th_L, c_v_L)
        c_max_R = max(c_R, c_th_R, c_v_R)
        
        # HLL wave speed estimates
        S_L = min(u_L - c_max_L, u_R - c_max_R)
        S_R = max(u_L + c_max_L, u_R + c_max_R)
        
        # Physical fluxes
        F_L = LNSNumerics._compute_physical_flux_1d(Q_left, p_L)
        F_R = LNSNumerics._compute_physical_flux_1d(Q_right, p_R)
        
        # HLL flux formula
        if S_L >= 0:
            flux = F_L
        elif S_R <= 0:
            flux = F_R
        else:
            # Mixed state flux
            flux = (S_R * F_L - S_L * F_R + S_L * S_R * (Q_right - Q_left)) / (S_R - S_L)
        
        return flux
    
    @staticmethod
    def _compute_physical_flux_1d(Q: np.ndarray, pressure: float) -> np.ndarray:
        """
        Compute physical flux for 1D LNS system.
        
        Args:
            Q: Conservative state [ρ, ρu, E, q_x, σ'_xx]
            pressure: Pressure
            
        Returns:
            Physical flux F(Q)
        """
        rho, rho_u, E, q_x, sigma_xx = Q
        
        if rho <= 0:
            return np.zeros_like(Q)
        
        u = rho_u / rho
        
        # Physical flux components
        F = np.array([
            rho_u,                           # Mass flux
            rho_u * u + pressure - sigma_xx, # Momentum flux (includes viscous stress)
            (E + pressure) * u - sigma_xx * u + q_x,  # Energy flux
            q_x * u,                         # Heat flux convection
            sigma_xx * u                     # Stress convection
        ])
        
        return F
    
    @staticmethod
    def lax_friedrichs_flux_1d(
        Q_left: np.ndarray,
        Q_right: np.ndarray,
        physics_params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """
        Lax-Friedrichs flux (fallback for HLL solver).
        
        Args:
            Q_left: Left state
            Q_right: Right state
            physics_params: Physical parameters
            
        Returns:
            Lax-Friedrichs numerical flux
        """
        # Estimate maximum wave speed (simplified)
        gamma = physics_params.get('specific_heat_ratio', 1.4)
        R = physics_params.get('gas_constant', 287.0)
        
        rho_avg = 0.5 * (Q_left[0] + Q_right[0])
        if rho_avg <= 0:
            return np.zeros_like(Q_left)
        
        # Rough pressure estimate
        E_avg = 0.5 * (Q_left[2] + Q_right[2])
        u_avg = 0.5 * (Q_left[1]/Q_left[0] + Q_right[1]/Q_right[0])
        e_internal = E_avg - 0.5 * rho_avg * u_avg**2
        
        if e_internal <= 0:
            return np.zeros_like(Q_left)
        
        T_avg = e_internal / (rho_avg * R / (gamma - 1))
        p_avg = rho_avg * R * T_avg
        c_avg = np.sqrt(gamma * p_avg / rho_avg)
        
        # Maximum wave speed estimate
        lambda_max = abs(u_avg) + c_avg
        
        # Physical fluxes
        F_L = LNSNumerics._compute_physical_flux_1d(Q_left, 0.5 * (gamma - 1) * Q_left[2])
        F_R = LNSNumerics._compute_physical_flux_1d(Q_right, 0.5 * (gamma - 1) * Q_right[2])
        
        # Lax-Friedrichs flux
        flux = 0.5 * (F_L + F_R) - 0.5 * lambda_max * (Q_right - Q_left)
        
        return flux
    
    @staticmethod
    def semi_implicit_source_update(
        Q_old: np.ndarray,
        Q_nsf_targets: np.ndarray,
        objective_derivatives: np.ndarray,
        relaxation_times: Dict[str, float],
        dt: float,
        variable_indices: Dict[str, Union[int, List[int]]]
    ) -> np.ndarray:
        """
        Semi-implicit update for source terms to handle stiff relaxation.
        
        This addresses the mixed-order accuracy issue in the previous implementation
        by properly handling the relaxation terms implicitly.
        
        Args:
            Q_old: Previous state
            Q_nsf_targets: NSF target values
            objective_derivatives: Convective parts of objective derivatives
            relaxation_times: Relaxation time constants
            dt: Time step
            variable_indices: Mapping of variables to indices
            
        Returns:
            Updated state after source term integration
        """
        Q_new = Q_old.copy()
        
        # Heat flux update (semi-implicit)
        if 'heat_flux' in variable_indices:
            idx = variable_indices['heat_flux']
            tau_q = relaxation_times['tau_q']
            
            if isinstance(idx, list):
                for i in idx:
                    q_old = Q_old[i]
                    q_nsf = Q_nsf_targets[i]
                    D_conv = objective_derivatives[i]
                    
                    # Semi-implicit: (q_new - q_old)/dt = -(q_new - q_nsf)/tau_q - D_conv
                    # Solve: q_new = (q_old + dt*(q_nsf/tau_q - D_conv)) / (1 + dt/tau_q)
                    numerator = q_old + dt * (q_nsf / tau_q - D_conv)
                    Q_new[i] = numerator / (1.0 + dt / tau_q)
            else:
                q_old = Q_old[idx]
                q_nsf = Q_nsf_targets[idx]
                D_conv = objective_derivatives[idx]
                
                numerator = q_old + dt * (q_nsf / tau_q - D_conv)
                Q_new[idx] = numerator / (1.0 + dt / tau_q)
        
        # Stress update (semi-implicit)
        if 'stress' in variable_indices:
            idx = variable_indices['stress']
            tau_sigma = relaxation_times['tau_sigma']
            
            if isinstance(idx, list):
                for i in idx:
                    sigma_old = Q_old[i]
                    sigma_nsf = Q_nsf_targets[i]
                    D_conv = objective_derivatives[i]
                    
                    numerator = sigma_old + dt * (sigma_nsf / tau_sigma - D_conv)
                    Q_new[i] = numerator / (1.0 + dt / tau_sigma)
            else:
                sigma_old = Q_old[idx]
                sigma_nsf = Q_nsf_targets[idx]
                D_conv = objective_derivatives[idx]
                
                numerator = sigma_old + dt * (sigma_nsf / tau_sigma - D_conv)
                Q_new[idx] = numerator / (1.0 + dt / tau_sigma)
        
        return Q_new
    
    @staticmethod
    def ssp_rk2_step(
        Q_current: np.ndarray,
        rhs_function: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> np.ndarray:
        """
        Strong Stability Preserving Runge-Kutta 2nd order time step with positivity preservation.
        
        Args:
            Q_current: Current state
            rhs_function: Function that computes dQ/dt
            dt: Time step
            
        Returns:
            Updated state after SSP-RK2 step
        """
        # Stage 1: Forward Euler
        Q1 = Q_current + dt * rhs_function(Q_current)
        
        # Apply positivity limiter to intermediate stage
        Q1 = LNSNumerics._apply_positivity_limiter(Q1)
        
        # Stage 2: Average with corrected step
        Q2 = Q1 + dt * rhs_function(Q1)
        Q_new = 0.5 * (Q_current + Q2)
        
        # Apply positivity limiter to final result
        Q_new = LNSNumerics._apply_positivity_limiter(Q_new)
        
        return Q_new
    
    @staticmethod
    def _apply_positivity_limiter(Q: np.ndarray) -> np.ndarray:
        """
        Apply positivity-preserving limiter to conserved variables.
        
        Ensures that density > 0 and internal energy > 0.
        
        Args:
            Q: State array [N_cells, N_vars] where vars = [ρ, ρu, E, q, σ]
            
        Returns:
            Limited state array with physical values
        """
        Q_limited = Q.copy()
        
        # Minimum values for stability (more aggressive)
        rho_min = 1e-10
        e_min = 1e-3  # Higher minimum energy for stability
        
        # Limit density
        Q_limited[:, 0] = np.maximum(Q_limited[:, 0], rho_min)
        
        # Limit internal energy (total energy - kinetic energy > e_min)
        rho = Q_limited[:, 0]
        rho_u = Q_limited[:, 1]
        E_total = Q_limited[:, 2]
        
        u = rho_u / rho  # Safe since rho > rho_min
        kinetic_energy = 0.5 * rho * u**2
        e_internal = E_total - kinetic_energy
        
        # If internal energy is too small, adjust total energy
        mask = e_internal < e_min
        if np.any(mask):
            Q_limited[mask, 2] = kinetic_energy[mask] + e_min
        
        return Q_limited
    
    @staticmethod
    def compute_time_step_cfl(
        state: LNSState,
        physics_params: Dict[str, float],
        dx: float,
        dy: Optional[float] = None,
        cfl_target: float = 0.8
    ) -> float:
        """
        Compute adaptive time step based on CFL condition.
        
        Args:
            state: Current state
            physics_params: Physical parameters
            dx, dy: Grid spacing
            cfl_target: Target CFL number
            
        Returns:
            Maximum stable time step
        """
        # Get primitive variables
        primitives = state.get_primitive_variables()
        
        # Maximum velocity
        if state.grid.ndim == 1:
            max_velocity = np.max(np.abs(primitives['velocity']))
        else:
            max_velocity = np.max(np.sqrt(
                primitives['velocity_x']**2 + primitives['velocity_y']**2
            ))
        
        # Acoustic speed
        gamma = physics_params.get('specific_heat_ratio', 1.4)
        c_sound = np.sqrt(gamma * np.max(primitives['pressure']) / np.max(primitives['density']))
        
        # LNS wave speeds
        tau_q = physics_params.get('tau_q', 1e-6)
        tau_sigma = physics_params.get('tau_sigma', 1e-6)
        k_thermal = physics_params.get('k_thermal', 0.025)
        mu_viscous = physics_params.get('mu_viscous', 1e-3)
        
        rho_max = np.max(primitives['density'])
        cv = physics_params.get('gas_constant', 287.0) / (gamma - 1)
        
        c_thermal = np.sqrt(k_thermal / (rho_max * cv * tau_q)) if tau_q > 0 else 0
        c_viscous = np.sqrt(mu_viscous / (rho_max * tau_sigma)) if tau_sigma > 0 else 0
        
        # Maximum wave speed
        c_max = max(c_sound + max_velocity, c_thermal, c_viscous)
        
        # CFL condition
        dt_cfl = cfl_target * dx / c_max
        
        if dy is not None:
            dt_cfl = min(dt_cfl, cfl_target * dy / c_max)
        
        logger.debug(f"CFL time step: {dt_cfl:.2e} s (c_max={c_max:.2e} m/s)")
        
        return dt_cfl
    
    def __repr__(self) -> str:
        """String representation of numerics."""
        return f"LNSNumerics(numba={'enabled' if self.use_numba else 'disabled'})"