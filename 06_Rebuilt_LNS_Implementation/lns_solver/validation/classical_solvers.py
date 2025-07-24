"""
Classical Reference Solvers for LNS Validation.

This module provides reference implementations of classical fluid dynamics
equations for comparison with the LNS solver:
- Euler equations (inviscid gas dynamics)
- Classical Navier-Stokes equations
- Heat diffusion equation (Fourier)

These serve as baseline comparisons to demonstrate LNS improvements.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging

from lns_solver.core.grid import LNSGrid
from lns_solver.core.numerics import LNSNumerics

logger = logging.getLogger(__name__)

# Type aliases
StateArray = np.ndarray
ClassicalSolution = Dict[str, np.ndarray]


class EulerSolver1D:
    """
    1D Euler equations solver for inviscid gas dynamics.
    
    Solves the system:
    ∂ρ/∂t + ∂(ρu)/∂x = 0
    ∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0  
    ∂E/∂t + ∂((E + p)u)/∂x = 0
    
    This provides the inviscid baseline for comparison with LNS.
    """
    
    def __init__(self, grid: LNSGrid, gamma: float = 1.4, R: float = 287.0):
        """
        Initialize Euler solver.
        
        Args:
            grid: Computational grid
            gamma: Specific heat ratio
            R: Gas constant
        """
        if grid.ndim != 1:
            raise ValueError("EulerSolver1D requires 1D grid")
            
        self.grid = grid
        self.gamma = gamma
        self.R = R
        self.numerics = LNSNumerics()
        
        # State vector: [ρ, ρu, E]
        self.Q = np.zeros((grid.nx, 3))
        self.t_current = 0.0
        
    def initialize_sod_shock_tube(self) -> None:
        """Initialize standard Sod shock tube."""
        # Left and right states (using physical units)
        rho_L, u_L, p_L = 1.0, 0.0, 101325.0
        rho_R, u_R, p_R = 0.125, 0.0, 10132.5
        
        # Convert to conservative variables
        for i, x in enumerate(self.grid.x):
            if x < 0.5:  # Left state
                rho, u, p = rho_L, u_L, p_L
            else:  # Right state
                rho, u, p = rho_R, u_R, p_R
            
            # Conservative variables
            E = p / (self.gamma - 1) + 0.5 * rho * u**2
            self.Q[i, :] = [rho, rho * u, E]
    
    def get_primitive_variables(self) -> Dict[str, np.ndarray]:
        """Convert conservative to primitive variables."""
        rho = self.Q[:, 0]
        rho_u = self.Q[:, 1]
        E = self.Q[:, 2]
        
        # Validate
        if np.any(rho <= 0):
            raise ValueError("Non-physical density in Euler solver")
        
        u = rho_u / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * u**2)
        
        if np.any(p <= 0):
            raise ValueError("Non-physical pressure in Euler solver")
        
        T = p / (rho * self.R)
        
        return {
            'density': rho,
            'velocity': u,
            'pressure': p,
            'temperature': T
        }
    
    def compute_euler_flux(self, Q_L: np.ndarray, Q_R: np.ndarray) -> np.ndarray:
        """Compute Euler flux using HLL solver."""
        # Left state
        rho_L, rho_u_L, E_L = Q_L
        if rho_L <= 0:
            return np.zeros(3)
        u_L = rho_u_L / rho_L
        p_L = (self.gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
        c_L = np.sqrt(self.gamma * p_L / rho_L) if p_L > 0 else 0
        
        # Right state
        rho_R, rho_u_R, E_R = Q_R
        if rho_R <= 0:
            return np.zeros(3)
        u_R = rho_u_R / rho_R
        p_R = (self.gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)
        c_R = np.sqrt(self.gamma * p_R / rho_R) if p_R > 0 else 0
        
        # Wave speed estimates
        S_L = min(u_L - c_L, u_R - c_R)
        S_R = max(u_L + c_L, u_R + c_R)
        
        # Physical fluxes
        F_L = np.array([
            rho_u_L,
            rho_u_L * u_L + p_L,
            (E_L + p_L) * u_L
        ])
        
        F_R = np.array([
            rho_u_R,
            rho_u_R * u_R + p_R,
            (E_R + p_R) * u_R
        ])
        
        # HLL flux
        if S_L >= 0:
            return F_L
        elif S_R <= 0:
            return F_R
        else:
            return (S_R * F_L - S_L * F_R + S_L * S_R * (Q_R - Q_L)) / (S_R - S_L)
    
    def compute_time_step(self, cfl: float = 0.8) -> float:
        """Compute stable time step."""
        try:
            primitives = self.get_primitive_variables()
            max_speed = np.max(np.abs(primitives['velocity']) + 
                             np.sqrt(self.gamma * primitives['pressure'] / primitives['density']))
            return cfl * self.grid.dx / max_speed if max_speed > 0 else 1e-6
        except:
            return 1e-6
    
    def take_time_step(self, dt: float) -> None:
        """Take single time step using SSP-RK2."""
        def euler_rhs(Q):
            return self.numerics.compute_hyperbolic_rhs_1d(Q, self.compute_euler_flux, self.grid.dx)
        
        self.Q = self.numerics.ssp_rk2_step(self.Q, euler_rhs, dt)
        self.t_current += dt
    
    def solve(self, t_final: float, cfl: float = 0.8) -> ClassicalSolution:
        """Solve Euler equations to final time."""
        times = [self.t_current]
        solutions = [self.get_primitive_variables()]
        
        while self.t_current < t_final:
            dt = min(self.compute_time_step(cfl), t_final - self.t_current)
            self.take_time_step(dt)
            
            if len(times) % 10 == 0 or self.t_current >= t_final:
                times.append(self.t_current)
                try:
                    solutions.append(self.get_primitive_variables())
                except:
                    logger.warning(f"Euler solver became unstable at t={self.t_current:.3e}")
                    break
        
        return {
            'times': times,
            'solutions': solutions,
            'grid': self.grid
        }


class NavierStokesSolver1D:
    """
    1D compressible Navier-Stokes solver with classical constitutive relations.
    
    Solves the system with classical NSF relations:
    ∂ρ/∂t + ∂(ρu)/∂x = 0
    ∂(ρu)/∂t + ∂(ρu² + p - σ)/∂x = 0
    ∂E/∂t + ∂((E + p)u - σu + q)/∂x = 0
    
    where:
    σ = (4/3)μ ∂u/∂x  (viscous stress)
    q = -k ∂T/∂x      (heat flux)
    """
    
    def __init__(
        self, 
        grid: LNSGrid, 
        gamma: float = 1.4, 
        R: float = 287.0,
        mu: float = 1e-5,
        k_thermal: float = 0.025
    ):
        """
        Initialize classical Navier-Stokes solver.
        
        Args:
            grid: Computational grid
            gamma: Specific heat ratio
            R: Gas constant  
            mu: Dynamic viscosity
            k_thermal: Thermal conductivity
        """
        if grid.ndim != 1:
            raise ValueError("NavierStokesSolver1D requires 1D grid")
            
        self.grid = grid
        self.gamma = gamma
        self.R = R
        self.mu = mu
        self.k_thermal = k_thermal
        self.numerics = LNSNumerics()
        
        # State vector: [ρ, ρu, E]
        self.Q = np.zeros((grid.nx, 3))
        self.t_current = 0.0
        
    def initialize_sod_shock_tube(self) -> None:
        """Initialize standard Sod shock tube."""
        # Same as Euler solver
        rho_L, u_L, p_L = 1.0, 0.0, 101325.0
        rho_R, u_R, p_R = 0.125, 0.0, 10132.5
        
        for i, x in enumerate(self.grid.x):
            if x < 0.5:
                rho, u, p = rho_L, u_L, p_L
            else:
                rho, u, p = rho_R, u_R, p_R
            
            E = p / (self.gamma - 1) + 0.5 * rho * u**2
            self.Q[i, :] = [rho, rho * u, E]
    
    def initialize_heat_conduction(
        self, 
        T_left: float = 400.0, 
        T_right: float = 300.0
    ) -> None:
        """Initialize heat conduction test case."""
        rho_init = 1.0
        u_init = 0.0
        
        # Linear temperature profile
        x_min, x_max = np.min(self.grid.x), np.max(self.grid.x)
        T_profile = T_left + (T_right - T_left) * (self.grid.x - x_min) / (x_max - x_min)
        
        for i, T in enumerate(T_profile):
            p = rho_init * self.R * T
            E = p / (self.gamma - 1) + 0.5 * rho_init * u_init**2
            self.Q[i, :] = [rho_init, rho_init * u_init, E]
    
    def get_primitive_variables(self) -> Dict[str, np.ndarray]:
        """Convert conservative to primitive variables."""
        rho = self.Q[:, 0]
        rho_u = self.Q[:, 1]
        E = self.Q[:, 2]
        
        if np.any(rho <= 0):
            raise ValueError("Non-physical density in NS solver")
        
        u = rho_u / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * u**2)
        
        if np.any(p <= 0):
            raise ValueError("Non-physical pressure in NS solver")
        
        T = p / (rho * self.R)
        
        return {
            'density': rho,
            'velocity': u,
            'pressure': p,
            'temperature': T
        }
    
    def compute_viscous_flux(self, Q: np.ndarray) -> np.ndarray:
        """Compute viscous flux terms."""
        try:
            primitives = self.get_primitive_variables()
            u = primitives['velocity']
            T = primitives['temperature']
        except:
            return np.zeros_like(Q)
        
        # Compute gradients
        du_dx = np.gradient(u, self.grid.dx)
        dT_dx = np.gradient(T, self.grid.dx)
        
        # Classical NSF relations
        sigma = (4.0/3.0) * self.mu * du_dx  # Viscous stress
        q = -self.k_thermal * dT_dx         # Heat flux
        
        # Viscous flux contributions
        F_viscous = np.zeros_like(Q)
        F_viscous[:, 0] = 0                    # No mass flux
        F_viscous[:, 1] = sigma                # Momentum flux
        F_viscous[:, 2] = sigma * u + q        # Energy flux
        
        return F_viscous
    
    def compute_ns_flux(self, Q_L: np.ndarray, Q_R: np.ndarray) -> np.ndarray:
        """Compute Navier-Stokes flux (inviscid part only for interface)."""
        # Use Euler flux for convective terms
        # Viscous terms are handled as source terms
        return EulerSolver1D.compute_euler_flux(self, Q_L, Q_R)
    
    def compute_time_step(self, cfl: float = 0.5) -> float:
        """Compute stable time step including viscous constraints."""
        try:
            primitives = self.get_primitive_variables()
            
            # Convective time step
            max_speed = np.max(np.abs(primitives['velocity']) + 
                             np.sqrt(self.gamma * primitives['pressure'] / primitives['density']))
            dt_convective = cfl * self.grid.dx / max_speed if max_speed > 0 else 1e-6
            
            # Viscous time step constraint
            rho_min = np.min(primitives['density'])
            dt_viscous = 0.5 * rho_min * self.grid.dx**2 / self.mu if self.mu > 0 else np.inf
            
            # Thermal time step constraint  
            cv = self.R / (self.gamma - 1)
            dt_thermal = 0.5 * rho_min * cv * self.grid.dx**2 / self.k_thermal if self.k_thermal > 0 else np.inf
            
            return min(dt_convective, dt_viscous, dt_thermal)
        except:
            return 1e-8
    
    def take_time_step(self, dt: float) -> None:
        """Take single time step with operator splitting."""
        # Step 1: Convective terms (hyperbolic)
        def convective_rhs(Q):
            return self.numerics.compute_hyperbolic_rhs_1d(Q, self.compute_ns_flux, self.grid.dx)
        
        Q_intermediate = self.numerics.ssp_rk2_step(self.Q, convective_rhs, dt)
        
        # Step 2: Viscous terms (parabolic) - explicit for simplicity
        try:
            self.Q = Q_intermediate
            F_viscous = self.compute_viscous_flux(self.Q)
            
            # Add viscous contribution as source term
            dF_dx = np.gradient(F_viscous, self.grid.dx, axis=0)
            self.Q += dt * dF_dx
            
        except Exception as e:
            logger.warning(f"Viscous step failed: {e}")
            self.Q = Q_intermediate
        
        self.t_current += dt
    
    def solve(self, t_final: float, cfl: float = 0.5) -> ClassicalSolution:
        """Solve Navier-Stokes equations to final time."""
        times = [self.t_current]
        solutions = [self.get_primitive_variables()]
        
        while self.t_current < t_final:
            dt = min(self.compute_time_step(cfl), t_final - self.t_current)
            
            try:
                self.take_time_step(dt)
                
                if len(times) % 10 == 0 or self.t_current >= t_final:
                    times.append(self.t_current)
                    solutions.append(self.get_primitive_variables())
                    
            except Exception as e:
                logger.warning(f"NS solver became unstable at t={self.t_current:.3e}: {e}")
                break
        
        return {
            'times': times,
            'solutions': solutions,
            'grid': self.grid
        }


class HeatDiffusionSolver1D:
    """
    1D heat diffusion equation solver (Fourier's law).
    
    Solves: ∂T/∂t = α ∇²T
    where α = k/(ρc_p) is thermal diffusivity.
    
    This provides the classical parabolic baseline for heat transport.
    """
    
    def __init__(
        self, 
        grid: LNSGrid, 
        thermal_diffusivity: float = 1e-5
    ):
        """
        Initialize heat diffusion solver.
        
        Args:
            grid: Computational grid
            thermal_diffusivity: α = k/(ρc_p)
        """
        if grid.ndim != 1:
            raise ValueError("HeatDiffusionSolver1D requires 1D grid")
            
        self.grid = grid
        self.alpha = thermal_diffusivity
        
        # Temperature field
        self.T = np.zeros(grid.nx)
        self.t_current = 0.0
    
    def initialize_step(
        self, 
        T_left: float = 400.0, 
        T_right: float = 300.0, 
        x_step: float = 0.5
    ) -> None:
        """Initialize with step function."""
        self.T = T_left * (self.grid.x < x_step) + T_right * (self.grid.x >= x_step)
    
    def initialize_linear(
        self, 
        T_left: float = 400.0, 
        T_right: float = 300.0
    ) -> None:
        """Initialize with linear profile."""
        x_min, x_max = np.min(self.grid.x), np.max(self.grid.x)
        self.T = T_left + (T_right - T_left) * (self.grid.x - x_min) / (x_max - x_min)
    
    def apply_boundary_conditions(self, T_left: float, T_right: float) -> None:
        """Apply Dirichlet boundary conditions."""
        self.T[0] = T_left
        self.T[-1] = T_right
    
    def compute_time_step(self, cfl: float = 0.5) -> float:
        """Compute stable time step for diffusion."""
        return cfl * self.grid.dx**2 / (2 * self.alpha)
    
    def take_time_step(self, dt: float, T_left: float = None, T_right: float = None) -> None:
        """Take single time step using explicit finite difference."""
        # Second derivative (central difference)
        d2T_dx2 = np.zeros_like(self.T)
        d2T_dx2[1:-1] = (self.T[2:] - 2*self.T[1:-1] + self.T[:-2]) / self.grid.dx**2
        
        # Time step
        self.T += dt * self.alpha * d2T_dx2
        
        # Apply boundary conditions
        if T_left is not None:
            self.T[0] = T_left
        if T_right is not None:
            self.T[-1] = T_right
            
        self.t_current += dt
    
    def solve(
        self, 
        t_final: float, 
        T_left: float = 400.0, 
        T_right: float = 300.0,
        cfl: float = 0.5
    ) -> ClassicalSolution:
        """Solve heat diffusion equation to final time."""
        times = [self.t_current]
        temperatures = [self.T.copy()]
        
        while self.t_current < t_final:
            dt = min(self.compute_time_step(cfl), t_final - self.t_current)
            self.take_time_step(dt, T_left, T_right)
            
            if len(times) % 10 == 0 or self.t_current >= t_final:
                times.append(self.t_current)
                temperatures.append(self.T.copy())
        
        return {
            'times': times,
            'temperatures': temperatures,
            'grid': self.grid
        }


def compare_euler_vs_lns():
    """Quick comparison test."""
    from lns_solver import LNSSolver1D, LNSGrid
    
    # Create grid
    grid = LNSGrid.create_uniform_1d(50, 0.0, 1.0)
    
    # Euler solver
    euler = EulerSolver1D(grid)
    euler.initialize_sod_shock_tube()
    
    # LNS solver
    lns = LNSSolver1D.create_sod_shock_tube(nx=50)
    
    print("Comparison Setup Complete:")
    print(f"  Grid: {grid.nx} cells from {grid.x[0]:.1f} to {grid.x[-1]:.1f}")
    print(f"  Euler solver: {len(euler.Q)} cells, {euler.Q.shape[1]} variables")
    print(f"  LNS solver: {lns.state.Q.shape[0]} cells, {lns.state.Q.shape[1]} variables")


if __name__ == "__main__":
    compare_euler_vs_lns()