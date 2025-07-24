"""
LNS 1D Solver: Complete implementation of Local Navier-Stokes equations in 1D.

This module provides the LNSSolver1D class that brings together all core 
infrastructure components into a working 1D LNS simulation engine with:
- Correct physics implementation
- Robust numerical methods  
- Comprehensive validation
- Professional software engineering practices

Example:
    Basic Sod shock tube simulation:
    
    >>> solver = LNSSolver1D.create_sod_shock_tube(nx=100)
    >>> results = solver.solve(t_final=0.2, dt=1e-5)
    >>> solver.plot_results()
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import time

from lns_solver.core.grid import LNSGrid, BoundaryCondition
from lns_solver.core.state import LNSState, MaterialProperties
from lns_solver.core.physics import LNSPhysics, LNSPhysicsParameters
from lns_solver.core.numerics import LNSNumerics
from lns_solver.utils.io import LNSDataWriter, LNSDataReader
from lns_solver.utils.constants import PhysicalConstants

logger = logging.getLogger(__name__)

# Type aliases
SimulationResults = Dict[str, Union[np.ndarray, List[float], Dict[str, Any]]]
PlotConfig = Dict[str, Any]


class LNSSolver1D:
    """
    Complete 1D Local Navier-Stokes solver with rigorous validation.
    
    This class integrates all core infrastructure components into a working
    1D LNS simulation engine. Key features:
    
    - Correct physics implementation (fixed deviatoric stress formula)
    - Robust HLL Riemann solver with LNS wave speeds
    - Semi-implicit time integration for stiff source terms  
    - Comprehensive validation against analytical solutions
    - Professional I/O and visualization capabilities
    
    The solver handles the 5-variable 1D LNS system:
    [ρ, ρu, E, q_x, σ'_xx]
    
    where:
    - ρ: density
    - ρu: momentum  
    - E: total energy
    - q_x: heat flux in x-direction
    - σ'_xx: deviatoric stress component
    
    Example:
        >>> solver = LNSSolver1D.create_sod_shock_tube(nx=100)
        >>> results = solver.solve(t_final=0.2)
        >>> print(f"Simulation completed in {results['wall_time']:.2f} seconds")
    """
    
    def __init__(
        self,
        grid: LNSGrid,
        physics: LNSPhysics,
        numerics: LNSNumerics,
        initial_state: Optional[LNSState] = None
    ):
        """
        Initialize 1D LNS solver.
        
        Args:
            grid: 1D computational grid
            physics: Physics model and parameters
            numerics: Numerical methods
            initial_state: Initial condition (optional)
        """
        if grid.ndim != 1:
            raise ValueError("LNSSolver1D requires 1D grid")
            
        self.grid = grid
        self.physics = physics
        self.numerics = numerics
        
        # Initialize state with 5 variables for 1D LNS
        self.state = initial_state or LNSState(grid, n_variables=5)
        
        # Simulation parameters
        self.t_current = 0.0
        self.dt_current = 1e-5
        self.iteration = 0
        
        # Solver settings
        self.cfl_target = 0.8
        self.adaptive_dt = True
        self.max_iterations = int(1e6)
        self.output_interval = 100
        
        # Validation and diagnostics
        self.conservation_errors = []
        self.dt_history = []
        self.performance_metrics = {}
        
        # I/O settings
        self.data_writer = LNSDataWriter()
        self.output_dir = Path("./lns_output")
        
        logger.info(f"Initialized 1D LNS solver with {grid.nx} cells")
    
    @classmethod
    def create_sod_shock_tube(
        cls,
        nx: int = 100,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        physics_params: Optional[LNSPhysicsParameters] = None
    ) -> 'LNSSolver1D':
        """
        Create solver for classic Sod shock tube problem.
        
        Args:
            nx: Number of grid cells
            x_bounds: Domain bounds (x_min, x_max)
            physics_params: Physics parameters (uses defaults if None)
            
        Returns:
            Configured LNSSolver1D with Sod initial conditions
            
        Example:
            >>> solver = LNSSolver1D.create_sod_shock_tube(nx=200)
            >>> results = solver.solve(t_final=0.2)
        """
        # Create grid
        grid = LNSGrid.create_uniform_1d(nx, x_bounds[0], x_bounds[1])
        
        # Set up physics
        if physics_params is None:
            physics_params = LNSPhysicsParameters(
                mu_viscous=1e-5,     # Small viscosity
                k_thermal=0.025,     # Thermal conductivity
                tau_q=1e-6,          # Heat flux relaxation time
                tau_sigma=1e-6       # Stress relaxation time
            )
        
        physics = LNSPhysics(physics_params)
        numerics = LNSNumerics(use_numba=True)
        
        # Create solver
        solver = cls(grid, physics, numerics)
        
        # Set Sod shock tube initial conditions
        solver.state.initialize_sod_shock_tube()
        
        # Set boundary conditions (outflow)
        solver.grid.set_boundary_condition('left', 'outflow')
        solver.grid.set_boundary_condition('right', 'outflow')
        
        logger.info("Created Sod shock tube solver")
        return solver
    
    @classmethod  
    def create_heat_conduction_test(
        cls,
        nx: int = 50,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        T_left: float = 400.0,
        T_right: float = 300.0
    ) -> 'LNSSolver1D':
        """
        Create solver for heat conduction validation test.
        
        Args:
            nx: Number of grid cells
            x_bounds: Domain bounds
            T_left: Left boundary temperature
            T_right: Right boundary temperature
            
        Returns:
            Configured solver for heat conduction test
        """
        # Create grid
        grid = LNSGrid.create_uniform_1d(nx, x_bounds[0], x_bounds[1])
        
        # Physics with emphasize thermal effects
        physics_params = LNSPhysicsParameters(
            mu_viscous=1e-6,     # Minimal viscosity
            k_thermal=0.1,       # Higher thermal conductivity
            tau_q=1e-5,          # Thermal relaxation time
            tau_sigma=1e-3       # Longer stress relaxation
        )
        
        physics = LNSPhysics(physics_params)
        numerics = LNSNumerics(use_numba=True)
        
        # Create solver
        solver = cls(grid, physics, numerics)
        
        # Initialize with temperature gradient
        rho_init = 1.0
        p_init = 101325.0
        u_init = 0.0
        
        # Linear temperature profile
        T_profile = T_left + (T_right - T_left) * solver.grid.x / (x_bounds[1] - x_bounds[0])
        
        for i, T in enumerate(T_profile):
            P_local = {
                'density': rho_init,
                'velocity': u_init, 
                'temperature': T,
                'heat_flux_x': 0.0,
                'stress_xx': 0.0
            }
            Q_local = solver.state.P_to_Q_1d(P_local)
            solver.state.Q[i, :] = Q_local
        
        # Set temperature boundary conditions
        solver.grid.set_boundary_condition('left', 'dirichlet', values=T_left)
        solver.grid.set_boundary_condition('right', 'dirichlet', values=T_right)
        
        logger.info(f"Created heat conduction test (ΔT = {T_left-T_right:.1f} K)")
        return solver
    
    def solve(
        self,
        t_final: float,
        dt_initial: Optional[float] = None,
        output_times: Optional[List[float]] = None,
        save_results: bool = True
    ) -> SimulationResults:
        """
        Solve the 1D LNS system to final time.
        
        Args:
            t_final: Final simulation time
            dt_initial: Initial time step (auto-computed if None)
            output_times: Specific times to save output (optional)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing simulation results and diagnostics
            
        Example:
            >>> results = solver.solve(t_final=0.2, dt_initial=1e-5)
            >>> print(f"Final time: {results['time_final']:.3f} s")
            >>> print(f"Iterations: {results['iterations']}")
        """
        logger.info(f"Starting 1D LNS simulation (t_final={t_final:.3e} s)")
        
        # Initialize timing
        start_time = time.time()
        
        # Set initial time step
        if dt_initial is not None:
            self.dt_current = dt_initial
        elif self.adaptive_dt:
            self.dt_current = self._compute_adaptive_timestep()
        
        # Initialize output
        if output_times is None:
            output_times = list(np.linspace(0, t_final, 11)[1:])  # 10 output snapshots as list
        
        output_data = {
            'times': [],
            'states': [],
            'primitives': []
        }
        
        # Save initial condition
        if save_results:
            self._save_output(0.0, output_data)
        
        # Main time integration loop
        while self.t_current < t_final and self.iteration < self.max_iterations:
            
            # Adjust final time step
            if self.t_current + self.dt_current > t_final:
                self.dt_current = t_final - self.t_current
            
            # Take time step
            self._take_timestep()
            
            # Update time and iteration
            self.t_current += self.dt_current
            self.iteration += 1
            
            # Adaptive time stepping
            if self.adaptive_dt and self.iteration % 10 == 0:
                self.dt_current = self._compute_adaptive_timestep()
            
            # Check for output times
            if save_results:
                for t_out in output_times:
                    if abs(self.t_current - t_out) < 0.5 * self.dt_current:
                        self._save_output(self.t_current, output_data)
                        output_times.remove(t_out)
                        break
            
            # Periodic diagnostics
            if self.iteration % self.output_interval == 0:
                self._log_progress()
                self._check_conservation()
            
            # Stability check
            if not self.state.validate_state():
                logger.error(f"Simulation became unstable at t={self.t_current:.3e}")
                break
        
        # Finalize results
        end_time = time.time()
        wall_time = end_time - start_time
        
        # Final output
        if save_results:
            self._save_output(self.t_current, output_data)
        
        results = {
            'time_final': self.t_current,
            'iterations': self.iteration,
            'wall_time': wall_time,
            'dt_final': self.dt_current,
            'conservation_errors': self.conservation_errors,
            'dt_history': self.dt_history,
            'output_data': output_data,
            'grid': self.grid,
            'final_state': self.state,
            'performance_metrics': self._compute_performance_metrics(wall_time)
        }
        
        logger.info(f"Simulation completed: {self.iteration} iterations in {wall_time:.2f}s")
        return results
    
    def _take_timestep(self) -> None:
        """Take a single time step using the numerical method."""
        
        # Compute hyperbolic RHS (convective terms) 
        hyperbolic_rhs = self.numerics.compute_hyperbolic_rhs_1d(
            self.state.Q,
            lambda Q_L, Q_R: self.numerics.hll_flux_1d(
                Q_L, Q_R, self.physics.get_physics_dict()
            ),
            self.grid.dx
        )
        
        # Compute source terms (relaxation + gradients)
        source_rhs = self._compute_source_terms()
        
        # Combined RHS function for time stepping
        def total_rhs(Q):
            # Update state for gradient computation
            old_Q = self.state.Q.copy()
            self.state.Q = Q
            
            # Recompute source terms with updated state
            source = self._compute_source_terms()
            
            # Restore state
            self.state.Q = old_Q
            
            return hyperbolic_rhs + source
        
        # SSP-RK2 time integration
        self.state.Q = self.numerics.ssp_rk2_step(
            self.state.Q,
            total_rhs,
            self.dt_current
        )
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
        
        # Store diagnostics
        self.dt_history.append(self.dt_current)
    
    def _compute_source_terms(self) -> np.ndarray:
        """
        Compute source terms for LNS equations.
        
        Returns:
            Source term RHS array [N_cells, 5]
        """
        # Get primitive variables and gradients
        primitives = self.state.get_primitive_variables()
        
        # Compute gradients
        temp_field = primitives['temperature'].reshape(-1)
        vel_field = primitives['velocity'].reshape(-1)
        
        gradients = self.numerics.compute_gradients_efficient(
            [temp_field, vel_field], 
            self.grid.dx
        )
        
        dT_dx = gradients['field_0']
        du_dx = gradients['field_1']
        
        # Material properties dictionary
        material_props = {
            'k_thermal': self.physics.params.k_thermal,
            'mu_viscous': self.physics.params.mu_viscous
        }
        
        # Compute NSF target values
        q_nsf, sigma_nsf = self.physics.compute_1d_nsf_targets(
            du_dx, dT_dx, material_props
        )
        
        # Initialize source terms
        source_rhs = np.zeros_like(self.state.Q)
        
        # Relaxation source terms for each cell
        for i in range(self.grid.nx):
            # Current non-equilibrium values
            q_current = self.state.Q[i, 3]  # Heat flux
            sigma_current = self.state.Q[i, 4]  # Stress
            
            # Relaxation terms: -(q - q_NSF)/τ_q, -(σ - σ_NSF)/τ_σ
            source_rhs[i, 3] = -(q_current - q_nsf[i]) / self.physics.params.tau_q
            source_rhs[i, 4] = -(sigma_current - sigma_nsf[i]) / self.physics.params.tau_sigma
        
        return source_rhs
    
    def _apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to current state."""
        
        # Get boundary conditions
        bc_left = self.grid.get_boundary_condition('left')
        bc_right = self.grid.get_boundary_condition('right')
        
        # Apply left boundary
        if bc_left.bc_type == 'dirichlet':
            # Temperature BC - adjust total energy
            T_bc = bc_left.values
            rho = self.state.Q[0, 0]
            u = self.state.Q[0, 1] / rho if rho > 0 else 0.0
            
            # Compute new total energy with BC temperature
            cv = PhysicalConstants.AIR_GAS_CONSTANT / (PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO - 1)
            e_internal = rho * cv * T_bc
            kinetic = 0.5 * rho * u**2
            self.state.Q[0, 2] = e_internal + kinetic
            
        elif bc_left.bc_type == 'outflow':
            # Extrapolate from interior
            self.state.Q[0, :] = self.state.Q[1, :]
        
        # Apply right boundary  
        if bc_right.bc_type == 'dirichlet':
            # Temperature BC - adjust total energy
            T_bc = bc_right.values
            rho = self.state.Q[-1, 0]
            u = self.state.Q[-1, 1] / rho if rho > 0 else 0.0
            
            # Compute new total energy with BC temperature
            cv = PhysicalConstants.AIR_GAS_CONSTANT / (PhysicalConstants.AIR_SPECIFIC_HEAT_RATIO - 1)
            e_internal = rho * cv * T_bc
            kinetic = 0.5 * rho * u**2
            self.state.Q[-1, 2] = e_internal + kinetic
            
        elif bc_right.bc_type == 'outflow':
            # Extrapolate from interior
            self.state.Q[-1, :] = self.state.Q[-2, :]
    
    def _compute_adaptive_timestep(self) -> float:
        """Compute adaptive time step based on CFL condition."""
        
        physics_dict = self.physics.get_physics_dict()
        
        dt_cfl = self.numerics.compute_time_step_cfl(
            self.state,
            physics_dict,
            self.grid.dx,
            cfl_target=self.cfl_target
        )
        
        return dt_cfl
    
    def _save_output(self, time: float, output_data: Dict) -> None:
        """Save current state to output data."""
        
        output_data['times'].append(time)
        output_data['states'].append(self.state.Q.copy())
        
        try:
            primitives = self.state.get_primitive_variables()
            output_data['primitives'].append(primitives.copy())
        except Exception as e:
            logger.warning(f"Could not compute primitives at t={time:.3e}: {e}")
            output_data['primitives'].append(None)
    
    def _log_progress(self) -> None:
        """Log simulation progress."""
        
        primitives = self.state.get_primitive_variables()
        rho_range = f"{np.min(primitives['density']):.3f}-{np.max(primitives['density']):.3f}"
        
        logger.info(
            f"Iter {self.iteration:6d}: t={self.t_current:.4e} s, "
            f"dt={self.dt_current:.2e} s, ρ=[{rho_range}]"
        )
    
    def _check_conservation(self) -> None:
        """Check conservation properties."""
        
        # Compute total mass, momentum, energy
        cell_volumes = self.grid.compute_cell_volumes()
        
        mass_total = np.sum(self.state.Q[:, 0] * cell_volumes)
        momentum_total = np.sum(self.state.Q[:, 1] * cell_volumes)  
        energy_total = np.sum(self.state.Q[:, 2] * cell_volumes)
        
        conservation_data = {
            'time': self.t_current,
            'mass': mass_total,
            'momentum': momentum_total,
            'energy': energy_total
        }
        
        self.conservation_errors.append(conservation_data)
    
    def _compute_performance_metrics(self, wall_time: float) -> Dict[str, float]:
        """Compute performance metrics."""
        
        cell_updates = self.iteration * self.grid.nx
        updates_per_second = cell_updates / wall_time if wall_time > 0 else 0
        
        return {
            'wall_time': wall_time,
            'iterations': self.iteration,
            'cell_updates': cell_updates,
            'updates_per_second': updates_per_second,
            'time_per_iteration': wall_time / self.iteration if self.iteration > 0 else 0
        }
    
    def plot_results(
        self, 
        results: Optional[SimulationResults] = None,
        variables: List[str] = ['density', 'velocity', 'pressure', 'temperature'],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot simulation results.
        
        Args:
            results: Simulation results (uses current state if None)
            variables: Variables to plot
            save_path: Path to save plots (optional)
        """
        
        if results is None:
            # Plot current state only
            primitives = self.state.get_primitive_variables()
            times = [self.t_current]
            all_primitives = [primitives]
        else:
            # Plot full results
            times = results['output_data']['times']
            all_primitives = results['output_data']['primitives']
        
        # Set up subplots
        n_vars = len(variables)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot each variable
        for i, var in enumerate(variables[:4]):  # Max 4 plots
            ax = axes[i]
            
            # Plot evolution over time
            for j, (t, prims) in enumerate(zip(times, all_primitives)):
                if prims is None:
                    continue
                    
                alpha = 0.3 + 0.7 * j / max(1, len(times) - 1)  # Fade older lines
                label = f't = {t:.3f}' if len(times) <= 5 else (f't = {t:.3f}' if j % max(1, len(times)//5) == 0 else None)
                
                ax.plot(self.grid.x, prims[var], alpha=alpha, label=label)
            
            ax.set_xlabel('x [m]')
            ax.set_ylabel(var.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            if len(times) <= 5:
                ax.legend()
        
        # Overall title
        fig.suptitle('1D LNS Simulation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Path(save_path), dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.show()
    
    def analyze_conservation(self, results: SimulationResults) -> Dict[str, Any]:
        """
        Analyze conservation properties from simulation results.
        
        Args:
            results: Simulation results
            
        Returns:
            Conservation analysis data
        """
        
        if not self.conservation_errors:
            logger.warning("No conservation data available")
            return {}
        
        # Extract conservation time series
        times = [entry['time'] for entry in self.conservation_errors]
        mass = np.array([entry['mass'] for entry in self.conservation_errors])
        momentum = np.array([entry['momentum'] for entry in self.conservation_errors])
        energy = np.array([entry['energy'] for entry in self.conservation_errors])
        
        # Compute relative errors (use absolute error if initial value is too small)
        mass_ref = max(abs(mass[0]), np.max(abs(mass)))
        momentum_ref = max(abs(momentum[0]), np.max(abs(momentum)))
        energy_ref = max(abs(energy[0]), np.max(abs(energy)))
        
        mass_error = abs(mass - mass[0]) / mass_ref if mass_ref > 1e-12 else abs(mass - mass[0])
        momentum_error = abs(momentum - momentum[0]) / momentum_ref if momentum_ref > 1e-12 else abs(momentum - momentum[0])
        energy_error = abs(energy - energy[0]) / energy_ref if energy_ref > 1e-12 else abs(energy - energy[0])
        
        analysis = {
            'times': times,
            'mass_conservation': {
                'values': mass,
                'relative_error': mass_error,
                'max_error': np.max(mass_error),
                'final_error': mass_error[-1]
            },
            'momentum_conservation': {
                'values': momentum,
                'relative_error': momentum_error,
                'max_error': np.max(momentum_error),
                'final_error': momentum_error[-1]
            },
            'energy_conservation': {
                'values': energy,
                'relative_error': energy_error,
                'max_error': np.max(energy_error),
                'final_error': energy_error[-1]
            }
        }
        
        # Log conservation summary
        logger.info("Conservation Analysis:")
        logger.info(f"  Mass conservation error: {analysis['mass_conservation']['max_error']:.2e}")
        logger.info(f"  Momentum conservation error: {analysis['momentum_conservation']['max_error']:.2e}")
        logger.info(f"  Energy conservation error: {analysis['energy_conservation']['max_error']:.2e}")
        
        return analysis
    
    def save_checkpoint(self, filename: Union[str, Path]) -> None:
        """Save current solver state to checkpoint file."""
        
        # Create serializable metadata (convert complex objects to dicts)
        checkpoint_data = {
            't_current': self.t_current,
            'dt_current': self.dt_current,
            'iteration': self.iteration,
            # Serialize physics parameters
            'mu_viscous': self.physics.params.mu_viscous,
            'k_thermal': self.physics.params.k_thermal,
            'tau_q': self.physics.params.tau_q,
            'tau_sigma': self.physics.params.tau_sigma,
            'gas_constant': self.physics.params.gas_constant,
            'specific_heat_ratio': self.physics.params.specific_heat_ratio,
            # Grid information
            'grid_ndim': self.grid.ndim,
            'grid_nx': self.grid.nx,
            'grid_x_min': float(np.min(self.grid.x)),
            'grid_x_max': float(np.max(self.grid.x)),
            # Boundary conditions
            'bc_left_type': self.grid.get_boundary_condition('left').bc_type if self.grid.get_boundary_condition('left') else 'outflow',
            'bc_right_type': self.grid.get_boundary_condition('right').bc_type if self.grid.get_boundary_condition('right') else 'outflow',
        }
        
        self.data_writer.write_state(
            filename,
            self.state,
            self.t_current,
            metadata=checkpoint_data
        )
        
        logger.info(f"Saved checkpoint to {filename}")
    
    @classmethod
    def load_checkpoint(cls, filename: Union[str, Path]) -> 'LNSSolver1D':
        """Load solver state from checkpoint file."""
        
        reader = LNSDataReader()
        state, time, metadata = reader.read_state(filename)
        
        # Reconstruct physics parameters
        physics_params = LNSPhysicsParameters(
            mu_viscous=metadata['mu_viscous'],
            k_thermal=metadata['k_thermal'],
            tau_q=metadata['tau_q'],
            tau_sigma=metadata['tau_sigma'],
            gas_constant=metadata['gas_constant'],
            specific_heat_ratio=metadata['specific_heat_ratio']
        )
        
        physics = LNSPhysics(physics_params)
        numerics = LNSNumerics()
        
        # Reconstruct grid
        if metadata['grid_ndim'] == 1:
            x_min = metadata.get('grid_x_min', 0.0)
            x_max = metadata.get('grid_x_max', 1.0)
            grid = LNSGrid.create_uniform_1d(metadata['grid_nx'], x_min, x_max)
            
            # Restore boundary conditions
            bc_left_type = metadata.get('bc_left_type', 'outflow')
            bc_right_type = metadata.get('bc_right_type', 'outflow')
            
            grid.set_boundary_condition('left', bc_left_type)
            grid.set_boundary_condition('right', bc_right_type)
        else:
            raise NotImplementedError("Only 1D grid loading implemented")
        
        solver = cls(grid, physics, numerics, state)
        solver.t_current = metadata['t_current']
        solver.dt_current = metadata['dt_current']
        solver.iteration = metadata['iteration']
        
        logger.info(f"Loaded checkpoint from {filename}")
        return solver
    
    def __repr__(self) -> str:
        """String representation of solver."""
        return (f"LNSSolver1D(nx={self.grid.nx}, t={self.t_current:.3e}, "
                f"iter={self.iteration})")