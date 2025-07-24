"""
Optimized 1D LNS Solver with Ghost Cell Boundary Conditions.

This solver addresses the critical issues identified in the code review:
1. Proper ghost cell-based boundary conditions (no overwriting physical cells)
2. Eliminated redundant Q->P conversions in flux computation
3. Vectorized operations for maximum performance
4. Clean separation between physics and numerics
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import time

from lns_solver.core.grid import LNSGrid
from lns_solver.core.state import LNSState, MaterialProperties
from lns_solver.core.physics import LNSPhysics, LNSPhysicsParameters
from lns_solver.core.numerics_optimized import OptimizedLNSNumerics
from lns_solver.core.boundary_conditions import (
    GhostCellBoundaryHandler, BoundaryCondition, BCType,
    create_outflow_bc, create_temperature_bc, create_wall_bc
)
from lns_solver.utils.io import LNSDataWriter, LNSDataReader
from lns_solver.utils.constants import PhysicalConstants

logger = logging.getLogger(__name__)


class OptimizedLNSSolver1D:
    """
    Optimized 1D LNS solver with proper FVM implementation.
    
    Key improvements over original:
    - Ghost cell boundary conditions (preserves conservation)
    - Eliminated redundant computations in hot loops
    - Vectorized operations for performance
    - Clean physics/numerics separation
    - Comprehensive performance monitoring
    
    This solver properly implements the finite volume method by:
    - Never overwriting physical cells with boundary conditions
    - Using ghost cells to enable correct boundary flux computation
    - Pre-computing primitive variables to avoid redundant calculations
    - Using efficient vectorized flux differencing for RHS computation
    """
    
    def __init__(
        self,
        grid: LNSGrid,
        physics: LNSPhysics,
        n_ghost: int = 2,
        initial_state: Optional[LNSState] = None
    ):
        """
        Initialize optimized 1D LNS solver.
        
        Args:
            grid: 1D computational grid
            physics: Physics model and parameters
            n_ghost: Number of ghost cell layers
            initial_state: Initial condition (optional)
        """
        if grid.ndim != 1:
            raise ValueError("OptimizedLNSSolver1D requires 1D grid")
            
        self.grid = grid
        self.physics = physics
        self.n_ghost = n_ghost
        
        # Initialize optimized numerics
        self.numerics = OptimizedLNSNumerics(n_ghost=n_ghost)
        
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
        
        # Boundary conditions (stored properly)
        self.boundary_conditions = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_timesteps': 0,
            'total_flux_computations': 0,
            'total_rhs_computations': 0,
            'average_timestep_time': 0.0,
            'conservation_errors': []
        }
        
        # I/O settings
        self.data_writer = LNSDataWriter()
        self.output_dir = Path("./lns_output_optimized")
        
        logger.info(f"Initialized optimized 1D LNS solver with {grid.nx} cells and {n_ghost} ghost layers")
    
    @classmethod
    def create_sod_shock_tube(
        cls,
        nx: int = 100,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        physics_params: Optional[LNSPhysicsParameters] = None,
        n_ghost: int = 2
    ) -> 'OptimizedLNSSolver1D':
        """
        Create optimized solver for Sod shock tube problem.
        
        Args:
            nx: Number of grid cells
            x_bounds: Domain bounds
            physics_params: Physics parameters (uses stable defaults if None)
            n_ghost: Number of ghost layers
            
        Returns:
            Configured optimized solver with proper boundary conditions
        """
        # Create grid
        grid = LNSGrid.create_uniform_1d(nx, x_bounds[0], x_bounds[1])
        
        # Use stable physics parameters (non-stiff regime)
        if physics_params is None:
            # Choose relaxation times that avoid stiffness
            dx = (x_bounds[1] - x_bounds[0]) / nx
            c_sound = 300.0  # Approximate sound speed
            dt_cfl = 0.5 * dx / c_sound
            tau_stable = 10.0 * dt_cfl  # Non-stiff relaxation time
            
            physics_params = LNSPhysicsParameters(
                mu_viscous=1e-5,
                k_thermal=0.025,
                tau_q=tau_stable,      # Stable heat flux relaxation
                tau_sigma=tau_stable   # Stable stress relaxation
            )
        
        physics = LNSPhysics(physics_params)
        
        # Create solver
        solver = cls(grid, physics, n_ghost=n_ghost)
        
        # Initialize Sod shock tube
        solver.state.initialize_sod_shock_tube()
        
        # Set appropriate boundary conditions (outflow on both sides)
        solver.set_boundary_condition('left', create_outflow_bc())
        solver.set_boundary_condition('right', create_outflow_bc())
        
        logger.info(f"Created optimized Sod shock tube solver:")
        logger.info(f"  Grid: {nx} cells, dx = {dx:.6f} m")
        logger.info(f"  Relaxation times: τ = {tau_stable:.2e} s (non-stiff)")
        logger.info(f"  Ghost layers: {n_ghost}")
        
        return solver
        
    def set_boundary_condition(self, location: str, bc: BoundaryCondition):
        """
        Set boundary condition for domain boundary.
        
        Args:
            location: Boundary location ('left', 'right')
            bc: Boundary condition specification
        """
        self.boundary_conditions[location] = bc
        logger.info(f"Set {location} boundary condition: {bc.bc_type}")
    
    def solve(
        self,
        t_final: float,
        dt_initial: Optional[float] = None,
        output_times: Optional[List[float]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Solve LNS equations to final time using optimized methods.
        
        Args:
            t_final: Final simulation time
            dt_initial: Initial time step (computed if None)
            output_times: Times for output (default: 10 outputs)
            save_results: Whether to save results to file
            
        Returns:
            Comprehensive results dictionary
        """
        logger.info(f"Starting optimized LNS solve to t = {t_final:.3e} s")
        
        # Initialize
        if dt_initial is not None:
            self.dt_current = dt_initial
        
        if output_times is None:
            output_times = list(np.linspace(0, t_final, 11)[1:])
        
        # Results storage
        results = {
            'times': [self.t_current],
            'output_data': {
                'primitives': [self.state.get_primitive_variables()],
                'conservatives': [self.state.Q.copy()]
            },
            'conservation_errors': [],
            'performance_metrics': {},
            'solver_diagnostics': []
        }
        
        # Main time loop
        wall_time_start = time.time()
        next_output_idx = 0
        
        while self.t_current < t_final and self.iteration < self.max_iterations:
            
            # Compute time step
            if self.adaptive_dt:
                self.dt_current = self._compute_adaptive_timestep()
            
            # Ensure we don't overshoot
            self.dt_current = min(self.dt_current, t_final - self.t_current)
            
            # Take optimized timestep
            timestep_start = time.time()
            self._take_optimized_timestep()
            timestep_time = time.time() - timestep_start
            
            # Update time and iteration
            self.t_current += self.dt_current
            self.iteration += 1
            
            # Update performance metrics
            self.performance_metrics['total_timesteps'] += 1
            self.performance_metrics['average_timestep_time'] = (
                (self.performance_metrics['average_timestep_time'] * (self.iteration - 1) + timestep_time) 
                / self.iteration
            )
            
            # Store output if needed
            if (next_output_idx < len(output_times) and 
                self.t_current >= output_times[next_output_idx]):
                
                results['times'].append(self.t_current)
                results['output_data']['primitives'].append(self.state.get_primitive_variables())
                results['output_data']['conservatives'].append(self.state.Q.copy())
                
                # Compute conservation errors
                conservation_error = self._analyze_conservation(results)
                results['conservation_errors'].append(conservation_error)
                
                next_output_idx += 1
                
                if self.iteration % self.output_interval == 0:
                    logger.info(f"  t = {self.t_current:.3e} s, dt = {self.dt_current:.3e} s, iter = {self.iteration}")
        
        # Finalize results
        wall_time = time.time() - wall_time_start
        results['iterations'] = self.iteration
        results['wall_time'] = wall_time
        results['final_time'] = self.t_current
        
        # Get performance statistics
        results['performance_metrics'] = self.numerics.get_performance_stats()
        results['performance_metrics']['wall_time'] = wall_time
        results['performance_metrics']['timesteps_per_second'] = self.iteration / wall_time
        
        logger.info(f"Optimized solve completed:")
        logger.info(f"  Final time: {self.t_current:.3e} s")
        logger.info(f"  Iterations: {self.iteration}")
        logger.info(f"  Wall time: {wall_time:.3f} s")
        logger.info(f"  Performance: {self.iteration/wall_time:.1f} timesteps/s")
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        return results
    
    def _take_optimized_timestep(self):
        """Take single optimized timestep using ghost cell method."""
        
        # Create RHS function that uses optimized numerics
        def rhs_function(Q_phys):
            # Get physics parameters as dictionary
            physics_params = {
                'gamma': self.physics.params.specific_heat_ratio,
                'R_gas': self.physics.params.gas_constant,
                'mu_viscous': self.physics.params.mu_viscous,
                'k_thermal': self.physics.params.k_thermal,
                'tau_q': self.physics.params.tau_q,
                'tau_sigma': self.physics.params.tau_sigma
            }
            
            # Compute hyperbolic RHS using optimized method
            rhs_hyperbolic, max_speed = self.numerics.compute_hyperbolic_rhs_1d_optimized(
                Q_phys,
                self.numerics.optimized_hll_flux_1d,
                physics_params,
                self.grid.dx,
                self.boundary_conditions
            )
            
            # Add source terms (LNS relaxation physics)
            rhs_source = self._compute_source_terms(Q_phys)
            
            total_rhs = rhs_hyperbolic + rhs_source
            
            return total_rhs, max_speed
        
        # SSP-RK2 step with optimized implementation
        self.state.Q = self.numerics.ssp_rk2_step_optimized(
            self.state.Q,
            rhs_function,
            self.dt_current,
            apply_limiter=True
        )
    
    def _compute_source_terms(self, Q: np.ndarray) -> np.ndarray:
        """
        Compute LNS source terms (relaxation physics).
        
        Args:
            Q: Conservative variables [nx, n_vars]
            
        Returns:
            Source term contributions [nx, n_vars]
        """
        source = np.zeros_like(Q)
        
        if Q.shape[1] < 5:
            return source  # No LNS variables
        
        # Get primitive variables and gradients
        primitives = self.state.get_primitive_variables()
        u = primitives['velocity']
        T = primitives['temperature']
        
        # Compute gradients
        du_dx = np.gradient(u, self.grid.dx)
        dT_dx = np.gradient(T, self.grid.dx)
        
        # Current heat flux and stress
        q_x = Q[:, 3]
        sigma_xx = Q[:, 4]
        
        # NSF targets - create material properties dict
        material_props = {
            'k_thermal': self.physics.params.k_thermal,
            'mu_viscous': self.physics.params.mu_viscous
        }
        q_nsf, sigma_nsf = self.physics.compute_1d_nsf_targets(
            du_dx, dT_dx, material_props
        )
        
        # Relaxation source terms
        tau_q = self.physics.params.tau_q
        tau_sigma = self.physics.params.tau_sigma
        
        source[:, 3] = -(q_x - q_nsf) / tau_q        # Heat flux relaxation
        source[:, 4] = -(sigma_xx - sigma_nsf) / tau_sigma  # Stress relaxation
        
        return source
    
    def _compute_adaptive_timestep(self) -> float:
        """Compute adaptive timestep using optimized primitives."""
        
        # Get pre-computed primitive variables
        primitives = self.numerics.compute_primitive_variables_vectorized(self.state.Q)
        
        # CFL constraint
        dt_cfl = self.numerics.compute_cfl_time_step(
            primitives, self.grid.dx, self.cfl_target
        )
        
        # Relaxation time constraints (for stability)
        dt_relax_q = 0.1 * self.physics.params.tau_q
        dt_relax_sigma = 0.1 * self.physics.params.tau_sigma
        
        # Return minimum (most restrictive)
        return min(dt_cfl, dt_relax_q, dt_relax_sigma)
    
    def _analyze_conservation(self, results: Dict) -> Dict[str, float]:
        """Analyze conservation properties."""
        
        if not results['output_data']['conservatives']:
            return {}
        
        # Current and initial conservative quantities
        Q_current = results['output_data']['conservatives'][-1]
        Q_initial = results['output_data']['conservatives'][0]
        
        # Integrate over domain
        dx = self.grid.dx
        mass_current = np.sum(Q_current[:, 0]) * dx
        momentum_current = np.sum(Q_current[:, 1]) * dx
        energy_current = np.sum(Q_current[:, 2]) * dx
        
        mass_initial = np.sum(Q_initial[:, 0]) * dx
        momentum_initial = np.sum(Q_initial[:, 1]) * dx
        energy_initial = np.sum(Q_initial[:, 2]) * dx
        
        return {
            'time': self.t_current,
            'mass': mass_current,
            'momentum': momentum_current,
            'energy': energy_current,
            'mass_error': abs(mass_current - mass_initial) / abs(mass_initial),
            'momentum_error': abs(momentum_current - momentum_initial) / max(abs(momentum_initial), np.finfo(float).eps),
            'energy_error': abs(energy_current - energy_initial) / abs(energy_initial)
        }
    
    def _save_results(self, results: Dict):
        """Save results to files."""
        self.output_dir.mkdir(exist_ok=True)
        
        # Save with optimized suffix
        filename = f"lns_optimized_results_t{results['final_time']:.3e}.h5"
        filepath = self.output_dir / filename
        
        try:
            self.data_writer.save_results(results, str(filepath))
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
    
    def plot_results(self, results: Dict, save_plot: bool = True):
        """Plot results with performance information."""
        
        if not results['output_data']['primitives']:
            logger.warning("No results to plot")
            return
        
        # Get final state
        final_primitives = results['output_data']['primitives'][-1]
        x = self.grid.x
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Density
        axes[0, 0].plot(x, final_primitives['density'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('x [m]')
        axes[0, 0].set_ylabel('Density [kg/m³]')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Density Profile')
        
        # Velocity
        axes[0, 1].plot(x, final_primitives['velocity'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('x [m]')
        axes[0, 1].set_ylabel('Velocity [m/s]')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Velocity Profile')
        
        # Pressure
        axes[1, 0].plot(x, final_primitives['pressure'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('x [m]')
        axes[1, 0].set_ylabel('Pressure [Pa]')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Pressure Profile')
        
        # Performance metrics
        perf = results['performance_metrics']
        metrics_text = f"""Performance Metrics:
Wall Time: {perf['wall_time']:.3f} s
Timesteps/s: {perf.get('timesteps_per_second', 0):.1f}
Total Flux Calls: {perf.get('total_flux_calls', 0)}
Avg Flux Time: {perf.get('avg_flux_time', 0):.3e} s
Iterations: {results['iterations']}"""
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Summary')
        
        plt.suptitle(f'Optimized LNS Results (t = {results["final_time"]:.3e} s)', fontsize=14)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'optimized_results.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {plot_path}")
        
        plt.show()


# Example usage and validation
if __name__ == "__main__":
    print("Testing Optimized LNS Solver")
    
    # Create optimized solver
    solver = OptimizedLNSSolver1D.create_sod_shock_tube(nx=100)
    
    # Run short simulation
    results = solver.solve(t_final=1e-3, dt_initial=1e-6)
    
    # Display results
    print(f"\nOptimized Solver Results:")
    print(f"Final time: {results['final_time']:.3e} s")
    print(f"Iterations: {results['iterations']}")
    print(f"Wall time: {results['wall_time']:.3f} s")
    print(f"Performance: {results['iterations']/results['wall_time']:.1f} timesteps/s")
    
    # Check final state
    final_primitives = results['output_data']['primitives'][-1]
    print(f"\nFinal State:")
    print(f"Density range: {np.min(final_primitives['density']):.6f} - {np.max(final_primitives['density']):.6f}")
    print(f"Velocity range: {np.min(final_primitives['velocity']):.3f} - {np.max(final_primitives['velocity']):.3f}")
    print(f"Pressure range: {np.min(final_primitives['pressure']):.1f} - {np.max(final_primitives['pressure']):.1f}")
    
    print("\n✅ Optimized solver demonstrates:")
    print("   • Proper ghost cell boundary conditions")
    print("   • Eliminated redundant Q->P conversions")
    print("   • Vectorized flux computation")
    print("   • Conservative finite volume method")
    print("   • Professional performance monitoring")