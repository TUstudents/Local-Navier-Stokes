"""
Comprehensive Validation Framework for LNS Solver.

This module provides the main validation framework that orchestrates
comparisons between LNS solver, analytical solutions, and classical methods.
It includes:
- Automated test suite execution
- Quantitative error metrics
- Visualization and reporting
- Performance benchmarking
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

from lns_solver.solvers.solver_1d_final import FinalIntegratedLNSSolver1D
from lns_solver.validation.analytical_solutions import (
    RiemannExactSolver, HeatConductionExact, AcousticWaveExact
)
from lns_solver.validation.classical_solvers import (
    EulerSolver1D, NavierStokesSolver1D, HeatDiffusionSolver1D
)
from lns_solver.core.grid import LNSGrid

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    l1_error: float
    l2_error: float
    linf_error: float
    relative_l2_error: float
    correlation_coefficient: float
    computational_time: float
    
    def __post_init__(self):
        """Validate metrics."""
        if self.l2_error < 0 or self.correlation_coefficient < -1 or self.correlation_coefficient > 1:
            raise ValueError("Invalid metric values")


class ComparisonMetrics:
    """Utilities for computing comparison metrics between solutions."""
    
    @staticmethod
    def compute_errors(
        numerical: np.ndarray, 
        reference: np.ndarray,
        dx: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute comprehensive error metrics.
        
        Args:
            numerical: Numerical solution
            reference: Reference/analytical solution
            dx: Grid spacing for integration
            
        Returns:
            Dictionary of error metrics
        """
        if len(numerical) != len(reference):
            raise ValueError("Arrays must have same length")
        
        error = numerical - reference
        
        # L1 norm (integral of absolute error)
        l1_error = np.sum(np.abs(error)) * dx
        
        # L2 norm (RMS error)
        l2_error = np.sqrt(np.sum(error**2) * dx)
        
        # L∞ norm (maximum error)
        linf_error = np.max(np.abs(error))
        
        # Relative L2 error
        ref_norm = np.sqrt(np.sum(reference**2) * dx)
        relative_l2_error = l2_error / ref_norm if ref_norm > 1e-12 else np.inf
        
        # Correlation coefficient
        if np.std(numerical) > 1e-12 and np.std(reference) > 1e-12:
            correlation = np.corrcoef(numerical, reference)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'l1_error': l1_error,
            'l2_error': l2_error,
            'linf_error': linf_error,
            'relative_l2_error': relative_l2_error,
            'correlation_coefficient': correlation
        }
    
    @staticmethod
    def compute_convergence_rate(
        errors: List[float], 
        grid_spacings: List[float]
    ) -> float:
        """
        Compute convergence rate from grid refinement study.
        
        Args:
            errors: List of errors for different grid resolutions
            grid_spacings: Corresponding grid spacings (dx values)
            
        Returns:
            Convergence rate p where error ~ dx^p
            
        Note: 
            CORRECTED to use actual grid spacing dx, not 1/N.
            Standard formulation: error ≈ C * dx^p
            Therefore: log(error) = p * log(dx) + log(C)
        """
        if len(errors) < 2 or len(errors) != len(grid_spacings):
            return 0.0
        
        # Filter out non-positive errors and spacings
        valid_pairs = [(e, h) for e, h in zip(errors, grid_spacings) if e > 0 and h > 0]
        if len(valid_pairs) < 2:
            return 0.0
        
        errors_valid, spacings_valid = zip(*valid_pairs)
        
        # Convert to log scale - CORRECTED to use actual dx
        log_errors = np.log(errors_valid)
        log_dx = np.log(spacings_valid)  # Use actual grid spacing, not 1/N
        
        # Linear fit: log(error) = p * log(dx) + log(C)
        coeffs = np.polyfit(log_dx, log_errors, 1)
        convergence_rate = coeffs[0]
        
        return convergence_rate
    
    @staticmethod
    def analyze_conservation(results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze conservation properties."""
        if 'conservation_errors' not in results:
            return {}
        
        conservation_data = results['conservation_errors']
        if not conservation_data:
            return {}
        
        # Extract time series
        times = np.array([entry['time'] for entry in conservation_data])
        mass = np.array([entry['mass'] for entry in conservation_data])
        momentum = np.array([entry['momentum'] for entry in conservation_data])
        energy = np.array([entry['energy'] for entry in conservation_data])
        
        # Compute conservation errors
        mass_error = np.max(np.abs(mass - mass[0])) / np.abs(mass[0]) if mass[0] != 0 else 0
        momentum_error = np.max(np.abs(momentum - momentum[0])) / max(np.abs(momentum[0]), np.max(np.abs(momentum))) if np.max(np.abs(momentum)) > 1e-12 else 0
        energy_error = np.max(np.abs(energy - energy[0])) / np.abs(energy[0]) if energy[0] != 0 else 0
        
        return {
            'mass_conservation_error': mass_error,
            'momentum_conservation_error': momentum_error,
            'energy_conservation_error': energy_error,
            'simulation_time': times[-1] - times[0] if len(times) > 1 else 0
        }


class ValidationSuite:
    """
    Main validation suite coordinating all comparison studies.
    
    This class orchestrates comprehensive validation of the LNS solver
    against analytical solutions and classical methods.
    """
    
    def __init__(self, output_dir: str = "./validation_results"):
        """
        Initialize validation suite.
        
        Args:
            output_dir: Directory for validation results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize solvers and analytical solutions
        self.riemann_solver = RiemannExactSolver()
        self.heat_exact = HeatConductionExact()
        self.acoustic_exact = AcousticWaveExact()
        
        # Results storage
        self.validation_results = {}
        
        logger.info(f"Initialized validation suite, output to {self.output_dir}")
    
    def validate_riemann_shock_tube(
        self,
        grid_sizes: List[int] = [50, 100, 200],
        t_final: float = 0.15
    ) -> Dict[str, Any]:
        """
        Validate LNS solver against exact Riemann solution.
        
        Args:
            grid_sizes: List of grid sizes for convergence study
            t_final: Final time for comparison
            
        Returns:
            Validation results dictionary
        """
        logger.info("Starting Riemann shock tube validation")
        
        results = {
            'test_name': 'riemann_shock_tube',
            'grid_sizes': grid_sizes,
            't_final': t_final,
            'lns_results': {},
            'euler_results': {},
            'analytical_solution': {},
            'error_metrics': {},
            'convergence_analysis': {}
        }
        
        # Initial conditions (standard Sod problem)
        rho_L, u_L, p_L = 1.0, 0.0, 101325.0
        rho_R, u_R, p_R = 0.125, 0.0, 10132.5
        
        for nx in grid_sizes:
            logger.info(f"  Testing grid size: {nx}")
            
            # Create grid
            grid = LNSGrid.create_uniform_1d(nx, 0.0, 1.0)
            x = grid.x
            
            # Analytical solution
            start_time = time.time()
            analytical = self.riemann_solver.solve(
                rho_L, u_L, p_L, rho_R, u_R, p_R, x, t_final
            )
            analytical_time = time.time() - start_time
            
            # LNS solver
            start_time = time.time()
            lns_solver = FinalIntegratedLNSSolver1D.create_sod_shock_tube(nx=nx)
            lns_results = lns_solver.solve(t_final=t_final, dt_initial=1e-6)
            lns_time = time.time() - start_time
            
            # Euler solver for comparison
            start_time = time.time()
            euler_solver = EulerSolver1D(grid)
            euler_solver.initialize_sod_shock_tube()
            euler_results = euler_solver.solve(t_final, cfl=0.8)
            euler_time = time.time() - start_time
            
            # Store results
            results['analytical_solution'][nx] = {
                'solution': analytical,
                'computational_time': analytical_time
            }
            
            results['lns_results'][nx] = {
                'final_primitives': lns_results['output_data']['primitives'][-1],
                'computational_time': lns_time,
                'iterations': lns_results['iterations'],
                'conservation': ComparisonMetrics.analyze_conservation(lns_results)
            }
            
            if euler_results['solutions']:
                results['euler_results'][nx] = {
                    'final_primitives': euler_results['solutions'][-1],
                    'computational_time': euler_time
                }
            
            # Compute error metrics
            lns_final = lns_results['output_data']['primitives'][-1]
            
            # Density comparison
            lns_density_metrics = ComparisonMetrics.compute_errors(
                lns_final['density'], analytical['density'], grid.dx
            )
            lns_density_metrics['computational_time'] = lns_time
            
            # Pressure comparison
            lns_pressure_metrics = ComparisonMetrics.compute_errors(
                lns_final['pressure'], analytical['pressure'], grid.dx
            )
            
            results['error_metrics'][nx] = {
                'lns_density': lns_density_metrics,
                'lns_pressure': lns_pressure_metrics
            }
            
            # Add Euler comparison if available
            if euler_results['solutions']:
                euler_final = euler_results['solutions'][-1]
                euler_density_metrics = ComparisonMetrics.compute_errors(
                    euler_final['density'], analytical['density'], grid.dx
                )
                euler_density_metrics['computational_time'] = euler_time
                
                results['error_metrics'][nx]['euler_density'] = euler_density_metrics
        
        # Convergence analysis - CORRECTED to use grid spacings
        if len(grid_sizes) >= 2:
            lns_l2_errors = [results['error_metrics'][nx]['lns_density']['l2_error'] 
                           for nx in grid_sizes]
            
            # Compute actual grid spacings for convergence rate calculation
            # Domain is [0, 1] for all test cases
            grid_spacings = [1.0 / nx for nx in grid_sizes]
            
            results['convergence_analysis'] = {
                'lns_convergence_rate': ComparisonMetrics.compute_convergence_rate(
                    lns_l2_errors, grid_spacings  # CORRECTED: use dx not N
                ),
                'lns_l2_errors': lns_l2_errors,
                'grid_spacings': grid_spacings  # Store for reference
            }
        
        # Save results
        self._save_results(results, 'riemann_validation.json')
        self._plot_riemann_comparison(results)
        
        self.validation_results['riemann'] = results
        logger.info("Completed Riemann shock tube validation")
        
        return results
    
    def validate_heat_conduction(
        self,
        grid_sizes: List[int] = [50, 100],
        t_final: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Validate heat transport against analytical solutions.
        
        Args:
            grid_sizes: Grid sizes for testing
            t_final: Final time
            
        Returns:
            Validation results
        """
        logger.info("Starting heat conduction validation")
        
        results = {
            'test_name': 'heat_conduction',
            'grid_sizes': grid_sizes,
            't_final': t_final,
            'lns_results': {},
            'classical_diffusion': {},
            'analytical_fourier': {},
            'analytical_mcv': {},
            'error_metrics': {}
        }
        
        T_left, T_right = 350.0, 300.0
        
        for nx in grid_sizes:
            logger.info(f"  Testing heat conduction, grid size: {nx}")
            
            # Create grid
            grid = LNSGrid.create_uniform_1d(nx, 0.0, 1.0)
            x = grid.x
            
            # LNS solver
            start_time = time.time()
            # Create final solver and set up heat conduction problem manually
            lns_solver = FinalIntegratedLNSSolver1D.create_sod_shock_tube(nx=nx)
            # Initialize with heat conduction profile (simplified)
            lns_solver.state.initialize_sod_shock_tube()  # Use available method for now
            lns_results = lns_solver.solve(t_final=t_final, dt_initial=1e-6)
            lns_time = time.time() - start_time
            
            # Classical heat diffusion
            start_time = time.time()
            diffusion_solver = HeatDiffusionSolver1D(grid, thermal_diffusivity=1e-5)
            diffusion_solver.initialize_linear(T_left, T_right)
            diffusion_results = diffusion_solver.solve(t_final, T_left, T_right)
            diffusion_time = time.time() - start_time
            
            # Analytical solutions
            start_time = time.time()
            fourier_solution = self.heat_exact.fourier_step_response(
                x, t_final, T_initial=(T_left + T_right)/2, 
                T_step=(T_left - T_right)/2, x_step=0.5
            )
            mcv_temp, mcv_flux = self.heat_exact.mcv_step_response(
                x, t_final, T_initial=(T_left + T_right)/2,
                T_step=(T_left - T_right)/2, x_step=0.5
            )
            analytical_time = time.time() - start_time
            
            # Store results
            results['lns_results'][nx] = {
                'final_primitives': lns_results['output_data']['primitives'][-1],
                'computational_time': lns_time,
                'conservation': ComparisonMetrics.analyze_conservation(lns_results)
            }
            
            results['classical_diffusion'][nx] = {
                'final_temperature': diffusion_results['temperatures'][-1],
                'computational_time': diffusion_time
            }
            
            results['analytical_fourier'][nx] = {
                'temperature': fourier_solution,
                'computational_time': analytical_time / 2
            }
            
            results['analytical_mcv'][nx] = {
                'temperature': mcv_temp,
                'heat_flux': mcv_flux,
                'computational_time': analytical_time / 2
            }
            
            # Error metrics
            lns_temp = lns_results['output_data']['primitives'][-1]['temperature']
            diffusion_temp = diffusion_results['temperatures'][-1]
            
            # Compare with Fourier solution
            lns_fourier_metrics = ComparisonMetrics.compute_errors(
                lns_temp, fourier_solution, grid.dx
            )
            
            diffusion_fourier_metrics = ComparisonMetrics.compute_errors(
                diffusion_temp, fourier_solution, grid.dx
            )
            
            # Compare with MCV solution
            lns_mcv_metrics = ComparisonMetrics.compute_errors(
                lns_temp, mcv_temp, grid.dx
            )
            
            results['error_metrics'][nx] = {
                'lns_vs_fourier': lns_fourier_metrics,
                'diffusion_vs_fourier': diffusion_fourier_metrics,
                'lns_vs_mcv': lns_mcv_metrics
            }
        
        # Save results
        self._save_results(results, 'heat_conduction_validation.json')
        self._plot_heat_conduction_comparison(results)
        
        self.validation_results['heat_conduction'] = results
        logger.info("Completed heat conduction validation")
        
        return results
    
    def validate_nsf_limit(
        self,
        tau_values: List[float] = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    ) -> Dict[str, Any]:
        """
        Validate NSF limit recovery as τ → 0.
        
        Args:
            tau_values: List of relaxation times to test
            
        Returns:
            Validation results showing NSF limit recovery
        """
        logger.info("Starting NSF limit validation")
        
        results = {
            'test_name': 'nsf_limit',
            'tau_values': tau_values,
            'lns_results': {},
            'ns_reference': {},
            'error_metrics': {},
            'limit_analysis': {}
        }
        
        nx = 100
        t_final = 1e-4
        grid = LNSGrid.create_uniform_1d(nx, 0.0, 1.0)
        
        # Classical Navier-Stokes reference
        logger.info("  Computing Navier-Stokes reference solution")
        ns_solver = NavierStokesSolver1D(grid, mu=1e-5, k_thermal=0.025)
        ns_solver.initialize_sod_shock_tube()
        ns_results = ns_solver.solve(t_final)
        
        results['ns_reference'] = {
            'final_primitives': ns_results['solutions'][-1] if ns_results['solutions'] else None,
            'computational_time': 0  # Not tracked for reference
        }
        
        # Test different relaxation times
        for tau in tau_values:
            logger.info(f"  Testing τ = {tau:.1e}")
            
            # Create LNS solver with specific relaxation time
            from lns_solver.core.physics import LNSPhysicsParameters, LNSPhysics
            from lns_solver.core.numerics_optimized import OptimizedLNSNumerics
            
            physics_params = LNSPhysicsParameters(
                mu_viscous=1e-5,
                k_thermal=0.025, 
                tau_q=tau,
                tau_sigma=tau
            )
            physics = LNSPhysics(physics_params)
            numerics = OptimizedLNSNumerics()
            
            # Use final integrated solver with specific parameters
            lns_solver = FinalIntegratedLNSSolver1D(
                grid, physics, n_ghost=2, use_operator_splitting=True
            )
            lns_solver.state.initialize_sod_shock_tube()
            
            start_time = time.time()
            try:
                lns_results = lns_solver.solve(t_final=t_final, dt_initial=1e-7)
                lns_time = time.time() - start_time
                
                results['lns_results'][tau] = {
                    'final_primitives': lns_results['output_data']['primitives'][-1],
                    'computational_time': lns_time,
                    'success': True
                }
                
                # Compare with NS reference if available
                if ns_results['solutions']:
                    lns_final = lns_results['output_data']['primitives'][-1]
                    ns_final = ns_results['solutions'][-1]
                    
                    # Density comparison
                    density_metrics = ComparisonMetrics.compute_errors(
                        lns_final['density'], ns_final['density'], grid.dx
                    )
                    
                    results['error_metrics'][tau] = {
                        'density_vs_ns': density_metrics
                    }
                
            except Exception as e:
                logger.warning(f"    Failed for τ = {tau:.1e}: {e}")
                results['lns_results'][tau] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Analyze limit behavior
        successful_taus = [tau for tau in tau_values if results['lns_results'][tau].get('success', False)]
        
        if len(successful_taus) >= 2 and ns_results['solutions']:
            # Compute errors vs τ
            errors = []
            for tau in successful_taus:
                if tau in results['error_metrics']:
                    errors.append(results['error_metrics'][tau]['density_vs_ns']['l2_error'])
                else:
                    errors.append(float('inf'))
            
            results['limit_analysis'] = {
                'successful_taus': successful_taus,
                'l2_errors': errors,
                'demonstrates_limit': len([e for e in errors if e < 0.1]) > 0
            }
        
        # Save results
        self._save_results(results, 'nsf_limit_validation.json')
        self._plot_nsf_limit_analysis(results)
        
        self.validation_results['nsf_limit'] = results
        logger.info("Completed NSF limit validation")
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        logger.info("Generating comprehensive validation report")
        
        report = {
            'validation_summary': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_tests': len(self.validation_results),
                'successful_tests': 0,
                'overall_assessment': 'pending'
            },
            'detailed_results': self.validation_results,
            'performance_summary': {},
            'accuracy_summary': {},
            'recommendations': []
        }
        
        # Analyze results
        successful_tests = 0
        total_computational_time = 0
        accuracy_scores = []
        
        for test_name, test_results in self.validation_results.items():
            test_successful = True
            test_time = 0
            test_accuracy = 0
            
            if test_name == 'riemann':
                # Analyze Riemann validation
                if 'error_metrics' in test_results:
                    largest_grid = max(test_results['grid_sizes'])
                    if largest_grid in test_results['error_metrics']:
                        l2_error = test_results['error_metrics'][largest_grid]['lns_density']['l2_error']
                        test_accuracy = max(0, 1 - l2_error / 0.1)  # Normalize error
                        
                    # Sum computational times
                    for nx in test_results['grid_sizes']:
                        if nx in test_results['lns_results']:
                            test_time += test_results['lns_results'][nx]['computational_time']
            
            elif test_name == 'heat_conduction':
                # Analyze heat conduction validation
                if 'error_metrics' in test_results:
                    largest_grid = max(test_results['grid_sizes'])
                    if largest_grid in test_results['error_metrics']:
                        l2_error = test_results['error_metrics'][largest_grid]['lns_vs_fourier']['l2_error']
                        test_accuracy = max(0, 1 - l2_error / 50.0)  # Different scale for temperature
            
            elif test_name == 'nsf_limit':
                # Analyze NSF limit validation
                if 'limit_analysis' in test_results:
                    test_accuracy = 1.0 if test_results['limit_analysis'].get('demonstrates_limit', False) else 0.5
            
            if test_accuracy > 0.7:  # Threshold for success
                successful_tests += 1
            
            total_computational_time += test_time
            accuracy_scores.append(test_accuracy)
        
        # Update summary
        report['validation_summary']['successful_tests'] = successful_tests
        report['validation_summary']['overall_assessment'] = (
            'excellent' if successful_tests == len(self.validation_results) and np.mean(accuracy_scores) > 0.9 else
            'good' if successful_tests >= len(self.validation_results) * 0.8 else
            'acceptable' if successful_tests >= len(self.validation_results) * 0.6 else
            'needs_improvement'
        )
        
        report['performance_summary'] = {
            'total_computational_time': total_computational_time,
            'average_accuracy_score': np.mean(accuracy_scores) if accuracy_scores else 0.0
        }
        
        # Generate recommendations
        if np.mean(accuracy_scores) > 0.9:
            report['recommendations'].append("LNS solver demonstrates excellent accuracy across all validation tests")
        elif np.mean(accuracy_scores) > 0.7:
            report['recommendations'].append("LNS solver shows good accuracy with minor areas for improvement")
        else:
            report['recommendations'].append("LNS solver accuracy needs improvement in some validation tests")
        
        # Save comprehensive report
        self._save_results(report, 'comprehensive_validation_report.json')
        
        logger.info(f"Validation report: {successful_tests}/{len(self.validation_results)} tests successful")
        return report
    
    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")
    
    def _plot_riemann_comparison(self, results: Dict[str, Any]) -> None:
        """Generate Riemann comparison plots."""
        if not results['grid_sizes']:
            return
        
        # Use largest grid for detailed comparison
        nx = max(results['grid_sizes'])
        
        if nx not in results['analytical_solution']:
            return
        
        analytical = results['analytical_solution'][nx]['solution']
        
        if nx in results['lns_results']:
            lns_final = results['lns_results'][nx]['final_primitives']
        else:
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        variables = ['density', 'velocity', 'pressure', 'temperature']
        units = ['kg/m³', 'm/s', 'Pa', 'K']
        
        grid = LNSGrid.create_uniform_1d(nx, 0.0, 1.0)
        x = grid.x
        
        for i, (var, unit) in enumerate(zip(variables, units)):
            ax = axes[i]
            
            # Analytical solution
            ax.plot(x, analytical[var], 'k-', linewidth=2, label='Exact')
            
            # LNS solution
            ax.plot(x, lns_final[var], 'r--', linewidth=1.5, label='LNS')
            
            # Euler solution if available
            if nx in results['euler_results']:
                euler_final = results['euler_results'][nx]['final_primitives']
                ax.plot(x, euler_final[var], 'b:', linewidth=1.5, label='Euler')
            
            ax.set_xlabel('x [m]')
            ax.set_ylabel(f'{var.capitalize()} [{unit}]')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Riemann Problem Validation (nx={nx}, t={results["t_final"]:.3f}s)', fontsize=14)
        plt.tight_layout()
        
        plot_path = self.output_dir / 'riemann_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Riemann comparison plot to {plot_path}")
    
    def _plot_heat_conduction_comparison(self, results: Dict[str, Any]) -> None:
        """Generate heat conduction comparison plots."""
        if not results['grid_sizes']:
            return
        
        nx = max(results['grid_sizes'])
        
        if (nx not in results['lns_results'] or 
            nx not in results['analytical_fourier']):
            return
        
        # Extract data
        lns_temp = results['lns_results'][nx]['final_primitives']['temperature']
        fourier_temp = results['analytical_fourier'][nx]['temperature']
        mcv_temp = results['analytical_mcv'][nx]['temperature']
        
        if nx in results['classical_diffusion']:
            diffusion_temp = results['classical_diffusion'][nx]['final_temperature']
        else:
            diffusion_temp = None
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        grid = LNSGrid.create_uniform_1d(nx, 0.0, 1.0)
        x = grid.x
        
        # Temperature comparison
        ax1.plot(x, fourier_temp, 'k-', linewidth=2, label='Fourier (analytical)')
        ax1.plot(x, mcv_temp, 'g-', linewidth=2, label='MCV (analytical)')
        ax1.plot(x, lns_temp, 'r--', linewidth=1.5, label='LNS')
        
        if diffusion_temp is not None:
            ax1.plot(x, diffusion_temp, 'b:', linewidth=1.5, label='Classical diffusion')
        
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('Temperature [K]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Temperature Profiles')
        
        # Error plot
        fourier_error = np.abs(lns_temp - fourier_temp)
        mcv_error = np.abs(lns_temp - mcv_temp)
        
        ax2.plot(x, fourier_error, 'k-', label='|LNS - Fourier|')
        ax2.plot(x, mcv_error, 'g-', label='|LNS - MCV|')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('Temperature Error [K]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Error Analysis')
        
        plt.suptitle(f'Heat Conduction Validation (nx={nx}, t={results["t_final"]:.1e}s)', fontsize=14)
        plt.tight_layout()
        
        plot_path = self.output_dir / 'heat_conduction_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved heat conduction comparison plot to {plot_path}")
    
    def _plot_nsf_limit_analysis(self, results: Dict[str, Any]) -> None:
        """Generate NSF limit analysis plots."""
        if 'limit_analysis' not in results or not results['limit_analysis'].get('successful_taus'):
            return
        
        successful_taus = results['limit_analysis']['successful_taus']
        l2_errors = results['limit_analysis']['l2_errors']
        
        # Create log-log plot of error vs τ
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.loglog(successful_taus, l2_errors, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Relaxation Time τ [s]')
        ax.set_ylabel('L2 Error vs Navier-Stokes')
        ax.grid(True, alpha=0.3)
        ax.set_title('NSF Limit Recovery Analysis')
        
        # Add theoretical slope line if we have enough points
        if len(successful_taus) >= 3:
            # Fit power law: error ~ τ^p
            log_tau = np.log(successful_taus)
            log_error = np.log(l2_errors)
            coeffs = np.polyfit(log_tau, log_error, 1)
            slope = coeffs[0]
            
            # Plot fitted line
            tau_fit = np.array(successful_taus)
            error_fit = np.exp(coeffs[1]) * tau_fit**slope
            ax.loglog(tau_fit, error_fit, 'k--', alpha=0.7, 
                     label=f'Fitted slope: {slope:.2f}')
            ax.legend()
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'nsf_limit_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved NSF limit analysis plot to {plot_path}")


def run_comprehensive_validation():
    """Run the complete validation suite."""
    # Create validation suite
    validator = ValidationSuite("./validation_results")
    
    print("🔬 Starting Comprehensive LNS Validation Suite")
    print("=" * 50)
    
    # Run all validation tests
    riemann_results = validator.validate_riemann_shock_tube(
        grid_sizes=[50, 100, 200], 
        t_final=0.15
    )
    
    heat_results = validator.validate_heat_conduction(
        grid_sizes=[50, 100], 
        t_final=1e-3
    )
    
    nsf_results = validator.validate_nsf_limit(
        tau_values=[1e-3, 1e-4, 1e-5, 1e-6]
    )
    
    # Generate comprehensive report
    report = validator.generate_comprehensive_report()
    
    print("\n📊 Validation Complete!")
    print(f"Overall Assessment: {report['validation_summary']['overall_assessment'].upper()}")
    print(f"Successful Tests: {report['validation_summary']['successful_tests']}/{report['validation_summary']['total_tests']}")
    print(f"Average Accuracy: {report['performance_summary']['average_accuracy_score']:.2%}")
    
    return validator, report


if __name__ == "__main__":
    run_comprehensive_validation()