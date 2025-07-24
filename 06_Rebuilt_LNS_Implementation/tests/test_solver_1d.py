"""
Unit tests for LNSSolver1D class.

Tests the complete 1D LNS solver functionality including:
- Initialization and setup
- Time integration
- Boundary conditions
- Conservation properties
- Standard test cases (Sod shock tube, heat conduction)
"""

import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import tempfile

from lns_solver.solvers.solver_1d import LNSSolver1D
from lns_solver.core.grid import LNSGrid
from lns_solver.core.physics import LNSPhysics, LNSPhysicsParameters
from lns_solver.core.numerics import LNSNumerics
from lns_solver.core.state import LNSState


class TestLNSSolver1D:
    """Test cases for LNSSolver1D class."""
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        # Create components
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        physics = LNSPhysics()
        numerics = LNSNumerics()
        
        # Create solver
        solver = LNSSolver1D(grid, physics, numerics)
        
        assert solver.grid == grid
        assert solver.physics == physics
        assert solver.numerics == numerics
        assert solver.state.n_variables == 5
        assert solver.t_current == 0.0
        assert solver.iteration == 0
        
    def test_invalid_grid_dimension(self):
        """Test that 2D grid raises error."""
        grid = LNSGrid.create_uniform_2d(nx=5, ny=5, x_bounds=(0, 1), y_bounds=(0, 1))
        physics = LNSPhysics()
        numerics = LNSNumerics()
        
        with pytest.raises(ValueError, match="LNSSolver1D requires 1D grid"):
            LNSSolver1D(grid, physics, numerics)
    
    def test_create_sod_shock_tube(self):
        """Test Sod shock tube creation."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=50)
        
        assert solver.grid.nx == 50
        assert solver.grid.ndim == 1
        
        # Check initial conditions
        primitives = solver.state.get_primitive_variables()
        
        # Left state (x < 0.5)
        left_indices = solver.grid.x < 0.5
        npt.assert_array_almost_equal(primitives['density'][left_indices], 1.0)
        npt.assert_array_almost_equal(primitives['velocity'][left_indices], 0.0)
        
        # Right state (x >= 0.5)  
        right_indices = solver.grid.x >= 0.5
        npt.assert_array_almost_equal(primitives['density'][right_indices], 0.125)
        npt.assert_array_almost_equal(primitives['velocity'][right_indices], 0.0)
        
        # Check boundary conditions
        bc_left = solver.grid.get_boundary_condition('left')
        bc_right = solver.grid.get_boundary_condition('right')
        assert bc_left.bc_type == 'outflow'
        assert bc_right.bc_type == 'outflow'
    
    def test_create_heat_conduction_test(self):
        """Test heat conduction test creation."""
        T_left, T_right = 400.0, 300.0
        solver = LNSSolver1D.create_heat_conduction_test(
            nx=20, T_left=T_left, T_right=T_right
        )
        
        assert solver.grid.nx == 20
        
        # Check temperature profile
        primitives = solver.state.get_primitive_variables()
        T_profile = primitives['temperature']
        
        # Should be approximately linear
        assert T_profile[0] >= T_right  # Left side hotter
        assert T_profile[-1] <= T_left  # Right side cooler
        
        # Check boundary conditions
        bc_left = solver.grid.get_boundary_condition('left') 
        bc_right = solver.grid.get_boundary_condition('right')
        assert bc_left.bc_type == 'dirichlet'
        assert bc_right.bc_type == 'dirichlet'
        assert bc_left.values == T_left
        assert bc_right.values == T_right
    
    def test_adaptive_timestep_computation(self):
        """Test adaptive time step computation."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=50)
        
        dt = solver._compute_adaptive_timestep()
        
        # Should be positive and reasonable
        assert dt > 0
        assert dt < 1e-2  # Should be reasonably small for stability
        
        # Test with different CFL targets
        solver.cfl_target = 0.5
        dt_conservative = solver._compute_adaptive_timestep()
        
        solver.cfl_target = 0.9
        dt_aggressive = solver._compute_adaptive_timestep()
        
        assert dt_conservative < dt_aggressive
    
    def test_boundary_condition_application(self):
        """Test boundary condition application."""
        solver = LNSSolver1D.create_heat_conduction_test(nx=10, T_left=400.0, T_right=300.0)
        
        # Store original values
        original_Q = solver.state.Q.copy()
        
        # Apply boundary conditions
        solver._apply_boundary_conditions()
        
        # Check that boundary temperatures are enforced
        primitives = solver.state.get_primitive_variables()
        
        # Left boundary should be close to 400K (within tolerance)
        assert abs(primitives['temperature'][0] - 400.0) < 5.0
        
        # Right boundary should be close to 300K
        assert abs(primitives['temperature'][-1] - 300.0) < 5.0
    
    def test_source_term_computation(self):
        """Test source term computation."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=10)
        
        # Set non-zero heat flux and stress to test relaxation
        solver.state.Q[:, 3] = 100.0  # Heat flux
        solver.state.Q[:, 4] = 50.0   # Stress
        
        source_terms = solver._compute_source_terms()
        
        # Source terms should have correct shape
        assert source_terms.shape == (10, 5)
        
        # Only heat flux and stress should have non-zero source terms
        npt.assert_array_almost_equal(source_terms[:, 0], 0.0)  # Density
        npt.assert_array_almost_equal(source_terms[:, 1], 0.0)  # Momentum
        npt.assert_array_almost_equal(source_terms[:, 2], 0.0)  # Energy
        
        # Heat flux and stress should have relaxation source terms
        assert np.any(source_terms[:, 3] != 0.0)  # Heat flux source
        assert np.any(source_terms[:, 4] != 0.0)  # Stress source
    
    def test_single_timestep(self):
        """Test taking a single time step."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=20)
        
        # Store initial state
        Q_initial = solver.state.Q.copy()
        t_initial = solver.t_current
        iteration_initial = solver.iteration
        
        # Take a time step
        solver.dt_current = 1e-6
        solver._take_timestep()
        
        # Check that state changed
        assert not np.allclose(solver.state.Q, Q_initial)
        
        # Check that diagnostics are updated
        assert len(solver.dt_history) == 1
        assert solver.dt_history[0] == 1e-6
    
    def test_short_simulation(self):
        """Test short simulation run."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=50)
        
        # Run short simulation
        results = solver.solve(t_final=1e-4, dt_initial=1e-6, save_results=True)
        
        # Check results structure
        assert 'time_final' in results
        assert 'iterations' in results
        assert 'wall_time' in results
        assert 'conservation_errors' in results
        assert 'output_data' in results
        
        # Check simulation progressed
        assert results['time_final'] > 0
        assert results['iterations'] > 0
        assert results['wall_time'] > 0
        
        # Check that we have output data
        output_data = results['output_data']
        assert len(output_data['times']) > 1
        assert len(output_data['states']) > 1
        assert len(output_data['primitives']) > 1
    
    def test_conservation_checking(self):
        """Test conservation checking."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=30)
        
        # Run simulation with conservation checking
        solver.output_interval = 5  # Check conservation frequently
        results = solver.solve(t_final=1e-5, dt_initial=1e-7)
        
        # Check that conservation data was collected
        conservation_errors = results['conservation_errors']
        assert len(conservation_errors) > 0
        
        # Each entry should have required fields
        for entry in conservation_errors:
            assert 'time' in entry
            assert 'mass' in entry
            assert 'momentum' in entry
            assert 'energy' in entry
        
        # Analyze conservation
        conservation_analysis = solver.analyze_conservation(results)
        
        assert 'mass_conservation' in conservation_analysis
        assert 'momentum_conservation' in conservation_analysis
        assert 'energy_conservation' in conservation_analysis
        
        # Conservation errors should be small for short simulation
        mass_error = conservation_analysis['mass_conservation']['max_error']
        momentum_error = conservation_analysis['momentum_conservation']['max_error']
        energy_error = conservation_analysis['energy_conservation']['max_error']
        
        # Mass and energy should be well conserved
        assert mass_error < 1e-10, f"Mass conservation error too large: {mass_error}"
        assert energy_error < 1e-10, f"Energy conservation error too large: {energy_error}"
        
        # Momentum conservation can be violated in shock problems (momentum is created from pressure forces)
        # Just check that it's finite and not growing wildly
        assert np.isfinite(momentum_error), "Momentum error should be finite"
        assert momentum_error < 10.0, f"Momentum error too large: {momentum_error}"
    
    def test_performance_metrics(self):
        """Test performance metrics computation."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=20)
        
        results = solver.solve(t_final=1e-5, dt_initial=1e-7)
        
        metrics = results['performance_metrics']
        
        # Check required metrics
        assert 'wall_time' in metrics
        assert 'iterations' in metrics  
        assert 'cell_updates' in metrics
        assert 'updates_per_second' in metrics
        assert 'time_per_iteration' in metrics
        
        # Values should be reasonable
        assert metrics['wall_time'] > 0
        assert metrics['iterations'] > 0
        assert metrics['cell_updates'] > 0
        assert metrics['updates_per_second'] > 0
        assert metrics['time_per_iteration'] > 0
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        # Create solver and run briefly
        solver1 = LNSSolver1D.create_sod_shock_tube(nx=20)
        solver1.solve(t_final=1e-5, dt_initial=1e-7)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.h5"
            solver1.save_checkpoint(checkpoint_path)
            
            # Load checkpoint
            solver2 = LNSSolver1D.load_checkpoint(checkpoint_path)
            
            # Check that states match
            npt.assert_array_almost_equal(solver1.state.Q, solver2.state.Q)
            assert abs(solver1.t_current - solver2.t_current) < 1e-10
            assert solver1.iteration == solver2.iteration
            
            # Check that solver can continue
            solver2.solve(t_final=solver2.t_current + 1e-5, dt_initial=1e-7)
            assert solver2.t_current > solver1.t_current
    
    def test_heat_conduction_physics_validation(self):
        """Test heat conduction physics validation."""
        # Create heat conduction test with known parameters
        solver = LNSSolver1D.create_heat_conduction_test(
            nx=50, T_left=350.0, T_right=300.0
        )
        
        # Run for some time to allow heat diffusion
        results = solver.solve(t_final=1e-3, dt_initial=1e-6)
        
        # Get final temperature profile
        final_primitives = results['output_data']['primitives'][-1]
        T_final = final_primitives['temperature']
        
        # Temperature should be smoothing out (less gradient than initial)
        # Initial profile was linear, final should be approaching equilibrium
        initial_gradient = abs(350.0 - 300.0) / solver.grid.dx
        final_gradient = abs(np.max(np.gradient(T_final, solver.grid.dx)))
        
        # Final gradient should be smaller (heat diffusion smooths temperature)
        assert final_gradient < initial_gradient
        
        # Temperatures should be bounded by initial values
        assert np.all(T_final >= 300.0 - 1.0)  # Allow small numerical error
        assert np.all(T_final <= 350.0 + 1.0)
    
    def test_solver_stability(self):
        """Test solver stability over longer runs."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=100)
        
        # Run for moderate time
        results = solver.solve(t_final=1e-3, dt_initial=1e-6)
        
        # Check that simulation completed
        assert results['time_final'] >= 1e-3 - 1e-6  # Should reach target time
        
        # Check that final state is physical
        final_primitives = results['output_data']['primitives'][-1]
        
        # Density should be positive
        assert np.all(final_primitives['density'] > 0)
        
        # Pressure should be positive
        assert np.all(final_primitives['pressure'] > 0)
        
        # Temperature should be positive
        assert np.all(final_primitives['temperature'] > 0)
        
        # No NaN or inf values
        for var_name, var_data in final_primitives.items():
            assert np.all(np.isfinite(var_data)), f"Non-finite values in {var_name}"
    
    def test_plot_functionality(self):
        """Test plotting functionality (without displaying)."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=30)
        results = solver.solve(t_final=1e-4, dt_initial=1e-6)
        
        # Test plotting without errors (we can't test visual output)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            # This should not raise errors
            solver.plot_results(results, save_path=None)
            
            # Test current state plotting
            solver.plot_results(save_path=None)
            
        except ImportError:
            # Skip test if matplotlib not available
            pytest.skip("Matplotlib not available for plotting test")
    
    def test_string_representations(self):
        """Test string representations."""
        solver = LNSSolver1D.create_sod_shock_tube(nx=25)
        
        # Test __repr__
        repr_str = repr(solver)
        assert "LNSSolver1D" in repr_str
        assert "nx=25" in repr_str
        assert "t=" in repr_str
        assert "iter=" in repr_str


# Integration tests
class TestLNSSolver1DIntegration:
    """Integration tests for complete solver workflows."""
    
    def test_sod_shock_tube_complete_workflow(self):
        """Test complete Sod shock tube workflow."""
        # Create solver
        solver = LNSSolver1D.create_sod_shock_tube(nx=100)
        
        # Run simulation
        results = solver.solve(t_final=0.2, dt_initial=1e-5)
        
        # Check that we captured shock physics
        final_primitives = results['output_data']['primitives'][-1]
        
        # Should have density variations (shock, contact, rarefaction)
        density_range = np.ptp(final_primitives['density'])
        assert density_range > 0.1, "Should have significant density variations"
        
        # Should have velocity variations
        velocity_range = np.ptp(final_primitives['velocity'])
        assert velocity_range > 0.1, "Should have significant velocity variations"
        
        # Should have pressure variations  
        pressure_range = np.ptp(final_primitives['pressure'])
        assert pressure_range > 1000, "Should have significant pressure variations"
    
    def test_heat_conduction_complete_workflow(self):
        """Test complete heat conduction workflow."""
        # Create solver
        solver = LNSSolver1D.create_heat_conduction_test(
            nx=50, T_left=400.0, T_right=300.0
        )
        
        # Capture initial profile
        initial_primitives = solver.state.get_primitive_variables()
        T_initial = initial_primitives['temperature']
        
        # Run simulation
        results = solver.solve(t_final=0.01, dt_initial=1e-5)
        
        # Get final profile
        final_primitives = results['output_data']['primitives'][-1]
        T_final = final_primitives['temperature']
        
        # Heat should have diffused (profile should be smoother)
        initial_std = np.std(T_initial)
        final_std = np.std(T_final)
        
        # Standard deviation should decrease (profile flattening)
        assert final_std < initial_std, "Temperature profile should smooth out"
        
        # Average temperature should be conserved (approximately)
        T_avg_initial = np.mean(T_initial)
        T_avg_final = np.mean(T_final)
        assert abs(T_avg_final - T_avg_initial) < 5.0, "Average temperature should be conserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])