"""
Unit tests for modern LNS infrastructure.

Tests the production components: LNSGrid, EnhancedLNSState, LNSPhysics, OptimizedLNSNumerics.
"""

import pytest
import numpy as np
import numpy.testing as npt

from lns_solver.core.grid import LNSGrid
from lns_solver.core.state_enhanced import EnhancedLNSState, StateConfiguration, LNSVariables
from lns_solver.core.physics import LNSPhysics, LNSPhysicsParameters
from lns_solver.core.numerics_optimized import OptimizedLNSNumerics
from lns_solver.core.boundary_conditions import GhostCellBoundaryHandler, BCType, create_outflow_bc
from lns_solver.solvers.solver_1d_final import FinalIntegratedLNSSolver1D


class TestLNSGrid:
    """Test cases for LNSGrid class."""
    
    def test_create_uniform_1d(self):
        """Test 1D uniform grid creation."""
        grid = LNSGrid.create_uniform_1d(nx=100, x_min=0.0, x_max=1.0)
        
        assert grid.ndim == 1
        assert grid.nx == 100
        assert len(grid.x) == 100
        assert grid.dx == 0.01
        npt.assert_almost_equal(grid.x[0], 0.005, decimal=6)  # Cell center
        npt.assert_almost_equal(grid.x[-1], 0.995, decimal=6)
        
    def test_create_uniform_2d(self):
        """Test 2D uniform grid creation."""
        grid = LNSGrid.create_uniform_2d(
            nx=50, ny=25, 
            x_bounds=(0.0, 2.0), 
            y_bounds=(-1.0, 1.0)
        )
        
        assert grid.ndim == 2
        assert grid.nx == 50
        assert grid.ny == 25
        assert grid.dx == 0.04
        assert grid.dy == 0.08
        
    def test_boundary_conditions(self):
        """Test boundary condition management."""
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        
        # Set boundary conditions
        grid.set_boundary_condition('left', 'dirichlet', values=300.0)
        grid.set_boundary_condition('right', 'outflow')
        
        # Check boundary conditions
        bc_left = grid.get_boundary_condition('left')
        bc_right = grid.get_boundary_condition('right')
        
        assert bc_left.bc_type == 'dirichlet'
        assert bc_left.values == 300.0
        assert bc_right.bc_type == 'outflow'
        
    def test_boundary_condition_retrieval(self):
        """Test boundary condition retrieval."""
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        grid.set_boundary_condition('left', 'dirichlet', values=100.0)
        grid.set_boundary_condition('right', 'neumann')
        
        # Test retrieval of boundary conditions
        bc_left = grid.get_boundary_condition('left')
        bc_right = grid.get_boundary_condition('right')
        
        assert bc_left is not None
        assert bc_left.bc_type == 'dirichlet'
        assert bc_left.values == 100.0
        
        assert bc_right is not None
        assert bc_right.bc_type == 'neumann'
        
    def test_cell_volumes(self):
        """Test cell volume computation."""
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        volumes = grid.compute_cell_volumes()
        
        assert len(volumes) == 10
        npt.assert_array_almost_equal(volumes, 0.1 * np.ones(10))
        
    def test_grid_validation_errors(self):
        """Test grid validation and error handling."""
        with pytest.raises(ValueError):
            LNSGrid.create_uniform_1d(nx=0, x_min=0.0, x_max=1.0)  # Invalid nx
            
        with pytest.raises(ValueError):
            LNSGrid.create_uniform_1d(nx=10, x_min=1.0, x_max=0.0)  # Invalid bounds


class TestEnhancedLNSState:
    """Test cases for EnhancedLNSState class."""
    
    def test_state_initialization(self):
        """Test enhanced state initialization."""
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        config = StateConfiguration(include_heat_flux=True, include_stress=True)
        state = EnhancedLNSState(grid, config)
        
        assert state.grid == grid
        assert state.config == config
        assert state.Q.shape == (10, 5)  # density, momentum_x, total_energy, heat_flux_x, stress_xx
        assert len(state.config.variable_names) == 5
        
    def test_named_accessors(self):
        """Test named accessor properties."""
        grid = LNSGrid.create_uniform_1d(nx=5, x_min=0.0, x_max=1.0)
        state = EnhancedLNSState(grid)
        
        # Set values using direct array access
        state.Q[:, LNSVariables.DENSITY] = 1.2
        state.Q[:, LNSVariables.MOMENTUM_X] = 0.5
        state.Q[:, LNSVariables.TOTAL_ENERGY] = 250000.0
        
        # Test named accessors
        npt.assert_array_almost_equal(state.density, 1.2 * np.ones(5))
        npt.assert_array_almost_equal(state.momentum_x, 0.5 * np.ones(5))
        npt.assert_array_almost_equal(state.total_energy, 250000.0 * np.ones(5))
        
        # Test computed properties
        velocity = state.velocity_x
        assert velocity.shape == (5,)
        npt.assert_array_almost_equal(velocity, 0.5/1.2 * np.ones(5))  # u = momentum/density
        
    def test_variable_enums(self):
        """Test LNSVariables enum consistency.""" 
        grid = LNSGrid.create_uniform_1d(nx=3, x_min=0.0, x_max=1.0)
        state = EnhancedLNSState(grid)
        
        # Test enum values match expected indices
        assert LNSVariables.DENSITY == 0
        assert LNSVariables.MOMENTUM_X == 1
        assert LNSVariables.TOTAL_ENERGY == 2
        assert LNSVariables.HEAT_FLUX_X == 3
        assert LNSVariables.STRESS_XX == 4
        
        # Test enum access matches direct indexing
        state.Q[:, 0] = 1.0
        state.Q[:, 1] = 0.1
        
        npt.assert_array_equal(state.Q[:, LNSVariables.DENSITY], state.Q[:, 0])
        npt.assert_array_equal(state.Q[:, LNSVariables.MOMENTUM_X], state.Q[:, 1])
        
    def test_primitive_variables(self):
        """Test primitive variable computation."""
        grid = LNSGrid.create_uniform_1d(nx=3, x_min=0.0, x_max=1.0)
        state = EnhancedLNSState(grid)
        
        # Set up realistic state with known values
        state.Q[:, LNSVariables.DENSITY] = 1.0
        state.Q[:, LNSVariables.MOMENTUM_X] = 0.1
        state.Q[:, LNSVariables.TOTAL_ENERGY] = 215007.5  # For T=300K, u=0.1
        
        # Test primitive variable computation
        primitives = state.get_primitive_variables()
        
        npt.assert_array_almost_equal(primitives['density'], 1.0)
        npt.assert_array_almost_equal(primitives['velocity'], 0.1)
        assert primitives['pressure'].shape == (3,)
        assert primitives['temperature'].shape == (3,)
        
    def test_state_initialization_methods(self):
        """Test state initialization methods."""
        grid = LNSGrid.create_uniform_1d(nx=5, x_min=0.0, x_max=1.0)
        state = EnhancedLNSState(grid)
        
        # Test Sod shock tube initialization
        state.initialize_sod_shock_tube()
        
        # Check that left and right states are different
        left_density = state.density[0]
        right_density = state.density[-1]
        assert left_density != right_density
        assert left_density > 0
        assert right_density > 0
        
    def test_state_validation(self):
        """Test state validation functionality."""
        grid = LNSGrid.create_uniform_1d(nx=3, x_min=0.0, x_max=1.0)
        state = EnhancedLNSState(grid)
        
        # Set valid state
        state.Q[:, LNSVariables.DENSITY] = 1.0
        state.Q[:, LNSVariables.MOMENTUM_X] = 0.1
        state.Q[:, LNSVariables.TOTAL_ENERGY] = 215000.0
        
        validation = state.validate_state()
        
        assert isinstance(validation, dict)
        assert 'positive_density' in validation
        assert 'positive_pressure' in validation
        assert 'finite_values' in validation




class TestLNSPhysics:
    """Test cases for LNSPhysics class."""
    
    def test_physics_initialization(self):
        """Test physics initialization."""
        params = LNSPhysicsParameters(
            mu_viscous=1e-3,
            k_thermal=0.025,
            tau_q=1e-6,
            tau_sigma=1e-6
        )
        physics = LNSPhysics(params)
        
        assert physics.params.mu_viscous == 1e-3
        assert physics.thermal_wave_speed > 0
        
    def test_1d_nsf_targets_correct_formula(self):
        """Test CORRECTED 1D NSF targets with proper deviatoric stress."""
        du_dx = 0.1
        dT_dx = -10.0
        material_props = {
            'k_thermal': 0.025,
            'mu_viscous': 1e-3
        }
        
        q_nsf, sigma_nsf = LNSPhysics.compute_1d_nsf_targets(du_dx, dT_dx, material_props)
        
        # Heat flux should be -k * dT/dx
        expected_q = -0.025 * (-10.0)
        npt.assert_almost_equal(q_nsf, expected_q)
        
        # CRITICAL TEST: Deviatoric stress should have 4/3 factor
        expected_sigma = (4.0/3.0) * 1e-3 * 0.1
        npt.assert_almost_equal(sigma_nsf, expected_sigma)
        
        # Verify this is NOT the old incorrect formula
        wrong_sigma = 2.0 * 1e-3 * 0.1
        assert abs(sigma_nsf - wrong_sigma) > 1e-10, "Still using wrong stress formula!"
        
    def test_2d_nsf_targets(self):
        """Test 2D NSF targets computation."""
        # Create test gradients
        shape = (10, 10)
        velocity_gradients = {
            'dux_dx': np.ones(shape) * 0.1,
            'dux_dy': np.ones(shape) * 0.05,
            'duy_dx': np.ones(shape) * 0.02,
            'duy_dy': np.ones(shape) * 0.08,
        }
        temperature_gradients = (
            np.ones(shape) * (-5.0),  # dT_dx
            np.ones(shape) * (-3.0)   # dT_dy
        )
        material_props = {
            'k_thermal': 0.025,
            'mu_viscous': 1e-3
        }
        
        q_nsf, sigma_nsf = LNSPhysics.compute_2d_nsf_targets(
            velocity_gradients, temperature_gradients, material_props
        )
        
        # Check heat flux
        assert q_nsf.shape == (10, 10, 2)
        npt.assert_almost_equal(q_nsf[:, :, 0], 0.025 * 5.0)  # -k * dT_dx
        npt.assert_almost_equal(q_nsf[:, :, 1], 0.025 * 3.0)  # -k * dT_dy
        
        # Check stress tensor (should be 3 components)
        assert sigma_nsf.shape == (10, 10, 3)
        
    def test_2d_objective_derivatives_complete(self):
        """Test COMPLETE 2D objective derivatives (critical fix)."""
        # Create test state and velocity fields
        N_x, N_y = 5, 5
        state_field = np.random.rand(N_x, N_y, 5)  # [q_x, q_y, σ_xx, σ_yy, σ_xy]
        velocity_field = np.random.rand(N_x, N_y, 2)  # [u_x, u_y]
        
        dx, dy = 0.1, 0.1
        
        # Compute objective derivatives
        result = LNSPhysics.compute_2d_objective_derivatives_complete(
            state_field, velocity_field, dx, dy
        )
        
        # Check that we get non-zero results (critical test)
        assert 'heat_flux' in result
        assert 'stress' in result
        
        heat_flux_derivs = result['heat_flux']
        stress_derivs = result['stress']
        
        assert heat_flux_derivs.shape == (N_x, N_y, 2)
        assert stress_derivs.shape == (N_x, N_y, 3)
        
        # CRITICAL: Verify that derivatives are not all zero (the main bug)
        assert np.any(heat_flux_derivs != 0), "Heat flux derivatives are zero - transport terms missing!"
        assert np.any(stress_derivs != 0), "Stress derivatives are zero - transport terms missing!"
        
    def test_equation_of_state(self):
        """Test equation of state computation."""
        physics = LNSPhysics()
        
        density = np.array([1.0, 1.2, 0.8])
        temperature = np.array([300.0, 350.0, 250.0])
        
        pressure = physics.compute_equation_of_state(density, temperature)
        
        expected = density * 287.0 * temperature
        npt.assert_array_almost_equal(pressure, expected)


class TestOptimizedLNSNumerics:
    """Test cases for OptimizedLNSNumerics class."""
    
    def test_numerics_initialization(self):
        """Test optimized numerics initialization."""
        numerics = OptimizedLNSNumerics(n_ghost=2)
        assert numerics.n_ghost == 2
        assert numerics.flux_call_count == 0
        
    def test_primitive_variable_computation(self):
        """Test vectorized primitive variable computation."""
        numerics = OptimizedLNSNumerics()
        
        # Create test conservative state
        Q = np.array([
            [1.0, 0.1, 215007.5, 0.0, 0.0],  # Standard test state
            [1.2, 0.2, 258009.0, 10.0, 5.0]  # Different state
        ])
        
        primitives = numerics.compute_primitive_variables_vectorized(Q)
        
        # Check that all expected keys are present
        expected_keys = ['density', 'velocity', 'pressure', 'temperature', 'sound_speed', 'heat_flux', 'stress']
        for key in expected_keys:
            assert key in primitives
        
        # Check shapes
        assert primitives['density'].shape == (2,)
        assert primitives['velocity'].shape == (2,)
        assert primitives['pressure'].shape == (2,)
        
        # Check physical validity
        assert np.all(primitives['density'] > 0)
        assert np.all(primitives['pressure'] > 0)
        assert np.all(primitives['temperature'] > 0)
        
    def test_optimized_hll_flux(self):
        """Test optimized HLL flux computation."""
        numerics = OptimizedLNSNumerics()
        
        # Sod shock tube states
        Q_L = np.array([1.0, 0.0, 2.5, 0.0, 0.0])
        Q_R = np.array([0.125, 0.0, 0.25, 0.0, 0.0])
        
        # Pre-computed primitives
        P_L = {
            'density': 1.0, 'velocity': 0.0, 'pressure': 1.0, 'sound_speed': 1.18
        }
        P_R = {
            'density': 0.125, 'velocity': 0.0, 'pressure': 0.1, 'sound_speed': 1.06
        }
        
        physics_params = {'gamma': 1.4}
        
        flux, wave_speed = numerics.optimized_hll_flux_1d(Q_L, Q_R, P_L, P_R, physics_params)
        
        assert len(flux) == 5
        assert np.all(np.isfinite(flux))
        assert wave_speed > 0
        
    def test_ssp_rk2_step_optimized(self):
        """Test optimized SSP-RK2 time stepping."""
        numerics = OptimizedLNSNumerics()
        
        # Simple test state
        Q = np.array([[1.0, 0.1, 215000.0, 0.0, 0.0]])
        
        def simple_rhs(Q_in):
            # Simple decay RHS for testing
            return -0.1 * Q_in, 1.0
        
        dt = 1e-6
        Q_new = numerics.ssp_rk2_step_optimized(Q, simple_rhs, dt)
        
        assert Q_new.shape == Q.shape
        assert np.all(np.isfinite(Q_new))
        assert np.all(Q_new[:, 0] > 0)  # Positive density maintained
        
    def test_cfl_time_step_computation(self):
        """Test CFL time step computation."""
        numerics = OptimizedLNSNumerics()
        
        primitives = {
            'velocity': np.array([0.1, 0.2]),
            'sound_speed': np.array([340.0, 350.0])
        }
        
        dx = 0.01
        dt = numerics.compute_cfl_time_step(primitives, dx, cfl_target=0.8)
        
        assert dt > 0
        assert dt < 1e-3  # Should be small for reasonable CFL
        
    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        numerics = OptimizedLNSNumerics()
        
        # Initially zero
        stats = numerics.get_performance_stats()
        assert stats['total_flux_calls'] == 0
        assert stats['total_flux_time'] == 0.0


# Integration tests
class TestCoreIntegration:
    """Integration tests combining multiple core components."""
    
    def test_complete_1d_setup(self):
        """Test complete 1D setup with all modern core components."""
        # Create grid
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        
        # Create enhanced state
        config = StateConfiguration(include_heat_flux=True, include_stress=True)
        state = EnhancedLNSState(grid, config)
        state.initialize_sod_shock_tube()
        
        # Create physics
        physics_params = LNSPhysicsParameters(
            mu_viscous=1e-5, k_thermal=0.025, tau_q=1e-6, tau_sigma=1e-6
        )
        physics = LNSPhysics(physics_params)
        
        # Create optimized numerics
        numerics = OptimizedLNSNumerics(n_ghost=2)
        
        # Test that everything works together
        primitives = state.get_primitive_variables()
        assert 'density' in primitives
        assert 'velocity' in primitives
        assert 'pressure' in primitives
        assert 'temperature' in primitives
        
        # Test physics computation
        du_dx = 0.1
        dT_dx = -5.0
        material_props = {'k_thermal': 0.025, 'mu_viscous': 1e-3}
        
        q_nsf, sigma_nsf = physics.compute_1d_nsf_targets(du_dx, dT_dx, material_props)
        
        assert np.isfinite(q_nsf)
        assert np.isfinite(sigma_nsf)
        
        # Test primitive variable computation with numerics
        Q_primitives = numerics.compute_primitive_variables_vectorized(state.Q)
        assert 'density' in Q_primitives
        assert 'velocity' in Q_primitives
        
        # Verify enhanced state features
        assert hasattr(state, 'density')
        assert hasattr(state, 'velocity_x')
        assert state.config.n_variables == 5
        
    def test_solver_integration(self):
        """Test integration with FinalIntegratedLNSSolver1D."""
        # Create solver using modern API
        solver = FinalIntegratedLNSSolver1D.create_sod_shock_tube(nx=20)
        
        # Verify solver components
        assert isinstance(solver.state, EnhancedLNSState)
        assert isinstance(solver.numerics, OptimizedLNSNumerics)
        assert isinstance(solver.physics, LNSPhysics)
        
        # Test initial state
        primitives = solver.state.get_primitive_variables()
        assert 'density' in primitives
        assert 'velocity' in primitives
        assert np.any(primitives['density'] > 0)
        
        # Test that solver has the expected public API
        assert hasattr(solver, 'solve')
        assert callable(getattr(solver, 'solve'))
        
        # Test basic solver functionality with small simulation
        try:
            results = solver.solve(t_final=1e-6, dt_initial=1e-9)
            solver_functional = True
            has_output = 'output_data' in results and len(results['output_data']) > 0
        except Exception as e:
            solver_functional = False
            has_output = False
            
        # Solver should be functional for basic operations
        assert solver_functional, "Solver should successfully run basic simulation"
        assert has_output, "Solver should produce meaningful output data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])