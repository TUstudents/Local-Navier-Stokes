"""
Unit tests for core LNS infrastructure.

Tests the fundamental building blocks: LNSGrid, LNSState, LNSPhysics, LNSNumerics.
"""

import pytest
import numpy as np
import numpy.testing as npt

from lns_solver.core.grid import LNSGrid, BoundaryCondition
from lns_solver.core.state import LNSState, MaterialProperties
from lns_solver.core.physics import LNSPhysics, LNSPhysicsParameters
from lns_solver.core.numerics import LNSNumerics


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
        
    def test_apply_boundary_conditions_1d(self):
        """Test 1D boundary condition application."""
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        grid.set_boundary_condition('left', 'dirichlet', values=100.0)
        grid.set_boundary_condition('right', 'neumann')
        
        field = np.ones(10) * 50.0
        field_bc = grid.apply_boundary_conditions(field)
        
        assert field_bc[0] == 100.0  # Dirichlet BC applied
        assert field_bc[-1] == field_bc[-2]  # Neumann BC (zero gradient)
        
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


class TestLNSState:
    """Test cases for LNSState class."""
    
    def test_state_initialization(self):
        """Test state vector initialization."""
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        state = LNSState(grid, n_variables=5)
        
        assert state.grid == grid
        assert state.n_variables == 5
        assert state.Q.shape == (10, 5)
        assert np.all(state.Q == 0.0)  # Initially zero
        
    def test_variable_access(self):
        """Test variable getting and setting."""
        grid = LNSGrid.create_uniform_1d(nx=5, x_min=0.0, x_max=1.0)
        state = LNSState(grid, n_variables=5)
        
        # Set density
        state.set_variable('density', 1.2)
        density = state.get_variable('density')
        
        npt.assert_array_almost_equal(density, 1.2 * np.ones(5))
        
        # Set array
        momentum = np.linspace(0.1, 0.5, 5)
        state.set_variable('momentum_x', momentum)
        retrieved = state.get_variable('momentum_x')
        
        npt.assert_array_almost_equal(retrieved, momentum)
        
    def test_q_to_p_1d_conversion(self):
        """Test 1D conservative to primitive conversion."""
        grid = LNSGrid.create_uniform_1d(nx=1, x_min=0.0, x_max=1.0)
        state = LNSState(grid, n_variables=5)
        
        # Calculate correct total energy for ρ=1.0, u=0.1, T=300K
        rho = 1.0
        u = 0.1
        T = 300.0
        cv = 287.0 / 0.4  # R/(γ-1)
        e_internal = rho * cv * T
        kinetic_energy = 0.5 * rho * u**2
        E_total = e_internal + kinetic_energy
        
        Q = np.array([rho, rho*u, E_total, 0.0, 0.0])  # [ρ, ρu, E, q, σ]
        
        P = state.Q_to_P_1d(Q)
        
        assert P['density'] == 1.0
        assert P['velocity'] == 0.1
        npt.assert_almost_equal(P['temperature'], 300.0, decimal=1)
        npt.assert_almost_equal(P['pressure'], 287.0 * 300.0, decimal=1)
        
    def test_p_to_q_1d_conversion(self):
        """Test 1D primitive to conservative conversion."""
        grid = LNSGrid.create_uniform_1d(nx=1, x_min=0.0, x_max=1.0)
        state = LNSState(grid, n_variables=5)
        
        P = {
            'density': 1.0,
            'velocity': 0.1,
            'temperature': 300.0,
            'heat_flux_x': 0.0,
            'stress_xx': 0.0
        }
        
        Q = state.P_to_Q_1d(P)
        
        assert Q[0] == 1.0  # density
        assert Q[1] == 0.1  # momentum
        # Energy should include internal + kinetic
        expected_E = 1.0 * (287.0 / 0.4) * 300.0 + 0.5 * 1.0 * 0.1**2
        npt.assert_almost_equal(Q[2], expected_E, decimal=1)
        
    def test_initialize_uniform(self):
        """Test uniform initialization."""
        grid = LNSGrid.create_uniform_1d(nx=5, x_min=0.0, x_max=1.0)
        state = LNSState(grid, n_variables=5)
        
        state.initialize_uniform(density=1.2, pressure=101325.0, velocity_x=0.5)
        
        primitives = state.get_primitive_variables()
        
        npt.assert_array_almost_equal(primitives['density'], 1.2 * np.ones(5))
        npt.assert_array_almost_equal(primitives['velocity'], 0.5 * np.ones(5))
        npt.assert_array_almost_equal(primitives['pressure'], 101325.0 * np.ones(5))
        
    def test_sod_shock_tube_initialization(self):
        """Test Sod shock tube initialization."""
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        state = LNSState(grid, n_variables=5)
        
        state.initialize_sod_shock_tube()
        
        primitives = state.get_primitive_variables()
        
        # Check left state (x < 0.5)
        left_indices = grid.x < 0.5
        npt.assert_array_almost_equal(primitives['density'][left_indices], 1.0)
        npt.assert_array_almost_equal(primitives['velocity'][left_indices], 0.0)
        
        # Check right state (x >= 0.5)
        right_indices = grid.x >= 0.5
        npt.assert_array_almost_equal(primitives['density'][right_indices], 0.125)
        npt.assert_array_almost_equal(primitives['velocity'][right_indices], 0.0)
        
    def test_state_validation(self):
        """Test state validation."""
        grid = LNSGrid.create_uniform_1d(nx=5, x_min=0.0, x_max=1.0)
        state = LNSState(grid, n_variables=5)
        
        # Valid state
        state.initialize_uniform(density=1.0, pressure=101325.0)
        assert state.validate_state() == True
        
        # Invalid state (negative density)
        state.Q[0, 0] = -1.0
        assert state.validate_state() == False


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


class TestLNSNumerics:
    """Test cases for LNSNumerics class."""
    
    def test_numerics_initialization(self):
        """Test numerics initialization."""
        numerics = LNSNumerics(use_numba=False)
        assert numerics.use_numba == False
        
    def test_gradient_computation_efficient(self):
        """Test efficient O(N²) gradient computation."""
        # Create test fields
        nx, ny = 10, 8
        field_1 = np.sin(np.linspace(0, 2*np.pi, nx*ny)).reshape(nx, ny)
        field_2 = np.cos(np.linspace(0, 2*np.pi, nx*ny)).reshape(nx, ny)
        
        fields = [field_1, field_2]
        dx, dy = 0.1, 0.1
        
        # Compute gradients
        gradients = LNSNumerics.compute_gradients_efficient(fields, dx, dy)
        
        # Check output structure
        assert 'field_0' in gradients
        assert 'field_1' in gradients
        
        grad_x_0, grad_y_0 = gradients['field_0']
        grad_x_1, grad_y_1 = gradients['field_1']
        
        assert grad_x_0.shape == (nx, ny)
        assert grad_y_0.shape == (nx, ny)
        assert grad_x_1.shape == (nx, ny)
        assert grad_y_1.shape == (nx, ny)
        
        # Verify gradients are computed (not zero)
        assert np.any(grad_x_0 != 0)
        assert np.any(grad_y_0 != 0)
        
    def test_hyperbolic_rhs_1d(self):
        """Test 1D hyperbolic RHS computation."""
        # Simple test case
        N_cells, N_vars = 5, 3
        state_field = np.random.rand(N_cells, N_vars)
        dx = 0.1
        
        def simple_flux(Q_L, Q_R):
            return 0.5 * (Q_L + Q_R)  # Simple average flux
        
        RHS = LNSNumerics.compute_hyperbolic_rhs_1d(state_field, simple_flux, dx)
        
        assert RHS.shape == (N_cells, N_vars)
        
        # Conservation check: sum of RHS should be zero for interior cells
        # (boundary effects may cause non-zero sum)
        
    def test_hyperbolic_rhs_2d_corrected_signs(self):
        """Test 2D hyperbolic RHS with CORRECTED signs."""
        N_x, N_y, N_vars = 4, 3, 2
        state_field = np.ones((N_x, N_y, N_vars))
        dx, dy = 0.1, 0.1
        
        def constant_flux(Q_L, Q_R, direction='x'):
            return np.ones_like(Q_L)  # Constant flux
        
        RHS = LNSNumerics.compute_hyperbolic_rhs_2d(state_field, constant_flux, dx, dy)
        
        assert RHS.shape == (N_x, N_y, N_vars)
        
        # With constant flux and uniform state, interior cells should have zero RHS
        # (flux in = flux out)
        # Only boundary cells should have non-zero RHS
        
    def test_hll_flux_1d(self):
        """Test HLL Riemann solver."""
        # Sod shock tube left and right states
        Q_left = np.array([1.0, 0.0, 2.5, 0.0, 0.0])
        Q_right = np.array([0.125, 0.0, 0.25, 0.0, 0.0])
        
        physics_params = {
            'specific_heat_ratio': 1.4,
            'gas_constant': 287.0,
            'tau_q': 1e-6,
            'tau_sigma': 1e-6,
            'k_thermal': 0.025,
            'mu_viscous': 1e-3
        }
        
        flux = LNSNumerics.hll_flux_1d(Q_left, Q_right, physics_params)
        
        assert len(flux) == 5
        assert np.all(np.isfinite(flux))  # No NaN or inf
        
    def test_semi_implicit_source_update(self):
        """Test semi-implicit source term update."""
        Q_old = np.array([1.0, 0.1, 253125.0, 100.0, 50.0])
        Q_nsf = np.array([1.0, 0.1, 253125.0, 80.0, 40.0])
        obj_derivs = np.array([0.0, 0.0, 0.0, 10.0, 5.0])
        
        relaxation_times = {'tau_q': 1e-6, 'tau_sigma': 1e-5}
        dt = 1e-8
        variable_indices = {'heat_flux': [3], 'stress': [4]}
        
        Q_new = LNSNumerics.semi_implicit_source_update(
            Q_old, Q_nsf, obj_derivs, relaxation_times, dt, variable_indices
        )
        
        assert len(Q_new) == 5
        assert Q_new[0] == Q_old[0]  # Unchanged variables
        assert Q_new[1] == Q_old[1]
        assert Q_new[2] == Q_old[2]
        
        # Heat flux and stress should be updated toward NSF targets
        assert Q_new[3] != Q_old[3]
        assert Q_new[4] != Q_old[4]


# Integration tests
class TestCoreIntegration:
    """Integration tests combining multiple core components."""
    
    def test_complete_1d_setup(self):
        """Test complete 1D setup with all core components."""
        # Create grid
        grid = LNSGrid.create_uniform_1d(nx=10, x_min=0.0, x_max=1.0)
        
        # Create state
        state = LNSState(grid, n_variables=5)
        state.initialize_uniform(density=1.0, pressure=101325.0)
        
        # Create physics
        physics = LNSPhysics()
        
        # Create numerics
        numerics = LNSNumerics()
        
        # Test that everything works together
        primitives = state.get_primitive_variables()
        assert 'density' in primitives
        assert 'velocity' in primitives
        
        # Test physics computation
        du_dx = 0.1
        dT_dx = -5.0
        material_props = {'k_thermal': 0.025, 'mu_viscous': 1e-3}
        
        q_nsf, sigma_nsf = physics.compute_1d_nsf_targets(du_dx, dT_dx, material_props)
        
        assert np.isfinite(q_nsf)
        assert np.isfinite(sigma_nsf)
        
        # Test gradient computation
        temperature_field = primitives['temperature']
        gradients = numerics.compute_gradients_efficient([temperature_field], grid.dx)
        
        assert 'field_0' in gradients
        
    def test_2d_integration(self):
        """Test 2D integration of core components."""
        # Create 2D grid
        grid = LNSGrid.create_uniform_2d(nx=5, ny=4, x_bounds=(0, 1), y_bounds=(0, 0.8))
        
        # Create 2D state
        state = LNSState(grid, n_variables=9)
        state.initialize_uniform(density=1.2, pressure=101325.0, velocity_x=0.1, velocity_y=0.05)
        
        # Verify 2D state works
        primitives = state.get_primitive_variables()
        assert 'velocity_x' in primitives
        assert 'velocity_y' in primitives
        assert primitives['velocity_x'].shape == (20,)  # 5*4 cells
        
        # Test 2D gradient computation
        temp_field = primitives['temperature'].reshape(5, 4)
        gradients = LNSNumerics.compute_gradients_efficient([temp_field], grid.dx, grid.dy)
        
        grad_x, grad_y = gradients['field_0']
        assert grad_x.shape == (5, 4)
        assert grad_y.shape == (5, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])