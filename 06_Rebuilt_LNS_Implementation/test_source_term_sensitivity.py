#!/usr/bin/env python3
"""
Test source term parameter sensitivity.

This script tests whether the source terms actually depend on τ parameters.
"""

import numpy as np
from lns_solver.solvers.solver_1d_final import FinalIntegratedLNSSolver1D
from lns_solver.core.grid import LNSGrid
from lns_solver.core.physics import LNSPhysicsParameters, LNSPhysics
from lns_solver.core.boundary_conditions import create_outflow_bc

def test_source_term_sensitivity():
    """Test that source terms depend on τ parameters."""
    print("🔬 Testing Source Term Parameter Sensitivity")
    print("=" * 60)
    
    # Create test setup
    grid = LNSGrid.create_uniform_1d(10, 0.0, 1.0)
    
    # Create test state with non-zero heat flux and stress
    nx = 10
    Q_test = np.ones((nx, 5))
    Q_test[:, 0] = 1.0     # density
    Q_test[:, 1] = 100.0   # momentum (significant velocity)
    Q_test[:, 2] = 250000.0 # energy
    Q_test[:, 3] = 200.0   # heat flux (non-zero)
    Q_test[:, 4] = 150.0   # stress (non-zero)
    
    # Create velocity gradient to generate production terms
    u = Q_test[:, 1] / Q_test[:, 0]  # 100 m/s velocity
    u[5:] = 50.0  # Create velocity gradient at midpoint
    Q_test[:, 1] = Q_test[:, 0] * u  # Update momentum
    
    # Test different relaxation times
    tau_values = [1e-2, 1e-3, 1e-4, 1e-5]
    
    print(f"Test state setup:")
    print(f"  Velocity range: {np.min(u):.1f} - {np.max(u):.1f} m/s")
    print(f"  Heat flux: {Q_test[0, 3]:.1f}")
    print(f"  Stress: {Q_test[0, 4]:.1f}")
    print(f"  Velocity gradient: {np.gradient(u, grid.dx)[0]:.1f} 1/s")
    print()
    
    for tau in tau_values:
        print(f"Testing τ = {tau:.1e} s:")
        
        # Create physics with this relaxation time
        physics_params_dict = {
            'mu_viscous': 1e-5,
            'k_thermal': 0.025,
            'tau_q': tau,
            'tau_sigma': tau,
            'gamma': 1.4,
            'R_gas': 287.0,
            'dx': grid.dx
        }
        
        physics_params = LNSPhysicsParameters(
            mu_viscous=1e-5,
            k_thermal=0.025,
            tau_q=tau,
            tau_sigma=tau
        )
        physics = LNSPhysics(physics_params)
        
        # Create solver
        solver = FinalIntegratedLNSSolver1D(
            grid, physics, n_ghost=2, use_operator_splitting=False  # Use direct method for clarity
        )
        solver.set_boundary_condition('left', create_outflow_bc())
        solver.set_boundary_condition('right', create_outflow_bc())
        
        # Test source term computation
        source_terms = solver._compute_source_terms(Q_test, physics_params_dict)
        
        # Analyze source terms
        heat_flux_source = source_terms[0, 3]  # Heat flux source at first cell
        stress_source = source_terms[0, 4]     # Stress source at first cell
        
        print(f"  Heat flux source: {heat_flux_source:.6e}")
        print(f"  Stress source: {stress_source:.6e}")
        
        # Check for τ dependence (relaxation terms should scale as 1/τ)
        if tau > 1e-6:
            expected_relaxation_scale = 1.0 / tau
            print(f"  Expected relaxation scale: ~{expected_relaxation_scale:.1e}")
        
        print()
    
    print("🔍 Analysis:")
    print("  If source terms are identical across τ values, there's a bug.")
    print("  If source terms scale with 1/τ, relaxation physics is working.")
    print("  If source terms have τ-independent parts, production physics is working.")

def test_production_vs_relaxation():
    """Separate test for production vs relaxation terms."""
    print("\n🔬 Testing Production vs Relaxation Separation")
    print("=" * 60)
    
    # Create test with strong velocity gradient
    grid = LNSGrid.create_uniform_1d(10, 0.0, 1.0)
    nx = 10
    Q_test = np.ones((nx, 5))
    Q_test[:, 0] = 1.0     # density
    Q_test[:, 2] = 250000.0 # energy
    Q_test[:, 3] = 100.0   # heat flux
    Q_test[:, 4] = 50.0    # stress
    
    # Create strong velocity gradient
    x = grid.x
    u_profile = 100.0 * np.sin(2 * np.pi * x)  # Sinusoidal velocity
    Q_test[:, 1] = Q_test[:, 0] * u_profile
    
    physics_params_dict = {
        'mu_viscous': 1e-5,
        'k_thermal': 0.025,
        'tau_q': 1e-4,
        'tau_sigma': 1e-4,
        'gamma': 1.4,
        'R_gas': 287.0,
        'dx': grid.dx
    }
    
    physics_params = LNSPhysicsParameters(
        mu_viscous=1e-5,
        k_thermal=0.025,
        tau_q=1e-4,
        tau_sigma=1e-4
    )
    physics = LNSPhysics(physics_params)
    
    solver = FinalIntegratedLNSSolver1D(
        grid, physics, n_ghost=2, use_operator_splitting=False
    )
    solver.set_boundary_condition('left', create_outflow_bc())
    solver.set_boundary_condition('right', create_outflow_bc())
    
    # Compute source terms
    source_terms = solver._compute_source_terms(Q_test, physics_params_dict)
    
    print(f"Velocity profile: {u_profile[0]:.1f} to {u_profile[-1]:.1f} m/s")
    print(f"Velocity gradient range: {np.min(np.gradient(u_profile, grid.dx)):.1f} to {np.max(np.gradient(u_profile, grid.dx)):.1f} 1/s")
    print()
    
    print("Source terms across domain:")
    for i in range(0, nx, 2):  # Sample every other point
        print(f"  Cell {i}: heat_flux_src = {source_terms[i, 3]:.3e}, stress_src = {source_terms[i, 4]:.3e}")
    
    # Check if source terms vary across domain (production terms should create spatial variation)
    heat_flux_variation = np.std(source_terms[:, 3])
    stress_variation = np.std(source_terms[:, 4])
    
    print(f"\nSpatial variation:")
    print(f"  Heat flux source std: {heat_flux_variation:.3e}")
    print(f"  Stress source std: {stress_variation:.3e}")
    
    if heat_flux_variation > 1e-10 or stress_variation > 1e-10:
        print("✅ Source terms show spatial variation - production terms working")
    else:
        print("❌ Source terms uniform - production terms missing")

if __name__ == "__main__":
    test_source_term_sensitivity()
    test_production_vs_relaxation()