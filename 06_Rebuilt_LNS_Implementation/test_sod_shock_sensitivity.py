#!/usr/bin/env python3
"""
Test Sod shock tube parameter sensitivity.

This script specifically tests the Sod shock tube setup used in validation
to see if the LNS terms have any effect in that particular configuration.
"""

import numpy as np
from lns_solver.solvers.solver_1d_final import FinalIntegratedLNSSolver1D
from lns_solver.core.grid import LNSGrid
from lns_solver.core.physics import LNSPhysicsParameters, LNSPhysics
from lns_solver.core.boundary_conditions import create_outflow_bc

def test_sod_shock_lns_sensitivity():
    """Test LNS parameter sensitivity in Sod shock tube."""
    print("🔬 Testing Sod Shock Tube LNS Sensitivity")
    print("=" * 60)
    
    nx = 50
    t_final = 2e-3  # Increase simulation time to see shock propagation
    
    # Test different relaxation times
    tau_values = [1e-2, 1e-4, 1e-6]
    
    results = {}
    
    for tau in tau_values:
        print(f"\nTesting τ = {tau:.1e} s:")
        
        # Create solver with this relaxation time
        grid = LNSGrid.create_uniform_1d(nx, 0.0, 1.0)
        
        physics_params = LNSPhysicsParameters(
            mu_viscous=1e-5,
            k_thermal=0.025,
            tau_q=tau,
            tau_sigma=tau
        )
        physics = LNSPhysics(physics_params)
        
        solver = FinalIntegratedLNSSolver1D(
            grid, physics, n_ghost=2, use_operator_splitting=True  # Use operator splitting like stable validation
        )
        solver.state.initialize_sod_shock_tube()
        solver.set_boundary_condition('left', create_outflow_bc())
        solver.set_boundary_condition('right', create_outflow_bc())
        
        print(f"  Initial conditions set up")
        
        # Check initial source terms
        initial_Q = solver.state.Q.copy()
        physics_params_dict = {
            'mu_viscous': 1e-5,
            'k_thermal': 0.025,
            'tau_q': tau,
            'tau_sigma': tau,
            'gamma': 1.4,
            'R_gas': 287.0,
            'dx': grid.dx
        }
        
        initial_source = solver._compute_source_terms(initial_Q, physics_params_dict)
        
        # Check if source terms are non-zero
        max_heat_source = np.max(np.abs(initial_source[:, 3]))
        max_stress_source = np.max(np.abs(initial_source[:, 4]))
        
        print(f"  Initial max heat flux source: {max_heat_source:.3e}")
        print(f"  Initial max stress source: {max_stress_source:.3e}")
        
        # Run short simulation
        sim_results = solver.solve(t_final=t_final, dt_initial=1e-6)
        final_primitives = sim_results['output_data']['primitives'][-1]
        
        max_vel = np.max(np.abs(final_primitives['velocity']))
        min_density = np.min(final_primitives['density'])
        
        results[tau] = {
            'max_velocity': max_vel,
            'min_density': min_density,
            'initial_heat_source': max_heat_source,
            'initial_stress_source': max_stress_source,
            'iterations': sim_results.get('iterations', 0)
        }
        
        print(f"  Final max velocity: {max_vel:.1f} m/s")
        print(f"  Final min density: {min_density:.6f} kg/m³")
        print(f"  Iterations: {results[tau]['iterations']}")
    
    # Analyze results
    print(f"\n🔍 SENSITIVITY ANALYSIS:")
    print("=" * 40)
    
    velocities = [results[tau]['max_velocity'] for tau in tau_values]
    densities = [results[tau]['min_density'] for tau in tau_values]
    
    vel_variation = np.std(velocities)
    density_variation = np.std(densities)
    
    print(f"Max velocity variation: {vel_variation:.6f} m/s")
    print(f"Min density variation: {density_variation:.9f} kg/m³")
    
    print(f"\nDetailed results:")
    for tau in tau_values:
        r = results[tau]
        print(f"  τ = {tau:.1e}: vel = {r['max_velocity']:.3f} m/s, ρ_min = {r['min_density']:.6f}")
    
    # Check for meaningful differences
    if vel_variation > 0.1 or density_variation > 1e-6:
        print(f"\n✅ SENSITIVITY DETECTED: LNS effects are visible")
        print(f"   Velocity changes by {vel_variation:.3f} m/s across τ values")
        print(f"   Density changes by {density_variation:.2e} kg/m³ across τ values")
    else:
        print(f"\n❌ NO SENSITIVITY: Results identical across τ values")
        print(f"   This suggests LNS terms may not be properly integrated")
    
    # Check initial source terms
    print(f"\nInitial source term analysis:")
    source_terms_present = any(results[tau]['initial_heat_source'] > 1e-10 or 
                             results[tau]['initial_stress_source'] > 1e-10 
                             for tau in tau_values)
    
    if source_terms_present:
        print(f"✅ Non-zero initial source terms detected")
    else:
        print(f"❌ Initial source terms are zero - may explain lack of sensitivity")

def analyze_sod_shock_gradients():
    """Analyze gradients in Sod shock tube to understand LNS activation."""
    print(f"\n🔬 Analyzing Sod Shock Tube Gradients")
    print("=" * 60)
    
    # Create standard Sod shock setup
    grid = LNSGrid.create_uniform_1d(50, 0.0, 1.0)
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
    solver.state.initialize_sod_shock_tube()
    
    # Get initial state
    Q_initial = solver.state.Q
    primitives = solver.numerics.compute_primitive_variables_vectorized(
        Q_initial,
        gamma=1.4,
        R_gas=287.0
    )
    
    u = primitives['velocity']
    T = primitives['temperature']
    
    # Compute gradients
    du_dx = np.gradient(u, grid.dx)
    dT_dx = np.gradient(T, grid.dx)
    
    print(f"Initial conditions analysis:")
    print(f"  Velocity range: {np.min(u):.3f} to {np.max(u):.3f} m/s")
    print(f"  Temperature range: {np.min(T):.1f} to {np.max(T):.1f} K")
    print(f"  Max velocity gradient: {np.max(np.abs(du_dx)):.3e} 1/s")
    print(f"  Max temperature gradient: {np.max(np.abs(dT_dx)):.3e} K/m")
    
    # Check heat flux and stress initial values
    q_initial = Q_initial[:, 3]
    sigma_initial = Q_initial[:, 4]
    
    print(f"  Initial heat flux range: {np.min(q_initial):.3e} to {np.max(q_initial):.3e}")
    print(f"  Initial stress range: {np.min(sigma_initial):.3e} to {np.max(sigma_initial):.3e}")
    
    # The issue might be that Sod shock tube starts with zero heat flux and stress
    if np.max(np.abs(q_initial)) < 1e-10 and np.max(np.abs(sigma_initial)) < 1e-10:
        print(f"\n⚠️  POTENTIAL ISSUE: Initial heat flux and stress are zero!")
        print(f"   With zero initial LNS variables, production terms may be minimal")
        print(f"   LNS effects might only appear after some evolution time")

if __name__ == "__main__":
    test_sod_shock_lns_sensitivity()
    analyze_sod_shock_gradients()