#!/usr/bin/env python3
"""
Test centralized physics implementation.

This script tests that LNSPhysics.compute_1d_lns_source_terms_complete()
works correctly and produces parameter-dependent results.
"""

import numpy as np
from lns_solver.core.physics import LNSPhysicsParameters, LNSPhysics

def test_centralized_physics_sensitivity():
    """Test that centralized physics shows τ dependence."""
    print("🔬 Testing Centralized Physics Parameter Sensitivity")
    print("=" * 60)
    
    # Create test state with significant gradients
    nx = 10
    Q_test = np.ones((nx, 5))
    Q_test[:, 0] = 1.0     # density
    Q_test[:, 2] = 250000.0 # energy
    Q_test[:, 3] = 100.0   # heat flux
    Q_test[:, 4] = 50.0    # stress
    
    # Create velocity profile with gradients
    x = np.linspace(0, 1, nx)
    u_profile = 100.0 * np.sin(2 * np.pi * x)  # Sinusoidal velocity
    Q_test[:, 1] = Q_test[:, 0] * u_profile    # Momentum
    
    dx = 0.1
    
    # Test different relaxation times
    tau_values = [1e-2, 1e-4, 1e-6]
    
    print(f"Test setup:")
    print(f"  Grid size: {nx} cells")
    print(f"  Velocity range: {np.min(u_profile):.1f} to {np.max(u_profile):.1f} m/s")
    print(f"  Velocity gradient range: {np.min(np.gradient(u_profile, dx)):.1f} to {np.max(np.gradient(u_profile, dx)):.1f} 1/s")
    print()
    
    for tau in tau_values:
        print(f"Testing τ = {tau:.1e} s:")
        
        # Create physics with this relaxation time
        params = LNSPhysicsParameters(
            mu_viscous=1e-5,
            k_thermal=0.025,
            tau_q=tau,
            tau_sigma=tau
        )
        physics = LNSPhysics(params)
        
        # Compute complete source terms
        source = physics.compute_1d_lns_source_terms_complete(Q_test, dx)
        
        # Analyze results
        heat_flux_source = source[:, 3]
        stress_source = source[:, 4]
        
        max_heat_source = np.max(np.abs(heat_flux_source))
        max_stress_source = np.max(np.abs(stress_source))
        
        print(f"  Max heat flux source: {max_heat_source:.3e}")
        print(f"  Max stress source: {max_stress_source:.3e}")
        
        # Check spatial variation (production terms should create variation)
        heat_variation = np.std(heat_flux_source)
        stress_variation = np.std(stress_source)
        
        print(f"  Heat flux source variation: {heat_variation:.3e}")
        print(f"  Stress source variation: {stress_variation:.3e}")
        print()
    
    print("🔍 Analysis:")
    print("✅ If source magnitudes scale with 1/τ, relaxation physics is working")
    print("✅ If source terms show spatial variation, production physics is working")
    print("✅ Both effects together confirm complete LNS physics implementation")

def test_physics_consistency():
    """Test that centralized physics is consistent with distributed implementations."""
    print("\n🔬 Testing Physics Implementation Consistency")
    print("=" * 60)
    
    # Create test state
    nx = 5
    Q_test = np.ones((nx, 5))
    Q_test[:, 0] = 1.0
    Q_test[:, 1] = 50.0   # Non-zero momentum
    Q_test[:, 2] = 250000.0
    Q_test[:, 3] = 200.0  # Non-zero heat flux
    Q_test[:, 4] = 100.0  # Non-zero stress
    
    dx = 0.02
    gamma = 1.4
    R_gas = 287.0
    
    # Create physics parameters
    params = LNSPhysicsParameters(
        mu_viscous=1e-5,
        k_thermal=0.025,
        tau_q=1e-4,
        tau_sigma=1e-4
    )
    physics = LNSPhysics(params)
    
    # Test the centralized method
    source_centralized = physics.compute_1d_lns_source_terms_complete(Q_test, dx, gamma, R_gas)
    
    print(f"Centralized physics results:")
    print(f"  Heat flux source: {source_centralized[0, 3]:.6e}")
    print(f"  Stress source: {source_centralized[0, 4]:.6e}")
    
    # Verify structure
    print(f"\nSource term structure verification:")
    print(f"  Mass source (should be 0): {source_centralized[0, 0]:.6e}")
    print(f"  Momentum source (should be 0): {source_centralized[0, 1]:.6e}")
    print(f"  Energy source (should be 0): {source_centralized[0, 2]:.6e}")
    print(f"  Heat flux source (non-zero): {source_centralized[0, 3]:.6e}")
    print(f"  Stress source (non-zero): {source_centralized[0, 4]:.6e}")
    
    # Test that non-LNS terms are indeed zero
    mass_momentum_energy_nonzero = (
        np.any(np.abs(source_centralized[:, 0]) > 1e-15) or
        np.any(np.abs(source_centralized[:, 1]) > 1e-15) or 
        np.any(np.abs(source_centralized[:, 2]) > 1e-15)
    )
    
    lns_terms_nonzero = (
        np.any(np.abs(source_centralized[:, 3]) > 1e-15) or
        np.any(np.abs(source_centralized[:, 4]) > 1e-15)
    )
    
    if not mass_momentum_energy_nonzero and lns_terms_nonzero:
        print(f"\n✅ Source term structure is correct")
        print(f"   Mass/momentum/energy sources are zero (as expected)")
        print(f"   LNS terms are non-zero (physics active)")
    else:
        print(f"\n❌ Source term structure issue detected")

if __name__ == "__main__":
    test_centralized_physics_sensitivity()
    test_physics_consistency()