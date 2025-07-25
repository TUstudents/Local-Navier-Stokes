#!/usr/bin/env python3
"""
Test periodic boundary conditions with actual simulation to check for conservation violations.
"""

import numpy as np
from lns_solver.solvers.solver_1d_final import FinalIntegratedLNSSolver1D
from lns_solver.core.grid import LNSGrid
from lns_solver.core.physics import LNSPhysics, LNSPhysicsParameters
from lns_solver.core.boundary_conditions import create_periodic_bc, BoundaryCondition, BCType

def create_periodic_bc():
    """Create periodic boundary condition."""
    return BoundaryCondition(BCType.PERIODIC)

def test_periodic_conservation():
    """Test periodic BC with actual LNS simulation."""
    print("🔬 Testing Periodic BC with LNS Simulation")
    print("=" * 50)
    
    # Create small domain with periodic boundaries
    nx = 20
    grid = LNSGrid.create_uniform_1d(nx, 0.0, 1.0)
    
    # Create physics
    physics_params = LNSPhysicsParameters(
        tau_q=1e-4,
        tau_sigma=1e-4,
        mu_viscous=1e-5,
        k_thermal=0.025
    )
    physics = LNSPhysics(physics_params)
    
    # Create solver with periodic boundaries
    solver = FinalIntegratedLNSSolver1D(grid, physics, n_ghost=2, use_operator_splitting=False)
    
    # Set periodic boundary conditions
    solver.set_boundary_condition('left', create_periodic_bc())
    solver.set_boundary_condition('right', create_periodic_bc())
    
    # Initialize with a smooth sinusoidal wave (should wrap around cleanly)
    x = solver.grid.x
    
    # Set up initial conditions with smooth periodic function
    for i in range(nx):
        # Sinusoidal density and pressure variations
        rho = 1.0 + 0.1 * np.sin(2 * np.pi * x[i])
        u = 0.1 * np.cos(2 * np.pi * x[i])  # Periodic velocity
        p = 101325.0 + 1000.0 * np.sin(2 * np.pi * x[i])
        T = p / (rho * 287.0)
        E = p / (1.4 - 1) + 0.5 * rho * u**2
        
        solver.state.Q[i, 0] = rho
        solver.state.Q[i, 1] = rho * u
        solver.state.Q[i, 2] = E
        solver.state.Q[i, 3] = 10.0 * np.sin(2 * np.pi * x[i])  # Periodic heat flux
        solver.state.Q[i, 4] = 5.0 * np.cos(2 * np.pi * x[i])   # Periodic stress
    
    print(f"Initial conditions:")
    print(f"  Domain: [{x[0]:.3f}, {x[-1]:.3f}] with {nx} cells")
    print(f"  Density range: [{np.min(solver.state.density):.6f}, {np.max(solver.state.density):.6f}]")
    print(f"  Velocity range: [{np.min(solver.state.velocity_x):.6f}, {np.max(solver.state.velocity_x):.6f}]")
    
    # Store initial values for conservation check
    initial_mass = np.sum(solver.state.density) * solver.grid.dx
    initial_momentum = np.sum(solver.state.momentum_x) * solver.grid.dx
    initial_energy = np.sum(solver.state.total_energy) * solver.grid.dx
    
    print(f"\nInitial conserved quantities:")
    print(f"  Mass: {initial_mass:.10f}")
    print(f"  Momentum: {initial_momentum:.10f}")
    print(f"  Energy: {initial_energy:.10f}")
    
    # Check periodicity of initial conditions
    # For truly periodic conditions, values at boundaries should wrap correctly
    print(f"\nPeriodicity check (initial):")
    print(f"  Left boundary density: {solver.state.density[0]:.6f}")
    print(f"  Right boundary density: {solver.state.density[-1]:.6f}")
    print(f"  Should be similar for smooth periodic function")
    
    # Run short simulation
    try:
        results = solver.solve(t_final=1e-5, dt_initial=1e-7)
        
        # Check conservation
        final_mass = np.sum(solver.state.density) * solver.grid.dx
        final_momentum = np.sum(solver.state.momentum_x) * solver.grid.dx
        final_energy = np.sum(solver.state.total_energy) * solver.grid.dx
        
        mass_error = abs(final_mass - initial_mass) / initial_mass
        momentum_error = abs(final_momentum - initial_momentum) / (abs(initial_momentum) + 1e-12)
        energy_error = abs(final_energy - initial_energy) / initial_energy
        
        print(f"\nSimulation completed:")
        print(f"  Time steps: {results['iterations']}")
        print(f"  Final time: {results['final_time']:.2e} s")
        
        print(f"\nConservation check:")
        print(f"  Mass error: {mass_error:.2e}")
        print(f"  Momentum error: {momentum_error:.2e}")
        print(f"  Energy error: {energy_error:.2e}")
        
        # Check if conservation errors are reasonable for periodic BC
        mass_ok = mass_error < 1e-12
        momentum_ok = momentum_error < 1e-10  # Momentum can have larger errors due to dynamics
        energy_ok = energy_error < 1e-10
        
        print(f"\nConservation assessment:")
        print(f"  Mass conservation: {'✅' if mass_ok else '❌'}")
        print(f"  Momentum conservation: {'✅' if momentum_ok else '❌'}")
        print(f"  Energy conservation: {'✅' if energy_ok else '❌'}")
        
        # Check boundary values for periodicity
        print(f"\nPeriodicity check (final):")
        print(f"  Left boundary density: {solver.state.density[0]:.6f}")
        print(f"  Right boundary density: {solver.state.density[-1]:.6f}")
        print(f"  Difference: {abs(solver.state.density[0] - solver.state.density[-1]):.2e}")
        
        if mass_ok and energy_ok:
            print(f"\n✅ Periodic BC appears to conserve properly")
            return True
        else:
            print(f"\n❌ Periodic BC shows conservation violations")
            print(f"This could indicate incorrect ghost cell population")
            return False
            
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        return False

if __name__ == "__main__":
    test_periodic_conservation()