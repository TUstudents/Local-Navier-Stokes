#!/usr/bin/env python3
"""
Test for the operator splitting API fix.

This test validates that the misleading function arguments issue has been resolved
and that the source_rhs function provided to step() is now actually used.
"""

import numpy as np
from lns_solver.core.operator_splitting import StrangSplitting, ImplicitRelaxationSolver, AdaptiveOperatorSplitting

def test_source_rhs_actually_used():
    """Test that the provided source_rhs function is actually used."""
    print("🔧 Testing Source RHS Function Usage Fix")
    print("-" * 50)
    
    # Create a simple test case
    nx = 10
    n_vars = 5
    Q_test = np.ones((nx, n_vars))
    Q_test[:, 0] = 1.0    # density
    Q_test[:, 1] = 0.1    # momentum  
    Q_test[:, 2] = 250000 # energy
    Q_test[:, 3] = 100.0  # heat flux
    Q_test[:, 4] = 50.0   # stress
    
    dt = 1e-6
    physics_params = {
        'gamma': 1.4,
        'R_gas': 287.0,
        'tau_q': 1e-5,
        'tau_sigma': 1e-5,
        'dx': 0.1
    }
    
    # Flag to track if our source_rhs function was called
    source_rhs_called = False
    source_rhs_call_count = 0
    
    def custom_source_rhs(Q):
        """Custom source function that sets a flag when called."""
        nonlocal source_rhs_called, source_rhs_call_count
        source_rhs_called = True
        source_rhs_call_count += 1
        
        # Return some non-zero source terms for testing
        source = np.zeros_like(Q)
        source[:, 3] = 1000.0  # Heat flux source
        source[:, 4] = 500.0   # Stress source
        return source
    
    def dummy_hyperbolic_rhs(Q):
        """Dummy hyperbolic function."""
        return np.zeros_like(Q)
    
    print("Testing FIXED operator splitting (use_advanced_source_solver=False):")
    
    # Test with FIXED API (should use provided source_rhs)
    splitter_fixed = StrangSplitting(
        implicit_solver=ImplicitRelaxationSolver(),
        use_advanced_source_solver=False  # Use provided source_rhs
    )
    
    # Reset flags
    source_rhs_called = False
    source_rhs_call_count = 0
    
    Q_result_fixed = splitter_fixed.step(
        Q_test.copy(), dt, dummy_hyperbolic_rhs, custom_source_rhs, physics_params
    )
    
    print(f"   Custom source_rhs called: {'✅' if source_rhs_called else '❌'}")
    print(f"   Call count: {source_rhs_call_count} (expected: 2 for Strang splitting)")
    print(f"   Result shape: {Q_result_fixed.shape}")
    
    # Store results from fixed test
    fixed_called_source = source_rhs_called
    fixed_call_count = source_rhs_call_count
    
    # Verify that the source terms were applied (use appropriate tolerance)
    heat_flux_changed = np.max(np.abs(Q_result_fixed[:, 3] - Q_test[:, 3])) > 1e-6
    stress_changed = np.max(np.abs(Q_result_fixed[:, 4] - Q_test[:, 4])) > 1e-6
    
    print(f"   Heat flux modified: {'✅' if heat_flux_changed else '❌'}")
    print(f"   Stress modified: {'✅' if stress_changed else '❌'}")
    
    # Debug: Show actual changes
    heat_flux_change = np.max(np.abs(Q_result_fixed[:, 3] - Q_test[:, 3]))
    stress_change = np.max(np.abs(Q_result_fixed[:, 4] - Q_test[:, 4]))
    print(f"   Max heat flux change: {heat_flux_change:.6f}")
    print(f"   Max stress change: {stress_change:.6f}")
    
    # Test backward compatibility (should NOT call provided source_rhs)
    print(f"\nTesting backward compatibility (use_advanced_source_solver=True):")
    
    splitter_compat = StrangSplitting(
        implicit_solver=ImplicitRelaxationSolver(),
        use_advanced_source_solver=True  # Use internal solver, ignore source_rhs
    )
    
    # Reset flags
    source_rhs_called = False
    source_rhs_call_count = 0
    
    Q_result_compat = splitter_compat.step(
        Q_test.copy(), dt, dummy_hyperbolic_rhs, custom_source_rhs, physics_params
    )
    
    print(f"   Custom source_rhs called: {'❌' if not source_rhs_called else '✅'} (should be ❌)")
    print(f"   Call count: {source_rhs_call_count} (expected: 0 for backward compatibility)")
    print(f"   Result shape: {Q_result_compat.shape}")
    
    # Store results from compat test
    compat_called_source = source_rhs_called
    compat_call_count = source_rhs_call_count
    
    # Both modes should still modify the state (just using different source terms)
    heat_flux_changed_compat = not np.allclose(Q_result_compat[:, 3], Q_test[:, 3])
    stress_changed_compat = not np.allclose(Q_result_compat[:, 4], Q_test[:, 4])
    
    print(f"   Heat flux modified: {'✅' if heat_flux_changed_compat else '❌'}")
    print(f"   Stress modified: {'✅' if stress_changed_compat else '❌'}")
    
    # Validation: Check that the fix works correctly
    fixed_api_works = fixed_called_source and fixed_call_count > 0  # Fixed version should call source_rhs
    compat_api_works = not compat_called_source and compat_call_count == 0  # Compat should not call source_rhs
    both_modify_state = heat_flux_changed and heat_flux_changed_compat  # Both should modify state
    
    return fixed_api_works and compat_api_works and both_modify_state

def test_adaptive_splitting_integration():
    """Test that AdaptiveOperatorSplitting works with the API fix."""
    print(f"\n🔧 Testing Adaptive Splitting Integration")
    print("-" * 50)
    
    # Test data
    nx = 8
    n_vars = 5
    Q_test = np.ones((nx, n_vars))
    Q_test[:, 0] = 1.0
    Q_test[:, 1] = 0.05
    Q_test[:, 2] = 250000
    Q_test[:, 3] = 80.0
    Q_test[:, 4] = 40.0
    
    dt = 1e-3  # Larger timestep to force stiff conditions
    physics_params = {
        'gamma': 1.4,
        'R_gas': 287.0,
        'tau_q': 1e-7,    # Very small tau to force stiffness
        'tau_sigma': 1e-7, # Very small tau to force stiffness
        'dx': 0.1
    }
    
    # Track function calls
    source_calls = 0
    hyperbolic_calls = 0
    
    def tracked_source_rhs(Q):
        nonlocal source_calls
        source_calls += 1
        source = np.zeros_like(Q)
        source[:, 3] = 2000.0
        source[:, 4] = 1000.0
        return source
    
    def tracked_hyperbolic_rhs(Q):
        nonlocal hyperbolic_calls
        hyperbolic_calls += 1
        return np.zeros_like(Q)
    
    print("Testing with corrected API (use_advanced_source_solver=False):")
    
    # Test corrected API
    adaptive_fixed = AdaptiveOperatorSplitting(use_advanced_source_solver=False)
    
    source_calls = 0
    hyperbolic_calls = 0
    
    Q_result, diagnostics = adaptive_fixed.adaptive_step(
        Q_test.copy(), dt, tracked_hyperbolic_rhs, tracked_source_rhs, physics_params
    )
    
    print(f"   Method used: {diagnostics['method_used']}")
    print(f"   Source RHS calls: {source_calls}")
    print(f"   Hyperbolic RHS calls: {hyperbolic_calls}")
    print(f"   Splitting required: {diagnostics['splitting_required']}")
    
    source_used = source_calls > 0
    print(f"   Provided source_rhs used: {'✅' if source_used else '❌'}")
    
    print(f"\nTesting backward compatibility (use_advanced_source_solver=True):")
    
    # Test backward compatibility
    adaptive_compat = AdaptiveOperatorSplitting(use_advanced_source_solver=True)
    
    source_calls = 0
    hyperbolic_calls = 0
    
    Q_result_compat, diagnostics_compat = adaptive_compat.adaptive_step(
        Q_test.copy(), dt, tracked_hyperbolic_rhs, tracked_source_rhs, physics_params
    )
    
    print(f"   Method used: {diagnostics_compat['method_used']}")
    print(f"   Source RHS calls: {source_calls}")
    print(f"   Hyperbolic RHS calls: {hyperbolic_calls}")
    print(f"   Splitting required: {diagnostics_compat['splitting_required']}")
    
    # In backward compatibility mode, source_rhs should be ignored when splitting is used
    if diagnostics_compat['splitting_required']:
        # When splitting is required, backward compat should ignore source_rhs
        source_properly_ignored = source_calls == 0
        print(f"   Source_rhs properly ignored (splitting mode): {'✅' if source_properly_ignored else '❌'}")
        compat_result = source_properly_ignored
    else:
        # When no splitting required, both modes use explicit method with source_rhs
        source_used_when_appropriate = source_calls > 0
        print(f"   Source_rhs used (explicit method): {'✅' if source_used_when_appropriate else '❌'}")
        compat_result = source_used_when_appropriate
    
    return source_used and compat_result

def main():
    """Run all operator splitting API fix tests."""
    print("🔧 OPERATOR SPLITTING API FIX VALIDATION")
    print("=" * 60)
    print("Testing the fix for misleading function arguments where")
    print("source_rhs was accepted but never used.")
    print("=" * 60)
    
    try:
        # Test basic functionality
        basic_test_passed = test_source_rhs_actually_used()
        
        # Test adaptive integration
        adaptive_test_passed = test_adaptive_splitting_integration()
        
        # Final assessment
        print(f"\n🏆 OPERATOR SPLITTING API FIX RESULTS:")
        print("=" * 50)
        
        if basic_test_passed:
            print("✅ Basic API fix: PASSED")
            print("   - Fixed version uses provided source_rhs function")
            print("   - Backward compatibility mode ignores source_rhs")
        else:
            print("❌ Basic API fix: FAILED")
        
        if adaptive_test_passed:
            print("✅ Adaptive integration: PASSED")
            print("   - Adaptive splitter respects use_advanced_source_solver flag")
            print("   - Both modes work correctly")
        else:
            print("❌ Adaptive integration: FAILED")
        
        overall_success = basic_test_passed and adaptive_test_passed
        
        print(f"\n🎯 API FIX STATUS:")
        if overall_success:
            print("✅ Misleading function arguments: FIXED")
            print("✅ Source RHS function now properly used")
            print("✅ Backward compatibility maintained")
            print("✅ Principle of least astonishment: RESTORED")
            assessment = "SUCCESS"
        else:
            print("❌ Some tests failed - fix needs investigation")
            assessment = "NEEDS_WORK"
        
        print(f"\n📈 FINAL ASSESSMENT: {assessment}")
        
        if overall_success:
            print(f"\n🎉 The misleading API has been fixed!")
            print("   Users can now trust that their source_rhs function will be used.")
        
    except Exception as e:
        print(f"❌ Error during API fix validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()