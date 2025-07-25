#!/usr/bin/env python3
"""
Test the suggested periodic BC fix to see if it's different.
"""

import numpy as np

def test_suggested_fix():
    """Test the suggested periodic BC fix."""
    print("🔍 Testing Suggested Periodic BC Fix")
    print("=" * 50)
    
    n_ghost = 2
    nx_phys = 4  # 4 physical cells
    n_vars = 1
    
    # Total array size: 2 (left ghost) + 4 (physical) + 2 (right ghost) = 8
    Q_ghost = np.zeros((nx_phys + 2*n_ghost, n_vars))
    
    # Initialize physical cells with distinctive values: [10, 20, 30, 40]
    phys_start = n_ghost  # Index 2
    for i in range(nx_phys):
        Q_ghost[phys_start + i, 0] = (i + 1) * 10
    
    print(f"Initial physical cells: {Q_ghost[phys_start:phys_start+nx_phys, 0]}")
    
    # Apply suggested fix
    phys_start = n_ghost
    phys_end_idx = Q_ghost.shape[0] - n_ghost - 1  # This is the LAST physical cell index
    nx_phys = phys_end_idx - phys_start + 1
    
    print(f"Suggested fix parameters:")
    print(f"  phys_start = {phys_start}")
    print(f"  phys_end_idx = {phys_end_idx} (last physical cell)")
    print(f"  nx_phys = {nx_phys}")
    
    # Left ghost cells get data from the RIGHT of the physical domain
    for g in range(n_ghost):
        src_idx = phys_end_idx - (n_ghost - 1) + g
        dest_idx = g
        print(f"  Left ghost {dest_idx} <- physical {src_idx} (value {Q_ghost[src_idx, 0]})")
        Q_ghost[dest_idx, :] = Q_ghost[src_idx, :]

    # Right ghost cells get data from the LEFT of the physical domain
    for g in range(n_ghost):
        src_idx = phys_start + g
        dest_idx = phys_end_idx + 1 + g
        print(f"  Right ghost {dest_idx} <- physical {src_idx} (value {Q_ghost[src_idx, 0]})")
        Q_ghost[dest_idx, :] = Q_ghost[src_idx, :]
    
    print(f"\nResult with suggested fix:")
    for i in range(Q_ghost.shape[0]):
        print(f"  Index {i}: {Q_ghost[i, 0]}")
    
    # Compare to expected
    print(f"\nExpected:")
    print(f"  Left ghost should be [30, 40] (rightmost physical)")
    print(f"  Right ghost should be [10, 20] (leftmost physical)")
    
    # Check
    left_correct = (Q_ghost[0, 0] == 30.0 and Q_ghost[1, 0] == 40.0)
    right_correct = (Q_ghost[6, 0] == 10.0 and Q_ghost[7, 0] == 20.0)  
    
    print(f"\nValidation:")
    print(f"  Left ghost correct: {'✅' if left_correct else '❌'}")
    print(f"  Right ghost correct: {'✅' if right_correct else '❌'}")

if __name__ == "__main__":
    test_suggested_fix()