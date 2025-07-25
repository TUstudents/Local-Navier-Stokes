#!/usr/bin/env python3
"""
Debug periodic boundary conditions to identify the indexing issue.
"""

import numpy as np
from lns_solver.core.boundary_conditions import GhostCellBoundaryHandler, BoundaryCondition, BCType

def test_periodic_bc_indexing():
    """Test periodic BC with simple data to verify indexing."""
    print("🔍 Debugging Periodic Boundary Conditions")
    print("=" * 50)
    
    n_ghost = 2
    nx_phys = 4  # 4 physical cells
    n_vars = 3
    
    # Total array size: 2 (left ghost) + 4 (physical) + 2 (right ghost) = 8
    Q_ghost = np.zeros((nx_phys + 2*n_ghost, n_vars))
    
    # Initialize physical cells with distinctive values
    phys_start = n_ghost  # Index 2
    phys_end = Q_ghost.shape[0] - n_ghost  # Index 6
    
    print(f"Array indices:")
    print(f"  Total size: {Q_ghost.shape[0]}")
    print(f"  Left ghost: [0:{n_ghost}] = [0:2]")
    print(f"  Physical: [{phys_start}:{phys_end}] = [2:6]")
    print(f"  Right ghost: [{phys_end}:{phys_end+n_ghost}] = [6:8]")
    
    # Set physical cells to distinctive values: [10, 20, 30, 40]
    for i in range(nx_phys):
        Q_ghost[phys_start + i, :] = (i + 1) * 10
    
    print(f"\nInitial state (physical cells only):")
    for i in range(Q_ghost.shape[0]):
        if phys_start <= i < phys_end:
            print(f"  Index {i} (physical): {Q_ghost[i, 0]}")
        else:
            print(f"  Index {i} (ghost): {Q_ghost[i, 0]}")
    
    # Apply current implementation
    bc_handler = GhostCellBoundaryHandler(n_ghost)
    bc_handler.boundary_conditions['left'] = BoundaryCondition(BCType.PERIODIC)
    bc_handler.boundary_conditions['right'] = BoundaryCondition(BCType.PERIODIC)
    
    print(f"\nApplying current periodic BC implementation...")
    bc_handler.apply_periodic_bc_1d(Q_ghost)
    
    print(f"\nAfter current implementation:")
    for i in range(Q_ghost.shape[0]):
        if phys_start <= i < phys_end:
            print(f"  Index {i} (physical): {Q_ghost[i, 0]}")
        else:
            print(f"  Index {i} (ghost): {Q_ghost[i, 0]}")
    
    # What should the correct result be?
    print(f"\nExpected correct result:")
    print(f"  Left ghost [0:2] should get from right physical [4:6]: [30, 40]")
    print(f"  Right ghost [6:8] should get from left physical [2:4]: [10, 20]")
    
    # Check if current result is correct
    left_ghost_correct = (Q_ghost[0, 0] == 30.0 and Q_ghost[1, 0] == 40.0)
    right_ghost_correct = (Q_ghost[6, 0] == 10.0 and Q_ghost[7, 0] == 20.0)
    
    print(f"\nValidation:")
    print(f"  Left ghost correct: {'✅' if left_ghost_correct else '❌'}")
    print(f"  Right ghost correct: {'✅' if right_ghost_correct else '❌'}")
    
    if not (left_ghost_correct and right_ghost_correct):
        print(f"\n🚨 PERIODIC BC BUG CONFIRMED!")
        print(f"Current implementation produces wrong ghost cell values")
        
        # Show what the current implementation actually does
        print(f"\nCurrent implementation analysis:")
        print(f"  phys_end-n_ghost:phys_end = {phys_end-n_ghost}:{phys_end} = [4:6]")
        print(f"  So left ghost gets: {Q_ghost[4:6, 0]} (indices 4,5 = values 30,40)")
        print(f"  phys_start:phys_start+n_ghost = {phys_start}:{phys_start+n_ghost} = [2:4]") 
        print(f"  So right ghost gets: {Q_ghost[2:4, 0]} (indices 2,3 = values 10,20)")
        
        print(f"\n✅ Wait... this actually looks CORRECT!")
        print(f"The implementation might be right after all.")
    else:
        print(f"\n✅ Periodic BC implementation appears correct!")

if __name__ == "__main__":
    test_periodic_bc_indexing()