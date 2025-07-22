# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational fluid dynamics research repository focused on the Local Navier-Stokes (LNS) equations - a novel approach to fluid dynamics that addresses fundamental limitations of the classical Navier-Stokes equations. The project is structured as a series of Jupyter notebooks that progressively develop and implement LNS theory and numerical methods.

## Repository Structure

The repository is organized into progressive series:

- `00_Motivation/` - Critiques of classical Navier-Stokes equations and motivation for LNS
- `01_Foundation.md` - Core theoretical foundation and critique of N-S equations
- `01_LNS/` - Basic 1D LNS implementations with enhanced physics
- `03_LNS_Series2/` - Advanced numerical methods and 3D implementations
- `04_LNS_for_Complex_Fluids/` - LNS applications to non-Newtonian and complex fluids
- `05_From_LNS_to Einstein's_Universe/` - Relativistic fluid dynamics applications

## Development Environment

This is a Python-based project with the following configuration:

### Python Environment
- Requires Python >=3.12
- Project name: `local-navier-stokes`
- Version: 0.1.0
- Currently has minimal dependencies (empty dependencies list in pyproject.toml)

### Key Libraries Used in Notebooks
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- Jupyter notebooks for interactive development and presentation

## Core Concepts

### Local Navier-Stokes (LNS) Theory
The LNS equations address fundamental flaws in classical Navier-Stokes:

1. **Incompressibility Constraint Issues**: Classical N-S incompressibility implies infinite speed of sound, which is unphysical
2. **Finite Relaxation Times**: LNS introduces relaxation times (`τ_q` for heat flux, `τ_σ` for stress) to model realistic physical behavior
3. **Local vs Non-local Physics**: LNS ensures locality of physical interactions

### Key Parameters
- `TAU_Q`: Thermal relaxation time (typical: 1e-5 s)
- `TAU_SIGMA`: Stress relaxation time (typical: 1e-5 s) 
- `MU_VISC`: Dynamic viscosity
- `K_THERM`: Thermal conductivity

## Implementation Details

### State Vector Structure (1D Enhanced)
The 1D LNS system uses a 5-variable state vector:
```
Q = [ρ, m_x, E_T, q_x, σ'_xx]
```
Where:
- ρ: density
- m_x: momentum in x-direction  
- E_T: total energy
- q_x: heat flux in x-direction
- σ'_xx: deviatoric stress component

### Numerical Methods
- Finite Volume Method (FVM) with cell-centered approach
- HLL (Harten-Lax-van Leer) flux for improved accuracy over Lax-Friedrichs
- SSP-RK2 time integration for stability
- Source terms handle relaxation physics

## Working with the Code

### Notebook Organization
Each notebook is self-contained but builds on previous concepts:
1. Start with motivation notebooks to understand the theoretical foundation
2. Progress through 1D implementations before attempting 2D/3D
3. Complex fluids applications require understanding of basic LNS

### Common Development Tasks
- Modify relaxation parameters (`TAU_Q`, `TAU_SIGMA`) to study different regimes
- Implement new test cases by defining initial condition functions
- Extend solvers by adding new source terms or flux computations
- Visualize results using the provided plotting functions

### Key Functions and Classes
- `Q_to_P_1D_enh()`: Convert conserved to primitive variables
- `flux_1D_LNS_enh()`: Compute flux vectors
- `source_1D_LNS_enh()`: Compute source terms including relaxation
- `hll_flux_1D_LNS_enh()`: HLL numerical flux implementation
- `solve_1D_LNS_FVM_enh()`: Main FVM solver loop

## Architecture Notes

### Theoretical Foundation
The project critiques classical fluid dynamics approaches and develops alternatives based on:
- Finite speed of information propagation (vs infinite in classical N-S)
- Memory effects through relaxation equations
- Proper treatment of compressibility effects

### Numerical Approach
- Conservative finite volume methods preserve physical conservation laws
- Characteristic-based schemes (HLL) respect wave propagation physics
- Stiff source term handling for small relaxation times
- Modular design allows easy modification of physics models

## Future Extensions

The codebase is designed for extension to:
- Higher-order numerical schemes (MUSCL, WENO)
- 2D/3D implementations with full tensor algebra
- Complex fluids (viscoelastic, multi-phase)
- Relativistic fluid dynamics
- Turbulence simulations with LNS effects

## Important Notes

- This is research-grade code focused on exploring new physics models
- Numerical stability may require careful parameter tuning
- Small relaxation times can create stiff systems requiring implicit methods
- The 1D implementations are simplified versions of full 3D tensor equations