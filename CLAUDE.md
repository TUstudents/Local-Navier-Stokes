# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational fluid dynamics research repository focused on the Local Navier-Stokes (LNS) equations - a novel approach to fluid dynamics that addresses fundamental limitations of the classical Navier-Stokes equations. The project is structured as a series of Jupyter notebooks that progressively develop and implement LNS theory and numerical methods.

## Repository Structure

The repository is organized into progressive series:

- `00_Motivation/` - Critiques of classical Navier-Stokes equations and motivation for LNS
  - `00a_NS_Critique.ipynb`, `00b_NS_Critique.ipynb`, `00c_NS_Critique.ipynb` - Theoretical critique series
  - `01_LNS_Solver_1D_Toy model.ipynb` - Initial LNS implementation concepts
- `01_Foundation.md` - Core theoretical foundation and critique of N-S equations  
- `01_LNS/` - Core 1D LNS implementations with enhanced physics ✅ **FULLY FUNCTIONAL**
  - `01_LNS_Solver_1D_EnhancedPhysics.ipynb` - Complete 5-variable LNS system with HLL flux
  - `02_LNS_Transonic_Analysis.ipynb` - Transonic flow analysis and parameter studies
- `03_LNS_Series2/` - Advanced numerical methods and 3D implementations
  - `LNS_Series2_NB1_AdvancedNumerics.ipynb` - Higher-order schemes
  - `LNS_Series2_NB2_3D_Implementation.ipynb` - 3D tensor implementations
  - `LNS_Series2_NB3_Transition.ipynb` - Transition to turbulence
  - `LNS_Series2_NB4_Developed_Turbulence.ipynb` - Fully developed turbulence
- `04_LNS_for_Complex_Fluids/` - LNS applications to non-Newtonian and complex fluids
  - `outline.md` - Research plan for complex fluids
- `05_From_LNS_to Einstein's_Universe/` - Relativistic fluid dynamics applications
  - Three notebooks covering relativistic extensions of LNS theory

## Development Environment

This is a Python-based project using modern dependency management:

### Python Environment & Dependencies
- **Python**: Requires >=3.12
- **Package Manager**: Uses `uv` for fast, reliable dependency management
- **Project**: `local-navier-stokes` v0.1.0
- **Dependencies** (managed via `pyproject.toml`):
  - `numpy>=1.24.0` - Numerical computations and array operations
  - `matplotlib>=3.6.0` - Plotting and visualization
  - `jupyter>=1.0.0` - Interactive notebook environment
  - `scipy>=1.10.0` - Scientific computing functions

### Virtual Environment Setup
The project uses a local virtual environment (`.venv/`) managed by `uv`:

```bash
# Using uv (recommended) - creates and syncs .venv automatically
uv sync

# Activate the virtual environment manually if needed
source .venv/bin/activate

# Verify correct Python version (should be 3.12.9)
python --version
```

**Environment Details:**
- **Virtual Environment**: `.venv/` directory (created by uv)
- **Python Version**: 3.12.9 (in .venv), system has 3.9.18 via miniconda3
- **Package Manager**: uv 0.6.8
- **Activation**: Use `source .venv/bin/activate` or uv handles activation automatically

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

### Notebook Status and Quality
**✅ Ready-to-Use Implementations:**
- **01_LNS_Solver_1D_EnhancedPhysics.ipynb**: Complete 5-variable LNS system with HLL flux, SSP-RK2 time stepping, and comprehensive test cases
- **02_LNS_Transonic_Analysis.ipynb**: Transonic analysis with parameter sweeps, proper theoretical framework, and visualization

**🔧 Development Series:**
- **00_Motivation/** series: Theoretical foundation and critique (documentation-focused)
- **03_LNS_Series2/** series: Advanced numerical methods (various completion levels)
- **05_Relativistic/** series: Relativistic extensions (conceptual)

### Jupyter Notebook JSON Formatting Standards ✅

**CRITICAL: Proper JSON Structure and Escaping**
Jupyter notebooks are JSON files with specific formatting requirements:

**LaTeX Formula Requirements:**
- **Raw Strings MANDATORY**: All Python strings containing LaTeX MUST use `r''` or `rf''` format
- **Escape Sequence Handling**: Avoid `\'` sequences in LaTeX - use `\prime` or simplified notation
- **No Unicode Math**: Use ASCII + LaTeX syntax instead of Unicode symbols (√, ∂, ∫, etc.)
- **JSON Escaping**: In notebook cells, backslashes must be properly escaped for JSON compliance

**Mathematical Expression Standards:**
```python
# ✅ CORRECT - Raw strings with LaTeX
title = r'1D LNS: $\rho$, $u_x$, $p$, $T$'
ylabel = rf'$\sigma_{xx}$ (Pa) - $\tau_\sigma = {TAU_SIGMA:.1e}$'
equation = r'$$\frac{\partial \rho}{\partial t} + \frac{\partial m_x}{\partial x} = 0$$'

# ❌ INCORRECT - Will cause JSON parsing errors
title = '1D LNS: $\\rho$, $u_x$, $p$, $T$'  # Double escaping issues
ylabel = f'$\sigma\'_{xx}$ (Pa)'              # Prime notation issues
equation = '$$\frac{\partial \rho}{\partial t} = 0$$'  # Missing raw string
```

**Notebook Cell Structure Requirements:**
- **Markdown cells**: Use proper JSON escaping for LaTeX content
- **Code cells**: All string literals with math must use raw strings
- **Cell metadata**: Maintain proper JSON structure for Jupyter compatibility
- **Output cells**: Ensure matplotlib renders LaTeX correctly without parser conflicts

**Critical Escaping Rules:**
1. **Python code in cells**: Always use `r''` for LaTeX strings
2. **Markdown cells**: LaTeX works directly, but avoid complex escape sequences
3. **f-strings with LaTeX**: Use `rf''` format consistently
4. **Variable names**: Avoid prime symbols in subscripts (`\sigma'_{xx}` → `\sigma_{xx}`)
5. **Complex fractions**: Prefer `\frac{a}{b}` over `a/b` for clarity

### Notebook Organization
Each notebook is self-contained but builds on previous concepts:
1. Start with `01_Foundation.md` for theoretical understanding
2. Use `01_LNS/01_LNS_Solver_1D_EnhancedPhysics.ipynb` as the reference implementation
3. Study `01_LNS/02_LNS_Transonic_Analysis.ipynb` for parameter studies and advanced analysis
4. Progress through series in numerical order for increasing complexity

### Common Development Tasks
- **Parameter Studies**: Modify relaxation parameters (`TAU_Q`, `TAU_SIGMA`) to study different physical regimes
- **Test Cases**: Implement new initial condition functions for different flow scenarios
- **Solver Extensions**: Add new source terms or flux computations to the modular framework
- **Visualization**: Use provided plotting functions with proper LaTeX formatting
- **Numerical Methods**: Extend HLL flux to HLLC or implement higher-order schemes

### Core Functions and Their Roles

#### Variable Conversion Functions
- `Q_to_P_1D_enh(Q_vec)`: Convert conserved variables [ρ,mx,ET,qx,σ'xx] to primitives [ρ,ux,p,T]
- `P_and_fluxes_to_Q_1D_enh(rho,ux,p,T,qx,sxx)`: Convert primitives + fluxes to conserved variables

#### Physics and Numerical Methods
- `flux_1D_LNS_enh(Q_vec)`: Compute LNS flux vector F(Q) for all 5 variables
- `source_1D_LNS_enh(Q_cell,Q_L,Q_R,dx,tau_q,tau_sigma)`: Compute relaxation source terms
- `hll_flux_1D_LNS_enh(Q_L,Q_R)`: HLL Riemann solver for interface fluxes
- `solve_1D_LNS_FVM_enh()`: Main finite volume solver with configurable time stepping

#### Solver Capabilities
- **Boundary Conditions**: Periodic, outflow (easily extensible)
- **Time Integration**: Forward Euler, SSP-RK2 (2nd-order accurate)
- **Flux Methods**: HLL (characteristic-based), Lax-Friedrichs fallback
- **Parameter Control**: Independent τ_q and τ_σ for thermal and viscous relaxation

## Architecture Notes

### Theoretical Foundation
The project critiques classical fluid dynamics approaches and develops alternatives based on:
- **Finite speed of information propagation** (vs infinite in classical N-S incompressible limit)
- **Memory effects** through dynamic relaxation equations for heat flux and stress
- **Proper compressibility treatment** avoiding unphysical infinite sound speed
- **Local physics** ensuring causality and finite response times

### Numerical Architecture ✅
**Current Implementation Status:**
- ✅ **Conservative finite volume methods**: Preserve physical conservation laws exactly
- ✅ **Characteristic-based schemes**: HLL respects wave propagation physics  
- ✅ **Relaxation source terms**: Proper handling of τ_q and τ_σ physics
- ✅ **Modular design**: Easy modification of physics models and numerical methods
- ✅ **Error handling**: Robust bounds checking and stability monitoring
- ✅ **Multiple test cases**: Sod shock tube, parameter sweeps, validation problems

### Code Quality and Reliability ✅
**Recently Completed Improvements:**
- ✅ **Syntax standardization**: All LaTeX expressions use proper raw string formatting
- ✅ **Function completeness**: All critical functions defined and working
- ✅ **Cross-platform compatibility**: Mathematical notation renders consistently
- ✅ **Error elimination**: No remaining syntax errors or undefined variables
- ✅ **Documentation standards**: Clear parameter descriptions and usage examples
- ✅ **LaTeX parsing fixes**: Resolved matplotlib math parser issues with simplified notation
- ✅ **Variable scope issues**: Fixed undefined variables in solver functions
- ✅ **Plot rendering**: Eliminated string formatting conflicts in visualization code

## Development Guidelines

### For New Implementations
1. **Start with 1D**: Use `01_LNS_Solver_1D_EnhancedPhysics.ipynb` as template
2. **Follow patterns**: Maintain the Q_to_P conversion structure and source term organization
3. **MANDATORY: Raw strings for LaTeX**: All mathematical expressions MUST use `r''` or `rf''` format
4. **JSON compliance**: Test notebook cells for proper JSON structure and escaping
5. **Modular design**: Keep physics separate from numerics for easy extension
6. **Parameter validation**: Include bounds checking for relaxation times
7. **Mathematical notation**: Use simplified LaTeX without prime symbols in subscripts

### For Testing and Validation
- **Sod shock tube**: Standard test case implemented and working
- **Parameter sweeps**: Test multiple τ regimes from NSF limit to highly elastic
- **Visualization**: Comprehensive plotting with proper mathematical notation
- **Stability monitoring**: Built-in NaN detection and stability warnings

## Future Extensions

The codebase is designed for extension to:
- **Higher-order schemes**: MUSCL with limiters, WENO reconstruction
- **Advanced Riemann solvers**: HLLC, Roe solver with entropy fix
- **2D/3D tensor implementations**: Full objective derivative treatments
- **Complex fluids**: Viscoelastic models (Oldroyd-B, Giesekus, FENE-P)
- **Relativistic extensions**: Israel-Stewart theory and beyond
- **Turbulence simulations**: Direct numerical simulation with LNS effects

## Current Limitations and Opportunities

### Known Limitations
- **Stiff relaxation**: Small τ values may require implicit/semi-implicit methods
- **Gradient computation**: Source terms use simple finite differences (could be improved)
- **Wave speed estimates**: HLL uses simplified acoustic speeds (full Jacobian analysis needed)
- **1D simplification**: Objective derivatives greatly simplified compared to 3D tensor algebra

### Immediate Opportunities
- **Operator splitting**: IMEX schemes for stiff source terms
- **Higher-order accuracy**: Implement MUSCL reconstruction for flux interfaces
- **Advanced test cases**: Kelvin-Helmholtz instability, Taylor-Couette flow
- **Performance optimization**: Vectorization and computational efficiency improvements

## Support and Maintenance

### Environment Requirements
- **Python ≥3.12**: Required for modern syntax and performance
- **uv package manager**: Fast, reliable dependency resolution
- **Jupyter environment**: Interactive development and research presentation

### Installation and Setup
```bash
# Clone repository
git clone <repository-url>
cd Local-Navier-Stokes

# Install dependencies and create virtual environment (recommended)
uv sync

# Activate virtual environment (if needed manually)
source .venv/bin/activate

# Verify setup
python --version  # Should show 3.12.9
uv --version      # Should show 0.6.8

# Launch Jupyter (from activated environment)
jupyter lab
```

**Important Notes:**
- The project requires Python >=3.12 but system Python is 3.9.18 (via miniconda3)
- The `.venv/` directory contains Python 3.12.9 and all required dependencies
- Always ensure you're using the virtual environment when running notebooks
- Use `which python` to verify you're using `.venv/bin/python` (not system Python)

### Troubleshooting

**JSON and LaTeX Formatting Issues:**
- **CRITICAL: Raw string requirement**: ALL Python strings with LaTeX MUST use `r''` or `rf''` format
- **JSON parsing errors**: Caused by improper escaping in notebook cells - check for `\'` sequences
- **Matplotlib LaTeX errors**: Use `\prime` instead of `\'` for prime notation, avoid `\sigma\'_{xx}`
- **Double escaping problems**: Don't use `\\` in raw strings - use single `\` with raw string prefix
- **f-string LaTeX mixing**: Always use `rf'text with {variable} and $\LaTeX$'` format consistently

**Development Environment Issues:**
- **Python version conflicts**: Ensure using `.venv/bin/python` (3.12.9) not system Python (3.9.18)
- **Import errors**: Verify all dependencies installed via `uv sync` and virtual environment activated
- **Virtual environment**: Use `source .venv/bin/activate` and verify with `which python`

**Numerical Solver Issues:**
- **Stability problems**: Check relaxation time values and CFL constraints
- **Performance issues**: Consider smaller time steps or implicit methods for small τ
- **Variable scope errors**: Ensure all variables (especially `x_coords`) are defined before use in solver functions

**Mathematical Notation Standards:**
- **Prime symbols**: Use `\sigma_{xx}` instead of `\sigma'_{xx}` for subscripts
- **Unicode symbols**: Replace √, ∂, ∫ with LaTeX equivalents (`\sqrt{}`, `\partial`, `\int`)
- **Complex expressions**: Break into multiple lines for readability and JSON compliance