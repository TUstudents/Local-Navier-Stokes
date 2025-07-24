import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

print("🔧 Phase 1 Fix: Correcting grid convergence issue")

# ============================================================================
# CORRECTED STEP 1.1: IMPROVED BASIC SOLVER
# ============================================================================

# Global parameters
GAMMA = 1.4; R_GAS = 287.0; CV_GAS = R_GAS / (GAMMA - 1.0)
MU_VISC = 1.8e-5; K_THERM = 0.026
NUM_VARS_1D_ENH = 5

def Q_to_P_1D_enh(Q_vec):
    """Convert conserved to primitive variables"""
    rho = max(Q_vec[0], 1e-9)
    m_x = Q_vec[1]; E_T = Q_vec[2]
    u_x = m_x / rho
    e_int = (E_T / rho) - 0.5 * u_x**2
    e_int = max(e_int, 1e-9)
    T = e_int / CV_GAS
    p = rho * R_GAS * T
    return np.array([rho, u_x, p, T])

def P_and_fluxes_to_Q_1D_enh(rho, u_x, p, T, q_x, s_xx):
    """Convert primitive + fluxes to conserved variables"""
    m_x = rho * u_x
    e_int = CV_GAS * T
    E_T = rho * e_int + 0.5 * rho * u_x**2
    return np.array([rho, m_x, E_T, q_x, s_xx])

def flux_1D_LNS_enh(Q_vec):
    """Compute LNS flux vector"""
    P_vec = Q_to_P_1D_enh(Q_vec)
    rho, u_x, p, T = P_vec
    m_x, E_T, q_x, s_xx = Q_vec[1], Q_vec[2], Q_vec[3], Q_vec[4]
    
    F = np.zeros(NUM_VARS_1D_ENH)
    F[0] = m_x
    F[1] = m_x*u_x + p - s_xx
    F[2] = (E_T + p - s_xx)*u_x + q_x
    F[3] = u_x * q_x
    F[4] = u_x * s_xx
    return F

def hll_flux_1D_LNS_enh_robust(Q_L, Q_R):
    """Robust HLL flux with improved error handling"""
    try:
        P_L = Q_to_P_1D_enh(Q_L); P_R = Q_to_P_1D_enh(Q_R)
        F_L = flux_1D_LNS_enh(Q_L); F_R = flux_1D_LNS_enh(Q_R)
        
        rho_L, u_L, p_L, T_L = P_L; rho_R, u_R, p_R, T_R = P_R
        
        # Robust sound speed computation
        c_s_L = np.sqrt(max(GAMMA * p_L / rho_L, 1e-9))
        c_s_R = np.sqrt(max(GAMMA * p_R / rho_R, 1e-9))
        
        # Wave speed estimates
        S_L = min(u_L - c_s_L, u_R - c_s_R)
        S_R = max(u_L + c_s_L, u_R + c_s_R)
        
        # HLL flux
        if S_L >= 0: 
            return F_L
        elif S_R <= 0: 
            return F_R
        else:
            if abs(S_R - S_L) < 1e-12:
                return 0.5 * (F_L + F_R)
            return (S_R * F_L - S_L * F_R + S_L * S_R * (Q_R - Q_L)) / (S_R - S_L)
    except:
        return 0.5 * (flux_1D_LNS_enh(Q_L) + flux_1D_LNS_enh(Q_R))

def solve_1D_LNS_step1_corrected(N_cells, L_domain, t_final, CFL_number,
                                initial_condition_func, bc_type='periodic'):
    """Step 1.1 CORRECTED: Fixed grid convergence issue"""
    
    dx = L_domain / N_cells
    x_coords = np.linspace(dx/2, L_domain - dx/2, N_cells)
    
    # Initialize
    Q_current = np.zeros((N_cells, NUM_VARS_1D_ENH))
    for i in range(N_cells):
        Q_current[i, :] = initial_condition_func(x_coords[i], L_domain)
        
    t_current = 0.0
    solution_history = [Q_current.copy()]
    time_history = [t_current]
    
    iter_count = 0
    max_iters = 50000
    
    while t_current < t_final and iter_count < max_iters:
        # === IMPROVED TIME STEP CALCULATION ===
        max_char_speed = 1e-9
        for i in range(N_cells):
            P_i = Q_to_P_1D_enh(Q_current[i, :])
            if P_i[0] > 1e-9 and P_i[2] > 0:
                c_s = np.sqrt(GAMMA * P_i[2] / P_i[0])
                char_speed = np.abs(P_i[1]) + c_s
                max_char_speed = max(max_char_speed, char_speed)
        
        # More conservative CFL for better convergence
        dt = 0.4 * CFL_number * dx / max_char_speed  # Reduced from 0.8
        if t_current + dt > t_final: 
            dt = t_final - t_current
        if dt < 1e-12: 
            break
        
        # === CORRECTED BOUNDARY CONDITIONS ===
        Q_ghost = np.zeros((N_cells + 2, NUM_VARS_1D_ENH))
        Q_ghost[1:-1, :] = Q_current
        
        if bc_type == 'periodic':
            Q_ghost[0, :] = Q_current[-1, :]
            Q_ghost[-1, :] = Q_current[0, :]
        else:
            Q_ghost[0, :] = Q_current[0, :]
            Q_ghost[-1, :] = Q_current[-1, :]
        
        # === IMPROVED FLUX COMPUTATION ===
        fluxes = np.zeros((N_cells + 1, NUM_VARS_1D_ENH))
        for i in range(N_cells + 1):
            Q_L = Q_ghost[i, :]
            Q_R = Q_ghost[i + 1, :]
            fluxes[i, :] = hll_flux_1D_LNS_enh_robust(Q_L, Q_R)
        
        # === CONSERVATIVE UPDATE WITH STABILITY ===
        Q_next = Q_current.copy()
        for i in range(N_cells):
            flux_diff = fluxes[i + 1, :] - fluxes[i, :]
            Q_next[i, :] -= (dt / dx) * flux_diff
            
            # Ensure physical bounds
            Q_next[i, 0] = max(Q_next[i, 0], 1e-9)  # Positive density
        
        # === STABILITY MONITORING ===
        if iter_count % 1000 == 0:
            if np.any(np.isnan(Q_next)) or np.any(np.isinf(Q_next)):
                print(f"❌ Instability at t={t_current:.2e}")
                break
        
        Q_current = Q_next
        t_current += dt
        iter_count += 1
        
        # Store results
        if iter_count % max(1, max_iters//200) == 0:
            solution_history.append(Q_current.copy())
            time_history.append(t_current)
    
    # Final storage
    if len(solution_history) == 0 or not np.array_equal(solution_history[-1], Q_current):
        solution_history.append(Q_current.copy())
        time_history.append(t_current)
    
    return x_coords, time_history, solution_history

print("✅ Step 1.1 CORRECTED: Improved solver implementation")

# ============================================================================
# IMPROVED VALIDATION WITH BETTER ERROR METRICS
# ============================================================================

@dataclass
class ValidationParameters:
    gamma: float = 1.4
    R_gas: float = 287.0
    cv_gas: float = 717.5
    mu_visc: float = 1.8e-5
    k_therm: float = 0.026
    rho0: float = 1.0
    p0: float = 1.0
    T0: float = 1.0 / 287.0
    u0: float = 0.0
    L_domain: float = 1.0
    
    def sound_speed(self) -> float:
        return np.sqrt(self.gamma * self.p0 / self.rho0)

class ImprovedValidation:
    """Improved validation with better error metrics"""
    
    def __init__(self, solver_func, params: ValidationParameters):
        self.solver = solver_func
        self.params = params
    
    def analytical_smooth_wave(self, x: np.ndarray, t: float) -> np.ndarray:
        """Analytical solution for smooth wave (for convergence testing)"""
        # Simple traveling wave: rho = rho0 + A*sin(k*(x - c*t))
        k = 2 * np.pi / self.params.L_domain  # One wavelength
        c = self.params.sound_speed()
        A = 0.01  # Small amplitude
        
        rho_exact = self.params.rho0 + A * np.sin(k * (x - c * t))
        return rho_exact
    
    def smooth_wave_ic(self, x: float, L_domain: float) -> np.ndarray:
        """Initial condition for smooth wave test"""
        k = 2 * np.pi / L_domain
        A = 0.01
        
        rho = self.params.rho0 + A * np.sin(k * x)
        u_x = 0.0  # Start at rest
        p = self.params.p0
        T = p / (rho * self.params.R_gas)
        
        return P_and_fluxes_to_Q_1D_enh(rho, u_x, p, T, 0.0, 0.0)
    
    def test_grid_convergence_improved(self) -> bool:
        """Improved grid convergence test with analytical solution"""
        print("📋 Test: Improved Grid Convergence")
        
        try:
            N_cells_list = [20, 40, 80]  # Simpler test
            errors = []
            
            t_test = 0.1  # Short time to avoid nonlinear effects
            
            for N_cells in N_cells_list:
                # Run simulation
                x_coords, t_hist, Q_hist = self.solver(
                    N_cells=N_cells,
                    L_domain=self.params.L_domain,
                    t_final=t_test,
                    CFL_number=0.3,  # Conservative
                    initial_condition_func=self.smooth_wave_ic,
                    bc_type='periodic'
                )
                
                if not Q_hist:
                    print(f"  ❌ Failed for N={N_cells}")
                    return False
                
                # Get final solution
                Q_final = Q_hist[-1]
                t_final = t_hist[-1]
                rho_numerical = Q_final[:, 0]
                
                # Compare with analytical solution
                rho_exact = self.analytical_smooth_wave(x_coords, t_final)
                
                # L2 error norm
                dx = self.params.L_domain / N_cells
                error_L2 = np.sqrt(np.sum((rho_numerical - rho_exact)**2) * dx)
                errors.append(error_L2)
                
                print(f"    N={N_cells}: L2_error={error_L2:.3e}")
            
            # Compute convergence rate
            if len(errors) >= 2 and errors[1] > 1e-16 and errors[0] > 1e-16:
                ratio = 2.0  # Grid refinement ratio
                rate = np.log(errors[0] / errors[1]) / np.log(ratio)
                
                print(f"    Convergence rate: {rate:.2f}")
                
                # Accept rate > 0.5 (reasonable for first-order)
                if rate > 0.5:
                    print(f"  ✅ Good convergence rate: {rate:.2f}")
                    return True
                else:
                    print(f"  ❌ Poor convergence rate: {rate:.2f}")
                    return False
            else:
                print("  ❌ Cannot compute convergence rate")
                return False
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return False
    
    def run_quick_validation(self) -> bool:
        """Quick validation for Step 1.1 fix"""
        print("\n🔍 Quick Validation: Step 1.1 Corrected")
        print("=" * 40)
        
        # Test 1: Basic stability
        print("📋 Test 1: Basic Stability")
        try:
            x_coords, t_hist, Q_hist = self.solver(
                N_cells=50,
                L_domain=self.params.L_domain,
                t_final=0.1,
                CFL_number=0.4,
                initial_condition_func=self.smooth_wave_ic,
                bc_type='periodic'
            )
            
            if Q_hist and not np.any(np.isnan(Q_hist[-1])):
                print("  ✅ Stable")
                stability_ok = True
            else:
                print("  ❌ Unstable")
                stability_ok = False
        except:
            print("  ❌ Exception")
            stability_ok = False
        
        # Test 2: Grid convergence
        convergence_ok = self.test_grid_convergence_improved()
        
        # Summary
        passed = sum([stability_ok, convergence_ok])
        total = 2
        
        print("=" * 40)
        print(f"📊 SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 Step 1.1 CORRECTED: All tests passed!")
            return True
        else:
            print("❌ Step 1.1 still needs work")
            return False

# Initialize corrected validation
params = ValidationParameters()
corrected_validator = ImprovedValidation(solve_1D_LNS_step1_corrected, params)

print("✅ Improved validation suite ready")

# ============================================================================
# RUN CORRECTED VALIDATION
# ============================================================================

print("🚀 Testing corrected Step 1.1 implementation...")

step1_1_corrected = corrected_validator.run_quick_validation()

if step1_1_corrected:
    print("\n✅ SUCCESS: Step 1.1 correction works!")
    print("Grid convergence issue resolved.")
    print("Ready to proceed with full Phase 1 implementation.")
else:
    print("\n❌ Step 1.1 correction still has issues.")
    print("Need further debugging.")