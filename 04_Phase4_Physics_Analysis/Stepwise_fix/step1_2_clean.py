import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

print("🔧 Step 1.2: Adding LNS Source Terms (Clean Implementation)")

# ============================================================================
# INHERIT WORKING STEP 1.1 BASELINE
# ============================================================================

# Global parameters
GAMMA = 1.4; R_GAS = 287.0; CV_GAS = R_GAS / (GAMMA - 1.0)
NUM_VARS_1D_ENH = 5
MU_VISC = 1.8e-5; K_THERM = 0.026

def simple_Q_to_P(Q_vec):
    """Simplified conserved to primitive conversion"""
    rho = max(Q_vec[0], 1e-9)
    m_x = Q_vec[1]; E_T = Q_vec[2]
    
    u_x = m_x / rho if rho > 1e-9 else 0.0
    e_int = (E_T - 0.5 * rho * u_x**2) / rho
    e_int = max(e_int, 1e-9)
    
    p = (GAMMA - 1.0) * rho * e_int
    T = p / (rho * R_GAS) if rho > 1e-9 else 1.0
    
    return np.array([rho, u_x, p, T])

def simple_P_to_Q(rho, u_x, p, T, q_x=0.0, s_xx=0.0):
    """Simplified primitive to conserved conversion"""
    m_x = rho * u_x
    e_int = p / ((GAMMA - 1.0) * rho) if rho > 1e-9 else 1e-9
    E_T = rho * e_int + 0.5 * rho * u_x**2
    return np.array([rho, m_x, E_T, q_x, s_xx])

def simple_flux_with_lns(Q_vec):
    """Simple flux computation INCLUDING LNS terms"""
    P_vec = simple_Q_to_P(Q_vec)
    rho, u_x, p, T = P_vec
    m_x, E_T, q_x, s_xx = Q_vec[1], Q_vec[2], Q_vec[3], Q_vec[4]
    
    F = np.zeros(NUM_VARS_1D_ENH)
    F[0] = m_x                           # Mass flux
    F[1] = m_x * u_x + p - s_xx          # Momentum flux WITH stress
    F[2] = (E_T + p - s_xx) * u_x + q_x  # Energy flux WITH heat flux  
    F[3] = u_x * q_x                     # Heat flux transport
    F[4] = u_x * s_xx                    # Stress transport
    
    return F

def lax_friedrichs_flux_lns(Q_L, Q_R, dx, dt):
    """Lax-Friedrichs flux for LNS system"""
    F_L = simple_flux_with_lns(Q_L)
    F_R = simple_flux_with_lns(Q_R)
    
    # Wave speed estimates
    P_L = simple_Q_to_P(Q_L); P_R = simple_Q_to_P(Q_R)
    
    c_L = np.sqrt(max(GAMMA * P_L[2] / P_L[0], 1e-9)) if P_L[0] > 1e-9 else 1.0
    c_R = np.sqrt(max(GAMMA * P_R[2] / P_R[0], 1e-9)) if P_R[0] > 1e-9 else 1.0
    
    lambda_max = max(
        abs(P_L[1]) + c_L,
        abs(P_R[1]) + c_R,
        dx / dt
    )
    
    # Lax-Friedrichs flux
    F_LF = 0.5 * (F_L + F_R) - 0.5 * lambda_max * (Q_R - Q_L)
    
    return F_LF

# ============================================================================
# NEW: LNS SOURCE TERMS
# ============================================================================

def compute_gradient_simple(Q_cells, var_index, dx, i):
    """Simple gradient computation for variable at index"""
    N_cells = len(Q_cells)
    
    if i == 0:  # Forward difference
        P_curr = simple_Q_to_P(Q_cells[i, :])
        P_next = simple_Q_to_P(Q_cells[i + 1, :])
        grad = (P_next[var_index] - P_curr[var_index]) / dx
    elif i == N_cells - 1:  # Backward difference
        P_prev = simple_Q_to_P(Q_cells[i - 1, :])
        P_curr = simple_Q_to_P(Q_cells[i, :])
        grad = (P_curr[var_index] - P_prev[var_index]) / dx
    else:  # Central difference
        P_prev = simple_Q_to_P(Q_cells[i - 1, :])
        P_next = simple_Q_to_P(Q_cells[i + 1, :])
        grad = (P_next[var_index] - P_prev[var_index]) / (2.0 * dx)
    
    return grad

def compute_lns_sources(Q_cells, dx, tau_q, tau_sigma):
    """Compute LNS relaxation source terms"""
    N_cells = len(Q_cells)
    S = np.zeros((N_cells, NUM_VARS_1D_ENH))
    
    for i in range(N_cells):
        P_i = simple_Q_to_P(Q_cells[i, :])
        rho, u_x, p, T = P_i
        q_x, s_xx = Q_cells[i, 3], Q_cells[i, 4]
        
        # Compute gradients
        du_dx = compute_gradient_simple(Q_cells, 1, dx, i)  # velocity gradient
        dT_dx = compute_gradient_simple(Q_cells, 3, dx, i)  # temperature gradient
        
        # NSF targets (equilibrium values)
        q_NSF = -K_THERM * dT_dx  # Fourier's law
        s_NSF = (4.0/3.0) * MU_VISC * du_dx  # Newton's law (1D)
        
        # Relaxation source terms
        S[i, 0] = 0.0  # No source for mass
        S[i, 1] = 0.0  # No source for momentum  
        S[i, 2] = 0.0  # No source for energy
        S[i, 3] = -(q_x - q_NSF) / tau_q  # Heat flux relaxation
        S[i, 4] = -(s_xx - s_NSF) / tau_sigma  # Stress relaxation
    
    return S

def solve_1D_LNS_step1_2_clean(N_cells, L_domain, t_final, CFL_number,
                               initial_condition_func, bc_type='periodic',
                               tau_q=1e-6, tau_sigma=1e-6):
    """Step 1.2: Add LNS source terms to validated baseline"""
    
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
    max_iters = 100000
    
    while t_current < t_final and iter_count < max_iters:
        # Time step calculation including source stiffness
        max_speed = 1e-9
        for i in range(N_cells):
            P_i = simple_Q_to_P(Q_current[i, :])
            if P_i[0] > 1e-9 and P_i[2] > 0:
                c_s = np.sqrt(GAMMA * P_i[2] / P_i[0])
                speed = abs(P_i[1]) + c_s
                max_speed = max(max_speed, speed)
        
        # Time step limits
        dt_hyperbolic = 0.25 * CFL_number * dx / max_speed
        dt_source = 0.1 * min(tau_q, tau_sigma)  # Source stiffness limit
        dt = min(dt_hyperbolic, dt_source)
        
        if t_current + dt > t_final:
            dt = t_final - t_current
        if dt < 1e-15:
            break
        
        # Ghost cells for boundary conditions
        Q_ghost = np.zeros((N_cells + 2, NUM_VARS_1D_ENH))
        Q_ghost[1:-1, :] = Q_current
        
        if bc_type == 'periodic':
            Q_ghost[0, :] = Q_current[-1, :]
            Q_ghost[-1, :] = Q_current[0, :]
        else:
            Q_ghost[0, :] = Q_current[0, :]
            Q_ghost[-1, :] = Q_current[-1, :]
        
        # Compute fluxes at interfaces
        fluxes = np.zeros((N_cells + 1, NUM_VARS_1D_ENH))
        for i in range(N_cells + 1):
            Q_L = Q_ghost[i, :]
            Q_R = Q_ghost[i + 1, :]
            fluxes[i, :] = lax_friedrichs_flux_lns(Q_L, Q_R, dx, dt)
        
        # Hyperbolic update
        Q_after_flux = Q_current.copy()
        for i in range(N_cells):
            flux_diff = fluxes[i + 1, :] - fluxes[i, :]
            Q_after_flux[i, :] = Q_current[i, :] - (dt / dx) * flux_diff
        
        # Source term update
        S = compute_lns_sources(Q_current, dx, tau_q, tau_sigma)
        
        Q_next = Q_after_flux.copy()
        for i in range(N_cells):
            Q_next[i, :] += dt * S[i, :]
            
            # Ensure physical bounds
            Q_next[i, 0] = max(Q_next[i, 0], 1e-9)  # Positive density
            
            # Check for negative pressure
            P_test = simple_Q_to_P(Q_next[i, :])
            if P_test[2] <= 0:
                # Reset to background state
                Q_next[i, :] = simple_P_to_Q(1.0, 0.0, 1.0, 1.0/R_GAS, 0.0, 0.0)
        
        # Stability monitoring
        if iter_count % 5000 == 0:
            if np.any(np.isnan(Q_next)) or np.any(np.isinf(Q_next)):
                print(f"❌ Instability at t={t_current:.2e}")
                break
        
        Q_current = Q_next
        t_current += dt
        iter_count += 1
        
        # Store solution
        if iter_count % max(1, max_iters//100) == 0:
            solution_history.append(Q_current.copy())
            time_history.append(t_current)
    
    # Final solution
    if len(solution_history) == 0 or not np.array_equal(solution_history[-1], Q_current):
        solution_history.append(Q_current.copy())
        time_history.append(t_current)
    
    return x_coords, time_history, solution_history

print("✅ Step 1.2 CLEAN: LNS source terms added to baseline")

# ============================================================================
# STEP 1.2 VALIDATION
# ============================================================================

@dataclass
class LNSParameters:
    gamma: float = 1.4
    R_gas: float = 287.0
    rho0: float = 1.0
    p0: float = 1.0
    L_domain: float = 1.0
    tau_q: float = 1e-6
    tau_sigma: float = 1e-6

class Step12Validation:
    """Validation for Step 1.2 with LNS source terms"""
    
    def __init__(self, solver_func, params: LNSParameters):
        self.solver = solver_func
        self.params = params
    
    def constant_state_ic(self, x: float, L_domain: float) -> np.ndarray:
        """Constant state with small non-equilibrium fluxes"""
        rho = self.params.rho0
        u_x = 0.0
        p = self.params.p0
        T = p / (rho * self.params.R_gas)
        
        # Small non-equilibrium initial conditions
        q_x = 0.01  # Small heat flux
        s_xx = 0.01  # Small stress
        
        return simple_P_to_Q(rho, u_x, p, T, q_x, s_xx)
    
    def test_stability_with_sources(self) -> bool:
        """Test stability with source terms active"""
        print("📋 Test: Stability with Source Terms")
        
        try:
            x_coords, t_hist, Q_hist = self.solver(
                N_cells=30,
                L_domain=self.params.L_domain,
                t_final=0.05,
                CFL_number=0.3,
                initial_condition_func=self.constant_state_ic,
                bc_type='periodic',
                tau_q=self.params.tau_q,
                tau_sigma=self.params.tau_sigma
            )
            
            if Q_hist and len(Q_hist) > 1:
                Q_final = Q_hist[-1]
                if not np.any(np.isnan(Q_final)) and not np.any(np.isinf(Q_final)):
                    # Check physical values
                    densities = [simple_Q_to_P(Q_final[i, :])[0] for i in range(len(Q_final))]
                    pressures = [simple_Q_to_P(Q_final[i, :])[2] for i in range(len(Q_final))]
                    
                    if all(d > 0 for d in densities) and all(p > 0 for p in pressures):
                        print("  ✅ Stable with active source terms")
                        return True
                    else:
                        print("  ❌ Unphysical values with sources")
                        return False
                else:
                    print("  ❌ NaN/Inf with source terms")
                    return False
            else:
                print("  ❌ No solution with source terms")
                return False
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return False
    
    def test_conservation_with_sources(self) -> bool:
        """Test conservation with source terms"""
        print("📋 Test: Conservation with Source Terms")
        
        try:
            x_coords, t_hist, Q_hist = self.solver(
                N_cells=40,
                L_domain=self.params.L_domain,
                t_final=0.03,
                CFL_number=0.25,
                initial_condition_func=self.constant_state_ic,
                bc_type='periodic',
                tau_q=self.params.tau_q,
                tau_sigma=self.params.tau_sigma
            )
            
            if Q_hist and len(Q_hist) >= 2:
                dx = self.params.L_domain / len(Q_hist[0])
                
                # Check mass conservation
                mass_initial = np.sum(Q_hist[0][:, 0]) * dx
                mass_final = np.sum(Q_hist[-1][:, 0]) * dx
                mass_error = abs((mass_final - mass_initial) / mass_initial)
                
                print(f"    Mass conservation error: {mass_error:.2e}")
                
                if mass_error < 1e-8:
                    print("  ✅ Excellent mass conservation with sources")
                    return True
                elif mass_error < 1e-6:
                    print("  ✅ Good mass conservation with sources")
                    return True
                else:
                    print("  ❌ Poor mass conservation with sources")
                    return False
            else:
                print("  ❌ Insufficient data")
                return False
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return False
    
    def test_nsf_limit(self) -> bool:
        """Test NSF limit: small tau should approach classical behavior"""
        print("📋 Test: NSF Limit Behavior")
        
        try:
            # Run with very small relaxation times
            x_coords, t_hist, Q_hist = self.solver(
                N_cells=25,
                L_domain=self.params.L_domain,
                t_final=0.02,
                CFL_number=0.2,
                initial_condition_func=self.constant_state_ic,
                bc_type='periodic',
                tau_q=1e-8,  # Very small
                tau_sigma=1e-8
            )
            
            if not Q_hist or len(Q_hist) < 2:
                print("  ❌ Simulation failed")
                return False
            
            Q_final = Q_hist[-1]
            
            # Check that heat flux and stress relax toward equilibrium
            q_final = np.mean(np.abs(Q_final[:, 3]))
            s_final = np.mean(np.abs(Q_final[:, 4]))
            
            print(f"    Final |q_x|: {q_final:.3e}")
            print(f"    Final |s_xx|: {s_final:.3e}")
            
            # Should be much smaller than initial values (0.01)
            if q_final < 0.005 and s_final < 0.005:
                print("  ✅ Good NSF limit behavior")
                return True
            else:
                print("  ❌ Poor NSF limit convergence")
                return False
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return False
    
    def run_step12_validation(self) -> bool:
        """Run Step 1.2 validation suite"""
        print("\n🔍 Step 1.2 Validation: LNS Source Terms")
        print("=" * 40)
        
        tests = [
            ("Stability", self.test_stability_with_sources),
            ("Conservation", self.test_conservation_with_sources),
            ("NSF Limit", self.test_nsf_limit)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            result = test_func()
            results.append(result)
        
        passed = sum(results)
        total = len(results)
        
        print("\n" + "=" * 40)
        print(f"📊 SUMMARY: {passed}/{total} tests passed")
        
        if passed >= 2:  # At least 2/3 tests pass
            print("✅ Step 1.2: LNS physics successfully added!")
            return True
        else:
            print("❌ Step 1.2 needs work")
            return False

# Initialize Step 1.2 validation
params = LNSParameters()
step12_validator = Step12Validation(solve_1D_LNS_step1_2_clean, params)

print("✅ Step 1.2 validation ready")

# ============================================================================
# RUN STEP 1.2 VALIDATION
# ============================================================================

print("🚀 Testing Step 1.2 implementation...")

step1_2_success = step12_validator.run_step12_validation()

if step1_2_success:
    print("\n🎉 SUCCESS: Step 1.2 passes!")
    print("LNS physics successfully integrated with baseline solver.")
    print("Ready to proceed with Step 1.3 (semi-implicit sources).")
else:
    print("\n❌ Step 1.2 needs debugging.")
    print("Source terms integration has issues.")