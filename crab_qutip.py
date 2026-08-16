import numpy as np
import qutip as qt
import qutip_qoc as qoc
from scipy.constants import h, k
np.random.seed(5040) # For reproducibility

# =====================================================================
# 1. Physics & Hardware System Parameters
# =====================================================================
N = 3
gate_time = 20.0
num_tslots = 200
tlist = np.linspace(0, gate_time, num_tslots)
dt = tlist[1] - tlist[0]

qubit_freq_Hz = 5.0e9
alpha_ghz = -0.300
alpha = 2 * np.pi * alpha_ghz  

T_eff = 0.050                
T1 = 50000.0                 
T2 = 20000.0                 
gamma_phi = (1.0 / T2) - (1.0 / (2.0 * T1))

exponent = (h * qubit_freq_Hz) / (k * T_eff)
n_th = 1.0 / (np.exp(exponent) - 1.0)

# =====================================================================
# 2. Define Operators & Hamiltonians
# =====================================================================
a = qt.destroy(N)
ad = a.dag()
n_op = ad * a

P0 = qt.basis(N, 0) * qt.basis(N, 0).dag()
P1 = qt.basis(N, 1) * qt.basis(N, 1).dag()
P2 = qt.basis(N, 2) * qt.basis(N, 2).dag()

H_drift = 0.5 * alpha * ad * ad * a * a
H_I = 0.5 * (a + ad)
H_Q = 0.5 * 1j * (ad - a)

H_qoc = [
    H_drift,
    [H_I, "ctrl_I"],
    [H_Q, "ctrl_Q"]
]

# =====================================================================
# 3. Generate a Gaussian Initial Guess
# =====================================================================
# CRAB needs a non-zero starting point. We give it a standard Gaussian pi-pulse.
sigma = gate_time / 5.0
center = gate_time / 2.0
guess_I = np.exp(-0.5 * ((tlist - center) / sigma)**2)
area = np.trapezoid(guess_I, dx=dt)
guess_I = guess_I * (np.pi / area)  # Calibrate to Pi area
guess_Q = np.zeros_like(tlist)      # Q channel starts at zero


# =====================================================================
# 4. CRAB Optimization
# =====================================================================
print("\nRunning CRAB optimization...")

U_targ_matrix = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
U_targ = qt.Qobj(U_targ_matrix)
U_0 = qt.qeye(N)

objective = qoc.Objective(initial=U_0, target=U_targ, H=H_qoc)

# Provide the Gaussian guess instead of np.zeros
control_parameters = {
    "ctrl_I": {"guess": guess_I, "bounds": [-2.0, 2.0]},
    "ctrl_Q": {"guess": guess_Q, "bounds": [-2.0, 2.0]}
}

res_crab = qoc.optimize_pulses(
    objectives=[objective],
    control_parameters=control_parameters,
    tlist=tlist,
    algorithm_kwargs={
        "alg": "CRAB",
        "fid_err_targ": 1e-4,   
        "max_iter": 1000,
    }
)

print(f"Optimization finished. Final Ideal Infidelity: {res_crab.infidelity:.2e}")

# qutip-qoc returns the final pulses here
I_t = res_crab.optimized_controls[0]
Q_t = res_crab.optimized_controls[1]


# =====================================================================
# 5. Open System Simulation (Adding Thermal & Dephasing Noise)
# =====================================================================
print("\nTesting optimized CRAB pulse in noisy thermal environment...")

H_noisy = [
    H_drift,
    [H_I, I_t],
    [H_Q, Q_t]
]

c_ops = [
    np.sqrt((1.0 / T1) * (1.0 + n_th)) * a, 
    np.sqrt((1.0 / T1) * n_th) * ad,        
    np.sqrt(gamma_phi) * n_op               
]

psi0 = qt.basis(N, 0)
result = qt.mesolve(H_noisy, psi0, tlist, c_ops, e_ops=[P0, P1, P2])

pop_0 = result.expect[0]
pop_1 = result.expect[1]
pop_2 = result.expect[2]


# =====================================================================
# 6. Plotting
# =====================================================================
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(tlist, I_t, label='I(t) - In Phase', color='blue', lw=2)
ax1.plot(tlist, Q_t, label='Q(t) - Quadrature', color='red', lw=2)
ax1.set_ylabel("Drive Amplitude (rad/ns)")
ax1.set_title("CRAB Optimized Pulses")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(tlist, pop_0, label=r'Ground State $|0\rangle$', color='black', lw=2)
ax2.plot(tlist, pop_1, label=r'Excited State $|1\rangle$', color='green', lw=2)
ax2.plot(tlist, pop_2, label=r'Leakage State $|2\rangle$', color='purple', lw=2)
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Population")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- Final Open-System (Thermal + T1 + Tphi) Results ---")
print(f"Final |1> Fidelity: {pop_1[-1]:.5f}")
print(f"Final |2> Leakage:  {pop_2[-1]:.5e}")