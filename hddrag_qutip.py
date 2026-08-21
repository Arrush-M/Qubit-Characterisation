import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
from scipy.constants import h, k

# =====================================================================
# 1. Pulse Envelope Definitions
# =====================================================================

def generate_gaussian(tlist, sigma, center, target_area=np.pi):
    """Generates a Gaussian pulse envelope calibrated to a target area."""
    dt = tlist[1] - tlist[0]
    envelope = np.exp(-0.5 * ((tlist - center) / sigma)**2)
    # Normalize area to exactly target_area (e.g., Pi for an X gate)
    area = np.trapezoid(envelope, dx=dt)
    return envelope * (target_area / area)

def generate_slepian(tlist, NW=4, target_area=np.pi):
    """Generates a Slepian (DPSS) window pulse envelope."""
    dt = tlist[1] - tlist[0]
    num_samples = len(tlist)
    # Slepian window maximizes energy in the main frequency band
    envelope = dpss(num_samples, NW=NW)
    area = np.trapezoid(envelope, dx=dt)
    return envelope * (target_area / area)

def apply_hd_drag(I_t, dt, alpha):
    """
    Applies Higher-Derivative (HD) DRAG to a base envelope I(t).
    
    Parameters:
    - I_t: The base in-phase envelope.
    - dt: Time step size.
    - alpha: Transmon anharmonicity (in angular frequency, rad/ns).
    - delta: Static detuning (if the drive is slightly off-resonance).
    
    Returns:
    - I_corr: The corrected In-phase envelope.
    - Q_corr: The corrected Quadrature envelope.
    """
    # Calculate numerical derivatives
    dI_dt = np.gradient(I_t, dt)
    ddI_dt2 = np.gradient(dI_dt, dt)
    dddI_dt3 = np.gradient(ddI_dt2, dt)
    
    # 1st-Order DRAG Q-channel correction (prevents leakage to |2>)
    Q_corr = - (1.0 / alpha) * dI_dt - (1.0 / (alpha**2)) * dddI_dt3
    
    # HD-DRAG I-channel correction (corrects phase accumulation and higher-order errors)
    # The second derivative term accounts for dynamic Stark shifts.
    I_corr = I_t - (1.0 / (alpha**2)) * ddI_dt2 # Swap 
    
    return I_corr, Q_corr


# =====================================================================
# 2. Physics & System Parameters
# =====================================================================

N = 3                   # Truncate transmon to 3 levels: |0>, |1>, |2>
gate_time = 20.0        # ns
num_points = 1000       # AWG sample points
tlist = np.linspace(0, gate_time, num_points)
dt = tlist[1] - tlist[0]

# Frequencies (Angular = 2 * pi * GHz = rad/ns)
# Note: Anharmonicity is usually negative (~ -300 MHz)
qubit_freq_Hz = 5.1e9
alpha_ghz = -0.300
alpha = 2 * np.pi * alpha_ghz  

# Noise Parameters
T_eff = 0.050 # Effective temperature (K)
T1 = 50000.0  # Relaxation time (ns) -> 50 us
T2 = 20000.0  # Ramsey dephasing time (ns) -> 20 us
# Calculate pure dephasing rate: 1/T2 = 1/(2*T1) + gamma_phi
gamma_phi = (1.0 / T2) - (1.0 / (2.0 * T1))

exponent = (h * qubit_freq_Hz) / (k * T_eff)
n_th = 1.0 / (np.exp(exponent) - 1.0)

# Choose Pulse Shape: 'gaussian' or 'slepian'
PULSE_SHAPE = 'gaussian' 
USE_DRAG = True


# =====================================================================
# 3. Generate the AWG Pulses
# =====================================================================

# 3a. Generate Base I(t)
if PULSE_SHAPE == 'gaussian':
    # Sigma is typically 1/4th or 1/5th of the total gate time
    sigma = gate_time / 5.0
    center = gate_time / 2.0
    base_I = generate_gaussian(tlist, sigma, center, target_area=np.pi)
elif PULSE_SHAPE == 'slepian':
    base_I = generate_slepian(tlist, NW=4, target_area=np.pi)

# 3b. Apply HD-DRAG
if USE_DRAG:
    I_t, Q_t = apply_hd_drag(base_I, dt, alpha=alpha)
else:
    I_t = base_I
    Q_t = np.zeros_like(tlist)


# =====================================================================
# 4. QuTiP Setup
# =====================================================================

# Operators
a = qt.destroy(N)
ad = a.dag()

# Number operator (n) and projectors for measuring states
n_op = ad * a
P0 = qt.basis(N, 0) * qt.basis(N, 0).dag()
P1 = qt.basis(N, 1) * qt.basis(N, 1).dag()
P2 = qt.basis(N, 2) * qt.basis(N, 2).dag()

# 4a. Hamiltonians
# Drift Hamiltonian: Transmon Anharmonicity
# H_drift = (alpha / 2) * a^dag * a^dag * a * a
H_drift = 0.5 * alpha * ad * ad * a * a

# Drive Operators
H_I = 0.5 * (a + ad)
H_Q = 0.5 * 1j * (ad - a)

# Full Time-Dependent Hamiltonian for mesolve
H_total = [
    H_drift,
    [H_I, I_t],
    [H_Q, Q_t]
]

# 4b. Dissipators (Collapse Operators)
c_ops = [
    np.sqrt((1.0 / T1) * (1.0 + n_th)) * a, 
    np.sqrt((1.0 / T1) * n_th) * ad,        
    np.sqrt(gamma_phi) * n_op               
]

# 4c. Initial State
# Start in the Ground State |0>
psi0 = qt.basis(N, 0)


# =====================================================================
# 5. Run Simulation
# =====================================================================

print(f"Simulating {gate_time} ns X-Gate using {PULSE_SHAPE.capitalize()} pulse...")
if USE_DRAG: print("HD-DRAG is Active.")

# e_ops measures the population of each state over time
result = qt.mesolve(H_total, psi0, tlist, c_ops, e_ops=[P0, P1, P2])

pop_0 = result.expect[0]
pop_1 = result.expect[1]
pop_2 = result.expect[2] # Leakage!


# =====================================================================
# 6. Plotting Results
# =====================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: The AWG Pulses
ax1.plot(tlist, I_t, label='I(t) - In Phase', color='blue', lw=2)
ax1.plot(tlist, Q_t, label='Q(t) - Quadrature', color='red', lw=2)
ax1.set_ylabel("Drive Amplitude (rad/ns)")
ax1.set_title(f"{PULSE_SHAPE.capitalize()} $\pi$-pulse with HD-DRAG")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Transmon State Populations
ax2.plot(tlist, pop_0, label=r'Ground State $|0\rangle$', color='black', lw=2)
ax2.plot(tlist, pop_1, label=r'Excited State $|1\rangle$', color='green', lw=2)
ax2.plot(tlist, pop_2, label=r'Leakage State $|2\rangle$', color='purple', lw=2)
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Population")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final Gate Metrics
print(f"Final |1> Fidelity:  {pop_1[-1]:.5f}")
print(f"Final |2> Leakage:   {pop_2[-1]:.5e}")
