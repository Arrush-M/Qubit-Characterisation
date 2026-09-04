import numpy as np
import qutip as qt
from qutip import Bloch
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
from scipy.constants import h, k
import time
import matplotlib.lines as mlines

# =====================================================================
# 1. Pulse Envelope Definitions
# =====================================================================

def generate_gaussian(tlist, sigma, center, target_area=np.pi):
    """Generates a Gaussian pulse envelope calibrated to a target area."""
    dt = tlist[1] - tlist[0]
    envelope = np.exp(-0.5 * ((tlist - center) / sigma)**2)
    area = np.trapezoid(envelope, dx=dt)
    return envelope * (target_area / area)

def generate_slepian(tlist, NW=4, target_area=np.pi):
    """Generates a Slepian (DPSS) window pulse envelope."""
    dt = tlist[1] - tlist[0]
    num_samples = len(tlist)
    envelope = dpss(num_samples, NW=NW)
    area = np.trapezoid(envelope, dx=dt)
    return envelope * (target_area / area)

def apply_hd_drag(I_t, dt, alpha):
    """Applies Higher-Derivative (HD) DRAG to a base envelope I(t)."""
    dI_dt = np.gradient(I_t, dt)
    Q_corr = - (0.5 / alpha) * dI_dt
    I_corr = I_t
    return I_corr, Q_corr


# =====================================================================
# 2. Physics & System Parameters
# =====================================================================

N = 3                   # Truncate transmon to 3 levels
gate_time = 20.0        # ns
num_points = 1000       # AWG sample points
tlist = np.linspace(0, gate_time, num_points)
dt = tlist[1] - tlist[0]

# Frequencies (Angular = 2 * pi * GHz = rad/ns)
qubit_freq_Hz = 5.1e9
alpha_ghz = -0.300
alpha = 2 * np.pi * alpha_ghz  

# Noise Parameters
T_eff = 0.050 # Effective temperature (K)
T1 = 50000.0  # Relaxation time (ns) -> 50 us
T2 = 20000.0  # Ramsey dephasing time (ns) -> 20 us
gamma_phi = (1.0 / T2) - (1.0 / (2.0 * T1))

exponent = (h * qubit_freq_Hz) / (k * T_eff)
n_th = 1.0 / (np.exp(exponent) - 1.0)

PULSE_SHAPE = 'gaussian' 


# =====================================================================
# 3. Generate the AWG Pulses (DRAG vs Non-DRAG)
# =====================================================================

# 3a. Generate Base I(t)
if PULSE_SHAPE == 'gaussian':
    sigma = gate_time / 5.0
    center = gate_time / 2.0
    base_I = generate_gaussian(tlist, sigma, center, target_area=np.pi)
elif PULSE_SHAPE == 'slepian':
    base_I = generate_slepian(tlist, NW=4, target_area=np.pi)

# 3b. Non-DRAG Envelopes
I_t_nodrag = base_I
Q_t_nodrag = np.zeros_like(tlist)

# 3c. DRAG Envelopes
I_t_drag, Q_t_drag = apply_hd_drag(base_I, dt, alpha=alpha)


# =====================================================================
# 4. QuTiP Setup
# =====================================================================

a = qt.destroy(N)
ad = a.dag()
n_op = ad * a

H_drift = 0.5 * alpha * ad * ad * a * a
H_I = 0.5 * (a + ad)
H_Q = 0.5 * 1j * (ad - a)

# Hamiltonians for both cases
H_total_nodrag = [H_drift, [H_I, I_t_nodrag], [H_Q, Q_t_nodrag]]
H_total_drag = [H_drift, [H_I, I_t_drag], [H_Q, Q_t_drag]]

c_ops = [
    np.sqrt((1.0 / T1) * (1.0 + n_th)) * a, 
    np.sqrt((1.0 / T1) * n_th) * ad,        
    np.sqrt(gamma_phi) * n_op               
]

psi0 = qt.basis(N, 0)


# =====================================================================
# 5. Run Simulations
# =====================================================================

print(f"Simulating {gate_time} ns X-Gate using {PULSE_SHAPE.capitalize()} pulse...")

t0 = time.time()
print("1/2: Simulating Non-DRAG...")
result_nodrag = qt.mesolve(H_total_nodrag, psi0, tlist, c_ops)

print("2/2: Simulating DRAG...")
result_drag = qt.mesolve(H_total_drag, psi0, tlist, c_ops)

print(f"Total Simulation Time: {time.time() - t0:.3f} seconds")


# =====================================================================
# 6. Plotting Results
# =====================================================================

def extract_qubit_state(state):
    """Truncates 3-level Qobj to 2-level and renormalizes."""
    if state.type in ['ket', 'bra']:
        mat = state.full()[:N-1]
        norm = np.linalg.norm(mat)
        if norm > 0: mat = mat / norm
        return qt.Qobj(mat)
    else:
        mat = state.full()[:N-1, :N-1]
        tr = np.real(np.trace(mat))
        if tr > 0: mat = mat / tr
        return qt.Qobj(mat)

# Extract states for both
states_nodrag = [extract_qubit_state(rho) for rho in result_nodrag.states]
states_drag = [extract_qubit_state(rho) for rho in result_drag.states]

sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()

# Expectation values (Trajectories)
pts_nodrag = [qt.expect(sx, states_nodrag), qt.expect(sy, states_nodrag), qt.expect(sz, states_nodrag)]
pts_drag = [qt.expect(sx, states_drag), qt.expect(sy, states_drag), qt.expect(sz, states_drag)]

# Setup Bloch sphere
b = Bloch()


# Add Trajectories (lines)
b.add_points(pts_nodrag, meth='l', colors=['#1f77b4'])
b.add_points(pts_drag, meth='l', colors=['#d62728'])

# Add Final States (arrows)
b.add_states(states_nodrag[-1])
b.add_states(states_drag[-1])

# Render Bloch Sphere
b.show()

# Add Custom Legend and Title
blue_line = mlines.Line2D([], [], color='#1f77b4', label='Non-DRAG')
red_line = mlines.Line2D([], [], color='#d62728', label='DRAG')
b.fig.suptitle("Pulse Trajectories on the Bloch Sphere")

# Keeps the window open!
plt.show()