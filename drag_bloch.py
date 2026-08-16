import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# =====================================================================
# 1. Pulse Definitions
# =====================================================================
gate_time = 10.0  # ns (Shorter gate makes the DRAG effect more visually obvious)
tlist = np.linspace(0, gate_time, 1000)
dt = tlist[1] - tlist[0]

# Standard Gaussian (Zeroed so it starts and ends smoothly)
sigma = gate_time / 5.0
center = gate_time / 2.0
I_t = np.exp(-0.5 * ((tlist - center) / sigma)**2)
I_t = I_t - I_t[0] 

# Normalize area to Pi (for an X-gate)
area = np.trapezoid(I_t, dx=dt)
I_t = I_t * (np.pi / area)

# Standard DRAG (1st-Order Derivative)
# Note: In a pure 2-level system, alpha doesn't physically exist. 
# We define an arbitrary alpha just to set the amplitude of the Q-channel correction.
alpha = -2 * np.pi * 0.300 
Q_t = -(1.0 / alpha) * np.gradient(I_t, dt)

# =====================================================================
# 2. System Setup (2-Level System)
# =====================================================================
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

# Drive Hamiltonians
H_I = 0.5 * sx
H_Q = 0.5 * sy

# Two different simulations for comparison
H_gaussian = [[H_I, I_t]]
H_drag = [[H_I, I_t], [H_Q, Q_t]]

# Start at North Pole |0>
psi0 = qt.basis(2, 0)

# =====================================================================
# 3. Simulate (Schrodinger Evolution - No Noise)
# =====================================================================
# We use e_ops to automatically extract the X, Y, Z coordinates for the Bloch sphere
res_gaussian = qt.sesolve(H_gaussian, psi0, tlist, e_ops=[sx, sy, sz])
res_drag = qt.sesolve(H_drag, psi0, tlist, e_ops=[sx, sy, sz])

# =====================================================================
# 4. Visualize on the Bloch Sphere
# =====================================================================
b = qt.Bloch()
b.figsize = [8, 8]

# Render the blank sphere FIRST
b.render()

# Extract the underlying 3D Matplotlib axis from the Bloch object
ax = b.axes

# Plot the trajectories directly using standard Matplotlib 3D plotting
ax.plot(res_gaussian.expect[0], res_gaussian.expect[1], res_gaussian.expect[2], 
        color='blue', lw=2, label='Standard Gaussian')

ax.plot(res_drag.expect[0], res_drag.expect[1], res_drag.expect[2], 
        color='red', lw=2, label='Gaussian + DRAG')

# Standard Matplotlib formatting
plt.legend(loc='upper right')
plt.title(f"DRAG Trajectory during a {gate_time} ns X-Gate")

plt.show()