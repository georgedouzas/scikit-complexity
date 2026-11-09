"""
# PID-stabilised inverted pendulum

A PID controller of the angle pushes a cart to keep the pendulum on top of it upright. From a tilt of 0.1 radians the
pendulum swings back to upright within a second and stays there, while the cart drifts away, since the controller
does not watch its position.
"""

# %%
# The system is built from its class with a cart of mass 1, a point mass 0.1 on a rod of length 0.5, and
# the gains `kp = 40`, `ki = 1`, and `kd = 8`. RK4 integrates it with steps of 1 ms, and the controller sets the force
# after every step. The default blueprint shows the angle, the position, and the force in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.control import PidInvertedPendulumSystem

SYSTEM = 'pid_inverted_pendulum'

system = PidInvertedPendulumSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer])

# %%
# The closed loop near upright
# ----------------------------
# Linearised near upright and without the small integral term, the angle follows
# `M l theta'' = -kd theta' - (kp - (M + m) g) theta`, a damped oscillator. Its solution from the tilt 0.1 should
# follow the simulated angle closely.

M, m, l, g, kp, kd = 1.0, 0.1, 0.5, 9.81, 40.0, 8.0
roots = np.roots([M * l, kd, kp - (M + m) * g])
times = trajectory.times
first, second = roots
linear = 0.1 * (second * np.exp(first * times) - first * np.exp(second * times)) / (second - first)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, trajectory.get_values('cart.theta'), label='simulated angle')
ax.plot(times, np.real(linear), '--', label='linear closed loop')
ax.axhspan(-0.01, 0.01, color='gray', alpha=0.2, label='within 0.01 of upright')
ax.set_xlabel('time (s)')
ax.set_ylabel('angle theta from upright')
ax.set_title('The controller restores the upright pendulum')
ax.legend()
plt.show()
