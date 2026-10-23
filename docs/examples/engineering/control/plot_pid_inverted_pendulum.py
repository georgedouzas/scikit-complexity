"""
# PID-stabilised inverted pendulum

A PID controller of the angle pushes a cart to keep the pendulum on top of it upright. From a tilt of 0.1 radians the
pendulum swings back to upright within a second and stays there, while the cart drifts away, since the controller
does not watch its position.
"""

# %%
# The model is created by name from the catalog with a cart of mass 1, a point mass 0.1 on a rod of length 0.5, and
# the gains `kp = 40`, `ki = 1`, and `kd = 8`. RK4 integrates it with steps of 1 ms, and the controller sets the force
# after every step. The default blueprint shows the angle, the position, and the force in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'pid_inverted_pendulum'

system = create(MODEL)
entry = describe(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=0.05,
    solver=entry.defaults['solver'],
    observers=[observer],
    observables=entry.observables_of(system),
).run(until=5)

# %%
# The closed loop near upright
# ----------------------------
# Linearised near upright and without the small integral term, the angle follows
# `M l theta'' = -kd theta' - (kp - (M + m) g) theta`, a damped oscillator. Its solution from the tilt 0.1 should
# follow the simulated angle closely.

M, m, l, g, kp, kd = 1.0, 0.1, 0.5, 9.81, 40.0, 8.0
roots = np.roots([M * l, kd, kp - (M + m) * g])
trajectory = simulation.trajectory_
times = trajectory.times
first, second = roots
linear = 0.1 * (second * np.exp(first * times) - first * np.exp(second * times)) / (second - first)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, trajectory['cart.theta'], label='simulated angle')
ax.plot(times, np.real(linear), '--', label='linear closed loop')
ax.axhspan(-0.01, 0.01, color='gray', alpha=0.2, label='within 0.01 of upright')
ax.set_xlabel('time (s)')
ax.set_ylabel('angle theta from upright')
ax.set_title('The controller restores the upright pendulum')
ax.legend()
plt.show()
