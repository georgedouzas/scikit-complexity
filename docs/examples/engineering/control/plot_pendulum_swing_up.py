"""
# Pendulum swing-up

Pushing the pivot back and forth in time with the swing, the controller pumps energy into a hanging pendulum. Each
swing goes higher than the last until the pendulum reaches its inverted position, with the energy of the upright
pendulum.
"""

# %%
# The model is created by name from the catalog with the pendulum hanging at rest, the natural frequency
# `omega = 1.565`, and the largest pivot acceleration `mu = 0.1`, and is integrated with RK4 for 100 time units. The
# default blueprint shows the angle and the pivot position with their rates in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'pendulum_swing_up'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.1, solver=RK4(dt=0.01), observers=[observer]).run(until=100)

# %%
# Energy pumping
# --------------
# The control law always pushes the energy `E = theta'^2 / 2 - omega^2 cos(theta)` towards `E_s = omega^2`, the energy
# of the inverted pendulum, so `E` should rise in steps from near `-omega^2`, the pendulum hanging almost at rest, to
# `omega^2` and stay there, while the swings in the phase plane grow until they reach the separatrix through the
# inverted position.

omega = 1.5652475842498528
trajectory = simulation.trajectory_
theta, rate = trajectory['pendulum.theta'], trajectory['pendulum.theta_d1']
energy = rate**2 / 2 - omega**2 * np.cos(theta)
fig, (pumping, phase) = plt.subplots(1, 2, figsize=(10, 4))
pumping.plot(trajectory.times, energy)
pumping.axhline(omega**2, color='black', linestyle='--', linewidth=0.8, label='inverted position')
pumping.axhline(-(omega**2), color='gray', linestyle=':', linewidth=0.8, label='hanging at rest')
pumping.set_xlabel('time')
pumping.set_ylabel('energy E')
pumping.set_title('The controller pumps the energy up')
pumping.legend()
grid = np.linspace(-np.pi, np.pi, 300)
phase.plot(np.angle(np.exp(1j * theta)), rate, '.', markersize=1)
for sign in (-1, 1):
    phase.plot(grid, sign * 2 * omega * np.cos(grid / 2), color='black', linewidth=0.8)
phase.set_xlabel('angle theta (wrapped)')
phase.set_ylabel("angular velocity theta'")
phase.set_title('Swings grow to the separatrix')
fig.tight_layout()
plt.show()
