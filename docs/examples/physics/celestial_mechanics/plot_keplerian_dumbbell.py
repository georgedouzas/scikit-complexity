"""
# Keplerian dumbbell and spin-orbit coupling

Two masses joined by a rigid rod orbit a heavy primary. The gravity gradient across the rod pulls harder on the near
mass, so the spin of the dumbbell and its orbit exchange angular momentum, the simplest spin-orbit coupling of a rigid
body, like an elongated asteroid.
"""

# %%
# The dumbbell of equal masses and length `epsilon = 0.1` starts on a circular orbit with its arm in the orbital plane,
# spinning three times faster than it orbits. The default blueprint shows its coordinates over time.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation, create_system

SYSTEM = 'keplerian_dumbbell'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=8)
simulation = Simulation(system, dt=0.05, observers=[observer]).prepare().run(until=60)

# %%
# Exchange of angular momentum
# ----------------------------
# The orbital angular momentum `u² theta'` and the spin `I phi'` should oscillate in opposite phase, while their sum,
# the total angular momentum, stays constant to the accuracy of the solver.

run = create_simulation(SYSTEM, epsilon=0.3, spin=1.5).prepare().run(until=60)
trajectory = run.trajectory_
times = np.asarray(trajectory.times)
u, theta_rate, phi_rate = (np.ravel(trajectory.values(f'dumbbell.{name}')) for name in ('u', 'theta_d1', 'phi_d1'))
inertia = 0.25 / 0.9 * 0.3**2
orbit, spin = u**2 * theta_rate, inertia * phi_rate
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(times, orbit - orbit[0], label='orbital')
ax.plot(times, spin - spin[0], label='spin')
ax.plot(times, orbit + spin - orbit[0] - spin[0], '--', color='black', label='total')
ax.set_xlabel('time')
ax.set_ylabel('change of angular momentum')
ax.set_title('Spin-orbit exchange, epsilon = 0.3')
ax.legend()
plt.show()
