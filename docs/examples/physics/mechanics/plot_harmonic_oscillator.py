"""
# Harmonic oscillator

A mass on a spring swings back and forth through its rest position forever: the displacement is a cosine of constant
amplitude and period `2π sqrt(m / k)`, and the velocity follows a quarter of a period behind.
"""

# %%
# The mass starts at rest one unit from equilibrium on an undamped spring with `k = m = 1`, so its period is `2π`. The
# run covers a little more than three periods, and the default blueprint shows the displacement first, with the
# velocity in a second time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'harmonic_oscillator'

system = create(MODEL, x=1.0, v=0.0, k=1.0, mass=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=20)

# %%
# Exact solution and energy
# -------------------------
# The exact motion is `x(t) = cos t`. The simulated displacement should lie on it, and the energy
# `E = m v^2 / 2 + k x^2 / 2` should stay at its initial value `1 / 2` up to the error of the solver.

times = np.asarray(simulation.trajectory_.times)
position = np.ravel(simulation.trajectory_['mass.x'])
velocity = np.ravel(simulation.trajectory_['mass.v'])
energy = velocity**2 / 2 + position**2 / 2
fig, (motion, drift) = plt.subplots(1, 2, figsize=(10, 4))
motion.plot(times, position, label='simulated')
motion.plot(times, np.cos(times), '--', color='black', label='cos t')
motion.set_xlabel('time')
motion.set_ylabel('displacement')
motion.set_title('Displacement against the exact solution')
motion.legend()
drift.plot(times, (energy - energy[0]) / energy[0])
drift.set_xlabel('time')
drift.set_ylabel('relative change of the energy')
drift.set_title('The energy is conserved')
fig.tight_layout()
plt.show()
